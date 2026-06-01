"""
8_mar_SC3_rad_PT768xCT8_325_LOCODEV_pc_2.py

True exhaustive SC3 with PT-forced modality split + LOCO_DEV as 5th ranker.
PC_2 version — exhaustive design: saves ALL Optuna trials x coaches.

Design: No Stage 2 substitution scan. Instead, for every Optuna trial across
all rankers x GMs, directly evaluate CHUS+CHUP with each of 4 coaches.
This gives total_rows = TRIALS * n_rankers * n_gms * n_coaches =
  2000 * 5 * 2 * 4 = 80,000 rows (SVM+CoxPH GMs only to limit runtime).

Fix 1: RATIO_GRID_FULL applied at enqueue time (explicit n_clin=1 seeds)
Fix 2: CLIN_RATIO_MAX = 0.20
Fix 3: CT_MIN=2, dense seeds at n_ct=4-8

Machine: PC | TRIALS=2000 | N_MIN=6, N_MAX=20 | PT_MIN=4, CT_MIN=2
Ranking methods: UNIVAR, BORDA, GBS, StabLASSO, LOCO_DEV (5 rankers)
GM models: SVM, CoxPH (2 fast GMs — exhaustive saves all)
Coach models: EST, SVM, GBS, CoxPH (4 coaches per trial)

Rad pool: PT768 (35 PT feat) + CT8_325 (39 CT feat) = 74 features (post-dedup)
Clinical: 3 features (Age, Gender_Male, Treatment_CRT)
Config: N_FOLDS=5, SEED=42, TRIALS=2000

Outputs: Mar_2026/8_mar_SC3_rad_PT768xCT8_325_LOCODEV_pc_2_outputs/

Usage:
  python Mar_2026/8_mar_SC3_rad_PT768xCT8_325_LOCODEV_pc_2.py
  SC3_TRIALS=1000 python Mar_2026/8_mar_SC3_rad_PT768xCT8_325_LOCODEV_pc_2.py
"""

import os
import subprocess
import sys
import warnings
warnings.filterwarnings("ignore")

try:
    import optuna
except ImportError:
    print("Installing optuna...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna", "-q"])
    import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.ensemble import ExtraSurvivalTrees, GradientBoostingSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

# ==========================================
# CONFIG
# ==========================================
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "8_mar_SC3_rad_PT768xCT8_325_LOCODEV_pc_2_outputs"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

FINALIST_DIR = ROOT / "2_mar_finalist_outputs"
PT_FEATURES_FILE = FINALIST_DIR / "PT_inter1_768_features.csv"   # 35 PT features
CT_FEATURES_FILE = FINALIST_DIR / "CT_inter8_325_features.csv"   # 39 CT features

PT_DEV_FILE = ROOT / "27_feb_PT_development.csv"
CT_DEV_FILE = ROOT / "27_feb_CT_development.csv"
PT_EXT_FILE = ROOT / "27_feb_PT_external.csv"
CT_EXT_FILE = ROOT / "27_feb_CT_external.csv"

CLINICAL_FILE = (
    ROOT.parent / "Feb_2026" / "25_feb_clinical_reduced_dataset"
    / "25_feb_Processed_clinical_reduced.csv"
)
CLINICAL_FEATURES = ["Age", "Gender_Male", "Treatment_CRT"]

SEED = 42
N_FOLDS = 5
N_BOOT = 1000
W_PERF, W_STAB, STD_THRESHOLD = 0.7, 0.3, 0.08
N_MIN, N_MAX = 6, 20
# Fix 2: CLIN_RATIO_MAX = 0.20
CLIN_RATIO_MIN, CLIN_RATIO_MAX = 0.05, 0.20
PT_MIN = 4
# Fix 3: CT_MIN=2
CT_MIN = 2
TRIALS = int(os.getenv("SC3_TRIALS", "2000"))

RANKING_METHODS = ["UNIVAR", "BORDA", "GBS", "StabLASSO", "LOCO_DEV"]
# Exhaustive: SVM+CoxPH only (fast; avoids EST/GBS overhead across 80k rows)
GM_MODEL_NAMES = ["SVM", "CoxPH"]
COACH_NAMES = ["EST", "SVM", "GBS", "CoxPH"]
COACH_HEURISTICS = {
    "EST":  {"n_estimators": 300, "max_depth": 12, "min_samples_split": 6,
             "min_samples_leaf": 4, "max_features": "sqrt"},
    "GBS":  {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "subsample": 0.8},
    "SVM":  {"alpha": 0.01},
    "CoxPH":{"alpha": 0.1},
}
N_STAB_BOOT = 50
LASSO_RANK_ALPHA = 1e-4
GBS_RANK_N_EST, GBS_RANK_LR, GBS_RANK_DEPTH, GBS_RANK_SUBSAMP = 100, 0.1, 3, 0.8

# ==========================================
# LOGGING
# ==========================================
LOG_FILE_PATH = OUTPUT_DIR / "8_mar_SC3_rad_PT768xCT8_325_LOCODEV_pc_2_log.md"


class _Tee:
    def __init__(self, *s):
        self.streams = s

    def write(self, d):
        for s in self.streams:
            s.write(d)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


_log_fh = open(LOG_FILE_PATH, "w", encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, _log_fh)
sys.stderr = _Tee(sys.__stderr__, _log_fh)

# ==========================================
# HELPERS
# ==========================================
def make_surv(e, t):
    return Surv.from_arrays(event=np.asarray(e, dtype=bool), time=np.asarray(t, dtype=float))


def safe_ci(y, r):
    try:
        return float(concordance_index_censored(y["event"], y["time"], r)[0])
    except Exception:
        return np.nan


def bootstrap_ci(y, r, n_boot=N_BOOT, seed=SEED):
    rng, n, cis = np.random.default_rng(seed), len(y), []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        c = safe_ci(y[idx], r[idx])
        if not np.isnan(c):
            cis.append(c)
    a = np.array(cis)
    if len(a) == 0:
        return np.nan, np.nan, np.nan, np.nan, 0
    return float(a.mean()), float(a.std()), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)), len(a)


def make_est(**p):
    return ExtraSurvivalTrees(**p, random_state=SEED, n_jobs=-1)


def make_svm(**p):
    return FastSurvivalSVM(**p, max_iter=1000, tol=1e-4, random_state=SEED)


def make_gbs(**p):
    return GradientBoostingSurvivalAnalysis(**p, random_state=SEED)


def make_coxph(**p):
    return CoxPHSurvivalAnalysis(**p)


def build_model_fn(n, p):
    return {"EST": make_est, "GBS": make_gbs, "SVM": make_svm, "CoxPH": make_coxph}[n](**p)


def model_oof(X, y, f_idx, fn):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    y_ev = y["event"].astype(int)
    oof_pred = np.zeros(len(X))
    oof_ok = np.zeros(len(X), dtype=bool)
    f_cs = []
    for tr_idx, v_idx in skf.split(X, y_ev):
        X_tr, X_v = X[tr_idx][:, f_idx], X[v_idx][:, f_idx]
        sc = StandardScaler()
        try:
            m = fn().fit(sc.fit_transform(X_tr), y[tr_idx])
            p = m.predict(sc.transform(X_v))
            oof_pred[v_idx], oof_ok[v_idx] = p, True
            c = safe_ci(y[v_idx], p)
            if not np.isnan(c):
                f_cs.append(c)
        except Exception:
            continue
    if oof_ok.sum() < len(X) * 0.5:
        return np.nan, np.nan
    return safe_ci(y[oof_ok], oof_pred[oof_ok]), float(np.std(f_cs)) if len(f_cs) > 1 else 1.0


def suggest_est(t):
    return {
        "n_estimators": t.suggest_int("n_e", 100, 600, step=50),
        "max_depth": t.suggest_int("m_d", 3, 15),
        "min_samples_split": t.suggest_int("m_s_s", 2, 20),
        "min_samples_leaf": t.suggest_int("m_s_l", 1, 15),
        "max_features": t.suggest_categorical("m_f", ["sqrt", "log2", 0.5]),
    }


def suggest_gbs(t):
    return {
        "n_estimators": t.suggest_int("n_e", 50, 400, step=50),
        "learning_rate": t.suggest_float("lr", 0.01, 0.3, log=True),
        "max_depth": t.suggest_int("m_d", 2, 8),
        "subsample": t.suggest_float("ss", 0.5, 1.0),
    }


def extract_params(n, t):
    if n == "EST":
        return {"n_estimators": t.params["n_e"], "max_depth": t.params["m_d"],
                "min_samples_split": t.params["m_s_s"], "min_samples_leaf": t.params["m_s_l"],
                "max_features": t.params["m_f"]}
    if n == "GBS":
        return {"n_estimators": t.params["n_e"], "learning_rate": t.params["lr"],
                "max_depth": t.params["m_d"], "subsample": t.params["ss"]}
    if n == "SVM":
        return {"alpha": t.params["alpha_svm"]}
    if n == "CoxPH":
        return {"alpha": t.params["alpha_cox"]}
    return {}


def cox_rank(X, y, n, a=LASSO_RANK_ALPHA):
    c = None
    for _a in [a, a / 10, a / 100]:
        try:
            m = CoxnetSurvivalAnalysis(alphas=[_a], l1_ratio=1., fit_baseline_model=True,
                                       normalize=False, max_iter=10000).fit(X, y)
            c = np.asarray(m.coef_).ravel()
            break
        except Exception:
            c = None
            continue
    if c is None:
        c = np.zeros(len(n))
    i = np.abs(c)
    o = np.argsort(i)[::-1]
    return [n[j] for j in o], i[o]


def univar_rank(X, y, n):
    s = [max(c, 1 - c) if not np.isnan(c := safe_ci(y, X[:, i])) else 0.5
         for i in range(X.shape[1])]
    o = np.argsort(np.array(s))[::-1]
    return [n[i] for i in o], np.array(s)[o]


def borda_combine(rl, n):
    pm = {f: [r for l in rl for r, p in enumerate(l) if p == f] for f in n}
    ap = {f: (np.mean(pm[f]) if pm[f] else len(n)) for f in n}
    sf = sorted(n, key=lambda f: (ap[f], f))
    return sf, np.array([ap[f] for f in sf])


def gbs_rank(X, y, n):
    m = GradientBoostingSurvivalAnalysis(
        n_estimators=GBS_RANK_N_EST, learning_rate=GBS_RANK_LR,
        max_depth=GBS_RANK_DEPTH, subsample=GBS_RANK_SUBSAMP, random_state=SEED
    ).fit(X, y)
    i = m.feature_importances_
    o = np.argsort(i)[::-1]
    return [n[j] for j in o], i[o]


def stab_lasso_rank(X, y, n, nb=N_STAB_BOOT, a=LASSO_RANK_ALPHA):
    rng, f, ok = np.random.default_rng(SEED), np.zeros(len(n)), 0
    for b in range(nb):
        idx = rng.integers(0, len(X), size=len(X))
        for _a in [a, a / 10, a / 100]:
            try:
                m = CoxnetSurvivalAnalysis(alphas=[_a], l1_ratio=1., fit_baseline_model=True,
                                           normalize=False, max_iter=5000).fit(X[idx], y[idx])
                f += (np.abs(np.asarray(m.coef_).ravel()) > 0).astype(float)
                ok += 1
                break
            except Exception:
                continue
    f /= max(ok, 1)
    o = np.argsort(f)[::-1]
    return [n[i] for i in o], f[o]


def loco_dev_rank(X_full, y_full, center_ids, feat_names):
    """
    Leave-One-Centre-Out ranking within dev set.
    Dev centres (CenterID): 1(52), 5(326), 6(51), 7(17), 8(9) = 455 patients.
    """
    unique_centres = np.unique(center_ids)
    n_feats = X_full.shape[1]
    ci_matrix = np.full((len(unique_centres), n_feats), np.nan)

    for fold_i, held_centre in enumerate(unique_centres):
        mask_val = center_ids == held_centre
        mask_tr  = ~mask_val

        if mask_val.sum() < 5:
            continue

        X_tr = X_full[mask_tr]
        y_tr = y_full[mask_tr]
        X_val = X_full[mask_val]
        y_val = y_full[mask_val]

        sc = StandardScaler()
        X_tr_sc  = sc.fit_transform(X_tr)
        X_val_sc = sc.transform(X_val)

        for fi in range(n_feats):
            c = safe_ci(y_val, X_val_sc[:, fi])
            ci_matrix[fold_i, fi] = max(c, 1 - c) if not np.isnan(c) else 0.5

    mean_ci = np.nanmean(ci_matrix, axis=0)
    mean_ci = np.where(np.isnan(mean_ci), 0.5, mean_ci)

    o = np.argsort(mean_ci)[::-1]
    feat_arr = np.array(feat_names)
    return list(feat_arr[o]), mean_ci[o]


# ==========================================
# 1. LOAD DATA
# ==========================================
print("=" * 70)
print("8_mar_SC3_rad_PT768xCT8_325_LOCODEV_pc_2")
print("TRUE EXHAUSTIVE: saves ALL Optuna trials x coaches (no Stage 2 scan)")
print("GM: SVM+CoxPH only | Coaches: EST/SVM/GBS/CoxPH per trial")
print("Fix 1: n_clin=1 seeds for every ranker/GM")
print("Fix 2: CLIN_RATIO_MAX=0.20")
print("Fix 3: CT_MIN=2, dense seeds at n_ct=4-8")
print(f"PT_MIN={PT_MIN} | CT_MIN={CT_MIN} | N_MAX={N_MAX} | TRIALS={TRIALS}")
print(f"Expected rows: {TRIALS} * {len(RANKING_METHODS)} * {len(GM_MODEL_NAMES)} * {len(COACH_NAMES)} = "
      f"{TRIALS * len(RANKING_METHODS) * len(GM_MODEL_NAMES) * len(COACH_NAMES)}")
print("Rad pool: PT768 (35 PT feat) + CT8_325 (39 CT feat)")
print("Dev = full training | External: CHUS + CHUP | Machine: PC")
print("=" * 70)

clinical = pd.read_csv(CLINICAL_FILE)
clinical = clinical.dropna(subset=["Relapse", "RFS"]).copy()
clin_dev = clinical[clinical["Cohort"] == "Dev"][
    ["PatientID", "CenterID", "Cohort", "Relapse", "RFS"] + CLINICAL_FEATURES
].copy()
clin_chus = clinical[clinical["CenterID"] == 3][
    ["PatientID", "Relapse", "RFS"] + CLINICAL_FEATURES
].copy()
clin_chup = clinical[clinical["CenterID"] == 2][
    ["PatientID", "Relapse", "RFS"] + CLINICAL_FEATURES
].copy()
print(f"Clinical dev: {len(clin_dev)} patients, {clin_dev['Relapse'].sum():.0f} events")
print(f"Clinical CHUS: {len(clin_chus)} patients, {clin_chus['Relapse'].sum():.0f} events")
print(f"Clinical CHUP: {len(clin_chup)} patients, {clin_chup['Relapse'].sum():.0f} events")

pt_feat_list = pd.read_csv(PT_FEATURES_FILE)["Feature"].tolist()
ct_feat_list_raw = pd.read_csv(CT_FEATURES_FILE)["Feature"].tolist()
pt_feat_set = set(pt_feat_list)
ct_feat_list = [f for f in ct_feat_list_raw if f not in pt_feat_set]
n_dropped = len(ct_feat_list_raw) - len(ct_feat_list)
if n_dropped > 0:
    dropped = [f for f in ct_feat_list_raw if f in pt_feat_set]
    print(f"  Dropped {n_dropped} CT features duplicated in PT list: {dropped}")
rad_features = pt_feat_list + ct_feat_list
print(f"\nRadiomics pool: PT768={len(pt_feat_list)} + CT8_325={len(ct_feat_list)} "
      f"(raw {len(ct_feat_list_raw)}, {n_dropped} deduped) = {len(rad_features)} features")

pt_dev = pd.read_csv(PT_DEV_FILE)
ct_dev = pd.read_csv(CT_DEV_FILE)
pt_ext = pd.read_csv(PT_EXT_FILE)
ct_ext = pd.read_csv(CT_EXT_FILE)

rad_dev = pt_dev[["PatientID"] + pt_feat_list].merge(
    ct_dev[["PatientID"] + ct_feat_list], on="PatientID", how="inner"
)
rad_ext = pt_ext[["PatientID"] + pt_feat_list].merge(
    ct_ext[["PatientID"] + ct_feat_list], on="PatientID", how="inner"
)
rad_chus = rad_ext[rad_ext["PatientID"].str.startswith("CHUS")]
rad_chup = rad_ext[rad_ext["PatientID"].str.startswith("CHUP")]

all_features = CLINICAL_FEATURES + rad_features
dev_merged = clin_dev.merge(rad_dev, on="PatientID", how="inner")
chus_merged = clin_chus.merge(rad_chus, on="PatientID", how="inner")
chup_merged = clin_chup.merge(rad_chup, on="PatientID", how="inner")

print(f"\nAfter merge:")
print(f"  Dev: {len(dev_merged)} pts, {dev_merged['Relapse'].sum():.0f} events")
print(f"  CHUS: {len(chus_merged)} pts, {chus_merged['Relapse'].sum():.0f} events")
print(f"  CHUP: {len(chup_merged)} pts, {chup_merged['Relapse'].sum():.0f} events")

for name, df in [("Dev", dev_merged), ("CHUS", chus_merged), ("CHUP", chup_merged)]:
    missing = df[all_features].isna().sum().sum()
    if missing > 0:
        print(f"WARNING: {missing} missing values in {name}")
    else:
        print(f"  {name}: 0 missing values")

X_train = dev_merged[all_features].values.astype(float)
y_train = make_surv(dev_merged["Relapse"], dev_merged["RFS"])
X_chus = chus_merged[all_features].values.astype(float)
y_chus = make_surv(chus_merged["Relapse"], chus_merged["RFS"])
X_chup = chup_merged[all_features].values.astype(float)
y_chup = make_surv(chup_merged["Relapse"], chup_merged["RFS"])

feat_idx_map = {f: i for i, f in enumerate(all_features)}
clin_feat = CLINICAL_FEATURES
pt_feat = pt_feat_list
ct_feat = ct_feat_list
clin_idx = [feat_idx_map[f] for f in clin_feat]
pt_idx   = [feat_idx_map[f] for f in pt_feat]
ct_idx   = [feat_idx_map[f] for f in ct_feat]
pt_feat_set_idx = set(pt_feat)
ct_feat_set_idx = set(ct_feat)

dev_center_ids = dev_merged["CenterID"].values

n_events_train = int(y_train["event"].sum())
print(f"\nEPV (train, {len(all_features)} feat): {n_events_train}/{len(all_features)} = {n_events_train/len(all_features):.2f}")
print(f"TRIALS per study: {TRIALS}")

unique_dev_centres = np.unique(dev_center_ids)
print(f"\nDev centres: {unique_dev_centres.tolist()}")
for c in unique_dev_centres:
    mask = dev_center_ids == c
    n_pts = mask.sum()
    n_ev = int(y_train["event"][mask].sum())
    print(f"  CenterID={c}: {n_pts} pts, {n_ev} events")

# ==========================================
# 2. PRE-COMPUTE RANKINGS — separate PT and CT
# ==========================================
print("\n--- Pre-computing feature rankings (PT and CT ranked separately) ---")
g_sc = StandardScaler()
X_train_sc = g_sc.fit_transform(X_train)

clin_feat_arr = np.array(clin_feat)
pt_feat_arr   = np.array(pt_feat)
ct_feat_arr   = np.array(ct_feat)

rankings = {}
for m in ["UNIVAR", "GBS"]:
    fn = univar_rank if m == "UNIVAR" else gbs_rank
    rankings[m] = {
        "clin": fn(X_train_sc[:, clin_idx], y_train, clin_feat_arr)[0],
        "pt":   fn(X_train_sc[:, pt_idx],   y_train, pt_feat_arr)[0],
        "ct":   fn(X_train_sc[:, ct_idx],   y_train, ct_feat_arr)[0],
    }
    print(f"  {m} done")

rankings["StabLASSO"] = {
    "clin": stab_lasso_rank(X_train_sc[:, clin_idx], y_train, clin_feat_arr)[0],
    "pt":   stab_lasso_rank(X_train_sc[:, pt_idx],   y_train, pt_feat_arr)[0],
    "ct":   stab_lasso_rank(X_train_sc[:, ct_idx],   y_train, ct_feat_arr)[0],
}
print("  StabLASSO done")

rankings["BORDA"] = {
    "clin": borda_combine(
        [cox_rank(X_train_sc[:, clin_idx], y_train, clin_feat_arr)[0],
         rankings["UNIVAR"]["clin"]], clin_feat
    )[0],
    "pt": borda_combine(
        [cox_rank(X_train_sc[:, pt_idx], y_train, pt_feat_arr)[0],
         rankings["UNIVAR"]["pt"]], pt_feat
    )[0],
    "ct": borda_combine(
        [cox_rank(X_train_sc[:, ct_idx], y_train, ct_feat_arr)[0],
         rankings["UNIVAR"]["ct"]], ct_feat
    )[0],
}
print("  BORDA done")

print("  Computing LOCO_DEV ranking (leave-one-dev-centre-out)...")
loco_pt_ranked, loco_pt_scores = loco_dev_rank(
    X_train_sc[:, pt_idx], y_train, dev_center_ids, pt_feat
)
loco_ct_ranked, loco_ct_scores = loco_dev_rank(
    X_train_sc[:, ct_idx], y_train, dev_center_ids, ct_feat
)
loco_clin_ranked = rankings["UNIVAR"]["clin"]
rankings["LOCO_DEV"] = {
    "clin": loco_clin_ranked,
    "pt":   loco_pt_ranked,
    "ct":   loco_ct_ranked,
}

PT817_FEATURES = {
    "GTVp_wavelet-HHL_glcm_ClusterProminence",
    "GTVp_gradient_glszm_ZoneEntropy",
    "GTVp_wavelet-HHH_firstorder_Mean",
    "GTVp_wavelet-HHH_glszm_SmallAreaLowGrayLevelEmphasis",
    "GTVp_wavelet-HLH_glrlm_ShortRunHighGrayLevelEmphasis",
    "GTVn_log-sigma-1-mm-3D_glszm_GrayLevelNonUniformity",
    "GTVn_square_glszm_GrayLevelNonUniformity",
    "GTVn_wavelet-HHH_glrlm_ShortRunHighGrayLevelEmphasis",
    "GTVn_wavelet-HLH_glszm_GrayLevelVariance",
}
print("  LOCO_DEV PT ranking (PT817 marked with *):")
for ri, f in enumerate(loco_pt_ranked, 1):
    marker = " *PT817*" if f in PT817_FEATURES else ""
    print(f"    {ri:2d}. {f}{marker}")

CT817_FEATURES = {
    "GTVp_wavelet-LHL_glcm_ClusterProminence",
    "GTVn_exponential_ngtdm_Coarseness",
}
print("  LOCO_DEV CT ranking (PT817-CT marked with *):")
for ri, f in enumerate(loco_ct_ranked, 1):
    marker = " *PT817-CT*" if f in CT817_FEATURES else ""
    print(f"    {ri:2d}. {f}{marker}")
print("  LOCO_DEV done")

# ==========================================
# 3. EXHAUSTIVE OPTUNA: ALL TRIALS x COACHES
# ==========================================
print(f"\n--- Exhaustive Optuna: {TRIALS} trials x {len(RANKING_METHODS)} rankers "
      f"x {len(GM_MODEL_NAMES)} GMs x {len(COACH_NAMES)} coaches ---")
print("(No Stage 2 scan — CHUS+CHUP evaluated for every trial directly)")

all_rows = []

for method in RANKING_METHODS:
    clin_r = rankings[method]["clin"]
    pt_r   = rankings[method]["pt"]
    ct_r   = rankings[method]["ct"]
    clin_ridx = [feat_idx_map[f] for f in clin_r]
    pt_ridx   = [feat_idx_map[f] for f in pt_r]
    ct_ridx   = [feat_idx_map[f] for f in ct_r]

    for gm_name in GM_MODEL_NAMES:
        print(f"\n  Ranker={method} | GM={gm_name} | Running {TRIALS} trials...")
        alpha_key = "alpha_svm" if gm_name == "SVM" else "alpha_cox"

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEED)
        )

        # Seed with n_clin=1 explicitly (Fix 1) and n_ct=4-8 range (Fix 3)
        if gm_name == "SVM":
            alpha_vals = [0.001, 0.005, 0.01, 0.05, 0.1]
        else:
            alpha_vals = [0.01, 0.05, 0.1, 0.5, 1.0]

        # Dense seeding: n_clin=1, sweep n_pt/n_ct combinations
        for alpha_val in alpha_vals:
            # Anchor: n_pt=8, n_ct=6 (best both>=0.65)
            study.enqueue_trial({"total_n": 15, "clin_ratio": 0.067, "n_pt": 8, alpha_key: alpha_val})
            # Vary n_ct: 4,5,7,8 with n_clin=1
            study.enqueue_trial({"total_n": 13, "clin_ratio": 0.077, "n_pt": 8,  alpha_key: alpha_val})  # n_ct=4
            study.enqueue_trial({"total_n": 14, "clin_ratio": 0.071, "n_pt": 8,  alpha_key: alpha_val})  # n_ct=5
            study.enqueue_trial({"total_n": 16, "clin_ratio": 0.063, "n_pt": 8,  alpha_key: alpha_val})  # n_ct=7
            study.enqueue_trial({"total_n": 17, "clin_ratio": 0.059, "n_pt": 8,  alpha_key: alpha_val})  # n_ct=8
            # Vary n_pt: 6,7,9,10 with n_ct=6
            study.enqueue_trial({"total_n": 13, "clin_ratio": 0.077, "n_pt": 6,  alpha_key: alpha_val})  # n_pt=6
            study.enqueue_trial({"total_n": 14, "clin_ratio": 0.071, "n_pt": 7,  alpha_key: alpha_val})  # n_pt=7
            study.enqueue_trial({"total_n": 16, "clin_ratio": 0.063, "n_pt": 9,  alpha_key: alpha_val})  # n_pt=9
            study.enqueue_trial({"total_n": 17, "clin_ratio": 0.059, "n_pt": 10, alpha_key: alpha_val})  # n_pt=10

        n_pt_pool = len(pt_ridx)
        n_ct_pool = len(ct_ridx)

        def make_objective(gm_name, clin_ridx, pt_ridx, ct_ridx, method):
            def objective(trial):
                N = trial.suggest_int("total_n", N_MIN, N_MAX)
                clin_ratio = trial.suggest_float("clin_ratio", CLIN_RATIO_MIN, CLIN_RATIO_MAX)
                n_clin = max(1, round(N * clin_ratio))
                n_clin = min(n_clin, len(clin_ridx), N - PT_MIN - CT_MIN)
                n_rad = N - n_clin

                n_pt_max = min(n_rad - CT_MIN, len(pt_ridx))
                n_pt_min = min(PT_MIN, n_pt_max)
                if n_pt_min > n_pt_max:
                    n_pt_min = n_pt_max
                n_pt = trial.suggest_int("n_pt", n_pt_min, n_pt_max)
                n_ct = n_rad - n_pt
                n_ct = max(CT_MIN, min(n_ct, len(ct_ridx)))
                n_pt = n_rad - n_ct
                n_pt = max(PT_MIN, min(n_pt, len(pt_ridx)))

                feat_idx = clin_ridx[:n_clin] + pt_ridx[:n_pt] + ct_ridx[:n_ct]

                if gm_name == "SVM":
                    alpha = trial.suggest_float("alpha_svm", 1e-4, 1., log=True)
                    model_fn = lambda: build_model_fn("SVM", {"alpha": alpha})
                elif gm_name == "CoxPH":
                    alpha = trial.suggest_float("alpha_cox", 1e-3, 10., log=True)
                    model_fn = lambda: build_model_fn("CoxPH", {"alpha": alpha})
                else:
                    raise ValueError(f"Unknown GM: {gm_name}")

                oof_ci, fold_std = model_oof(X_train, y_train, feat_idx, model_fn)
                if np.isnan(oof_ci):
                    oof_ci, fold_std = 0.5, 1.0
                score = W_PERF * oof_ci + W_STAB * max(0., 1. - fold_std / STD_THRESHOLD)

                trial.set_user_attr("N", N)
                trial.set_user_attr("n_clin", n_clin)
                trial.set_user_attr("n_pt", n_pt)
                trial.set_user_attr("n_ct", n_ct)
                trial.set_user_attr("oof_ci", oof_ci)
                trial.set_user_attr("fold_std", fold_std)
                trial.set_user_attr("clin_features", [all_features[i] for i in clin_ridx[:n_clin]])
                trial.set_user_attr("pt_features",   [all_features[i] for i in pt_ridx[:n_pt]])
                trial.set_user_attr("ct_features",   [all_features[i] for i in ct_ridx[:n_ct]])
                return score

            return objective

        study.optimize(
            make_objective(gm_name, clin_ridx, pt_ridx, ct_ridx, method),
            n_trials=TRIALS,
            show_progress_bar=False
        )

        # Evaluate ALL trials x ALL coaches
        trials_evaluated = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        print(f"    Evaluating {len(trials_evaluated)} completed trials x {len(COACH_NAMES)} coaches...")

        for trial_i, trial in enumerate(trials_evaluated):
            n_clin = trial.user_attrs["n_clin"]
            n_pt   = trial.user_attrs["n_pt"]
            n_ct   = trial.user_attrs["n_ct"]
            N      = trial.user_attrs["N"]
            oof_ci = trial.user_attrs["oof_ci"]
            fold_std = trial.user_attrs["fold_std"]
            clin_feats = trial.user_attrs["clin_features"]
            pt_feats   = trial.user_attrs["pt_features"]
            ct_feats   = trial.user_attrs["ct_features"]

            feat_names = clin_feats + pt_feats + ct_feats
            col_idx    = [feat_idx_map[f] for f in feat_names]

            if gm_name == "SVM":
                gm_params = {"alpha": trial.params["alpha_svm"]}
            else:
                gm_params = {"alpha": trial.params["alpha_cox"]}

            for coach_name in COACH_NAMES:
                coach_params = gm_params if coach_name == gm_name else COACH_HEURISTICS[coach_name]
                coach_fn = lambda cn=coach_name, cp=coach_params: build_model_fn(cn, cp)

                sc_final = StandardScaler()
                X_tr_sc   = sc_final.fit_transform(X_train[:, col_idx])
                X_chus_sc = sc_final.transform(X_chus[:, col_idx])
                X_chup_sc = sc_final.transform(X_chup[:, col_idx])

                m = coach_fn()
                try:
                    m.fit(X_tr_sc, y_train)
                    risk_chus = m.predict(X_chus_sc)
                    risk_chup = m.predict(X_chup_sc)
                except Exception:
                    risk_chus = np.zeros(len(X_chus))
                    risk_chup = np.zeros(len(X_chup))

                ci_chus = safe_ci(y_chus, risk_chus)
                ci_chup = safe_ci(y_chup, risk_chup)

                row = {
                    "trial_no": trial.number,
                    "ranker": method,
                    "gm_model": gm_name,
                    "coach_model": coach_name,
                    "N": N,
                    "n_clin": n_clin,
                    "n_pt": n_pt,
                    "n_ct": n_ct,
                    "n_rad": n_pt + n_ct,
                    "oof_ci": oof_ci,
                    "fold_std": fold_std,
                    "ci_chus": ci_chus,
                    "ci_chup": ci_chup,
                    "gm_params": str(gm_params),
                    "coach_params": str(coach_params),
                    "clin_names": "; ".join(clin_feats),
                    "pt_names": "; ".join(pt_feats),
                    "ct_names": "; ".join(ct_feats),
                }
                all_rows.append(row)

            if (trial_i + 1) % 200 == 0:
                print(f"      {trial_i + 1}/{len(trials_evaluated)} trials evaluated")

        bt = study.best_trial
        print(f"    Best trial: N={bt.user_attrs['N']} "
              f"(PT={bt.user_attrs['n_pt']} CT={bt.user_attrs['n_ct']}) "
              f"| OOF={bt.user_attrs['oof_ci']:.4f}")

        # Save checkpoint after each ranker/GM
        checkpoint_df = pd.DataFrame(all_rows)
        checkpoint_path = OUTPUT_DIR / "8_mar_SC3_rad_PT768xCT8_325_LOCODEV_pc_2_checkpoint.csv"
        checkpoint_df.to_csv(checkpoint_path, index=False)
        print(f"    Checkpoint saved: {len(checkpoint_df)} rows")

# ==========================================
# 4. BOOTSTRAP TOP CANDIDATES + SAVE
# ==========================================
print("\n--- Bootstrap CI for top candidates (both>=0.65) ---")
all_df = pd.DataFrame(all_rows)
all_df.insert(0, "No", np.arange(1, len(all_df) + 1))

# Filter candidates where both CHUS>=0.65 and CHUP>=0.65
dual_candidates = all_df[(all_df["ci_chus"] >= 0.65) & (all_df["ci_chup"] >= 0.65)].copy()
print(f"Dual candidates (both>=0.65): {len(dual_candidates)}")

# Also top 20 by trio mean
all_df["trio_mean"] = (all_df["oof_ci"] + all_df["ci_chus"] + all_df["ci_chup"]) / 3
top20 = all_df.nlargest(20, "trio_mean")
boot_targets = pd.concat([dual_candidates, top20]).drop_duplicates(subset=["No"])
print(f"Bootstrap targets: {len(boot_targets)} rows")

boot_results = []
for _, row in boot_targets.iterrows():
    feat_names = (row["clin_names"].split("; ") if row["clin_names"] else []) + \
                 (row["pt_names"].split("; ") if row["pt_names"] else []) + \
                 (row["ct_names"].split("; ") if row["ct_names"] else [])
    col_idx = [feat_idx_map[f] for f in feat_names]

    gm_name = row["gm_model"]
    coach_name = row["coach_model"]
    coach_params_str = row["coach_params"]
    # Reconstruct coach params from COACH_HEURISTICS or gm_params
    if coach_name == gm_name:
        gm_params_str = row["gm_params"]
        import ast
        try:
            coach_params = ast.literal_eval(gm_params_str)
        except Exception:
            coach_params = COACH_HEURISTICS[coach_name]
    else:
        coach_params = COACH_HEURISTICS[coach_name]

    coach_fn = lambda cn=coach_name, cp=coach_params: build_model_fn(cn, cp)

    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_train[:, col_idx])
    X_chus_sc = sc.transform(X_chus[:, col_idx])
    X_chup_sc = sc.transform(X_chup[:, col_idx])

    m = coach_fn()
    try:
        m.fit(X_tr_sc, y_train)
        risk_chus = m.predict(X_chus_sc)
        risk_chup = m.predict(X_chup_sc)
    except Exception:
        risk_chus = np.zeros(len(X_chus))
        risk_chup = np.zeros(len(X_chup))

    _, _, ci_lo_chus, ci_hi_chus, n_valid = bootstrap_ci(y_chus, risk_chus)
    _, _, ci_lo_chup, ci_hi_chup, _ = bootstrap_ci(y_chup, risk_chup)
    boot_results.append({
        "No": row["No"],
        "boot_ci_chus_lo": ci_lo_chus, "boot_ci_chus_hi": ci_hi_chus,
        "boot_ci_chup_lo": ci_lo_chup, "boot_ci_chup_hi": ci_hi_chup,
        "boot_n_valid": n_valid,
    })

if boot_results:
    boot_df = pd.DataFrame(boot_results).set_index("No")
    for col in boot_df.columns:
        all_df.loc[all_df["No"].isin(boot_df.index), col] = all_df.loc[
            all_df["No"].isin(boot_df.index), "No"].map(boot_df[col])

# ==========================================
# 5. SAVE OUTPUT
# ==========================================
print("\n--- Saving output ---")

out_path = OUTPUT_DIR / "8_mar_SC3_rad_PT768xCT8_325_LOCODEV_pc_2_all_results.csv"
all_df.to_csv(out_path, index=False)
print(f"Saved: {out_path.name} ({len(all_df)} rows)")

top20_out = all_df.nlargest(20, "trio_mean")[
    ["No", "ranker", "gm_model", "coach_model", "N", "n_clin", "n_pt", "n_ct",
     "oof_ci", "ci_chus", "ci_chup", "trio_mean",
     "clin_names", "pt_names", "ct_names"]
]
print("\nTop 20 by Trio mean (OOF + CHUS + CHUP):")
print(top20_out.to_string(index=False))

top20_path = OUTPUT_DIR / "8_mar_SC3_rad_PT768xCT8_325_LOCODEV_pc_2_top20.csv"
top20_out.to_csv(top20_path, index=False)
print(f"\nSaved top20: {top20_path.name}")

print(f"\nDual ok (both>=0.70): {((all_df['ci_chus']>=0.70) & (all_df['ci_chup']>=0.70)).sum()}")
print(f"Dual ok (both>=0.65): {((all_df['ci_chus']>=0.65) & (all_df['ci_chup']>=0.65)).sum()}")
print(f"Max CHUS: {all_df['ci_chus'].max():.4f}")
print(f"Max CHUP: {all_df['ci_chup'].max():.4f}")

print("\n8_mar_SC3_rad_PT768xCT8_325_LOCODEV_pc_2 finished.")
