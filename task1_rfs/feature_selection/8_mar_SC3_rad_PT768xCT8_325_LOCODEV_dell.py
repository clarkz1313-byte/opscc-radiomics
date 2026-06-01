"""
8_mar_SC3_rad_PT768xCT8_325_LOCODEV_dell.py

Traditional SC3 with PT-forced modality split + LOCO_DEV as 5th ranker.
Dell version — two fixes vs Thinkpad:

Fix 1: RATIO_GRID_FULL = [0.0] + RATIO_GRID
  - When ratio == 0.0, hardcode target_nc = 1 (guarantees n_clin=1 is evaluated)
  - Motivation: deepPT_pc 64k rows best near-dual at n_clin=1 but OOF never selects it

Fix 2: CLIN_RATIO_MAX = 0.20 (down from 0.50)
  - Concentrates Optuna in low-clin territory

Machine: Dell | TRIALS=1000 | N_MIN=6, N_MAX=20 | PT_MIN=4, CT_MIN=1
Ranking methods: UNIVAR, BORDA, GBS, StabLASSO, LOCO_DEV (5 rankers)
GM models: EST, SVM, GBS, CoxPH (4 traditional)

Rad pool: PT768 (35 PT feat) + CT8_325 (39 CT feat) = 74 features (post-dedup)
Clinical: 3 features (Age, Gender_Male, Treatment_CRT)
Config: N_FOLDS=5, SEED=42, TRIALS=1000

Outputs: Mar_2026/8_mar_SC3_rad_PT768xCT8_325_LOCODEV_dell_outputs/

Usage:
  python Mar_2026/8_mar_SC3_rad_PT768xCT8_325_LOCODEV_dell.py
  SC3_TRIALS=800 python Mar_2026/8_mar_SC3_rad_PT768xCT8_325_LOCODEV_dell.py
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
from sklearn.inspection import permutation_importance
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.ensemble import ExtraSurvivalTrees, GradientBoostingSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

# ==========================================
# CONFIG
# ==========================================
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "8_mar_SC3_rad_PT768xCT8_325_LOCODEV_dell_outputs"
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
# Fix 2: CLIN_RATIO_MAX reduced to 0.20 (was 0.50)
CLIN_RATIO_MIN, CLIN_RATIO_MAX = 0.05, 0.20
PT_MIN = 4
CT_MIN = 1
TRIALS = int(os.getenv("SC3_TRIALS", "1000"))
RATIO_GRID = [round(r * 0.05, 2) for r in range(1, 13)]
# Fix 1: prepend 0.0 to guarantee n_clin=1 is evaluated
RATIO_GRID_FULL = [0.0] + RATIO_GRID

RANKING_METHODS = ["UNIVAR", "BORDA", "GBS", "StabLASSO", "LOCO_DEV"]
GM_MODEL_NAMES = ["EST", "SVM", "GBS", "CoxPH"]
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
PI_N_REPEATS = 10

# ==========================================
# LOGGING
# ==========================================
LOG_FILE_PATH = OUTPUT_DIR / "8_mar_SC3_rad_PT768xCT8_325_LOCODEV_dell_log.md"


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

    For each dev centre c:
      - Train CoxPH on all other centres combined.
      - Predict risk on centre c.
      - Record per-feature univariate C-index on held-out centre c.

    Final rank: features sorted by mean C-index across the 5 LOCO folds.
    Uses only dev data -- no external data touched.

    Dev centres (CenterID): 1(52), 5(326), 6(51), 7(17), 8(9) = 455 patients.
    Centre 7 and 8 are small; univar C-index may be unstable there but averaged out.
    """
    unique_centres = np.unique(center_ids)
    n_feats = X_full.shape[1]
    ci_matrix = np.full((len(unique_centres), n_feats), np.nan)

    for fold_i, held_centre in enumerate(unique_centres):
        mask_val = center_ids == held_centre
        mask_tr  = ~mask_val

        if mask_val.sum() < 5:
            # Too small to compute reliable C-index; skip this fold
            continue

        X_tr = X_full[mask_tr]
        y_tr = y_full[mask_tr]
        X_val = X_full[mask_val]
        y_val = y_full[mask_val]

        sc = StandardScaler()
        X_tr_sc  = sc.fit_transform(X_tr)
        X_val_sc = sc.transform(X_val)

        # Univariate C-index per feature on held-out centre
        for fi in range(n_feats):
            c = safe_ci(y_val, X_val_sc[:, fi])
            ci_matrix[fold_i, fi] = max(c, 1 - c) if not np.isnan(c) else 0.5

    # Mean C-index across LOCO folds (ignoring nan rows from skipped centres)
    mean_ci = np.nanmean(ci_matrix, axis=0)
    mean_ci = np.where(np.isnan(mean_ci), 0.5, mean_ci)

    o = np.argsort(mean_ci)[::-1]
    feat_arr = np.array(feat_names)
    return list(feat_arr[o]), mean_ci[o]


def get_least_important_feature(X, y, model, features_to_check, all_lineup):
    X_df = pd.DataFrame(X, columns=all_lineup)
    pi_result = permutation_importance(
        model, X_df, y, n_repeats=PI_N_REPEATS,
        random_state=SEED, n_jobs=-1, scoring=surv_ci_scorer
    )
    importances = pd.Series(pi_result.importances_mean, index=all_lineup)
    return importances[features_to_check].idxmin()


def surv_ci_scorer(estimator, X, y):
    try:
        return safe_ci(y, estimator.predict(X))
    except Exception:
        return 0.5


def next_available_feature(ranked_features, current_features):
    current_set = set(current_features)
    for feat in ranked_features:
        if feat not in current_set:
            return feat
    return None


# ==========================================
# 1. LOAD DATA
# ==========================================
print("=" * 70)
print("8_mar_SC3_rad_PT768xCT8_325_LOCODEV_dell")
print("PT-forced modality split + LOCO_DEV as 5th ranker")
print("Fix 1: RATIO_GRID_FULL includes 0.0 (n_clin=1 guaranteed)")
print("Fix 2: CLIN_RATIO_MAX=0.20 (concentrates Optuna in low-clin territory)")
print(f"PT_MIN={PT_MIN} | CT_MIN={CT_MIN} | N_MAX={N_MAX} | TRIALS={TRIALS}")
print("Rad pool: PT768 (35 PT feat) + CT8_325 (39 CT feat)")
print("Dev = full training | External: CHUS + CHUP | Machine: Dell")
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
print(f"PT features: {len(pt_feat_list)} | CT features: {len(ct_feat_list)}")

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
print(f"  Total features: {len(all_features)} ({len(CLINICAL_FEATURES)} clin + {len(rad_features)} rad)")

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
rad_feat = rad_features
clin_idx = [feat_idx_map[f] for f in clin_feat]
pt_idx   = [feat_idx_map[f] for f in pt_feat]
ct_idx   = [feat_idx_map[f] for f in ct_feat]
rad_idx  = [feat_idx_map[f] for f in rad_feat]
pt_feat_set_idx = set(pt_feat)
ct_feat_set_idx = set(ct_feat)

# Store CenterID array aligned with dev_merged for LOCO_DEV
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

# LOCO_DEV ranking: computed separately for PT and CT
print("  Computing LOCO_DEV ranking (leave-one-dev-centre-out)...")

loco_pt_ranked, loco_pt_scores = loco_dev_rank(
    X_train_sc[:, pt_idx], y_train, dev_center_ids, pt_feat
)
loco_ct_ranked, loco_ct_scores = loco_dev_rank(
    X_train_sc[:, ct_idx], y_train, dev_center_ids, ct_feat
)
# Use UNIVAR for clinical (small count, LOCO unstable)
loco_clin_ranked = rankings["UNIVAR"]["clin"]

rankings["LOCO_DEV"] = {
    "clin": loco_clin_ranked,
    "pt":   loco_pt_ranked,
    "ct":   loco_ct_ranked,
}

# Print LOCO_DEV PT ranking with PT817 markers
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
print("  LOCO_DEV PT ranking (top 35, PT817 marked with *):")
for ri, f in enumerate(loco_pt_ranked, 1):
    marker = " *PT817*" if f in PT817_FEATURES else ""
    print(f"    {ri:2d}. {f}{marker}")

CT817_FEATURES = {
    "GTVp_wavelet-LHL_glcm_ClusterProminence",
    "GTVn_exponential_ngtdm_Coarseness",
}
print("  LOCO_DEV CT ranking (all 39, PT817-CT marked with *):")
for ri, f in enumerate(loco_ct_ranked, 1):
    marker = " *PT817-CT*" if f in CT817_FEATURES else ""
    print(f"    {ri:2d}. {f}{marker}")
print("  LOCO_DEV done")

# ==========================================
# 3. STAGE 1: GM OPTUNA SEARCH (PT-forced split)
# ==========================================
print(f"\n--- Stage 1: Optuna GM search ({TRIALS} trials each, PT-forced split) ---")


def make_gm_objective(gm_name, clin_ridx, pt_ridx, ct_ridx):
    n_pt_pool = len(pt_ridx)
    n_ct_pool = len(ct_ridx)

    def objective(trial):
        N = trial.suggest_int("total_n", N_MIN, N_MAX)
        clin_ratio = trial.suggest_float("clin_ratio", CLIN_RATIO_MIN, CLIN_RATIO_MAX)
        n_clin = max(1, round(N * clin_ratio))
        n_clin = min(n_clin, len(clin_ridx), N - PT_MIN - CT_MIN)
        n_rad = N - n_clin

        # PT/CT split: sample n_pt independently; n_ct = remainder
        n_pt_max = min(n_rad - CT_MIN, n_pt_pool)
        n_pt_min = min(PT_MIN, n_pt_max)
        if n_pt_min > n_pt_max:
            n_pt_min = n_pt_max
        n_pt = trial.suggest_int("n_pt", n_pt_min, n_pt_max)
        n_ct = n_rad - n_pt
        n_ct = max(CT_MIN, min(n_ct, n_ct_pool))
        n_pt = n_rad - n_ct
        n_pt = max(PT_MIN, min(n_pt, n_pt_pool))

        feat_idx = clin_ridx[:n_clin] + pt_ridx[:n_pt] + ct_ridx[:n_ct]

        if gm_name == "EST":
            model_fn = lambda: build_model_fn("EST", suggest_est(trial))
        elif gm_name == "GBS":
            model_fn = lambda: build_model_fn("GBS", suggest_gbs(trial))
        elif gm_name == "SVM":
            model_fn = lambda: build_model_fn("SVM", {"alpha": trial.suggest_float("alpha_svm", 1e-4, 1., log=True)})
        elif gm_name == "CoxPH":
            model_fn = lambda: build_model_fn("CoxPH", {"alpha": trial.suggest_float("alpha_cox", 1e-3, 10., log=True)})
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
        trial.set_user_attr("features", [all_features[i] for i in feat_idx])
        return score

    return objective


stage1_results = {}
for method in RANKING_METHODS:
    clin_r = rankings[method]["clin"]
    pt_r   = rankings[method]["pt"]
    ct_r   = rankings[method]["ct"]
    clin_ridx = [feat_idx_map[f] for f in clin_r]
    pt_ridx   = [feat_idx_map[f] for f in pt_r]
    ct_ridx   = [feat_idx_map[f] for f in ct_r]

    for gm_name in GM_MODEL_NAMES:
        print(f"  Optuna: {method}/{gm_name} ...", end=" ", flush=True)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEED)
        )

        # Seed enqueues targeting deep PT regions per ranker.
        # LOCO_DEV PT817 ranks: 1,4,7,12,21,23,24,29,31
        # GBS PT817 ranks: 1,2,4,6,9,12,16,18,34 -> n_pt>=16 covers 7/9
        # UNIVAR PT817 ranks: 1,2,5,8,11,19,30,31,32 -> n_pt>=19 covers 7/9
        # StabLASSO PT817 ranks: 5,6,14,20,22,27,29,30,34 -> n_pt>=30 covers 8/9
        if method == "LOCO_DEV":
            if gm_name in ("SVM", "CoxPH"):
                alpha_key = "alpha_svm" if gm_name == "SVM" else "alpha_cox"
                for alpha_val in ([0.001, 0.01, 0.1] if gm_name == "SVM" else [0.01, 0.1, 1.0]):
                    study.enqueue_trial({"total_n": 14, "clin_ratio": 0.07, "n_pt": 12, alpha_key: alpha_val})
                    study.enqueue_trial({"total_n": 20, "clin_ratio": 0.05, "n_pt": 18, alpha_key: alpha_val})
                study.enqueue_trial({"total_n": 24, "clin_ratio": 0.05, "n_pt": 22,
                                     alpha_key: (0.01 if gm_name == "SVM" else 0.1)})
            elif gm_name == "GBS":
                study.enqueue_trial({"total_n": 14, "clin_ratio": 0.07, "n_pt": 12,
                                     "n_e": 100, "lr": 0.1, "m_d": 3, "ss": 0.8})
                study.enqueue_trial({"total_n": 20, "clin_ratio": 0.05, "n_pt": 18,
                                     "n_e": 100, "lr": 0.1, "m_d": 3, "ss": 0.8})
                study.enqueue_trial({"total_n": 14, "clin_ratio": 0.07, "n_pt": 12,
                                     "n_e": 200, "lr": 0.05, "m_d": 4, "ss": 0.9})
            elif gm_name == "EST":
                study.enqueue_trial({"total_n": 14, "clin_ratio": 0.07, "n_pt": 12,
                                     "n_e": 300, "m_d": 12, "m_s_s": 6, "m_s_l": 4, "m_f": "sqrt"})
                study.enqueue_trial({"total_n": 20, "clin_ratio": 0.05, "n_pt": 18,
                                     "n_e": 300, "m_d": 12, "m_s_s": 6, "m_s_l": 4, "m_f": "sqrt"})
        elif method == "GBS":
            if gm_name in ("SVM", "CoxPH"):
                alpha_key = "alpha_svm" if gm_name == "SVM" else "alpha_cox"
                alpha_val = 0.01 if gm_name == "SVM" else 0.1
                study.enqueue_trial({"total_n": 18, "clin_ratio": 0.06, "n_pt": 16, alpha_key: alpha_val})
                study.enqueue_trial({"total_n": 12, "clin_ratio": 0.08, "n_pt": 10, alpha_key: alpha_val})
                study.enqueue_trial({"total_n": 10, "clin_ratio": 0.10, "n_pt": 8,  alpha_key: alpha_val})
            elif gm_name == "GBS":
                study.enqueue_trial({"total_n": 18, "clin_ratio": 0.06, "n_pt": 16,
                                     "n_e": 100, "lr": 0.1, "m_d": 3, "ss": 0.8})
                study.enqueue_trial({"total_n": 12, "clin_ratio": 0.08, "n_pt": 10,
                                     "n_e": 100, "lr": 0.1, "m_d": 3, "ss": 0.8})
            elif gm_name == "EST":
                study.enqueue_trial({"total_n": 18, "clin_ratio": 0.06, "n_pt": 16,
                                     "n_e": 300, "m_d": 12, "m_s_s": 6, "m_s_l": 4, "m_f": "sqrt"})
                study.enqueue_trial({"total_n": 12, "clin_ratio": 0.08, "n_pt": 10,
                                     "n_e": 300, "m_d": 12, "m_s_s": 6, "m_s_l": 4, "m_f": "sqrt"})
        elif method == "UNIVAR":
            if gm_name in ("SVM", "CoxPH"):
                alpha_key = "alpha_svm" if gm_name == "SVM" else "alpha_cox"
                alpha_val = 0.01 if gm_name == "SVM" else 0.1
                study.enqueue_trial({"total_n": 20, "clin_ratio": 0.05, "n_pt": 18, alpha_key: alpha_val})
                study.enqueue_trial({"total_n": 15, "clin_ratio": 0.07, "n_pt": 13, alpha_key: alpha_val})
                study.enqueue_trial({"total_n": 11, "clin_ratio": 0.09, "n_pt": 9,  alpha_key: alpha_val})
            elif gm_name == "GBS":
                study.enqueue_trial({"total_n": 20, "clin_ratio": 0.05, "n_pt": 18,
                                     "n_e": 100, "lr": 0.1, "m_d": 3, "ss": 0.8})
                study.enqueue_trial({"total_n": 15, "clin_ratio": 0.07, "n_pt": 13,
                                     "n_e": 100, "lr": 0.1, "m_d": 3, "ss": 0.8})
            elif gm_name == "EST":
                study.enqueue_trial({"total_n": 20, "clin_ratio": 0.05, "n_pt": 18,
                                     "n_e": 300, "m_d": 12, "m_s_s": 6, "m_s_l": 4, "m_f": "sqrt"})
        elif method == "StabLASSO":
            if gm_name in ("SVM", "CoxPH"):
                alpha_key = "alpha_svm" if gm_name == "SVM" else "alpha_cox"
                alpha_val = 0.01 if gm_name == "SVM" else 0.1
                study.enqueue_trial({"total_n": 20, "clin_ratio": 0.05, "n_pt": 18, alpha_key: alpha_val})
                study.enqueue_trial({"total_n": 15, "clin_ratio": 0.07, "n_pt": 13, alpha_key: alpha_val})
            elif gm_name == "GBS":
                study.enqueue_trial({"total_n": 20, "clin_ratio": 0.05, "n_pt": 18,
                                     "n_e": 100, "lr": 0.1, "m_d": 3, "ss": 0.8})
            elif gm_name == "EST":
                study.enqueue_trial({"total_n": 20, "clin_ratio": 0.05, "n_pt": 18,
                                     "n_e": 300, "m_d": 12, "m_s_s": 6, "m_s_l": 4, "m_f": "sqrt"})
        elif method == "BORDA":
            if gm_name in ("SVM", "CoxPH"):
                alpha_key = "alpha_svm" if gm_name == "SVM" else "alpha_cox"
                alpha_val = 1e-3 if gm_name == "SVM" else 0.1
                study.enqueue_trial({"total_n": 18, "clin_ratio": 0.06, "n_pt": 16, alpha_key: alpha_val})
                study.enqueue_trial({"total_n": 10, "clin_ratio": 0.10, "n_pt": 8,  alpha_key: alpha_val})
            elif gm_name == "GBS":
                study.enqueue_trial({"total_n": 18, "clin_ratio": 0.06, "n_pt": 16,
                                     "n_e": 100, "lr": 0.1, "m_d": 3, "ss": 0.8})
            elif gm_name == "EST":
                study.enqueue_trial({"total_n": 18, "clin_ratio": 0.06, "n_pt": 16,
                                     "n_e": 300, "m_d": 12, "m_s_s": 6, "m_s_l": 4, "m_f": "sqrt"})

        study.optimize(
            make_gm_objective(gm_name, clin_ridx, pt_ridx, ct_ridx),
            n_trials=TRIALS,
            show_progress_bar=False
        )
        bt = study.best_trial
        bt_feats = bt.user_attrs["features"]
        bt_n_pt = bt.user_attrs.get("n_pt", sum(1 for f in bt_feats if f in pt_feat_set_idx))
        bt_n_ct = bt.user_attrs.get("n_ct", sum(1 for f in bt_feats if f in ct_feat_set_idx))
        stage1_results[(method, gm_name)] = {
            "N": bt.user_attrs["N"],
            "gm_params": extract_params(gm_name, bt),
            "clin_r": clin_r,
            "pt_r": pt_r,
            "ct_r": ct_r,
            "anchor_features": bt_feats,
            "anchor_n_pt": bt_n_pt,
            "anchor_n_ct": bt_n_ct,
        }
        print(f"N={bt.user_attrs['N']} (PT={bt_n_pt} CT={bt_n_ct}) | OOF={bt.user_attrs['oof_ci']:.4f}")

# ==========================================
# 4. STAGE 2: SUBSTITUTION SCAN (PT/CT split aware, RATIO_GRID_FULL)
# ==========================================
print("\n--- Stage 2: Substitution scan across clin_ratio grid (Fix 1: includes ratio=0.0) ---")

all_scan_rows = []
main_loop_feature_store = {}

for method in RANKING_METHODS:
    for gm_name in GM_MODEL_NAMES:
        s1 = stage1_results[(method, gm_name)]
        N_star = s1["N"]
        gm_params = s1["gm_params"]
        clin_r = s1["clin_r"]
        pt_r   = s1["pt_r"]
        ct_r   = s1["ct_r"]
        anchor_feats = s1["anchor_features"]

        for ratio in RATIO_GRID_FULL:
            current_clin_feats = [f for f in anchor_feats if f in set(clin_feat)]
            current_pt_feats   = [f for f in anchor_feats if f in pt_feat_set_idx]
            current_ct_feats   = [f for f in anchor_feats if f in ct_feat_set_idx]

            # Fix 1: when ratio == 0.0, hardcode target_nc = 1
            if ratio == 0.0:
                target_nc = 1
            else:
                target_nc = min(max(1, round(N_star * ratio)), len(clin_r), N_star - PT_MIN - CT_MIN)
            temp_model_fn = lambda: build_model_fn(gm_name, gm_params)

            while len(current_clin_feats) < target_nc:
                if len(current_pt_feats) + len(current_ct_feats) <= PT_MIN + CT_MIN:
                    break
                next_clin = next_available_feature(clin_r, current_clin_feats)
                if next_clin is None:
                    break
                current_clin_feats.append(next_clin)
                lineup = current_clin_feats + current_pt_feats + current_ct_feats
                col_idx_temp = [feat_idx_map[f] for f in lineup]
                X_temp_sc = StandardScaler().fit_transform(X_train[:, col_idx_temp])
                temp_model = temp_model_fn().fit(X_temp_sc, y_train)
                rad_candidate = []
                if len(current_pt_feats) > PT_MIN:
                    rad_candidate += current_pt_feats
                if len(current_ct_feats) > CT_MIN:
                    rad_candidate += current_ct_feats
                if not rad_candidate:
                    break
                lvi = get_least_important_feature(X_temp_sc, y_train, temp_model, rad_candidate, lineup)
                if lvi in current_pt_feats:
                    current_pt_feats.remove(lvi)
                else:
                    current_ct_feats.remove(lvi)

            while len(current_clin_feats) > target_nc:
                if len(current_clin_feats) == 1:
                    break
                lineup = current_clin_feats + current_pt_feats + current_ct_feats
                col_idx_temp = [feat_idx_map[f] for f in lineup]
                X_temp_sc = StandardScaler().fit_transform(X_train[:, col_idx_temp])
                temp_model = temp_model_fn().fit(X_temp_sc, y_train)
                lvi_clin = get_least_important_feature(X_temp_sc, y_train, temp_model,
                                                        current_clin_feats, lineup)
                current_clin_feats.remove(lvi_clin)
                pt_frac = len(current_pt_feats) / max(1, len(current_pt_feats) + len(current_ct_feats))
                if pt_frac < 0.5 or len(current_ct_feats) >= len(current_pt_feats):
                    next_pt = next_available_feature(pt_r, current_pt_feats)
                    if next_pt is not None:
                        current_pt_feats.append(next_pt)
                        continue
                next_ct = next_available_feature(ct_r, current_ct_feats)
                if next_ct is not None:
                    current_ct_feats.append(next_ct)

            feat_names = current_clin_feats + current_pt_feats + current_ct_feats
            col_idx = [feat_idx_map[f] for f in feat_names]
            main_loop_feature_store[(method, gm_name, ratio)] = list(feat_names)

            for coach_name in COACH_NAMES:
                coach_params = gm_params if coach_name == gm_name else COACH_HEURISTICS[coach_name]
                coach_fn = lambda: build_model_fn(coach_name, coach_params)

                oof_ci, fstd = model_oof(X_train, y_train, col_idx, coach_fn)

                sc_final = StandardScaler()
                X_tr_sc = sc_final.fit_transform(X_train[:, col_idx])
                X_ch_sc_chus = sc_final.transform(X_chus[:, col_idx])
                X_ch_sc_chup = sc_final.transform(X_chup[:, col_idx])

                m = coach_fn()
                try:
                    m.fit(X_tr_sc, y_train)
                    risk_tr = m.predict(X_tr_sc)
                    risk_chus = m.predict(X_ch_sc_chus)
                    risk_chup = m.predict(X_ch_sc_chup)
                except Exception:
                    risk_tr = np.zeros(len(X_tr_sc))
                    risk_chus = np.zeros(len(X_chus))
                    risk_chup = np.zeros(len(X_chup))

                ci_chus = safe_ci(y_chus, risk_chus)
                ci_chup = safe_ci(y_chup, risk_chup)

                row = {
                    "ranker": method,
                    "gm_model": gm_name,
                    "coach_model": coach_name,
                    "gm_N_star": N_star,
                    "N": len(feat_names),
                    "n_clin": len(current_clin_feats),
                    "n_pt": len(current_pt_feats),
                    "n_ct": len(current_ct_feats),
                    "n_rad": len(current_pt_feats) + len(current_ct_feats),
                    "clin_ratio": ratio,
                    "oof_ci": oof_ci,
                    "fold_std": fstd,
                    "ci_train": safe_ci(y_train, risk_tr),
                    "ci_chus": ci_chus,
                    "ci_chup": ci_chup,
                    "boot_ci_chus_lo": np.nan,
                    "boot_ci_chus_hi": np.nan,
                    "boot_ci_chup_lo": np.nan,
                    "boot_ci_chup_hi": np.nan,
                    "boot_n_valid": np.nan,
                    "gm_params": str(gm_params),
                    "coach_params": str(coach_params),
                    "clin_names": "; ".join(current_clin_feats),
                    "pt_names": "; ".join(current_pt_feats),
                    "ct_names": "; ".join(current_ct_feats),
                    "rad_names": "; ".join(
                        [f"PT:{f}" for f in current_pt_feats] +
                        [f"CT:{f}" for f in current_ct_feats]
                    ),
                }
                all_scan_rows.append(row)
                print(
                    f"  {method}/{gm_name}/{coach_name} | R={ratio:.2f} "
                    f"| N={len(feat_names)} ({len(current_clin_feats)}c "
                    f"{len(current_pt_feats)}pt+{len(current_ct_feats)}ct) "
                    f"| OOF={oof_ci:.4f} | CHUS={ci_chus:.4f} | CHUP={ci_chup:.4f}"
                )

# ==========================================
# 5. BOOTSTRAP CI FOR ALL ROWS
# ==========================================
print("\n--- Final evaluation: Bootstrap CI (CHUS + CHUP) ---")
scan_df = pd.DataFrame(all_scan_rows)

for row_idx in scan_df.index:
    method = scan_df.loc[row_idx, "ranker"]
    gm_name = scan_df.loc[row_idx, "gm_model"]
    coach_name = scan_df.loc[row_idx, "coach_model"]
    winning_ratio = scan_df.loc[row_idx, "clin_ratio"]
    feat_names = main_loop_feature_store[(method, gm_name, winning_ratio)]
    col_idx = [feat_idx_map[f] for f in feat_names]

    print(f"Bootstrap {row_idx + 1}/{len(scan_df)}: {method}/{gm_name}/{coach_name} r={winning_ratio:.2f}")

    s1 = stage1_results[(method, gm_name)]
    coach_params = s1["gm_params"] if coach_name == gm_name else COACH_HEURISTICS[coach_name]
    coach_fn = lambda: build_model_fn(coach_name, coach_params)

    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_train[:, col_idx])
    X_chus_sc = sc.transform(X_chus[:, col_idx])
    X_chup_sc = sc.transform(X_chup[:, col_idx])

    m = coach_fn().fit(X_tr_sc, y_train)
    risk_chus = m.predict(X_chus_sc)
    risk_chup = m.predict(X_chup_sc)

    _, _, ci_lo_chus, ci_hi_chus, n_valid = bootstrap_ci(y_chus, risk_chus)
    _, _, ci_lo_chup, ci_hi_chup, _ = bootstrap_ci(y_chup, risk_chup)
    scan_df.loc[row_idx, ["boot_ci_chus_lo", "boot_ci_chus_hi",
                           "boot_ci_chup_lo", "boot_ci_chup_hi",
                           "boot_n_valid"]] = ci_lo_chus, ci_hi_chus, ci_lo_chup, ci_hi_chup, n_valid

# ==========================================
# 6. SAVE OUTPUT
# ==========================================
print("\n--- Saving output ---")
scan_df.insert(0, "No", np.arange(1, len(scan_df) + 1))

out_path = OUTPUT_DIR / "8_mar_SC3_rad_PT768xCT8_325_LOCODEV_dell_all_results.csv"
scan_df.to_csv(out_path, index=False)
print(f"Saved: {out_path.name} ({len(scan_df)} rows)")

scan_df["trio_mean"] = (scan_df["oof_ci"] + scan_df["ci_chus"] + scan_df["ci_chup"]) / 3
scan_df["trio_gap"] = scan_df[["oof_ci", "ci_chus", "ci_chup"]].max(axis=1) - \
                      scan_df[["oof_ci", "ci_chus", "ci_chup"]].min(axis=1)
top10 = scan_df.nlargest(10, "trio_mean")[
    ["No", "ranker", "gm_model", "coach_model", "N", "n_clin", "n_pt", "n_ct",
     "oof_ci", "ci_chus", "ci_chup", "trio_mean", "trio_gap",
     "boot_ci_chus_lo", "boot_ci_chus_hi", "boot_ci_chup_lo", "boot_ci_chup_hi",
     "clin_names", "pt_names", "ct_names"]
]
print("\nTop 10 by Trio mean (OOF + CHUS + CHUP):")
print(top10.to_string(index=False))

top10_path = OUTPUT_DIR / "8_mar_SC3_rad_PT768xCT8_325_LOCODEV_dell_top10.csv"
top10.to_csv(top10_path, index=False)
print(f"\nSaved top10: {top10_path.name}")

print("\n8_mar_SC3_rad_PT768xCT8_325_LOCODEV_dell finished.")
