"""
8_mar_SC3_rad_PT768xCT8_325_LOCODEV_pc_6b.py

Exhaustive SC3 EPV-cut control run.
pc_6b: full-pool EPV-cut rankers only (shrinkage and direct variants).
No bootstrap. Semi-guided bounds for PT768-only search.
Saves ALL Optuna trials x 4 coaches per trial (no Stage 2 substitution scan).

Rankers:
  FULL_WLOCO_epv_cut
  FULL_LOCO_EPV_CUT
GM: SVM, CoxPH | Coaches: EST, SVM, GBS, CoxPH
Expected rows: TRIALS * 4 * 2 * 4 (e.g. 2000*32 = 64k)

Outputs: Mar_2026/8_mar_SC3_rad_PT768xCT8_325_LOCODEV_pc_6b_outputs/

Usage:
  python Mar_2026/8_mar_SC3_rad_PT768xCT8_325_LOCODEV_pc_6b.py
  SC3_TRIALS=1000 python Mar_2026/8_mar_SC3_rad_PT768xCT8_325_LOCODEV_pc_6b.py
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
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import ExtraSurvivalTrees, GradientBoostingSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

# ==========================================
# CONFIG
# ==========================================
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "8_mar_SC3_rad_PT768xCT8_325_LOCODEV_pc_6b_outputs"
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
N_BOOT = 0
N_MIN, N_MAX = 8, 12
N_CLIN_MIN, N_CLIN_MAX = 1, 2
PT_MIN = 6
CT_MIN = 1
PT_MAX = 10
CT_MAX = 2
TRIALS = int(os.getenv("SC3_TRIALS", "2000"))
PHASE_A_TRIALS = int(os.getenv("SC3_PHASE_A_TRIALS", "300"))

# Ranking config
LOCO_LAMBDA = 0.35
LOCO_KAPPA = 5.0
PREFILTER_K = 30
LOCO_EVT_MIN_EVENTS = 10
EPV_CUT_EVENTS_MIN = 8
EPV_CUT_ENON_MIN = 300
EPV_CUT_RATIO_MIN = 0.0

# GM objective weights: J = OOF - a*OOF_std - b*center_gap - c*feature_penalty
OBJ_A_OOF_STD = 0.10
OBJ_B_CENTER_GAP = 0.00
OBJ_C_FEAT_PEN = 0.00

RANKING_METHODS = [
    "FULL_WLOCO_epv_cut",
    "FULL_LOCO_EPV_CUT",
]
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
# ==========================================
# LOGGING
# ==========================================
LOG_FILE_PATH = OUTPUT_DIR / "8_mar_SC3_rad_PT768xCT8_325_LOCODEV_pc_6b_log.md"


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


def model_oof_with_center_gap(X, y, center_ids, f_idx, fn):
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
        return np.nan, np.nan, 1.0

    oof_ci = safe_ci(y[oof_ok], oof_pred[oof_ok])
    fold_std = float(np.std(f_cs)) if len(f_cs) > 1 else 1.0

    center_cis = []
    for c in np.unique(center_ids):
        mask_c = (center_ids == c) & oof_ok
        if mask_c.sum() < 5:
            continue
        if int(y["event"][mask_c].sum()) == 0:
            continue
        if int((~y["event"][mask_c]).sum()) == 0:
            continue
        c_ci = safe_ci(y[mask_c], oof_pred[mask_c])
        if not np.isnan(c_ci):
            center_cis.append(c_ci)
    center_gap = (float(max(center_cis) - min(center_cis)) if len(center_cis) >= 2 else 1.0)
    return oof_ci, fold_std, center_gap


def univar_rank(X, y, n):
    s = [max(c, 1 - c) if not np.isnan(c := safe_ci(y, X[:, i])) else 0.5
         for i in range(X.shape[1])]
    o = np.argsort(np.array(s))[::-1]
    return [n[i] for i in o], np.array(s)[o]


def split_ranked_pt_ct(ranked_features, pt_set, ct_set):
    pt_r = [f for f in ranked_features if f in pt_set]
    ct_r = [f for f in ranked_features if f in ct_set]
    return pt_r, ct_r


def loco_rank_mode(
    X_full,
    y_full,
    center_ids,
    feat_names,
    mode,
    loco_lambda=LOCO_LAMBDA,
    kappa=LOCO_KAPPA,
):
    """
    LOCO rankers used in _5:
    - loco_evt: hard cutoff by min events
    - w_epv_ratio: weight=e/n
    - w_enon: weight=e*(n-e)
    - w_epv_cut: hard cutoff by (events, e*(n-e), e/n), equal weights among kept centres
    - w_epv_x_enon: weight=(e/n)*e*(n-e)
    - loco_epv_cut: hard EPV cutoff + direct centre mean CI (no shrinkage, no weighting)
    """
    centres = np.unique(center_ids)
    n_feats = X_full.shape[1]
    ci_matrix = np.full((len(centres), n_feats), np.nan)
    centre_weights = np.zeros(len(centres), dtype=float)

    for fold_i, held_centre in enumerate(centres):
        mask_val = center_ids == held_centre
        mask_tr = ~mask_val
        n_c = int(mask_val.sum())
        if n_c < 5:
            continue
        e_c = int(y_full["event"][mask_val].sum())
        ne_c = n_c - e_c
        epv_ratio = (e_c / n_c) if n_c > 0 else 0.0
        enon = float(e_c * ne_c)

        include = True
        if mode == "loco_evt":
            include = e_c >= LOCO_EVT_MIN_EVENTS
            w_c = 1.0
        elif mode == "w_epv_ratio":
            w_c = max(1e-9, epv_ratio)
        elif mode == "w_enon":
            w_c = max(1.0, enon)
        elif mode == "w_epv_x_enon":
            w_c = max(1e-9, epv_ratio * enon)
        elif mode == "w_epv_cut":
            include = (
                e_c >= EPV_CUT_EVENTS_MIN
                and enon >= EPV_CUT_ENON_MIN
                and epv_ratio >= EPV_CUT_RATIO_MIN
            )
            w_c = 1.0
        elif mode == "loco_epv_cut":
            include = (
                e_c >= EPV_CUT_EVENTS_MIN
                and enon >= EPV_CUT_ENON_MIN
                and epv_ratio >= EPV_CUT_RATIO_MIN
            )
            w_c = 1.0
        else:
            raise ValueError(f"Unknown LOCO mode: {mode}")

        if not include:
            continue
        centre_weights[fold_i] = w_c

        X_tr = X_full[mask_tr]
        X_val = X_full[mask_val]
        y_val = y_full[mask_val]
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_val_sc = sc.transform(X_val)
        for fi in range(n_feats):
            c = safe_ci(y_val, X_val_sc[:, fi])
            c = max(c, 1 - c) if not np.isnan(c) else 0.5
            if mode == "loco_epv_cut":
                ci_matrix[fold_i, fi] = c
            else:
                ci_matrix[fold_i, fi] = (e_c * c + kappa * 0.5) / (e_c + kappa)

    valid = centre_weights > 0
    if not np.any(valid):
        feat_arr = np.array(feat_names)
        return list(feat_arr), np.full(len(feat_names), 0.5)

    W = centre_weights[valid]
    M = np.where(np.isnan(ci_matrix[valid, :]), 0.5, ci_matrix[valid, :])
    w_sum = float(np.sum(W))
    mean_ci = np.sum(M * W[:, None], axis=0) / w_sum

    if mode in ("w_epv_ratio", "w_enon", "w_epv_x_enon"):
        var_ci = np.sum(W[:, None] * (M - mean_ci[None, :]) ** 2, axis=0) / w_sum
        std_ci = np.sqrt(np.maximum(var_ci, 0.0))
        score = mean_ci - loco_lambda * std_ci
    else:
        score = mean_ci

    o = np.argsort(score)[::-1]
    feat_arr = np.array(feat_names)
    return list(feat_arr[o]), score[o]


def enqueue_structural_seeds(study, alpha_key):
    alpha_vals = [0.001, 0.01, 0.05, 0.1, 0.3]
    structures = [
        {"total_n": 10, "n_clin": 1, "n_pt": 8},
        {"total_n": 9, "n_clin": 1, "n_pt": 7},
        {"total_n": 11, "n_clin": 1, "n_pt": 8},
        {"total_n": 11, "n_clin": 2, "n_pt": 8},
        {"total_n": 12, "n_clin": 1, "n_pt": 9},
        {"total_n": 12, "n_clin": 2, "n_pt": 9},
    ]
    for s in structures:
        n_ct = (s["total_n"] - s["n_clin"]) - s["n_pt"]
        if n_ct < CT_MIN or n_ct > CT_MAX:
            continue
        for a in alpha_vals:
            p = dict(s)
            p[alpha_key] = a
            study.enqueue_trial(p)


def enqueue_phase_b_focus(study, alpha_key, n_enqueue, seed=SEED):
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) < 30 or n_enqueue <= 0:
        return
    completed = sorted(completed, key=lambda t: t.value if t.value is not None else -1e9, reverse=True)
    top_k = max(1, int(0.2 * len(completed)))
    top = completed[:top_k]
    n_vals = [int(t.params.get("total_n", N_MIN)) for t in top]
    nc_vals = [int(t.params.get("n_clin", N_CLIN_MIN)) for t in top]
    npt_vals = [int(t.params.get("n_pt", PT_MIN)) for t in top]
    alpha_vals = [float(t.params.get(alpha_key, 0.01)) for t in top if alpha_key in t.params]
    if len(alpha_vals) == 0:
        alpha_vals = [0.01]

    n_low, n_high = int(np.percentile(n_vals, 20)), int(np.percentile(n_vals, 80))
    nc_low, nc_high = int(np.percentile(nc_vals, 20)), int(np.percentile(nc_vals, 80))
    npt_low, npt_high = int(np.percentile(npt_vals, 20)), int(np.percentile(npt_vals, 80))
    a_low, a_high = float(np.percentile(alpha_vals, 10)), float(np.percentile(alpha_vals, 90))
    a_low = max(a_low, 1e-4 if alpha_key == "alpha_svm" else 1e-3)
    a_high = max(a_high, a_low * 1.05)

    rng = np.random.default_rng(seed)
    for _ in range(n_enqueue):
        N = int(rng.integers(max(N_MIN, n_low), min(N_MAX, n_high) + 1))
        n_clin = int(rng.integers(max(N_CLIN_MIN, nc_low), min(N_CLIN_MAX, nc_high) + 1))
        n_rad = N - n_clin
        n_pt_lo = max(PT_MIN, n_rad - CT_MAX, npt_low)
        n_pt_hi = min(PT_MAX, n_rad - CT_MIN, npt_high)
        if n_pt_lo > n_pt_hi:
            continue
        n_pt = int(rng.integers(n_pt_lo, n_pt_hi + 1))
        n_ct = n_rad - n_pt
        if n_ct < CT_MIN or n_ct > CT_MAX:
            continue
        loga = float(rng.uniform(np.log(a_low), np.log(a_high)))
        p = {"total_n": N, "n_clin": n_clin, "n_pt": n_pt, alpha_key: float(np.exp(loga))}
        study.enqueue_trial(p)


# ==========================================
# 1. LOAD DATA
# ==========================================
print("=" * 70)
print("8_mar_SC3_rad_PT768xCT8_325_LOCODEV_pc_6b")
print("Exhaustive: pc_6b EPV-cut controls (full pool) | N_BOOT=0")
print("GM: SVM+CoxPH | Coaches: EST/SVM/GBS/CoxPH")
print(f"N in [{N_MIN},{N_MAX}] | n_clin in [{N_CLIN_MIN},{N_CLIN_MAX}] | "
      f"n_pt in [{PT_MIN},{PT_MAX}] | n_ct in [{CT_MIN},{CT_MAX}] | TRIALS={TRIALS}")
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
# 2. PRE-COMPUTE RANKINGS — FULL EPV-cut rankers
# ==========================================
print("\n--- Pre-computing feature rankings (FULL_WLOCO_epv_cut + FULL_LOCO_EPV_CUT) ---")
g_sc = StandardScaler()
X_train_sc = g_sc.fit_transform(X_train)

clin_feat_arr = np.array(clin_feat)
pt_feat_arr   = np.array(pt_feat)
ct_feat_arr   = np.array(ct_feat)

rankings = {}
clin_univar_ranked = univar_rank(X_train_sc[:, clin_idx], y_train, clin_feat_arr)[0]
print("  Clin UNIVAR (for all rankers) done")

rad_feat = pt_feat + ct_feat
rad_idx_list = pt_idx + ct_idx
ranked_rad_univar, _ = univar_rank(X_train_sc[:, rad_idx_list], y_train, rad_feat)
univar30 = ranked_rad_univar[:PREFILTER_K]
univar30_idx = [feat_idx_map[f] for f in univar30]
print(f"  UNIVAR top-{PREFILTER_K} pool prepared")

ranker_mode_map = {
    "FULL_WLOCO_epv_cut": ("w_epv_cut", "full"),
    "FULL_LOCO_EPV_CUT": ("loco_epv_cut", "full"),
}
for rk in RANKING_METHODS:
    mode, pool_kind = ranker_mode_map[rk]
    if pool_kind == "full":
        pool_names = rad_feat
        pool_idx = rad_idx_list
    else:
        pool_names = univar30
        pool_idx = univar30_idx
    ranked_rk, _ = loco_rank_mode(
        X_train_sc[:, pool_idx],
        y_train,
        dev_center_ids,
        pool_names,
        mode=mode,
    )
    pt_r, ct_r = split_ranked_pt_ct(ranked_rk, pt_feat_set_idx, ct_feat_set_idx)
    rankings[rk] = {"clin": clin_univar_ranked, "pt": pt_r, "ct": ct_r}
    print(f"  {rk} done [{pool_kind}] (pt={len(pt_r)}, ct={len(ct_r)})")

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
CT817_FEATURES = {
    "GTVp_wavelet-LHL_glcm_ClusterProminence",
    "GTVn_exponential_ngtdm_Coarseness",
}
print("\n  Rank preview by ranker (PT top-8 / CT top-5):")
for rk in RANKING_METHODS:
    pt_preview = rankings[rk]["pt"][:8]
    ct_preview = rankings[rk]["ct"][:5]
    print(f"    {rk}:")
    print("      PT:", ", ".join([f"{x}{'*' if x in PT817_FEATURES else ''}" for x in pt_preview]))
    print("      CT:", ", ".join([f"{x}{'*' if x in CT817_FEATURES else ''}" for x in ct_preview]))
print("  pc_6b ranker setup done")

# ==========================================
# 3. EXHAUSTIVE OPTUNA: ALL TRIALS x COACHES
# ==========================================
print(f"\n--- Exhaustive Optuna (2-phase TPE): {TRIALS} trials x {len(RANKING_METHODS)} rankers "
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
    if len(pt_ridx) < PT_MIN or len(ct_ridx) < CT_MIN:
        print(
            f"\n  Skip ranker={method}: insufficient pool "
            f"(pt={len(pt_ridx)} need>={PT_MIN}, ct={len(ct_ridx)} need>={CT_MIN})"
        )
        continue

    for gm_name in GM_MODEL_NAMES:
        print(f"\n  Ranker={method} | GM={gm_name} | Running {TRIALS} trials...")
        alpha_key = "alpha_svm" if gm_name == "SVM" else "alpha_cox"

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEED)
        )

        enqueue_structural_seeds(study, alpha_key)

        n_pt_pool = len(pt_ridx)
        n_ct_pool = len(ct_ridx)

        def make_objective(gm_name, clin_ridx, pt_ridx, ct_ridx, method):
            def objective(trial):
                N = trial.suggest_int("total_n", N_MIN, N_MAX)
                n_clin_hi = min(N_CLIN_MAX, len(clin_ridx), N - PT_MIN - CT_MIN)
                n_clin = trial.suggest_int("n_clin", N_CLIN_MIN, n_clin_hi)
                n_rad = N - n_clin

                n_pt_min = max(PT_MIN, n_rad - CT_MAX)
                n_pt_max = min(PT_MAX, n_rad - CT_MIN, len(pt_ridx))
                if n_pt_min > n_pt_max:
                    n_pt_min, n_pt_max = n_pt_max, n_pt_max
                n_pt = trial.suggest_int("n_pt", n_pt_min, n_pt_max)
                n_ct = n_rad - n_pt
                n_ct = max(CT_MIN, min(n_ct, CT_MAX, len(ct_ridx)))
                n_pt = n_rad - n_ct
                n_pt = max(PT_MIN, min(n_pt, PT_MAX, len(pt_ridx)))

                feat_idx = clin_ridx[:n_clin] + pt_ridx[:n_pt] + ct_ridx[:n_ct]

                if gm_name == "SVM":
                    alpha = trial.suggest_float("alpha_svm", 1e-4, 1., log=True)
                    model_fn = lambda: build_model_fn("SVM", {"alpha": alpha})
                elif gm_name == "CoxPH":
                    alpha = trial.suggest_float("alpha_cox", 1e-3, 10., log=True)
                    model_fn = lambda: build_model_fn("CoxPH", {"alpha": alpha})
                else:
                    raise ValueError(f"Unknown GM: {gm_name}")

                oof_ci, fold_std, center_gap = model_oof_with_center_gap(
                    X_train, y_train, dev_center_ids, feat_idx, model_fn
                )
                if np.isnan(oof_ci):
                    oof_ci, fold_std, center_gap = 0.5, 1.0, 1.0
                feat_pen = (N - N_MIN) / max(1, (N_MAX - N_MIN))
                score = oof_ci - OBJ_A_OOF_STD * fold_std - OBJ_B_CENTER_GAP * center_gap - OBJ_C_FEAT_PEN * feat_pen

                trial.set_user_attr("N", N)
                trial.set_user_attr("n_clin", n_clin)
                trial.set_user_attr("n_pt", n_pt)
                trial.set_user_attr("n_ct", n_ct)
                trial.set_user_attr("oof_ci", oof_ci)
                trial.set_user_attr("fold_std", fold_std)
                trial.set_user_attr("center_gap", center_gap)
                trial.set_user_attr("objective_score", score)
                trial.set_user_attr("clin_features", [all_features[i] for i in clin_ridx[:n_clin]])
                trial.set_user_attr("pt_features",   [all_features[i] for i in pt_ridx[:n_pt]])
                trial.set_user_attr("ct_features",   [all_features[i] for i in ct_ridx[:n_ct]])
                return score

            return objective

        objective_fn = make_objective(gm_name, clin_ridx, pt_ridx, ct_ridx, method)
        n_a = min(PHASE_A_TRIALS, TRIALS)
        study.optimize(objective_fn, n_trials=n_a, show_progress_bar=False)
        remaining = TRIALS - n_a
        if remaining > 0:
            enqueue_phase_b_focus(study, alpha_key, n_enqueue=min(remaining, max(100, remaining // 2)))
            study.optimize(objective_fn, n_trials=remaining, show_progress_bar=False)

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
            center_gap = trial.user_attrs.get("center_gap", np.nan)
            objective_score = trial.user_attrs.get("objective_score", trial.value if trial.value is not None else np.nan)
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
                    "center_gap": center_gap,
                    "objective_score": objective_score,
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
              f"| OOF={bt.user_attrs['oof_ci']:.4f} | GAP={bt.user_attrs.get('center_gap', np.nan):.4f}")

# ==========================================
# 4. SAVE OUTPUT
# ==========================================
all_df = pd.DataFrame(all_rows)
all_df.insert(0, "No", np.arange(1, len(all_df) + 1))
all_df["trio_mean"] = (all_df["oof_ci"] + all_df["ci_chus"] + all_df["ci_chup"]) / 3

# ==========================================
# 5. SAVE OUTPUT
# ==========================================
print("\n--- Saving output ---")

out_path = OUTPUT_DIR / "8_mar_SC3_rad_PT768xCT8_325_LOCODEV_pc_6b_all_results.csv"
all_df.to_csv(out_path, index=False)
print(f"Saved: {out_path.name} ({len(all_df)} rows)")

top20_out = all_df.nlargest(20, "trio_mean")[
    ["No", "ranker", "gm_model", "coach_model", "N", "n_clin", "n_pt", "n_ct",
     "oof_ci", "ci_chus", "ci_chup", "trio_mean",
     "clin_names", "pt_names", "ct_names"]
]
print("\nTop 20 by Trio mean (OOF + CHUS + CHUP):")
print(top20_out.to_string(index=False))

pass_primary = all_df[(all_df["oof_ci"] > 0.68) & (all_df["ci_chus"] >= 0.70) & (all_df["ci_chup"] >= 0.70)]
print(f"\nPrimary target rows (OOF>0.68 & CHUS>=0.70 & CHUP>=0.70): {len(pass_primary)}")
if len(pass_primary) > 0:
    best_mm = pass_primary.assign(min_ext=np.minimum(pass_primary["ci_chus"], pass_primary["ci_chup"])) \
        .sort_values(["min_ext", "trio_mean"], ascending=False).head(5)
    print(best_mm[["No", "ranker", "gm_model", "coach_model", "N", "n_clin", "n_pt", "n_ct", "oof_ci", "ci_chus", "ci_chup", "trio_mean"]].to_string(index=False))

print("\n8_mar_SC3_rad_PT768xCT8_325_LOCODEV_pc_6b finished.")


