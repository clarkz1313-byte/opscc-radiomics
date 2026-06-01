"""
8_mar_SC3_rad_PT768xCT8_325_LOCODEV_dell_6a.py

Traditional SC3 comparison run with 4 rankers. No bootstrap.
Dell_4: N in [6,10], n_clin in [1,2], n_pt in [4,6], n_ct in [1,3].

Rankers:
  FULL_LOCO_evt
  FULL_LOCO_EPV_CUT
  FILTER_P20C10_LOCO_evt
  FILTER_P20C10_LOCO_EPV_CUT
Fix: RATIO_GRID_FULL = [0.0] + RATIO_GRID (n_clin=1 at ratio=0.0)

Machine: Dell | TRIALS=1000 | N_MIN=8, N_MAX=12 | PT_MIN=6, CT_MIN=1
N_BOOT=0 (omit bootstrap to save time)

Outputs: Mar_2026/8_mar_SC3_rad_PT768xCT8_325_LOCODEV_dell_6a_outputs/

Usage:
  python Mar_2026/8_mar_SC3_rad_PT768xCT8_325_LOCODEV_dell_6a.py
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
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import ExtraSurvivalTrees, GradientBoostingSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

# ==========================================
# CONFIG
# ==========================================
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "8_mar_SC3_rad_PT768xCT8_325_LOCODEV_dell_6a_outputs"
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
N_BOOT = 0  # Omit bootstrap to save time
N_MIN, N_MAX = 8, 12
N_CLIN_MIN, N_CLIN_MAX = 1, 2
PT_MIN = 6
CT_MIN = 1
PT_MAX = 10
CT_MAX = 2
TRIALS = int(os.getenv("SC3_TRIALS", "1000"))
RATIO_GRID = [round(r * 0.05, 2) for r in range(1, 13)]
# Fix 1: prepend 0.0 to guarantee n_clin=1 is evaluated
RATIO_GRID_FULL = [0.0] + RATIO_GRID

LOCO_LAMBDA = 0.35
LOCO_KAPPA = 5.0
PREFILTER_PT_K = 20
PREFILTER_CT_K = 10
LOCO_EVT_MIN_EVENTS = 10
EPV_CUT_EVENTS_MIN = 8
EPV_CUT_ENON_MIN = 300
EPV_CUT_RATIO_MIN = 0.0

OBJ_A_OOF_STD = 0.10
OBJ_B_CENTER_GAP = 0.00
OBJ_C_FEAT_PEN = 0.00

RANKING_METHODS = [
    "FULL_LOCO_evt",
    "FULL_LOCO_EPV_CUT",
    "FILTER_P20C10_LOCO_evt",
    "FILTER_P20C10_LOCO_EPV_CUT",
]
GM_MODEL_NAMES = ["EST", "SVM", "GBS", "CoxPH"]
COACH_NAMES = ["EST", "SVM", "GBS", "CoxPH"]
COACH_HEURISTICS = {
    "EST":  {"n_estimators": 300, "max_depth": 12, "min_samples_split": 6,
             "min_samples_leaf": 4, "max_features": "sqrt"},
    "GBS":  {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "subsample": 0.8},
    "SVM":  {"alpha": 0.01},
    "CoxPH":{"alpha": 0.1},
}
PI_N_REPEATS = 10

# ==========================================
# LOGGING
# ==========================================
LOG_FILE_PATH = OUTPUT_DIR / "8_mar_SC3_rad_PT768xCT8_325_LOCODEV_dell_6a_log.md"


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


def univar_rank(X, y, n):
    s = [max(c, 1 - c) if not np.isnan(c := safe_ci(y, X[:, i])) else 0.5
         for i in range(X.shape[1])]
    o = np.argsort(np.array(s))[::-1]
    return [n[i] for i in o], np.array(s)[o]


def loco_rank_mode(
    X_full,
    y_full,
    center_ids,
    feat_names,
    mode,
    loco_lambda=LOCO_LAMBDA,
    kappa=LOCO_KAPPA,
):
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
print("8_mar_SC3_rad_PT768xCT8_325_LOCODEV_dell_6a")
print("Traditional SC3: 4-ranker comparison (full + filtered evt/epv-cut); N_BOOT=0")
print("Rankers: FULL_LOCO_evt / FULL_LOCO_EPV_CUT / FILTER_P20C10_LOCO_evt / FILTER_P20C10_LOCO_EPV_CUT")
print(f"N[{N_MIN},{N_MAX}] n_clin[{N_CLIN_MIN},{N_CLIN_MAX}] n_pt[{PT_MIN},{PT_MAX}] n_ct[{CT_MIN},{CT_MAX}] | TRIALS={TRIALS}")
print("Rad pool: PT768 (35) + CT8_325 (39) | Dev = full train | CHUS+CHUP")
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

# Store CenterID array aligned with dev_merged for LOCO rankers
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
# 2. PRE-COMPUTE RANKINGS — full + separate-prefilter rankers
# ==========================================
print("\n--- Pre-computing feature rankings (4 rankers; full and PT20+CT10 filtered pools) ---")
g_sc = StandardScaler()
X_train_sc = g_sc.fit_transform(X_train)

clin_feat_arr = np.array(clin_feat)
rankings = {}
clin_univar_ranked = univar_rank(X_train_sc[:, clin_idx], y_train, clin_feat_arr)[0]
print("  Clin UNIVAR (for all rankers) done")

rad_feat = pt_feat + ct_feat
rad_idx_list = pt_idx + ct_idx
pt_ranked_univar, _ = univar_rank(X_train_sc[:, pt_idx], y_train, pt_feat)
ct_ranked_univar, _ = univar_rank(X_train_sc[:, ct_idx], y_train, ct_feat)
pref_pt = pt_ranked_univar[:PREFILTER_PT_K]
pref_ct = ct_ranked_univar[:PREFILTER_CT_K]
pref_pool = pref_pt + pref_ct
pref_idx = [feat_idx_map[f] for f in pref_pool]
print(f"  Separate prefilter prepared: PT top-{PREFILTER_PT_K} + CT top-{PREFILTER_CT_K} = {len(pref_pool)}")

ranker_mode_map = {
    "FULL_LOCO_evt": ("loco_evt", "full"),
    "FULL_LOCO_EPV_CUT": ("loco_epv_cut", "full"),
    "FILTER_P20C10_LOCO_evt": ("loco_evt", "pref"),
    "FILTER_P20C10_LOCO_EPV_CUT": ("loco_epv_cut", "pref"),
}
for rk in RANKING_METHODS:
    mode, pool_kind = ranker_mode_map[rk]
    if pool_kind == "full":
        pool_names = rad_feat
        pool_idx = rad_idx_list
    else:
        pool_names = pref_pool
        pool_idx = pref_idx
    ranked_rk, _ = loco_rank_mode(
        X_train_sc[:, pool_idx],
        y_train,
        dev_center_ids,
        pool_names,
        mode=mode,
    )
    pt_r = [f for f in ranked_rk if f in pt_feat_set_idx]
    ct_r = [f for f in ranked_rk if f in ct_feat_set_idx]
    rankings[rk] = {"clin": clin_univar_ranked, "pt": pt_r, "ct": ct_r}
    print(f"  {rk} done [{pool_kind}] (pt={len(pt_r)}, ct={len(ct_r)})")

# ==========================================
# 3. STAGE 1: GM OPTUNA SEARCH (PT-forced split, robust objective)
# ==========================================
print(f"\n--- Stage 1: Optuna GM search ({TRIALS} trials each, PT-forced split, robust objective) ---")


def make_gm_objective(gm_name, clin_ridx, pt_ridx, ct_ridx):
    def objective(trial):
        N = trial.suggest_int("total_n", N_MIN, N_MAX)
        n_clin_hi = min(N_CLIN_MAX, len(clin_ridx), N - PT_MIN - CT_MIN)
        n_clin = trial.suggest_int("n_clin", N_CLIN_MIN, n_clin_hi)
        n_rad = N - n_clin

        n_pt_min = max(PT_MIN, n_rad - CT_MAX)
        n_pt_max = min(PT_MAX, n_rad - CT_MIN, len(pt_ridx))
        if n_pt_min > n_pt_max:
            n_pt_min = n_pt_max
        n_pt = trial.suggest_int("n_pt", n_pt_min, n_pt_max)
        n_ct = n_rad - n_pt
        n_ct = max(CT_MIN, min(n_ct, CT_MAX, len(ct_ridx)))
        n_pt = n_rad - n_ct
        n_pt = max(PT_MIN, min(n_pt, PT_MAX, len(pt_ridx)))

        feat_idx = clin_ridx[:n_clin] + pt_ridx[:n_pt] + ct_ridx[:n_ct]

        if gm_name == "EST":
            model_fn = lambda: build_model_fn("EST", suggest_est(trial))
        elif gm_name == "GBS":
            model_fn = lambda: build_model_fn("GBS", suggest_gbs(trial))
        elif gm_name == "SVM":
            alpha = trial.suggest_float("alpha_svm", 1e-4, 1.0, log=True)
            model_fn = lambda: build_model_fn("SVM", {"alpha": alpha})
        elif gm_name == "CoxPH":
            alpha = trial.suggest_float("alpha_cox", 1e-3, 10.0, log=True)
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
        trial.set_user_attr("features", [all_features[i] for i in feat_idx])
        return score

    return objective


stage1_results = {}
for method in RANKING_METHODS:
    clin_r = rankings[method]["clin"]
    pt_r = rankings[method]["pt"]
    ct_r = rankings[method]["ct"]
    clin_ridx = [feat_idx_map[f] for f in clin_r]
    pt_ridx = [feat_idx_map[f] for f in pt_r]
    ct_ridx = [feat_idx_map[f] for f in ct_r]

    for gm_name in GM_MODEL_NAMES:
        print(f"  Optuna: {method}/{gm_name} ...", end=" ", flush=True)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEED)
        )

        if gm_name in ("SVM", "CoxPH"):
            alpha_key = "alpha_svm" if gm_name == "SVM" else "alpha_cox"
            for alpha_val in ([0.001, 0.01, 0.1] if gm_name == "SVM" else [0.01, 0.1, 1.0]):
                study.enqueue_trial({"total_n": 10, "n_clin": 1, "n_pt": 8, alpha_key: alpha_val})
                study.enqueue_trial({"total_n": 11, "n_clin": 1, "n_pt": 8, alpha_key: alpha_val})
                study.enqueue_trial({"total_n": 12, "n_clin": 2, "n_pt": 9, alpha_key: alpha_val})
        elif gm_name == "GBS":
            study.enqueue_trial({"total_n": 10, "n_clin": 1, "n_pt": 8,
                                 "n_e": 100, "lr": 0.1, "m_d": 3, "ss": 0.8})
            study.enqueue_trial({"total_n": 12, "n_clin": 2, "n_pt": 9,
                                 "n_e": 100, "lr": 0.1, "m_d": 3, "ss": 0.8})
        elif gm_name == "EST":
            study.enqueue_trial({"total_n": 10, "n_clin": 1, "n_pt": 8,
                                 "n_e": 300, "m_d": 12, "m_s_s": 6, "m_s_l": 4, "m_f": "sqrt"})
            study.enqueue_trial({"total_n": 12, "n_clin": 2, "n_pt": 9,
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
        print(
            f"N={bt.user_attrs['N']} (PT={bt_n_pt} CT={bt_n_ct}) "
            f"| OOF={bt.user_attrs['oof_ci']:.4f} | GAP={bt.user_attrs.get('center_gap', np.nan):.4f}"
        )

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
# 5. BOOTSTRAP CI FOR ALL ROWS (skipped when N_BOOT=0)
# ==========================================
scan_df = pd.DataFrame(all_scan_rows)
if N_BOOT > 0:
    print("\n--- Final evaluation: Bootstrap CI (CHUS + CHUP) ---")
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
else:
    print("\n--- Bootstrap skipped (N_BOOT=0) ---")

# ==========================================
# 6. SAVE OUTPUT
# ==========================================
print("\n--- Saving output ---")
scan_df.insert(0, "No", np.arange(1, len(scan_df) + 1))

out_path = OUTPUT_DIR / "8_mar_SC3_rad_PT768xCT8_325_LOCODEV_dell_6a_all_results.csv"
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

top10_path = OUTPUT_DIR / "8_mar_SC3_rad_PT768xCT8_325_LOCODEV_dell_6a_top10.csv"
top10.to_csv(top10_path, index=False)
print(f"\nSaved top10: {top10_path.name}")

print("\n8_mar_SC3_rad_PT768xCT8_325_LOCODEV_dell_6a finished.")


