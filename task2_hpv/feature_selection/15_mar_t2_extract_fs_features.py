# -*- coding: utf-8 -*-
"""
Task 2 Final Feature Extraction — Full Pipeline Reconstruction (Task 1 methodology)
=====================================================================================
For every CT (7) and PT (9) pipeline winner (best trial by AUC_CV in ALLtrials CSVs):
  1. Re-run S1->S2->S3 on full X_train using the stored Optuna params  (identical to
     the Task 1 approach in 4_mar_feature_selections_scripts/2_mar_extract_finalist_features.py)
  2. Save per-pipeline feature CSVs in 15_mar_T2_final_features/
  3. For CT_S3_2 and PT_S3_5 (which have user_attrs_selected_features):
     - Overlap analysis: stored vs reconstructed
  4. Bootstrap eval (CV5 / Test / Ext + 95% CI) for the PRIMARY winners:
     CT_S3_2 (reconstructed), PT_S3_5 (reconstructed), CT+PT combined (reconstructed)
  5. Comparison table: stored-features performance vs reconstructed-features performance

Run:
    cd "D:/Uppsala thesis" && python Mar_2026_task2/15_mar_t2_extract_fs_features.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from Mar_2026_task2.fs_task2_utils import (
    MRMR_OK,
    RELIEFF_OK,
    XGB_OK,
    correlation_filter,
    elasticnet_logistic_selection,
    lasso_logistic_selection,
    nested_cv_auc,
    evaluate_auc_test,
)

if MRMR_OK:
    from mrmr import mrmr_classif
if RELIEFF_OK:
    from skrebate import ReliefF
if XGB_OK:
    from xgboost import XGBClassifier

# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------
OUTPUT_DIR  = SCRIPT_DIR / "15_mar_T2_final_features"
DATA_DIR    = SCRIPT_DIR / "12_mar_task2_rad_data"
OPTUNA_DIR  = SCRIPT_DIR / "14_mar_t2_optuna_outputs"

CT_TRAIN = DATA_DIR / "13_mar_task2_CT_primary_train.csv"
CT_TEST  = DATA_DIR / "13_mar_task2_CT_primary_test.csv"
CT_EXT   = DATA_DIR / "12_mar_task2_CT_primary_ext.csv"

PT_TRAIN = DATA_DIR / "13_mar_task2_PT_primary_train.csv"
PT_TEST  = DATA_DIR / "13_mar_task2_PT_primary_test.csv"
PT_EXT   = DATA_DIR / "12_mar_task2_PT_primary_ext.csv"

CT_ALLTRIALS = OPTUNA_DIR / "14_mar_task2_stage4_CT_ALLtrials_20260315.csv"
PT_ALLTRIALS = OPTUNA_DIR / "14_mar_task2_stage4_PT_ALLtrials_20260316.csv"

CT_S3_2_TRIALS = OPTUNA_DIR / "CT_S3_2_trials_20260315.csv"
PT_S3_5_TRIALS = OPTUNA_DIR / "PT_S3_5_trials_20260316.csv"

EXCLUDE = [
    "PatientID", "HPV_binary", "Relapse", "RFS",
    "Age", "Gender_Male", "Treatment_CRT", "prefix",
]

SEED    = 42
N_BOOT  = 1000
N_FOLDS = 5

# -----------------------------------------------------------------------
# CT pipeline definitions (rank -> pipeline string tokens from Stage 3 CSV)
# -----------------------------------------------------------------------
CT_PIPELINES = {
    1:  {"S1": "ANOVA",   "S2": "RF PermImp",  "S3": "GB PermImp"},
    2:  {"S1": "ReliefF", "S2": "ElasticNet",  "S3": "RF PermImp"},
    3:  {"S1": "ANOVA",   "S2": "GB PermImp",  "S3": "RF PermImp"},
    6:  {"S1": "ANOVA",   "S2": "GB PermImp",  "S3": "LASSO"},
    7:  {"S1": "mRMR",    "S2": "LASSO",       "S3": "GB PermImp"},
    8:  {"S1": "ANOVA",   "S2": "RF PermImp",  "S3": "LASSO"},
    12: {"S1": "ReliefF", "S2": "RF PermImp",  "S3": "LASSO"},
}

# PT pipeline definitions (rank -> pipeline string tokens from Stage 3 CSV)
PT_PIPELINES = {
    1:  {"S1": "mRMR",  "S2": "GB PermImp",  "S3": "XGB PermImp"},
    3:  {"S1": "MWU",   "S2": "ElasticNet",  "S3": "XGB PermImp"},
    4:  {"S1": "MWU",   "S2": "XGB PermImp", "S3": "GB PermImp"},
    5:  {"S1": "mRMR",  "S2": "ElasticNet",  "S3": "XGB PermImp"},
    6:  {"S1": "ANOVA", "S2": "XGB PermImp", "S3": "RF PermImp"},
    7:  {"S1": "MWU",   "S2": "XGB PermImp", "S3": "LASSO"},
    10: {"S1": "ANOVA", "S2": "ElasticNet",  "S3": "XGB PermImp"},
    13: {"S1": "MWU",   "S2": "ElasticNet",  "S3": "GB PermImp"},
    15: {"S1": "ANOVA", "S2": "GB PermImp",  "S3": "XGB PermImp"},
}

# Stored (user_attrs) features for the two primary winners
CT_STORED_FEATURES = [
    "GTVp_log-sigma-1-mm-3D_firstorder_Maximum",
    "GTVn_wavelet-HLH_gldm_DependenceNonUniformityNormalized",
    "GTVn_wavelet-HHL_glszm_SmallAreaLowGrayLevelEmphasis",
    "GTVn_wavelet-HLH_ngtdm_Contrast",
    "GTVp_gradient_glrlm_ShortRunHighGrayLevelEmphasis",
    "GTVp_logarithm_glcm_InverseVariance",
    "GTVp_gradient_glrlm_HighGrayLevelRunEmphasis",
    "GTVp_squareroot_glszm_SizeZoneNonUniformity",
    "GTVp_logarithm_glcm_JointEnergy",
    "GTVp_gradient_glcm_ClusterProminence",
    "GTVp_gradient_glcm_ClusterShade",
]

PT_STORED_FEATURES = [
    "GTVn_wavelet-LHL_glszm_GrayLevelVariance",
    "GTVp_wavelet-LLH_firstorder_Median",
    "GTVn_logarithm_glszm_SmallAreaLowGrayLevelEmphasis",
    "GTVp_wavelet-HLL_firstorder_90Percentile",
    "GTVp_wavelet-HHL_glrlm_ShortRunHighGrayLevelEmphasis",
    "GTVp_wavelet-HHL_firstorder_Skewness",
    "GTVp_wavelet-LLH_glcm_ClusterShade",
    "GTVp_wavelet-HHL_glcm_InverseVariance",
]


# -----------------------------------------------------------------------
# Data loaders
# -----------------------------------------------------------------------
def _load_split(train_f: Path, test_f: Path, ext_f: Path):
    df_tr = pd.read_csv(train_f)
    feat_cols = [c for c in df_tr.columns if c not in EXCLUDE]
    X_tr = df_tr[feat_cols].replace([np.inf, -np.inf], np.nan)
    X_tr = X_tr.fillna(X_tr.median())
    y_tr = df_tr["HPV_binary"].values.astype(int)

    df_te = pd.read_csv(test_f)
    X_te = df_te[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(X_tr.median())
    y_te = df_te["HPV_binary"].values.astype(int)

    df_ext = pd.read_csv(ext_f)
    ext_cols = [c for c in feat_cols if c in df_ext.columns]
    X_ext = df_ext[ext_cols].replace([np.inf, -np.inf], np.nan).fillna(X_tr[ext_cols].median())
    y_ext = df_ext["HPV_binary"].values.astype(int)

    return X_tr, y_tr, X_te, y_te, X_ext, y_ext, feat_cols


# -----------------------------------------------------------------------
# Atomic selectors (operate on full X_train, same as _s{1,2,3}_sampler)
# -----------------------------------------------------------------------
def _sel_anova(X_df: pd.DataFrame, y: np.ndarray, k: int) -> list[str]:
    scores, _ = f_classif(X_df.values, y)
    scores = np.nan_to_num(scores, nan=0.0)
    idx = np.argsort(scores)[::-1][: min(k, X_df.shape[1])]
    return [X_df.columns[i] for i in idx]


def _sel_mwu(X_df: pd.DataFrame, y: np.ndarray, k: int) -> list[str]:
    pos, neg = X_df.values[y == 1], X_df.values[y == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return list(X_df.columns[:k])
    aucs = []
    for j in range(X_df.shape[1]):
        try:
            stat, _ = mannwhitneyu(pos[:, j], neg[:, j], alternative="two-sided")
            aucs.append(abs(stat / (n_pos * n_neg) - 0.5))
        except Exception:
            aucs.append(0.0)
    idx = np.argsort(np.array(aucs))[::-1][: min(k, X_df.shape[1])]
    return [X_df.columns[i] for i in idx]


def _sel_mrmr(X_df: pd.DataFrame, y: np.ndarray, k: int) -> list[str]:
    if not MRMR_OK:
        raise RuntimeError("mrmr-selection not installed")
    return list(mrmr_classif(X=X_df, y=pd.Series(y), K=min(k, X_df.shape[1])))


def _sel_relieff(X_df: pd.DataFrame, y: np.ndarray, k: int,
                 global_order: list[str] | None = None) -> list[str]:
    if global_order is not None:
        cols = set(X_df.columns)
        return [c for c in global_order if c in cols][:k]
    if not RELIEFF_OK:
        raise RuntimeError("skrebate not installed")
    n_neighbors = min(50, len(y) - 1)
    rf = ReliefF(n_features_to_select=X_df.shape[1], n_neighbors=n_neighbors)
    rf.fit(X_df.values, y)
    order = np.argsort(rf.feature_importances_)[::-1]
    return [X_df.columns[i] for i in order[:min(k, X_df.shape[1])]]


def _sel_lasso(X_df: pd.DataFrame, y: np.ndarray, target: int) -> list[str]:
    return lasso_logistic_selection(X_df, y, target_features=min(target, X_df.shape[1]),
                                    random_state=SEED)


def _sel_en(X_df: pd.DataFrame, y: np.ndarray, target: int, l1_ratio: float) -> list[str]:
    return elasticnet_logistic_selection(X_df, y, target_features=min(target, X_df.shape[1]),
                                         l1_ratio=l1_ratio, random_state=SEED)


def _sel_rf_perm(X_df: pd.DataFrame, y: np.ndarray, k: int, n_estimators: int) -> list[str]:
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=SEED, n_jobs=1)
    rf.fit(X_df.values, y)
    r = permutation_importance(rf, X_df.values, y, n_repeats=5, random_state=SEED, n_jobs=1)
    idx = np.argsort(r.importances_mean)[::-1][: min(k, X_df.shape[1])]
    return [X_df.columns[i] for i in idx]


def _sel_xgb_perm(X_df: pd.DataFrame, y: np.ndarray, k: int, n_estimators: int) -> list[str]:
    if not XGB_OK:
        raise RuntimeError("xgboost not installed")
    clf = XGBClassifier(n_estimators=n_estimators, random_state=SEED,
                        eval_metric="logloss", verbosity=0, n_jobs=1)
    clf.fit(X_df.values, y)
    r = permutation_importance(clf, X_df.values, y, n_repeats=5, random_state=SEED, n_jobs=1)
    idx = np.argsort(r.importances_mean)[::-1][: min(k, X_df.shape[1])]
    return [X_df.columns[i] for i in idx]


def _sel_gb_perm(X_df: pd.DataFrame, y: np.ndarray, k: int, n_estimators: int) -> list[str]:
    clf = GradientBoostingClassifier(n_estimators=n_estimators, random_state=SEED)
    clf.fit(X_df.values, y)
    r = permutation_importance(clf, X_df.values, y, n_repeats=5, random_state=SEED, n_jobs=1)
    idx = np.argsort(r.importances_mean)[::-1][: min(k, X_df.shape[1])]
    return [X_df.columns[i] for i in idx]


# -----------------------------------------------------------------------
# Full pipeline reconstruction on X_train
# -----------------------------------------------------------------------
def _run_pipeline(X_tr: pd.DataFrame, y_tr: np.ndarray,
                  s1_tok: str, s2_tok: str, s3_tok: str,
                  params: dict,
                  global_relieff_order: list[str] | None = None) -> list[str]:
    """Run S1->S2->S3 on full X_train with stored params. Returns final feature list."""
    # --- S1 ---
    if "ANOVA" in s1_tok:
        s1 = _sel_anova(X_tr, y_tr, int(params["anova_k"]))
    elif "ReliefF" in s1_tok:
        s1 = _sel_relieff(X_tr, y_tr, int(params["relief_k"]), global_relieff_order)
    elif "mRMR" in s1_tok:
        s1 = _sel_mrmr(X_tr, y_tr, int(params["mrmr_k"]))
    elif "MWU" in s1_tok:
        s1 = _sel_mwu(X_tr, y_tr, int(params["mwu_k"]))
    elif "LASSO" in s1_tok:
        s1 = _sel_lasso(X_tr, y_tr, int(params["lasso_target"]))
    elif "ElasticNet" in s1_tok:
        s1 = _sel_en(X_tr, y_tr, int(params["en_target"]), float(params["en_l1_ratio"]))
    elif "Corr" in s1_tok:
        s1 = correlation_filter(X_tr, threshold=float(params["corr_threshold"]))
    else:
        s1 = list(X_tr.columns)
    if not s1:
        return []

    # --- S2 ---
    X_s1 = X_tr[s1]
    if not s2_tok:
        s2 = s1
    elif "ANOVA" in s2_tok:
        s2 = _sel_anova(X_s1, y_tr, int(params["s2_anova_k"]))
    elif "ReliefF" in s2_tok:
        k = int(params.get("s2_relief_k", 30))
        if RELIEFF_OK:
            rf_s2 = ReliefF(n_features_to_select=min(k, len(s1)),
                            n_neighbors=min(50, len(y_tr) - 1))
            rf_s2.fit(X_s1.values, y_tr)
            order = np.argsort(rf_s2.feature_importances_)[::-1][: min(k, len(s1))]
            s2 = [X_s1.columns[i] for i in order]
        else:
            s2 = s1[:k]
    elif "mRMR" in s2_tok:
        s2 = _sel_mrmr(X_s1, y_tr, int(params["s2_mrmr_k"]))
    elif "ElasticNet" in s2_tok:
        s2 = _sel_en(X_s1, y_tr, int(params["s2_en_target"]), float(params["s2_en_l1_ratio"]))
    elif "LASSO" in s2_tok:
        s2 = _sel_lasso(X_s1, y_tr, int(params["s2_lasso_target"]))
    elif "XGB PermImp" in s2_tok or "XGBoost" in s2_tok:
        s2 = _sel_xgb_perm(X_s1, y_tr, int(params["s2_xgb_k"]), int(params["s2_xgb_n_estimators"]))
    elif "RF PermImp" in s2_tok:
        s2 = _sel_rf_perm(X_s1, y_tr, int(params["s2_rf_k"]), int(params["s2_rf_n_estimators"]))
    elif "GB PermImp" in s2_tok:
        s2 = _sel_gb_perm(X_s1, y_tr, int(params["s2_gb_k"]), int(params["s2_gb_n_estimators"]))
    else:
        s2 = s1
    if not s2:
        return []

    # --- S3 ---
    X_s2 = X_tr[s2]
    if not s3_tok:
        return s2
    elif "LASSO" in s3_tok:
        return _sel_lasso(X_s2, y_tr, int(params["s3_lasso_target"]))
    elif "ElasticNet" in s3_tok:
        return _sel_en(X_s2, y_tr, int(params["s3_en_target"]), float(params["s3_en_l1_ratio"]))
    elif "XGB PermImp" in s3_tok or "XGBoost" in s3_tok:
        return _sel_xgb_perm(X_s2, y_tr, int(params["s3_xgb_k"]), int(params["s3_xgb_n_estimators"]))
    elif "RF PermImp" in s3_tok:
        return _sel_rf_perm(X_s2, y_tr, int(params["s3_rf_k"]), int(params["s3_rf_n_estimators"]))
    elif "GB PermImp" in s3_tok:
        return _sel_gb_perm(X_s2, y_tr, int(params["s3_gb_k"]), int(params["s3_gb_n_estimators"]))
    return s2


# -----------------------------------------------------------------------
# AUC evaluation helpers
# -----------------------------------------------------------------------
def _logit_auc(X_tr, y_tr, X_ev, y_ev) -> float:
    if len(np.unique(y_ev)) < 2:
        return float("nan")
    scaler = StandardScaler()
    clf = LogisticRegression(C=1.0, class_weight="balanced",
                             solver="saga", max_iter=3000, random_state=SEED)
    clf.fit(scaler.fit_transform(X_tr), y_tr)
    return float(roc_auc_score(y_ev, clf.predict_proba(scaler.transform(X_ev))[:, 1]))


def _cv_auc(X, y) -> tuple[float, float]:
    clf = SkPipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, class_weight="balanced",
                                  solver="saga", max_iter=3000, random_state=SEED)),
    ])
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    return float(scores.mean()), float(scores.std())


def _bootstrap(X_tr, y_tr, X_ev, y_ev, n=N_BOOT) -> tuple[float, float, float, list]:
    rng = np.random.default_rng(SEED)
    n_ev = len(y_ev)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, n_ev, size=n_ev)
        yb, Xb = y_ev[idx], X_ev[idx]
        if len(np.unique(yb)) < 2:
            continue
        aucs.append(_logit_auc(X_tr, y_tr, Xb, yb))
    aucs = [a for a in aucs if not np.isnan(a)]
    return (float(np.mean(aucs)), float(np.percentile(aucs, 2.5)),
            float(np.percentile(aucs, 97.5)), aucs)


# -----------------------------------------------------------------------
# Extract winner params from ALLtrials CSV
# -----------------------------------------------------------------------
def _get_winner_params(alltrials_path: Path, intra_no: str) -> tuple[dict, pd.Series]:
    df = pd.read_csv(alltrials_path)
    row = df[df["intra_no"] == intra_no].sort_values("AUC_CV", ascending=False).iloc[0]
    # Collect all params_* columns
    param_cols = [c for c in df.columns if c.startswith("params_")]
    params = {}
    for col in param_cols:
        key = col.replace("params_", "")
        val = row[col]
        if pd.notna(val):
            params[key] = val
    return params, row


# -----------------------------------------------------------------------
# Reconstruct features for one pipeline
# -----------------------------------------------------------------------
def reconstruct_pipeline(
    modality: str,
    rank: int,
    intra_no: str,
    alltrials_path: Path,
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    pipe_def: dict,
    global_relieff_order: list[str] | None = None,
) -> tuple[list[str], dict, pd.Series]:
    params, row = _get_winner_params(alltrials_path, intra_no)
    print(f"  {intra_no}: AUC_CV={row['AUC_CV']:.4f}, Fea={int(row['Fea'])}, "
          f"params={params}")
    features = _run_pipeline(
        X_tr, y_tr,
        pipe_def["S1"], pipe_def["S2"], pipe_def["S3"],
        params, global_relieff_order,
    )
    print(f"    -> Reconstructed {len(features)} features")
    return features, params, row


# -----------------------------------------------------------------------
# Bootstrap evaluation + CSV save for one pipeline
# -----------------------------------------------------------------------
def evaluate_and_save(
    label: str,
    intra_no: str,
    features: list[str],
    auc_cv_optuna: float,
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_te: pd.DataFrame,
    y_te: np.ndarray,
    X_ext: pd.DataFrame,
    y_ext: np.ndarray,
    source: str = "reconstructed",
) -> dict:
    if not features:
        print(f"    [WARNING] No features for {intra_no} — skipping eval")
        return {}

    f_te  = [f for f in features if f in X_tr.columns and f in X_te.columns]
    f_ext = [f for f in features if f in X_tr.columns and f in X_ext.columns]

    cv_mean, cv_std = _cv_auc(X_tr[f_te].values, y_tr)
    test_pt = _logit_auc(X_tr[f_te].values, y_tr, X_te[f_te].values, y_te)
    ext_pt  = _logit_auc(X_tr[f_ext].values, y_tr, X_ext[f_ext].values, y_ext)

    t_mean, t_lo, t_hi, t_aucs = _bootstrap(
        X_tr[f_te].values, y_tr, X_te[f_te].values, y_te)
    e_mean, e_lo, e_hi, e_aucs = _bootstrap(
        X_tr[f_ext].values, y_tr, X_ext[f_ext].values, y_ext)

    print(f"    CV5={cv_mean:.4f}±{cv_std:.4f} | Test={test_pt:.4f} "
          f"[{t_lo:.4f},{t_hi:.4f}] | Ext={ext_pt:.4f} [{e_lo:.4f},{e_hi:.4f}]")

    # Feature CSV
    feat_df = pd.DataFrame({
        "Rank": range(1, len(features) + 1),
        "Feature": features,
        "Modality": label,
        "intra_no": intra_no,
        "Source": source,
        "AUC_CV_optuna": auc_cv_optuna,
        "CV5_mean": round(cv_mean, 4),
        "CV5_std": round(cv_std, 4),
        "Test_pointest": round(test_pt, 4),
        "Ext_pointest": round(ext_pt, 4),
        "Test_boot_mean": round(t_mean, 4),
        "Test_CI_lo": round(t_lo, 4),
        "Test_CI_hi": round(t_hi, 4),
        "Ext_boot_mean": round(e_mean, 4),
        "Ext_CI_lo": round(e_lo, 4),
        "Ext_CI_hi": round(e_hi, 4),
        "N_features": len(features),
        "N_bootstrap": N_BOOT,
    })
    suffix = f"_{source}" if source != "reconstructed" else ""
    out_feat = OUTPUT_DIR / f"T2_{intra_no}{suffix}.csv"
    feat_df.to_csv(out_feat, index=False)

    # Bootstrap CSV
    maxlen = max(len(t_aucs), len(e_aucs))
    boot_df = pd.DataFrame({
        "rep":      range(1, maxlen + 1),
        "Test_AUC": t_aucs + [float("nan")] * (maxlen - len(t_aucs)),
        "Ext_AUC":  e_aucs + [float("nan")] * (maxlen - len(e_aucs)),
    })
    out_boot = OUTPUT_DIR / f"T2_{intra_no}{suffix}_bootstrap.csv"
    boot_df.to_csv(out_boot, index=False)

    return {
        "intra_no": intra_no, "source": source, "n_features": len(features),
        "auc_cv_optuna": auc_cv_optuna,
        "cv_mean": round(cv_mean, 4), "cv_std": round(cv_std, 4),
        "test_pt": round(test_pt, 4),
        "t_mean": round(t_mean, 4), "t_lo": round(t_lo, 4), "t_hi": round(t_hi, 4),
        "ext_pt": round(ext_pt, 4),
        "e_mean": round(e_mean, 4), "e_lo": round(e_lo, 4), "e_hi": round(e_hi, 4),
        "t_aucs": t_aucs, "e_aucs": e_aucs,
    }


# -----------------------------------------------------------------------
# Combined evaluation (merge CT + PT on PatientID)
# -----------------------------------------------------------------------
def evaluate_combined(
    ct_features: list[str],
    pt_features: list[str],
    source: str = "reconstructed",
) -> dict:
    print(f"\n  Building combined CT+PT dataset (merge on PatientID)...")

    def _merge_split(ct_file, pt_file, ct_feats, pt_feats):
        df_ct = pd.read_csv(ct_file)
        df_pt = pd.read_csv(pt_file)
        ct_avail = [f for f in ct_feats if f in df_ct.columns]
        pt_avail = [f for f in pt_feats if f in df_pt.columns]
        ct_sel = df_ct[ct_avail + ["PatientID", "HPV_binary"]].copy()
        pt_sel = df_pt[pt_avail + ["PatientID"]].copy()
        return ct_sel.merge(pt_sel, on="PatientID", how="inner")

    merged_tr  = _merge_split(CT_TRAIN, PT_TRAIN, ct_features, pt_features)
    merged_te  = _merge_split(CT_TEST,  PT_TEST,  ct_features, pt_features)
    merged_ext = _merge_split(CT_EXT,   PT_EXT,   ct_features, pt_features)

    overlap = [f for f in pt_features if f in ct_features]
    comb_feats = list(dict.fromkeys(ct_features + [f for f in pt_features if f not in ct_features]))
    comb_feats = [f for f in comb_feats if
                  f in merged_tr.columns and f in merged_te.columns and f in merged_ext.columns]

    print(f"    CT={len(ct_features)}, PT={len(pt_features)}, overlap={len(overlap)}, "
          f"combined={len(comb_feats)}")
    print(f"    Patients: Train={len(merged_tr)}, Test={len(merged_te)}, Ext={len(merged_ext)}")

    def _prep(df, ref_df=None):
        X = df[comb_feats].replace([np.inf, -np.inf], np.nan)
        y = df["HPV_binary"].values.astype(int)
        return X, y

    X_tr_c, y_tr_c = _prep(merged_tr)
    X_tr_c = X_tr_c.fillna(X_tr_c.median())
    X_te_c, y_te_c = _prep(merged_te)
    X_te_c = X_te_c.fillna(X_tr_c.median())
    X_ext_c, y_ext_c = _prep(merged_ext)
    X_ext_c = X_ext_c.fillna(X_tr_c.median())

    cv_mean, cv_std = _cv_auc(X_tr_c.values, y_tr_c)
    test_pt = _logit_auc(X_tr_c.values, y_tr_c, X_te_c.values, y_te_c)
    ext_pt  = _logit_auc(X_tr_c.values, y_tr_c, X_ext_c.values, y_ext_c)

    print(f"    CV5={cv_mean:.4f}±{cv_std:.4f} | Test={test_pt:.4f} | Ext={ext_pt:.4f}")
    print(f"    Bootstrap Test...")
    t_mean, t_lo, t_hi, t_aucs = _bootstrap(X_tr_c.values, y_tr_c, X_te_c.values, y_te_c)
    print(f"      mean={t_mean:.4f} [{t_lo:.4f},{t_hi:.4f}]")
    print(f"    Bootstrap Ext...")
    e_mean, e_lo, e_hi, e_aucs = _bootstrap(X_tr_c.values, y_tr_c, X_ext_c.values, y_ext_c)
    print(f"      mean={e_mean:.4f} [{e_lo:.4f},{e_hi:.4f}]")

    # Feature CSV
    suffix = f"_{source}" if source != "reconstructed" else ""
    feat_df = pd.DataFrame({
        "Rank":     range(1, len(comb_feats) + 1),
        "Feature":  comb_feats,
        "Source_modality": ["CT" if f in ct_features else "PT" for f in comb_feats],
        "intra_no": f"CT_S3_2+PT_S3_5",
        "Source":   source,
        "CV5_mean": round(cv_mean, 4), "CV5_std": round(cv_std, 4),
        "Test_pointest": round(test_pt, 4), "Ext_pointest": round(ext_pt, 4),
        "Test_boot_mean": round(t_mean, 4),
        "Test_CI_lo": round(t_lo, 4), "Test_CI_hi": round(t_hi, 4),
        "Ext_boot_mean": round(e_mean, 4),
        "Ext_CI_lo": round(e_lo, 4), "Ext_CI_hi": round(e_hi, 4),
        "N_bootstrap": N_BOOT,
    })
    out_feat = OUTPUT_DIR / f"T2_combined_CT_PT{suffix}.csv"
    feat_df.to_csv(out_feat, index=False)

    maxlen = max(len(t_aucs), len(e_aucs))
    boot_df = pd.DataFrame({
        "rep":      range(1, maxlen + 1),
        "Test_AUC": t_aucs + [float("nan")] * (maxlen - len(t_aucs)),
        "Ext_AUC":  e_aucs + [float("nan")] * (maxlen - len(e_aucs)),
    })
    (OUTPUT_DIR / f"T2_combined_CT_PT{suffix}_bootstrap.csv").write_text("")
    boot_df.to_csv(OUTPUT_DIR / f"T2_combined_CT_PT{suffix}_bootstrap.csv", index=False)

    print(f"    Saved: {out_feat.name}")
    return {
        "intra_no": "CT_S3_2+PT_S3_5", "source": source,
        "n_features": len(comb_feats),
        "cv_mean": round(cv_mean, 4), "cv_std": round(cv_std, 4),
        "test_pt": round(test_pt, 4),
        "t_mean": round(t_mean, 4), "t_lo": round(t_lo, 4), "t_hi": round(t_hi, 4),
        "ext_pt": round(ext_pt, 4),
        "e_mean": round(e_mean, 4), "e_lo": round(e_lo, 4), "e_hi": round(e_hi, 4),
    }


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("Task 2 Final Feature Extraction — Full Reconstruction (Task 1 method)")
    print("=" * 70)

    # ===================================================================
    # STEP 1: Load data
    # ===================================================================
    print("\n[1/6] Loading CT data...")
    X_ct_tr, y_ct_tr, X_ct_te, y_ct_te, X_ct_ext, y_ct_ext, _ = _load_split(
        CT_TRAIN, CT_TEST, CT_EXT)
    print(f"  Train={len(y_ct_tr)} Test={len(y_ct_te)} Ext={len(y_ct_ext)}")

    print("[1/6] Loading PT data...")
    X_pt_tr, y_pt_tr, X_pt_te, y_pt_te, X_pt_ext, y_pt_ext, _ = _load_split(
        PT_TRAIN, PT_TEST, PT_EXT)
    print(f"  Train={len(y_pt_tr)} Test={len(y_pt_te)} Ext={len(y_pt_ext)}")

    # ===================================================================
    # STEP 2: Precompute ReliefF for CT (ranks 2 and 12 use ReliefF S1)
    # ===================================================================
    print("\n[2/6] Precomputing ReliefF on CT X_train...")
    if RELIEFF_OK:
        n_neighbors = min(50, len(y_ct_tr) - 1)
        rf_global = ReliefF(n_features_to_select=X_ct_tr.shape[1], n_neighbors=n_neighbors)
        rf_global.fit(X_ct_tr.values, y_ct_tr)
        ct_relieff_order = [X_ct_tr.columns[i]
                            for i in np.argsort(rf_global.feature_importances_)[::-1]]
        print(f"  ReliefF done. Top-3: {ct_relieff_order[:3]}")
    else:
        ct_relieff_order = None
        print("  [WARNING] skrebate not installed; ReliefF will be skipped for CT_S3_2/CT_S3_12")

    # ===================================================================
    # STEP 3: Reconstruct all CT pipeline features
    # ===================================================================
    print("\n[3/6] Reconstructing CT pipeline winners...")
    CT_WINNER_ORDER = [2, 12, 1, 3, 6, 8, 7]  # descending AUC_CV order
    CT_INTRA_MAP = {
        1:  "CT_S3_1_344",
        2:  "CT_S3_2_460",
        3:  "CT_S3_3_426",
        6:  "CT_S3_6_215",
        7:  "CT_S3_7_486",
        8:  "CT_S3_8_78",
        12: "CT_S3_12_142",
    }
    CT_AUC_CV_MAP = {
        1:  0.7922, 2: 0.8244, 3: 0.7406,
        6:  0.7200, 7: 0.6722, 8: 0.7161, 12: 0.8017,
    }

    ct_reconstructed: dict[int, list[str]] = {}
    ct_all_results: list[dict] = []

    for rank in CT_WINNER_ORDER:
        intra_no = CT_INTRA_MAP[rank]
        print(f"\n  [CT_S3_{rank}] {intra_no}")
        pipe_def = CT_PIPELINES[rank]
        try:
            features, params, row = reconstruct_pipeline(
                "CT", rank, intra_no, CT_ALLTRIALS,
                X_ct_tr, y_ct_tr, pipe_def,
                global_relieff_order=ct_relieff_order,
            )
            ct_reconstructed[rank] = features

            # Save lightweight per-pipeline feature CSV (no bootstrap for non-primary)
            feat_df = pd.DataFrame({
                "Feature":   features,
                "Modality":  "CT",
                "intra_no":  intra_no,
                "Rank":      range(1, len(features) + 1),
                "Pipeline":  f"{pipe_def['S1']}->{pipe_def['S2']}->{pipe_def['S3']}",
                "AUC_CV":    CT_AUC_CV_MAP[rank],
                "Source":    "reconstructed",
            })
            feat_df.to_csv(OUTPUT_DIR / f"T2_{intra_no}_reconstructed_features.csv", index=False)
        except Exception as e:
            print(f"    [ERROR] {e}")
            ct_reconstructed[rank] = []

    # Primary CT winner: full bootstrap eval for reconstructed features
    print(f"\n  [CT PRIMARY] Full bootstrap eval for CT_S3_2_460 (reconstructed)...")
    ct2_recon_feats = ct_reconstructed.get(2, [])
    ct2_recon_res = evaluate_and_save(
        "CT", "CT_S3_2_460", ct2_recon_feats, CT_AUC_CV_MAP[2],
        X_ct_tr, y_ct_tr, X_ct_te, y_ct_te, X_ct_ext, y_ct_ext,
        source="reconstructed",
    )
    ct_all_results.append(ct2_recon_res)

    # ===================================================================
    # STEP 4: Reconstruct all PT pipeline features
    # ===================================================================
    print("\n[4/6] Reconstructing PT pipeline winners...")
    PT_WINNER_ORDER = [5, 10, 6, 3, 13, 4, 7, 1, 15]  # descending AUC_CV order
    PT_INTRA_MAP = {
        1:  "PT_S3_1_309",
        3:  "PT_S3_3_475",
        4:  "PT_S3_4_249",
        5:  "PT_S3_5_531",
        6:  "PT_S3_6_266",
        7:  "PT_S3_7_527",
        10: "PT_S3_10_390",
        13: "PT_S3_13_266",
        15: "PT_S3_15_235",
    }
    PT_AUC_CV_MAP = {
        1: 0.7706, 3: 0.7767, 4: 0.7633, 5: 0.8367,
        6: 0.7839, 7: 0.7828, 10: 0.7872, 13: 0.7411, 15: 0.7344,
    }

    pt_reconstructed: dict[int, list[str]] = {}
    pt_all_results: list[dict] = []

    for rank in PT_WINNER_ORDER:
        intra_no = PT_INTRA_MAP[rank]
        print(f"\n  [PT_S3_{rank}] {intra_no}")
        pipe_def = PT_PIPELINES[rank]
        try:
            features, params, row = reconstruct_pipeline(
                "PT", rank, intra_no, PT_ALLTRIALS,
                X_pt_tr, y_pt_tr, pipe_def,
                global_relieff_order=None,
            )
            pt_reconstructed[rank] = features

            feat_df = pd.DataFrame({
                "Feature":   features,
                "Modality":  "PT",
                "intra_no":  intra_no,
                "Rank":      range(1, len(features) + 1),
                "Pipeline":  f"{pipe_def['S1']}->{pipe_def['S2']}->{pipe_def['S3']}",
                "AUC_CV":    PT_AUC_CV_MAP[rank],
                "Source":    "reconstructed",
            })
            feat_df.to_csv(OUTPUT_DIR / f"T2_{intra_no}_reconstructed_features.csv", index=False)
        except Exception as e:
            print(f"    [ERROR] {e}")
            pt_reconstructed[rank] = []

    # Primary PT winner: full bootstrap eval for reconstructed features
    print(f"\n  [PT PRIMARY] Full bootstrap eval for PT_S3_5_531 (reconstructed)...")
    pt5_recon_feats = pt_reconstructed.get(5, [])
    pt5_recon_res = evaluate_and_save(
        "PT", "PT_S3_5_531", pt5_recon_feats, PT_AUC_CV_MAP[5],
        X_pt_tr, y_pt_tr, X_pt_te, y_pt_te, X_pt_ext, y_pt_ext,
        source="reconstructed",
    )
    pt_all_results.append(pt5_recon_res)

    # ===================================================================
    # STEP 5: Combined CT+PT (reconstructed)
    # ===================================================================
    print("\n[5/6] Combined CT+PT evaluation (reconstructed)...")
    comb_recon_res = evaluate_combined(ct2_recon_feats, pt5_recon_feats, source="reconstructed")

    # ===================================================================
    # STEP 6: Stored vs reconstructed overlap analysis + comparison table
    # ===================================================================
    print("\n[6/6] Overlap analysis and comparison table...")

    # CT_S3_2 overlap
    ct_stored_set = set(CT_STORED_FEATURES)
    ct_recon_set  = set(ct2_recon_feats)
    ct_overlap    = ct_stored_set & ct_recon_set
    ct_stored_only = ct_stored_set - ct_recon_set
    ct_recon_only  = ct_recon_set  - ct_stored_set

    print(f"\n  CT_S3_2_460 overlap analysis:")
    print(f"    Stored ({len(CT_STORED_FEATURES)}): {sorted(CT_STORED_FEATURES)}")
    print(f"    Reconstructed ({len(ct2_recon_feats)}): {sorted(ct2_recon_feats)}")
    print(f"    Overlap ({len(ct_overlap)}): {sorted(ct_overlap)}")
    print(f"    Stored-only ({len(ct_stored_only)}): {sorted(ct_stored_only)}")
    print(f"    Recon-only ({len(ct_recon_only)}): {sorted(ct_recon_only)}")

    # PT_S3_5 overlap
    pt_stored_set = set(PT_STORED_FEATURES)
    pt_recon_set  = set(pt5_recon_feats)
    pt_overlap    = pt_stored_set & pt_recon_set
    pt_stored_only = pt_stored_set - pt_recon_set
    pt_recon_only  = pt_recon_set  - pt_stored_set

    print(f"\n  PT_S3_5_531 overlap analysis:")
    print(f"    Stored ({len(PT_STORED_FEATURES)}): {sorted(PT_STORED_FEATURES)}")
    print(f"    Reconstructed ({len(pt5_recon_feats)}): {sorted(pt5_recon_feats)}")
    print(f"    Overlap ({len(pt_overlap)}): {sorted(pt_overlap)}")
    print(f"    Stored-only ({len(pt_stored_only)}): {sorted(pt_stored_only)}")
    print(f"    Recon-only ({len(pt_recon_only)}): {sorted(pt_recon_only)}")

    # Overlap CSV
    overlap_rows = []
    for f in sorted(ct_stored_set | ct_recon_set):
        overlap_rows.append({
            "Feature": f, "Modality": "CT", "intra_no": "CT_S3_2_460",
            "In_stored": f in ct_stored_set, "In_reconstructed": f in ct_recon_set,
            "Status": ("both" if f in ct_overlap
                       else "stored_only" if f in ct_stored_set else "recon_only"),
        })
    for f in sorted(pt_stored_set | pt_recon_set):
        overlap_rows.append({
            "Feature": f, "Modality": "PT", "intra_no": "PT_S3_5_531",
            "In_stored": f in pt_stored_set, "In_reconstructed": f in pt_recon_set,
            "Status": ("both" if f in pt_overlap
                       else "stored_only" if f in pt_stored_set else "recon_only"),
        })
    overlap_df = pd.DataFrame(overlap_rows)
    overlap_df.to_csv(OUTPUT_DIR / "T2_stored_vs_reconstructed_overlap.csv", index=False)
    print(f"\n  Saved overlap CSV: T2_stored_vs_reconstructed_overlap.csv")

    # --- Stored features performance (for comparison) ---
    print("\n  Evaluating stored features (CT_S3_2 and PT_S3_5)...")
    ct2_stored_res = evaluate_and_save(
        "CT", "CT_S3_2_460", CT_STORED_FEATURES, CT_AUC_CV_MAP[2],
        X_ct_tr, y_ct_tr, X_ct_te, y_ct_te, X_ct_ext, y_ct_ext,
        source="stored",
    )
    pt5_stored_res = evaluate_and_save(
        "PT", "PT_S3_5_531", PT_STORED_FEATURES, PT_AUC_CV_MAP[5],
        X_pt_tr, y_pt_tr, X_pt_te, y_pt_te, X_pt_ext, y_pt_ext,
        source="stored",
    )

    print("\n  Evaluating stored combined CT+PT...")
    comb_stored_res = evaluate_combined(CT_STORED_FEATURES, PT_STORED_FEATURES, source="stored")

    # ===================================================================
    # Comparison table
    # ===================================================================
    def _fmt(res, split):
        if not res:
            return "N/A", "N/A"
        if split == "CV5":
            return f"{res['cv_mean']:.4f}", f"±{res['cv_std']:.4f}"
        elif split == "Test":
            return f"{res['test_pt']:.4f}", f"[{res['t_lo']:.4f},{res['t_hi']:.4f}]"
        else:  # Ext
            return f"{res['ext_pt']:.4f}", f"[{res['e_lo']:.4f},{res['e_hi']:.4f}]"

    print("\n" + "=" * 90)
    print("COMPARISON TABLE: Stored vs Reconstructed features (primary winners)")
    print("=" * 90)
    header = f"{'Model':<28} {'N_fea':<7} {'CV5 AUC':<10} {'Test AUC':<10} {'Test 95%CI':<20} {'Ext AUC':<10} {'Ext 95%CI'}"
    print(header)
    print("-" * 90)

    rows_data = [
        ("CT_S3_2 [stored 11]",       ct2_stored_res),
        ("CT_S3_2 [reconstructed]",    ct2_recon_res),
        ("PT_S3_5 [stored 8]",         pt5_stored_res),
        ("PT_S3_5 [reconstructed]",    pt5_recon_res),
        ("CT+PT [stored]",             comb_stored_res),
        ("CT+PT [reconstructed]",      comb_recon_res),
    ]
    comparison_records = []
    for label, res in rows_data:
        if not res:
            print(f"  {label:<26} — no result")
            continue
        n_fea = res.get("n_features", "?")
        cv_v, cv_s = _fmt(res, "CV5")
        te_v, te_ci = _fmt(res, "Test")
        ex_v, ex_ci = _fmt(res, "Ext")
        print(f"  {label:<26} {str(n_fea):<7} {cv_v:<10} {te_v:<10} {te_ci:<20} {ex_v:<10} {ex_ci}")
        comparison_records.append({
            "Model": label,
            "N_features": n_fea,
            "CV5_AUC": res.get("cv_mean"),
            "CV5_std": res.get("cv_std"),
            "Test_AUC": res.get("test_pt"),
            "Test_CI_lo": res.get("t_lo"),
            "Test_CI_hi": res.get("t_hi"),
            "Ext_AUC": res.get("ext_pt"),
            "Ext_CI_lo": res.get("e_lo"),
            "Ext_CI_hi": res.get("e_hi"),
        })

    pd.DataFrame(comparison_records).to_csv(
        OUTPUT_DIR / "T2_stored_vs_reconstructed_comparison.csv", index=False)
    print("\nSaved: T2_stored_vs_reconstructed_comparison.csv")

    # ===================================================================
    # All-pipelines summary
    # ===================================================================
    print("\n" + "=" * 70)
    print("ALL PIPELINE RECONSTRUCTION SUMMARY")
    print("=" * 70)
    print(f"{'intra_no':<20} {'N_fea_recon':<14} {'AUC_CV_optuna'}")
    print("-" * 55)
    for rank in CT_WINNER_ORDER:
        intra_no = CT_INTRA_MAP[rank]
        n_f = len(ct_reconstructed.get(rank, []))
        print(f"  {intra_no:<18} {n_f:<14} {CT_AUC_CV_MAP[rank]:.4f}")
    for rank in PT_WINNER_ORDER:
        intra_no = PT_INTRA_MAP[rank]
        n_f = len(pt_reconstructed.get(rank, []))
        print(f"  {intra_no:<18} {n_f:<14} {PT_AUC_CV_MAP[rank]:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
