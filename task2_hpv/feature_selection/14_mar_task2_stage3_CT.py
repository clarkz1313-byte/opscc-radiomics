# -*- coding: utf-8 -*-
"""
Task 2 Stage 3: CT three-step feature-selection pipelines (exploratory).

Adds a third selector (S3) to each Stage 2 shortlisted pipeline.
All three steps (S1, S2, S3) are recomputed end-to-end inside each outer fold.

CT Stage 3 is exploratory / limited-scope only: Stage 2 signal is marginal
(only CT_S2_1 clearly beats the Stage 1 global best).

Usage:
    cd "D:/Uppsala thesis"
    python Mar_2026_task2/14_mar_task2_stage3_CT.py
"""

import json
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning,ignore::UserWarning")

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings(
    "ignore",
    message=r".*'penalty' was deprecated in version 1\.8.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*sklearn\.utils\.parallel\.delayed.*sklearn\.utils\.parallel\.Parallel.*",
    category=UserWarning,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pearsonr
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from statsmodels.stats.multitest import multipletests

from fs_task2_utils import (
    MRMR_OK,
    RELIEFF_OK,
    XGB_OK,
    correlation_filter,
    elasticnet_logistic_selection,
    evaluate_auc_test,
    lasso_logistic_selection,
    nested_cv_auc,
)

if RELIEFF_OK:
    from skrebate import ReliefF

if MRMR_OK:
    from mrmr import mrmr_classif

if XGB_OK:
    from xgboost import XGBClassifier


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
MODALITY = "CT"
PROTOCOL_VERSION = "stage3_end_to_end_v1"
SAVE_EVERY = 10

RESULTS_DIR = SCRIPT_DIR / "13_mar_t2_fs_results"
DATA_DIR = SCRIPT_DIR / "12_mar_task2_rad_data"

TRAIN_FILE = DATA_DIR / "13_mar_task2_CT_primary_train.csv"
TEST_FILE = DATA_DIR / "13_mar_task2_CT_primary_test.csv"

OUT_CSV = RESULTS_DIR / "14_mar_task2_stage3_CT_result.csv"
OUT_PKL = RESULTS_DIR / "14_mar_task2_stage3_CT_features.pkl"
OUT_META = RESULTS_DIR / "14_mar_task2_stage3_CT_metadata.json"
CHECKPOINT = RESULTS_DIR / "14_mar_task2_stage3_CT_checkpoint.pkl"

BASELINE_AUC_S1 = 0.6328   # CT Stage 1 baseline (CT_2, C1 LASSO-60)
BASELINE_FEAT_S1 = 60

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE_COLS = [
    "PatientID", "HPV_binary", "Relapse", "RFS",
    "Age", "Gender_Male", "Treatment_CRT", "prefix",
]

np.random.seed(SEED)


# ============================================================
# STAGE 2 SHORTLIST — hardcoded (12 pipelines, AUC2 >= 0.60)
# Source: 13_mar_t2_fs_results/13_mar_task2_stage2_CT_result_processed.csv
# Processed CSV is a review artifact only; this list is authoritative.
# ============================================================

S2_SHORTLIST = [
    # s2_base_rank, category, s1_rank, s1_label, s1_selector_key, s1_selector_param,
    #   s2_method, s2_k, s2_label, auc2, fea2
    {
        "s2_base_rank": "CT_S2_1",
        "category": "A->D",
        "s1_rank": "CT_6",
        "s1_label": "A7: ANOVA (top 100)",
        "s1_key": "A7", "s1_param": 100,
        "s2_method": "D3", "s2_k": 20,
        "s2_label": "D3: GB PermImp (top 20)",
        "auc2": 0.6889, "fea2": 20,
    },
    {
        "s2_base_rank": "CT_S2_2",
        "category": "B->C",
        "s1_rank": "CT_30",
        "s1_label": "B1: Corr filter (r<0.85)",
        "s1_key": "B1", "s1_param": 0.85,
        "s2_method": "C1", "s2_k": 30,
        "s2_label": "C1: LASSO-Logistic (target 30)",
        "auc2": 0.6767, "fea2": 30,
    },
    {
        "s2_base_rank": "CT_S2_3",
        "category": "C->D",
        "s1_rank": "CT_4",
        "s1_label": "C1: LASSO-Logistic (target 30)",
        "s1_key": "C1", "s1_param": 30,
        "s2_method": "D1", "s2_k": 20,
        "s2_label": "D1: RF PermImp (top 20)",
        "auc2": 0.6767, "fea2": 20,
    },
    {
        "s2_base_rank": "CT_S2_4",
        "category": "A->D",
        "s1_rank": "CT_6",
        "s1_label": "A7: ANOVA (top 100)",
        "s1_key": "A7", "s1_param": 100,
        "s2_method": "D1", "s2_k": 20,
        "s2_label": "D1: RF PermImp (top 20)",
        "auc2": 0.6744, "fea2": 20,
    },
    {
        "s2_base_rank": "CT_S2_6",
        "category": "B->C",
        "s1_rank": "CT_5",
        "s1_label": "B3: mRMR classif (top 50)",
        "s1_key": "B3", "s1_param": 50,
        "s2_method": "C2", "s2_k": 30,
        "s2_label": "C2: ElasticNet-Logistic (target 30)",
        "auc2": 0.6467, "fea2": 29,
    },
    {
        "s2_base_rank": "CT_S2_7",
        "category": "A->D",
        "s1_rank": "CT_7",
        "s1_label": "A6: ReliefF (top 50)",
        "s1_key": "A6", "s1_param": 50,
        "s2_method": "D1", "s2_k": 30,
        "s2_label": "D1: RF PermImp (top 30)",
        "auc2": 0.6383, "fea2": 30,
    },
    {
        "s2_base_rank": "CT_S2_9",
        "category": "B->D",
        "s1_rank": "CT_5",
        "s1_label": "B3: mRMR classif (top 50)",
        "s1_key": "B3", "s1_param": 50,
        "s2_method": "D1", "s2_k": 20,
        "s2_label": "D1: RF PermImp (top 20)",
        "auc2": 0.6278, "fea2": 20,
    },
    {
        "s2_base_rank": "CT_S2_11",
        "category": "B->C",
        "s1_rank": "CT_5",
        "s1_label": "B3: mRMR classif (top 50)",
        "s1_key": "B3", "s1_param": 50,
        "s2_method": "C1", "s2_k": 20,
        "s2_label": "C1: LASSO-Logistic (target 20)",
        "auc2": 0.6256, "fea2": 20,
    },
    {
        "s2_base_rank": "CT_S2_13",
        "category": "C->D",
        "s1_rank": "CT_8",
        "s1_label": "C2: ElasticNet-Logistic (target 60)",
        "s1_key": "C2", "s1_param": 60,
        "s2_method": "D1", "s2_k": 20,
        "s2_label": "D1: RF PermImp (top 20)",
        "auc2": 0.6233, "fea2": 20,
    },
    {
        "s2_base_rank": "CT_S2_16",
        "category": "A->C",
        "s1_rank": "CT_6",
        "s1_label": "A7: ANOVA (top 100)",
        "s1_key": "A7", "s1_param": 100,
        "s2_method": "C1", "s2_k": 20,
        "s2_label": "C1: LASSO-Logistic (target 20)",
        "auc2": 0.6189, "fea2": 19,
    },
    {
        "s2_base_rank": "CT_S2_18",
        "category": "A->C",
        "s1_rank": "CT_7",
        "s1_label": "A6: ReliefF (top 50)",
        "s1_key": "A6", "s1_param": 50,
        "s2_method": "C2", "s2_k": 20,
        "s2_label": "C2: ElasticNet-Logistic (target 20)",
        "auc2": 0.6128, "fea2": 20,
    },
    {
        "s2_base_rank": "CT_S2_19",
        "category": "B->C",
        "s1_rank": "CT_3",
        "s1_label": "B3: mRMR classif (top 30)",
        "s1_key": "B3", "s1_param": 30,
        "s2_method": "C1", "s2_k": 20,
        "s2_label": "C1: LASSO-Logistic (target 20)",
        "auc2": 0.6094, "fea2": 19,
    },
]

# S3 k values: {10, 20}; smaller than S2 {20, 30}
S3_K_VALUES = [10, 20]


# ============================================================
# HELPERS
# ============================================================

def format_seconds(seconds: float) -> str:
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def selector_mwu(X_f: pd.DataFrame, y_f: np.ndarray, use_fdr: bool = False) -> list[str]:
    pos_mask = y_f == 1
    neg_mask = y_f == 0
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return []
    rows = []
    for col in X_f.columns:
        stat, p_val = mannwhitneyu(
            X_f[col].values[pos_mask],
            X_f[col].values[neg_mask],
            alternative="two-sided",
        )
        rows.append({"feature": col, "p": p_val})
    df = pd.DataFrame(rows)
    if use_fdr:
        _, p_fdr, _, _ = multipletests(df["p"].values, method="fdr_bh")
        df["p_sel"] = p_fdr
    else:
        df["p_sel"] = df["p"]
    return df[df["p_sel"] < 0.05]["feature"].tolist()


def selector_pearson(X_f: pd.DataFrame, y_f: np.ndarray, r_thresh: float) -> list[str]:
    selected = []
    for col in X_f.columns:
        try:
            r_val, p_val = pearsonr(X_f[col].values, y_f)
        except Exception:
            continue
        if np.isnan(r_val) or np.isnan(p_val):
            continue
        if abs(r_val) > r_thresh and p_val < 0.05:
            selected.append(col)
    return selected


def selector_relief(X_f: pd.DataFrame, y_f: np.ndarray, k: int) -> list[str]:
    if not RELIEFF_OK:
        raise RuntimeError("skrebate is not available.")
    n_neighbors = max(1, min(50, len(y_f) - 1))
    relief = ReliefF(n_features_to_select=min(k, X_f.shape[1]), n_neighbors=n_neighbors)
    relief.fit(X_f.values, y_f)
    order = np.argsort(relief.feature_importances_)[::-1][: min(k, X_f.shape[1])]
    return [X_f.columns[i] for i in order]


def selector_anova(X_f: pd.DataFrame, y_f: np.ndarray, k: int) -> list[str]:
    scores, _ = f_classif(X_f.values, y_f)
    scores = np.nan_to_num(scores, nan=-np.inf)
    order = np.argsort(scores)[::-1][: min(k, X_f.shape[1])]
    return [X_f.columns[i] for i in order]


def selector_b3_mrmr(X_f: pd.DataFrame, y_f: np.ndarray, k: int) -> list[str]:
    if not MRMR_OK:
        raise RuntimeError("mrmr-selection is not available.")
    return list(mrmr_classif(X=X_f, y=pd.Series(y_f), K=min(k, X_f.shape[1])))


def selector_corr(X_f: pd.DataFrame, _: np.ndarray, threshold: float = 0.85) -> list[str]:
    return correlation_filter(X_f, threshold)


def selector_lasso(X_f: pd.DataFrame, y_f: np.ndarray, k: int) -> list[str]:
    return lasso_logistic_selection(X_f, y_f, target_features=k, random_state=SEED)


def selector_en(X_f: pd.DataFrame, y_f: np.ndarray, k: int) -> list[str]:
    return elasticnet_logistic_selection(X_f, y_f, target_features=k, random_state=SEED)


def _holdout_split(X_f: pd.DataFrame, y_f: np.ndarray):
    return train_test_split(
        X_f.values, y_f, test_size=0.25, random_state=SEED, stratify=y_f,
    )


def selector_rf_perm(X_f: pd.DataFrame, y_f: np.ndarray, k: int) -> list[str]:
    X_sub_tr, X_sub_val, y_sub_tr, y_sub_val = _holdout_split(X_f, y_f)
    clf = RandomForestClassifier(
        n_estimators=500, class_weight="balanced", random_state=SEED, n_jobs=-1,
    )
    clf.fit(X_sub_tr, y_sub_tr)
    imp = permutation_importance(clf, X_sub_val, y_sub_val, n_repeats=10, random_state=SEED, n_jobs=1)
    order = np.argsort(np.nan_to_num(imp.importances_mean, nan=-np.inf))[::-1][: min(k, X_f.shape[1])]
    return [X_f.columns[i] for i in order]


def selector_xgb_perm(X_f: pd.DataFrame, y_f: np.ndarray, k: int) -> list[str]:
    if not XGB_OK:
        raise RuntimeError("xgboost is not available.")
    X_sub_tr, X_sub_val, y_sub_tr, y_sub_val = _holdout_split(X_f, y_f)
    neg = int((y_sub_tr == 0).sum())
    pos = int((y_sub_tr == 1).sum())
    scale_pos = neg / max(pos, 1)
    clf = XGBClassifier(
        n_estimators=100, scale_pos_weight=scale_pos,
        use_label_encoder=False, eval_metric="logloss",
        random_state=SEED, verbosity=0, n_jobs=-1,
    )
    clf.fit(X_sub_tr, y_sub_tr)
    imp = permutation_importance(clf, X_sub_val, y_sub_val, n_repeats=10, random_state=SEED, n_jobs=1)
    order = np.argsort(np.nan_to_num(imp.importances_mean, nan=-np.inf))[::-1][: min(k, X_f.shape[1])]
    return [X_f.columns[i] for i in order]


def selector_gb_perm(X_f: pd.DataFrame, y_f: np.ndarray, k: int) -> list[str]:
    X_sub_tr, X_sub_val, y_sub_tr, y_sub_val = _holdout_split(X_f, y_f)
    clf = GradientBoostingClassifier(n_estimators=200, random_state=SEED)
    clf.fit(X_sub_tr, y_sub_tr)
    imp = permutation_importance(clf, X_sub_val, y_sub_val, n_repeats=10, random_state=SEED, n_jobs=1)
    order = np.argsort(np.nan_to_num(imp.importances_mean, nan=-np.inf))[::-1][: min(k, X_f.shape[1])]
    return [X_f.columns[i] for i in order]


def make_selector(method: str, param):
    """Return a selector function for a given method key and parameter."""
    if method == "A0":
        return lambda X_f, y_f, use_fdr=(param == "fdr"): selector_mwu(X_f, y_f, use_fdr=use_fdr)
    if method == "A2":
        return lambda X_f, y_f, r=param: selector_pearson(X_f, y_f, r)
    if method == "A6":
        return lambda X_f, y_f, k=param: selector_relief(X_f, y_f, k)
    if method == "A7":
        return lambda X_f, y_f, k=param: selector_anova(X_f, y_f, k)
    if method == "B1":
        return lambda X_f, y_f, t=param: selector_corr(X_f, y_f, t)
    if method == "B3":
        return lambda X_f, y_f, k=param: selector_b3_mrmr(X_f, y_f, k)
    if method == "C1":
        return lambda X_f, y_f, k=param: selector_lasso(X_f, y_f, k)
    if method == "C2":
        return lambda X_f, y_f, k=param: selector_en(X_f, y_f, k)
    if method == "D1":
        return lambda X_f, y_f, k=param: selector_rf_perm(X_f, y_f, k)
    if method == "D2":
        return lambda X_f, y_f, k=param: selector_xgb_perm(X_f, y_f, k)
    if method == "D3":
        return lambda X_f, y_f, k=param: selector_gb_perm(X_f, y_f, k)
    raise ValueError(f"Unknown method: {method}")


def compose_three_step_selector(s1_sel, s2_sel, s3_sel):
    def _selector(X_f: pd.DataFrame, y_f: np.ndarray) -> list[str]:
        f1 = s1_sel(X_f, y_f)
        if len(f1) == 0:
            return []
        f2 = s2_sel(X_f[f1].copy(), y_f)
        if len(f2) == 0:
            return []
        return s3_sel(X_f[f2].copy(), y_f)
    return _selector


def method_category(method: str) -> str:
    if method.startswith("C"):
        return "C"
    if method.startswith("D"):
        return "D"
    return method[0]


def format_s3_label(method: str, k: int) -> str:
    labels = {
        "C1": f"C1: LASSO-Logistic (target {k})",
        "C2": f"C2: ElasticNet-Logistic (target {k})",
        "D1": f"D1: RF PermImp (top {k})",
        "D2": f"D2: XGBoost PermImp (top {k})",
        "D3": f"D3: GB PermImp (top {k})",
    }
    return labels[method]


def get_s2_category(s2_method: str) -> str:
    return "C" if s2_method.startswith("C") else "D"


def allowed_s3_methods(s2_base: dict) -> list[tuple[str, int]]:
    """
    Return list of (s3_method, s3_k) pairs allowed for this S2 base.

    Direction rules (from plan Section 4):
    - A->C base: S3 in C (!=S2 method) or D; k in {10,20}
    - A->D base: S3 in C or D (!=S2 method); k in {10,20}
    - B->C base: S3 in C (!=S2 method) or D; k in {10,20}
    - B->D base: S3 in C or D (!=S2 method); k in {10,20}
    - C->D base: S3 in D (!=S2 method) only; k in {10,20}

    Constraints:
    - S3 method != S2 method (no same method repeated)
    - No A or B as S3
    - C->D->C is forbidden (C->D base cannot use C as S3)
    - A->D->C and B->D->C are allowed
    - Guard: skip if fea2 <= s3_k
    """
    cat = s2_base["category"]  # e.g. "A->D", "C->D"
    s2_method = s2_base["s2_method"]
    fea2 = s2_base["fea2"]

    s3_c_methods = [m for m in ["C1", "C2"] if m != s2_method]
    s3_d_methods = [m for m in ["D1", "D2", "D3"] if m != s2_method]
    if not XGB_OK:
        s3_d_methods = [m for m in s3_d_methods if m != "D2"]

    allowed = []

    if cat in ("A->C", "B->C"):
        # S3: C (!=S2) or D
        candidate_methods = s3_c_methods + s3_d_methods
    elif cat in ("A->D", "B->D"):
        # S3: C or D (!=S2)
        candidate_methods = s3_c_methods + s3_d_methods
    elif cat == "C->D":
        # S3: D (!=S2) only — C->D->C is forbidden
        candidate_methods = s3_d_methods
    else:
        candidate_methods = []

    for method in candidate_methods:
        for k in S3_K_VALUES:
            if fea2 > k:  # guard: must have more features entering than k
                allowed.append((method, k))

    return allowed


def build_stage3_pipelines() -> list[dict]:
    pipelines: list[dict] = []
    idx = 1

    for s2_base in S2_SHORTLIST:
        s3_options = allowed_s3_methods(s2_base)
        for s3_method, s3_k in s3_options:
            s2_cat = s2_base["category"]
            s3_cat = method_category(s3_method)
            pattern = f"{s2_cat}->{s3_cat}"

            full_label = (
                f"{s2_base['s1_label']} -> {s2_base['s2_label']} -> "
                f"{format_s3_label(s3_method, s3_k)}"
            )
            pipelines.append({
                "pipeline_id": f"CT_S3_{idx:03d}",
                "pattern": pattern,
                "s1_rank": s2_base["s1_rank"],
                "s2_base_rank": s2_base["s2_base_rank"],
                "s2_base_auc2": s2_base["auc2"],
                "s1_label": s2_base["s1_label"],
                "s2_label": s2_base["s2_label"],
                "s3_label": format_s3_label(s3_method, s3_k),
                "full_label": full_label,
                "s1_key": s2_base["s1_key"],
                "s1_param": s2_base["s1_param"],
                "s2_method": s2_base["s2_method"],
                "s2_k": s2_base["s2_k"],
                "s3_method": s3_method,
                "s3_k": s3_k,
                "fea2_expected": s2_base["fea2"],
            })
            idx += 1

    return pipelines


def load_checkpoint() -> tuple[set[str], list[dict], dict[str, list[str]]]:
    if not CHECKPOINT.exists():
        return set(), [], {}
    with open(CHECKPOINT, "rb") as f:
        ckpt = pickle.load(f)
    if ckpt.get("protocol_version") != PROTOCOL_VERSION:
        print("[RESUME] Ignoring legacy checkpoint from different protocol version")
        return set(), [], {}
    return set(ckpt["done_ids"]), ckpt["results"], ckpt["features"]


def save_checkpoint(done_ids: set[str], results: list[dict], features: dict[str, list[str]]) -> None:
    with open(CHECKPOINT, "wb") as f:
        pickle.dump(
            {
                "protocol_version": PROTOCOL_VERSION,
                "done_ids": sorted(done_ids),
                "results": results,
                "features": features,
            },
            f,
        )


# ============================================================
# MAIN
# ============================================================

print("=" * 80)
print(f"Task 2 Stage 3: {MODALITY} three-step pipelines (exploratory scope)")
print("=" * 80)

for path in [TRAIN_FILE, TEST_FILE]:
    if not path.exists():
        print(f"[ERROR] Missing required input: {path}")
        sys.exit(1)
    print(f"[OK] {path.name}")

print(f"[INFO] Dependency flags: ReliefF={RELIEFF_OK} mRMR={MRMR_OK} XGBoost={XGB_OK}")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

rad_cols = [c for c in train_df.columns if c not in EXCLUDE_COLS]
X_train = train_df[rad_cols].copy()
X_test = test_df[rad_cols].copy()
y_train = train_df["HPV_binary"].values.astype(int)
y_test = test_df["HPV_binary"].values.astype(int)

X_train = X_train.replace([np.inf, -np.inf], np.nan)
train_medians = X_train.median()
X_train = X_train.fillna(train_medians)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(train_medians)

print(
    f"\n[INFO] Data: train={len(train_df)}, test={len(test_df)}, "
    f"radiomics={len(rad_cols)}, S1_baseline_AUC={BASELINE_AUC_S1}"
)

PIPELINES = build_stage3_pipelines()
print(f"[INFO] Built {len(PIPELINES)} Stage 3 pipelines from {len(S2_SHORTLIST)} S2 bases")

done_ids, results, selected_features_storage = load_checkpoint()
if done_ids:
    print(f"[RESUME] Loaded checkpoint with {len(done_ids)} completed pipelines")

start_time = time.time()
completed_since_save = 0

for i, pipeline in enumerate(PIPELINES, start=1):
    pid = pipeline["pipeline_id"]
    if pid in done_ids:
        print(f"[{i:03d}/{len(PIPELINES)}] [SKIP] {pid}")
        continue

    pipeline_start = time.time()

    s1_sel = make_selector(pipeline["s1_key"], pipeline["s1_param"])
    s2_sel = make_selector(pipeline["s2_method"], pipeline["s2_k"])
    s3_sel = make_selector(pipeline["s3_method"], pipeline["s3_k"])
    three_step = compose_three_step_selector(s1_sel, s2_sel, s3_sel)

    try:
        auc3, std3 = nested_cv_auc(X_train, y_train, three_step, random_state=SEED)

        # Compute feature counts on full train set (for reporting)
        f1_full = s1_sel(X_train, y_train)
        f2_full = s2_sel(X_train[f1_full].copy(), y_train) if f1_full else []
        f3_full = s3_sel(X_train[f2_full].copy(), y_train) if f2_full else []

        test3 = evaluate_auc_test(X_train, y_train, X_test, y_test, f3_full)
        fea1 = len(f1_full)
        fea2 = len(f2_full)
        fea3 = len(f3_full)
        gap = test3 - auc3 if not np.isnan(test3) and not np.isnan(auc3) else float("nan")

    except Exception as exc:
        print(f"[{i:03d}/{len(PIPELINES)}] [ERROR] {pid} -> {type(exc).__name__}: {exc}")
        done_ids.add(pid)
        continue

    pipeline_s = time.time() - pipeline_start
    elapsed_s = time.time() - start_time

    result_row = {
        "Pattern": pipeline["pattern"],
        "S1_Rank": pipeline["s1_rank"],
        "S2_Base_Rank": pipeline["s2_base_rank"],
        "Pipeline": pipeline["full_label"],
        "AUC3": auc3,
        "Std3": std3,
        "Delta_AUC_S1": auc3 - BASELINE_AUC_S1,
        "Delta_AUC_S2": auc3 - pipeline["s2_base_auc2"],
        "Test3": test3,
        "Gap": gap,
        "Fea3": fea3,
        "Fea2": fea2,
        "Fea1": fea1,
        "S2_Base_AUC2": pipeline["s2_base_auc2"],
        "Pipeline_s": pipeline_s,
        "Elapsed_s": elapsed_s,
    }
    results.append(result_row)
    selected_features_storage[pid] = f3_full
    done_ids.add(pid)
    completed_since_save += 1

    print(
        f"[{i:03d}/{len(PIPELINES)}] {pid} {pipeline['pattern']} "
        f"AUC3={auc3:.4f} Test3={test3:.4f} Gap={gap:+.4f} "
        f"Fea3={fea3} Pipeline={format_seconds(pipeline_s)} "
        f"Elapsed={format_seconds(elapsed_s)}"
    )

    if completed_since_save >= SAVE_EVERY:
        save_checkpoint(done_ids, results, selected_features_storage)
        completed_since_save = 0

save_checkpoint(done_ids, results, selected_features_storage)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    ["AUC3", "S2_Base_Rank"], ascending=[False, True], na_position="last"
).reset_index(drop=True)
results_df.insert(0, "Rank", [f"CT_S3_{i}" for i in range(1, len(results_df) + 1)])
results_df = results_df[
    [
        "Rank", "Pattern", "S1_Rank", "S2_Base_Rank", "Pipeline",
        "AUC3", "Std3", "Delta_AUC_S1", "Delta_AUC_S2",
        "Test3", "Gap", "Fea3", "Fea2", "Fea1", "S2_Base_AUC2",
        "Pipeline_s", "Elapsed_s",
    ]
]

results_df.to_csv(OUT_CSV, index=False)
with open(OUT_PKL, "wb") as f:
    pickle.dump(selected_features_storage, f)

metadata = {
    "modality": MODALITY,
    "protocol_version": PROTOCOL_VERSION,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "execution_time_s": time.time() - start_time,
    "seed": SEED,
    "train_file": TRAIN_FILE.name,
    "test_file": TEST_FILE.name,
    "stage1_source": "recomputed_from_train",
    "n_train": len(train_df),
    "n_test": len(test_df),
    "n_radiomics_feats": len(rad_cols),
    "baseline_auc_s1": BASELINE_AUC_S1,
    "n_s2_bases": len(S2_SHORTLIST),
    "dependency_flags": {
        "relieff_ok": RELIEFF_OK,
        "mrmr_ok": MRMR_OK,
        "xgboost_ok": XGB_OK,
    },
    "n_pipelines_built": len(PIPELINES),
    "n_results": len(results_df),
    "top_pipeline": results_df.iloc[0]["Pipeline"] if len(results_df) else "",
    "top_auc3": float(results_df.iloc[0]["AUC3"]) if len(results_df) else float("nan"),
}
with open(OUT_META, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n{'=' * 60}")
print("SANITY CHECKS")
print(f"{'=' * 60}")
print(f"Expected pipelines: {len(PIPELINES)}")
print(f"Completed rows:     {len(results_df)}")
if len(results_df):
    print(f"AUC3 range:         {results_df['AUC3'].min():.4f} - {results_df['AUC3'].max():.4f}")
    print(f"Test3 range:        {results_df['Test3'].min():.4f} - {results_df['Test3'].max():.4f}")
    print(f"Gap range:          {results_df['Gap'].min():+.4f} - {results_df['Gap'].max():+.4f}")
    print(f"Fea3 range:         {int(results_df['Fea3'].min())} - {int(results_df['Fea3'].max())}")

print(f"\nResults CSV:   {OUT_CSV.name}")
print(f"Features PKL:  {OUT_PKL.name}")
print(f"Metadata JSON: {OUT_META.name}")
print(f"Checkpoint:    {CHECKPOINT.name}")
print(f"\nTop 10 pipelines:")
if len(results_df):
    print(
        results_df.head(10)[
            ["Rank", "Pattern", "S2_Base_Rank", "Pipeline", "AUC3", "Test3", "Gap", "Fea3"]
        ].to_string(index=False)
    )
else:
    print("<no completed pipelines>")
