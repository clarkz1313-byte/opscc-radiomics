# -*- coding: utf-8 -*-
"""
Task 2 Stage 2: CT two-step feature-selection pipelines.

This revised version recomputes Stage 1 subsets directly from the fixed train
set and evaluates Stage 2 pipelines with a true two-step nested CV path.

Usage:
    cd "D:/Uppsala thesis"
    python Mar_2026_task2/13_mar_task2_stage2_CT.py
"""

import json
import pickle
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

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
PROTOCOL_VERSION = "stage2_end_to_end_v2"
SAVE_EVERY = 10

RESULTS_DIR = SCRIPT_DIR / "13_mar_t2_fs_results"
DATA_DIR = SCRIPT_DIR / "12_mar_task2_rad_data"

TRAIN_FILE = DATA_DIR / "13_mar_task2_CT_primary_train.csv"
TEST_FILE = DATA_DIR / "13_mar_task2_CT_primary_test.csv"

OUT_CSV = RESULTS_DIR / "13_mar_task2_stage2_CT_result.csv"
OUT_PKL = RESULTS_DIR / "13_mar_task2_stage2_CT_features.pkl"
OUT_META = RESULTS_DIR / "13_mar_task2_stage2_CT_metadata.json"
CHECKPOINT = RESULTS_DIR / "13_mar_task2_stage2_CT_checkpoint.pkl"

BASELINE_AUC = 0.6328
BASELINE_FEAT = 60

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE_COLS = [
    "PatientID", "HPV_binary", "Relapse", "RFS",
    "Age", "Gender_Male", "Treatment_CRT", "prefix",
]

S1_CANDIDATES = [
    {"rank": "CT_6", "cat": "A", "label": "A7: ANOVA (top 100)", "selector_key": "A7", "selector_param": 100},
    {"rank": "CT_7", "cat": "A", "label": "A6: ReliefF (top 50)", "selector_key": "A6", "selector_param": 50},
    {"rank": "CT_10", "cat": "A", "label": "A2: Pearson (r>0.15 p<0.05)", "selector_key": "A2", "selector_param": 0.15},
    {"rank": "CT_3", "cat": "B", "label": "B3: mRMR classif (top 30)", "selector_key": "B3", "selector_param": 30},
    {"rank": "CT_5", "cat": "B", "label": "B3: mRMR classif (top 50)", "selector_key": "B3", "selector_param": 50},
    {"rank": "CT_30", "cat": "B", "label": "B1: Corr filter (r<0.85)", "selector_key": "B1", "selector_param": 0.85},
    {"rank": "CT_2", "cat": "C", "label": "C1: LASSO-Logistic (target 60)", "selector_key": "C1", "selector_param": 60},
    {"rank": "CT_4", "cat": "C", "label": "C1: LASSO-Logistic (target 30)", "selector_key": "C1", "selector_param": 30},
    {"rank": "CT_8", "cat": "C", "label": "C2: ElasticNet-Logistic (target 60)", "selector_key": "C2", "selector_param": 60},
]

np.random.seed(SEED)


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


def _holdout_split(X_f: pd.DataFrame, y_f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(
        X_f.values,
        y_f,
        test_size=0.25,
        random_state=SEED,
        stratify=y_f,
    )


def selector_rf_perm(X_f: pd.DataFrame, y_f: np.ndarray, k: int) -> list[str]:
    X_sub_tr, X_sub_val, y_sub_tr, y_sub_val = _holdout_split(X_f, y_f)
    clf = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    clf.fit(X_sub_tr, y_sub_tr)
    imp = permutation_importance(
        clf, X_sub_val, y_sub_val, n_repeats=10, random_state=SEED, n_jobs=-1
    )
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
        n_estimators=100,
        scale_pos_weight=scale_pos,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=SEED,
        verbosity=0,
        n_jobs=-1,
    )
    clf.fit(X_sub_tr, y_sub_tr)
    imp = permutation_importance(
        clf, X_sub_val, y_sub_val, n_repeats=10, random_state=SEED, n_jobs=-1
    )
    order = np.argsort(np.nan_to_num(imp.importances_mean, nan=-np.inf))[::-1][: min(k, X_f.shape[1])]
    return [X_f.columns[i] for i in order]


def selector_gb_perm(X_f: pd.DataFrame, y_f: np.ndarray, k: int) -> list[str]:
    X_sub_tr, X_sub_val, y_sub_tr, y_sub_val = _holdout_split(X_f, y_f)
    clf = GradientBoostingClassifier(n_estimators=200, random_state=SEED)
    clf.fit(X_sub_tr, y_sub_tr)
    imp = permutation_importance(
        clf, X_sub_val, y_sub_val, n_repeats=10, random_state=SEED, n_jobs=-1
    )
    order = np.argsort(np.nan_to_num(imp.importances_mean, nan=-np.inf))[::-1][: min(k, X_f.shape[1])]
    return [X_f.columns[i] for i in order]


def format_s2_label(s2_method: str, s2_k: int | None) -> str:
    if s2_method == "B1_corr":
        return "B1: Corr filter (r<0.85)"
    if s2_method == "C1":
        return f"C1: LASSO-Logistic (target {s2_k})"
    if s2_method == "C2":
        return f"C2: ElasticNet-Logistic (target {s2_k})"
    if s2_method == "D1":
        return f"D1: RF PermImp (top {s2_k})"
    if s2_method == "D2":
        return f"D2: XGBoost PermImp (top {s2_k})"
    if s2_method == "D3":
        return f"D3: GB PermImp (top {s2_k})"
    raise ValueError(f"Unknown S2 method: {s2_method}")


def make_s1_selector(candidate: dict):
    key = candidate["selector_key"]
    param = candidate["selector_param"]
    if key == "A0":
        return lambda X_f, y_f, use_fdr=(param == "fdr"): selector_mwu(X_f, y_f, use_fdr=use_fdr)
    if key == "A2":
        return lambda X_f, y_f, r_thresh=param: selector_pearson(X_f, y_f, r_thresh)
    if key == "A6":
        return lambda X_f, y_f, k=param: selector_relief(X_f, y_f, k)
    if key == "A7":
        return lambda X_f, y_f, k=param: selector_anova(X_f, y_f, k)
    if key == "B1":
        return lambda X_f, y_f, threshold=param: selector_corr(X_f, y_f, threshold)
    if key == "B3":
        return lambda X_f, y_f, k=param: selector_b3_mrmr(X_f, y_f, k)
    if key == "C1":
        return lambda X_f, y_f, k=param: selector_lasso(X_f, y_f, k)
    if key == "C2":
        return lambda X_f, y_f, k=param: selector_en(X_f, y_f, k)
    raise ValueError(f"Unsupported S1 selector: {key}")


def make_s2_selector(s2_method: str, s2_k: int | None):
    if s2_method == "B1_corr":
        return lambda X_f, y_f: selector_corr(X_f, y_f, 0.85)
    if s2_method == "C1":
        return lambda X_f, y_f, k=s2_k: selector_lasso(X_f, y_f, k)
    if s2_method == "C2":
        return lambda X_f, y_f, k=s2_k: selector_en(X_f, y_f, k)
    if s2_method == "D1":
        return lambda X_f, y_f, k=s2_k: selector_rf_perm(X_f, y_f, k)
    if s2_method == "D2":
        return lambda X_f, y_f, k=s2_k: selector_xgb_perm(X_f, y_f, k)
    if s2_method == "D3":
        return lambda X_f, y_f, k=s2_k: selector_gb_perm(X_f, y_f, k)
    raise ValueError(f"Unknown S2 method: {s2_method}")


def compose_two_step_selector(s1_selector, s2_selector):
    def _selector(X_f: pd.DataFrame, y_f: np.ndarray) -> list[str]:
        s1_features = s1_selector(X_f, y_f)
        if len(s1_features) == 0:
            return []
        X_s1 = X_f[s1_features].copy()
        return s2_selector(X_s1, y_f)

    return _selector


def compute_s1_feature_map(X_df: pd.DataFrame, y_arr: np.ndarray) -> tuple[dict[str, list[str]], dict[str, object]]:
    feature_map: dict[str, list[str]] = {}
    selector_map: dict[str, object] = {}
    print("\n[INFO] Recomputing Stage 1 feature subsets from fixed train set")
    for candidate in S1_CANDIDATES:
        selector_fn = make_s1_selector(candidate)
        try:
            selected = selector_fn(X_df, y_arr)
            selector_map[candidate["rank"]] = selector_fn
            feature_map[candidate["rank"]] = selected
            print(f"  [S1] {candidate['rank']} {candidate['label']} -> {len(selected)} features")
        except Exception as exc:
            print(f"  [WARN] {candidate['rank']} {candidate['label']} skipped: {type(exc).__name__}: {exc}")
    return feature_map, selector_map


def build_pipelines(s1_feature_map: dict[str, list[str]], selector_map: dict[str, object]) -> list[dict]:
    by_cat = {
        "A": [x for x in S1_CANDIDATES if x["cat"] == "A" and x["rank"] in selector_map],
        "B": [x for x in S1_CANDIDATES if x["cat"] == "B" and x["rank"] in selector_map],
        "C": [x for x in S1_CANDIDATES if x["cat"] == "C" and x["rank"] in selector_map],
    }

    pipelines: list[dict] = []
    idx = 1

    def add_pipeline(direction: str, s1: dict, s2_method: str, s2_k: int | None) -> None:
        nonlocal idx
        s1_features = s1_feature_map.get(s1["rank"], [])
        if len(s1_features) == 0:
            return
        if s2_method == "D2" and not XGB_OK:
            return
        if s2_k is not None and len(s1_features) <= s2_k:
            return
        label = f"{s1['label']} -> {format_s2_label(s2_method, s2_k)}"
        pipelines.append(
            {
                "pipeline_id": f"CT_S2_{idx:03d}",
                "direction": direction,
                "s1": s1,
                "s1_selector": selector_map[s1["rank"]],
                "s2_method": s2_method,
                "s2_k": s2_k,
                "label": label,
            }
        )
        idx += 1

    for s1 in by_cat["A"]:
        add_pipeline("A->B", s1, "B1_corr", None)
        for s2_method in ["C1", "C2"]:
            for k in [20, 30]:
                add_pipeline("A->C", s1, s2_method, k)
        for s2_method in ["D1", "D2", "D3"]:
            for k in [20, 30]:
                add_pipeline("A->D", s1, s2_method, k)

    for s1 in by_cat["B"]:
        for s2_method in ["C1", "C2"]:
            for k in [20, 30]:
                add_pipeline("B->C", s1, s2_method, k)
        for s2_method in ["D1", "D2", "D3"]:
            for k in [20, 30]:
                add_pipeline("B->D", s1, s2_method, k)

    for s1 in by_cat["C"]:
        for s2_method in ["D1", "D2", "D3"]:
            for k in [20, 30]:
                add_pipeline("C->D", s1, s2_method, k)

    return pipelines


def load_checkpoint() -> tuple[set[str], list[dict], dict[str, list[str]]]:
    if not CHECKPOINT.exists():
        return set(), [], {}
    with open(CHECKPOINT, "rb") as f:
        ckpt = pickle.load(f)
    if ckpt.get("protocol_version") != PROTOCOL_VERSION:
        print("[RESUME] Ignoring legacy checkpoint from older Stage 2 protocol")
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
print(f"Task 2 Stage 2: {MODALITY} two-step pipelines")
print("=" * 80)

for path in [TRAIN_FILE, TEST_FILE]:
    if not path.exists():
        print(f"[ERROR] Missing required input: {path}")
        sys.exit(1)
    print(f"[OK] {path.name}")

print(
    f"[INFO] Dependency flags: ReliefF={RELIEFF_OK} mRMR={MRMR_OK} XGBoost={XGB_OK}"
)

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

s1_feature_map, s1_selector_map = compute_s1_feature_map(X_train, y_train)
PIPELINES = build_pipelines(s1_feature_map, s1_selector_map)
print(f"\n[INFO] Built {len(PIPELINES)} pipelines under {PROTOCOL_VERSION}")

done_ids, results, selected_features_storage = load_checkpoint()
if done_ids:
    print(f"[RESUME] Loaded checkpoint with {len(done_ids)} completed pipelines")

start_time = time.time()
completed_since_save = 0

for i, pipeline in enumerate(PIPELINES, start=1):
    pid = pipeline["pipeline_id"]
    if pid in done_ids:
        print(f"[{i:02d}/{len(PIPELINES)}] [SKIP] {pipeline['label']}")
        continue

    pipeline_start = time.time()
    s1 = pipeline["s1"]
    s1_selector = pipeline["s1_selector"]
    s2_selector = make_s2_selector(pipeline["s2_method"], pipeline["s2_k"])
    two_step_selector = compose_two_step_selector(s1_selector, s2_selector)

    try:
        auc2, std2 = nested_cv_auc(X_train, y_train, two_step_selector, random_state=SEED)
        s1_features_full = s1_feature_map[s1["rank"]]
        feats_full = s2_selector(X_train[s1_features_full].copy(), y_train)
        test2 = evaluate_auc_test(X_train, y_train, X_test, y_test, feats_full)
        fea1 = len(s1_features_full)
        fea2 = len(feats_full)
        gap = test2 - auc2 if not np.isnan(test2) and not np.isnan(auc2) else float("nan")
    except Exception as exc:
        print(f"[{i:02d}/{len(PIPELINES)}] [ERROR] {pipeline['label']} -> {type(exc).__name__}: {exc}")
        done_ids.add(pid)
        continue

    pipeline_s = time.time() - pipeline_start
    elapsed_s = time.time() - start_time

    result_row = {
        "Category": pipeline["direction"],
        "S1_Rank": s1["rank"],
        "Pipeline": pipeline["label"],
        "AUC2": auc2,
        "Std2": std2,
        "Delta_AUC": auc2 - BASELINE_AUC,
        "Test2": test2,
        "Gap_Test_minus_AUC2": gap,
        "Fea2": fea2,
        "Fea1": fea1,
        "Delta_Fea": fea2 - BASELINE_FEAT,
        "Pipeline_s": pipeline_s,
        "Elapsed_s": elapsed_s,
    }
    results.append(result_row)
    selected_features_storage[pid] = feats_full
    done_ids.add(pid)
    completed_since_save += 1

    print(
        f"[{i:02d}/{len(PIPELINES)}] {pipeline['label']} "
        f"AUC2={auc2:.4f} Test2={test2:.4f} Gap={gap:+.4f} "
        f"Fea2={fea2} Pipeline={format_seconds(pipeline_s)} "
        f"Elapsed={format_seconds(elapsed_s)}"
    )

    if completed_since_save >= SAVE_EVERY:
        save_checkpoint(done_ids, results, selected_features_storage)
        completed_since_save = 0

save_checkpoint(done_ids, results, selected_features_storage)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(["AUC2", "Test2"], ascending=[False, False], na_position="last").reset_index(drop=True)
results_df.insert(0, "Rank", [f"CT_S2_{i}" for i in range(1, len(results_df) + 1)])
results_df = results_df[
    [
        "Rank",
        "Category",
        "S1_Rank",
        "Pipeline",
        "AUC2",
        "Std2",
        "Delta_AUC",
        "Test2",
        "Gap_Test_minus_AUC2",
        "Fea2",
        "Fea1",
        "Delta_Fea",
        "Pipeline_s",
        "Elapsed_s",
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
    "baseline_auc": BASELINE_AUC,
    "baseline_feat": BASELINE_FEAT,
    "dependency_flags": {
        "relieff_ok": RELIEFF_OK,
        "mrmr_ok": MRMR_OK,
        "xgboost_ok": XGB_OK,
    },
    "n_pipelines_built": len(PIPELINES),
    "n_results": len(results_df),
    "top_pipeline": results_df.iloc[0]["Pipeline"] if len(results_df) else "",
    "top_auc2": float(results_df.iloc[0]["AUC2"]) if len(results_df) else float("nan"),
}
with open(OUT_META, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n{'=' * 60}")
print("SANITY CHECKS")
print(f"{'=' * 60}")
print(f"Expected pipelines: {len(PIPELINES)}")
print(f"Completed rows:     {len(results_df)}")
if len(results_df):
    print(f"AUC2 range:         {results_df['AUC2'].min():.4f} - {results_df['AUC2'].max():.4f}")
    print(f"Test2 range:        {results_df['Test2'].min():.4f} - {results_df['Test2'].max():.4f}")
    print(
        f"Gap range:          "
        f"{results_df['Gap_Test_minus_AUC2'].min():+.4f} - "
        f"{results_df['Gap_Test_minus_AUC2'].max():+.4f}"
    )
    print(f"Fea2 range:         {int(results_df['Fea2'].min())} - {int(results_df['Fea2'].max())}")

print(f"\nResults CSV:   {OUT_CSV.name}")
print(f"Features PKL:  {OUT_PKL.name}")
print(f"Metadata JSON: {OUT_META.name}")
print(f"Checkpoint:    {CHECKPOINT.name}")
print(f"\nTop 10 pipelines:")
if len(results_df):
    print(
        results_df.head(10)[
            ["Rank", "Pipeline", "AUC2", "Test2", "Gap_Test_minus_AUC2", "Fea2"]
        ].to_string(index=False)
    )
else:
    print("<no completed pipelines>")
