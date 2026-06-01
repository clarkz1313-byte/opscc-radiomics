# -*- coding: utf-8 -*-
"""
Stage 2: PT Radiomics Two-Step Pipelines - CHUS+CHUP External Split (2v)
=======================================================================
Uses 12 selected Stage 1 methods from 27_feb_PT_2v. No train/test split:
evaluation via CV only. 54 pipelines: A->B, A->C, A->D, B->C, B->D, C->D.

Input:  Mar_2026/27_feb_PT_development.csv
        Mar_2026/27_feb_PT_2v_checkpoint_features.pkl (or recalc if missing)

Output: Mar_2026/27_feb_PT_Stage2_2v_result.csv
        Mar_2026/27_feb_PT_Stage2_2v_result_metadata.json

Only COX-starting S1 (A1_Cox_p0.001): precomputed univariate Cox p-values once.
C1_LASSO, C2_ElasticNet, C3_Stability computed on-demand. ReliefF and MI: on-demand.

Usage:
    cd "D:/Uppsala thesis"
    python Mar_2026/27_feb_PT_Stage2_2v.py
"""

import pandas as pd
import numpy as np
import pickle
import hashlib
import json
import random
import time
import sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

# ============================================================
# PATH SETUP
# ============================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR

sys.path.insert(0, str(PROJECT_ROOT))
from fs_utils import (
    univariate_cox_selection,
    mutual_info_selection,
    relieff_selection,
    mrmr_selection,
    lasso_cox_selection,
    elasticnet_cox_selection,
    stability_selection_lasso,
    xgboost_survival_selection,
    permutation_importance_survival,
    rsf_permutation_importance,
    evaluate_features_cv,
)

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
RESULTS_BASE = "27_feb_PT_Stage2_2v_result"
DATA_FILE = SCRIPT_DIR / "27_feb_PT_development.csv"
STAGE1_CHECKPOINT_FEATURES = SCRIPT_DIR / "27_feb_PT_2v_checkpoint_features.pkl"
STAGE1_CHECKPOINT_RESULTS = SCRIPT_DIR / "27_feb_PT_2v_checkpoint_results.pkl"
CHECKPOINT_RESULTS = SCRIPT_DIR / "27_feb_PT_Stage2_2v_checkpoint_results.pkl"
CHECKPOINT_FEATURES = SCRIPT_DIR / "27_feb_PT_Stage2_2v_checkpoint_features.pkl"

# Reference baseline for delta columns (PT_1 LASSO from Stage 1)
BASELINE_CV1 = 0.754
BASELINE_FEA1 = 30
BASELINE_STD1 = 0.065

random.seed(SEED)
np.random.seed(SEED)

# Rank -> Stage 1 storage key (12 selected methods)
RANK_TO_S1_KEY = {
    "PT_15": "A1_Cox_p0.001",
    "PT_18": "A6_ReliefF_50_n100",
    "PT_20": "A4_MI_100",
    "PT_3": "B3_mRMR_30",
    "PT_4": "B3_mRMR_40",
    "PT_6": "B3_mRMR_50",
    "PT_1": "C1_LASSO",
    "PT_2": "C2_ElasticNet",
    "PT_28": "C3_Stability_50_0.7",
    "PT_5": "D2_XGBoost_50",
    "PT_11": "D3_PermImp_50",
    "PT_13": "D1_RSF_PermImp_50",
}

# Category membership for pipeline grid
S1_CATEGORIES = {
    "PT_15": "A", "PT_18": "A", "PT_20": "A",
    "PT_3": "B", "PT_4": "B", "PT_6": "B",
    "PT_1": "C", "PT_2": "C", "PT_28": "C",
    "PT_5": "D", "PT_11": "D", "PT_13": "D",
}

# S1 key -> (func, kwargs) for recalculation when checkpoint missing.
# A1_Cox: COX-starter, uses precomputed p-values only. C1/C2/C3, ReliefF, MI: on-demand.
S1_RECALC_SPEC = {
    "A1_Cox_p0.001": (None, {"p_threshold": 0.001}),  # COX-starter: precomputed p-values
    "A6_ReliefF_50_n100": (relieff_selection, {"n_features": 50, "n_neighbors": 100}),
    "A4_MI_100": (mutual_info_selection, {"k_features": 100}),
    "B3_mRMR_30": (mrmr_selection, {"n_features": 30}),
    "B3_mRMR_40": (mrmr_selection, {"n_features": 40}),
    "B3_mRMR_50": (mrmr_selection, {"n_features": 50}),
    "C1_LASSO": (lasso_cox_selection, {"target_features": 100, "n_alphas": 100}),
    "C2_ElasticNet": (elasticnet_cox_selection, {"l1_ratio": 0.5, "target_features": 100, "n_alphas": 100}),
    "C3_Stability_50_0.7": (stability_selection_lasso, {"n_bootstrap": 100, "stability_threshold": 0.7, "n_features": 50,
                                                       "selection_strategy": "threshold", "random_state": SEED}),
    "D2_XGBoost_50": (xgboost_survival_selection, {"n_features": 50, "n_estimators": 100, "random_state": SEED}),
    "D3_PermImp_50": (permutation_importance_survival, {"n_features": 50, "n_estimators": 500, "random_state": SEED}),
    "D1_RSF_PermImp_50": (rsf_permutation_importance, {"n_features": 50, "n_estimators": 500, "random_state": SEED}),
}

# S2 methods: (category, func, kwargs) for each S1 key when it acts as S2
S2_METHODS = {
    "PT_18": (relieff_selection, {"n_features": 50, "n_neighbors": 100}),
    "PT_20": (mutual_info_selection, {"k_features": 100}),
    "PT_3": (mrmr_selection, {"n_features": 30}),
    "PT_4": (mrmr_selection, {"n_features": 40}),
    "PT_6": (mrmr_selection, {"n_features": 50}),
    "PT_1": (lasso_cox_selection, {"target_features": 100, "n_alphas": 100}),
    "PT_2": (elasticnet_cox_selection, {"l1_ratio": 0.5, "target_features": 100, "n_alphas": 100}),
    "PT_28": (stability_selection_lasso, {"n_bootstrap": 100, "stability_threshold": 0.7, "n_features": 50,
                                          "selection_strategy": "threshold", "random_state": SEED}),
    "PT_5": (xgboost_survival_selection, {"n_features": 50, "n_estimators": 100, "random_state": SEED}),
    "PT_11": (permutation_importance_survival, {"n_features": 50, "n_estimators": 500, "random_state": SEED}),
    "PT_13": (rsf_permutation_importance, {"n_features": 50, "n_estimators": 500, "random_state": SEED}),
}

# Only A1_Cox uses precomputed Cox p-values (as in 10_feb_CT_REstage4.py)
COX_PRECOMPUTE_S1_KEY = "A1_Cox_p0.001"

try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm

try:
    from skrebate import ReliefF
    RELIEFF_AVAILABLE = True
except ImportError:
    RELIEFF_AVAILABLE = False

try:
    from mrmr import mrmr_classif
    MRMR_AVAILABLE = True
except ImportError:
    MRMR_AVAILABLE = False

START_TIME = time.time()


def precompute_cox_p_values(X, y_df):
    """Precompute univariate Cox p-values for all features once (~27 min for 2818)."""
    from lifelines import CoxPHFitter
    y_time = y_df["RFS_time"].values
    y_event = y_df["event"].values
    p_values = {}
    for col in tqdm(X.columns, desc="Precomputing Cox p-values"):
        try:
            df = pd.DataFrame({"T": y_time, "E": y_event, "X": X[col]})
            cph = CoxPHFitter()
            cph.fit(df, duration_col="T", event_col="E", show_progress=False)
            p_values[col] = float(cph.summary.loc["X", "p"])
        except Exception:
            p_values[col] = 1.0
    return pd.Series(p_values)


def _cox_pvalues_to_a1_features(cox_p_values):
    """Convert precomputed Cox p-values to A1_Cox_p0.001 feature list."""
    return cox_p_values[cox_p_values < 0.001].index.tolist()


def load_stage1_features(X_train, y_train, precomputed_cox, cox_p_values):
    """Load S1 features from checkpoint or recalculate (using precomputed Cox when applicable)."""
    if STAGE1_CHECKPOINT_FEATURES.exists():
        try:
            with open(STAGE1_CHECKPOINT_FEATURES, "rb") as f:
                storage = pickle.load(f)
        except Exception:
            storage = {}
            print("[INFO] Stage 1 checkpoint unreadable - will recalculate S1 features")
    else:
        storage = {}
        print("[INFO] Stage 1 checkpoint missing - will recalculate S1 features as needed")

    recalc_cache = {}

    def get_s1_features(s1_key):
        if s1_key in storage and len(storage[s1_key]) > 0:
            return storage[s1_key]
        if s1_key in recalc_cache and len(recalc_cache[s1_key]) > 0:
            return recalc_cache[s1_key]
        if s1_key in precomputed_cox and len(precomputed_cox.get(s1_key, [])) > 0:
            recalc_cache[s1_key] = precomputed_cox[s1_key]
            return precomputed_cox[s1_key]
        func, kwargs = S1_RECALC_SPEC.get(s1_key, (None, {}))
        if func is not None:
            features = func(X_train, y_train, **kwargs)
            recalc_cache[s1_key] = features
            return features
        if s1_key == "A1_Cox_p0.001" and cox_p_values is not None:
            features = cox_p_values[cox_p_values < 0.001].index.tolist()
            recalc_cache[s1_key] = features
            return features
        return []

    return get_s1_features


def load_checkpoint():
    if CHECKPOINT_RESULTS.exists() and CHECKPOINT_FEATURES.exists():
        with open(CHECKPOINT_RESULTS, "rb") as f:
            results = pickle.load(f)
        with open(CHECKPOINT_FEATURES, "rb") as f:
            features = pickle.load(f)
        print(f"  [RESUME] Loaded checkpoint: {len(results)} pipelines completed")
        return results, features
    return [], {}


def save_checkpoint(results, features):
    with open(CHECKPOINT_RESULTS, "wb") as f:
        pickle.dump(results, f)
    with open(CHECKPOINT_FEATURES, "wb") as f:
        pickle.dump(features, f)


def is_completed(results, pipeline_name):
    return any(r.get("Pipeline") == pipeline_name for r in results)


def build_pipeline_grid():
    """Build 54 pipelines: A->B, A->C, A->D, B->C, B->D, C->D."""
    A = [r for r, cat in S1_CATEGORIES.items() if cat == "A"]
    B = [r for r, cat in S1_CATEGORIES.items() if cat == "B"]
    C = [r for r, cat in S1_CATEGORIES.items() if cat == "C"]
    D = [r for r, cat in S1_CATEGORIES.items() if cat == "D"]
    grid = []
    for s1_list, s2_list in [(A, B), (A, C), (A, D), (B, C), (B, D), (C, D)]:
        for r1 in s1_list:
            for r2 in s2_list:
                if r1 != r2:
                    grid.append((r1, r2))
    return grid


def run_pipeline(r1, r2, get_s1_features):
    """Run S1 -> S2 pipeline. Returns (result_dict, s2_features) or (None, None)."""
    s1_key = RANK_TO_S1_KEY[r1]
    s1_features = get_s1_features(s1_key)
    if len(s1_features) == 0:
        return None, None

    X_s1 = X_train[s1_features]
    func2, kwargs2 = S2_METHODS[r2]
    try:
        s2_features = func2(X_s1, y_train, **kwargs2)
    except Exception:
        return None, None

    if len(s2_features) == 0:
        return None, None

    cv_cindex, cv_std = evaluate_features_cv(
        X_train, y_train, s2_features,
        method_name=f"{s1_key}->{RANK_TO_S1_KEY[r2]}", random_state=SEED
    )

    result = {
        "Category": f"{S1_CATEGORIES[r1]}-{S1_CATEGORIES[r2]}",
        "Pipeline": f"{RANK_TO_S1_KEY[r1]} -> {RANK_TO_S1_KEY[r2]}",
        "CV2": cv_cindex,
        "Std2": cv_std,
        "Fea2": len(s2_features),
        "Fea1": len(s1_features),
    }
    return result, s2_features


# ============================================================
# MAIN
# ============================================================

print("=" * 80)
print("PT STAGE 2: Two-Step Pipelines (2v) - 12 Selected S1 Methods")
print("=" * 80)

if not DATA_FILE.exists():
    print(f"[ERROR] Data not found: {DATA_FILE}")
    sys.exit(1)

print(f"\n[Step 1] Loading data...")
df = pd.read_csv(DATA_FILE)
feature_cols = [c for c in df.columns if c not in ["PatientID", "Relapse", "RFS"]]
X = df[feature_cols].copy()
y_time = df["RFS"].values
y_event = df["Relapse"].values.astype(bool)
X_train = X.replace([np.inf, -np.inf], np.nan)
train_medians = X_train.median()
X_train = X_train.fillna(train_medians)
y_train = pd.DataFrame({"RFS_time": y_time, "event": y_event})
print(f"  Patients: {len(df)}, Features: {len(feature_cols)}, Events: {int(y_event.sum())}")

# Precompute only Univariate Cox p-values (for A1_Cox_p0.001 when checkpoint missing)
print(f"\n[Step 2] Precomputing Univariate Cox p-values (if A1_Cox needed)...")
cox_p_values = None
precomputed_cox = {}
needs_recalc = not STAGE1_CHECKPOINT_FEATURES.exists()
needs_a1_cox = False
if needs_recalc:
    needs_a1_cox = True
else:
    try:
        with open(STAGE1_CHECKPOINT_FEATURES, "rb") as f:
            storage = pickle.load(f)
        if COX_PRECOMPUTE_S1_KEY not in storage or len(storage.get(COX_PRECOMPUTE_S1_KEY, [])) == 0:
            needs_a1_cox = True
    except Exception:
        needs_a1_cox = True
if needs_a1_cox:
    print("  Precomputing Cox p-values for A1_Cox_p0.001 (COX-starting pipelines only)...")
    cox_p_values = precompute_cox_p_values(X_train, y_train)
    precomputed_cox[COX_PRECOMPUTE_S1_KEY] = _cox_pvalues_to_a1_features(cox_p_values)

get_s1_features = load_stage1_features(X_train, y_train, precomputed_cox, cox_p_values)

PIPELINE_GRID = build_pipeline_grid()
print(f"\n[Step 3] Running {len(PIPELINE_GRID)} pipelines...")

results, selected_features_storage = load_checkpoint()

for i, (r1, r2) in enumerate(PIPELINE_GRID):
    pipeline_name = f"{RANK_TO_S1_KEY[r1]} -> {RANK_TO_S1_KEY[r2]}"
    if is_completed(results, pipeline_name):
        print(f"  [{i+1:2d}/{len(PIPELINE_GRID)}] [SKIP] {pipeline_name}")
        continue

    needs_relieff = r1 == "PT_18" or r2 == "PT_18"
    needs_mrmr = r1 in ("PT_3", "PT_4", "PT_6") or r2 in ("PT_3", "PT_4", "PT_6")
    if needs_relieff and not RELIEFF_AVAILABLE:
        print(f"  [{i+1:2d}/{len(PIPELINE_GRID)}] [SKIP] {pipeline_name} (ReliefF unavailable)")
        continue
    if needs_mrmr and not MRMR_AVAILABLE:
        print(f"  [{i+1:2d}/{len(PIPELINE_GRID)}] [SKIP] {pipeline_name} (mRMR unavailable)")
        continue

    print(f"  [{i+1:2d}/{len(PIPELINE_GRID)}] {pipeline_name}")
    result, features = run_pipeline(r1, r2, get_s1_features)
    if result is not None:
        results.append(result)
        selected_features_storage[pipeline_name] = features
        print(f"    CV={result['CV2']:.4f} +/- {result['Std2']:.4f}, Fea2={result['Fea2']}")
        save_checkpoint(results, selected_features_storage)

# ============================================================
# SAVE RESULTS
# ============================================================

print(f"\n{'='*60}")
print("SAVING RESULTS")
print(f"{'='*60}")

execution_time = time.time() - START_TIME
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("CV2", ascending=False).reset_index(drop=True)
results_df["Rank"] = [f"PT_S2_{i}" for i in range(1, len(results_df) + 1)]
results_df["DeCV2-B1"] = results_df["CV2"] - BASELINE_CV1
results_df["DeFea2-B1"] = results_df["Fea2"] - BASELINE_FEA1
results_df["DeStd2-Std1"] = results_df["Std2"] - BASELINE_STD1
results_df["Efficiency"] = results_df["CV2"] / (results_df["Fea2"] / 100)

export_cols = ["Rank", "Category", "Pipeline", "CV2", "DeCV2-B1", "Std2", "DeStd2-Std1",
               "Fea2", "DeFea2-B1", "Fea1", "Efficiency"]
results_df = results_df[[c for c in export_cols if c in results_df.columns]]

output_file = OUTPUT_DIR / f"{RESULTS_BASE}.csv"
results_df.to_csv(output_file, index=False)

metadata = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "execution_time_seconds": execution_time,
    "seed": SEED,
    "total_pipelines": len(results_df),
    "top_pipeline": results_df.iloc[0]["Pipeline"] if len(results_df) > 0 else None,
    "top_cv2": float(results_df.iloc[0]["CV2"]) if len(results_df) > 0 else None,
    "baseline_cv1": BASELINE_CV1,
    "baseline_fea1": BASELINE_FEA1,
    "data_file": "Mar_2026/27_feb_PT_development.csv",
    "n_patients_development": len(df),
    "n_features": len(feature_cols),
}
with open(OUTPUT_DIR / f"{RESULTS_BASE}_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"  Results: {output_file.name}")
print(f"  Pipelines: {len(results_df)}")
print(f"  Time: {execution_time/60:.1f} min")
if len(results_df) > 0:
    print(f"\n  Top 5:")
    print(results_df.head()[["Rank", "Pipeline", "CV2", "Fea2"]].to_string(index=False))

# Remove checkpoints on success
if CHECKPOINT_RESULTS.exists():
    CHECKPOINT_RESULTS.unlink()
if CHECKPOINT_FEATURES.exists():
    CHECKPOINT_FEATURES.unlink()

# ============================================================
# SANITY AND LOGIC CHECKS
# ============================================================

print(f"\n{'='*60}")
print("SANITY AND LOGIC CHECKS")
print(f"{'='*60}")

checks_ok = True
if len(results_df) != 54:
    print(f"  [CHECK] Expected 54 pipelines, got {len(results_df)}")
    checks_ok = False
else:
    print(f"  [OK] 54 pipelines completed")

rank_col = results_df.get("Rank")
if rank_col is not None:
    expected_ranks = [f"PT_S2_{i}" for i in range(1, 55)]
    if not all(rank_col.iloc[i] == expected_ranks[i] for i in range(len(rank_col))):
        print(f"  [CHECK] Rank column format may be incorrect")
        checks_ok = False
    else:
        print(f"  [OK] Rank column format PT_S2_1, PT_S2_2, ...")

if "CV2" in results_df.columns:
    cv_range = (results_df["CV2"].min(), results_df["CV2"].max())
    if cv_range[0] < 0 or cv_range[1] > 1:
        print(f"  [CHECK] CV2 out of [0,1]: min={cv_range[0]}, max={cv_range[1]}")
        checks_ok = False
    else:
        print(f"  [OK] CV2 in [0,1]: [{cv_range[0]:.4f}, {cv_range[1]:.4f}]")

if "Fea2" in results_df.columns:
    fea_min, fea_max = results_df["Fea2"].min(), results_df["Fea2"].max()
    if fea_min < 1 or fea_max > len(feature_cols):
        print(f"  [CHECK] Fea2 out of expected range: min={fea_min}, max={fea_max}")
        checks_ok = False
    else:
        print(f"  [OK] Fea2 in valid range: [{fea_min}, {fea_max}]")

cats = results_df["Category"].unique()
expected_cats = {"A-B", "A-C", "A-D", "B-C", "B-D", "C-D"}
if not set(cats) <= expected_cats:
    print(f"  [CHECK] Unexpected category: {set(cats) - expected_cats}")
    checks_ok = False
else:
    print(f"  [OK] Categories: {sorted(cats)}")

if checks_ok:
    print(f"\n  All sanity checks passed.")
else:
    print(f"\n  Some checks failed - please review.")

print(f"\n{'='*80}")
print("PT STAGE 2 COMPLETE")
print(f"{'='*80}")
print(f"  Results:  {RESULTS_BASE}.csv")
print(f"  Metadata: {RESULTS_BASE}_metadata.json")
print(f"  Total time: {execution_time/60:.1f} min")
if len(results_df) > 0:
    print(f"  Top pipeline: {results_df.iloc[0]['Pipeline']} (CV={results_df.iloc[0]['CV2']:.4f})")
print(f"{'='*80}")
