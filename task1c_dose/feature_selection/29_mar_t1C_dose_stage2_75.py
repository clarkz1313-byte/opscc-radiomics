# -*- coding: utf-8 -*-
"""
29_mar_t1C_dose_stage2_75.py
============================
Branch C - Step 2: Dose Feature Selection Stage 2

Two-step pipelines — consistent with Task 1 PT/CT Stage 2 methodology
(4_mar_feature_selections_scripts/27_feb_PT_Stage2_2v.py).

Evaluation pattern (matches Task 1 exactly):
  1. S1 features computed ONCE on full dev (loaded from Stage 1 checkpoint)
  2. S2 selector runs on full dev restricted to S1 feature subset
  3. evaluate_features_cv() does 5-fold CV only on the final S2 feature set
     — no re-selection inside folds (same as Task 1)

Pipeline grid: A->B, A->C, A->D, B->C, B->D, C->D  (no same-method)

Input:
  Mar_2026_task1C/Dose_development_75.csv
  Mar_2026_task1C/29_mar_T1C_fs_script_results/Dose_stage1_checkpoint_features_75.pkl  (S1 feature lists)

Output:
  Mar_2026_task1C/29_mar_T1C_fs_script_results/Dose_stage2_result_75.csv
  Mar_2026_task1C/29_mar_T1C_fs_script_results/Dose_stage2_result_metadata_75.json
  Mar_2026_task1C/29_mar_T1C_fs_script_results/Dose_stage2_checkpoint_results_75.pkl
  Mar_2026_task1C/29_mar_T1C_fs_script_results/Dose_stage2_checkpoint_features_75.pkl

Usage:
    cd "D:/Uppsala thesis"
    python Mar_2026_task1C/29_mar_T1C_fs_script_results/29_mar_t1C_dose_stage2_75.py
"""

import hashlib
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv

try:
    from lifelines.utils import concordance_index
except ImportError:
    from sksurv.metrics import concordance_index_censored

    def concordance_index(times, scores, events):
        return concordance_index_censored(events.astype(bool), times, -scores)[0]

SCRIPT_DIR   = Path(__file__).resolve().parent
TASK1C_ROOT  = SCRIPT_DIR.parent
PROJECT_ROOT = TASK1C_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fs_utils import (
    MRMR_AVAILABLE,
    RELIEFF_AVAILABLE,
    correlation_filter_fixed,
    elasticnet_cox_selection,
    evaluate_features_cv,
    lasso_cox_selection,
    mrmr_selection,
    permutation_importance_survival,
    relieff_selection,
    rsf_permutation_importance,
    stability_selection_lasso,
    univariate_cox_selection,
    xgboost_survival_selection,
    mutual_info_selection,
)

# ============================================================
# CONFIGURATION
# ============================================================
SEED = 42
RESULTS_CSV = SCRIPT_DIR / "Dose_stage2_result_75.csv"
METADATA_JSON = SCRIPT_DIR / "Dose_stage2_result_metadata_75.json"

DATA_FILE              = TASK1C_ROOT / "Dose_development_75.csv"
S1_CHECKPOINT_FEATURES = SCRIPT_DIR / "Dose_stage1_checkpoint_features_75.pkl"
CHECKPOINT_RESULTS     = SCRIPT_DIR / "Dose_stage2_checkpoint_results_75.pkl"
CHECKPOINT_FEATURES    = SCRIPT_DIR / "Dose_stage2_checkpoint_features_75.pkl"

# Baseline from top Stage 1-75 result (Dose_1: mRMR 20, CV=0.8863, Fea=20)
BASELINE_CV1  = 0.8862577509636331
BASELINE_FEA1 = 20
BASELINE_STD1 = 0.1200437449481557

NON_FEATURE_COLS = {"PatientID", "CenterID", "Relapse", "RFS", "Gender_Male"}

random.seed(SEED)
np.random.seed(SEED)

# ============================================================
# S1 CANDIDATES (top performers from Stage 1, one per category group)
# Mirrors the Task 1 PT Stage 2 selection of 12 S1 methods.
# Keys match Dose_stage1_checkpoint_features_75.pkl storage keys.
# ============================================================
RANK_TO_S1_KEY = {
    # Category A
    "Dose_6":  "A6_ReliefF_30_n30",     # ReliefF top30 n30
    "Dose_9":  "A1_Cox_p0.1",           # Univariate Cox p<0.1
    "Dose_14": "A7_ANOVA_200",          # ANOVA top 200
    # Category B
    "Dose_1":  "B3_mRMR_20",            # mRMR 20
    "Dose_2":  "B3_mRMR_50",            # mRMR 50
    "Dose_3":  "B3_mRMR_30",            # mRMR 30
    # Category C
    "Dose_4":  "C2_ElasticNet",         # ElasticNet
    "Dose_7":  "C1_LASSO",              # LASSO-Cox
    "Dose_29": "C3_Stability_ranked_75", # Stability top 75 ranked
    # Category D
    "Dose_12": "D3_PermImp_30",         # GBS PermImp 30
    "Dose_21": "D3_PermImp_50",         # GBS PermImp 50
    "Dose_23": "D2_XGBoost_50",         # XGBoost 50
}

S1_CATEGORIES = {
    "Dose_6": "A", "Dose_9": "A", "Dose_14": "A",
    "Dose_1": "B", "Dose_2": "B", "Dose_3": "B",
    "Dose_4": "C", "Dose_7": "C", "Dose_29": "C",
    "Dose_12": "D", "Dose_21": "D", "Dose_23": "D",
}

# S2 methods: (func, kwargs) — applied to X_train restricted to S1 features
def lasso_cox_highalpha(X, y_df, l1_ratio=1.0, alpha_min_ratio=0.5, n_alphas=50):
    """Coxnet selector with conservative alpha path for small-N dose data."""
    y_time = y_df["RFS_time"].values
    y_event = y_df["event"].values

    X_tr, X_val, yt_tr, yt_val, ye_tr, ye_val = train_test_split(
        X, y_time, y_event, test_size=0.2, random_state=SEED, stratify=y_event
    )
    y_surv_tr = Surv.from_arrays(event=ye_tr, time=yt_tr)

    scaler = StandardScaler()
    Xtr_sc = scaler.fit_transform(X_tr)
    Xval_sc = scaler.transform(X_val)

    for amr in [alpha_min_ratio, 0.6, 0.7, 0.8, 0.9]:
        try:
            model = CoxnetSurvivalAnalysis(
                l1_ratio=l1_ratio,
                alpha_min_ratio=amr,
                n_alphas=n_alphas,
                max_iter=10000,
            )
            model.fit(Xtr_sc, y_surv_tr)

            best_score, best_alpha = -np.inf, None
            for i, alpha in enumerate(model.alphas_):
                coef = model.coef_[:, i]
                n_sel = np.sum(coef != 0)
                if 0 < n_sel < X.shape[1]:
                    risk_val = Xval_sc @ coef
                    try:
                        ci = concordance_index(yt_val, -risk_val, ye_val)
                        if ci > best_score:
                            best_score = ci
                            best_alpha = alpha
                    except Exception:
                        continue

            if best_alpha is None:
                continue

            y_surv_full = Surv.from_arrays(event=y_event, time=y_time)
            scaler_full = StandardScaler()
            X_full_sc = scaler_full.fit_transform(X)
            final_model = CoxnetSurvivalAnalysis(
                l1_ratio=l1_ratio,
                alphas=[best_alpha],
                max_iter=10000,
            )
            final_model.fit(X_full_sc, y_surv_full)
            coef = final_model.coef_[:, 0]
            selected = X.columns[coef != 0].tolist()
            if selected:
                return selected
        except Exception:
            continue

    return []


# Caps reduced vs Task 1 to match small-cohort dose rerun (same rationale as Stage 1)
S2_METHODS = {
    # B-group S2 methods
    "Dose_1":  (mrmr_selection,                  {"n_features": 20}),
    "Dose_2":  (mrmr_selection,                  {"n_features": 50}),
    "Dose_3":  (mrmr_selection,                  {"n_features": 30}),
    # C-group S2 methods
    "Dose_4":  (lasso_cox_highalpha,             {"l1_ratio": 0.5, "alpha_min_ratio": 0.5, "n_alphas": 50}),
    "Dose_7":  (lasso_cox_highalpha,             {"l1_ratio": 1.0, "alpha_min_ratio": 0.5, "n_alphas": 50}),
    "Dose_29": (stability_selection_lasso,       {"n_bootstrap": 100, "stability_threshold": 0.0,
                                                   "n_features": 75, "selection_strategy": "top_k",
                                                   "random_state": SEED}),
    # D-group S2 methods
    "Dose_12": (permutation_importance_survival, {"n_features": 30, "n_estimators": 500, "random_state": SEED}),
    "Dose_21": (permutation_importance_survival, {"n_features": 50, "n_estimators": 500, "random_state": SEED}),
    "Dose_23": (xgboost_survival_selection,      {"n_features": 50, "n_estimators": 500, "random_state": SEED}),
}

# ============================================================
# HELPERS
# ============================================================
def load_s1_features():
    """Load Stage 1 feature lists from checkpoint."""
    if not S1_CHECKPOINT_FEATURES.exists():
        print(f"[ERROR] Stage 1 checkpoint not found: {S1_CHECKPOINT_FEATURES}")
        print("[INFO] Run 29_mar_t1C_dose_stage1_75.py first.")
        sys.exit(1)
    with open(S1_CHECKPOINT_FEATURES, "rb") as f:
        storage = pickle.load(f)
    print(f"  Loaded Stage 1 checkpoint: {len(storage)} feature sets")
    return storage


def get_s1_features(storage, rank):
    """Retrieve S1 features for a given rank; recalculate if missing."""
    s1_key = RANK_TO_S1_KEY[rank]
    if s1_key in storage and len(storage[s1_key]) > 0:
        return storage[s1_key]
    # Not in checkpoint — recalculate on full X_train
    print(f"  [RECALC] {s1_key} not in Stage 1 checkpoint, recalculating...")
    key = s1_key
    if key.startswith("A1_Cox"):
        p = float(key.split("_p")[1])
        return univariate_cox_selection(X_train, y_train, p_threshold=p)
    if key.startswith("A7_ANOVA"):
        k = int(key.split("_")[-1])
        from sklearn.feature_selection import f_classif
        scores, _ = f_classif(X_train.values, y_train["event"].values.astype(int))
        order = np.argsort(np.nan_to_num(scores, nan=-np.inf))[::-1][:k]
        return [X_train.columns[i] for i in order]
    if key.startswith("A6_ReliefF"):
        parts = key.split("_")
        k, n = int(parts[2]), int(parts[4][1:])
        return relieff_selection(X_train, y_train, n_features=k, n_neighbors=n)
    if key.startswith("B3_mRMR"):
        k = int(key.split("_")[-1])
        return list(mrmr_selection(X_train, y_train, n_features=k))
    if key == "C1_LASSO":
        return lasso_cox_highalpha(X_train, y_train, l1_ratio=1.0,
                                   alpha_min_ratio=0.5, n_alphas=50)
    if key == "C2_ElasticNet":
        return lasso_cox_highalpha(X_train, y_train, l1_ratio=0.5,
                                   alpha_min_ratio=0.5, n_alphas=50)
    if key.startswith("C3_Stability_ranked"):
        n = int(key.split("_")[-1])
        return stability_selection_lasso(X_train, y_train, n_features=n,
                                         n_bootstrap=100, stability_threshold=0.0,
                                         selection_strategy="top_k", random_state=SEED)
    if key.startswith("C3_Stability"):
        parts = key.split("_")
        n = int(parts[2])
        t = float(parts[3])
        return stability_selection_lasso(X_train, y_train, n_features=n,
                                         n_bootstrap=100, stability_threshold=t,
                                         selection_strategy="threshold", random_state=SEED)
    if key.startswith("D1_RSF"):
        n = int(key.split("_")[-1])
        return rsf_permutation_importance(X_train, y_train, n_features=n,
                                          n_estimators=500, random_state=SEED)
    if key.startswith("D2_XGBoost"):
        n = int(key.split("_")[-1])
        return xgboost_survival_selection(X_train, y_train, n_features=n,
                                          n_estimators=500, random_state=SEED)
    if key.startswith("D3_PermImp"):
        n = int(key.split("_")[-1])
        return permutation_importance_survival(X_train, y_train, n_features=n,
                                               n_estimators=500, random_state=SEED)
    print(f"  [WARNING] Cannot recalculate {s1_key} — returning empty")
    return []


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
    """A->B, A->C, A->D, B->C, B->D, C->D — no same-method pairs."""
    A = [r for r, c in S1_CATEGORIES.items() if c == "A"]
    B = [r for r, c in S1_CATEGORIES.items() if c == "B"]
    C = [r for r, c in S1_CATEGORIES.items() if c == "C"]
    D = [r for r, c in S1_CATEGORIES.items() if c == "D"]
    grid = []
    for s1_list, s2_list in [(A, B), (A, C), (A, D), (B, C), (B, D), (C, D)]:
        for r1 in s1_list:
            for r2 in s2_list:
                if r2 in S2_METHODS:   # only include ranks that have an S2 func defined
                    grid.append((r1, r2))
    return grid


def run_pipeline(r1, r2, s1_feature_cache):
    """S1 on full dev (cached) -> S2 on S1 subset -> evaluate_features_cv on S2 features."""
    s1_features = s1_feature_cache.get(r1, [])
    if len(s1_features) == 0:
        return None, None

    X_s1 = X_train[s1_features]
    func2, kwargs2 = S2_METHODS[r2]
    try:
        s2_features = func2(X_s1, y_train, **kwargs2)
    except Exception as e:
        print(f"    S2 ERROR ({type(e).__name__}): {e}")
        return None, None

    if len(s2_features) == 0:
        return None, None

    s1_key = RANK_TO_S1_KEY[r1]
    s2_key = RANK_TO_S1_KEY[r2]
    pipeline_label = f"{s1_key} -> {s2_key}"

    cv2, std2 = evaluate_features_cv(X_train, y_train, s2_features,
                                      method_name=pipeline_label,
                                      random_state=SEED)
    result = {
        "Category":    f"{S1_CATEGORIES[r1]}->{S1_CATEGORIES[r2]}",
        "S1_Rank":     r1,
        "Pipeline":    pipeline_label,
        "CV2":         cv2,
        "Std2":        std2,
        "Delta_CV":    cv2 - BASELINE_CV1,
        "Fea2":        len(s2_features),
        "Fea1":        len(s1_features),
        "Delta_Fea":   len(s2_features) - BASELINE_FEA1,
    }
    return result, s2_features


# ============================================================
# MAIN
# ============================================================
print("=" * 80)
print("DOSE STAGE 2: Two-Step Pipelines — Branch C (Task 1 methodology)")
print("=" * 80)

for f in [DATA_FILE, S1_CHECKPOINT_FEATURES]:
    if not f.exists():
        print(f"[ERROR] Missing: {f}")
        sys.exit(1)
    print(f"[OK] {f.name}")

# Load data
print(f"\n[Step 1] Loading data...")
df = pd.read_csv(DATA_FILE)
feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
X_train = df[feature_cols].copy().replace([np.inf, -np.inf], np.nan)
X_train = X_train.fillna(X_train.median())
y_train = pd.DataFrame({"RFS_time": df["RFS"].values,
                         "event": df["Relapse"].values.astype(bool)})
print(f"  Patients: {len(df)}, Features: {len(feature_cols)}, "
      f"Events: {int(df['Relapse'].sum())}")

# Load S1 features (computed once on full dev)
print(f"\n[Step 2] Loading Stage 1 feature sets...")
s1_storage = load_s1_features()

# Build per-rank cache (retrieve or recalculate)
s1_feature_cache = {}
for rank in RANK_TO_S1_KEY:
    feats = get_s1_features(s1_storage, rank)
    s1_feature_cache[rank] = feats
    s1_key = RANK_TO_S1_KEY[rank]
    print(f"  {rank} ({s1_key}): {len(feats)} features")

# Build pipeline grid
PIPELINE_GRID = build_pipeline_grid()
print(f"\n[Step 3] Running {len(PIPELINE_GRID)} pipelines "
      f"(A->B, A->C, A->D, B->C, B->D, C->D)...")

results, selected_features_storage = load_checkpoint()
START_TIME = time.time()

for i, (r1, r2) in enumerate(PIPELINE_GRID):
    s1_key = RANK_TO_S1_KEY[r1]
    s2_key = RANK_TO_S1_KEY[r2]
    pipeline_label = f"{s1_key} -> {s2_key}"

    if is_completed(results, pipeline_label):
        print(f"  [{i+1:2d}/{len(PIPELINE_GRID)}] [SKIP] {pipeline_label}")
        continue

    needs_mrmr = r2 in ("Dose_1", "Dose_2", "Dose_3")
    needs_relieff = r1 == "Dose_6"
    if needs_mrmr and not MRMR_AVAILABLE:
        print(f"  [{i+1:2d}/{len(PIPELINE_GRID)}] [SKIP] {pipeline_label} (mRMR unavailable)")
        continue
    if needs_relieff and not RELIEFF_AVAILABLE:
        print(f"  [{i+1:2d}/{len(PIPELINE_GRID)}] [SKIP] {pipeline_label} (ReliefF unavailable)")
        continue

    print(f"  [{i+1:2d}/{len(PIPELINE_GRID)}] {pipeline_label}")
    result, features = run_pipeline(r1, r2, s1_feature_cache)
    if result is not None:
        results.append(result)
        selected_features_storage[pipeline_label] = features
        print(f"    CV2={result['CV2']:.4f} +/- {result['Std2']:.4f}  "
              f"Fea1={result['Fea1']} -> Fea2={result['Fea2']}")
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
results_df["Rank"] = [f"Dose_S2_{i}" for i in range(1, len(results_df) + 1)]
cols = ["Rank", "Category", "S1_Rank", "Pipeline",
        "CV2", "Std2", "Delta_CV", "Fea2", "Fea1", "Delta_Fea"]
results_df = results_df[cols]

out_csv = RESULTS_CSV
results_df.to_csv(out_csv, index=False)

results_hash = hashlib.md5(
    results_df[["Pipeline", "CV2", "Std2", "Fea2"]].to_json().encode()
).hexdigest()

metadata = {
    "timestamp":              time.strftime("%Y-%m-%d %H:%M:%S"),
    "execution_time_seconds": execution_time,
    "seed":                   SEED,
    "n_pipelines":            len(results_df),
    "top_pipeline":           results_df.iloc[0]["Pipeline"] if len(results_df) > 0 else "N/A",
    "top_cv2":                float(results_df.iloc[0]["CV2"]) if len(results_df) > 0 else float("nan"),
    "baseline_cv1":           BASELINE_CV1,
    "n_patients":             len(df),
    "n_events":               int(df["Relapse"].sum()),
    "methodology":            "S1 on full dev (cached) -> S2 on S1 subset -> evaluate_features_cv",
    "consistent_with":        "4_mar_feature_selections_scripts/27_feb_PT_Stage2_2v.py",
    "results_hash":           results_hash,
    "branch":                 "Task1C dosiomics subset extension, HMR-expanded N=75 rerun",
}
with open(METADATA_JSON, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"  Saved: {out_csv.name} ({len(results_df)} pipelines)")
print(f"  Execution time: {execution_time / 60:.1f} min")
print(f"\n  Top 10 pipelines:")
print(results_df.head(10)[["Rank", "Pipeline", "CV2", "Std2", "Fea2"]].to_string(index=False))
print("\n[DONE] Stage 2 complete.")
print(f"  Next: fork/run Stage 3 with _75 inputs and outputs.")
