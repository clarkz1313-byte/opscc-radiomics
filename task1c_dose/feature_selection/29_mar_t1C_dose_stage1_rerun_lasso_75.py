# -*- coding: utf-8 -*-
"""
29_mar_t1C_dose_stage1_rerun_lasso_75.py
=========================================
Re-run C1/C2 LASSO-Cox and Elastic Net Cox for dose Stage 1 on the
HMR-expanded 75-patient sensitivity cohort only.

Problem: fs_utils.lasso_cox_selection hardcodes alpha_min_ratio=0.01,
which can cause ArithmeticError ("weights too large") on the small-N /
2818-feature dose dataset.

Fix: Inline re-implementation using CoxnetSurvivalAnalysis with progressively
larger alpha_min_ratio values (0.5 → 0.7 → 0.9) until no arithmetic error.
Does NOT modify fs_utils.py.

This script:
  1. Loads existing _75 checkpoint pkl files if present, otherwise starts fresh
  2. Removes any existing C1/C2 _75 entries
  3. Re-runs C1 LASSO-Cox and C2 Elastic Net with high-alpha-min-ratio approach
  4. Saves _75 checkpoint files and Dose_stage1_result_75.csv

Usage:
    cd "D:/Uppsala thesis"
    python Mar_2026_task1C/29_mar_T1C_fs_script_results/29_mar_t1C_dose_stage1_rerun_lasso_75.py
"""

import json
import pickle
import sys
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

# ============================================================
# PATHS
# ============================================================
SCRIPT_DIR   = Path(__file__).resolve().parent
TASK1C_ROOT  = SCRIPT_DIR.parent
PROJECT_ROOT = TASK1C_ROOT.parent

sys.path.insert(0, str(PROJECT_ROOT))
from fs_utils import evaluate_features_cv

DATA_FILE           = TASK1C_ROOT / "Dose_development_75.csv"
CHECKPOINT_RESULTS  = SCRIPT_DIR / "Dose_stage1_checkpoint_results_75.pkl"
CHECKPOINT_FEATURES = SCRIPT_DIR / "Dose_stage1_checkpoint_features_75.pkl"
RESULTS_CSV         = SCRIPT_DIR / "Dose_stage1_result_75.csv"
SEED = 42

print("=" * 70)
print("Dose Stage 1 — C1/C2 LASSO/ElasticNet re-run (high alpha_min_ratio)")
print("=" * 70)

# ============================================================
# VERIFY
# ============================================================
for f in [DATA_FILE]:
    if not f.exists():
        print(f"[ERROR] Missing: {f}")
        sys.exit(1)
    print(f"[OK] {f.name}")

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(DATA_FILE)
non_feature = {"PatientID", "CenterID", "Relapse", "RFS", "Gender_Male"}
feature_cols = [c for c in df.columns if c not in non_feature]

X_train = df[feature_cols].copy().replace([np.inf, -np.inf], np.nan)
X_train = X_train.fillna(X_train.median())
y_train = pd.DataFrame({"RFS_time": df["RFS"].values,
                         "event": df["Relapse"].values.astype(bool)})

print(f"\nData: {len(df)} patients, {int(df['Relapse'].sum())} events, "
      f"{len(feature_cols)} features")

# ============================================================
# LOAD CHECKPOINT IF PRESENT
# ============================================================
if CHECKPOINT_RESULTS.exists() and CHECKPOINT_FEATURES.exists():
    with open(CHECKPOINT_RESULTS, "rb") as f:
        results = pickle.load(f)
    with open(CHECKPOINT_FEATURES, "rb") as f:
        selected_features_storage = pickle.load(f)
    print(f"Checkpoint loaded: {len(results)} existing methods")
else:
    results = []
    selected_features_storage = {}
    print("No _75 checkpoint found; starting fresh with C1/C2 only")

if len(df) != 75 or int(df["Relapse"].sum()) != 13:
    print(f"[WARNING] Expected 75 patients / 13 events, got "
          f"{len(df)} / {int(df['Relapse'].sum())}")

# Remove any prior C1/C2 entries (failed or partial)
TARGET_METHODS = {"C1: LASSO-Cox", "C2: Elastic Net Cox"}
results = [r for r in results if r["Method"] not in TARGET_METHODS]
for k in list(selected_features_storage.keys()):
    if k in {"C1_LASSO", "C2_ElasticNet"}:
        del selected_features_storage[k]
print(f"Removed prior C1/C2 entries. Remaining: {len(results)} methods")

# ============================================================
# INLINE LASSO-COX WITH HIGH ALPHA_MIN_RATIO
# ============================================================
def lasso_cox_highalpha(X, y_df, l1_ratio=1.0, alpha_min_ratio=0.5, n_alphas=50):
    """
    LASSO-Cox (l1_ratio=1.0) or ElasticNet (0<l1_ratio<1) with configurable
    alpha_min_ratio. Tries the given ratio; falls back to larger values if
    ArithmeticError persists.
    """
    y_time  = y_df["RFS_time"].values
    y_event = y_df["event"].values

    # 80/20 split for alpha selection (same as fs_utils version)
    X_tr, X_val, yt_tr, yt_val, ye_tr, ye_val = train_test_split(
        X, y_time, y_event, test_size=0.2, random_state=SEED, stratify=y_event
    )
    y_surv_tr = Surv.from_arrays(event=ye_tr, time=yt_tr)

    scaler = StandardScaler()
    Xtr_sc = scaler.fit_transform(X_tr)
    Xval_sc = scaler.transform(X_val)

    # Try progressively larger alpha_min_ratio until no ArithmeticError
    for amr in [alpha_min_ratio, 0.6, 0.7, 0.8, 0.9]:
        try:
            lasso = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio,
                                           alpha_min_ratio=amr,
                                           n_alphas=n_alphas,
                                           max_iter=10000)
            lasso.fit(Xtr_sc, y_surv_tr)

            best_score, best_alpha = -np.inf, None
            for i, alpha in enumerate(lasso.alphas_):
                coef = lasso.coef_[:, i]
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
                best_alpha = lasso.alphas_[len(lasso.alphas_) // 2]

            # Retrain on full data with best alpha
            y_surv_full = Surv.from_arrays(event=y_event, time=y_time)
            scaler_full = StandardScaler()
            X_sc_full = scaler_full.fit_transform(X)
            lasso_full = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio,
                                                 alphas=[best_alpha],
                                                 max_iter=10000)
            lasso_full.fit(X_sc_full, y_surv_full)
            best_coef = lasso_full.coef_[:, 0]
            selected = X.columns[best_coef != 0].tolist()
            print(f"    Succeeded with alpha_min_ratio={amr}, "
                  f"best_alpha={best_alpha:.6f}, selected={len(selected)}")
            return selected

        except ArithmeticError as e:
            print(f"    alpha_min_ratio={amr} -> ArithmeticError: {e} — trying larger...")
        except Exception as e:
            print(f"    alpha_min_ratio={amr} -> {type(e).__name__}: {e} — trying larger...")

    print("    All alpha_min_ratio attempts failed — returning empty list")
    return []

# ============================================================
# C1: LASSO-Cox
# ============================================================
print("\n--- C1: LASSO-Cox ---")
try:
    selected_c1 = lasso_cox_highalpha(X_train, y_train, l1_ratio=1.0,
                                       alpha_min_ratio=0.5, n_alphas=50)
    if len(selected_c1) > 0:
        cindex, std = evaluate_features_cv(X_train, y_train, selected_c1,
                                            "C1_LASSO", random_state=SEED)
        results.append({"Category": "C", "Method": "C1: LASSO-Cox",
                        "C-index": cindex, "Std": std, "Features": len(selected_c1)})
        selected_features_storage["C1_LASSO"] = selected_c1
        print(f"  C-index: {cindex:.4f} +/- {std:.4f}, Features: {len(selected_c1)}")
    else:
        print("  No features selected — recording NaN")
        results.append({"Category": "C", "Method": "C1: LASSO-Cox",
                        "C-index": float("nan"), "Std": float("nan"), "Features": 0})
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")
    results.append({"Category": "C", "Method": "C1: LASSO-Cox",
                    "C-index": float("nan"), "Std": float("nan"), "Features": 0})

# ============================================================
# C2: Elastic Net Cox
# ============================================================
print("\n--- C2: Elastic Net Cox ---")
try:
    selected_c2 = lasso_cox_highalpha(X_train, y_train, l1_ratio=0.5,
                                       alpha_min_ratio=0.5, n_alphas=50)
    if len(selected_c2) > 0:
        cindex, std = evaluate_features_cv(X_train, y_train, selected_c2,
                                            "C2_ElasticNet", random_state=SEED)
        results.append({"Category": "C", "Method": "C2: Elastic Net Cox",
                        "C-index": cindex, "Std": std, "Features": len(selected_c2)})
        selected_features_storage["C2_ElasticNet"] = selected_c2
        print(f"  C-index: {cindex:.4f} +/- {std:.4f}, Features: {len(selected_c2)}")
    else:
        print("  No features selected — recording NaN")
        results.append({"Category": "C", "Method": "C2: Elastic Net Cox",
                        "C-index": float("nan"), "Std": float("nan"), "Features": 0})
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")
    results.append({"Category": "C", "Method": "C2: Elastic Net Cox",
                    "C-index": float("nan"), "Std": float("nan"), "Features": 0})

# ============================================================
# SAVE UPDATED CHECKPOINT
# ============================================================
with open(CHECKPOINT_RESULTS, "wb") as f:
    pickle.dump(results, f)
with open(CHECKPOINT_FEATURES, "wb") as f:
    pickle.dump(selected_features_storage, f)
print(f"\nCheckpoint updated: {len(results)} methods total")

# ============================================================
# REBUILD AND SAVE RESULTS CSV
# ============================================================
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("C-index", ascending=False).reset_index(drop=True)
results_df["Rank"] = [f"Dose_{i}" for i in range(1, len(results_df) + 1)]
results_df = results_df[["Rank", "Category", "Method", "C-index", "Std", "Features"]]
results_df["Efficiency"] = results_df["C-index"] / (results_df["Features"] / 100)

results_df.to_csv(RESULTS_CSV, index=False)
print(f"Results CSV updated: {RESULTS_CSV.name} ({len(results_df)} rows)")

print(f"\nTop 5 methods:")
print(results_df.head()[["Rank", "Method", "C-index", "Features"]].to_string(index=False))
print("\n[DONE]")
