# -*- coding: utf-8 -*-
"""
29_mar_t1C_dose_stage1_75.py
============================
Branch C — Step 1: Dose Feature Selection Stage 1

Fork of 4_mar_feature_selections_scripts/27_feb_PT_2v.py, adapted for:
  - Input:  Mar_2026_task1C/Dose_development_75.csv  (75 patients, 13 events)
  - Modality: dose (GTVp + GTVn combined, 2818 features)
  - Feature count caps retained from the small-cohort N=60 script for comparability
  - Rank prefix: Dose_1, Dose_2, ... (instead of PT_*)

Key cap changes vs Task 1 PT Stage 1 (retained from N=60 pipeline):
  - MI / ReliefF: top-k capped at 50/100/150 (was 100/200/300)
  - mRMR: k capped at 20/30/50 (was 30/40/50/60/100/200)
  - RSF/XGB/PermImp: n_features capped at 30/50 (was 50/60)
  - ANOVA: k=200 (was 500)
  - Stability: n_feat capped at 30/50/75 (was 50/75/100/125)
  - Baseline: random 200 features (was 1000)

Checkpoint: resumes from last completed category if interrupted.

Output:
  Mar_2026_task1C/29_mar_T1C_fs_script_results/Dose_stage1_result_75.csv
  Mar_2026_task1C/29_mar_T1C_fs_script_results/Dose_stage1_result_metadata_75.json
  Mar_2026_task1C/29_mar_T1C_fs_script_results/Dose_stage1_checkpoint_results_75.pkl
  Mar_2026_task1C/29_mar_T1C_fs_script_results/Dose_stage1_checkpoint_features_75.pkl

Usage:
    cd "D:/Uppsala thesis"
    python Mar_2026_task1C/29_mar_T1C_fs_script_results/29_mar_t1C_dose_stage1_75.py
"""

import hashlib
import json
import pickle
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
# PATH SETUP
# ============================================================
SCRIPT_DIR   = Path(__file__).resolve().parent
TASK1C_ROOT  = SCRIPT_DIR.parent
PROJECT_ROOT = TASK1C_ROOT.parent

sys.path.insert(0, str(PROJECT_ROOT))
from fs_utils import *

# ============================================================
# CONFIGURATION
# ============================================================
SEED         = 42
RESULTS_CSV  = SCRIPT_DIR / "Dose_stage1_result_75.csv"
METADATA_JSON = SCRIPT_DIR / "Dose_stage1_result_metadata_75.json"
SUMMARY_PNG  = SCRIPT_DIR / "Dose_stage1_summary_75.png"
DATA_FILE    = TASK1C_ROOT / "Dose_development_75.csv"

CHECKPOINT_RESULTS  = SCRIPT_DIR / "Dose_stage1_checkpoint_results_75.pkl"
CHECKPOINT_FEATURES = SCRIPT_DIR / "Dose_stage1_checkpoint_features_75.pkl"

random.seed(SEED)
np.random.seed(SEED)

print("=" * 80)
print("DOSE STAGE 1: Feature Selection — Branch C dosiomics subset")
print("=" * 80)

# ============================================================
# VERIFY INPUT
# ============================================================
if not DATA_FILE.exists():
    print(f"[ERROR] Data not found: {DATA_FILE}")
    print("[INFO] Build Mar_2026_task1C/Dose_development_75.csv first")
    sys.exit(1)

if not (PROJECT_ROOT / "fs_utils.py").exists():
    print(f"[ERROR] fs_utils.py not found in {PROJECT_ROOT}")
    sys.exit(1)

print(f"[OK] Data: {DATA_FILE.name}")
print(f"[OK] fs_utils.py found")

# ============================================================
# OPTIONAL IMPORTS
# ============================================================
try:
    from tqdm.auto import tqdm
    import tqdm as _tqdm
    _tqdm.tqdm.monitor_interval = 0
except ImportError:
    from tqdm import tqdm

try:
    from skrebate import ReliefF
    RELIEFF_AVAILABLE = True
except ImportError:
    print("[WARNING] ReliefF not available (pip install skrebate)")
    RELIEFF_AVAILABLE = False

try:
    from mrmr import mrmr_classif
    MRMR_AVAILABLE = True
except ImportError:
    print("[WARNING] mRMR not available (pip install mrmr-selection)")
    MRMR_AVAILABLE = False

plt.style.use("default")
sns.set_palette("husl")
START_TIME = time.time()

# ============================================================
# DATA LOADING
# ============================================================
print(f"\n[Step 1] Loading dose development data...")

df = pd.read_csv(DATA_FILE)

# Dose features: all GTVp_* and GTVn_* columns
feature_cols_gtvp = [c for c in df.columns if c.startswith("GTVp_")]
feature_cols_gtvn = [c for c in df.columns if c.startswith("GTVn_")]
feature_cols = feature_cols_gtvp + feature_cols_gtvn

non_feature = {"PatientID", "CenterID", "Relapse", "RFS", "Gender_Male"}
feature_cols = [c for c in df.columns if c not in non_feature]

print(f"  Patients : {len(df)}")
print(f"  GTVp dose features: {len(feature_cols_gtvp)}")
print(f"  GTVn dose features: {len(feature_cols_gtvn)}")
print(f"  Total dose features: {len(feature_cols)}")
print(f"  Events: {int(df['Relapse'].sum())} ({100 * df['Relapse'].mean():.1f}%)")
if len(df) != 75 or int(df["Relapse"].sum()) != 13:
    print(f"  [WARNING] Expected 75 patients / 13 events, got "
          f"{len(df)} / {int(df['Relapse'].sum())}")
print(f"  EPV note: {int(df['Relapse'].sum())} events / {len(feature_cols)} features — "
      f"aggressive filtering is required")

X_train = df[feature_cols].copy()
X_train = X_train.replace([np.inf, -np.inf], np.nan)
train_medians = X_train.median()
X_train = X_train.fillna(train_medians)

missing_count = X_train.isnull().sum().sum()
print(f"  Missing values after inf/NaN handling: {missing_count}")

y_train = pd.DataFrame({"RFS_time": df["RFS"].values, "event": df["Relapse"].values.astype(bool)})
print(f"  Development (all): {len(X_train)} patients ({y_train['event'].sum()} events)")

# ============================================================
# CHECKPOINT HELPERS
# ============================================================
def load_checkpoint():
    if CHECKPOINT_RESULTS.exists() and CHECKPOINT_FEATURES.exists():
        with open(CHECKPOINT_RESULTS, "rb") as f:
            results = pickle.load(f)
        with open(CHECKPOINT_FEATURES, "rb") as f:
            features = pickle.load(f)
        print(f"  [RESUME] Loaded checkpoint: {len(results)} methods completed")
        return results, features
    return [], {}

def save_checkpoint(results, features):
    with open(CHECKPOINT_RESULTS, "wb") as f:
        pickle.dump(results, f)
    with open(CHECKPOINT_FEATURES, "wb") as f:
        pickle.dump(features, f)

def is_completed(results, method_name):
    return any(r["Method"] == method_name for r in results)

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
                best_alpha = model.alphas_[len(model.alphas_) // 2]

            y_surv_full = Surv.from_arrays(event=y_event, time=y_time)
            scaler_full = StandardScaler()
            X_sc_full = scaler_full.fit_transform(X)
            final_model = CoxnetSurvivalAnalysis(
                l1_ratio=l1_ratio,
                alphas=[best_alpha],
                max_iter=10000,
            )
            final_model.fit(X_sc_full, y_surv_full)
            selected = X.columns[final_model.coef_[:, 0] != 0].tolist()
            print(f"    Succeeded with alpha_min_ratio={amr}, "
                  f"best_alpha={best_alpha:.6f}, selected={len(selected)}")
            return selected
        except ArithmeticError as e:
            print(f"    alpha_min_ratio={amr} -> ArithmeticError: {e}; trying larger")
        except Exception as e:
            print(f"    alpha_min_ratio={amr} -> {type(e).__name__}: {e}; trying larger")

    print("    All alpha_min_ratio attempts failed; returning empty list")
    return []

# ============================================================
# LOAD OR INITIALIZE
# ============================================================
print(f"\n[Step 2] Initializing results storage...")
results, selected_features_storage = load_checkpoint()

# ============================================================
# CATEGORY A: Statistical / Filter Methods
# ============================================================
print(f"\n{'='*60}")
print("CATEGORY A: Statistical / Filter Methods")
print(f"{'='*60}")

# A1: Univariate Cox
for p_thresh in [0.001, 0.01, 0.05, 0.1]:
    method_name = f"A1: Univariate Cox (p<{p_thresh})"
    if is_completed(results, method_name):
        print(f"  [SKIP] {method_name}")
        continue
    print(f"  Running {method_name}...")
    try:
        selected = univariate_cox_selection(X_train, y_train, p_threshold=p_thresh)
        if len(selected) > 0:
            cindex, std = evaluate_features_cv(X_train, y_train, selected,
                                               f"A1_Cox_p{p_thresh}", random_state=SEED)
            results.append({"Category": "A", "Method": method_name,
                             "C-index": cindex, "Std": std, "Features": len(selected)})
            selected_features_storage[f"A1_Cox_p{p_thresh}"] = selected
            print(f"    C-index: {cindex:.4f} +/- {std:.4f}, Features: {len(selected)}")
        else:
            print(f"    No features selected")
    except Exception as e:
        print(f"    ERROR: {type(e).__name__}: {e}")
        results.append({"Category": "A", "Method": method_name,
                        "C-index": float("nan"), "Std": float("nan"), "Features": 0})

# A2: Pearson Correlation
method_name = "A2: Pearson Correlation"
if not is_completed(results, method_name):
    print(f"  Running {method_name}...")
    try:
        selected = pearson_selection(X_train, y_train, r_threshold=0.1, p_threshold=0.05)
        if len(selected) > 0:
            cindex, std = evaluate_features_cv(X_train, y_train, selected,
                                               "A2_Pearson", random_state=SEED)
            results.append({"Category": "A", "Method": method_name,
                             "C-index": cindex, "Std": std, "Features": len(selected)})
            selected_features_storage["A2_Pearson"] = selected
            print(f"    C-index: {cindex:.4f} +/- {std:.4f}, Features: {len(selected)}")
        else:
            print(f"    No features selected")
    except Exception as e:
        print(f"    ERROR: {type(e).__name__}: {e}")
        results.append({"Category": "A", "Method": method_name,
                        "C-index": float("nan"), "Std": float("nan"), "Features": 0})
else:
    print(f"  [SKIP] {method_name}")

# A3: Spearman Correlation
method_name = "A3: Spearman Correlation"
if not is_completed(results, method_name):
    print(f"  Running {method_name}...")
    try:
        selected = spearman_selection(X_train, y_train, r_threshold=0.1, p_threshold=0.05)
        if len(selected) > 0:
            cindex, std = evaluate_features_cv(X_train, y_train, selected,
                                               "A3_Spearman", random_state=SEED)
            results.append({"Category": "A", "Method": method_name,
                             "C-index": cindex, "Std": std, "Features": len(selected)})
            selected_features_storage["A3_Spearman"] = selected
            print(f"    C-index: {cindex:.4f} +/- {std:.4f}, Features: {len(selected)}")
        else:
            print(f"    No features selected")
    except Exception as e:
        print(f"    ERROR: {type(e).__name__}: {e}")
        results.append({"Category": "A", "Method": method_name,
                        "C-index": float("nan"), "Std": float("nan"), "Features": 0})
else:
    print(f"  [SKIP] {method_name}")

# A4: Mutual Information — cap at 50/100/150 (was 100/200/300 for 455 pts)
for k in [50, 100, 150]:
    method_name = f"A4: Mutual Information (top {k})"
    if is_completed(results, method_name):
        print(f"  [SKIP] {method_name}")
        continue
    print(f"  Running {method_name}...")
    try:
        selected = mutual_info_selection(X_train, y_train, k_features=k)
        if len(selected) > 0:
            cindex, std = evaluate_features_cv(X_train, y_train, selected,
                                               f"A4_MI_{k}", random_state=SEED)
            results.append({"Category": "A", "Method": method_name,
                             "C-index": cindex, "Std": std, "Features": len(selected)})
            selected_features_storage[f"A4_MI_{k}"] = selected
            print(f"    C-index: {cindex:.4f} +/- {std:.4f}, Features: {len(selected)}")
    except Exception as e:
        print(f"    ERROR: {type(e).__name__}: {e}")
        results.append({"Category": "A", "Method": method_name,
                        "C-index": float("nan"), "Std": float("nan"), "Features": 0})

# A6: ReliefF — k capped at 30/50/100, n_neighbors capped at 30/50 (< n_patients=60)
if RELIEFF_AVAILABLE:
    for k in [30, 50, 100]:
        for n_neighbors in [30, 50]:
            method_name = f"A6: ReliefF (top {k}, n_neighbors={n_neighbors})"
            if is_completed(results, method_name):
                print(f"  [SKIP] {method_name}")
                continue
            print(f"  Running {method_name}...")
            try:
                selected = relieff_selection(X_train, y_train,
                                             n_features=k, n_neighbors=n_neighbors)
                if len(selected) > 0:
                    cindex, std = evaluate_features_cv(X_train, y_train, selected,
                                                       f"A6_ReliefF_{k}_n{n_neighbors}",
                                                       random_state=SEED)
                    results.append({"Category": "A", "Method": method_name,
                                    "C-index": cindex, "Std": std, "Features": len(selected)})
                    selected_features_storage[f"A6_ReliefF_{k}_n{n_neighbors}"] = selected
                    print(f"    C-index: {cindex:.4f} +/- {std:.4f}, Features: {len(selected)}")
            except Exception as e:
                print(f"    ERROR: {type(e).__name__}: {e}")
                results.append({"Category": "A", "Method": method_name,
                                "C-index": float("nan"), "Std": float("nan"), "Features": 0})
else:
    print("  [SKIP] A6: ReliefF not available")

# A7: ANOVA F-test — k=200 (was 500)
method_name = "A7: ANOVA F-test"
if not is_completed(results, method_name):
    k = 200
    print(f"  Running {method_name} (k={k})...")
    try:
        selected = anova_selection(X_train, y_train, k_features=k)
        if len(selected) > 0:
            cindex, std = evaluate_features_cv(X_train, y_train, selected,
                                               f"A7_ANOVA_{k}", random_state=SEED)
            results.append({"Category": "A", "Method": method_name,
                             "C-index": cindex, "Std": std, "Features": len(selected)})
            selected_features_storage[f"A7_ANOVA_{k}"] = selected
            print(f"    C-index: {cindex:.4f} +/- {std:.4f}, Features: {len(selected)}")
    except Exception as e:
        print(f"    ERROR: {type(e).__name__}: {e}")
        results.append({"Category": "A", "Method": method_name,
                        "C-index": float("nan"), "Std": float("nan"), "Features": 0})
else:
    print(f"  [SKIP] {method_name}")

save_checkpoint(results, selected_features_storage)
print(f"  Category A checkpoint saved. ({len(results)} methods total)")

# ============================================================
# CATEGORY B: Redundancy Removal Methods
# ============================================================
print(f"\n{'='*60}")
print("CATEGORY B: Redundancy Removal Methods")
print(f"{'='*60}")

# B1: Correlation Filter
for thresh in [0.85, 0.90, 0.95]:
    method_name = f"B1: Correlation Filter (r<{thresh})"
    if is_completed(results, method_name):
        print(f"  [SKIP] {method_name}")
        continue
    print(f"  Running {method_name}...")
    try:
        selected = correlation_filter_fixed(X_train, y_train,
                                            threshold=thresh, max_features=300)
        if len(selected) > 0:
            cindex, std = evaluate_features_cv(X_train, y_train, selected,
                                               f"B1_Corr_{thresh}", random_state=SEED)
            results.append({"Category": "B", "Method": method_name,
                             "C-index": cindex, "Std": std, "Features": len(selected)})
            selected_features_storage[f"B1_Corr_{thresh}"] = selected
            print(f"    C-index: {cindex:.4f} +/- {std:.4f}, Features: {len(selected)}")
    except Exception as e:
        print(f"    ERROR: {type(e).__name__}: {e}")
        results.append({"Category": "B", "Method": method_name,
                        "C-index": float("nan"), "Std": float("nan"), "Features": 0})

# B2: Variance Threshold
method_name = "B2: Variance Threshold"
if not is_completed(results, method_name):
    print(f"  Running {method_name}...")
    try:
        selected = variance_filter(X_train, threshold=0.01)
        if len(selected) > 0:
            cindex, std = evaluate_features_cv(X_train, y_train, selected,
                                               "B2_Variance", random_state=SEED)
            results.append({"Category": "B", "Method": method_name,
                             "C-index": cindex, "Std": std, "Features": len(selected)})
            selected_features_storage["B2_Variance"] = selected
            print(f"    C-index: {cindex:.4f} +/- {std:.4f}, Features: {len(selected)}")
    except Exception as e:
        print(f"    ERROR: {type(e).__name__}: {e}")
        results.append({"Category": "B", "Method": method_name,
                        "C-index": float("nan"), "Std": float("nan"), "Features": 0})
else:
    print(f"  [SKIP] {method_name}")

# B3: mRMR — k capped at 20/30/50 (was 30/40/50/60/100/200)
if MRMR_AVAILABLE:
    for k in [20, 30, 50]:
        method_name = f"B3: mRMR ({k})"
        if is_completed(results, method_name):
            print(f"  [SKIP] {method_name}")
            continue
        print(f"  Running {method_name}...")
        try:
            selected = mrmr_selection(X_train, y_train, n_features=k)
            if len(selected) > 0:
                cindex, std = evaluate_features_cv(X_train, y_train, selected,
                                                   f"B3_mRMR_{k}", random_state=SEED)
                results.append({"Category": "B", "Method": method_name,
                                 "C-index": cindex, "Std": std, "Features": len(selected)})
                selected_features_storage[f"B3_mRMR_{k}"] = selected
                print(f"    C-index: {cindex:.4f} +/- {std:.4f}, Features: {len(selected)}")
        except Exception as e:
            print(f"    ERROR: {type(e).__name__}: {e}")
            results.append({"Category": "B", "Method": method_name,
                            "C-index": float("nan"), "Std": float("nan"), "Features": 0})
else:
    print("  [SKIP] B3: mRMR not available")

save_checkpoint(results, selected_features_storage)
print(f"  Category B checkpoint saved. ({len(results)} methods total)")

# ============================================================
# CATEGORY C: Regularization Methods
# ============================================================
print(f"\n{'='*60}")
print("CATEGORY C: Regularization Methods")
print(f"{'='*60}")

# C1: LASSO-Cox
method_name = "C1: LASSO-Cox"
if not is_completed(results, method_name):
    print(f"  Running {method_name}...")
    try:
        selected = lasso_cox_highalpha(
            X_train, y_train, l1_ratio=1.0, alpha_min_ratio=0.5, n_alphas=50
        )
        if len(selected) > 0:
            cindex, std = evaluate_features_cv(X_train, y_train, selected,
                                               "C1_LASSO", random_state=SEED)
            results.append({"Category": "C", "Method": method_name,
                             "C-index": cindex, "Std": std, "Features": len(selected)})
            selected_features_storage["C1_LASSO"] = selected
            print(f"    C-index: {cindex:.4f} +/- {std:.4f}, Features: {len(selected)}")
    except Exception as e:
        print(f"    ERROR: {type(e).__name__}: {e}")
        results.append({"Category": "C", "Method": method_name,
                        "C-index": float("nan"), "Std": float("nan"), "Features": 0})
else:
    print(f"  [SKIP] {method_name}")

# C2: Elastic Net Cox
method_name = "C2: Elastic Net Cox"
if not is_completed(results, method_name):
    print(f"  Running {method_name}...")
    try:
        selected = lasso_cox_highalpha(
            X_train, y_train, l1_ratio=0.5, alpha_min_ratio=0.5, n_alphas=50
        )
        if len(selected) > 0:
            cindex, std = evaluate_features_cv(X_train, y_train, selected,
                                               "C2_ElasticNet", random_state=SEED)
            results.append({"Category": "C", "Method": method_name,
                             "C-index": cindex, "Std": std, "Features": len(selected)})
            selected_features_storage["C2_ElasticNet"] = selected
            print(f"    C-index: {cindex:.4f} +/- {std:.4f}, Features: {len(selected)}")
    except Exception as e:
        print(f"    ERROR: {type(e).__name__}: {e}")
        results.append({"Category": "C", "Method": method_name,
                        "C-index": float("nan"), "Std": float("nan"), "Features": 0})
else:
    print(f"  [SKIP] {method_name}")

# C3: Stability Selection LASSO (threshold strategy) — n_feat capped at 30/50/75
for n_feat in [30, 50, 75]:
    for thresh in [0.6, 0.7]:
        method_name = f"C3: Stability ({n_feat}, thresh={thresh})"
        if is_completed(results, method_name):
            print(f"  [SKIP] {method_name}")
            continue
        print(f"  Running {method_name}...")
        try:
            selected = stability_selection_lasso(X_train, y_train,
                                                  n_bootstrap=100,
                                                  stability_threshold=thresh,
                                                  n_features=n_feat,
                                                  selection_strategy="threshold",
                                                  random_state=SEED)
            if len(selected) > 0:
                cindex, std = evaluate_features_cv(X_train, y_train, selected,
                                                   f"C3_Stability_{n_feat}_{thresh}",
                                                   random_state=SEED)
                results.append({"Category": "C", "Method": method_name,
                                 "C-index": cindex, "Std": std, "Features": len(selected)})
                selected_features_storage[f"C3_Stability_{n_feat}_{thresh}"] = selected
                print(f"    C-index: {cindex:.4f} +/- {std:.4f}, Features: {len(selected)}")
        except Exception as e:
            print(f"    ERROR: {type(e).__name__}: {e}")
            results.append({"Category": "C", "Method": method_name,
                            "C-index": float("nan"), "Std": float("nan"), "Features": 0})

# C3: Stability Selection LASSO (top_k strategy) — n_feat capped at 30/50/75
for n_feat in [30, 50, 75]:
    method_name = f"C3: Stability (top {n_feat}, ranked)"
    if is_completed(results, method_name):
        print(f"  [SKIP] {method_name}")
        continue
    print(f"  Running {method_name}...")
    try:
        selected = stability_selection_lasso(X_train, y_train,
                                              n_bootstrap=100,
                                              stability_threshold=0.0,
                                              n_features=n_feat,
                                              selection_strategy="top_k",
                                              random_state=SEED)
        if len(selected) > 0:
            cindex, std = evaluate_features_cv(X_train, y_train, selected,
                                               f"C3_Stability_ranked_{n_feat}",
                                               random_state=SEED)
            results.append({"Category": "C", "Method": method_name,
                             "C-index": cindex, "Std": std, "Features": len(selected)})
            selected_features_storage[f"C3_Stability_ranked_{n_feat}"] = selected
            print(f"    C-index: {cindex:.4f} +/- {std:.4f}, Features: {len(selected)}")
    except Exception as e:
        print(f"    ERROR: {type(e).__name__}: {e}")
        results.append({"Category": "C", "Method": method_name,
                        "C-index": float("nan"), "Std": float("nan"), "Features": 0})

save_checkpoint(results, selected_features_storage)
print(f"  Category C checkpoint saved. ({len(results)} methods total)")

# ============================================================
# CATEGORY D: ML-Based Importance Ranking
# ============================================================
print(f"\n{'='*60}")
print("CATEGORY D: ML-Based Importance Ranking")
print(f"{'='*60}")

# D1: RSF Permutation Importance — n_features capped at 30/50
for n_feat in [30, 50]:
    method_name = f"D1: RSF PermImp ({n_feat})"
    if is_completed(results, method_name):
        print(f"  [SKIP] {method_name}")
        continue
    print(f"  Running {method_name}...")
    try:
        selected = rsf_permutation_importance(X_train, y_train,
                                              n_features=n_feat,
                                              n_estimators=500,
                                              random_state=SEED)
        if len(selected) > 0:
            cindex, std = evaluate_features_cv(X_train, y_train, selected,
                                               f"D1_RSF_PermImp_{n_feat}",
                                               random_state=SEED)
            results.append({"Category": "D", "Method": method_name,
                             "C-index": cindex, "Std": std, "Features": len(selected)})
            selected_features_storage[f"D1_RSF_PermImp_{n_feat}"] = selected
            print(f"    C-index: {cindex:.4f} +/- {std:.4f}, Features: {len(selected)}")
    except Exception as e:
        print(f"    ERROR: {type(e).__name__}: {e}")
        results.append({"Category": "D", "Method": method_name,
                        "C-index": float("nan"), "Std": float("nan"), "Features": 0})

# D2: XGBoost Survival — n_features capped at 30/50
for n_feat in [30, 50]:
    method_name = f"D2: XGBoost ({n_feat})"
    if is_completed(results, method_name):
        print(f"  [SKIP] {method_name}")
        continue
    print(f"  Running {method_name}...")
    try:
        selected = xgboost_survival_selection(X_train, y_train,
                                              n_features=n_feat,
                                              n_estimators=100,
                                              random_state=SEED)
        if len(selected) > 0:
            cindex, std = evaluate_features_cv(X_train, y_train, selected,
                                               f"D2_XGBoost_{n_feat}",
                                               random_state=SEED)
            results.append({"Category": "D", "Method": method_name,
                             "C-index": cindex, "Std": std, "Features": len(selected)})
            selected_features_storage[f"D2_XGBoost_{n_feat}"] = selected
            print(f"    C-index: {cindex:.4f} +/- {std:.4f}, Features: {len(selected)}")
    except Exception as e:
        print(f"    ERROR: {type(e).__name__}: {e}")
        results.append({"Category": "D", "Method": method_name,
                        "C-index": float("nan"), "Std": float("nan"), "Features": 0})

# D3: Permutation Importance (GradientBoostingSurvivalAnalysis) — n_features capped at 30/50
for n_feat in [30, 50]:
    method_name = f"D3: Permutation Importance ({n_feat})"
    if is_completed(results, method_name):
        print(f"  [SKIP] {method_name}")
        continue
    print(f"  Running {method_name}...")
    try:
        selected = permutation_importance_survival(X_train, y_train,
                                                   n_features=n_feat,
                                                   n_estimators=500,
                                                   random_state=SEED)
        if len(selected) > 0:
            cindex, std = evaluate_features_cv(X_train, y_train, selected,
                                               f"D3_PermImp_{n_feat}",
                                               random_state=SEED)
            results.append({"Category": "D", "Method": method_name,
                             "C-index": cindex, "Std": std, "Features": len(selected)})
            selected_features_storage[f"D3_PermImp_{n_feat}"] = selected
            print(f"    C-index: {cindex:.4f} +/- {std:.4f}, Features: {len(selected)}")
    except Exception as e:
        print(f"    ERROR: {type(e).__name__}: {e}")
        results.append({"Category": "D", "Method": method_name,
                        "C-index": float("nan"), "Std": float("nan"), "Features": 0})

save_checkpoint(results, selected_features_storage)
print(f"  Category D checkpoint saved. ({len(results)} methods total)")

# ============================================================
# BASELINE: Random Feature Selection
# ============================================================
print(f"\n{'='*60}")
print("BASELINE: Random Feature Selection")
print(f"{'='*60}")

method_name = "Method 0: Random 200 Features"
if not is_completed(results, method_name):
    np.random.seed(42)
    n_random = 200
    print(f"  Running baseline (random {n_random} features)...")
    try:
        random_features = np.random.choice(X_train.columns, size=n_random, replace=False).tolist()
        cindex, std = evaluate_features_cv(X_train, y_train, random_features,
                                           "Baseline_Random", random_state=SEED)
        results.append({"Category": "Baseline", "Method": method_name,
                        "C-index": cindex, "Std": std, "Features": n_random})
        print(f"    C-index: {cindex:.4f} +/- {std:.4f}")
    except Exception as e:
        print(f"    ERROR: {type(e).__name__}: {e}")
        results.append({"Category": "Baseline", "Method": method_name,
                        "C-index": float("nan"), "Std": float("nan"), "Features": 0})
else:
    print(f"  [SKIP] {method_name}")

save_checkpoint(results, selected_features_storage)

# ============================================================
# SAVE RESULTS
# ============================================================
print(f"\n{'='*60}")
print("SAVING RESULTS")
print(f"{'='*60}")

execution_time = time.time() - START_TIME

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("C-index", ascending=False).reset_index(drop=True)
results_df["Rank"] = [f"Dose_{i}" for i in range(1, len(results_df) + 1)]
results_df = results_df[["Rank", "Category", "Method", "C-index", "Std", "Features"]]
results_df["Efficiency"] = results_df["C-index"] / (results_df["Features"] / 100)

output_file = RESULTS_CSV
results_df.to_csv(output_file, index=False)

results_hash = hashlib.md5(
    results_df[["Method", "C-index", "Std", "Features"]].to_json().encode()
).hexdigest()

metadata = {
    "run": 1,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "execution_time_seconds": execution_time,
    "seed": SEED,
    "total_methods": len(results_df),
    "top_method": results_df.iloc[0]["Method"] if len(results_df) > 0 else "N/A",
    "top_cindex": float(results_df.iloc[0]["C-index"]) if len(results_df) > 0 else float("nan"),
    "results_hash": results_hash,
    "data_file": "Mar_2026_task1C/Dose_development_75.csv",
    "n_patients_development": len(df),
    "n_events_development": int(df["Relapse"].sum()),
    "n_features": len(feature_cols),
    "epv_note": f"{int(df['Relapse'].sum())} events / {len(feature_cols)} features",
    "external_center": "CHUS only (CHUP has no dose data)",
    "branch": "Task1C dosiomics subset extension - HMR75 sensitivity",
}

with open(METADATA_JSON, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"  Results: {output_file.name}")
print(f"  Total methods: {len(results_df)}")
print(f"  Execution time: {execution_time / 60:.1f} min")
print(f"\n  Top 10 methods:")
print(results_df.head(10)[["Rank", "Method", "C-index", "Std", "Features"]].to_string(index=False))

# ============================================================
# VISUALIZATION
# ============================================================
try:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax1 = axes[0, 0]
    top10 = results_df.head(10).sort_values("C-index")
    ax1.barh(range(len(top10)), top10["C-index"], color="steelblue", alpha=0.8)
    ax1.set_yticks(range(len(top10)))
    ax1.set_yticklabels([f"{r} ({m[:30]})" for r, m in zip(top10["Rank"], top10["Method"])],
                        fontsize=8)
    ax1.set_xlabel("C-index (5-fold CV)")
    ax1.set_title("Top 10 Methods by C-index")
    ax1.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="Random")
    ax1.legend(fontsize=8)

    ax2 = axes[0, 1]
    cat_data = results_df.groupby("Category")["C-index"].mean().sort_values(ascending=False)
    ax2.bar(cat_data.index, cat_data.values, color="coral", alpha=0.8)
    ax2.set_xlabel("Category")
    ax2.set_ylabel("Mean C-index")
    ax2.set_title("Mean C-index by Category")
    ax2.axhline(y=0.5, color="red", linestyle="--", alpha=0.5)

    ax3 = axes[1, 0]
    valid = results_df.dropna(subset=["C-index", "Features"])
    ax3.scatter(valid["Features"], valid["C-index"], alpha=0.6, color="green")
    ax3.set_xlabel("Number of Features")
    ax3.set_ylabel("C-index")
    ax3.set_title("C-index vs Feature Count")
    ax3.axhline(y=0.5, color="red", linestyle="--", alpha=0.5)

    ax4 = axes[1, 1]
    valid2 = results_df.dropna(subset=["C-index", "Std"])
    ax4.scatter(valid2["Std"], valid2["C-index"], alpha=0.6, color="purple")
    ax4.set_xlabel("Std (CV fold variability)")
    ax4.set_ylabel("C-index")
    ax4.set_title("C-index vs Stability")

    plt.suptitle("Dose Stage 1 Feature Selection - Branch C HMR75 sensitivity (75 pts, 13 events)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plot_file = SUMMARY_PNG
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  [SAVED] {plot_file.name}")
except Exception as e:
    print(f"\n  [WARNING] Plot failed: {e}")

print("\n[DONE] Stage 1 complete.")
print(f"  Next: review Dose_stage1_result_75.csv, then fork/run Stage 2 with _75 outputs.")
