# -*- coding: utf-8 -*-
"""
Stage 1: CT Radiomics Feature Selection - CHUS+CHUP External Split (2v)
========================================================================
Single run version (no reproducibility loop).
Development cohort: 457 patients (CHUS, CHUP excluded; remaining centers)
External holdout: 90 patients (CHUS + CHUP)

Input:  Mar_2026/27_feb_CT_development.csv
Output: Mar_2026/27_feb_CT_2v_result.csv
        Mar_2026/27_feb_CT_2v_result_metadata.json

No train/test split: trains on ALL dev samples (saves events). Evaluation via CV only.

Checkpoint: Resumes from last completed category if interrupted.

Usage:
    cd "D:/Uppsala thesis/Mar_2026"
    python 27_feb_CT_2v.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import hashlib
import json
import random
import time
import sys
from pathlib import Path

# ============================================================
# PATH SETUP
# ============================================================

SCRIPT_DIR = Path(__file__).parent                # Mar_2026
PROJECT_ROOT = SCRIPT_DIR.parent                  # D:\Uppsala thesis
OUTPUT_DIR = SCRIPT_DIR                           # Mar_2026

sys.path.insert(0, str(PROJECT_ROOT))
from fs_utils import *

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
RESULTS_BASE = '27_feb_CT_2v_result'
DATA_FILE = PROJECT_ROOT / 'Mar_2026' / '27_feb_CT_development.csv'
CHECKPOINT_RESULTS = OUTPUT_DIR / '27_feb_CT_2v_checkpoint_results.pkl'
CHECKPOINT_FEATURES = OUTPUT_DIR / '27_feb_CT_2v_checkpoint_features.pkl'

random.seed(SEED)
np.random.seed(SEED)

print("=" * 80)
print("CT STAGE 1: Feature Selection - CHUS+CHUP External Split 2v (Single Run)")
print("=" * 80)

# ============================================================
# VERIFY INPUT
# ============================================================

if not DATA_FILE.exists():
    print(f"[ERROR] Data not found: {DATA_FILE}")
    print("[INFO] Run: python Feb_2026/27_feb_create_CHUS_CHUP_external_split.py (creates Mar_2026 datasets)")
    sys.exit(1)

if not (PROJECT_ROOT / 'fs_utils.py').exists():
    print(f"[ERROR] fs_utils.py not found in {PROJECT_ROOT}")
    sys.exit(1)

print(f"[OK] Data: {DATA_FILE.name}")
print(f"[OK] fs_utils.py found")

# ============================================================
# IMPORTS (optional libraries)
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

plt.style.use('default')
sns.set_palette("husl")
START_TIME = time.time()

# ============================================================
# DATA LOADING
# ============================================================

print(f"\n[Step 1] Loading CT development data...")

df = pd.read_csv(DATA_FILE)

feature_cols_gtvp = [c for c in df.columns if c.startswith('GTVp_')]
feature_cols_gtvn = [c for c in df.columns if c.startswith('GTVn_')]

print(f"  Patients: {len(df)}")
print(f"  GTVp features: {len(feature_cols_gtvp)}")
print(f"  GTVn features: {len(feature_cols_gtvn)}")
print(f"  Events: {int(df['Relapse'].sum())} ({100*df['Relapse'].mean():.1f}%)")

feature_cols = [c for c in df.columns if c not in ['PatientID', 'Relapse', 'RFS']]
X = df[feature_cols].copy()
y_time = df['RFS'].values
y_event = df['Relapse'].values.astype(bool)

# ============================================================
# USE ALL DEV DATA (no train/test split - saves events)
# ============================================================

X_train = X.replace([np.inf, -np.inf], np.nan)
train_medians = X_train.median()
X_train = X_train.fillna(train_medians)

missing_count = X_train.isnull().sum().sum()
print(f"  Missing values after inf handling: {missing_count}")

y_train = pd.DataFrame({'RFS_time': y_time, 'event': y_event})

print(f"  Development (all): {len(X_train)} patients ({y_train['event'].sum()} events)")
print(f"  Features: {len(feature_cols)}")

# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def load_checkpoint():
    if CHECKPOINT_RESULTS.exists() and CHECKPOINT_FEATURES.exists():
        with open(CHECKPOINT_RESULTS, 'rb') as f:
            results = pickle.load(f)
        with open(CHECKPOINT_FEATURES, 'rb') as f:
            features = pickle.load(f)
        print(f"  [RESUME] Loaded checkpoint: {len(results)} methods completed")
        return results, features
    return [], {}

def save_checkpoint(results, features):
    with open(CHECKPOINT_RESULTS, 'wb') as f:
        pickle.dump(results, f)
    with open(CHECKPOINT_FEATURES, 'wb') as f:
        pickle.dump(features, f)

def is_completed(results, method_name):
    return any(r['Method'] == method_name for r in results)

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
    method_name = f'A1: Univariate Cox (p<{p_thresh})'
    if is_completed(results, method_name):
        print(f"  [SKIP] {method_name}")
        continue
    print(f"  Running {method_name}...")
    try:
        selected = univariate_cox_selection(X_train, y_train, p_threshold=p_thresh)
        if len(selected) > 0:
            cindex, std = evaluate_features_cv(X_train, y_train, selected, f"A1_Cox_p{p_thresh}", random_state=SEED)
            results.append({'Category': 'A', 'Method': method_name, 'C-index': cindex, 'Std': std, 'Features': len(selected)})
            selected_features_storage[f'A1_Cox_p{p_thresh}'] = selected
            print(f"    C-index: {cindex:.4f} +/- {std:.4f}, Features: {len(selected)}")
        else:
            print(f"    No features selected")
    except Exception as e:
        print(f"    ERROR: {type(e).__name__}: {e}")
        results.append({'Category': 'A', 'Method': method_name, 'C-index': float('nan'), 'Std': float('nan'), 'Features': 0})

# A2: Pearson Correlation
method_name = 'A2: Pearson Correlation'
if not is_completed(results, method_name):
    print(f"  Running {method_name}...")
    try:
        selected = pearson_selection(X_train, y_train, r_threshold=0.1, p_threshold=0.05)
        if len(selected) > 0:
            cindex, std = evaluate_features_cv(X_train, y_train, selected, "A2_Pearson", random_state=SEED)
            results.append({'Category': 'A', 'Method': method_name, 'C-index': cindex, 'Std': std, 'Features': len(selected)})
            selected_features_storage['A2_Pearson'] = selected
            print(f"    C-index: {cindex:.4f} +/- {std:.4f}, Features: {len(selected)}")
    except Exception as e:
        print(f"    ERROR: {type(e).__name__}: {e}")
        results.append({'Category': 'A', 'Method': method_name, 'C-index': float('nan'), 'Std': float('nan'), 'Features': 0})
else:
    print(f"  [SKIP] {method_name}")

# A3: Spearman Correlation
method_name = 'A3: Spearman Correlation'
if not is_completed(results, method_name):
    print(f"  Running {method_name}...")
    try:
        selected = spearman_selection(X_train, y_train, r_threshold=0.1, p_threshold=0.05)
        if len(selected) > 0:
            cindex, std = evaluate_features_cv(X_train, y_train, selected, "A3_Spearman", random_state=SEED)
            results.append({'Category': 'A', 'Method': method_name, 'C-index': cindex, 'Std': std, 'Features': len(selected)})
            selected_features_storage['A3_Spearman'] = selected
            print(f"    C-index: {cindex:.4f} +/- {std:.4f}, Features: {len(selected)}")
    except Exception as e:
        print(f"    ERROR: {type(e).__name__}: {e}")
        results.append({'Category': 'A', 'Method': method_name, 'C-index': float('nan'), 'Std': float('nan'), 'Features': 0})
else:
    print(f"  [SKIP] {method_name}")

# A4: Mutual Information
for k in [100, 200, 300]:
    method_name = f'A4: Mutual Information (top {k})'
    if is_completed(results, method_name):
        print(f"  [SKIP] {method_name}")
        continue
    print(f"  Running {method_name}...")
    try:
        selected = mutual_info_selection(X_train, y_train, k_features=k)
        if len(selected) > 0:
            cindex, std = evaluate_features_cv(X_train, y_train, selected, f"A4_MI_{k}", random_state=SEED)
            results.append({'Category': 'A', 'Method': method_name, 'C-index': cindex, 'Std': std, 'Features': len(selected)})
            selected_features_storage[f'A4_MI_{k}'] = selected
            print(f"    C-index: {cindex:.4f} +/- {std:.4f}, Features: {len(selected)}")
    except Exception as e:
        print(f"    ERROR: {type(e).__name__}: {e}")
        results.append({'Category': 'A', 'Method': method_name, 'C-index': float('nan'), 'Std': float('nan'), 'Features': 0})

# A6: ReliefF
if RELIEFF_AVAILABLE:
    for k in [50, 100, 200]:
        for n_neighbors in [50, 100]:
            method_name = f'A6: ReliefF (top {k}, n_neighbors={n_neighbors})'
            if is_completed(results, method_name):
                print(f"  [SKIP] {method_name}")
                continue
            print(f"  Running {method_name}...")
            try:
                selected = relieff_selection(X_train, y_train, n_features=k, n_neighbors=n_neighbors)
                if len(selected) > 0:
                    cindex, std = evaluate_features_cv(X_train, y_train, selected, f"A6_ReliefF_{k}_n{n_neighbors}", random_state=SEED)
                    results.append({'Category': 'A', 'Method': method_name, 'C-index': cindex, 'Std': std, 'Features': len(selected)})
                    selected_features_storage[f'A6_ReliefF_{k}_n{n_neighbors}'] = selected
                    print(f"    C-index: {cindex:.4f} +/- {std:.4f}, Features: {len(selected)}")
            except Exception as e:
                print(f"    ERROR: {type(e).__name__}: {e}")
                results.append({'Category': 'A', 'Method': method_name, 'C-index': float('nan'), 'Std': float('nan'), 'Features': 0})
else:
    print("  [SKIP] A6: ReliefF not available")

# A7: ANOVA F-test
method_name = 'A7: ANOVA F-test'
if not is_completed(results, method_name):
    k = 500
    print(f"  Running {method_name} (k={k})...")
    try:
        selected = anova_selection(X_train, y_train, k_features=k)
        if len(selected) > 0:
            cindex, std = evaluate_features_cv(X_train, y_train, selected, f"A7_ANOVA_{k}", random_state=SEED)
            results.append({'Category': 'A', 'Method': method_name, 'C-index': cindex, 'Std': std, 'Features': len(selected)})
            selected_features_storage[f'A7_ANOVA_{k}'] = selected
            print(f"    C-index: {cindex:.4f} +/- {std:.4f}, Features: {len(selected)}")
    except Exception as e:
        print(f"    ERROR: {type(e).__name__}: {e}")
        results.append({'Category': 'A', 'Method': method_name, 'C-index': float('nan'), 'Std': float('nan'), 'Features': 0})
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
    method_name = f'B1: Correlation Filter (r<{thresh})'
    if is_completed(results, method_name):
        print(f"  [SKIP] {method_name}")
        continue
    print(f"  Running {method_name}...")
    try:
        selected = correlation_filter_fixed(X_train, y_train, threshold=thresh, max_features=300)
        if len(selected) > 0:
            cindex, std = evaluate_features_cv(X_train, y_train, selected, f"B1_Corr_{thresh}", random_state=SEED)
            results.append({'Category': 'B', 'Method': method_name, 'C-index': cindex, 'Std': std, 'Features': len(selected)})
            selected_features_storage[f'B1_Corr_{thresh}'] = selected
            print(f"    C-index: {cindex:.4f} +/- {std:.4f}, Features: {len(selected)}")
    except Exception as e:
        print(f"    ERROR: {type(e).__name__}: {e}")
        results.append({'Category': 'B', 'Method': method_name, 'C-index': float('nan'), 'Std': float('nan'), 'Features': 0})

# B2: Variance Threshold
method_name = 'B2: Variance Threshold'
if not is_completed(results, method_name):
    print(f"  Running {method_name}...")
    try:
        selected = variance_filter(X_train, threshold=0.01)
        if len(selected) > 500:
            print(f"    Skipped CV: {len(selected)} features (p >> n, CoxPH convergence impractical)")
            results.append({'Category': 'B', 'Method': method_name, 'C-index': float('nan'), 'Std': float('nan'), 'Features': len(selected)})
        elif len(selected) > 0:
            cindex, std = evaluate_features_cv(X_train, y_train, selected, "B2_Variance", random_state=SEED)
            results.append({'Category': 'B', 'Method': method_name, 'C-index': cindex, 'Std': std, 'Features': len(selected)})
            selected_features_storage['B2_Variance'] = selected
            print(f"    C-index: {cindex:.4f} +/- {std:.4f}, Features: {len(selected)}")
    except Exception as e:
        print(f"    ERROR: {type(e).__name__}: {e}")
        results.append({'Category': 'B', 'Method': method_name, 'C-index': float('nan'), 'Std': float('nan'), 'Features': 0})
else:
    print(f"  [SKIP] {method_name}")

# B3: mRMR
if MRMR_AVAILABLE:
    for k in [30, 40, 50, 60, 100, 200]:
        method_name = f'B3: mRMR ({k})'
        if is_completed(results, method_name):
            print(f"  [SKIP] {method_name}")
            continue
        print(f"  Running {method_name}...")
        try:
            selected = mrmr_selection(X_train, y_train, n_features=k)
            if len(selected) > 0:
                cindex, std = evaluate_features_cv(X_train, y_train, selected, f"B3_mRMR_{k}", random_state=SEED)
                results.append({'Category': 'B', 'Method': method_name, 'C-index': cindex, 'Std': std, 'Features': len(selected)})
                selected_features_storage[f'B3_mRMR_{k}'] = selected
                print(f"    C-index: {cindex:.4f} +/- {std:.4f}, Features: {len(selected)}")
        except Exception as e:
            print(f"    ERROR: {type(e).__name__}: {e}")
            results.append({'Category': 'B', 'Method': method_name, 'C-index': float('nan'), 'Std': float('nan'), 'Features': 0})
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
method_name = 'C1: LASSO-Cox'
if not is_completed(results, method_name):
    print(f"  Running {method_name}...")
    try:
        selected = lasso_cox_selection(X_train, y_train, target_features=100, n_alphas=100)
        if len(selected) > 0:
            cindex, std = evaluate_features_cv(X_train, y_train, selected, "C1_LASSO", random_state=SEED)
            results.append({'Category': 'C', 'Method': method_name, 'C-index': cindex, 'Std': std, 'Features': len(selected)})
            selected_features_storage['C1_LASSO'] = selected
            print(f"    C-index: {cindex:.4f} +/- {std:.4f}, Features: {len(selected)}")
    except Exception as e:
        print(f"    ERROR: {type(e).__name__}: {e}")
        results.append({'Category': 'C', 'Method': method_name, 'C-index': float('nan'), 'Std': float('nan'), 'Features': 0})
else:
    print(f"  [SKIP] {method_name}")

# C2: Elastic Net Cox
method_name = 'C2: Elastic Net Cox'
if not is_completed(results, method_name):
    print(f"  Running {method_name}...")
    try:
        selected = elasticnet_cox_selection(X_train, y_train, l1_ratio=0.5, target_features=100, n_alphas=100)
        if len(selected) > 0:
            cindex, std = evaluate_features_cv(X_train, y_train, selected, "C2_ElasticNet", random_state=SEED)
            results.append({'Category': 'C', 'Method': method_name, 'C-index': cindex, 'Std': std, 'Features': len(selected)})
            selected_features_storage['C2_ElasticNet'] = selected
            print(f"    C-index: {cindex:.4f} +/- {std:.4f}, Features: {len(selected)}")
    except Exception as e:
        print(f"    ERROR: {type(e).__name__}: {e}")
        results.append({'Category': 'C', 'Method': method_name, 'C-index': float('nan'), 'Std': float('nan'), 'Features': 0})
else:
    print(f"  [SKIP] {method_name}")

# C3: Stability Selection LASSO (threshold strategy)
for n_feat in [50, 75, 100, 125]:
    for thresh in [0.6, 0.7]:
        method_name = f'C3: Stability ({n_feat}, thresh={thresh})'
        if is_completed(results, method_name):
            print(f"  [SKIP] {method_name}")
            continue
        print(f"  Running {method_name}...")
        try:
            selected = stability_selection_lasso(X_train, y_train, n_bootstrap=100,
                                                  stability_threshold=thresh, n_features=n_feat,
                                                  selection_strategy='threshold', random_state=SEED)
            if len(selected) > 0:
                cindex, std = evaluate_features_cv(X_train, y_train, selected, f"C3_Stability_{n_feat}_{thresh}", random_state=SEED)
                results.append({'Category': 'C', 'Method': method_name, 'C-index': cindex, 'Std': std, 'Features': len(selected)})
                selected_features_storage[f'C3_Stability_{n_feat}_{thresh}'] = selected
                print(f"    C-index: {cindex:.4f} +/- {std:.4f}, Features: {len(selected)}")
        except Exception as e:
            print(f"    ERROR: {type(e).__name__}: {e}")
            results.append({'Category': 'C', 'Method': method_name, 'C-index': float('nan'), 'Std': float('nan'), 'Features': 0})

# C3: Stability Selection LASSO (top_k strategy)
for n_feat in [50, 75, 100]:
    method_name = f'C3: Stability (top {n_feat}, ranked)'
    if is_completed(results, method_name):
        print(f"  [SKIP] {method_name}")
        continue
    print(f"  Running {method_name}...")
    try:
        selected = stability_selection_lasso(X_train, y_train, n_bootstrap=100,
                                              stability_threshold=0.0, n_features=n_feat,
                                              selection_strategy='top_k', random_state=SEED)
        if len(selected) > 0:
            cindex, std = evaluate_features_cv(X_train, y_train, selected, f"C3_Stability_ranked_{n_feat}", random_state=SEED)
            results.append({'Category': 'C', 'Method': method_name, 'C-index': cindex, 'Std': std, 'Features': len(selected)})
            selected_features_storage[f'C3_Stability_ranked_{n_feat}'] = selected
            print(f"    C-index: {cindex:.4f} +/- {std:.4f}, Features: {len(selected)}")
    except Exception as e:
        print(f"    ERROR: {type(e).__name__}: {e}")
        results.append({'Category': 'C', 'Method': method_name, 'C-index': float('nan'), 'Std': float('nan'), 'Features': 0})

save_checkpoint(results, selected_features_storage)
print(f"  Category C checkpoint saved. ({len(results)} methods total)")

# ============================================================
# CATEGORY D: ML-Based Importance Ranking
# ============================================================

print(f"\n{'='*60}")
print("CATEGORY D: ML-Based Importance Ranking")
print(f"{'='*60}")

# D1: Random Survival Forest (Permutation Importance)
for n_feat in [50, 60]:
    method_name = f'D1: RSF PermImp ({n_feat})'
    if is_completed(results, method_name):
        print(f"  [SKIP] {method_name}")
        continue
    print(f"  Running {method_name}...")
    try:
        selected = rsf_permutation_importance(X_train, y_train, n_features=n_feat, n_estimators=500, random_state=SEED)
        if len(selected) > 0:
            cindex, std = evaluate_features_cv(X_train, y_train, selected, f"D1_RSF_PermImp_{n_feat}", random_state=SEED)
            results.append({'Category': 'D', 'Method': method_name, 'C-index': cindex, 'Std': std, 'Features': len(selected)})
            selected_features_storage[f'D1_RSF_PermImp_{n_feat}'] = selected
            print(f"    C-index: {cindex:.4f} +/- {std:.4f}, Features: {len(selected)}")
    except Exception as e:
        print(f"    ERROR: {type(e).__name__}: {e}")
        results.append({'Category': 'D', 'Method': method_name, 'C-index': float('nan'), 'Std': float('nan'), 'Features': 0})

# D2: XGBoost Survival
for n_feat in [50, 60]:
    method_name = f'D2: XGBoost ({n_feat})'
    if is_completed(results, method_name):
        print(f"  [SKIP] {method_name}")
        continue
    print(f"  Running {method_name}...")
    try:
        selected = xgboost_survival_selection(X_train, y_train, n_features=n_feat, n_estimators=100, random_state=SEED)
        if len(selected) > 0:
            cindex, std = evaluate_features_cv(X_train, y_train, selected, f"D2_XGBoost_{n_feat}", random_state=SEED)
            results.append({'Category': 'D', 'Method': method_name, 'C-index': cindex, 'Std': std, 'Features': len(selected)})
            selected_features_storage[f'D2_XGBoost_{n_feat}'] = selected
            print(f"    C-index: {cindex:.4f} +/- {std:.4f}, Features: {len(selected)}")
    except Exception as e:
        print(f"    ERROR: {type(e).__name__}: {e}")
        results.append({'Category': 'D', 'Method': method_name, 'C-index': float('nan'), 'Std': float('nan'), 'Features': 0})

# D3: Permutation Importance (GradientBoostingSurvivalAnalysis)
for n_feat in [50, 60]:
    method_name = f'D3: Permutation Importance ({n_feat})'
    if is_completed(results, method_name):
        print(f"  [SKIP] {method_name}")
        continue
    print(f"  Running {method_name}...")
    try:
        selected = permutation_importance_survival(X_train, y_train, n_features=n_feat, n_estimators=500, random_state=SEED)
        if len(selected) > 0:
            cindex, std = evaluate_features_cv(X_train, y_train, selected, f"D3_PermImp_{n_feat}", random_state=SEED)
            results.append({'Category': 'D', 'Method': method_name, 'C-index': cindex, 'Std': std, 'Features': len(selected)})
            selected_features_storage[f'D3_PermImp_{n_feat}'] = selected
            print(f"    C-index: {cindex:.4f} +/- {std:.4f}, Features: {len(selected)}")
    except Exception as e:
        print(f"    ERROR: {type(e).__name__}: {e}")
        results.append({'Category': 'D', 'Method': method_name, 'C-index': float('nan'), 'Std': float('nan'), 'Features': 0})

save_checkpoint(results, selected_features_storage)
print(f"  Category D checkpoint saved. ({len(results)} methods total)")

# ============================================================
# BASELINE: Random Feature Selection
# ============================================================

print(f"\n{'='*60}")
print("BASELINE: Random Feature Selection")
print(f"{'='*60}")

method_name = 'Method 0: Random 1000 Features'
if not is_completed(results, method_name):
    np.random.seed(42)
    n_random = 1000
    print(f"  Running baseline (random {n_random} features)...")
    try:
        random_features = np.random.choice(X_train.columns, size=n_random, replace=False).tolist()
        cindex, std = evaluate_features_cv(X_train, y_train, random_features, "Baseline_Random", random_state=SEED)
        results.append({'Category': 'Baseline', 'Method': method_name, 'C-index': cindex, 'Std': std, 'Features': n_random})
        print(f"    C-index: {cindex:.4f} +/- {std:.4f}")
    except Exception as e:
        print(f"    ERROR: {type(e).__name__}: {e}")
        results.append({'Category': 'Baseline', 'Method': method_name, 'C-index': float('nan'), 'Std': float('nan'), 'Features': 0})
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
results_df = results_df.sort_values('C-index', ascending=False).reset_index(drop=True)
results_df['Rank'] = [f'CT_{i}' for i in range(1, len(results_df) + 1)]
results_df = results_df[['Rank', 'Category', 'Method', 'C-index', 'Std', 'Features']]
results_df['Efficiency'] = results_df['C-index'] / (results_df['Features'] / 100)

output_file = OUTPUT_DIR / f'{RESULTS_BASE}.csv'
results_df.to_csv(output_file, index=False)

results_hash = hashlib.md5(
    results_df[['Method', 'C-index', 'Std', 'Features']].to_json().encode()
).hexdigest()

metadata = {
    'run': 1,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'execution_time_seconds': execution_time,
    'seed': SEED,
    'total_methods': len(results_df),
    'top_method': results_df.iloc[0]['Method'],
    'top_cindex': float(results_df.iloc[0]['C-index']),
    'results_hash': results_hash,
    'data_file': 'Mar_2026/27_feb_CT_development.csv',
    'n_patients_development': len(df),
    'n_patients_external': 90,
    'n_features': len(feature_cols),
    'external_center': 'CHUS+CHUP'
}

with open(OUTPUT_DIR / f'{RESULTS_BASE}_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"  Results: {output_file.name}")
print(f"  Total methods: {len(results_df)}")
print(f"  Execution time: {execution_time/60:.1f} min")
print(f"\n  Top 5 methods:")
print(results_df.head()[['Rank', 'Method', 'C-index', 'Features']].to_string(index=False))

# ============================================================
# VISUALIZATION
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax1 = axes[0, 0]
top10 = results_df.head(10).sort_values('C-index')
colors = sns.color_palette("husl", len(top10))
ax1.barh(range(len(top10)), top10['C-index'], color=colors)
ax1.set_yticks(range(len(top10)))
ax1.set_yticklabels(top10['Method'], fontsize=9)
ax1.set_xlabel('C-index', fontsize=12)
ax1.set_title('CT Stage 1 (CHUS+CHUP Split 2v): Top 10 Methods', fontsize=14, fontweight='bold')
ax1.axvline(x=0.6, color='red', linestyle='--', alpha=0.5)

ax2 = axes[0, 1]
category_summary = results_df.groupby('Category')['C-index'].mean().sort_values(ascending=False)
ax2.bar(range(len(category_summary)), category_summary.values)
ax2.set_xticks(range(len(category_summary)))
ax2.set_xticklabels(category_summary.index)
ax2.set_ylabel('Mean C-index')
ax2.set_title('CT Stage 1 (CHUS+CHUP Split 2v): Category Performance', fontsize=14, fontweight='bold')

ax3 = axes[1, 0]
for cat in results_df['Category'].unique():
    cat_data = results_df[results_df['Category'] == cat]
    ax3.scatter(cat_data['Features'], cat_data['C-index'], label=cat, s=100, alpha=0.6)
ax3.set_xlabel('Number of Features')
ax3.set_ylabel('C-index')
ax3.set_title('CT Stage 1 (CHUS+CHUP Split 2v): Features vs Performance', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

ax4 = axes[1, 1]
top_eff = results_df.head(10).sort_values('Efficiency')
ax4.barh(range(len(top_eff)), top_eff['Efficiency'])
ax4.set_yticks(range(len(top_eff)))
ax4.set_yticklabels(top_eff['Method'], fontsize=8)
ax4.set_xlabel('Efficiency')
ax4.set_title('CT Stage 1 (CHUS+CHUP Split 2v): Efficiency (Top 10)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / f'{RESULTS_BASE}_summary.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# CLEANUP AND SUMMARY
# ============================================================

# Remove checkpoints on successful completion
if CHECKPOINT_RESULTS.exists():
    CHECKPOINT_RESULTS.unlink()
if CHECKPOINT_FEATURES.exists():
    CHECKPOINT_FEATURES.unlink()

print(f"\n{'='*80}")
print("CT STAGE 1 COMPLETE")
print(f"{'='*80}")
print(f"  Results:     {RESULTS_BASE}.csv")
print(f"  Metadata:    {RESULTS_BASE}_metadata.json")
print(f"  Visualization: {RESULTS_BASE}_summary.png")
print(f"  Total time:  {execution_time/60:.1f} min")
print(f"  Methods:     {len(results_df)}")
print(f"  Top method:  {results_df.iloc[0]['Method']} (C-index={results_df.iloc[0]['C-index']:.4f})")
print(f"{'='*80}")
