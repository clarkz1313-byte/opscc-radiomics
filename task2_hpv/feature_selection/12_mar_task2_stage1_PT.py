# -*- coding: utf-8 -*-
"""
Task 2 Stage 1: PT Radiomics Feature Selection — HPV Classification (no-MDA A1 split)
=======================================================================================
Task 2 adaptations (see Mar_2026_task2/6_mar_task2.md, Section 4 Phase 2):

- Outcome: HPV_binary (0/1), NOT RFS/Relapse
- All survival rankers replaced with classification equivalents
- Evaluation: nested 5-fold stratified CV ROC-AUC (leakage-free)
  Feature selection runs inside each outer fold so the held-out fold
  labels are never seen during selection.
- Additional Test_AUC column: model trained on full X_train, evaluated on
  X_test (80/20 split). For diagnostic/overfit-check only — not used for
  ranking or selection.
- Ranking metric: CV_AUC descending
- No pre-filtering — all methods start from full 2818-feature pool
- Radiomics only (no clinical features)

Dev cohort: 87 HPV-known patients (CID1+CID6+CID2+CID7, no MDA)
  Input train: Mar_2026_task2/12_mar_task2_rad_data/13_mar_task2_PT_primary_train.csv
  Input test:  Mar_2026_task2/12_mar_task2_rad_data/13_mar_task2_PT_primary_test.csv
External: CHUS (27 HPV-known)
  Input: Mar_2026_task2/12_mar_task2_rad_data/12_mar_task2_PT_primary_ext.csv

Output: Mar_2026_task2/12_mar_task2_PT_stage1_result.csv
        Mar_2026_task2/12_mar_task2_PT_stage1_features.pkl  (for Stage 2 input)
        Mar_2026_task2/12_mar_task2_PT_stage1_metadata.json

Usage:
    cd "D:/Uppsala thesis"
    python Mar_2026_task2/12_mar_task2_stage1_PT.py
"""

import json
import pickle
import random
import time
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# sys.path must be configured BEFORE importing fs_task2_utils
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.inspection import permutation_importance
from scipy.stats import mannwhitneyu, spearmanr, pearsonr
from statsmodels.stats.multitest import multipletests

from fs_task2_utils import (
    nested_cv_auc,
    evaluate_auc_cv,
    evaluate_auc_test,
    record,
    correlation_filter,
    lasso_logistic_selection,
    elasticnet_logistic_selection,
    stability_selection_logistic,
    RELIEFF_OK,
    MRMR_OK,
    XGB_OK,
    BORUTA_OK,
)
if RELIEFF_OK:
    from skrebate import ReliefF
if MRMR_OK:
    from mrmr import mrmr_classif
if XGB_OK:
    from xgboost import XGBClassifier
if BORUTA_OK:
    from boruta import BorutaPy

# ============================================================
# CONFIGURATION
# ============================================================

SEED           = 42
MODALITY       = 'PT'
RESULTS_BASE   = '12_mar_task2_PT_stage1'
DATA_DIR       = SCRIPT_DIR / '12_mar_task2_rad_data'
OUTPUT_DIR     = SCRIPT_DIR

TRAIN_FILE = DATA_DIR / '13_mar_task2_PT_primary_train.csv'
TEST_FILE  = DATA_DIR / '13_mar_task2_PT_primary_test.csv'
EXT_FILE = DATA_DIR / '12_mar_task2_PT_primary_ext.csv'

EXCLUDE_COLS = ['PatientID', 'HPV_binary', 'Relapse', 'RFS',
                'Age', 'Gender_Male', 'Treatment_CRT']

random.seed(SEED)
np.random.seed(SEED)

print('=' * 80)
print(f'Task 2 Stage 1: {MODALITY} Feature Selection — HPV Classification')
print('=' * 80)

# ============================================================
# VERIFY INPUTS
# ============================================================

for p in [TRAIN_FILE, TEST_FILE, EXT_FILE]:
    if not p.exists():
        print(f'[ERROR] Not found: {p}')
        print('[INFO] Run 13_mar_task2_make_split.py first.')
        sys.exit(1)
    print(f'[OK] {p.name}')

# ============================================================
# LOAD DATA
# ============================================================

print(f'\n[Step 1] Loading {MODALITY} fixed train/test data...')

dev_train = pd.read_csv(TRAIN_FILE)
dev_test  = pd.read_csv(TEST_FILE)
dev = pd.concat([dev_train, dev_test], ignore_index=True)

rad_cols = [c for c in dev_train.columns if c not in EXCLUDE_COLS and c != 'prefix']

X_train = dev_train[rad_cols].copy().reset_index(drop=True)
X_test  = dev_test[rad_cols].copy().reset_index(drop=True)
y_train = dev_train['HPV_binary'].values.astype(int)
y_test  = dev_test['HPV_binary'].values.astype(int)

# Impute on train; apply same medians to test (no leakage).
X_train       = X_train.replace([np.inf, -np.inf], np.nan)
train_medians = X_train.median()
X_train       = X_train.fillna(train_medians)
X_test        = X_test.replace([np.inf, -np.inf], np.nan).fillna(train_medians)

n_pos      = int(y_train.sum())
n_neg      = int((y_train == 0).sum())
n_pos_test = int(y_test.sum())
n_neg_test = int((y_test == 0).sum())

print(f'  Patients:           {len(dev)}')
print(f'  Train:              {len(X_train)} (HPV+: {n_pos}, HPV-: {n_neg})')
print(f'  Internal test:      {len(X_test)} (HPV+: {n_pos_test}, HPV-: {n_neg_test})')
print(f'  Radiomics features: {len(rad_cols)}')

# Secondary split for D-category importance ranking only.
# Importance is computed on X_fs_val (not seen during model fit).
X_fs_train, X_fs_val, y_fs_train, y_fs_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=SEED, stratify=y_train
)
X_fs_train = X_fs_train.reset_index(drop=True)
X_fs_val   = X_fs_val.reset_index(drop=True)

START_TIME = time.time()

results: list[dict] = []
selected_features_storage: dict[str, list] = {}

print('\n[NOTE] All CV_AUC values use nested 5-fold CV (leakage-free, consistent across all methods).')
print('[NOTE] Test_AUC = train-on-X_train, predict-on-X_test — diagnostic only.\n')

# ============================================================
# CATEGORY A: Statistical / Filter Methods
# ============================================================

print(f"{'=' * 60}")
print('CATEGORY A: Statistical / Filter Methods')
print(f"{'=' * 60}")

# ----------------------------------------------------------------
# A0: Mann-Whitney U AUC
# ----------------------------------------------------------------
print('  Running A0: Mann-Whitney U AUC...')
try:
    # Compute full-train ranking once to get the feature list for Test_AUC and pkl.
    mwu_rows = []
    for col in X_train.columns:
        pos_vals = X_train.loc[y_train == 1, col].values
        neg_vals = X_train.loc[y_train == 0, col].values
        stat, p  = mannwhitneyu(pos_vals, neg_vals, alternative='two-sided')
        auc_val  = stat / (len(pos_vals) * len(neg_vals))
        auc_val  = max(auc_val, 1 - auc_val)
        mwu_rows.append({'feature': col, 'auc': auc_val, 'p': p})
    mwu_df = pd.DataFrame(mwu_rows).sort_values('auc', ascending=False)
    _, p_fdr, _, _ = multipletests(mwu_df['p'].values, method='fdr_bh')
    mwu_df['p_fdr'] = p_fdr

    for p_col, label, key in [
        ('p',     'A0: MWU AUC (p<0.05 uncorr)', 'A0_MWU_uncorr'),
        ('p_fdr', 'A0: MWU AUC (p<0.05 BH-FDR)', 'A0_MWU_fdr'),
    ]:
        selected_full = mwu_df[mwu_df[p_col] < 0.05]['feature'].tolist()
        if len(selected_full) == 0:
            print(f'    [{label}] No features passed threshold')
            continue

        def _sel_mwu(X_f, y_f, _p=p_col):
            rows = []
            for col in X_f.columns:
                pv = X_f.loc[y_f == 1, col].values
                nv = X_f.loc[y_f == 0, col].values
                s, p = mannwhitneyu(pv, nv, alternative='two-sided')
                rows.append({'feature': col, 'p': p})
            df = pd.DataFrame(rows)
            if _p == 'p_fdr':
                _, pfdr, _, _ = multipletests(df['p'].values, method='fdr_bh')
                df['p_fdr'] = pfdr
            return df[df[_p] < 0.05]['feature'].tolist()

        auc, std = nested_cv_auc(X_train, y_train, _sel_mwu)
        test_auc = evaluate_auc_test(X_train, y_train, X_test, y_test, selected_full)
        record(results, selected_features_storage, 'A', label, key, selected_full, auc, std, test_auc)
except Exception as e:
    print(f'    ERROR A0: {type(e).__name__}: {e}')

# ----------------------------------------------------------------
# A2: Pearson vs HPV_binary
# ----------------------------------------------------------------
print('  Running A2: Pearson vs HPV_binary...')
try:
    for r_thresh, label, key in [
        (0.10, 'A2: Pearson (r>0.10 p<0.05)', 'A2_Pearson_r10'),
        (0.15, 'A2: Pearson (r>0.15 p<0.05)', 'A2_Pearson_r15'),
    ]:
        selected_full = [c for c in X_train.columns
                         if abs(pearsonr(X_train[c].values, y_train)[0]) > r_thresh
                         and pearsonr(X_train[c].values, y_train)[1] < 0.05]
        if len(selected_full) == 0:
            print(f'    [{label}] No features passed threshold')
            continue

        def _sel_pearson(X_f, y_f, _rt=r_thresh):
            return [c for c in X_f.columns
                    if abs(pearsonr(X_f[c].values, y_f)[0]) > _rt
                    and pearsonr(X_f[c].values, y_f)[1] < 0.05]

        auc, std = nested_cv_auc(X_train, y_train, _sel_pearson)
        test_auc = evaluate_auc_test(X_train, y_train, X_test, y_test, selected_full)
        record(results, selected_features_storage, 'A', label, key, selected_full, auc, std, test_auc)
except Exception as e:
    print(f'    ERROR A2: {type(e).__name__}: {e}')

# ----------------------------------------------------------------
# A3: Spearman vs HPV_binary
# ----------------------------------------------------------------
print('  Running A3: Spearman vs HPV_binary...')
try:
    for r_thresh, label, key in [
        (0.10, 'A3: Spearman (r>0.10 p<0.05)', 'A3_Spearman_r10'),
        (0.15, 'A3: Spearman (r>0.15 p<0.05)', 'A3_Spearman_r15'),
    ]:
        selected_full = [c for c in X_train.columns
                         if abs(spearmanr(X_train[c].values, y_train)[0]) > r_thresh
                         and spearmanr(X_train[c].values, y_train)[1] < 0.05]
        if len(selected_full) == 0:
            print(f'    [{label}] No features passed threshold')
            continue

        def _sel_spearman(X_f, y_f, _rt=r_thresh):
            return [c for c in X_f.columns
                    if abs(spearmanr(X_f[c].values, y_f)[0]) > _rt
                    and spearmanr(X_f[c].values, y_f)[1] < 0.05]

        auc, std = nested_cv_auc(X_train, y_train, _sel_spearman)
        test_auc = evaluate_auc_test(X_train, y_train, X_test, y_test, selected_full)
        record(results, selected_features_storage, 'A', label, key, selected_full, auc, std, test_auc)
except Exception as e:
    print(f'    ERROR A3: {type(e).__name__}: {e}')

# ----------------------------------------------------------------
# A4: Mutual Information (classif)
# ----------------------------------------------------------------
print('  Running A4: Mutual Information (classif)...')
try:
    mi_scores_full = mutual_info_classif(X_train.values, y_train, random_state=SEED)
    mi_order_full  = np.argsort(mi_scores_full)[::-1]
    for k, label, key in [
        (60,  'A4: MI classif (top 60)',  'A4_MI_60'),
        (100, 'A4: MI classif (top 100)', 'A4_MI_100'),
    ]:
        selected_full = [X_train.columns[i] for i in mi_order_full[:k]]

        def _sel_mi(X_f, y_f, _k=k):
            sc = mutual_info_classif(X_f.values, y_f, random_state=SEED)
            return [X_f.columns[i] for i in np.argsort(sc)[::-1][:_k]]

        auc, std = nested_cv_auc(X_train, y_train, _sel_mi)
        test_auc = evaluate_auc_test(X_train, y_train, X_test, y_test, selected_full)
        record(results, selected_features_storage, 'A', label, key, selected_full, auc, std, test_auc)
except Exception as e:
    print(f'    ERROR A4: {type(e).__name__}: {e}')

# ----------------------------------------------------------------
# A6: ReliefF (classification)
# ----------------------------------------------------------------
print('  Running A6: ReliefF (classification)...')
if not RELIEFF_OK:
    print('    [WARNING] skrebate not available — skipping A6')
if RELIEFF_OK:
    try:
        for k, label, key in [
            (50,  'A6: ReliefF (top 50,  n=50)', 'A6_ReliefF_50'),
            (100, 'A6: ReliefF (top 100, n=50)', 'A6_ReliefF_100'),
        ]:
            rf_sel = ReliefF(n_features_to_select=k, n_neighbors=50)
            rf_sel.fit(X_train.values, y_train)
            selected_full = [X_train.columns[i]
                             for i in np.argsort(rf_sel.feature_importances_)[::-1][:k]]

            def _sel_relief(X_f, y_f, _k=k):
                rs = ReliefF(n_features_to_select=_k, n_neighbors=min(50, len(y_f) - 1))
                rs.fit(X_f.values, y_f)
                return [X_f.columns[i]
                        for i in np.argsort(rs.feature_importances_)[::-1][:_k]]

            auc, std = nested_cv_auc(X_train, y_train, _sel_relief)
            test_auc = evaluate_auc_test(X_train, y_train, X_test, y_test, selected_full)
            record(results, selected_features_storage, 'A', label, key, selected_full, auc, std, test_auc)
    except Exception as e:
        print(f'    ERROR A6: {type(e).__name__}: {e}')

# ----------------------------------------------------------------
# A7: ANOVA F-test (f_classif)
# ----------------------------------------------------------------
print('  Running A7: ANOVA f_classif...')
try:
    f_scores_full, _ = f_classif(X_train.values, y_train)
    f_order_full     = np.argsort(f_scores_full)[::-1]
    for k, label, key in [
        (100, 'A7: ANOVA (top 100)', 'A7_ANOVA_100'),
        (200, 'A7: ANOVA (top 200)', 'A7_ANOVA_200'),
    ]:
        selected_full = [X_train.columns[i] for i in f_order_full[:k]]

        def _sel_anova(X_f, y_f, _k=k):
            fs, _ = f_classif(X_f.values, y_f)
            return [X_f.columns[i] for i in np.argsort(fs)[::-1][:_k]]

        auc, std = nested_cv_auc(X_train, y_train, _sel_anova)
        test_auc = evaluate_auc_test(X_train, y_train, X_test, y_test, selected_full)
        record(results, selected_features_storage, 'A', label, key, selected_full, auc, std, test_auc)
except Exception as e:
    print(f'    ERROR A7: {type(e).__name__}: {e}')

print(f'  Category A done. ({len(results)} methods so far)')

# ============================================================
# CATEGORY B: Redundancy Removal Methods
# ============================================================

print(f"\n{'=' * 60}")
print('CATEGORY B: Redundancy Removal Methods')
print(f"{'=' * 60}")

# ----------------------------------------------------------------
# B1: Correlation filter (label-agnostic)
# ----------------------------------------------------------------
print('  Running B1: Correlation filter...')
try:
    for thresh, label, key in [
        (0.85, 'B1: Corr filter (r<0.85)', 'B1_Corr_085'),
        (0.90, 'B1: Corr filter (r<0.90)', 'B1_Corr_090'),
    ]:
        selected_full = correlation_filter(X_train, thresh)
        if len(selected_full) > 500:
            print(f'    [{label}] {len(selected_full)} features — too many for CV, recording count only')
            results.append({'Category': 'B', 'Method': label,
                            'CV_AUC': float('nan'), 'CV_Std': float('nan'),
                            'Test_AUC': float('nan'), 'Features': len(selected_full)})
            selected_features_storage[key] = selected_full
            continue
        # Label-agnostic selector — same threshold applied inside each fold
        def _sel_corr(X_f, y_f, _t=thresh):
            return correlation_filter(X_f, _t)
        auc, std = nested_cv_auc(X_train, y_train, _sel_corr)
        test_auc = evaluate_auc_test(X_train, y_train, X_test, y_test, selected_full)
        record(results, selected_features_storage, 'B', label, key, selected_full, auc, std, test_auc)
except Exception as e:
    print(f'    ERROR B1: {type(e).__name__}: {e}')

# ----------------------------------------------------------------
# B2: Variance threshold (label-agnostic, percentile-based)
# ----------------------------------------------------------------
print('  Running B2: Variance threshold...')
try:
    variances = X_train.var()
    for pct, label, key in [
        (50, 'B2: Variance threshold (top 50%ile)', 'B2_Variance_p50'),
        (75, 'B2: Variance threshold (top 25%ile)', 'B2_Variance_p75'),
    ]:
        thresh_v  = np.percentile(variances.values, pct)
        selected_full = variances[variances >= thresh_v].index.tolist()
        if len(selected_full) > 500:
            print(f'    [{label}] {len(selected_full)} features — recording count only')
            results.append({'Category': 'B', 'Method': label,
                            'CV_AUC': float('nan'), 'CV_Std': float('nan'),
                            'Test_AUC': float('nan'), 'Features': len(selected_full)})
            selected_features_storage[key] = selected_full
        else:
            def _sel_var(X_f, y_f, _pct=pct):
                vv = X_f.var()
                return vv[vv >= np.percentile(vv.values, _pct)].index.tolist()
            auc, std = nested_cv_auc(X_train, y_train, _sel_var)
            test_auc = evaluate_auc_test(X_train, y_train, X_test, y_test, selected_full)
            record(results, selected_features_storage, 'B', label, key, selected_full, auc, std, test_auc)
except Exception as e:
    print(f'    ERROR B2: {type(e).__name__}: {e}')

# ----------------------------------------------------------------
# B3: mRMR classification
# ----------------------------------------------------------------
print('  Running B3: mRMR classif...')
if not MRMR_OK:
    print('    [WARNING] mrmr-selection not available — skipping B3')
if MRMR_OK:
    try:
        for k, label, key in [
            (30, 'B3: mRMR classif (top 30)', 'B3_mRMR_30'),
            (50, 'B3: mRMR classif (top 50)', 'B3_mRMR_50'),
        ]:
            selected_full = mrmr_classif(X=X_train, y=pd.Series(y_train), K=k)

            def _sel_mrmr(X_f, y_f, _k=k):
                return mrmr_classif(X=X_f, y=pd.Series(y_f), K=_k)

            auc, std = nested_cv_auc(X_train, y_train, _sel_mrmr)
            test_auc = evaluate_auc_test(X_train, y_train, X_test, y_test, selected_full)
            record(results, selected_features_storage, 'B', label, key, selected_full, auc, std, test_auc)
    except Exception as e:
        print(f'    ERROR B3: {type(e).__name__}: {e}')

print(f'  Category B done. ({len(results)} methods so far)')

# ============================================================
# CATEGORY C: Regularisation Methods (nested CV)
# ============================================================

print(f"\n{'=' * 60}")
print('CATEGORY C: Regularisation Methods (Logistic)')
print(f"{'=' * 60}")

# C1: LASSO-Logistic
print('  Running C1: LASSO-Logistic...')
try:
    for target, label, key in [
        (30, 'C1: LASSO-Logistic (target 30)', 'C1_LASSO_30'),
        (60, 'C1: LASSO-Logistic (target 60)', 'C1_LASSO_60'),
    ]:
        # Full-train selection for the pkl/test_auc reference feature list
        selected_full = lasso_logistic_selection(X_train, y_train, target_features=target)
        print(f'      [C1 full-train target={target}] selected {len(selected_full)} features')
        if len(selected_full) == 0:
            print(f'    [{label}] No features selected')
            continue

        def _sel_lasso(X_f, y_f, _t=target):
            return lasso_logistic_selection(X_f, y_f, target_features=_t)

        auc, std = nested_cv_auc(X_train, y_train, _sel_lasso)
        test_auc = evaluate_auc_test(X_train, y_train, X_test, y_test, selected_full)
        record(results, selected_features_storage, 'C', label, key, selected_full, auc, std, test_auc)
except Exception as e:
    print(f'    ERROR C1: {type(e).__name__}: {e}')

# C2: ElasticNet-Logistic
print('  Running C2: ElasticNet-Logistic...')
try:
    for target, label, key in [
        (30, 'C2: ElasticNet-Logistic (target 30)', 'C2_EN_30'),
        (60, 'C2: ElasticNet-Logistic (target 60)', 'C2_EN_60'),
    ]:
        selected_full = elasticnet_logistic_selection(X_train, y_train, target_features=target)
        print(f'      [C2 full-train target={target}] selected {len(selected_full)} features')
        if len(selected_full) == 0:
            print(f'    [{label}] No features selected')
            continue

        def _sel_en(X_f, y_f, _t=target):
            return elasticnet_logistic_selection(X_f, y_f, target_features=_t)

        auc, std = nested_cv_auc(X_train, y_train, _sel_en)
        test_auc = evaluate_auc_test(X_train, y_train, X_test, y_test, selected_full)
        record(results, selected_features_storage, 'C', label, key, selected_full, auc, std, test_auc)
except Exception as e:
    print(f'    ERROR C2: {type(e).__name__}: {e}')

# C3: Stability Selection (top-30 by frequency — threshold always empty at this n/p)
print('  Running C3: Stability Selection (logistic base, top-30 fallback)...')
try:
    selected_full = stability_selection_logistic(X_train, y_train, n_bootstrap=100,
                                                 stability_threshold=0, n_features=30)
    if len(selected_full) == 0:
        print('    [C3] No features selected')
    else:
        def _sel_stab(X_f, y_f):
            return stability_selection_logistic(X_f, y_f, n_bootstrap=50,
                                               stability_threshold=0, n_features=30)

        auc, std = nested_cv_auc(X_train, y_train, _sel_stab)
        test_auc = evaluate_auc_test(X_train, y_train, X_test, y_test, selected_full)
        record(results, selected_features_storage, 'C',
               'C3: Stability (top-30 fallback)', 'C3_Stab_top30',
               selected_full, auc, std, test_auc)
except Exception as e:
    print(f'    ERROR C3: {type(e).__name__}: {e}')

print(f'  Category C done. ({len(results)} methods so far)')

# ============================================================
# CATEGORY D: ML-Based Importance Ranking
# ============================================================

print(f"\n{'=' * 60}")
print('CATEGORY D: ML-Based Importance (Classification)')
print(f"{'=' * 60}")
print('  [D] Importance ranked on X_fs_val (held-out) for selected_full/Test_AUC reference.')
print('  [D] CV_AUC uses nested 5-fold CV with inner fs split inside each fold (leakage-free).')

# D1: Random Forest PermImp
print('  Running D1: RF PermImp...')
try:
    for k, label, key in [
        (50, 'D1: RF PermImp (top 50)', 'D1_RF_50'),
        (80, 'D1: RF PermImp (top 80)', 'D1_RF_80'),
    ]:
        # Full-train selection for selected_full/test_auc reference
        clf = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                     random_state=SEED, n_jobs=-1)
        clf.fit(X_fs_train.values, y_fs_train)
        imp = permutation_importance(clf, X_fs_val.values, y_fs_val, n_repeats=10,
                                     random_state=SEED, n_jobs=-1)
        order    = np.argsort(imp.importances_mean)[::-1]
        selected = [X_train.columns[i] for i in order[:k]]
        # Nested CV — inner fs split inside each fold
        def _sel_d1(X_f, y_f, _k=k):
            Xfst, Xfsv, yfst, yfsv = train_test_split(
                X_f.values, y_f, test_size=0.25, random_state=SEED, stratify=y_f)
            c = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                       random_state=SEED, n_jobs=-1)
            c.fit(Xfst, yfst)
            pi = permutation_importance(c, Xfsv, yfsv, n_repeats=10,
                                        random_state=SEED, n_jobs=-1)
            return [X_f.columns[i] for i in np.argsort(pi.importances_mean)[::-1][:_k]]
        auc, std = nested_cv_auc(X_train, y_train, _sel_d1)
        test_auc = evaluate_auc_test(X_train, y_train, X_test, y_test, selected)
        record(results, selected_features_storage, 'D', label, key, selected, auc, std, test_auc)
except Exception as e:
    print(f'    ERROR D1: {type(e).__name__}: {e}')

# D2: XGBoost
print('  Running D2: XGBoost classif...')
if not XGB_OK:
    print('    [WARNING] xgboost not available — skipping D2')
if XGB_OK:
    try:
        scale_pos = n_neg / n_pos if n_pos > 0 else 1.0
        for k, label, key in [
            (50, 'D2: XGBoost classif (top 50)', 'D2_XGB_50'),
            (80, 'D2: XGBoost classif (top 80)', 'D2_XGB_80'),
        ]:
            # Full-train selection for selected_full/test_auc reference
            clf = XGBClassifier(n_estimators=100, scale_pos_weight=scale_pos,
                                use_label_encoder=False, eval_metric='logloss',
                                random_state=SEED, verbosity=0)
            clf.fit(X_fs_train.values, y_fs_train)
            imp = permutation_importance(clf, X_fs_val.values, y_fs_val, n_repeats=10,
                                         random_state=SEED, n_jobs=-1)
            order    = np.argsort(imp.importances_mean)[::-1]
            selected = [X_train.columns[i] for i in order[:k]]
            # Nested CV — inner fs split inside each fold
            def _sel_d2(X_f, y_f, _k=k):
                sp = int((y_f == 0).sum()) / max(int((y_f == 1).sum()), 1)
                Xfst, Xfsv, yfst, yfsv = train_test_split(
                    X_f.values, y_f, test_size=0.25, random_state=SEED, stratify=y_f)
                c = XGBClassifier(n_estimators=100, scale_pos_weight=sp,
                                  use_label_encoder=False, eval_metric='logloss',
                                  random_state=SEED, verbosity=0)
                c.fit(Xfst, yfst)
                pi = permutation_importance(c, Xfsv, yfsv, n_repeats=10,
                                            random_state=SEED, n_jobs=-1)
                return [X_f.columns[i] for i in np.argsort(pi.importances_mean)[::-1][:_k]]
            auc, std = nested_cv_auc(X_train, y_train, _sel_d2)
            test_auc = evaluate_auc_test(X_train, y_train, X_test, y_test, selected)
            record(results, selected_features_storage, 'D', label, key, selected, auc, std, test_auc)
    except Exception as e:
        print(f'    ERROR D2: {type(e).__name__}: {e}')

# D3: GradientBoosting PermImp
print('  Running D3: GradientBoosting PermImp...')
try:
    for k, label, key in [
        (50, 'D3: GB PermImp (top 50)', 'D3_GB_50'),
        (80, 'D3: GB PermImp (top 80)', 'D3_GB_80'),
    ]:
        # Full-train selection for selected_full/test_auc reference
        clf = GradientBoostingClassifier(n_estimators=500, random_state=SEED)
        clf.fit(X_fs_train.values, y_fs_train)
        imp  = permutation_importance(clf, X_fs_val.values, y_fs_val, n_repeats=10,
                                      random_state=SEED, n_jobs=-1)
        order    = np.argsort(imp.importances_mean)[::-1]
        selected = [X_train.columns[i] for i in order[:k]]
        # Nested CV — inner fs split inside each fold (200 estimators for speed)
        def _sel_d3(X_f, y_f, _k=k):
            Xfst, Xfsv, yfst, yfsv = train_test_split(
                X_f.values, y_f, test_size=0.25, random_state=SEED, stratify=y_f)
            c = GradientBoostingClassifier(n_estimators=200, random_state=SEED)
            c.fit(Xfst, yfst)
            pi = permutation_importance(c, Xfsv, yfsv, n_repeats=10,
                                        random_state=SEED, n_jobs=-1)
            return [X_f.columns[i] for i in np.argsort(pi.importances_mean)[::-1][:_k]]
        auc, std = nested_cv_auc(X_train, y_train, _sel_d3)
        test_auc = evaluate_auc_test(X_train, y_train, X_test, y_test, selected)
        record(results, selected_features_storage, 'D', label, key, selected, auc, std, test_auc)
except Exception as e:
    print(f'    ERROR D3: {type(e).__name__}: {e}')

# D4: Extra Trees PermImp
print('  Running D4: Extra Trees PermImp...')
try:
    for k, label, key in [
        (50, 'D4: ExtraTrees PermImp (top 50)', 'D4_ET_50'),
        (80, 'D4: ExtraTrees PermImp (top 80)', 'D4_ET_80'),
    ]:
        # Full-train selection for selected_full/test_auc reference
        clf = ExtraTreesClassifier(n_estimators=500, class_weight='balanced',
                                   random_state=SEED, n_jobs=-1)
        clf.fit(X_fs_train.values, y_fs_train)
        imp  = permutation_importance(clf, X_fs_val.values, y_fs_val, n_repeats=10,
                                      random_state=SEED, n_jobs=-1)
        order    = np.argsort(imp.importances_mean)[::-1]
        selected = [X_train.columns[i] for i in order[:k]]
        # Nested CV — inner fs split inside each fold
        def _sel_d4(X_f, y_f, _k=k):
            Xfst, Xfsv, yfst, yfsv = train_test_split(
                X_f.values, y_f, test_size=0.25, random_state=SEED, stratify=y_f)
            c = ExtraTreesClassifier(n_estimators=500, class_weight='balanced',
                                     random_state=SEED, n_jobs=-1)
            c.fit(Xfst, yfst)
            pi = permutation_importance(c, Xfsv, yfsv, n_repeats=10,
                                        random_state=SEED, n_jobs=-1)
            return [X_f.columns[i] for i in np.argsort(pi.importances_mean)[::-1][:_k]]
        auc, std = nested_cv_auc(X_train, y_train, _sel_d4)
        test_auc = evaluate_auc_test(X_train, y_train, X_test, y_test, selected)
        record(results, selected_features_storage, 'D', label, key, selected, auc, std, test_auc)
except Exception as e:
    print(f'    ERROR D4: {type(e).__name__}: {e}')

# D5: Boruta — importance-ranked top-k
# At n~52 (X_fs_train), Boruta's shadow-feature test is too strict to confirm
# features at p<0.05. Instead we run Boruta to obtain its RF importance ranking,
# then take top-30 and top-50 by rank. Confirmed/tentative flags are reported
# for transparency but do not gate selection.
print('  Running D5: Boruta (top-k by importance rank)...')
if not BORUTA_OK:
    print('    [WARNING] boruta not available — skipping D5 (pip install boruta)')
if BORUTA_OK:
    try:
        rf_base = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                         random_state=SEED, n_jobs=-1)
        boruta_sel = BorutaPy(rf_base, n_estimators='auto', verbose=0,
                              random_state=SEED, max_iter=100)
        boruta_sel.fit(X_fs_train.values, y_fs_train)
        n_confirmed  = int(boruta_sel.support_.sum())
        n_tentative  = int(boruta_sel.support_weak_.sum())
        print(f'    [D5] Boruta confirmed: {n_confirmed}, tentative: {n_tentative}')
        # Rank all features by Boruta's internal RF importance
        imp_order = np.argsort(boruta_sel.importance_history_.mean(axis=0))[::-1]
        for k, label, key in [
            (30, 'D5: Boruta rank (top 30)', 'D5_Boruta_top30'),
            (50, 'D5: Boruta rank (top 50)', 'D5_Boruta_top50'),
        ]:
            selected = [X_fs_train.columns[i] for i in imp_order[:k]]
            # Nested CV — inner Boruta fit inside each fold
            def _sel_d5(X_f, y_f, _k=k):
                Xfst, _, yfst, _ = train_test_split(
                    X_f.values, y_f, test_size=0.25, random_state=SEED, stratify=y_f)
                rf_i = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                              random_state=SEED, n_jobs=-1)
                bor_i = BorutaPy(rf_i, n_estimators='auto', verbose=0,
                                 random_state=SEED, max_iter=100)
                bor_i.fit(Xfst, yfst)
                imp_o = np.argsort(bor_i.importance_history_.mean(axis=0))[::-1]
                return [X_f.columns[i] for i in imp_o[:_k]]
            auc, std = nested_cv_auc(X_train, y_train, _sel_d5)
            test_auc = evaluate_auc_test(X_train, y_train, X_test, y_test, selected)
            record(results, selected_features_storage, 'D', label, key, selected, auc, std, test_auc)
    except Exception as e:
        print(f'    ERROR D5: {type(e).__name__}: {e}')

print(f'  Category D done. ({len(results)} methods so far)')

# ============================================================
# BASELINE: Random 500 features
# ============================================================

print(f"\n{'=' * 60}")
print('BASELINE: Random 500 features from full pool')
print(f"{'=' * 60}")

try:
    rng_cols  = random.sample(list(X_train.columns), min(500, len(X_train.columns)))
    # Nested CV — random selector lambda (fresh random sample per fold)
    def _sel_baseline(X_f, y_f):
        return random.sample(list(X_f.columns), min(500, X_f.shape[1]))
    auc, std  = nested_cv_auc(X_train, y_train, _sel_baseline)
    test_auc  = evaluate_auc_test(X_train, y_train, X_test, y_test, rng_cols)
    record(results, selected_features_storage, 'Baseline',
           'Baseline: Random 500', 'Baseline_Random500', rng_cols, auc, std, test_auc)
except Exception as e:
    print(f'    ERROR Baseline: {type(e).__name__}: {e}')

# ============================================================
# SAVE RESULTS
# ============================================================

print(f"\n{'=' * 60}")
print('SAVING RESULTS')
print(f"{'=' * 60}")

execution_time = time.time() - START_TIME

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('CV_AUC', ascending=False).reset_index(drop=True)
results_df.insert(0, 'Rank', [f'PT_{i}' for i in range(1, len(results_df) + 1)])
results_df = results_df[['Rank', 'Category', 'Method', 'CV_AUC', 'CV_Std', 'Test_AUC', 'Features']]

out_csv  = OUTPUT_DIR / f'{RESULTS_BASE}_result.csv'
out_feat = OUTPUT_DIR / f'{RESULTS_BASE}_features.pkl'
out_meta = OUTPUT_DIR / f'{RESULTS_BASE}_metadata.json'

results_df.to_csv(out_csv, index=False)

with open(out_feat, 'wb') as f:
    pickle.dump(selected_features_storage, f)

metadata = {
    'modality':           MODALITY,
    'timestamp':          time.strftime('%Y-%m-%d %H:%M:%S'),
    'execution_time_s':   execution_time,
    'seed':               SEED,
    'n_patients_dev':     len(dev),
    'n_train':            len(X_train),
    'n_test_internal':    len(X_test),
    'n_hpv_pos_train':    n_pos,
    'n_hpv_neg_train':    n_neg,
    'n_hpv_pos_test':     n_pos_test,
    'n_hpv_neg_test':     n_neg_test,
    'n_radiomics_feats':  len(rad_cols),
    'n_total_candidates': X_train.shape[1],
    'n_methods':          len(results_df),
    'top_method':         results_df.iloc[0]['Method'] if len(results_df) else '',
    'top_cv_auc':         float(results_df.iloc[0]['CV_AUC']) if len(results_df) else float('nan'),
    'cv_note':            'nested 5-fold: FS runs inside each outer fold',
    'test_auc_note':      'test AUC = train-on-X_train predict-on-X_test, diagnostic only',
    'dev_file':           f'{TRAIN_FILE.name} + {TEST_FILE.name}',
    'ext_file':           EXT_FILE.name,
    'split':              'no-MDA A1 primary',
    'split_source':       'fixed train/test csv from 13_mar_task2_make_split.py',
    'external_centre':    'CHUS',
}

with open(out_meta, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f'  Results CSV:     {out_csv.name}')
print(f'  Features pkl:    {out_feat.name}')
print(f'  Metadata JSON:   {out_meta.name}')
print(f'  Total methods:   {len(results_df)}')
print(f'  Execution time:  {execution_time / 60:.1f} min')
print(f'\n  Top 10 methods:')
print(results_df.head(10)[['Rank', 'Method', 'CV_AUC', 'CV_Std', 'Test_AUC', 'Features']].to_string(index=False))

print(f'\n{"=" * 80}')
print(f'PT STAGE 1 COMPLETE')
print(f'{"=" * 80}')
print(f'  Next: 12_mar_task2_stage1_CT.py (parallel run on another machine)')
print(f'  Then: Stage 2 — two-step pipeline chaining on top-ranked Stage 1 methods')
print(f'{"=" * 80}')
