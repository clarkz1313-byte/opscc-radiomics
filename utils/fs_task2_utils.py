# -*- coding: utf-8 -*-
"""
fs_task2_utils.py - Shared Feature Selection Utilities for Task 2 (HPV Classification)
========================================================================================
Classification-specific helpers for the 4-stage FS pipeline (Stage 1-4).
All functions operate on binary HPV classification (HPV_binary 0/1).

Distinct from fs_utils.py which is survival-analysis only (Task 1).

Usage:
    from Mar_2026_task2.fs_task2_utils import (
        evaluate_auc_cv, nested_cv_auc, evaluate_auc_test, record,
        correlation_filter,
        lasso_logistic_selection, elasticnet_logistic_selection,
        stability_selection_logistic,
    )
"""

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# Optional imports - callers check the *_OK flags before using
try:
    from skrebate import ReliefF
    RELIEFF_OK = True
except ImportError:
    RELIEFF_OK = False

try:
    from mrmr import mrmr_classif
    MRMR_OK = True
except ImportError:
    MRMR_OK = False

try:
    from xgboost import XGBClassifier
    XGB_OK = True
except ImportError:
    XGB_OK = False

try:
    from boruta import BorutaPy
    BORUTA_OK = True
except ImportError:
    BORUTA_OK = False


# ============================================================
# CORE EVALUATION
# ============================================================

def evaluate_auc_cv(X_df: pd.DataFrame,
                    y_arr: np.ndarray,
                    selected: list,
                    random_state: int = 42) -> tuple[float, float]:
    """5-fold stratified CV AUC via Pipeline(StandardScaler + LogisticRegression).

    NOTE: This function is LEAKY for any method whose feature selection used
    y_arr over the full X_df (C methods, A filter methods, B3 mRMR, etc.).
    Use nested_cv_auc() instead for those methods.
    This function is only appropriate when the feature set was selected on
    truly independent data (e.g. D methods with X_fs_train/X_fs_val split).

    Parameters
    ----------
    X_df : pd.DataFrame
        Full feature matrix (train partition only).
    y_arr : np.ndarray
        Binary label array aligned with X_df rows.
    selected : list
        Column names to evaluate.
    random_state : int

    Returns
    -------
    (mean_auc, std_auc) - both NaN if selected is empty.
    """
    if len(selected) == 0:
        return float('nan'), float('nan')
    Xs = X_df[selected].values
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('logit', LogisticRegression(C=1.0, class_weight='balanced',
                                     solver='saga', max_iter=3000,
                                     random_state=random_state)),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scores = cross_val_score(clf, Xs, y_arr, cv=cv, scoring='roc_auc')
    return float(scores.mean()), float(scores.std())


def nested_cv_auc(X_df: pd.DataFrame,
                  y_arr: np.ndarray,
                  selector_fn: Callable[[pd.DataFrame, np.ndarray], list],
                  random_state: int = 42) -> tuple[float, float]:
    """Leakage-free nested 5-fold CV AUC.

    For each outer fold, the selector_fn runs only on that fold's train split,
    then the classifier is evaluated on the held-out fold. This means feature
    selection never sees the test fold labels — eliminating selection bias.

    Parameters
    ----------
    X_df : pd.DataFrame
        Full feature matrix (train partition only — never include test data).
    y_arr : np.ndarray
        Binary label array aligned with X_df rows.
    selector_fn : callable(X_fold_train_df, y_fold_train) -> list[str]
        Function that receives the fold's train DataFrame and labels, returns
        a list of selected column names. Must not access any data outside its
        two arguments.
    random_state : int

    Returns
    -------
    (mean_auc, std_auc) — NaN if all folds produced empty selections.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    X_arr = X_df.values
    cols  = list(X_df.columns)
    fold_aucs = []

    for train_idx, val_idx in cv.split(X_arr, y_arr):
        X_fold_tr = pd.DataFrame(X_arr[train_idx], columns=cols)
        y_fold_tr = y_arr[train_idx]
        X_fold_val = X_arr[val_idx]
        y_fold_val = y_arr[val_idx]

        selected = selector_fn(X_fold_tr, y_fold_tr)
        if len(selected) == 0:
            continue

        col_idx = [cols.index(c) for c in selected]
        scaler = StandardScaler()
        Xtr_s  = scaler.fit_transform(X_fold_tr[selected].values)
        Xval_s = scaler.transform(X_fold_val[:, col_idx])

        clf = LogisticRegression(C=1.0, class_weight='balanced',
                                 solver='saga', max_iter=3000,
                                 random_state=random_state)
        clf.fit(Xtr_s, y_fold_tr)

        if len(np.unique(y_fold_val)) < 2:
            continue  # can't compute AUC with one class in fold
        proba = clf.predict_proba(Xval_s)[:, 1]
        fold_aucs.append(roc_auc_score(y_fold_val, proba))

    if len(fold_aucs) == 0:
        return float('nan'), float('nan')
    return float(np.mean(fold_aucs)), float(np.std(fold_aucs))


def evaluate_auc_test(X_train_df: pd.DataFrame,
                      y_train: np.ndarray,
                      X_test_df: pd.DataFrame,
                      y_test: np.ndarray,
                      selected: list,
                      random_state: int = 42) -> float:
    """Evaluate a fixed feature set on the held-out test set.

    Fits StandardScaler + LogisticRegression on the full train set, predicts
    on test set. Used only for leakage/overfit diagnostics — never for
    selection or ranking.

    Parameters
    ----------
    X_train_df : pd.DataFrame
    y_train : np.ndarray
    X_test_df : pd.DataFrame
    y_test : np.ndarray
    selected : list of column names
    random_state : int

    Returns
    -------
    float AUC on test set, or NaN if selected is empty or test has one class.
    """
    if len(selected) == 0:
        return float('nan')
    if len(np.unique(y_test)) < 2:
        return float('nan')
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train_df[selected].values)
    Xte = scaler.transform(X_test_df[selected].values)
    clf = LogisticRegression(C=1.0, class_weight='balanced',
                             solver='saga', max_iter=3000,
                             random_state=random_state)
    clf.fit(Xtr, y_train)
    proba = clf.predict_proba(Xte)[:, 1]
    return float(roc_auc_score(y_test, proba))


def record(results: list, selected_features_storage: dict,
           category: str, method_name: str, storage_key: str,
           selected: list, auc: float, std: float,
           test_auc: float = float('nan')) -> None:
    """Append one result row and store selected feature list."""
    results.append({
        'Category': category,
        'Method':   method_name,
        'CV_AUC':   auc,
        'CV_Std':   std,
        'Test_AUC': test_auc,
        'Features': len(selected),
    })
    selected_features_storage[storage_key] = selected
    test_str = f'  Test: {test_auc:.4f}' if not np.isnan(test_auc) else ''
    print(f'    CV: {auc:.4f} +/- {std:.4f}  |{test_str}  Features: {len(selected)}')


# ============================================================
# CATEGORY B: REDUNDANCY REMOVAL
# ============================================================

def correlation_filter(X_df: pd.DataFrame, threshold: float) -> list:
    """Remove features with pairwise Pearson |r| > threshold.

    Greedy: for each pair above threshold, drops the second column encountered
    in upper-triangle order (same as fs_utils.py behaviour).

    Parameters
    ----------
    X_df : pd.DataFrame
        Feature matrix (train only).
    threshold : float
        Correlation threshold, e.g. 0.85 or 0.90.

    Returns
    -------
    list of surviving column names.
    """
    corr_matrix = X_df.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return [c for c in X_df.columns if c not in drop]


# ============================================================
# CATEGORY C: REGULARISED LOGISTIC SELECTION
# ============================================================

def lasso_logistic_selection(X_df: pd.DataFrame,
                              y_arr: np.ndarray,
                              target_features: int,
                              random_state: int = 42) -> list:
    """LASSO-Logistic: StandardScale then sweep C grid to hit target_features.

    Uses a dense C grid in the sparse-to-mid regularisation zone where non-zero
    solutions emerge at p >> n (2818 features, n=69 samples).

    Parameters
    ----------
    X_df : pd.DataFrame
        Feature matrix (fold train only when called from nested_cv_auc).
    y_arr : np.ndarray
        Binary label array.
    target_features : int
        Approximate number of non-zero features desired.
    random_state : int

    Returns
    -------
    list of selected column names (closest to target_features found).
    """
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_df.values)
    cols = list(X_df.columns)
    best_selected, best_diff = [], float('inf')
    best_c = None
    for C in np.logspace(-4, 0, 240):
        clf = LogisticRegression(l1_ratio=1, C=C, solver='saga',
                                 class_weight='balanced', max_iter=5000,
                                 random_state=random_state)
        clf.fit(Xs, y_arr)
        sel = [cols[i] for i, c in enumerate(clf.coef_[0]) if c != 0]
        diff = abs(len(sel) - target_features)
        if diff < best_diff:
            best_diff, best_selected = diff, sel
            best_c = C
            if best_diff == 0:
                break   # exact match - no need to continue
    return best_selected


def elasticnet_logistic_selection(X_df: pd.DataFrame,
                                   y_arr: np.ndarray,
                                   target_features: int,
                                   l1_ratio: float = 0.5,
                                   random_state: int = 42) -> list:
    """ElasticNet-Logistic: StandardScale then sweep C grid to hit target_features.

    Parameters
    ----------
    X_df : pd.DataFrame
    y_arr : np.ndarray
    target_features : int
    l1_ratio : float
        Mix between L1 (1.0) and L2 (0.0). Default 0.5.
    random_state : int

    Returns
    -------
    list of selected column names.
    """
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_df.values)
    cols = list(X_df.columns)
    best_selected, best_diff = [], float('inf')
    best_c = None
    for C in np.logspace(-4, 0, 240):
        clf = LogisticRegression(l1_ratio=l1_ratio, C=C,
                                 solver='saga', class_weight='balanced',
                                 max_iter=5000, random_state=random_state)
        clf.fit(Xs, y_arr)
        sel = [cols[i] for i, c in enumerate(clf.coef_[0]) if c != 0]
        diff = abs(len(sel) - target_features)
        if diff < best_diff:
            best_diff, best_selected = diff, sel
            best_c = C
            if best_diff == 0:
                break
    return best_selected


def stability_selection_logistic(X_df: pd.DataFrame,
                                  y_arr: np.ndarray,
                                  n_bootstrap: int = 100,
                                  stability_threshold: float = 0.6,
                                  n_features: int = 30,
                                  random_state: int = 42) -> list:
    """Bootstrap stability selection with L1-Logistic.

    Scales once on full X_df, then runs n_bootstrap subsamples (80%).
    Uses low-to-mid C values [1e-4, 5e-4, 1e-3, 5e-3, 1e-2] to
    produce sparse per-bootstrap selections in the p >> n regime.

    Parameters
    ----------
    X_df : pd.DataFrame
    y_arr : np.ndarray
    n_bootstrap : int
    stability_threshold : float
        Fraction of bootstraps a feature must be selected in.
        If 0, returns top n_features by frequency instead.
    n_features : int
        Used when stability_threshold=0 (top-k fallback).
    random_state : int

    Returns
    -------
    list of selected column names.
    """
    scaler = StandardScaler()
    Xs_full = scaler.fit_transform(X_df.values)
    counts = np.zeros(X_df.shape[1])
    rng = np.random.RandomState(random_state)
    c_values = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_arr), size=int(0.8 * len(y_arr)), replace=False)
        Xb, yb = Xs_full[idx], y_arr[idx]
        for C in c_values:
            clf = LogisticRegression(l1_ratio=1, C=C, solver='saga',
                                     class_weight='balanced', max_iter=5000,
                                     random_state=random_state)
            try:
                clf.fit(Xb, yb)
                counts += (clf.coef_[0] != 0).astype(int)
            except Exception:
                pass
    freq = counts / (n_bootstrap * len(c_values))
    if stability_threshold > 0:
        selected_idx = np.where(freq >= stability_threshold)[0]
    else:
        selected_idx = np.argsort(freq)[::-1][:n_features]
    return [X_df.columns[i] for i in selected_idx]


# ============================================================
# AVAILABILITY FLAGS (re-exported for callers)
# ============================================================
# Callers can do:
#   from Mar_2026_task2.fs_task2_utils import RELIEFF_OK, MRMR_OK, XGB_OK, BORUTA_OK
