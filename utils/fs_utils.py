"""
Feature Selection Utility Functions
Extracted from 11_dec_fs.ipynb for clean, reusable implementation.

Date: December 12, 2024
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, mutual_info_regression, VarianceThreshold
from sklearn.inspection import permutation_importance
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.util import Surv
from scipy.stats import spearmanr, pearsonr
from collections import Counter
# Try to use notebook tqdm if available (better for Jupyter), otherwise use regular tqdm
try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm

# Configure tqdm for clean output - disable monitor to avoid nested bars
try:
    tqdm.tqdm.monitor_interval = 0
except:
    pass  # Some tqdm versions don't have this attribute
import warnings
warnings.filterwarnings('ignore')

# Optional imports with try-except blocks
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


# ============================================================
# CORE EVALUATION FUNCTION
# ============================================================

def evaluate_features_cv(X, y_df, selected_features, method_name="", n_splits=5, random_state=42):
    """
    Evaluate C-index using stratified cross-validation with Cox model.
    Uses FIXED penalizer=0.1 (not adaptive).
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y_df : pd.DataFrame
        Must contain 'RFS_time' and 'event' columns
    selected_features : list
        List of feature names to evaluate
    method_name : str
        Name of method (for logging)
    n_splits : int
        Number of CV folds
    random_state : int
        Random seed
        
    Returns:
    --------
    c_mean : float
        Mean C-index across folds
    c_std : float
        Standard deviation of C-index
    """
    # Extract survival data
    y_time = y_df['RFS_time'].values
    y_event = y_df['event'].values

    # Select features
    X_selected = X[selected_features]

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    c_indices = []

    for train_idx, test_idx in kf.split(X_selected, y_event):
        X_train_cv = X_selected.iloc[train_idx]
        X_test_cv = X_selected.iloc[test_idx]
        y_time_train = y_time[train_idx]
        y_time_test = y_time[test_idx]
        y_event_train = y_event[train_idx]
        y_event_test = y_event[test_idx]

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train_cv),
            columns=X_train_cv.columns
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test_cv),
            columns=X_test_cv.columns
        )

        # Prepare training data
        train_df = X_train_scaled.copy()
        train_df['T'] = y_time_train
        train_df['E'] = y_event_train

        try:
            # Fit Cox model with FIXED penalizer=0.1
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(train_df, duration_col='T', event_col='E', show_progress=False)

            # Predict risk scores
            risk_scores = cph.predict_partial_hazard(X_test_scaled)

            # Calculate C-index
            c_idx = concordance_index(y_time_test, -risk_scores.values.flatten(), y_event_test)
            c_indices.append(c_idx)

        except Exception as e:
            # Silently skip failed folds
            continue

    # Calculate statistics
    if len(c_indices) == 0:
        return np.nan, np.nan

    c_mean = np.mean(c_indices)
    c_std = np.std(c_indices)

    return c_mean, c_std


# ============================================================
# CATEGORY A: STATISTICAL/FILTER METHODS
# ============================================================

def univariate_cox_selection(X, y_df, p_threshold=0.05):
    """
    Select features based on univariate Cox regression p-values.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y_df : pd.DataFrame
        Must contain 'RFS_time' and 'event' columns
    p_threshold : float
        P-value threshold for selection
        
    Returns:
    --------
    selected : list
        List of selected feature names
    """
    y_time = y_df['RFS_time'].values
    y_event = y_df['event'].values

    p_values = []
    for col in tqdm(X.columns, desc="Univariate Cox", leave=False, dynamic_ncols=True):
        try:
            df = pd.DataFrame({'T': y_time, 'E': y_event, 'X': X[col]})
            cph = CoxPHFitter()
            cph.fit(df, duration_col='T', event_col='E', show_progress=False)
            p_val = cph.summary.loc['X', 'p']
            p_values.append((col, p_val))
        except:
            p_values.append((col, 1.0))

    selected = [col for col, p in p_values if p < p_threshold]
    return selected


def pearson_selection(X, y_df, r_threshold=0.1, p_threshold=0.05):
    """
    Select features correlated with survival time (ignores censoring).
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y_df : pd.DataFrame
        Must contain 'RFS_time' column
    r_threshold : float
        Minimum absolute correlation coefficient
    p_threshold : float
        Maximum p-value
        
    Returns:
    --------
    selected : list
        List of selected feature names
    """
    y_time = y_df['RFS_time'].values

    correlations = []
    for col in X.columns:
        r, p = pearsonr(X[col], y_time)
        if abs(r) > r_threshold and p < p_threshold:
            correlations.append(col)

    return correlations


def spearman_selection(X, y_df, r_threshold=0.1, p_threshold=0.05):
    """
    Select features with Spearman correlation to survival time.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y_df : pd.DataFrame
        Must contain 'RFS_time' column
    r_threshold : float
        Minimum absolute correlation coefficient
    p_threshold : float
        Maximum p-value
        
    Returns:
    --------
    selected : list
        List of selected feature names
    """
    y_time = y_df['RFS_time'].values

    correlations = []
    for col in X.columns:
        r, p = spearmanr(X[col], y_time)
        if abs(r) > r_threshold and p < p_threshold:
            correlations.append(col)

    return correlations


def mutual_info_selection(X, y_df, k_features=300):
    """
    Select features using mutual information.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y_df : pd.DataFrame
        Must contain 'RFS_time' column
    k_features : int
        Number of top features to select
        
    Returns:
    --------
    selected : list
        List of selected feature names
    """
    y_time = y_df['RFS_time'].values

    mi_scores = mutual_info_regression(X, y_time, random_state=42)
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
    selected = mi_series.head(k_features).index.tolist()

    return selected


def relieff_selection(X, y_df, n_features=100, n_neighbors=100):
    """
    ReliefF feature selection.
    Popular in radiomics, captures feature interactions.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y_df : pd.DataFrame
        Must contain 'RFS_time' and 'event' columns
    n_features : int
        Number of features to select
    n_neighbors : int
        Number of neighbors for ReliefF
        
    Returns:
    --------
    selected : list
        List of selected feature names
    """
    if not RELIEFF_AVAILABLE:
        print("  SKIPPED: skrebate not installed")
        return []
    
    # ReliefF expects continuous target, use RFS time weighted by event
    y_time = y_df['RFS_time'].values
    y_event = y_df['event'].values

    # Weight: Positive time for events, negative for censored
    y_weighted = y_time * (2 * y_event - 1)

    relief = ReliefF(n_features_to_select=n_features, n_neighbors=n_neighbors)
    relief.fit(X.values, y_weighted)

    # Get feature scores
    feature_scores = pd.Series(relief.feature_importances_, index=X.columns)
    selected = feature_scores.nlargest(n_features).index.tolist()

    return selected


def anova_selection(X, y_df, k_features=500):
    """
    ANOVA F-test (treats as classification, ignores time).
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y_df : pd.DataFrame
        Must contain 'event' column
    k_features : int
        Number of top features to select
        
    Returns:
    --------
    selected : list
        List of selected feature names
    """
    y_event = y_df['event'].values

    f_scores, p_values = f_classif(X, y_event)
    f_series = pd.Series(f_scores, index=X.columns).sort_values(ascending=False)
    selected = f_series.head(k_features).index.tolist()

    return selected


# ============================================================
# CATEGORY B: REDUNDANCY REMOVAL
# ============================================================

def correlation_filter(X, threshold=0.90, method='pearson'):
    """
    Remove highly correlated features.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    threshold : float
        Correlation threshold (features with |r| > threshold are removed)
    method : str
        Correlation method ('pearson' or 'spearman')
        
    Returns:
    --------
    selected : list
        List of selected feature names
    """
    corr_matrix = X.corr(method=method).abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    selected = [col for col in X.columns if col not in to_drop]

    return selected


def correlation_filter_fixed(X, y_df, threshold=0.90, max_features=300):
    """
    Remove highly correlated features, then select top features by univariate Cox.
    FIXED VERSION: After correlation filtering, select top max_features by univariate Cox p-value.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y_df : pd.DataFrame
        Must contain 'RFS_time' and 'event' columns
    threshold : float
        Correlation threshold
    max_features : int
        Maximum number of features to keep after filtering
        
    Returns:
    --------
    selected : list
        List of selected feature names
    """
    # Step 1: Remove correlated features
    corr_matrix = X.corr(method='pearson').abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    X_decorr = X.drop(columns=to_drop)
    
    # Step 2: If still too many, select top features by univariate Cox p-value
    if X_decorr.shape[1] > max_features:
        y_time = y_df['RFS_time'].values
        y_event = y_df['event'].values
        
        p_values = []
        for col in tqdm(X_decorr.columns, desc="Corr Filter", leave=False, dynamic_ncols=True):
            try:
                df = pd.DataFrame({'T': y_time, 'E': y_event, 'X': X_decorr[col]})
                cph = CoxPHFitter()
                cph.fit(df, duration_col='T', event_col='E', show_progress=False)
                p_val = cph.summary.loc['X', 'p']
                p_values.append((col, p_val))
            except:
                p_values.append((col, 1.0))
        
        # Sort by p-value and select top max_features
        p_values_sorted = sorted(p_values, key=lambda x: x[1])
        selected = [col for col, _ in p_values_sorted[:max_features]]
    else:
        selected = X_decorr.columns.tolist()
    
    return selected


def variance_filter(X, threshold=0.01):
    """
    Remove low-variance features.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    threshold : float
        Variance threshold
        
    Returns:
    --------
    selected : list
        List of selected feature names
    """
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X)
    selected = X.columns[selector.get_support()].tolist()
    return selected


def mrmr_selection(X, y_df, n_features=50):
    """
    Minimum Redundancy Maximum Relevance.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y_df : pd.DataFrame
        Must contain 'event' column
    n_features : int
        Number of features to select
        
    Returns:
    --------
    selected : list
        List of selected feature names
    """
    if not MRMR_AVAILABLE:
        print("  SKIPPED: mrmr not installed")
        return []
    
    y_event = y_df['event'].values

    selected = mrmr_classif(X, y_event, K=n_features)
    return selected


# ============================================================
# CATEGORY C: REGULARIZATION METHODS
# ============================================================

def lasso_cox_selection(X, y_df, target_features=100, n_alphas=100):
    """
    LASSO-Cox with L1 regularization.
    FIXED: Uses coef_[:, i] instead of coef_path_[i].
    FIXED (19 Dec): Uses internal CV to select alpha (avoid data leakage).

    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y_df : pd.DataFrame
        Must contain 'RFS_time' and 'event' columns
    target_features : int
        Target number of features
    n_alphas : int
        Number of alpha values to test

    Returns:
    --------
    selected : list
        List of selected feature names
    """
    from sklearn.model_selection import train_test_split

    y_time = y_df['RFS_time'].values
    y_event = y_df['event'].values

    # Split into train/validation (80/20) for alpha selection
    X_train, X_val, y_train_time, y_val_time, y_train_event, y_val_event = train_test_split(
        X, y_time, y_event, test_size=0.2, random_state=42, stratify=y_event
    )

    y_train_surv = Surv.from_arrays(event=y_train_event, time=y_train_time)

    # Standardize features (fit on train, apply to both)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Fit LASSO on training set
    lasso = CoxnetSurvivalAnalysis(l1_ratio=1.0, alpha_min_ratio=0.01, n_alphas=n_alphas)
    lasso.fit(X_train_scaled, y_train_surv)

    # Select alpha using VALIDATION set C-index
    best_score, best_alpha_value = -np.inf, None
    for i, alpha in enumerate(lasso.alphas_):
        coef = lasso.coef_[:, i]
        n_selected = np.sum(coef != 0)
        if 0 < n_selected < len(X.columns):
            # Calculate C-index on VALIDATION set (not training)
            risk_val = X_val_scaled @ coef
            c_idx = concordance_index(y_val_time, -risk_val, y_val_event)
            if c_idx > best_score:
                best_score = c_idx
                best_alpha_value = alpha  # Store VALUE not index

    # Fallback: if no valid alpha found, use middle of grid
    if best_alpha_value is None:
        best_alpha_value = lasso.alphas_[len(lasso.alphas_) // 2]

    # Retrain on full training set with selected alpha VALUE
    y_surv_full = Surv.from_arrays(event=y_event, time=y_time)
    scaler_full = StandardScaler()
    X_scaled_full = scaler_full.fit_transform(X)

    # Use SPECIFIC alpha value, not auto-generated grid
    lasso_full = CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=[best_alpha_value])
    lasso_full.fit(X_scaled_full, y_surv_full)

    # Get features (only one alpha, so index=0)
    best_coef = lasso_full.coef_[:, 0]
    selected_features = X.columns[best_coef != 0].tolist()

    return selected_features


def elasticnet_cox_selection(X, y_df, target_features=100, l1_ratio=0.5, n_alphas=100, alpha_min_ratio=0.01):
    """
    Elastic Net Cox (L1 + L2 regularization).
    FIXED: Uses coef_[:, i] instead of coef_path_[i].
    FIXED (19 Dec): Uses internal CV to select alpha (avoid data leakage).

    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y_df : pd.DataFrame
        Must contain 'RFS_time' and 'event' columns
    target_features : int
        Target number of features
    l1_ratio : float
        L1 ratio (0.0 = Ridge, 1.0 = LASSO)
    n_alphas : int
        Number of alpha values to test

    Returns:
    --------
    selected : list
        List of selected feature names
    """
    from sklearn.model_selection import train_test_split

    y_time = y_df['RFS_time'].values
    y_event = y_df['event'].values

    # Split into train/validation (80/20) for alpha selection
    X_train, X_val, y_train_time, y_val_time, y_train_event, y_val_event = train_test_split(
        X, y_time, y_event, test_size=0.2, random_state=42, stratify=y_event
    )

    y_train_surv = Surv.from_arrays(event=y_train_event, time=y_train_time)

    # Standardize features (fit on train, apply to both)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Fit Elastic Net on training set
    enet = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alpha_min_ratio=alpha_min_ratio, n_alphas=n_alphas)
    enet.fit(X_train_scaled, y_train_surv)

    # Select alpha using VALIDATION set C-index
    best_score, best_alpha_value = -np.inf, None
    for i in range(len(enet.alphas_)):
        coef = enet.coef_[:, i]
        n_selected = np.sum(coef != 0)
        if 0 < n_selected < len(X.columns):
            # Calculate C-index on VALIDATION set (not training)
            risk_val = X_val_scaled @ coef
            c_idx = concordance_index(y_val_time, -risk_val, y_val_event)
            if c_idx > best_score:
                best_score = c_idx
                best_alpha_value = enet.alphas_[i]  # Store VALUE not index

    # Fallback: if no valid alpha found, use middle of grid
    if best_alpha_value is None:
        best_alpha_value = enet.alphas_[len(enet.alphas_) // 2]

    # Retrain on full training set with selected alpha VALUE
    y_surv_full = Surv.from_arrays(event=y_event, time=y_time)
    scaler_full = StandardScaler()
    X_scaled_full = scaler_full.fit_transform(X)

    # Use SPECIFIC alpha value, not auto-generated grid
    enet_full = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alphas=[best_alpha_value])
    enet_full.fit(X_scaled_full, y_surv_full)

    # Get features (only one alpha, so index=0)
    best_coef = enet_full.coef_[:, 0]
    selected_features = X.columns[best_coef != 0].tolist()

    return selected_features


def stability_selection_lasso(X, y_df, n_features=50, n_bootstrap=100, stability_threshold=0.6,
                               selection_strategy='threshold', random_state=42):
    """
    Stability Selection with LASSO.
    FIXED: Uses coef_ not coef_path_, updates selected_mask after fallback.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y_df : pd.DataFrame
        Must contain 'RFS_time' and 'event' columns
    n_features : int
        Target number of features
    n_bootstrap : int
        Number of bootstrap samples
    stability_threshold : float
        Stability threshold (fraction of bootstraps)
    selection_strategy : str
        'threshold' (select all above threshold) or 'top_k' (select exactly n_features)
        
    Returns:
    --------
    selected : list
        List of selected feature names
    """
    from sklearn.utils import resample

    y_time = y_df['RFS_time'].values
    y_event = y_df['event'].values

    n_samples = X.shape[0]
    selection_counts = np.zeros(X.shape[1])

    for i in tqdm(range(n_bootstrap), desc="Stability Selection", leave=False, dynamic_ncols=True):
        # Progress handled by tqdm, no need for manual print

        # Bootstrap sample (50% of data for stability)
        sample_size = n_samples // 2
        indices = resample(range(n_samples), n_samples=sample_size, random_state=random_state+i)

        X_boot = X.iloc[indices]
        y_time_boot = y_time[indices]
        y_event_boot = y_event[indices]
        y_surv_boot = Surv.from_arrays(event=y_event_boot, time=y_time_boot)

        try:
            # Standardize
            scaler = StandardScaler()
            X_boot_scaled = scaler.fit_transform(X_boot)

            # Fit LASSO
            lasso = CoxnetSurvivalAnalysis(l1_ratio=1.0, alpha_min_ratio=0.01, n_alphas=50)
            lasso.fit(X_boot_scaled, y_surv_boot)

            # Find alpha that gives ~n_features
            # coef_ is shape (n_features, n_alphas)
            best_idx = 0
            for j in range(len(lasso.alphas_)):
                n_selected = np.sum(lasso.coef_[:, j] != 0)  # FIXED: use coef_ not coef_path_
                if n_selected <= n_features:
                    best_idx = j
                    break

            # Get selected features
            coef = lasso.coef_[:, best_idx]
            selected_mask = (coef != 0)
            selection_counts += selected_mask.astype(int)
        except:
            continue

    # Calculate selection frequency
    selection_freq = selection_counts / n_bootstrap

    # Selection strategy
    if selection_strategy == 'threshold':
        # Strategy 1: Select all features above threshold
        selected_mask = selection_freq >= stability_threshold
        selected_features = X.columns[selected_mask].tolist()

        if len(selected_features) == 0:
            print(f"  WARNING: No features above threshold {stability_threshold}")
            print(f"  Falling back to top {n_features} features")
            top_indices = np.argsort(selection_freq)[-n_features:]
            selected_features = X.columns[top_indices].tolist()
            # FIXED: Update selected_mask after fallback to prevent ValueError
            selected_mask = np.isin(np.arange(len(X.columns)), top_indices)

    elif selection_strategy == 'top_k':
        # Strategy 2: Select exactly top K by stability score
        top_indices = np.argsort(selection_freq)[-n_features:]
        selected_features = X.columns[top_indices].tolist()
    else:
        raise ValueError("selection_strategy must be 'threshold' or 'top_k'")

    return selected_features


# ============================================================
# CATEGORY D: ML-BASED IMPORTANCE RANKING
# ============================================================

def rsf_permutation_importance(X, y_df, n_features=60, n_estimators=100, n_repeats=5, random_state=42):
    """
    Random Survival Forest with permutation importance.
    FIXED: Uses permutation importance instead of feature_importances_.
    FIXED (19 Dec): Calculates importance on validation set (avoid overfitting).

    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y_df : pd.DataFrame
        Must contain 'RFS_time' and 'event' columns
    n_features : int
        Number of features to select
    n_estimators : int
        Number of trees
    n_repeats : int
        Number of permutation repeats
    random_state : int
        Random state for reproducibility

    Returns:
    --------
    selected : list
        List of selected feature names
    """
    from sklearn.model_selection import train_test_split

    y_time = y_df['RFS_time'].values
    y_event = y_df['event'].values

    # Split into train/validation (80/20) for importance calculation
    X_train, X_val, y_train_time, y_val_time, y_train_event, y_val_event = train_test_split(
        X, y_time, y_event, test_size=0.2, random_state=random_state, stratify=y_event
    )

    y_train_surv = Surv.from_arrays(event=y_train_event, time=y_train_time)
    y_val_surv = Surv.from_arrays(event=y_val_event, time=y_val_time)

    # Standardize features (fit on train, apply to both)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train RSF on training set
    rsf = RandomSurvivalForest(
        n_estimators=n_estimators,
        max_depth=5,
        min_samples_split=15,
        min_samples_leaf=10,
        max_features='sqrt',
        n_jobs=-1,
        random_state=random_state,
        verbose=0
    )
    rsf.fit(X_train_scaled, y_train_surv)

    # Calculate permutation importance on VALIDATION set (not training)
    perm_importance = permutation_importance(
        rsf, X_val_scaled, y_val_surv,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )

    # Use mean importance
    importances = perm_importance.importances_mean

    # Select top features
    importance_series = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    selected = importance_series.head(n_features).index.tolist()

    return selected


def xgboost_survival_selection(X, y_df, n_features=60, n_estimators=500, random_state=42):
    """
    XGBoost-style survival model with feature importance.
    FIXED (19 Dec): Uses validation set for importance calculation (avoid overfitting).

    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y_df : pd.DataFrame
        Must contain 'RFS_time' and 'event' columns
    n_features : int
        Number of features to select
    n_estimators : int
        Number of boosting rounds
    random_state : int
        Random state for reproducibility

    Returns:
    --------
    selected : list
        List of selected feature names
    """
    from sklearn.model_selection import train_test_split

    y_time = y_df['RFS_time'].values
    y_event = y_df['event'].values

    # Split into train/validation (80/20)
    X_train, X_val, y_train_time, y_val_time, y_train_event, y_val_event = train_test_split(
        X, y_time, y_event, test_size=0.2, random_state=random_state, stratify=y_event
    )

    y_train_surv = Surv.from_arrays(event=y_train_event, time=y_train_time)
    y_val_surv = Surv.from_arrays(event=y_val_event, time=y_val_time)

    # Train on training set
    gbsa = GradientBoostingSurvivalAnalysis(
        n_estimators=n_estimators,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        random_state=random_state,
        verbose=0
    )
    gbsa.fit(X_train, y_train_surv)

    # Calculate permutation importance on VALIDATION set
    perm_importance = permutation_importance(
        gbsa, X_val, y_val_surv,
        n_repeats=5,
        random_state=random_state,
        n_jobs=-1
    )

    # Use mean importance
    importances = perm_importance.importances_mean
    importance_series = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    selected = importance_series.head(n_features).index.tolist()

    return selected


def permutation_importance_survival(X, y_df, n_features=60, n_estimators=500, n_repeats=10, random_state=42):
    """
    Permutation-based feature importance for survival models.
    FIXED: Uses permutation importance for GradientBoostingSurvivalAnalysis.
    FIXED (19 Dec): Calculates importance on validation set (avoid overfitting).

    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y_df : pd.DataFrame
        Must contain 'RFS_time' and 'event' columns
    n_features : int
        Number of features to select
    n_estimators : int
        Number of boosting rounds
    n_repeats : int
        Number of permutation repeats
    random_state : int
        Random state for reproducibility

    Returns:
    --------
    selected : list
        List of selected feature names
    """
    from sklearn.model_selection import train_test_split

    y_time = y_df['RFS_time'].values
    y_event = y_df['event'].values

    # Split into train/validation (80/20)
    X_train, X_val, y_train_time, y_val_time, y_train_event, y_val_event = train_test_split(
        X, y_time, y_event, test_size=0.2, random_state=random_state, stratify=y_event
    )

    y_train_surv = Surv.from_arrays(event=y_train_event, time=y_train_time)
    y_val_surv = Surv.from_arrays(event=y_val_event, time=y_val_time)

    # Train on training set
    gbsa = GradientBoostingSurvivalAnalysis(
        n_estimators=n_estimators,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        random_state=random_state,
        verbose=0
    )
    gbsa.fit(X_train, y_train_surv)

    # Calculate permutation importance on VALIDATION set (not training)
    perm_importance = permutation_importance(
        gbsa, X_val, y_val_surv,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )

    # Use mean importance
    importances = perm_importance.importances_mean

    # Select top features
    importance_series = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    selected = importance_series.head(n_features).index.tolist()

    return selected


# ============================================================
# CATEGORY F: ENSEMBLE METHODS
# ============================================================

def voting_ensemble_selection(feature_selections, min_votes=3):
    """
    Select features that appear in at least min_votes different methods.
    
    Parameters:
    -----------
    feature_selections : dict
        Dictionary of {method_name: [selected_features]}
    min_votes : int
        Minimum number of methods that must select a feature
        
    Returns:
    --------
    selected : list
        List of selected feature names
    """
    # Count votes for each feature
    feature_votes = Counter()
    for method, features in feature_selections.items():
        for feat in features:
            feature_votes[feat] += 1

    # Select features with enough votes
    selected = [feat for feat, votes in feature_votes.items() if votes >= min_votes]

    return selected
