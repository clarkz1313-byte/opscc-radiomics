# -*- coding: utf-8 -*-
"""
PT Task 2 Stage 4 Optuna: 9 pipelines, CV-only, full-chain tuning

Per Mar_2026_task2/14_mar_t2_optuna_plan.md:
- 9 pipelines (PT_S3 ranks: 1, 3, 4, 5, 6, 7, 10, 13, 15)
- Full-chain tuning across S1/S2/S3 selectors
- Objective: WEIGHT_CV * AUC_CV - WEIGHT_FEA * Fea_penalty (no Test term)
- No ReliefF precompute needed (no ReliefF S1 in PT shortlist)
- HEAVY_PIPELINE_N_JOBS for ranks with GB PermImp as S2 (ranks 1, 15)
- Diagnostic Test4 exported, not used in objective

Run:
    cd "D:/Uppsala thesis" && python Mar_2026_task2/14_mar_task2_stage4_PT.py
"""

from __future__ import annotations

import argparse
import joblib
import logging
import os
import random
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import optuna
import pandas as pd

try:
    import psycopg2

    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from Mar_2026_task2.fs_task2_utils import (
    MRMR_OK,
    XGB_OK,
    correlation_filter,
    elasticnet_logistic_selection,
    evaluate_auc_test,
    lasso_logistic_selection,
    nested_cv_auc,
)

if MRMR_OK:
    from mrmr import mrmr_classif
if XGB_OK:
    from xgboost import XGBClassifier

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.inspection import permutation_importance

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
OUTPUT_DIR = SCRIPT_DIR / "14_mar_t2_optuna_outputs"
LOG_DIR = SCRIPT_DIR / "14_mar_t2_optuna_outputs"
CHECKPOINT_DIR = SCRIPT_DIR / "14_mar_t2_optuna_outputs"

TRAIN_FILE = SCRIPT_DIR / "12_mar_task2_rad_data" / "13_mar_task2_PT_primary_train.csv"
TEST_FILE = SCRIPT_DIR / "12_mar_task2_rad_data" / "13_mar_task2_PT_primary_test.csv"
EXT_FILE = SCRIPT_DIR / "12_mar_task2_rad_data" / "12_mar_task2_PT_primary_ext.csv"
STAGE3_CSV = SCRIPT_DIR / "13_mar_t2_fs_results" / "14_mar_task2_stage3_PT_result.csv"

logger = None

# ---------------------------------------------------------------------
# PostgreSQL
# ---------------------------------------------------------------------
DB_HOST = os.environ.get("OPTUNA_DB_HOST", "localhost")
DB_PORT = int(os.environ.get("OPTUNA_DB_PORT", "5432"))
DB_USER = os.environ.get("OPTUNA_DB_USER", "postgres")
DB_PASSWORD = os.environ.get("OPTUNA_DB_PASSWORD", "userdefined")
DB_NAME = os.environ.get("OPTUNA_DB_NAME", "optuna_db")
DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
STORAGE_TIMEOUT = 600

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
SEED = 42
N_FOLDS = 5
CV_SEED = 42
FEA_MAX = 50
WEIGHT_CV = 0.8
WEIGHT_FEA = 0.2

N_TRIALS_DEFAULT = 700   # rank 10 (parsimony, ANOVA->EN->XGB, fast)
N_TRIALS_HEAVY = 500     # ranks 1, 15 (GB PermImp as S2)
N_TRIALS_PARSIMON = 600  # ranks 3,4,5,6,7,13

N_JOBS = 8
HEAVY_PIPELINE_N_JOBS = 1

# PT_S3_14 excluded (too slow for overnight nested CV: ~57s/trial)
PIPELINES = {
    1: {"group": "performance"},   # HEAVY: mRMR->GB->XGB
    3: {"group": "performance"},   # MWU->EN->XGB
    4: {"group": "performance"},   # MWU->XGB->GB
    5: {"group": "performance"},   # mRMR->EN->XGB
    6: {"group": "performance"},   # ANOVA->XGB->RF
    7: {"group": "parsimony"},     # MWU->XGB->LASSO
    10: {"group": "parsimony"},    # ANOVA->EN->XGB (fast ~9s)
    13: {"group": "parsimony"},    # MWU->EN->GB
    15: {"group": "parsimony"},    # HEAVY: ANOVA->GB->XGB
}
HEAVY_RANKS = frozenset({1, 15})

# ---------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------
X_train: pd.DataFrame
y_train: np.ndarray
X_test: pd.DataFrame
y_test: np.ndarray
X_ext: pd.DataFrame
y_ext: np.ndarray
STAGE3_METHODS: dict[int, dict] = {}


def _parse_pipeline(pipeline_str: str) -> tuple[str, str, str]:
    parts = [p.strip() for p in str(pipeline_str).split("->")]
    if len(parts) != 3:
        return "", "", ""
    return parts[0], parts[1], parts[2]


def _load_stage3_methods() -> None:
    global STAGE3_METHODS
    df = pd.read_csv(STAGE3_CSV)
    for _, r in df.iterrows():
        rank_str = str(r.get("Rank", ""))
        if not rank_str.startswith("PT_S3_"):
            continue
        try:
            rank = int(rank_str.replace("PT_S3_", ""))
        except ValueError:
            continue
        s1, s2, s3 = _parse_pipeline(str(r.get("Pipeline", "")))
        STAGE3_METHODS[rank] = {
            "S1": s1,
            "S2": s2,
            "S3": s3,
            "Pipeline": str(r.get("Pipeline", "")),
            "AUC3": float(r.get("AUC3", 0)),
            "Fea3": int(r.get("Fea3", 0)),
            "Test3": float(r.get("Test3", 0)),
        }
    logger.info("Loaded %s Stage 3 pipelines", len(STAGE3_METHODS))


def _load_data() -> None:
    global X_train, y_train, X_test, y_test, X_ext, y_ext
    exclude = [
        "PatientID",
        "HPV_binary",
        "Relapse",
        "RFS",
        "Age",
        "Gender_Male",
        "Treatment_CRT",
        "prefix",
    ]

    df_tr = pd.read_csv(TRAIN_FILE)
    feat_cols = [c for c in df_tr.columns if c not in exclude]
    X_train = df_tr[feat_cols].replace([np.inf, -np.inf], np.nan)
    X_train = X_train.fillna(X_train.median())
    y_train = df_tr["HPV_binary"].values.astype(int)

    df_te = pd.read_csv(TEST_FILE)
    X_test = df_te[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
    y_test = df_te["HPV_binary"].values.astype(int)

    df_ext = pd.read_csv(EXT_FILE)
    ext_feat_cols = [c for c in feat_cols if c in df_ext.columns]
    X_ext = df_ext[ext_feat_cols].replace([np.inf, -np.inf], np.nan).fillna(X_train[ext_feat_cols].median())
    y_ext = df_ext["HPV_binary"].values.astype(int)

    logger.info(
        "Train: %s patients, %s features, %s HPV+",
        len(df_tr),
        len(feat_cols),
        int(y_train.sum()),
    )
    logger.info("Test: %s patients, %s HPV+", len(df_te), int(y_test.sum()))
    logger.info("Ext:  %s patients, %s HPV+", len(df_ext), int(y_ext.sum()))


def _build_selector_fn(params: dict, rank: int):
    """Return a fold-local selector_fn(X_fold_df, y_fold) -> list[str].

    Uses the hyperparameters already suggested by the trial (fixed values from
    `params`). Re-runs the full S1->S2->S3 chain on each fold's train split so
    that feature selection never sees the held-out fold — eliminating selection
    bias. Called once per trial inside _common_objective.
    """
    s1_tok = STAGE3_METHODS.get(rank, {}).get("S1", "")
    s2_tok = STAGE3_METHODS.get(rank, {}).get("S2", "")
    s3_tok = STAGE3_METHODS.get(rank, {}).get("S3", "")

    def selector_fn(X_fold_df: pd.DataFrame, y_fold: np.ndarray) -> list[str]:
        # --- S1 ---
        if "ANOVA" in s1_tok:
            k = int(params.get("anova_k", 100))
            s1 = _selector_anova(X_fold_df, y_fold, k)
        elif "MWU" in s1_tok:
            k = int(params.get("mwu_k", 100))
            s1 = _selector_mwu(X_fold_df, y_fold, k)
        elif "Corr" in s1_tok:
            thr = float(params.get("corr_threshold", 0.90))
            s1 = _selector_corr(X_fold_df, threshold=thr)
        elif "mRMR" in s1_tok:
            k = int(params.get("mrmr_k", 30))
            s1 = _selector_mrmr(X_fold_df, y_fold, k)
        elif "LASSO" in s1_tok:
            t = int(params.get("lasso_target", 30))
            s1 = _selector_lasso(X_fold_df, y_fold, target=t)
        elif "ElasticNet" in s1_tok:
            t = int(params.get("en_target", 30))
            l1 = float(params.get("en_l1_ratio", 0.5))
            s1 = _selector_en(X_fold_df, y_fold, target=t, l1_ratio=l1)
        else:
            s1 = list(X_fold_df.columns)
        if not s1:
            return []

        # --- S2 ---
        X_s1 = X_fold_df[s1]
        if not s2_tok:
            s2 = s1
        elif "ANOVA" in s2_tok:
            k = int(params.get("s2_anova_k", 50))
            s2 = _selector_anova(X_s1, y_fold, k)
        elif "Corr" in s2_tok:
            thr = float(params.get("s2_corr_threshold", 0.90))
            s2 = _selector_corr(X_s1, threshold=thr)
        elif "mRMR" in s2_tok:
            k = int(params.get("s2_mrmr_k", 30))
            s2 = _selector_mrmr(X_s1, y_fold, k)
        elif "LASSO" in s2_tok:
            t = int(params.get("s2_lasso_target", 20))
            s2 = _selector_lasso(X_s1, y_fold, target=t)
        elif "ElasticNet" in s2_tok:
            t = int(params.get("s2_en_target", 20))
            l1 = float(params.get("s2_en_l1_ratio", 0.5))
            s2 = _selector_en(X_s1, y_fold, target=t, l1_ratio=l1)
        elif "XGBoost" in s2_tok or "D2" in s2_tok:
            k = int(params.get("s2_xgb_k", 20))
            ne = int(params.get("s2_xgb_n_estimators", 100))
            s2 = _selector_xgb_perm(X_s1, y_fold, k, n_estimators=ne)
        elif "RF PermImp" in s2_tok or "D1" in s2_tok:
            k = int(params.get("s2_rf_k", 20))
            ne = int(params.get("s2_rf_n_estimators", 100))
            s2 = _selector_rf_perm(X_s1, y_fold, k, n_estimators=ne)
        elif "GB PermImp" in s2_tok or "D3" in s2_tok:
            k = int(params.get("s2_gb_k", 20))
            ne = int(params.get("s2_gb_n_estimators", 100))
            s2 = _selector_gb_perm(X_s1, y_fold, k, n_estimators=ne)
        else:
            s2 = s1
        if not s2:
            return []

        # --- S3 ---
        X_s2 = X_fold_df[s2]
        if not s3_tok:
            return s2
        elif "LASSO" in s3_tok:
            t = int(params.get("s3_lasso_target", 10))
            return _selector_lasso(X_s2, y_fold, target=t)
        elif "ElasticNet" in s3_tok:
            t = int(params.get("s3_en_target", 10))
            l1 = float(params.get("s3_en_l1_ratio", 0.5))
            return _selector_en(X_s2, y_fold, target=t, l1_ratio=l1)
        elif "XGBoost" in s3_tok or "D2" in s3_tok:
            k = int(params.get("s3_xgb_k", 10))
            ne = int(params.get("s3_xgb_n_estimators", 100))
            return _selector_xgb_perm(X_s2, y_fold, k, n_estimators=ne)
        elif "RF PermImp" in s3_tok or "D1" in s3_tok:
            k = int(params.get("s3_rf_k", 10))
            ne = int(params.get("s3_rf_n_estimators", 100))
            return _selector_rf_perm(X_s2, y_fold, k, n_estimators=ne)
        elif "GB PermImp" in s3_tok or "D3" in s3_tok:
            k = int(params.get("s3_gb_k", 10))
            ne = int(params.get("s3_gb_n_estimators", 100))
            return _selector_gb_perm(X_s2, y_fold, k, n_estimators=ne)
        return s2

    return selector_fn


def _common_objective(trial, s3_features: list[str], rank: int) -> float:
    n_features = len(s3_features)
    if n_features == 0 or n_features > FEA_MAX:
        return 0.0
    selector_fn = _build_selector_fn(trial.params, rank)
    auc_cv, _ = nested_cv_auc(X_train, y_train, selector_fn, random_state=CV_SEED)
    if np.isnan(auc_cv):
        auc_cv = 0.0
    fea_penalty = min(n_features / FEA_MAX, 1.0)
    objective = WEIGHT_CV * auc_cv - WEIGHT_FEA * fea_penalty
    trial.set_user_attr("auc_cv", round(auc_cv, 4))
    trial.set_user_attr("n_features", n_features)
    trial.set_user_attr("fea_penalty", round(fea_penalty, 4))
    trial.set_user_attr("selected_features", s3_features)
    logger.info(
        "PT_S3_%s Trial %s: nested_AUC=%.4f Fea=%s Obj=%.4f",
        rank,
        trial.number,
        auc_cv,
        n_features,
        objective,
    )
    return objective


def _selector_anova(X_df: pd.DataFrame, y_arr: np.ndarray, k: int) -> list[str]:
    scores, _ = f_classif(X_df.values, y_arr)
    scores = np.nan_to_num(scores, nan=0.0)
    idx = np.argsort(scores)[::-1][: min(k, X_df.shape[1])]
    return [X_df.columns[i] for i in idx]


def _selector_mwu(X_df: pd.DataFrame, y_arr: np.ndarray, k: int) -> list[str]:
    """Mann-Whitney U AUC filter: keep top-k features by AUC score."""
    from scipy.stats import mannwhitneyu

    y = y_arr
    pos = X_df.values[y == 1]
    neg = X_df.values[y == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return list(X_df.columns[:k])
    aucs = []
    for j in range(X_df.shape[1]):
        try:
            stat, _ = mannwhitneyu(pos[:, j], neg[:, j], alternative="two-sided")
            auc_j = stat / (n_pos * n_neg)
            aucs.append(abs(auc_j - 0.5))
        except Exception:
            aucs.append(0.0)
    aucs = np.array(aucs)
    idx = np.argsort(aucs)[::-1][: min(k, X_df.shape[1])]
    return [X_df.columns[i] for i in idx]


def _selector_mrmr(X_df: pd.DataFrame, y_arr: np.ndarray, k: int) -> list[str]:
    if not MRMR_OK:
        raise RuntimeError("mrmr package not installed; PT_S3_1/5 requires mRMR. pip install mrmr-selection")
    return list(mrmr_classif(X=X_df, y=pd.Series(y_arr), K=min(k, X_df.shape[1])))


def _selector_corr(X_df: pd.DataFrame, threshold: float) -> list[str]:
    return correlation_filter(X_df, threshold=threshold)


def _selector_lasso(X_df: pd.DataFrame, y_arr: np.ndarray, target: int) -> list[str]:
    return lasso_logistic_selection(
        X_df,
        y_arr,
        target_features=min(target, X_df.shape[1]),
        random_state=CV_SEED,
    )


def _selector_en(X_df: pd.DataFrame, y_arr: np.ndarray, target: int, l1_ratio: float) -> list[str]:
    return elasticnet_logistic_selection(
        X_df,
        y_arr,
        target_features=min(target, X_df.shape[1]),
        l1_ratio=l1_ratio,
        random_state=CV_SEED,
    )


def _selector_rf_perm(
    X_df: pd.DataFrame,
    y_arr: np.ndarray,
    k: int,
    n_estimators: int = 100,
) -> list[str]:
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=CV_SEED, n_jobs=1)
    rf.fit(X_df.values, y_arr)
    r = permutation_importance(
        rf,
        X_df.values,
        y_arr,
        n_repeats=5,
        random_state=CV_SEED,
        n_jobs=1,
    )
    idx = np.argsort(r.importances_mean)[::-1][: min(k, X_df.shape[1])]
    return [X_df.columns[i] for i in idx]


def _selector_xgb_perm(
    X_df: pd.DataFrame,
    y_arr: np.ndarray,
    k: int,
    n_estimators: int = 100,
) -> list[str]:
    if not XGB_OK:
        raise RuntimeError("xgboost not installed; required for XGBoost PermImp selector. pip install xgboost")
    clf = XGBClassifier(
        n_estimators=n_estimators,
        random_state=CV_SEED,
        eval_metric="logloss",
        verbosity=0,
        n_jobs=1,
    )
    clf.fit(X_df.values, y_arr)
    r = permutation_importance(
        clf,
        X_df.values,
        y_arr,
        n_repeats=5,
        random_state=CV_SEED,
        n_jobs=1,
    )
    idx = np.argsort(r.importances_mean)[::-1][: min(k, X_df.shape[1])]
    return [X_df.columns[i] for i in idx]


def _selector_gb_perm(
    X_df: pd.DataFrame,
    y_arr: np.ndarray,
    k: int,
    n_estimators: int = 100,
) -> list[str]:
    clf = GradientBoostingClassifier(n_estimators=n_estimators, random_state=CV_SEED)
    clf.fit(X_df.values, y_arr)
    r = permutation_importance(
        clf,
        X_df.values,
        y_arr,
        n_repeats=5,
        random_state=CV_SEED,
        n_jobs=1,
    )
    idx = np.argsort(r.importances_mean)[::-1][: min(k, X_df.shape[1])]
    return [X_df.columns[i] for i in idx]


def _s1_sampler(trial, rank: int) -> list[str]:
    s1_tok = STAGE3_METHODS.get(rank, {}).get("S1", "")

    if "ANOVA" in s1_tok:
        k = trial.suggest_int("anova_k", 50, 300)
        return _selector_anova(X_train, y_train, k)

    if "MWU" in s1_tok:
        k = trial.suggest_int("mwu_k", 50, 700)
        return _selector_mwu(X_train, y_train, k)

    if "Corr" in s1_tok:
        thr = trial.suggest_float("corr_threshold", 0.80, 0.95)
        return _selector_corr(X_train, threshold=thr)

    if "mRMR" in s1_tok:
        k = trial.suggest_int("mrmr_k", 20, 80)
        return _selector_mrmr(X_train, y_train, k)

    if "LASSO" in s1_tok:
        t = trial.suggest_int("lasso_target", 10, 60)
        return _selector_lasso(X_train, y_train, target=t)

    if "ElasticNet" in s1_tok:
        t = trial.suggest_int("en_target", 10, 60)
        l1 = trial.suggest_float("en_l1_ratio", 0.1, 0.9)
        return _selector_en(X_train, y_train, target=t, l1_ratio=l1)

    return list(X_train.columns)


def _s2_sampler(trial, s1: list[str], rank: int) -> list[str]:
    s2_tok = STAGE3_METHODS.get(rank, {}).get("S2", "")
    X_s1 = X_train[s1]
    if not s2_tok:
        return s1

    if "ANOVA" in s2_tok:
        k = trial.suggest_int("s2_anova_k", 20, 100)
        return _selector_anova(X_s1, y_train, k)

    if "Corr" in s2_tok:
        thr = trial.suggest_float("s2_corr_threshold", 0.80, 0.95)
        return _selector_corr(X_s1, threshold=thr)

    if "mRMR" in s2_tok:
        k = trial.suggest_int("s2_mrmr_k", 10, 60)
        return _selector_mrmr(X_s1, y_train, k)

    if "LASSO" in s2_tok:
        t = trial.suggest_int("s2_lasso_target", 10, 60)
        return _selector_lasso(X_s1, y_train, target=t)

    if "ElasticNet" in s2_tok:
        t = trial.suggest_int("s2_en_target", 10, 60)
        l1 = trial.suggest_float("s2_en_l1_ratio", 0.1, 0.9)
        return _selector_en(X_s1, y_train, target=t, l1_ratio=l1)

    if "XGBoost" in s2_tok or "D2" in s2_tok:
        k = trial.suggest_int("s2_xgb_k", 10, 60)
        ne = trial.suggest_int("s2_xgb_n_estimators", 50, 200)
        return _selector_xgb_perm(X_s1, y_train, k, n_estimators=ne)

    if "RF PermImp" in s2_tok or "D1" in s2_tok:
        k = trial.suggest_int("s2_rf_k", 10, 60)
        ne = trial.suggest_int("s2_rf_n_estimators", 50, 200)
        return _selector_rf_perm(X_s1, y_train, k, n_estimators=ne)

    if "GB PermImp" in s2_tok or "D3" in s2_tok:
        k = trial.suggest_int("s2_gb_k", 10, 60)
        ne = trial.suggest_int("s2_gb_n_estimators", 50, 200)
        return _selector_gb_perm(X_s1, y_train, k, n_estimators=ne)

    return s1


def _s3_sampler(trial, s2: list[str], rank: int) -> list[str]:
    s3_tok = STAGE3_METHODS.get(rank, {}).get("S3", "")
    X_s2 = X_train[s2]
    if not s3_tok:
        return s2

    if "LASSO" in s3_tok:
        t = trial.suggest_int("s3_lasso_target", 5, 40)
        return _selector_lasso(X_s2, y_train, target=t)

    if "ElasticNet" in s3_tok:
        t = trial.suggest_int("s3_en_target", 5, 40)
        l1 = trial.suggest_float("s3_en_l1_ratio", 0.1, 0.9)
        return _selector_en(X_s2, y_train, target=t, l1_ratio=l1)

    if "XGBoost" in s3_tok or "D2" in s3_tok:
        k = trial.suggest_int("s3_xgb_k", 5, 40)
        ne = trial.suggest_int("s3_xgb_n_estimators", 50, 200)
        return _selector_xgb_perm(X_s2, y_train, k, n_estimators=ne)

    if "RF PermImp" in s3_tok or "D1" in s3_tok:
        k = trial.suggest_int("s3_rf_k", 5, 40)
        ne = trial.suggest_int("s3_rf_n_estimators", 50, 200)
        return _selector_rf_perm(X_s2, y_train, k, n_estimators=ne)

    if "GB PermImp" in s3_tok or "D3" in s3_tok:
        k = trial.suggest_int("s3_gb_k", 5, 40)
        ne = trial.suggest_int("s3_gb_n_estimators", 50, 200)
        return _selector_gb_perm(X_s2, y_train, k, n_estimators=ne)

    return s2


def _objective(trial, rank: int) -> float:
    try:
        s1 = _s1_sampler(trial, rank)
        if not s1:
            return 0.0
        s2 = _s2_sampler(trial, s1, rank)
        if not s2:
            return 0.0
        s3 = _s3_sampler(trial, s2, rank)
        if not s3:
            return 0.0
        return _common_objective(trial, s3, rank)
    except Exception as e:
        logger.error("PT_S3_%s Trial %s failed: %s", rank, trial.number, e)
        return 0.0


def _n_trials(rank: int) -> int:
    if rank == 10:
        return N_TRIALS_DEFAULT   # fast pipeline (~9s/trial), give extra budget
    if rank in HEAVY_RANKS:
        return N_TRIALS_HEAVY     # GB PermImp S2, single-threaded
    return N_TRIALS_PARSIMON      # all other pipelines


def run_optimization(rank: int, n_trials: int) -> optuna.Study:
    study_name = f"PT_S3_{rank}_Task2_Stage4"
    n_jobs = HEAVY_PIPELINE_N_JOBS if rank in HEAVY_RANKS else N_JOBS

    logger.info("\n%s", "=" * 70)
    logger.info("Starting: %s", study_name)
    if rank in STAGE3_METHODS:
        b = STAGE3_METHODS[rank]
        logger.info("Stage 3 baseline: AUC3=%.4f Fea3=%s", b["AUC3"], b["Fea3"])
        logger.info("Pipeline: %s", b["Pipeline"])
    logger.info(
        "Trials: %s n_jobs=%s | Objective: %.2f*AUC - %.2f*Fea",
        n_trials,
        n_jobs,
        WEIGHT_CV,
        WEIGHT_FEA,
    )
    logger.info("%s\n", "=" * 70)

    storage = optuna.storages.RDBStorage(
        url=DB_URL,
        engine_kwargs={"connect_args": {"connect_timeout": STORAGE_TIMEOUT}},
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )

    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if completed >= n_trials:
        logger.info("Study already has %s completed trials. Skipping.", completed)
        return study

    remaining = n_trials - completed
    logger.info("Resuming from %s completed, running %s more trials", completed, remaining)

    try:
        study.optimize(
            lambda t, r=rank: _objective(t, r),
            n_trials=remaining,
            n_jobs=n_jobs,
            show_progress_bar=True,
            timeout=None,
        )
    except KeyboardInterrupt:
        logger.warning("Optimization interrupted. Results saved to PostgreSQL.")

    CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
    cp_path = CHECKPOINT_DIR / f"{study_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    joblib.dump(study, cp_path)
    logger.info("Checkpoint saved: %s", cp_path)
    return study


def _compute_test4(features: list[str]) -> float:
    if not features:
        return float("nan")
    avail = [f for f in features if f in X_train.columns and f in X_test.columns]
    if len(avail) < 2:
        return float("nan")
    try:
        auc = evaluate_auc_test(X_train, y_train, X_test, y_test, avail)
        return round(float(auc), 4)
    except Exception:
        return float("nan")


def _compute_ext_auc(features: list[str]) -> float:
    if not features:
        return float("nan")
    avail = [f for f in features if f in X_train.columns and f in X_ext.columns]
    if len(avail) < 2:
        return float("nan")
    try:
        auc = evaluate_auc_test(X_train, y_train, X_ext, y_ext, avail)
        return round(float(auc), 4)
    except Exception:
        return float("nan")


def save_results(studies: dict[int, optuna.Study]) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    run_date = datetime.now().strftime("%Y%m%d")
    results = []
    all_ranks = sorted(PIPELINES.keys())

    for rank in all_ranks:
        study = studies.get(rank)
        if study is None:
            continue

        df_trials = study.trials_dataframe()
        b = STAGE3_METHODS.get(rank, {})
        pipeline_str = b.get("Pipeline", f"PT_S3_{rank}")

        df_trials["Rank"] = rank
        df_trials["Pipeline"] = pipeline_str
        df_trials["AUC3"] = b.get("AUC3", 0)
        df_trials["Fea3"] = b.get("Fea3", 0)
        df_trials["AUC_CV"] = df_trials["user_attrs_auc_cv"] if "user_attrs_auc_cv" in df_trials.columns else 0.0
        df_trials["Fea"] = df_trials["user_attrs_n_features"] if "user_attrs_n_features" in df_trials.columns else 0
        df_trials["Objective"] = df_trials["value"] if "value" in df_trials.columns else 0.0
        df_trials = df_trials.sort_values("Objective", ascending=False).reset_index(drop=True)

        per_pipeline = OUTPUT_DIR / f"PT_S3_{rank}_trials_{run_date}.csv"
        df_trials.to_csv(per_pipeline, index=False)
        logger.info("Trials saved: %s", per_pipeline)

        best = study.best_trial
        best_feats = best.user_attrs.get("selected_features", [])
        test4 = _compute_test4(best_feats)
        ext_auc = _compute_ext_auc(best_feats)
        auc4 = float(best.user_attrs.get("auc_cv", 0))
        gap4 = round(test4 - auc4, 4) if not np.isnan(test4) else float("nan")

        results.append(
            {
                "Rank": f"PT_S3_{rank}",
                "Pipeline": pipeline_str,
                "AUC3": b.get("AUC3", 0),
                "Fea3": b.get("Fea3", 0),
                "Test3": b.get("Test3", 0),
                "AUC4": round(auc4, 4),
                "Fea4": best.user_attrs.get("n_features", 0),
                "Objective": round(best.value, 4),
                "Best_params": str(best.params),
                "Test4": test4,
                "Gap4": gap4,
                "extAUC": ext_auc,
            }
        )

    res_df = pd.DataFrame(results)
    res_df["_rank_int"] = res_df["Rank"].str.replace("PT_S3_", "").astype(int)
    res_df = res_df.sort_values("_rank_int").drop(columns=["_rank_int"]).reset_index(drop=True)
    summary_path = OUTPUT_DIR / f"14_mar_task2_stage4_PT_results_{run_date}.csv"
    res_df.to_csv(summary_path, index=False)
    logger.info("\n%s\nResults saved: %s\n%s", "=" * 70, summary_path, "=" * 70)

    all_trials = []
    inter_counter = 1
    for rank in all_ranks:
        study = studies.get(rank)
        if study is None:
            continue
        b = STAGE3_METHODS.get(rank, {})
        pipeline_str = b.get("Pipeline", f"PT_S3_{rank}")
        for t in sorted(study.trials, key=lambda x: x.number):
            if t.state != optuna.trial.TrialState.COMPLETE:
                continue
            t_feats = t.user_attrs.get("selected_features", [])
            row = {
                "inter_no": inter_counter,
                "intra_no": f"PT_S3_{rank}_{t.number}",
                "S3_rank": rank,
                "Pipeline": pipeline_str,
                "AUC_CV": t.user_attrs.get("auc_cv"),
                "Fea": t.user_attrs.get("n_features"),
                "Objective": t.value,
                "Test4": _compute_test4(t_feats),
                "extAUC": _compute_ext_auc(t_feats),
                "Stage3_AUC3": b.get("AUC3"),
                "Stage3_Fea3": b.get("Fea3"),
            }
            row.update({f"params_{k}": v for k, v in t.params.items()})
            all_trials.append(row)
            inter_counter += 1

    df_all = pd.DataFrame(all_trials)
    all_out = OUTPUT_DIR / f"14_mar_task2_stage4_PT_ALLtrials_{run_date}.csv"
    df_all.to_csv(all_out, index=False)
    logger.info("All trials combined CSV saved: %s (%s rows)", all_out, len(df_all))


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(exist_ok=True, parents=True)
    log_file = LOG_DIR / f"14_mar_task2_stage4_PT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logging.captureWarnings(False)
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)
    log.info("Logging initialized: %s", log_file)
    return log


def test_db_connection() -> bool:
    if not PSYCOPG2_AVAILABLE:
        print("\nERROR: psycopg2-binary not installed. pip install psycopg2-binary\n")
        return False
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            connect_timeout=10,
        )
        conn.close()
        return True
    except Exception as e:
        print(f"\nERROR: Cannot connect to PostgreSQL: {e}\n")
        return False


def main() -> None:
    global logger

    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=None, help="Single pipeline rank (PT_S3_#)")
    parser.add_argument("--trials", type=int, default=None, help="Override n_trials")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("PT Task 2 Stage 4 Optuna: 9 pipelines, CV-only, full-chain tuning")
    print("=" * 70 + "\n")

    if not test_db_connection():
        sys.exit(1)
    print("Database connection OK\n")

    logger = setup_logging()
    random.seed(SEED)
    np.random.seed(SEED)

    _load_stage3_methods()
    _load_data()
    # No ReliefF precompute needed: PT shortlist has no ReliefF S1 pipelines
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # Sanity check: verify all selected ranks are in STAGE3_METHODS
    for r in PIPELINES:
        assert r in STAGE3_METHODS, f"Rank {r} not found in Stage 3 CSV"
        m = STAGE3_METHODS[r]
        assert m["S1"] and m["S2"] and m["S3"], f"Rank {r} has empty pipeline token"
    logger.info("Sanity check passed: all %s ranks found in Stage 3 methods", len(PIPELINES))

    ranks_to_run = [args.rank] if args.rank is not None else sorted(PIPELINES.keys())

    studies: dict[int, optuna.Study] = {}
    for rank in ranks_to_run:
        if rank not in STAGE3_METHODS:
            logger.warning("Rank %s not in Stage3, skip", rank)
            continue
        n_trials = args.trials if args.trials is not None else _n_trials(rank)
        studies[rank] = run_optimization(rank, n_trials)

    save_results(studies)

    logger.info("\n" + "=" * 70)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 70)
    logger.info("Total pipelines: %s", len(ranks_to_run))
    logger.info("Output: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
