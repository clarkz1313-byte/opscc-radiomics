# -*- coding: utf-8 -*-
"""
PT Stage 4 Optuna: 17 Pipelines, Full-Chain Tuning, CV-Only (2v)

Per Mar_2026/27_feb_PT_Stage4_Optuna_plan.md:
- 17 pipelines: 9 Performance, 5 Parsimony, 3 Exploratory
- Full-chain tuning: every S1/S2/S3 step has trial.suggest_* params
- CV-only objective: WEIGHT_CV * CV - WEIGHT_FEA * Fea_penalty (no Test term)
- Cox precompute mandatory before any pipeline run
- N_TRIALS=500, PT_S3_25 override 300 (RSF slow with n_jobs=1)
- HEAVY_PIPELINE_N_JOBS: PT_S3_25 uses n_jobs=1 to avoid RSF deadlock

Data: Mar_2026/27_feb_PT_development.csv (full dev, no test set)
Stage 3: Mar_2026/27_feb_PT_Stage3_2v_Processed_result.csv

Run: cd "D:/Uppsala thesis" && python Mar_2026/27_feb_PT_Stage4_2v.py
"""

from __future__ import annotations

import warnings
import sys
from pathlib import Path

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
try:
    from pandas.errors import Pandas4Warning
    warnings.filterwarnings("ignore", category=Pandas4Warning)
except ImportError:
    pass
import os
os.environ["PYTHONWARNINGS"] = "ignore"

import optuna
import numpy as np
import pandas as pd
import logging
import random
import joblib
from datetime import datetime
from typing import Tuple, Dict, Any

# Check PostgreSQL
try:
    import psycopg2
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fs_utils import (
    mrmr_selection,
    mutual_info_selection,
    lasso_cox_selection,
    elasticnet_cox_selection,
    xgboost_survival_selection,
    permutation_importance_survival,
    rsf_permutation_importance,
    evaluate_features_cv,
)

# ========== CONFIGURATION ==========

DEV_DATA_FILE = SCRIPT_DIR / "27_feb_PT_development.csv"
STAGE3_FILE = SCRIPT_DIR / "27_feb_PT_Stage3_2v_Processed_result.csv"
OUTPUT_DIR = SCRIPT_DIR / "27_feb_PT_Stage4_2v_outputs"
LOG_DIR = SCRIPT_DIR / "27_feb_PT_Stage4_2v_outputs"
CHECKPOINT_DIR = SCRIPT_DIR / "27_feb_PT_Stage4_2v_outputs"

EXTERNAL_FILE = SCRIPT_DIR / "27_feb_PT_external.csv"

DB_HOST = os.environ.get("OPTUNA_DB_HOST", "localhost")
DB_PORT = int(os.environ.get("OPTUNA_DB_PORT", "5432"))
DB_USER = os.environ.get("OPTUNA_DB_USER", "postgres")
DB_PASSWORD = os.environ.get("OPTUNA_DB_PASSWORD", "userdefined")
DB_NAME = os.environ.get("OPTUNA_DB_NAME", "optuna_db")
DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
STORAGE_TIMEOUT = 600

SEED = 42
N_TRIALS = 850
N_TRIALS_RANK_OVERRIDES = {"PT_S3_25": 300}
N_JOBS = 16
HEAVY_PIPELINE_N_JOBS = {
    "PT_S3_25": 1,   # D1_RSF: RSF internal parallelism deadlocks with n_jobs>1
    "PT_S3_1": 1,    # D3_PermImp as S2 (permimp_n_estimators 300-600)
    "PT_S3_2": 1,    # D3_PermImp as S2
    "PT_S3_6": 1,    # D3_PermImp as S2
    "PT_S3_9": 1,    # D3_PermImp as S2
    "PT_S3_14": 1,   # D3_PermImp as S2
}

N_FOLDS = 5
CV_SEED = 42

WEIGHT_CV = 0.65
WEIGHT_FEA = 0.35
FEA_MAX = 50

# 17 pipelines: PT_S3_1, 2, 3, 4, 5, 6, 9, 10, 14 (Perf), 17, 21, 25, 38, 40 (Pars), 20, 24, 32 (Explor)
PT_S3_RANKS = [1, 2, 3, 4, 5, 6, 9, 10, 14, 17, 20, 21, 24, 25, 32, 38, 40]

# ========== GLOBALS ==========

X_train = None
y_train = None
X_external_CHUS = None
X_external_CHUP = None
y_external_CHUS = None
y_external_CHUP = None
logger = None
GLOBAL_COX_P_VALUES = None
stage3_baselines: Dict[int, Dict[str, Any]] = {}


def load_external_data() -> None:
    """Load PT external CSV, split into CHUS and CHUP subsets. Logging only."""
    global X_external_CHUS, X_external_CHUP, y_external_CHUS, y_external_CHUP

    if not EXTERNAL_FILE.exists():
        logger.warning("External file not found: %s", EXTERNAL_FILE)
        return

    df = pd.read_csv(EXTERNAL_FILE)
    df["Center"] = df["PatientID"].str[:4]
    feature_cols = [c for c in df.columns if c not in ["PatientID", "Relapse", "RFS", "Center"]]
    train_medians = X_train[feature_cols].median()

    for center, attr_x, attr_y in [
        ("CHUS", "X_external_CHUS", "y_external_CHUS"),
        ("CHUP", "X_external_CHUP", "y_external_CHUP"),
    ]:
        sub = df[df["Center"] == center].copy()
        X = sub[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(train_medians)
        y = pd.DataFrame({"RFS_time": sub["RFS"].values, "event": sub["Relapse"].values.astype(bool)})
        globals()[attr_x] = X.reset_index(drop=True)
        globals()[attr_y] = y.reset_index(drop=True)

    logger.info(
        "External loaded: CHUS=%s pts (%s events), CHUP=%s pts (%s events)",
        len(X_external_CHUS), int(y_external_CHUS["event"].sum()),
        len(X_external_CHUP), int(y_external_CHUP["event"].sum()),
    )


def evaluate_external_cindex(features: list, X_ext: pd.DataFrame, y_ext: pd.DataFrame) -> float:
    """Fit CoxPH on full dev set with selected features, return c-index on external set. Logging only."""
    try:
        from lifelines import CoxPHFitter
        from lifelines.utils import concordance_index

        if X_ext is None or y_ext is None or len(X_ext) == 0:
            return float("nan")
        avail = [f for f in features if f in X_ext.columns and f in X_train.columns]
        if len(avail) < 2:
            return float("nan")

        df_fit = X_train[avail].copy()
        df_fit["T"] = y_train["RFS_time"].values
        df_fit["E"] = y_train["event"].values.astype(int)

        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(df_fit, duration_col="T", event_col="E", show_progress=False)

        risk = cph.predict_partial_hazard(X_ext[avail])
        ci = concordance_index(y_ext["RFS_time"].values, -risk.values, y_ext["event"].values)
        return round(float(ci), 4)
    except Exception as e:
        logger.debug("External cindex failed (%s pts, %s features): %s", len(X_ext), len(avail), e)
        return float("nan")


def precompute_cox_p_values(X: pd.DataFrame, y_df: pd.DataFrame) -> pd.Series:
    """Precompute univariate Cox p-values for all features (A1_Cox pipelines)."""
    try:
        from lifelines import CoxPHFitter
        from tqdm import tqdm
    except ImportError as e:
        raise ImportError("lifelines and tqdm required for Cox precompute") from e

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


def setup_logging() -> logging.Logger:
    """Setup logging to file and console."""
    LOG_DIR.mkdir(exist_ok=True, parents=True)
    log_file = LOG_DIR / f"27_feb_PT_Stage4_2v_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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


def load_data() -> None:
    """Load full development data (no test set). CV-only evaluation."""
    global X_train, y_train

    if not DEV_DATA_FILE.exists():
        raise FileNotFoundError(f"Data not found: {DEV_DATA_FILE}")

    df = pd.read_csv(DEV_DATA_FILE)
    feature_cols = [c for c in df.columns if c not in ["PatientID", "Relapse", "RFS"]]
    X = df[feature_cols]
    y_time = df["RFS"].values
    y_event = df["Relapse"].values.astype(bool)

    X_train = X.replace([np.inf, -np.inf], np.nan)
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    y_train = pd.DataFrame({"RFS_time": y_time, "event": y_event})

    logger.info("Loaded dev data: %s patients, %s features, %s events",
                len(df), len(feature_cols), int(y_event.sum()))


def load_stage3_baselines() -> None:
    """Load Stage 3 Processed result. Parse Rank as PT_S3_X -> rank num."""
    global stage3_baselines

    if not STAGE3_FILE.exists():
        logger.warning("Stage 3 file not found: %s", STAGE3_FILE)
        return

    df = pd.read_csv(STAGE3_FILE)
    for _, row in df.iterrows():
        rank_str = str(row.get("Rank", ""))
        if not rank_str.startswith("PT_S3_"):
            continue
        try:
            rank = int(rank_str.replace("PT_S3_", ""))
        except ValueError:
            continue
        stage3_baselines[rank] = {
            "CV3": float(row.get("CV3", 0)),
            "Fea3": int(row.get("Fea3", 0)),
            "Pipeline": str(row.get("Pipeline", "")),
        }
    logger.info("Loaded %s Stage 3 baselines", len(stage3_baselines))


def evaluate_cv(X_sel: pd.DataFrame, y: pd.DataFrame) -> float:
    """5-fold CV C-index."""
    if len(X_sel.columns) == 0:
        return 0.0
    features = list(X_sel.columns)
    cv_mean, _ = evaluate_features_cv(
        X_sel, y, features, n_splits=N_FOLDS, random_state=CV_SEED
    )
    return float(cv_mean) if not np.isnan(cv_mean) else 0.0


def _common_objective(trial, s1: list, s2: list, s3: list, rank: int, fea_max: int) -> float:
    """Shared CV-only objective. No test set."""
    n_features = len(s3)
    if n_features > fea_max:
        return 0.0
    X_sel = X_train[s1][s2][s3]
    selected_features = list(X_sel.columns)
    cv_score = evaluate_cv(X_sel, y_train)
    fea_penalty = min(n_features / FEA_MAX, 1.0)
    objective = WEIGHT_CV * cv_score - WEIGHT_FEA * fea_penalty
    trial.set_user_attr("cv_score", round(cv_score, 4))
    trial.set_user_attr("n_features", n_features)
    trial.set_user_attr("fea_penalty", round(fea_penalty, 4))
    trial.set_user_attr("selected_features", selected_features)
    logger.info(
        "PT_S3_%s Trial %s: CV=%.4f Fea=%s Obj=%.4f",
        rank, trial.number, cv_score, n_features, objective
    )
    return objective


# ========== OBJECTIVE FUNCTIONS (17 pipelines) ==========

def objective_PT_S3_1(trial: optuna.Trial) -> float:
    """C1_LASSO -> D3_PermImp -> D2_XGBoost (6 params)."""
    try:
        lt = trial.suggest_int("lasso_target_features", 50, 150)
        la = trial.suggest_int("lasso_n_alphas", 50, 200)
        s1 = lasso_cox_selection(X_train, y_train, target_features=lt, n_alphas=la)
        if len(s1) < 10:
            return 0.0

        pn = trial.suggest_int("permimp_n_features", 30, 70)
        pne = trial.suggest_int("permimp_n_estimators", 300, 600)
        s2 = permutation_importance_survival(
            X_train[s1], y_train, n_features=min(pn, len(s1)),
            n_estimators=pne, random_state=SEED
        )
        if len(s2) == 0:
            return 0.0

        xn = trial.suggest_int("xgb_n_features", 30, 70)
        xne = trial.suggest_int("xgb_n_estimators", 80, 150)
        s3 = xgboost_survival_selection(
            X_train[s1][s2], y_train, n_features=min(xn, len(s2)),
            n_estimators=xne, random_state=SEED
        )
        if len(s3) == 0:
            s3 = s2[:20]
        return _common_objective(trial, s1, s2, s3, 1, 80)
    except Exception as e:
        logger.error("PT_S3_1 Trial %s failed: %s", trial.number, e)
        return 0.0


def objective_PT_S3_2(trial: optuna.Trial) -> float:
    """C2_ElasticNet -> D3_PermImp -> D2_XGBoost (8 params)."""
    try:
        et = trial.suggest_int("elasticnet_target_features", 50, 150)
        el1 = trial.suggest_float("elasticnet_l1_ratio", 0.3, 0.9)
        ea = trial.suggest_int("elasticnet_n_alphas", 50, 200)
        s1 = elasticnet_cox_selection(
            X_train, y_train, target_features=et, l1_ratio=el1, n_alphas=ea
        )
        if len(s1) < 10:
            return 0.0

        pn = trial.suggest_int("permimp_n_features", 30, 70)
        pne = trial.suggest_int("permimp_n_estimators", 300, 600)
        s2 = permutation_importance_survival(
            X_train[s1], y_train, n_features=min(pn, len(s1)),
            n_estimators=pne, random_state=SEED
        )
        if len(s2) == 0:
            return 0.0

        xn = trial.suggest_int("xgb_n_features", 30, 70)
        xne = trial.suggest_int("xgb_n_estimators", 80, 150)
        s3 = xgboost_survival_selection(
            X_train[s1][s2], y_train, n_features=min(xn, len(s2)),
            n_estimators=xne, random_state=SEED
        )
        if len(s3) == 0:
            s3 = s2[:20]
        return _common_objective(trial, s1, s2, s3, 2, 80)
    except Exception as e:
        logger.error("PT_S3_2 Trial %s failed: %s", trial.number, e)
        return 0.0


def objective_PT_S3_3(trial: optuna.Trial) -> float:
    """B3_mRMR -> C1_LASSO -> D2_XGBoost (5 params)."""
    try:
        mn = trial.suggest_int("mrmr_n_features", 20, 80)
        s1 = mrmr_selection(X_train, y_train, n_features=mn)
        if len(s1) < 10:
            return 0.0

        lt = trial.suggest_int("lasso_target_features", 50, 150)
        la = trial.suggest_int("lasso_n_alphas", 50, 200)
        s2 = lasso_cox_selection(X_train[s1], y_train, target_features=lt, n_alphas=la)
        if len(s2) == 0:
            s2 = s1[:20]

        xn = trial.suggest_int("xgb_n_features", 30, 70)
        xne = trial.suggest_int("xgb_n_estimators", 80, 150)
        s3 = xgboost_survival_selection(
            X_train[s1][s2], y_train, n_features=min(xn, len(s2)),
            n_estimators=xne, random_state=SEED
        )
        if len(s3) == 0:
            s3 = s2[:15]
        return _common_objective(trial, s1, s2, s3, 3, 80)
    except Exception as e:
        logger.error("PT_S3_3 Trial %s failed: %s", trial.number, e)
        return 0.0


def objective_PT_S3_4(trial: optuna.Trial) -> float:
    """B3_mRMR -> C1_LASSO -> C2_ElasticNet (8 params)."""
    try:
        mn = trial.suggest_int("mrmr_n_features", 20, 80)
        s1 = mrmr_selection(X_train, y_train, n_features=mn)
        if len(s1) < 10:
            return 0.0

        lt = trial.suggest_int("lasso_target_features", 50, 150)
        la = trial.suggest_int("lasso_n_alphas", 50, 200)
        s2 = lasso_cox_selection(X_train[s1], y_train, target_features=lt, n_alphas=la)
        if len(s2) == 0:
            s2 = s1[:20]

        et = trial.suggest_int("elasticnet_target_features", 50, 150)
        el1 = trial.suggest_float("elasticnet_l1_ratio", 0.3, 0.9)
        ea = trial.suggest_int("elasticnet_n_alphas", 50, 200)
        s3 = elasticnet_cox_selection(
            X_train[s1][s2], y_train, target_features=et, l1_ratio=el1, n_alphas=ea
        )
        if len(s3) == 0:
            s3 = s2[:15]
        return _common_objective(trial, s1, s2, s3, 4, 80)
    except Exception as e:
        logger.error("PT_S3_4 Trial %s failed: %s", trial.number, e)
        return 0.0


def objective_PT_S3_5(trial: optuna.Trial) -> float:
    """B3_mRMR -> C2_ElasticNet -> C1_LASSO (8 params)."""
    try:
        mn = trial.suggest_int("mrmr_n_features", 20, 80)
        s1 = mrmr_selection(X_train, y_train, n_features=mn)
        if len(s1) < 10:
            return 0.0

        et = trial.suggest_int("elasticnet_target_features", 50, 150)
        el1 = trial.suggest_float("elasticnet_l1_ratio", 0.3, 0.9)
        ea = trial.suggest_int("elasticnet_n_alphas", 50, 200)
        s2 = elasticnet_cox_selection(
            X_train[s1], y_train, target_features=et, l1_ratio=el1, n_alphas=ea
        )
        if len(s2) == 0:
            s2 = s1[:20]

        lt = trial.suggest_int("lasso_target_features", 50, 150)
        la = trial.suggest_int("lasso_n_alphas", 50, 200)
        s3 = lasso_cox_selection(X_train[s1][s2], y_train, target_features=lt, n_alphas=la)
        if len(s3) == 0:
            s3 = s2[:15]
        return _common_objective(trial, s1, s2, s3, 5, 80)
    except Exception as e:
        logger.error("PT_S3_5 Trial %s failed: %s", trial.number, e)
        return 0.0


def objective_PT_S3_6(trial: optuna.Trial) -> float:
    """B3_mRMR -> D3_PermImp -> C1_LASSO (6 params)."""
    try:
        mn = trial.suggest_int("mrmr_n_features", 20, 80)
        s1 = mrmr_selection(X_train, y_train, n_features=mn)
        if len(s1) < 10:
            return 0.0

        pn = trial.suggest_int("permimp_n_features", 30, 70)
        pne = trial.suggest_int("permimp_n_estimators", 300, 600)
        s2 = permutation_importance_survival(
            X_train[s1], y_train, n_features=min(pn, len(s1)),
            n_estimators=pne, random_state=SEED
        )
        if len(s2) == 0:
            return 0.0

        lt = trial.suggest_int("lasso_target_features", 50, 150)
        la = trial.suggest_int("lasso_n_alphas", 50, 200)
        s3 = lasso_cox_selection(X_train[s1][s2], y_train, target_features=lt, n_alphas=la)
        if len(s3) == 0:
            s3 = s2[:15]
        return _common_objective(trial, s1, s2, s3, 6, 80)
    except Exception as e:
        logger.error("PT_S3_6 Trial %s failed: %s", trial.number, e)
        return 0.0


def objective_PT_S3_9(trial: optuna.Trial) -> float:
    """B3_mRMR -> D3_PermImp -> C2_ElasticNet (7 params)."""
    try:
        mn = trial.suggest_int("mrmr_n_features", 20, 80)
        s1 = mrmr_selection(X_train, y_train, n_features=mn)
        if len(s1) < 10:
            return 0.0

        pn = trial.suggest_int("permimp_n_features", 30, 70)
        pne = trial.suggest_int("permimp_n_estimators", 300, 600)
        s2 = permutation_importance_survival(
            X_train[s1], y_train, n_features=min(pn, len(s1)),
            n_estimators=pne, random_state=SEED
        )
        if len(s2) == 0:
            return 0.0

        et = trial.suggest_int("elasticnet_target_features", 50, 150)
        el1 = trial.suggest_float("elasticnet_l1_ratio", 0.3, 0.9)
        ea = trial.suggest_int("elasticnet_n_alphas", 50, 200)
        s3 = elasticnet_cox_selection(
            X_train[s1][s2], y_train, target_features=et, l1_ratio=el1, n_alphas=ea
        )
        if len(s3) == 0:
            s3 = s2[:15]
        return _common_objective(trial, s1, s2, s3, 9, 80)
    except Exception as e:
        logger.error("PT_S3_9 Trial %s failed: %s", trial.number, e)
        return 0.0


def objective_PT_S3_10(trial: optuna.Trial) -> float:
    """B3_mRMR -> C2_ElasticNet -> D2_XGBoost (6 params)."""
    try:
        mn = trial.suggest_int("mrmr_n_features", 20, 80)
        s1 = mrmr_selection(X_train, y_train, n_features=mn)
        if len(s1) < 10:
            return 0.0

        et = trial.suggest_int("elasticnet_target_features", 50, 150)
        el1 = trial.suggest_float("elasticnet_l1_ratio", 0.3, 0.9)
        ea = trial.suggest_int("elasticnet_n_alphas", 50, 200)
        s2 = elasticnet_cox_selection(
            X_train[s1], y_train, target_features=et, l1_ratio=el1, n_alphas=ea
        )
        if len(s2) == 0:
            s2 = s1[:20]

        xn = trial.suggest_int("xgb_n_features", 30, 70)
        xne = trial.suggest_int("xgb_n_estimators", 80, 150)
        s3 = xgboost_survival_selection(
            X_train[s1][s2], y_train, n_features=min(xn, len(s2)),
            n_estimators=xne, random_state=SEED
        )
        if len(s3) == 0:
            s3 = s2[:15]
        return _common_objective(trial, s1, s2, s3, 10, 80)
    except Exception as e:
        logger.error("PT_S3_10 Trial %s failed: %s", trial.number, e)
        return 0.0


def objective_PT_S3_14(trial: optuna.Trial) -> float:
    """B3_mRMR -> D3_PermImp -> D2_XGBoost (5 params)."""
    try:
        mn = trial.suggest_int("mrmr_n_features", 20, 80)
        s1 = mrmr_selection(X_train, y_train, n_features=mn)
        if len(s1) < 10:
            return 0.0

        pn = trial.suggest_int("permimp_n_features", 30, 70)
        pne = trial.suggest_int("permimp_n_estimators", 300, 600)
        s2 = permutation_importance_survival(
            X_train[s1], y_train, n_features=min(pn, len(s1)),
            n_estimators=pne, random_state=SEED
        )
        if len(s2) == 0:
            return 0.0

        xn = trial.suggest_int("xgb_n_features", 30, 70)
        xne = trial.suggest_int("xgb_n_estimators", 80, 150)
        s3 = xgboost_survival_selection(
            X_train[s1][s2], y_train, n_features=min(xn, len(s2)),
            n_estimators=xne, random_state=SEED
        )
        if len(s3) == 0:
            s3 = s2[:15]
        return _common_objective(trial, s1, s2, s3, 14, 80)
    except Exception as e:
        logger.error("PT_S3_14 Trial %s failed: %s", trial.number, e)
        return 0.0


def objective_PT_S3_17(trial: optuna.Trial) -> float:
    """A1_Cox -> B3_mRMR -> C1_LASSO (5 params)."""
    try:
        cp = trial.suggest_float("cox_p_threshold", 0.0005, 0.05, log=True)
        s1 = GLOBAL_COX_P_VALUES[GLOBAL_COX_P_VALUES < cp].index.tolist()
        if len(s1) < 10:
            return 0.0

        mn = trial.suggest_int("mrmr_n_features", 20, 80)
        s2 = mrmr_selection(X_train[s1], y_train, n_features=min(mn, len(s1)))
        if len(s2) == 0:
            return 0.0

        lt = trial.suggest_int("lasso_target_features", 50, 150)
        la = trial.suggest_int("lasso_n_alphas", 50, 200)
        s3 = lasso_cox_selection(X_train[s1][s2], y_train, target_features=lt, n_alphas=la)
        if len(s3) == 0:
            s3 = s2[:15]
        return _common_objective(trial, s1, s2, s3, 17, 50)
    except Exception as e:
        logger.error("PT_S3_17 Trial %s failed: %s", trial.number, e)
        return 0.0


def objective_PT_S3_20(trial: optuna.Trial) -> float:
    """A1_Cox -> C1_LASSO -> C2_ElasticNet (8 params)."""
    try:
        cp = trial.suggest_float("cox_p_threshold", 0.0005, 0.05, log=True)
        s1 = GLOBAL_COX_P_VALUES[GLOBAL_COX_P_VALUES < cp].index.tolist()
        if len(s1) < 10:
            return 0.0

        lt = trial.suggest_int("lasso_target_features", 50, 150)
        la = trial.suggest_int("lasso_n_alphas", 50, 200)
        s2 = lasso_cox_selection(X_train[s1], y_train, target_features=lt, n_alphas=la)
        if len(s2) == 0:
            s2 = s1[:30]

        et = trial.suggest_int("elasticnet_target_features", 50, 150)
        el1 = trial.suggest_float("elasticnet_l1_ratio", 0.3, 0.9)
        ea = trial.suggest_int("elasticnet_n_alphas", 50, 200)
        s3 = elasticnet_cox_selection(
            X_train[s1][s2], y_train, target_features=et, l1_ratio=el1, n_alphas=ea
        )
        if len(s3) == 0:
            s3 = s2[:15]
        return _common_objective(trial, s1, s2, s3, 20, 80)
    except Exception as e:
        logger.error("PT_S3_20 Trial %s failed: %s", trial.number, e)
        return 0.0


def objective_PT_S3_21(trial: optuna.Trial) -> float:
    """A1_Cox -> D2_XGBoost -> C1_LASSO (6 params)."""
    try:
        cp = trial.suggest_float("cox_p_threshold", 0.0005, 0.05, log=True)
        s1 = GLOBAL_COX_P_VALUES[GLOBAL_COX_P_VALUES < cp].index.tolist()
        if len(s1) < 10:
            return 0.0

        xn = trial.suggest_int("xgb_n_features", 30, 70)
        xne = trial.suggest_int("xgb_n_estimators", 80, 150)
        s2 = xgboost_survival_selection(
            X_train[s1], y_train, n_features=min(xn, len(s1)),
            n_estimators=xne, random_state=SEED
        )
        if len(s2) == 0:
            return 0.0

        lt = trial.suggest_int("lasso_target_features", 50, 150)
        la = trial.suggest_int("lasso_n_alphas", 50, 200)
        s3 = lasso_cox_selection(X_train[s1][s2], y_train, target_features=lt, n_alphas=la)
        if len(s3) == 0:
            s3 = s2[:15]
        return _common_objective(trial, s1, s2, s3, 21, 50)
    except Exception as e:
        logger.error("PT_S3_21 Trial %s failed: %s", trial.number, e)
        return 0.0


def objective_PT_S3_24(trial: optuna.Trial) -> float:
    """A1_Cox -> C2_ElasticNet -> C1_LASSO (8 params)."""
    try:
        cp = trial.suggest_float("cox_p_threshold", 0.0005, 0.05, log=True)
        s1 = GLOBAL_COX_P_VALUES[GLOBAL_COX_P_VALUES < cp].index.tolist()
        if len(s1) < 10:
            return 0.0

        et = trial.suggest_int("elasticnet_target_features", 50, 150)
        el1 = trial.suggest_float("elasticnet_l1_ratio", 0.3, 0.9)
        ea = trial.suggest_int("elasticnet_n_alphas", 50, 200)
        s2 = elasticnet_cox_selection(
            X_train[s1], y_train, target_features=et, l1_ratio=el1, n_alphas=ea
        )
        if len(s2) == 0:
            s2 = s1[:30]

        lt = trial.suggest_int("lasso_target_features", 50, 150)
        la = trial.suggest_int("lasso_n_alphas", 50, 200)
        s3 = lasso_cox_selection(X_train[s1][s2], y_train, target_features=lt, n_alphas=la)
        if len(s3) == 0:
            s3 = s2[:15]
        return _common_objective(trial, s1, s2, s3, 24, 80)
    except Exception as e:
        logger.error("PT_S3_24 Trial %s failed: %s", trial.number, e)
        return 0.0


def objective_PT_S3_25(trial: optuna.Trial) -> float:
    """A1_Cox -> D1_RSF_PermImp -> C1_LASSO (6 params). HEAVY: n_jobs=1."""
    try:
        cp = trial.suggest_float("cox_p_threshold", 0.0005, 0.05, log=True)
        s1 = GLOBAL_COX_P_VALUES[GLOBAL_COX_P_VALUES < cp].index.tolist()
        if len(s1) < 10:
            return 0.0

        rn = trial.suggest_int("rsf_n_features", 30, 70)
        rne = trial.suggest_int("rsf_n_estimators", 300, 600)
        s2 = rsf_permutation_importance(
            X_train[s1], y_train, n_features=min(rn, len(s1)),
            n_estimators=rne, random_state=SEED
        )
        if len(s2) == 0:
            return 0.0

        lt = trial.suggest_int("lasso_target_features", 50, 150)
        la = trial.suggest_int("lasso_n_alphas", 50, 200)
        s3 = lasso_cox_selection(X_train[s1][s2], y_train, target_features=lt, n_alphas=la)
        if len(s3) == 0:
            s3 = s2[:15]
        return _common_objective(trial, s1, s2, s3, 25, 50)
    except Exception as e:
        logger.error("PT_S3_25 Trial %s failed: %s", trial.number, e)
        return 0.0


def objective_PT_S3_32(trial: optuna.Trial) -> float:
    """A1_Cox -> D2_XGBoost -> B3_mRMR (5 params)."""
    try:
        cp = trial.suggest_float("cox_p_threshold", 0.0005, 0.05, log=True)
        s1 = GLOBAL_COX_P_VALUES[GLOBAL_COX_P_VALUES < cp].index.tolist()
        if len(s1) < 10:
            return 0.0

        xn = trial.suggest_int("xgb_n_features", 30, 70)
        xne = trial.suggest_int("xgb_n_estimators", 80, 150)
        s2 = xgboost_survival_selection(
            X_train[s1], y_train, n_features=min(xn, len(s1)),
            n_estimators=xne, random_state=SEED
        )
        if len(s2) == 0:
            return 0.0

        mn = trial.suggest_int("mrmr_n_features", 20, 80)
        s3 = mrmr_selection(X_train[s1][s2], y_train, n_features=min(mn, len(s2)))
        if len(s3) == 0:
            s3 = s2[:15]
        return _common_objective(trial, s1, s2, s3, 32, 80)
    except Exception as e:
        logger.error("PT_S3_32 Trial %s failed: %s", trial.number, e)
        return 0.0


def objective_PT_S3_38(trial: optuna.Trial) -> float:
    """A4_MI -> C1_LASSO -> C2_ElasticNet (8 params)."""
    try:
        mk = trial.suggest_int("mi_k_features", 50, 200)
        s1 = mutual_info_selection(X_train, y_train, k_features=mk)
        if len(s1) < 10:
            return 0.0

        lt = trial.suggest_int("lasso_target_features", 50, 150)
        la = trial.suggest_int("lasso_n_alphas", 50, 200)
        s2 = lasso_cox_selection(X_train[s1], y_train, target_features=lt, n_alphas=la)
        if len(s2) == 0:
            s2 = s1[:20]

        et = trial.suggest_int("elasticnet_target_features", 50, 150)
        el1 = trial.suggest_float("elasticnet_l1_ratio", 0.3, 0.9)
        ea = trial.suggest_int("elasticnet_n_alphas", 50, 200)
        s3 = elasticnet_cox_selection(
            X_train[s1][s2], y_train, target_features=et, l1_ratio=el1, n_alphas=ea
        )
        if len(s3) == 0:
            s3 = s2[:15]
        return _common_objective(trial, s1, s2, s3, 38, 80)
    except Exception as e:
        logger.error("PT_S3_38 Trial %s failed: %s", trial.number, e)
        return 0.0


def objective_PT_S3_40(trial: optuna.Trial) -> float:
    """A4_MI -> C1_LASSO -> D2_XGBoost (6 params)."""
    try:
        mk = trial.suggest_int("mi_k_features", 50, 200)
        s1 = mutual_info_selection(X_train, y_train, k_features=mk)
        if len(s1) < 10:
            return 0.0

        lt = trial.suggest_int("lasso_target_features", 50, 150)
        la = trial.suggest_int("lasso_n_alphas", 50, 200)
        s2 = lasso_cox_selection(X_train[s1], y_train, target_features=lt, n_alphas=la)
        if len(s2) == 0:
            s2 = s1[:20]

        xn = trial.suggest_int("xgb_n_features", 30, 70)
        xne = trial.suggest_int("xgb_n_estimators", 80, 150)
        s3 = xgboost_survival_selection(
            X_train[s1][s2], y_train, n_features=min(xn, len(s2)),
            n_estimators=xne, random_state=SEED
        )
        if len(s3) == 0:
            s3 = s2[:15]
        return _common_objective(trial, s1, s2, s3, 40, 80)
    except Exception as e:
        logger.error("PT_S3_40 Trial %s failed: %s", trial.number, e)
        return 0.0


# ========== OBJECTIVE MAP ==========

OBJECTIVE_MAP = {
    1: objective_PT_S3_1,
    2: objective_PT_S3_2,
    3: objective_PT_S3_3,
    4: objective_PT_S3_4,
    5: objective_PT_S3_5,
    6: objective_PT_S3_6,
    9: objective_PT_S3_9,
    10: objective_PT_S3_10,
    14: objective_PT_S3_14,
    17: objective_PT_S3_17,
    20: objective_PT_S3_20,
    21: objective_PT_S3_21,
    24: objective_PT_S3_24,
    25: objective_PT_S3_25,
    32: objective_PT_S3_32,
    38: objective_PT_S3_38,
    40: objective_PT_S3_40,
}


def run_optimization(rank: int, n_trials: int) -> optuna.Study:
    """Run Optuna for one pipeline."""
    study_name = f"PT_S3_{rank}_Stage4"
    n_jobs = HEAVY_PIPELINE_N_JOBS.get(f"PT_S3_{rank}", N_JOBS)

    logger.info("\n%s", "=" * 70)
    logger.info("Starting: %s", study_name)
    if rank in stage3_baselines:
        b = stage3_baselines[rank]
        logger.info("Stage 3 baseline: CV3=%.4f Fea3=%s", b["CV3"], b["Fea3"])
        logger.info("Pipeline: %s", b["Pipeline"])
    logger.info("Trials: %s n_jobs=%s | Objective: %.2f*CV - %.2f*Fea", n_trials, n_jobs, WEIGHT_CV, WEIGHT_FEA)
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
            OBJECTIVE_MAP[rank],
            n_trials=remaining,
            n_jobs=n_jobs,
            show_progress_bar=True,
            timeout=None,
        )
    except KeyboardInterrupt:
        logger.warning("Optimization interrupted. Results saved.")

    CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
    cp_path = CHECKPOINT_DIR / f"{study_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    joblib.dump(study, cp_path)
    logger.info("Checkpoint saved: %s", cp_path)
    return study


def save_results(studies: Dict[int, optuna.Study]) -> None:
    """Save per-pipeline trials and summary CSV."""
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    run_date = datetime.now().strftime("%Y%m%d")
    results = []

    for rank in PT_S3_RANKS:
        study = studies.get(rank)
        if study is None:
            continue
        df_trials = study.trials_dataframe()
        b = stage3_baselines.get(rank, {})
        pipeline = b.get("Pipeline", f"PT_S3_{rank}")

        df_trials["Rank"] = rank
        df_trials["Pipeline"] = f"PT_S3_{rank}:{pipeline}"
        df_trials["Stage3_CV"] = b.get("CV3", 0)
        df_trials["Stage3_Fea"] = b.get("Fea3", 0)
        if "user_attrs_cv_score" in df_trials.columns:
            df_trials["CV"] = df_trials["user_attrs_cv_score"]
        else:
            df_trials["CV"] = 0.0
        if "user_attrs_n_features" in df_trials.columns:
            df_trials["Fea"] = df_trials["user_attrs_n_features"]
        else:
            df_trials["Fea"] = 0
        if "value" in df_trials.columns:
            df_trials["Objective"] = df_trials["value"]
        else:
            df_trials["Objective"] = 0.0

        df_trials = df_trials.sort_values("Objective", ascending=False).reset_index(drop=True)
        pf = OUTPUT_DIR / f"PT_S3_{rank}_trials_{run_date}.csv"
        df_trials.to_csv(pf, index=False)
        logger.info("Trials saved: %s", pf)

        best = study.best_trial
        features = best.user_attrs.get("selected_features", [])
        ext_chus = float("nan")
        ext_chup = float("nan")
        if features:
            if X_external_CHUS is not None:
                ext_chus = evaluate_external_cindex(features, X_external_CHUS, y_external_CHUS)
            if X_external_CHUP is not None:
                ext_chup = evaluate_external_cindex(features, X_external_CHUP, y_external_CHUP)
        logger.info("  PT_S3_%s best: Ext CHUS=%.4f  Ext CHUP=%.4f", rank, ext_chus, ext_chup)

        results.append({
            "Rank": rank,
            "Pipeline": f"PT_S3_{rank}:{pipeline}",
            "Stage3_CV": b.get("CV3", 0),
            "Stage3_Fea": b.get("Fea3", 0),
            "Optuna_CV": round(best.user_attrs.get("cv_score", 0), 4),
            "Optuna_Fea": best.user_attrs.get("n_features", 0),
            "Objective": round(best.value, 4),
            "Best_params": str(best.params),
            "Ext_CHUS": ext_chus,
            "Ext_CHUP": ext_chup,
        })

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values(["Rank"]).reset_index(drop=True)
    out_file = OUTPUT_DIR / f"27_feb_PT_Stage4_2v_results_{run_date}.csv"
    res_df.to_csv(out_file, index=False)
    logger.info("\n%s\nResults saved: %s\n%s", "=" * 70, out_file, "=" * 70)

    # Combined all-trials CSV with inter_no, intra_no, Ext_CHUS, Ext_CHUP per row
    all_trials = []
    inter_counter = 1
    logger.info("Building combined CSV with Ext_CHUS/Ext_CHUP per trial...")
    for rank in PT_S3_RANKS:
        study = studies.get(rank)
        if study is None:
            continue
        for t in sorted(study.trials, key=lambda x: x.number):
            if t.state != optuna.trial.TrialState.COMPLETE:
                continue
            b = stage3_baselines.get(rank, {})
            pipeline = b.get("Pipeline", f"PT_S3_{rank}")
            features = t.user_attrs.get("selected_features", [])
            ext_chus = float("nan")
            ext_chup = float("nan")
            if features:
                if X_external_CHUS is not None:
                    ext_chus = evaluate_external_cindex(features, X_external_CHUS, y_external_CHUS)
                if X_external_CHUP is not None:
                    ext_chup = evaluate_external_cindex(features, X_external_CHUP, y_external_CHUP)
            row = {
                "inter_no": inter_counter,
                "intra_no": f"PT_S3_{rank}_{t.number}",
                "S3_rank": rank,
                "Pipeline": f"PT_S3_{rank}:{pipeline}",
                "CV": t.user_attrs.get("cv_score"),
                "Fea": t.user_attrs.get("n_features"),
                "Objective": t.value,
                "Stage3_CV": b.get("CV3"),
                "Stage3_Fea": b.get("Fea3"),
                "Ext_CHUS": ext_chus,
                "Ext_CHUP": ext_chup,
            }
            row.update({f"params_{k}": v for k, v in t.params.items()})
            all_trials.append(row)
            inter_counter += 1

    df_all = pd.DataFrame(all_trials)
    all_out = OUTPUT_DIR / f"27_feb_PT_Stage4_2v_ALLtrials_{run_date}.csv"
    df_all.to_csv(all_out, index=False)
    logger.info("All trials combined CSV saved: %s (%s rows)", all_out, len(df_all))


def test_db_connection() -> bool:
    """Test PostgreSQL connection."""
    if not PSYCOPG2_AVAILABLE:
        print("\nERROR: psycopg2-binary not installed. pip install psycopg2-binary\n")
        return False
    try:
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT, user=DB_USER,
            password=DB_PASSWORD, database=DB_NAME, connect_timeout=10
        )
        conn.close()
        return True
    except Exception as e:
        print(f"\nERROR: Cannot connect to PostgreSQL: {e}\n")
        return False


def main() -> None:
    global logger, GLOBAL_COX_P_VALUES

    print("\n" + "=" * 70)
    print("PT Stage 4 Optuna (2v): 17 pipelines, CV-only, full-chain tuning")
    print("=" * 70 + "\n")

    if not test_db_connection():
        sys.exit(1)
    print("Database connection OK\n")

    logger = setup_logging()
    random.seed(SEED)
    np.random.seed(SEED)

    load_data()
    load_external_data()
    load_stage3_baselines()

    GLOBAL_COX_P_VALUES = precompute_cox_p_values(X_train, y_train)
    logger.info("Precomputed Cox p-values for %s features", len(GLOBAL_COX_P_VALUES))

    studies = {}
    for rank in PT_S3_RANKS:
        n_trials = N_TRIALS_RANK_OVERRIDES.get(f"PT_S3_{rank}", N_TRIALS)
        studies[rank] = run_optimization(rank, n_trials)

    save_results(studies)

    logger.info("\n" + "=" * 70)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 70)
    logger.info("Total pipelines: %s", len(PT_S3_RANKS))
    logger.info("Output: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
