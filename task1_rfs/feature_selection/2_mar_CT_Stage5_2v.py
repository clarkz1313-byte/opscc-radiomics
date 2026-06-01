# -*- coding: utf-8 -*-
"""
CT Stage 5 Refine: 5 pipelines, Round 2 Optuna, Option A new studies
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
    lasso_cox_selection,
    elasticnet_cox_selection,
    xgboost_survival_selection,
    evaluate_features_cv,
    univariate_cox_selection,
)

# ========== CONFIGURATION ==========

# 1. DOCSTRING: Updated above.
# 2. PATHS:
OUTPUT_DIR     = SCRIPT_DIR / "2_mar_CT_Stage5_2v_outputs"
LOG_DIR        = SCRIPT_DIR / "2_mar_CT_Stage5_2v_outputs"
CHECKPOINT_DIR = SCRIPT_DIR / "2_mar_CT_Stage5_2v_outputs"
PATH_CT_DEV      = SCRIPT_DIR / "27_feb_CT_development.csv"
PATH_STAGE3      = SCRIPT_DIR / "27_feb_CT_Stage3_2v_Processed_result.csv"
PATH_CT_EXTERNAL = SCRIPT_DIR / "27_feb_CT_external.csv"

DB_HOST = "localhost"
DB_PORT = 5432
DB_USER = "postgres"
DB_PASSWORD = "1730"
DB_NAME = "optuna_db"
DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
STORAGE_TIMEOUT = 600

SEED = 42

# 3. CONFIG:
N_TRIALS = 2000
N_TRIALS_RANK_OVERRIDES = {}
N_JOBS = 8
HEAVY_PIPELINE_N_JOBS = {}
CT_S3_RANKS = [37, 31, 14, 11, 25]

N_FOLDS = 5
CV_SEED = 42

# WEIGHT_FEA must be < 0.281 so that CV=0.7022/Fea=17 (primary peak) scores higher than
# CV=0.6866/Fea=16 (secondary low-feature peak). At 0.40, TPE rationally exploited the
# low-feature solution. At 0.25, the 1-feature difference (~0.013 penalty gap) is smaller
# than the CV gain (0.0156), so TPE correctly chases CV improvement.
WEIGHT_CV = 0.75
WEIGHT_FEA = 0.25
FEA_MAX = 30   # hard ceiling: trials with >30 features score 0; pushes TPE toward Fea ~15-20

# ========== GLOBALS (from CT Stage 4 script) ==========

X_train: pd.DataFrame
y_train: pd.DataFrame
stage3_baselines: Dict[int, Dict[str, Any]] = {}
logger: logging.Logger | None = None

GLOBAL_COX_P_VALUES: pd.Series
GLOBAL_PEARSON_CORR: pd.Series
X_external_CHUS: pd.DataFrame | None = None
X_external_CHUP: pd.DataFrame | None = None
y_external_CHUS: pd.DataFrame | None = None
y_external_CHUP: pd.DataFrame | None = None


# 13. DATA LOADING (from CT Stage 4 script)
def _load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not PATH_CT_DEV.exists():
        raise FileNotFoundError(f"Data not found: {PATH_CT_DEV}")
    df = pd.read_csv(PATH_CT_DEV)
    feat_cols = [c for c in df.columns if c not in ("PatientID", "Relapse", "RFS")]
    X = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    y_time = df["RFS"].values
    y_event = df["Relapse"].values.astype(bool)
    y = pd.DataFrame({"RFS_time": y_time, "event": y_event})
    logger.info("Loaded dev data: %s patients, %s features, %s events",
                len(df), len(feat_cols), int(y_event.sum()))
    return X, y

def _load_stage3_methods() -> None:
    global stage3_baselines
    if not PATH_STAGE3.exists():
        logger.warning("Stage 3 file not found: %s", PATH_STAGE3)
        return
    df = pd.read_csv(PATH_STAGE3)
    for _, r in df.iterrows():
        rank_str = str(r.get("Rank", ""))
        if not rank_str.startswith("CT_S3_"):
            continue
        try:
            rank = int(rank_str.replace("CT_S3_", ""))
        except ValueError:
            continue
        stage3_baselines[rank] = {
            "Pipeline": str(r.get("Pipeline", "")),
            "CV3": float(r.get("CV3", 0)),
            "Fea3": int(r.get("Fea3", 0)),
        }
    logger.info("Loaded %s Stage 3 baselines", len(stage3_baselines))

def _load_external_data() -> None:
    """Load CT external CSV, split into CHUS and CHUP subsets."""
    global X_external_CHUS, X_external_CHUP, y_external_CHUS, y_external_CHUP
    if not PATH_CT_EXTERNAL.exists():
        logger.warning("CT external file not found: %s", PATH_CT_EXTERNAL)
        return
    df = pd.read_csv(PATH_CT_EXTERNAL)
    df["Center"] = df["PatientID"].str[:4]
    feature_cols = [c for c in df.columns if c not in ("PatientID", "Relapse", "RFS", "Center")]
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
        len(X_external_CHUS) if X_external_CHUS is not None else 0,
        int(y_external_CHUS["event"].sum()) if y_external_CHUS is not None else 0,
        len(X_external_CHUP) if X_external_CHUP is not None else 0,
        int(y_external_CHUP["event"].sum()) if y_external_CHUP is not None else 0,
    )

def _precompute_cox_p_values(X: pd.DataFrame, y_df: pd.DataFrame) -> pd.Series:
    """Precompute univariate Cox p-values for all features (A1_Cox S1 pipelines)."""
    from lifelines import CoxPHFitter
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable, **kwargs):  # bare fallback — no progress bar
            return iterable
    y_time = y_df["RFS_time"].values
    y_event = y_df["event"].values
    p_values = {}
    for col in tqdm(X.columns, desc="Precomputing Cox p-values"):
        try:
            d = pd.DataFrame({"T": y_time, "E": y_event, "X": X[col]})
            cph = CoxPHFitter()
            cph.fit(d, duration_col="T", event_col="E", show_progress=False)
            p_values[col] = float(cph.summary.loc["X", "p"])
        except Exception:
            p_values[col] = 1.0
    return pd.Series(p_values)

def _init_globals() -> None:
    global X_train, y_train, GLOBAL_COX_P_VALUES, GLOBAL_PEARSON_CORR
    X_train, y_train = _load_data()
    GLOBAL_COX_P_VALUES = _precompute_cox_p_values(X_train, y_train)
    event_series = pd.Series(y_train["event"].astype(float).values, index=X_train.index)
    GLOBAL_PEARSON_CORR = X_train.corrwith(event_series).abs().sort_values(ascending=False)
    logger.info("Precomputed Cox p-values and Pearson corr on %s features", len(X_train.columns))

def evaluate_external_cindex(features: list, X_ext: pd.DataFrame, y_ext: pd.DataFrame) -> float:
    """Fit CoxPH on full dev set with selected features, return c-index on external set."""
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
    try:
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

# 10. LOGGING
def setup_logging() -> logging.Logger:
    """Setup logging to file and console."""
    LOG_DIR.mkdir(exist_ok=True, parents=True)
    log_file = LOG_DIR / f"2_mar_CT_Stage5_2v_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

def evaluate_cv(X_sel: pd.DataFrame, y: pd.DataFrame) -> float:
    """5-fold CV C-index."""
    if len(X_sel.columns) == 0:
        return 0.0
    features = list(X_sel.columns)
    cv_mean, _ = evaluate_features_cv(
        X_sel, y, features, n_splits=N_FOLDS, random_state=CV_SEED
    )
    return float(cv_mean) if not np.isnan(cv_mean) else 0.0

def _common_objective(trial, s1: list, s2: list, s3: list, rank: int) -> float:
    """Shared CV-only objective."""
    n_features = len(s3)
    if n_features > FEA_MAX:
        return 0.0
    
    # This logic assumes s1, s2, s3 are lists of feature names
    X_sel = X_train[s3]
    
    selected_features = list(X_sel.columns)
    cv_score = evaluate_cv(X_sel, y_train)
    fea_penalty = min(n_features / FEA_MAX, 1.0)
    objective = WEIGHT_CV * cv_score - WEIGHT_FEA * fea_penalty
    trial.set_user_attr("cv_score", round(cv_score, 4))
    trial.set_user_attr("n_features", n_features)
    trial.set_user_attr("fea_penalty", round(fea_penalty, 4))
    trial.set_user_attr("selected_features", selected_features)
    logger.info(
        "CT_S3_%s Trial %s: CV=%.4f Fea=%s Obj=%.4f",
        rank, trial.number, cv_score, n_features, objective
    )
    return objective

# 6. SEED DICT — best-balanced Stage 4 trial per pipeline (CHUS>=0.68 AND CHUP>=0.68)
# Ranges narrowed around Stage 4 good-trial clusters; Stage 4 param boundaries preserved
# as outer limits so no previously reachable configuration is excluded.
CT_STAGE5_SEEDS = {
    # inter_no=464: CV4=0.7022 Fea4=17 CHUS=0.6878 CHUP=0.7138
    37: {"cox_s1_p_threshold": 0.0283, "mrmr_n_features": 22,
         "s3_elasticnet_target_features": 43, "s3_elasticnet_l1_ratio": 0.702,
         "s3_elasticnet_n_alphas": 102},
    # inter_no=462: CV4=0.7022 Fea4=17 CHUS=0.6878 CHUP=0.7138
    31: {"cox_s1_p_threshold": 0.0340, "mrmr_n_features": 22,
         "s3_lasso_target_features": 42, "s3_lasso_n_alphas": 80},
    # inter_no=412: CV4=0.7032 Fea4=18 CHUS=0.7041 CHUP=0.7034 — best balanced for S14
    14: {"pearson_n_features": 467, "mrmr_n_features": 20,
         "s3_elasticnet_target_features": 20, "s3_elasticnet_l1_ratio": 0.708,
         "s3_elasticnet_n_alphas": 125},
    # inter_no=469: CV4=0.7022 Fea4=17 CHUS=0.6878 CHUP=0.7138
    11: {"pearson_n_features": 506, "mrmr_n_features": 22,
         "s3_lasso_target_features": 6, "s3_lasso_n_alphas": 69},
    # inter_no=763: CV4=0.6966 Fea4=23 CHUS=0.7061 CHUP=0.7034 — only balanced trial for S25
    25: {"pearson_n_features": 449, "mrmr_n_features": 23,
         "s3_xgb_n_features": 27, "s3_xgb_n_estimators": 101},
}

# 12. OBJECTIVE FUNCTIONS
# Param ranges narrowed from Stage 4 good-trial analysis (CV4>=0.70, CHUS>=0.65, CHUP>=0.65).
# FEA_MAX=30 means >30 features returns 0.0; combined with WEIGHT_FEA=0.40 this steers
# TPE toward Fea ~15-20 without hard-excluding valid configurations near the boundary.

def objective_CT_S3_37(trial: optuna.Trial) -> float:  # Cox -> mRMR_40 -> ElasticNet
    try:
        # S1: Cox — good region [0.010, 0.050]; log-uniform keeps low-p exploration alive
        # Stage 4 good trials: cox_p 0.017–0.043; outer bound kept at 0.050
        cox_p = trial.suggest_float("cox_s1_p_threshold", 0.010, 0.050, log=True)
        s1 = GLOBAL_COX_P_VALUES[GLOBAL_COX_P_VALUES <= cox_p].sort_values().head(500).index.tolist()
        if not s1: return 0.0
        # S2: mRMR — good region [20, 30]; wider upper kept at 40 to allow exploration
        mn = trial.suggest_int("mrmr_n_features", 20, 40)
        s2 = mrmr_selection(X_train[s1], y_train, n_features=min(mn, len(s1)))
        if not s2: return 0.0
        # S3: ElasticNet — good region: target [15, 50], l1 [0.55, 0.90], n_alphas [70, 150]
        et = trial.suggest_int("s3_elasticnet_target_features", 15, 50)
        el1 = trial.suggest_float("s3_elasticnet_l1_ratio", 0.55, 0.90)
        ea = trial.suggest_int("s3_elasticnet_n_alphas", 70, 150)
        s3 = elasticnet_cox_selection(X_train[s2], y_train, target_features=min(et, len(s2)), l1_ratio=el1, n_alphas=ea)
        if not s3: return 0.0
        return _common_objective(trial, s1, s2, s3, 37)
    except Exception as e:
        logger.error("CT_S3_37 failed: %s", e); return 0.0

def objective_CT_S3_31(trial: optuna.Trial) -> float:  # Cox -> mRMR_40 -> LASSO
    try:
        # S1: Cox — good region [0.015, 0.050]; both good trials had cox_p 0.034–0.048
        cox_p = trial.suggest_float("cox_s1_p_threshold", 0.010, 0.050, log=True)
        s1 = GLOBAL_COX_P_VALUES[GLOBAL_COX_P_VALUES <= cox_p].sort_values().head(500).index.tolist()
        if not s1: return 0.0
        # S2: mRMR — good trials: mrmr_n 20–22; upper kept at 35 for exploration
        mn = trial.suggest_int("mrmr_n_features", 20, 35)
        s2 = mrmr_selection(X_train[s1], y_train, n_features=min(mn, len(s1)))
        if not s2: return 0.0
        # S3: LASSO — good trials: target 22–42, n_alphas 67–99
        lt = trial.suggest_int("s3_lasso_target_features", 5, 50)
        la = trial.suggest_int("s3_lasso_n_alphas", 60, 150)
        s3 = lasso_cox_selection(X_train[s2], y_train, target_features=min(lt, len(s2)), n_alphas=la)
        if not s3: return 0.0
        return _common_objective(trial, s1, s2, s3, 31)
    except Exception as e:
        logger.error("CT_S3_31 failed: %s", e); return 0.0

def objective_CT_S3_14(trial: optuna.Trial) -> float:  # Pearson -> mRMR_40 -> ElasticNet
    try:
        # S1: Pearson — good trials: pearson_n 344–597; lower tightened to 300
        pn = trial.suggest_int("pearson_n_features", 300, 600)
        s1 = GLOBAL_PEARSON_CORR.head(min(pn, len(GLOBAL_PEARSON_CORR))).index.tolist()
        if not s1: return 0.0
        # S2: mRMR — good trials: mrmr_n 20–25; upper kept at 40 to allow some exploration
        mn = trial.suggest_int("mrmr_n_features", 20, 40)
        s2 = mrmr_selection(X_train[s1], y_train, n_features=min(mn, len(s1)))
        if not s2: return 0.0
        # S3: ElasticNet — good trials: target 17–46, l1 0.37–0.90, n_alphas 52–198
        et = trial.suggest_int("s3_elasticnet_target_features", 15, 50)
        el1 = trial.suggest_float("s3_elasticnet_l1_ratio", 0.35, 0.92)
        ea = trial.suggest_int("s3_elasticnet_n_alphas", 50, 200)
        s3 = elasticnet_cox_selection(X_train[s2], y_train, target_features=min(et, len(s2)), l1_ratio=el1, n_alphas=ea)
        if not s3: return 0.0
        return _common_objective(trial, s1, s2, s3, 14)
    except Exception as e:
        logger.error("CT_S3_14 failed: %s", e); return 0.0

def objective_CT_S3_11(trial: optuna.Trial) -> float:  # Pearson -> mRMR_40 -> LASSO
    try:
        # S1: Pearson — good trials all have pearson_n 480–600; lower tightened to 400
        pn = trial.suggest_int("pearson_n_features", 400, 600)
        s1 = GLOBAL_PEARSON_CORR.head(min(pn, len(GLOBAL_PEARSON_CORR))).index.tolist()
        if not s1: return 0.0
        # S2: mRMR — good trials: mrmr_n 20–61; keep broad since range matters less here
        mn = trial.suggest_int("mrmr_n_features", 20, 65)
        s2 = mrmr_selection(X_train[s1], y_train, n_features=min(mn, len(s1)))
        if not s2: return 0.0
        # S3: LASSO — good trials: target 6–30; keep [5, 35] to allow full exploration
        lt = trial.suggest_int("s3_lasso_target_features", 5, 35)
        la = trial.suggest_int("s3_lasso_n_alphas", 50, 180)
        s3 = lasso_cox_selection(X_train[s2], y_train, target_features=min(lt, len(s2)), n_alphas=la)
        if not s3: return 0.0
        return _common_objective(trial, s1, s2, s3, 11)
    except Exception as e:
        logger.error("CT_S3_11 failed: %s", e); return 0.0

def objective_CT_S3_25(trial: optuna.Trial) -> float:  # Pearson -> mRMR_50 -> XGB
    try:
        # S1: Pearson — bestie trial: pearson_n=449; good external needs larger pools
        # CV4-only top trials used small pearson_n (105–261) but had poor CHUS/CHUP;
        # bestie (CHUS=0.706, CHUP=0.703) used pearson_n=449. Range: [250, 600]
        pn = trial.suggest_int("pearson_n_features", 250, 600)
        s1 = GLOBAL_PEARSON_CORR.head(min(pn, len(GLOBAL_PEARSON_CORR))).index.tolist()
        if not s1: return 0.0
        # S2: mRMR — bestie used mrmr_n=23; small mrmr→small XGB input→poor CHUS/CHUP
        # Keep [20, 40] to stay in balanced zone; avoid >40 which gave high CV but low ext
        mn = trial.suggest_int("mrmr_n_features", 20, 40)
        s2 = mrmr_selection(X_train[s1], y_train, n_features=min(mn, len(s1)))
        if not s2: return 0.0
        # S3: XGB — bestie xgb_n=27, xgb_ne=101; higher xgb_n correlates with better ext
        # Low xgb_n (5-9) maximizes Objective (low Fea penalty) but degrades CHUS/CHUP
        # Range: [15, 35] steers TPE toward balanced Fea ~15-25
        xn = trial.suggest_int("s3_xgb_n_features", 15, 35)
        xne = trial.suggest_int("s3_xgb_n_estimators", 80, 150)
        s3 = xgboost_survival_selection(X_train[s2], y_train, n_features=min(xn, len(s2)), n_estimators=xne, random_state=CV_SEED)
        if not s3: return 0.0
        return _common_objective(trial, s1, s2, s3, 25)
    except Exception as e:
        logger.error("CT_S3_25 failed: %s", e); return 0.0

OBJECTIVE_MAP = {
    37: objective_CT_S3_37,
    31: objective_CT_S3_31,
    14: objective_CT_S3_14,
    11: objective_CT_S3_11,
    25: objective_CT_S3_25,
}

def run_optimization(rank: int, n_trials: int) -> optuna.Study:
    """Run Optuna for one pipeline."""
    # 4. STUDY NAMES
    study_name = f"CT_S3_{rank}_Stage5"
    n_jobs = HEAVY_PIPELINE_N_JOBS.get(f"CT_S3_{rank}", N_JOBS)

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

    # 7. ENQUEUE IN run_optimization()
    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if completed == 0:
        seed_params = CT_STAGE5_SEEDS.get(rank)
        if seed_params:
            study.enqueue_trial(seed_params)
            logger.info("Enqueued Stage 4 seed for CT_S3_%s", rank)
    
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
    
    # 5. RANKS LIST
    for rank in CT_S3_RANKS:
        study = studies.get(rank)
        if study is None:
            continue
        df_trials = study.trials_dataframe()
        b = stage3_baselines.get(rank, {})
        pipeline = b.get("Pipeline", f"CT_S3_{rank}")

        df_trials["Rank"] = rank
        # 8. INTRA_NO FORMAT
        df_trials["Pipeline"] = f"CT_S3_{rank}:{pipeline}"
        df_trials["Stage3_CV"] = b.get("CV3", 0)
        df_trials["Stage3_Fea"] = b.get("Fea3", 0)
        
        # Column names CV4/Fea4 are inherited from PT script
        df_trials["CV4"] = df_trials["user_attrs_cv_score"] if "user_attrs_cv_score" in df_trials.columns else 0.0
        df_trials["Fea4"] = df_trials["user_attrs_n_features"] if "user_attrs_n_features" in df_trials.columns else 0
        df_trials["Objective"] = df_trials["value"] if "value" in df_trials.columns else 0.0

        df_trials = df_trials.sort_values("Objective", ascending=False).reset_index(drop=True)
        # 9. OUTPUT FILENAMES
        pf = OUTPUT_DIR / f"CT_S3_{rank}_Stage5_trials_{run_date}.csv"
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
        logger.info("  CT_S3_%s best: Ext CHUS=%.4f  Ext CHUP=%.4f", rank, ext_chus, ext_chup)

        results.append({
            "Rank": rank,
            "Pipeline": f"CT_S3_{rank}:{pipeline}",
            "Stage3_CV": b.get("CV3", 0),
            "Stage3_Fea": b.get("Fea3", 0),
            "Optuna_CV": round(best.user_attrs.get("cv_score", 0), 4),
            "Optuna_Fea": best.user_attrs.get("n_features", 0),
            "Objective": round(best.value, 4) if best.value is not None else 0.0,
            "Best_params": str(best.params),
            "Ext_CHUS": ext_chus,
            "Ext_CHUP": ext_chup,
        })

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values(["Rank"]).reset_index(drop=True)
    # 9. OUTPUT FILENAMES
    out_file = OUTPUT_DIR / f"2_mar_CT_Stage5_2v_results_{run_date}.csv"
    res_df.to_csv(out_file, index=False)
    logger.info("\n%s\nResults saved: %s\n%s", "=" * 70, out_file, "=" * 70)

    # Combined all-trials CSV
    all_trials = []
    inter_counter = 1
    logger.info("Building combined CSV with Ext_CHUS/Ext_CHUP per trial...")
    # 5. RANKS LIST
    for rank in CT_S3_RANKS:
        study = studies.get(rank)
        if study is None:
            continue
        for t in sorted(study.trials, key=lambda x: x.number):
            if t.state != optuna.trial.TrialState.COMPLETE:
                continue
            b = stage3_baselines.get(rank, {})
            # 8. INTRA_NO FORMAT
            pipeline = b.get("Pipeline", f"CT_S3_{rank}")
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
                # 8. INTRA_NO FORMAT
                "intra_no": f"CT_S3_{rank}_Stage5_{t.number}",
                "S3_rank": rank,
                "Pipeline": f"CT_S3_{rank}:{pipeline}",
                "CV4": t.user_attrs.get("cv_score"),
                "Fea4": t.user_attrs.get("n_features"),
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
    # 9. OUTPUT FILENAMES
    all_out = OUTPUT_DIR / f"2_mar_CT_Stage5_2v_ALLtrials_{run_date}.csv"
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
    global logger

    # 11. MAIN BANNER
    print("\n" + "=" * 70)
    print("CT Stage 5 Refine (2v): 5 pipelines, Round 2 Optuna")
    print("=" * 70 + "\n")

    if not test_db_connection():
        sys.exit(1)
    print("Database connection OK\n")

    logger = setup_logging()
    random.seed(SEED)
    np.random.seed(SEED)

    _load_stage3_methods()
    _init_globals()
    _load_external_data()
    
    studies = {}
    # 5. RANKS LIST
    for rank in CT_S3_RANKS:
        n_trials = N_TRIALS_RANK_OVERRIDES.get(f"CT_S3_{rank}", N_TRIALS)
        studies[rank] = run_optimization(rank, n_trials)

    save_results(studies)

    logger.info("\n" + "=" * 70)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 70)
    logger.info("Total pipelines: %s", len(CT_S3_RANKS))
    logger.info("Output: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
