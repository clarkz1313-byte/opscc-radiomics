# -*- coding: utf-8 -*-
"""
CT Stage 4 Optuna: 19 Pipelines, Full-Chain Tuning, CV-Only (2v)

Per Mar_2026/27_feb_CT_PT_Stage4_Optuna_plan.md (Part II, lines 259-597):
- 19 pipelines: 9 Performance, 5 Parsimony, 5 Exploratory
- Full-chain tuning: every S1/S2/S3 step has trial.suggest_* params
- CV-only objective: WEIGHT_CV * CV - WEIGHT_FEA * Fea_penalty (no Test term)
- Cox precompute mandatory (S1-Cox pipelines); Pearson precompute (A2_Pearson S1)
- Mid-chain Cox (CT_S3_28, 29, 41): fresh univariate_cox_selection on S1 output
- HEAVY_PIPELINE_N_JOBS: D3_PermImp and D1_RSF use n_jobs=1
- External validation: CHUS and CHUP from 27_feb_CT_external.csv (Ext_CHUS, Ext_CHUP in all exports)

Data: Mar_2026/27_feb_CT_development.csv
Stage 3: Mar_2026/27_feb_CT_Stage3_2v_Processed_result.csv (67 rows, Cox merged)

Run: cd "D:/Uppsala thesis" && python Mar_2026/28_feb_CT_Stage4_2v.py
"""

from __future__ import annotations

import argparse
import joblib
import logging
import random
import sys
import warnings
from datetime import datetime
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

import numpy as np
import pandas as pd
import optuna

# Check PostgreSQL
try:
    import psycopg2
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fs_utils import (
    evaluate_features_cv,
    univariate_cox_selection,
    mrmr_selection,
    lasso_cox_selection,
    elasticnet_cox_selection,
    xgboost_survival_selection,
    permutation_importance_survival,
    rsf_permutation_importance,
)

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
OUTPUT_DIR = SCRIPT_DIR / "27_feb_CT_Stage4_2v_outputs"
LOG_DIR = SCRIPT_DIR / "27_feb_CT_Stage4_2v_outputs"
CHECKPOINT_DIR = SCRIPT_DIR / "27_feb_CT_Stage4_2v_outputs"

# Data
PATH_CT_DEV = SCRIPT_DIR / "27_feb_CT_development.csv"
PATH_STAGE3 = SCRIPT_DIR / "27_feb_CT_Stage3_2v_Processed_result.csv"
PATH_CT_EXTERNAL = SCRIPT_DIR / "27_feb_CT_external.csv"

logger = None

# -----------------------------------------------------------------------------
# PostgreSQL (same as PT script)
# -----------------------------------------------------------------------------
DB_HOST = "localhost"
DB_PORT = 5432
DB_USER = "postgres"
DB_PASSWORD = "1730"
DB_NAME = "optuna_db"
DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
STORAGE_TIMEOUT = 600

SEED = 42

# -----------------------------------------------------------------------------
# Config (from plan lines 259-597)
# -----------------------------------------------------------------------------
N_FOLDS = 5
CV_SEED = 42
FEA_MAX = 50
WEIGHT_CV = 0.65
WEIGHT_FEA = 0.35
N_TRIALS_DEFAULT = 850
N_TRIALS_LOW = 300   # CT_S3_12, 21 (D1_RSF; slow at n_jobs=1)
N_TRIALS_MED = 500   # CT_S3_8, 9, 34, 1 (D3_PermImp; slow at n_jobs=1)
N_TRIALS_PARS_EXPLOR = 600

N_JOBS = 8  # Match N_JOBS = 12  # i9-12900K: 8 P-cores × 2 HT = 16 threads; 12 leaves headroom

HEAVY_PIPELINE_N_JOBS = 1
# D3_PermImp and D1_RSF use n_jobs=1 to prevent deadlock (same issue as PT_S3_32).
# XGBoost (D2) does NOT deadlock and runs at full N_JOBS.
HEAVY_RANKS = frozenset({1, 8, 9, 12, 21, 31, 34})

PIPELINES = {
    3: {"perf": 1}, 5: {"perf": 1}, 7: {"perf": 1}, 8: {"perf": 1},
    9: {"perf": 1}, 10: {"perf": 1}, 12: {"perf": 1}, 13: {"perf": 1},
    14: {"perf": 1},
    21: {"pars": 1}, 28: {"pars": 1}, 29: {"pars": 1},
    34: {"pars": 1}, 41: {"pars": 1},
    1: {"explor": 1}, 2: {"explor": 1}, 4: {"explor": 1},
    31: {"explor": 1}, 37: {"explor": 1},
    # Supplementary: post-hoc inclusions from Stage 4 alt-run (see plan CT.1.4)
    11: {"supp_perf": 1},   # A2_Pearson -> B3_mRMR_40 -> C1_LASSO; CV3=0.7100; strong ext. validation
    25: {"supp_pars": 1},   # A2_Pearson -> B3_mRMR_50 -> D2_XGBoost_50; unique A-B-D; CHUS/CHUP > 0.70
}

PERF_RANKS = frozenset({3, 5, 7, 8, 9, 10, 12, 13, 14})
PARS_RANKS = frozenset({21, 28, 29, 34, 41})
EXPLOR_RANKS = frozenset({1, 2, 4, 31, 37})
SUPP_RANKS = frozenset({11, 25})  # supplementary; included in output but not original plan groups

# S1/S2/S3 method IDs from Stage 3 CSV
STAGE3_METHODS = {}


def _parse_pipeline(pipeline_str: str) -> tuple[str, str, str]:
    """Parse 'S1 -> S2 -> S3' into (S1, S2, S3) tokens."""
    parts = [p.strip() for p in str(pipeline_str).split("->")]
    if len(parts) != 3:
        return ("", "", "")
    return (parts[0], parts[1], parts[2])


def _load_stage3_methods() -> None:
    global STAGE3_METHODS
    df = pd.read_csv(PATH_STAGE3)
    for _, r in df.iterrows():
        rank_str = str(r.get("Rank", ""))
        if not rank_str.startswith("CT_S3_"):
            continue
        try:
            rank = int(rank_str.replace("CT_S3_", ""))
        except ValueError:
            continue
        pipeline_str = str(r.get("Pipeline", ""))
        s1, s2, s3 = _parse_pipeline(pipeline_str)
        STAGE3_METHODS[rank] = {
            "S1": s1, "S2": s2, "S3": s3, "Pipeline": pipeline_str,
            "CV3": float(r.get("CV3", 0)), "Fea3": int(r.get("Fea3", 0)),
        }
    logger.info("Loaded %s Stage 3 pipelines", len(STAGE3_METHODS))


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


def _precompute_cox_p_values(X: pd.DataFrame, y_df: pd.DataFrame) -> pd.Series:
    """Precompute univariate Cox p-values for all features (A1_Cox S1 pipelines)."""
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
            d = pd.DataFrame({"T": y_time, "E": y_event, "X": X[col]})
            cph = CoxPHFitter()
            cph.fit(d, duration_col="T", event_col="E", show_progress=False)
            p_values[col] = float(cph.summary.loc["X", "p"])
        except Exception:
            p_values[col] = 1.0
    return pd.Series(p_values)


# -----------------------------------------------------------------------------
# Global precomputes (loaded once)
# -----------------------------------------------------------------------------
X_train: pd.DataFrame
y_train: pd.DataFrame
GLOBAL_COX_P_VALUES: pd.Series
GLOBAL_PEARSON_CORR: pd.Series
X_external_CHUS: pd.DataFrame | None = None
X_external_CHUP: pd.DataFrame | None = None
y_external_CHUS: pd.DataFrame | None = None
y_external_CHUP: pd.DataFrame | None = None


def _load_external_data() -> None:
    """Load CT external CSV, split into CHUS and CHUP subsets. Same logic as PT."""
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


def _evaluate_external_cindex(
    features: list[str],
    X_ext: pd.DataFrame | None,
    y_ext: pd.DataFrame | None,
) -> float:
    """Fit CoxPH on full dev set with selected features, return c-index on external set. Same as PT."""
    try:
        from lifelines import CoxPHFitter
        from lifelines.utils import concordance_index
    except ImportError:
        return float("nan")
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
        ci = concordance_index(
            y_ext["RFS_time"].values, -risk.values, y_ext["event"].values
        )
        return round(float(ci), 4)
    except Exception:
        return float("nan")


def _init_globals() -> None:
    global X_train, y_train, GLOBAL_COX_P_VALUES, GLOBAL_PEARSON_CORR
    X_train, y_train = _load_data()
    GLOBAL_COX_P_VALUES = _precompute_cox_p_values(X_train, y_train)
    event_series = pd.Series(y_train["event"].astype(float).values, index=X_train.index)
    GLOBAL_PEARSON_CORR = X_train.corrwith(event_series).abs().sort_values(ascending=False)
    logger.info("Precomputed Cox p-values and Pearson corr on %s features", len(X_train.columns))


def evaluate_cv(X_sel: pd.DataFrame, y: pd.DataFrame) -> float:
    if len(X_sel.columns) == 0:
        return 0.0
    features = list(X_sel.columns)
    cv_mean, _ = evaluate_features_cv(
        X_sel, y, features, n_splits=N_FOLDS, random_state=CV_SEED
    )
    return float(cv_mean) if not np.isnan(cv_mean) else 0.0


def _common_objective(trial, s1: list, s2: list, s3: list, rank: int, fea_max: int) -> float:
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
        "CT_S3_%s Trial %s: CV=%.4f Fea=%s Obj=%.4f",
        rank, trial.number, cv_score, n_features, objective
    )
    return objective


# -----------------------------------------------------------------------------
# S1 samplers (match Stage 3 Pipeline tokens: A2_Pearson, A1_Cox*, B3_mRMR*, C1, C2)
# -----------------------------------------------------------------------------
def _s1_sampler(trial, rank: int) -> list[str]:
    m = STAGE3_METHODS.get(rank, {})
    s1 = str(m.get("S1", ""))
    if "A2_Pearson" in s1:
        pn = trial.suggest_int("pearson_n_features", 100, 600)
        return GLOBAL_PEARSON_CORR.head(min(pn, len(GLOBAL_PEARSON_CORR))).index.tolist()
    if "A1_Cox" in s1:
        cox_p = trial.suggest_float("cox_s1_p_threshold", 0.0005, 0.05, log=True)
        passed = GLOBAL_COX_P_VALUES[GLOBAL_COX_P_VALUES <= cox_p].sort_values()
        return passed.head(500).index.tolist()
    if "B3_mRMR" in s1:
        mn = trial.suggest_int("mrmr_n_features", 20, 80)
        return mrmr_selection(X_train, y_train, n_features=min(mn, X_train.shape[1]))
    if "C1_LASSO" in s1:
        lt = trial.suggest_int("lasso_target_features", 50, 150)
        la = trial.suggest_int("lasso_n_alphas", 50, 200)
        return lasso_cox_selection(X_train, y_train, target_features=lt, n_alphas=la)
    if "C2_ElasticNet" in s1:
        et = trial.suggest_int("elasticnet_target_features", 50, 150)
        el1 = trial.suggest_float("elasticnet_l1_ratio", 0.3, 0.9)
        ea = trial.suggest_int("elasticnet_n_alphas", 50, 200)
        return elasticnet_cox_selection(X_train, y_train, target_features=et, l1_ratio=el1, n_alphas=ea)
    return list(X_train.columns)


# -----------------------------------------------------------------------------
# S2 samplers (B3_mRMR, A1_Cox mid-chain, C1_LASSO, C2_ElasticNet, D2_XGBoost, D3_PermImp, D1_RSF)
# -----------------------------------------------------------------------------
def _s2_sampler(trial, s1: list[str], rank: int) -> list[str]:
    m = STAGE3_METHODS.get(rank, {})
    s2 = str(m.get("S2", ""))
    X_s1 = X_train[s1]
    if not s2:
        return s1
    if "B3_mRMR" in s2:
        mn = trial.suggest_int("mrmr_n_features", 20, 80)
        return mrmr_selection(X_s1, y_train, n_features=min(mn, len(s1)))
    if "A1_Cox" in s2:
        cox_p = trial.suggest_float("cox_mid_p_threshold", 0.0005, 0.05, log=True)
        return univariate_cox_selection(X_s1, y_train, p_threshold=cox_p)
    if "C1_LASSO" in s2:
        lt = trial.suggest_int("lasso_target_features", 50, 150)
        la = trial.suggest_int("lasso_n_alphas", 50, 200)
        return lasso_cox_selection(X_s1, y_train, target_features=lt, n_alphas=la)
    if "C2_ElasticNet" in s2:
        et = trial.suggest_int("elasticnet_target_features", 50, 150)
        el1 = trial.suggest_float("elasticnet_l1_ratio", 0.3, 0.9)
        ea = trial.suggest_int("elasticnet_n_alphas", 50, 200)
        return elasticnet_cox_selection(X_s1, y_train, target_features=et, l1_ratio=el1, n_alphas=ea)
    if "D2_XGBoost" in s2:
        xn = trial.suggest_int("xgb_n_features", 30, 70)
        xne = trial.suggest_int("xgb_n_estimators", 80, 150)
        return xgboost_survival_selection(
            X_s1, y_train, n_features=min(xn, len(s1)),
            n_estimators=xne, random_state=CV_SEED
        )
    if "D3_PermImp" in s2:
        pn = trial.suggest_int("permimp_n_features", 30, 70)
        pne = trial.suggest_int("permimp_n_estimators", 300, 600)
        return permutation_importance_survival(
            X_s1, y_train, n_features=min(pn, len(s1)),
            n_estimators=pne, random_state=CV_SEED
        )
    if "D1_RSF" in s2:
        rn = trial.suggest_int("rsf_n_features", 30, 80)
        rne = trial.suggest_int("rsf_n_estimators", 200, 600)
        return rsf_permutation_importance(
            X_s1, y_train, n_features=min(rn, len(s1)),
            n_estimators=rne, random_state=CV_SEED
        )
    return s1


# -----------------------------------------------------------------------------
# S3 samplers (C1_LASSO, C2_ElasticNet, D2_XGBoost for CT Stage 4 pipelines)
# -----------------------------------------------------------------------------
def _s3_sampler(trial, s2: list[str], rank: int) -> list[str]:
    m = STAGE3_METHODS.get(rank, {})
    s3 = str(m.get("S3", ""))
    X_s2 = X_train[s2]
    if not s3:
        return s2
    if "C1_LASSO" in s3:
        lt = trial.suggest_int("s3_lasso_target_features", 5, 50)
        la = trial.suggest_int("s3_lasso_n_alphas", 50, 200)
        return lasso_cox_selection(
            X_s2, y_train, target_features=min(lt, len(s2)), n_alphas=la
        )
    if "C2_ElasticNet" in s3:
        et = trial.suggest_int("s3_elasticnet_target_features", 5, 50)
        el1 = trial.suggest_float("s3_elasticnet_l1_ratio", 0.3, 0.9)
        ea = trial.suggest_int("s3_elasticnet_n_alphas", 50, 200)
        return elasticnet_cox_selection(
            X_s2, y_train, target_features=min(et, len(s2)),
            l1_ratio=el1, n_alphas=ea
        )
    if "D2_XGBoost" in s3:
        xn = trial.suggest_int("s3_xgb_n_features", 5, 50)
        xne = trial.suggest_int("s3_xgb_n_estimators", 80, 150)
        return xgboost_survival_selection(
            X_s2, y_train, n_features=min(xn, len(s2)),
            n_estimators=xne, random_state=CV_SEED
        )
    return s2


# -----------------------------------------------------------------------------
# Objective
# -----------------------------------------------------------------------------
def _objective(trial, rank: int, fea_max: int) -> float:
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
        return _common_objective(trial, s1, s2, s3, rank, fea_max)
    except Exception as e:
        logger.error("CT_S3_%s Trial %s failed: %s", rank, trial.number, e)
        return 0.0


# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
def _n_trials(rank: int) -> int:
    if rank in (12, 21):
        return N_TRIALS_LOW
    if rank in (8, 9, 34, 1):
        return N_TRIALS_MED
    if rank in PARS_RANKS | EXPLOR_RANKS | SUPP_RANKS:
        return N_TRIALS_PARS_EXPLOR
    return N_TRIALS_DEFAULT


def run_optimization(rank: int, n_trials: int) -> optuna.Study:
    """Run Optuna for one pipeline. Mirrors PT run_optimization()."""
    study_name = f"CT_S3_{rank}_Stage4"
    n_jobs = HEAVY_PIPELINE_N_JOBS if rank in HEAVY_RANKS else N_JOBS

    logger.info("\n%s", "=" * 70)
    logger.info("Starting: %s", study_name)
    if rank in STAGE3_METHODS:
        b = STAGE3_METHODS[rank]
        logger.info("Stage 3 baseline: CV3=%.4f Fea3=%s", b["CV3"], b["Fea3"])
        logger.info("Pipeline: %s", b["Pipeline"])
    logger.info(
        "Trials: %s n_jobs=%s | Objective: %.2f*CV - %.2f*Fea",
        n_trials, n_jobs, WEIGHT_CV, WEIGHT_FEA,
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
            lambda t, r=rank: _objective(t, r, FEA_MAX),
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


def save_results(studies: dict[int, optuna.Study]) -> None:
    """Save per-pipeline trial CSV, summary CSV, and ALLtrials CSV. Mirrors PT save_results()."""
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    run_date = datetime.now().strftime("%Y%m%d")
    results = []
    all_ranks = sorted(PIPELINES.keys())

    for rank in all_ranks:
        study = studies.get(rank)
        if study is None:
            continue

        # Per-pipeline trials CSV (all trials sorted by Objective)
        df_trials = study.trials_dataframe()
        b = STAGE3_METHODS.get(rank, {})
        pipeline = b.get("Pipeline", f"CT_S3_{rank}")

        df_trials["Rank"] = rank
        df_trials["Pipeline"] = f"CT_S3_{rank}:{pipeline}"
        df_trials["Stage3_CV"] = b.get("CV3", 0)
        df_trials["Stage3_Fea"] = b.get("Fea3", 0)
        df_trials["CV"] = df_trials["user_attrs_cv_score"] if "user_attrs_cv_score" in df_trials.columns else 0.0
        df_trials["Fea"] = df_trials["user_attrs_n_features"] if "user_attrs_n_features" in df_trials.columns else 0
        df_trials["Objective"] = df_trials["value"] if "value" in df_trials.columns else 0.0

        df_trials = df_trials.sort_values("Objective", ascending=False).reset_index(drop=True)
        pf = OUTPUT_DIR / f"CT_S3_{rank}_trials_{run_date}.csv"
        df_trials.to_csv(pf, index=False)
        logger.info("Trials saved: %s", pf)

        # Best trial for summary
        best = study.best_trial
        features = best.user_attrs.get("selected_features", [])
        ext_chus = float("nan")
        ext_chup = float("nan")
        if features:
            if X_external_CHUS is not None:
                ext_chus = _evaluate_external_cindex(features, X_external_CHUS, y_external_CHUS)
            if X_external_CHUP is not None:
                ext_chup = _evaluate_external_cindex(features, X_external_CHUP, y_external_CHUP)
        logger.info("  CT_S3_%s best: CV=%.4f Fea=%s Obj=%.4f Ext_CHUS=%.4f Ext_CHUP=%.4f",
                    rank,
                    best.user_attrs.get("cv_score", 0),
                    best.user_attrs.get("n_features", 0),
                    best.value, ext_chus, ext_chup)

        results.append({
            "Rank": rank,
            "Pipeline": f"CT_S3_{rank}:{pipeline}",
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
    res_df = res_df.sort_values("Rank").reset_index(drop=True)
    out_file = OUTPUT_DIR / f"28_feb_CT_Stage4_2v_results_{run_date}.csv"
    res_df.to_csv(out_file, index=False)
    logger.info("\n%s\nResults saved: %s\n%s", "=" * 70, out_file, "=" * 70)

    # ALLtrials CSV with inter_no, intra_no, Ext_CHUS, Ext_CHUP per trial
    all_trials = []
    inter_counter = 1
    logger.info("Building combined ALLtrials CSV with Ext_CHUS/Ext_CHUP per trial...")
    for rank in all_ranks:
        study = studies.get(rank)
        if study is None:
            continue
        b = STAGE3_METHODS.get(rank, {})
        pipeline = b.get("Pipeline", f"CT_S3_{rank}")
        for t in sorted(study.trials, key=lambda x: x.number):
            if t.state != optuna.trial.TrialState.COMPLETE:
                continue
            features = t.user_attrs.get("selected_features", [])
            ext_chus = float("nan")
            ext_chup = float("nan")
            if features:
                if X_external_CHUS is not None:
                    ext_chus = _evaluate_external_cindex(features, X_external_CHUS, y_external_CHUS)
                if X_external_CHUP is not None:
                    ext_chup = _evaluate_external_cindex(features, X_external_CHUP, y_external_CHUP)
            row = {
                "inter_no": inter_counter,
                "intra_no": f"CT_S3_{rank}_{t.number}",
                "S3_rank": rank,
                "Pipeline": f"CT_S3_{rank}:{pipeline}",
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
    all_out = OUTPUT_DIR / f"28_feb_CT_Stage4_2v_ALLtrials_{run_date}.csv"
    df_all.to_csv(all_out, index=False)
    logger.info("All trials combined CSV saved: %s (%s rows)", all_out, len(df_all))


def setup_logging() -> logging.Logger:
    """Setup logging to file and console (same as PT)."""
    LOG_DIR.mkdir(exist_ok=True, parents=True)
    log_file = LOG_DIR / f"28_feb_CT_Stage4_2v_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=None, help="Single pipeline rank (CT_S3_#)")
    parser.add_argument("--trials", type=int, default=None, help="Override n_trials")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("CT Stage 4 Optuna (2v): 19 pipelines, CV-only, full-chain tuning")
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
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

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

