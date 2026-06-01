# -*- coding: utf-8 -*-
"""
29_mar_t1C_dose_stage4_75.py
T1C Dose Stage 4: Optuna full-chain tuning for shortlisted Stage 3 pipelines.
Survival C-index (CoxPH), Task 1 methodology. Performance group only.
Template: 28_feb_CT_Stage4_2v.py

Usage:
    cd "D:/Uppsala thesis"
    python Mar_2026_task1C/29_mar_T1C_fs_script_results/29_mar_t1C_dose_stage4_75.py
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

try:
    import psycopg2
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

SCRIPT_DIR = Path(__file__).resolve().parent
TASK1C_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = TASK1C_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fs_utils import (
    anova_selection,
    evaluate_features_cv,
    mrmr_selection,
    permutation_importance_survival,
    rsf_permutation_importance,
    stability_selection_lasso,
    univariate_cox_selection,
)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
DATA_FILE = TASK1C_ROOT / "Dose_development_75.csv"
EXTERNAL_FILE = TASK1C_ROOT / "Dose_external_CHUS.csv"
STAGE3_CSV = SCRIPT_DIR / "Dose_stage3_result_75.csv"
OUTPUT_DIR = SCRIPT_DIR / "Dose_stage4_outputs_75"
LOG_DIR = OUTPUT_DIR
CHECKPOINT_DIR = OUTPUT_DIR

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
N_REPEATS = 3          # repeated CV to stabilize C-index on small N
FEA_MAX = 12           # keep downstream exhaustive dose sweep at <= 2^12 combinations
WEIGHT_CV = 0.55       # reduced: CV is noisy on 75 pts / 13 events
WEIGHT_STD = 0.15      # new: penalize high fold-to-fold variance (noise signature)
WEIGHT_FEA = 0.30      # increased: stronger parsimony pressure
# Post-hoc best-trial selection filter for low-feature but non-trivial dose sets.
# Trials with fewer features than this floor are excluded when picking the best
# trial per rank in save_results. Optuna optimization is unaffected.
MIN_FEA_BEST = 8
N_JOBS = 8
HEAVY_PIPELINE_N_JOBS = 1

DOSE_RANKS = [1, 5, 6, 16, 19]
# Selected N=75 Stage 3 candidates:
# S3_1:  A1_Cox -> mRMR_50 -> D3_PermImp_10
# S3_5:  A1_Cox -> mRMR_20 -> D3_PermImp_10
# S3_6:  A1_Cox -> mRMR_30 -> D1_RSF_10
# S3_16: A7_ANOVA -> mRMR_30 -> D1_RSF_20
# S3_19: A7_ANOVA -> mRMR_20 -> D3_PermImp_10
# All selected pipelines have D1/D3 as the final selector, so run each study serially.
# Reduced from 500-700: TPE exploits CV noise beyond ~150 trials on this sample size
N_TRIALS_MAP = {
    1: 50,    # HEAVY: D3_PermImp S3
    5: 50,    # HEAVY: D3_PermImp S3
    6: 50,    # HEAVY: D1_RSF S3
    16: 50,   # HEAVY: D1_RSF S3
    19: 50,   # HEAVY: D3_PermImp S3
}
HEAVY_RANKS = frozenset({1, 5, 6, 16, 19})

NON_FEATURE_COLS = {"PatientID", "CenterID", "Relapse", "RFS", "Gender_Male"}


# ---------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------
X_train: pd.DataFrame
y_train: pd.DataFrame
GLOBAL_COX_P_VALUES: pd.Series
X_external_CHUS: pd.DataFrame | None = None
y_external_CHUS: pd.DataFrame | None = None
STAGE3_METHODS: dict[int, dict] = {}


def _parse_pipeline(pipeline_str: str) -> tuple[str, str, str]:
    parts = [p.strip() for p in str(pipeline_str).split("->")]
    if len(parts) != 3:
        return "", "", ""
    return parts[0], parts[1], parts[2]


def _load_stage3_methods() -> None:
    global STAGE3_METHODS
    df = pd.read_csv(STAGE3_CSV)
    for _, row in df.iterrows():
        rank_str = str(row.get("Rank", ""))
        if not rank_str.startswith("Dose_S3_"):
            continue
        try:
            rank = int(rank_str.replace("Dose_S3_", ""))
        except ValueError:
            continue
        s1, s2, s3 = _parse_pipeline(str(row.get("Pipeline", "")))
        # B-start pipelines (e.g. B3_mRMR -> C3 -> D1) have no A-step;
        # shift slots so s1=passthrough, s2=B-step, s3=C/D-step.
        b_start = s1.startswith("B")
        if b_start:
            s1, s2, s3 = "", s1, s2
        STAGE3_METHODS[rank] = {
            "S1": s1,
            "S2": s2,
            "S3": s3,
            "b_start": b_start,
            "Pipeline": str(row.get("Pipeline", "")),
            "CV3": float(row.get("CV3", 0)),
            "Fea3": int(row.get("Fea3", 0)),
        }
    logger.info("Loaded %s Stage 3 dose pipelines", len(STAGE3_METHODS))


def _load_data() -> None:
    global X_train, y_train
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Data not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    feat_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X_train = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    X_train = X_train.fillna(X_train.median())
    y_train = pd.DataFrame(
        {
            "RFS_time": df["RFS"].values,
            "event": df["Relapse"].values.astype(bool),
        }
    )

    logger.info(
        "Train: %s patients, %s features, %s events",
        len(df),
        len(feat_cols),
        int(y_train["event"].sum()),
    )


def _load_external_data() -> None:
    global X_external_CHUS, y_external_CHUS
    if not EXTERNAL_FILE.exists():
        logger.warning("Dose external file not found: %s", EXTERNAL_FILE)
        return

    df = pd.read_csv(EXTERNAL_FILE)
    df["Center"] = df["PatientID"].astype(str).str[:4]
    df = df[df["Center"] == "CHUS"].copy()
    feat_cols = [c for c in X_train.columns if c in df.columns]
    train_medians = X_train[feat_cols].median()

    X_external_CHUS = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(train_medians)
    y_external_CHUS = pd.DataFrame(
        {
            "RFS_time": df["RFS"].values,
            "event": df["Relapse"].values.astype(bool),
        }
    )

    logger.info(
        "External CHUS loaded: %s patients, %s events",
        len(X_external_CHUS),
        int(y_external_CHUS["event"].sum()),
    )


def _precompute_cox_p_values(X: pd.DataFrame, y_df: pd.DataFrame) -> pd.Series:
    from lifelines import CoxPHFitter
    from tqdm import tqdm

    y_time = y_df["RFS_time"].values
    y_event = y_df["event"].values
    p_values = {}
    for col in tqdm(X.columns, desc="Precomputing Cox p-values"):
        try:
            frame = pd.DataFrame({"T": y_time, "E": y_event, "X": X[col]})
            cph = CoxPHFitter()
            cph.fit(frame, duration_col="T", event_col="E", show_progress=False)
            p_values[col] = float(cph.summary.loc["X", "p"])
        except Exception:
            p_values[col] = 1.0
    return pd.Series(p_values)


def _precompute_cox_p_values_global() -> None:
    global GLOBAL_COX_P_VALUES
    logger.info("Starting global Cox p-value precompute on %s features", X_train.shape[1])
    t0 = datetime.now()
    GLOBAL_COX_P_VALUES = _precompute_cox_p_values(X_train, y_train)
    dt = datetime.now() - t0
    logger.info("Finished Cox p-value precompute in %s", str(dt).split(".")[0])


def _evaluate_external_cindex(
    features: list[str],
    X_ext: pd.DataFrame | None,
    y_ext: pd.DataFrame | None,
) -> float:
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

        fit_df = X_train[avail].copy()
        fit_df["T"] = y_train["RFS_time"].values
        fit_df["E"] = y_train["event"].values.astype(int)
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(fit_df, duration_col="T", event_col="E", show_progress=False)
        risk = cph.predict_partial_hazard(X_ext[avail])
        return round(
            float(
                concordance_index(
                    y_ext["RFS_time"].values,
                    -risk.values,
                    y_ext["event"].values.astype(bool),
                )
            ),
            4,
        )
    except Exception:
        return float("nan")


def _s1_sampler(trial, rank: int) -> list[str]:
    meta = STAGE3_METHODS.get(rank, {})
    s1 = str(meta.get("S1", ""))

    # B-start pipeline: no A-step, pass all training features through
    if meta.get("b_start", False):
        return X_train.columns.tolist()

    if "A7_ANOVA" in s1:
        anova_k = trial.suggest_int("anova_n_features", 100, 400)
        return anova_selection(X_train, y_train, k_features=anova_k)

    if "A1_Cox" in s1:
        cox_p = trial.suggest_float("cox_p_threshold", 0.005, 0.2, log=True)
        passed = GLOBAL_COX_P_VALUES[GLOBAL_COX_P_VALUES <= cox_p].sort_values()
        return passed.head(500).index.tolist()

    return []


def _s2_sampler(trial, s1_features: list[str], rank: int) -> list[str]:
    if len(s1_features) == 0:
        return []
    mn = trial.suggest_int("mrmr_n_features", 15, 60)
    return mrmr_selection(
        X_train[s1_features],
        y_train,
        n_features=min(mn, len(s1_features)),
    )


def _s3_sampler(trial, s2_features: list[str], rank: int) -> list[str]:
    if len(s2_features) == 0:
        return []

    meta = STAGE3_METHODS.get(rank, {})
    s3 = str(meta.get("S3", ""))
    X_s2 = X_train[s2_features]

    if "C3_Stability" in s3:
        stab_n = trial.suggest_int("stab_n_features", 5, 12)
        stab_b = trial.suggest_int("stab_n_bootstrap", 30, 80)
        stab_thr = trial.suggest_float("stab_threshold", 0.5, 0.85)
        return stability_selection_lasso(
            X_s2,
            y_train,
            n_features=min(stab_n, len(s2_features)),
            n_bootstrap=stab_b,
            stability_threshold=stab_thr,
            selection_strategy="threshold",
            random_state=SEED,
        )

    if "D1_RSF" in s3:
        rsf_n = trial.suggest_int("rsf_n_features", 5, 12)
        rsf_ne = trial.suggest_int("rsf_n_estimators", 200, 500)
        return rsf_permutation_importance(
            X_s2,
            y_train,
            n_features=min(rsf_n, len(s2_features)),
            n_estimators=rsf_ne,
            random_state=SEED,
        )

    if "D3_PermImp" in s3:
        pn = trial.suggest_int("permimp_n_features", 5, 12)
        pne = trial.suggest_int("permimp_n_estimators", 200, 500)
        return permutation_importance_survival(
            X_s2,
            y_train,
            n_features=min(pn, len(s2_features)),
            n_estimators=pne,
            random_state=SEED,
        )

    return []


def _common_objective(trial, s3_features: list[str], rank: int) -> float:
    n_features = len(s3_features)
    if n_features == 0 or n_features > FEA_MAX:
        return 0.0

    cv_scores = []
    for repeat in range(N_REPEATS):
        mean, _ = evaluate_features_cv(
            X_train,
            y_train,
            s3_features,
            method_name=f"Dose_S3_{rank}",
            n_splits=N_FOLDS,
            random_state=SEED + repeat,
        )
        if not np.isnan(mean):
            cv_scores.append(mean)

    if not cv_scores:
        return 0.0

    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))
    fea_penalty = min(n_features / FEA_MAX, 1.0)
    objective = WEIGHT_CV * cv_mean - WEIGHT_STD * cv_std - WEIGHT_FEA * fea_penalty

    trial.set_user_attr("cv_score", round(cv_mean, 4))
    trial.set_user_attr("cv_std", round(cv_std, 4))
    trial.set_user_attr("n_features", n_features)
    trial.set_user_attr("selected_features", s3_features)
    return objective


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
        logger.error("Dose_S3_%s Trial %s failed: %s", rank, trial.number, e)
        return 0.0


def run_optimization(rank: int, n_trials: int) -> optuna.Study:
    study_name = f"Dose75_S3_{rank}_Stage4"
    n_jobs = HEAVY_PIPELINE_N_JOBS if rank in HEAVY_RANKS else N_JOBS

    logger.info("\n%s", "=" * 70)
    logger.info("Starting: %s", study_name)
    if rank in STAGE3_METHODS:
        base = STAGE3_METHODS[rank]
        logger.info("Stage 3 baseline: CV3=%.4f Fea3=%s", base["CV3"], base["Fea3"])
        logger.info("Pipeline: %s", base["Pipeline"])
    logger.info(
        "Trials: %s n_jobs=%s | Objective: %.2f*CV_mean - %.2f*CV_std - %.2f*Fea | Repeats: %s",
        n_trials, n_jobs, WEIGHT_CV, WEIGHT_STD, WEIGHT_FEA, N_REPEATS,
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
        sampler=optuna.samplers.CmaEsSampler(seed=SEED, warn_independent_sampling=False),
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


def save_results(studies: dict[int, optuna.Study]) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    run_date = datetime.now().strftime("%Y%m%d")
    summary_rows = []
    all_trials = []
    inter_counter = 1

    for rank in sorted(DOSE_RANKS):
        study = studies.get(rank)
        if study is None:
            continue

        base = STAGE3_METHODS.get(rank, {})
        pipeline = base.get("Pipeline", f"Dose_S3_{rank}")

        df_trials = study.trials_dataframe()
        df_trials["Rank"] = rank
        df_trials["Pipeline"] = f"Dose_S3_{rank}:{pipeline}"
        df_trials["CV3"] = base.get("CV3", 0)
        df_trials["Fea3"] = base.get("Fea3", 0)
        df_trials["CV"] = df_trials["user_attrs_cv_score"] if "user_attrs_cv_score" in df_trials.columns else 0.0
        df_trials["Fea"] = df_trials["user_attrs_n_features"] if "user_attrs_n_features" in df_trials.columns else 0
        df_trials["Objective"] = df_trials["value"] if "value" in df_trials.columns else 0.0
        df_trials = df_trials.sort_values("Objective", ascending=False).reset_index(drop=True)
        pf = OUTPUT_DIR / f"Dose_S3_{rank}_trials_{run_date}.csv"
        df_trials.to_csv(pf, index=False)
        logger.info("Trials saved: %s", pf)

        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed_trials:
            logger.warning("Dose_S3_%s has no completed trials, skipping summary row.", rank)
            continue

        # Post-hoc best selection: prefer trials with n_features >= MIN_FEA_BEST
        # (EPV rule) to avoid noise-fitting on near-empty feature sets.
        # Falls back to unconstrained best if no trial meets the floor.
        eligible = [
            t for t in completed_trials
            if t.user_attrs.get("n_features", 0) >= MIN_FEA_BEST
        ]
        pool = eligible if eligible else completed_trials
        best = max(pool, key=lambda t: t.user_attrs.get("cv_score", 0.0))
        if not eligible:
            logger.warning(
                "Dose_S3_%s: no trial meets MIN_FEA_BEST=%s, falling back to unconstrained best.",
                rank, MIN_FEA_BEST,
            )
        else:
            logger.info(
                "Dose_S3_%s: selected best from %s eligible trials (Fea>=%s), trial #%s CV=%.4f Fea=%s",
                rank, len(eligible), MIN_FEA_BEST, best.number,
                best.user_attrs.get("cv_score", 0), best.user_attrs.get("n_features", 0),
            )

        features = best.user_attrs.get("selected_features", [])
        ext_chus = float("nan")
        if features:
            ext_chus = _evaluate_external_cindex(features, X_external_CHUS, y_external_CHUS)

        summary_rows.append(
            {
                "Rank": rank,
                "Pipeline": f"Dose_S3_{rank}:{pipeline}",
                "CV3": base.get("CV3", 0),
                "Fea3": base.get("Fea3", 0),
                "CV4": round(best.user_attrs.get("cv_score", 0), 4),
                "Fea4": best.user_attrs.get("n_features", 0),
                "Delta_CV": round(best.user_attrs.get("cv_score", 0) - base.get("CV3", 0), 4),
                "Objective": round(best.value, 4),
                "Best_params": str(best.params),
                "Ext_CHUS": ext_chus,
            }
        )

        for t in sorted(study.trials, key=lambda x: x.number):
            if t.state != optuna.trial.TrialState.COMPLETE:
                continue
            row = {
                "inter_no": inter_counter,
                "intra_no": f"Dose_S3_{rank}_{t.number}",
                "Rank": rank,
                "Pipeline": f"Dose_S3_{rank}:{pipeline}",
                "CV": t.user_attrs.get("cv_score"),
                "CV_std": t.user_attrs.get("cv_std"),
                "Fea": t.user_attrs.get("n_features"),
                "Objective": t.value,
            }
            row.update({f"params_{k}": v for k, v in t.params.items()})
            all_trials.append(row)
            inter_counter += 1

    summary_df = pd.DataFrame(summary_rows).sort_values("Rank").reset_index(drop=True)
    out_file = OUTPUT_DIR / f"Dose_stage4_results_75_{run_date}.csv"
    summary_df.to_csv(out_file, index=False)
    logger.info("Summary results saved: %s", out_file)

    df_all = pd.DataFrame(all_trials)
    all_out = OUTPUT_DIR / f"Dose_stage4_ALLtrials_75_{run_date}.csv"
    df_all.to_csv(all_out, index=False)
    logger.info("All trials combined CSV saved: %s (%s rows)", all_out, len(df_all))


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(exist_ok=True, parents=True)
    log_file = LOG_DIR / f"29_mar_t1C_dose_stage4_75_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    parser.add_argument("--rank", type=int, default=None, help="Single pipeline rank (Dose_S3_#)")
    parser.add_argument("--trials", type=int, default=None, help="Override n_trials")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("T1C Dose Stage 4-75 Optuna: full-chain tuning, CV-only")
    print("=" * 70 + "\n")

    if not test_db_connection():
        sys.exit(1)

    logger = setup_logging()
    random.seed(SEED)
    np.random.seed(SEED)

    _load_data()
    _load_external_data()
    _precompute_cox_p_values_global()
    _load_stage3_methods()
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    ranks_to_run = [args.rank] if args.rank is not None else DOSE_RANKS
    studies: dict[int, optuna.Study] = {}

    for rank in ranks_to_run:
        if rank not in DOSE_RANKS:
            logger.warning("Rank %s is not in the configured Stage 4-75 shortlist, skip", rank)
            continue
        if rank not in STAGE3_METHODS:
            logger.warning("Rank %s not in Stage 3 shortlist, skip", rank)
            continue
        n_trials = args.trials if args.trials is not None else N_TRIALS_MAP.get(rank, 250)
        studies[rank] = run_optimization(rank, n_trials)

    save_results(studies)
    logger.info("Done.")


if __name__ == "__main__":
    main()
