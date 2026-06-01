# -*- coding: utf-8 -*-
"""
15_mar_task2_stage4_CT_add_test4_ext.py

Post-processing patch: add Test4 and extAUC columns to the existing
14_mar_task2_stage4_CT_ALLtrials and results CSVs without rerunning Optuna.

Loads all CT Stage 4 studies from PostgreSQL, recomputes Test4 (internal test
set) and extAUC (external set) for every completed trial using the
`selected_features` user attribute, then patches both CSVs in-place.

Usage:
    cd "D:/Uppsala thesis"
    python Mar_2026_task2/15_mar_task2_stage4_CT_add_test4_ext.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import optuna
import pandas as pd

from Mar_2026_task2.fs_task2_utils import evaluate_auc_test

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------
# Paths — mirror 14_mar_task2_stage4_CT.py exactly
# ---------------------------------------------------------------------
OUTPUT_DIR   = SCRIPT_DIR / "14_mar_t2_optuna_outputs"
TRAIN_FILE   = SCRIPT_DIR / "12_mar_task2_rad_data" / "13_mar_task2_CT_primary_train.csv"
TEST_FILE    = SCRIPT_DIR / "12_mar_task2_rad_data" / "13_mar_task2_CT_primary_test.csv"
EXT_FILE     = SCRIPT_DIR / "12_mar_task2_rad_data" / "12_mar_task2_CT_primary_ext.csv"

DB_HOST     = "localhost"
DB_PORT     = 5432
DB_USER     = "postgres"
DB_PASSWORD = "1730"
DB_NAME     = "optuna_db"
DB_URL      = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

PIPELINES = [1, 2, 3, 6, 7, 8, 12]
SEED = 42

EXCLUDE = ["PatientID", "HPV_binary", "Relapse", "RFS",
           "Age", "Gender_Male", "Treatment_CRT", "prefix"]


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
def _load_data():
    df_tr = pd.read_csv(TRAIN_FILE)
    feat_cols = [c for c in df_tr.columns if c not in EXCLUDE]
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

    print(f"Train: {len(df_tr)} | Test: {len(df_te)} | Ext: {len(df_ext)}")
    return X_train, y_train, X_test, y_test, X_ext, y_ext


def _safe_auc(X_train, y_train, X_eval, y_eval, features) -> float:
    avail = [f for f in features if f in X_train.columns and f in X_eval.columns]
    if len(avail) < 2:
        return float("nan")
    try:
        return round(float(evaluate_auc_test(X_train, y_train, X_eval, y_eval, avail, random_state=SEED)), 4)
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print("=" * 70)
    print("CT Stage 4 post-patch: adding Test4 + extAUC")
    print("=" * 70)

    X_train, y_train, X_test, y_test, X_ext, y_ext = _load_data()

    storage = optuna.storages.RDBStorage(
        url=DB_URL,
        engine_kwargs={"connect_args": {"connect_timeout": 60}},
    )

    # Build lookup: (rank, trial_number) -> {test4, ext_auc}
    lookup: dict[str, dict] = {}   # key = intra_no "CT_S3_{rank}_{t.number}"

    for rank in PIPELINES:
        study_name = f"CT_S3_{rank}_Task2_Stage4"
        try:
            study = optuna.load_study(study_name=study_name, storage=storage)
        except Exception as e:
            print(f"  Could not load {study_name}: {e}")
            continue

        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        print(f"  CT_S3_{rank}: {len(completed)} completed trials — computing Test4 + extAUC...")

        for t in completed:
            feats = t.user_attrs.get("selected_features", [])
            test4   = _safe_auc(X_train, y_train, X_test,  y_test,  feats)
            ext_auc = _safe_auc(X_train, y_train, X_ext,   y_ext,   feats)
            lookup[f"CT_S3_{rank}_{t.number}"] = {"Test4": test4, "extAUC": ext_auc}

    print(f"\nTotal trials computed: {len(lookup)}")

    # --- Patch ALLtrials CSV ---
    all_csvs = sorted(OUTPUT_DIR.glob("14_mar_task2_stage4_CT_ALLtrials_*.csv"))
    if not all_csvs:
        print("ERROR: No ALLtrials CSV found in", OUTPUT_DIR)
        return
    all_csv = all_csvs[-1]   # most recent
    print(f"\nPatching ALLtrials: {all_csv.name}")
    df_all = pd.read_csv(all_csv)

    df_all["Test4"]  = df_all["intra_no"].map(lambda k: lookup.get(k, {}).get("Test4",  float("nan")))
    df_all["extAUC"] = df_all["intra_no"].map(lambda k: lookup.get(k, {}).get("extAUC", float("nan")))

    df_all.to_csv(all_csv, index=False)
    print(f"  Saved ({len(df_all)} rows, Test4 non-null={df_all['Test4'].notna().sum()}, extAUC non-null={df_all['extAUC'].notna().sum()})")

    # --- Patch results summary CSV ---
    res_csvs = sorted(OUTPUT_DIR.glob("14_mar_task2_stage4_CT_results_*.csv"))
    if res_csvs:
        res_csv = res_csvs[-1]
        print(f"\nPatching results summary: {res_csv.name}")
        df_res = pd.read_csv(res_csv)

        ext_by_rank = {}
        for rank in PIPELINES:
            study_name = f"CT_S3_{rank}_Task2_Stage4"
            try:
                study = optuna.load_study(study_name=study_name, storage=storage)
                best_feats = study.best_trial.user_attrs.get("selected_features", [])
                ext_by_rank[f"CT_S3_{rank}"] = _safe_auc(X_train, y_train, X_ext, y_ext, best_feats)
            except Exception:
                ext_by_rank[f"CT_S3_{rank}"] = float("nan")

        df_res["extAUC"] = df_res["Rank"].map(ext_by_rank)
        df_res.to_csv(res_csv, index=False)
        print(f"  Saved ({len(df_res)} rows)")
        print("\n  Results summary with extAUC:")
        print(df_res[["Rank", "AUC4", "Test4", "extAUC"]].to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
