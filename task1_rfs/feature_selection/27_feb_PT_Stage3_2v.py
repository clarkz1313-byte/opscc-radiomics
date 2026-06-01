# -*- coding: utf-8 -*-
"""
Stage 3: PT Radiomics Three-Step Pipelines - CHUS+CHUP External Split (2v)
==========================================================================
Expands top 19 Stage 2 pipelines (PT_S2_1 to PT_S2_19) from Processed result to S1->S2->S3.
A/B prefilters for C/D. No C_C_C, D_D_D, C/D-C/D-A/B. No same method.

Input:  Mar_2026/27_feb_PT_development.csv
        Mar_2026/27_feb_PT_Stage2_2v_Processed_result.csv

Output: Mar_2026/27_feb_PT_Stage3_2v_result.csv
        Mar_2026/27_feb_PT_Stage3_2v_result_metadata.json

No train/test split. CV only (no Test3). Cox-starting: precomputed p-values.
When filtering/deduping downstream: if CV3/Std3/Fea3 identical, use S2_Base_Rank
(lower=better) to prefer the pipeline from the better-performing S2 base.

Usage:
    cd "D:/Uppsala thesis"
    python Mar_2026/27_feb_PT_Stage3_2v.py
"""

from __future__ import annotations

import json
import pickle
import random
import time
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fs_utils import (
    mutual_info_selection,
    relieff_selection,
    mrmr_selection,
    lasso_cox_selection,
    elasticnet_cox_selection,
    stability_selection_lasso,
    xgboost_survival_selection,
    permutation_importance_survival,
    rsf_permutation_importance,
    univariate_cox_selection,
    evaluate_features_cv,
)

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
RESULTS_BASE = "27_feb_PT_Stage3_2v_result"
DATA_FILE = SCRIPT_DIR / "27_feb_PT_development.csv"
STAGE2_RESULT = SCRIPT_DIR / "27_feb_PT_Stage2_2v_Processed_result.csv"
STAGE2_RANK_MIN = 1
STAGE2_RANK_MAX = 19
EXPECTED_PIPELINES = 94  # sanity check: 4 C-D-D + 30 B-C + 15 B-D + 19 A-C + 18 A-B + 12 A-D
CHECKPOINT_PATH = SCRIPT_DIR / f"{RESULTS_BASE}_checkpoint.pkl"
CHECKPOINT_FAILED = SCRIPT_DIR / f"{RESULTS_BASE}_checkpoint_failed.pkl"

BASELINE_CV1 = 0.754
BASELINE_FEA1 = 30
BASELINE_STD1 = 0.065
BASELINE_CV2 = 0.7544
BASELINE_FEA2 = 15

COX_PRECOMPUTE_S1_KEY = "A1_Cox_p0.001"

random.seed(SEED)
np.random.seed(SEED)

# Storage keys by category (PT Stage 2 format)
C_KEYS = ["C1_LASSO", "C2_ElasticNet", "C3_Stability_50_0.7"]
D_KEYS = ["D1_RSF_PermImp_50", "D2_XGBoost_50", "D3_PermImp_50"]
B_KEYS = ["B3_mRMR_30", "B3_mRMR_40", "B3_mRMR_50"]

try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm

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
# DATACLASS
# ============================================================


@dataclass(frozen=True)
class Stage3Pipeline:
    pipeline_name: str
    pattern: str
    s1_key: str
    s2_key: str
    s3_key: str
    s2_base_rank: int
    s2_base_cv: float
    s2_base_fea: int


# ============================================================
# METHOD REGISTRY (storage keys -> callable)
# ============================================================


def build_method_funcs(seed: int) -> Dict[str, callable]:
    """Map storage keys to callable (matching 27_feb_PT_Stage2_2v)."""
    return {
        "A1_Cox_p0.001": lambda X, y, pv=None: (
            pv[pv < 0.001].index.tolist() if pv is not None
            else univariate_cox_selection(X, y, p_threshold=0.001)
        ),
        "A6_ReliefF_50_n100": lambda X, y, pv=None: relieff_selection(X, y, n_features=50, n_neighbors=100),
        "A4_MI_100": lambda X, y, pv=None: mutual_info_selection(X, y, k_features=100),
        "B3_mRMR_30": lambda X, y, pv=None: mrmr_selection(X, y, n_features=30),
        "B3_mRMR_40": lambda X, y, pv=None: mrmr_selection(X, y, n_features=40),
        "B3_mRMR_50": lambda X, y, pv=None: mrmr_selection(X, y, n_features=50),
        "C1_LASSO": lambda X, y, pv=None: lasso_cox_selection(X, y, target_features=100, n_alphas=100),
        "C2_ElasticNet": lambda X, y, pv=None: elasticnet_cox_selection(
            X, y, l1_ratio=0.5, target_features=100, n_alphas=100
        ),
        "C3_Stability_50_0.7": lambda X, y, pv=None: stability_selection_lasso(
            X, y, n_bootstrap=100, stability_threshold=0.7, n_features=50,
            selection_strategy="threshold", random_state=seed
        ),
        "D1_RSF_PermImp_50": lambda X, y, pv=None: rsf_permutation_importance(
            X, y, n_features=50, n_estimators=500, random_state=seed
        ),
        "D2_XGBoost_50": lambda X, y, pv=None: xgboost_survival_selection(
            X, y, n_features=50, n_estimators=100, random_state=seed
        ),
        "D3_PermImp_50": lambda X, y, pv=None: permutation_importance_survival(
            X, y, n_features=50, n_estimators=500, random_state=seed
        ),
    }


# ============================================================
# PIPELINE GENERATION
# ============================================================
# Rules: A/B prefilters for C/D. No C_C_C, D_D_D. No C/D-C/D-A/B.
# C-D bases: S3 in D only (excl S2) — no C-D-B.
# No same method (S3 != S2 when same category).


def _parse_rank_num(rank_str: str) -> int:
    """PT_S2_1 -> 1"""
    try:
        return int(rank_str.replace("PT_S2_", ""))
    except (ValueError, AttributeError):
        return -1


def load_stage2_bases() -> List[Tuple[int, str, str, str, float, float, int]]:
    """Load Processed Stage 2 result, filter ranks 1-19. Returns (rank, category, s1_key, s2_key, cv2, std2, fea2)."""
    if not STAGE2_RESULT.exists():
        raise FileNotFoundError(f"Stage 2 result not found: {STAGE2_RESULT}")

    df = pd.read_csv(STAGE2_RESULT)
    bases = []
    for _, row in df.iterrows():
        rank_str = str(row.get("Rank", ""))
        rank_num = _parse_rank_num(rank_str)
        if rank_num < STAGE2_RANK_MIN or rank_num > STAGE2_RANK_MAX:
            continue
        pipeline = str(row.get("Pipeline", ""))
        parts = [p.strip() for p in pipeline.split("->")]
        if len(parts) != 2:
            continue
        s1_key, s2_key = parts[0], parts[1]
        cat = str(row.get("Category", ""))
        cv2 = float(row.get("CV2", 0))
        std2 = float(row.get("Std2", 0))
        fea2 = int(row.get("Fea2", 0))
        bases.append((rank_num, cat, s1_key, s2_key, cv2, std2, fea2))
    return sorted(bases, key=lambda x: x[0])


def generate_stage3_pipelines(bases: List[Tuple]) -> List[Stage3Pipeline]:
    """Expand each base to S3. No C-D-B. No same method."""
    pipelines: List[Stage3Pipeline] = []
    seen: set = set()

    for rank, category, s1_key, s2_key, cv2, std2, fea2 in bases:
        s3_options: List[Tuple[str, str]] = []

        if category == "B-C":
            for c in C_KEYS:
                if c != s2_key:
                    s3_options.append((c, "B-C-C"))
            for d in D_KEYS:
                s3_options.append((d, "B-C-D"))
        elif category == "B-D":
            for c in C_KEYS:
                s3_options.append((c, "B-D-C"))
            for d in D_KEYS:
                if d != s2_key:
                    s3_options.append((d, "B-D-D"))
        elif category == "C-D":
            # No C/D-C/D-A/B: C-D expands to D only (not B)
            for d in D_KEYS:
                if d != s2_key:
                    s3_options.append((d, "C-D-D"))
        elif category == "A-C":
            for c in C_KEYS:
                if c != s2_key:
                    s3_options.append((c, "A-C-C"))
            for d in D_KEYS:
                s3_options.append((d, "A-C-D"))
        elif category == "A-B":
            for c in C_KEYS:
                s3_options.append((c, "A-B-C"))
            for d in D_KEYS:
                s3_options.append((d, "A-B-D"))
        elif category == "A-D":
            for c in C_KEYS:
                s3_options.append((c, "A-D-C"))
            for b in B_KEYS:
                s3_options.append((b, "A-D-B"))
        else:
            continue

        for s3_key, pattern in s3_options:
            name = f"{s1_key} -> {s2_key} -> {s3_key}"
            if name in seen:
                continue
            seen.add(name)
            pipelines.append(Stage3Pipeline(
                pipeline_name=name,
                pattern=pattern,
                s1_key=s1_key,
                s2_key=s2_key,
                s3_key=s3_key,
                s2_base_rank=rank,
                s2_base_cv=cv2,
                s2_base_fea=fea2,
            ))

    return pipelines


# ============================================================
# COX P-VALUE PRECOMPUTATION
# ============================================================


def precompute_cox_p_values(X, y_df):
    """Precompute univariate Cox p-values for all features (Cox-starting pipelines)."""
    from lifelines import CoxPHFitter
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


# ============================================================
# DATA LOADING
# ============================================================


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load PT development data. No train/test split — all dev."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Data not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    feature_cols = [c for c in df.columns if c not in ["PatientID", "Relapse", "RFS"]]
    X = df[feature_cols].copy()
    y_time = df["RFS"].values
    y_event = df["Relapse"].values.astype(bool)

    X_train = X.replace([np.inf, -np.inf], np.nan)
    train_medians = X_train.median()
    X_train = X_train.fillna(train_medians)
    y_train = pd.DataFrame({"RFS_time": y_time, "event": y_event})

    print(f"  Patients: {len(df)} | Features: {len(feature_cols)} | Events: {int(y_event.sum())}")
    return X_train, y_train


# ============================================================
# PIPELINE RUNNER
# ============================================================


def _run_method(func, X, y, cox_p_values, key):
    """Run method; use precomputed p-values for A1_Cox."""
    if key == COX_PRECOMPUTE_S1_KEY and cox_p_values is not None:
        return list(cox_p_values[cox_p_values < 0.001].index)
    return list(func(X, y, cox_p_values))


def run_three_stage_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    s1_key: str,
    s2_key: str,
    s3_key: str,
    method_funcs: Dict[str, callable],
    cox_p_values,
) -> Optional[Dict]:
    """Run S1 -> S2 -> S3. Returns dict or None."""
    try:
        f1 = method_funcs.get(s1_key)
        f2 = method_funcs.get(s2_key)
        f3 = method_funcs.get(s3_key)
        if not all([f1, f2, f3]):
            return None

        features_s1 = _run_method(f1, X_train, y_train, cox_p_values, s1_key)
        if len(features_s1) == 0:
            return None

        X_s1 = X_train[features_s1]
        features_s2 = _run_method(f2, X_s1, y_train, None, s2_key)
        if len(features_s2) == 0:
            return None

        X_s2 = X_train[features_s2]
        features_s3 = _run_method(f3, X_s2, y_train, None, s3_key)
        if len(features_s3) == 0:
            return None

        cv_score, cv_std = evaluate_features_cv(
            X_train, y_train, features_s3,
            method_name=f"{s1_key}->{s2_key}->{s3_key}",
            random_state=SEED,
        )

        return {
            "CV3": cv_score,
            "Std3": cv_std,
            "Fea3": len(features_s3),
            "Fea1": len(features_s1),
            "Fea2": len(features_s2),
        }

    except Exception:
        return None


# ============================================================
# CHECKPOINT
# ============================================================


def load_checkpoint() -> Tuple[Dict[str, Dict], set]:
    completed, failed = {}, set()
    if CHECKPOINT_PATH.exists():
        try:
            with open(CHECKPOINT_PATH, "rb") as f:
                completed = pickle.load(f)
        except Exception:
            completed = {}
    if CHECKPOINT_FAILED.exists():
        try:
            with open(CHECKPOINT_FAILED, "rb") as f:
                failed = pickle.load(f)
        except Exception:
            failed = set()
    return completed, failed


def save_checkpoint(completed: Dict, failed: set) -> None:
    with open(CHECKPOINT_PATH, "wb") as f:
        pickle.dump(completed, f)
    with open(CHECKPOINT_FAILED, "wb") as f:
        pickle.dump(failed, f)


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    print("=" * 80)
    print("PT STAGE 3: Three-Step Pipelines (2v) - Top 19 Stage 2 Bases (Processed)")
    print("=" * 80)
    start_time = time.time()

    X_train, y_train = load_data()

    bases = load_stage2_bases()
    print(f"  Stage 2 bases loaded: {len(bases)} (ranks {STAGE2_RANK_MIN}-{STAGE2_RANK_MAX})")

    pipelines = generate_stage3_pipelines(bases)
    print(f"  Stage 3 pipelines generated: {len(pipelines)}")
    if len(pipelines) != EXPECTED_PIPELINES:
        print(f"  [WARN] Expected {EXPECTED_PIPELINES} pipelines (sanity check)")

    needs_cox = any(b[2] == COX_PRECOMPUTE_S1_KEY for b in bases)
    cox_p_values = None
    if needs_cox:
        print("  Precomputing Cox p-values for A1_Cox-starting pipelines...")
        cox_p_values = precompute_cox_p_values(X_train, y_train)

    method_funcs = build_method_funcs(SEED)
    completed, failed = load_checkpoint()

    print(f"\n[Running pipelines]\n")

    try:
        for idx, pipe in enumerate(pipelines, start=1):
            name = pipe.pipeline_name
            if name in completed:
                print(f"  [{idx:3d}/{len(pipelines)}] [SKIP] {name}")
                continue
            if name in failed:
                print(f"  [{idx:3d}/{len(pipelines)}] [FAIL-SKIP] {name}")
                continue

            needs_relieff = "A6_ReliefF" in name
            needs_mrmr = "B3_mRMR" in name
            if needs_relieff and not RELIEFF_AVAILABLE:
                failed.add(name)
                save_checkpoint(completed, failed)
                continue
            if needs_mrmr and not MRMR_AVAILABLE:
                failed.add(name)
                save_checkpoint(completed, failed)
                continue

            print(f"  [{idx:3d}/{len(pipelines)}] {name}")
            result = run_three_stage_pipeline(
                X_train, y_train,
                pipe.s1_key, pipe.s2_key, pipe.s3_key,
                method_funcs, cox_p_values,
            )

            if result is None:
                failed.add(name)
            else:
                result.update({
                    "Pipeline": name,
                    "Pattern": pipe.pattern,
                    "S2_Base_Rank": pipe.s2_base_rank,
                    "S2_Base_CV": pipe.s2_base_cv,
                    "S2_Base_Fea": pipe.s2_base_fea,
                })
                completed[name] = result
                print(f"    CV3={result['CV3']:.4f} Fea3={result['Fea3']}")

            save_checkpoint(completed, failed)

    except KeyboardInterrupt:
        print("\n  [INTERRUPTED] Saving checkpoint...")
    finally:
        save_checkpoint(completed, failed)

    execution_time = time.time() - start_time

    if len(completed) > 0:
        df = pd.DataFrame(list(completed.values()))
        # Primary: CV3 desc. Tie-break: Fea3 asc, Std3 asc, S2_Base_Rank asc (better S2 base first)
        df = df.sort_values(
            by=["CV3", "Fea3", "Std3", "S2_Base_Rank"],
            ascending=[False, True, True, True],
        ).reset_index(drop=True)
        df["Rank"] = [f"PT_S3_{i}" for i in range(1, len(df) + 1)]
        df["DeCV3-B1"] = df["CV3"] - BASELINE_CV1
        df["DeCV3-B2"] = df["CV3"] - BASELINE_CV2
        df["DeStd3-Std1"] = df["Std3"] - BASELINE_STD1
        df["DeFea3-B1"] = df["Fea3"] - BASELINE_FEA1
        df["DeFea3-B2"] = df["Fea3"] - BASELINE_FEA2

        export_cols = [
            "Rank", "Pattern", "Pipeline",
            "CV3", "DeCV3-B1", "DeCV3-B2", "Std3", "DeStd3-Std1",
            "Fea3", "DeFea3-B1", "DeFea3-B2", "Fea1", "Fea2",
            "S2_Base_Rank", "S2_Base_CV", "S2_Base_Fea",
        ]
        df = df[export_cols]
    else:
        df = pd.DataFrame()

    output_file = SCRIPT_DIR / f"{RESULTS_BASE}.csv"
    df.to_csv(output_file, index=False)

    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
    if CHECKPOINT_FAILED.exists():
        CHECKPOINT_FAILED.unlink()

    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "execution_time_seconds": round(execution_time, 1),
        "seed": SEED,
        "n_stage2_bases": len(bases),
        "n_pipelines": len(pipelines),
        "completed": len(completed),
        "failed": len(failed),
        "data_file": "Mar_2026/27_feb_PT_development.csv",
        "stage2_file": "Mar_2026/27_feb_PT_Stage2_2v_Processed_result.csv",
        "stage2_rank_range": [STAGE2_RANK_MIN, STAGE2_RANK_MAX],
        "baseline_cv1": BASELINE_CV1,
        "baseline_fea1": BASELINE_FEA1,
        "baseline_cv2": BASELINE_CV2,
        "baseline_fea2": BASELINE_FEA2,
    }
    with open(SCRIPT_DIR / f"{RESULTS_BASE}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*80}")
    print("PT STAGE 3 COMPLETE")
    print(f"{'='*80}")
    print(f"  Completed: {len(completed)} | Failed: {len(failed)} | Total: {len(pipelines)}")
    print(f"  Time: {execution_time/60:.1f} min")
    print(f"  Results: {output_file.name}")
    if len(df) > 0:
        print(f"\n  Top 5:")
        print(df.head()[["Rank", "Pattern", "Pipeline", "CV3", "Fea3"]].to_string(index=False))
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
