# -*- coding: utf-8 -*-
"""
29_mar_t1C_dose_stage3_75.py
============================
T1C Dose Stage 3: add S3 selector to each deduplicated Stage 2 pipeline.

Methodology: Task 1 pattern - S1/S2 loaded from checkpoints (full-dev),
S3 runs on X[s2_features], evaluate_features_cv scores S3 features only.

Usage:
    cd "D:/Uppsala thesis"
    python Mar_2026_task1C/29_mar_T1C_fs_script_results/29_mar_t1C_dose_stage3_75.py
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
TASK1C_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = TASK1C_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fs_utils import (
    MRMR_AVAILABLE,
    RELIEFF_AVAILABLE,
    elasticnet_cox_selection,
    evaluate_features_cv,
    permutation_importance_survival,
    rsf_permutation_importance,
    stability_selection_lasso,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = TASK1C_ROOT / "Dose_development_75.csv"
S1_CHECKPOINT_FEAT = SCRIPT_DIR / "Dose_stage1_checkpoint_features_75.pkl"
S2_CHECKPOINT_FEAT = SCRIPT_DIR / "Dose_stage2_checkpoint_features_75.pkl"
S2_RESULT_CSV = SCRIPT_DIR / "Dose_stage2_result_75.csv"
OUT_CSV = SCRIPT_DIR / "Dose_stage3_result_75.csv"
OUT_PKL = SCRIPT_DIR / "Dose_stage3_checkpoint_features_75.pkl"
CHECKPOINT = SCRIPT_DIR / "Dose_stage3_checkpoint_results_75.pkl"

SEED = 42
SAVE_EVERY = 10
S2_CUTOFF_RANK = "Dose_S2_20"
NON_FEATURE_COLS = {"PatientID", "CenterID", "Relapse", "RFS", "Gender_Male"}

RANK_TO_S1_KEY = {
    "Dose_6": "A6_ReliefF_30_n30",
    "Dose_9": "A1_Cox_p0.1",
    "Dose_14": "A7_ANOVA_200",
    "Dose_1": "B3_mRMR_20",
    "Dose_2": "B3_mRMR_50",
    "Dose_3": "B3_mRMR_30",
    "Dose_4": "C2_ElasticNet",
    "Dose_7": "C1_LASSO",
    "Dose_29": "C3_Stability_ranked_75",
    "Dose_12": "D3_PermImp_30",
    "Dose_21": "D3_PermImp_50",
    "Dose_23": "D2_XGBoost_50",
}


# ============================================================
# HELPERS
# ============================================================

def format_seconds(seconds: float) -> str:
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def parse_rank_num(rank_str: str) -> int:
    try:
        return int(str(rank_str).split("_")[-1])
    except Exception:
        return 10**9


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not DATA_FILE.exists():
        print(f"[ERROR] Missing input: {DATA_FILE}")
        sys.exit(1)

    df = pd.read_csv(DATA_FILE)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X_train = df[feature_cols].copy()
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    train_medians = X_train.median()
    X_train = X_train.fillna(train_medians)

    y_train = pd.DataFrame(
        {
            "RFS_time": df["RFS"].values,
            "event": df["Relapse"].values.astype(bool),
        }
    )

    print(
        f"  Patients: {len(df)} | Features: {len(feature_cols)} | "
        f"Events: {int(y_train['event'].sum())}"
    )
    return X_train, y_train


def load_feature_cache(path: Path, label: str) -> dict:
    if not path.exists():
        print(f"[ERROR] Missing {label}: {path}")
        sys.exit(1)
    with open(path, "rb") as f:
        cache = pickle.load(f)
    print(f"  Loaded {label}: {len(cache)} entries")
    return cache


def load_stage2_bases() -> list[dict]:
    if not S2_RESULT_CSV.exists():
        print(f"[ERROR] Missing Stage 2 CSV: {S2_RESULT_CSV}")
        sys.exit(1)

    df = pd.read_csv(S2_RESULT_CSV).copy()
    df["rank_num"] = df["Rank"].apply(parse_rank_num)
    cutoff_num = parse_rank_num(S2_CUTOFF_RANK)
    df = df[df["rank_num"] <= cutoff_num].sort_values("rank_num").reset_index(drop=True)

    exact_keep = []
    seen_exact = set()
    for _, row in df.iterrows():
        key = (float(row["CV2"]), int(row["Fea2"]))
        if key in seen_exact:
            continue
        exact_keep.append(row)
        seen_exact.add(key)
    exact_df = pd.DataFrame(exact_keep).reset_index(drop=True)

    near_keep = []
    for _, row in exact_df.iterrows():
        row_cv = float(row["CV2"])
        row_fea2 = int(row["Fea2"])
        is_near_dup = False
        for kept in near_keep:
            if int(kept["Fea2"]) != row_fea2:
                continue
            if abs(float(kept["CV2"]) - row_cv) <= 0.001:
                is_near_dup = True
                break
        if not is_near_dup:
            near_keep.append(row)

    dedup_df = pd.DataFrame(near_keep).sort_values("rank_num").reset_index(drop=True)

    bases = []
    for _, row in dedup_df.iterrows():
        pipeline_label = str(row["Pipeline"])
        parts = [p.strip() for p in pipeline_label.split("->")]
        s2_method = parts[1] if len(parts) >= 2 else ""
        bases.append(
            {
                "rank": str(row["Rank"]),
                "rank_num": int(row["rank_num"]),
                "category": str(row["Category"]),
                "s1_rank": str(row["S1_Rank"]),
                "pipeline_label": pipeline_label,
                "s2_method": s2_method,
                "cv2": float(row["CV2"]),
                "std2": float(row["Std2"]),
                "fea2": int(row["Fea2"]),
            }
        )

    print(f"  Stage 2 bases before dedup: {len(df)}")
    print(f"  After exact-dup dedup:      {len(exact_df)}")
    print(f"  After near-dup dedup:       {len(bases)}")
    return bases


def s3_method_specs() -> dict[str, dict]:
    return {
        "C2_ElasticNet": {
            "category": "C",
            "k": None,
            "func": lambda X, y: elasticnet_cox_selection(
                X, y, l1_ratio=0.5, target_features=100, n_alphas=100
            ),
        },
        "C3_Stability_30_0.6": {
            "category": "C",
            "k": 30,
            "func": lambda X, y: stability_selection_lasso(
                X,
                y,
                n_features=30,
                n_bootstrap=50,
                stability_threshold=0.6,
                selection_strategy="threshold",
                random_state=SEED,
            ),
        },
        "C3_Stability_30_0.7": {
            "category": "C",
            "k": 30,
            "func": lambda X, y: stability_selection_lasso(
                X,
                y,
                n_features=30,
                n_bootstrap=50,
                stability_threshold=0.7,
                selection_strategy="threshold",
                random_state=SEED,
            ),
        },
        "D1_RSF_PermImp_10": {
            "category": "D",
            "k": 10,
            "func": lambda X, y: rsf_permutation_importance(
                X, y, n_features=10, n_estimators=500, random_state=SEED
            ),
        },
        "D1_RSF_PermImp_20": {
            "category": "D",
            "k": 20,
            "func": lambda X, y: rsf_permutation_importance(
                X, y, n_features=20, n_estimators=500, random_state=SEED
            ),
        },
        "D3_PermImp_10": {
            "category": "D",
            "k": 10,
            "func": lambda X, y: permutation_importance_survival(
                X, y, n_features=10, n_estimators=500, random_state=SEED
            ),
        },
        "D3_PermImp_20": {
            "category": "D",
            "k": 20,
            "func": lambda X, y: permutation_importance_survival(
                X, y, n_features=20, n_estimators=500, random_state=SEED
            ),
        },
    }


def valid_s3_methods(base: dict) -> list[tuple[str, callable, int | None]]:
    s2_method = base["s2_method"]
    s2_category = s2_method[:1]
    fea2 = base["fea2"]
    specs = s3_method_specs()
    valid = []

    for s3_key, spec in specs.items():
        s3_category = spec["category"]
        s3_k = spec["k"]

        if s3_category not in {"C", "D"}:
            continue
        if s3_key == s2_method:
            continue
        if s2_category == "C" and s3_category != "D":
            continue
        if s3_k is not None and fea2 <= s3_k:
            continue

        valid.append((s3_key, spec["func"], s3_k))

    return valid


def load_checkpoint() -> tuple[list[dict], dict[str, list[str]]]:
    results = []
    features = {}
    if CHECKPOINT.exists():
        with open(CHECKPOINT, "rb") as f:
            payload = pickle.load(f)
        results = payload.get("results", [])
        features = payload.get("features", {})
        print(f"  [RESUME] Loaded checkpoint: {len(results)} completed pipelines")
    return results, features


def save_checkpoint(results: list[dict], features: dict[str, list[str]]) -> None:
    with open(CHECKPOINT, "wb") as f:
        pickle.dump({"results": results, "features": features}, f)


# ============================================================
# MAIN
# ============================================================

print("=" * 80)
print("T1C Dose Stage 3-75: three-step pipelines")
print("=" * 80)
start_time = time.time()

X_train, y_train = load_data()
s1_cache = load_feature_cache(S1_CHECKPOINT_FEAT, "Stage 1 feature cache")
s2_cache = load_feature_cache(S2_CHECKPOINT_FEAT, "Stage 2 feature cache")
bases = load_stage2_bases()
results, s3_features_storage = load_checkpoint()
completed_pipelines = {row["Pipeline"] for row in results}

if not MRMR_AVAILABLE:
    print("  [INFO] mRMR unavailable, but not needed for Stage 3.")
if not RELIEFF_AVAILABLE:
    print("  [INFO] ReliefF unavailable, but not needed for Stage 3.")

expected_total = sum(len(valid_s3_methods(base)) for base in bases)
print(f"  Expected Stage 3 pipelines after guard rules: {expected_total}")
print("\n[Running pipelines]\n")

done_since_save = 0
counter = 0

for base in bases:
    pipeline_label = base["pipeline_label"]
    s2_features = s2_cache.get(pipeline_label, [])
    if len(s2_features) == 0:
        print(f"  [SKIP] No S2 features cached for {pipeline_label}")
        continue

    s1_key = RANK_TO_S1_KEY.get(base["s1_rank"], "")
    if s1_key and s1_key not in s1_cache:
        print(f"  [WARN] S1 cache missing key for {base['s1_rank']} ({s1_key})")

    X_s2 = X_train[s2_features]
    for s3_key, s3_func, s3_k in valid_s3_methods(base):
        full_label = f"{pipeline_label} -> {s3_key}"
        if full_label in completed_pipelines:
            continue
        counter += 1

        try:
            s3_features = s3_func(X_s2, y_train)
        except Exception as exc:
            print(f"  [{counter}/{expected_total}] [ERROR] {full_label} | {type(exc).__name__}: {exc}")
            continue

        if len(s3_features) == 0:
            continue

        cv3, std3 = evaluate_features_cv(
            X_train,
            y_train,
            s3_features,
            method_name=full_label,
            random_state=SEED,
        )

        row = {
            "S2_Rank": base["rank"],
            "S1_Rank": base["s1_rank"],
            "Category": base["category"],
            "Pipeline": full_label,
            "CV3": cv3,
            "Std3": std3,
            "Delta_CV": cv3 - base["cv2"] if not np.isnan(cv3) else float("nan"),
            "Fea3": len(s3_features),
            "Fea2": base["fea2"],
            "Delta_Fea": len(s3_features) - base["fea2"],
            "S2_Rank_Num": base["rank_num"],
        }
        results.append(row)
        s3_features_storage[full_label] = s3_features
        completed_pipelines.add(full_label)
        done_since_save += 1

        delta_cv = cv3 - base["cv2"] if not np.isnan(cv3) else float("nan")
        print(
            f"  [{counter}/{expected_total}] {full_label} | "
            f"CV3={cv3:.4f} | Fea3={len(s3_features)} | "
            f"Delta_CV={delta_cv:+.4f}"
        )

        if done_since_save >= SAVE_EVERY:
            save_checkpoint(results, s3_features_storage)
            done_since_save = 0

save_checkpoint(results, s3_features_storage)

if len(results) > 0:
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by=["CV3", "Fea3", "Std3", "S2_Rank_Num"],
        ascending=[False, True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    results_df.insert(0, "Rank", [f"Dose_S3_{i}" for i in range(1, len(results_df) + 1)])
    results_df = results_df[
        [
            "Rank",
            "Category",
            "S2_Rank",
            "S1_Rank",
            "Pipeline",
            "CV3",
            "Std3",
            "Delta_CV",
            "Fea3",
            "Fea2",
            "Delta_Fea",
        ]
    ]
else:
    results_df = pd.DataFrame(
        columns=[
            "Rank",
            "Category",
            "S2_Rank",
            "S1_Rank",
            "Pipeline",
            "CV3",
            "Std3",
            "Delta_CV",
            "Fea3",
            "Fea2",
            "Delta_Fea",
        ]
    )

results_df.to_csv(OUT_CSV, index=False)
with open(OUT_PKL, "wb") as f:
    pickle.dump(s3_features_storage, f)

elapsed = time.time() - start_time
print(f"\n{'=' * 80}")
print("T1C DOSE STAGE 3 COMPLETE")
print(f"{'=' * 80}")
print(f"  Pipelines completed: {len(results_df)}")
print(f"  Elapsed time: {format_seconds(elapsed)}")
print(f"  Results CSV: {OUT_CSV.name}")
print(f"  Features PKL: {OUT_PKL.name}")
if len(results_df):
    print("\n  Top 5 by CV3:")
    print(results_df.head(5).to_string(index=False))
print(f"{'=' * 80}")
