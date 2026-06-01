# -*- coding: utf-8 -*-
"""
Task 2: create a frozen optimized internal train/test split map.

Final design:
- one shared CT/PT PatientID split
- fixed n_test=20, n_train=67
- exact test composition: 14 HPV+ / 6 HPV-
- singleton CHUM HPV- forced to train
- HMR kept train-only
- split search is deterministic with SEED=42

Usage:
    cd "D:/Uppsala thesis"
    python Mar_2026_task2/13_mar_task2_make_split.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


SEED = 42
N_ITER = 5000

W1 = 1.0
W2 = 2.0
W3 = 3.0

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "12_mar_task2_rad_data"

CT_DEV_FILE = DATA_DIR / "12_mar_task2_CT_primary_dev.csv"
PT_DEV_FILE = DATA_DIR / "12_mar_task2_PT_primary_dev.csv"
OUT_SPLIT_FILE = DATA_DIR / "13_mar_task2_split_map.csv"
CT_TRAIN_FILE = DATA_DIR / "13_mar_task2_CT_primary_train.csv"
CT_TEST_FILE = DATA_DIR / "13_mar_task2_CT_primary_test.csv"
PT_TRAIN_FILE = DATA_DIR / "13_mar_task2_PT_primary_train.csv"
PT_TEST_FILE = DATA_DIR / "13_mar_task2_PT_primary_test.csv"


@dataclass(frozen=True)
class Candidate:
    score: float
    test_ids: tuple[str, ...]
    count_key: tuple[tuple[str, int], ...]


def load_primary_dev() -> pd.DataFrame:
    ct = pd.read_csv(CT_DEV_FILE)
    pt = pd.read_csv(PT_DEV_FILE)

    ct_ids = set(ct["PatientID"])
    pt_ids = set(pt["PatientID"])
    if ct_ids != pt_ids:
        raise ValueError("CT/PT primary dev files do not contain the same PatientIDs.")

    ct_check = ct[["PatientID", "HPV_binary"]].copy().sort_values("PatientID").reset_index(drop=True)
    pt_check = pt[["PatientID", "HPV_binary"]].copy().sort_values("PatientID").reset_index(drop=True)
    if not ct_check.equals(pt_check):
        raise ValueError("CT/PT primary dev files disagree on PatientID/HPV labels.")

    # Preserve CT row order so the reconstructed baseline split matches the
    # current Stage 1 scripts, which split directly on the dev file as loaded.
    meta = ct[["PatientID", "HPV_binary"]].copy()
    meta["HPV_binary"] = meta["HPV_binary"].astype(int)
    meta["prefix"] = meta["PatientID"].str[:4]
    return meta


def save_split_datasets(test_ids: set[str]) -> None:
    split_map = pd.read_csv(OUT_SPLIT_FILE)
    train_ids = set(split_map.loc[split_map["split"] == "train", "PatientID"])
    test_ids_from_map = set(split_map.loc[split_map["split"] == "test", "PatientID"])
    if test_ids_from_map != test_ids:
        raise ValueError("Split map and chosen test IDs disagree.")

    ct = pd.read_csv(CT_DEV_FILE)
    pt = pd.read_csv(PT_DEV_FILE)

    ct_train = ct[ct["PatientID"].isin(train_ids)].copy()
    ct_test = ct[ct["PatientID"].isin(test_ids)].copy()
    pt_train = pt[pt["PatientID"].isin(train_ids)].copy()
    pt_test = pt[pt["PatientID"].isin(test_ids)].copy()

    ct_train.to_csv(CT_TRAIN_FILE, index=False)
    ct_test.to_csv(CT_TEST_FILE, index=False)
    pt_train.to_csv(PT_TRAIN_FILE, index=False)
    pt_test.to_csv(PT_TEST_FILE, index=False)


def build_cell_pools(meta: pd.DataFrame) -> dict[tuple[str, int], list[str]]:
    pools: dict[tuple[str, int], list[str]] = {}
    for (prefix, hpv), sub_df in meta.groupby(["prefix", "HPV_binary"]):
        pools[(prefix, int(hpv))] = sorted(sub_df["PatientID"].tolist())
    return pools


def get_count_key(test_ids: set[str], meta: pd.DataFrame) -> tuple[tuple[str, int], ...]:
    test_df = meta[meta["PatientID"].isin(test_ids)]
    counts = (
        test_df.groupby(["prefix", "HPV_binary"])
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["prefix", "HPV_binary"])
    )
    return tuple((f"{row.prefix}|{int(row.HPV_binary)}", int(row.n)) for row in counts.itertuples())


def summarize_split(meta: pd.DataFrame, test_ids: set[str]) -> dict[str, object]:
    split_df = meta.copy()
    split_df["split"] = np.where(split_df["PatientID"].isin(test_ids), "test", "train")

    train_df = split_df[split_df["split"] == "train"].copy()
    test_df = split_df[split_df["split"] == "test"].copy()

    train_neg = int((train_df["HPV_binary"] == 0).sum())
    test_neg = int((test_df["HPV_binary"] == 0).sum())
    train_pos = int((train_df["HPV_binary"] == 1).sum())
    test_pos = int((test_df["HPV_binary"] == 1).sum())

    center_prop_train = train_df["prefix"].value_counts(normalize=True).to_dict()
    center_prop_test = test_df["prefix"].value_counts(normalize=True).to_dict()
    all_centers = sorted(split_df["prefix"].unique())

    hpv_rate_train = float(train_df["HPV_binary"].mean())
    hpv_rate_test = float(test_df["HPV_binary"].mean())

    score_center = 0.0
    for center in all_centers:
        score_center += abs(center_prop_train.get(center, 0.0) - center_prop_test.get(center, 0.0))

    score_within = 0.0
    for center in ["CHUP", "HGJ-"]:
        tr_center = train_df[train_df["prefix"] == center]
        te_center = test_df[test_df["prefix"] == center]
        if len(tr_center) == 0 or len(te_center) == 0:
            score_within += 1.0
        else:
            score_within += abs(float(tr_center["HPV_binary"].mean()) - float(te_center["HPV_binary"].mean()))

    score = (
        W1 * abs(hpv_rate_train - hpv_rate_test)
        + W2 * score_center
        + W3 * score_within
    )

    return {
        "split_df": split_df,
        "train_df": train_df,
        "test_df": test_df,
        "train_pos": train_pos,
        "train_neg": train_neg,
        "test_pos": test_pos,
        "test_neg": test_neg,
        "score": score,
    }


def is_valid_candidate(meta: pd.DataFrame, test_ids: set[str]) -> bool:
    summary = summarize_split(meta, test_ids)
    train_df = summary["train_df"]
    test_df = summary["test_df"]

    if len(train_df) != 67 or len(test_df) != 20:
        return False
    if summary["test_pos"] != 14 or summary["test_neg"] != 6:
        return False
    if summary["train_pos"] != 47 or summary["train_neg"] != 20:
        return False

    # Singleton CHUM HPV- forced to train.
    chumn_neg = meta[(meta["prefix"] == "CHUM") & (meta["HPV_binary"] == 0)]["PatientID"].tolist()
    if any(pid in test_ids for pid in chumn_neg):
        return False

    # Largest center must be represented in test.
    if int(((test_df["prefix"] == "CHUM") & (test_df["HPV_binary"] == 1)).sum()) < 3:
        return False

    # HMR remains train-only in the final design.
    if any(test_df["prefix"] == "HMR-"):
        return False

    # Every center with >=2 patients must have at least one patient in train.
    center_sizes = meta["prefix"].value_counts()
    for center, size in center_sizes.items():
        if size >= 2 and int((train_df["prefix"] == center).sum()) == 0:
            return False

    return True


def sample_candidate(rng: np.random.Generator, pools: dict[tuple[str, int], list[str]]) -> set[str]:
    # Fixed choices.
    counts = {
        ("CHUM", 1): 5,
        ("CHUM", 0): 0,
        ("CHUP", 1): int(rng.choice([3, 4])),
        ("HGJ-", 1): 0,  # filled below to preserve total test HPV+ = 14
        ("CHUP", 0): int(rng.choice([3, 4])),
        ("HGJ-", 0): 0,  # filled below to preserve total test HPV- = 6
        ("HMR-", 1): 0,
    }
    counts[("HGJ-", 1)] = 14 - counts[("CHUM", 1)] - counts[("CHUP", 1)]
    counts[("HGJ-", 0)] = 6 - counts[("CHUP", 0)]

    test_ids: set[str] = set()
    for cell, n_take in counts.items():
        pool = pools.get(cell, [])
        if n_take < 0 or n_take > len(pool):
            return set()
        if n_take == 0:
            continue
        chosen = rng.choice(pool, size=n_take, replace=False)
        test_ids.update(str(pid) for pid in chosen)
    return test_ids


def reconstruct_baseline_seed42(meta: pd.DataFrame) -> set[str]:
    train_idx, test_idx = train_test_split(
        meta.index,
        test_size=0.2,
        random_state=SEED,
        stratify=meta["HPV_binary"],
    )
    return set(meta.loc[test_idx, "PatientID"].tolist())


def print_cell_table(title: str, df: pd.DataFrame) -> None:
    print(f"\n{title}")
    cell_table = (
        df.groupby(["prefix", "HPV_binary"])
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["prefix", "HPV_binary"])
    )
    if cell_table.empty:
        print("  <empty>")
        return
    print(cell_table.to_string(index=False))


def print_split_summary(title: str, meta: pd.DataFrame, test_ids: set[str]) -> None:
    summary = summarize_split(meta, test_ids)
    train_df = summary["train_df"]
    test_df = summary["test_df"]

    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}")
    print(
        f"Train: {len(train_df)}  (HPV+: {summary['train_pos']}, HPV-: {summary['train_neg']})"
    )
    print(
        f"Test:  {len(test_df)}  (HPV+: {summary['test_pos']}, HPV-: {summary['test_neg']})"
    )
    print(f"Score: {summary['score']:.6f}")

    print_cell_table("Train cell table", train_df)
    print_cell_table("Test cell table", test_df)

    print("\nCenter totals")
    center_summary = pd.DataFrame({
        "train": train_df["prefix"].value_counts(),
        "test": test_df["prefix"].value_counts(),
    }).fillna(0).astype(int)
    center_summary = center_summary.sort_index()
    print(center_summary.to_string())


def main() -> None:
    print("=" * 80)
    print("Task 2: Optimized Internal Split Map")
    print("=" * 80)
    print(f"SEED: {SEED}")
    print(f"Iterations: {N_ITER}")

    for path in [CT_DEV_FILE, PT_DEV_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")
        print(f"[OK] {path.name}")

    meta = load_primary_dev()
    pools = build_cell_pools(meta)

    print(f"\nPrimary dev cohort: {len(meta)} patients")
    print_cell_table("Verified dev cohort cell table", meta)

    baseline_test_ids = reconstruct_baseline_seed42(meta)
    print_split_summary("Baseline split reconstructed from train_test_split(seed=42)", meta, baseline_test_ids)

    rng = np.random.default_rng(SEED)
    best: Candidate | None = None
    top_unique: dict[tuple[tuple[str, int], ...], float] = {}

    for _ in range(N_ITER):
        test_ids = sample_candidate(rng, pools)
        if not test_ids:
            continue
        if not is_valid_candidate(meta, test_ids):
            continue

        summary = summarize_split(meta, test_ids)
        count_key = get_count_key(test_ids, meta)
        score = float(summary["score"])

        prev = top_unique.get(count_key)
        if prev is None or score < prev:
            top_unique[count_key] = score

        candidate = Candidate(
            score=score,
            test_ids=tuple(sorted(test_ids)),
            count_key=count_key,
        )
        if best is None or candidate.score < best.score:
            best = candidate

    if best is None:
        raise RuntimeError("No valid candidate split found.")

    print(f"\n{'=' * 80}")
    print("Top unique candidate cell allocations")
    print(f"{'=' * 80}")
    for rank, (count_key, score) in enumerate(sorted(top_unique.items(), key=lambda x: x[1])[:10], start=1):
        print(f"\n[{rank}] score={score:.6f}")
        for cell, n_val in count_key:
            print(f"  {cell}: {n_val}")

    best_test_ids = set(best.test_ids)
    print_split_summary("Chosen optimized split", meta, best_test_ids)

    split_map = meta[["PatientID"]].copy()
    split_map["split"] = np.where(split_map["PatientID"].isin(best_test_ids), "test", "train")
    split_map = split_map.sort_values("PatientID").reset_index(drop=True)
    split_map.to_csv(OUT_SPLIT_FILE, index=False)
    save_split_datasets(best_test_ids)

    print(f"\nSaved split map: {OUT_SPLIT_FILE}")
    print(f"Saved CT train set: {CT_TRAIN_FILE}")
    print(f"Saved CT test set:  {CT_TEST_FILE}")
    print(f"Saved PT train set: {PT_TRAIN_FILE}")
    print(f"Saved PT test set:  {PT_TEST_FILE}")


if __name__ == "__main__":
    main()
