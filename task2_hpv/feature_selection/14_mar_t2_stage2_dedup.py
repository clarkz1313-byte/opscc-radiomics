# -*- coding: utf-8 -*-
"""
Process Task 2 Stage 2 results for Stage 3 cutoff review.

Deduplicates rows with identical AUC2 / Test2 / Fea2, keeps the first row
encountered (therefore preserving the original Rank ordering), and drops the
timing columns from the processed outputs.

Inputs:
    Mar_2026_task2/13_mar_t2_fs_results/13_mar_task2_stage2_CT_result.csv
    Mar_2026_task2/13_mar_t2_fs_results/13_mar_task2_stage2_PT_result.csv

Outputs:
    Mar_2026_task2/13_mar_t2_fs_results/13_mar_task2_stage2_CT_Processed.csv
    Mar_2026_task2/13_mar_t2_fs_results/13_mar_task2_stage2_PT_Processed.csv

Usage:
    cd "D:/Uppsala thesis"
    python Mar_2026_task2/14_mar_t2_stage3_dedup.py
"""

from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "13_mar_t2_fs_results"

KEY_COLS = ["AUC2", "Test2", "Fea2"]
DROP_COLS = ["Pipeline_s", "Elapsed_s"]


def process_file(in_path: Path, out_path: Path) -> None:
    df = pd.read_csv(in_path)

    keep_cols = [col for col in df.columns if col not in DROP_COLS]
    df = df[keep_cols].copy()

    out = df.drop_duplicates(subset=KEY_COLS, keep="first").copy()
    out.to_csv(out_path, index=False)

    print(f"{in_path.name}: {len(df)} -> {len(out)} rows after dedup")
    print(f"  Output: {out_path.name}")


def main() -> None:
    print("=" * 60)
    print("Task 2 Stage 2 dedup for Stage 3 cutoff review")
    print("Dedup key: AUC2 / Test2 / Fea2")
    print("Keep policy: first row only, original Rank preserved")
    print("Dropped columns: Pipeline_s / Elapsed_s")
    print("=" * 60)

    ct_in = RESULTS_DIR / "13_mar_task2_stage2_CT_result.csv"
    ct_out = RESULTS_DIR / "13_mar_task2_stage2_CT_Processed.csv"
    pt_in = RESULTS_DIR / "13_mar_task2_stage2_PT_result.csv"
    pt_out = RESULTS_DIR / "13_mar_task2_stage2_PT_Processed.csv"

    for in_path, out_path in [(ct_in, ct_out), (pt_in, pt_out)]:
        if not in_path.exists():
            print(f"[SKIP] Missing input: {in_path.name}")
            continue
        process_file(in_path, out_path)

    print("Done.")


if __name__ == "__main__":
    main()
