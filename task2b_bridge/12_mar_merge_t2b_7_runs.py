from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "2_apr_T2B_outputs"

ALL_COLUMNS = [
    "trial_no", "combo_id", "n_total", "n_pt", "n_ct", "surv_gm", "hpv_gm", "pair_key",
    "oof_ci_s1", "oof_auc_s1", "joint_s1", "feat_pt", "feat_ct",
    "oof_ci", "oof_auc", "joint_score",
    "ext_ci", "ext_auc", "ext_ba", "ext_spe", "ext_sen",
    "boot_ci_lo", "boot_ci_hi", "boot_auc_lo", "boot_auc_hi",
]

DEFAULT_GROUPS = {
    "7ab": ["7", "7b"],
    "7cd": ["7c", "7d"],
    "7abcd": ["7", "7b", "7c", "7d"],
}


def _all_results_path(tag: str) -> Path:
    return OUT_DIR / f"t2b_all_results_{tag}.csv"


def _top20_joint_path(tag: str) -> Path:
    return OUT_DIR / f"t2b_top20_joint_{tag}.csv"


def _top20_rfs_path(tag: str) -> Path:
    return OUT_DIR / f"t2b_top20_rfs_{tag}.csv"


def _top20_hpv_path(tag: str) -> Path:
    return OUT_DIR / f"t2b_top20_hpv_{tag}.csv"


def _log_path(tag: str) -> Path:
    return OUT_DIR / f"t2b_log_{tag}.md"


def _load_all_results(tag: str) -> pd.DataFrame:
    path = _all_results_path(tag)
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing input file: {path}")
    df = pd.read_csv(path)
    missing = [col for col in ALL_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Input file has missing columns: {path} | missing={missing}")
    return df[ALL_COLUMNS].copy()


def _with_trial_no(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "trial_no" in out.columns:
        out = out.drop(columns=["trial_no"])
    out.insert(0, "trial_no", np.arange(1, len(out) + 1, dtype=int))
    return out


def _merge_runs(tags: list[str]) -> pd.DataFrame:
    frames = []
    for tag in tags:
        df = _load_all_results(tag)
        df.insert(0, "source_tag", tag)
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["combo_id", "pair_key"], keep="first").reset_index(drop=True)
    merged = merged.sort_values(["joint_score", "ext_ci", "ext_auc"], ascending=False).reset_index(drop=True)
    merged = merged.drop(columns=["source_tag"])
    return _with_trial_no(merged)


def _write_outputs(tag: str, df: pd.DataFrame, sources: list[str]) -> None:
    all_csv = _all_results_path(tag)
    top_joint_csv = _top20_joint_path(tag)
    top_rfs_csv = _top20_rfs_path(tag)
    top_hpv_csv = _top20_hpv_path(tag)
    log_md = _log_path(tag)

    df.to_csv(all_csv, index=False)

    top_joint = df.sort_values(["joint_score", "ext_ci", "ext_auc"], ascending=False).head(20)
    top_rfs = df.sort_values(["ext_ci", "joint_score", "ext_auc"], ascending=False).head(20)
    top_hpv = df.sort_values(["ext_auc", "joint_score", "ext_ci"], ascending=False).head(20)
    top_joint.to_csv(top_joint_csv, index=False)
    top_rfs.to_csv(top_rfs_csv, index=False)
    top_hpv.to_csv(top_hpv_csv, index=False)

    dual = df[(df["ext_ci"] >= 0.60) & (df["ext_auc"] >= 0.70)]
    summary_lines = [
        f"# Task 2B merged {tag} Log",
        "",
        f"- Sources merged: {', '.join(sources)}",
        f"- Total merged rows: {len(df):,}",
        f"- Dual-floor candidates (ext_ci >= 0.60 and ext_auc >= 0.70): {len(dual):,}",
        "",
        "## Top 5 by joint_score",
    ]

    for _, row in df.head(5).iterrows():
        summary_lines.append(
            f"- trial_no={int(row['trial_no'])} | combo_id={row['combo_id']} | "
            f"{row['surv_gm']} + {row['hpv_gm']} | "
            f"joint_score={row['joint_score']:.3f} | ext_ci={row['ext_ci']:.3f} | ext_auc={row['ext_auc']:.3f}"
        )

    log_md.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge Task 2B v7-family result files into a combined label.")
    parser.add_argument("--group", choices=sorted(DEFAULT_GROUPS), help="Use a predefined merge group.")
    parser.add_argument("--tags", nargs="+", help="Explicit source tags such as: 7 7b 7c 7d")
    parser.add_argument("--out-tag", help="Output tag such as: 7ab, 7cd, or 7abcd")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.group:
        tags = DEFAULT_GROUPS[args.group]
        out_tag = args.group
    else:
        if not args.tags or not args.out_tag:
            raise SystemExit("Provide either --group <name> or both --tags <...> and --out-tag <tag>.")
        tags = args.tags
        out_tag = args.out_tag

    print(f"Merging tags={tags} -> out_tag={out_tag}")
    merged = _merge_runs(tags)
    _write_outputs(out_tag, merged, tags)
    print(f"Done. Wrote {len(merged):,} rows to {_all_results_path(out_tag)}")


if __name__ == "__main__":
    main()
