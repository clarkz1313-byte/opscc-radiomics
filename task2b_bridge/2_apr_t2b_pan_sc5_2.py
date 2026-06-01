"""
2_apr_t2b_pan_sc5_2.py

Stricter Task 2B pan-feature joint-objective SC5 search.
Forked from 2_apr_t2b_pan_sc5.py with:
  - narrower structure range
  - stronger survival weighting
  - narrower LR C range
  - hard internal floors on oof_ci / oof_auc
  - dedicated _2 outputs under Apr_2026_task2B/2_apr_T2B_outputs/
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE_SCRIPT = ROOT / "2_apr_t2b_pan_sc5.py"
OUT_DIR = ROOT / "2_apr_T2B_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_base_module():
    spec = importlib.util.spec_from_file_location("t2b_pan_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load base script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_base_module()

# ---------------------------------------------------------------------------
# Stricter v2 configuration
# ---------------------------------------------------------------------------
base.N_MIN = 7
base.N_MAX = 12
base.PT_MIN = 4
base.CT_MIN = 2
base.ALPHA = 0.70

OOF_CI_MIN = 0.55
OOF_AUC_MIN = 0.70
LR_C_MIN = 1e-3
LR_C_MAX = 1.0

base.SEEDS = [
    {"n_total": 12, "n_pt": 7, "lr_C": 0.32},
    {"n_total": 12, "n_pt": 8, "lr_C": 0.37},
    {"n_total": 8, "n_pt": 4, "lr_C": 0.32},
    {"n_total": 10, "n_pt": 5, "lr_C": 0.32},
    {"n_total": 10, "n_pt": 6, "lr_C": 0.37},
    {"n_total": 9, "n_pt": 4, "lr_C": 0.20},
    {"n_total": 7, "n_pt": 4, "lr_C": 0.10},
    {"n_total": 12, "n_pt": 6, "lr_C": 0.37},
]

base.ALL_CSV = OUT_DIR / "t2b_all_results_2.csv"
base.CKPT_CSV = OUT_DIR / "t2b_checkpoint_2.csv"
base.TOP20_JOINT_CSV = OUT_DIR / "t2b_top20_joint_2.csv"
base.TOP20_RFS_CSV = OUT_DIR / "t2b_top20_rfs_2.csv"
base.TOP20_HPV_CSV = OUT_DIR / "t2b_top20_hpv_2.csv"
base.LOG_MD = OUT_DIR / "t2b_log_2.md"


def make_objective(
    ranker_name: str,
    rankings: dict[str, dict[str, list[int]]],
    train_df,
):
    pt_order = rankings[ranker_name]["pt"]
    ct_order = rankings[ranker_name]["ct"]

    def objective(trial):
        n_total = trial.suggest_int("n_total", base.N_MIN, base.N_MAX)
        n_pt_max = min(len(pt_order), n_total - base.CT_MIN - 1)
        if n_pt_max < base.PT_MIN:
            trial.set_user_attr("rejected_reason", "n_pt_max_below_PT_MIN")
            return -1.0

        n_pt = trial.suggest_int("n_pt", base.PT_MIN, n_pt_max)
        n_ct = n_total - 1 - n_pt
        if n_ct < base.CT_MIN or n_ct > len(ct_order):
            trial.set_user_attr("rejected_reason", "n_ct_out_of_bounds")
            return -1.0

        lr_c = trial.suggest_float("lr_C", LR_C_MIN, LR_C_MAX, log=True)
        feat_pt = [base.PAN_PT[i] for i in pt_order[:n_pt]]
        feat_ct = [base.PAN_CT[i] for i in ct_order[:n_ct]]

        kf = base.KFold(n_splits=base.N_FOLDS, shuffle=True, random_state=base.SEED)
        oof_ci_list: list[float] = []
        oof_auc_list: list[float] = []

        for tr_idx, vl_idx in kf.split(train_df):
            tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
            vl_df = train_df.iloc[vl_idx].reset_index(drop=True)
            x_tr, x_vl = base._scale_blocks(tr_df, vl_df, feat_pt, feat_ct)

            y_surv_tr = base.Surv.from_arrays(event=tr_df["Relapse"].astype(bool), time=tr_df["RFS"])
            y_surv_vl = base.Surv.from_arrays(event=vl_df["Relapse"].astype(bool), time=vl_df["RFS"])
            y_hpv_tr = tr_df["HPV_binary"].to_numpy(dtype=int)
            y_hpv_vl = vl_df["HPV_binary"].to_numpy(dtype=int)

            est = base.ExtraSurvivalTrees(
                n_estimators=base.N_EST,
                random_state=base.SEED,
                n_jobs=-1,
            )
            est.fit(x_tr, y_surv_tr)
            risk = est.predict(x_vl)
            oof_ci_list.append(base._safe_ci(y_surv_vl, risk))

            clf = base.LogisticRegression(
                C=lr_c,
                penalty="l2",
                solver="lbfgs",
                class_weight="balanced",
                max_iter=2000,
                random_state=base.SEED,
            )
            clf.fit(x_tr, y_hpv_tr)
            proba = clf.predict_proba(x_vl)[:, 1]
            try:
                auc = float(base.roc_auc_score(y_hpv_vl, proba))
            except Exception:
                auc = 0.5
            oof_auc_list.append(auc)

        oof_ci = float(base.np.mean(oof_ci_list))
        oof_auc = float(base.np.mean(oof_auc_list))
        fold_std_ci = float(base.np.std(oof_ci_list))
        fold_std_auc = float(base.np.std(oof_auc_list))
        joint = base.ALPHA * oof_ci + (1.0 - base.ALPHA) * oof_auc
        stab_penalty = max(0.0, 1.0 - max(fold_std_ci, fold_std_auc) / base.STD_THRESHOLD)
        score = base.W_PERF * joint + base.W_STAB * stab_penalty

        trial.set_user_attr("oof_ci", oof_ci)
        trial.set_user_attr("oof_auc", oof_auc)
        trial.set_user_attr("joint_score", joint)
        trial.set_user_attr("fold_std_ci", fold_std_ci)
        trial.set_user_attr("fold_std_auc", fold_std_auc)
        trial.set_user_attr("n_total", n_total)
        trial.set_user_attr("n_pt", n_pt)
        trial.set_user_attr("n_ct", n_ct)
        trial.set_user_attr("lr_C", lr_c)
        trial.set_user_attr("feat_pt", "|".join(feat_pt))
        trial.set_user_attr("feat_ct", "|".join(feat_ct))
        trial.set_user_attr("oof_gate_pass", int((oof_ci >= OOF_CI_MIN) and (oof_auc >= OOF_AUC_MIN)))

        if oof_ci < OOF_CI_MIN:
            trial.set_user_attr("rejected_reason", "oof_ci_below_floor")
            return -1.0
        if oof_auc < OOF_AUC_MIN:
            trial.set_user_attr("rejected_reason", "oof_auc_below_floor")
            return -1.0

        return score

    return objective


def _summary_table(df):
    rows = []
    for ranker in base.RANKER_NAMES:
        sub = df[df["ranker"] == ranker].copy()
        if sub.empty:
            continue
        best = sub.sort_values(["joint_score", "ext_ci", "ext_auc"], ascending=False).iloc[0]
        rows.append(
            {
                "ranker": ranker,
                "best_oof_joint": float(best["joint_score"]),
                "best_ext_ci": float(best["ext_ci"]),
                "best_ext_auc": float(best["ext_auc"]),
                "n_total": int(best["n_total"]),
                "n_pt": int(best["n_pt"]),
                "n_ct": int(best["n_ct"]),
            }
        )
    lines = [
        "=== Task 2B Pan-SC5 v2 Summary ===",
        "Training: n=87 (HPV-=26, Relapses=20) | External CHUS: n=27 (HPV-=7, Rel=5)",
        "Feature pool: 1 clin + 15 PT + 11 CT = 27",
        f"Config: N=7..12 | PT_MIN=4 | CT_MIN=2 | ALPHA=0.70 | lr_C in [{LR_C_MIN}, {LR_C_MAX}]",
        f"Internal floors: oof_ci >= {OOF_CI_MIN:.2f} and oof_auc >= {OOF_AUC_MIN:.2f}",
        "",
        f"{'Ranker':20s} {'Best_OOF_joint':>14s} {'Best_ext_CI':>12s} {'Best_ext_AUC':>13s} {'N':>4s} {'n_PT':>5s} {'n_CT':>5s}",
    ]
    for row in rows:
        lines.append(
            f"{row['ranker']:20s} {row['best_oof_joint']:14.3f} {row['best_ext_ci']:12.3f} "
            f"{row['best_ext_auc']:13.3f} {row['n_total']:4d} {row['n_pt']:5d} {row['n_ct']:5d}"
        )
    lines.extend(
        [
            "",
            "Target: ext_CI >= 0.70 AND ext_AUC >= 0.70",
            "Ref:    Task1 CHUS=0.7429 | T68357 CHUS=0.7786",
            "",
            f"Outputs: {OUT_DIR}",
        ]
    )
    return "\n".join(lines), base.pd.DataFrame(rows)


base.make_objective = make_objective
base._summary_table = _summary_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 2B pan-feature joint-objective SC5 search v2")
    parser.add_argument("--smoke", action="store_true", help="Run a short validation pass")
    parser.add_argument("--rankers", type=str, default="", help="Comma-separated ranker subset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"ROOT={ROOT}")
    print(f"DATA_DIR={base.DATA_DIR}")
    print(f"TRAIN_FILE={base.TRAIN_FILE}")
    print(f"EXT_FILE={base.EXT_FILE}")
    print(f"OUT_DIR={OUT_DIR}")
    print(
        "v2 config: "
        f"N={base.N_MIN}..{base.N_MAX} | PT_MIN={base.PT_MIN} | CT_MIN={base.CT_MIN} | "
        f"ALPHA={base.ALPHA:.2f} | lr_C=[{LR_C_MIN}, {LR_C_MAX}] | "
        f"oof floors=({OOF_CI_MIN:.2f}, {OOF_AUC_MIN:.2f})"
    )
    base.run(args)


if __name__ == "__main__":
    main()
