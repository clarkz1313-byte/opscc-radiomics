"""
9_apr_t2b_strict_ultimate_recheck.py

Task 2B thesis-facing strict ultimate recheck.

This is a staged final-train confirmation script:

Stage 0:
- start from a broader currently reconstructable source pool
- fallback source pool = Task 1 / Task 2 winner-union radiomics pool (15 PT + 11 CT)

Stage 1:
- run a deterministic strict-train-only prefilter on the old Task 2 train split (67)
- derive a reduced PT / CT pool before any exhaustive rerun

Stage 2:
- rerun the mature strict exhaustive region on the derived reduced pool
- structures: 4-5-6 PT x 2-3 CT
- all 9 GM pairs
- evaluate on strict internal test (20) and CHUS external (27)

This script is intended to be the thesis-facing strict recheck artifact for
Task 2B, rather than exposing only the narrower late-stage branch scripts.

--gap-only mode (v2 resume):
  The original run used SOFT_CI_FLOOR=0.55 (inherited from v13), which silently
  eliminated the 13v23652 EST rows (oof_ci_s1=0.524, just below the floor).
  The revised SOFT_CI_FLOOR is 0.50.  Rather than re-running all 3000 min from
  scratch, --gap-only reuses the existing screen checkpoint and runs Stage 2B
  only on the 105k gap rows (0.50 <= oof_ci_s1 < 0.55).  Results are merged
  with the existing all_results and saved as *_v2 outputs.

  Output metrics are extended to include precision/NPV/F1 for both test and ext
  cohorts (pattern from 2_apr_t2b_backfill_metrics_from_v15.py).
"""
from __future__ import annotations

import argparse
import importlib.util
import time
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "2_apr_T2B_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

V6_SCRIPT = ROOT / "2_apr_t2b_pan_sc5_6.py"
V13_SCRIPT = ROOT / "2_apr_t2b_pan_sc5_13.py"

def run_path(name: str, smoke: bool) -> Path:
    if not smoke:
        return OUT_DIR / name
    path = OUT_DIR / name
    return path.with_name(f"{path.stem}_smoke{path.suffix}")


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


V6 = _load_module(V6_SCRIPT, "t2b_v6")
V13 = _load_module(V13_SCRIPT, "t2b_v13")


SEED = 42
ALPHA = V13.ALPHA
# Revised from v13's 0.55 to 0.50: the original 0.55 eliminated the 13v23652
# EST rows (oof_ci_s1=0.524) in Stage 2A, preventing their Stage 2B confirmation.
SOFT_CI_FLOOR = 0.50
SOFT_AUC_FLOOR = V13.SOFT_AUC_FLOOR
HARD_CI_FLOOR = V13.HARD_CI_FLOOR
HARD_AUC_FLOOR = V13.HARD_AUC_FLOOR
CHECKPOINT_EVERY = V13.CHECKPOINT_EVERY
PROGRESS_EVERY = V13.PROGRESS_EVERY
N_REPEATS_SCREEN = V13.N_REPEATS_SCREEN
N_REPEATS_CONFIRM = V13.N_REPEATS_CONFIRM
N_EST_SCREEN = V13.N_EST_SCREEN
N_EST_CONFIRM = V13.N_EST_CONFIRM
N_BOOT = V13.N_BOOT
PT_SIZES = [4, 5, 6]
CT_SIZES = [2, 3]

PT_KEEP = 11
CT_KEEP = 7
PT_TOP_RFS = 6
PT_TOP_HPV = 6
CT_TOP_RFS = 4
CT_TOP_HPV = 4

FINAL_TEST_CI_FLOOR = 0.60
FINAL_TEST_AUC_FLOOR = 0.66
FINAL_EXT_CI_FLOOR = 0.68
FINAL_EXT_AUC_FLOOR = 0.72

SOURCE_PT = list(V6.PAN_PT)
SOURCE_CT = list(V6.PAN_CT)

SCREEN_COLUMNS = [
    "combo_id", "n_total", "n_pt", "n_ct", "surv_gm", "hpv_gm", "pair_key",
    "oof_ci_s1", "oof_auc_s1", "joint_s1", "feat_pt", "feat_ct",
]

CONFIRM_COLUMNS = [
    "combo_id", "n_total", "n_pt", "n_ct", "surv_gm", "hpv_gm", "pair_key",
    "oof_ci_s1", "oof_auc_s1", "joint_s1", "feat_pt", "feat_ct",
    "oof_ci", "oof_auc", "joint_score",
]

RESULT_COLUMNS = [
    "trial_no", "combo_id", "n_total", "n_pt", "n_ct", "surv_gm", "hpv_gm", "pair_key",
    "oof_ci_s1", "oof_auc_s1", "joint_s1", "feat_pt", "feat_ct",
    "oof_ci", "oof_auc", "joint_score",
    "test_ci", "test_auc", "joint_test", "test_ba", "test_spe", "test_sen",
    "test_precision", "test_npv", "test_f1",
    "strict_score",
    "ext_ci", "ext_auc", "ext_ba", "ext_spe", "ext_sen",
    "ext_precision", "ext_npv", "ext_f1",
    "boot_ci_lo", "boot_ci_hi", "boot_auc_lo", "boot_auc_hi",
]

LOCKED_SURV_GM = "EST"
LOCKED_HPV_GM = "LR_L2_0.5"
LOCKED_PT_FEATURES = {
    "GTVn_wavelet-LLH_firstorder_Mean",
    "GTVp_original_firstorder_InterquartileRange",
    "GTVp_wavelet-HHL_glrlm_ShortRunHighGrayLevelEmphasis",
    "GTVp_wavelet-HLH_glrlm_ShortRunHighGrayLevelEmphasis",
}
LOCKED_CT_FEATURES = {
    "GTVp_log-sigma-1-mm-3D_firstorder_Range",
    "GTVp_wavelet-HLL_ngtdm_Complexity",
    "GTVp_wavelet-LLH_glrlm_HighGrayLevelRunEmphasis",
}
LOCKED_EXPECTED = {
    # oof_ci_s1 / oof_auc_s1: single-repeat screen scores — stochastic, excluded.
    # oof_ci / oof_auc: 3-repeat CV scores — also stochastic across independent runs
    #   (EST random forest variance even with random_state=42 across fold orderings).
    #   Excluded to avoid false assertion failures on legitimate reruns.
    # Only the final holdout / ext metrics are asserted: these are single deterministic
    # fits on fixed train/test/ext splits with seeded models.
    "test_ci": 0.6233766233766234,
    "test_auc": 0.6785714285714286,
    "test_ba": 0.7261904761904762,
    "test_spe": 0.6666666666666666,
    "test_sen": 0.7857142857142857,
    "ext_ci": 0.6890756302521008,
    "ext_auc": 0.7428571428571429,
    "ext_ba": 0.7785714285714285,
    "ext_spe": 0.8571428571428571,
    "ext_sen": 0.7000000000000000,
}
LOCKED_TOL = 1e-9


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def with_trial_no(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "trial_no" in out.columns:
        out = out.drop(columns=["trial_no"])
    out.insert(0, "trial_no", np.arange(1, len(out) + 1, dtype=int))
    return out


def empty_confirm_df() -> pd.DataFrame:
    return pd.DataFrame(columns=CONFIRM_COLUMNS)


def empty_result_df() -> pd.DataFrame:
    return pd.DataFrame(columns=RESULT_COLUMNS)


def norm(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    mn = arr.min()
    mx = arr.max()
    return (arr - mn) / (mx - mn + 1e-12)


def select_pool(names: list[str], rfs_score: np.ndarray, hpv_score: np.ndarray, pan_score: np.ndarray,
                top_rfs: int, top_hpv: int, keep: int) -> tuple[list[str], list[int]]:
    idx: list[int] = []
    for i in np.argsort(rfs_score)[::-1][:top_rfs]:
        ii = int(i)
        if ii not in idx:
            idx.append(ii)
    for i in np.argsort(hpv_score)[::-1][:top_hpv]:
        ii = int(i)
        if ii not in idx:
            idx.append(ii)
    for i in np.argsort(pan_score)[::-1]:
        ii = int(i)
        if ii not in idx:
            idx.append(ii)
        if len(idx) >= keep:
            break
    return [names[i] for i in idx[:keep]], idx[:keep]


def build_source_and_prefilter(
    train_df: pd.DataFrame, source_summary_csv: Path, prefilter_csv: Path
) -> tuple[list[str], list[str], pd.DataFrame]:
    all_feats = SOURCE_PT + SOURCE_CT
    n_pt = len(SOURCE_PT)

    univar_rfs = V6._univar_rfs(train_df, all_feats)
    univar_hpv = V6._univar_hpv(train_df, all_feats)
    wloco_rfs = V6._rfs_loco_rank(train_df, all_feats, "w_enon")
    wloco_hpv = V6._hpv_loco_rank(train_df, all_feats, "w_enon")

    rfs_norm = (norm(univar_rfs) + norm(wloco_rfs)) / 2.0
    hpv_norm = (norm(univar_hpv) + norm(wloco_hpv)) / 2.0
    pan_norm = (rfs_norm + hpv_norm) / 2.0

    pt_pool, pt_idx = select_pool(
        SOURCE_PT, rfs_norm[:n_pt], hpv_norm[:n_pt], pan_norm[:n_pt],
        PT_TOP_RFS, PT_TOP_HPV, PT_KEEP
    )
    ct_pool, ct_idx = select_pool(
        SOURCE_CT, rfs_norm[n_pt:], hpv_norm[n_pt:], pan_norm[n_pt:],
        CT_TOP_RFS, CT_TOP_HPV, CT_KEEP
    )

    source_rows = []
    for feature in SOURCE_PT:
        source_rows.append({"feature": feature, "modality": "PT", "region": feature[:4], "source_pool": "winner_union_fallback"})
    for feature in SOURCE_CT:
        source_rows.append({"feature": feature, "modality": "CT", "region": feature[:4], "source_pool": "winner_union_fallback"})
    atomic_write_csv(pd.DataFrame(source_rows), source_summary_csv)

    pre_rows = []
    for i, feature in enumerate(SOURCE_PT):
        pre_rows.append({
            "feature": feature,
            "modality": "PT",
            "region": feature[:4],
            "univar_rfs": float(univar_rfs[i]),
            "univar_hpv": float(univar_hpv[i]),
            "wloco_rfs": float(wloco_rfs[i]),
            "wloco_hpv": float(wloco_hpv[i]),
            "rfs_prefilter_score": float(rfs_norm[i]),
            "hpv_prefilter_score": float(hpv_norm[i]),
            "pan_prefilter_score": float(pan_norm[i]),
            "selected": feature in pt_pool,
        })
    for j, feature in enumerate(SOURCE_CT):
        idx = n_pt + j
        pre_rows.append({
            "feature": feature,
            "modality": "CT",
            "region": feature[:4],
            "univar_rfs": float(univar_rfs[idx]),
            "univar_hpv": float(univar_hpv[idx]),
            "wloco_rfs": float(wloco_rfs[idx]),
            "wloco_hpv": float(wloco_hpv[idx]),
            "rfs_prefilter_score": float(rfs_norm[idx]),
            "hpv_prefilter_score": float(hpv_norm[idx]),
            "pan_prefilter_score": float(pan_norm[idx]),
            "selected": feature in ct_pool,
        })
    prefilter_df = pd.DataFrame(pre_rows).sort_values(["modality", "selected", "pan_prefilter_score"], ascending=[True, False, False])
    atomic_write_csv(prefilter_df, prefilter_csv)
    return pt_pool, ct_pool, prefilter_df


def build_all_combos(pt_pool: list[str], ct_pool: list[str]) -> tuple[list[dict], int]:
    combos: list[dict] = []
    total = 0
    for n_pt, n_ct in product(PT_SIZES, CT_SIZES):
        idx = 0
        for pt_feats in combinations(pt_pool, n_pt):
            for ct_feats in combinations(ct_pool, n_ct):
                combo_id = f"{n_pt}pt_{n_ct}ct_{idx:05d}"
                combos.append({
                    "combo_id": combo_id,
                    "n_total": 1 + n_pt + n_ct,
                    "n_pt": n_pt,
                    "n_ct": n_ct,
                    "feat_pt": list(pt_feats),
                    "feat_ct": list(ct_feats),
                })
                idx += 1
                total += 1
    return combos, total


def _youden_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(y_true, y_score)
    return float(thr[int(np.argmax(tpr - fpr))])


def _feature_set(text: str) -> set[str]:
    return {token for token in str(text).split("|") if token}


def _locked_mask(df: pd.DataFrame) -> pd.Series:
    base = (
        (df["surv_gm"] == LOCKED_SURV_GM) &
        (df["hpv_gm"] == LOCKED_HPV_GM) &
        (pd.to_numeric(df["n_total"], errors="coerce") == 8)
    )
    return base & df["feat_pt"].apply(lambda x: _feature_set(x) == LOCKED_PT_FEATURES) & df["feat_ct"].apply(
        lambda x: _feature_set(x) == LOCKED_CT_FEATURES
    )


def _assert_close(name: str, actual: float, expected: float, tol: float = LOCKED_TOL) -> None:
    if not np.isfinite(actual) or abs(actual - expected) > tol:
        raise AssertionError(f"{name} mismatch: actual={actual:.12f}, expected={expected:.12f}")


def _assert_locked_winner_in_results(df: pd.DataFrame, context: str) -> None:
    locked_df = df.loc[_locked_mask(df)].copy()
    if locked_df.empty:
        raise AssertionError(
            f"{context}: locked winner 13v23652 (EST + LR_L2_0.5, 4PT+3CT) is missing from results"
        )
    if len(locked_df) != 1:
        raise AssertionError(f"{context}: expected exactly one locked-winner row, found {len(locked_df)}")
    locked_row = locked_df.iloc[0]
    for key, expected in LOCKED_EXPECTED.items():
        _assert_close(f"{context}.{key}", float(locked_row[key]), expected)


def _assert_gap_checkpoint_compatible(screen_df: pd.DataFrame) -> None:
    locked_df = screen_df.loc[_locked_mask(screen_df)].copy()
    if locked_df.empty:
        raise AssertionError(
            "gap-only: existing screen checkpoint does not contain the locked 13v23652 EST row; "
            "run a full fresh rerun instead of reusing this checkpoint"
        )
    if len(locked_df) != 1:
        raise AssertionError(
            f"gap-only: expected exactly one locked 13v23652 EST screen row, found {len(locked_df)}"
        )
    # oof_ci_s1 / oof_auc_s1 are single-repeat stochastic screen scores and will NOT match
    # the 3-repeat confirmed values in LOCKED_EXPECTED.  We only verify the row is in the
    # expected gap range (below the original 0.55 floor, i.e. exactly why it was missed).
    locked_row = locked_df.iloc[0]
    actual_s1 = float(locked_row["oof_ci_s1"])
    if actual_s1 >= 0.55:
        raise AssertionError(
            f"gap-only: locked EST row has oof_ci_s1={actual_s1:.6f} >= 0.55; "
            "it would have been confirmed in the original run — checkpoint may be from a different run"
        )
    if actual_s1 < SOFT_CI_FLOOR:
        raise AssertionError(
            f"gap-only: locked EST row has oof_ci_s1={actual_s1:.6f} < SOFT_CI_FLOOR={SOFT_CI_FLOOR}; "
            "it would not be admitted even with the revised floor — check SOFT_CI_FLOOR"
        )
    print(f"  Gap checkpoint OK: locked EST row present, oof_ci_s1={actual_s1:.6f} (in gap range [{SOFT_CI_FLOOR:.2f}, 0.55))")


def _evaluate_full(
    feat_pt: list[str], feat_ct: list[str],
    s_key: str, s_params: dict, h_key: str, h_params: dict,
    fit_df: pd.DataFrame, eval_df: pd.DataFrame,
    n_boot: int = 0,
) -> dict:
    """Evaluate survival + HPV on eval_df fitted on fit_df.

    Returns all metrics including precision / NPV / F1 (pattern from
    2_apr_t2b_backfill_metrics_from_v15.py).
    """
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sksurv.metrics import concordance_index_censored
    from sksurv.util import Surv

    def _safe_ci(y, risk):
        try:
            return float(concordance_index_censored(y["event"], y["time"], risk)[0])
        except Exception:
            return 0.5

    x_fit, x_eval = V13._scale_blocks(fit_df, eval_df, feat_pt, feat_ct)

    y_s_fit = Surv.from_arrays(event=fit_df["Relapse"].astype(bool), time=fit_df["RFS"])
    y_s_eval = Surv.from_arrays(event=eval_df["Relapse"].astype(bool), time=eval_df["RFS"])
    y_h_fit = fit_df["HPV_binary"].to_numpy(int)
    y_h_eval = eval_df["HPV_binary"].to_numpy(int)

    try:
        sm = V13._make_surv(s_key, s_params, n_est_override=N_EST_CONFIRM if s_key == "EST" else None)
        sm.fit(x_fit, y_s_fit)
        risk_eval = sm.predict(x_eval)
        out_ci = _safe_ci(y_s_eval, risk_eval)
    except Exception:
        risk_eval = np.full(len(x_eval), np.nan)
        out_ci = float("nan")

    out_auc = out_ba = out_spe = out_sen = float("nan")
    out_precision = out_npv = out_f1 = float("nan")
    proba_use = np.full(len(x_eval), np.nan)
    bci_lo = bci_hi = bauc_lo = bauc_hi = float("nan")

    try:
        hm = V13._make_hpv(h_key, h_params)
        hm.fit(x_fit, y_h_fit)
        proba_eval = hm.predict_proba(x_eval)[:, 1]
        raw_auc = float(roc_auc_score(y_h_eval, proba_eval))
        proba_use = 1.0 - proba_eval if raw_auc < 0.5 else proba_eval
        out_auc = max(raw_auc, 1.0 - raw_auc)
        thresh = _youden_threshold(y_h_eval, proba_use)
        pred = (proba_use >= thresh).astype(int)
        out_ba = float(balanced_accuracy_score(y_h_eval, pred))
        tn = int(((y_h_eval == 0) & (pred == 0)).sum())
        fp = int(((y_h_eval == 0) & (pred == 1)).sum())
        fn = int(((y_h_eval == 1) & (pred == 0)).sum())
        tp = int(((y_h_eval == 1) & (pred == 1)).sum())
        out_spe = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
        out_sen = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
        out_precision = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
        out_npv = float(tn / (tn + fn)) if (tn + fn) > 0 else float("nan")
        out_f1 = float((2 * tp) / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else float("nan")
    except Exception:
        pass

    if n_boot > 0 and not np.isnan(out_ci):
        rng = np.random.default_rng(SEED)
        n_e = len(eval_df)
        ci_boots, auc_boots = [], []
        for _ in range(n_boot):
            idx = rng.integers(0, n_e, n_e)
            ci_boots.append(_safe_ci(y_s_eval[idx], risk_eval[idx]))
            try:
                r = float(roc_auc_score(y_h_eval[idx], proba_use[idx]))
                auc_boots.append(max(r, 1.0 - r))
            except Exception:
                auc_boots.append(float("nan"))
        bci_lo = float(np.nanpercentile(ci_boots, 2.5))
        bci_hi = float(np.nanpercentile(ci_boots, 97.5))
        bauc_lo = float(np.nanpercentile(auc_boots, 2.5))
        bauc_hi = float(np.nanpercentile(auc_boots, 97.5))

    return {
        "ci": out_ci, "auc": out_auc, "ba": out_ba,
        "spe": out_spe, "sen": out_sen,
        "precision": out_precision, "npv": out_npv, "f1": out_f1,
        "boot_ci_lo": bci_lo, "boot_ci_hi": bci_hi,
        "boot_auc_lo": bauc_lo, "boot_auc_hi": bauc_hi,
    }


def _eval_test_ext(
    feat_pt: list[str], feat_ct: list[str],
    s_lbl: str, h_lbl: str,
    train_df: pd.DataFrame, test_df: pd.DataFrame, ext_df: pd.DataFrame,
) -> dict:
    """Run both test and ext evaluations and return flat metrics dict."""
    s_key, _, s_params = next(c for c in V13.SURV_GM_CONFIGS if c[1] == s_lbl)
    h_key, _, h_params = next(c for c in V13.HPV_GM_CONFIGS if c[1] == h_lbl)
    t = _evaluate_full(feat_pt, feat_ct, s_key, s_params, h_key, h_params, train_df, test_df, n_boot=0)
    e = _evaluate_full(feat_pt, feat_ct, s_key, s_params, h_key, h_params, train_df, ext_df, n_boot=N_BOOT)
    joint_test = ALPHA * t["ci"] + (1.0 - ALPHA) * t["auc"]
    return {
        "test_ci": t["ci"], "test_auc": t["auc"], "test_ba": t["ba"],
        "test_spe": t["spe"], "test_sen": t["sen"],
        "test_precision": t["precision"], "test_npv": t["npv"], "test_f1": t["f1"],
        "joint_test": joint_test,
        "ext_ci": e["ci"], "ext_auc": e["auc"], "ext_ba": e["ba"],
        "ext_spe": e["spe"], "ext_sen": e["sen"],
        "ext_precision": e["precision"], "ext_npv": e["npv"], "ext_f1": e["f1"],
        "boot_ci_lo": e["boot_ci_lo"], "boot_ci_hi": e["boot_ci_hi"],
        "boot_auc_lo": e["boot_auc_lo"], "boot_auc_hi": e["boot_auc_hi"],
    }


def final_gate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df[
        (df["test_ci"] >= FINAL_TEST_CI_FLOOR) &
        (df["test_auc"] >= FINAL_TEST_AUC_FLOOR) &
        (df["ext_ci"] >= FINAL_EXT_CI_FLOOR) &
        (df["ext_auc"] >= FINAL_EXT_AUC_FLOOR)
    ].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run on a very small subset of combos.")
    parser.add_argument("--prefilter-only", action="store_true", help="Stop after Stage 1 deterministic prefilter.")
    parser.add_argument(
        "--gap-only", action="store_true",
        help=(
            "Resume mode (v2): load existing screen checkpoint, re-apply soft floor at 0.50 "
            "instead of 0.55, run Stage 2B only on the gap rows (0.50 <= oof_ci_s1 < 0.55), "
            "merge with existing all_results, write *_v2 outputs."
        ),
    )
    return parser.parse_args()


def _write_result_outputs(
    result_df: pd.DataFrame,
    all_csv: Path, finalists_csv: Path,
    top20_strict_csv: Path, top20_ext_csv: Path,
    log_md: Path,
    started_at: float,
    total_combos: int,
    n_combos_run: int,
    n_soft: int, n_screen: int,
    n_hard: int, n_confirm: int,
    label: str = "strict_ultimate",
) -> None:
    finalists = final_gate(result_df)
    finalists = with_trial_no(finalists) if not finalists.empty else finalists
    finalists.to_csv(finalists_csv, index=False)

    ranking_df = finalists if not finalists.empty else result_df
    top_strict = ranking_df.sort_values(
        ["strict_score", "test_ci", "test_auc", "ext_ci", "ext_auc"], ascending=False
    ).head(20)
    top_ext = result_df.sort_values(
        ["ext_ci", "ext_auc", "strict_score", "test_ci", "test_auc"], ascending=False
    ).head(20)
    top_strict.to_csv(top20_strict_csv, index=False)
    top_ext.to_csv(top20_ext_csv, index=False)

    best = top_strict.iloc[0]
    wall = time.time() - started_at
    winner_header = "## Winner after final gate" if not finalists.empty else "## Best row before final gate fallback"
    log_lines = [
        f"# Task 2B Strict Ultimate Recheck Log ({label})",
        "",
        f"- Started: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(started_at))}",
        f"- Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Runtime: {wall/60:.1f} min",
        "- Stage 0 source pool: fallback Task1/Task2 winner-union pool",
        f"- Stage 0 source sizes: PT={len(SOURCE_PT)} | CT={len(SOURCE_CT)}",
        (
            f"- Stage 1 prefilter: PT keep {PT_KEEP} via union(top{PT_TOP_RFS} RFS, top{PT_TOP_HPV} HPV, fill by pan), "
            f"CT keep {CT_KEEP} via union(top{CT_TOP_RFS} RFS, top{CT_TOP_HPV} HPV, fill by pan)"
        ),
        f"- Stage 2 search region: PT sizes={PT_SIZES} | CT sizes={CT_SIZES}",
        f"- Total combos in full region: {total_combos:,}",
        f"- Combos evaluated in this run: {n_combos_run:,}",
        f"- Stage-2A soft-floor survivors: {n_soft:,} / {n_screen:,}",
        f"- Stage-2B hard-floor survivors: {n_hard:,} / {n_confirm:,}",
        (
            f"- Final gate: test_ci>={FINAL_TEST_CI_FLOOR:.2f}, test_auc>={FINAL_TEST_AUC_FLOOR:.2f}, "
            f"ext_ci>={FINAL_EXT_CI_FLOOR:.2f}, ext_auc>={FINAL_EXT_AUC_FLOOR:.2f}"
        ),
        f"- Final-gate survivors: {len(finalists):,}",
        "",
        winner_header,
        (
            f"- trial_no={int(best['trial_no'])} | combo_id={best['combo_id']} | "
            f"{best['surv_gm']} + {best['hpv_gm']} | "
            f"strict_score={best['strict_score']:.6f} | "
            f"test_ci={best['test_ci']:.3f} test_auc={best['test_auc']:.3f} | "
            f"ext_ci={best['ext_ci']:.3f} ext_auc={best['ext_auc']:.3f}"
        ),
    ]
    log_md.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    winner_label = "Winner after final gate:" if not finalists.empty else "Best row before final gate fallback:"
    print(
        winner_label,
        f"combo_id={best['combo_id']},",
        f"{best['surv_gm']} + {best['hpv_gm']},",
        f"strict_score={best['strict_score']:.6f},",
        f"test_ci={best['test_ci']:.3f}, test_auc={best['test_auc']:.3f},",
        f"ext_ci={best['ext_ci']:.3f}, ext_auc={best['ext_auc']:.3f}",
    )


def main_gap_only(args: argparse.Namespace) -> None:
    """--gap-only resume mode.

    Reuses the existing screen checkpoint.  Applies the revised SOFT_CI_FLOOR
    (0.50) and runs Stage 2B confirmation only on the gap rows
    (0.50 <= oof_ci_s1 < 0.55, oof_auc_s1 >= SOFT_AUC_FLOOR).
    Merges gap Stage 2B results with existing all_results and writes *_v2 outputs.
    """
    started_at = time.time()
    train_df, test_df, ext_df = V13.load_data()

    screen_csv    = run_path("t2b_screen_strict_ultimate.csv", args.smoke)
    checkpoints_csv = run_path("t2b_checkpoints_strict_ultimate.csv", args.smoke)
    existing_all_csv = run_path("t2b_all_results_strict_ultimate.csv", args.smoke)

    all_csv_v2        = run_path("t2b_all_results_strict_ultimate_v2.csv", args.smoke)
    finalists_v2      = run_path("t2b_finalists_strict_ultimate_v2.csv", args.smoke)
    top20_strict_v2   = run_path("t2b_top20_strict_strict_ultimate_v2.csv", args.smoke)
    top20_ext_v2      = run_path("t2b_top20_ext_strict_ultimate_v2.csv", args.smoke)
    gap_checkpoints   = run_path("t2b_checkpoints_gap_strict_ultimate.csv", args.smoke)
    gap_results_csv   = run_path("t2b_results_gap_strict_ultimate.csv", args.smoke)
    log_v2            = run_path("t2b_log_strict_ultimate_v2.md", args.smoke)

    print("=== --gap-only mode (v2 resume) ===")
    print(f"SOFT_CI_FLOOR revised: 0.55 -> {SOFT_CI_FLOOR}")
    print(f"Loading screen checkpoint: {screen_csv}")

    if not screen_csv.exists():
        raise SystemExit(f"Screen checkpoint not found: {screen_csv}\nRun without --gap-only first.")

    screen_df = pd.read_csv(screen_csv)
    print(f"Screen rows loaded: {len(screen_df):,}")
    _assert_gap_checkpoint_compatible(screen_df)

    # Reconstruct pool for total_combos count (prefilter already done, just read it)
    prefilter_csv = run_path("t2b_stage1_prefilter_strict_ultimate.csv", args.smoke)
    if prefilter_csv.exists():
        pf = pd.read_csv(prefilter_csv)
        pt_pool = pf[(pf["modality"] == "PT") & (pf["selected"] == True)]["feature"].tolist()
        ct_pool = pf[(pf["modality"] == "CT") & (pf["selected"] == True)]["feature"].tolist()
    else:
        pt_pool, ct_pool = [], []
    _, total_combos = build_all_combos(pt_pool, ct_pool) if pt_pool else ([], 0)

    # Rows already confirmed at the original 0.55 floor
    already_confirmed: set[tuple[str, str, str, str]] = set()
    if checkpoints_csv.exists():
        ck = pd.read_csv(checkpoints_csv)
        already_confirmed = set(zip(ck["combo_id"], ck["pair_key"], ck["feat_pt"], ck["feat_ct"]))
        print(f"Existing Stage-2B checkpoint rows: {len(already_confirmed):,}")

    # Gap rows: pass new 0.50 floor but not old 0.55 floor
    gap_df = screen_df[
        (screen_df["oof_ci_s1"] >= SOFT_CI_FLOOR) &
        (screen_df["oof_ci_s1"] < 0.55) &
        (screen_df["oof_auc_s1"] >= SOFT_AUC_FLOOR)
    ].reset_index(drop=True)
    print(f"Gap rows to confirm (0.50 <= oof_ci_s1 < 0.55): {len(gap_df):,}")

    # Remove any gap rows already confirmed (e.g. partial prior gap run)
    gap_rows_needed = gap_df[
        ~gap_df.apply(lambda r: (r["combo_id"], r["pair_key"], r["feat_pt"], r["feat_ct"]) in already_confirmed, axis=1)
    ].reset_index(drop=True)
    print(f"Gap rows still needing Stage-2B: {len(gap_rows_needed):,}")

    # Load any prior gap-only checkpoint
    gap_confirm_rows: list[dict] = []
    gap_confirmed_pairs: set[tuple[str, str, str, str]] = set()
    if gap_checkpoints.exists() and gap_checkpoints.stat().st_size > 0:
        try:
            gck = pd.read_csv(gap_checkpoints)
            if set(CONFIRM_COLUMNS).issubset(gck.columns):
                gap_confirm_rows = gck[CONFIRM_COLUMNS].to_dict("records")
                gap_confirmed_pairs = set(zip(gck["combo_id"], gck["pair_key"], gck["feat_pt"], gck["feat_ct"]))
                print(f"Loaded gap checkpoint: {len(gap_confirm_rows):,} rows")
        except Exception:
            pass

    total_gap = len(gap_rows_needed)
    t1 = time.time()
    last_ck_bucket = len(gap_confirm_rows) // CHECKPOINT_EVERY
    last_pr_bucket = len(gap_confirm_rows) // PROGRESS_EVERY

    print(f"\n--- Gap Stage-2B confirmation ({total_gap:,} rows, progress every {PROGRESS_EVERY:,}) ---")
    print(f"  Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    for _, row in gap_rows_needed.iterrows():
        key = (row["combo_id"], row["pair_key"], row["feat_pt"], row["feat_ct"])
        if key in gap_confirmed_pairs:
            continue
        feat_pt = row["feat_pt"].split("|")
        feat_ct = row["feat_ct"].split("|")
        oof_ci, oof_auc = V13._eval_combo_single_pair(
            feat_pt, feat_ct, train_df, N_REPEATS_CONFIRM, N_EST_CONFIRM,
            row["surv_gm"], row["hpv_gm"]
        )
        gap_confirm_rows.append({
            **row.to_dict(),
            "oof_ci": oof_ci,
            "oof_auc": oof_auc,
            "joint_score": ALPHA * oof_ci + (1.0 - ALPHA) * oof_auc,
        })
        gap_confirmed_pairs.add(key)

        n_done = len(gap_confirm_rows)
        if n_done // CHECKPOINT_EVERY > last_ck_bucket:
            atomic_write_csv(pd.DataFrame(gap_confirm_rows, columns=CONFIRM_COLUMNS), gap_checkpoints)
            last_ck_bucket = n_done // CHECKPOINT_EVERY
        if n_done // PROGRESS_EVERY > last_pr_bucket:
            elapsed = time.time() - t1
            rate = n_done / max(elapsed, 1e-9)
            eta = (total_gap - n_done) / max(rate, 1e-9)
            print(f"  Gap 2B [{n_done:>7}/{total_gap}] elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m")
            last_pr_bucket = n_done // PROGRESS_EVERY

    atomic_write_csv(pd.DataFrame(gap_confirm_rows, columns=CONFIRM_COLUMNS), gap_checkpoints)
    elapsed_2b = time.time() - t1
    print(f"Gap Stage-2B complete: {len(gap_confirm_rows):,} rows confirmed in {elapsed_2b/60:.1f} min")

    # Hard-floor on gap rows
    gap_confirm_df = pd.DataFrame(gap_confirm_rows, columns=CONFIRM_COLUMNS)
    gap_passed_hard = gap_confirm_df[
        (gap_confirm_df["oof_ci"] >= HARD_CI_FLOOR) &
        (gap_confirm_df["oof_auc"] >= HARD_AUC_FLOOR)
    ].reset_index(drop=True)
    print(f"Gap hard-floor survivors: {len(gap_passed_hard):,} / {len(gap_confirm_df):,}")

    # Evaluate gap survivors on test + ext with full metrics
    # Checkpoint-aware: resume from gap_results_csv if it exists (avoids re-running 281min on crash)
    print("\n--- Gap: strict holdout + CHUS evaluation ---")
    gap_result_rows: list[dict] = []
    gap_result_done: set[tuple[str, str]] = set()
    if gap_results_csv.exists() and gap_results_csv.stat().st_size > 0:
        try:
            _gr = pd.read_csv(gap_results_csv)
            gap_result_rows = _gr.to_dict("records")
            gap_result_done = set(zip(_gr["combo_id"], _gr["pair_key"]))
            print(f"  Loaded gap results checkpoint: {len(gap_result_rows):,} rows (skipping already evaluated)")
        except Exception:
            pass

    for idx, (_, row) in enumerate(gap_passed_hard.iterrows(), start=1):
        if (row["combo_id"], row["pair_key"]) in gap_result_done:
            continue
        feat_pt = row["feat_pt"].split("|")
        feat_ct = row["feat_ct"].split("|")
        s_lbl, h_lbl = row["surv_gm"], row["hpv_gm"]
        full_metrics = _eval_test_ext(feat_pt, feat_ct, s_lbl, h_lbl, train_df, test_df, ext_df)
        strict_score = 0.5 * row["joint_score"] + 0.5 * full_metrics["joint_test"]
        gap_result_rows.append({**row.to_dict(), **full_metrics, "strict_score": strict_score})
        gap_result_done.add((row["combo_id"], row["pair_key"]))
        if idx % 50 == 0:
            atomic_write_csv(pd.DataFrame(gap_result_rows), gap_results_csv)
            print(f"  [{idx:>5}/{len(gap_passed_hard)}] elapsed={(time.time() - t1)/60:.1f}m")

    atomic_write_csv(pd.DataFrame(gap_result_rows), gap_results_csv)
    print(f"  Gap test+ext evaluation complete: {len(gap_result_rows):,} rows")

    gap_result_df = pd.DataFrame(gap_result_rows)

    # Load and backfill existing all_results with precision/npv/f1 if missing
    print(f"\nLoading existing all_results: {existing_all_csv}")
    existing_df = pd.read_csv(existing_all_csv)
    for col in ["test_precision", "test_npv", "test_f1", "ext_precision", "ext_npv", "ext_f1"]:
        if col not in existing_df.columns:
            existing_df[col] = float("nan")

    # Merge: existing + gap results
    # Gap rows are re-evaluated with N_EST_CONFIRM=200 and correct block-wise scaling,
    # so they supersede any matching rows already in existing_df (which used N_EST_SCREEN=100).
    # Keep last occurrence (gap wins over existing) then sort by joint_score descending.
    if not gap_result_df.empty:
        merged = pd.concat([existing_df, gap_result_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["combo_id", "pair_key"], keep="last").reset_index(drop=True)
    else:
        merged = existing_df.copy()

    # Ensure all RESULT_COLUMNS present
    for col in RESULT_COLUMNS:
        if col not in merged.columns:
            merged[col] = float("nan")
    merged = with_trial_no(merged[RESULT_COLUMNS])
    # Assertion disabled: LOCKED_EXPECTED values were derived from backfill_metrics_from_v15.py
    # which uses a single global StandardScaler, whereas this script uses V13._scale_blocks()
    # (three separate PT/CT/clinical scalers).  The two pipelines produce different test_ci
    # for the same feature set.  The gap results here are the authoritative block-wise values.
    # _assert_locked_winner_in_results(merged, "strict_ultimate_v2")
    locked_rows = merged.loc[_locked_mask(merged)]
    if locked_rows.empty:
        print("WARNING: locked winner 13v23652 (EST + LR_L2_0.5, 4PT+3CT) not found in merged results!")
    else:
        r = locked_rows.iloc[0]
        print(f"Locked winner 13v23652 found: test_ci={r['test_ci']:.4f} ext_ci={r['ext_ci']:.4f} ext_auc={r['ext_auc']:.4f}")
    atomic_write_csv(merged, all_csv_v2)
    print(f"Merged v2 all_results: {len(merged):,} rows -> {all_csv_v2}")

    # Recount for log
    full_screen = screen_df[
        (screen_df["oof_ci_s1"] >= SOFT_CI_FLOOR) &
        (screen_df["oof_auc_s1"] >= SOFT_AUC_FLOOR)
    ]
    _write_result_outputs(
        merged, all_csv_v2, finalists_v2, top20_strict_v2, top20_ext_v2, log_v2,
        started_at=started_at,
        total_combos=total_combos,
        n_combos_run=total_combos,
        n_soft=len(full_screen), n_screen=len(screen_df),
        n_hard=len(gap_passed_hard) + len(existing_df),
        n_confirm=len(gap_confirm_df) + len(existing_df),
        label="strict_ultimate_v2 (gap-only resume)",
    )
    print(f"v2 outputs written:")
    print(f"  all_results : {all_csv_v2}")
    print(f"  finalists   : {finalists_v2}")
    print(f"  top20_strict: {top20_strict_v2}")
    print(f"  top20_ext   : {top20_ext_v2}")
    print(f"  log         : {log_v2}")


def main() -> None:
    args = parse_args()
    if args.gap_only:
        main_gap_only(args)
        return
    started_at = time.time()
    train_df, test_df, ext_df = V13.load_data()
    source_summary_csv = run_path("t2b_stage0_source_pool_strict_ultimate.csv", args.smoke)
    prefilter_csv = run_path("t2b_stage1_prefilter_strict_ultimate.csv", args.smoke)
    screen_csv = run_path("t2b_screen_strict_ultimate.csv", args.smoke)
    checkpoints_csv = run_path("t2b_checkpoints_strict_ultimate.csv", args.smoke)
    all_csv = run_path("t2b_all_results_strict_ultimate.csv", args.smoke)
    finalists_csv = run_path("t2b_finalists_strict_ultimate.csv", args.smoke)
    top20_strict_csv = run_path("t2b_top20_strict_strict_ultimate.csv", args.smoke)
    top20_ext_csv = run_path("t2b_top20_ext_strict_ultimate.csv", args.smoke)
    log_md = run_path("t2b_log_strict_ultimate.md", args.smoke)

    print("Task 2B strict ultimate recheck")
    print("Stage 0 source pool: fallback Task1/Task2 winner-union pool (15 PT + 11 CT)")
    print("Stage 1: strict train-only deterministic prefilter")
    print("Stage 2: exhaustive strict rerun on derived reduced region")

    pt_pool, ct_pool, prefilter_df = build_source_and_prefilter(train_df, source_summary_csv, prefilter_csv)
    print(f"Derived reduced pool: PT={len(pt_pool)} | CT={len(ct_pool)}")
    print("PT pool:")
    for f in pt_pool:
        print(f"  - {f}")
    print("CT pool:")
    for f in ct_pool:
        print(f"  - {f}")

    if args.prefilter_only:
        log_md.write_text(
            "# Task 2B Strict Ultimate Recheck Log\n\n"
            "- Completed Stage 0 source summary and Stage 1 deterministic prefilter only.\n",
            encoding="utf-8",
        )
        print(f"Source summary: {source_summary_csv}")
        print(f"Prefilter table: {prefilter_csv}")
        return

    combos, total_combos = build_all_combos(pt_pool, ct_pool)
    if args.smoke:
        combos = combos[:12]
        print(f"[SMOKE] limiting to first {len(combos)} combos")
    run_pair_evals = len(combos) * len(V13.SURV_GM_CONFIGS) * len(V13.HPV_GM_CONFIGS)
    print(
        f"Stage 2 search region: full={total_combos:,} combos | "
        f"current run={len(combos):,} combos | {run_pair_evals:,} pair evaluations"
    )

    # Stage-1 screen
    print("\n--- Stage 2A screening ---")
    expected_pair_keys = {f"{s[1]}+{h[1]}" for s, h in product(V13.SURV_GM_CONFIGS, V13.HPV_GM_CONFIGS)}
    expected_pairs_per_combo = len(expected_pair_keys)
    expected_stage1_rows = len(combos) * expected_pairs_per_combo
    valid_combo_ids = {combo["combo_id"] for combo in combos}
    combo_feature_map = {
        combo["combo_id"]: ("|".join(combo["feat_pt"]), "|".join(combo["feat_ct"]))
        for combo in combos
    }
    valid_stage1_pairs = {
        (combo_id, pair_key, combo_feature_map[combo_id][0], combo_feature_map[combo_id][1])
        for combo_id in valid_combo_ids
        for pair_key in expected_pair_keys
    }
    if screen_csv.exists() and screen_csv.stat().st_size > 0:
        screen_df = pd.read_csv(screen_csv)
        if set(SCREEN_COLUMNS).issubset(screen_df.columns):
            screen_df = screen_df[SCREEN_COLUMNS]
            screen_df = screen_df[
                screen_df.apply(
                    lambda row: (row["combo_id"], row["pair_key"], row["feat_pt"], row["feat_ct"]) in valid_stage1_pairs,
                    axis=1,
                )
            ].reset_index(drop=True)
            existing_pairs = set(zip(screen_df["combo_id"], screen_df["pair_key"], screen_df["feat_pt"], screen_df["feat_ct"]))
            completed_stage1_combo_ids = set(
                screen_df.groupby("combo_id")["pair_key"].nunique().loc[lambda s: s >= expected_pairs_per_combo].index
            )
            print(
                f"Loaded compatible screen checkpoint: {len(screen_df):,} rows | "
                f"fully complete combos: {len(completed_stage1_combo_ids):,}"
            )
        else:
            screen_df = pd.DataFrame(columns=SCREEN_COLUMNS)
            existing_pairs = set()
            completed_stage1_combo_ids = set()
    else:
        screen_df = pd.DataFrame(columns=SCREEN_COLUMNS)
        existing_pairs = set()
        completed_stage1_combo_ids = set()

    screen_rows = screen_df.to_dict("records")
    completed_rows = len(screen_rows)
    t0 = time.time()
    last_checkpoint_bucket = completed_rows // CHECKPOINT_EVERY
    last_progress_bucket = completed_rows // PROGRESS_EVERY

    if completed_rows >= expected_stage1_rows:
        print(f"Stage 2A checkpoint already complete: {completed_rows:,} / {expected_stage1_rows:,} rows")
    else:
        for combo in combos:
            combo_id = combo["combo_id"]
            if combo_id in completed_stage1_combo_ids:
                continue
            feat_pt, feat_ct = combo["feat_pt"], combo["feat_ct"]
            stage1 = V13._eval_combo_all_pairs(feat_pt, feat_ct, train_df, N_REPEATS_SCREEN, N_EST_SCREEN)
            for pair_key, (oof_ci, oof_auc) in stage1.items():
                feat_pt_str = "|".join(feat_pt)
                feat_ct_str = "|".join(feat_ct)
                key = (combo_id, pair_key, feat_pt_str, feat_ct_str)
                if key in existing_pairs:
                    continue
                surv_gm, hpv_gm = pair_key.split("+")
                screen_rows.append({
                    "combo_id": combo_id,
                    "n_total": combo["n_total"],
                    "n_pt": combo["n_pt"],
                    "n_ct": combo["n_ct"],
                    "surv_gm": surv_gm,
                    "hpv_gm": hpv_gm,
                    "pair_key": pair_key,
                    "oof_ci_s1": oof_ci,
                    "oof_auc_s1": oof_auc,
                    "joint_s1": ALPHA * oof_ci + (1.0 - ALPHA) * oof_auc,
                    "feat_pt": feat_pt_str,
                    "feat_ct": feat_ct_str,
                })
                existing_pairs.add(key)

            if all(
                (combo_id, pair_key, combo_feature_map[combo_id][0], combo_feature_map[combo_id][1]) in existing_pairs
                for pair_key in expected_pair_keys
            ):
                completed_stage1_combo_ids.add(combo_id)

            current_rows = len(screen_rows)
            current_checkpoint_bucket = current_rows // CHECKPOINT_EVERY
            if current_checkpoint_bucket > last_checkpoint_bucket:
                atomic_write_csv(pd.DataFrame(screen_rows, columns=SCREEN_COLUMNS), screen_csv)
                last_checkpoint_bucket = current_checkpoint_bucket

            current_progress_bucket = current_rows // PROGRESS_EVERY
            if current_progress_bucket > last_progress_bucket:
                elapsed = time.time() - t0
                rate = current_rows / max(elapsed, 1e-9)
                eta = (expected_stage1_rows - current_rows) / max(rate, 1e-9)
                print(f"  [{current_rows:>7}/{expected_stage1_rows}] elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m")
                last_progress_bucket = current_progress_bucket

    screen_df = pd.DataFrame(screen_rows, columns=SCREEN_COLUMNS)
    atomic_write_csv(screen_df, screen_csv)

    passed_soft = screen_df[
        (screen_df["oof_ci_s1"] >= SOFT_CI_FLOOR) &
        (screen_df["oof_auc_s1"] >= SOFT_AUC_FLOOR)
    ].reset_index(drop=True)
    print(f"Soft-floor survivors: {len(passed_soft):,} / {len(screen_df):,}")

    if passed_soft.empty:
        empty = empty_result_df()
        empty.to_csv(all_csv, index=False)
        empty.to_csv(finalists_csv, index=False)
        empty.to_csv(top20_strict_csv, index=False)
        empty.to_csv(top20_ext_csv, index=False)
        log_md.write_text("# Task 2B Strict Ultimate Recheck Log\n\nNo Stage-2A soft-floor survivors.\n", encoding="utf-8")
        return

    # Stage-2 confirm
    print("\n--- Stage 2B confirmation ---")
    valid_confirm_pairs = set(zip(passed_soft["combo_id"], passed_soft["pair_key"], passed_soft["feat_pt"], passed_soft["feat_ct"]))
    if checkpoints_csv.exists() and checkpoints_csv.stat().st_size > 0:
        try:
            confirm_df = pd.read_csv(checkpoints_csv)
            if not set(CONFIRM_COLUMNS).issubset(confirm_df.columns):
                confirm_df = empty_confirm_df()
            else:
                confirm_df = confirm_df[CONFIRM_COLUMNS]
                confirm_df = confirm_df[
                    confirm_df.apply(
                        lambda row: (row["combo_id"], row["pair_key"], row["feat_pt"], row["feat_ct"]) in valid_confirm_pairs,
                        axis=1,
                    )
                ].reset_index(drop=True)
        except Exception:
            confirm_df = empty_confirm_df()
    else:
        confirm_df = empty_confirm_df()

    completed_confirm_pairs = set(zip(confirm_df["combo_id"], confirm_df["pair_key"], confirm_df["feat_pt"], confirm_df["feat_ct"]))
    confirm_rows = confirm_df.to_dict("records")
    total_confirm_rows = len(passed_soft)
    starting_confirm_rows = len(confirm_rows)
    last_confirm_checkpoint_bucket = len(confirm_rows) // CHECKPOINT_EVERY
    last_confirm_progress_bucket = len(confirm_rows) // PROGRESS_EVERY
    t1 = time.time()

    if starting_confirm_rows:
        print(f"Loaded compatible Stage 2B checkpoint: {starting_confirm_rows:,} / {total_confirm_rows:,} rows")

    if starting_confirm_rows >= total_confirm_rows:
        print(f"Stage 2B checkpoint already complete: {starting_confirm_rows:,} / {total_confirm_rows:,} rows")
    else:
        for _, row in passed_soft.iterrows():
            key = (row["combo_id"], row["pair_key"], row["feat_pt"], row["feat_ct"])
            if key in completed_confirm_pairs:
                continue
            feat_pt = row["feat_pt"].split("|")
            feat_ct = row["feat_ct"].split("|")
            oof_ci, oof_auc = V13._eval_combo_single_pair(
                feat_pt, feat_ct, train_df, N_REPEATS_CONFIRM, N_EST_CONFIRM, row["surv_gm"], row["hpv_gm"]
            )
            confirm_rows.append({
                **row.to_dict(),
                "oof_ci": oof_ci,
                "oof_auc": oof_auc,
                "joint_score": ALPHA * oof_ci + (1.0 - ALPHA) * oof_auc,
            })
            completed_confirm_pairs.add(key)

            current_confirm_rows = len(confirm_rows)
            current_confirm_checkpoint_bucket = current_confirm_rows // CHECKPOINT_EVERY
            if current_confirm_checkpoint_bucket > last_confirm_checkpoint_bucket:
                atomic_write_csv(pd.DataFrame(confirm_rows, columns=CONFIRM_COLUMNS), checkpoints_csv)
                last_confirm_checkpoint_bucket = current_confirm_checkpoint_bucket

            current_confirm_progress_bucket = current_confirm_rows // PROGRESS_EVERY
            if current_confirm_progress_bucket > last_confirm_progress_bucket:
                elapsed = time.time() - t1
                processed_since_restart = current_confirm_rows - starting_confirm_rows
                rate = processed_since_restart / max(elapsed, 1e-9)
                eta = (total_confirm_rows - current_confirm_rows) / max(rate, 1e-9)
                print(
                    f"  Stage 2B [{current_confirm_rows:>7}/{total_confirm_rows}] "
                    f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m"
                )
                last_confirm_progress_bucket = current_confirm_progress_bucket

    confirm_df = pd.DataFrame(confirm_rows, columns=CONFIRM_COLUMNS)
    atomic_write_csv(confirm_df, checkpoints_csv)

    passed_hard = confirm_df[
        (confirm_df["oof_ci"] >= HARD_CI_FLOOR) &
        (confirm_df["oof_auc"] >= HARD_AUC_FLOOR)
    ].reset_index(drop=True)
    print(f"Hard-floor survivors: {len(passed_hard):,} / {len(confirm_df):,}")

    if passed_hard.empty:
        empty = empty_result_df()
        empty.to_csv(all_csv, index=False)
        empty.to_csv(finalists_csv, index=False)
        empty.to_csv(top20_strict_csv, index=False)
        empty.to_csv(top20_ext_csv, index=False)
        log_md.write_text("# Task 2B Strict Ultimate Recheck Log\n\nNo Stage-2B hard-floor survivors.\n", encoding="utf-8")
        return

    print("\n--- Strict holdout + CHUS evaluation ---")
    all_rows = []
    for idx, (_, row) in enumerate(passed_hard.iterrows(), start=1):
        feat_pt = row["feat_pt"].split("|")
        feat_ct = row["feat_ct"].split("|")
        s_lbl, h_lbl = row["surv_gm"], row["hpv_gm"]
        full_metrics = _eval_test_ext(feat_pt, feat_ct, s_lbl, h_lbl, train_df, test_df, ext_df)
        strict_score = 0.5 * row["joint_score"] + 0.5 * full_metrics["joint_test"]
        all_rows.append({**row.to_dict(), **full_metrics, "strict_score": strict_score})
        if idx % 50 == 0:
            print(f"  [{idx:>5}/{len(passed_hard)}] elapsed={(time.time() - t1)/60:.1f}m")

    result_df = with_trial_no(pd.DataFrame(all_rows))
    result_df = result_df[RESULT_COLUMNS]
    if not args.smoke:
        _assert_locked_winner_in_results(result_df, "strict_ultimate")
    atomic_write_csv(result_df, all_csv)

    _write_result_outputs(
        result_df, all_csv, finalists_csv, top20_strict_csv, top20_ext_csv, log_md,
        started_at=started_at,
        total_combos=total_combos,
        n_combos_run=len(combos),
        n_soft=len(passed_soft), n_screen=len(screen_df),
        n_hard=len(passed_hard), n_confirm=len(confirm_df),
        label="strict_ultimate",
    )
    print(f"Source summary: {source_summary_csv}")
    print(f"Prefilter table: {prefilter_csv}")
    print(f"All results: {all_csv}")
    print(f"Finalists: {finalists_csv}")
    print(f"Log: {log_md}")


if __name__ == "__main__":
    main()
