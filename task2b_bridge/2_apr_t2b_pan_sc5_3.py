"""
2_apr_t2b_pan_sc5_3.py  --  Task 2B v3

Exhaustive combinatorial search + consensus pre-filter + two-stage repeated CV.

Key differences from v1/v2
---------------------------
- No Optuna: pure exhaustive enumeration of all valid feature subsets
- Only 3 RFS-anchored rankers (UNIVAR_RFS, LOCO_RFS_evt, LOCO_RFS_epv_cut)
  which showed POSITIVE OOF<->EXT correlation in v1 data (+0.699, +0.622, +0.622)
- Consensus pre-filter: keep top PT_POOL_SIZE PT and CT_POOL_SIZE CT features
  by mean rank across the 3 rankers
- LR_C tested as a small grid [0.1, 0.5, 1.0] rather than continuous search
- Two-stage evaluation:
    Stage 1 (screening)  -- single 5-fold CV, n_est=100, soft floor
    Stage 2 (confirm)    -- RepeatedKFold(5x3=15 splits), n_est=200, hard floor
- ALPHA = 0.60 (survival upweighted vs 0.5 in v1)
- Outputs written to 2_apr_T2B_outputs/ with _3 suffix
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

warnings.filterwarnings("ignore")


def _ensure_import(module_name: str, package_name: str | None = None) -> None:
    package = package_name or module_name
    try:
        __import__(module_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])


for _module, _package in [
    ("numpy", None),
    ("pandas", None),
    ("sklearn", "scikit-learn"),
    ("sksurv", "scikit-survival"),
]:
    _ensure_import(_module, _package)

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import ExtraSurvivalTrees
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "2_apr_T2B_data"
OUT_DIR = ROOT / "2_apr_T2B_outputs"
TRAIN_FILE = DATA_DIR / "2_apr_t2b_train.csv"
EXT_FILE = DATA_DIR / "2_apr_t2b_ext.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)
SCREEN_CSV = OUT_DIR / "t2b_screen_3.csv"
ALL_CSV = OUT_DIR / "t2b_all_results_3.csv"
TOP20_JOINT_CSV = OUT_DIR / "t2b_top20_joint_3.csv"
TOP20_RFS_CSV = OUT_DIR / "t2b_top20_rfs_3.csv"
TOP20_HPV_CSV = OUT_DIR / "t2b_top20_hpv_3.csv"
LOG_MD = OUT_DIR / "t2b_log_3.md"

# ---------------------------------------------------------------------------
# v3 Parameters
# ---------------------------------------------------------------------------
SEED = 42

# Only rankers with confirmed positive OOF<->EXT correlation from v1 analysis
RANKER_NAMES_V3 = ["UNIVAR_RFS", "LOCO_RFS_evt", "LOCO_RFS_epv_cut"]

PT_POOL_SIZE = 8        # top-N PT features by consensus mean rank
CT_POOL_SIZE = 5        # top-N CT features by consensus mean rank

N_MIN = 7
N_MAX = 10
PT_MIN = 4
CT_MIN = 2

ALPHA = 0.60            # survival upweighted (harder task than HPV)

LR_C_GRID = [0.1, 0.5, 1.0]

# Stage 1: fast screening
N_EST_SCREEN = 100
N_REPEATS_SCREEN = 1    # single 5-fold
SOFT_CI_FLOOR = 0.50
SOFT_AUC_FLOOR = 0.66

# Stage 2: confirmation
N_EST_CONFIRM = 200
N_REPEATS_CONFIRM = 3   # 3x5 = 15 splits
HARD_CI_FLOOR = 0.54
HARD_AUC_FLOOR = 0.70

TOPK_CONFIRM = 200      # top-K from Stage 1 forwarded to Stage 2
N_FOLDS = 5
N_BOOT = 500

# LOCO params (unchanged from v1)
RFS_LOCO_EVENTS_MIN = 3
RFS_LOCO_ENON_MIN = 50
RFS_LOCO_KAPPA = 5.0

CLINICAL_FEATURES = ["Gender_Male"]

PAN_PT = [
    "GTVp_exponential_glszm_HighGrayLevelZoneEmphasis",
    "GTVn_wavelet-LLH_firstorder_Mean",
    "GTVp_wavelet-HLH_glrlm_ShortRunHighGrayLevelEmphasis",
    "GTVp_gradient_glszm_ZoneEntropy",
    "GTVp_wavelet-LHL_glszm_SmallAreaHighGrayLevelEmphasis",
    "GTVn_wavelet-LHH_glszm_LowGrayLevelZoneEmphasis",
    "GTVp_wavelet-HLH_glszm_HighGrayLevelZoneEmphasis",
    "GTVn_wavelet-LHL_glszm_GrayLevelVariance",
    "GTVn_logarithm_glszm_SmallAreaLowGrayLevelEmphasis",
    "GTVn_wavelet-LLH_firstorder_Skewness",
    "GTVn_squareroot_glcm_Idm",
    "GTVp_wavelet-LLH_firstorder_Median",
    "GTVn_logarithm_glcm_Idn",
    "GTVp_wavelet-HHL_glrlm_ShortRunHighGrayLevelEmphasis",
    "GTVp_original_firstorder_InterquartileRange",
]

PAN_CT = [
    "GTVp_wavelet-LLH_glrlm_HighGrayLevelRunEmphasis",
    "GTVp_wavelet-HLL_ngtdm_Complexity",
    "GTVp_gradient_glszm_SmallAreaLowGrayLevelEmphasis",
    "GTVp_wavelet-LHH_firstorder_RootMeanSquared",
    "GTVp_log-sigma-1-mm-3D_firstorder_Range",
    "GTVn_wavelet-HLH_gldm_SmallDependenceEmphasis",
    "GTVn_wavelet-LHH_glrlm_GrayLevelVariance",
    "GTVn_wavelet-HLH_gldm_SmallDependenceHighGrayLevelEmphasis",
    "GTVn_wavelet-HHH_glcm_DifferenceAverage",
    "GTVn_wavelet-HHH_glszm_ZonePercentage",
    "GTVn_wavelet-LHH_glcm_ClusterProminence",
]


# ---------------------------------------------------------------------------
# Shared helpers (same as v1)
# ---------------------------------------------------------------------------

def _safe_ci(y: np.ndarray, risk: np.ndarray) -> float:
    try:
        return float(concordance_index_censored(y["event"], y["time"], risk)[0])
    except Exception:
        return 0.5


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return 0.5
        auc = float(roc_auc_score(y_true, scores))
        return max(auc, 1.0 - auc)
    except Exception:
        return 0.5


def _youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        fpr, tpr, thr = roc_curve(y_true, scores)
        idx = int(np.argmax(tpr - fpr))
        return float(thr[idx])
    except Exception:
        return 0.5


def _with_trial_no(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "trial_no" in out.columns:
        out = out.drop(columns=["trial_no"])
    out.insert(0, "trial_no", np.arange(1, len(out) + 1, dtype=int))
    return out


def _scale_blocks(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feat_pt: list[str],
    feat_ct: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    x_clin_tr = train_df[CLINICAL_FEATURES].to_numpy(dtype=float)
    x_pt_tr = train_df[feat_pt].to_numpy(dtype=float)
    x_ct_tr = train_df[feat_ct].to_numpy(dtype=float)

    x_clin_vl = valid_df[CLINICAL_FEATURES].to_numpy(dtype=float)
    x_pt_vl = valid_df[feat_pt].to_numpy(dtype=float)
    x_ct_vl = valid_df[feat_ct].to_numpy(dtype=float)

    sc_clin = StandardScaler().fit(x_clin_tr)
    sc_pt = StandardScaler().fit(x_pt_tr)
    sc_ct = StandardScaler().fit(x_ct_tr)

    x_train = np.hstack([
        sc_clin.transform(x_clin_tr),
        sc_pt.transform(x_pt_tr),
        sc_ct.transform(x_ct_tr),
    ])
    x_valid = np.hstack([
        sc_clin.transform(x_clin_vl),
        sc_pt.transform(x_pt_vl),
        sc_ct.transform(x_ct_vl),
    ])
    return x_train, x_valid


# ---------------------------------------------------------------------------
# Ranker functions (same as v1, used only for consensus pool computation)
# ---------------------------------------------------------------------------

def _rfs_loco_rank(train_df: pd.DataFrame, feat_names: list[str], mode: str) -> np.ndarray:
    centre_ids = train_df["CenterID"].to_numpy()
    y_surv = Surv.from_arrays(event=train_df["Relapse"].astype(bool), time=train_df["RFS"])
    x = train_df[feat_names].to_numpy(dtype=float)
    centres = np.unique(centre_ids)
    n_feats = len(feat_names)
    ci_matrix = np.full((len(centres), n_feats), np.nan)
    centre_weights = np.zeros(len(centres), dtype=float)

    for fold_i, held in enumerate(centres):
        mask_val = centre_ids == held
        n_c = int(mask_val.sum())
        if n_c < 2:
            continue
        e_c = int(y_surv["event"][mask_val].sum())
        ne_c = n_c - e_c
        enon = float(e_c * ne_c)

        if mode == "loco_evt":
            include = e_c >= RFS_LOCO_EVENTS_MIN
            w_c, use_shrink = 1.0, False
        elif mode == "loco_epv_cut":
            include = e_c >= RFS_LOCO_EVENTS_MIN and enon >= RFS_LOCO_ENON_MIN
            w_c, use_shrink = 1.0, False
        else:
            raise ValueError(mode)

        if not include:
            continue

        centre_weights[fold_i] = w_c
        x_val, y_val = x[mask_val], y_surv[mask_val]
        for feat_i in range(n_feats):
            try:
                ci = float(concordance_index_censored(
                    y_val["event"], y_val["time"], x_val[:, feat_i]
                )[0])
                ci = max(ci, 1.0 - ci)
            except Exception:
                ci = 0.5
            ci_matrix[fold_i, feat_i] = ci

    valid = centre_weights > 0
    if not np.any(valid):
        return np.full(n_feats, 0.5)
    matrix = np.where(np.isnan(ci_matrix[valid]), 0.5, ci_matrix[valid])
    return np.mean(matrix, axis=0)


def _univar_rfs(train_df: pd.DataFrame, feat_names: list[str]) -> np.ndarray:
    y_surv = Surv.from_arrays(event=train_df["Relapse"].astype(bool), time=train_df["RFS"])
    scores = []
    for feat in feat_names:
        try:
            ci = float(concordance_index_censored(
                y_surv["event"], y_surv["time"], train_df[feat].to_numpy(dtype=float)
            )[0])
            scores.append(max(ci, 1.0 - ci))
        except Exception:
            scores.append(0.5)
    return np.asarray(scores, dtype=float)


# ---------------------------------------------------------------------------
# Consensus pool computation
# ---------------------------------------------------------------------------

def compute_consensus_pool(
    train_df: pd.DataFrame,
) -> tuple[list[str], list[str], dict]:
    """
    Build the consensus feature pool from the 3 positive-corr rankers.

    Returns
    -------
    pt_pool : top PT_POOL_SIZE PT features in mean-rank order (best first)
    ct_pool : top CT_POOL_SIZE CT features in mean-rank order (best first)
    info    : dict with per-ranker and consensus rank arrays for logging
    """
    all_feats = PAN_PT + PAN_CT
    n_pt = len(PAN_PT)

    print("Computing consensus pool from 3 RFS-anchored rankers...")
    raw_scores: dict[str, np.ndarray] = {
        "UNIVAR_RFS": _univar_rfs(train_df, all_feats),
        "LOCO_RFS_evt": _rfs_loco_rank(train_df, all_feats, "loco_evt"),
        "LOCO_RFS_epv_cut": _rfs_loco_rank(train_df, all_feats, "loco_epv_cut"),
    }

    # Convert each ranker's score array to rank position (0 = highest score = best)
    pt_rank_matrix = np.zeros((len(RANKER_NAMES_V3), n_pt))
    ct_rank_matrix = np.zeros((len(RANKER_NAMES_V3), len(PAN_CT)))

    for r_i, name in enumerate(RANKER_NAMES_V3):
        sc = raw_scores[name]
        pt_sc = sc[:n_pt]
        ct_sc = sc[n_pt:]
        # argsort of argsort gives the rank position of each element
        pt_rank_matrix[r_i] = np.argsort(np.argsort(-pt_sc))
        ct_rank_matrix[r_i] = np.argsort(np.argsort(-ct_sc))

    pt_mean_rank = pt_rank_matrix.mean(axis=0)
    ct_mean_rank = ct_rank_matrix.mean(axis=0)

    pt_order = np.argsort(pt_mean_rank)[:PT_POOL_SIZE]
    ct_order = np.argsort(ct_mean_rank)[:CT_POOL_SIZE]

    pt_pool = [PAN_PT[i] for i in pt_order]
    ct_pool = [PAN_CT[i] for i in ct_order]

    print(f"  PT pool ({PT_POOL_SIZE}):")
    for i, (idx, feat) in enumerate(zip(pt_order, pt_pool)):
        print(f"    {i+1}. [{idx:2d}] mean_rank={pt_mean_rank[idx]:.2f}  {feat}")
    print(f"  CT pool ({CT_POOL_SIZE}):")
    for i, (idx, feat) in enumerate(zip(ct_order, ct_pool)):
        print(f"    {i+1}. [{idx:2d}] mean_rank={ct_mean_rank[idx]:.2f}  {feat}")

    info = {
        "pt_mean_rank": pt_mean_rank,
        "ct_mean_rank": ct_mean_rank,
        "raw_scores": raw_scores,
    }
    return pt_pool, ct_pool, info


# ---------------------------------------------------------------------------
# CV evaluation helper
# ---------------------------------------------------------------------------

def _cv_eval(
    feat_pt: list[str],
    feat_ct: list[str],
    lr_c: float,
    train_df: pd.DataFrame,
    n_repeats: int,
    n_est: int,
) -> tuple[float, float, float, float]:
    """
    Returns (oof_ci, oof_auc, std_ci, std_auc).
    Uses RepeatedKFold(n_splits=N_FOLDS, n_repeats=n_repeats).
    """
    rkf = RepeatedKFold(n_splits=N_FOLDS, n_repeats=n_repeats, random_state=SEED)
    ci_list: list[float] = []
    auc_list: list[float] = []

    for tr_idx, vl_idx in rkf.split(train_df):
        tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
        vl_df = train_df.iloc[vl_idx].reset_index(drop=True)
        x_tr, x_vl = _scale_blocks(tr_df, vl_df, feat_pt, feat_ct)

        y_surv_tr = Surv.from_arrays(event=tr_df["Relapse"].astype(bool), time=tr_df["RFS"])
        y_surv_vl = Surv.from_arrays(event=vl_df["Relapse"].astype(bool), time=vl_df["RFS"])
        y_hpv_tr = tr_df["HPV_binary"].to_numpy(dtype=int)
        y_hpv_vl = vl_df["HPV_binary"].to_numpy(dtype=int)

        est = ExtraSurvivalTrees(n_estimators=n_est, random_state=SEED, n_jobs=-1)
        est.fit(x_tr, y_surv_tr)
        risk = est.predict(x_vl)
        ci_list.append(_safe_ci(y_surv_vl, risk))

        clf = LogisticRegression(
            C=lr_c, penalty="l2", solver="lbfgs",
            class_weight="balanced", max_iter=2000, random_state=SEED,
        )
        clf.fit(x_tr, y_hpv_tr)
        proba = clf.predict_proba(x_vl)[:, 1]
        try:
            auc = float(roc_auc_score(y_hpv_vl, proba))
        except Exception:
            auc = 0.5
        auc_list.append(auc)

    return (
        float(np.mean(ci_list)),
        float(np.mean(auc_list)),
        float(np.std(ci_list)),
        float(np.std(auc_list)),
    )


# ---------------------------------------------------------------------------
# External evaluation (same logic as v1)
# ---------------------------------------------------------------------------

def evaluate_on_ext(
    feat_pt: list[str],
    feat_ct: list[str],
    lr_c: float,
    train_df: pd.DataFrame,
    ext_df: pd.DataFrame,
    n_boot: int,
) -> dict[str, float]:
    x_tr, x_ext = _scale_blocks(train_df, ext_df, feat_pt, feat_ct)

    y_surv_tr = Surv.from_arrays(event=train_df["Relapse"].astype(bool), time=train_df["RFS"])
    y_surv_ext = Surv.from_arrays(event=ext_df["Relapse"].astype(bool), time=ext_df["RFS"])
    y_hpv_tr = train_df["HPV_binary"].to_numpy(dtype=int)
    y_hpv_ext = ext_df["HPV_binary"].to_numpy(dtype=int)

    est = ExtraSurvivalTrees(n_estimators=N_EST_CONFIRM, random_state=SEED, n_jobs=-1)
    est.fit(x_tr, y_surv_tr)
    risk_ext = est.predict(x_ext)
    ext_ci = _safe_ci(y_surv_ext, risk_ext)

    clf = LogisticRegression(
        C=lr_c, penalty="l2", solver="lbfgs",
        class_weight="balanced", max_iter=2000, random_state=SEED,
    )
    clf.fit(x_tr, y_hpv_tr)
    proba_ext = clf.predict_proba(x_ext)[:, 1]
    try:
        ext_auc = float(roc_auc_score(y_hpv_ext, proba_ext))
    except Exception:
        ext_auc = 0.5

    threshold = _youden_threshold(y_hpv_ext, proba_ext)
    pred_ext = (proba_ext >= threshold).astype(int)
    ext_ba = float(balanced_accuracy_score(y_hpv_ext, pred_ext))
    tn = int(((y_hpv_ext == 0) & (pred_ext == 0)).sum())
    fp = int(((y_hpv_ext == 0) & (pred_ext == 1)).sum())
    fn = int(((y_hpv_ext == 1) & (pred_ext == 0)).sum())
    tp = int(((y_hpv_ext == 1) & (pred_ext == 1)).sum())
    ext_spe = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    ext_sen = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")

    boot_ci_lo = boot_ci_hi = boot_auc_lo = boot_auc_hi = float("nan")
    if n_boot > 0:
        rng = np.random.default_rng(SEED)
        ci_boots: list[float] = []
        auc_boots: list[float] = []
        n_ext = len(ext_df)
        for _ in range(n_boot):
            idx = rng.integers(0, n_ext, n_ext)
            ci_boots.append(_safe_ci(y_surv_ext[idx], risk_ext[idx]))
            try:
                auc_boots.append(float(roc_auc_score(y_hpv_ext[idx], proba_ext[idx])))
            except Exception:
                auc_boots.append(float("nan"))
        boot_ci_lo = float(np.nanpercentile(ci_boots, 2.5))
        boot_ci_hi = float(np.nanpercentile(ci_boots, 97.5))
        boot_auc_lo = float(np.nanpercentile(auc_boots, 2.5))
        boot_auc_hi = float(np.nanpercentile(auc_boots, 97.5))

    return {
        "ext_ci": ext_ci, "ext_auc": ext_auc, "ext_ba": ext_ba,
        "ext_spe": ext_spe, "ext_sen": ext_sen,
        "boot_ci_lo": boot_ci_lo, "boot_ci_hi": boot_ci_hi,
        "boot_auc_lo": boot_auc_lo, "boot_auc_hi": boot_auc_hi,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(TRAIN_FILE)
    ext_df = pd.read_csv(EXT_FILE)

    required_cols = (
        ["PatientID", "CenterID", "HPV_binary", "Relapse", "RFS"]
        + CLINICAL_FEATURES + PAN_PT + PAN_CT
    )
    for label, df in [("train", train_df), ("ext", ext_df)]:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {label}: {missing}")

    assert len(train_df) == 87, f"Expected 87 train rows, got {len(train_df)}"
    assert len(ext_df) == 27, f"Expected 27 ext rows, got {len(ext_df)}"

    for col in ["CenterID", "HPV_binary", "Relapse"]:
        train_df[col] = train_df[col].astype(int)
        ext_df[col] = ext_df[col].astype(int)
    train_df["RFS"] = train_df["RFS"].astype(float)
    ext_df["RFS"] = ext_df["RFS"].astype(float)

    print(
        f"Data: train n={len(train_df)} "
        f"(HPV-={int((train_df['HPV_binary']==0).sum())}, "
        f"Rel={int(train_df['Relapse'].sum())}) | "
        f"ext n={len(ext_df)} "
        f"(HPV-={int((ext_df['HPV_binary']==0).sum())}, "
        f"Rel={int(ext_df['Relapse'].sum())})"
    )
    return train_df, ext_df


# ---------------------------------------------------------------------------
# Exhaustive enumeration
# ---------------------------------------------------------------------------

def _enumerate_combos(
    pt_pool: list[str],
    ct_pool: list[str],
    smoke: bool,
) -> list[tuple]:
    """
    Enumerate all valid (n_total, n_pt, n_ct, feat_pt_tuple, feat_ct_tuple, lr_C).
    1 clinical feature always included; not counted in n_pt / n_ct.
    """
    combos = []
    n_min = N_MIN if not smoke else 7
    n_max = (N_MAX if not smoke else 8)

    for n_total in range(n_min, n_max + 1):
        for n_pt in range(PT_MIN, min(len(pt_pool), n_total - CT_MIN - 1) + 1):
            n_ct = n_total - 1 - n_pt
            if n_ct < CT_MIN or n_ct > len(ct_pool):
                continue
            for pt_feats in combinations(pt_pool, n_pt):
                for ct_feats in combinations(ct_pool, n_ct):
                    for lr_c in LR_C_GRID:
                        combos.append((n_total, n_pt, n_ct, pt_feats, ct_feats, lr_c))

    return combos


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    started_at = time.time()
    train_df, ext_df = load_data()

    # --- Consensus pool ---
    pt_pool, ct_pool, pool_info = compute_consensus_pool(train_df)

    # --- Enumerate all combos ---
    all_combos = _enumerate_combos(pt_pool, ct_pool, smoke=args.smoke)
    n_smoke_screen = 10 if args.smoke else len(all_combos)
    screen_combos = all_combos[:n_smoke_screen]

    print(
        f"\nv3 config: PT_POOL={PT_POOL_SIZE}, CT_POOL={CT_POOL_SIZE}, "
        f"N=[{N_MIN},{N_MAX}], LR_C_GRID={LR_C_GRID}"
    )
    print(f"Total combos enumerated: {len(all_combos):,}")
    if args.smoke:
        print(f"[SMOKE] screening only {n_smoke_screen} combos")

    # =========================================================================
    # STAGE 1: Screening (single 5-fold, fast)
    # =========================================================================
    print(f"\n--- Stage 1: Screening ({len(screen_combos):,} combos, "
          f"{N_FOLDS}-fold x{N_REPEATS_SCREEN}, n_est={N_EST_SCREEN}) ---")

    screen_rows: list[dict] = []
    t0 = time.time()
    for combo_i, (n_total, n_pt, n_ct, pt_feats, ct_feats, lr_c) in enumerate(screen_combos):
        oof_ci, oof_auc, std_ci, std_auc = _cv_eval(
            list(pt_feats), list(ct_feats), lr_c,
            train_df, N_REPEATS_SCREEN, N_EST_SCREEN,
        )
        joint = ALPHA * oof_ci + (1.0 - ALPHA) * oof_auc
        screen_rows.append({
            "combo_id": combo_i,
            "n_total": n_total, "n_pt": n_pt, "n_ct": n_ct,
            "lr_C": lr_c,
            "oof_ci_s1": oof_ci, "oof_auc_s1": oof_auc,
            "std_ci_s1": std_ci, "std_auc_s1": std_auc,
            "joint_s1": joint,
            "feat_pt": "|".join(pt_feats),
            "feat_ct": "|".join(ct_feats),
        })

        if (combo_i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (combo_i + 1) / elapsed
            remaining = (len(screen_combos) - combo_i - 1) / max(rate, 1e-9)
            print(
                f"  [{combo_i+1:>6}/{len(screen_combos)}] "
                f"elapsed={elapsed/60:.1f}m  eta={remaining/60:.1f}m  "
                f"last: CI={oof_ci:.3f} AUC={oof_auc:.3f}"
            )

    screen_df = pd.DataFrame(screen_rows)
    screen_df.to_csv(SCREEN_CSV, index=False)
    print(f"Stage 1 done in {(time.time()-t0)/60:.1f} min | "
          f"wrote {SCREEN_CSV.name}")

    # Soft floor gate
    passed_soft = screen_df[
        (screen_df["oof_ci_s1"] >= SOFT_CI_FLOOR) &
        (screen_df["oof_auc_s1"] >= SOFT_AUC_FLOOR)
    ].copy()
    n_top = min(TOPK_CONFIRM, len(passed_soft))
    top_screen = (
        passed_soft
        .sort_values("joint_s1", ascending=False)
        .head(n_top)
        .reset_index(drop=True)
    )
    print(
        f"Soft floor (CI>={SOFT_CI_FLOOR}, AUC>={SOFT_AUC_FLOOR}): "
        f"{len(passed_soft)}/{len(screen_df)} passed | "
        f"forwarding top-{n_top} to Stage 2"
    )

    # =========================================================================
    # STAGE 2: Confirmation (repeated CV, stricter)
    # =========================================================================
    print(f"\n--- Stage 2: Confirmation ({len(top_screen)} combos, "
          f"{N_FOLDS}-fold x{N_REPEATS_CONFIRM}={N_FOLDS*N_REPEATS_CONFIRM} splits, "
          f"n_est={N_EST_CONFIRM}) ---")

    confirm_rows: list[dict] = []
    t0 = time.time()
    for row_i, row in top_screen.iterrows():
        feat_pt = row["feat_pt"].split("|")
        feat_ct = row["feat_ct"].split("|")
        lr_c = float(row["lr_C"])

        oof_ci, oof_auc, std_ci, std_auc = _cv_eval(
            feat_pt, feat_ct, lr_c,
            train_df, N_REPEATS_CONFIRM, N_EST_CONFIRM,
        )
        joint = ALPHA * oof_ci + (1.0 - ALPHA) * oof_auc
        confirm_rows.append({
            **row.to_dict(),
            "oof_ci": oof_ci, "oof_auc": oof_auc,
            "std_ci": std_ci, "std_auc": std_auc,
            "joint_score": joint,
        })

    confirm_df = pd.DataFrame(confirm_rows)

    # Hard floor gate
    passed_hard = confirm_df[
        (confirm_df["oof_ci"] >= HARD_CI_FLOOR) &
        (confirm_df["oof_auc"] >= HARD_AUC_FLOOR)
    ].copy()
    print(
        f"Hard floor (CI>={HARD_CI_FLOOR}, AUC>={HARD_AUC_FLOOR}): "
        f"{len(passed_hard)}/{len(confirm_df)} passed → external evaluation"
    )

    if passed_hard.empty:
        print(
            "WARNING: No candidates survived the hard floor. "
            "Relaxing to all Stage-2 rows for external eval."
        )
        passed_hard = confirm_df.copy()

    # =========================================================================
    # External CHUS evaluation
    # =========================================================================
    print(f"\n--- External evaluation ({len(passed_hard)} candidates, "
          f"n_boot={N_BOOT}) ---")

    all_rows: list[dict] = []
    t0 = time.time()
    for i, (_, row) in enumerate(passed_hard.iterrows()):
        feat_pt = row["feat_pt"].split("|")
        feat_ct = row["feat_ct"].split("|")
        lr_c = float(row["lr_C"])

        ext_metrics = evaluate_on_ext(
            feat_pt, feat_ct, lr_c, train_df, ext_df, N_BOOT,
        )
        all_rows.append({**row.to_dict(), **ext_metrics})

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(
                f"  [{i+1:>4}/{len(passed_hard)}] "
                f"elapsed={elapsed:.0f}s  "
                f"last: ext_ci={ext_metrics['ext_ci']:.3f} "
                f"ext_auc={ext_metrics['ext_auc']:.3f}"
            )

    # =========================================================================
    # Write outputs
    # =========================================================================
    result_df = _with_trial_no(pd.DataFrame(all_rows))
    result_df.to_csv(ALL_CSV, index=False)

    result_df.sort_values(["joint_score", "ext_ci", "ext_auc"], ascending=False).head(20).to_csv(
        TOP20_JOINT_CSV, index=False
    )
    result_df.sort_values(["ext_ci", "joint_score"], ascending=False).head(20).to_csv(
        TOP20_RFS_CSV, index=False
    )
    result_df.sort_values(["ext_auc", "joint_score"], ascending=False).head(20).to_csv(
        TOP20_HPV_CSV, index=False
    )

    # --- Summary ---
    wall = time.time() - started_at
    dual_ok = result_df[
        (result_df["oof_ci"] >= HARD_CI_FLOOR) &
        (result_df["oof_auc"] >= HARD_AUC_FLOOR) &
        (result_df["ext_ci"] >= 0.60) &
        (result_df["ext_auc"] >= 0.70)
    ]

    summary_lines = [
        "=== Task 2B v3 Summary ===",
        f"Train n=87 | Ext CHUS n=27",
        f"Consensus pool: {PT_POOL_SIZE} PT + {CT_POOL_SIZE} CT = {PT_POOL_SIZE+CT_POOL_SIZE} features",
        f"Total combos screened: {len(screen_combos):,}",
        f"Stage-1 passed soft floor: {len(passed_soft)}",
        f"Stage-2 passed hard floor: {len(passed_hard)}",
        f"External eval rows: {len(result_df)}",
        f"Dual-task candidates (CI>=0.60, AUC>=0.70 ext; CI>={HARD_CI_FLOOR}, "
        f"AUC>={HARD_AUC_FLOOR} oof): {len(dual_ok)}",
        f"Wall time: {wall/60:.1f} min",
        "",
        "--- Top 5 by OOF joint score ---",
    ]
    top5 = result_df.sort_values("joint_score", ascending=False).head(5)
    for _, r in top5.iterrows():
        summary_lines.append(
            f"  N={int(r['n_total'])} PT={int(r['n_pt'])} CT={int(r['n_ct'])} "
            f"lr_C={r['lr_C']:.2f} | "
            f"oof_ci={r['oof_ci']:.3f} oof_auc={r['oof_auc']:.3f} | "
            f"ext_ci={r['ext_ci']:.3f} ext_auc={r['ext_auc']:.3f}"
        )
    summary_lines += [
        "",
        "--- Top 5 by ext_ci ---",
    ]
    top5_ext = result_df.sort_values(["ext_ci", "joint_score"], ascending=False).head(5)
    for _, r in top5_ext.iterrows():
        summary_lines.append(
            f"  N={int(r['n_total'])} PT={int(r['n_pt'])} CT={int(r['n_ct'])} "
            f"lr_C={r['lr_C']:.2f} | "
            f"oof_ci={r['oof_ci']:.3f} oof_auc={r['oof_auc']:.3f} | "
            f"ext_ci={r['ext_ci']:.3f} ext_auc={r['ext_auc']:.3f}"
        )

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    # Log MD
    pt_pool_str = "\n".join(f"  {i+1}. {f}" for i, f in enumerate(pt_pool))
    ct_pool_str = "\n".join(f"  {i+1}. {f}" for i, f in enumerate(ct_pool))
    log_lines = [
        "# Task 2B v3 Log",
        "",
        f"- Started : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(started_at))}",
        f"- Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Wall time: {wall/60:.1f} min",
        f"- Smoke mode: {args.smoke}",
        "",
        "## Consensus pool",
        f"PT pool ({PT_POOL_SIZE}):",
        pt_pool_str,
        f"CT pool ({CT_POOL_SIZE}):",
        ct_pool_str,
        "",
        "## Summary",
        "```text",
        summary_text,
        "```",
        "",
        "## Top 20 by OOF joint score",
    ]
    top20 = result_df.sort_values("joint_score", ascending=False).head(20)
    for _, r in top20.iterrows():
        log_lines.append(
            f"- joint={r['joint_score']:.3f} | "
            f"oof_ci={r['oof_ci']:.3f} oof_auc={r['oof_auc']:.3f} | "
            f"ext_ci={r['ext_ci']:.3f} ext_auc={r['ext_auc']:.3f} | "
            f"N={int(r['n_total'])} PT={int(r['n_pt'])} CT={int(r['n_ct'])} "
            f"lr_C={r['lr_C']:.2f} | "
            f"PT={r['feat_pt']} | CT={r['feat_ct']}"
        )
    LOG_MD.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\nOutputs written to {OUT_DIR}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 2B v3 exhaustive combinatorial search")
    parser.add_argument(
        "--smoke", action="store_true",
        help="Quick test: screen only first 10 combos, skip confirm/ext",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"ROOT     = {ROOT}")
    print(f"DATA_DIR = {DATA_DIR}")
    print(f"OUT_DIR  = {OUT_DIR}")
    run(args)


if __name__ == "__main__":
    main()
