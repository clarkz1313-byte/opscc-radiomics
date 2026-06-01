"""
2_apr_t2b_pan_sc5_4.py  --  Task 2B v4

Same RFS-consensus pool as v3 (top-8 PT + top-5 CT).
Extended N = [7, 12]  (minimum imposed by 1-clin + PT_MIN=4 + CT_MIN=2).
GM tournament: 3 survival x 3 HPV = 9 model pairs.
Tests the same GM families used in Task 1 and Task 2 winning pipelines:
  - EST(n=200)  = Task 1 winning coach
  - SVM(a=0.001)= Task 1 winning GM
  - CoxPH       = linear survival (generalises to small CHUS)
  - LR_L2(C=0.5)= Task 2 T68357 primary winner
  - LR_EN       = Task 2 T113130 secondary winner (coach)
  - SVM_L       = Task 2 T113130 secondary GM

Efficient screening: for each feature combo, all 6 models are fitted once per
fold then cross-producted into 9 (oof_ci, oof_auc) pairs. This is 3x faster
than evaluating each pair independently.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import warnings
from itertools import combinations, product
from pathlib import Path

warnings.filterwarnings("ignore")


def _ensure_import(module_name: str, package_name: str | None = None) -> None:
    package = package_name or module_name
    try:
        __import__(module_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])


for _module, _package in [
    ("numpy", None), ("pandas", None),
    ("sklearn", "scikit-learn"), ("sksurv", "scikit-survival"),
]:
    _ensure_import(_module, _package)

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sksurv.ensemble import ExtraSurvivalTrees
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.svm import FastSurvivalSVM
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

SCREEN_CSV = OUT_DIR / "t2b_screen_4.csv"
ALL_CSV = OUT_DIR / "t2b_all_results_4.csv"
TOP20_JOINT_CSV = OUT_DIR / "t2b_top20_joint_4.csv"
TOP20_RFS_CSV = OUT_DIR / "t2b_top20_rfs_4.csv"
TOP20_HPV_CSV = OUT_DIR / "t2b_top20_hpv_4.csv"
LOG_MD = OUT_DIR / "t2b_log_4.md"

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
SEED = 42
N_MIN = 7   # minimum feasible: 1 clin + PT_MIN(4) + CT_MIN(2) = 7
N_MAX = 12
PT_MIN = 4
CT_MIN = 2
ALPHA = 0.60
N_FOLDS = 5
N_REPEATS_SCREEN = 1
N_REPEATS_CONFIRM = 3
N_EST_SCREEN = 100
N_EST_CONFIRM = 200
SOFT_CI_FLOOR = 0.50
SOFT_AUC_FLOOR = 0.66
HARD_CI_FLOOR = 0.54
HARD_AUC_FLOOR = 0.70
TOPK_CONFIRM = 150
N_BOOT = 500
PT_POOL_SIZE = 8
CT_POOL_SIZE = 5
RFS_LOCO_EVENTS_MIN = 3
RFS_LOCO_ENON_MIN = 50
RFS_LOCO_KAPPA = 5.0
CLINICAL_FEATURES = ["Gender_Male"]

# ---------------------------------------------------------------------------
# GM definitions  (covering T1 + T2 winning pipelines)
# ---------------------------------------------------------------------------
# Label convention: EST uses N_EST_SCREEN(100) in Stage 1 and N_EST_CONFIRM(200) in Stage 2.
# Label is just "EST" to avoid implying a fixed tree count in Stage-1 logs.
SURV_GM_CONFIGS = [
    # key, label, fixed_params
    ("EST",   "EST",      {"n_estimators": N_EST_SCREEN}),  # T1 coach; Stage 2 uses N_EST_CONFIRM
    ("SVM",   "SVM_0001", {"alpha": 0.001}),                # T1 winning GM (alpha=0.001)
    ("CoxPH", "CoxPH01",  {"alpha": 0.1}),                  # linear survival (alpha=0.1)
]
HPV_GM_CONFIGS = [
    # key, label, fixed_params
    ("LR_L2", "LR_L2_0.5",  {"C": 0.5, "penalty": "l2", "solver": "lbfgs", "max_iter": 2000}),  # T2 T68357 (C≈0.37)
    ("LR_EN", "LR_EN_1.0",  {"C": 1.0, "penalty": "elasticnet", "solver": "saga",
                              "l1_ratio": 0.5, "max_iter": 5000}),                               # T2 T113130 coach
    ("SVM_L", "SVM_L_001",  {"C": 0.01}),                                                        # T2 T113130 GM
]

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
# Helpers
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
        return max(float(roc_auc_score(y_true, scores)), 1.0 - float(roc_auc_score(y_true, scores)))
    except Exception:
        return 0.5


def _youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        fpr, tpr, thr = roc_curve(y_true, scores)
        return float(thr[int(np.argmax(tpr - fpr))])
    except Exception:
        return 0.5


def _with_trial_no(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "trial_no" in out.columns:
        out = out.drop(columns=["trial_no"])
    out.insert(0, "trial_no", np.arange(1, len(out) + 1, dtype=int))
    return out


def _scale_blocks(tr_df, vl_df, feat_pt, feat_ct):
    def _arr(df, cols):
        return df[cols].to_numpy(dtype=float)
    x_clin_tr, x_pt_tr, x_ct_tr = _arr(tr_df, CLINICAL_FEATURES), _arr(tr_df, feat_pt), _arr(tr_df, feat_ct)
    x_clin_vl, x_pt_vl, x_ct_vl = _arr(vl_df, CLINICAL_FEATURES), _arr(vl_df, feat_pt), _arr(vl_df, feat_ct)
    sc_c, sc_p, sc_t = StandardScaler(), StandardScaler(), StandardScaler()
    x_tr = np.hstack([sc_c.fit_transform(x_clin_tr), sc_p.fit_transform(x_pt_tr), sc_t.fit_transform(x_ct_tr)])
    x_vl = np.hstack([sc_c.transform(x_clin_vl),    sc_p.transform(x_pt_vl),    sc_t.transform(x_ct_vl)])
    return x_tr, x_vl


# ---------------------------------------------------------------------------
# GM factories
# ---------------------------------------------------------------------------
def _make_surv(key: str, params: dict, n_est_override: int | None = None):
    if key == "EST":
        n = n_est_override if n_est_override is not None else params["n_estimators"]
        return ExtraSurvivalTrees(n_estimators=n, random_state=SEED, n_jobs=-1)
    if key == "SVM":
        return FastSurvivalSVM(alpha=params["alpha"], max_iter=1000, tol=1e-4, random_state=SEED)
    if key == "CoxPH":
        return CoxPHSurvivalAnalysis(alpha=params["alpha"])
    raise ValueError(key)


def _make_hpv(key: str, params: dict):
    if key == "LR_L2":
        return LogisticRegression(C=params["C"], penalty="l2", solver="lbfgs",
                                  class_weight="balanced", max_iter=2000, random_state=SEED)
    if key == "LR_EN":
        return LogisticRegression(C=params["C"], penalty="elasticnet", solver="saga",
                                  l1_ratio=params["l1_ratio"], class_weight="balanced",
                                  max_iter=5000, random_state=SEED)
    if key == "SVM_L":
        base = LinearSVC(C=params["C"], class_weight="balanced", max_iter=5000, random_state=SEED)
        return CalibratedClassifierCV(base, cv=3)
    raise ValueError(key)


# ---------------------------------------------------------------------------
# Ranker functions (same as v3)
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
            w_c = 1.0
        elif mode == "loco_epv_cut":
            include = e_c >= RFS_LOCO_EVENTS_MIN and enon >= RFS_LOCO_ENON_MIN
            w_c = 1.0
        else:
            raise ValueError(mode)
        if not include:
            continue
        centre_weights[fold_i] = w_c
        x_val, y_val = x[mask_val], y_surv[mask_val]
        for feat_i in range(n_feats):
            try:
                ci = float(concordance_index_censored(y_val["event"], y_val["time"], x_val[:, feat_i])[0])
                ci_matrix[fold_i, feat_i] = max(ci, 1.0 - ci)
            except Exception:
                ci_matrix[fold_i, feat_i] = 0.5
    valid = centre_weights > 0
    if not np.any(valid):
        return np.full(n_feats, 0.5)
    return np.mean(np.where(np.isnan(ci_matrix[valid]), 0.5, ci_matrix[valid]), axis=0)


def _univar_rfs(train_df: pd.DataFrame, feat_names: list[str]) -> np.ndarray:
    y_surv = Surv.from_arrays(event=train_df["Relapse"].astype(bool), time=train_df["RFS"])
    scores = []
    for feat in feat_names:
        try:
            ci = float(concordance_index_censored(y_surv["event"], y_surv["time"],
                                                  train_df[feat].to_numpy(dtype=float))[0])
            scores.append(max(ci, 1.0 - ci))
        except Exception:
            scores.append(0.5)
    return np.asarray(scores, dtype=float)


def compute_consensus_pool(train_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    all_feats = PAN_PT + PAN_CT
    n_pt = len(PAN_PT)
    print("Building RFS consensus pool (same as v3)...")
    raw = {
        "UNIVAR_RFS": _univar_rfs(train_df, all_feats),
        "LOCO_RFS_evt": _rfs_loco_rank(train_df, all_feats, "loco_evt"),
        "LOCO_RFS_epv_cut": _rfs_loco_rank(train_df, all_feats, "loco_epv_cut"),
    }
    RANKER_NAMES = list(raw.keys())
    pt_ranks = np.zeros((len(RANKER_NAMES), n_pt))
    ct_ranks = np.zeros((len(RANKER_NAMES), len(PAN_CT)))
    for r_i, name in enumerate(RANKER_NAMES):
        sc = raw[name]
        pt_ranks[r_i] = np.argsort(np.argsort(-sc[:n_pt]))
        ct_ranks[r_i] = np.argsort(np.argsort(-sc[n_pt:]))
    pt_pool = [PAN_PT[i] for i in np.argsort(pt_ranks.mean(axis=0))[:PT_POOL_SIZE]]
    ct_pool = [PAN_CT[i] for i in np.argsort(ct_ranks.mean(axis=0))[:CT_POOL_SIZE]]
    print(f"  PT pool ({PT_POOL_SIZE}): {[f.split('_')[0]+'...' for f in pt_pool]}")
    print(f"  CT pool ({CT_POOL_SIZE}): {[f.split('_')[0]+'...' for f in ct_pool]}")
    return pt_pool, ct_pool


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(TRAIN_FILE)
    ext_df = pd.read_csv(EXT_FILE)
    assert len(train_df) == 87, f"Expected 87 train rows, got {len(train_df)}"
    assert len(ext_df) == 27, f"Expected 27 ext rows, got {len(ext_df)}"
    for col in ["CenterID", "HPV_binary", "Relapse"]:
        train_df[col] = train_df[col].astype(int)
        ext_df[col] = ext_df[col].astype(int)
    train_df["RFS"] = train_df["RFS"].astype(float)
    ext_df["RFS"] = ext_df["RFS"].astype(float)
    print(f"Train n={len(train_df)} (HPV-={(train_df['HPV_binary']==0).sum()}, "
          f"Rel={train_df['Relapse'].sum()}) | Ext n={len(ext_df)}")
    return train_df, ext_df


# ---------------------------------------------------------------------------
# Core: evaluate all 9 GM pairs for one feature combo in a single CV pass
# ---------------------------------------------------------------------------
def _eval_combo_all_pairs(
    feat_pt: list[str],
    feat_ct: list[str],
    train_df: pd.DataFrame,
    n_repeats: int,
    est_n: int,
) -> dict[str, tuple[float, float]]:
    """
    Returns dict: pair_key -> (mean_oof_ci, mean_oof_auc).
    Fits all 3 survival and 3 HPV models per fold, then cross-products.
    Stratified on combined label (2*Relapse + HPV_binary) to ensure each fold
    has representative relapse events and HPV classes (reduces fold instability).
    NOTE: unstratified repeats still have some variance at n~17/fold; this is a
    known limitation at this sample size.
    """
    # Combined strat label: 0=no-rel/HPV-, 1=no-rel/HPV+, 2=rel/HPV-, 3=rel/HPV+
    y_strat = (train_df["Relapse"].to_numpy(int) * 2
               + train_df["HPV_binary"].to_numpy(int)).clip(0, 3)
    rkf = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=n_repeats, random_state=SEED)
    # accumulate fold predictions per model
    s_ci_lists: dict[str, list[float]] = {s[1]: [] for s in SURV_GM_CONFIGS}
    h_auc_lists: dict[str, list[float]] = {h[1]: [] for h in HPV_GM_CONFIGS}

    for tr_idx, vl_idx in rkf.split(train_df, y_strat):
        tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
        vl_df = train_df.iloc[vl_idx].reset_index(drop=True)
        x_tr, x_vl = _scale_blocks(tr_df, vl_df, feat_pt, feat_ct)

        y_surv_tr = Surv.from_arrays(event=tr_df["Relapse"].astype(bool), time=tr_df["RFS"])
        y_surv_vl = Surv.from_arrays(event=vl_df["Relapse"].astype(bool), time=vl_df["RFS"])
        y_hpv_tr = tr_df["HPV_binary"].to_numpy(dtype=int)
        y_hpv_vl = vl_df["HPV_binary"].to_numpy(dtype=int)

        # -- Survival GMs --
        for (s_key, s_lbl, s_params) in SURV_GM_CONFIGS:
            try:
                m = _make_surv(s_key, s_params, n_est_override=est_n if s_key == "EST" else None)
                m.fit(x_tr, y_surv_tr)
                risk = m.predict(x_vl)
                s_ci_lists[s_lbl].append(_safe_ci(y_surv_vl, risk))
            except Exception:
                s_ci_lists[s_lbl].append(0.5)

        # -- HPV GMs --
        for (h_key, h_lbl, h_params) in HPV_GM_CONFIGS:
            try:
                m = _make_hpv(h_key, h_params)
                m.fit(x_tr, y_hpv_tr)
                proba = m.predict_proba(x_vl)[:, 1]
                try:
                    auc = float(roc_auc_score(y_hpv_vl, proba))
                except Exception:
                    auc = 0.5
                h_auc_lists[h_lbl].append(auc)
            except Exception:
                h_auc_lists[h_lbl].append(0.5)

    # cross-product
    results: dict[str, tuple[float, float]] = {}
    for (_, s_lbl, _), (_, h_lbl, _) in product(SURV_GM_CONFIGS, HPV_GM_CONFIGS):
        oof_ci = float(np.mean(s_ci_lists[s_lbl]))
        oof_auc = float(np.mean(h_auc_lists[h_lbl]))
        results[f"{s_lbl}|{h_lbl}"] = (oof_ci, oof_auc)
    return results


# ---------------------------------------------------------------------------
# External evaluation (single GM pair)
# ---------------------------------------------------------------------------
def _eval_ext(feat_pt, feat_ct, surv_key, surv_params, hpv_key, hpv_params,
              train_df, ext_df, n_boot: int) -> dict:
    x_tr, x_ext = _scale_blocks(train_df, ext_df, feat_pt, feat_ct)
    y_surv_tr = Surv.from_arrays(event=train_df["Relapse"].astype(bool), time=train_df["RFS"])
    y_surv_ext = Surv.from_arrays(event=ext_df["Relapse"].astype(bool), time=ext_df["RFS"])
    y_hpv_tr = train_df["HPV_binary"].to_numpy(dtype=int)
    y_hpv_ext = ext_df["HPV_binary"].to_numpy(dtype=int)

    try:
        sm = _make_surv(surv_key, surv_params, n_est_override=N_EST_CONFIRM if surv_key == "EST" else None)
        sm.fit(x_tr, y_surv_tr)
        risk_ext = sm.predict(x_ext)
        ext_ci = _safe_ci(y_surv_ext, risk_ext)
    except Exception:
        risk_ext = np.full(len(x_ext), np.nan)
        ext_ci = float("nan")

    try:
        hm = _make_hpv(hpv_key, hpv_params)
        hm.fit(x_tr, y_hpv_tr)
        proba_ext = hm.predict_proba(x_ext)[:, 1]
        ext_auc = float(roc_auc_score(y_hpv_ext, proba_ext))
        thresh = _youden_threshold(y_hpv_ext, proba_ext)
        pred = (proba_ext >= thresh).astype(int)
        ext_ba = float(balanced_accuracy_score(y_hpv_ext, pred))
    except Exception:
        proba_ext = np.full(len(x_ext), np.nan)
        ext_auc = ext_ba = float("nan")

    bci_lo = bci_hi = bauc_lo = bauc_hi = float("nan")
    if n_boot > 0 and not np.isnan(ext_ci):
        rng = np.random.default_rng(SEED)
        n_ext = len(ext_df)
        ci_b, auc_b = [], []
        for _ in range(n_boot):
            idx = rng.integers(0, n_ext, n_ext)
            ci_b.append(_safe_ci(y_surv_ext[idx], risk_ext[idx]))
            try:
                auc_b.append(float(roc_auc_score(y_hpv_ext[idx], proba_ext[idx])))
            except Exception:
                auc_b.append(float("nan"))
        bci_lo, bci_hi = float(np.nanpercentile(ci_b, 2.5)), float(np.nanpercentile(ci_b, 97.5))
        bauc_lo, bauc_hi = float(np.nanpercentile(auc_b, 2.5)), float(np.nanpercentile(auc_b, 97.5))

    return {"ext_ci": ext_ci, "ext_auc": ext_auc, "ext_ba": ext_ba,
            "boot_ci_lo": bci_lo, "boot_ci_hi": bci_hi,
            "boot_auc_lo": bauc_lo, "boot_auc_hi": bauc_hi}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> None:
    started_at = time.time()
    train_df, ext_df = load_data()
    pt_pool, ct_pool = compute_consensus_pool(train_df)

    # Enumerate all feature combos
    all_combos: list[tuple] = []
    for n_total in range(N_MIN, N_MAX + 1):
        for n_pt in range(PT_MIN, min(len(pt_pool), n_total - CT_MIN - 1) + 1):
            n_ct = n_total - 1 - n_pt
            if CT_MIN <= n_ct <= len(ct_pool):
                for pt_f in combinations(pt_pool, n_pt):
                    for ct_f in combinations(ct_pool, n_ct):
                        all_combos.append((n_total, n_pt, n_ct, pt_f, ct_f))

    screen_combos = all_combos[:10] if args.smoke else all_combos
    print(f"\nv4: {len(all_combos):,} feature combos | 9 GM pairs | N=[{N_MIN},{N_MAX}]")
    if args.smoke:
        print(f"[SMOKE] testing first 10 combos only")

    # ===========================================================================
    # STAGE 1: Screening
    # ===========================================================================
    print(f"\n--- Stage 1 ({len(screen_combos):,} combos × 9 GM pairs, "
          f"{N_FOLDS}f×{N_REPEATS_SCREEN}, EST n={N_EST_SCREEN}) ---")
    screen_rows: list[dict] = []
    t0 = time.time()

    for ci, (n_total, n_pt, n_ct, pt_f, ct_f) in enumerate(screen_combos):
        pair_results = _eval_combo_all_pairs(
            list(pt_f), list(ct_f), train_df, N_REPEATS_SCREEN, N_EST_SCREEN
        )
        for pair_key, (oof_ci, oof_auc) in pair_results.items():
            s_lbl, h_lbl = pair_key.split("|")
            screen_rows.append({
                "combo_id": ci,
                "n_total": n_total, "n_pt": n_pt, "n_ct": n_ct,
                "surv_gm": s_lbl, "hpv_gm": h_lbl,
                "pair_key": pair_key,
                "oof_ci_s1": oof_ci, "oof_auc_s1": oof_auc,
                "joint_s1": ALPHA * oof_ci + (1 - ALPHA) * oof_auc,
                "feat_pt": "|".join(pt_f), "feat_ct": "|".join(ct_f),
            })

        if (ci + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (ci + 1) / elapsed
            eta = (len(screen_combos) - ci - 1) / max(rate, 1e-9)
            print(f"  [{ci+1:>5}/{len(screen_combos)}] {elapsed/60:.1f}m  eta={eta/60:.1f}m")

    screen_df = pd.DataFrame(screen_rows)
    screen_df.to_csv(SCREEN_CSV, index=False)
    print(f"Stage 1 done in {(time.time()-t0)/60:.1f} min | {len(screen_df):,} rows")

    if screen_df.empty:
        print("[WARN] Stage 1 produced zero rows. Writing empty outputs and exiting.")
        _with_trial_no(screen_df).to_csv(ALL_CSV, index=False)
        pd.DataFrame().to_csv(TOP20_JOINT_CSV, index=False)
        pd.DataFrame().to_csv(TOP20_RFS_CSV, index=False)
        pd.DataFrame().to_csv(TOP20_HPV_CSV, index=False)
        LOG_MD.write_text("# v4 Log\n\nStage 1 produced zero rows.\n", encoding="utf-8")
        return

    passed_soft = screen_df[
        (screen_df["oof_ci_s1"] >= SOFT_CI_FLOOR) &
        (screen_df["oof_auc_s1"] >= SOFT_AUC_FLOOR)
    ]
    if passed_soft.empty:
        print("[WARN] No soft-floor survivors. Writing screen CSV and exiting without Stage 2.")
        screen_df.to_csv(SCREEN_CSV, index=False)
        pd.DataFrame().to_csv(ALL_CSV, index=False)
        LOG_MD.write_text("# v4 Log\n\nNo soft-floor survivors in Stage 1.\n", encoding="utf-8")
        return

    top_screen = (passed_soft.sort_values("joint_s1", ascending=False)
                  .head(TOPK_CONFIRM).reset_index(drop=True))
    print(f"Soft floor passed: {len(passed_soft):,} | forwarding top-{len(top_screen)} to Stage 2")

    # ===========================================================================
    # STAGE 2: Confirmation
    # ===========================================================================
    print(f"\n--- Stage 2 ({len(top_screen)} rows, {N_FOLDS}f×{N_REPEATS_CONFIRM}="
          f"{N_FOLDS*N_REPEATS_CONFIRM} splits, EST n={N_EST_CONFIRM}) ---")
    confirm_rows: list[dict] = []
    t0 = time.time()

    for _, row in top_screen.iterrows():
        feat_pt = row["feat_pt"].split("|")
        feat_ct = row["feat_ct"].split("|")
        s_lbl = row["surv_gm"]
        h_lbl = row["hpv_gm"]
        # find config dicts
        s_key, _, s_params = next(c for c in SURV_GM_CONFIGS if c[1] == s_lbl)
        h_key, _, h_params = next(c for c in HPV_GM_CONFIGS if c[1] == h_lbl)

        pair_res = _eval_combo_all_pairs(feat_pt, feat_ct, train_df, N_REPEATS_CONFIRM, N_EST_CONFIRM)
        pair_k = f"{s_lbl}|{h_lbl}"
        oof_ci, oof_auc = pair_res.get(pair_k, (0.5, 0.5))
        joint = ALPHA * oof_ci + (1 - ALPHA) * oof_auc
        confirm_rows.append({**row.to_dict(), "oof_ci": oof_ci, "oof_auc": oof_auc, "joint_score": joint})

    confirm_df = pd.DataFrame(confirm_rows)
    if confirm_df.empty:
        print("[WARN] Stage 2 produced zero rows. Writing empty outputs and exiting.")
        pd.DataFrame().to_csv(ALL_CSV, index=False)
        LOG_MD.write_text("# v4 Log\n\nStage 2 produced zero rows.\n", encoding="utf-8")
        return

    passed_hard = confirm_df[
        (confirm_df["oof_ci"] >= HARD_CI_FLOOR) &
        (confirm_df["oof_auc"] >= HARD_AUC_FLOOR)
    ]
    # Hard floor is enforced strictly — no fallback to weaker rows.
    # A zero-survivor result here is the negative conclusion (no dual-task winner exists).
    if passed_hard.empty:
        print(f"[RESULT] Hard-floor survivors: 0/{len(confirm_df)} — negative conclusion for this pool.")
        _with_trial_no(confirm_df).to_csv(ALL_CSV, index=False)
        LOG_MD.write_text(
            f"# v4 Log\n\nHard-floor survivors: 0/{len(confirm_df)}.\n"
            f"Negative conclusion: no dual-task winner clears "
            f"oof_ci>={HARD_CI_FLOOR} AND oof_auc>={HARD_AUC_FLOOR}.\n",
            encoding="utf-8"
        )
        return
    print(f"Hard floor passed: {len(passed_hard)}/{len(confirm_df)} | "
          f"elapsed={(time.time()-t0)/60:.1f}m")

    # ===========================================================================
    # External evaluation
    # ===========================================================================
    print(f"\n--- External eval ({len(passed_hard)} rows, n_boot={N_BOOT}) ---")
    all_rows: list[dict] = []
    t0 = time.time()

    for i, (_, row) in enumerate(passed_hard.iterrows()):
        feat_pt = row["feat_pt"].split("|")
        feat_ct = row["feat_ct"].split("|")
        s_lbl, h_lbl = row["surv_gm"], row["hpv_gm"]
        s_key, _, s_params = next(c for c in SURV_GM_CONFIGS if c[1] == s_lbl)
        h_key, _, h_params = next(c for c in HPV_GM_CONFIGS if c[1] == h_lbl)

        ext = _eval_ext(feat_pt, feat_ct, s_key, s_params, h_key, h_params,
                        train_df, ext_df, N_BOOT)
        all_rows.append({**row.to_dict(), **ext})

        if (i + 1) % 20 == 0:
            print(f"  [{i+1:>4}/{len(passed_hard)}] elapsed={time.time()-t0:.0f}s | "
                  f"last: ext_ci={ext['ext_ci']:.3f} ext_auc={ext['ext_auc']:.3f}")

    # ===========================================================================
    # Outputs
    # ===========================================================================
    result_df = _with_trial_no(pd.DataFrame(all_rows))
    if result_df.empty:
        print("[WARN] External eval produced zero rows. Writing empty outputs.")
        result_df.to_csv(ALL_CSV, index=False)
        pd.DataFrame().to_csv(TOP20_JOINT_CSV, index=False)
        pd.DataFrame().to_csv(TOP20_RFS_CSV, index=False)
        pd.DataFrame().to_csv(TOP20_HPV_CSV, index=False)
        LOG_MD.write_text("# v4 Log\n\nExternal eval produced zero rows.\n", encoding="utf-8")
        return

    result_df.to_csv(ALL_CSV, index=False)
    result_df.sort_values(["joint_score", "ext_ci", "ext_auc"], ascending=False).head(20).to_csv(TOP20_JOINT_CSV, index=False)
    result_df.sort_values(["ext_ci", "joint_score"], ascending=False).head(20).to_csv(TOP20_RFS_CSV, index=False)
    result_df.sort_values(["ext_auc", "joint_score"], ascending=False).head(20).to_csv(TOP20_HPV_CSV, index=False)

    wall = time.time() - started_at
    dual_ok = result_df[
        (result_df["oof_ci"] >= HARD_CI_FLOOR) & (result_df["oof_auc"] >= HARD_AUC_FLOOR) &
        (result_df["ext_ci"] >= 0.60) & (result_df["ext_auc"] >= 0.70)
    ]

    summary = (
        f"=== v4 Summary ===\n"
        f"Pool: {PT_POOL_SIZE} PT + {CT_POOL_SIZE} CT | N=[{N_MIN},{N_MAX}] | "
        f"{len(all_combos):,} combos | 9 GM pairs\n"
        f"Stage-1 passed: {len(passed_soft):,} | Stage-2 passed: {len(passed_hard)} | "
        f"Ext evaluated: {len(result_df)}\n"
        f"Dual-task candidates (ext_ci>=0.60, ext_auc>=0.70): {len(dual_ok)}\n"
        f"Wall time: {wall/60:.1f} min\n\n"
        f"--- Top 5 by OOF joint ---\n"
    )
    for _, r in result_df.sort_values("joint_score", ascending=False).head(5).iterrows():
        summary += (f"  {r['surv_gm']} | {r['hpv_gm']} | N={int(r['n_total'])} "
                    f"PT={int(r['n_pt'])} CT={int(r['n_ct'])} | "
                    f"oof_ci={r['oof_ci']:.3f} oof_auc={r['oof_auc']:.3f} | "
                    f"ext_ci={r['ext_ci']:.3f} ext_auc={r['ext_auc']:.3f}\n")
    summary += "\n--- Best per GM pair (by ext_ci) ---\n"
    if "pair_key" in result_df.columns:
        for pair_key, grp in result_df.groupby("pair_key"):
            best = grp.sort_values("ext_ci", ascending=False).iloc[0]
            summary += (f"  {pair_key:35s}: ext_ci={best['ext_ci']:.3f} "
                        f"ext_auc={best['ext_auc']:.3f} | oof_ci={best['oof_ci']:.3f}\n")

    print("\n" + summary)
    LOG_MD.write_text(
        f"# Task 2B v4 Log\n\n"
        f"- Started: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(started_at))}\n"
        f"- Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- Wall: {wall/60:.1f} min\n\n"
        f"## Summary\n```\n{summary}\n```\n",
        encoding="utf-8"
    )
    print(f"Outputs in {OUT_DIR}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="Test with first 10 combos")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"ROOT={ROOT}\nv4: 9 GM pairs | N=[{N_MIN},{N_MAX}] | pool={PT_POOL_SIZE}+{CT_POOL_SIZE}")
    run(args)


if __name__ == "__main__":
    main()
