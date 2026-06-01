"""
2_apr_t2b_pan_sc5_5.py  --  Task 2B v5

Four feature pools x 9 GM pairs (3 survival x 3 HPV).
Designed for PC i9-12900K (~10-12h).

Pool definitions (what v3/v4 don't cover):
  RFS_consensus  : top-8 PT + top-5 CT by mean rank across
                   [UNIVAR_RFS, LOCO_evt, LOCO_epv_cut]  -- same as v3/v4, for comparison
  UNIVAR_HPV     : top-8 PT + top-5 CT by UNIVAR HPV AUC rank
  UNIVAR_JOINT   : top-8 PT + top-5 CT by 0.5*UNIVAR_RFS + 0.5*UNIVAR_HPV rank
  GBC_HPV        : top-8 PT + top-5 CT by GBC feature importance (T2 INDIV winner ranker)

Run a subset of pools with --pools (space-separated):
  python 2_apr_t2b_pan_sc5_5.py --pools UNIVAR_HPV GBC_HPV

GM pairs: same 9 as v4 (covering T1 + T2 winning pipeline combinations).
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


for _m, _p in [("numpy", None), ("pandas", None),
                ("sklearn", "scikit-learn"), ("sksurv", "scikit-survival")]:
    _ensure_import(_m, _p)

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
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
LOG_MD = OUT_DIR / "t2b_log_5.md"

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
SEED = 42
N_MIN, N_MAX = 7, 12   # N_MIN=7: minimum feasible 1-clin + PT_MIN(4) + CT_MIN(2)
PT_MIN, CT_MIN = 4, 2
ALPHA = 0.60
N_FOLDS = 5
N_REPEATS_SCREEN, N_REPEATS_CONFIRM = 1, 3
N_EST_SCREEN, N_EST_CONFIRM = 100, 200
SOFT_CI_FLOOR, SOFT_AUC_FLOOR = 0.50, 0.66
HARD_CI_FLOOR, HARD_AUC_FLOOR = 0.54, 0.70
TOPK_CONFIRM = 150
N_BOOT = 500
PT_POOL_SIZE, CT_POOL_SIZE = 8, 5
RFS_LOCO_EVENTS_MIN, RFS_LOCO_ENON_MIN, RFS_LOCO_KAPPA = 3, 50, 5.0
HPV_LOCO_HPV_MIN, HPV_LOCO_ENON_MIN, HPV_LOCO_KAPPA = 2, 50, 5.0
GBC_N_EST, GBC_LR_RATE, GBC_DEPTH, GBC_SUB = 100, 0.1, 3, 0.8
CLINICAL_FEATURES = ["Gender_Male"]

# ---------------------------------------------------------------------------
# Pool definitions
# ---------------------------------------------------------------------------
ALL_POOL_NAMES = ["RFS_consensus", "UNIVAR_HPV", "UNIVAR_JOINT", "GBC_HPV"]

# ---------------------------------------------------------------------------
# GM configs (same as v4)
# ---------------------------------------------------------------------------
# EST label is just "EST" (no tree count suffix) to avoid implying a fixed count;
# Stage 1 uses N_EST_SCREEN(100), Stage 2 uses N_EST_CONFIRM(200).
SURV_GM_CONFIGS = [
    ("EST",   "EST",      {"n_estimators": N_EST_SCREEN}),
    ("SVM",   "SVM_0001", {"alpha": 0.001}),
    ("CoxPH", "CoxPH01",  {"alpha": 0.1}),
]
HPV_GM_CONFIGS = [
    ("LR_L2", "LR_L2_0.5", {"C": 0.5,  "penalty": "l2",          "solver": "lbfgs", "max_iter": 2000}),
    ("LR_EN", "LR_EN_1.0", {"C": 1.0,  "penalty": "elasticnet",  "solver": "saga",
                             "l1_ratio": 0.5, "max_iter": 5000}),
    ("SVM_L", "SVM_L_001", {"C": 0.01}),
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
def _safe_ci(y, risk):
    try:
        return float(concordance_index_censored(y["event"], y["time"], risk)[0])
    except Exception:
        return 0.5


def _safe_auc(y_true, scores):
    try:
        if len(np.unique(y_true)) < 2:
            return 0.5
        auc = float(roc_auc_score(y_true, scores))
        return max(auc, 1.0 - auc)
    except Exception:
        return 0.5


def _youden_threshold(y_true, scores):
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
    def _a(df, cols): return df[cols].to_numpy(dtype=float)
    stacks = []
    for cols in [CLINICAL_FEATURES, feat_pt, feat_ct]:
        sc = StandardScaler()
        stacks.append((sc.fit_transform(_a(tr_df, cols)), sc.transform(_a(vl_df, cols))))
    x_tr = np.hstack([s[0] for s in stacks])
    x_vl = np.hstack([s[1] for s in stacks])
    return x_tr, x_vl


def _make_surv(key, params, n_est_override=None):
    if key == "EST":
        n = n_est_override if n_est_override is not None else params["n_estimators"]
        return ExtraSurvivalTrees(n_estimators=n, random_state=SEED, n_jobs=-1)
    if key == "SVM":
        return FastSurvivalSVM(alpha=params["alpha"], max_iter=1000, tol=1e-4, random_state=SEED)
    if key == "CoxPH":
        return CoxPHSurvivalAnalysis(alpha=params["alpha"])
    raise ValueError(key)


def _make_hpv(key, params):
    if key == "LR_L2":
        return LogisticRegression(C=params["C"], penalty="l2", solver="lbfgs",
                                  class_weight="balanced", max_iter=2000, random_state=SEED)
    if key == "LR_EN":
        return LogisticRegression(C=params["C"], penalty="elasticnet", solver="saga",
                                  l1_ratio=params["l1_ratio"], class_weight="balanced",
                                  max_iter=5000, random_state=SEED)
    if key == "SVM_L":
        return CalibratedClassifierCV(
            LinearSVC(C=params["C"], class_weight="balanced", max_iter=5000, random_state=SEED), cv=3)
    raise ValueError(key)


# ---------------------------------------------------------------------------
# Ranker functions
# ---------------------------------------------------------------------------
def _univar_rfs(train_df, feat_names):
    y = Surv.from_arrays(event=train_df["Relapse"].astype(bool), time=train_df["RFS"])
    sc = []
    for f in feat_names:
        try:
            ci = float(concordance_index_censored(y["event"], y["time"], train_df[f].to_numpy(float))[0])
            sc.append(max(ci, 1.0 - ci))
        except Exception:
            sc.append(0.5)
    return np.asarray(sc, dtype=float)


def _univar_hpv(train_df, feat_names):
    y = train_df["HPV_binary"].to_numpy(int)
    return np.asarray([_safe_auc(y, train_df[f].to_numpy(float)) for f in feat_names], dtype=float)


def _rfs_loco_rank(train_df, feat_names, mode):
    cids = train_df["CenterID"].to_numpy()
    y = Surv.from_arrays(event=train_df["Relapse"].astype(bool), time=train_df["RFS"])
    x = train_df[feat_names].to_numpy(float)
    centres = np.unique(cids)
    n = len(feat_names)
    ci_mat = np.full((len(centres), n), np.nan)
    wts = np.zeros(len(centres))
    for i, held in enumerate(centres):
        mask = cids == held
        nc = int(mask.sum())
        if nc < 2: continue
        ec = int(y["event"][mask].sum())
        enon = float(ec * (nc - ec))
        if mode == "loco_evt":
            include = ec >= RFS_LOCO_EVENTS_MIN; w = 1.0
        elif mode == "loco_epv_cut":
            include = ec >= RFS_LOCO_EVENTS_MIN and enon >= RFS_LOCO_ENON_MIN; w = 1.0
        else:
            raise ValueError(mode)
        if not include: continue
        wts[i] = w
        for fi in range(n):
            try:
                ci = float(concordance_index_censored(y["event"][mask], y["time"][mask], x[mask, fi])[0])
                ci_mat[i, fi] = max(ci, 1.0 - ci)
            except Exception:
                ci_mat[i, fi] = 0.5
    valid = wts > 0
    if not np.any(valid): return np.full(n, 0.5)
    return np.mean(np.where(np.isnan(ci_mat[valid]), 0.5, ci_mat[valid]), axis=0)


def _gbc_hpv_rank(train_df, feat_names):
    x = train_df[feat_names].to_numpy(float)
    y = train_df["HPV_binary"].to_numpy(int)
    sc = StandardScaler().fit(x)
    try:
        gbc = GradientBoostingClassifier(n_estimators=GBC_N_EST, learning_rate=GBC_LR_RATE,
                                         max_depth=GBC_DEPTH, subsample=GBC_SUB, random_state=SEED)
        gbc.fit(sc.transform(x), y)
        return gbc.feature_importances_
    except Exception:
        return np.full(len(feat_names), 1.0 / len(feat_names))


# ---------------------------------------------------------------------------
# Pool computation
# ---------------------------------------------------------------------------
def compute_pool(pool_name: str, train_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    all_feats = PAN_PT + PAN_CT
    n_pt = len(PAN_PT)
    print(f"  Computing pool '{pool_name}'...")

    if pool_name == "RFS_consensus":
        scores = {
            "UNIVAR_RFS":    _univar_rfs(train_df, all_feats),
            "LOCO_evt":      _rfs_loco_rank(train_df, all_feats, "loco_evt"),
            "LOCO_epv_cut":  _rfs_loco_rank(train_df, all_feats, "loco_epv_cut"),
        }
        weights = [1/3, 1/3, 1/3]

    elif pool_name == "UNIVAR_HPV":
        scores = {"UNIVAR_HPV": _univar_hpv(train_df, all_feats)}
        weights = [1.0]

    elif pool_name == "UNIVAR_JOINT":
        rfs_sc  = _univar_rfs(train_df, all_feats)
        hpv_sc  = _univar_hpv(train_df, all_feats)
        scores  = {"UNIVAR_RFS": rfs_sc, "UNIVAR_HPV": hpv_sc}
        weights = [0.5, 0.5]

    elif pool_name == "GBC_HPV":
        scores = {"GBC_HPV": _gbc_hpv_rank(train_df, all_feats)}
        weights = [1.0]

    else:
        raise ValueError(f"Unknown pool: {pool_name}")

    sc_list = list(scores.values())
    # normalise each ranker to [0,1] before weighting
    def _norm(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-12)
    combined = sum(w * _norm(sc) for w, sc in zip(weights, sc_list))

    pt_combined = combined[:n_pt]
    ct_combined = combined[n_pt:]
    pt_order = np.argsort(pt_combined)[::-1][:PT_POOL_SIZE]
    ct_order = np.argsort(ct_combined)[::-1][:CT_POOL_SIZE]

    pt_pool = [PAN_PT[i] for i in pt_order]
    ct_pool = [PAN_CT[i] for i in ct_order]
    print(f"    PT: {[f[:30] for f in pt_pool]}")
    print(f"    CT: {[f[:30] for f in ct_pool]}")
    return pt_pool, ct_pool


# ---------------------------------------------------------------------------
# Core: evaluate all 9 GM pairs for one feature combo (single CV pass)
# ---------------------------------------------------------------------------
def _eval_combo_all_pairs(feat_pt, feat_ct, train_df, n_repeats, est_n):
    """
    Returns dict: pair_key -> (mean_oof_ci, mean_oof_auc).
    Stratified on combined label (2*Relapse + HPV_binary) to ensure each fold
    has representative relapse events and HPV classes (reduces fold instability).
    NOTE: variance at n~17/fold is a known limitation at this sample size.
    """
    y_strat = (train_df["Relapse"].to_numpy(int) * 2
               + train_df["HPV_binary"].to_numpy(int)).clip(0, 3)
    rkf = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=n_repeats, random_state=SEED)
    s_ci  = {s[1]: [] for s in SURV_GM_CONFIGS}
    h_auc = {h[1]: [] for h in HPV_GM_CONFIGS}

    for tr_idx, vl_idx in rkf.split(train_df, y_strat):
        tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
        vl_df = train_df.iloc[vl_idx].reset_index(drop=True)
        x_tr, x_vl = _scale_blocks(tr_df, vl_df, feat_pt, feat_ct)
        y_s_tr = Surv.from_arrays(event=tr_df["Relapse"].astype(bool), time=tr_df["RFS"])
        y_s_vl = Surv.from_arrays(event=vl_df["Relapse"].astype(bool), time=vl_df["RFS"])
        y_h_tr = tr_df["HPV_binary"].to_numpy(int)
        y_h_vl = vl_df["HPV_binary"].to_numpy(int)

        for (sk, sl, sp) in SURV_GM_CONFIGS:
            try:
                m = _make_surv(sk, sp, n_est_override=est_n if sk == "EST" else None)
                m.fit(x_tr, y_s_tr)
                s_ci[sl].append(_safe_ci(y_s_vl, m.predict(x_vl)))
            except Exception:
                s_ci[sl].append(0.5)

        for (hk, hl, hp) in HPV_GM_CONFIGS:
            try:
                m = _make_hpv(hk, hp)
                m.fit(x_tr, y_h_tr)
                proba = m.predict_proba(x_vl)[:, 1]
                try:
                    auc = float(roc_auc_score(y_h_vl, proba))
                except Exception:
                    auc = 0.5
                h_auc[hl].append(auc)
            except Exception:
                h_auc[hl].append(0.5)

    results = {}
    for (_, sl, _), (_, hl, _) in product(SURV_GM_CONFIGS, HPV_GM_CONFIGS):
        ci  = float(np.mean(s_ci[sl]))
        auc = float(np.mean(h_auc[hl]))
        results[f"{sl}|{hl}"] = (ci, auc)
    return results


# ---------------------------------------------------------------------------
# External evaluation (single pair)
# ---------------------------------------------------------------------------
def _eval_ext(feat_pt, feat_ct, s_key, s_params, h_key, h_params, train_df, ext_df, n_boot):
    x_tr, x_ext = _scale_blocks(train_df, ext_df, feat_pt, feat_ct)
    y_s_tr  = Surv.from_arrays(event=train_df["Relapse"].astype(bool), time=train_df["RFS"])
    y_s_ext = Surv.from_arrays(event=ext_df["Relapse"].astype(bool), time=ext_df["RFS"])
    y_h_tr  = train_df["HPV_binary"].to_numpy(int)
    y_h_ext = ext_df["HPV_binary"].to_numpy(int)

    try:
        sm = _make_surv(s_key, s_params, n_est_override=N_EST_CONFIRM if s_key == "EST" else None)
        sm.fit(x_tr, y_s_tr)
        risk_ext = sm.predict(x_ext)
        ext_ci = _safe_ci(y_s_ext, risk_ext)
    except Exception:
        risk_ext = np.full(len(x_ext), np.nan)
        ext_ci = float("nan")

    try:
        hm = _make_hpv(h_key, h_params)
        hm.fit(x_tr, y_h_tr)
        proba_ext = hm.predict_proba(x_ext)[:, 1]
        ext_auc = float(roc_auc_score(y_h_ext, proba_ext))
        thresh = _youden_threshold(y_h_ext, proba_ext)
        ext_ba = float(balanced_accuracy_score(y_h_ext, (proba_ext >= thresh).astype(int)))
    except Exception:
        proba_ext = np.full(len(x_ext), np.nan)
        ext_auc = ext_ba = float("nan")

    bci_lo = bci_hi = bauc_lo = bauc_hi = float("nan")
    if n_boot > 0 and not np.isnan(ext_ci):
        rng = np.random.default_rng(SEED)
        n_e = len(ext_df)
        cib, ab = [], []
        for _ in range(n_boot):
            idx = rng.integers(0, n_e, n_e)
            cib.append(_safe_ci(y_s_ext[idx], risk_ext[idx]))
            try:
                ab.append(float(roc_auc_score(y_h_ext[idx], proba_ext[idx])))
            except Exception:
                ab.append(float("nan"))
        bci_lo, bci_hi = float(np.nanpercentile(cib, 2.5)), float(np.nanpercentile(cib, 97.5))
        bauc_lo, bauc_hi = float(np.nanpercentile(ab, 2.5)), float(np.nanpercentile(ab, 97.5))

    return {"ext_ci": ext_ci, "ext_auc": ext_auc, "ext_ba": ext_ba,
            "boot_ci_lo": bci_lo, "boot_ci_hi": bci_hi,
            "boot_auc_lo": bauc_lo, "boot_auc_hi": bauc_hi}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data():
    train_df = pd.read_csv(TRAIN_FILE)
    ext_df   = pd.read_csv(EXT_FILE)
    assert len(train_df) == 87 and len(ext_df) == 27
    for col in ["CenterID", "HPV_binary", "Relapse"]:
        train_df[col] = train_df[col].astype(int)
        ext_df[col]   = ext_df[col].astype(int)
    train_df["RFS"] = train_df["RFS"].astype(float)
    ext_df["RFS"]   = ext_df["RFS"].astype(float)
    print(f"Train n=87 (HPV-={(train_df['HPV_binary']==0).sum()}, Rel={train_df['Relapse'].sum()}) "
          f"| Ext n=27")
    return train_df, ext_df


# ---------------------------------------------------------------------------
# Per-pool search
# ---------------------------------------------------------------------------
def run_pool(pool_name: str, pt_pool: list[str], ct_pool: list[str],
             train_df: pd.DataFrame, ext_df: pd.DataFrame,
             smoke: bool) -> pd.DataFrame:
    """Run full exhaustive search for one pool. Returns result_df."""
    # Build feature combos
    all_combos = []
    for n_total in range(N_MIN, N_MAX + 1):
        for n_pt in range(PT_MIN, min(len(pt_pool), n_total - CT_MIN - 1) + 1):
            n_ct = n_total - 1 - n_pt
            if CT_MIN <= n_ct <= len(ct_pool):
                for pt_f in combinations(pt_pool, n_pt):
                    for ct_f in combinations(ct_pool, n_ct):
                        all_combos.append((n_total, n_pt, n_ct, pt_f, ct_f))

    screen_combos = all_combos[:10] if smoke else all_combos
    suffix = f"5_{pool_name}"
    screen_csv = OUT_DIR / f"t2b_screen_{suffix}.csv"
    all_csv    = OUT_DIR / f"t2b_all_results_{suffix}.csv"

    print(f"\n  [Pool={pool_name}] {len(all_combos):,} combos | Stage 1 screen={len(screen_combos)}")

    # Stage 1
    screen_rows = []
    t0 = time.time()
    for ci, (n_total, n_pt, n_ct, pt_f, ct_f) in enumerate(screen_combos):
        pair_res = _eval_combo_all_pairs(list(pt_f), list(ct_f), train_df, N_REPEATS_SCREEN, N_EST_SCREEN)
        for pair_key, (oof_ci, oof_auc) in pair_res.items():
            s_lbl, h_lbl = pair_key.split("|")
            screen_rows.append({
                "pool": pool_name, "combo_id": ci,
                "n_total": n_total, "n_pt": n_pt, "n_ct": n_ct,
                "surv_gm": s_lbl, "hpv_gm": h_lbl, "pair_key": pair_key,
                "oof_ci_s1": oof_ci, "oof_auc_s1": oof_auc,
                "joint_s1": ALPHA * oof_ci + (1 - ALPHA) * oof_auc,
                "feat_pt": "|".join(pt_f), "feat_ct": "|".join(ct_f),
            })
        if (ci + 1) % 500 == 0:
            elapsed = time.time() - t0
            eta = (len(screen_combos) - ci - 1) / max((ci + 1) / elapsed, 1e-9)
            print(f"    [{ci+1:>5}/{len(screen_combos)}] {elapsed/60:.1f}m  eta={eta/60:.1f}m")

    screen_df = pd.DataFrame(screen_rows)
    screen_df.to_csv(screen_csv, index=False)
    print(f"  Stage 1 done in {(time.time()-t0)/60:.1f} min | {len(screen_df):,} rows")

    passed_soft = screen_df[
        (screen_df["oof_ci_s1"] >= SOFT_CI_FLOOR) &
        (screen_df["oof_auc_s1"] >= SOFT_AUC_FLOOR)
    ] if not screen_df.empty else screen_df

    if screen_df.empty or passed_soft.empty:
        reason = "Stage 1 zero rows" if screen_df.empty else "no soft-floor survivors"
        print(f"  [Pool={pool_name}] WARN: {reason}. Skipping Stage 2.")
        screen_df.to_csv(screen_csv, index=False)
        pd.DataFrame().to_csv(all_csv, index=False)
        return pd.DataFrame()

    top_screen = passed_soft.sort_values("joint_s1", ascending=False).head(TOPK_CONFIRM)
    print(f"  Soft floor: {len(passed_soft):,} passed | forwarding {len(top_screen)} to Stage 2")

    # Stage 2
    confirm_rows = []
    t0 = time.time()
    for _, row in top_screen.iterrows():
        feat_pt = row["feat_pt"].split("|")
        feat_ct = row["feat_ct"].split("|")
        s_lbl, h_lbl = row["surv_gm"], row["hpv_gm"]
        pair_res = _eval_combo_all_pairs(feat_pt, feat_ct, train_df, N_REPEATS_CONFIRM, N_EST_CONFIRM)
        oof_ci, oof_auc = pair_res.get(f"{s_lbl}|{h_lbl}", (0.5, 0.5))
        confirm_rows.append({
            **row.to_dict(),
            "oof_ci": oof_ci, "oof_auc": oof_auc,
            "joint_score": ALPHA * oof_ci + (1 - ALPHA) * oof_auc,
        })

    confirm_df = pd.DataFrame(confirm_rows)
    if confirm_df.empty:
        print(f"  [Pool={pool_name}] WARN: Stage 2 zero rows. Writing empty CSV.")
        pd.DataFrame().to_csv(all_csv, index=False)
        return pd.DataFrame()

    passed_hard = confirm_df[
        (confirm_df["oof_ci"] >= HARD_CI_FLOOR) &
        (confirm_df["oof_auc"] >= HARD_AUC_FLOOR)
    ]
    # Hard floor enforced strictly — no fallback. Zero survivors = negative result for this pool.
    if passed_hard.empty:
        print(f"  [Pool={pool_name}] RESULT: Hard-floor survivors 0/{len(confirm_df)} — negative conclusion.")
        _with_trial_no(confirm_df).to_csv(all_csv, index=False)
        return pd.DataFrame()
    print(f"  Stage 2 done in {(time.time()-t0)/60:.1f} min | "
          f"Hard floor: {len(passed_hard)}/{len(confirm_df)}")

    # External evaluation
    print(f"  External eval ({len(passed_hard)} rows)...")
    all_rows = []
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
        if (i + 1) % 10 == 0:
            print(f"    [{i+1:>3}/{len(passed_hard)}] elapsed={time.time()-t0:.0f}s")

    result_df = _with_trial_no(pd.DataFrame(all_rows))
    if result_df.empty:
        print(f"  [Pool={pool_name}] WARN: External eval zero rows. Writing empty CSV.")
        pd.DataFrame().to_csv(all_csv, index=False)
        return pd.DataFrame()

    result_df.to_csv(all_csv, index=False)

    # Pool summary
    dual = result_df[
        (result_df["oof_ci"] >= HARD_CI_FLOOR) & (result_df["oof_auc"] >= HARD_AUC_FLOOR) &
        (result_df["ext_ci"] >= 0.60) & (result_df["ext_auc"] >= 0.70)
    ]
    print(f"  [Pool={pool_name}] dual-task candidates: {len(dual)} | "
          f"top ext_ci={result_df['ext_ci'].max():.3f} | "
          f"top ext_auc={result_df['ext_auc'].max():.3f}")
    return result_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> None:
    started_at = time.time()
    train_df, ext_df = load_data()

    pool_names = args.pools if args.pools else ALL_POOL_NAMES
    print(f"\nv5: {len(pool_names)} pools × 9 GM pairs | N=[{N_MIN},{N_MAX}]")
    print(f"Pools: {pool_names}")

    # Compute pools
    pools: dict[str, tuple[list, list]] = {}
    print("\n--- Computing feature pools ---")
    for pname in pool_names:
        pools[pname] = compute_pool(pname, train_df)

    # Run search per pool
    all_results: list[pd.DataFrame] = []
    log_lines = [
        "# Task 2B v5 Log",
        f"- Started: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(started_at))}",
        f"- Pools: {pool_names}",
        f"- GMs: {[s[1] for s in SURV_GM_CONFIGS]} x {[h[1] for h in HPV_GM_CONFIGS]}",
        "",
    ]

    print("\n--- Running exhaustive search per pool ---")
    for pname in pool_names:
        pt_pool, ct_pool = pools[pname]
        result_df = run_pool(pname, pt_pool, ct_pool, train_df, ext_df, smoke=args.smoke)
        all_results.append(result_df)

        # Pool summary for log
        if len(result_df) > 0:
            top5 = result_df.sort_values("joint_score", ascending=False).head(5)
            log_lines.append(f"## Pool: {pname}")
            log_lines.append(f"Ext rows: {len(result_df)} | "
                             f"best ext_ci={result_df['ext_ci'].max():.3f} | "
                             f"best ext_auc={result_df['ext_auc'].max():.3f}")
            for _, r in top5.iterrows():
                log_lines.append(
                    f"  {r['surv_gm']} | {r['hpv_gm']} | N={int(r['n_total'])} | "
                    f"oof_ci={r['oof_ci']:.3f} oof_auc={r['oof_auc']:.3f} | "
                    f"ext_ci={r['ext_ci']:.3f} ext_auc={r['ext_auc']:.3f}"
                )
            log_lines.append("")

    # Merged output
    ext_results = [df for df in all_results if not df.empty]
    if ext_results:
        merged = _with_trial_no(pd.concat(ext_results, ignore_index=True))
        merged_csv = OUT_DIR / "t2b_all_results_5_merged.csv"
        merged.to_csv(merged_csv, index=False)
        merged.sort_values(["joint_score", "ext_ci", "ext_auc"], ascending=False).head(20).to_csv(
            OUT_DIR / "t2b_top20_joint_5.csv", index=False)
        merged.sort_values(["ext_ci", "joint_score"], ascending=False).head(20).to_csv(
            OUT_DIR / "t2b_top20_rfs_5.csv", index=False)

        dual_total = merged[
            (merged["oof_ci"] >= HARD_CI_FLOOR) & (merged["oof_auc"] >= HARD_AUC_FLOOR) &
            (merged["ext_ci"] >= 0.60) & (merged["ext_auc"] >= 0.70)
        ]
        wall = time.time() - started_at
        summary = (
            f"=== v5 Summary ===\n"
            f"Pools: {pool_names}\n"
            f"Total ext rows: {len(merged):,} | "
            f"Dual-task candidates: {len(dual_total)}\n"
            f"Wall time: {wall/60:.1f} min\n\n"
            f"Best ext_ci by pool:\n"
        )
        for pname in pool_names:
            sub = merged[merged["pool"] == pname]
            if len(sub):
                best = sub.sort_values("ext_ci", ascending=False).iloc[0]
                summary += (f"  {pname:20s}: ext_ci={best['ext_ci']:.3f} "
                            f"ext_auc={best['ext_auc']:.3f} "
                            f"({best['surv_gm']}|{best['hpv_gm']})\n")
        print("\n" + summary)
        log_lines += ["## Overall Summary", "```", summary, "```",
                      f"- Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}"]
        LOG_MD.write_text("\n".join(log_lines), encoding="utf-8")
        print(f"Merged CSV: {merged_csv}")
        print(f"Log: {LOG_MD}")
    else:
        print("\n[RESULT] No pools produced externally evaluated rows after hard-floor filtering.")
        merged_csv = OUT_DIR / "t2b_all_results_5_merged.csv"
        pd.DataFrame().to_csv(merged_csv, index=False)
        pd.DataFrame().to_csv(OUT_DIR / "t2b_top20_joint_5.csv", index=False)
        pd.DataFrame().to_csv(OUT_DIR / "t2b_top20_rfs_5.csv", index=False)
        log_lines += [
            "## Overall Summary",
            "No pools produced externally evaluated rows after hard-floor filtering.",
            f"- Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        LOG_MD.write_text("\n".join(log_lines), encoding="utf-8")
        print(f"Merged CSV: {merged_csv}")
        print(f"Log: {LOG_MD}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--pools", nargs="+", choices=ALL_POOL_NAMES,
                   help="Space-separated pool names (default: all 4). "
                        "Example: --pools UNIVAR_HPV GBC_HPV")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"ROOT={ROOT}\nv5: 4 pools x 9 GM pairs | N=[{N_MIN},{N_MAX}]")
    run(args)


if __name__ == "__main__":
    main()
