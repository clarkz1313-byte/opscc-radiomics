"""
2_apr_t2b_pan_sc5_14.py  --  Task 2B v14 strict 6PT gap-fill rerun

Strict gap-fill rerun forked from v7b.

Search space:
- same curated v7/v7b union feature space (11 PT + 7 CT)
- same structures: 6 PT x 2-3 CT
- same 9 GM pairs (3 survival x 3 HPV)

Strict cohort design:
- train on old Task 2 train split (67)
- select/confirm by CV on that train split
- report fixed internal holdout metrics on old Task 2 test split (20)
- report CHUS external metrics separately (27)

This script is the strict companion to v13, covering the remaining 6 PT x 2-3 CT
gap-fill tier from the original v7b design under the same old Task 2 split.

Run:
    python Apr_2026_task2B/2_apr_t2b_pan_sc5_14.py
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
    ("numpy", None),
    ("pandas", None),
    ("sklearn", "scikit-learn"),
    ("sksurv", "scikit-survival"),
]:
    _ensure_import(_module, _package)

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
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
TASK2_DIR = ROOT.parent / "Mar_2026_task2" / "12_mar_task2_rad_data"
TRAIN_FILE = DATA_DIR / "2_apr_t2b_train.csv"
EXT_FILE = DATA_DIR / "2_apr_t2b_ext.csv"
SPLIT_MAP_FILE = TASK2_DIR / "13_mar_task2_split_map.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)
SCREEN_CSV = OUT_DIR / "t2b_screen_14.csv"
CHECKPOINTS_CSV = OUT_DIR / "t2b_checkpoints_14.csv"
ALL_CSV = OUT_DIR / "t2b_all_results_14.csv"
TOP20_STRICT_CSV = OUT_DIR / "t2b_top20_strict_14.csv"
TOP20_EXT_CSV = OUT_DIR / "t2b_top20_ext_14.csv"
LOG_MD = OUT_DIR / "t2b_log_14.md"

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
SEED = 42
ALPHA = 0.60
N_FOLDS = 5
N_REPEATS_SCREEN = 1
N_REPEATS_CONFIRM = 3
N_EST_SCREEN = 100
N_EST_CONFIRM = 200
SOFT_CI_FLOOR = 0.55
SOFT_AUC_FLOOR = 0.69
HARD_CI_FLOOR = 0.55
HARD_AUC_FLOOR = 0.69
N_BOOT = 100
CHECKPOINT_EVERY = 2000
PROGRESS_EVERY = 5000
CLINICAL_FEATURES = ["Gender_Male"]
PT_SIZES = [6]
CT_SIZES = [2, 3]

UNION_PT = [
    "GTVn_logarithm_glszm_SmallAreaLowGrayLevelEmphasis",
    "GTVn_wavelet-LHH_glszm_LowGrayLevelZoneEmphasis",
    "GTVn_wavelet-LLH_firstorder_Mean",
    "GTVn_wavelet-LLH_firstorder_Skewness",
    "GTVp_exponential_glszm_HighGrayLevelZoneEmphasis",
    "GTVp_gradient_glszm_ZoneEntropy",
    "GTVp_original_firstorder_InterquartileRange",
    "GTVp_wavelet-HHL_glrlm_ShortRunHighGrayLevelEmphasis",
    "GTVp_wavelet-HLH_glrlm_ShortRunHighGrayLevelEmphasis",
    "GTVp_wavelet-HLH_glszm_HighGrayLevelZoneEmphasis",
    "GTVp_wavelet-LLH_firstorder_Median",
]

UNION_CT = [
    "GTVn_wavelet-LHH_glcm_ClusterProminence",
    "GTVn_wavelet-LHH_glrlm_GrayLevelVariance",
    "GTVp_gradient_glszm_SmallAreaLowGrayLevelEmphasis",
    "GTVp_log-sigma-1-mm-3D_firstorder_Range",
    "GTVp_wavelet-HLL_ngtdm_Complexity",
    "GTVp_wavelet-LHH_firstorder_RootMeanSquared",
    "GTVp_wavelet-LLH_glrlm_HighGrayLevelRunEmphasis",
]

SURV_GM_CONFIGS = [
    ("EST", "EST", {"n_estimators": N_EST_SCREEN}),
    ("SVM", "SVM_0001", {"alpha": 0.001}),
    ("CoxPH", "CoxPH01", {"alpha": 0.1}),
]

HPV_GM_CONFIGS = [
    ("LR_L2", "LR_L2_0.5", {"C": 0.5, "penalty": "l2", "solver": "lbfgs", "max_iter": 2000}),
    ("LR_EN", "LR_EN_1.0", {"C": 1.0, "penalty": "elasticnet", "solver": "saga", "l1_ratio": 0.5, "max_iter": 5000}),
    ("SVM_L", "SVM_L_001", {"C": 0.01}),
]

SCREEN_COLUMNS = [
    "combo_id", "n_total", "n_pt", "n_ct", "surv_gm", "hpv_gm", "pair_key",
    "oof_ci_s1", "oof_auc_s1", "joint_s1", "feat_pt", "feat_ct",
]

RESULT_COLUMNS = [
    "trial_no", "combo_id", "n_total", "n_pt", "n_ct", "surv_gm", "hpv_gm", "pair_key",
    "oof_ci_s1", "oof_auc_s1", "joint_s1", "feat_pt", "feat_ct",
    "oof_ci", "oof_auc", "joint_score",
    "test_ci", "test_auc", "joint_test", "test_ba", "test_spe", "test_sen",
    "strict_score",
    "ext_ci", "ext_auc", "ext_ba", "ext_spe", "ext_sen",
    "boot_ci_lo", "boot_ci_hi", "boot_auc_lo", "boot_auc_hi",
]

CONFIRM_COLUMNS = [
    "combo_id", "n_total", "n_pt", "n_ct", "surv_gm", "hpv_gm", "pair_key",
    "oof_ci_s1", "oof_auc_s1", "joint_s1", "feat_pt", "feat_ct",
    "oof_ci", "oof_auc", "joint_score",
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


def _scale_blocks(tr_df, vl_df, feat_pt, feat_ct):
    def _arr(df, cols):
        return df[cols].to_numpy(dtype=float)

    x_clin_tr, x_pt_tr, x_ct_tr = _arr(tr_df, CLINICAL_FEATURES), _arr(tr_df, feat_pt), _arr(tr_df, feat_ct)
    x_clin_vl, x_pt_vl, x_ct_vl = _arr(vl_df, CLINICAL_FEATURES), _arr(vl_df, feat_pt), _arr(vl_df, feat_ct)
    sc_c, sc_p, sc_t = StandardScaler(), StandardScaler(), StandardScaler()
    x_tr = np.hstack([sc_c.fit_transform(x_clin_tr), sc_p.fit_transform(x_pt_tr), sc_t.fit_transform(x_ct_tr)])
    x_vl = np.hstack([sc_c.transform(x_clin_vl), sc_p.transform(x_pt_vl), sc_t.transform(x_ct_vl)])
    return x_tr, x_vl


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _with_trial_no(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "trial_no" in out.columns:
        out = out.drop(columns=["trial_no"])
    out.insert(0, "trial_no", np.arange(1, len(out) + 1, dtype=int))
    return out


def _empty_result_df() -> pd.DataFrame:
    return pd.DataFrame(columns=RESULT_COLUMNS)


def _empty_confirm_df() -> pd.DataFrame:
    return pd.DataFrame(columns=CONFIRM_COLUMNS)


def _load_legacy_confirm_df(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return _empty_confirm_df()
    try:
        df = pd.read_csv(path)
    except Exception:
        return _empty_confirm_df()
    missing = [col for col in CONFIRM_COLUMNS if col not in df.columns]
    if missing:
        return _empty_confirm_df()
    return df[CONFIRM_COLUMNS].copy()


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
        return LogisticRegression(
            C=params["C"], penalty="l2", solver="lbfgs",
            class_weight="balanced", max_iter=2000, random_state=SEED
        )
    if key == "LR_EN":
        return LogisticRegression(
            C=params["C"], penalty="elasticnet", solver="saga",
            l1_ratio=params["l1_ratio"], class_weight="balanced",
            max_iter=5000, random_state=SEED
        )
    if key == "SVM_L":
        base = LinearSVC(C=params["C"], class_weight="balanced", max_iter=5000, random_state=SEED)
        return CalibratedClassifierCV(base, cv=3)
    raise ValueError(key)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------
def _eval_combo_all_pairs(feat_pt, feat_ct, train_df, n_repeats, est_n):
    """
    Returns dict: pair_key -> (mean_oof_ci, mean_oof_auc).
    Stratified on combined label (2*Relapse + HPV_binary) to reduce fold instability.
    """
    y_strat = (train_df["Relapse"].to_numpy(int) * 2 + train_df["HPV_binary"].to_numpy(int)).clip(0, 3)
    rkf = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=n_repeats, random_state=SEED)
    s_ci = {s[1]: [] for s in SURV_GM_CONFIGS}
    h_auc = {h[1]: [] for h in HPV_GM_CONFIGS}

    for tr_idx, vl_idx in rkf.split(train_df, y_strat):
        tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
        vl_df = train_df.iloc[vl_idx].reset_index(drop=True)
        x_tr, x_vl = _scale_blocks(tr_df, vl_df, feat_pt, feat_ct)
        y_s_tr = Surv.from_arrays(event=tr_df["Relapse"].astype(bool), time=tr_df["RFS"])
        y_s_vl = Surv.from_arrays(event=vl_df["Relapse"].astype(bool), time=vl_df["RFS"])
        y_h_tr = tr_df["HPV_binary"].to_numpy(int)
        y_h_vl = vl_df["HPV_binary"].to_numpy(int)

        for sk, sl, sp in SURV_GM_CONFIGS:
            try:
                model = _make_surv(sk, sp, n_est_override=est_n if sk == "EST" else None)
                model.fit(x_tr, y_s_tr)
                s_ci[sl].append(_safe_ci(y_s_vl, model.predict(x_vl)))
            except Exception:
                s_ci[sl].append(0.5)

        for hk, hl, hp in HPV_GM_CONFIGS:
            try:
                model = _make_hpv(hk, hp)
                model.fit(x_tr, y_h_tr)
                proba = model.predict_proba(x_vl)[:, 1]
                try:
                    auc = float(roc_auc_score(y_h_vl, proba))
                except Exception:
                    auc = 0.5
                h_auc[hl].append(auc)
            except Exception:
                h_auc[hl].append(0.5)

    results = {}
    for (_, sl, _), (_, hl, _) in product(SURV_GM_CONFIGS, HPV_GM_CONFIGS):
        results[f"{sl}+{hl}"] = (float(np.mean(s_ci[sl])), float(np.mean(h_auc[hl])))
    return results


def _eval_combo_single_pair(feat_pt, feat_ct, train_df, n_repeats, est_n, surv_label, hpv_label):
    y_strat = (train_df["Relapse"].to_numpy(int) * 2 + train_df["HPV_binary"].to_numpy(int)).clip(0, 3)
    rkf = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=n_repeats, random_state=SEED)
    s_key, _, s_params = next(c for c in SURV_GM_CONFIGS if c[1] == surv_label)
    h_key, _, h_params = next(c for c in HPV_GM_CONFIGS if c[1] == hpv_label)
    s_ci, h_auc = [], []

    for tr_idx, vl_idx in rkf.split(train_df, y_strat):
        tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
        vl_df = train_df.iloc[vl_idx].reset_index(drop=True)
        x_tr, x_vl = _scale_blocks(tr_df, vl_df, feat_pt, feat_ct)
        y_s_tr = Surv.from_arrays(event=tr_df["Relapse"].astype(bool), time=tr_df["RFS"])
        y_s_vl = Surv.from_arrays(event=vl_df["Relapse"].astype(bool), time=vl_df["RFS"])
        y_h_tr = tr_df["HPV_binary"].to_numpy(int)
        y_h_vl = vl_df["HPV_binary"].to_numpy(int)

        try:
            sm = _make_surv(s_key, s_params, n_est_override=est_n if s_key == "EST" else None)
            sm.fit(x_tr, y_s_tr)
            s_ci.append(_safe_ci(y_s_vl, sm.predict(x_vl)))
        except Exception:
            s_ci.append(0.5)

        try:
            hm = _make_hpv(h_key, h_params)
            hm.fit(x_tr, y_h_tr)
            proba = hm.predict_proba(x_vl)[:, 1]
            try:
                h_auc.append(float(roc_auc_score(y_h_vl, proba)))
            except Exception:
                h_auc.append(0.5)
        except Exception:
            h_auc.append(0.5)

    return float(np.mean(s_ci)), float(np.mean(h_auc))


def _evaluate_on_dataset(feat_pt, feat_ct, s_key, s_params, h_key, h_params, fit_df, eval_df, n_boot=0):
    x_fit, x_eval = _scale_blocks(fit_df, eval_df, feat_pt, feat_ct)
    y_s_fit = Surv.from_arrays(event=fit_df["Relapse"].astype(bool), time=fit_df["RFS"])
    y_s_eval = Surv.from_arrays(event=eval_df["Relapse"].astype(bool), time=eval_df["RFS"])
    y_h_fit = fit_df["HPV_binary"].to_numpy(int)
    y_h_eval = eval_df["HPV_binary"].to_numpy(int)

    try:
        sm = _make_surv(s_key, s_params, n_est_override=N_EST_CONFIRM if s_key == "EST" else None)
        sm.fit(x_fit, y_s_fit)
        risk_eval = sm.predict(x_eval)
        out_ci = _safe_ci(y_s_eval, risk_eval)
    except Exception:
        risk_eval = np.full(len(x_eval), np.nan)
        out_ci = float("nan")

    out_spe = out_sen = float("nan")
    try:
        hm = _make_hpv(h_key, h_params)
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
    except Exception:
        proba_use = np.full(len(x_eval), np.nan)
        out_auc = out_ba = float("nan")

    bci_lo = bci_hi = bauc_lo = bauc_hi = float("nan")
    if n_boot > 0 and not np.isnan(out_ci):
        rng = np.random.default_rng(SEED)
        n_e = len(eval_df)
        ci_boots, auc_boots = [], []
        for _ in range(n_boot):
            idx = rng.integers(0, n_e, n_e)
            ci_boots.append(_safe_ci(y_s_eval[idx], risk_eval[idx]))
            try:
                raw_auc = float(roc_auc_score(y_h_eval[idx], proba_use[idx]))
                auc_boots.append(max(raw_auc, 1.0 - raw_auc))
            except Exception:
                auc_boots.append(float("nan"))
        bci_lo = float(np.nanpercentile(ci_boots, 2.5))
        bci_hi = float(np.nanpercentile(ci_boots, 97.5))
        bauc_lo = float(np.nanpercentile(auc_boots, 2.5))
        bauc_hi = float(np.nanpercentile(auc_boots, 97.5))

    return {
        "ci": out_ci,
        "auc": out_auc,
        "ba": out_ba,
        "spe": out_spe,
        "sen": out_sen,
        "boot_ci_lo": bci_lo,
        "boot_ci_hi": bci_hi,
        "boot_auc_lo": bauc_lo,
        "boot_auc_hi": bauc_hi,
    }


def evaluate_on_test(feat_pt, feat_ct, s_key, s_params, h_key, h_params, train_df, test_df):
    m = _evaluate_on_dataset(feat_pt, feat_ct, s_key, s_params, h_key, h_params, train_df, test_df, n_boot=0)
    return {
        "test_ci": m["ci"],
        "test_auc": m["auc"],
        "test_ba": m["ba"],
        "test_spe": m["spe"],
        "test_sen": m["sen"],
    }


def evaluate_on_ext(feat_pt, feat_ct, s_key, s_params, h_key, h_params, train_df, ext_df, n_boot):
    m = _evaluate_on_dataset(feat_pt, feat_ct, s_key, s_params, h_key, h_params, train_df, ext_df, n_boot=n_boot)
    return {
        "ext_ci": m["ci"],
        "ext_auc": m["auc"],
        "ext_ba": m["ba"],
        "ext_spe": m["spe"],
        "ext_sen": m["sen"],
        "boot_ci_lo": m["boot_ci_lo"],
        "boot_ci_hi": m["boot_ci_hi"],
        "boot_auc_lo": m["boot_auc_lo"],
        "boot_auc_hi": m["boot_auc_hi"],
    }


# ---------------------------------------------------------------------------
# Data / combinations
# ---------------------------------------------------------------------------
def _coerce_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["CenterID", "HPV_binary", "Relapse"]:
        out[col] = out[col].astype(int)
    out["RFS"] = out["RFS"].astype(float)
    return out


def load_data():
    pooled_df = _coerce_df(pd.read_csv(TRAIN_FILE))
    ext_df = _coerce_df(pd.read_csv(EXT_FILE))
    split_map = pd.read_csv(SPLIT_MAP_FILE)
    split_map["PatientID"] = split_map["PatientID"].astype(str)
    split_map["split"] = split_map["split"].astype(str).str.lower()

    strict_train_ids = set(split_map.loc[split_map["split"] == "train", "PatientID"])
    strict_test_ids = set(split_map.loc[split_map["split"] == "test", "PatientID"])

    train_df = pooled_df[pooled_df["PatientID"].isin(strict_train_ids)].reset_index(drop=True)
    test_df = pooled_df[pooled_df["PatientID"].isin(strict_test_ids)].reset_index(drop=True)

    assert len(pooled_df) == 87 and len(train_df) == 67 and len(test_df) == 20 and len(ext_df) == 27

    print(
        f"Strict train n={len(train_df)} (HPV-={(train_df['HPV_binary'] == 0).sum()}, Rel={train_df['Relapse'].sum()}) | "
        f"Strict test n={len(test_df)} (HPV-={(test_df['HPV_binary'] == 0).sum()}, Rel={test_df['Relapse'].sum()}) | "
        f"CHUS ext n={len(ext_df)} (HPV-={(ext_df['HPV_binary'] == 0).sum()}, Rel={ext_df['Relapse'].sum()})"
    )
    return train_df, test_df, ext_df


# ---------------------------------------------------------------------------
# Data / combinations
# ---------------------------------------------------------------------------
def build_all_combos():
    combos = []
    total = 0
    for n_pt, n_ct in product(PT_SIZES, CT_SIZES):
        idx = 0
        for pt_feats in combinations(UNION_PT, n_pt):
            for ct_feats in combinations(UNION_CT, n_ct):
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


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> None:
    started_at = time.time()
    train_df, test_df, ext_df = load_data()
    combos, total_combos = build_all_combos()
    total_pair_evals = total_combos * len(SURV_GM_CONFIGS) * len(HPV_GM_CONFIGS)

    print(f"Total feature combos: {total_combos:,} | Total stage-1 pair evals: {total_pair_evals:,}")
    if args.smoke:
        combos = combos[:12]
        print(f"[SMOKE] limiting to first {len(combos)} feature combos")

    print("\n--- Stage 1 screening ---")
    if SCREEN_CSV.exists() and SCREEN_CSV.stat().st_size > 0:
        screen_df = pd.read_csv(SCREEN_CSV)
        if list(screen_df.columns) != SCREEN_COLUMNS:
            screen_df = screen_df[SCREEN_COLUMNS]
        existing_pairs = set(zip(screen_df["combo_id"], screen_df["pair_key"]))
        print(f"Loaded checkpoint: {len(screen_df):,} stage-1 rows")
    else:
        screen_df = pd.DataFrame(columns=SCREEN_COLUMNS)
        existing_pairs = set()

    expected_pair_keys = {f"{s[1]}+{h[1]}" for s, h in product(SURV_GM_CONFIGS, HPV_GM_CONFIGS)}
    expected_stage1_rows = len(combos) * len(expected_pair_keys)
    screen_rows = screen_df.to_dict("records")
    completed_rows = len(screen_rows)

    if completed_rows >= expected_stage1_rows:
        print(f"Stage 1 already complete: {completed_rows:,} rows loaded")
    else:
        t0 = time.time()
        last_checkpoint_bucket = completed_rows // CHECKPOINT_EVERY
        last_progress_bucket = completed_rows // PROGRESS_EVERY

        for combo in combos:
            feat_pt, feat_ct = combo["feat_pt"], combo["feat_ct"]
            stage1 = _eval_combo_all_pairs(feat_pt, feat_ct, train_df, N_REPEATS_SCREEN, N_EST_SCREEN)
            for pair_key, (oof_ci, oof_auc) in stage1.items():
                key = (combo["combo_id"], pair_key)
                if key in existing_pairs:
                    continue
                surv_gm, hpv_gm = pair_key.split("+")
                screen_rows.append({
                    "combo_id": combo["combo_id"],
                    "n_total": combo["n_total"],
                    "n_pt": combo["n_pt"],
                    "n_ct": combo["n_ct"],
                    "surv_gm": surv_gm,
                    "hpv_gm": hpv_gm,
                    "pair_key": pair_key,
                    "oof_ci_s1": oof_ci,
                    "oof_auc_s1": oof_auc,
                    "joint_s1": ALPHA * oof_ci + (1.0 - ALPHA) * oof_auc,
                    "feat_pt": "|".join(feat_pt),
                    "feat_ct": "|".join(feat_ct),
                })
                existing_pairs.add(key)

            current_rows = len(screen_rows)
            current_checkpoint_bucket = current_rows // CHECKPOINT_EVERY
            if current_checkpoint_bucket > last_checkpoint_bucket:
                _atomic_write_csv(pd.DataFrame(screen_rows, columns=SCREEN_COLUMNS), SCREEN_CSV)
                last_checkpoint_bucket = current_checkpoint_bucket

            current_progress_bucket = current_rows // PROGRESS_EVERY
            if current_progress_bucket > last_progress_bucket:
                elapsed = time.time() - t0
                rate = current_rows / max(elapsed, 1e-9)
                eta = (total_pair_evals - current_rows) / max(rate, 1e-9)
                temp_df = pd.DataFrame(screen_rows, columns=SCREEN_COLUMNS)
                hit_rate = float(((temp_df["oof_ci_s1"] >= SOFT_CI_FLOOR) & (temp_df["oof_auc_s1"] >= SOFT_AUC_FLOOR)).mean())
                print(
                    f"  [{current_rows:>7}/{total_pair_evals}] elapsed={elapsed/60:.1f}m "
                    f"eta={eta/60:.1f}m soft_hit={100*hit_rate:.2f}%"
                )
                last_progress_bucket = current_progress_bucket

        screen_df = pd.DataFrame(screen_rows, columns=SCREEN_COLUMNS)
        _atomic_write_csv(screen_df, SCREEN_CSV)
        print(f"Stage 1 done in {(time.time() - t0)/60:.1f} min | rows={len(screen_df):,}")

    passed_soft = screen_df[
        (screen_df["oof_ci_s1"] >= SOFT_CI_FLOOR) &
        (screen_df["oof_auc_s1"] >= SOFT_AUC_FLOOR)
    ].reset_index(drop=True)

    print(f"Stage 1 soft-floor survivors: {len(passed_soft):,} / {len(screen_df):,}")
    if passed_soft.empty:
        empty = _empty_result_df()
        _atomic_write_csv(_empty_confirm_df(), CHECKPOINTS_CSV)
        empty.to_csv(ALL_CSV, index=False)
        empty.to_csv(TOP20_STRICT_CSV, index=False)
        empty.to_csv(TOP20_EXT_CSV, index=False)
        LOG_MD.write_text("# Task 2B v14 Log\n\nNo Stage-1 soft-floor survivors.\n", encoding="utf-8")
        return

    print("\n--- Stage 2 confirmation ---")
    valid_confirm_pairs = set(zip(passed_soft["combo_id"], passed_soft["pair_key"]))
    if CHECKPOINTS_CSV.exists() and CHECKPOINTS_CSV.stat().st_size > 0:
        confirm_df = pd.read_csv(CHECKPOINTS_CSV)
        if list(confirm_df.columns) != CONFIRM_COLUMNS:
            confirm_df = confirm_df[CONFIRM_COLUMNS]
        confirm_df = confirm_df[
            confirm_df.apply(lambda row: (row["combo_id"], row["pair_key"]) in valid_confirm_pairs, axis=1)
        ].reset_index(drop=True)
        print(f"Loaded checkpoint: {len(confirm_df):,} stage-2 rows")
    else:
        confirm_df = _empty_confirm_df()

    completed_confirm_pairs = set(zip(confirm_df["combo_id"], confirm_df["pair_key"]))
    confirm_rows = confirm_df.to_dict("records")
    total_confirm_rows = len(passed_soft)
    starting_confirm_rows = len(confirm_rows)
    last_confirm_checkpoint_bucket = len(confirm_rows) // CHECKPOINT_EVERY
    last_confirm_progress_bucket = len(confirm_rows) // 200
    t1 = time.time()
    for _, row in passed_soft.iterrows():
        key = (row["combo_id"], row["pair_key"])
        if key in completed_confirm_pairs:
            continue
        feat_pt = row["feat_pt"].split("|")
        feat_ct = row["feat_ct"].split("|")
        oof_ci, oof_auc = _eval_combo_single_pair(
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
            _atomic_write_csv(pd.DataFrame(confirm_rows, columns=CONFIRM_COLUMNS), CHECKPOINTS_CSV)
            last_confirm_checkpoint_bucket = current_confirm_checkpoint_bucket

        current_confirm_progress_bucket = current_confirm_rows // 200
        if current_confirm_progress_bucket > last_confirm_progress_bucket:
            elapsed = time.time() - t1
            completed_this_session = current_confirm_rows - starting_confirm_rows
            rate = completed_this_session / max(elapsed, 1e-9)
            eta = (total_confirm_rows - current_confirm_rows) / max(rate, 1e-9)
            temp_df = pd.DataFrame(confirm_rows, columns=CONFIRM_COLUMNS)
            hard_hit = float(((temp_df["oof_ci"] >= HARD_CI_FLOOR) & (temp_df["oof_auc"] >= HARD_AUC_FLOOR)).mean())
            print(
                f"  [{current_confirm_rows:>7}/{total_confirm_rows}] elapsed={elapsed/60:.1f}m "
                f"eta={eta/60:.1f}m hard_hit={100*hard_hit:.2f}%"
            )
            last_confirm_progress_bucket = current_confirm_progress_bucket

    confirm_df = pd.DataFrame(confirm_rows, columns=CONFIRM_COLUMNS)
    _atomic_write_csv(confirm_df, CHECKPOINTS_CSV)
    passed_hard = confirm_df[
        (confirm_df["oof_ci"] >= HARD_CI_FLOOR) &
        (confirm_df["oof_auc"] >= HARD_AUC_FLOOR)
    ].reset_index(drop=True)
    print(
        f"Stage 2 done in {(time.time() - t1)/60:.1f} min | "
        f"hard-floor survivors={len(passed_hard):,} / {len(confirm_df):,}"
    )

    if passed_hard.empty:
        empty = _empty_result_df()
        empty.to_csv(ALL_CSV, index=False)
        empty.to_csv(TOP20_STRICT_CSV, index=False)
        empty.to_csv(TOP20_EXT_CSV, index=False)
        summary = (
            "# Task 2B v14 Log\n\n"
            f"- Total feature combos evaluated: {total_combos:,}\n"
            f"- Strict train/test/ext: {len(train_df)} / {len(test_df)} / {len(ext_df)}\n"
            f"- Stage-1 soft-floor survivors: {len(passed_soft):,} / {len(screen_df):,}\n"
            f"- Stage-2 hard-floor survivors: 0 / {len(confirm_df):,}\n"
            "- No rows reached holdout/external evaluation.\n"
        )
        LOG_MD.write_text(summary, encoding="utf-8")
        return

    print("\n--- Strict holdout + CHUS evaluation ---")
    all_rows = []
    t2 = time.time()
    for idx, (_, row) in enumerate(passed_hard.iterrows(), start=1):
        feat_pt = row["feat_pt"].split("|")
        feat_ct = row["feat_ct"].split("|")
        s_lbl, h_lbl = row["surv_gm"], row["hpv_gm"]
        s_key, _, s_params = next(c for c in SURV_GM_CONFIGS if c[1] == s_lbl)
        h_key, _, h_params = next(c for c in HPV_GM_CONFIGS if c[1] == h_lbl)
        test_metrics = evaluate_on_test(feat_pt, feat_ct, s_key, s_params, h_key, h_params, train_df, test_df)
        ext_metrics = evaluate_on_ext(feat_pt, feat_ct, s_key, s_params, h_key, h_params, train_df, ext_df, N_BOOT)
        joint_test = ALPHA * test_metrics["test_ci"] + (1.0 - ALPHA) * test_metrics["test_auc"]
        strict_score = 0.5 * row["joint_score"] + 0.5 * joint_test
        all_rows.append({
            **row.to_dict(),
            **test_metrics,
            "joint_test": joint_test,
            "strict_score": strict_score,
            **ext_metrics,
        })
        if idx % 50 == 0:
            print(f"  [{idx:>5}/{len(passed_hard)}] elapsed={(time.time() - t2)/60:.1f}m")

    result_df = _with_trial_no(pd.DataFrame(all_rows))
    result_df = result_df[RESULT_COLUMNS]
    result_df.to_csv(ALL_CSV, index=False)

    top_strict = result_df.sort_values(["strict_score", "joint_test", "ext_ci", "ext_auc"], ascending=False).head(20)
    top_ext = result_df.sort_values(["ext_ci", "ext_auc", "strict_score"], ascending=False).head(20)
    top_strict.to_csv(TOP20_STRICT_CSV, index=False)
    top_ext.to_csv(TOP20_EXT_CSV, index=False)

    dual = result_df[(result_df["ext_ci"] >= 0.60) & (result_df["ext_auc"] >= 0.70)]
    wall = time.time() - started_at
    best_strict = result_df.sort_values(["strict_score", "joint_test", "ext_ci", "ext_auc"], ascending=False).iloc[0]
    best_ext = result_df.sort_values(["ext_ci", "ext_auc", "strict_score"], ascending=False).iloc[0]

    summary_lines = [
        "# Task 2B v14 Log",
        "",
        f"- Started: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(started_at))}",
        f"- Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Runtime: {wall/60:.1f} min",
        f"- Strict train/test/ext: {len(train_df)} / {len(test_df)} / {len(ext_df)}",
        f"- Total feature combos evaluated: {total_combos:,}",
        f"- Stage-1 soft-floor survivors: {len(passed_soft):,} / {len(screen_df):,} ({100*len(passed_soft)/max(len(screen_df),1):.2f}%)",
        f"- Stage-2 hard-floor survivors: {len(passed_hard):,} / {len(confirm_df):,} ({100*len(passed_hard)/max(len(confirm_df),1):.2f}%)",
        f"- Strict evaluated rows: {len(result_df):,}",
        f"- Dual-floor CHUS candidates (ext_ci >= 0.60 and ext_auc >= 0.70): {len(dual):,}",
        "",
        "## Best row by strict_score",
        (
            f"- trial_no={int(best_strict['trial_no'])} | combo_id={best_strict['combo_id']} | "
            f"{best_strict['surv_gm']} + {best_strict['hpv_gm']} | "
            f"N={int(best_strict['n_total'])} PT={int(best_strict['n_pt'])} CT={int(best_strict['n_ct'])} | "
            f"oof_ci={best_strict['oof_ci']:.3f} oof_auc={best_strict['oof_auc']:.3f} | "
            f"test_ci={best_strict['test_ci']:.3f} test_auc={best_strict['test_auc']:.3f} | "
            f"ext_ci={best_strict['ext_ci']:.3f} ext_auc={best_strict['ext_auc']:.3f}"
        ),
        "",
        "## Best row by ext_ci",
        (
            f"- trial_no={int(best_ext['trial_no'])} | combo_id={best_ext['combo_id']} | "
            f"{best_ext['surv_gm']} + {best_ext['hpv_gm']} | "
            f"ext_ci={best_ext['ext_ci']:.3f} ext_auc={best_ext['ext_auc']:.3f} | "
            f"test_ci={best_ext['test_ci']:.3f} test_auc={best_ext['test_auc']:.3f} | "
            f"oof_ci={best_ext['oof_ci']:.3f} oof_auc={best_ext['oof_auc']:.3f}"
        ),
    ]

    summary = "\n".join(summary_lines)
    print("\n" + summary)
    LOG_MD.write_text(summary + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run on a small subset of feature combos.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        f"ROOT={ROOT}\n"
        f"v14 strict rerun from v7b 6PT gap-fill | PT sizes={PT_SIZES} | CT sizes={CT_SIZES} | "
        f"soft/hard CI floors={SOFT_CI_FLOOR:.2f}/{HARD_CI_FLOOR:.2f} | AUC floors={SOFT_AUC_FLOOR:.2f}/{HARD_AUC_FLOOR:.2f}"
    )
    run(args)


if __name__ == "__main__":
    main()
