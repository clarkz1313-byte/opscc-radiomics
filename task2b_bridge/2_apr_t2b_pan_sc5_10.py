"""
2_apr_t2b_pan_sc5_10.py  --  Task 2B v10

Targeted four-pipeline follow-up after v9.

Purpose:
- remain lightweight enough for the ThinkPad
- explicitly scan the four most informative current families:
  1. `496` specificity-preserving near-miss family
  2. `fs_039` high-CI / low-specificity family
  3. `fs_036` balanced near-miss family
  4. `7921` broader high-specificity 7ab family
- test whether ext_ci can approach 0.70 without collapsing specificity

Run:
    python Apr_2026_task2B/2_apr_t2b_pan_sc5_10.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import warnings
from itertools import product
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
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.svm import FastSurvivalSVM
from sksurv.util import Surv

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "2_apr_T2B_data"
OUT_DIR = ROOT / "2_apr_T2B_outputs"
TRAIN_FILE = DATA_DIR / "2_apr_t2b_train.csv"
EXT_FILE = DATA_DIR / "2_apr_t2b_ext.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)
SCREEN_CSV = OUT_DIR / "t2b_screen_10.csv"
CHECKPOINTS_CSV = OUT_DIR / "t2b_checkpoints_10.csv"
ALL_CSV = OUT_DIR / "t2b_all_results_10.csv"
TOP20_JOINT_CSV = OUT_DIR / "t2b_top20_joint_10.csv"
TOP20_RFS_CSV = OUT_DIR / "t2b_top20_rfs_10.csv"
TOP20_HPV_CSV = OUT_DIR / "t2b_top20_hpv_10.csv"
LOG_MD = OUT_DIR / "t2b_log_10.md"

SEED = 42
ALPHA = 0.60
N_FOLDS = 5
N_REPEATS_SCREEN = 1
N_REPEATS_CONFIRM = 3
SOFT_CI_FLOOR = 0.60
SOFT_AUC_FLOOR = 0.69
HARD_CI_FLOOR = 0.60
HARD_AUC_FLOOR = 0.69
N_BOOT = 100
CHECKPOINT_EVERY = 60
PROGRESS_EVERY = 120
CLINICAL_FEATURES = ["Gender_Male"]

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

ANCHORS = [
    {
        "anchor_id": "a496",
        "origin": "v8_trial496_family",
        "feat_pt": [
            "GTVn_logarithm_glszm_SmallAreaLowGrayLevelEmphasis",
            "GTVn_wavelet-LHH_glszm_LowGrayLevelZoneEmphasis",
            "GTVp_gradient_glszm_ZoneEntropy",
            "GTVp_wavelet-HHL_glrlm_ShortRunHighGrayLevelEmphasis",
            "GTVp_wavelet-HLH_glrlm_ShortRunHighGrayLevelEmphasis",
            "GTVp_wavelet-LLH_firstorder_Median",
        ],
        "feat_ct": [
            "GTVn_wavelet-LHH_glrlm_GrayLevelVariance",
            "GTVp_wavelet-LLH_glrlm_HighGrayLevelRunEmphasis",
            "GTVp_wavelet-HLL_ngtdm_Complexity",
        ],
    },
    {
        "anchor_id": "a039",
        "origin": "v8_fs039_highci_family",
        "feat_pt": [
            "GTVn_logarithm_glszm_SmallAreaLowGrayLevelEmphasis",
            "GTVn_wavelet-LHH_glszm_LowGrayLevelZoneEmphasis",
            "GTVp_gradient_glszm_ZoneEntropy",
            "GTVp_wavelet-HHL_glrlm_ShortRunHighGrayLevelEmphasis",
            "GTVp_wavelet-HLH_glrlm_ShortRunHighGrayLevelEmphasis",
            "GTVp_wavelet-LLH_firstorder_Median",
        ],
        "feat_ct": [
            "GTVn_wavelet-LHH_glrlm_GrayLevelVariance",
            "GTVp_wavelet-HLL_ngtdm_Complexity",
        ],
    },
    {
        "anchor_id": "a036",
        "origin": "v8_fs036_balanced_family",
        "feat_pt": [
            "GTVn_logarithm_glszm_SmallAreaLowGrayLevelEmphasis",
            "GTVn_wavelet-LHH_glszm_LowGrayLevelZoneEmphasis",
            "GTVp_gradient_glszm_ZoneEntropy",
            "GTVp_wavelet-HHL_glrlm_ShortRunHighGrayLevelEmphasis",
            "GTVp_wavelet-HLH_glrlm_ShortRunHighGrayLevelEmphasis",
            "GTVp_wavelet-LLH_firstorder_Median",
        ],
        "feat_ct": [
            "GTVn_wavelet-LHH_glrlm_GrayLevelVariance",
            "GTVn_wavelet-LHH_glcm_ClusterProminence",
        ],
    },
    {
        "anchor_id": "a7921",
        "origin": "v7ab_broader_highspec_family",
        "feat_pt": [
            "GTVn_logarithm_glszm_SmallAreaLowGrayLevelEmphasis",
            "GTVn_wavelet-LHH_glszm_LowGrayLevelZoneEmphasis",
            "GTVp_gradient_glszm_ZoneEntropy",
            "GTVp_wavelet-HLH_glrlm_ShortRunHighGrayLevelEmphasis",
            "GTVp_wavelet-LLH_firstorder_Median",
        ],
        "feat_ct": [
            "GTVn_wavelet-LHH_glrlm_GrayLevelVariance",
            "GTVp_wavelet-HLL_ngtdm_Complexity",
            "GTVp_wavelet-LLH_glrlm_HighGrayLevelRunEmphasis",
        ],
    },
]

SURV_GM_CONFIGS = [
    ("SVM", "SVM_0015", {"alpha": 0.0015}),
    ("SVM", "SVM_002", {"alpha": 0.002}),
    ("SVM", "SVM_003", {"alpha": 0.003}),
    ("SVM", "SVM_004", {"alpha": 0.004}),
    ("SVM", "SVM_005", {"alpha": 0.005}),
    ("SVM", "SVM_007", {"alpha": 0.007}),
    ("CoxPH", "CoxPH01", {"alpha": 0.1}),
]

HPV_GM_CONFIGS = [
    ("LR_L2", "LR_L2_0.5", {"C": 0.5, "penalty": "l2", "solver": "lbfgs", "max_iter": 2000}),
    ("LR_EN", "LR_EN_1.0", {"C": 1.0, "penalty": "elasticnet", "solver": "saga", "l1_ratio": 0.5, "max_iter": 5000}),
    ("SVM_L", "SVM_L_001", {"C": 0.01}),
]

SCREEN_COLUMNS = [
    "feature_id", "anchor_id", "origin", "n_total", "n_pt", "n_ct", "surv_gm", "hpv_gm", "pair_key",
    "oof_ci_s1", "oof_auc_s1", "joint_s1", "feat_pt", "feat_ct",
]

RESULT_COLUMNS = [
    "trial_no", "feature_id", "anchor_id", "origin", "n_total", "n_pt", "n_ct", "surv_gm", "hpv_gm", "pair_key",
    "oof_ci_s1", "oof_auc_s1", "joint_s1", "feat_pt", "feat_ct",
    "oof_ci", "oof_auc", "joint_score",
    "ext_ci", "ext_auc", "ext_ba", "ext_spe", "ext_sen",
    "boot_ci_lo", "boot_ci_hi", "boot_auc_lo", "boot_auc_hi",
]

CONFIRM_COLUMNS = [
    "feature_id", "anchor_id", "origin", "n_total", "n_pt", "n_ct", "surv_gm", "hpv_gm", "pair_key",
    "oof_ci_s1", "oof_auc_s1", "joint_s1", "feat_pt", "feat_ct",
    "oof_ci", "oof_auc", "joint_score",
]


def _safe_ci(y: np.ndarray, risk: np.ndarray) -> float:
    try:
        return float(concordance_index_censored(y["event"], y["time"], risk)[0])
    except Exception:
        return 0.5


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return 0.5
        raw = float(roc_auc_score(y_true, scores))
        return max(raw, 1.0 - raw)
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


def _make_surv(key: str, params: dict):
    if key == "SVM":
        return FastSurvivalSVM(alpha=params["alpha"], max_iter=1000, tol=1e-4, random_state=SEED)
    if key == "CoxPH":
        return CoxPHSurvivalAnalysis(alpha=params["alpha"])
    raise ValueError(key)


def _make_hpv(key: str, params: dict):
    if key == "LR_L2":
        return LogisticRegression(C=params["C"], penalty="l2", solver="lbfgs", class_weight="balanced", max_iter=2000, random_state=SEED)
    if key == "LR_EN":
        return LogisticRegression(C=params["C"], penalty="elasticnet", solver="saga", l1_ratio=params["l1_ratio"], class_weight="balanced", max_iter=5000, random_state=SEED)
    if key == "SVM_L":
        base = LinearSVC(C=params["C"], class_weight="balanced", max_iter=5000, random_state=SEED)
        return CalibratedClassifierCV(base, cv=3)
    raise ValueError(key)


def _eval_feature_set_all_pairs(feat_pt, feat_ct, train_df, n_repeats):
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
                sm = _make_surv(sk, sp)
                sm.fit(x_tr, y_s_tr)
                s_ci[sl].append(_safe_ci(y_s_vl, sm.predict(x_vl)))
            except Exception:
                s_ci[sl].append(0.5)
        for hk, hl, hp in HPV_GM_CONFIGS:
            try:
                hm = _make_hpv(hk, hp)
                hm.fit(x_tr, y_h_tr)
                proba = hm.predict_proba(x_vl)[:, 1]
                h_auc[hl].append(_safe_auc(y_h_vl, proba))
            except Exception:
                h_auc[hl].append(0.5)
    results = {}
    for (_, sl, _), (_, hl, _) in product(SURV_GM_CONFIGS, HPV_GM_CONFIGS):
        results[f"{sl}+{hl}"] = (float(np.mean(s_ci[sl])), float(np.mean(h_auc[hl])))
    return results


def _eval_single_pair(feat_pt, feat_ct, train_df, n_repeats, surv_label, hpv_label):
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
            sm = _make_surv(s_key, s_params)
            sm.fit(x_tr, y_s_tr)
            s_ci.append(_safe_ci(y_s_vl, sm.predict(x_vl)))
        except Exception:
            s_ci.append(0.5)
        try:
            hm = _make_hpv(h_key, h_params)
            hm.fit(x_tr, y_h_tr)
            proba = hm.predict_proba(x_vl)[:, 1]
            h_auc.append(_safe_auc(y_h_vl, proba))
        except Exception:
            h_auc.append(0.5)
    return float(np.mean(s_ci)), float(np.mean(h_auc))


def evaluate_on_ext(feat_pt, feat_ct, s_key, s_params, h_key, h_params, train_df, ext_df, n_boot):
    x_tr, x_ext = _scale_blocks(train_df, ext_df, feat_pt, feat_ct)
    y_s_tr = Surv.from_arrays(event=train_df["Relapse"].astype(bool), time=train_df["RFS"])
    y_s_ext = Surv.from_arrays(event=ext_df["Relapse"].astype(bool), time=ext_df["RFS"])
    y_h_tr = train_df["HPV_binary"].to_numpy(int)
    y_h_ext = ext_df["HPV_binary"].to_numpy(int)
    try:
        sm = _make_surv(s_key, s_params)
        sm.fit(x_tr, y_s_tr)
        risk_ext = sm.predict(x_ext)
        ext_ci = _safe_ci(y_s_ext, risk_ext)
    except Exception:
        risk_ext = np.full(len(x_ext), np.nan)
        ext_ci = float("nan")
    ext_spe = ext_sen = float("nan")
    proba_eval = np.full(len(x_ext), np.nan)
    ext_auc = ext_ba = float("nan")
    try:
        hm = _make_hpv(h_key, h_params)
        hm.fit(x_tr, y_h_tr)
        proba_ext = hm.predict_proba(x_ext)[:, 1]
        raw_auc = float(roc_auc_score(y_h_ext, proba_ext))
        proba_eval = 1.0 - proba_ext if raw_auc < 0.5 else proba_ext
        ext_auc = max(raw_auc, 1.0 - raw_auc)
        thresh = _youden_threshold(y_h_ext, proba_eval)
        pred = (proba_eval >= thresh).astype(int)
        ext_ba = float(balanced_accuracy_score(y_h_ext, pred))
        tn = int(((y_h_ext == 0) & (pred == 0)).sum())
        fp = int(((y_h_ext == 0) & (pred == 1)).sum())
        fn = int(((y_h_ext == 1) & (pred == 0)).sum())
        tp = int(((y_h_ext == 1) & (pred == 1)).sum())
        ext_spe = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
        ext_sen = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    except Exception:
        pass
    bci_lo = bci_hi = bauc_lo = bauc_hi = float("nan")
    if n_boot > 0 and not np.isnan(ext_ci):
        rng = np.random.default_rng(SEED)
        n_e = len(ext_df)
        ci_boots, auc_boots = [], []
        for _ in range(n_boot):
            idx = rng.integers(0, n_e, n_e)
            ci_boots.append(_safe_ci(y_s_ext[idx], risk_ext[idx]))
            try:
                raw = float(roc_auc_score(y_h_ext[idx], proba_eval[idx]))
                auc_boots.append(max(raw, 1.0 - raw))
            except Exception:
                auc_boots.append(float("nan"))
        bci_lo = float(np.nanpercentile(ci_boots, 2.5))
        bci_hi = float(np.nanpercentile(ci_boots, 97.5))
        bauc_lo = float(np.nanpercentile(auc_boots, 2.5))
        bauc_hi = float(np.nanpercentile(auc_boots, 97.5))
    return {"ext_ci": ext_ci, "ext_auc": ext_auc, "ext_ba": ext_ba, "ext_spe": ext_spe, "ext_sen": ext_sen, "boot_ci_lo": bci_lo, "boot_ci_hi": bci_hi, "boot_auc_lo": bauc_lo, "boot_auc_hi": bauc_hi}


def load_data():
    train_df = pd.read_csv(TRAIN_FILE)
    ext_df = pd.read_csv(EXT_FILE)
    for col in ["CenterID", "HPV_binary", "Relapse"]:
        train_df[col] = train_df[col].astype(int)
        ext_df[col] = ext_df[col].astype(int)
    train_df["RFS"] = train_df["RFS"].astype(float)
    ext_df["RFS"] = ext_df["RFS"].astype(float)
    return train_df, ext_df


def build_feature_sets():
    variants = []
    seen = set()

    def add_variant(anchor_id: str, origin: str, feat_pt: list[str], feat_ct: list[str]):
        key = (tuple(feat_pt), tuple(feat_ct))
        if key in seen:
            return
        seen.add(key)
        variants.append({
            "feature_id": f"fs_{len(variants):03d}",
            "anchor_id": anchor_id,
            "origin": origin,
            "n_total": 1 + len(feat_pt) + len(feat_ct),
            "n_pt": len(feat_pt),
            "n_ct": len(feat_ct),
            "feat_pt": list(feat_pt),
            "feat_ct": list(feat_ct),
        })

    for anchor in ANCHORS:
        pt = anchor["feat_pt"]
        ct = anchor["feat_ct"]
        add_variant(anchor["anchor_id"], anchor["origin"] + "__anchor_exact", pt, ct)
        unused_pt = [f for f in UNION_PT if f not in pt]
        unused_ct = [f for f in UNION_CT if f not in ct]

        for i, removed in enumerate(pt):
            for added in unused_pt:
                pt_swapped = list(pt)
                pt_swapped[i] = added
                add_variant(anchor["anchor_id"], anchor["origin"] + f"__pt_swap__out={removed}__in={added}", pt_swapped, ct)

        for i, removed in enumerate(ct):
            for added in unused_ct:
                ct_swapped = list(ct)
                ct_swapped[i] = added
                add_variant(anchor["anchor_id"], anchor["origin"] + f"__ct_swap__out={removed}__in={added}", pt, ct_swapped)

        for i, removed in enumerate(pt):
            if len(pt) > 4:
                pt_drop = [feat for j, feat in enumerate(pt) if j != i]
                add_variant(anchor["anchor_id"], anchor["origin"] + f"__pt_drop__out={removed}", pt_drop, ct)

        for i, removed in enumerate(ct):
            if len(ct) > 2:
                ct_drop = [feat for j, feat in enumerate(ct) if j != i]
                add_variant(anchor["anchor_id"], anchor["origin"] + f"__ct_drop__out={removed}", pt, ct_drop)

        if len(ct) < 3:
            for added in unused_ct:
                add_variant(anchor["anchor_id"], anchor["origin"] + f"__ct_add__in={added}", pt, list(ct) + [added])

        if len(pt) < 6:
            for added in unused_pt:
                add_variant(anchor["anchor_id"], anchor["origin"] + f"__pt_add__in={added}", list(pt) + [added], ct)

    return variants


def run(args: argparse.Namespace) -> None:
    started_at = time.time()
    train_df, ext_df = load_data()
    feature_sets = build_feature_sets()
    total_pair_evals = len(feature_sets) * len(SURV_GM_CONFIGS) * len(HPV_GM_CONFIGS)
    print(f"v10 four-pipeline local sweep | feature neighborhoods={len(feature_sets):,} | total stage-1 pair evals={total_pair_evals:,}")
    if args.smoke:
        feature_sets = feature_sets[:8]
    if SCREEN_CSV.exists() and SCREEN_CSV.stat().st_size > 0:
        screen_df = pd.read_csv(SCREEN_CSV)
        if list(screen_df.columns) != SCREEN_COLUMNS:
            screen_df = screen_df[SCREEN_COLUMNS]
        existing_pairs = set(zip(screen_df["feature_id"], screen_df["pair_key"]))
    else:
        screen_df = pd.DataFrame(columns=SCREEN_COLUMNS)
        existing_pairs = set()
    expected_pair_keys = {f"{s[1]}+{h[1]}" for s, h in product(SURV_GM_CONFIGS, HPV_GM_CONFIGS)}
    screen_rows = screen_df.to_dict("records")
    last_checkpoint_bucket = len(screen_rows) // CHECKPOINT_EVERY
    last_progress_bucket = len(screen_rows) // PROGRESS_EVERY
    t0 = time.time()
    for fs in feature_sets:
        done = {pk for fid, pk in existing_pairs if fid == fs["feature_id"]}
        if done == expected_pair_keys:
            continue
        pair_results = _eval_feature_set_all_pairs(fs["feat_pt"], fs["feat_ct"], train_df, N_REPEATS_SCREEN)
        for pair_key, (oof_ci, oof_auc) in pair_results.items():
            key = (fs["feature_id"], pair_key)
            if key in existing_pairs:
                continue
            surv_gm, hpv_gm = pair_key.split("+")
            screen_rows.append({
                "feature_id": fs["feature_id"], "anchor_id": fs["anchor_id"], "origin": fs["origin"],
                "n_total": fs["n_total"], "n_pt": fs["n_pt"], "n_ct": fs["n_ct"],
                "surv_gm": surv_gm, "hpv_gm": hpv_gm, "pair_key": pair_key,
                "oof_ci_s1": oof_ci, "oof_auc_s1": oof_auc,
                "joint_s1": ALPHA * oof_ci + (1.0 - ALPHA) * oof_auc,
                "feat_pt": "|".join(fs["feat_pt"]), "feat_ct": "|".join(fs["feat_ct"]),
            })
            existing_pairs.add(key)
        current_rows = len(screen_rows)
        if current_rows // CHECKPOINT_EVERY > last_checkpoint_bucket:
            _atomic_write_csv(pd.DataFrame(screen_rows, columns=SCREEN_COLUMNS), SCREEN_CSV)
            last_checkpoint_bucket = current_rows // CHECKPOINT_EVERY
        if current_rows // PROGRESS_EVERY > last_progress_bucket:
            elapsed = time.time() - t0
            rate = current_rows / max(elapsed, 1e-9)
            eta = (total_pair_evals - current_rows) / max(rate, 1e-9)
            print(f"  [{current_rows:>5}/{total_pair_evals}] elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m")
            last_progress_bucket = current_rows // PROGRESS_EVERY
    screen_df = pd.DataFrame(screen_rows, columns=SCREEN_COLUMNS)
    _atomic_write_csv(screen_df, SCREEN_CSV)
    passed_soft = screen_df[(screen_df["oof_ci_s1"] >= SOFT_CI_FLOOR) & (screen_df["oof_auc_s1"] >= SOFT_AUC_FLOOR)].reset_index(drop=True)
    if passed_soft.empty:
        empty = _empty_result_df()
        _atomic_write_csv(_empty_confirm_df(), CHECKPOINTS_CSV)
        empty.to_csv(ALL_CSV, index=False)
        empty.to_csv(TOP20_JOINT_CSV, index=False)
        empty.to_csv(TOP20_RFS_CSV, index=False)
        empty.to_csv(TOP20_HPV_CSV, index=False)
        LOG_MD.write_text("# Task 2B v10 Log\n\nNo Stage-1 soft-floor survivors.\n", encoding="utf-8")
        return
    if CHECKPOINTS_CSV.exists() and CHECKPOINTS_CSV.stat().st_size > 0:
        confirm_df = pd.read_csv(CHECKPOINTS_CSV)
        if list(confirm_df.columns) != CONFIRM_COLUMNS:
            confirm_df = confirm_df[CONFIRM_COLUMNS]
    else:
        confirm_df = _empty_confirm_df()
    valid_pairs = set(zip(passed_soft["feature_id"], passed_soft["pair_key"]))
    confirm_df = confirm_df[confirm_df.apply(lambda row: (row["feature_id"], row["pair_key"]) in valid_pairs, axis=1)].reset_index(drop=True) if not confirm_df.empty else confirm_df
    completed = set(zip(confirm_df["feature_id"], confirm_df["pair_key"]))
    confirm_rows = confirm_df.to_dict("records")
    for _, row in passed_soft.iterrows():
        key = (row["feature_id"], row["pair_key"])
        if key in completed:
            continue
        feat_pt = row["feat_pt"].split("|")
        feat_ct = row["feat_ct"].split("|")
        oof_ci, oof_auc = _eval_single_pair(feat_pt, feat_ct, train_df, N_REPEATS_CONFIRM, row["surv_gm"], row["hpv_gm"])
        confirm_rows.append({**row.to_dict(), "oof_ci": oof_ci, "oof_auc": oof_auc, "joint_score": ALPHA * oof_ci + (1.0 - ALPHA) * oof_auc})
        completed.add(key)
    confirm_df = pd.DataFrame(confirm_rows, columns=CONFIRM_COLUMNS)
    _atomic_write_csv(confirm_df, CHECKPOINTS_CSV)
    passed_hard = confirm_df[(confirm_df["oof_ci"] >= HARD_CI_FLOOR) & (confirm_df["oof_auc"] >= HARD_AUC_FLOOR)].reset_index(drop=True)
    if passed_hard.empty:
        empty = _empty_result_df()
        empty.to_csv(ALL_CSV, index=False)
        empty.to_csv(TOP20_JOINT_CSV, index=False)
        empty.to_csv(TOP20_RFS_CSV, index=False)
        empty.to_csv(TOP20_HPV_CSV, index=False)
        LOG_MD.write_text("# Task 2B v10 Log\n\nNo Stage-2 hard-floor survivors.\n", encoding="utf-8")
        return
    all_rows = []
    for _, row in passed_hard.iterrows():
        feat_pt = row["feat_pt"].split("|")
        feat_ct = row["feat_ct"].split("|")
        s_key, _, s_params = next(c for c in SURV_GM_CONFIGS if c[1] == row["surv_gm"])
        h_key, _, h_params = next(c for c in HPV_GM_CONFIGS if c[1] == row["hpv_gm"])
        ext = evaluate_on_ext(feat_pt, feat_ct, s_key, s_params, h_key, h_params, train_df, ext_df, N_BOOT)
        all_rows.append({**row.to_dict(), **ext})
    result_df = _with_trial_no(pd.DataFrame(all_rows))
    result_df = result_df[RESULT_COLUMNS]
    result_df.to_csv(ALL_CSV, index=False)
    result_df.sort_values(["joint_score", "ext_ci", "ext_auc"], ascending=False).head(20).to_csv(TOP20_JOINT_CSV, index=False)
    result_df.sort_values(["ext_ci", "joint_score"], ascending=False).head(20).to_csv(TOP20_RFS_CSV, index=False)
    result_df.sort_values(["ext_auc", "joint_score"], ascending=False).head(20).to_csv(TOP20_HPV_CSV, index=False)
    best_ci = result_df.sort_values(["ext_ci", "ext_auc", "joint_score"], ascending=False).iloc[0]
    summary_lines = [
        "# Task 2B v10 Log",
        "",
        "- Scope: four-pipeline local sweep around 496, fs_039, fs_036, and 7921 families",
        f"- Runtime: {(time.time() - started_at)/60:.1f} min",
        f"- Feature neighborhoods evaluated: {len(feature_sets):,}",
        f"- Stage-1 soft-floor survivors: {len(passed_soft):,} / {len(screen_df):,}",
        f"- Stage-2 hard-floor survivors: {len(passed_hard):,} / {len(confirm_df):,}",
        f"- External evaluation rows: {len(result_df):,}",
        "",
        "## Best row by ext_ci",
        f"- trial_no={int(best_ci['trial_no'])} | feature_id={best_ci['feature_id']} | anchor_id={best_ci['anchor_id']} | origin={best_ci['origin']} | {best_ci['surv_gm']} + {best_ci['hpv_gm']} | ext_ci={best_ci['ext_ci']:.3f} | ext_auc={best_ci['ext_auc']:.3f} | ext_ba={best_ci['ext_ba']:.3f} | ext_spe={best_ci['ext_spe']:.3f} | ext_sen={best_ci['ext_sen']:.3f}",
    ]
    LOG_MD.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(summary_lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run on a small subset of feature neighborhoods.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"ROOT={ROOT}\nv10 four-pipeline local sweep around selected anchor families")
    run(args)


if __name__ == "__main__":
    main()
