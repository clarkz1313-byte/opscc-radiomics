"""
2_apr_t2b_pan_sc5.py

Task 2B pan-feature joint-objective SC5 search.
Uses staged local datasets built by:
    Apr_2026_task2B/2_apr_T2B_data/2_apr_t2b_build_data.py

Outputs are written directly to Apr_2026_task2B/.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import warnings
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
    ("optuna", None),
    ("sklearn", "scikit-learn"),
    ("sksurv", "scikit-survival"),
]:
    _ensure_import(_module, _package)

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import ExtraSurvivalTrees
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

optuna.logging.set_verbosity(optuna.logging.WARNING)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "2_apr_T2B_data"
TRAIN_FILE = DATA_DIR / "2_apr_t2b_train.csv"
EXT_FILE = DATA_DIR / "2_apr_t2b_ext.csv"

ALL_CSV = ROOT / "t2b_all_results.csv"
CKPT_CSV = ROOT / "t2b_checkpoint.csv"
TOP20_JOINT_CSV = ROOT / "t2b_top20_joint.csv"
TOP20_RFS_CSV = ROOT / "t2b_top20_rfs.csv"
TOP20_HPV_CSV = ROOT / "t2b_top20_hpv.csv"
LOG_MD = ROOT / "t2b_log.md"

SEED = 42
N_FOLDS = 5
N_MIN = 4
N_MAX = 16
PT_MIN = 4
CT_MIN = 1
ALPHA = 0.5
TRIALS = 500
N_BOOT = 1000
N_EST = 200
W_PERF = 0.8
W_STAB = 0.2
STD_THRESHOLD = 0.10

CLINICAL_FEATURES = ["Gender_Male"]

RFS_LOCO_EVENTS_MIN = 3
RFS_LOCO_ENON_MIN = 50
RFS_LOCO_KAPPA = 5.0

HPV_LOCO_HPV_MIN = 2
HPV_LOCO_ENON_MIN = 50
HPV_LOCO_KAPPA = 5.0

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

RANKER_NAMES = [
    "LOCO_RFS_evt",
    "LOCO_RFS_epv_cut",
    "WLOCO_RFS_enon",
    "WLOCO_RFS_epv_cut",
    "LOCO_HPV_evt",
    "LOCO_HPV_epv_cut",
    "WLOCO_HPV_enon",
    "WLOCO_HPV_epv_cut",
    "UNIVAR_RFS",
    "UNIVAR_HPV",
    "UNIVAR_JOINT",
    "GBC_HPV",
    "GBC_JOINT",
    "WLOCO_JOINT_enon",
]

SEEDS = [
    {"n_total": 12, "n_pt": 7, "lr_C": 0.32},
    {"n_total": 16, "n_pt": 8, "lr_C": 0.37},
    {"n_total": 8, "n_pt": 4, "lr_C": 0.32},
    {"n_total": 10, "n_pt": 5, "lr_C": 0.32},
    {"n_total": 12, "n_pt": 8, "lr_C": 0.32},
    {"n_total": 12, "n_pt": 5, "lr_C": 0.32},
    {"n_total": 6, "n_pt": 4, "lr_C": 0.10},
    {"n_total": 16, "n_pt": 9, "lr_C": 0.37},
]


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


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


def _shorten_feature_name(name: str, limit: int = 44) -> str:
    if len(name) <= limit:
        return name
    return name[: limit - 3] + "..."


def _youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        fpr, tpr, thr = roc_curve(y_true, scores)
        youden = tpr - fpr
        idx = int(np.argmax(youden))
        return float(thr[idx])
    except Exception:
        return 0.5


def _smoke_trials(args: argparse.Namespace) -> tuple[int, int]:
    if args.smoke:
        return 2, 50
    return TRIALS, N_BOOT


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(TRAIN_FILE)
    ext_df = pd.read_csv(EXT_FILE)

    required_cols = ["PatientID", "CenterID", "HPV_binary", "Relapse", "RFS"] + CLINICAL_FEATURES + PAN_PT + PAN_CT
    for label, df in [("train", train_df), ("ext", ext_df)]:
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {label}: {missing}")

    assert len(train_df) == 87, f"Expected 87 train rows, got {len(train_df)}"
    assert len(ext_df) == 27, f"Expected 27 ext rows, got {len(ext_df)}"

    train_df["CenterID"] = train_df["CenterID"].astype(int)
    train_df["HPV_binary"] = train_df["HPV_binary"].astype(int)
    train_df["Relapse"] = train_df["Relapse"].astype(int)
    train_df["RFS"] = train_df["RFS"].astype(float)
    ext_df["CenterID"] = ext_df["CenterID"].astype(int)
    ext_df["HPV_binary"] = ext_df["HPV_binary"].astype(int)
    ext_df["Relapse"] = ext_df["Relapse"].astype(int)
    ext_df["RFS"] = ext_df["RFS"].astype(float)

    print(
        f"Data loaded: train n={len(train_df)} (HPV-={int((train_df['HPV_binary'] == 0).sum())}, "
        f"Rel={int(train_df['Relapse'].sum())}) | ext n={len(ext_df)} "
        f"(HPV-={int((ext_df['HPV_binary'] == 0).sum())}, Rel={int(ext_df['Relapse'].sum())})"
    )
    return train_df, ext_df


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

    x_train = np.hstack([sc_clin.transform(x_clin_tr), sc_pt.transform(x_pt_tr), sc_ct.transform(x_ct_tr)])
    x_valid = np.hstack([sc_clin.transform(x_clin_vl), sc_pt.transform(x_pt_vl), sc_ct.transform(x_ct_vl)])
    return x_train, x_valid


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
        elif mode == "w_enon":
            include = True
            w_c, use_shrink = max(1.0, enon), True
        elif mode == "w_epv_cut":
            include = e_c >= RFS_LOCO_EVENTS_MIN and enon >= RFS_LOCO_ENON_MIN
            w_c, use_shrink = max(1.0, enon), True
        else:
            raise ValueError(mode)

        if not include:
            continue

        centre_weights[fold_i] = w_c
        x_val, y_val = x[mask_val], y_surv[mask_val]
        for feat_i in range(n_feats):
            try:
                ci = float(concordance_index_censored(y_val["event"], y_val["time"], x_val[:, feat_i])[0])
                ci = max(ci, 1.0 - ci)
            except Exception:
                ci = 0.5
            if use_shrink:
                ci = (e_c * ci + RFS_LOCO_KAPPA * 0.5) / (e_c + RFS_LOCO_KAPPA)
            ci_matrix[fold_i, feat_i] = ci

    valid = centre_weights > 0
    if not np.any(valid):
        return np.full(n_feats, 0.5)
    matrix = np.where(np.isnan(ci_matrix[valid]), 0.5, ci_matrix[valid])
    weights = centre_weights[valid]
    if mode in {"w_enon", "w_epv_cut"}:
        return np.sum(matrix * weights[:, None], axis=0) / weights.sum()
    return np.mean(matrix, axis=0)


def _hpv_loco_rank(train_df: pd.DataFrame, feat_names: list[str], mode: str) -> np.ndarray:
    centre_ids = train_df["CenterID"].to_numpy()
    y_hpv = train_df["HPV_binary"].to_numpy(dtype=int)
    x = train_df[feat_names].to_numpy(dtype=float)
    centres = np.unique(centre_ids)
    n_feats = len(feat_names)
    auc_matrix = np.full((len(centres), n_feats), np.nan)
    centre_weights = np.zeros(len(centres), dtype=float)

    for fold_i, held in enumerate(centres):
        mask_val = centre_ids == held
        n_c = int(mask_val.sum())
        if n_c < 2:
            continue
        y_val = y_hpv[mask_val]
        hpv_neg = int((y_val == 0).sum())
        hpv_pos = int((y_val == 1).sum())
        enon = float(hpv_neg * hpv_pos)

        if mode == "loco_evt":
            include = hpv_neg >= HPV_LOCO_HPV_MIN
            w_c, use_shrink = 1.0, False
        elif mode == "loco_epv_cut":
            include = hpv_neg >= HPV_LOCO_HPV_MIN and enon >= HPV_LOCO_ENON_MIN
            w_c, use_shrink = 1.0, False
        elif mode == "w_enon":
            include = True
            w_c, use_shrink = max(1.0, enon), True
        elif mode == "w_epv_cut":
            include = hpv_neg >= HPV_LOCO_HPV_MIN and enon >= HPV_LOCO_ENON_MIN
            w_c, use_shrink = max(1.0, enon), True
        else:
            raise ValueError(mode)

        if not include:
            continue

        centre_weights[fold_i] = w_c
        x_val = x[mask_val]
        for feat_i in range(n_feats):
            auc = _safe_auc(y_val, x_val[:, feat_i])
            if use_shrink:
                auc = (hpv_neg * auc + HPV_LOCO_KAPPA * 0.5) / (hpv_neg + HPV_LOCO_KAPPA)
            auc_matrix[fold_i, feat_i] = auc

    valid = centre_weights > 0
    if not np.any(valid):
        return np.full(n_feats, 0.5)
    matrix = np.where(np.isnan(auc_matrix[valid]), 0.5, auc_matrix[valid])
    weights = centre_weights[valid]
    if mode in {"w_enon", "w_epv_cut"}:
        return np.sum(matrix * weights[:, None], axis=0) / weights.sum()
    return np.mean(matrix, axis=0)


def _univar_rfs(train_df: pd.DataFrame, feat_names: list[str]) -> np.ndarray:
    y_surv = Surv.from_arrays(event=train_df["Relapse"].astype(bool), time=train_df["RFS"])
    scores = []
    for feat in feat_names:
        try:
            ci = float(concordance_index_censored(y_surv["event"], y_surv["time"], train_df[feat].to_numpy(dtype=float))[0])
            scores.append(max(ci, 1.0 - ci))
        except Exception:
            scores.append(0.5)
    return np.asarray(scores, dtype=float)


def _univar_hpv(train_df: pd.DataFrame, feat_names: list[str]) -> np.ndarray:
    y = train_df["HPV_binary"].to_numpy(dtype=int)
    scores = []
    for feat in feat_names:
        scores.append(_safe_auc(y, train_df[feat].to_numpy(dtype=float)))
    return np.asarray(scores, dtype=float)


def _gbc_hpv_rank(train_df: pd.DataFrame, feat_names: list[str]) -> np.ndarray:
    x = train_df[feat_names].to_numpy(dtype=float)
    y = train_df["HPV_binary"].to_numpy(dtype=int)
    scaler = StandardScaler().fit(x)
    x_sc = scaler.transform(x)
    gbc = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        random_state=SEED,
    )
    gbc.fit(x_sc, y)
    return gbc.feature_importances_


def compute_all_rankings(train_df: pd.DataFrame) -> dict[str, dict[str, list[int]]]:
    all_feats = PAN_PT + PAN_CT
    print("Pre-computing rankers...")

    scores_rfs = {
        "LOCO_RFS_evt": _rfs_loco_rank(train_df, all_feats, "loco_evt"),
        "LOCO_RFS_epv_cut": _rfs_loco_rank(train_df, all_feats, "loco_epv_cut"),
        "WLOCO_RFS_enon": _rfs_loco_rank(train_df, all_feats, "w_enon"),
        "WLOCO_RFS_epv_cut": _rfs_loco_rank(train_df, all_feats, "w_epv_cut"),
    }
    scores_hpv = {
        "LOCO_HPV_evt": _hpv_loco_rank(train_df, all_feats, "loco_evt"),
        "LOCO_HPV_epv_cut": _hpv_loco_rank(train_df, all_feats, "loco_epv_cut"),
        "WLOCO_HPV_enon": _hpv_loco_rank(train_df, all_feats, "w_enon"),
        "WLOCO_HPV_epv_cut": _hpv_loco_rank(train_df, all_feats, "w_epv_cut"),
    }
    rfs_u = _univar_rfs(train_df, all_feats)
    hpv_u = _univar_hpv(train_df, all_feats)
    scores_univar = {
        "UNIVAR_RFS": rfs_u,
        "UNIVAR_HPV": hpv_u,
        "UNIVAR_JOINT": 0.5 * rfs_u + 0.5 * hpv_u,
    }

    gbc_hpv_raw = _gbc_hpv_rank(train_df, all_feats)

    def _norm(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)
        mn = float(arr.min())
        mx = float(arr.max())
        return (arr - mn) / (mx - mn + 1e-12)

    scores_gbc = {
        "GBC_HPV": gbc_hpv_raw,
        "GBC_JOINT": 0.5 * _norm(gbc_hpv_raw) + 0.5 * _norm(rfs_u),
    }
    scores_joint = {
        "WLOCO_JOINT_enon": 0.5 * scores_rfs["WLOCO_RFS_enon"] + 0.5 * scores_hpv["WLOCO_HPV_enon"],
    }

    all_scores = {**scores_rfs, **scores_hpv, **scores_univar, **scores_gbc, **scores_joint}
    n_pt = len(PAN_PT)
    rankings: dict[str, dict[str, list[int]]] = {}
    for name, scores in all_scores.items():
        pt_scores = scores[:n_pt]
        ct_scores = scores[n_pt:]
        pt_order = np.argsort(pt_scores)[::-1].tolist()
        ct_order = np.argsort(ct_scores)[::-1].tolist()
        rankings[name] = {"pt": pt_order, "ct": ct_order}
        print(
            f"  {name:18s} top PT={_shorten_feature_name(PAN_PT[pt_order[0]])} | "
            f"top CT={_shorten_feature_name(PAN_CT[ct_order[0]])}"
        )
    return rankings


def make_objective(
    ranker_name: str,
    rankings: dict[str, dict[str, list[int]]],
    train_df: pd.DataFrame,
) -> callable:
    pt_order = rankings[ranker_name]["pt"]
    ct_order = rankings[ranker_name]["ct"]

    def objective(trial: optuna.Trial) -> float:
        n_total = trial.suggest_int("n_total", N_MIN, N_MAX)
        n_pt_max = min(len(pt_order), n_total - CT_MIN - 1)
        if n_pt_max < PT_MIN:
            return -1.0
        n_pt = trial.suggest_int("n_pt", PT_MIN, n_pt_max)
        n_ct = n_total - 1 - n_pt
        if n_ct < CT_MIN or n_ct > len(ct_order):
            return -1.0

        lr_c = trial.suggest_float("lr_C", 1e-4, 10.0, log=True)
        feat_pt = [PAN_PT[i] for i in pt_order[:n_pt]]
        feat_ct = [PAN_CT[i] for i in ct_order[:n_ct]]

        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        oof_ci_list: list[float] = []
        oof_auc_list: list[float] = []

        for tr_idx, vl_idx in kf.split(train_df):
            tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
            vl_df = train_df.iloc[vl_idx].reset_index(drop=True)
            x_tr, x_vl = _scale_blocks(tr_df, vl_df, feat_pt, feat_ct)

            y_surv_tr = Surv.from_arrays(event=tr_df["Relapse"].astype(bool), time=tr_df["RFS"])
            y_surv_vl = Surv.from_arrays(event=vl_df["Relapse"].astype(bool), time=vl_df["RFS"])
            y_hpv_tr = tr_df["HPV_binary"].to_numpy(dtype=int)
            y_hpv_vl = vl_df["HPV_binary"].to_numpy(dtype=int)

            est = ExtraSurvivalTrees(n_estimators=N_EST, random_state=SEED, n_jobs=-1)
            est.fit(x_tr, y_surv_tr)
            risk = est.predict(x_vl)
            oof_ci_list.append(_safe_ci(y_surv_vl, risk))

            clf = LogisticRegression(
                C=lr_c,
                penalty="l2",
                solver="lbfgs",
                class_weight="balanced",
                max_iter=2000,
                random_state=SEED,
            )
            clf.fit(x_tr, y_hpv_tr)
            proba = clf.predict_proba(x_vl)[:, 1]
            try:
                auc = float(roc_auc_score(y_hpv_vl, proba))
            except Exception:
                auc = 0.5
            oof_auc_list.append(auc)

        oof_ci = float(np.mean(oof_ci_list))
        oof_auc = float(np.mean(oof_auc_list))
        fold_std_ci = float(np.std(oof_ci_list))
        fold_std_auc = float(np.std(oof_auc_list))
        joint = ALPHA * oof_ci + (1.0 - ALPHA) * oof_auc
        stab_penalty = max(0.0, 1.0 - max(fold_std_ci, fold_std_auc) / STD_THRESHOLD)
        score = W_PERF * joint + W_STAB * stab_penalty

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

        return score

    return objective


def evaluate_on_ext(
    feat_pt: list[str],
    feat_ct: list[str],
    lr_c: float,
    train_df: pd.DataFrame,
    ext_df: pd.DataFrame,
    n_boot: int,
) -> dict[str, float]:
    """Evaluate a feature set on external cohort.

    When n_boot=0 only point estimates are returned (boots set to NaN).
    Bootstrap is reserved for per-ranker best trials to keep runtime tractable.
    """
    x_tr, x_ext = _scale_blocks(train_df, ext_df, feat_pt, feat_ct)

    y_surv_tr = Surv.from_arrays(event=train_df["Relapse"].astype(bool), time=train_df["RFS"])
    y_surv_ext = Surv.from_arrays(event=ext_df["Relapse"].astype(bool), time=ext_df["RFS"])
    y_hpv_tr = train_df["HPV_binary"].to_numpy(dtype=int)
    y_hpv_ext = ext_df["HPV_binary"].to_numpy(dtype=int)

    est = ExtraSurvivalTrees(n_estimators=N_EST, random_state=SEED, n_jobs=-1)
    est.fit(x_tr, y_surv_tr)
    risk_ext = est.predict(x_ext)
    ext_ci = _safe_ci(y_surv_ext, risk_ext)

    clf = LogisticRegression(
        C=lr_c,
        penalty="l2",
        solver="lbfgs",
        class_weight="balanced",
        max_iter=2000,
        random_state=SEED,
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
    ext_spe = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
    ext_sen = float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan

    boot_ci_lo = boot_ci_hi = boot_auc_lo = boot_auc_hi = np.nan
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
                auc_boots.append(np.nan)
        boot_ci_lo = float(np.nanpercentile(ci_boots, 2.5))
        boot_ci_hi = float(np.nanpercentile(ci_boots, 97.5))
        boot_auc_lo = float(np.nanpercentile(auc_boots, 2.5))
        boot_auc_hi = float(np.nanpercentile(auc_boots, 97.5))

    return {
        "ext_ci": ext_ci,
        "ext_auc": ext_auc,
        "ext_ba": ext_ba,
        "ext_spe": ext_spe,
        "ext_sen": ext_sen,
        "boot_ci_lo": boot_ci_lo,
        "boot_ci_hi": boot_ci_hi,
        "boot_auc_lo": boot_auc_lo,
        "boot_auc_hi": boot_auc_hi,
    }


def _write_outputs(all_rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(all_rows)
    df.to_csv(ALL_CSV, index=False)
    df.sort_values(["joint_score", "ext_ci", "ext_auc"], ascending=False).head(20).to_csv(TOP20_JOINT_CSV, index=False)
    df.sort_values(["ext_ci", "joint_score"], ascending=False).head(20).to_csv(TOP20_RFS_CSV, index=False)
    df.sort_values(["ext_auc", "joint_score"], ascending=False).head(20).to_csv(TOP20_HPV_CSV, index=False)
    return df


def _trial_has_feature_attrs(trial: optuna.trial.FrozenTrial) -> bool:
    required = {"feat_pt", "feat_ct", "lr_C", "n_total", "n_pt", "n_ct", "joint_score", "oof_ci", "oof_auc"}
    return required.issubset(trial.user_attrs.keys())


def _summary_table(df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    rows = []
    for ranker in RANKER_NAMES:
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
    summary_df = pd.DataFrame(rows)
    lines = [
        "=== Task 2B Pan-SC5 Summary ===",
        "Training: n=87 (HPV-=26, Relapses=20) | External CHUS: n=27 (HPV-=7, Rel=5)",
        "Feature pool: 1 clin + 15 PT + 11 CT = 27",
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
            f"Outputs: {ROOT}",
        ]
    )
    return "\n".join(lines), summary_df


def _write_log(df: pd.DataFrame, started_at: float, smoke: bool) -> None:
    summary_text, _ = _summary_table(df)
    lines = [
        "# Task 2B Pan-SC5 Log",
        "",
        f"- Started: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(started_at))}",
        f"- Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Smoke mode: {smoke}",
        f"- Train rows: 87",
        f"- Ext rows: 27",
        "",
        "## Summary",
        "```text",
        summary_text,
        "```",
        "",
        "## Top 3 per Ranker",
    ]
    for ranker in RANKER_NAMES:
        sub = df[df["ranker"] == ranker].sort_values(["joint_score", "ext_ci", "ext_auc"], ascending=False).head(3)
        if sub.empty:
            continue
        lines.append(f"### {ranker}")
        for _, row in sub.iterrows():
            lines.append(
                f"- joint={row['joint_score']:.3f} | ext_ci={row['ext_ci']:.3f} | ext_auc={row['ext_auc']:.3f} | "
                f"N={int(row['n_total'])} | PT={int(row['n_pt'])} | CT={int(row['n_ct'])} | "
                f"PT={row['feat_pt']} | CT={row['feat_ct']}"
            )
        lines.append("")
    LOG_MD.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    started_at = time.time()
    trials, n_boot = _smoke_trials(args)
    train_df, ext_df = load_data()
    rankings = compute_all_rankings(train_df)
    if args.rankers:
        requested = [name.strip() for name in args.rankers.split(",") if name.strip()]
        ranker_names = [name for name in RANKER_NAMES if name in requested]
    else:
        ranker_names = list(RANKER_NAMES)

    print(f"Running {len(ranker_names)} rankers | trials per ranker={trials} | bootstrap={n_boot}")
    all_rows: list[dict] = []

    for ranker_name in ranker_names:
        print(f"\n=== Ranker: {ranker_name} ({trials} trials) ===")
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
        for seed_cfg in SEEDS:
            n_ct = seed_cfg["n_total"] - 1 - seed_cfg["n_pt"]
            if n_ct >= CT_MIN:
                study.enqueue_trial(seed_cfg)
        objective = make_objective(ranker_name, rankings, train_df)
        study.optimize(objective, n_trials=trials, show_progress_bar=False)

        # Evaluate all trials with point estimates only (n_boot=0 — fast)
        # Bootstrap CI95 is reserved for the per-ranker best trial only
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        complete_with_attrs = [t for t in completed if _trial_has_feature_attrs(t)]
        skipped = len(completed) - len(complete_with_attrs)
        if not complete_with_attrs:
            raise RuntimeError(f"No valid completed trials with feature metadata for ranker {ranker_name}")

        best_trial = max(complete_with_attrs, key=lambda t: t.value or -1.0)

        ranker_rows: list[dict] = []
        for trial in complete_with_attrs:
            feat_pt = trial.user_attrs["feat_pt"].split("|")
            feat_ct = trial.user_attrs["feat_ct"].split("|")
            is_best = trial.number == best_trial.number
            # Bootstrap only for the best trial to keep runtime tractable
            ext_metrics = evaluate_on_ext(
                feat_pt, feat_ct, float(trial.user_attrs["lr_C"]),
                train_df, ext_df, n_boot if is_best else 0,
            )
            row = {
                "trial_no": int(trial.number),
                "ranker": ranker_name,
                **trial.user_attrs,
                **ext_metrics,
            }
            ranker_rows.append(row)

        all_rows.extend(ranker_rows)
        pd.DataFrame(all_rows).to_csv(CKPT_CSV, index=False)
        best_row = next(r for r in ranker_rows if r["trial_no"] == best_trial.number)
        print(
            f"  Best obj={study.best_value:.4f} | ext_ci={best_row['ext_ci']:.3f} "
            f"ext_auc={best_row['ext_auc']:.3f} | N={int(best_row['n_total'])} "
            f"PT={int(best_row['n_pt'])} CT={int(best_row['n_ct'])} | skipped_no_attrs={skipped}"
        )

    df = _write_outputs(all_rows)
    summary_text, _ = _summary_table(df)
    print("\n" + summary_text)
    _write_log(df, started_at, args.smoke)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 2B pan-feature joint-objective SC5 search")
    parser.add_argument("--smoke", action="store_true", help="Run a short validation pass")
    parser.add_argument("--rankers", type=str, default="", help="Comma-separated ranker subset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"ROOT={ROOT}")
    print(f"DATA_DIR={DATA_DIR}")
    print(f"TRAIN_FILE={TRAIN_FILE}")
    print(f"EXT_FILE={EXT_FILE}")
    run(args)


if __name__ == "__main__":
    main()
