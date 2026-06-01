"""
27_mar_T2_dev_ultimate_recheck.py

Task 2 — Ultimate Reproducibility Recheck Script
HPV status binary classification (HPV+ vs HPV−)

Purpose: Unified single-script recheck that replaces the two separate
17_mar_task2_model_dev_LOCO.py and 17_mar_task2_model_dev_INDIV.py.
Confirms selected winners (T31763 INDIV-GBC, T77583 LOCO-WLOCO_enon) remain
top performers under a reduced, representative 5-ranker set. Adds full
classification metrics (Sen/Spec/BA/F1/Precision/NPV/Acc + Youden threshold)
for both test and ext splits. Saves per-patient probabilities for ALL
trio_ok candidates for post-hoc analysis.

Design (mirrors 11_mar_SC3_rad_PT768xCT8_235_9_ultimate_recheck.py structure):
  Rankers: LOCO_evt, LOCO_EPV_CUT, WLOCO_enon  (3 LOCO — Task 1 trio analogs)
           GBC, UNIVAR                          (2 INDIV — primary + simple baseline)
  GMs:     LR_L2, LR_EN, SVM_L, RF
  Coaches: LR_L2, LR_EN, SVM_L, RF
  N_TRIALS=2000 per (ranker × GM) block
  Expected rows: 5 × 4 × 2000 × 4 = 160,000 main rows

New output columns vs original scripts:
  16 classification metrics (8 per split, computed at Youden threshold)
  4 boolean filter flags: trio_ok, balanced_ok, ci_ok, top_candidate

Key winners to confirm:
  INDIV primary:     T31763 GBC/LR_L2×SVM_L 13CT+6PT+2clin ext=0.814 lo95=0.636
  LOCO concordance:  T77583 WLOCO_enon/SVM_L×SVM_L 2CT+8PT+1clin ext=0.793 lo95=0.592

Reference plan:  Mar_2026_task2/27_mar_t2_dev_finals_discussion.md §8
Task 1 analog:   Mar_2026/11_mar_SC3_rad_PT768xCT8_235_9_ultimate_recheck.py

Run:
    cd "D:/Uppsala thesis" && python Mar_2026_task2/27_mar_T2_dev_ultimate_recheck.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("default")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except ImportError:
    pass

import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    balanced_accuracy_score, f1_score,
    precision_score, recall_score, accuracy_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================
# PATHS
# ============================================================
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

CT_FEATURES_FILE = SCRIPT_DIR / "15_mar_T2_final_features" / "T2_CT_S3_2_460_reconstructed_features.csv"
PT_FEATURES_FILE = SCRIPT_DIR / "15_mar_T2_final_features" / "T2_PT_S3_5_531_reconstructed_features.csv"

CT_TRAIN_FILE = SCRIPT_DIR / "12_mar_task2_rad_data" / "13_mar_task2_CT_primary_train.csv"
CT_TEST_FILE  = SCRIPT_DIR / "12_mar_task2_rad_data" / "13_mar_task2_CT_primary_test.csv"
CT_EXT_FILE   = SCRIPT_DIR / "12_mar_task2_rad_data" / "12_mar_task2_CT_primary_ext.csv"
PT_TRAIN_FILE = SCRIPT_DIR / "12_mar_task2_rad_data" / "13_mar_task2_PT_primary_train.csv"
PT_TEST_FILE  = SCRIPT_DIR / "12_mar_task2_rad_data" / "13_mar_task2_PT_primary_test.csv"
PT_EXT_FILE   = SCRIPT_DIR / "12_mar_task2_rad_data" / "12_mar_task2_PT_primary_ext.csv"

# Unified clinical source — same file used by Task 1 final script
# (25_feb_Processed_clinical_reduced.csv is 25_feb_clinical_reduced_dataset stripped to 3 cols)
CLINICAL_FILE = PROJECT_ROOT / "Feb_2026" / "25_feb_clinical_reduced_dataset" / "25_feb_Processed_clinical_reduced.csv"

OUT_DIR  = SCRIPT_DIR / "27_mar_T2_recheck_outputs"
ALL_CSV  = OUT_DIR / "27_mar_t2_recheck_all_results.csv"
CKPT_CSV = OUT_DIR / "27_mar_t2_recheck_checkpoint.csv"
TOP_CSV  = OUT_DIR / "27_mar_t2_recheck_top_results.csv"
PRED_CSV = OUT_DIR / "27_mar_t2_recheck_all_pass_predictions.csv"
LOG_FILE = OUT_DIR / "27_mar_t2_recheck_log.txt"
# OUT_DIR and log file are created inside main() to avoid import-time side effects

# ============================================================
# CONFIG
# ============================================================
SEED              = 42
N_FOLDS           = 5
N_BOOT            = 1000
N_TRIALS          = 2000      # per (ranker × GM) Optuna block
LABEL_COL         = "HPV_binary"
CLINICAL_FEATURES = ["Age", "Gender_Male", "Treatment_CRT"]

TOTAL_N_MIN = 7    # MIN_CLIN(1) + MIN_PT(4) + MIN_CT(2)
TOTAL_N_MAX = 25
MIN_CLIN    = 1
MIN_PT      = 4    # PT generalises better to ext
MIN_CT      = 2

W_PERF, W_STAB, STD_THRESHOLD = 0.7, 0.3, 0.08
PASS_THRESHOLD = 0.70

# LOCO params (same as 17_mar scripts)
LOCO_KAPPA    = 5.0
LOCO_HPV_MIN  = 2
LOCO_ENON_MIN = 50

# GBC ranker params (same as 17_mar INDIV script)
GBC_N_EST   = 100
GBC_LR      = 0.1
GBC_DEPTH   = 3
GBC_SUBSAMP = 0.8

# 5 rankers: 3 LOCO (Task 1 trio analogs) + GBC (primary INDIV) + UNIVAR (simple INDIV)
RANKER_NAMES = ["LOCO_evt", "LOCO_EPV_CUT", "WLOCO_enon", "GBC", "UNIVAR"]
GM_NAMES     = ["LR_L2", "LR_EN", "SVM_L", "RF"]
COACH_NAMES  = ["LR_L2", "LR_EN", "SVM_L", "RF"]

EXCLUDE_COLS = {"PatientID", LABEL_COL, "Relapse", "RFS",
                "Age", "Gender_Male", "Treatment_CRT", "prefix", "centre"}

# Boolean filter thresholds (see §8.4 of 27_mar_t2_dev_finals_discussion.md)
TRIO_OK_OOF_MIN  = 0.70
TRIO_OK_TEST_MIN = 0.70
TRIO_OK_EXT_MIN  = 0.70
BALANCED_OK_BA_EXT_MIN  = 0.70
BALANCED_OK_BA_TEST_MIN = 0.65
CI_OK_LO95_MIN   = 0.57
TOP_CANDIDATE_BA_EXT_MIN  = 0.75
TOP_CANDIDATE_LO95_MIN    = 0.63

# ============================================================
# LOGGING
# ============================================================
class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

# Log handle opened in main() after OUT_DIR is created
_log_fh = None

# ============================================================
# DATA LOADING
# ============================================================
def _load_data():
    ct_feat_list = pd.read_csv(CT_FEATURES_FILE)["Feature"].tolist()
    pt_feat_list = pd.read_csv(PT_FEATURES_FILE)["Feature"].tolist()

    ct_train = pd.read_csv(CT_TRAIN_FILE)
    ct_test  = pd.read_csv(CT_TEST_FILE)
    ct_ext   = pd.read_csv(CT_EXT_FILE)
    pt_train = pd.read_csv(PT_TRAIN_FILE)
    pt_test  = pd.read_csv(PT_TEST_FILE)
    pt_ext   = pd.read_csv(PT_EXT_FILE)

    def _merge(ct_df, pt_df):
        merged = ct_df.merge(
            pt_df[["PatientID"] + pt_feat_list], on="PatientID",
            how="inner", suffixes=("", "_PT"),
        )
        rename = {c: c[:-3] for c in merged.columns if c.endswith("_PT")}
        if rename:
            merged = merged.drop(columns=list(rename.values()), errors="ignore")
            merged = merged.rename(columns=rename)
        return merged

    train_df = _merge(ct_train, pt_train).copy()
    test_df  = _merge(ct_test,  pt_test).copy()
    ext_df   = _merge(ct_ext,   pt_ext).copy()

    # Load clinical from unified source (same as Task 1 final script).
    # Drop any embedded clinical columns from the radiomics CSVs first to
    # avoid ambiguity, then left-merge from the canonical file.
    clin_df = pd.read_csv(CLINICAL_FILE)[["PatientID"] + CLINICAL_FEATURES]
    for df in (train_df, test_df, ext_df):
        drop_cols = [c for c in CLINICAL_FEATURES if c in df.columns]
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)
    train_df = train_df.merge(clin_df, on="PatientID", how="left")
    test_df  = test_df.merge(clin_df,  on="PatientID", how="left")
    ext_df   = ext_df.merge(clin_df,   on="PatientID", how="left")

    def _centre(pid):
        pfx = str(pid).split("-")[0].upper()
        return {"CHUM": "CHUM", "CHUP": "CHUP", "HGJ": "HGJ",
                "HMR": "HMR", "CHUS": "CHUS"}.get(pfx, pfx)

    train_df["centre"] = train_df["PatientID"].apply(_centre)
    return train_df, test_df, ext_df, ct_feat_list, pt_feat_list


# ============================================================
# RANKER IMPLEMENTATIONS
# ============================================================
def _safe_auc(y_true, scores):
    """Directional AUC: max(auc, 1-auc) or 0.5 on failure."""
    try:
        if len(np.unique(y_true)) < 2:
            return 0.5
        auc = float(roc_auc_score(y_true, scores))
        return max(auc, 1.0 - auc)
    except Exception:
        return 0.5


def _loco_rank(X_train: np.ndarray, y_train: np.ndarray,
               centre_ids: np.ndarray, feat_names: list[str],
               mode: str) -> tuple[list[str], np.ndarray]:
    """
    LOCO ranking for binary classification.
    Modes:
      loco_evt    : HPV− ≥ LOCO_HPV_MIN hard cut, unweighted, no shrinkage
      loco_epv_cut: HPV− ≥ LOCO_HPV_MIN AND HPV−×HPV+ ≥ LOCO_ENON_MIN, unweighted
      w_enon      : no cut, w=HPV−×HPV+, shrinkage κ=LOCO_KAPPA
    """
    centres    = np.unique(centre_ids)
    n_feats    = X_train.shape[1]
    auc_matrix = np.full((len(centres), n_feats), np.nan)
    c_weights  = np.zeros(len(centres), dtype=float)

    for ci, held in enumerate(centres):
        mask    = centre_ids == held
        n_c     = int(mask.sum())
        if n_c < 2:
            continue
        y_val   = y_train[mask]
        hpv_neg = int((y_val == 0).sum())
        hpv_pos = int((y_val == 1).sum())
        enon    = float(hpv_neg * hpv_pos)

        if mode == "loco_evt":
            include       = hpv_neg >= LOCO_HPV_MIN
            w_c           = 1.0
            use_shrinkage = False
        elif mode == "loco_epv_cut":
            include       = (hpv_neg >= LOCO_HPV_MIN) and (enon >= LOCO_ENON_MIN)
            w_c           = 1.0
            use_shrinkage = False
        elif mode == "w_enon":
            include       = True
            w_c           = max(0.0, enon)
            use_shrinkage = True
        else:
            raise ValueError(f"Unknown LOCO mode: {mode}")

        if not include:
            continue

        c_weights[ci] = w_c
        X_val = X_train[mask]
        for fi in range(n_feats):
            auc_c = _safe_auc(y_val, X_val[:, fi])
            if use_shrinkage:
                auc_c = (hpv_neg * auc_c + LOCO_KAPPA * 0.5) / (hpv_neg + LOCO_KAPPA)
            auc_matrix[ci, fi] = auc_c

    valid = c_weights > 0
    if not np.any(valid):
        return list(feat_names), np.full(n_feats, 0.5)

    M = np.where(np.isnan(auc_matrix[valid]), 0.5, auc_matrix[valid])
    W = c_weights[valid]

    if mode == "w_enon":
        total_w = float(W.sum())
        score   = np.sum(M * W[:, None], axis=0) / total_w if total_w > 0 else np.mean(M, axis=0)
    else:
        score = np.mean(M, axis=0)

    order = np.argsort(score)[::-1]
    return [feat_names[i] for i in order], score[order]


RANKER_MODE_MAP = {
    "LOCO_evt":     "loco_evt",
    "LOCO_EPV_CUT": "loco_epv_cut",
    "WLOCO_enon":   "w_enon",
}


def _univar_rank(X: np.ndarray, y: np.ndarray, feat_names: list[str]) -> list[str]:
    """UNIVAR: directional AUC per feature on all-train."""
    scores = [_safe_auc(y, X[:, i]) for i in range(X.shape[1])]
    return [feat_names[i] for i in np.argsort(np.array(scores))[::-1]]


def _gbc_rank(X: np.ndarray, y: np.ndarray, feat_names: list[str]) -> list[str]:
    """GBC: GradientBoostingClassifier feature importances on all-train."""
    try:
        model = GradientBoostingClassifier(
            n_estimators=GBC_N_EST, learning_rate=GBC_LR,
            max_depth=GBC_DEPTH, subsample=GBC_SUBSAMP,
            random_state=SEED,
        )
        model.fit(X, y)
        imp = model.feature_importances_
    except Exception:
        imp = np.zeros(len(feat_names))
    return [feat_names[i] for i in np.argsort(imp)[::-1]]


# ============================================================
# CLINICAL RANKER — LASSO-logistic (preserved from 17_mar scripts)
# ============================================================
def _clinical_rank(X_clin: np.ndarray, y: np.ndarray,
                   feat_names: list[str]) -> list[str]:
    """Rank clinical features by |LASSO coefficient|. Returns ordered list.
    Harmonised LASSO-logistic clinical ranker (aligned with LOCO script version).
    """
    Xs = StandardScaler().fit_transform(X_clin)
    for C in np.logspace(-2, 1, 60):
        try:
            clf = LogisticRegression(
                penalty="l1", solver="saga", C=C,
                class_weight="balanced", random_state=SEED, max_iter=5000,
            )
            clf.fit(Xs, y)
            coefs = np.abs(clf.coef_[0])
            if coefs.sum() > 0:
                return [feat_names[i] for i in np.argsort(coefs)[::-1]]
        except Exception:
            continue
    return list(feat_names)


# ============================================================
# GM / COACH CONSTRUCTORS
# ============================================================
def _make_model(name: str, params: dict):
    if name == "LR_L2":
        return LogisticRegression(
            penalty="l2", solver="lbfgs", class_weight="balanced",
            C=params.get("C", 1.0), max_iter=2000, random_state=SEED,
        )
    if name == "LR_EN":
        return LogisticRegression(
            penalty="elasticnet", solver="saga", class_weight="balanced",
            C=params.get("C", 1.0), l1_ratio=params.get("l1_ratio", 0.5),
            max_iter=5000, random_state=SEED,
        )
    if name == "SVM_L":
        base = LinearSVC(
            C=params.get("C", 0.01), class_weight="balanced",
            max_iter=5000, random_state=SEED,
        )
        return CalibratedClassifierCV(base, cv=3)
    if name == "RF":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=params.get("max_depth", 4),
            min_samples_leaf=params.get("min_samples_leaf", 2),
            class_weight="balanced", random_state=SEED, n_jobs=-1,
        )
    raise ValueError(f"Unknown model: {name}")


def _suggest_gm_params(trial: optuna.Trial, gm_name: str) -> dict:
    if gm_name == "LR_L2":
        return {"C": trial.suggest_float("C", 1e-4, 10.0, log=True)}
    if gm_name == "LR_EN":
        return {"C": trial.suggest_float("C", 1e-4, 10.0, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0)}
    if gm_name == "SVM_L":
        return {"C": trial.suggest_float("C", 1e-4, 10.0, log=True)}
    if gm_name == "RF":
        return {"max_depth": trial.suggest_int("max_depth", 2, 6),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10)}
    raise ValueError(f"Unknown GM: {gm_name}")


def _coach_params_from_trial(coach_name: str, gm_name: str, trial_params: dict) -> dict:
    if coach_name == "LR_L2":
        return {"C": trial_params.get("C", 1.0)} if gm_name == "LR_L2" else {"C": 1.0}
    if coach_name == "LR_EN":
        if gm_name == "LR_EN":
            return {"C": trial_params.get("C", 1.0),
                    "l1_ratio": trial_params.get("l1_ratio", 0.5)}
        return {"C": 1.0, "l1_ratio": 0.5}
    if coach_name == "SVM_L":
        return {"C": trial_params.get("C", 0.01)} if gm_name == "SVM_L" else {"C": 0.01}
    if coach_name == "RF":
        if gm_name == "RF":
            return {"max_depth": trial_params.get("max_depth", 4),
                    "min_samples_leaf": trial_params.get("min_samples_leaf", 2)}
        return {"max_depth": 4, "min_samples_leaf": 2}
    raise ValueError(f"Unknown coach: {coach_name}")


# ============================================================
# OOF ENGINE (GM objective)
# ============================================================
def _model_oof_auc(
    X_ct: np.ndarray, X_pt: np.ndarray, X_clin: np.ndarray,
    y: np.ndarray,
    ct_idx: list[int], pt_idx: list[int], clin_idx: list[int],
    gm_name: str, gm_params: dict,
) -> tuple[float, float]:
    """
    5-fold stratified OOF AUC for GM Optuna objective.
    Scalers fitted on fold-train only (no leakage).
    Returns (oof_auc, fold_std).
    """
    X_ct_s   = X_ct[:, ct_idx]
    X_pt_s   = X_pt[:, pt_idx]
    X_clin_s = X_clin[:, clin_idx]

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_proba = np.zeros(len(y), dtype=float)
    fold_aucs = []

    for tr, vl in skf.split(X_ct_s, y):
        sc_ct   = StandardScaler().fit(X_ct_s[tr])
        sc_pt   = StandardScaler().fit(X_pt_s[tr])
        sc_clin = StandardScaler().fit(X_clin_s[tr])

        X_tr  = np.hstack([sc_ct.transform(X_ct_s[tr]),
                            sc_pt.transform(X_pt_s[tr]),
                            sc_clin.transform(X_clin_s[tr])])
        X_val = np.hstack([sc_ct.transform(X_ct_s[vl]),
                            sc_pt.transform(X_pt_s[vl]),
                            sc_clin.transform(X_clin_s[vl])])
        try:
            clf = _make_model(gm_name, gm_params)
            clf.fit(X_tr, y[tr])
            oof_proba[vl] = clf.predict_proba(X_val)[:, 1]
            fold_aucs.append(float(roc_auc_score(y[vl], oof_proba[vl])))
        except Exception as e:
            print(f"[WARN] OOF fold failed gm={gm_name}: {type(e).__name__}: {e}")
            oof_proba[vl] = 0.5

    if len(fold_aucs) < max(2, N_FOLDS - 1):
        raise optuna.TrialPruned()

    try:
        oof_auc = float(roc_auc_score(y, oof_proba))
    except Exception:
        oof_auc = 0.5
    fold_std = float(np.std(fold_aucs)) if len(fold_aucs) > 1 else STD_THRESHOLD
    return oof_auc, fold_std


# ============================================================
# YOUDEN CLASSIFICATION METRICS
# ============================================================
def _youden_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """
    Compute classification metrics at the Youden-optimal threshold.

    Youden threshold = argmax(TPR - FPR) = argmax(Sensitivity + Specificity - 1).
    This is the standard clinical operating point for imbalanced data.
    Using 0.5 is inappropriate: HPV+ prevalence ~70-74% and Platt-scaled
    probabilities shift all outputs upward, making 0.5 a near-all-positive rule.

    Returns dict with keys:
      youden_thresh, ba, sen, spe, f1, prec, npv, acc
    All NaN-safe (returns NaN dict on failure).
    """
    nan_result = {k: np.nan for k in
                  ("youden_thresh", "ba", "sen", "spe", "f1", "prec", "npv", "acc")}
    try:
        if len(np.unique(y_true)) < 2:
            return nan_result
        fpr, tpr, thresholds = roc_curve(y_true, proba)
        j_idx  = int(np.argmax(tpr - fpr))
        thresh = float(thresholds[j_idx])
        y_pred = (proba >= thresh).astype(int)

        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))

        sen  = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        spe  = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        npv  = tn / (tn + fn) if (tn + fn) > 0 else np.nan
        ba   = (sen + spe) / 2.0 if not (np.isnan(sen) or np.isnan(spe)) else np.nan
        f1   = float(f1_score(y_true, y_pred, zero_division=0))
        acc  = float(accuracy_score(y_true, y_pred))

        return {"youden_thresh": thresh, "ba": ba, "sen": sen, "spe": spe,
                "f1": f1, "prec": prec, "npv": npv, "acc": acc}
    except Exception:
        return nan_result


# ============================================================
# BOOTSTRAP AUC
# ============================================================
def _bootstrap_auc(y: np.ndarray, proba: np.ndarray,
                   n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    rng  = np.random.default_rng(seed)
    aucs = []
    n    = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            aucs.append(float(roc_auc_score(y[idx], proba[idx])))
        except Exception:
            continue
    if not aucs:
        return {"mean": np.nan, "lo95": np.nan, "hi95": np.nan, "n": 0}
    a = np.array(aucs)
    return {"mean": float(a.mean()),
            "lo95": float(np.percentile(a, 2.5)),
            "hi95": float(np.percentile(a, 97.5)),
            "n":    len(a)}


# ============================================================
# COACH EVALUATION (Test + Ext simultaneously)
# ============================================================
def _coach_eval(
    X_ct_tr, X_pt_tr, X_clin_tr, y_tr,
    X_ct_te, X_pt_te, X_clin_te, y_te,
    X_ct_ext, X_pt_ext, X_clin_ext, y_ext,
    ct_idx: list[int], pt_idx: list[int], clin_idx: list[int],
    coach_name: str, coach_params: dict,
) -> tuple:
    """
    Fit coach on full X_train. Evaluate Test AND Ext in same pass.
    Scalers fitted on X_train only, applied to both Test and Ext (no leakage).
    Returns (auc_test, auc_ext, boot_ext_dict,
             proba_te, proba_ext,
             metrics_test_dict, metrics_ext_dict).
    """
    sc_ct   = StandardScaler().fit(X_ct_tr[:, ct_idx])
    sc_pt   = StandardScaler().fit(X_pt_tr[:, pt_idx])
    sc_clin = StandardScaler().fit(X_clin_tr[:, clin_idx])

    def _tf(Xct, Xpt, Xcl):
        return np.hstack([sc_ct.transform(Xct[:, ct_idx]),
                          sc_pt.transform(Xpt[:, pt_idx]),
                          sc_clin.transform(Xcl[:, clin_idx])])

    nan_m = {k: np.nan for k in
             ("youden_thresh", "ba", "sen", "spe", "f1", "prec", "npv", "acc")}
    try:
        clf = _make_model(coach_name, coach_params)
        clf.fit(_tf(X_ct_tr, X_pt_tr, X_clin_tr), y_tr)
        proba_te  = clf.predict_proba(_tf(X_ct_te,  X_pt_te,  X_clin_te))[:, 1]
        proba_ext = clf.predict_proba(_tf(X_ct_ext, X_pt_ext, X_clin_ext))[:, 1]
    except Exception as e:
        print(f"[WARN] Coach failed coach={coach_name} params={coach_params}: "
              f"{type(e).__name__}: {e}")
        return (np.nan, np.nan,
                {"mean": np.nan, "lo95": np.nan, "hi95": np.nan, "n": 0},
                np.full(len(y_te), np.nan), np.full(len(y_ext), np.nan),
                nan_m, nan_m)

    auc_test = float(roc_auc_score(y_te,  proba_te))  if len(np.unique(y_te))  > 1 else np.nan
    auc_ext  = float(roc_auc_score(y_ext, proba_ext)) if len(np.unique(y_ext)) > 1 else np.nan
    boot_ext = _bootstrap_auc(y_ext, proba_ext)

    m_test = _youden_metrics(y_te,  proba_te)
    m_ext  = _youden_metrics(y_ext, proba_ext)

    return auc_test, auc_ext, boot_ext, proba_te, proba_ext, m_test, m_ext


# ============================================================
# BOOLEAN FILTER FLAGS
# ============================================================
def _compute_flags(oof_auc: float, auc_test: float, auc_ext: float,
                   ba_test: float, ba_ext: float,
                   boot_ext_lo95: float) -> dict:
    """
    Tiered boolean filter flags for Excel sorting of 160k rows.
    Each flag depends on the prior (nested filter funnel).

    trio_ok:       All 3 AUC ≥ 0.70 — base credible generalisation floor
    balanced_ok:   trio_ok + BA_ext ≥ 0.70 + BA_test ≥ 0.65
    ci_ok:         balanced_ok + boot_ext_lo95 ≥ 0.57 (captures LOCO lo95=0.592 and INDIV lo95=0.636)
    top_candidate: ci_ok + (BA_ext ≥ 0.75 OR lo95 ≥ 0.63)
    """
    def _ok(v):
        return not (v is None or (isinstance(v, float) and np.isnan(v)))

    trio_ok = (
        _ok(oof_auc)   and oof_auc   >= TRIO_OK_OOF_MIN and
        _ok(auc_test)  and auc_test  >= TRIO_OK_TEST_MIN and
        _ok(auc_ext)   and auc_ext   >= TRIO_OK_EXT_MIN
    )
    balanced_ok = (
        trio_ok and
        _ok(ba_ext)  and ba_ext  >= BALANCED_OK_BA_EXT_MIN and
        _ok(ba_test) and ba_test >= BALANCED_OK_BA_TEST_MIN
    )
    ci_ok = (
        balanced_ok and
        _ok(boot_ext_lo95) and boot_ext_lo95 >= CI_OK_LO95_MIN
    )
    top_candidate = (
        ci_ok and (
            (_ok(ba_ext)        and ba_ext        >= TOP_CANDIDATE_BA_EXT_MIN) or
            (_ok(boot_ext_lo95) and boot_ext_lo95 >= TOP_CANDIDATE_LO95_MIN)
        )
    )
    return {
        "trio_ok":       int(trio_ok),
        "balanced_ok":   int(balanced_ok),
        "ci_ok":         int(ci_ok),
        "top_candidate": int(top_candidate),
    }


# ============================================================
# STRUCTURAL SEEDS (mirroring 17_mar scripts)
# ============================================================
def _enqueue_structural_seeds(study: optuna.Study, gm_name: str) -> None:
    """Pre-enqueue structurally-varied compositions before TPE warmup.
    All combos satisfy TOTAL_N_MIN=7, MIN_PT=4, MIN_CT=2, MIN_CLIN=1
    so none will be pruned immediately by the objective constraints.
    """
    base_combos = [
        # n_total, n_clin, n_pt  (n_ct = n_total - n_clin - n_pt, must be >= MIN_CT=2)
        {"n_total": 7,  "n_clin": 1, "n_pt": 4},   # minimal: 1clin+4pt+2ct
        {"n_total": 9,  "n_clin": 1, "n_pt": 4},   # small: 1clin+4pt+4ct
        {"n_total": 10, "n_clin": 1, "n_pt": 5},   # balanced: 1clin+5pt+4ct
        {"n_total": 11, "n_clin": 2, "n_pt": 4},   # 2clin+4pt+5ct
        {"n_total": 13, "n_clin": 2, "n_pt": 5},   # medium: 2clin+5pt+6ct
        {"n_total": 15, "n_clin": 2, "n_pt": 6},   # larger: 2clin+6pt+7ct
    ]
    if gm_name in ("LR_L2", "SVM_L"):
        hparam_sets = [{"C": v} for v in [1e-3, 1e-2, 1e-1, 1.0]]
    elif gm_name == "LR_EN":
        hparam_sets = [{"C": c, "l1_ratio": l1}
                       for c, l1 in [(1e-2, 0.3), (1e-1, 0.5), (1.0, 0.7)]]
    elif gm_name == "RF":
        hparam_sets = [{"max_depth": md, "min_samples_leaf": msl}
                       for md, msl in [(3, 2), (4, 2), (5, 3)]]
    else:
        return
    for combo in base_combos:
        for hp in hparam_sets:
            try:
                study.enqueue_trial({**combo, **hp})
            except Exception:
                pass


# ============================================================
# BASELINE ROWS (cheap single-modality LR_L2 reference)
# ============================================================
def _baseline_rows(
    ranker: str,
    ct_ranked, pt_ranked, clin_ranked,
    ct_feat_list, pt_feat_list,
    X_ct_tr, X_ct_te, X_ct_ext,
    X_pt_tr, X_pt_te, X_pt_ext,
    X_clin_tr, X_clin_te, X_clin_ext,
    y_tr, y_te, y_ext,
    start_no: int = 0,
) -> list[dict]:
    rows = []
    configs = [
        ("CT_only",   ct_ranked[:12],  ct_feat_list,       X_ct_tr,   X_ct_te,   X_ct_ext,   12, 0, 0),
        ("PT_only",   pt_ranked[:6],   pt_feat_list,       X_pt_tr,   X_pt_te,   X_pt_ext,   0,  6, 0),
        ("Clin_only", clin_ranked[:2], CLINICAL_FEATURES,  X_clin_tr, X_clin_te, X_clin_ext, 0,  0, 2),
    ]
    for bno, (block, feats, feat_list, Xtr, Xte, Xext, nct, npt, ncl) in enumerate(configs):
        n_f = len(feats)
        idx = [feat_list.index(f) for f in feats]
        sc  = StandardScaler().fit(Xtr[:, idx])
        clf = LogisticRegression(
            penalty="l2", solver="lbfgs", class_weight="balanced",
            C=1.0, max_iter=2000, random_state=SEED,
        )
        clf.fit(sc.transform(Xtr[:, idx]), y_tr)
        proba_te  = clf.predict_proba(sc.transform(Xte[:, idx]))[:, 1]
        proba_ext = clf.predict_proba(sc.transform(Xext[:, idx]))[:, 1]
        auc_test  = float(roc_auc_score(y_te,  proba_te))  if len(np.unique(y_te))  > 1 else np.nan
        auc_ext   = float(roc_auc_score(y_ext, proba_ext)) if len(np.unique(y_ext)) > 1 else np.nan
        boot      = _bootstrap_auc(y_ext, proba_ext)
        m_te  = _youden_metrics(y_te,  proba_te)
        m_ext = _youden_metrics(y_ext, proba_ext)
        flags = _compute_flags(np.nan, auc_test, auc_ext,
                               m_te["ba"], m_ext["ba"], boot["lo95"])
        rows.append({
            "trial_no":    start_no + bno + 1,
            "ranker":      ranker,
            "gm":          "baseline",
            "coach":       "LR_L2",
            "block":       block,
            "n_ct":        nct, "n_pt": npt, "n_clin": ncl,
            "n_total":     n_f,
            "ct_features":   "|".join(feats) if block == "CT_only"   else "",
            "pt_features":   "|".join(feats) if block == "PT_only"   else "",
            "clin_features": "|".join(feats) if block == "Clin_only" else "",
            "gm_params":   str({"C": 1.0}),
            "coach_params":str({"C": 1.0}),
            "oof_auc":     np.nan, "fold_std": np.nan,
            "auc_test":    auc_test, "auc_ext": auc_ext,
            "boot_ext_mean": boot["mean"],
            "boot_ext_lo95": boot["lo95"],
            "boot_ext_hi95": boot["hi95"],
            "pass_test":   int(not np.isnan(auc_test)  and auc_test  >= PASS_THRESHOLD),
            "pass_ext":    int(not np.isnan(auc_ext)   and auc_ext   >= PASS_THRESHOLD),
            "youden_thresh_test": m_te["youden_thresh"],
            "ba_test":  m_te["ba"],  "sen_test":  m_te["sen"],  "spe_test":  m_te["spe"],
            "f1_test":  m_te["f1"],  "prec_test": m_te["prec"], "npv_test":  m_te["npv"],
            "acc_test": m_te["acc"],
            "youden_thresh_ext":  m_ext["youden_thresh"],
            "ba_ext":   m_ext["ba"],  "sen_ext":   m_ext["sen"],  "spe_ext":   m_ext["spe"],
            "f1_ext":   m_ext["f1"],  "prec_ext":  m_ext["prec"], "npv_ext":   m_ext["npv"],
            "acc_ext":  m_ext["acc"],
            **flags,
        })
    return rows


# ============================================================
# MAIN
# ============================================================
def main():
    global _log_fh
    # Create output dir and open log here (not at import time)
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    _log_fh = open(LOG_FILE, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, _log_fh)
    sys.stderr = _Tee(sys.__stderr__, _log_fh)

    t0 = time.time()
    print("=" * 80)
    print("27_mar_T2_dev_ultimate_recheck.py  —  Task 2 HPV Classification Recheck")
    print(f"Rankers: {RANKER_NAMES}")
    print(f"GMs:     {GM_NAMES}")
    print(f"Coaches: {COACH_NAMES}")
    print(f"N_TRIALS={N_TRIALS} | N_BOOT={N_BOOT} | SEED={SEED}")
    print(f"Composition: TOTAL=[{TOTAL_N_MIN},{TOTAL_N_MAX}] | "
          f"MIN_CLIN={MIN_CLIN} MIN_PT={MIN_PT} MIN_CT={MIN_CT}")
    expected_main     = len(RANKER_NAMES) * len(GM_NAMES) * N_TRIALS * len(COACH_NAMES)
    expected_baseline = len(RANKER_NAMES) * 3
    print(f"Expected main rows:     "
          f"{len(RANKER_NAMES)}x{len(GM_NAMES)}x{N_TRIALS}x{len(COACH_NAMES)} = {expected_main:,}")
    print(f"Expected baseline rows: {len(RANKER_NAMES)}x3 = {expected_baseline:,}")
    print(f"Expected total rows:    {expected_main + expected_baseline:,}")
    print("Winners to confirm: INDIV T31763 (GBC/LR_L2xSVM_L, ext=0.814, lo95=0.636) "
          "| LOCO T77583 (WLOCO_enon/SVM_LxSVM_L, ext=0.793, lo95=0.592)")
    print("=" * 80)

    # ----------------------------------------------------------
    # Load data
    # ----------------------------------------------------------
    print("\n--- Loading data ---")
    train_df, test_df, ext_df, ct_feat_list, pt_feat_list = _load_data()
    print(f"Train={len(train_df)} | Test={len(test_df)} | Ext={len(ext_df)}")

    # Preflight: CT/PT inner-join must not silently drop patients
    # Expected counts from data preparation (adjust if cohort changes)
    EXPECTED_TRAIN, EXPECTED_TEST, EXPECTED_EXT = 67, 20, 27
    for split_name, df, exp in [("train", train_df, EXPECTED_TRAIN),
                                 ("test",  test_df,  EXPECTED_TEST),
                                 ("ext",   ext_df,   EXPECTED_EXT)]:
        if len(df) != exp:
            raise ValueError(
                f"CT/PT merge dropped patients in {split_name}: "
                f"got {len(df)}, expected {exp}. Check PatientID alignment."
            )

    ct_feat_list = [f for f in ct_feat_list if f not in EXCLUDE_COLS]
    pt_feat_list = [f for f in pt_feat_list if f not in EXCLUDE_COLS]
    dup = set(ct_feat_list) & set(pt_feat_list)
    if dup:
        raise ValueError(f"CT/PT feature overlap: {sorted(dup)}")
    for split_name, df in [("train", train_df), ("test", test_df), ("ext", ext_df)]:
        miss_ct = [f for f in ct_feat_list if f not in df.columns]
        miss_pt = [f for f in pt_feat_list if f not in df.columns]
        if miss_ct or miss_pt:
            raise ValueError(
                f"Missing features in {split_name}. CT:{miss_ct[:5]} PT:{miss_pt[:5]}"
            )
    print(f"  CT={len(ct_feat_list)} features | PT={len(pt_feat_list)} features")

    y_train = train_df[LABEL_COL].values.astype(int)
    y_test  = test_df[LABEL_COL].values.astype(int)
    y_ext   = ext_df[LABEL_COL].values.astype(int)
    print(f"Train: HPV-={(y_train==0).sum()}, HPV+={(y_train==1).sum()}")
    print(f"Test:  HPV-={(y_test==0).sum()},  HPV+={(y_test==1).sum()}")
    print(f"Ext:   HPV-={(y_ext==0).sum()},   HPV+={(y_ext==1).sum()}")

    X_ct_train   = train_df[ct_feat_list].values.astype(float)
    X_ct_test    = test_df[ct_feat_list].values.astype(float)
    X_ct_ext     = ext_df[ct_feat_list].values.astype(float)
    X_pt_train   = train_df[pt_feat_list].values.astype(float)
    X_pt_test    = test_df[pt_feat_list].values.astype(float)
    X_pt_ext     = ext_df[pt_feat_list].values.astype(float)
    X_clin_train = train_df[CLINICAL_FEATURES].values.astype(float)
    X_clin_test  = test_df[CLINICAL_FEATURES].values.astype(float)
    X_clin_ext   = ext_df[CLINICAL_FEATURES].values.astype(float)
    centre_ids   = train_df["centre"].values

    print("\nTrain centre breakdown:")
    for c, g in train_df.groupby("centre"):
        neg  = int((g[LABEL_COL] == 0).sum())
        pos  = int((g[LABEL_COL] == 1).sum())
        enon = neg * pos
        print(f"  {c}: n={len(g)}, HPV-={neg}, HPV+={pos}, enon={enon}")

    # ----------------------------------------------------------
    # Clinical ranking (once, all-train LASSO-logistic)
    # ----------------------------------------------------------
    print("\n--- Clinical ranking (LASSO-logistic, all-train) ---")
    clin_ranked = _clinical_rank(X_clin_train, y_train, CLINICAL_FEATURES)
    print(f"  Rank: {clin_ranked}")

    # ----------------------------------------------------------
    # Pre-compute all ranker orderings (once each)
    # ----------------------------------------------------------
    print("\n--- Pre-computing all rankers ---")
    sc_ct_g = StandardScaler().fit(X_ct_train)
    sc_pt_g = StandardScaler().fit(X_pt_train)
    X_ct_sc = sc_ct_g.transform(X_ct_train)
    X_pt_sc = sc_pt_g.transform(X_pt_train)

    # Combined scaled array for LOCO (needs centre structure)
    X_rad_sc  = np.hstack([X_ct_sc, X_pt_sc])
    rad_names = ct_feat_list + pt_feat_list
    ct_set    = set(ct_feat_list)

    rankings: dict[str, dict] = {}

    # LOCO rankers
    for rk in ["LOCO_evt", "LOCO_EPV_CUT", "WLOCO_enon"]:
        mode = RANKER_MODE_MAP[rk]
        ranked_rad, _ = _loco_rank(X_rad_sc, y_train, centre_ids, rad_names, mode)
        ct_ranked_rk  = [f for f in ranked_rad if f in ct_set]
        pt_ranked_rk  = [f for f in ranked_rad if f not in ct_set]
        rankings[rk]  = {"ct": ct_ranked_rk, "pt": pt_ranked_rk, "clin": clin_ranked}
        valid_centres = []
        for c_name in np.unique(centre_ids):
            mask = centre_ids == c_name
            neg  = int((y_train[mask] == 0).sum())
            pos  = int((y_train[mask] == 1).sum())
            enon = neg * pos
            if mode == "loco_evt":
                active = neg >= LOCO_HPV_MIN
            elif mode == "loco_epv_cut":
                active = (neg >= LOCO_HPV_MIN) and (enon >= LOCO_ENON_MIN)
            elif mode == "w_enon":
                active = enon > 0
            else:
                active = True
            if active:
                valid_centres.append(f"{c_name}(neg={neg},enon={enon})")
        print(f"  {rk}: ct={len(ct_ranked_rk)}, pt={len(pt_ranked_rk)} | "
              f"voting: {valid_centres}")

    # INDIV rankers
    print("  Computing GBC (applied to CT and PT pools separately, ~10s)...")
    ct_gbc = _gbc_rank(X_ct_sc, y_train, ct_feat_list)
    pt_gbc = _gbc_rank(X_pt_sc, y_train, pt_feat_list)
    rankings["GBC"] = {"ct": ct_gbc, "pt": pt_gbc, "clin": clin_ranked}
    print(f"  GBC: ct={len(ct_gbc)}, pt={len(pt_gbc)}")

    ct_univar = _univar_rank(X_ct_sc, y_train, ct_feat_list)
    pt_univar = _univar_rank(X_pt_sc, y_train, pt_feat_list)
    rankings["UNIVAR"] = {"ct": ct_univar, "pt": pt_univar, "clin": clin_ranked}
    print(f"  UNIVAR: ct={len(ct_univar)}, pt={len(pt_univar)}")

    # ----------------------------------------------------------
    # Checkpoint resume
    # ----------------------------------------------------------
    all_rows: list[dict]          = []
    done_blocks: set[tuple]       = set()
    row_counter: int              = 0
    exp_block_rows                = N_TRIALS * len(COACH_NAMES)

    if CKPT_CSV.exists():
        print(f"\nCheckpoint found: {CKPT_CSV} — loading...")
        ckpt_df  = pd.read_csv(CKPT_CSV)
        all_rows = ckpt_df.to_dict("records")
        if "trial_no" in ckpt_df.columns and len(ckpt_df):
            row_counter = int(
                pd.to_numeric(ckpt_df["trial_no"], errors="coerce").max(skipna=True)
            )
        main_ckpt = (ckpt_df[ckpt_df["block"] == "CT_PT_Clin"]
                     if "block" in ckpt_df.columns else pd.DataFrame())
        if len(main_ckpt):
            for (rk, gm), g in main_ckpt.groupby(["ranker", "gm"]):
                if len(g) >= int(exp_block_rows * 0.9):
                    done_blocks.add((rk, gm))
        print(f"  Loaded {len(all_rows):,} rows | row_counter={row_counter} | "
              f"done_blocks={len(done_blocks)}/{len(RANKER_NAMES)*len(GM_NAMES)}")
    else:
        print("\nNo checkpoint found — starting fresh.")

    total_blocks = len(RANKER_NAMES) * len(GM_NAMES)
    block_no     = 0

    # ----------------------------------------------------------
    # Main loop: 5 rankers × 4 GMs = 20 blocks
    # ----------------------------------------------------------
    for ranker in RANKER_NAMES:
        ct_ranked_r   = rankings[ranker]["ct"]
        pt_ranked_r   = rankings[ranker]["pt"]
        clin_ranked_r = rankings[ranker]["clin"]

        ct_rank_idx   = [ct_feat_list.index(f)   for f in ct_ranked_r]
        pt_rank_idx   = [pt_feat_list.index(f)   for f in pt_ranked_r]
        clin_rank_idx = [CLINICAL_FEATURES.index(f) for f in clin_ranked_r]

        for gm in GM_NAMES:
            block_no += 1
            if (ranker, gm) in done_blocks:
                print(f"\n[{block_no}/{total_blocks}] Ranker={ranker} | GM={gm} — "
                      f"SKIPPED (checkpoint complete)")
                continue

            block_start = time.time()
            print(f"\n[{block_no}/{total_blocks}] Ranker={ranker} | GM={gm} | "
                  f"Trials={N_TRIALS}")

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=SEED),
            )
            _enqueue_structural_seeds(study, gm)

            def objective(
                trial,
                _ct_ri=ct_rank_idx, _pt_ri=pt_rank_idx, _cl_ri=clin_rank_idx,
                _ct_r=ct_ranked_r,  _pt_r=pt_ranked_r,  _cl_r=clin_ranked_r,
                _gm=gm,
            ):
                n_total = trial.suggest_int(
                    "n_total", TOTAL_N_MIN,
                    min(TOTAL_N_MAX, len(_ct_ri) + len(_pt_ri) + len(_cl_ri)),
                )
                max_clin = min(len(_cl_ri), n_total - MIN_PT - MIN_CT)
                if max_clin < MIN_CLIN:
                    raise optuna.TrialPruned()
                n_clin = trial.suggest_int("n_clin", MIN_CLIN, max_clin)

                n_rad    = n_total - n_clin
                n_pt_max = min(len(_pt_ri), n_rad - MIN_CT)
                if n_pt_max < MIN_PT:
                    raise optuna.TrialPruned()
                n_pt = trial.suggest_int("n_pt", MIN_PT, n_pt_max)

                n_ct = n_rad - n_pt
                if n_ct < MIN_CT or n_ct > len(_ct_ri):
                    raise optuna.TrialPruned()

                gm_params = _suggest_gm_params(trial, _gm)
                oof_auc, fold_std = _model_oof_auc(
                    X_ct_train, X_pt_train, X_clin_train, y_train,
                    _ct_ri[:n_ct], _pt_ri[:n_pt], _cl_ri[:n_clin],
                    _gm, gm_params,
                )
                score = W_PERF * oof_auc + W_STAB * max(0.0, 1.0 - fold_std / STD_THRESHOLD)

                trial.set_user_attr("n_ct",       n_ct)
                trial.set_user_attr("n_pt",       n_pt)
                trial.set_user_attr("n_clin",     n_clin)
                trial.set_user_attr("ct_feats",   "|".join(_ct_r[:n_ct]))
                trial.set_user_attr("pt_feats",   "|".join(_pt_r[:n_pt]))
                trial.set_user_attr("clin_feats", "|".join(_cl_r[:n_clin]))
                trial.set_user_attr("gm_params",  str(gm_params))
                trial.set_user_attr("oof_auc",    float(oof_auc))
                trial.set_user_attr("fold_std",   float(fold_std))

                if (trial.number + 1) % 150 == 0:
                    print(f"  Trial {trial.number+1}/{N_TRIALS} | "
                          f"OOF={oof_auc:.4f} | score={score:.4f}")
                return score

            study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

            completed = [t for t in study.trials
                         if t.state == optuna.trial.TrialState.COMPLETE]
            print(f"  Evaluating {len(completed)} trials x {len(COACH_NAMES)} coaches "
                  f"(Test + Ext simultaneously)...")

            block_rows: list[dict] = []
            for t in completed:
                n_ct       = int(t.user_attrs["n_ct"])
                n_pt       = int(t.user_attrs["n_pt"])
                n_clin     = int(t.user_attrs["n_clin"])
                ct_feats   = t.user_attrs["ct_feats"].split("|")
                pt_feats   = t.user_attrs["pt_feats"].split("|")
                clin_feats = t.user_attrs["clin_feats"].split("|")
                oof_auc    = float(t.user_attrs["oof_auc"])
                fold_std   = float(t.user_attrs["fold_std"])
                gm_params_str = t.user_attrs["gm_params"]

                ct_idx_c   = [ct_feat_list.index(f)   for f in ct_feats]
                pt_idx_c   = [pt_feat_list.index(f)   for f in pt_feats]
                clin_idx_c = [CLINICAL_FEATURES.index(f) for f in clin_feats]

                for coach in COACH_NAMES:
                    coach_params = _coach_params_from_trial(coach, gm, t.params)

                    (auc_test, auc_ext, boot_ext,
                     proba_te, proba_ext,
                     m_test, m_ext) = _coach_eval(
                        X_ct_train, X_pt_train, X_clin_train, y_train,
                        X_ct_test,  X_pt_test,  X_clin_test,  y_test,
                        X_ct_ext,   X_pt_ext,   X_clin_ext,   y_ext,
                        ct_idx_c, pt_idx_c, clin_idx_c,
                        coach, coach_params,
                    )

                    flags = _compute_flags(
                        oof_auc, auc_test, auc_ext,
                        m_test["ba"], m_ext["ba"], boot_ext["lo95"],
                    )

                    row_counter += 1
                    row: dict = {
                        "trial_no":    row_counter,
                        "ranker":      ranker,
                        "gm":          gm,
                        "coach":       coach,
                        "block":       "CT_PT_Clin",
                        "n_ct":        n_ct,
                        "n_pt":        n_pt,
                        "n_clin":      n_clin,
                        "n_total":     n_ct + n_pt + n_clin,
                        "ct_features":   "|".join(ct_feats),
                        "pt_features":   "|".join(pt_feats),
                        "clin_features": "|".join(clin_feats),
                        "gm_params":     gm_params_str,
                        "coach_params":  str(coach_params),
                        "oof_auc":     oof_auc,
                        "fold_std":    fold_std,
                        "auc_test":    auc_test,
                        "auc_ext":     auc_ext,
                        "boot_ext_mean": boot_ext["mean"],
                        "boot_ext_lo95": boot_ext["lo95"],
                        "boot_ext_hi95": boot_ext["hi95"],
                        "pass_test":   int(not np.isnan(auc_test) and auc_test >= PASS_THRESHOLD),
                        "pass_ext":    int(not np.isnan(auc_ext)  and auc_ext  >= PASS_THRESHOLD),
                        # 16 classification metrics (8 per split, at Youden threshold)
                        "youden_thresh_test": m_test["youden_thresh"],
                        "ba_test":  m_test["ba"],  "sen_test":  m_test["sen"],
                        "spe_test": m_test["spe"], "f1_test":   m_test["f1"],
                        "prec_test":m_test["prec"],"npv_test":  m_test["npv"],
                        "acc_test": m_test["acc"],
                        "youden_thresh_ext":  m_ext["youden_thresh"],
                        "ba_ext":   m_ext["ba"],   "sen_ext":   m_ext["sen"],
                        "spe_ext":  m_ext["spe"],  "f1_ext":    m_ext["f1"],
                        "prec_ext": m_ext["prec"], "npv_ext":   m_ext["npv"],
                        "acc_ext":  m_ext["acc"],
                        # 4 tiered boolean filter flags
                        **flags,
                    }
                    block_rows.append(row)

            all_rows.extend(block_rows)

            # Block summary
            bdf      = pd.DataFrame(block_rows)
            n_trio   = int(bdf["trio_ok"].sum())       if len(bdf) else 0
            n_top    = int(bdf["top_candidate"].sum()) if len(bdf) else 0
            best_ext = float(bdf["auc_ext"].max())     if len(bdf) else np.nan
            elapsed  = time.time() - block_start
            print(f"  Block: {len(block_rows):,} rows | trio_ok={n_trio} | "
                  f"top_candidate={n_top} | best_ext={best_ext:.4f} | "
                  f"elapsed={elapsed/60:.1f} min")

            pd.DataFrame(all_rows).to_csv(CKPT_CSV, index=False)
            print(f"  Checkpoint saved ({len(all_rows):,} total rows)")

    # ----------------------------------------------------------
    # Incremental baselines
    # ----------------------------------------------------------
    print("\n--- Incremental baselines ---")
    for rk in RANKER_NAMES:
        bl_rows = _baseline_rows(
            rk,
            rankings[rk]["ct"], rankings[rk]["pt"], rankings[rk]["clin"],
            ct_feat_list, pt_feat_list,
            X_ct_train, X_ct_test, X_ct_ext,
            X_pt_train, X_pt_test, X_pt_ext,
            X_clin_train, X_clin_test, X_clin_ext,
            y_train, y_test, y_ext,
            start_no=row_counter,
        )
        row_counter += len(bl_rows)
        all_rows.extend(bl_rows)
        for r in bl_rows:
            print(f"  {rk} {r['block']}: test={r['auc_test']:.4f} ext={r['auc_ext']:.4f}")

    # ----------------------------------------------------------
    # Save final outputs
    # ----------------------------------------------------------
    print("\n--- Saving final outputs ---")
    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(ALL_CSV, index=False)
    print(f"  All results:  {ALL_CSV} ({len(all_df):,} rows)")

    # Top results: trio_ok rows sorted by tiered flags then boot_ext_lo95
    main_df  = all_df[all_df["block"] == "CT_PT_Clin"].copy()
    trio_df  = main_df[main_df["trio_ok"] == 1].copy()
    if len(trio_df) > 0:
        top_df = trio_df.sort_values(
            ["top_candidate", "ci_ok", "boot_ext_lo95", "ba_ext", "oof_auc"],
            ascending=[False, False, False, False, False],
        )
    else:
        main_df["_sort"] = main_df["auc_ext"] + main_df["oof_auc"].fillna(0)
        top_df = main_df.sort_values("_sort", ascending=False).head(100).drop(columns=["_sort"])
    top_df.to_csv(TOP_CSV, index=False)
    print(f"  Top results:  {TOP_CSV} "
          f"(trio_ok={len(trio_df):,} rows | top_candidate={int(trio_df['top_candidate'].sum()) if len(trio_df) else 0})")

    # All-pass predictions: save per-patient probabilities for all trio_ok rows
    print(f"\n--- Saving all-pass predictions (trio_ok rows) ---")
    pred_rows: list[dict] = []
    trio_ok_main = main_df[main_df["trio_ok"] == 1].copy()

    for _, row in trio_ok_main.iterrows():
        ct_feats   = row["ct_features"].split("|") if row["ct_features"] else []
        pt_feats   = row["pt_features"].split("|") if row["pt_features"] else []
        clin_feats = row["clin_features"].split("|") if row["clin_features"] else []
        try:
            import ast
            coach_params = ast.literal_eval(row["coach_params"])
        except Exception:
            coach_params = {}

        ct_idx_r   = [ct_feat_list.index(f)   for f in ct_feats   if f in ct_feat_list]
        pt_idx_r   = [pt_feat_list.index(f)   for f in pt_feats   if f in pt_feat_list]
        clin_idx_r = [CLINICAL_FEATURES.index(f) for f in clin_feats if f in CLINICAL_FEATURES]

        if not ct_idx_r and not pt_idx_r and not clin_idx_r:
            continue

        try:
            (_, _, _,
             proba_te, proba_ext, _, _) = _coach_eval(
                X_ct_train, X_pt_train, X_clin_train, y_train,
                X_ct_test,  X_pt_test,  X_clin_test,  y_test,
                X_ct_ext,   X_pt_ext,   X_clin_ext,   y_ext,
                ct_idx_r, pt_idx_r, clin_idx_r,
                row["coach"], coach_params,
            )
            trial_id = int(row["trial_no"])
            for i, pt_row in enumerate(test_df.itertuples()):
                pred_rows.append({
                    "trial_no": trial_id, "ranker": row["ranker"],
                    "gm": row["gm"], "coach": row["coach"],
                    "PatientID": pt_row.PatientID, "split": "test",
                    "y_true": int(y_test[i]),
                    "proba_HPVpos": float(proba_te[i]),
                })
            for i, pt_row in enumerate(ext_df.itertuples()):
                pred_rows.append({
                    "trial_no": trial_id, "ranker": row["ranker"],
                    "gm": row["gm"], "coach": row["coach"],
                    "PatientID": pt_row.PatientID, "split": "ext",
                    "y_true": int(y_ext[i]),
                    "proba_HPVpos": float(proba_ext[i]),
                })
        except Exception as e:
            print(f"[WARN] Prediction refit failed trial_no={row['trial_no']}: "
                  f"{type(e).__name__}: {e}")

    if pred_rows:
        pd.DataFrame(pred_rows).to_csv(PRED_CSV, index=False)
        n_trio_trials = len(trio_ok_main)
        print(f"  Predictions: {PRED_CSV} "
              f"({len(pred_rows):,} rows | {n_trio_trials} trio_ok models x {len(test_df)+len(ext_df)} patients)")
    else:
        print("  No trio_ok predictions to save.")

    # ----------------------------------------------------------
    # Summary report
    # ----------------------------------------------------------
    print("\n" + "=" * 80)
    print("RECHECK SUMMARY")
    print("=" * 80)
    top10 = top_df.head(10)
    for _, r in top10.iterrows():
        print(f"  trial={int(r['trial_no'])} | {r['ranker']}/{r['gm']}x{r['coach']} | "
              f"n={int(r['n_ct'])}CT+{int(r['n_pt'])}PT+{int(r['n_clin'])}Clin | "
              f"OOF={r['oof_auc']:.4f} | test={r['auc_test']:.4f} | ext={r['auc_ext']:.4f} | "
              f"lo95={r['boot_ext_lo95']:.4f} | "
              f"ba_ext={r['ba_ext']:.3f} | top_cand={int(r['top_candidate'])}")

    print(f"\nFilter funnel (main CT_PT_Clin rows only):")
    for flag in ["trio_ok", "balanced_ok", "ci_ok", "top_candidate"]:
        n = int(main_df[flag].sum()) if flag in main_df.columns else 0
        print(f"  {flag}: {n:,}")

    total = time.time() - t0
    print(f"\nTotal runtime: {total/60:.1f} min ({total/3600:.2f} h)")
    print("27_mar_T2_dev_ultimate_recheck.py finished.")


if __name__ == "__main__":
    main()
