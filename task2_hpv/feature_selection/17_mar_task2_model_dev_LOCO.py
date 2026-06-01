"""
17_mar_task2_model_dev_LOCO.py

Task 2 Model Development — Script A (Dell i9-11950H)
LOCO-based radiomics rankers: LOCO_evt, LOCO_EPV_CUT, WLOCO_enon, WLOCO_EPV_CUT

Reference plan: Mar_2026_task2/17_mar_task2_model_dev_plan.md (v5)

Design (GM + Coach exhaustive system, mirroring Task 1):
  - 4 LOCO rankers × 4 GMs × 1500 trials × 4 Coaches = 96,000 result rows
  - GM: runs inside Optuna OOF objective (5-fold, SEED=42), score = 0.7*AUC + 0.3*stability
  - Coach: fits on full X_train after Optuna, evaluates Test AND Ext simultaneously
  - Separate StandardScaler per block (CT / PT / Clin), fitted on fold-train for GM OOF
  - Clinical features ranked by LASSO-logistic (all-train), top-2 cap
  - Feature subset params: n_ct ∈ [8,16], n_pt ∈ [4,8], n_clin ∈ [1,2]
  - pass_both = 1 iff auc_test >= 0.70 AND auc_ext >= 0.70

LOCO variants (Task 1 naming preserved):
  LOCO_evt      → FULL_LOCO_evt analog: HPV− ≥ 2 hard cut, unweighted, no shrinkage
  LOCO_EPV_CUT  → FULL_LOCO_EPV_CUT analog: HPV− ≥ 2 AND HPV−×HPV+ ≥ 50, unweighted
  WLOCO_enon    → FULL_WLOCO_enon analog: no cut, w=HPV−×HPV+, shrinkage κ=5
  WLOCO_EPV_CUT → FULL_WLOCO_epv_cut analog: composite cut, equal weights, shrinkage κ=5

Input data:
  Train: 67 pts (CHUM=16/1HPV−, CHUP=22/11HPV−, HGJ=27/8HPV−, HMR=2/0HPV−)
  Test:  20 pts (internal held-out, evaluated simultaneously with Ext by Coach)
  Ext:   27 pts (CHUS, external validation)

Run:
    cd "D:/Uppsala thesis" && python Mar_2026_task2/17_mar_task2_model_dev_LOCO.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("default")
warnings.filterwarnings("ignore", category=UserWarning)          # sklearn convergence noise
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
from sklearn.metrics import roc_auc_score
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

OUT_DIR = SCRIPT_DIR / "17_mar_model_dev_outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

ALL_CSV  = OUT_DIR / "17_mar_LOCO_all_results.csv"
CKPT_CSV = OUT_DIR / "17_mar_LOCO_checkpoint.csv"
TOP_CSV  = OUT_DIR / "17_mar_LOCO_top_results.csv"
PRED_CSV = OUT_DIR / "17_mar_LOCO_best_predictions.csv"
LOG_FILE = OUT_DIR / "17_mar_LOCO_log.txt"

# ============================================================
# CONFIG
# ============================================================
SEED             = 42
N_FOLDS          = 5
N_BOOT           = 1000
N_TRIALS         = 2000          # per (ranker × GM) Optuna study
LABEL_COL        = "HPV_binary"
CLINICAL_FEATURES = ["Age", "Gender_Male", "Treatment_CRT"]

# Composition-based feature search: guarantee minimums per block, n_total free
# PT min > CT min: PT generalises better to ext (PT-alone ext=0.693 > CT-alone ~0.65 local)
# Mirrors Task 1 dataset preference where PT dominated ext performance
TOTAL_N_MIN = 7    # floor: MIN_CLIN(1) + MIN_PT(4) + MIN_CT(2)
TOTAL_N_MAX = 25   # ceiling: allows full 16CT+8PT+3clin=27 minus tiny slack
MIN_CLIN = 1
MIN_PT   = 4       # 50% of PT pool (8); higher floor reflecting PT ext strength
MIN_CT   = 2       # 12.5% of CT pool (16)

W_PERF, W_STAB, STD_THRESHOLD = 0.7, 0.3, 0.08
PASS_THRESHOLD = 0.70

# LOCO
LOCO_KAPPA       = 5.0   # shrinkage anchor (κ), same as Task 1
LOCO_HPV_MIN     = 2     # min HPV− in held-out fold for hard/composite cut
LOCO_ENON_MIN    = 50    # min HPV−×HPV+ for composite EPV cut

RANKER_NAMES = ["LOCO_evt", "LOCO_EPV_CUT", "WLOCO_enon", "WLOCO_EPV_CUT"]
GM_NAMES     = ["LR_L2", "LR_EN", "SVM_L", "RF"]
COACH_NAMES  = ["LR_L2", "LR_EN", "SVM_L", "RF"]

EXCLUDE_COLS = {"PatientID", LABEL_COL, "Relapse", "RFS",
                "Age", "Gender_Male", "Treatment_CRT", "prefix", "centre"}

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

_log_fh = open(LOG_FILE, "w", encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, _log_fh)
sys.stderr = _Tee(sys.__stderr__, _log_fh)

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
        merged = ct_df.merge(pt_df[["PatientID"] + pt_feat_list], on="PatientID",
                             how="inner", suffixes=("", "_PT"))
        # Rename any _PT-suffixed columns back (collision when a PT feat also exists in CT data)
        rename = {c: c[:-3] for c in merged.columns if c.endswith("_PT")}
        if rename:
            merged = merged.drop(columns=list(rename.values()), errors="ignore")
            merged = merged.rename(columns=rename)
        return merged

    train_df = _merge(ct_train, pt_train).copy()
    test_df  = _merge(ct_test,  pt_test).copy()
    ext_df   = _merge(ct_ext,   pt_ext).copy()

    def _centre(pid):
        pfx = str(pid).split("-")[0].upper()
        return {"CHUM": "CHUM", "CHUP": "CHUP", "HGJ": "HGJ",
                "HMR": "HMR", "CHUS": "CHUS"}.get(pfx, pfx)

    train_df["centre"] = train_df["PatientID"].apply(_centre)
    return train_df, test_df, ext_df, ct_feat_list, pt_feat_list


# ============================================================
# LOCO RANKERS
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
    LOCO ranking for binary classification. Adapted from Task 1's loco_rank_mode().

    Modes (named to match Task 1 FULL_LOCO_* analogs):
      loco_evt      : HPV− ≥ LOCO_HPV_MIN hard cut, unweighted, no shrinkage
      loco_epv_cut  : HPV− ≥ LOCO_HPV_MIN AND HPV−×HPV+ ≥ LOCO_ENON_MIN, unweighted
      w_enon        : no cut, w=HPV−×HPV+, shrinkage κ=LOCO_KAPPA
      w_epv_cut     : composite cut, equal weights, shrinkage κ=LOCO_KAPPA
    """
    centres   = np.unique(centre_ids)
    n_feats   = X_train.shape[1]
    auc_matrix = np.full((len(centres), n_feats), np.nan)
    c_weights  = np.zeros(len(centres), dtype=float)

    for ci, held in enumerate(centres):
        mask  = centre_ids == held
        n_c   = int(mask.sum())
        if n_c < 2:
            continue
        y_val    = y_train[mask]
        hpv_neg  = int((y_val == 0).sum())
        hpv_pos  = int((y_val == 1).sum())
        enon     = float(hpv_neg * hpv_pos)

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
            w_c           = max(0.0, enon)   # 0 if either class absent → auto-excluded
            use_shrinkage = True
        elif mode == "w_epv_cut":
            include       = (hpv_neg >= LOCO_HPV_MIN) and (enon >= LOCO_ENON_MIN)
            w_c           = 1.0              # equal weights for kept centres
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
                # Shrink toward 0.5 weighted by HPV− count, κ=5 (same as Task 1)
                auc_c = (hpv_neg * auc_c + LOCO_KAPPA * 0.5) / (hpv_neg + LOCO_KAPPA)
            auc_matrix[ci, fi] = auc_c

    valid = c_weights > 0
    if not np.any(valid):
        return list(feat_names), np.full(n_feats, 0.5)

    M = np.where(np.isnan(auc_matrix[valid]), 0.5, auc_matrix[valid])
    W = c_weights[valid]

    if mode in ("w_enon", "w_epv_cut"):
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
    "WLOCO_EPV_CUT":"w_epv_cut",
}


# ============================================================
# CLINICAL RANKER — LASSO-logistic
# ============================================================
def _clinical_rank(X_clin: np.ndarray, y: np.ndarray,
                   feat_names: list[str]) -> list[str]:
    """Rank clinical features by |LASSO coefficient|. Returns ordered list."""
    Xs = StandardScaler().fit_transform(X_clin)
    for C in np.logspace(-2, 1, 60):
        try:
            clf = LogisticRegression(penalty="l1", solver="saga", C=C,
                                     class_weight="balanced", random_state=SEED,
                                     max_iter=5000)
            clf.fit(Xs, y)
            coefs = np.abs(clf.coef_[0])
            if coefs.sum() > 0:
                return [feat_names[i] for i in np.argsort(coefs)[::-1]]
        except Exception:
            continue
    return list(feat_names)  # fallback: original order


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
        base = LinearSVC(C=params.get("C", 0.01), class_weight="balanced",
                         max_iter=5000, random_state=SEED)
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
    """Use trial's GM params if coach type matches GM; else fallback defaults."""
    if coach_name == "LR_L2":
        return {"C": trial_params["C"]} if gm_name == "LR_L2" else {"C": 1.0}
    if coach_name == "LR_EN":
        if gm_name == "LR_EN":
            return {"C": trial_params["C"], "l1_ratio": trial_params["l1_ratio"]}
        return {"C": 1.0, "l1_ratio": 0.5}
    if coach_name == "SVM_L":
        return {"C": trial_params["C"]} if gm_name == "SVM_L" else {"C": 0.01}
    if coach_name == "RF":
        if gm_name == "RF":
            return {"max_depth": trial_params["max_depth"],
                    "min_samples_leaf": trial_params["min_samples_leaf"]}
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
    5-fold OOF AUC for GM objective.
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
            print(f"[WARN] OOF fold failed gm={gm_name} params={gm_params}: "
                  f"{type(e).__name__}: {e}")
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
    Fit coach on full X_train, evaluate Test AND Ext in same pass.
    Returns (auc_test, auc_ext, boot_ext_dict, proba_te, proba_ext).
    """
    sc_ct   = StandardScaler().fit(X_ct_tr[:, ct_idx])
    sc_pt   = StandardScaler().fit(X_pt_tr[:, pt_idx])
    sc_clin = StandardScaler().fit(X_clin_tr[:, clin_idx])

    def _tf(Xct, Xpt, Xcl):
        return np.hstack([sc_ct.transform(Xct[:, ct_idx]),
                          sc_pt.transform(Xpt[:, pt_idx]),
                          sc_clin.transform(Xcl[:, clin_idx])])

    X_tr_sc  = _tf(X_ct_tr,  X_pt_tr,  X_clin_tr)
    X_te_sc  = _tf(X_ct_te,  X_pt_te,  X_clin_te)
    X_ext_sc = _tf(X_ct_ext, X_pt_ext, X_clin_ext)

    try:
        clf = _make_model(coach_name, coach_params)
        clf.fit(X_tr_sc, y_tr)
        proba_te  = clf.predict_proba(X_te_sc)[:, 1]
        proba_ext = clf.predict_proba(X_ext_sc)[:, 1]
    except Exception as e:
        print(f"[WARN] Coach failed coach={coach_name} params={coach_params}: "
              f"{type(e).__name__}: {e}")
        return np.nan, np.nan, {"mean": np.nan, "lo95": np.nan, "hi95": np.nan, "n": 0}, \
               np.full(len(y_te), np.nan), np.full(len(y_ext), np.nan)

    auc_test = float(roc_auc_score(y_te,  proba_te))  if len(np.unique(y_te))  > 1 else np.nan
    auc_ext  = float(roc_auc_score(y_ext, proba_ext)) if len(np.unique(y_ext)) > 1 else np.nan
    boot_ext = _bootstrap_auc(y_ext, proba_ext)
    return auc_test, auc_ext, boot_ext, proba_te, proba_ext


def _bootstrap_auc(y: np.ndarray, proba: np.ndarray,
                   n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    aucs = []
    n = len(y)
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
# INCREMENTAL BASELINES (cheap, no Optuna)
# ============================================================
def _baseline_rows(ranker: str,
                   ct_ranked, pt_ranked, clin_ranked,
                   ct_feat_list, pt_feat_list,
                   X_ct_tr, X_ct_te, X_ct_ext,
                   X_pt_tr, X_pt_te, X_pt_ext,
                   X_clin_tr, X_clin_te, X_clin_ext,
                   y_tr, y_te, y_ext,
                   start_no: int = 0) -> list[dict]:
    rows = []
    configs = [
        ("CT_only",   ct_ranked[:12],  ct_feat_list,  X_ct_tr,   X_ct_te,   X_ct_ext,   12, 0, 0),
        ("PT_only",   pt_ranked[:6],   pt_feat_list,  X_pt_tr,   X_pt_te,   X_pt_ext,   0,  6, 0),
        ("Clin_only", clin_ranked[:2], ["Age","Gender_Male","Treatment_CRT"],
                                                       X_clin_tr, X_clin_te, X_clin_ext, 0,  0, 2),
    ]
    for bno, (block, feats, feat_list, Xtr, Xte, Xext, nct, npt, ncl) in enumerate(configs):
        n_f = len(feats)
        idx = [feat_list.index(f) for f in feats]
        sc  = StandardScaler().fit(Xtr[:, idx])
        clf = LogisticRegression(penalty="l2", solver="lbfgs",
                                 class_weight="balanced", C=1.0,
                                 max_iter=2000, random_state=SEED)
        clf.fit(sc.transform(Xtr[:, idx]), y_tr)
        proba_te  = clf.predict_proba(sc.transform(Xte[:, idx]))[:, 1]
        proba_ext = clf.predict_proba(sc.transform(Xext[:, idx]))[:, 1]
        auc_test  = float(roc_auc_score(y_te,  proba_te))  if len(np.unique(y_te))  > 1 else np.nan
        auc_ext   = float(roc_auc_score(y_ext, proba_ext)) if len(np.unique(y_ext)) > 1 else np.nan
        boot      = _bootstrap_auc(y_ext, proba_ext)
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
            "pass_both":   int(not np.isnan(auc_test)  and not np.isnan(auc_ext)
                               and auc_test >= PASS_THRESHOLD and auc_ext >= PASS_THRESHOLD),
        })
    return rows


# ============================================================
# STRUCTURAL SEEDS (Task 1 soft-fork: pre-enqueue known-good combos)
# ============================================================
def _enqueue_structural_seeds(study: optuna.Study, gm_name: str) -> None:
    """
    Pre-enqueue a small set of structurally-varied compositions before TPE warmup.
    Mirrors Task 1's enqueue_structural_seeds() adapted for classification params.
    n_ct is derived (= n_total - n_clin - n_pt), not enqueued directly.
    """
    base_combos = [
        {"n_total": 3,  "n_clin": 1, "n_pt": 1},
        {"n_total": 6,  "n_clin": 1, "n_pt": 2},
        {"n_total": 8,  "n_clin": 1, "n_pt": 3},
        {"n_total": 10, "n_clin": 2, "n_pt": 3},
        {"n_total": 12, "n_clin": 2, "n_pt": 4},
        {"n_total": 15, "n_clin": 2, "n_pt": 5},
    ]
    if gm_name in ("LR_L2", "SVM_L"):
        hparam_sets = [{"C": v} for v in [1e-3, 1e-2, 1e-1, 1.0]]
    elif gm_name == "LR_EN":
        hparam_sets = [{"C": c, "l1_ratio": l1} for c, l1 in [(1e-2, 0.3), (1e-1, 0.5), (1.0, 0.7)]]
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
# MAIN
# ============================================================
def main():
    t0 = time.time()
    print("=" * 80)
    print("17_mar_task2_model_dev_LOCO.py  —  Dell i9-11950H")
    print(f"Rankers: {RANKER_NAMES}")
    print(f"GMs:     {GM_NAMES}")
    print(f"Coaches: {COACH_NAMES}")
    print(f"N_TRIALS={N_TRIALS} | N_BOOT={N_BOOT} | SEED={SEED}")
    print(f"Composition search: TOTAL=[{TOTAL_N_MIN},{TOTAL_N_MAX}] | MIN_CLIN={MIN_CLIN} MIN_PT={MIN_PT} MIN_CT={MIN_CT}")
    expected_main     = len(RANKER_NAMES) * len(GM_NAMES) * N_TRIALS * len(COACH_NAMES)
    expected_baseline = len(RANKER_NAMES) * 3
    print(f"Expected main rows:     {len(RANKER_NAMES)}×{len(GM_NAMES)}×{N_TRIALS}×{len(COACH_NAMES)} = {expected_main:,}")
    print(f"Expected baseline rows: {len(RANKER_NAMES)}×3 = {expected_baseline:,}")
    print(f"Expected total rows:    {expected_main + expected_baseline:,}")
    print("=" * 80)

    # ----------------------------------------------------------
    # Load data
    # ----------------------------------------------------------
    print("\n--- Loading data ---")
    train_df, test_df, ext_df, ct_feat_list, pt_feat_list = _load_data()
    print(f"Train={len(train_df)} | Test={len(test_df)} | Ext={len(ext_df)}")

    # Feature list guards
    ct_feat_list = [f for f in ct_feat_list if f not in EXCLUDE_COLS]
    pt_feat_list = [f for f in pt_feat_list if f not in EXCLUDE_COLS]
    dup = set(ct_feat_list) & set(pt_feat_list)
    if dup:
        raise ValueError(f"CT/PT feature overlap detected: {sorted(dup)}")
    missing_ct  = [f for f in ct_feat_list if f not in train_df.columns]
    missing_pt  = [f for f in pt_feat_list if f not in train_df.columns]
    if missing_ct or missing_pt:
        raise ValueError(f"Missing feature columns. CT:{missing_ct[:5]} PT:{missing_pt[:5]}")
    print(f"  CT features: {len(ct_feat_list)} | PT features: {len(pt_feat_list)} (after EXCLUDE_COLS filter)")

    y_train = train_df[LABEL_COL].values.astype(int)
    y_test  = test_df[LABEL_COL].values.astype(int)
    y_ext   = ext_df[LABEL_COL].values.astype(int)
    print(f"Train: HPV−={(y_train==0).sum()}, HPV+={(y_train==1).sum()}")
    print(f"Test:  HPV−={(y_test==0).sum()},  HPV+={(y_test==1).sum()}")
    print(f"Ext:   HPV−={(y_ext==0).sum()},   HPV+={(y_ext==1).sum()}")

    X_ct_train  = train_df[ct_feat_list].values.astype(float)
    X_ct_test   = test_df[ct_feat_list].values.astype(float)
    X_ct_ext    = ext_df[ct_feat_list].values.astype(float)
    X_pt_train  = train_df[pt_feat_list].values.astype(float)
    X_pt_test   = test_df[pt_feat_list].values.astype(float)
    X_pt_ext    = ext_df[pt_feat_list].values.astype(float)
    X_clin_train = train_df[CLINICAL_FEATURES].values.astype(float)
    X_clin_test  = test_df[CLINICAL_FEATURES].values.astype(float)
    X_clin_ext   = ext_df[CLINICAL_FEATURES].values.astype(float)

    centre_ids = train_df["centre"].values

    print("\nTrain centre breakdown:")
    for c, g in train_df.groupby("centre"):
        neg = int((g[LABEL_COL] == 0).sum())
        pos = int((g[LABEL_COL] == 1).sum())
        enon = neg * pos
        print(f"  {c}: n={len(g)}, HPV−={neg}, HPV+={pos}, enon={enon}")

    # ----------------------------------------------------------
    # Clinical ranking (once, all-train)
    # ----------------------------------------------------------
    print("\n--- Clinical ranking (LASSO-logistic) ---")
    clin_ranked = _clinical_rank(X_clin_train, y_train, CLINICAL_FEATURES)
    print(f"  Rank: {clin_ranked}")

    # ----------------------------------------------------------
    # Pre-compute LOCO rankings (once per ranker)
    # ----------------------------------------------------------
    print("\n--- Pre-computing LOCO rankers ---")
    sc_rad = StandardScaler().fit(np.hstack([X_ct_train, X_pt_train]))
    X_rad_sc = sc_rad.transform(np.hstack([X_ct_train, X_pt_train]))
    rad_all = ct_feat_list + pt_feat_list
    ct_set  = set(ct_feat_list)

    rankings: dict[str, dict] = {}
    for rk in RANKER_NAMES:
        mode = RANKER_MODE_MAP[rk]
        ranked_rad, _ = _loco_rank(X_rad_sc, y_train, centre_ids, rad_all, mode)
        ct_ranked = [f for f in ranked_rad if f in ct_set]
        pt_ranked = [f for f in ranked_rad if f not in ct_set]
        rankings[rk] = {"ct": ct_ranked, "pt": pt_ranked, "clin": clin_ranked}
        # Annotate which centres were active
        valid_centres = []
        for c_name in np.unique(centre_ids):
            mask = centre_ids == c_name
            neg = int((y_train[mask] == 0).sum())
            pos = int((y_train[mask] == 1).sum())
            enon = neg * pos
            if mode in ("loco_evt",):
                active = neg >= LOCO_HPV_MIN
            elif mode in ("loco_epv_cut", "w_epv_cut"):
                active = (neg >= LOCO_HPV_MIN) and (enon >= LOCO_ENON_MIN)
            elif mode == "w_enon":
                active = enon > 0
            else:
                active = True
            if active:
                valid_centres.append(f"{c_name}(neg={neg},enon={enon})")
        print(f"  {rk}: ct={len(ct_ranked)}, pt={len(pt_ranked)} | voting: {valid_centres}")

    # ----------------------------------------------------------
    # Optuna studies: 4 rankers × 4 GMs
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    # Checkpoint resume: load completed blocks on restart
    # ----------------------------------------------------------
    all_rows: list[dict] = []
    done_blocks: set[tuple[str, str]] = set()
    row_counter = 0  # unique per (ranker, gm, trial, coach) row

    exp_block_rows = N_TRIALS * len(COACH_NAMES)
    if CKPT_CSV.exists():
        print(f"\nCheckpoint found: {CKPT_CSV} — loading...")
        ckpt_df = pd.read_csv(CKPT_CSV)
        all_rows = ckpt_df.to_dict("records")
        if "trial_no" in ckpt_df.columns and len(ckpt_df):
            row_counter = int(pd.to_numeric(ckpt_df["trial_no"], errors="coerce").max(skipna=True))
        main_ckpt = ckpt_df[ckpt_df.get("block", "") == "CT_PT_Clin"] if "block" in ckpt_df.columns else pd.DataFrame()
        for (rk, gm), g in main_ckpt.groupby(["ranker", "gm"]) if len(main_ckpt) else []:
            if len(g) >= int(exp_block_rows * 0.9):  # 90% threshold: robust to TrialPruned losses
                done_blocks.add((rk, gm))
        print(f"  Loaded {len(all_rows):,} rows | row_counter={row_counter} | "
              f"done_blocks={len(done_blocks)}/{len(RANKER_NAMES)*len(GM_NAMES)}")
    else:
        print("\nNo checkpoint found — starting fresh.")

    total_blocks = len(RANKER_NAMES) * len(GM_NAMES)
    block_no = 0

    for ranker in RANKER_NAMES:
        ct_ranked   = rankings[ranker]["ct"]
        pt_ranked   = rankings[ranker]["pt"]
        clin_ranked_r = rankings[ranker]["clin"]

        # Index maps for fast slicing
        ct_rank_idx  = [ct_feat_list.index(f)  for f in ct_ranked]
        pt_rank_idx  = [pt_feat_list.index(f)  for f in pt_ranked]
        clin_rank_idx = [CLINICAL_FEATURES.index(f) for f in clin_ranked_r]

        for gm in GM_NAMES:
            block_no += 1
            if (ranker, gm) in done_blocks:
                print(f"\n[{block_no}/{total_blocks}] Ranker={ranker} | GM={gm} — SKIPPED (checkpoint complete)")
                continue
            block_start = time.time()
            print(f"\n[{block_no}/{total_blocks}] Ranker={ranker} | GM={gm} | Trials={N_TRIALS}")

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=SEED),
            )
            _enqueue_structural_seeds(study, gm)

            def objective(trial,
                          _ct_ri=ct_rank_idx, _pt_ri=pt_rank_idx, _cl_ri=clin_rank_idx,
                          _gm=gm):
                # Composition search: sample n_total then allocate clin, pt, ct (each >=1)
                n_total = trial.suggest_int("n_total", TOTAL_N_MIN,
                                            min(TOTAL_N_MAX, len(_ct_ri) + len(_pt_ri) + len(_cl_ri)))

                max_clin = min(len(_cl_ri), n_total - MIN_PT - MIN_CT)
                if max_clin < MIN_CLIN:
                    raise optuna.TrialPruned()
                n_clin = trial.suggest_int("n_clin", MIN_CLIN, max_clin)

                n_rad = n_total - n_clin
                n_pt_max = min(len(_pt_ri), n_rad - MIN_CT)
                if n_pt_max < MIN_PT:
                    raise optuna.TrialPruned()
                n_pt = trial.suggest_int("n_pt", MIN_PT, n_pt_max)

                n_ct = n_rad - n_pt
                if n_ct < MIN_CT or n_ct > len(_ct_ri):
                    raise optuna.TrialPruned()

                ct_idx_fold   = _ct_ri[:n_ct]
                pt_idx_fold   = _pt_ri[:n_pt]
                clin_idx_fold = _cl_ri[:n_clin]

                gm_params = _suggest_gm_params(trial, _gm)

                oof_auc, fold_std = _model_oof_auc(
                    X_ct_train, X_pt_train, X_clin_train, y_train,
                    ct_idx_fold, pt_idx_fold, clin_idx_fold, _gm, gm_params,
                )

                score = W_PERF * oof_auc + W_STAB * max(0.0, 1.0 - fold_std / STD_THRESHOLD)

                trial.set_user_attr("n_ct",       n_ct)
                trial.set_user_attr("n_pt",       n_pt)
                trial.set_user_attr("n_clin",     n_clin)
                trial.set_user_attr("ct_feats",   "|".join(ct_ranked[:n_ct]))
                trial.set_user_attr("pt_feats",   "|".join(pt_ranked[:n_pt]))
                trial.set_user_attr("clin_feats", "|".join(clin_ranked_r[:n_clin]))
                trial.set_user_attr("gm_params",  str(gm_params))
                trial.set_user_attr("oof_auc",    float(oof_auc))
                trial.set_user_attr("fold_std",   float(fold_std))

                if (trial.number + 1) % 150 == 0:
                    print(f"  Trial {trial.number+1}/{N_TRIALS} | OOF={oof_auc:.4f} | score={score:.4f}")

                return score

            study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

            # --------------------------------------------------
            # Coach evaluation — every completed trial × all Coaches
            # --------------------------------------------------
            completed = [t for t in study.trials
                         if t.state == optuna.trial.TrialState.COMPLETE]
            print(f"  Evaluating {len(completed)} trials × {len(COACH_NAMES)} coaches "
                  f"(Test + Ext simultaneously)...")

            block_rows: list[dict] = []
            for t in completed:
                n_ct   = int(t.user_attrs["n_ct"])
                n_pt   = int(t.user_attrs["n_pt"])
                n_clin = int(t.user_attrs["n_clin"])
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

                    auc_test, auc_ext, boot_ext, _, _ = _coach_eval(
                        X_ct_train, X_pt_train, X_clin_train, y_train,
                        X_ct_test,  X_pt_test,  X_clin_test,  y_test,
                        X_ct_ext,   X_pt_ext,   X_clin_ext,   y_ext,
                        ct_idx_c, pt_idx_c, clin_idx_c,
                        coach, coach_params,
                    )

                    row_counter += 1  # unique per (ranker, gm, trial, coach) row
                    block_rows.append({
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
                        "pass_both":   int(not np.isnan(auc_test) and not np.isnan(auc_ext)
                                           and auc_test >= PASS_THRESHOLD
                                           and auc_ext  >= PASS_THRESHOLD),
                    })

            all_rows.extend(block_rows)

            # Block summary
            bdf = pd.DataFrame(block_rows)
            n_strict = int(bdf["pass_both"].sum()) if len(bdf) else 0
            best_ext = float(bdf["auc_ext"].max()) if len(bdf) else np.nan
            elapsed  = time.time() - block_start
            print(f"  Block: {len(block_rows):,} rows | pass_both={n_strict} | "
                  f"best_ext={best_ext:.4f} | elapsed={elapsed/60:.1f} min")

            # Checkpoint after every block
            pd.DataFrame(all_rows).to_csv(CKPT_CSV, index=False)
            print(f"  Checkpoint saved ({len(all_rows):,} total rows)")

    # ----------------------------------------------------------
    # Incremental baselines (per ranker, cheap LR_L2 C=1)
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
    print(f"  All results: {ALL_CSV} ({len(all_df):,} rows)")

    main_df = all_df[all_df["block"] == "CT_PT_Clin"].copy()
    pass_df = main_df[main_df["pass_both"] == 1].copy()
    if len(pass_df) > 0:
        top_df = pass_df.sort_values(["auc_ext", "oof_auc"], ascending=False)
    else:
        main_df["_auc_sum"] = main_df["auc_ext"] + main_df["oof_auc"]
        top_df = main_df.sort_values(["_auc_sum", "auc_ext", "oof_auc"], ascending=False).head(20).drop(columns=["_auc_sum"])
    top_df.to_csv(TOP_CSV, index=False)
    print(f"  Top results: {TOP_CSV} (pass_both={len(pass_df):,} rows saved; "
          f"{'all pass rows' if len(pass_df) > 0 else 'fallback top-20 by auc_ext+oof_auc'})")

    # Best model predictions
    if len(pass_df) > 0:
        bo_row = pass_df.sort_values(["auc_ext", "oof_auc"], ascending=False).iloc[0]
    else:
        _mdf = main_df.copy()
        _mdf["_auc_sum"] = _mdf["auc_ext"] + _mdf["oof_auc"]
        bo_row = _mdf.sort_values(["_auc_sum", "auc_ext", "oof_auc"], ascending=False).iloc[0]

    bo_ct    = bo_row["ct_features"].split("|")
    bo_pt    = bo_row["pt_features"].split("|")
    bo_clin  = bo_row["clin_features"].split("|")
    bo_coach = bo_row["coach"]
    try:
        import ast
        bo_cparams = ast.literal_eval(bo_row["coach_params"])
    except Exception:
        bo_cparams = {}

    bo_ct_idx   = [ct_feat_list.index(f)   for f in bo_ct]
    bo_pt_idx   = [pt_feat_list.index(f)   for f in bo_pt]
    bo_clin_idx = [CLINICAL_FEATURES.index(f) for f in bo_clin]

    _, _, _, proba_te, proba_ext = _coach_eval(
        X_ct_train, X_pt_train, X_clin_train, y_train,
        X_ct_test,  X_pt_test,  X_clin_test,  y_test,
        X_ct_ext,   X_pt_ext,   X_clin_ext,   y_ext,
        bo_ct_idx, bo_pt_idx, bo_clin_idx, bo_coach, bo_cparams,
    )
    pred_rows = []
    for i, row in enumerate(test_df.itertuples()):
        pred_rows.append({"PatientID": row.PatientID, "split": "test",
                          "y_true": int(y_test[i]),
                          "proba_HPVpos": float(proba_te[i]),
                          "predicted_label": int(proba_te[i] >= 0.5)})
    for i, row in enumerate(ext_df.itertuples()):
        pred_rows.append({"PatientID": row.PatientID, "split": "ext",
                          "y_true": int(y_ext[i]),
                          "proba_HPVpos": float(proba_ext[i]),
                          "predicted_label": int(proba_ext[i] >= 0.5)})
    pd.DataFrame(pred_rows).to_csv(PRED_CSV, index=False)
    print(f"  Predictions: {PRED_CSV} (ranker={bo_row['ranker']}, coach={bo_coach}, "
          f"ext={bo_row['auc_ext']:.4f})")

    total = time.time() - t0
    print(f"\nTotal: {total/60:.1f} min ({total/3600:.2f} h)")
    print("17_mar_task2_model_dev_LOCO.py finished.")


if __name__ == "__main__":
    main()
