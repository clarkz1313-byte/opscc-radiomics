from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import ExtraSurvivalTrees
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "2_apr_T2B_data"
TASK2_DIR = ROOT.parent / "Mar_2026_task2" / "12_mar_task2_rad_data"
TRAIN_FILE = DATA_DIR / "2_apr_t2b_train.csv"
EXT_FILE = DATA_DIR / "2_apr_t2b_ext.csv"
SPLIT_MAP_FILE = TASK2_DIR / "13_mar_task2_split_map.csv"

SEED = 42
N_FOLDS = 5
N_REPEATS_SCREEN = 1
N_REPEATS_CONFIRM = 3
N_EST_SCREEN = 100
N_EST_CONFIRM = 200
N_BOOT = 100
CLINICAL_FEATURES = ["Gender_Male"]
HPV_PARAMS = {"C": 0.5, "penalty": "l2", "solver": "lbfgs", "max_iter": 2000}
PT_FEATURES = [
    "GTVn_wavelet-LLH_firstorder_Mean",
    "GTVp_original_firstorder_InterquartileRange",
    "GTVp_wavelet-HHL_glrlm_ShortRunHighGrayLevelEmphasis",
    "GTVp_wavelet-HLH_glrlm_ShortRunHighGrayLevelEmphasis",
]
CT_FEATURES = [
    "GTVp_log-sigma-1-mm-3D_firstorder_Range",
    "GTVp_wavelet-HLL_ngtdm_Complexity",
    "GTVp_wavelet-LLH_glrlm_HighGrayLevelRunEmphasis",
]
EXPECTED = {
    "oof_ci_s1": 0.5693981481481482,
    "oof_auc_s1": 0.78,
    "oof_ci": 0.5868529271054571,
    "oof_auc": 0.7379629629629629,
    "test_ci": 0.6233766233766234,
    "test_auc": 0.6785714285714286,
    "test_ba": 0.7261904761904762,
    "test_spe": 0.6666666666666666,
    "test_sen": 0.7857142857142857,
    "ext_ci": 0.6890756302521008,
    "ext_auc": 0.7428571428571429,
    "ext_ba": 0.7785714285714285,
    "ext_spe": 0.8571428571428571,
    "ext_sen": 0.7,
    "boot_ci_lo": 0.5034188034188034,
    "boot_ci_hi": 0.9704999999999998,
    "boot_auc_lo": 0.5077721291866029,
    "boot_auc_hi": 0.9340422077922076,
}
TOL = 1e-9


def safe_ci(y: np.ndarray, risk: np.ndarray) -> float:
    try:
        return float(concordance_index_censored(y["event"], y["time"], risk)[0])
    except Exception:
        return 0.5


def youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, scores)
    return float(thr[int(np.argmax(tpr - fpr))])


def scale_blocks(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    def arr(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
        return df[cols].to_numpy(dtype=float)

    train_parts = [arr(train_df, CLINICAL_FEATURES), arr(train_df, PT_FEATURES), arr(train_df, CT_FEATURES)]
    eval_parts = [arr(eval_df, CLINICAL_FEATURES), arr(eval_df, PT_FEATURES), arr(eval_df, CT_FEATURES)]
    scaled_train = []
    scaled_eval = []
    for x_tr, x_ev in zip(train_parts, eval_parts):
        sc = StandardScaler()
        scaled_train.append(sc.fit_transform(x_tr))
        scaled_eval.append(sc.transform(x_ev))
    return np.hstack(scaled_train), np.hstack(scaled_eval)


def make_surv(n_estimators: int) -> ExtraSurvivalTrees:
    return ExtraSurvivalTrees(n_estimators=n_estimators, random_state=SEED, n_jobs=-1)


def make_hpv() -> LogisticRegression:
    return LogisticRegression(
        C=HPV_PARAMS["C"],
        penalty=HPV_PARAMS["penalty"],
        solver=HPV_PARAMS["solver"],
        class_weight="balanced",
        max_iter=HPV_PARAMS["max_iter"],
        random_state=SEED,
    )


def eval_oof(train_df: pd.DataFrame, n_repeats: int, n_estimators: int) -> tuple[float, float]:
    y_strat = (train_df["Relapse"].to_numpy(int) * 2 + train_df["HPV_binary"].to_numpy(int)).clip(0, 3)
    rkf = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=n_repeats, random_state=SEED)
    ci_scores: list[float] = []
    auc_scores: list[float] = []

    for tr_idx, vl_idx in rkf.split(train_df, y_strat):
        tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
        vl_df = train_df.iloc[vl_idx].reset_index(drop=True)
        x_tr, x_vl = scale_blocks(tr_df, vl_df)
        y_s_tr = Surv.from_arrays(event=tr_df["Relapse"].astype(bool), time=tr_df["RFS"])
        y_s_vl = Surv.from_arrays(event=vl_df["Relapse"].astype(bool), time=vl_df["RFS"])
        y_h_tr = tr_df["HPV_binary"].to_numpy(int)
        y_h_vl = vl_df["HPV_binary"].to_numpy(int)

        surv = make_surv(n_estimators)
        surv.fit(x_tr, y_s_tr)
        ci_scores.append(safe_ci(y_s_vl, surv.predict(x_vl)))

        hpv = make_hpv()
        hpv.fit(x_tr, y_h_tr)
        auc_scores.append(float(roc_auc_score(y_h_vl, hpv.predict_proba(x_vl)[:, 1])))

    return float(np.mean(ci_scores)), float(np.mean(auc_scores))


def evaluate_on_dataset(
    fit_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    n_estimators: int,
    n_boot: int = 0,
) -> dict[str, float]:
    x_fit, x_eval = scale_blocks(fit_df, eval_df)
    y_s_fit = Surv.from_arrays(event=fit_df["Relapse"].astype(bool), time=fit_df["RFS"])
    y_s_eval = Surv.from_arrays(event=eval_df["Relapse"].astype(bool), time=eval_df["RFS"])
    y_h_fit = fit_df["HPV_binary"].to_numpy(int)
    y_h_eval = eval_df["HPV_binary"].to_numpy(int)

    surv = make_surv(n_estimators)
    surv.fit(x_fit, y_s_fit)
    risk_eval = surv.predict(x_eval)
    out_ci = safe_ci(y_s_eval, risk_eval)

    hpv = make_hpv()
    hpv.fit(x_fit, y_h_fit)
    proba_eval = hpv.predict_proba(x_eval)[:, 1]
    raw_auc = float(roc_auc_score(y_h_eval, proba_eval))
    proba_use = 1.0 - proba_eval if raw_auc < 0.5 else proba_eval
    out_auc = max(raw_auc, 1.0 - raw_auc)
    thresh = youden_threshold(y_h_eval, proba_use)
    pred = (proba_use >= thresh).astype(int)
    out_ba = float(balanced_accuracy_score(y_h_eval, pred))
    tn = int(((y_h_eval == 0) & (pred == 0)).sum())
    fp = int(((y_h_eval == 0) & (pred == 1)).sum())
    fn = int(((y_h_eval == 1) & (pred == 0)).sum())
    tp = int(((y_h_eval == 1) & (pred == 1)).sum())
    out_spe = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    out_sen = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")

    out = {
        "ci": out_ci,
        "auc": out_auc,
        "ba": out_ba,
        "spe": out_spe,
        "sen": out_sen,
    }
    if n_boot <= 0:
        return out

    rng = np.random.default_rng(SEED)
    n_eval = len(eval_df)
    ci_boots = []
    auc_boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_eval, n_eval)
        ci_boots.append(safe_ci(y_s_eval[idx], risk_eval[idx]))
        raw_boot_auc = float(roc_auc_score(y_h_eval[idx], proba_use[idx]))
        auc_boots.append(max(raw_boot_auc, 1.0 - raw_boot_auc))
    out["boot_ci_lo"] = float(np.nanpercentile(ci_boots, 2.5))
    out["boot_ci_hi"] = float(np.nanpercentile(ci_boots, 97.5))
    out["boot_auc_lo"] = float(np.nanpercentile(auc_boots, 2.5))
    out["boot_auc_hi"] = float(np.nanpercentile(auc_boots, 97.5))
    return out


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pooled_df = pd.read_csv(TRAIN_FILE)
    ext_df = pd.read_csv(EXT_FILE)
    split_map = pd.read_csv(SPLIT_MAP_FILE)
    split_map["PatientID"] = split_map["PatientID"].astype(str)
    split_map["split"] = split_map["split"].astype(str).str.lower()

    train_ids = set(split_map.loc[split_map["split"] == "train", "PatientID"])
    test_ids = set(split_map.loc[split_map["split"] == "test", "PatientID"])

    for df in (pooled_df, ext_df):
        df["PatientID"] = df["PatientID"].astype(str)
        for col in ["CenterID", "HPV_binary", "Relapse"]:
            df[col] = df[col].astype(int)
        df["RFS"] = df["RFS"].astype(float)

    train_df = pooled_df[pooled_df["PatientID"].isin(train_ids)].reset_index(drop=True)
    test_df = pooled_df[pooled_df["PatientID"].isin(test_ids)].reset_index(drop=True)
    if not (len(train_df) == 67 and len(test_df) == 20 and len(ext_df) == 27):
        raise ValueError("Strict split sizes do not match expected 67 / 20 / 27")
    return train_df, test_df, ext_df


def assert_close(name: str, actual: float, expected: float) -> None:
    if not np.isfinite(actual) or abs(actual - expected) > TOL:
        raise AssertionError(f"{name} mismatch: actual={actual:.12f}, expected={expected:.12f}")


def main() -> None:
    train_df, test_df, ext_df = load_data()
    print("Locked Task 2B winner recheck: 13v23652")
    print("Features:")
    print(f"  PT: {PT_FEATURES}")
    print(f"  CT: {CT_FEATURES}")
    print("Model: EST(n=100/200 for CV; n=200 for final fit) + LR_L2_0.5")

    oof_ci_s1, oof_auc_s1 = eval_oof(train_df, N_REPEATS_SCREEN, N_EST_SCREEN)
    oof_ci, oof_auc = eval_oof(train_df, N_REPEATS_CONFIRM, N_EST_CONFIRM)
    test_metrics = evaluate_on_dataset(train_df, test_df, N_EST_CONFIRM, n_boot=0)
    ext_metrics = evaluate_on_dataset(train_df, ext_df, N_EST_CONFIRM, n_boot=N_BOOT)

    actual = {
        "oof_ci_s1": oof_ci_s1,
        "oof_auc_s1": oof_auc_s1,
        "oof_ci": oof_ci,
        "oof_auc": oof_auc,
        "test_ci": test_metrics["ci"],
        "test_auc": test_metrics["auc"],
        "test_ba": test_metrics["ba"],
        "test_spe": test_metrics["spe"],
        "test_sen": test_metrics["sen"],
        "ext_ci": ext_metrics["ci"],
        "ext_auc": ext_metrics["auc"],
        "ext_ba": ext_metrics["ba"],
        "ext_spe": ext_metrics["spe"],
        "ext_sen": ext_metrics["sen"],
        "boot_ci_lo": ext_metrics["boot_ci_lo"],
        "boot_ci_hi": ext_metrics["boot_ci_hi"],
        "boot_auc_lo": ext_metrics["boot_auc_lo"],
        "boot_auc_hi": ext_metrics["boot_auc_hi"],
    }

    for key, expected in EXPECTED.items():
        assert_close(key, actual[key], expected)

    print("\nReproduced metrics:")
    for key in [
        "oof_ci_s1", "oof_auc_s1", "oof_ci", "oof_auc",
        "test_ci", "test_auc", "test_ba", "test_spe", "test_sen",
        "ext_ci", "ext_auc", "ext_ba", "ext_spe", "ext_sen",
        "boot_ci_lo", "boot_ci_hi", "boot_auc_lo", "boot_auc_hi",
    ]:
        print(f"  {key}: {actual[key]:.12f}")
    print("\nStatus: exact reproduction passed.")


if __name__ == "__main__":
    main()
