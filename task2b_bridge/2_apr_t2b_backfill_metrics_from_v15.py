"""
Backfill strict test/ext precision, NPV, and F1 onto existing Task 2B result rows.

Default use:
    python Apr_2026_task2B/2_apr_t2b_backfill_metrics_from_v15.py \
        --in-csv Apr_2026_task2B/2_apr_T2B_outputs/t2b_all_results_13.csv \
        --out-csv Apr_2026_task2B/2_apr_T2B_outputs/t2b_all_results_13_plusmetrics.csv \
        --trial-nos 23652 23653
"""
from __future__ import annotations

import argparse
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sksurv.ensemble import ExtraSurvivalTrees
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.svm import FastSurvivalSVM
from sksurv.util import Surv

ROOT = Path(__file__).resolve().parent
TRAIN_FILE = ROOT / "2_apr_T2B_data" / "2_apr_t2b_train.csv"
EXT_FILE = ROOT / "2_apr_T2B_data" / "2_apr_t2b_ext.csv"
SPLIT_MAP_FILE = ROOT.parent / "Mar_2026_task2" / "12_mar_task2_rad_data" / "13_mar_task2_split_map.csv"
SEED = 42
N_EST_CONFIRM = 200
N_BOOT = 100

SURV_LABEL_TO_CONFIG = {
    "EST": ("EST", {"n_estimators": N_EST_CONFIRM}),
    "SVM_0001": ("SVM", {"alpha": 0.001}),
    "CoxPH01": ("CoxPH", {"alpha": 0.1}),
    "EST_120_sqrt": ("EST", {"n_estimators": 120, "max_features": "sqrt", "min_samples_leaf": 1, "min_samples_split": 2}),
    "EST_200_sqrt": ("EST", {"n_estimators": 200, "max_features": "sqrt", "min_samples_leaf": 1, "min_samples_split": 2}),
    "EST_320_sqrt": ("EST", {"n_estimators": 320, "max_features": "sqrt", "min_samples_leaf": 1, "min_samples_split": 2}),
    "EST_200_half": ("EST", {"n_estimators": 200, "max_features": 0.5, "min_samples_leaf": 1, "min_samples_split": 2}),
    "EST_320_half": ("EST", {"n_estimators": 320, "max_features": 0.5, "min_samples_leaf": 1, "min_samples_split": 2}),
    "EST_200_all": ("EST", {"n_estimators": 200, "max_features": 1.0, "min_samples_leaf": 1, "min_samples_split": 2}),
    "EST_200_leaf2": ("EST", {"n_estimators": 200, "max_features": "sqrt", "min_samples_leaf": 2, "min_samples_split": 4}),
    "EST_320_leaf2": ("EST", {"n_estimators": 320, "max_features": "sqrt", "min_samples_leaf": 2, "min_samples_split": 4}),
}

HPV_LABEL_TO_CONFIG = {
    "LR_L2_0.5": ("LR_L2", {"C": 0.5}),
    "LR_EN_1.0": ("LR_EN", {"C": 1.0, "l1_ratio": 0.5}),
    "SVM_L_001": ("SVM_L", {"C": 0.01}),
}


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
    return train_df, test_df, ext_df


def _safe_ci(y, risk):
    try:
        return float(concordance_index_censored(y["event"], y["time"], risk)[0])
    except Exception:
        return 0.5


def _youden_threshold(y_true, y_score):
    thresholds = np.unique(y_score)
    if len(thresholds) == 0:
        return 0.5
    best_t, best_j = 0.5, -1.0
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        tn = ((y_true == 0) & (pred == 0)).sum()
        fp = ((y_true == 0) & (pred == 1)).sum()
        fn = ((y_true == 1) & (pred == 0)).sum()
        tp = ((y_true == 1) & (pred == 1)).sum()
        spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        j = sen + spe - 1.0
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t


def _scale_blocks(train_df, eval_df, feat_pt, feat_ct):
    use_features = [*feat_pt, *feat_ct, "Gender_Male"]
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[use_features].to_numpy(float))
    x_eval = scaler.transform(eval_df[use_features].to_numpy(float))
    return x_train, x_eval


def _make_surv(key, params):
    if key == "EST":
        kwargs = {"n_estimators": params["n_estimators"], "random_state": SEED, "n_jobs": -1}
        for opt in ["max_depth", "min_samples_split", "min_samples_leaf", "max_features"]:
            if opt in params:
                kwargs[opt] = params[opt]
        return ExtraSurvivalTrees(**kwargs)
    if key == "SVM":
        return FastSurvivalSVM(alpha=params["alpha"], max_iter=1000, tol=1e-4, random_state=SEED)
    if key == "CoxPH":
        return CoxPHSurvivalAnalysis(alpha=params["alpha"])
    raise ValueError(key)


def _make_hpv(key, params):
    if key == "LR_L2":
        return LogisticRegression(C=params["C"], penalty="l2", solver="lbfgs", class_weight="balanced", max_iter=2000, random_state=SEED)
    if key == "LR_EN":
        return LogisticRegression(C=params["C"], penalty="elasticnet", solver="saga", l1_ratio=params["l1_ratio"], class_weight="balanced", max_iter=5000, random_state=SEED)
    if key == "SVM_L":
        base = LinearSVC(C=params["C"], class_weight="balanced", max_iter=5000, random_state=SEED)
        return CalibratedClassifierCV(base, cv=3)
    raise ValueError(key)


def _evaluate_on_dataset(feat_pt, feat_ct, surv_label, hpv_label, fit_df, eval_df, n_boot=0):
    s_key, s_params = SURV_LABEL_TO_CONFIG[surv_label]
    h_key, h_params = HPV_LABEL_TO_CONFIG[hpv_label]
    x_fit, x_eval = _scale_blocks(fit_df, eval_df, feat_pt, feat_ct)
    y_s_fit = Surv.from_arrays(event=fit_df["Relapse"].astype(bool), time=fit_df["RFS"])
    y_s_eval = Surv.from_arrays(event=eval_df["Relapse"].astype(bool), time=eval_df["RFS"])
    y_h_fit = fit_df["HPV_binary"].to_numpy(int)
    y_h_eval = eval_df["HPV_binary"].to_numpy(int)

    try:
        sm = _make_surv(s_key, s_params)
        sm.fit(x_fit, y_s_fit)
        risk_eval = sm.predict(x_eval)
        out_ci = _safe_ci(y_s_eval, risk_eval)
    except Exception:
        risk_eval = np.full(len(x_eval), np.nan)
        out_ci = float("nan")

    out_spe = out_sen = out_precision = out_npv = out_f1 = float("nan")
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
        out_precision = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
        out_npv = float(tn / (tn + fn)) if (tn + fn) > 0 else float("nan")
        out_f1 = float((2 * tp) / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else float("nan")
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
        "precision": out_precision,
        "npv": out_npv,
        "f1": out_f1,
        "boot_ci_lo": bci_lo,
        "boot_ci_hi": bci_hi,
        "boot_auc_lo": bauc_lo,
        "boot_auc_hi": bauc_hi,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-csv', required=True)
    parser.add_argument('--out-csv', required=True)
    parser.add_argument('--trial-nos', nargs='*', default=[])
    args = parser.parse_args()

    df = pd.read_csv(args.in_csv)
    if args.trial_nos:
        mask = df['trial_no'].astype(str).isin([str(x) for x in args.trial_nos])
    else:
        mask = pd.Series(True, index=df.index)

    train_df, test_df, ext_df = load_data()
    updates = 0
    for idx, row in df[mask].iterrows():
        feat_pt = str(row['feat_pt']).split('|')
        feat_ct = str(row['feat_ct']).split('|')
        surv_label = str(row['surv_gm'])
        hpv_label = str(row['hpv_gm'])
        t = _evaluate_on_dataset(feat_pt, feat_ct, surv_label, hpv_label, train_df, test_df, n_boot=0)
        e = _evaluate_on_dataset(feat_pt, feat_ct, surv_label, hpv_label, train_df, ext_df, n_boot=N_BOOT)
        df.loc[idx, 'test_precision'] = t['precision']
        df.loc[idx, 'test_npv'] = t['npv']
        df.loc[idx, 'test_f1'] = t['f1']
        df.loc[idx, 'ext_precision'] = e['precision']
        df.loc[idx, 'ext_npv'] = e['npv']
        df.loc[idx, 'ext_f1'] = e['f1']
        updates += 1

    df.to_csv(args.out_csv, index=False)
    print(f'Backfilled {updates} rows -> {args.out_csv}')


if __name__ == '__main__':
    main()
