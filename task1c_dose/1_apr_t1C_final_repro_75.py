"""
1_apr_t1C_final_repro_75.py

T1C-HMR75 Stage 4 finalist script: Arm A exhaustive sweep with full SC5 pipeline.
Mirrors 1_apr_t1C_exhaustive_sweep.py but replaces EST-only with the correct
SVM GM (alpha=0.001) + EST coach pipeline — identical to Task 1 SC5.

Arm A only: 7PT + 4CT + 1clin (all Task 1 winner features locked)
            + all C(12, k) dose combinations for k = 0..12
            → 4096 rows

Metrics per row:
  - oof_ci, fold_std   : dev 5-fold CV with SVM GM (same as SC5 model_oof)
  - objective          : W_PERF * oof_ci + W_STAB * max(0, 1 - fold_std/STD_THRESHOLD)
  - chus_ci            : EST coach on full 75-pt dev, evaluated on 44-pt CHUS
  - chus_boot_mean/lo/hi : 1000-sample bootstrap CI on CHUS
  - AUC/Brier/IBS     : CHUS time-dependent posthoc metrics at 1/2/3 years
  - dev_chus_gap       : chus_ci - oof_ci

Sanity checks:
  - k=0 row (baseline, no dose): expanded-cohort sensitivity baseline
  - k=12 row: full Stage 4-75 highest-CV finalist pool

Usage:
    cd "D:/Uppsala thesis"
    python Mar_2026_task1C/1_apr_t1C_final_repro_75.py
"""

from __future__ import annotations

import itertools
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import ExtraSurvivalTrees
from sksurv.metrics import (
    brier_score,
    concordance_index_censored,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.svm import FastSurvivalSVM
from sksurv.util import Surv

# ============================================================
# PATHS
# ============================================================
ROOT         = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

OUT_DIR  = ROOT / "Dose_SC5_outputs_75"
OUT_DIR.mkdir(exist_ok=True, parents=True)
OUT_CSV  = OUT_DIR / "1_apr_final_repro_results_75.csv"
TOP_CSV  = OUT_DIR / "1_apr_final_repro_top20_75.csv"
LOG_MD   = OUT_DIR / "1_apr_final_repro_log_75.md"

PT_DEV_FILE    = PROJECT_ROOT / "Mar_2026" / "27_feb_PT_development.csv"
CT_DEV_FILE    = PROJECT_ROOT / "Mar_2026" / "27_feb_CT_development.csv"
PT_EXT_FILE    = PROJECT_ROOT / "Mar_2026" / "27_feb_PT_external.csv"
CT_EXT_FILE    = PROJECT_ROOT / "Mar_2026" / "27_feb_CT_external.csv"
DOSE_DEV_FILE  = ROOT / "Dose_development_75.csv"
DOSE_CHUS_FILE = ROOT / "Dose_external_CHUS.csv"
FINALIST_FEATURE_FILE = (
    ROOT
    / "29_mar_T1C_fs_script_results"
    / "Dose_stage4_inter31_finalist_features_75.csv"
)
CLINICAL_FILE  = (
    PROJECT_ROOT
    / "Feb_2026"
    / "25_feb_clinical_reduced_dataset"
    / "25_feb_Processed_clinical_reduced.csv"
)

# ============================================================
# CONFIG — locked to Task 1 SC5 values
# ============================================================
SEED           = 42
N_FOLDS        = 5
N_EST          = 200
SVM_ALPHA      = 0.001      # locked from Task 1 winner
N_BOOT         = 1000
W_PERF         = 0.8
W_STAB         = 0.2
STD_THRESHOLD  = 0.08
SAVE_EVERY     = 500
TIME_POINTS_DAYS = np.array([365.0, 730.0, 1095.0])

# ============================================================
# FEATURE LISTS
# ============================================================
PT7_LOCKED = [
    "GTVp_exponential_glszm_HighGrayLevelZoneEmphasis",
    "GTVn_wavelet-LLH_firstorder_Mean",
    "GTVp_wavelet-HLH_glrlm_ShortRunHighGrayLevelEmphasis",
    "GTVp_gradient_glszm_ZoneEntropy",
    "GTVp_wavelet-LHL_glszm_SmallAreaHighGrayLevelEmphasis",
    "GTVn_wavelet-LHH_glszm_LowGrayLevelZoneEmphasis",
    "GTVp_wavelet-HLH_glszm_HighGrayLevelZoneEmphasis",
]

CT4_LOCKED = [
    "GTVp_wavelet-LLH_glrlm_HighGrayLevelRunEmphasis",
    "GTVp_wavelet-HLL_ngtdm_Complexity",
    "GTVp_gradient_glszm_SmallAreaLowGrayLevelEmphasis",
    "GTVp_wavelet-LHH_firstorder_RootMeanSquared",
]

CLIN_FIXED = ["Gender_Male"]

def load_dose_pool():
    if not FINALIST_FEATURE_FILE.exists():
        raise FileNotFoundError(f"Missing finalist feature manifest: {FINALIST_FEATURE_FILE}")

    manifest = pd.read_csv(FINALIST_FEATURE_FILE).sort_values("order")
    required_cols = {"order", "feature", "source_inter_no", "source_intra_no"}
    missing = required_cols.difference(manifest.columns)
    if missing:
        raise ValueError(
            f"Finalist feature manifest missing columns: {sorted(missing)}"
        )

    if len(manifest) != 12:
        raise ValueError(
            f"Expected 12 Stage 4-75 finalist features, found {len(manifest)}"
        )

    if not manifest["source_inter_no"].eq(31).all():
        raise ValueError("Finalist feature manifest is not locked to source_inter_no=31")

    if not manifest["source_intra_no"].eq("Dose_S3_1_30").all():
        raise ValueError("Finalist feature manifest is not locked to Dose_S3_1_30")

    return manifest["feature"].astype(str).tolist()


# Dose pool: 12-feature Stage 4-75 highest-CV finalist set.
# Loaded from the inter_no=31 finalist manifest for auditability.
DOSE_POOL = load_dose_pool()

# Dose feature names can collide with locked PET/CT radiomics names.
# Keep raw names for reporting, but namespace dose columns internally.
DOSE_PREFIX = "DOSE__"
DOSE_ALIASES = {f: f"{DOSE_PREFIX}{f}" for f in DOSE_POOL}
DOSE_POOL_MODEL = [DOSE_ALIASES[f] for f in DOSE_POOL]
DOSE_ORIGINAL_BY_MODEL = {v: k for k, v in DOSE_ALIASES.items()}

# Full Stage 4-75 finalist pool marker for sanity check.
DOSE_STAGE4_75_FINALIST = set(DOSE_POOL_MODEL)

ALL_FEATURES = CLIN_FIXED + PT7_LOCKED + CT4_LOCKED + DOSE_POOL_MODEL

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


_log_fh = open(LOG_MD, "w", encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, _log_fh)
sys.stderr = _Tee(sys.__stderr__, _log_fh)


# ============================================================
# HELPERS
# ============================================================
def make_surv(event, time):
    return Surv.from_arrays(
        event=np.asarray(event, dtype=bool),
        time=np.asarray(time, dtype=float),
    )


def safe_ci(y, risk):
    try:
        return float(concordance_index_censored(y["event"], y["time"], risk)[0])
    except Exception:
        return np.nan


def bootstrap_ci(y, risk, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    cis = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        c = safe_ci(y[idx], risk[idx])
        if not np.isnan(c):
            cis.append(c)
    arr = np.array(cis)
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def safe_td_auc(y_train_ref, y_test, risk, times=TIME_POINTS_DAYS):
    """Time-dependent cumulative/dynamic AUC on valid test time points."""
    try:
        t_min = float(y_test["time"].min())
        t_max = float(y_test["time"].max())
        valid_times = times[(times > t_min) & (times < t_max)]
        if len(valid_times) == 0:
            return [np.nan] * len(times), np.nan

        auc_vals, mean_auc = cumulative_dynamic_auc(
            y_train_ref, y_test, risk, valid_times
        )
        result = []
        vt_idx = 0
        for t in times:
            if t in valid_times:
                result.append(float(auc_vals[vt_idx]))
                vt_idx += 1
            else:
                result.append(np.nan)
        return result, float(mean_auc)
    except Exception:
        return [np.nan] * len(times), np.nan


def survfuncs_to_array(survfuncs, times):
    """Evaluate survival step functions for Brier/IBS calculations."""
    out = np.zeros((len(survfuncs), len(times)), dtype=float)
    for i, sf in enumerate(survfuncs):
        try:
            sf_times = sf.x
            sf_vals = sf.y
        except AttributeError:
            sf_times = None
            sf_vals = None

        for j, t in enumerate(times):
            try:
                val = float(sf(t))
                if np.isnan(val) and sf_times is not None:
                    val = 1.0 if t < sf_times[0] else float(sf_vals[-1])
            except Exception:
                val = np.nan
            out[i, j] = val

    for i in range(out.shape[0]):
        last_valid = 1.0
        for j in range(out.shape[1]):
            if np.isnan(out[i, j]):
                out[i, j] = last_valid
            else:
                last_valid = out[i, j]
    return out


def safe_brier(y_train_ref, y_test, survfuncs, times=TIME_POINTS_DAYS):
    """Brier score at fixed time points on valid test time points."""
    try:
        t_min = float(y_test["time"].min())
        t_max = float(y_test["time"].max())
        valid_times = times[(times > t_min) & (times < t_max)]
        if len(valid_times) == 0:
            return [np.nan] * len(times)

        surv_array = survfuncs_to_array(survfuncs, valid_times)
        _, bs_vals = brier_score(y_train_ref, y_test, surv_array, valid_times)
        result = []
        vt_idx = 0
        for t in times:
            if t in valid_times:
                result.append(float(bs_vals[vt_idx]))
                vt_idx += 1
            else:
                result.append(np.nan)
        return result
    except Exception:
        return [np.nan] * len(times)


def safe_ibs(y_train_ref, y_test, survfuncs):
    """Integrated Brier Score over the observed test follow-up range."""
    try:
        t_min = float(y_test["time"].min())
        t_max = float(y_test["time"].max())
        if t_max <= t_min:
            return np.nan
        times_grid = np.linspace(t_min + 1, t_max - 1, 50)
        times_grid = times_grid[times_grid > 0]
        if len(times_grid) < 2:
            return np.nan
        surv_array = survfuncs_to_array(survfuncs, times_grid)
        return float(integrated_brier_score(
            y_train_ref, y_test, surv_array, times_grid
        ))
    except Exception:
        return np.nan


def rounded_or_nan(value, ndigits=6):
    if value is None or np.isnan(value):
        return np.nan
    return round(float(value), ndigits)


def dev_cv_svm(X_train, y_train, col_idx):
    """5-fold CV on dev using SVM GM — mirrors model_oof() in 31_mar_t1C_dose_SC5.py."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    y_event  = y_train["event"].astype(int)
    oof_pred = np.zeros(len(X_train), dtype=float)
    oof_ok   = np.zeros(len(X_train), dtype=bool)
    fold_cis = []

    for tr_idx, vl_idx in skf.split(X_train, y_event):
        X_tr = X_train[tr_idx][:, col_idx]
        X_vl = X_train[vl_idx][:, col_idx]
        scaler = StandardScaler()
        try:
            svm = FastSurvivalSVM(alpha=SVM_ALPHA, max_iter=1000, tol=1e-4, random_state=SEED)
            svm.fit(scaler.fit_transform(X_tr), y_train[tr_idx])
            pred = svm.predict(scaler.transform(X_vl))
            oof_pred[vl_idx] = pred
            oof_ok[vl_idx]   = True
            fc = safe_ci(y_train[vl_idx], pred)
            if not np.isnan(fc):
                fold_cis.append(fc)
        except Exception:
            continue

    if oof_ok.sum() < len(X_train) * 0.5:
        return np.nan, np.nan
    oof_ci   = safe_ci(y_train[oof_ok], oof_pred[oof_ok])
    fold_std = float(np.std(fold_cis)) if len(fold_cis) > 1 else np.nan
    return round(oof_ci, 6), round(fold_std, 6)


def chus_est_coach(X_train, y_train, X_chus, y_chus, col_idx):
    """EST coach: fit on full dev, predict on CHUS."""
    empty_metrics = {
        "chus_ci": np.nan,
        "chus_boot_mean": np.nan,
        "chus_boot_lo": np.nan,
        "chus_boot_hi": np.nan,
        "AUC-1yr": np.nan,
        "AUC-2yr": np.nan,
        "AUC-3yr": np.nan,
        "Mean AUC": np.nan,
        "Brier-1yr": np.nan,
        "Brier-2yr": np.nan,
        "Brier-3yr": np.nan,
        "IBS": np.nan,
    }
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train[:, col_idx])
    X_ch_sc = scaler.transform(X_chus[:, col_idx])
    try:
        est = ExtraSurvivalTrees(n_estimators=N_EST, random_state=SEED, n_jobs=-1)
        est.fit(X_tr_sc, y_train)
        risk = est.predict(X_ch_sc)
        survfuncs = est.predict_survival_function(X_ch_sc)
    except Exception:
        return empty_metrics
    ci = safe_ci(y_chus, risk)
    boot_mean, boot_lo, boot_hi = bootstrap_ci(y_chus, risk)
    aucs, mean_auc = safe_td_auc(y_train, y_chus, risk)
    briers = safe_brier(y_train, y_chus, survfuncs)
    ibs = safe_ibs(y_train, y_chus, survfuncs)
    return {
        "chus_ci": rounded_or_nan(ci),
        "chus_boot_mean": rounded_or_nan(boot_mean),
        "chus_boot_lo": rounded_or_nan(boot_lo),
        "chus_boot_hi": rounded_or_nan(boot_hi),
        "AUC-1yr": rounded_or_nan(aucs[0]),
        "AUC-2yr": rounded_or_nan(aucs[1]),
        "AUC-3yr": rounded_or_nan(aucs[2]),
        "Mean AUC": rounded_or_nan(mean_auc),
        "Brier-1yr": rounded_or_nan(briers[0]),
        "Brier-2yr": rounded_or_nan(briers[1]),
        "Brier-3yr": rounded_or_nan(briers[2]),
        "IBS": rounded_or_nan(ibs),
    }


def sc5_objective(oof_ci, fold_std):
    """SC5 objective — mirrors compute_objective() in 31_mar_t1C_dose_SC5.py."""
    if np.isnan(oof_ci) or np.isnan(fold_std):
        return np.nan
    return round(W_PERF * oof_ci + W_STAB * max(0.0, 1.0 - fold_std / STD_THRESHOLD), 6)


# ============================================================
# DATA LOADING
# ============================================================
def load_data():
    for path in [PT_DEV_FILE, CT_DEV_FILE, PT_EXT_FILE, CT_EXT_FILE,
                 DOSE_DEV_FILE, DOSE_CHUS_FILE, CLINICAL_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")

    clinical = pd.read_csv(CLINICAL_FILE).dropna(subset=["Relapse", "RFS"])

    if "Cohort" in clinical.columns:
        clin_dev = clinical[clinical["Cohort"] == "Dev"][
            ["PatientID", "CenterID", "Relapse", "RFS", "Gender_Male"]
        ].copy()
    else:
        clin_dev = clinical[clinical["CenterID"].isin([1, 6, 7])][
            ["PatientID", "CenterID", "Relapse", "RFS", "Gender_Male"]
        ].copy()

    clin_chus = clinical[clinical["CenterID"] == 3][
        ["PatientID", "CenterID", "Relapse", "RFS", "Gender_Male"]
    ].copy()

    pt_dev   = pd.read_csv(PT_DEV_FILE)[["PatientID"] + PT7_LOCKED]
    ct_dev   = pd.read_csv(CT_DEV_FILE)[["PatientID"] + CT4_LOCKED]
    dose_dev = (
        pd.read_csv(DOSE_DEV_FILE)[["PatientID"] + DOSE_POOL]
        .rename(columns=DOSE_ALIASES)
    )

    pt_ext  = pd.read_csv(PT_EXT_FILE)
    ct_ext  = pd.read_csv(CT_EXT_FILE)
    pt_chus = pt_ext[pt_ext["PatientID"].astype(str).str.startswith("CHUS")][
        ["PatientID"] + PT7_LOCKED
    ].copy()
    ct_chus = ct_ext[ct_ext["PatientID"].astype(str).str.startswith("CHUS")][
        ["PatientID"] + CT4_LOCKED
    ].copy()
    dose_chus = (
        pd.read_csv(DOSE_CHUS_FILE)[["PatientID"] + DOSE_POOL]
        .rename(columns=DOSE_ALIASES)
    )

    dev_df = (
        clin_dev
        .merge(pt_dev,    on="PatientID", how="inner")
        .merge(ct_dev,    on="PatientID", how="inner")
        .merge(dose_dev,  on="PatientID", how="inner")
        .sort_values("PatientID").reset_index(drop=True)
    )
    chus_df = (
        clin_chus
        .merge(pt_chus,   on="PatientID", how="inner")
        .merge(ct_chus,   on="PatientID", how="inner")
        .merge(dose_chus, on="PatientID", how="inner")
        .sort_values("PatientID").reset_index(drop=True)
    )

    dev_n, dev_ev   = len(dev_df),  int(dev_df["Relapse"].sum())
    chus_n, chus_ev = len(chus_df), int(chus_df["Relapse"].sum())
    print(f"Dev={dev_n} ({dev_ev} events) | CHUS={chus_n} ({chus_ev} events)")
    if (dev_n, dev_ev) != (75, 13):
        raise ValueError(f"Unexpected dev cohort: {dev_n}/{dev_ev} (expected 75/13)")
    if (chus_n, chus_ev) != (44, 8):
        raise ValueError(f"Unexpected CHUS cohort: {chus_n}/{chus_ev} (expected 44/8)")

    for col in ALL_FEATURES:
        if col in dev_df.columns:
            med = dev_df[col].median()
            dev_df[col]  = dev_df[col].replace([np.inf, -np.inf], np.nan).fillna(med)
            chus_df[col] = chus_df[col].replace([np.inf, -np.inf], np.nan).fillna(med)

    X_train = dev_df[ALL_FEATURES].to_numpy(dtype=float)
    y_train = make_surv(dev_df["Relapse"], dev_df["RFS"])
    X_chus  = chus_df[ALL_FEATURES].to_numpy(dtype=float)
    y_chus  = make_surv(chus_df["Relapse"], chus_df["RFS"])
    feat_idx_map = {f: i for i, f in enumerate(ALL_FEATURES)}

    return X_train, y_train, X_chus, y_chus, feat_idx_map


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("1_apr_t1C_final_repro_75.py")
    print("T1C-HMR75: 7PT+4CT+1clin + Stage 4-75 C(12,k) dose pool, k=0..12")
    print("Pipeline: SVM GM (alpha=0.001) + EST coach — identical to Task 1 SC5")
    print(f"SEED={SEED} | N_FOLDS={N_FOLDS} | N_EST={N_EST} | N_BOOT={N_BOOT}")
    print("=" * 70)

    t0 = time.time()
    X_train, y_train, X_chus, y_chus, feat_idx_map = load_data()

    base_features = CLIN_FIXED + PT7_LOCKED + CT4_LOCKED
    total_combos  = sum(1 for k in range(len(DOSE_POOL_MODEL) + 1)
                        for _ in itertools.combinations(DOSE_POOL_MODEL, k))
    print(f"\nTotal combinations: {total_combos}  (4096 expected)")

    rows    = []
    counter = 0

    for k in range(len(DOSE_POOL_MODEL) + 1):
        for dose_combo in itertools.combinations(DOSE_POOL_MODEL, k):
            dose_combo  = list(dose_combo)
            dose_combo_raw = [DOSE_ORIGINAL_BY_MODEL[f] for f in dose_combo]
            feat_names  = base_features + dose_combo
            col_idx     = [feat_idx_map[f] for f in feat_names]

            oof_ci, fold_std   = dev_cv_svm(X_train, y_train, col_idx)
            objective          = sc5_objective(oof_ci, fold_std)
            chus_metrics = chus_est_coach(
                X_train, y_train, X_chus, y_chus, col_idx
            )
            chus_ci = chus_metrics["chus_ci"]
            dev_chus_gap = round(chus_ci - oof_ci, 6) if not (
                np.isnan(chus_ci) or np.isnan(oof_ci)
            ) else np.nan

            is_baseline = (k == 0)
            is_stage4_full_pool = (set(dose_combo) == DOSE_STAGE4_75_FINALIST)

            rows.append({
                "trial_no":       counter + 1,
                "n_dose":         k,
                "n_total":        len(feat_names),
                "dose_features":  "|".join(dose_combo_raw),
                "oof_ci":         oof_ci,
                "fold_std":       fold_std,
                "objective":      objective,
                "chus_ci":        chus_metrics["chus_ci"],
                "chus_boot_mean": chus_metrics["chus_boot_mean"],
                "chus_boot_lo":   chus_metrics["chus_boot_lo"],
                "chus_boot_hi":   chus_metrics["chus_boot_hi"],
                "AUC-1yr":        chus_metrics["AUC-1yr"],
                "AUC-2yr":        chus_metrics["AUC-2yr"],
                "AUC-3yr":        chus_metrics["AUC-3yr"],
                "Mean AUC":       chus_metrics["Mean AUC"],
                "Brier-1yr":      chus_metrics["Brier-1yr"],
                "Brier-2yr":      chus_metrics["Brier-2yr"],
                "Brier-3yr":      chus_metrics["Brier-3yr"],
                "IBS":            chus_metrics["IBS"],
                "dev_chus_gap":   dev_chus_gap,
                "is_baseline":    is_baseline,
                "is_stage4_full_pool": is_stage4_full_pool,
            })

            counter += 1
            if counter % 200 == 0:
                best = max((r["chus_ci"] for r in rows if not np.isnan(r["chus_ci"])),
                           default=np.nan)
                print(f"  {counter}/{total_combos} | k={k} | best CHUS so far: {best:.4f}")

            if counter % SAVE_EVERY == 0:
                pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
                print(f"  Checkpoint saved at {counter} rows")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    df.nlargest(20, "chus_ci").to_csv(TOP_CSV, index=False)

    elapsed = (time.time() - t0) / 60.0

    # --------------------------------------------------------
    # SANITY CHECKS
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("SANITY CHECKS")
    print("=" * 70)

    base_row = df[df["is_baseline"] == True].iloc[0]
    print(f"  k=0 baseline (7PT+4CT+1clin, no dose):")
    print(f"    oof_ci={base_row['oof_ci']:.4f}  chus_ci={base_row['chus_ci']:.4f}  "
          f"gap={base_row['dev_chus_gap']:+.4f}")
    print(f"    Expanded-cohort sensitivity baseline; no fixed reference expected")

    full_pool_rows = df[df["is_stage4_full_pool"] == True]
    if len(full_pool_rows):
        wr = full_pool_rows.iloc[0]
        status = "INFO"
        print(f"  Full Stage 4-75 finalist pool (k=12, inter_no=31 source):")
        print(f"    oof_ci={wr['oof_ci']:.4f}  chus_ci={wr['chus_ci']:.4f}  "
              f"gap={wr['dev_chus_gap']:+.4f}  [{status}]")
        print(f"    This is the highest-CV Stage 4-75 candidate pool before exhaustive subset sweep")
    else:
        print("  Full Stage 4-75 finalist pool row NOT FOUND; check DOSE_POOL")

    # --------------------------------------------------------
    # RESULTS SUMMARY
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    best_chus_row = df.nlargest(1, "chus_ci").iloc[0]
    best_obj_row  = df.nlargest(1, "objective").iloc[0]

    print(f"\nBest by CHUS:")
    print(f"  n_dose={int(best_chus_row['n_dose'])}  n_total={int(best_chus_row['n_total'])}")
    print(f"  oof_ci={best_chus_row['oof_ci']:.4f}  fold_std={best_chus_row['fold_std']:.4f}")
    print(f"  chus_ci={best_chus_row['chus_ci']:.4f}  boot_lo={best_chus_row['chus_boot_lo']:.4f}")
    print(f"  dev_chus_gap={best_chus_row['dev_chus_gap']:+.4f}")
    print(f"  dose_features: {best_chus_row['dose_features']}")

    print(f"\nBest by objective:")
    print(f"  n_dose={int(best_obj_row['n_dose'])}  n_total={int(best_obj_row['n_total'])}")
    print(f"  oof_ci={best_obj_row['oof_ci']:.4f}  objective={best_obj_row['objective']:.4f}")
    print(f"  chus_ci={best_obj_row['chus_ci']:.4f}")
    print(f"  dose_features: {best_obj_row['dose_features']}")

    print(f"\nCHUS by k (mean | max | best_oof | best_gap):")
    for k in range(len(DOSE_POOL_MODEL) + 1):
        grp = df[df["n_dose"] == k]
        if len(grp):
            best_g = grp.nlargest(1, "chus_ci").iloc[0]
            print(f"  k={k:2d} n={len(grp):4d} | "
                  f"mean_chus={grp['chus_ci'].mean():.4f}  "
                  f"max_chus={grp['chus_ci'].max():.4f}  "
                  f"oof@best={best_g['oof_ci']:.4f}  "
                  f"gap@best={best_g['dev_chus_gap']:+.4f}")

    print(f"\nSaved: {OUT_CSV}  ({len(df)} rows)")
    print(f"Saved: {TOP_CSV}  (top 20 by CHUS)")
    print(f"Log:   {LOG_MD}")
    print(f"Elapsed: {elapsed:.1f} minutes")


if __name__ == "__main__":
    main()
