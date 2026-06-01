"""
30_mar_task1_posthoc_metrics.py

Task 1 Post-Hoc Supplementary Metrics
Locked winner: PT_S3_1_768xCT_S3_8_235, N=12 (1 clinical + 7 PT + 4 CT)
Coach: ExtraSurvivalTrees(n_estimators=200, random_state=42)

Purpose:
    Compute time-dependent AUC (IPCW) and Integrated Brier Score (IBS)
    for the locked Task 1 winner on dev OOF, CHUS, and CHUP.
    No re-selection, no pipeline changes. Winner identity is frozen.
    Risk scores are generated identically to 28_mar_task1_branchB_KM.py.

Metrics computed:
    - Time-dependent AUC at 1, 2, 3 years (cumulative/dynamic, IPCW)
    - Integrated Brier Score (IBS) over the observed time range
    - Brier score at 1, 2, 3 years

Reproducibility gate:
    CHUS C-index must match 0.742857 (tol 0.001)
    CHUP C-index must match 0.727586 (tol 0.001)
    Script aborts if either delta exceeds tolerance.

Outputs (saved to Mar_2026/30_mar_task1_posthoc_metrics_outputs/):
    - posthoc_metrics_summary.csv   — all metrics in one table
    - posthoc_metrics_summary.md    — markdown table for thesis
    - posthoc_metrics_log.md        — full console log
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import ExtraSurvivalTrees
from sksurv.metrics import (
    concordance_index_censored,
    cumulative_dynamic_auc,
    integrated_brier_score,
    brier_score,
)
from sksurv.util import Surv

# ============================================================
# CONFIG
# ============================================================
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "30_mar_task1_posthoc_metrics_outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

LOG_PATH     = OUT_DIR / "posthoc_metrics_log.md"
CSV_PATH     = OUT_DIR / "posthoc_metrics_summary.csv"
MD_PATH      = OUT_DIR / "posthoc_metrics_summary.md"

PT_DEV_FILE   = ROOT / "27_feb_PT_development.csv"
CT_DEV_FILE   = ROOT / "27_feb_CT_development.csv"
PT_EXT_FILE   = ROOT / "27_feb_PT_external.csv"
CT_EXT_FILE   = ROOT / "27_feb_CT_external.csv"
CLINICAL_FILE = ROOT.parent / "Feb_2026" / "25_feb_clinical_reduced_dataset" / "25_feb_Processed_clinical_reduced.csv"
PT_FEAT_FILE  = ROOT / "2_mar_finalist_outputs" / "PT_inter1_768_features_recheck.csv"
CT_FEAT_FILE  = ROOT / "2_mar_finalist_outputs" / "CT_inter8_235_features_recheck.csv"

SEED    = 42
N_FOLDS = 5

# Locked winner feature set — identical to 28_mar_task1_branchB_KM.py
CLINICAL_FEATURES = ["Gender_Male"]
PT_WINNER = [
    "GTVp_exponential_glszm_HighGrayLevelZoneEmphasis",
    "GTVn_wavelet-LLH_firstorder_Mean",
    "GTVp_wavelet-HLH_glrlm_ShortRunHighGrayLevelEmphasis",
    "GTVp_gradient_glszm_ZoneEntropy",
    "GTVp_wavelet-LHL_glszm_SmallAreaHighGrayLevelEmphasis",
    "GTVn_wavelet-LHH_glszm_LowGrayLevelZoneEmphasis",
    "GTVp_wavelet-HLH_glszm_HighGrayLevelZoneEmphasis",
]
CT_WINNER = [
    "GTVp_wavelet-LLH_glrlm_HighGrayLevelRunEmphasis",
    "GTVp_wavelet-HLL_ngtdm_Complexity",
    "GTVp_gradient_glszm_SmallAreaLowGrayLevelEmphasis",
    "GTVp_wavelet-LHH_firstorder_RootMeanSquared",
]
WINNER_FEATURES = CLINICAL_FEATURES + PT_WINNER + CT_WINNER  # 12 total

LOCKED_CHUS = 0.742857142857143
LOCKED_CHUP = 0.727586206896552
REPRO_TOL   = 0.001

# Time points in days (1, 2, 3 years)
TIME_POINTS_DAYS = np.array([365.0, 730.0, 1095.0])
TIME_LABELS      = ["1yr", "2yr", "3yr"]

# ============================================================
# LOGGING
# ============================================================
_log_fh = open(LOG_PATH, "w", encoding="utf-8")

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
        return float("nan")


def safe_td_auc(y_train, y_test, risk, times):
    """
    Time-dependent cumulative/dynamic AUC (IPCW).
    y_train used to fit censoring distribution.
    Returns list of AUC values at each time point, and integrated mean.
    Clips times to be strictly within observed range of y_test.
    """
    try:
        t_min = float(y_test["time"].min())
        t_max = float(y_test["time"].max())
        # Use times strictly within (t_min, t_max) to avoid boundary issues
        valid_times = times[(times > t_min) & (times < t_max)]
        if len(valid_times) == 0:
            return [np.nan] * len(times), np.nan
        auc_vals, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk, valid_times)
        # Map back to full time list
        result = []
        vt_idx = 0
        for t in times:
            if t in valid_times:
                result.append(float(auc_vals[vt_idx]))
                vt_idx += 1
            else:
                result.append(np.nan)
        return result, float(mean_auc)
    except Exception as e:
        print(f"  [WARN] TD-AUC failed: {e}")
        return [np.nan] * len(times), np.nan


def survfuncs_to_array(survfuncs, times):
    """
    Convert list of sksurv StepFunction objects to a 2D array
    of shape (n_patients, n_times) by evaluating each function at query times.
    sksurv brier_score/integrated_brier_score expect this format.

    NaN handling: if a StepFunction cannot evaluate at a time point (out of
    its training range), we impute:
      - before first time point: 1.0 (no events yet, survival = 1)
      - after last time point:   last known survival value
    This prevents NaN propagation into IBS.
    """
    n = len(survfuncs)
    out = np.zeros((n, len(times)), dtype=float)
    for i, sf in enumerate(survfuncs):
        # Get the step function's own time grid and values
        try:
            sf_times = sf.x   # breakpoints
            sf_vals  = sf.y   # survival probabilities at breakpoints
        except AttributeError:
            sf_times = None
            sf_vals  = None

        for j, t in enumerate(times):
            try:
                val = float(sf(t))
                if np.isnan(val) and sf_times is not None:
                    # Impute: before first point → 1.0; after last → last value
                    if t < sf_times[0]:
                        val = 1.0
                    else:
                        val = float(sf_vals[-1])
            except Exception:
                val = np.nan
            out[i, j] = val

    # Final safety: forward-fill any remaining NaNs column-wise won't help;
    # instead clip per-patient: carry last valid value forward across times
    for i in range(n):
        last_valid = 1.0
        for j in range(out.shape[1]):
            if np.isnan(out[i, j]):
                out[i, j] = last_valid
            else:
                last_valid = out[i, j]
    return out


def safe_brier(y_train, y_test, survfuncs, times):
    """
    Brier score at specific time points.
    survfuncs: list of StepFunction objects (one per patient in y_test).
    Returns list of Brier scores at each time point.
    """
    try:
        t_min = float(y_test["time"].min())
        t_max = float(y_test["time"].max())
        valid_times = times[(times > t_min) & (times < t_max)]
        if len(valid_times) == 0:
            return [np.nan] * len(times)
        surv_array = survfuncs_to_array(survfuncs, valid_times)
        _, bs_vals = brier_score(y_train, y_test, surv_array, valid_times)
        result = []
        vt_idx = 0
        for t in times:
            if t in valid_times:
                result.append(float(bs_vals[vt_idx]))
                vt_idx += 1
            else:
                result.append(np.nan)
        return result
    except Exception as e:
        print(f"  [WARN] Brier score failed: {e}")
        return [np.nan] * len(times)


def safe_ibs(y_train, y_test, survfuncs):
    """
    Integrated Brier Score over the observed time range of y_test.
    """
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
        ibs_val = integrated_brier_score(y_train, y_test, surv_array, times_grid)
        return float(ibs_val)
    except Exception as e:
        print(f"  [WARN] IBS failed: {e}")
        return np.nan


def get_survival_functions(model, X_sc):
    """
    Extract survival functions from a fitted ExtraSurvivalTrees model.
    Returns list of StepFunction objects aligned to X_sc rows.
    """
    return model.predict_survival_function(X_sc)


# ============================================================
# LOAD DATA  (identical to 28_mar_task1_branchB_KM.py)
# ============================================================
print("=" * 70)
print("30_mar_task1_posthoc_metrics.py")
print("Task 1 Post-Hoc Supplementary Metrics")
print(f"Winner: N=12 (1 clin + 7 PT + 4 CT) | Coach: EST(200, seed={SEED})")
print("=" * 70)

clinical = pd.read_csv(CLINICAL_FILE).dropna(subset=["Relapse", "RFS"])

dev_clin  = clinical[clinical["Cohort"] == "Dev"][
    ["PatientID", "CenterID", "Relapse", "RFS"] + CLINICAL_FEATURES
].copy()
chus_clin = clinical[clinical["CenterID"] == 3][
    ["PatientID", "Relapse", "RFS"] + CLINICAL_FEATURES
].copy()
chup_clin = clinical[clinical["CenterID"] == 2][
    ["PatientID", "Relapse", "RFS"] + CLINICAL_FEATURES
].copy()

# Load radiomics
pt_dev = pd.read_csv(PT_DEV_FILE)
ct_dev = pd.read_csv(CT_DEV_FILE)
pt_ext = pd.read_csv(PT_EXT_FILE)
ct_ext = pd.read_csv(CT_EXT_FILE)

all_features = WINNER_FEATURES

def build_cohort(clin_df, pt_df, ct_df, id_filter=None):
    rad = pt_df[["PatientID"] + PT_WINNER].merge(
        ct_df[["PatientID"] + CT_WINNER], on="PatientID", how="inner"
    )
    if id_filter is not None:
        rad = rad[rad["PatientID"].str.startswith(id_filter)]
    df = clin_df.merge(rad, on="PatientID", how="inner")
    return df

dev_df  = build_cohort(dev_clin,  pt_dev, ct_dev)
chus_df = build_cohort(chus_clin, pt_ext, ct_ext, id_filter="CHUS")
chup_df = build_cohort(chup_clin, pt_ext, ct_ext, id_filter="CHUP")

print(
    f"Dev={len(dev_df)} ({int(dev_df['Relapse'].sum())} events) | "
    f"CHUS={len(chus_df)} ({int(chus_df['Relapse'].sum())} events) | "
    f"CHUP={len(chup_df)} ({int(chup_df['Relapse'].sum())} events)"
)

X_dev  = dev_df[all_features].values.astype(float)
y_dev  = make_surv(dev_df["Relapse"],  dev_df["RFS"])
X_chus = chus_df[all_features].values.astype(float)
y_chus = make_surv(chus_df["Relapse"], chus_df["RFS"])
X_chup = chup_df[all_features].values.astype(float)
y_chup = make_surv(chup_df["Relapse"], chup_df["RFS"])

# ============================================================
# LOCKED MODEL: fit on full dev, evaluate external
# ============================================================
print("\n--- Fitting locked model on full dev ---")
scaler_full = StandardScaler()
X_dev_sc  = scaler_full.fit_transform(X_dev)
X_chus_sc = scaler_full.transform(X_chus)
X_chup_sc = scaler_full.transform(X_chup)

model_full = ExtraSurvivalTrees(n_estimators=200, random_state=SEED, n_jobs=-1)
model_full.fit(X_dev_sc, y_dev)

risk_chus = model_full.predict(X_chus_sc)
risk_chup = model_full.predict(X_chup_sc)

ci_chus = safe_ci(y_chus, risk_chus)
ci_chup = safe_ci(y_chup, risk_chup)

print(f"  CHUS C-index: {ci_chus:.6f} (locked: {LOCKED_CHUS:.6f}, delta: {abs(ci_chus - LOCKED_CHUS):.6f})")
print(f"  CHUP C-index: {ci_chup:.6f} (locked: {LOCKED_CHUP:.6f}, delta: {abs(ci_chup - LOCKED_CHUP):.6f})")

if abs(ci_chus - LOCKED_CHUS) > REPRO_TOL:
    raise RuntimeError(
        f"CHUS C-index reproducibility FAILED: got {ci_chus:.6f}, expected {LOCKED_CHUS:.6f}"
    )
if abs(ci_chup - LOCKED_CHUP) > REPRO_TOL:
    raise RuntimeError(
        f"CHUP C-index reproducibility FAILED: got {ci_chup:.6f}, expected {LOCKED_CHUP:.6f}"
    )
print("  Reproducibility gate PASSED.")

# Survival functions for Brier / IBS (external cohorts)
survf_chus = get_survival_functions(model_full, X_chus_sc)
survf_chup = get_survival_functions(model_full, X_chup_sc)

# ============================================================
# OOF: generate dev OOF risk scores + survival functions
# ============================================================
print("\n--- Generating dev OOF risk scores (5-fold CV) ---")
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
y_event = y_dev["event"].astype(int)

oof_risk   = np.full(len(X_dev), np.nan)
# Survival functions per patient — stored as list indexed by position
oof_survf  = [None] * len(X_dev)
oof_y_true = np.empty(len(X_dev), dtype=y_dev.dtype)

for fold_i, (tr, vl) in enumerate(skf.split(X_dev, y_event)):
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_dev[tr])
    X_vl_sc = sc.transform(X_dev[vl])

    m = ExtraSurvivalTrees(n_estimators=200, random_state=SEED, n_jobs=-1)
    try:
        m.fit(X_tr_sc, y_dev[tr])
        oof_risk[vl]  = m.predict(X_vl_sc)
        sf_vl = get_survival_functions(m, X_vl_sc)
        for local_i, global_i in enumerate(vl):
            oof_survf[global_i]  = sf_vl[local_i]
            oof_y_true[global_i] = y_dev[global_i]
    except Exception as e:
        print(f"  [WARN] Fold {fold_i} failed: {e}")

valid_mask = ~np.isnan(oof_risk)
oof_risk_v  = oof_risk[valid_mask]
oof_y_v     = y_dev[valid_mask]
oof_survf_v = [oof_survf[i] for i in np.where(valid_mask)[0]]

ci_oof = safe_ci(oof_y_v, oof_risk_v)
print(f"  Dev OOF C-index: {ci_oof:.4f} (expected ~0.706)")

# ============================================================
# TIME-DEPENDENT AUC
# ============================================================
print("\n--- Time-dependent AUC (IPCW, cumulative/dynamic) ---")

# Dev OOF AUC: use oof_y_v as both train and test (censoring from same data)
auc_dev, mean_auc_dev = safe_td_auc(oof_y_v, oof_y_v, oof_risk_v, TIME_POINTS_DAYS)
print(f"  Dev OOF AUC@1yr={auc_dev[0]:.4f}, @2yr={auc_dev[1]:.4f}, @3yr={auc_dev[2]:.4f}, mean={mean_auc_dev:.4f}")

# CHUS AUC: train censoring from dev, test on CHUS
auc_chus, mean_auc_chus = safe_td_auc(y_dev, y_chus, risk_chus, TIME_POINTS_DAYS)
print(f"  CHUS AUC@1yr={auc_chus[0]:.4f}, @2yr={auc_chus[1]:.4f}, @3yr={auc_chus[2]:.4f}, mean={mean_auc_chus:.4f}")

# CHUP AUC: train censoring from dev, test on CHUP
auc_chup, mean_auc_chup = safe_td_auc(y_dev, y_chup, risk_chup, TIME_POINTS_DAYS)
print(f"  CHUP AUC@1yr={auc_chup[0]:.4f}, @2yr={auc_chup[1]:.4f}, @3yr={auc_chup[2]:.4f}, mean={mean_auc_chup:.4f}")

# ============================================================
# BRIER SCORE + IBS
# ============================================================
print("\n--- Brier Score and Integrated Brier Score ---")

bs_dev  = safe_brier(oof_y_v, oof_y_v,  oof_survf_v, TIME_POINTS_DAYS)
bs_chus = safe_brier(y_dev,   y_chus,   survf_chus,  TIME_POINTS_DAYS)
bs_chup = safe_brier(y_dev,   y_chup,   survf_chup,  TIME_POINTS_DAYS)

print(f"  Dev OOF Brier@1yr={bs_dev[0]:.4f}, @2yr={bs_dev[1]:.4f}, @3yr={bs_dev[2]:.4f}")
print(f"  CHUS Brier@1yr={bs_chus[0]:.4f}, @2yr={bs_chus[1]:.4f}, @3yr={bs_chus[2]:.4f}")
print(f"  CHUP Brier@1yr={bs_chup[0]:.4f}, @2yr={bs_chup[1]:.4f}, @3yr={bs_chup[2]:.4f}")

ibs_dev  = safe_ibs(oof_y_v, oof_y_v, oof_survf_v)
ibs_chus = safe_ibs(y_dev,   y_chus,  survf_chus)
ibs_chup = safe_ibs(y_dev,   y_chup,  survf_chup)

print(f"  Dev OOF IBS: {ibs_dev:.4f}")
print(f"  CHUS IBS:    {ibs_chus:.4f}")
print(f"  CHUP IBS:    {ibs_chup:.4f}")

# ============================================================
# SAVE RESULTS
# ============================================================
print("\n--- Saving outputs ---")

rows = []
for cohort, ci, aucs, mean_auc, bs, ibs in [
    ("Dev_OOF", ci_oof,  auc_dev,  mean_auc_dev,  bs_dev,  ibs_dev),
    ("CHUS",    ci_chus, auc_chus, mean_auc_chus, bs_chus, ibs_chus),
    ("CHUP",    ci_chup, auc_chup, mean_auc_chup, bs_chup, ibs_chup),
]:
    row = {
        "Cohort":       cohort,
        "C_index":      round(ci, 4),
        "AUC_1yr":      round(aucs[0], 4) if not np.isnan(aucs[0]) else "N/A",
        "AUC_2yr":      round(aucs[1], 4) if not np.isnan(aucs[1]) else "N/A",
        "AUC_3yr":      round(aucs[2], 4) if not np.isnan(aucs[2]) else "N/A",
        "AUC_mean":     round(mean_auc, 4) if not np.isnan(mean_auc) else "N/A",
        "Brier_1yr":    round(bs[0], 4) if not np.isnan(bs[0]) else "N/A",
        "Brier_2yr":    round(bs[1], 4) if not np.isnan(bs[1]) else "N/A",
        "Brier_3yr":    round(bs[2], 4) if not np.isnan(bs[2]) else "N/A",
        "IBS":          round(ibs, 4) if not np.isnan(ibs) else "N/A",
    }
    rows.append(row)

results_df = pd.DataFrame(rows)
results_df.to_csv(CSV_PATH, index=False)
print(f"  Saved CSV: {CSV_PATH}")

# Markdown table
md_lines = [
    "# Task 1 Post-Hoc Supplementary Metrics\n",
    "**Model:** Locked winner PT_S3_1_768xCT_S3_8_235, N=12 (1 clinical + 7 PET + 4 CT)",
    "**Coach:** ExtraSurvivalTrees(n_estimators=200, random_state=42)\n",
    "## C-index\n",
    "| Cohort | C-index |",
    "|---|---:|",
]
for r in rows:
    md_lines.append(f"| {r['Cohort']} | {r['C_index']} |")

md_lines += [
    "\n## Time-Dependent AUC (IPCW, cumulative/dynamic)\n",
    "| Cohort | AUC @1yr | AUC @2yr | AUC @3yr | Mean AUC |",
    "|---|---:|---:|---:|---:|",
]
for r in rows:
    md_lines.append(
        f"| {r['Cohort']} | {r['AUC_1yr']} | {r['AUC_2yr']} | {r['AUC_3yr']} | {r['AUC_mean']} |"
    )

md_lines += [
    "\n## Brier Score and Integrated Brier Score\n",
    "| Cohort | Brier @1yr | Brier @2yr | Brier @3yr | IBS |",
    "|---|---:|---:|---:|---:|",
]
for r in rows:
    md_lines.append(
        f"| {r['Cohort']} | {r['Brier_1yr']} | {r['Brier_2yr']} | {r['Brier_3yr']} | {r['IBS']} |"
    )

md_lines += [
    "\n## Notes",
    "- AUC/Brier N/A means the time point fell outside the observed survival range for that cohort.",
    "- Dev OOF uses the OOF risk scores and fold-held-out survival functions for both train and test censoring estimation.",
    "- CHUS and CHUP use the full dev cohort to fit the censoring distribution (IPCW).",
    "- IBS computed over a 50-point grid within observed time range per cohort.",
    "- Reproducibility gate: CHUS and CHUP C-indices verified against locked values (tol=0.001) before metric computation.",
]

MD_PATH.write_text("\n".join(md_lines), encoding="utf-8")
print(f"  Saved markdown: {MD_PATH}")

print("\n" + "=" * 70)
print("30_mar_task1_posthoc_metrics.py COMPLETE")
print(results_df.to_string(index=False))
print("=" * 70)
