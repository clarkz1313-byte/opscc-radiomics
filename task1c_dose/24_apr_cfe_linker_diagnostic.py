"""
24_apr_cfe_linker_diagnostic.py

Diagnostic: find cross-task CFE linker candidates for RPM vs DAE N75.

For every CHUS dose-eligible patient (N=44), predicts risk under:
  - RPM locked winner (PT_S3_1_768xCT_S3_8_235, N=12, no dose)
  - DAE N75 locked winner (1TC2298, N=18, 6 dose features)

Outputs a CSV ranked by "correction quality" and prints a shortlist of
candidates satisfying the cross-correction criterion:
  Event linker:    RPM under-call (risk < RPM threshold) AND DAE correct (risk >= DAE threshold)
  Non-event linker: RPM over-call  (risk >= RPM threshold) AND DAE correct (risk <  DAE threshold)

Output folder: Mar_2026_task1C/1_apr_t1C_post_study_outputs_75/branchA2/
Output file:   T1C_CFE_linker_diagnostic.csv
"""

from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import ExtraSurvivalTrees
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_ROOT = Path(__file__).resolve().parent          # Mar_2026_task1C/
MAR_ROOT    = SCRIPT_ROOT.parent / "Mar_2026"          # Mar_2026/
OUT_DIR     = SCRIPT_ROOT / "1_apr_t1C_post_study_outputs_75" / "branchA2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV     = OUT_DIR / "T1C_CFE_linker_diagnostic.csv"

CLINICAL_FILE = (
    SCRIPT_ROOT.parent / "Feb_2026"
    / "25_feb_clinical_reduced_dataset"
    / "25_feb_Processed_clinical_reduced.csv"
)

# RPM data files (full cohort, no dose requirement)
PT_DEV_FILE  = MAR_ROOT / "27_feb_PT_development.csv"
CT_DEV_FILE  = MAR_ROOT / "27_feb_CT_development.csv"
PT_EXT_FILE  = MAR_ROOT / "27_feb_PT_external.csv"
CT_EXT_FILE  = MAR_ROOT / "27_feb_CT_external.csv"

# DAE N75 data files (dose-eligible subset)
DAE_DEV_CSV  = SCRIPT_ROOT / "Dose_development_75.csv"
DAE_CHUS_CSV = SCRIPT_ROOT / "Dose_external_CHUS.csv"

TARGET_EVENT = "Relapse"
TARGET_TIME  = "RFS"
SEED         = 42
N_EST        = 200
DOSE_PREFIX  = "DOSE__"

# ---------------------------------------------------------------------------
# RPM locked winner feature set (PT_S3_1_768xCT_S3_8_235, N=12)
# ---------------------------------------------------------------------------
RPM_CLINICAL = ["Gender_Male"]
RPM_PT = [
    "GTVp_exponential_glszm_HighGrayLevelZoneEmphasis",
    "GTVn_wavelet-LLH_firstorder_Mean",
    "GTVp_wavelet-HLH_glrlm_ShortRunHighGrayLevelEmphasis",
    "GTVp_gradient_glszm_ZoneEntropy",
    "GTVp_wavelet-LHL_glszm_SmallAreaHighGrayLevelEmphasis",
    "GTVn_wavelet-LHH_glszm_LowGrayLevelZoneEmphasis",
    "GTVp_wavelet-HLH_glszm_HighGrayLevelZoneEmphasis",
]
RPM_CT = [
    "GTVp_wavelet-LLH_glrlm_HighGrayLevelRunEmphasis",
    "GTVp_wavelet-HLL_ngtdm_Complexity",
    "GTVp_gradient_glszm_SmallAreaLowGrayLevelEmphasis",
    "GTVp_wavelet-LHH_firstorder_RootMeanSquared",
]
RPM_FEATURES = RPM_CLINICAL + RPM_PT + RPM_CT

RPM_LOCKED_CHUS = 0.742857142857143
RPM_LOCKED_CHUP = 0.727586206896552
RPM_REPRO_TOL   = 0.001

# ---------------------------------------------------------------------------
# DAE N75 locked winner feature set (1TC2298, N=18)
# ---------------------------------------------------------------------------
DAE_CLINICAL = ["Gender_Male"]
DAE_PT = [
    "GTVp_exponential_glszm_HighGrayLevelZoneEmphasis",
    "GTVn_wavelet-LLH_firstorder_Mean",
    "GTVp_wavelet-HLH_glrlm_ShortRunHighGrayLevelEmphasis",
    "GTVp_gradient_glszm_ZoneEntropy",
    "GTVp_wavelet-LHL_glszm_SmallAreaHighGrayLevelEmphasis",
    "GTVn_wavelet-LHH_glszm_LowGrayLevelZoneEmphasis",
    "GTVp_wavelet-HLH_glszm_HighGrayLevelZoneEmphasis",
]
DAE_CT = [
    "GTVp_wavelet-LLH_glrlm_HighGrayLevelRunEmphasis",
    "GTVp_wavelet-HLL_ngtdm_Complexity",
    "GTVp_gradient_glszm_SmallAreaLowGrayLevelEmphasis",
    "GTVp_wavelet-LHH_firstorder_RootMeanSquared",
]
DAE_DOSE_RAW = [
    "GTVp_wavelet-HLH_firstorder_Median",
    "GTVn_wavelet-HLH_glszm_HighGrayLevelZoneEmphasis",
    "GTVn_wavelet-LHH_firstorder_Mean",
    "GTVn_gradient_firstorder_Maximum",
    "GTVn_gradient_firstorder_Range",
    "GTVp_wavelet-LLH_glrlm_GrayLevelVariance",
]
DAE_DOSE = [DOSE_PREFIX + f for f in DAE_DOSE_RAW]
DAE_FEATURES = DAE_CLINICAL + DAE_PT + DAE_CT + DAE_DOSE

DAE_LOCKED_OOF  = 0.8482
DAE_LOCKED_CHUS = 0.834921
DAE_REPRO_TOL   = 0.001

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_surv(event: pd.Series, time: pd.Series):
    return Surv.from_arrays(
        event=np.asarray(event, dtype=bool),
        time=np.asarray(time, dtype=float),
    )


def safe_ci(y, risk: np.ndarray) -> float:
    try:
        return float(concordance_index_censored(y["event"], y["time"], risk)[0])
    except Exception:
        return float("nan")


def build_est() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("est", ExtraSurvivalTrees(n_estimators=N_EST, random_state=SEED, n_jobs=1)),
    ])


def prefixed_dose_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [f for f in DAE_DOSE_RAW if f not in df.columns]
    if missing:
        raise KeyError(f"Missing dose features in {path}: {missing}")
    out = df[["PatientID"] + DAE_DOSE_RAW].copy()
    return out.rename(columns={f: DOSE_PREFIX + f for f in DAE_DOSE_RAW})

# ---------------------------------------------------------------------------
# Load RPM data
# ---------------------------------------------------------------------------
print("Loading RPM data...")
clinical_all = pd.read_csv(CLINICAL_FILE).dropna(subset=[TARGET_EVENT, TARGET_TIME])

rpm_clin_dev = clinical_all[clinical_all["Cohort"] == "Dev"][
    ["PatientID", "CenterID", TARGET_EVENT, TARGET_TIME] + RPM_CLINICAL
].copy()
rpm_clin_ext = clinical_all[clinical_all["CenterID"].isin([3, 4])][
    ["PatientID", "CenterID", TARGET_EVENT, TARGET_TIME] + RPM_CLINICAL
].copy()

rpm_pt_dev  = pd.read_csv(PT_DEV_FILE)[["PatientID"] + RPM_PT]
rpm_ct_dev  = pd.read_csv(CT_DEV_FILE)[["PatientID"] + RPM_CT]
rpm_pt_ext  = pd.read_csv(PT_EXT_FILE)
rpm_ct_ext  = pd.read_csv(CT_EXT_FILE)

rpm_pt_chus = rpm_pt_ext[rpm_pt_ext["PatientID"].str.startswith("CHUS")][["PatientID"] + RPM_PT].copy()
rpm_ct_chus = rpm_ct_ext[rpm_ct_ext["PatientID"].str.startswith("CHUS")][["PatientID"] + RPM_CT].copy()
rpm_rad_dev  = rpm_pt_dev.merge(rpm_ct_dev, on="PatientID")
rpm_rad_chus = rpm_pt_chus.merge(rpm_ct_chus, on="PatientID")

rpm_clin_chus = rpm_clin_ext[rpm_clin_ext["CenterID"] == 3].copy()

rpm_dev_df  = rpm_clin_dev.merge(rpm_rad_dev,  on="PatientID")
rpm_chus_df = rpm_clin_chus.merge(rpm_rad_chus, on="PatientID")

print(f"  RPM dev: {len(rpm_dev_df)} patients, CHUS: {len(rpm_chus_df)}")

# ---------------------------------------------------------------------------
# Load DAE N75 data
# ---------------------------------------------------------------------------
print("Loading DAE N75 data...")
dae_clin_dev = clinical_all[clinical_all["Cohort"] == "Dev"][
    ["PatientID", "CenterID", TARGET_EVENT, TARGET_TIME] + DAE_CLINICAL
].copy()
dae_clin_chus = clinical_all[clinical_all["CenterID"] == 3][
    ["PatientID", "CenterID", TARGET_EVENT, TARGET_TIME] + DAE_CLINICAL
].copy()

dose_dev  = prefixed_dose_frame(DAE_DEV_CSV)
dose_chus = prefixed_dose_frame(DAE_CHUS_CSV)

dae_pt_dev  = pd.read_csv(PT_DEV_FILE)[["PatientID"] + DAE_PT]
dae_ct_dev  = pd.read_csv(CT_DEV_FILE)[["PatientID"] + DAE_CT]
dae_pt_ext  = pd.read_csv(PT_EXT_FILE)
dae_ct_ext  = pd.read_csv(CT_EXT_FILE)
dae_pt_chus = dae_pt_ext[dae_pt_ext["PatientID"].str.startswith("CHUS")][["PatientID"] + DAE_PT].copy()
dae_ct_chus = dae_ct_ext[dae_ct_ext["PatientID"].str.startswith("CHUS")][["PatientID"] + DAE_CT].copy()

dae_rad_dev  = dae_pt_dev.merge(dae_ct_dev,  on="PatientID")
dae_rad_chus = dae_pt_chus.merge(dae_ct_chus, on="PatientID")

dae_dev_df = (
    dae_clin_dev
    .merge(dae_rad_dev,  on="PatientID")
    .merge(dose_dev,     on="PatientID")
)
dae_chus_df = (
    dae_clin_chus
    .merge(dae_rad_chus, on="PatientID")
    .merge(dose_chus,    on="PatientID")
)
print(f"  DAE dev: {len(dae_dev_df)} patients, CHUS: {len(dae_chus_df)}")

# ---------------------------------------------------------------------------
# Fit RPM pipeline and verify reproducibility
# ---------------------------------------------------------------------------
print("\nFitting RPM pipeline...")
rpm_pipeline = build_est()
y_rpm_dev = make_surv(rpm_dev_df[TARGET_EVENT], rpm_dev_df[TARGET_TIME])
rpm_pipeline.fit(rpm_dev_df[RPM_FEATURES], y_rpm_dev)

rpm_risk_dev = rpm_pipeline.predict(rpm_dev_df[RPM_FEATURES])
rpm_threshold = float(np.median(rpm_risk_dev))

y_rpm_chus = make_surv(rpm_chus_df[TARGET_EVENT], rpm_chus_df[TARGET_TIME])
rpm_risk_chus_full = rpm_pipeline.predict(rpm_chus_df[RPM_FEATURES])
rpm_ci_chus = safe_ci(y_rpm_chus, rpm_risk_chus_full)

print(f"  RPM threshold (dev median): {rpm_threshold:.6f}")
print(f"  RPM CHUS C-index: {rpm_ci_chus:.6f}  locked={RPM_LOCKED_CHUS:.6f}  delta={abs(rpm_ci_chus-RPM_LOCKED_CHUS):.6f}")
assert abs(rpm_ci_chus - RPM_LOCKED_CHUS) <= RPM_REPRO_TOL, "RPM CHUS reproducibility FAILED"
print("  RPM reproducibility: PASSED")

# ---------------------------------------------------------------------------
# Fit DAE N75 pipeline and verify reproducibility
# ---------------------------------------------------------------------------
print("\nFitting DAE N75 pipeline...")
dae_pipeline = build_est()
y_dae_dev = make_surv(dae_dev_df[TARGET_EVENT], dae_dev_df[TARGET_TIME])
dae_pipeline.fit(dae_dev_df[DAE_FEATURES], y_dae_dev)

dae_risk_dev = dae_pipeline.predict(dae_dev_df[DAE_FEATURES])
dae_threshold = float(np.median(dae_risk_dev))

y_dae_chus = make_surv(dae_chus_df[TARGET_EVENT], dae_chus_df[TARGET_TIME])
dae_risk_chus_full = dae_pipeline.predict(dae_chus_df[DAE_FEATURES])
dae_ci_chus = safe_ci(y_dae_chus, dae_risk_chus_full)

dae_oof_risk = np.zeros(len(dae_dev_df))
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for train_idx, val_idx in skf.split(dae_dev_df[DAE_FEATURES], dae_dev_df[TARGET_EVENT]):
    fold_pipe = build_est()
    fold_pipe.fit(
        dae_dev_df[DAE_FEATURES].iloc[train_idx],
        make_surv(dae_dev_df[TARGET_EVENT].iloc[train_idx], dae_dev_df[TARGET_TIME].iloc[train_idx]),
    )
    dae_oof_risk[val_idx] = fold_pipe.predict(dae_dev_df[DAE_FEATURES].iloc[val_idx])
dae_ci_oof = safe_ci(y_dae_dev, dae_oof_risk)

print(f"  DAE threshold (dev median): {dae_threshold:.6f}")
print(f"  DAE OOF C-index: {dae_ci_oof:.6f}  locked={DAE_LOCKED_OOF:.6f}  delta={abs(dae_ci_oof-DAE_LOCKED_OOF):.6f}")
print(f"  DAE CHUS C-index: {dae_ci_chus:.6f}  locked={DAE_LOCKED_CHUS:.6f}  delta={abs(dae_ci_chus-DAE_LOCKED_CHUS):.6f}")
assert abs(dae_ci_chus - DAE_LOCKED_CHUS) <= DAE_REPRO_TOL, "DAE CHUS reproducibility FAILED"
print("  DAE reproducibility: PASSED")

# ---------------------------------------------------------------------------
# Predict all CHUS dose-eligible patients under both models
# ---------------------------------------------------------------------------
print("\nPredicting CHUS dose-eligible patients under both models...")

# RPM risk for the dose-eligible CHUS subset only
chus_dose_ids = set(dae_chus_df["PatientID"].tolist())
rpm_chus_dose = rpm_chus_df[rpm_chus_df["PatientID"].isin(chus_dose_ids)].copy()
rpm_chus_dose = rpm_chus_dose.sort_values("PatientID").reset_index(drop=True)

dae_chus_sorted = dae_chus_df.sort_values("PatientID").reset_index(drop=True)

assert list(rpm_chus_dose["PatientID"]) == list(dae_chus_sorted["PatientID"]), \
    "Patient ID mismatch between RPM and DAE CHUS dose-eligible subsets"

rpm_risk_dose = rpm_pipeline.predict(rpm_chus_dose[RPM_FEATURES])
dae_risk_dose = dae_pipeline.predict(dae_chus_sorted[DAE_FEATURES])

results = pd.DataFrame({
    "PatientID":      dae_chus_sorted["PatientID"].values,
    "event":          dae_chus_sorted[TARGET_EVENT].values.astype(int),
    "rpm_risk":       np.round(rpm_risk_dose, 6),
    "dae_risk":       np.round(dae_risk_dose, 6),
    "rpm_threshold":  rpm_threshold,
    "dae_threshold":  dae_threshold,
    "rpm_class":      np.where(rpm_risk_dose >= rpm_threshold, "high", "low"),
    "dae_class":      np.where(dae_risk_dose >= dae_threshold, "high", "low"),
    "rpm_dist":       np.round(rpm_risk_dose - rpm_threshold, 6),
    "dae_dist":       np.round(dae_risk_dose - dae_threshold, 6),
})

# Classification outcomes
results["rpm_correct"] = np.where(
    results["event"] == 1,
    results["rpm_class"] == "high",
    results["rpm_class"] == "low",
)
results["dae_correct"] = np.where(
    results["event"] == 1,
    results["dae_class"] == "high",
    results["dae_class"] == "low",
)
results["rpm_correct"] = results["rpm_correct"].astype(bool)
results["dae_correct"] = results["dae_correct"].astype(bool)

# Cross-correction flag: RPM wrong, DAE right
results["cross_corrected"] = (~results["rpm_correct"]) & results["dae_correct"]

# Linker type
def linker_type(row):
    if row["event"] == 1 and row["rpm_class"] == "low" and row["dae_class"] == "high":
        return "event_linker"        # RPM under-call -> DAE correct high-risk
    if row["event"] == 0 and row["rpm_class"] == "high" and row["dae_class"] == "low":
        return "nonevent_linker"     # RPM over-call  -> DAE correct low-risk
    if row["event"] == 1 and row["rpm_class"] == "low" and row["dae_class"] == "low":
        return "event_both_undercall"
    if row["event"] == 1 and row["rpm_class"] == "high" and row["dae_class"] == "high":
        return "event_both_correct"
    if row["event"] == 0 and row["rpm_class"] == "high" and row["dae_class"] == "high":
        return "nonevent_both_overcall"
    if row["event"] == 0 and row["rpm_class"] == "low" and row["dae_class"] == "low":
        return "nonevent_both_correct"
    return "other"

results["linker_type"] = results.apply(linker_type, axis=1)

# Boundary proximity score: sum of abs distances from both thresholds (lower = better linker)
results["boundary_proximity_score"] = (
    results["rpm_dist"].abs() + results["dae_dist"].abs()
)

results = results.sort_values(
    ["cross_corrected", "boundary_proximity_score"],
    ascending=[False, True],
).reset_index(drop=True)

# ---------------------------------------------------------------------------
# Save and print
# ---------------------------------------------------------------------------
results.to_csv(OUT_CSV, index=False)
print(f"\nFull results saved to: {OUT_CSV}")

print("\n" + "="*70)
print("CROSS-CORRECTED PATIENTS (RPM wrong -> DAE right):")
print("="*70)
cross = results[results["cross_corrected"]]
if cross.empty:
    print("  None found.")
else:
    print(cross[[
        "PatientID","event","rpm_risk","dae_risk",
        "rpm_class","dae_class","linker_type","boundary_proximity_score"
    ]].to_string(index=False))

print("\n" + "="*70)
print("ALL CHUS DOSE-ELIGIBLE PATIENTS (sorted by cross-correction then proximity):")
print("="*70)
print(results[[
    "PatientID","event","rpm_risk","dae_risk",
    "rpm_class","dae_class","rpm_correct","dae_correct",
    "linker_type","boundary_proximity_score"
]].to_string(index=False))

print("\n" + "="*70)
print("SUMMARY:")
print("="*70)
print(f"  Total CHUS dose-eligible: {len(results)}")
print(f"  Events: {results['event'].sum()}  |  Non-events: {(results['event']==0).sum()}")
print(f"  RPM threshold: {rpm_threshold:.6f}")
print(f"  DAE threshold: {dae_threshold:.6f}")
print(f"  Cross-corrected (RPM wrong -> DAE right): {results['cross_corrected'].sum()}")
print(f"    - Event linkers   (RPM under-call -> DAE correct): {(results['linker_type']=='event_linker').sum()}")
print(f"    - Non-event linkers (RPM over-call -> DAE correct): {(results['linker_type']=='nonevent_linker').sum()}")

print(f"\nOutput: {OUT_CSV}")
