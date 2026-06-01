"""
1_apr_t1C_branchA2_CFE_75.py

T1C / DAE N75 Post-Study - Branch A2: Counterfactual Explanations (DiCE)
Locked winner: 1TC2298, n=18 (1 clinical + 7 PT + 4 CT + 6 dose)
Coach: ExtraSurvivalTrees(n_estimators=200, random_state=42)
Cohorts: Dev (75 pts, 13 events) + CHUS (44 pts, 8 events). No CHUP dose data.

Outputs:
  Mar_2026_task1C/1_apr_t1C_post_study_outputs_75/branchA2/
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import os
import warnings

SCRIPT_ROOT = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_ROOT / "1_apr_t1C_post_study_outputs_75" / "branchA2"
MPL_DIR = OUT_DIR / ".mplconfig"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_DIR)
warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import ExtraSurvivalTrees
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv


T1_ROOT = SCRIPT_ROOT.parent / "Mar_2026"
TARGET_EVENT = "Relapse"
TARGET_TIME = "RFS"

DEV_CSV = SCRIPT_ROOT / "Dose_development_75.csv"
CHUS_CSV = SCRIPT_ROOT / "Dose_external_CHUS.csv"
PT_DEV_FILE = T1_ROOT / "27_feb_PT_development.csv"
CT_DEV_FILE = T1_ROOT / "27_feb_CT_development.csv"
PT_EXT_FILE = T1_ROOT / "27_feb_PT_external.csv"
CT_EXT_FILE = T1_ROOT / "27_feb_CT_external.csv"
CLINICAL_FILE = (
    SCRIPT_ROOT.parent / "Feb_2026" / "25_feb_clinical_reduced_dataset"
    / "25_feb_Processed_clinical_reduced.csv"
)

SELECTED_PATIENTS_CSV = OUT_DIR / "T1C_CFE_selected_borderline_patients.csv"
SHIFT_TABLE_CSV = OUT_DIR / "T1C_CFE_counterfactual_feature_shifts_long.csv"
SHIFT_SUMMARY_CSV = OUT_DIR / "T1C_CFE_counterfactual_feature_shift_summary.csv"
RISK_TRANSITIONS_CSV = OUT_DIR / "T1C_CFE_counterfactual_risk_transitions.csv"
FEATURE_SHIFT_BAR_PNG = OUT_DIR / "T1C_CFE_feature_shift_summary_bar.png"
RISK_TRANSITIONS_PNG = OUT_DIR / "T1C_CFE_counterfactual_risk_transitions.png"
SHIFT_HEATMAP_PNG = OUT_DIR / "T1C_CFE_counterfactual_shift_heatmap.png"
SIGNED_SHIFTS_PNG = OUT_DIR / "T1C_CFE_signed_shifts_per_case.png"
FEATURE_SHIFT_MATCHED_PNG = OUT_DIR / "T1C_CFE_feature_shift_summary_matched.png"
FEATURE_GROUP_MATCHED_PNG = OUT_DIR / "T1C_CFE_feature_group_shift_summary_matched.png"
FEATURE_GROUP_MATCHED_CSV = OUT_DIR / "T1C_CFE_feature_group_shift_summary_matched.csv"
SHIFT_SUMMARY_MATCHED_CSV = OUT_DIR / "T1C_CFE_feature_shift_summary_matched.csv"
SELECTED_MATCHED_CSV = OUT_DIR / "T1C_CFE_selected_matched_patients.csv"
RISK_TRANSITIONS_MATCHED_CSV = OUT_DIR / "T1C_CFE_risk_transitions_matched.csv"
SHIFT_TABLE_MATCHED_CSV = OUT_DIR / "T1C_CFE_counterfactual_feature_shifts_matched_long.csv"

SEED = 42
N_EST = 200
TOTAL_CFS = 2
LOCKED_OOF = 0.8482
LOCKED_CHUS = 0.834921
REPRO_TOL = 0.001
WINNER_ID = "1TC2298"
MATCHED_PATIENT_IDS = ["CHUS-065"]
# Same-direction matched example:
# CHUS-065 is a non-event overcalled high-risk by both RPM and DAE, so both
# models cross downward in the CFE risk-transition view.
TRIAL_NO = 2298
DOSE_PREFIX = "DOSE__"

CLINICAL_FEATURES = ["Gender_Male"]
PT_LOCKED = [
    "GTVp_exponential_glszm_HighGrayLevelZoneEmphasis",
    "GTVn_wavelet-LLH_firstorder_Mean",
    "GTVp_wavelet-HLH_glrlm_ShortRunHighGrayLevelEmphasis",
    "GTVp_gradient_glszm_ZoneEntropy",
    "GTVp_wavelet-LHL_glszm_SmallAreaHighGrayLevelEmphasis",
    "GTVn_wavelet-LHH_glszm_LowGrayLevelZoneEmphasis",
    "GTVp_wavelet-HLH_glszm_HighGrayLevelZoneEmphasis",
]
CT_LOCKED = [
    "GTVp_wavelet-LLH_glrlm_HighGrayLevelRunEmphasis",
    "GTVp_wavelet-HLL_ngtdm_Complexity",
    "GTVp_gradient_glszm_SmallAreaLowGrayLevelEmphasis",
    "GTVp_wavelet-LHH_firstorder_RootMeanSquared",
]
DOSE_WINNER_6 = [
    "GTVp_wavelet-HLH_firstorder_Median",
    "GTVn_wavelet-HLH_glszm_HighGrayLevelZoneEmphasis",
    "GTVn_wavelet-LHH_firstorder_Mean",
    "GTVn_gradient_firstorder_Maximum",
    "GTVn_gradient_firstorder_Range",
    "GTVp_wavelet-LLH_glrlm_GrayLevelVariance",
]
DOSE_WINNER_6_PREFIXED = [DOSE_PREFIX + f for f in DOSE_WINNER_6]
FEATURES = CLINICAL_FEATURES + PT_LOCKED + CT_LOCKED + DOSE_WINNER_6_PREFIXED
MUTABLE_FEATURES = PT_LOCKED + CT_LOCKED + DOSE_WINNER_6_PREFIXED

plt.rcParams.update(
    {
        "figure.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# SHAP risk-association directions from Table 3.3.4 (v15 outline).
# +1 = higher feature value -> higher predicted risk (positive SHAP direction)
# -1 = higher feature value -> lower predicted risk (negative SHAP direction)
#  0 = weak/mixed (D2 only)
SHAP_DIRECTION: dict[str, int] = {
    # Dose features (DOSE__ prefix in data columns)
    "DOSE__GTVn_gradient_firstorder_Range": +1,  # D5
    "DOSE__GTVn_gradient_firstorder_Maximum": +1,  # D4
    "DOSE__GTVp_wavelet-HLH_firstorder_Median": +1,  # D1
    "DOSE__GTVn_wavelet-LHH_firstorder_Mean": +1,  # D3
    "DOSE__GTVp_wavelet-LLH_glrlm_GrayLevelVariance": -1,  # D6
    "DOSE__GTVn_wavelet-HLH_glszm_HighGrayLevelZoneEmphasis": 0,  # D2 weak/mixed
    # PT features (no prefix)
    "GTVp_gradient_glszm_ZoneEntropy": +1,  # P1
    "GTVn_wavelet-LLH_firstorder_Mean": -1,  # P4 (nodal mean)
    "GTVp_wavelet-HLH_glrlm_ShortRunHighGrayLevelEmphasis": -1,  # P3
    "GTVp_wavelet-LHL_glszm_SmallAreaHighGrayLevelEmphasis": -1,  # P5 (note: DAE direction)
    "GTVn_wavelet-LHH_glszm_LowGrayLevelZoneEmphasis": +1,  # P7 (DAE flipped vs RPM)
    "GTVp_wavelet-HLH_glszm_HighGrayLevelZoneEmphasis": -1,  # P6 (nodal mean alt)
    "GTVp_exponential_glszm_HighGrayLevelZoneEmphasis": -1,  # P2
    # CT features (no prefix)
    "GTVp_wavelet-LHH_firstorder_RootMeanSquared": -1,  # T4
    "GTVp_wavelet-HLL_ngtdm_Complexity": +1,  # T2
    "GTVp_gradient_glszm_SmallAreaLowGrayLevelEmphasis": -1,  # T3 (DAE flipped vs RPM)
    "GTVp_wavelet-LLH_glrlm_HighGrayLevelRunEmphasis": +1,  # T1 (DAE flipped vs RPM)
    # Clinical
    "Gender_Male": +1,  # C1
}

# Short human-readable labels for CFE bar panels.
# Dose features carry the DOSE__ prefix in the data; strip it in display.
_CFE_LABEL: dict[str, str] = {
    # Dose features
    "DOSE__GTVn_gradient_firstorder_Range": "GTVn Grad Range (D5)",
    "DOSE__GTVn_gradient_firstorder_Maximum": "GTVn Grad Max (D4)",
    "DOSE__GTVp_wavelet-HLH_firstorder_Median": "GTVp HLH Median (D1)",
    "DOSE__GTVn_wavelet-LHH_firstorder_Mean": "GTVn LHH Mean (D3)",
    "DOSE__GTVp_wavelet-LLH_glrlm_GrayLevelVariance": "GTVp LLH GLVar (D6)",
    "DOSE__GTVn_wavelet-HLH_glszm_HighGrayLevelZoneEmphasis": "GTVn HLH HGLZE (D2)",
    # PT features
    "GTVp_gradient_glszm_ZoneEntropy": "GTVp ZoneEntropy (P1)",
    "GTVn_wavelet-LLH_firstorder_Mean": "GTVn LLH Mean (P4)",
    "GTVp_wavelet-HLH_glrlm_ShortRunHighGrayLevelEmphasis": "GTVp HLH SRHGLE (P3)",
    "GTVp_wavelet-LHL_glszm_SmallAreaHighGrayLevelEmphasis": "GTVp LHL SAHGLE (P5)",
    "GTVn_wavelet-LHH_glszm_LowGrayLevelZoneEmphasis": "GTVn LHH LGLZE (P7)",
    "GTVp_wavelet-HLH_glszm_HighGrayLevelZoneEmphasis": "GTVp HLH HGLZE (P6)",
    "GTVp_exponential_glszm_HighGrayLevelZoneEmphasis": "GTVp Exp HGZE (P2)",
    # CT features
    "GTVp_wavelet-LHH_firstorder_RootMeanSquared": "GTVp LHH RMS (T4)",
    "GTVp_wavelet-HLL_ngtdm_Complexity": "GTVp HLL Cmplxty (T2)",
    "GTVp_gradient_glszm_SmallAreaLowGrayLevelEmphasis": "GTVp SALLGLE (T3)",
    "GTVp_wavelet-LLH_glrlm_HighGrayLevelRunEmphasis": "GTVp LLH HGRE (T1)",
    # Clinical
    "Gender_Male": "Gender Male (C1)",
}

_MODALITY_COLOR: dict[str, str] = {
    "PT": "#f58518",
    "CT": "#4c78a8",
    "Dose": "#7b61a8",
    "Clinical": "#777777",
    "Unknown": "#aaaaaa",
}


def _cfe_bar_label(feat: str) -> str:
    return _CFE_LABEL.get(feat, feat.split("__")[-1] if "__" in feat else feat)


def _cfe_bar_color(feat: str) -> str:
    return _MODALITY_COLOR.get(_feature_group(feat), "#aaaaaa")


def _feature_group(feat: str) -> str:
    """Return modality group for a feature name (handles DOSE__ prefix)."""
    if feat == "Gender_Male":
        return "Clinical"
    if feat.startswith(DOSE_PREFIX):
        return "Dose"
    if feat in PT_LOCKED:
        return "PT"
    if feat in CT_LOCKED:
        return "CT"
    return "Unknown"


@dataclass
class SelectedPatient:
    label: str
    patient_id: str
    event: int
    risk_score: float
    boundary_distance: float


def ensure_dice():
    try:
        import dice_ml  # type: ignore
    except ImportError as exc:
        raise ImportError("dice_ml not installed. Run: pip install dice-ml") from exc
    return dice_ml


def make_surv(event: pd.Series, time: pd.Series):
    return Surv.from_arrays(event=np.asarray(event, dtype=bool), time=np.asarray(time, dtype=float))


def safe_ci(y, risk: np.ndarray) -> float:
    try:
        return float(concordance_index_censored(y["event"], y["time"], risk)[0])
    except Exception:
        return float("nan")


def feature_label(feature: str) -> str:
    clean = feature.replace(DOSE_PREFIX, "")
    if clean == "Gender_Male":
        return clean
    parts = clean.split("_")
    if len(parts) >= 4 and parts[0].startswith("GTV"):
        modality = feature_group(feature)
        region = parts[0]
        transform = parts[1].replace("wavelet-", "wav-")
        metric = parts[-1]
        return f"{modality} {region} {transform} {metric}"
    return "_".join(parts[-2:]) if len(parts) >= 2 else clean


def feature_group(feature: str) -> str:
    if feature in CLINICAL_FEATURES:
        return "Clinical"
    if feature in PT_LOCKED:
        return "PT"
    if feature in CT_LOCKED:
        return "CT"
    if feature.startswith(DOSE_PREFIX):
        return "Dose"
    return "Unknown"


def patient_route_label(patient_id: str, label: str, cf_index: int) -> str:
    if label.startswith("event_") or label.startswith("matched_event_"):
        short = "ER"
    elif label.startswith("nonevent_") or label.startswith("matched_nonevent_"):
        short = "NER"
    else:
        short = "BD"
    route = chr(ord("A") + cf_index - 1)
    return f"{patient_id}\n{short} route {route}"


def patient_group_label(patient_id: str, label: str) -> str:
    if label.startswith("event_"):
        tag = "Event under-called"
    elif label.startswith("nonevent_"):
        tag = "Non-event over-called"
    elif label.startswith("matched_nonevent_"):
        tag = "Matched non-event over-called"
    elif label.startswith("matched_event_"):
        tag = "Matched event under-called"
    elif label.startswith("matched_"):
        tag = "Matched comparator"
    else:
        tag = "Boundary case"
    return f"{patient_id}\n{tag}"


def prefixed_dose_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [feature for feature in DOSE_WINNER_6 if feature not in df.columns]
    if missing:
        raise KeyError(f"Missing locked dose features in {path}: {missing}")
    out = df[["PatientID"] + DOSE_WINNER_6].copy()
    return out.rename(columns={feature: DOSE_PREFIX + feature for feature in DOSE_WINNER_6})


def load_t1c_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    clinical = pd.read_csv(CLINICAL_FILE).dropna(subset=[TARGET_EVENT, TARGET_TIME])
    clin_dev = clinical[clinical["Cohort"] == "Dev"][
        ["PatientID", "CenterID", TARGET_EVENT, TARGET_TIME] + CLINICAL_FEATURES
    ].copy()
    clin_chus = clinical[clinical["CenterID"] == 3][
        ["PatientID", "CenterID", TARGET_EVENT, TARGET_TIME] + CLINICAL_FEATURES
    ].copy()

    dose_dev = prefixed_dose_frame(DEV_CSV)
    dose_chus = prefixed_dose_frame(CHUS_CSV)

    pt_dev = pd.read_csv(PT_DEV_FILE)[["PatientID"] + PT_LOCKED]
    ct_dev = pd.read_csv(CT_DEV_FILE)[["PatientID"] + CT_LOCKED]
    pt_ext = pd.read_csv(PT_EXT_FILE)
    ct_ext = pd.read_csv(CT_EXT_FILE)
    pt_chus = pt_ext[pt_ext["PatientID"].str.startswith("CHUS")][["PatientID"] + PT_LOCKED].copy()
    ct_chus = ct_ext[ct_ext["PatientID"].str.startswith("CHUS")][["PatientID"] + CT_LOCKED].copy()

    rad_dev = pt_dev.merge(ct_dev, on="PatientID", how="inner")
    rad_chus = pt_chus.merge(ct_chus, on="PatientID", how="inner")

    dev_df = (
        clin_dev.merge(rad_dev, on="PatientID", how="inner")
        .merge(dose_dev, on="PatientID", how="inner")
    )
    chus_df = (
        clin_chus.merge(rad_chus, on="PatientID", how="inner")
        .merge(dose_chus, on="PatientID", how="inner")
    )
    return dev_df, chus_df


def build_est(n_jobs: int = 1) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("est", ExtraSurvivalTrees(n_estimators=N_EST, random_state=SEED, n_jobs=n_jobs)),
        ]
    )


def fit_locked_pipeline(dev_df: pd.DataFrame) -> tuple[Pipeline, np.ndarray, float]:
    x_dev = dev_df[FEATURES]
    y_dev = make_surv(dev_df[TARGET_EVENT], dev_df[TARGET_TIME])
    pipeline = build_est(n_jobs=1)
    pipeline.fit(x_dev, y_dev)
    risk_dev = pipeline.predict(x_dev)
    threshold_dev = float(np.median(risk_dev))
    return pipeline, risk_dev, threshold_dev


def select_borderline_cases(ext_df: pd.DataFrame, pipeline: Pipeline, threshold: float) -> list[SelectedPatient]:
    eval_df = ext_df.copy()
    eval_df["risk_score"] = pipeline.predict(eval_df[FEATURES])
    eval_df["boundary_distance"] = (eval_df["risk_score"] - threshold).abs()
    selected: list[SelectedPatient] = []

    event_lowrisk = eval_df[(eval_df[TARGET_EVENT] == 1) & (eval_df["risk_score"] < threshold)].copy()
    if not event_lowrisk.empty:
        row = event_lowrisk.sort_values("risk_score", ascending=False).iloc[0]
        selected.append(
            SelectedPatient(
                label="event_low_risk_near_boundary",
                patient_id=str(row["PatientID"]),
                event=int(row[TARGET_EVENT]),
                risk_score=float(row["risk_score"]),
                boundary_distance=float(row["boundary_distance"]),
            )
        )

    nonevent_highrisk = eval_df[(eval_df[TARGET_EVENT] == 0) & (eval_df["risk_score"] >= threshold)].copy()
    if not nonevent_highrisk.empty:
        row = nonevent_highrisk.sort_values("risk_score", ascending=True).iloc[0]
        selected.append(
            SelectedPatient(
                label="nonevent_high_risk_near_boundary",
                patient_id=str(row["PatientID"]),
                event=int(row[TARGET_EVENT]),
                risk_score=float(row["risk_score"]),
                boundary_distance=float(row["boundary_distance"]),
            )
        )

    if len(selected) < 2:
        fallback = (
            eval_df.sort_values("boundary_distance", ascending=True)
            .loc[lambda d: ~d["PatientID"].isin([s.patient_id for s in selected])]
            .head(2 - len(selected))
        )
        for i, (_, row) in enumerate(fallback.iterrows(), start=1):
            selected.append(
                SelectedPatient(
                    label=f"boundary_fallback_{i}",
                    patient_id=str(row["PatientID"]),
                    event=int(row[TARGET_EVENT]),
                    risk_score=float(row["risk_score"]),
                    boundary_distance=float(row["boundary_distance"]),
                )
            )
    return selected


def select_matched_cases(ext_df: pd.DataFrame, pipeline: Pipeline, threshold: float) -> list[SelectedPatient]:
    """Select the hardcoded cross-task linker patients and predict their DAE N75 risk."""
    eval_df = ext_df.copy()
    eval_df["risk_score"] = pipeline.predict(eval_df[FEATURES])
    eval_df["boundary_distance"] = (eval_df["risk_score"] - threshold).abs()
    selected: list[SelectedPatient] = []
    for pid in MATCHED_PATIENT_IDS:
        rows = eval_df[eval_df["PatientID"] == pid]
        if rows.empty:
            print(f"  WARNING: matched patient {pid} not found in CHUS dose-eligible subset")
            continue
        row = rows.iloc[0]
        side = "low_risk" if float(row["risk_score"]) < threshold else "high_risk"
        event_label = "event" if int(row[TARGET_EVENT]) == 1 else "nonevent"
        selected.append(
            SelectedPatient(
                label=f"matched_{event_label}_{side}",
                patient_id=pid,
                event=int(row[TARGET_EVENT]),
                risk_score=float(row["risk_score"]),
                boundary_distance=float(row["boundary_distance"]),
            )
        )
    return selected


def add_standardized_deltas(shift_df: pd.DataFrame, feature_scales: pd.Series) -> pd.DataFrame:
    out = shift_df.copy()
    if out.empty:
        out["feature_group"] = pd.Series(dtype=str)
        out["feature_scale_train"] = pd.Series(dtype=float)
        out["standardized_delta"] = pd.Series(dtype=float)
        out["abs_standardized_delta"] = pd.Series(dtype=float)
        return out
    out["feature_group"] = out["feature"].map(feature_group)
    out["feature_scale_train"] = out["feature"].map(feature_scales)
    out["standardized_delta"] = out["delta"] / out["feature_scale_train"]
    out["abs_standardized_delta"] = out["standardized_delta"].abs()
    return out


def summarise_standardized_shifts(shift_df: pd.DataFrame) -> pd.DataFrame:
    if shift_df.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "feature_group",
                "n_changes",
                "patients_changed",
                "mean_abs_delta",
                "median_abs_delta",
                "mean_delta",
                "mean_abs_standardized_delta",
                "median_abs_standardized_delta",
                "mean_standardized_delta",
            ]
        )
    grouped = (
        shift_df.groupby("feature", as_index=False)
        .agg(
            n_changes=("feature", "size"),
            patients_changed=("PatientID", "nunique"),
            mean_abs_delta=("delta", lambda s: s.abs().mean()),
            median_abs_delta=("delta", lambda s: s.abs().median()),
            mean_delta=("delta", "mean"),
            mean_abs_standardized_delta=("abs_standardized_delta", "mean"),
            median_abs_standardized_delta=("abs_standardized_delta", "median"),
            mean_standardized_delta=("standardized_delta", "mean"),
        )
        .sort_values(["n_changes", "mean_abs_standardized_delta"], ascending=[False, False])
        .reset_index(drop=True)
    )
    grouped["feature_group"] = grouped["feature"].map(feature_group)
    return grouped[
        [
            "feature",
            "feature_group",
            "n_changes",
            "patients_changed",
            "mean_abs_delta",
            "median_abs_delta",
            "mean_delta",
            "mean_abs_standardized_delta",
            "median_abs_standardized_delta",
            "mean_standardized_delta",
        ]
    ]


def generate_counterfactuals(
    dice_ml,
    dev_df: pd.DataFrame,
    chus_df: pd.DataFrame,
    pipeline: Pipeline,
    selected: Iterable[SelectedPatient],
    threshold: float,
    train_risk: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_for_dice = dev_df[FEATURES].copy()
    train_for_dice["risk_score"] = train_risk
    dice_data = dice_ml.Data(
        dataframe=train_for_dice,
        continuous_features=MUTABLE_FEATURES,
        categorical_features=CLINICAL_FEATURES,
        outcome_name="risk_score",
    )
    dice_model = dice_ml.Model(model=pipeline, backend="sklearn", model_type="regressor")
    explainer = dice_ml.Dice(dice_data, dice_model, method="random")

    risk_std = float(np.std(train_risk, ddof=0))
    margin = max(1e-4, 0.02 * risk_std)
    min_risk = float(np.min(train_risk))
    max_risk = float(np.max(train_risk))
    selected_rows: list[dict[str, object]] = []
    shifts: list[dict[str, object]] = []
    transitions: list[dict[str, object]] = []

    for patient in selected:
        row_df = chus_df.loc[chus_df["PatientID"] == patient.patient_id, ["PatientID", TARGET_EVENT] + FEATURES].copy()
        if row_df.empty:
            continue
        row = row_df.iloc[0]
        q_features = row_df[FEATURES].reset_index(drop=True)
        if patient.risk_score >= threshold:
            desired = [min_risk, threshold - margin]
            target_side = "below_threshold"
        else:
            desired = [threshold + margin, max_risk]
            target_side = "above_threshold"
        if desired[0] >= desired[1]:
            desired = [min_risk, max_risk]

        cf = explainer.generate_counterfactuals(
            query_instances=q_features,
            total_CFs=TOTAL_CFS,
            desired_range=desired,
            features_to_vary=MUTABLE_FEATURES,
            random_seed=SEED,
        )
        cf_rows = cf.cf_examples_list[0].final_cfs_df.reset_index(drop=True)
        selected_rows.append(
            {
                "label": patient.label,
                "PatientID": patient.patient_id,
                "ExternalCenter": "CHUS",
                "event": patient.event,
                "risk_score": round(patient.risk_score, 6),
                "boundary_distance": round(patient.boundary_distance, 6),
                "threshold_dev": round(threshold, 6),
                "target_side": target_side,
            }
        )

        for cf_idx, cf_row in cf_rows.iterrows():
            cf_features = cf_row[FEATURES].to_frame().T
            cf_risk = float(pipeline.predict(cf_features)[0])
            transitions.append(
                {
                    "label": patient.label,
                    "PatientID": patient.patient_id,
                    "ExternalCenter": "CHUS",
                    "cf_index": cf_idx + 1,
                    "baseline_risk_score": round(patient.risk_score, 6),
                    "counterfactual_risk_score": round(cf_risk, 6),
                    "risk_delta": round(cf_risk - patient.risk_score, 6),
                    "threshold_dev": round(threshold, 6),
                }
            )
            for feature in FEATURES:
                baseline = row[feature]
                current = cf_row[feature]
                if feature == "Gender_Male":
                    changed = bool(baseline != current)
                    delta = None
                else:
                    changed = abs(float(current) - float(baseline)) > 1e-12
                    delta = float(current) - float(baseline)
                if changed:
                    shifts.append(
                        {
                            "label": patient.label,
                            "PatientID": patient.patient_id,
                            "ExternalCenter": "CHUS",
                            "event": int(row[TARGET_EVENT]),
                            "cf_index": cf_idx + 1,
                            "feature": feature,
                            "baseline_value": baseline,
                            "counterfactual_value": current,
                            "delta": delta,
                            "baseline_risk_score": round(patient.risk_score, 6),
                            "counterfactual_risk_score": round(cf_risk, 6),
                        }
                    )

    return pd.DataFrame(selected_rows), pd.DataFrame(shifts), pd.DataFrame(transitions)


def plot_feature_shift_summary(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        return
    color_map = {"PT": "#2a9d8f", "CT": "#d55e00", "Dose": "#7b61a8"}
    plot_df = summary_df.sort_values("mean_abs_standardized_delta", ascending=True)
    vals = plot_df["mean_abs_standardized_delta"].to_numpy()
    labels = [feature_label(feature) for feature in plot_df["feature"]]
    colors = [color_map.get(group, "#5f5f5f") for group in plot_df["feature_group"]]
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    ax.barh(labels, vals, color=colors, edgecolor="none", height=0.66)
    ax.set_xlabel("Mean absolute standardized shift")
    ax.set_ylabel("Feature")
    ax.set_title(f"T1C N75 CFE Feature-Shift Summary - {WINNER_ID}")
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.35)
    fig.tight_layout()
    fig.savefig(FEATURE_SHIFT_BAR_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_signed_shifts_per_case(
    shift_df: pd.DataFrame,
    out_path: Path,
    title: str,
    top_n: int = 5,
) -> None:
    """Per-case per-route signed feature-shift bar chart.

    Bar colour = modality (PT orange / CT blue / Dose purple / Clinical grey).
    Magenta bar edge = shift direction is SHAP-consistent (moves feature
      in the direction that should help cross the threshold per SHAP).
    No edge = indirect / suppressor path.
    Panel title shows patient, route, and risk before -> after.
    """
    if shift_df.empty:
        return

    cases = (
        shift_df[["PatientID", "label"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    n_cases = len(cases)
    n_routes = int(shift_df["cf_index"].max()) if not shift_df.empty else 2

    fig, axes = plt.subplots(
        n_cases, n_routes,
        figsize=(5.2 * n_routes, 3.8 * n_cases),
        squeeze=False,
    )
    fig.subplots_adjust(hspace=0.55, wspace=0.45)

    for row_idx, (_, case_row) in enumerate(cases.iterrows()):
        pid = case_row["PatientID"]
        lbl = case_row["label"]
        needs_risk_up = "low_risk" in lbl or "under" in lbl
        panel_char = chr(ord("A") + row_idx)

        for col_idx in range(1, n_routes + 1):
            ax = axes[row_idx][col_idx - 1]
            sub = shift_df[
                (shift_df["PatientID"] == pid) & (shift_df["cf_index"] == col_idx)
            ].copy()
            if sub.empty:
                ax.set_visible(False)
                continue

            sub = sub.nlargest(top_n, "abs_standardized_delta")
            sub = sub.sort_values("abs_standardized_delta", ascending=True)

            vals = sub["standardized_delta"].fillna(0.0).to_numpy(dtype=float)
            feat_names = sub["feature"].tolist()
            ylabels = [_cfe_bar_label(f) for f in feat_names]

            bar_colors, edge_colors, edge_widths = [], [], []
            for feat, v in zip(feat_names, vals):
                bar_colors.append(_cfe_bar_color(feat))
                shap_dir = SHAP_DIRECTION.get(feat, 0)
                if shap_dir == 0 or np.isnan(v):
                    is_consistent = False
                elif needs_risk_up:
                    is_consistent = (v > 0 and shap_dir == +1) or (v < 0 and shap_dir == -1)
                else:
                    is_consistent = (v < 0 and shap_dir == +1) or (v > 0 and shap_dir == -1)
                edge_colors.append("#cc00cc" if is_consistent else "none")
                edge_widths.append(2.25 if is_consistent else 0.0)

            ax.barh(
                ylabels, vals,
                color=bar_colors,
                edgecolor=edge_colors,
                linewidth=edge_widths,
                height=0.55,
            )
            ax.axvline(0, color="black", linewidth=0.9)

            risk_str = ""
            if "baseline_risk_score" in sub.columns:
                risk_b = sub["baseline_risk_score"].iloc[0]
                risk_c = sub["counterfactual_risk_score"].iloc[0]
                arrow = "\u2191" if risk_c > risk_b else "\u2193"
                risk_str = f"Risk {risk_b:.3f} {arrow} {risk_c:.3f}"
            route_char = chr(ord("A") + col_idx - 1)
            ax.set_title(
                f"Panel {panel_char}  \u2022  {pid}  |  Route {route_char}\n{risk_str}",
                fontsize=8.5, pad=5,
            )
            ax.set_xlabel("Standardised shift (SD units)\n+\u202ffeature increased   \u2212\u202ffeature decreased",
                          fontsize=7.5)
            ax.tick_params(axis="y", labelsize=7.5)
            ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.4)
            ax.spines["left"].set_visible(False)

            xmax = float(np.nanmax(np.abs(vals))) if len(vals) else 1.0
            xmax = xmax if xmax > 0 else 1.0
            ax.set_xlim(-xmax * 1.6, xmax * 1.6)
            for y_pos, v in enumerate(vals):
                sym = "\u2191" if v > 0 else "\u2193"
                ha = "left" if v >= 0 else "right"
                offset = xmax * 0.09 if v >= 0 else -xmax * 0.09
                ax.text(v + offset, y_pos, f"{sym}{abs(v):.2f}",
                        va="center", ha=ha, fontsize=7.0)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#f58518", label="PET feature"),
        Patch(facecolor="#4c78a8", label="CT feature"),
        Patch(facecolor="#7b61a8", label="Dose feature"),
        Patch(facecolor="white", edgecolor="#cc00cc", linewidth=2.25,
              label="SHAP-consistent (magenta edge)"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center", ncol=4,
        frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(title, fontsize=10, y=1.01)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_risk_transitions(transitions_df: pd.DataFrame, threshold: float) -> None:
    if transitions_df.empty:
        return
    order = transitions_df[["label", "PatientID"]].drop_duplicates().reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for x_idx, row in order.iterrows():
        rows = transitions_df.loc[transitions_df["PatientID"] == row["PatientID"]].sort_values("cf_index")
        baseline = rows["baseline_risk_score"].iloc[0]
        ax.scatter([x_idx], [baseline], color="black", s=62, label="Observed" if x_idx == 0 else None, zorder=4)
        for _, cf in rows.iterrows():
            ax.scatter(
                [x_idx],
                [cf["counterfactual_risk_score"]],
                color="#c75b39",
                s=62,
                label="Counterfactual route" if x_idx == 0 and int(cf["cf_index"]) == 1 else None,
                zorder=4,
            )
            ax.plot([x_idx, x_idx], [baseline, cf["counterfactual_risk_score"]], color="#8f8f8f", linewidth=1.8)
    ax.axhline(threshold, color="#1f77b4", linestyle="--", linewidth=1.5, label="Dev risk threshold")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([patient_group_label(r["PatientID"], r["label"]) for _, r in order.iterrows()])
    ax.set_ylabel("Predicted survival risk score")
    ax.set_title("T1C N75 CFE risk transitions: native boundary cases plus CHUS-065")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(RISK_TRANSITIONS_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_shift_heatmap(shift_df: pd.DataFrame) -> None:
    if shift_df.empty:
        return
    hm = shift_df.copy()
    hm["patient_cf"] = hm.apply(
        lambda row: patient_route_label(str(row["PatientID"]), str(row["label"]), int(row["cf_index"])),
        axis=1,
    )
    hm["feature_short"] = hm["feature"].map(feature_label)
    pivot = (
        hm.pivot_table(
            index="feature_short",
            columns="patient_cf",
            values="standardized_delta",
            aggfunc="mean",
            fill_value=0.0,
        )
        .reindex(index=hm.groupby("feature_short")["abs_standardized_delta"].mean().sort_values(ascending=False).index)
    )
    fig_width = max(8.4, 1.55 * pivot.shape[1] + 2.4)
    fig_height = max(4.8, 0.64 * pivot.shape[0] + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    vmax = max(float(pivot.abs().to_numpy().max()), 1.0)
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=0)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Patient and alternative counterfactual route")
    ax.set_ylabel("Feature")
    ax.set_title(f"T1C N75 CFE Standardized Shift Map - {WINNER_ID}")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Standardized shift (SD units)")
    fig.tight_layout()
    fig.savefig(SHIFT_HEATMAP_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_feature_shift_summary_generic(summary_df: pd.DataFrame, out_path: Path, title: str) -> None:
    if summary_df.empty:
        return
    color_map = {"PT": "#2a9d8f", "CT": "#d55e00", "Dose": "#7b61a8"}
    plot_df = summary_df.sort_values("mean_abs_standardized_delta", ascending=True)
    vals = plot_df["mean_abs_standardized_delta"].to_numpy()
    labels = [feature_label(feature) for feature in plot_df["feature"]]
    colors = [color_map.get(group, "#5f5f5f") for group in plot_df["feature_group"]]
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    ax.barh(labels, vals, color=colors, edgecolor="none", height=0.66)
    ax.set_xlabel("Mean absolute standardized shift")
    ax.set_ylabel("Feature")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_feature_group_shift_matched(shift_df: pd.DataFrame) -> None:
    if shift_df.empty:
        return
    grouped = (
        shift_df.groupby("feature_group", as_index=False)
        .agg(total_abs_std_shift=("abs_standardized_delta", "sum"))
        .sort_values("total_abs_std_shift", ascending=False)
    )
    color_map = {"PT": "#2a9d8f", "CT": "#d55e00", "Dose": "#7b61a8", "Clinical": "#5f5f5f"}
    colors = [color_map.get(g, "#5f5f5f") for g in grouped["feature_group"]]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.bar(grouped["feature_group"], grouped["total_abs_std_shift"],
           color=colors, edgecolor="none", width=0.55)
    ax.set_xlabel("Feature group")
    ax.set_ylabel("Total absolute standardised shift")
    ax.set_title(f"T1C N75 Matched CFE Feature-Group Shift - {WINNER_ID}")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    fig.tight_layout()
    fig.savefig(FEATURE_GROUP_MATCHED_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    dice_ml = ensure_dice()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 88)
    print("T1C N75 BRANCH A2 CFE")
    print("=" * 88)

    dev_df, chus_df = load_t1c_frames()
    feature_scales = dev_df[MUTABLE_FEATURES].std(ddof=0).replace(0, 1.0)
    pipeline, risk_dev, threshold_dev = fit_locked_pipeline(dev_df)
    y_chus = make_surv(chus_df[TARGET_EVENT], chus_df[TARGET_TIME])
    risk_chus = pipeline.predict(chus_df[FEATURES])
    ci_chus = safe_ci(y_chus, risk_chus)
    delta = abs(ci_chus - LOCKED_CHUS)
    print(f"Reproduced CHUS={ci_chus:.6f}  locked={LOCKED_CHUS:.6f}  delta={delta:.6f}")
    if delta > REPRO_TOL:
        raise RuntimeError(
            f"Reproducibility check FAILED for {WINNER_ID}. "
            f"Expected {LOCKED_CHUS:.6f}, got {ci_chus:.6f}."
        )
    print(f"Reproducibility check PASSED for {WINNER_ID}. CHUS={ci_chus:.6f} delta={delta:.6f}")

    selected = select_borderline_cases(chus_df, pipeline, threshold_dev)
    matched = select_matched_cases(chus_df, pipeline, threshold_dev)
    selected_df, shift_df, transitions_df = generate_counterfactuals(
        dice_ml=dice_ml,
        dev_df=dev_df,
        chus_df=chus_df,
        pipeline=pipeline,
        selected=selected,
        threshold=threshold_dev,
        train_risk=risk_dev,
    )
    shift_df = add_standardized_deltas(shift_df, feature_scales)
    summary_df = summarise_standardized_shifts(shift_df)
    native_shift_df = shift_df.copy()

    matched_selected_df = pd.DataFrame()
    matched_shift_df = pd.DataFrame()
    matched_transitions_df = pd.DataFrame()
    matched_summary_df = pd.DataFrame()
    if matched:
        matched_selected_df, matched_shift_df, matched_transitions_df = generate_counterfactuals(
            dice_ml=dice_ml,
            dev_df=dev_df,
            chus_df=chus_df,
            pipeline=pipeline,
            selected=matched,
            threshold=threshold_dev,
            train_risk=risk_dev,
        )
        matched_shift_df = add_standardized_deltas(matched_shift_df, feature_scales)
        matched_summary_df = summarise_standardized_shifts(matched_shift_df)

    plot_transitions_df = pd.concat(
        [transitions_df, matched_transitions_df],
        ignore_index=True,
    )

    plot_feature_shift_summary(summary_df)
    plot_risk_transitions(plot_transitions_df, threshold_dev)
    plot_signed_shifts_per_case(
        pd.concat([native_shift_df, matched_shift_df], ignore_index=True),
        SIGNED_SHIFTS_PNG,
        "DAE CFE: which features DiCE changed and how (native + CHUS-065 cases)",
        top_n=5,
    )
    plot_shift_heatmap(shift_df)

    selected_df.to_csv(SELECTED_PATIENTS_CSV, index=False)
    shift_df.to_csv(SHIFT_TABLE_CSV, index=False)
    summary_df.to_csv(SHIFT_SUMMARY_CSV, index=False)
    plot_transitions_df.to_csv(RISK_TRANSITIONS_CSV, index=False)
    if matched:
        plot_feature_group_shift_matched(matched_shift_df)
        plot_feature_shift_summary_generic(
            matched_summary_df,
            FEATURE_SHIFT_MATCHED_PNG,
            f"T1C N75 matched CFE feature-shift - {WINNER_ID}",
        )
        matched_selected_df.to_csv(SELECTED_MATCHED_CSV, index=False)
        matched_shift_df.to_csv(SHIFT_TABLE_MATCHED_CSV, index=False)
        matched_summary_df.to_csv(SHIFT_SUMMARY_MATCHED_CSV, index=False)
        matched_transitions_df.to_csv(RISK_TRANSITIONS_MATCHED_CSV, index=False)
        group_totals = (
            matched_shift_df.groupby("feature_group")["abs_standardized_delta"]
            .sum()
            .reset_index()
            .rename(columns={"abs_standardized_delta": "total_abs_std_shift"})
        )
        group_totals.to_csv(FEATURE_GROUP_MATCHED_CSV, index=False)
        print("Matched group totals:")
        print(group_totals.to_string(index=False))
        print("Matched risk transitions:")
        print(
            matched_transitions_df[
                ["PatientID", "cf_index", "baseline_risk_score", "counterfactual_risk_score", "risk_delta"]
            ].to_string(index=False)
        )
    else:
        print("  No matched patients found in CHUS dose-eligible subset.")

    print(f"Dev n={len(dev_df)} | CHUS n={len(chus_df)}")
    print(f"Winner={WINNER_ID} trial_no={TRIAL_NO} OOF={LOCKED_OOF:.4f} CHUS={ci_chus:.6f}")
    print(f"Dev risk threshold (median): {threshold_dev:.6f}")
    print("Selected CHUS patients:")
    print(selected_df.to_string(index=False) if not selected_df.empty else "  none")
    print("Saved outputs:")
    for path in [
        SELECTED_PATIENTS_CSV,
        SHIFT_TABLE_CSV,
        SHIFT_SUMMARY_CSV,
        RISK_TRANSITIONS_CSV,
        FEATURE_SHIFT_BAR_PNG,
        RISK_TRANSITIONS_PNG,
        SIGNED_SHIFTS_PNG,
        SHIFT_HEATMAP_PNG,
    ]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
