"""
28_mar_task1_branchA2_CFE.py

Task 1 Post-Study - Branch A2: Counterfactual Explanations (DiCE)
Locked winner: PT_S3_1_768xCT_S3_8_235, N=12 (1 clinical + 7 PT + 4 CT)
Coach model: ExtraSurvivalTrees(n_estimators=200, random_state=42)

Design:
  - Refit the locked winner exactly as Branch A / Branch B.
  - Use DiCE in regression mode on survival risk score.
  - Define "flip" as crossing the dev-derived risk threshold used in KM calibration view.
  - Keep Gender_Male immutable; only radiomics are allowed to vary.
  - Generate 2 alternative counterfactual routes per selected patient.

Selected cases:
  - Native boundary cases: model-native event under-call and non-event over-call.
  - Matched dose-comparison cases: one event-rescue and one non-event-rescue
    shared CHUS dose-eligible patient also used by T1C.

Outputs (saved to Mar_2026/28_mar_task1_post_study_outputs/branchA2/):
  - selected_counterfactual_cases.csv
  - counterfactual_feature_shifts_long.csv
  - counterfactual_feature_shift_summary.csv
  - feature_shift_summary_all.csv / native.csv / matched.csv
  - counterfactual_risk_transitions.csv
  - risk_transitions_all/native/matched.png
  - shift_heatmap_all/native/matched.png
  - feature_shift_summary_native/matched.png
  - feature_group_shift_summary_matched.png
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import os
_SCRIPT_ROOT = Path(__file__).resolve().parent
_MPL_DIR = _SCRIPT_ROOT / "28_mar_task1_post_study_outputs" / "branchA2" / ".mplconfig"
_MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_MPL_DIR)

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


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "28_mar_task1_post_study_outputs" / "branchA2"

PT_DEV_FILE = ROOT / "27_feb_PT_development.csv"
CT_DEV_FILE = ROOT / "27_feb_CT_development.csv"
PT_EXT_FILE = ROOT / "27_feb_PT_external.csv"
CT_EXT_FILE = ROOT / "27_feb_CT_external.csv"
CLINICAL_FILE = ROOT.parent / "Feb_2026" / "25_feb_clinical_reduced_dataset" / "25_feb_Processed_clinical_reduced.csv"
PT_FEAT_FILE = ROOT / "2_mar_finalist_outputs" / "PT_inter1_768_features_recheck.csv"
CT_FEAT_FILE = ROOT / "2_mar_finalist_outputs" / "CT_inter8_235_features_recheck.csv"

TARGET_EVENT = "Relapse"
TARGET_TIME = "RFS"

SELECTED_PATIENTS_CSV = OUT_DIR / "T1_CFE_selected_counterfactual_cases.csv"
SELECTED_PATIENTS_LEGACY_CSV = OUT_DIR / "T1_CFE_selected_borderline_patients.csv"
SHIFT_TABLE_CSV = OUT_DIR / "T1_CFE_counterfactual_feature_shifts_long.csv"
SHIFT_SUMMARY_CSV = OUT_DIR / "T1_CFE_counterfactual_feature_shift_summary.csv"
SHIFT_SUMMARY_ALL_CSV = OUT_DIR / "T1_CFE_feature_shift_summary_all.csv"
SHIFT_SUMMARY_NATIVE_CSV = OUT_DIR / "T1_CFE_feature_shift_summary_native.csv"
SHIFT_SUMMARY_MATCHED_CSV = OUT_DIR / "T1_CFE_feature_shift_summary_matched.csv"
FEATURE_GROUP_MATCHED_CSV = OUT_DIR / "T1_CFE_feature_group_shift_summary_matched.csv"
RISK_TRANSITIONS_CSV = OUT_DIR / "T1_CFE_counterfactual_risk_transitions.csv"
FEATURE_SHIFT_BAR_PNG = OUT_DIR / "T1_CFE_feature_shift_summary_bar.png"
FEATURE_SHIFT_NATIVE_PNG = OUT_DIR / "T1_CFE_feature_shift_summary_native.png"
FEATURE_SHIFT_MATCHED_PNG = OUT_DIR / "T1_CFE_feature_shift_summary_matched.png"
FEATURE_GROUP_MATCHED_PNG = OUT_DIR / "T1_CFE_feature_group_shift_summary_matched.png"
RISK_TRANSITIONS_PNG = OUT_DIR / "T1_CFE_counterfactual_risk_transitions.png"
SHIFT_HEATMAP_PNG = OUT_DIR / "T1_CFE_counterfactual_shift_heatmap.png"
SHIFT_HEATMAP_ALL_PNG = OUT_DIR / "T1_CFE_shift_heatmap_all.png"
SHIFT_HEATMAP_NATIVE_PNG = OUT_DIR / "T1_CFE_shift_heatmap_native.png"
SHIFT_HEATMAP_MATCHED_PNG = OUT_DIR / "T1_CFE_shift_heatmap_matched.png"
SIGNED_SHIFTS_PNG = OUT_DIR / "T1_CFE_signed_shifts_per_case.png"

SEED = 42
N_EST = 200
TOTAL_CFS = 2
REPRO_TOL = 0.001
LOCKED_CHUS = 0.742857142857143
LOCKED_CHUP = 0.727586206896552
CASE_GROUP_NATIVE = "native_boundary"
CASE_GROUP_MATCHED = "matched_dose_comparison"

# SHAP risk-association directions from Table 3.2.5.
# +1 = higher feature value -> higher predicted risk (positive SHAP direction)
# -1 = higher feature value -> lower predicted risk (negative SHAP direction)
SHAP_DIRECTION: dict[str, int] = {
    "GTVp_gradient_glszm_ZoneEntropy": +1,
    "GTVn_wavelet-LLH_firstorder_Mean": -1,
    "GTVp_wavelet-LLH_glrlm_HighGrayLevelRunEmphasis": -1,
    "GTVp_exponential_glszm_HighGrayLevelZoneEmphasis": -1,
    "GTVp_wavelet-HLH_glrlm_ShortRunHighGrayLevelEmphasis": -1,
    "GTVn_wavelet-LHH_glszm_LowGrayLevelZoneEmphasis": -1,
    "GTVp_wavelet-HLH_glszm_HighGrayLevelZoneEmphasis": -1,
    "GTVp_wavelet-LHL_glszm_SmallAreaHighGrayLevelEmphasis": +1,
    "GTVp_wavelet-LHH_firstorder_RootMeanSquared": -1,
    "GTVp_gradient_glszm_SmallAreaLowGrayLevelEmphasis": +1,
    "GTVp_wavelet-HLL_ngtdm_Complexity": +1,
}

# Short human-readable labels for CFE bar panels (label code from Table 3.2.2)
_CFE_LABEL: dict[str, str] = {
    "GTVp_gradient_glszm_ZoneEntropy":                       "GTVp ZoneEntropy (P1)",
    "GTVn_wavelet-LLH_firstorder_Mean":                      "GTVn LLH Mean (P6)",
    "GTVp_wavelet-LLH_glrlm_HighGrayLevelRunEmphasis":       "GTVp LLH HGRE (T1)",
    "GTVp_exponential_glszm_HighGrayLevelZoneEmphasis":      "GTVp Exp HGZE (P2)",
    "GTVp_wavelet-HLH_glrlm_ShortRunHighGrayLevelEmphasis":  "GTVp HLH SRHGLE (P3)",
    "GTVn_wavelet-LHH_glszm_LowGrayLevelZoneEmphasis":       "GTVn LHH LGLZE (P7)",
    "GTVp_wavelet-HLH_glszm_HighGrayLevelZoneEmphasis":      "GTVp HLH HGLZE (P5)",
    "GTVp_wavelet-LHL_glszm_SmallAreaHighGrayLevelEmphasis": "GTVp LHL SAHGLE (P4)",
    "GTVp_wavelet-LHH_firstorder_RootMeanSquared":           "GTVp LHH RMS (T4)",
    "GTVp_gradient_glszm_SmallAreaLowGrayLevelEmphasis":     "GTVp SALLGLE (T3)",
    "GTVp_wavelet-HLL_ngtdm_Complexity":                     "GTVp HLL Cmplxty (T2)",
    "Gender_Male":                                            "Gender Male (C1)",
}

_MODALITY_COLOR: dict[str, str] = {
    "PT": "#f58518",
    "CT": "#4c78a8",
    "Clinical": "#777777",
    "Unknown": "#aaaaaa",
}


def _cfe_bar_label(feat: str) -> str:
    return _CFE_LABEL.get(feat, feature_label(feat))


def _cfe_bar_color(feat: str) -> str:
    return _MODALITY_COLOR.get(feature_group(feat), "#aaaaaa")
# Main-text same-direction matched example:
# - CHUS-065: non-event overcalled high-risk by both RPM and DAE.
#   Both models therefore cross downward in the CFE risk-transition view.
MATCHED_PATIENT_IDS = ["CHUS-065"]

# Locked winner features from Branch A/B
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
FEATURES = CLINICAL_FEATURES + PT_WINNER + CT_WINNER
RADIOMICS_FEATURES = PT_WINNER + CT_WINNER


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


@dataclass
class SelectedPatient:
    case_group: str
    label: str
    patient_id: str
    center: str
    event: int
    risk_score: float
    boundary_distance: float
    selection_reason: str


def ensure_dice():
    try:
        import dice_ml  # type: ignore
    except ImportError as exc:
        raise ImportError("dice_ml not installed. Run: pip install dice-ml") from exc
    return dice_ml


def ensure_output_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_surv(event: pd.Series, time: pd.Series):
    return Surv.from_arrays(event=np.asarray(event, dtype=bool), time=np.asarray(time, dtype=float))


def safe_ci(y, risk: np.ndarray) -> float:
    try:
        return float(concordance_index_censored(y["event"], y["time"], risk)[0])
    except Exception:
        return float("nan")


def feature_label(feature: str) -> str:
    if feature == "Gender_Male":
        return feature
    parts = feature.split("_")
    if len(parts) >= 2:
        return "_".join(parts[-2:])
    return feature


def feature_group(feature: str) -> str:
    if feature in CLINICAL_FEATURES:
        return "Clinical"
    if feature in PT_WINNER:
        return "PT"
    if feature in CT_WINNER:
        return "CT"
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
    elif label.startswith("matched_"):
        tag = "Matched comparator"
    else:
        tag = "Boundary case"
    return f"{patient_id}\n{tag}"


def load_task1_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clinical = pd.read_csv(CLINICAL_FILE).dropna(subset=[TARGET_EVENT, TARGET_TIME])
    pt_feat_pool = pd.read_csv(PT_FEAT_FILE)["Feature"].tolist()
    ct_feat_raw = pd.read_csv(CT_FEAT_FILE)["Feature"].tolist()
    pt_set = set(pt_feat_pool)
    ct_feat_pool = [f for f in ct_feat_raw if f not in pt_set]

    clin_dev = clinical[clinical["Cohort"] == "Dev"][
        ["PatientID", "CenterID", TARGET_EVENT, TARGET_TIME] + CLINICAL_FEATURES
    ].copy()
    clin_chus = clinical[clinical["CenterID"] == 3][
        ["PatientID", TARGET_EVENT, TARGET_TIME] + CLINICAL_FEATURES
    ].copy()
    clin_chup = clinical[clinical["CenterID"] == 2][
        ["PatientID", TARGET_EVENT, TARGET_TIME] + CLINICAL_FEATURES
    ].copy()

    pt_dev = pd.read_csv(PT_DEV_FILE)
    ct_dev = pd.read_csv(CT_DEV_FILE)
    pt_ext = pd.read_csv(PT_EXT_FILE)
    ct_ext = pd.read_csv(CT_EXT_FILE)

    # Mirror Branch A/B merge path exactly for strict reproducibility.
    rad_dev = pt_dev[["PatientID"] + pt_feat_pool].merge(
        ct_dev[["PatientID"] + ct_feat_pool], on="PatientID", how="inner"
    )
    rad_ext = pt_ext[["PatientID"] + pt_feat_pool].merge(
        ct_ext[["PatientID"] + ct_feat_pool], on="PatientID", how="inner"
    )
    rad_chus = rad_ext[rad_ext["PatientID"].str.startswith("CHUS")].copy()
    rad_chup = rad_ext[rad_ext["PatientID"].str.startswith("CHUP")].copy()

    dev_df = clin_dev.merge(rad_dev, on="PatientID", how="inner")
    chus_df = clin_chus.merge(rad_chus, on="PatientID", how="inner")
    chup_df = clin_chup.merge(rad_chup, on="PatientID", how="inner")

    ext_df = pd.concat(
        [
            chus_df.assign(ExternalCenter="CHUS"),
            chup_df.assign(ExternalCenter="CHUP"),
        ],
        ignore_index=True,
    )
    return dev_df, chus_df, ext_df


def fit_locked_pipeline(dev_df: pd.DataFrame) -> tuple[Pipeline, np.ndarray, float]:
    x_dev = dev_df[FEATURES].copy()
    y_dev = make_surv(dev_df[TARGET_EVENT], dev_df[TARGET_TIME])

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            # Use single-thread mode to avoid Windows IPC permission issues in some environments.
            ("est", ExtraSurvivalTrees(n_estimators=N_EST, random_state=SEED, n_jobs=1)),
        ]
    )
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
                case_group=CASE_GROUP_NATIVE,
                label="event_low_risk_near_boundary",
                patient_id=str(row["PatientID"]),
                center=str(row["ExternalCenter"]),
                event=int(row[TARGET_EVENT]),
                risk_score=float(row["risk_score"]),
                boundary_distance=float(row["boundary_distance"]),
                selection_reason="Native Task 1 event under-call closest to the dev risk threshold.",
            )
        )

    nonevent_highrisk = eval_df[(eval_df[TARGET_EVENT] == 0) & (eval_df["risk_score"] >= threshold)].copy()
    if not nonevent_highrisk.empty:
        row = nonevent_highrisk.sort_values("risk_score", ascending=True).iloc[0]
        selected.append(
            SelectedPatient(
                case_group=CASE_GROUP_NATIVE,
                label="nonevent_high_risk_near_boundary",
                patient_id=str(row["PatientID"]),
                center=str(row["ExternalCenter"]),
                event=int(row[TARGET_EVENT]),
                risk_score=float(row["risk_score"]),
                boundary_distance=float(row["boundary_distance"]),
                selection_reason="Native Task 1 non-event over-call closest to the dev risk threshold.",
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
                    case_group=CASE_GROUP_NATIVE,
                    label=f"boundary_fallback_{i}",
                    patient_id=str(row["PatientID"]),
                    center=str(row["ExternalCenter"]),
                    event=int(row[TARGET_EVENT]),
                    risk_score=float(row["risk_score"]),
                    boundary_distance=float(row["boundary_distance"]),
                    selection_reason="Native fallback case selected by nearest boundary distance.",
                )
            )

    return selected


def select_matched_cases(ext_df: pd.DataFrame, pipeline: Pipeline, threshold: float) -> list[SelectedPatient]:
    eval_df = ext_df.copy()
    eval_df["risk_score"] = pipeline.predict(eval_df[FEATURES])
    eval_df["boundary_distance"] = (eval_df["risk_score"] - threshold).abs()

    matched: list[SelectedPatient] = []
    for patient_id in MATCHED_PATIENT_IDS:
        row_df = eval_df.loc[eval_df["PatientID"] == patient_id]
        if row_df.empty:
            raise ValueError(f"Matched patient {patient_id} not found in Task 1 external dataframe.")
        row = row_df.iloc[0]
        side = "low_risk" if float(row["risk_score"]) < threshold else "high_risk"
        label = f"matched_{'event' if int(row[TARGET_EVENT]) == 1 else 'nonevent'}_{side}"
        matched.append(
            SelectedPatient(
                case_group=CASE_GROUP_MATCHED,
                label=label,
                patient_id=str(row["PatientID"]),
                center=str(row["ExternalCenter"]),
                event=int(row[TARGET_EVENT]),
                risk_score=float(row["risk_score"]),
                boundary_distance=float(row["boundary_distance"]),
                selection_reason=(
                    "Fixed shared CHUS dose-eligible rescue comparator selected for matched Task 1 vs T1C CFE."
                ),
            )
        )
    return matched


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

    out = (
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
    out["feature_group"] = out["feature"].map(feature_group)
    return out[
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
    ext_df: pd.DataFrame,
    pipeline: Pipeline,
    selected: Iterable[SelectedPatient],
    threshold: float,
    train_risk: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_for_dice = dev_df[FEATURES].copy()
    train_for_dice["risk_score"] = train_risk

    dice_data = dice_ml.Data(
        dataframe=train_for_dice,
        continuous_features=RADIOMICS_FEATURES,
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

    for p in selected:
        row_df = ext_df.loc[ext_df["PatientID"] == p.patient_id, ["PatientID", "ExternalCenter", TARGET_EVENT] + FEATURES].copy()
        if row_df.empty:
            continue
        q = row_df.iloc[0]
        q_features = row_df[FEATURES].copy().reset_index(drop=True)

        if p.risk_score >= threshold:
            desired = [min_risk, threshold - margin]
            target_side = "below_threshold"
        else:
            desired = [threshold + margin, max_risk]
            target_side = "above_threshold"

        if desired[0] >= desired[1]:
            # Extremely narrow range edge case; widen minimally.
            desired = [min_risk, max_risk]

        cf = explainer.generate_counterfactuals(
            query_instances=q_features,
            total_CFs=TOTAL_CFS,
            desired_range=desired,
            features_to_vary=RADIOMICS_FEATURES,
            random_seed=SEED,
        )

        cf_rows = cf.cf_examples_list[0].final_cfs_df.reset_index(drop=True)
        selected_rows.append(
            {
                "case_group": p.case_group,
                "label": p.label,
                "PatientID": p.patient_id,
                "ExternalCenter": p.center,
                "event": p.event,
                "risk_score": round(p.risk_score, 6),
                "boundary_distance": round(p.boundary_distance, 6),
                "threshold_dev": round(threshold, 6),
                "target_side": target_side,
                "selection_reason": p.selection_reason,
            }
        )

        for cf_idx, cf_row in cf_rows.iterrows():
            cf_features = cf_row[FEATURES].to_frame().T
            cf_risk = float(pipeline.predict(cf_features)[0])
            crossed_threshold = (p.risk_score < threshold <= cf_risk) or (p.risk_score >= threshold > cf_risk)
            transitions.append(
                {
                    "case_group": p.case_group,
                    "label": p.label,
                    "PatientID": p.patient_id,
                    "ExternalCenter": p.center,
                    "cf_index": cf_idx + 1,
                    "baseline_risk_score": round(p.risk_score, 6),
                    "counterfactual_risk_score": round(cf_risk, 6),
                    "risk_delta": round(cf_risk - p.risk_score, 6),
                    "threshold_dev": round(threshold, 6),
                    "crossed_threshold": crossed_threshold,
                }
            )

            for feat in FEATURES:
                b = q[feat]
                c = cf_row[feat]
                if feat == "Gender_Male":
                    changed = bool(b != c)
                    delta = None
                else:
                    changed = abs(float(c) - float(b)) > 1e-12
                    delta = float(c) - float(b)
                if changed:
                    shifts.append(
                        {
                            "case_group": p.case_group,
                            "label": p.label,
                            "PatientID": p.patient_id,
                            "ExternalCenter": p.center,
                            "event": int(q[TARGET_EVENT]),
                            "cf_index": cf_idx + 1,
                            "feature": feat,
                            "feature_group": feature_group(feat),
                            "baseline_value": b,
                            "counterfactual_value": c,
                            "delta": delta,
                            "baseline_risk_score": round(p.risk_score, 6),
                            "counterfactual_risk_score": round(cf_risk, 6),
                        }
                    )

    return pd.DataFrame(selected_rows), pd.DataFrame(shifts), pd.DataFrame(transitions)


def plot_feature_shift_summary(summary_df: pd.DataFrame, out_path: Path, title: str) -> None:
    if summary_df.empty:
        return
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    plot_df = summary_df.sort_values("mean_abs_standardized_delta", ascending=True)
    vals = plot_df["mean_abs_standardized_delta"].to_numpy()
    labels = [feature_label(f) for f in plot_df["feature"]]
    color_map = {"PT": "#2a9d8f", "CT": "#d55e00", "Dose": "#7b61a8"}
    colors = [color_map.get(g, "#2f6c8f") for g in plot_df.get("feature_group", pd.Series([""] * len(plot_df)))]
    ax.barh(labels, vals, color=colors, edgecolor="none", height=0.65)
    ax.set_xlabel("Mean absolute standardized shift")
    ax.set_ylabel("Feature")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.35)
    for y, v in enumerate(vals):
        ax.text(v + 0.04, y, f"{v:.2f}", va="center", ha="left", fontsize=8)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2a9d8f", label="PT"),
        Patch(facecolor="#d55e00", label="CT"),
        Patch(facecolor="#7b61a8", label="Dose"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_signed_shifts_per_case(
    shift_df: pd.DataFrame,
    out_path: Path,
    title: str,
    top_n: int = 5,
) -> None:
    """Per-case per-route signed feature-shift bar chart.

    Bar colour = modality (PT orange / CT blue / Clinical grey).
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

            # risk transition info in title
            risk_str = ""
            if "baseline_risk_score" in sub.columns:
                risk_b = sub["baseline_risk_score"].iloc[0]
                risk_c = sub["counterfactual_risk_score"].iloc[0]
                arrow = "\u2191" if risk_c > risk_b else "\u2193"
                risk_str = f"Risk {risk_b:.2f} {arrow} {risk_c:.2f}"
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

            # value annotations with direction arrows
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
        Patch(facecolor="white", edgecolor="#cc00cc", linewidth=2.25,
              label="SHAP-consistent (magenta edge)"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center", ncol=3,
        frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(title, fontsize=10, y=1.01)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_risk_transitions(transitions_df: pd.DataFrame, threshold: float, out_path: Path, title: str) -> None:
    if transitions_df.empty:
        return
    order = transitions_df[["label", "PatientID"]].drop_duplicates().reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for x_idx, row in order.iterrows():
        mask = transitions_df["PatientID"] == row["PatientID"]
        rows = transitions_df.loc[mask].sort_values("cf_index")
        baseline = rows["baseline_risk_score"].iloc[0]
        ax.scatter([x_idx], [baseline], color="black", s=62, label="Observed" if x_idx == 0 else None, zorder=4)
        for _, cf in rows.iterrows():
            ax.scatter(
                [x_idx],
                [cf["counterfactual_risk_score"]],
                color="#c75b39",
                s=62,
                label="Counterfactual route" if x_idx == 0 and cf["cf_index"] == 1 else None,
                zorder=4,
            )
            ax.plot([x_idx, x_idx], [baseline, cf["counterfactual_risk_score"]], color="#8f8f8f", linewidth=1.8, alpha=0.8, label=None)
            cf_risks = rows["counterfactual_risk_score"].to_numpy()
            overlap = len(cf_risks) == 2 and abs(cf_risks[0] - cf_risks[1]) < 0.02
            route_char = chr(ord("A") + int(cf["cf_index"]) - 1)
            if overlap and route_char == "B":
                txt_x, txt_ha = x_idx - 0.08, "right"
            else:
                txt_x, txt_ha = x_idx + 0.08, "left"
            ax.text(
                txt_x,
                cf["counterfactual_risk_score"],
                f"Route {route_char}",
                fontsize=8,
                va="center",
                ha=txt_ha,
                color="#6b3929",
            )

    ax.axhline(threshold, color="#1f77b4", linestyle="--", linewidth=1.5, label="Dev risk threshold")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([patient_group_label(r["PatientID"], r["label"]) for _, r in order.iterrows()])
    ax.set_ylabel("Predicted survival risk score")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_shift_heatmap(shift_df: pd.DataFrame, out_path: Path, title: str) -> None:
    if shift_df.empty:
        return
    hm = shift_df.copy()
    hm["patient_cf"] = hm.apply(
        lambda r: patient_route_label(str(r["PatientID"]), str(r["label"]), int(r["cf_index"])),
        axis=1,
    )
    hm["feature_short"] = hm["feature"].map(feature_label)
    pivot = (
        hm.pivot_table(index="feature_short", columns="patient_cf", values="standardized_delta", aggfunc="mean", fill_value=0.0)
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
    ax.set_title(title)

    ax.set_xticks([x - 0.5 for x in range(1, pivot.shape[1])], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, pivot.shape[0])], minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8, alpha=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = float(pivot.iloc[i, j])
            if abs(val) < 0.15:
                continue
            ax.text(
                j,
                i,
                f"{val:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="white" if abs(val) > vmax * 0.42 else "black",
            )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Standardized shift (SD units)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_feature_group_summary(shift_df: pd.DataFrame, out_path: Path, title: str) -> pd.DataFrame:
    if shift_df.empty:
        return pd.DataFrame(columns=["case_group", "feature_group", "total_abs_standardized_shift", "n_changes"])
    group_df = (
        shift_df.groupby(["case_group", "feature_group"], as_index=False)
        .agg(
            total_abs_standardized_shift=("abs_standardized_delta", "sum"),
            n_changes=("feature", "size"),
        )
        .sort_values("total_abs_standardized_shift", ascending=False)
    )
    plot_df = group_df.groupby("feature_group", as_index=False)["total_abs_standardized_shift"].sum()
    order = ["PT", "CT", "Dose", "Clinical", "Unknown"]
    plot_df["feature_group"] = pd.Categorical(plot_df["feature_group"], categories=order, ordered=True)
    plot_df = plot_df.sort_values("feature_group")
    colors = {"PT": "#2a9d8f", "CT": "#d55e00", "Dose": "#7b61a8", "Clinical": "#555555", "Unknown": "#999999"}

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.bar(
        plot_df["feature_group"].astype(str),
        plot_df["total_abs_standardized_shift"],
        color=[colors.get(str(g), "#999999") for g in plot_df["feature_group"]],
        edgecolor="none",
    )
    ax.set_ylabel("Total absolute standardized shift")
    ax.set_xlabel("Feature group")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return group_df


def main() -> None:
    dice_ml = ensure_dice()
    ensure_output_dir()

    dev_df, chus_df, ext_df = load_task1_frames()
    x_dev = dev_df[FEATURES].values.astype(float)
    y_dev = make_surv(dev_df[TARGET_EVENT], dev_df[TARGET_TIME])
    feature_scales = dev_df[RADIOMICS_FEATURES].std(ddof=0).replace(0, 1.0)

    pipeline, risk_dev, threshold_dev = fit_locked_pipeline(dev_df)
    ci_chus = safe_ci(
        make_surv(chus_df[TARGET_EVENT], chus_df[TARGET_TIME]),
        pipeline.predict(chus_df[FEATURES]),
    )
    chup_df = ext_df[ext_df["ExternalCenter"] == "CHUP"].copy()
    ci_chup = safe_ci(
        make_surv(chup_df[TARGET_EVENT], chup_df[TARGET_TIME]),
        pipeline.predict(chup_df[FEATURES]),
    )
    if abs(ci_chus - LOCKED_CHUS) > REPRO_TOL or abs(ci_chup - LOCKED_CHUP) > REPRO_TOL:
        raise RuntimeError(
            "Reproducibility check FAILED. "
            f"CHUS={ci_chus:.6f} (locked={LOCKED_CHUS:.6f}), "
            f"CHUP={ci_chup:.6f} (locked={LOCKED_CHUP:.6f}). "
            "Use the same Python environment as the locked Task 1 post-study runs."
        )
    print(
        "Reproducibility check PASSED. "
        f"CHUS={ci_chus:.6f} (locked={LOCKED_CHUS:.6f}), "
        f"CHUP={ci_chup:.6f} (locked={LOCKED_CHUP:.6f})."
    )

    selected = select_borderline_cases(ext_df, pipeline, threshold_dev)
    selected.extend(select_matched_cases(ext_df, pipeline, threshold_dev))
    selected_df, shift_df, transitions_df = generate_counterfactuals(
        dice_ml=dice_ml,
        dev_df=dev_df,
        ext_df=ext_df,
        pipeline=pipeline,
        selected=selected,
        threshold=threshold_dev,
        train_risk=risk_dev,
    )
    shift_df = add_standardized_deltas(shift_df, feature_scales)
    summary_df = summarise_standardized_shifts(shift_df)
    native_shift_df = shift_df[shift_df["case_group"] == CASE_GROUP_NATIVE].copy()
    matched_shift_df = shift_df[shift_df["case_group"] == CASE_GROUP_MATCHED].copy()
    native_summary_df = summarise_standardized_shifts(native_shift_df)
    matched_summary_df = summarise_standardized_shifts(matched_shift_df)

    plot_feature_shift_summary(summary_df, FEATURE_SHIFT_BAR_PNG, "Task 1 CFE feature-shift summary (all cases)")
    plot_feature_shift_summary(native_summary_df, FEATURE_SHIFT_NATIVE_PNG, "Task 1 native-boundary CFE feature shifts")
    plot_feature_shift_summary(matched_summary_df, FEATURE_SHIFT_MATCHED_PNG, "Task 1 matched-comparator CFE feature shifts")
    plot_risk_transitions(
        transitions_df,
        threshold_dev,
        RISK_TRANSITIONS_PNG,
        "Task 1 CFE risk transitions: native boundary cases plus CHUS-065 matched case",
    )
    plot_signed_shifts_per_case(
        pd.concat([native_shift_df, matched_shift_df], ignore_index=True),
        SIGNED_SHIFTS_PNG,
        "Task 1 CFE: which features DiCE changed and how (native + CHUS-065 cases)",
        top_n=5,
    )
    plot_shift_heatmap(shift_df, SHIFT_HEATMAP_PNG, "Task 1 CFE standardized shift map (all cases)")
    plot_shift_heatmap(shift_df, SHIFT_HEATMAP_ALL_PNG, "Task 1 CFE standardized shift map (all cases)")
    plot_shift_heatmap(native_shift_df, SHIFT_HEATMAP_NATIVE_PNG, "Task 1 native-boundary standardized shift map")
    plot_shift_heatmap(matched_shift_df, SHIFT_HEATMAP_MATCHED_PNG, "Task 1 matched-comparator standardized shift map")
    group_matched_df = plot_feature_group_summary(
        matched_shift_df,
        FEATURE_GROUP_MATCHED_PNG,
        "Task 1 matched-comparator feature-group shifts",
    )

    selected_df.to_csv(SELECTED_PATIENTS_CSV, index=False)
    selected_df.to_csv(SELECTED_PATIENTS_LEGACY_CSV, index=False)
    shift_df.to_csv(SHIFT_TABLE_CSV, index=False)
    summary_df.to_csv(SHIFT_SUMMARY_CSV, index=False)
    summary_df.to_csv(SHIFT_SUMMARY_ALL_CSV, index=False)
    native_summary_df.to_csv(SHIFT_SUMMARY_NATIVE_CSV, index=False)
    matched_summary_df.to_csv(SHIFT_SUMMARY_MATCHED_CSV, index=False)
    group_matched_df.to_csv(FEATURE_GROUP_MATCHED_CSV, index=False)
    transitions_df.to_csv(RISK_TRANSITIONS_CSV, index=False)

    print("=" * 88)
    print("TASK 1 BRANCH A2 CFE")
    print("=" * 88)
    print(f"Dev n={len(dev_df)} | CHUS n={len(chus_df)} | Ext total n={len(ext_df)}")
    print(f"Recomputed CHUS C-index (for anchor): {ci_chus:.6f}")
    print(f"Recomputed CHUP C-index (for anchor): {ci_chup:.6f}")
    print(f"Dev risk threshold (median): {threshold_dev:.6f}")
    print()
    print("Selected counterfactual external patients:")
    if selected_df.empty:
        print("  none")
    else:
        print(selected_df.to_string(index=False))
    print()
    print("Counterfactual transitions:")
    if transitions_df.empty:
        print("  none")
    else:
        print(transitions_df.to_string(index=False))
    print()
    print("Top standardized counterfactual levers:")
    if summary_df.empty:
        print("  none")
    else:
        print(summary_df.head(10).to_string(index=False))
    print()
    print("Saved outputs:")
    print(f"- {SELECTED_PATIENTS_CSV}")
    print(f"- {SHIFT_TABLE_CSV}")
    print(f"- {SHIFT_SUMMARY_CSV}")
    print(f"- {RISK_TRANSITIONS_CSV}")
    print(f"- {FEATURE_SHIFT_BAR_PNG}")
    print(f"- {RISK_TRANSITIONS_PNG}")
    print(f"- {SHIFT_HEATMAP_PNG}")


if __name__ == "__main__":
    main()
