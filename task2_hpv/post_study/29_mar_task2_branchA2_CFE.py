"""
29_mar_task2_branchA2_CFE.py

Task 2 Post-Study - Branch A2: Counterfactual Explanations (DiCE)
Locked winner: T68357, N=16 (1 clinical + 8 PT + 7 CT)
Coach: LR_L2, C=0.3689386715674865

Purpose:
  Generate patient-level counterfactual explanations for the locked Task 2 winner
  to show the minimal radiomics shifts required to flip the predicted HPV class.

Notes:
  - Model is trained strictly on the 67-patient Task 2 train split.
  - Borderline patients are selected from the fixed 20-patient Task 2 test split.
  - `Gender_Male` is treated as categorical and immutable.
  - Only the 15 radiomics features are allowed to vary.

Example:
  python Mar_2026_task2/29_mar_task2_branchA2_CFE.py
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
RAD_DIR = ROOT / "12_mar_task2_rad_data"
OUT_DIR = ROOT / "29_mar_task2_post_study_outputs" / "branchA2"

CT_TRAIN_FILE = RAD_DIR / "13_mar_task2_CT_primary_train.csv"
CT_TEST_FILE = RAD_DIR / "13_mar_task2_CT_primary_test.csv"
PT_TRAIN_FILE = RAD_DIR / "13_mar_task2_PT_primary_train.csv"
PT_TEST_FILE = RAD_DIR / "13_mar_task2_PT_primary_test.csv"

SELECTED_PATIENTS_CSV = OUT_DIR / "T2_CFE_selected_borderline_patients.csv"
SHIFT_TABLE_CSV = OUT_DIR / "T2_CFE_counterfactual_feature_shifts_long.csv"
SHIFT_SUMMARY_CSV = OUT_DIR / "T2_CFE_counterfactual_feature_shift_summary.csv"
PROBA_TRANSITIONS_CSV = OUT_DIR / "T2_CFE_counterfactual_probability_transitions.csv"
FEATURE_SHIFT_BAR_PNG = OUT_DIR / "T2_CFE_feature_shift_summary_bar.png"
PROBA_TRANSITIONS_PNG = OUT_DIR / "T2_CFE_counterfactual_probability_transitions.png"
SHIFT_HEATMAP_PNG = OUT_DIR / "T2_CFE_counterfactual_shift_heatmap.png"
SIGNED_SHIFTS_PNG = OUT_DIR / "T2_CFE_signed_shifts_per_case.png"

TARGET = "HPV_binary"
CLINICAL_FEATURES = ["Gender_Male"]
PT_FEATURES = [
    "GTVn_wavelet-LHL_glszm_GrayLevelVariance",
    "GTVn_logarithm_glszm_SmallAreaLowGrayLevelEmphasis",
    "GTVn_wavelet-LLH_firstorder_Skewness",
    "GTVn_squareroot_glcm_Idm",
    "GTVp_wavelet-LLH_firstorder_Median",
    "GTVn_logarithm_glcm_Idn",
    "GTVp_wavelet-HHL_glrlm_ShortRunHighGrayLevelEmphasis",
    "GTVp_original_firstorder_InterquartileRange",
]
CT_FEATURES = [
    "GTVp_log-sigma-1-mm-3D_firstorder_Range",
    "GTVn_wavelet-HLH_gldm_SmallDependenceEmphasis",
    "GTVn_wavelet-LHH_glrlm_GrayLevelVariance",
    "GTVn_wavelet-HLH_gldm_SmallDependenceHighGrayLevelEmphasis",
    "GTVn_wavelet-HHH_glcm_DifferenceAverage",
    "GTVn_wavelet-HHH_glszm_ZonePercentage",
    "GTVn_wavelet-LHH_glcm_ClusterProminence",
]
FEATURES = CLINICAL_FEATURES + PT_FEATURES + CT_FEATURES
RADIOMICS_FEATURES = PT_FEATURES + CT_FEATURES

LOCKED_C = 0.3689386715674865
SEED = 42
THRESHOLD = 0.5

# SHAP direction per feature (from Table 3.4.5):
# +1 = higher value → higher P(HPV+); -1 = higher value → lower P(HPV+)
SHAP_DIRECTION: dict[str, int] = {
    "GTVn_wavelet-LHL_glszm_GrayLevelVariance": -1,           # P1
    "GTVp_wavelet-HHL_glrlm_ShortRunHighGrayLevelEmphasis": +1, # P7
    "GTVp_wavelet-LLH_firstorder_Median": +1,                  # P5
    "GTVp_original_firstorder_InterquartileRange": -1,          # P8
    "GTVn_logarithm_glszm_SmallAreaLowGrayLevelEmphasis": -1,  # P2
    "GTVn_wavelet-LLH_firstorder_Skewness": -1,                # P3
    "GTVn_wavelet-HHH_glcm_DifferenceAverage": +1,             # C5
    "GTVn_wavelet-LHH_glrlm_GrayLevelVariance": +1,            # C3
    "Gender_Male": -1,                                          # Cl
    "GTVn_squareroot_glcm_Idm": +1,                            # P4
    "GTVn_logarithm_glcm_Idn": +1,                             # P6
    "GTVn_wavelet-HLH_gldm_SmallDependenceHighGrayLevelEmphasis": -1, # C4
    "GTVp_log-sigma-1-mm-3D_firstorder_Range": -1,             # C1
    "GTVn_wavelet-HLH_gldm_SmallDependenceEmphasis": -1,       # C2
    "GTVn_wavelet-LHH_glcm_ClusterProminence": -1,             # C7
    "GTVn_wavelet-HHH_glszm_ZonePercentage": +1,               # C6
}

_MODALITY_COLOR = {"PT": "#f58518", "CT": "#4c78a8", "Clinical": "#aaaaaa"}


def _feature_modality(feat: str) -> str:
    if feat in PT_FEATURES:
        return "PT"
    if feat in CT_FEATURES:
        return "CT"
    return "Clinical"


def _cfe_bar_label(feat: str) -> str:
    if feat == "Gender_Male":
        return "Gender_Male"
    parts = feat.split("_")
    return "_".join(parts[-2:]) if len(parts) >= 2 else feat


def _cfe_bar_color(feat: str) -> str:
    return _MODALITY_COLOR[_feature_modality(feat)]


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
    label: str
    patient_id: str
    y_true: int
    y_pred: int
    proba_hpv_pos: float


def feature_label(feature: str) -> str:
    if feature == "Gender_Male":
        return feature
    parts = feature.split("_")
    if len(parts) >= 2:
        return "_".join(parts[-2:])
    return feature


def patient_route_label(patient_id: str, label: str, cf_index: int) -> str:
    if label.startswith("false_negative"):
        base = "FN"
    elif label.startswith("true_positive"):
        base = "TP"
    else:
        base = "BD"
    route = chr(ord("A") + cf_index - 1)
    return f"{patient_id}\n{base} route {route}"


def patient_group_label(patient_id: str, label: str) -> str:
    if label.startswith("false_negative"):
        base = "False negative"
    elif label.startswith("true_positive"):
        base = "True positive"
    else:
        base = "Boundary case"
    return f"{patient_id}\n{base}"


def ensure_dice():
    try:
        import dice_ml  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "dice_ml is not installed. Install it with: pip install dice-ml"
        ) from exc
    return dice_ml


def ensure_output_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_split(ct_path: Path, pt_path: Path) -> pd.DataFrame:
    ct_df = pd.read_csv(ct_path)
    pt_df = pd.read_csv(pt_path)

    ct_keep = ["PatientID", TARGET] + CLINICAL_FEATURES + CT_FEATURES
    pt_keep = ["PatientID"] + PT_FEATURES

    merged = ct_df[ct_keep].merge(
        pt_df[pt_keep],
        on="PatientID",
        how="inner",
        validate="one_to_one",
    )
    return merged.copy()


def fit_locked_pipeline(train_df: pd.DataFrame) -> Pipeline:
    x_train = train_df[FEATURES].copy()
    y_train = train_df[TARGET].astype(int).copy()

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    penalty="l2",
                    C=LOCKED_C,
                    solver="lbfgs",
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=SEED,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    return pipeline


def get_feature_scales(train_df: pd.DataFrame) -> pd.Series:
    scales = train_df[RADIOMICS_FEATURES].std(ddof=0).replace(0, 1.0)
    return scales


def select_borderline_patients(test_df: pd.DataFrame, pipeline: Pipeline) -> list[SelectedPatient]:
    test_eval = test_df.copy()
    test_eval["proba_hpv_pos"] = pipeline.predict_proba(test_eval[FEATURES])[:, 1]
    test_eval["y_pred"] = (test_eval["proba_hpv_pos"] >= THRESHOLD).astype(int)
    test_eval["dist_to_boundary"] = (test_eval["proba_hpv_pos"] - THRESHOLD).abs()

    selected: list[SelectedPatient] = []

    false_negatives = test_eval[(test_eval[TARGET] == 1) & (test_eval["y_pred"] == 0)].copy()
    if not false_negatives.empty:
        row = false_negatives.sort_values("proba_hpv_pos", ascending=False).iloc[0]
        selected.append(
            SelectedPatient(
                label="false_negative_hpv_pos_near_boundary",
                patient_id=str(row["PatientID"]),
                y_true=int(row[TARGET]),
                y_pred=int(row["y_pred"]),
                proba_hpv_pos=float(row["proba_hpv_pos"]),
            )
        )

    true_positives = test_eval[(test_eval[TARGET] == 1) & (test_eval["y_pred"] == 1)].copy()
    if not true_positives.empty:
        row = true_positives.sort_values("dist_to_boundary", ascending=True).iloc[0]
        selected.append(
            SelectedPatient(
                label="true_positive_near_boundary",
                patient_id=str(row["PatientID"]),
                y_true=int(row[TARGET]),
                y_pred=int(row["y_pred"]),
                proba_hpv_pos=float(row["proba_hpv_pos"]),
            )
        )

    if len(selected) < 2:
        fallback = (
            test_eval.sort_values("dist_to_boundary", ascending=True)
            .loc[lambda df: ~df["PatientID"].isin([s.patient_id for s in selected])]
            .head(2 - len(selected))
        )
        for i, (_, row) in enumerate(fallback.iterrows(), start=1):
            selected.append(
                SelectedPatient(
                    label=f"boundary_fallback_{i}",
                    patient_id=str(row["PatientID"]),
                    y_true=int(row[TARGET]),
                    y_pred=int(row["y_pred"]),
                    proba_hpv_pos=float(row["proba_hpv_pos"]),
                )
            )

    return selected


def generate_counterfactuals(
    dice_ml,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    pipeline: Pipeline,
    selected_patients: Iterable[SelectedPatient],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dice_data = dice_ml.Data(
        dataframe=train_df[FEATURES + [TARGET]].copy(),
        continuous_features=RADIOMICS_FEATURES,
        categorical_features=CLINICAL_FEATURES,
        outcome_name=TARGET,
    )
    dice_model = dice_ml.Model(
        model=pipeline,
        backend="sklearn",
        model_type="classifier",
    )
    explainer = dice_ml.Dice(dice_data, dice_model, method="random")

    selected_rows: list[dict[str, object]] = []
    shifts: list[dict[str, object]] = []
    transitions: list[dict[str, object]] = []

    for patient in selected_patients:
        query = (
            test_df.loc[test_df["PatientID"] == patient.patient_id, ["PatientID"] + FEATURES]
            .copy()
            .reset_index(drop=True)
        )
        if query.empty:
            continue

        query_features = query[FEATURES].copy()

        cf = explainer.generate_counterfactuals(
            query_instances=query_features,
            total_CFs=2,
            desired_class="opposite",
            features_to_vary=RADIOMICS_FEATURES,
            random_seed=SEED,
        )

        cf_examples = cf.cf_examples_list[0].final_cfs_df.copy()
        cf_examples = cf_examples.reset_index(drop=True)

        selected_rows.append(
            {
                "label": patient.label,
                "PatientID": patient.patient_id,
                "y_true": patient.y_true,
                "y_pred": patient.y_pred,
                "proba_hpv_pos": round(patient.proba_hpv_pos, 6),
            }
        )

        baseline = query.iloc[0]
        for cf_idx, cf_row in cf_examples.iterrows():
            cf_features = cf_row[FEATURES].to_frame().T
            cf_proba = float(pipeline.predict_proba(cf_features)[0, 1])
            transitions.append(
                {
                    "label": patient.label,
                    "PatientID": patient.patient_id,
                    "cf_index": cf_idx + 1,
                    "baseline_proba_hpv_pos": round(patient.proba_hpv_pos, 6),
                    "counterfactual_proba_hpv_pos": round(cf_proba, 6),
                    "proba_delta": round(cf_proba - patient.proba_hpv_pos, 6),
                }
            )
            for feature in FEATURES:
                base_val = baseline[feature]
                cf_val = cf_row[feature]
                if pd.isna(base_val) and pd.isna(cf_val):
                    continue
                if feature == "Gender_Male":
                    changed = bool(base_val != cf_val)
                    delta = None
                else:
                    changed = abs(float(cf_val) - float(base_val)) > 1e-12
                    delta = float(cf_val) - float(base_val)
                if changed:
                    shifts.append(
                        {
                            "label": patient.label,
                            "PatientID": patient.patient_id,
                            "cf_index": cf_idx + 1,
                            "feature": feature,
                            "baseline_value": base_val,
                            "counterfactual_value": cf_val,
                            "delta": delta,
                            "baseline_proba_hpv_pos": round(patient.proba_hpv_pos, 6),
                            "counterfactual_proba_hpv_pos": round(cf_proba, 6),
                        }
                    )

    selected_df = pd.DataFrame(selected_rows)
    shift_df = pd.DataFrame(shifts)
    transitions_df = pd.DataFrame(transitions)
    return selected_df, shift_df, transitions_df


def summarise_shifts(shift_df: pd.DataFrame) -> pd.DataFrame:
    if shift_df.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "n_changes",
                "patients_changed",
                "mean_abs_delta",
                "median_abs_delta",
                "mean_delta",
            ]
        )

    tmp = shift_df.copy()
    tmp["abs_delta"] = tmp["delta"].abs()
    summary = (
        tmp.groupby("feature", as_index=False)
        .agg(
            n_changes=("feature", "size"),
            patients_changed=("PatientID", "nunique"),
            mean_abs_delta=("abs_delta", "mean"),
            median_abs_delta=("abs_delta", "median"),
            mean_delta=("delta", "mean"),
        )
        .sort_values(["n_changes", "mean_abs_delta"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return summary


def add_standardized_deltas(shift_df: pd.DataFrame, feature_scales: pd.Series) -> pd.DataFrame:
    if shift_df.empty:
        shift_df["standardized_delta"] = pd.Series(dtype=float)
        shift_df["abs_standardized_delta"] = pd.Series(dtype=float)
        return shift_df

    out = shift_df.copy()
    out["feature_scale_train"] = out["feature"].map(feature_scales)
    out["standardized_delta"] = out["delta"] / out["feature_scale_train"]
    out["abs_standardized_delta"] = out["standardized_delta"].abs()
    return out


def summarise_standardized_shifts(shift_df: pd.DataFrame) -> pd.DataFrame:
    if shift_df.empty:
        return pd.DataFrame(
            columns=[
                "feature",
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

    tmp = shift_df.copy()
    summary = (
        tmp.groupby("feature", as_index=False)
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
        .sort_values(
            ["n_changes", "mean_abs_standardized_delta"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )
    return summary


def plot_feature_shift_summary(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        return

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    plot_df = summary_df.sort_values("mean_abs_standardized_delta", ascending=True)
    vals = plot_df["mean_abs_standardized_delta"].to_numpy()
    labels = [feature_label(f) for f in plot_df["feature"]]
    ax.barh(labels, vals, color="#2f6c8f", edgecolor="none", height=0.65)
    ax.set_xlabel("Mean absolute standardized shift")
    ax.set_ylabel("Feature")
    ax.set_title("Counterfactual feature-shift summary")
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.35)
    for y, v in enumerate(vals):
        ax.text(v + 0.04, y, f"{v:.2f}", va="center", ha="left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FEATURE_SHIFT_BAR_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_probability_transitions(transitions_df: pd.DataFrame) -> None:
    if transitions_df.empty:
        return

    order = transitions_df[["label", "PatientID"]].drop_duplicates().reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for x_idx, row in order.iterrows():
        mask = transitions_df["PatientID"] == row["PatientID"]
        patient_rows = transitions_df.loc[mask].sort_values("cf_index")
        baseline = patient_rows["baseline_proba_hpv_pos"].iloc[0]
        ax.scatter([x_idx], [baseline], color="black", s=62, label="Observed" if x_idx == 0 else None, zorder=4)
        for _, cf_row in patient_rows.iterrows():
            ax.scatter(
                [x_idx],
                [cf_row["counterfactual_proba_hpv_pos"]],
                color="#c75b39",
                s=62,
                label="Counterfactual route" if x_idx == 0 and cf_row["cf_index"] == 1 else None,
                zorder=4,
            )
            ax.plot(
                [x_idx, x_idx],
                [baseline, cf_row["counterfactual_proba_hpv_pos"]],
                color="#8f8f8f",
                linewidth=1.8,
                alpha=0.8,
            )
            ax.text(
                x_idx + 0.08,
                cf_row["counterfactual_proba_hpv_pos"],
                f"Route {chr(ord('A') + int(cf_row['cf_index']) - 1)}",
                fontsize=8,
                va="center",
                ha="left",
                color="#6b3929",
            )
    ax.axhline(THRESHOLD, color="#1f77b4", linestyle="--", linewidth=1.5, label="Decision threshold")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([patient_group_label(r["PatientID"], r["label"]) for _, r in order.iterrows()], rotation=0)
    ax.set_ylabel("Predicted HPV+ probability")
    ax.set_ylim(0, 1.02)
    ax.set_title("Observed-to-counterfactual probability transitions")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(PROBA_TRANSITIONS_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_shift_heatmap(shift_df: pd.DataFrame) -> None:
    if shift_df.empty:
        return

    heatmap_df = shift_df.copy()
    heatmap_df["patient_cf"] = heatmap_df.apply(
        lambda r: patient_route_label(str(r["PatientID"]), str(r["label"]), int(r["cf_index"])),
        axis=1,
    )
    heatmap_df["feature_short"] = heatmap_df["feature"].map(feature_label)
    pivot = (
        heatmap_df.pivot_table(
            index="feature_short",
            columns="patient_cf",
            values="standardized_delta",
            aggfunc="mean",
            fill_value=0.0,
        )
        .reindex(index=heatmap_df.groupby("feature_short")["abs_standardized_delta"].mean().sort_values(ascending=False).index)
    )

    fig_width = max(8.4, 1.55 * pivot.shape[1] + 2.4)
    fig_height = max(4.8, 0.64 * pivot.shape[0] + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    vmax = float(pivot.abs().to_numpy().max()) if not pivot.empty else 1.0
    vmax = max(vmax, 1.0)
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=0)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Standardized counterfactual feature-shift map")
    ax.set_xlabel("Patient and alternative counterfactual route")
    ax.set_ylabel("Radiomic feature")

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
    fig.savefig(SHIFT_HEATMAP_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_signed_shifts_per_case(
    shift_df: pd.DataFrame,
    out_path: Path,
    top_n: int = 5,
) -> None:
    """Per-case per-route signed feature-shift bar chart (Figure 3.4.9b).

    Bar colour = modality (PT orange / CT blue / Clinical grey).
    Magenta bar edge = shift direction is SHAP-consistent (moves feature
      in the direction that should help cross the DiCE threshold).
    Panel title shows patient, route, and probability before → after.
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
        needs_hpv_pos_up = "false_negative" in lbl

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
                elif needs_hpv_pos_up:
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

            proba_b = float(sub["baseline_proba_hpv_pos"].iloc[0])
            proba_c = float(sub["counterfactual_proba_hpv_pos"].iloc[0])
            arrow = "↑" if proba_c > proba_b else "↓"
            proba_str = f"P(HPV+) {proba_b:.3f} {arrow} {proba_c:.3f}"
            route_char = chr(ord("A") + col_idx - 1)
            ax.set_title(
                f"{pid}  |  Route {route_char}\n{proba_str}",
                fontsize=8.5, pad=5,
            )
            ax.set_xlabel(
                "Standardised shift (SD units)\n+ feature increased   − feature decreased",
                fontsize=7.5,
            )
            ax.tick_params(axis="y", labelsize=7.5)
            ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.4)
            ax.spines["left"].set_visible(False)

            xmax = float(np.nanmax(np.abs(vals))) if len(vals) else 1.0
            xmax = xmax if xmax > 0 else 1.0
            ax.set_xlim(-xmax * 1.6, xmax * 1.6)
            for y_pos, v in enumerate(vals):
                sym = "↑" if v > 0 else "↓"
                ha = "left" if v >= 0 else "right"
                offset = xmax * 0.09 if v >= 0 else -xmax * 0.09
                ax.text(v + offset, y_pos, f"{sym}{abs(v):.2f}",
                        va="center", ha=ha, fontsize=7.0)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#f58518", label="PET feature"),
        Patch(facecolor="#4c78a8", label="CT feature"),
        Patch(facecolor="#aaaaaa", label="Clinical feature"),
        Patch(facecolor="white", edgecolor="#cc00cc", linewidth=2.25,
              label="SHAP-consistent direction (magenta edge)"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center", ncol=4,
        frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(
        "HCM CFE: per-case per-route signed feature shifts (top 5 by magnitude)",
        fontsize=10, y=1.01,
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    dice_ml = ensure_dice()
    ensure_output_dir()

    train_df = build_split(CT_TRAIN_FILE, PT_TRAIN_FILE)
    test_df = build_split(CT_TEST_FILE, PT_TEST_FILE)
    feature_scales = get_feature_scales(train_df)

    pipeline = fit_locked_pipeline(train_df)
    selected_patients = select_borderline_patients(test_df, pipeline)
    selected_df, shift_df, transitions_df = generate_counterfactuals(
        dice_ml=dice_ml,
        train_df=train_df,
        test_df=test_df,
        pipeline=pipeline,
        selected_patients=selected_patients,
    )
    shift_df = add_standardized_deltas(shift_df, feature_scales)
    summary_df = summarise_standardized_shifts(shift_df)

    plot_feature_shift_summary(summary_df)
    plot_probability_transitions(transitions_df)
    plot_shift_heatmap(shift_df)
    plot_signed_shifts_per_case(shift_df, SIGNED_SHIFTS_PNG)

    if not selected_df.empty:
        selected_df = pd.DataFrame([asdict(p) for p in selected_patients])
    selected_df.to_csv(SELECTED_PATIENTS_CSV, index=False)
    shift_df.to_csv(SHIFT_TABLE_CSV, index=False)
    summary_df.to_csv(SHIFT_SUMMARY_CSV, index=False)
    transitions_df.to_csv(PROBA_TRANSITIONS_CSV, index=False)

    print("=" * 80)
    print("SELECTED BORDERLINE TEST PATIENTS")
    print("=" * 80)
    print(selected_df.to_string(index=False))
    print()

    print("=" * 80)
    print("COUNTERFACTUAL FEATURE SHIFTS")
    print("=" * 80)
    if shift_df.empty:
        print("No feature changes were returned by DiCE.")
    else:
        print(shift_df.to_string(index=False))
        print()
        print("=" * 80)
        print("COUNTERFACTUAL FEATURE SHIFT SUMMARY")
        print("=" * 80)
        print(summary_df.to_string(index=False))

    print()
    print("Saved outputs:")
    print(f"- {SELECTED_PATIENTS_CSV}")
    print(f"- {SHIFT_TABLE_CSV}")
    print(f"- {SHIFT_SUMMARY_CSV}")
    print(f"- {PROBA_TRANSITIONS_CSV}")
    print(f"- {FEATURE_SHIFT_BAR_PNG}")
    print(f"- {PROBA_TRANSITIONS_PNG}")
    print(f"- {SHIFT_HEATMAP_PNG}")
    print(f"- {SIGNED_SHIFTS_PNG}")


if __name__ == "__main__":
    main()
