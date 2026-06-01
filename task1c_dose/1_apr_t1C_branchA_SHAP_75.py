"""
1_apr_t1C_branchA_SHAP_75.py

T1C / DAE N75 Post-Study - Branch A1: SHAP Interpretation
Locked winner: 1TC2298, n=18 (1 clinical + 7 PT + 4 CT + 6 dose)
Coach: ExtraSurvivalTrees(n_estimators=200, random_state=42)
Cohorts: Dev (75 pts, 13 events) + CHUS (44 pts, 8 events). No CHUP dose data.

Outputs:
  Mar_2026_task1C/1_apr_t1C_post_study_outputs_75/branchA/

Usage:
  python 1_apr_t1C_branchA_SHAP_75.py             # full run
  python 1_apr_t1C_branchA_SHAP_75.py --chus065   # fast mode: only add CHUS-065 waterfall
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    import shap
except ImportError:
    print("Installing shap...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shap", "-q"])
    import shap

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
T1_ROOT = ROOT.parent / "Mar_2026"
OUT_DIR = ROOT / "1_apr_t1C_post_study_outputs_75" / "branchA"

DEV_CSV = ROOT / "Dose_development_75.csv"
CHUS_CSV = ROOT / "Dose_external_CHUS.csv"
PT_DEV_FILE = T1_ROOT / "27_feb_PT_development.csv"
CT_DEV_FILE = T1_ROOT / "27_feb_CT_development.csv"
PT_EXT_FILE = T1_ROOT / "27_feb_PT_external.csv"
CT_EXT_FILE = T1_ROOT / "27_feb_CT_external.csv"
CLINICAL_FILE = (
    ROOT.parent / "Feb_2026" / "25_feb_clinical_reduced_dataset"
    / "25_feb_Processed_clinical_reduced.csv"
)

SEED = 42
N_EST = 200
LOCKED_OOF = 0.8482
LOCKED_CHUS = 0.834921
REPRO_TOL = 0.001
WINNER_ID = "1TC2298"
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
ALL_FEATURES = CLINICAL_FEATURES + PT_LOCKED + CT_LOCKED + DOSE_WINNER_6_PREFIXED


def make_surv(event, time):
    return Surv.from_arrays(
        event=np.asarray(event, dtype=bool),
        time=np.asarray(time, dtype=float),
    )


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
        modality = modality_of(feature)
        region = parts[0]
        transform = parts[1].replace("wavelet-", "wav-")
        metric = parts[-1]
        return f"{modality} {region} {transform} {metric}"
    return "_".join(parts[-2:]) if len(parts) >= 2 else clean


def modality_of(feature: str) -> str:
    if feature in CLINICAL_FEATURES:
        return "Clinical"
    if feature in PT_LOCKED:
        return "PET"
    if feature in CT_LOCKED:
        return "CT"
    if feature.startswith(DOSE_PREFIX):
        return "Dose"
    return "Unknown"


def region_of(feature: str) -> str:
    clean = feature.replace(DOSE_PREFIX, "")
    if feature in CLINICAL_FEATURES:
        return "Clinical"
    if "GTVp" in clean:
        return "GTVp"
    if "GTVn" in clean:
        return "GTVn"
    return "Unknown"


def prefixed_dose_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [feature for feature in DOSE_WINNER_6 if feature not in df.columns]
    if missing:
        raise KeyError(f"Missing locked dose features in {path}: {missing}")
    out = df[["PatientID"] + DOSE_WINNER_6].copy()
    return out.rename(columns={feature: DOSE_PREFIX + feature for feature in DOSE_WINNER_6})


def load_t1c_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    clinical = pd.read_csv(CLINICAL_FILE).dropna(subset=["Relapse", "RFS"])
    clin_dev = clinical[clinical["Cohort"] == "Dev"][
        ["PatientID", "CenterID", "Relapse", "RFS"] + CLINICAL_FEATURES
    ].copy()
    clin_chus = clinical[clinical["CenterID"] == 3][
        ["PatientID", "CenterID", "Relapse", "RFS"] + CLINICAL_FEATURES
    ].copy()

    dose_dev = prefixed_dose_frame(DEV_CSV)
    dose_chus = prefixed_dose_frame(CHUS_CSV)

    pt_dev = pd.read_csv(PT_DEV_FILE)[["PatientID"] + PT_LOCKED]
    ct_dev = pd.read_csv(CT_DEV_FILE)[["PatientID"] + CT_LOCKED]
    pt_ext = pd.read_csv(PT_EXT_FILE)
    ct_ext = pd.read_csv(CT_EXT_FILE)
    pt_chus = pt_ext[pt_ext["PatientID"].str.startswith("CHUS")][["PatientID"] + PT_LOCKED]
    ct_chus = ct_ext[ct_ext["PatientID"].str.startswith("CHUS")][["PatientID"] + CT_LOCKED]

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


def build_est(n_jobs: int = -1) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("est", ExtraSurvivalTrees(n_estimators=N_EST, random_state=SEED, n_jobs=n_jobs)),
        ]
    )


def validate_repro(pipeline: Pipeline, chus_df: pd.DataFrame) -> tuple[np.ndarray, float]:
    x_chus = chus_df[ALL_FEATURES]
    y_chus = make_surv(chus_df["Relapse"], chus_df["RFS"])
    risk_chus = pipeline.predict(x_chus)
    ci_chus = safe_ci(y_chus, risk_chus)
    delta = abs(ci_chus - LOCKED_CHUS)
    print(f"Reproduced CHUS={ci_chus:.6f}  locked={LOCKED_CHUS:.6f}  delta={delta:.6f}")
    if delta > REPRO_TOL:
        raise RuntimeError(
            f"Reproducibility check FAILED for {WINNER_ID}. "
            f"Expected CHUS={LOCKED_CHUS:.6f}, got {ci_chus:.6f}."
        )
    return risk_chus, ci_chus


def save_beeswarm(tag: str, shap_values: np.ndarray, x_values: pd.DataFrame, n_events: int) -> None:
    plt.figure(figsize=(20, 9))
    shap.summary_plot(
        shap_values,
        x_values,
        feature_names=[feature_label(feature) for feature in ALL_FEATURES],
        show=False,
        plot_type="dot",
        max_display=18,
    )
    plt.title(
        f"DAE - SHAP Beeswarm, {tag.upper()} Cohort "
        f"(N={len(x_values)}, events={n_events}, {WINNER_ID})",
        fontsize=11,
    )
    plt.tight_layout()
    out_path = OUT_DIR / f"T1C_SHAP_beeswarm_{tag}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved: {out_path.name}")


def save_mean_abs_bar(summary_df: pd.DataFrame) -> None:
    plot_df = summary_df.sort_values("mean_abs_shap_dev", ascending=False).reset_index(drop=True)
    x = np.arange(len(plot_df))
    width = 0.32
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        x - width / 2,
        plot_df["mean_abs_shap_dev"],
        width,
        label="Dev",
        color="#1f77b4",
        alpha=0.85,
    )
    ax.bar(
        x + width / 2,
        plot_df["mean_abs_shap_chus"],
        width,
        label="CHUS",
        color="#ff7f0e",
        alpha=0.85,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["feature_label"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("mean(|SHAP value|)", fontsize=10)
    ax.set_title(
        f"DAE - Mean |SHAP| per Feature across Development and CHUS Cohorts ({WINNER_ID})",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    out_path = OUT_DIR / "T1C_SHAP_bar_mean_abs_shap.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved: {out_path.name}")


def save_dependence_top4(x_dev: pd.DataFrame, shap_dev: np.ndarray, summary_df: pd.DataFrame) -> None:
    top_features = summary_df.sort_values("mean_abs_shap_dev", ascending=False)["feature"].head(4).tolist()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, feature in zip(axes.ravel(), top_features):
        idx = ALL_FEATURES.index(feature)
        shap.dependence_plot(
            idx,
            shap_dev,
            x_dev,
            feature_names=[feature_label(item) for item in ALL_FEATURES],
            ax=ax,
            show=False,
        )
        ax.set_title(feature_label(feature), fontsize=9)
    plt.suptitle(
        f"DAE - SHAP Dependence Plots for Top 4 Features "
        f"(Development Cohort, {WINNER_ID})",
        fontsize=11,
    )
    plt.tight_layout()
    out_path = OUT_DIR / "T1C_SHAP_dependence_top4_dev.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved: {out_path.name}")


def save_waterfall_like(
    label: str,
    patient_id: str,
    shap_row: np.ndarray,
    expected_value: float,
    data_row: np.ndarray,
) -> None:
    exp = shap.Explanation(
        values=shap_row,
        base_values=expected_value,
        data=data_row,
        feature_names=[feature_label(feature) for feature in ALL_FEATURES],
    )
    plt.figure(figsize=(15, 6))
    shap.waterfall_plot(exp, max_display=12, show=False)
    for txt in plt.gca().texts:
        txt.set_color("black")
        txt.set_fontsize(8)
    plt.title(
        f"DAE - SHAP Waterfall, {label.replace('_', ' ').title()} "
        f"Patient (Dev, PatientID={patient_id}, {WINNER_ID})",
        fontsize=10,
    )
    plt.tight_layout(pad=1.5)
    out_path = OUT_DIR / f"T1C_SHAP_waterfall_{label}_dev.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved: {out_path.name}")


def save_chus065_waterfall(
    chus_df: pd.DataFrame,
    shap_chus: np.ndarray,
    expected_value: float,
) -> None:
    mask = chus_df["PatientID"] == "CHUS-065"
    if not mask.any():
        print("  WARNING: CHUS-065 not found in CHUS cohort — skipping waterfall.")
        return
    idx = int(np.where(mask.values)[0][0])
    exp = shap.Explanation(
        values=shap_chus[idx],
        base_values=expected_value,
        data=chus_df[ALL_FEATURES].iloc[idx].to_numpy(dtype=float),
        feature_names=[feature_label(feature) for feature in ALL_FEATURES],
    )
    plt.figure(figsize=(15, 6))
    shap.waterfall_plot(exp, max_display=18, show=False)
    for txt in plt.gca().texts:
        txt.set_color("black")
        txt.set_fontsize(8)
    plt.title(
        f"DAE - SHAP Waterfall, CHUS-065 (External Cross-Task Linker, CHUS, {WINNER_ID})",
        fontsize=10,
    )
    plt.tight_layout(pad=1.5)
    out_path = OUT_DIR / "T1C_SHAP_waterfall_chus065.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved: {out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DAE Branch A1 SHAP")
    parser.add_argument(
        "--chus065",
        action="store_true",
        help="Fast mode: only add CHUS-065 waterfall (skips all other plots)",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 88)
    print("T1C N75 BRANCH A1 SHAP")
    print("=" * 88)

    dev_df, chus_df = load_t1c_frames()
    print(
        f"Dev={len(dev_df)} ({int(dev_df['Relapse'].sum())} events) | "
        f"CHUS={len(chus_df)} ({int(chus_df['Relapse'].sum())} events)"
    )

    x_dev = dev_df[ALL_FEATURES]
    y_dev = make_surv(dev_df["Relapse"], dev_df["RFS"])
    x_chus = chus_df[ALL_FEATURES]

    pipeline = build_est()
    pipeline.fit(x_dev, y_dev)
    risk_dev = pipeline.predict(x_dev)
    _, ci_chus = validate_repro(pipeline, chus_df)

    print("Computing SHAP values with PermutationExplainer...")
    background = x_dev.copy()
    explainer = shap.PermutationExplainer(pipeline.predict, background)

    if not args.chus065:
        exp_dev_obj = explainer(x_dev)
        shap_dev = np.asarray(exp_dev_obj.values, dtype=float)

    exp_chus_obj = explainer(x_chus)
    shap_chus = np.asarray(exp_chus_obj.values, dtype=float)

    expected_value = float(
        np.mean(exp_chus_obj.base_values)
        if hasattr(exp_chus_obj, "base_values")
        else float(explainer.expected_value)
    )

    if not args.chus065:
        save_beeswarm("dev", shap_dev, x_dev, int(dev_df["Relapse"].sum()))
        save_beeswarm("chus", shap_chus, x_chus, int(chus_df["Relapse"].sum()))

        summary_df = pd.DataFrame(
            {
                "feature": ALL_FEATURES,
                "feature_label": [feature_label(feature) for feature in ALL_FEATURES],
                "modality": [modality_of(feature) for feature in ALL_FEATURES],
                "region": [region_of(feature) for feature in ALL_FEATURES],
                "mean_abs_shap_dev": np.abs(shap_dev).mean(axis=0),
                "mean_abs_shap_chus": np.abs(shap_chus).mean(axis=0),
            }
        ).sort_values("mean_abs_shap_dev", ascending=False)
        summary_df.to_csv(OUT_DIR / "T1C_SHAP_summary_table.csv", index=False)
        print("  Saved: T1C_SHAP_summary_table.csv")

        rows = []
        for cohort, shap_values in [("dev", shap_dev), ("chus", shap_chus)]:
            abs_mean = np.abs(shap_values).mean(axis=0)
            for grouping_name, grouping_func in [("modality", modality_of), ("region", region_of)]:
                tmp = pd.DataFrame(
                    {
                        "group": [grouping_func(feature) for feature in ALL_FEATURES],
                        "mean_abs": abs_mean,
                    }
                )
                total = float(tmp["mean_abs"].sum())
                for group, value in tmp.groupby("group")["mean_abs"].sum().items():
                    rows.append(
                        {
                            "cohort": cohort,
                            "grouping": grouping_name,
                            "group": group,
                            "mean_abs_shap_sum": value,
                            "percent_total": 100.0 * value / total if total else 0.0,
                        }
                    )
        contrib_df = pd.DataFrame(rows)
        contrib_df.to_csv(OUT_DIR / "T1C_SHAP_modality_region_contribution.csv", index=False)
        print("  Saved: T1C_SHAP_modality_region_contribution.csv")

        save_mean_abs_bar(summary_df)
        save_dependence_top4(x_dev, shap_dev, summary_df)

        high_idx = int(np.argmax(risk_dev))
        low_idx = int(np.argmin(risk_dev))
        med_idx = int(np.argsort(np.abs(risk_dev - np.median(risk_dev)))[0])
        save_waterfall_like(
            "high_risk",
            str(dev_df.iloc[high_idx]["PatientID"]),
            shap_dev[high_idx],
            expected_value,
            x_dev.iloc[high_idx].to_numpy(dtype=float),
        )
        save_waterfall_like(
            "low_risk",
            str(dev_df.iloc[low_idx]["PatientID"]),
            shap_dev[low_idx],
            expected_value,
            x_dev.iloc[low_idx].to_numpy(dtype=float),
        )
        save_waterfall_like(
            "median_risk",
            str(dev_df.iloc[med_idx]["PatientID"]),
            shap_dev[med_idx],
            expected_value,
            x_dev.iloc[med_idx].to_numpy(dtype=float),
        )

    # Always plot CHUS-065 waterfall (in both full and --chus065 modes)
    save_chus065_waterfall(chus_df, shap_chus, expected_value)

    print("\nComplete.")
    print(f"Winner={WINNER_ID} trial_no={TRIAL_NO} OOF={LOCKED_OOF:.4f} CHUS={ci_chus:.6f}")
    print(f"Outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
