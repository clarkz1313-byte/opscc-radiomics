"""
1_apr_t1C_branchB_KM_75.py

T1C / DAE N75 Post-Study - Branch B: Kaplan-Meier Risk Stratification
Locked winner: 1TC2298, n=18 (1 clinical + 7 PT + 4 CT + 6 dose)
Coach: ExtraSurvivalTrees(n_estimators=200, random_state=42)
Cohorts: Dev (75 pts, 13 events) + CHUS (44 pts, 8 events). No CHUP dose data.

Outputs:
  Mar_2026_task1C/1_apr_t1C_post_study_outputs_75/branchB/
"""

from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    import lifelines  # noqa: F401
except ImportError:
    print("Installing lifelines...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lifelines", "-q"])

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import ExtraSurvivalTrees
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv


ROOT = Path(__file__).resolve().parent
T1_ROOT = ROOT.parent / "Mar_2026"
OUT_DIR = ROOT / "1_apr_t1C_post_study_outputs_75" / "branchB"

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
DAYS_TO_MONTHS = 1 / 30.44
LOCKED_OOF = 0.8482
LOCKED_CHUS = 0.834921
REPRO_TOL = 0.001
WINNER_ID = "1TC2298"
TRIAL_NO = 2298
DOSE_PREFIX = "DOSE__"

COLOR_HIGH = "#d62728"
COLOR_LOW = "#1f77b4"

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


def compute_hr(df_cohort: pd.DataFrame, risk_scores: np.ndarray, threshold: float) -> tuple[float, float, float, float]:
    group = (risk_scores >= threshold).astype(int)
    cox_df = pd.DataFrame(
        {
            "duration": df_cohort["RFS"].to_numpy(dtype=float) * DAYS_TO_MONTHS,
            "event": df_cohort["Relapse"].to_numpy(dtype=int),
            "high_risk": group,
        }
    )
    try:
        cph = CoxPHFitter()
        cph.fit(cox_df, duration_col="duration", event_col="event", formula="high_risk", show_progress=False)
        summary = cph.summary
        hr = float(np.exp(summary.loc["high_risk", "coef"]))
        hr_lo = float(np.exp(summary.loc["high_risk", "coef lower 95%"]))
        hr_hi = float(np.exp(summary.loc["high_risk", "coef upper 95%"]))
        p_cox = float(summary.loc["high_risk", "p"])
    except Exception as exc:
        print(f"  Cox HR failed: {exc}")
        hr, hr_lo, hr_hi, p_cox = float("nan"), float("nan"), float("nan"), float("nan")
    return hr, hr_lo, hr_hi, p_cox


def plot_km(
    df_cohort: pd.DataFrame,
    risk_scores: np.ndarray,
    threshold: float,
    cohort_name: str,
    out_path: Path,
    ci_val: float,
) -> tuple[int, int, float]:
    group = (risk_scores >= threshold).astype(int)
    n_high = int(group.sum())
    n_low = int((group == 0).sum())
    duration_months = df_cohort["RFS"].to_numpy(dtype=float) * DAYS_TO_MONTHS
    event = df_cohort["Relapse"].to_numpy(dtype=bool)

    lr = logrank_test(
        duration_months[group == 1],
        duration_months[group == 0],
        event_observed_A=event[group == 1],
        event_observed_B=event[group == 0],
    )
    p_val = float(lr.p_value)

    fig, ax = plt.subplots(figsize=(7, 5))
    for grp_val, label, color in [
        (1, f"High risk (n={n_high})", COLOR_HIGH),
        (0, f"Low risk (n={n_low})", COLOR_LOW),
    ]:
        mask = group == grp_val
        if int(mask.sum()) == 0:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(duration_months[mask], event_observed=event[mask], label=label)
        kmf.plot_survival_function(ax=ax, ci_show=True, color=color, linewidth=2)

    p_text = f"p = {p_val:.4f}" if p_val >= 0.0001 else "p < 0.0001"
    try:
        cox_df_inner = pd.DataFrame({
            "duration": df_cohort["RFS"].to_numpy(dtype=float) * DAYS_TO_MONTHS,
            "event": df_cohort["Relapse"].to_numpy(dtype=int),
            "high_risk": group,
        })
        cph_inner = CoxPHFitter()
        cph_inner.fit(cox_df_inner, duration_col="duration", event_col="event",
                      formula="high_risk", show_progress=False)
        hr_v = float(np.exp(cph_inner.summary.loc["high_risk", "coef"]))
        hr_lo_v = float(np.exp(cph_inner.summary.loc["high_risk", "coef lower 95%"]))
        hr_hi_v = float(np.exp(cph_inner.summary.loc["high_risk", "coef upper 95%"]))
        if hr_v > 999:
            hr_text = "HR>>1 (near-complete sep.)"
        else:
            hr_text = f"HR={hr_v:.2f} ({hr_lo_v:.2f}–{hr_hi_v:.2f})"
    except Exception:
        hr_text = "HR=N/A"
    ax.text(
        0.97,
        0.97,
        f"{p_text}\n{hr_text}\nC-index = {ci_val:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
    )
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Relapse-free survival probability")
    ax.set_title(f"T1C N75 Kaplan-Meier - {cohort_name} ({WINNER_ID}, N=18)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return n_high, n_low, p_val


def run_km_pass(
    cohort_specs: list[tuple[pd.DataFrame, np.ndarray, float, str, float]],
    suffix: str,
    out_2panel: Path,
    out_table: Path,
) -> pd.DataFrame:
    rows = []
    for df_c, risk_c, threshold_c, name, ci_c in cohort_specs:
        out_path = OUT_DIR / f"T1C_KM_km_{name.lower()}{suffix}.png"
        n_high, n_low, p_lr = plot_km(df_c, risk_c, threshold_c, name, out_path, ci_c)
        hr, hr_lo, hr_hi, p_cox = compute_hr(df_c, risk_c, threshold_c)
        rows.append(
            {
                "cohort": name,
                "n_total": len(df_c),
                "n_events": int(df_c["Relapse"].sum()),
                "n_high_risk": n_high,
                "n_low_risk": n_low,
                "threshold": round(threshold_c, 6),
                "logrank_p": round(p_lr, 6),
                "HR": round(hr, 4),
                "HR_CI_lower": round(hr_lo, 4),
                "HR_CI_upper": round(hr_hi, 4),
                "cox_p": round(p_cox, 6),
                "c_index": round(ci_c, 6),
            }
        )
        print(
            f"  {name}: n_high={n_high}, n_low={n_low}, "
            f"logrank_p={p_lr:.4f}, HR={hr:.2f}, C-index={ci_c:.4f}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (df_c, risk_c, threshold_c, name, ci_c) in zip(axes, cohort_specs):
        group = (risk_c >= threshold_c).astype(int)
        duration_months = df_c["RFS"].to_numpy(dtype=float) * DAYS_TO_MONTHS
        event = df_c["Relapse"].to_numpy(dtype=bool)
        lr = logrank_test(
            duration_months[group == 1],
            duration_months[group == 0],
            event_observed_A=event[group == 1],
            event_observed_B=event[group == 0],
        )
        for grp_val, label, color in [
            (1, f"High risk (n={int(group.sum())})", COLOR_HIGH),
            (0, f"Low risk (n={int((group == 0).sum())})", COLOR_LOW),
        ]:
            mask = group == grp_val
            if int(mask.sum()) == 0:
                continue
            kmf = KaplanMeierFitter()
            kmf.fit(duration_months[mask], event_observed=event[mask], label=label)
            kmf.plot_survival_function(ax=ax, ci_show=True, color=color, linewidth=2)
        p_text = f"p = {lr.p_value:.4f}" if lr.p_value >= 0.0001 else "p < 0.0001"
        try:
            _cox_df = pd.DataFrame({
                "duration": df_c["RFS"].to_numpy(dtype=float) * DAYS_TO_MONTHS,
                "event": df_c["Relapse"].to_numpy(dtype=int),
                "high_risk": group,
            })
            _cph = CoxPHFitter()
            _cph.fit(_cox_df, duration_col="duration", event_col="event",
                     formula="high_risk", show_progress=False)
            _hr = float(np.exp(_cph.summary.loc["high_risk", "coef"]))
            _lo = float(np.exp(_cph.summary.loc["high_risk", "coef lower 95%"]))
            _hi = float(np.exp(_cph.summary.loc["high_risk", "coef upper 95%"]))
            if _hr > 999:
                hr_text = "HR>>1 (near-complete sep.)"
            else:
                hr_text = f"HR={_hr:.2f} ({_lo:.2f}–{_hi:.2f})"
        except Exception:
            hr_text = "HR=N/A"
        ax.text(
            0.97,
            0.97,
            f"{p_text}\n{hr_text}\nC-index = {ci_c:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
        )
        ax.set_title(name)
        ax.set_xlabel("Time (months)")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc="lower left")
    axes[0].set_ylabel("Relapse-free survival probability")
    fig.suptitle(f"T1C N75 KM risk stratification - {WINNER_ID}", y=1.02)
    fig.tight_layout()
    fig.savefig(out_2panel, dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_table, index=False)
    print(f"  Saved: {out_2panel.name}")
    print(f"  Saved: {out_table.name}")
    return summary_df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 88)
    print("T1C N75 BRANCH B KM")
    print("=" * 88)

    dev_df, chus_df = load_t1c_frames()
    print(
        f"Dev={len(dev_df)} ({int(dev_df['Relapse'].sum())} events) | "
        f"CHUS={len(chus_df)} ({int(chus_df['Relapse'].sum())} events)"
    )

    x_dev = dev_df[ALL_FEATURES]
    y_dev = make_surv(dev_df["Relapse"], dev_df["RFS"])
    x_chus = chus_df[ALL_FEATURES]
    y_chus = make_surv(chus_df["Relapse"], chus_df["RFS"])

    pipeline = build_est()
    pipeline.fit(x_dev, y_dev)
    risk_dev = pipeline.predict(x_dev)
    risk_chus = pipeline.predict(x_chus)
    ci_chus = safe_ci(y_chus, risk_chus)
    delta = abs(ci_chus - LOCKED_CHUS)
    print(f"Reproduced CHUS={ci_chus:.6f}  locked={LOCKED_CHUS:.6f}  delta={delta:.6f}")
    if delta > REPRO_TOL:
        raise RuntimeError(f"Reproducibility check FAILED for {WINNER_ID}: CHUS={ci_chus:.6f}")

    threshold_dev = float(np.median(risk_dev))
    threshold_chus = float(np.median(risk_chus))
    print(f"Dev median threshold={threshold_dev:.6f}")
    print(f"CHUS median threshold={threshold_chus:.6f}")

    ci_dev = safe_ci(y_dev, risk_dev)

    print("\nPrimary: cohort-specific median split")
    run_km_pass(
        [
            (dev_df, risk_dev, threshold_dev, "Dev", ci_dev),
            (chus_df, risk_chus, threshold_chus, "CHUS", ci_chus),
        ],
        suffix="",
        out_2panel=OUT_DIR / "T1C_KM_km_combined_2panel.png",
        out_table=OUT_DIR / "T1C_KM_km_summary_table.csv",
    )

    print("\nSecondary: dev-threshold transport")
    run_km_pass(
        [
            (dev_df, risk_dev, threshold_dev, "Dev", ci_dev),
            (chus_df, risk_chus, threshold_dev, "CHUS", ci_chus),
        ],
        suffix="_devthresh",
        out_2panel=OUT_DIR / "T1C_KM_km_combined_2panel_devthresh.png",
        out_table=OUT_DIR / "T1C_KM_km_summary_table_devthresh.csv",
    )

    print("\nComplete.")
    print(f"Winner={WINNER_ID} trial_no={TRIAL_NO} OOF={LOCKED_OOF:.4f} CHUS={ci_chus:.6f}")
    print(f"Outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
