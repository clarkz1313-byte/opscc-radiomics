"""
28_mar_task1_branchA3_DCA.py

Task 1 Post-Study — Branch A3: Decision Curve Analysis (DCA)
Locked winner: 1T40001, N=12 (1 clinical + 7 PT + 4 CT)
Coach: ExtraSurvivalTrees(n_estimators=200, random_state=42)

Purpose:
    Compute Decision Curve Analysis for the locked RPM winner across three
    cohorts: development (N=455), CHUS external (N=55), and CHUP external
    (N=35). Compares RPM net benefit against a clinical-only baseline,
    treat-all, and treat-none strategies.

    Event class: RFS relapse (Relapse=1). Treat-all prevalence = P(Relapse=1).
    Threshold range: 0.05 to 0.50 (step 0.01) — clinically actionable range
    for a survival risk-escalation decision.

Outputs (saved to Mar_2026/28_mar_task1_post_study_outputs/branchA3/):
    - T1_DCA_net_benefit_dev.png
    - T1_DCA_net_benefit_chus.png        (primary thesis figure)
    - T1_DCA_net_benefit_chup.png        (sensitivity — annotated)
    - T1_DCA_net_benefit_combined.png    (three-panel overview)
    - T1_DCA_net_benefit_table_dev.csv
    - T1_DCA_net_benefit_table_chus.csv
    - T1_DCA_net_benefit_table_chup.csv

Reproducibility guard (aborts if either fails):
    CHUS C-index = 0.742857142857143  (tolerance 0.001)
    CHUP C-index = 0.727586206896552  (tolerance 0.001)

Run:
    python Mar_2026/28_mar_task1_branchA3_DCA.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import ExtraSurvivalTrees
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "28_mar_task1_post_study_outputs" / "branchA3"

PT_DEV_FILE   = ROOT / "27_feb_PT_development.csv"
CT_DEV_FILE   = ROOT / "27_feb_CT_development.csv"
PT_EXT_FILE   = ROOT / "27_feb_PT_external.csv"
CT_EXT_FILE   = ROOT / "27_feb_CT_external.csv"
CLINICAL_FILE = (
    ROOT.parent / "Feb_2026" / "25_feb_clinical_reduced_dataset"
    / "25_feb_Processed_clinical_reduced.csv"
)
PT_FEAT_FILE  = ROOT / "2_mar_finalist_outputs" / "PT_inter1_768_features_recheck.csv"
CT_FEAT_FILE  = ROOT / "2_mar_finalist_outputs" / "CT_inter8_235_features_recheck.csv"

# ============================================================
# LOCKED MODEL CONSTANTS
# ============================================================
SEED = 42

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

# Clinical-only comparator feature pool
CLINICAL_ONLY_FEATURES = ["Age", "Gender_Male", "Treatment_CRT"]

LOCKED_CHUS   = 0.742857142857143
LOCKED_CHUP   = 0.727586206896552
REPRO_TOL     = 0.001

# DCA threshold range: 0.05–0.50, step 0.01 (clinically meaningful for risk escalation)
THRESHOLDS = np.arange(0.05, 0.51, 0.01)

# ============================================================
# PLOT STYLE
# ============================================================
plt.rcParams.update(
    {
        "figure.dpi": 300,
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

COLOUR_RPM        = "#2166ac"   # solid blue — RPM model (11 rad. + 1 clin.)
COLOUR_CLIN       = "#d95f02"   # dashed orange — clinical model
COLOUR_TREAT_ALL  = "#636363"   # dotted grey — treat all
COLOUR_TREAT_NONE = "black"     # solid black at y=0 — treat none


# ============================================================
# HELPERS
# ============================================================
def make_surv(event: np.ndarray, time: np.ndarray):
    return Surv.from_arrays(event=np.asarray(event, dtype=bool),
                            time=np.asarray(time, dtype=float))


def safe_ci(y, risk: np.ndarray) -> float:
    try:
        return float(concordance_index_censored(y["event"], y["time"], risk)[0])
    except Exception:
        return float("nan")


# ============================================================
# DATA LOADING  (mirrors recheck script merge logic exactly)
# ============================================================
def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pt_feat_pool = pd.read_csv(PT_FEAT_FILE)["Feature"].tolist()
    ct_feat_raw  = pd.read_csv(CT_FEAT_FILE)["Feature"].tolist()
    pt_set       = set(pt_feat_pool)
    ct_feat_pool = [f for f in ct_feat_raw if f not in pt_set]

    clinical = pd.read_csv(CLINICAL_FILE).dropna(subset=["Relapse", "RFS"])

    # Pull clinical columns — include both HCM and clinical-only pool columns
    extra_clin = [c for c in CLINICAL_ONLY_FEATURES if c not in CLINICAL_FEATURES]
    clin_cols = ["PatientID", "CenterID", "Cohort", "Relapse", "RFS"] + CLINICAL_FEATURES + extra_clin

    clin_dev  = clinical[clinical["Cohort"] == "Dev"][
        [c for c in clin_cols if c in clinical.columns]].copy()
    clin_chus = clinical[clinical["CenterID"] == 3][
        [c for c in clin_cols if c in clinical.columns]].copy()
    clin_chup = clinical[clinical["CenterID"] == 2][
        [c for c in clin_cols if c in clinical.columns]].copy()

    pt_dev = pd.read_csv(PT_DEV_FILE)
    ct_dev = pd.read_csv(CT_DEV_FILE)
    pt_ext = pd.read_csv(PT_EXT_FILE)
    ct_ext = pd.read_csv(CT_EXT_FILE)

    rad_dev = pt_dev[["PatientID"] + pt_feat_pool].merge(
        ct_dev[["PatientID"] + ct_feat_pool], on="PatientID", how="inner")
    rad_ext = pt_ext[["PatientID"] + pt_feat_pool].merge(
        ct_ext[["PatientID"] + ct_feat_pool], on="PatientID", how="inner")

    rad_chus = rad_ext[rad_ext["PatientID"].str.startswith("CHUS")]
    rad_chup = rad_ext[rad_ext["PatientID"].str.startswith("CHUP")]

    dev_df  = clin_dev.merge(rad_dev,  on="PatientID", how="inner")
    chus_df = clin_chus.merge(rad_chus, on="PatientID", how="inner")
    chup_df = clin_chup.merge(rad_chup, on="PatientID", how="inner")
    return dev_df, chus_df, chup_df


# ============================================================
# MODEL FITTING
# ============================================================
def fit_rpm_model(
    dev_df: pd.DataFrame,
) -> tuple[StandardScaler, ExtraSurvivalTrees]:
    """Fit the locked RPM model on development only."""
    X = dev_df[WINNER_FEATURES].values.astype(float)
    y = make_surv(dev_df["Relapse"], dev_df["RFS"])
    scaler = StandardScaler().fit(X)
    model = ExtraSurvivalTrees(n_estimators=200, random_state=SEED, n_jobs=-1)
    model.fit(scaler.transform(X), y)
    return scaler, model


def fit_clinical_model(
    dev_df: pd.DataFrame,
) -> tuple[StandardScaler, LogisticRegression, list[str]]:
    """Fit clinical-only LR comparator on development only."""
    available = [f for f in CLINICAL_ONLY_FEATURES if f in dev_df.columns]
    X = dev_df[available].values.astype(float)
    y = dev_df["Relapse"].values.astype(int)
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs",
        max_iter=2000, random_state=SEED,
    )
    model.fit(scaler.transform(X), y)
    return scaler, model, available


# ============================================================
# REPRODUCIBILITY CHECK
# ============================================================
def run_reproducibility_check(
    rpm_scaler: StandardScaler,
    rpm_model: ExtraSurvivalTrees,
    dev_df: pd.DataFrame,
    chus_df: pd.DataFrame,
    chup_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return risk scores for all cohorts; abort if C-index deviates."""
    def risk(df: pd.DataFrame) -> np.ndarray:
        X = rpm_scaler.transform(df[WINNER_FEATURES].values.astype(float))
        return rpm_model.predict(X)

    risk_dev  = risk(dev_df)
    risk_chus = risk(chus_df)
    risk_chup = risk(chup_df)

    y_chus = make_surv(chus_df["Relapse"], chus_df["RFS"])
    y_chup = make_surv(chup_df["Relapse"], chup_df["RFS"])
    ci_chus = safe_ci(y_chus, risk_chus)
    ci_chup = safe_ci(y_chup, risk_chup)

    d_chus = abs(ci_chus - LOCKED_CHUS)
    d_chup = abs(ci_chup - LOCKED_CHUP)

    print(f"\nReproducibility check:")
    print(f"  CHUS C-index = {ci_chus:.6f}  locked={LOCKED_CHUS:.6f}  delta={d_chus:.6f}")
    print(f"  CHUP C-index = {ci_chup:.6f}  locked={LOCKED_CHUP:.6f}  delta={d_chup:.6f}")

    if d_chus > REPRO_TOL or d_chup > REPRO_TOL:
        raise RuntimeError(
            f"Reproducibility check FAILED. "
            f"CHUS delta={d_chus:.6f} (tol={REPRO_TOL}), "
            f"CHUP delta={d_chup:.6f} (tol={REPRO_TOL})."
        )
    print("  Reproducibility check PASSED.\n")
    return risk_dev, risk_chus, risk_chup


# ============================================================
# RPM RISK -> PROBABILITY CONVERSION
# ============================================================
def risk_to_proba(risk_scores: np.ndarray) -> np.ndarray:
    """
    Convert raw ESurvivalTrees risk scores to [0,1] probabilities for DCA.
    Uses min-max normalisation fitted on the same array (dev) or applied
    directly per cohort after dev-fitted min/max.
    Called with dev-fitted bounds so no leakage occurs.
    """
    lo = float(risk_scores.min())
    hi = float(risk_scores.max())
    if hi == lo:
        return np.full_like(risk_scores, 0.5)
    return (risk_scores - lo) / (hi - lo)


# ============================================================
# DCA COMPUTATION
# ============================================================
def compute_net_benefit(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Standard Vickers-Elkin net benefit: NB(t) = TP/N - FP/N * t/(1-t)."""
    n = len(y_true)
    nb = np.empty(len(thresholds))
    for i, t in enumerate(thresholds):
        pred = (proba >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        odds = t / (1.0 - t) if t < 1.0 else np.inf
        nb[i] = tp / n - fp / n * odds
    return nb


def compute_treat_all_nb(
    y_true: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Net benefit of 'treat all as high-risk (relapse)' strategy."""
    n = len(y_true)
    prevalence = (y_true == 1).mean()  # P(Relapse=1)
    nb = np.empty(len(thresholds))
    for i, t in enumerate(thresholds):
        odds = t / (1.0 - t) if t < 1.0 else np.inf
        nb[i] = prevalence - (1.0 - prevalence) * odds
    return nb


def build_dca_table(
    y_true: np.ndarray,
    proba_rpm: np.ndarray,
    proba_clin: np.ndarray,
    thresholds: np.ndarray,
    cohort_label: str,
) -> pd.DataFrame:
    nb_rpm      = compute_net_benefit(y_true, proba_rpm,  thresholds)
    nb_clin     = compute_net_benefit(y_true, proba_clin, thresholds)
    nb_treat_all  = compute_treat_all_nb(y_true, thresholds)
    nb_treat_none = np.zeros(len(thresholds))
    return pd.DataFrame({
        "cohort":           cohort_label,
        "threshold":        np.round(thresholds, 4),
        "nb_rpm_model":     np.round(nb_rpm,      6),
        "nb_clinical_only": np.round(nb_clin,      6),
        "nb_treat_all":     np.round(nb_treat_all, 6),
        "nb_treat_none":    np.round(nb_treat_none,6),
    })


# ============================================================
# PLOTTING
# ============================================================
def _draw_dca_panel(
    ax: plt.Axes,
    dca_df: pd.DataFrame,
    title: str,
    show_annotation: bool = False,
    n_patients: int | None = None,
    n_events: int | None = None,
    annotation_note: str | None = None,
) -> None:
    t       = dca_df["threshold"].to_numpy()
    nb_rpm  = dca_df["nb_rpm_model"].to_numpy()
    nb_clin = dca_df["nb_clinical_only"].to_numpy()
    nb_all  = dca_df["nb_treat_all"].to_numpy()

    ax.plot(t, nb_rpm,  color=COLOUR_RPM,       linewidth=2.2,
            label="RPM model (11 rad.\u202f+\u202f1 clin.)")
    ax.plot(t, nb_clin, color=COLOUR_CLIN,      linewidth=1.8, linestyle="--",
            label="Clinical model (Age, Sex, Treatment)")
    ax.plot(t, nb_all,  color=COLOUR_TREAT_ALL, linewidth=1.4, linestyle=":",
            label="Treat all as high-risk (assume everyone relapses)")
    ax.axhline(0.0, color=COLOUR_TREAT_NONE, linewidth=1.2,
               label="Treat none (assume nobody relapses)")

    ax.set_xlabel("Threshold probability, $p_t$")
    ax.set_ylabel("Net benefit")
    ax.set_title(title)
    ax.set_xlim(t[0] - 0.01, t[-1] + 0.01)

    all_vals = np.concatenate([nb_rpm, nb_clin, nb_all, [0.0]])
    ymin = max(float(np.nanmin(all_vals)), -0.25)
    ymax = float(np.nanmax(all_vals))
    margin = 0.05 * max(abs(ymax - ymin), 0.05)
    ax.set_ylim(ymin - margin, ymax + margin)

    ax.legend(loc="lower left", frameon=True, framealpha=0.9, fontsize=9,
              edgecolor="lightgrey")

    if show_annotation and n_patients is not None:
        lines = [f"N={n_patients}"]
        if n_events is not None:
            lines.append(f"RFS events={n_events}")
        if annotation_note:
            lines.append(annotation_note)
        ax.text(
            0.98, 0.97, "\n".join(lines),
            transform=ax.transAxes, fontsize=8,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="lightyellow", alpha=0.8, edgecolor="grey"),
        )


def save_single_panel(
    dca_df: pd.DataFrame,
    title: str,
    out_path: Path,
    show_annotation: bool = False,
    n_patients: int | None = None,
    n_events: int | None = None,
    annotation_note: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    _draw_dca_panel(ax, dca_df, title, show_annotation, n_patients, n_events, annotation_note)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def save_combined_panel(
    dca_dev: pd.DataFrame,
    dca_chus: pd.DataFrame,
    dca_chup: pd.DataFrame,
    n_dev: int, ev_dev: int,
    n_chus: int, ev_chus: int,
    n_chup: int, ev_chup: int,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)

    _draw_dca_panel(
        axes[0], dca_dev,
        f"Development cohort (N={n_dev})",
    )
    _draw_dca_panel(
        axes[1], dca_chus,
        f"External validation: CHUS (N={n_chus})",
    )
    _draw_dca_panel(
        axes[2], dca_chup,
        f"External validation: CHUP (N={n_chup})",
    )

    # Keep legend only in Dev panel (axes[0]); remove from CHUS and CHUP panels
    for ax in [axes[1], axes[2]]:
        legend = ax.get_legend()
        if legend:
            legend.remove()

    fig.suptitle(
        "Decision Curve Analysis \u2014 RPM Retained Winner (1T40001) across "
        "Development, CHUS, and CHUP Cohorts",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    print("Loading data splits...")
    dev_df, chus_df, chup_df = load_splits()
    print(
        f"  Dev N={len(dev_df)} ({int(dev_df['Relapse'].sum())} events) | "
        f"CHUS N={len(chus_df)} ({int(chus_df['Relapse'].sum())} events) | "
        f"CHUP N={len(chup_df)} ({int(chup_df['Relapse'].sum())} events)"
    )

    # ---- Fit locked RPM model ----
    print("\nFitting locked RPM model (1T40001)...")
    rpm_scaler, rpm_model = fit_rpm_model(dev_df)

    # ---- Reproducibility check ----
    risk_dev, risk_chus, risk_chup = run_reproducibility_check(
        rpm_scaler, rpm_model, dev_df, chus_df, chup_df
    )

    # ---- Fit clinical-only comparator ----
    print("Fitting clinical-only comparator model...")
    clin_scaler, clin_model, clin_cols = fit_clinical_model(dev_df)
    print(f"  Clinical features used: {clin_cols}")

    # ---- Convert risk scores to probabilities (dev-fitted min-max) ----
    risk_lo = float(risk_dev.min())
    risk_hi = float(risk_dev.max())

    def norm(r: np.ndarray) -> np.ndarray:
        return np.clip((r - risk_lo) / (risk_hi - risk_lo), 0.0, 1.0)

    proba_rpm_dev  = norm(risk_dev)
    proba_rpm_chus = norm(risk_chus)
    proba_rpm_chup = norm(risk_chup)

    # ---- Clinical model probabilities ----
    def clin_proba(df: pd.DataFrame) -> np.ndarray:
        avail = [f for f in clin_cols if f in df.columns]
        if not avail:
            return np.full(len(df), dev_df["Relapse"].mean())
        X = df[avail].values.astype(float)
        if len(avail) < len(clin_cols):
            padded = np.zeros((len(df), len(clin_cols)))
            for j, col in enumerate(clin_cols):
                if col in avail:
                    padded[:, j] = X[:, avail.index(col)]
            X = padded
        return clin_model.predict_proba(clin_scaler.transform(X))[:, 1]

    proba_clin_dev  = clin_proba(dev_df)
    proba_clin_chus = clin_proba(chus_df)
    proba_clin_chup = clin_proba(chup_df)

    y_dev  = dev_df["Relapse"].values.astype(int)
    y_chus = chus_df["Relapse"].values.astype(int)
    y_chup = chup_df["Relapse"].values.astype(int)

    # ---- Compute DCA tables ----
    print("Computing DCA net benefit curves...")
    dca_dev  = build_dca_table(y_dev,  proba_rpm_dev,  proba_clin_dev,  THRESHOLDS, "dev")
    dca_chus = build_dca_table(y_chus, proba_rpm_chus, proba_clin_chus, THRESHOLDS, "chus")
    dca_chup = build_dca_table(y_chup, proba_rpm_chup, proba_clin_chup, THRESHOLDS, "chup")

    # ---- Print summary ----
    for label, dca_df in [("Dev", dca_dev), ("CHUS", dca_chus), ("CHUP", dca_chup)]:
        rpm_beats_all  = (dca_df["nb_rpm_model"] > dca_df["nb_treat_all"]).sum()
        rpm_beats_clin = (dca_df["nb_rpm_model"] > dca_df["nb_clinical_only"]).sum()
        total = len(dca_df)
        print(
            f"  {label}: RPM > Treat-All at {rpm_beats_all}/{total} thresholds; "
            f"RPM > Clinical-only at {rpm_beats_clin}/{total} thresholds"
        )

    # ---- Save CSV tables ----
    print("\nSaving CSV tables...")
    dca_dev.to_csv( OUT_DIR / "T1_DCA_net_benefit_table_dev.csv",  index=False)
    dca_chus.to_csv(OUT_DIR / "T1_DCA_net_benefit_table_chus.csv", index=False)
    dca_chup.to_csv(OUT_DIR / "T1_DCA_net_benefit_table_chup.csv", index=False)
    print("  Saved: T1_DCA_net_benefit_table_dev.csv")
    print("  Saved: T1_DCA_net_benefit_table_chus.csv")
    print("  Saved: T1_DCA_net_benefit_table_chup.csv")

    # ---- Save single-panel plots ----
    print("\nSaving plots...")
    save_single_panel(
        dca_dev,
        title=f"Decision Curve Analysis \u2014 Recurrence Prediction Model (RPM), "
              f"Development Cohort (N={len(dev_df)})",
        out_path=OUT_DIR / "T1_DCA_net_benefit_dev.png",
    )
    save_single_panel(
        dca_chus,
        title=f"Decision Curve Analysis \u2014 Recurrence Prediction Model (RPM), "
              f"External Validation: CHUS (N={len(chus_df)})",
        out_path=OUT_DIR / "T1_DCA_net_benefit_chus.png",
    )
    save_single_panel(
        dca_chup,
        title=f"Decision Curve Analysis \u2014 Recurrence Prediction Model (RPM), "
              f"External Validation: CHUP (N={len(chup_df)})",
        out_path=OUT_DIR / "T1_DCA_net_benefit_chup.png",
    )
    save_combined_panel(
        dca_dev, dca_chus, dca_chup,
        n_dev=len(dev_df),   ev_dev=int(dev_df["Relapse"].sum()),
        n_chus=len(chus_df), ev_chus=int(chus_df["Relapse"].sum()),
        n_chup=len(chup_df), ev_chup=int(chup_df["Relapse"].sum()),
        out_path=OUT_DIR / "T1_DCA_net_benefit_combined.png",
    )

    print("\n" + "=" * 72)
    print("BRANCH A3 COMPLETE")
    print("=" * 72)
    print(f"All outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
