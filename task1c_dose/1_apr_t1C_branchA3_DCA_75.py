"""
1_apr_t1C_branchA3_DCA.py

Task 1C / DAE Post-Study - Branch A3: Decision Curve Analysis (DCA)
Locked winner: 1TC2298, N=18 (1 clinical + 7 PT + 4 CT + 6 dose)
Coach: ExtraSurvivalTrees(n_estimators=200, random_state=42)

Purpose:
    Compute Decision Curve Analysis for the locked N75 DAE winner across
    development (N=75) and CHUS external (N=44). Compares DAE net benefit
    against a no-dose k=0 baseline, clinical-only baseline, treat-all,
    and treat-none strategies.

    Event class: RFS relapse (Relapse=1). Treat-all prevalence = P(Relapse=1).
    Threshold range: 0.05 to 0.50 (step 0.01), matching the RPM Branch A3
    recurrence-risk escalation DCA.

Outputs (saved to Mar_2026_task1C/1_apr_t1C_post_study_outputs_75/branchA3/):
    - T1C_DCA_net_benefit_dev.png
    - T1C_DCA_net_benefit_chus.png
    - T1C_DCA_net_benefit_combined.png
    - T1C_DCA_net_benefit_table_dev.csv
    - T1C_DCA_net_benefit_table_chus.csv

Reproducibility guard:
    CHUS C-index = 0.834921 (tolerance 0.001)

Run:
    python Mar_2026_task1C/1_apr_t1C_branchA3_DCA.py
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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import ExtraSurvivalTrees
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================
ROOT = Path(__file__).resolve().parent
T1_ROOT = ROOT.parent / "Mar_2026"
OUT_DIR = ROOT / "1_apr_t1C_post_study_outputs_75" / "branchA3"

DOSE_DEV_FILE = ROOT / "Dose_development_75.csv"
DOSE_CHUS_FILE = ROOT / "Dose_external_CHUS.csv"
PT_DEV_FILE = T1_ROOT / "27_feb_PT_development.csv"
CT_DEV_FILE = T1_ROOT / "27_feb_CT_development.csv"
PT_EXT_FILE = T1_ROOT / "27_feb_PT_external.csv"
CT_EXT_FILE = T1_ROOT / "27_feb_CT_external.csv"
CLINICAL_FILE = (
    ROOT.parent / "Feb_2026" / "25_feb_clinical_reduced_dataset"
    / "25_feb_Processed_clinical_reduced.csv"
)

# ============================================================
# LOCKED MODEL CONSTANTS
# ============================================================
SEED = 42
N_EST = 200
N_FOLDS = 5

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
DOSE_WINNER_6 = [
    "GTVp_wavelet-HLH_firstorder_Median",
    "GTVn_wavelet-HLH_glszm_HighGrayLevelZoneEmphasis",
    "GTVn_wavelet-LHH_firstorder_Mean",
    "GTVn_gradient_firstorder_Maximum",
    "GTVn_gradient_firstorder_Range",
    "GTVp_wavelet-LLH_glrlm_GrayLevelVariance",
]
DOSE_PREFIX = "DOSE__"
DOSE_MODEL_FEATURES = [f"{DOSE_PREFIX}{f}" for f in DOSE_WINNER_6]
WINNER_FEATURES = CLINICAL_FEATURES + PT_WINNER + CT_WINNER + DOSE_MODEL_FEATURES
NODOSE_FEATURES = CLINICAL_FEATURES + PT_WINNER + CT_WINNER  # k=0: 12 features

CLINICAL_ONLY_FEATURES = ["Age", "Gender_Male", "Treatment_CRT"]

LOCKED_CHUS = 0.834921
LOCKED_NODOSE_CHUS = 0.7048
REPRO_TOL = 0.001
NODOSE_REPRO_TOL = 0.005
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

COLOUR_DAE    = "#762a83"   # purple — DAE model (11 rad. + 6 dose + 1 clin.), consistent with dose colour in CFE
COLOUR_NODOSE = "#2166ac"   # blue — RPM model (11 rad. + 1 clin., no dose), matches RPM DCA
COLOUR_CLIN   = "#d95f02"   # orange — clinical model, matches RPM DCA
COLOUR_TREAT_ALL  = "#636363"
COLOUR_TREAT_NONE = "black"


# ============================================================
# HELPERS
# ============================================================
def make_surv(event: np.ndarray, time: np.ndarray):
    return Surv.from_arrays(
        event=np.asarray(event, dtype=bool),
        time=np.asarray(time, dtype=float),
    )


def safe_ci(y, risk: np.ndarray) -> float:
    try:
        return float(concordance_index_censored(y["event"], y["time"], risk)[0])
    except Exception:
        return float("nan")


def _prefixed_dose_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [f for f in DOSE_WINNER_6 if f not in df.columns]
    if missing:
        raise KeyError(f"Missing locked dose features in {path}: {missing}")
    out = df[["PatientID"] + DOSE_WINNER_6].copy()
    return out.rename(columns={f: f"{DOSE_PREFIX}{f}" for f in DOSE_WINNER_6})


# ============================================================
# DATA LOADING
# ============================================================
def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    clinical = pd.read_csv(CLINICAL_FILE).dropna(subset=["Relapse", "RFS"])

    clin_cols = [
        "PatientID",
        "CenterID",
        "Cohort",
        "Relapse",
        "RFS",
    ] + sorted(set(CLINICAL_FEATURES + CLINICAL_ONLY_FEATURES))
    clin_cols = [c for c in clin_cols if c in clinical.columns]

    clin_dev = clinical[clinical["Cohort"] == "Dev"][clin_cols].copy()
    clin_chus = clinical[clinical["CenterID"] == 3][clin_cols].copy()

    dose_dev = _prefixed_dose_frame(DOSE_DEV_FILE)
    dose_chus = _prefixed_dose_frame(DOSE_CHUS_FILE)

    pt_dev = pd.read_csv(PT_DEV_FILE)[["PatientID"] + PT_WINNER]
    ct_dev = pd.read_csv(CT_DEV_FILE)[["PatientID"] + CT_WINNER]
    pt_ext = pd.read_csv(PT_EXT_FILE)
    ct_ext = pd.read_csv(CT_EXT_FILE)

    pt_chus = pt_ext[pt_ext["PatientID"].str.startswith("CHUS")][["PatientID"] + PT_WINNER]
    ct_chus = ct_ext[ct_ext["PatientID"].str.startswith("CHUS")][["PatientID"] + CT_WINNER]

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


# ============================================================
# MODEL FITTING
# ============================================================
def fit_dae_model(dev_df: pd.DataFrame) -> tuple[StandardScaler, ExtraSurvivalTrees]:
    x = dev_df[WINNER_FEATURES].to_numpy(dtype=float)
    y = make_surv(dev_df["Relapse"], dev_df["RFS"])
    scaler = StandardScaler().fit(x)
    model = ExtraSurvivalTrees(n_estimators=N_EST, random_state=SEED, n_jobs=-1)
    model.fit(scaler.transform(x), y)
    return scaler, model


def fit_survival_model(
    dev_df: pd.DataFrame,
    features: list[str],
) -> tuple[StandardScaler, ExtraSurvivalTrees]:
    x = dev_df[features].to_numpy(dtype=float)
    y = make_surv(dev_df["Relapse"], dev_df["RFS"])
    scaler = StandardScaler().fit(x)
    model = ExtraSurvivalTrees(n_estimators=N_EST, random_state=SEED, n_jobs=-1)
    model.fit(scaler.transform(x), y)
    return scaler, model


def compute_oof_risk(dev_df: pd.DataFrame, features: list[str]) -> np.ndarray:
    x_all = dev_df[features].to_numpy(dtype=float)
    y_event = dev_df["Relapse"].to_numpy(dtype=int)
    y_all = make_surv(dev_df["Relapse"], dev_df["RFS"])
    risk_oof = np.full(len(dev_df), np.nan, dtype=float)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for train_idx, test_idx in skf.split(x_all, y_event):
        scaler = StandardScaler().fit(x_all[train_idx])
        model = ExtraSurvivalTrees(n_estimators=N_EST, random_state=SEED, n_jobs=-1)
        model.fit(scaler.transform(x_all[train_idx]), y_all[train_idx])
        risk_oof[test_idx] = model.predict(scaler.transform(x_all[test_idx]))

    if np.isnan(risk_oof).any():
        raise RuntimeError("OOF prediction failed: at least one dev row has no risk score.")
    return risk_oof


def fit_clinical_model(
    dev_df: pd.DataFrame,
) -> tuple[StandardScaler, LogisticRegression, list[str], pd.Series]:
    available = [f for f in CLINICAL_ONLY_FEATURES if f in dev_df.columns]
    if not available:
        raise RuntimeError("No clinical-only comparator features found.")

    fill_values = dev_df[available].median(numeric_only=True)
    x = dev_df[available].fillna(fill_values).to_numpy(dtype=float)
    y = dev_df["Relapse"].to_numpy(dtype=int)

    scaler = StandardScaler().fit(x)
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        random_state=SEED,
    )
    model.fit(scaler.transform(x), y)
    return scaler, model, available, fill_values


def run_reproducibility_check(
    dae_scaler: StandardScaler,
    dae_model: ExtraSurvivalTrees,
    dev_df: pd.DataFrame,
    chus_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    def risk(df: pd.DataFrame) -> np.ndarray:
        x = dae_scaler.transform(df[WINNER_FEATURES].to_numpy(dtype=float))
        return dae_model.predict(x)

    risk_dev = risk(dev_df)
    risk_chus = risk(chus_df)

    y_chus = make_surv(chus_df["Relapse"], chus_df["RFS"])
    ci_chus = safe_ci(y_chus, risk_chus)
    delta = abs(ci_chus - LOCKED_CHUS)

    print("\nReproducibility check:")
    print(f"  CHUS C-index = {ci_chus:.6f}  locked={LOCKED_CHUS:.6f}  delta={delta:.6f}")
    if delta > REPRO_TOL:
        raise RuntimeError(
            f"Reproducibility check FAILED. CHUS delta={delta:.6f} "
            f"(tol={REPRO_TOL})."
        )
    print("  Reproducibility check PASSED.\n")
    return risk_dev, risk_chus


# ============================================================
# DCA COMPUTATION
# ============================================================
def compute_net_benefit(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    n = len(y_true)
    nb = np.empty(len(thresholds))
    for i, t in enumerate(thresholds):
        pred = (proba >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        odds = t / (1.0 - t)
        nb[i] = tp / n - fp / n * odds
    return nb


def compute_treat_all_nb(y_true: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    prevalence = float((y_true == 1).mean())
    nb = np.empty(len(thresholds))
    for i, t in enumerate(thresholds):
        odds = t / (1.0 - t)
        nb[i] = prevalence - (1.0 - prevalence) * odds
    return nb


def build_dca_table(
    y_true: np.ndarray,
    proba_dae: np.ndarray,
    proba_nodose: np.ndarray,
    proba_clin: np.ndarray,
    thresholds: np.ndarray,
    cohort_label: str,
) -> pd.DataFrame:
    nb_dae = compute_net_benefit(y_true, proba_dae, thresholds)
    nb_nodose = compute_net_benefit(y_true, proba_nodose, thresholds)
    nb_clin = compute_net_benefit(y_true, proba_clin, thresholds)
    nb_treat_all = compute_treat_all_nb(y_true, thresholds)
    nb_treat_none = np.zeros(len(thresholds))
    return pd.DataFrame(
        {
            "cohort": cohort_label,
            "threshold": np.round(thresholds, 4),
            "nb_dae_model": np.round(nb_dae, 6),
            "nb_nodose_baseline": np.round(nb_nodose, 6),
            "nb_clinical_only": np.round(nb_clin, 6),
            "nb_treat_all": np.round(nb_treat_all, 6),
            "nb_treat_none": np.round(nb_treat_none, 6),
        }
    )


# ============================================================
# PLOTTING
# ============================================================
def _draw_dca_panel(
    ax: plt.Axes,
    dca_df: pd.DataFrame,
    title: str,
    n_patients: int,
    n_events: int,
) -> None:
    t = dca_df["threshold"].to_numpy()
    nb_dae = dca_df["nb_dae_model"].to_numpy()
    nb_nodose = dca_df["nb_nodose_baseline"].to_numpy()
    nb_clin = dca_df["nb_clinical_only"].to_numpy()
    nb_all = dca_df["nb_treat_all"].to_numpy()

    ax.plot(
        t,
        nb_dae,
        color=COLOUR_DAE,
        linewidth=2.2,
        label="DAE model (11 rad. + 6 dose + 1 clin.)",
    )
    ax.plot(
        t,
        nb_nodose,
        color=COLOUR_NODOSE,
        linewidth=1.8,
        linestyle="--",
        label="RPM model (11 rad. + 1 clin., no dose)",
    )
    ax.plot(
        t,
        nb_clin,
        color=COLOUR_CLIN,
        linewidth=1.8,
        linestyle="--",
        label="Clinical model (Age, Sex, Treatment)",
    )
    ax.plot(
        t,
        nb_all,
        color=COLOUR_TREAT_ALL,
        linewidth=1.4,
        linestyle=":",
        label="Treat all as high-risk (assume everyone relapses)",
    )
    ax.axhline(0.0, color=COLOUR_TREAT_NONE, linewidth=1.2,
               label="Treat none (assume nobody relapses)")

    ax.set_xlabel("Threshold probability, $p_t$")
    ax.set_ylabel("Net benefit")
    ax.set_title(title)
    ax.set_xlim(t[0] - 0.01, t[-1] + 0.01)

    all_vals = np.concatenate([nb_dae, nb_nodose, nb_clin, nb_all, [0.0]])
    ymin = max(float(np.nanmin(all_vals)), -0.25)
    ymax = float(np.nanmax(all_vals))
    margin = 0.05 * max(abs(ymax - ymin), 0.05)
    ax.set_ylim(ymin - margin, ymax + margin)

    ax.legend(loc="lower left", frameon=True, framealpha=0.9, fontsize=9, edgecolor="lightgrey")


def save_single_panel(
    dca_df: pd.DataFrame,
    title: str,
    out_path: Path,
    n_patients: int,
    n_events: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    _draw_dca_panel(ax, dca_df, title, n_patients, n_events)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def save_combined_panel(
    dca_dev: pd.DataFrame,
    dca_chus: pd.DataFrame,
    n_dev: int,
    ev_dev: int,
    n_chus: int,
    ev_chus: int,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)

    _draw_dca_panel(axes[0], dca_dev, f"Development cohort (N={n_dev})", n_dev, ev_dev)
    _draw_dca_panel(axes[1], dca_chus, f"External validation: CHUS (N={n_chus})", n_chus, ev_chus)

    legend = axes[1].get_legend()
    if legend:
        legend.remove()

    fig.suptitle(
        "Decision Curve Analysis - DAE Retained Winner (1TC2298)",
        fontsize=12,
        y=1.01,
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

    print("Loading T1C N75 data splits...")
    dev_df, chus_df = load_splits()
    print(
        f"  Dev N={len(dev_df)} ({int(dev_df['Relapse'].sum())} events) | "
        f"CHUS N={len(chus_df)} ({int(chus_df['Relapse'].sum())} events)"
    )

    print("\nFitting locked DAE model (1TC2298)...")
    dae_scaler, dae_model = fit_dae_model(dev_df)
    risk_dev, risk_chus = run_reproducibility_check(dae_scaler, dae_model, dev_df, chus_df)

    print("Fitting no-dose baseline comparator (k=0, 12 features)...")
    risk_dev_nodose = compute_oof_risk(dev_df, NODOSE_FEATURES)
    nodose_scaler, nodose_model = fit_survival_model(dev_df, NODOSE_FEATURES)
    risk_chus_nodose = nodose_model.predict(
        nodose_scaler.transform(chus_df[NODOSE_FEATURES].to_numpy(dtype=float))
    )
    y_chus_surv = make_surv(chus_df["Relapse"], chus_df["RFS"])
    ci_chus_nodose = safe_ci(y_chus_surv, risk_chus_nodose)
    nodose_delta = abs(ci_chus_nodose - LOCKED_NODOSE_CHUS)
    print(
        f"  No-dose CHUS C-index = {ci_chus_nodose:.6f}  "
        f"expected~={LOCKED_NODOSE_CHUS:.4f}  delta={nodose_delta:.6f}"
    )
    if nodose_delta > NODOSE_REPRO_TOL:
        print(
            f"  WARNING: no-dose CHUS C-index deviates by {nodose_delta:.6f} "
            f"(tol={NODOSE_REPRO_TOL}). Continuing post-hoc DCA."
        )

    print("Fitting clinical-only comparator model...")
    clin_scaler, clin_model, clin_cols, clin_fill = fit_clinical_model(dev_df)
    print(f"  Clinical features used: {clin_cols}")

    risk_lo = float(risk_dev.min())
    risk_hi = float(risk_dev.max())

    def norm(risk_scores: np.ndarray) -> np.ndarray:
        if risk_hi == risk_lo:
            return np.full(len(risk_scores), 0.5)
        return np.clip((risk_scores - risk_lo) / (risk_hi - risk_lo), 0.0, 1.0)

    proba_dae_dev = norm(risk_dev)
    proba_dae_chus = norm(risk_chus)

    nodose_lo = float(risk_dev_nodose.min())
    nodose_hi = float(risk_dev_nodose.max())

    def norm_nodose(risk_scores: np.ndarray) -> np.ndarray:
        if nodose_hi == nodose_lo:
            return np.full(len(risk_scores), 0.5)
        return np.clip((risk_scores - nodose_lo) / (nodose_hi - nodose_lo), 0.0, 1.0)

    proba_nodose_dev = norm_nodose(risk_dev_nodose)
    proba_nodose_chus = norm_nodose(risk_chus_nodose)

    def clin_proba(df: pd.DataFrame) -> np.ndarray:
        x = df[clin_cols].fillna(clin_fill).to_numpy(dtype=float)
        return clin_model.predict_proba(clin_scaler.transform(x))[:, 1]

    proba_clin_dev = clin_proba(dev_df)
    proba_clin_chus = clin_proba(chus_df)

    y_dev = dev_df["Relapse"].to_numpy(dtype=int)
    y_chus = chus_df["Relapse"].to_numpy(dtype=int)

    print("Computing DCA net benefit curves...")
    dca_dev = build_dca_table(
        y_dev,
        proba_dae_dev,
        proba_nodose_dev,
        proba_clin_dev,
        THRESHOLDS,
        "dev",
    )
    dca_chus = build_dca_table(
        y_chus,
        proba_dae_chus,
        proba_nodose_chus,
        proba_clin_chus,
        THRESHOLDS,
        "chus",
    )

    for label, dca_df in [("Dev", dca_dev), ("CHUS", dca_chus)]:
        dae_beats_all = int((dca_df["nb_dae_model"] > dca_df["nb_treat_all"]).sum())
        dae_beats_nodose = int((dca_df["nb_dae_model"] > dca_df["nb_nodose_baseline"]).sum())
        dae_beats_clin = int((dca_df["nb_dae_model"] > dca_df["nb_clinical_only"]).sum())
        total = len(dca_df)
        print(
            f"  {label}: DAE > Treat-All at {dae_beats_all}/{total} thresholds; "
            f"DAE > No-dose at {dae_beats_nodose}/{total} thresholds; "
            f"DAE > Clinical-only at {dae_beats_clin}/{total} thresholds"
        )

    print("\nSaving CSV tables...")
    dca_dev.to_csv(OUT_DIR / "T1C_DCA_net_benefit_table_dev.csv", index=False)
    dca_chus.to_csv(OUT_DIR / "T1C_DCA_net_benefit_table_chus.csv", index=False)
    print("  Saved: T1C_DCA_net_benefit_table_dev.csv")
    print("  Saved: T1C_DCA_net_benefit_table_chus.csv")

    print("\nSaving plots...")
    save_single_panel(
        dca_dev,
        title=f"Decision Curve Analysis - DAE, Development Cohort (N={len(dev_df)})",
        out_path=OUT_DIR / "T1C_DCA_net_benefit_dev.png",
        n_patients=len(dev_df),
        n_events=int(dev_df["Relapse"].sum()),
    )
    save_single_panel(
        dca_chus,
        title=f"Decision Curve Analysis - DAE, External Validation: CHUS (N={len(chus_df)})",
        out_path=OUT_DIR / "T1C_DCA_net_benefit_chus.png",
        n_patients=len(chus_df),
        n_events=int(chus_df["Relapse"].sum()),
    )
    save_combined_panel(
        dca_dev,
        dca_chus,
        n_dev=len(dev_df),
        ev_dev=int(dev_df["Relapse"].sum()),
        n_chus=len(chus_df),
        ev_chus=int(chus_df["Relapse"].sum()),
        out_path=OUT_DIR / "T1C_DCA_net_benefit_combined.png",
    )

    print("\n" + "=" * 72)
    print("T1C BRANCH A3 DCA COMPLETE")
    print("=" * 72)
    print(f"All outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
