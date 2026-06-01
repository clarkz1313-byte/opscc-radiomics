"""
29_mar_task2_branchA3_DCA.py

Task 2 Post-Study - Branch A3: Decision Curve Analysis (DCA)
Locked winner: T68357, N=16 (1 clinical + 8 PT + 7 CT)
Coach: LR_L2, C=0.3689386715674865

Purpose:
    Compute Decision Curve Analysis (DCA) for the locked HCM winner across
    three cohorts: development (N=67), internal test (N=20), and CHUS external
    validation (N=27). Compares HCM model net benefit against a clinical-only
    baseline, treat-all, and treat-none strategies. The DCA action class is
    HPV-negative identification, using 1 - predict_proba(..., class 1).

Outputs (saved to Mar_2026_task2/29_mar_task2_post_study_outputs/branchA3/):
    - T2_DCA_net_benefit_dev.png         (single-panel, dev cohort)
    - T2_DCA_net_benefit_test.png        (single-panel, internal test cohort)
    - T2_DCA_net_benefit_chus.png        (single-panel, CHUS external — primary thesis figure)
    - T2_DCA_net_benefit_combined.png    (three-panel overview figure)
    - T2_DCA_net_benefit_table_dev.csv
    - T2_DCA_net_benefit_table_test.csv
    - T2_DCA_net_benefit_table_chus.csv

Reproducibility guard (aborts if either fails):
    AUC_ext  = 0.7785714285714286  (tolerance 0.001)
    BA_ext   = 0.775               (tolerance 0.01)
    Threshold= 0.8719759300122685

Run:
    python Mar_2026_task2/29_mar_task2_branchA3_DCA.py
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
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================
ROOT = Path(__file__).resolve().parent
RAD_DIR = ROOT / "12_mar_task2_rad_data"
OUT_DIR = ROOT / "29_mar_task2_post_study_outputs" / "branchA3"

CT_TRAIN_FILE = RAD_DIR / "13_mar_task2_CT_primary_train.csv"
CT_TEST_FILE = RAD_DIR / "13_mar_task2_CT_primary_test.csv"
CT_EXT_FILE = RAD_DIR / "12_mar_task2_CT_primary_ext.csv"
PT_TRAIN_FILE = RAD_DIR / "13_mar_task2_PT_primary_train.csv"
PT_TEST_FILE = RAD_DIR / "13_mar_task2_PT_primary_test.csv"
PT_EXT_FILE = RAD_DIR / "12_mar_task2_PT_primary_ext.csv"

# ============================================================
# LOCKED MODEL CONSTANTS
# ============================================================
TARGET = "HPV_binary"
CLINICAL_FEATURES_HCM = ["Gender_Male"]
PT_WINNER = [
    "GTVn_wavelet-LHL_glszm_GrayLevelVariance",
    "GTVn_logarithm_glszm_SmallAreaLowGrayLevelEmphasis",
    "GTVn_wavelet-LLH_firstorder_Skewness",
    "GTVn_squareroot_glcm_Idm",
    "GTVp_wavelet-LLH_firstorder_Median",
    "GTVn_logarithm_glcm_Idn",
    "GTVp_wavelet-HHL_glrlm_ShortRunHighGrayLevelEmphasis",
    "GTVp_original_firstorder_InterquartileRange",
]
CT_WINNER = [
    "GTVp_log-sigma-1-mm-3D_firstorder_Range",
    "GTVn_wavelet-HLH_gldm_SmallDependenceEmphasis",
    "GTVn_wavelet-LHH_glrlm_GrayLevelVariance",
    "GTVn_wavelet-HLH_gldm_SmallDependenceHighGrayLevelEmphasis",
    "GTVn_wavelet-HHH_glcm_DifferenceAverage",
    "GTVn_wavelet-HHH_glszm_ZonePercentage",
    "GTVn_wavelet-LHH_glcm_ClusterProminence",
]
HCM_FEATURES = CLINICAL_FEATURES_HCM + PT_WINNER + CT_WINNER

# Clinical-only model feature pool (available in CSV as standard columns)
CLINICAL_ONLY_FEATURES = ["Age", "Gender_Male", "Treatment_CRT"]

LOCKED_C = 0.3689386715674865
LOCKED_AUC_EXT = 0.7785714285714286
LOCKED_BA_EXT = 0.775
LOCKED_YOUDEN_EXT = 0.8719759300122685
REPRO_TOL_AUC = 0.001
REPRO_TOL_BA = 0.01
SEED = 42

# DCA threshold range — capped at 0.50: above 50% certainty is beyond the
# clinically useful range for a 26% prevalence HPV-negative decision.
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

COLOUR_HCM = "#2ca02c"       # solid green — HCM radiomics-clinical model
COLOUR_CLIN = "#d95f02"      # dashed orange — clinical-only model
COLOUR_TREAT_ALL = "#636363" # dotted grey — treat all
COLOUR_TREAT_NONE = "black"  # solid black at y=0 — treat none


# ============================================================
# DATA LOADING
# ============================================================
def build_split(ct_path: Path, pt_path: Path) -> pd.DataFrame:
    """Merge CT and PT files, keeping only required columns."""
    ct_df = pd.read_csv(ct_path)
    pt_df = pd.read_csv(pt_path)

    ct_keep = ["PatientID", TARGET] + CT_WINNER
    # Clinical features: pull from CT file (they appear there)
    for col in CLINICAL_FEATURES_HCM + CLINICAL_ONLY_FEATURES:
        if col in ct_df.columns and col not in ct_keep:
            ct_keep.append(col)

    pt_keep = ["PatientID"] + PT_WINNER

    merged = ct_df[ct_keep].merge(
        pt_df[pt_keep],
        on="PatientID",
        how="inner",
        validate="one_to_one",
    )
    return merged.copy()


def verify_inputs() -> None:
    print("=" * 72)
    print("INPUT FILE CHECK")
    print("=" * 72)
    paths = {
        "ct_train": CT_TRAIN_FILE,
        "ct_test": CT_TEST_FILE,
        "ct_ext": CT_EXT_FILE,
        "pt_train": PT_TRAIN_FILE,
        "pt_test": PT_TEST_FILE,
        "pt_ext": PT_EXT_FILE,
    }
    all_ok = True
    for label, path in paths.items():
        ok = path.exists()
        status = "PASS" if ok else "FAIL"
        print(f"  {label:10s} {status}  {path}")
        all_ok &= ok
    if not all_ok:
        raise FileNotFoundError("One or more required input files are missing.")

    # Check required columns
    required = set(HCM_FEATURES + [TARGET, "PatientID"])
    for label, path in paths.items():
        cols = set(pd.read_csv(path, nrows=1).columns.tolist())
        # Clinical-only features may be absent from ext files (Age, Treatment_CRT)
        hcm_missing = required - cols
        if hcm_missing:
            print(f"  WARNING: {label} missing columns: {sorted(hcm_missing)}")


# ============================================================
# MODEL FITTING
# ============================================================
def fit_hcm_model(train_df: pd.DataFrame) -> tuple[StandardScaler, LogisticRegression]:
    """Fit the locked HCM model on the training set (scaler fitted on train only)."""
    x_train = train_df[HCM_FEATURES].to_numpy(dtype=float)
    y_train = train_df[TARGET].to_numpy(dtype=int)
    scaler = StandardScaler().fit(x_train)
    x_sc = scaler.transform(x_train)
    model = LogisticRegression(
        penalty="l2",
        C=LOCKED_C,
        solver="lbfgs",
        class_weight="balanced",
        max_iter=2000,
        random_state=SEED,
    )
    model.fit(x_sc, y_train)
    return scaler, model


def fit_clinical_model(
    train_df: pd.DataFrame,
    clin_features: list[str],
) -> tuple[StandardScaler, LogisticRegression]:
    """Fit clinical-only baseline model. Falls back to available clinical columns."""
    available = [f for f in clin_features if f in train_df.columns]
    if not available:
        raise ValueError(f"None of {clin_features} found in training DataFrame.")
    x_train = train_df[available].to_numpy(dtype=float)
    y_train = train_df[TARGET].to_numpy(dtype=int)
    scaler = StandardScaler().fit(x_train)
    x_sc = scaler.transform(x_train)
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        class_weight="balanced",
        max_iter=2000,
        random_state=SEED,
    )
    model.fit(x_sc, y_train)
    return scaler, model, available


def run_reproducibility_check(
    model: LogisticRegression,
    scaler: StandardScaler,
    ext_df: pd.DataFrame,
) -> None:
    """Refit-and-check: abort if CHUS AUC or BA deviates from locked values."""
    x_ext = scaler.transform(ext_df[HCM_FEATURES].to_numpy(dtype=float))
    y_ext = ext_df[TARGET].to_numpy(dtype=int)
    proba = model.predict_proba(x_ext)[:, 1]
    auc = float(roc_auc_score(y_ext, proba))
    pred = (proba >= LOCKED_YOUDEN_EXT).astype(int)
    ba = float(balanced_accuracy_score(y_ext, pred))

    d_auc = abs(auc - LOCKED_AUC_EXT)
    d_ba = abs(ba - LOCKED_BA_EXT)

    print(f"\nReproducibility check:")
    print(f"  AUC_ext = {auc:.6f}  locked={LOCKED_AUC_EXT:.6f}  delta={d_auc:.6f}")
    print(f"  BA_ext  = {ba:.6f}  locked={LOCKED_BA_EXT:.6f}  delta={d_ba:.6f}")

    if d_auc > REPRO_TOL_AUC or d_ba > REPRO_TOL_BA:
        raise RuntimeError(
            f"Reproducibility check FAILED. "
            f"AUC delta={d_auc:.6f} (tol={REPRO_TOL_AUC}), "
            f"BA delta={d_ba:.6f} (tol={REPRO_TOL_BA})."
        )
    print("  Reproducibility check PASSED.\n")


# ============================================================
# DCA COMPUTATION
# ============================================================
def compute_net_benefit(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """
    Standard Vickers-Elkin net benefit at each threshold.
    NB(t) = TP/N - FP/N * t/(1-t)
    """
    n = len(y_true)
    nb = np.empty(len(thresholds))
    for i, t in enumerate(thresholds):
        pred = (proba >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        # Avoid division at t=1.0
        odds = t / (1.0 - t) if t < 1.0 else np.inf
        nb[i] = tp / n - fp / n * odds
    return nb


def compute_treat_all_nb(
    y_true: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Net benefit of the 'Treat All as HPV-negative' strategy."""
    n = len(y_true)
    prevalence = (y_true == 1).mean()  # P(HPV-negative) - the action class
    nb = np.empty(len(thresholds))
    for i, t in enumerate(thresholds):
        odds = t / (1.0 - t) if t < 1.0 else np.inf
        nb[i] = prevalence - (1.0 - prevalence) * odds
    return nb


def build_dca_table(
    y_true: np.ndarray,
    proba_hcm: np.ndarray,
    proba_clin: np.ndarray,
    thresholds: np.ndarray,
    cohort_label: str,
) -> pd.DataFrame:
    """Build a long-form DCA results table for one cohort."""
    nb_hcm = compute_net_benefit(y_true, proba_hcm, thresholds)
    nb_clin = compute_net_benefit(y_true, proba_clin, thresholds)
    nb_treat_all = compute_treat_all_nb(y_true, thresholds)
    nb_treat_none = np.zeros(len(thresholds))

    return pd.DataFrame(
        {
            "cohort": cohort_label,
            "threshold": np.round(thresholds, 4),
            "nb_hcm_model": np.round(nb_hcm, 6),
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
    show_annotation: bool = False,
    n_patients: int | None = None,
    n_hpvneg: int | None = None,
) -> None:
    """Draw one DCA panel onto ax."""
    t = dca_df["threshold"].to_numpy()
    nb_hcm = dca_df["nb_hcm_model"].to_numpy()
    nb_clin = dca_df["nb_clinical_only"].to_numpy()
    nb_all = dca_df["nb_treat_all"].to_numpy()

    ax.plot(
        t, nb_hcm,
        color=COLOUR_HCM, linewidth=2.2,
        label="HCM radiomics-clinical model (15 rad. + 1 clin.)\nAction: identify HPV-negative patients",
    )
    ax.plot(
        t, nb_clin,
        color=COLOUR_CLIN, linewidth=1.8, linestyle="--",
        label="Clinical-only model (Age, Sex, Treatment)",
    )
    ax.plot(
        t, nb_all,
        color=COLOUR_TREAT_ALL, linewidth=1.4, linestyle=":",
        label="Treat all as HPV-negative (full-treatment safeguard)",
    )
    ax.axhline(0.0, color=COLOUR_TREAT_NONE, linewidth=1.2,
               label="Treat none as HPV-negative")

    ax.set_xlabel("Threshold probability, $p_t$")
    ax.set_ylabel("Net benefit")
    ax.set_title(title)
    ax.set_xlim(0.04, 0.55)
    ax.set_ylim(-0.30, 0.30)

    ax.legend(loc="lower left", frameon=True, framealpha=0.9, fontsize=9,
              edgecolor="lightgrey")


def save_single_panel(
    dca_df: pd.DataFrame,
    title: str,
    out_path: Path,
    show_annotation: bool = False,
    n_patients: int | None = None,
    n_hpvneg: int | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    _draw_dca_panel(ax, dca_df, title, show_annotation, n_patients, n_hpvneg)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def save_combined_panel(
    dca_dev: pd.DataFrame,
    dca_test: pd.DataFrame,
    dca_chus: pd.DataFrame,
    out_path: Path,
) -> None:
    # Horizontal (1×3) layout: Development / Internal Test / CHUS side by side
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)

    _draw_dca_panel(axes[0], dca_dev, "Development cohort (N=67)")
    _draw_dca_panel(axes[1], dca_test, "Internal test cohort (N=20)")
    _draw_dca_panel(axes[2], dca_chus, "External validation: CHUS (N=27)")

    # Keep legend only in Dev panel (axes[0]); remove from test and CHUS panels
    for ax in [axes[1], axes[2]]:
        legend = ax.get_legend()
        if legend:
            legend.remove()

    fig.suptitle(
        "Decision Curve Analysis \u2014 HCM Retained Winner (T68357) across "
        "Development, Internal Test, and External Validation Cohorts",
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
    verify_inputs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load splits ----
    print("\nLoading data splits...")
    train_df = build_split(CT_TRAIN_FILE, PT_TRAIN_FILE)
    test_df = build_split(CT_TEST_FILE, PT_TEST_FILE)
    ext_df = build_split(CT_EXT_FILE, PT_EXT_FILE)

    print(f"  train N={len(train_df)} | test N={len(test_df)} | CHUS N={len(ext_df)}")
    n_hpvneg_chus = int((ext_df[TARGET] == 0).sum())
    print(f"  CHUS HPV-negative: {n_hpvneg_chus} / {len(ext_df)}")

    # ---- Fit HCM locked model ----
    print("\nFitting locked HCM model (T68357)...")
    hcm_scaler, hcm_model = fit_hcm_model(train_df)

    # ---- Reproducibility check ----
    run_reproducibility_check(hcm_model, hcm_scaler, ext_df)

    # ---- Fit clinical-only model ----
    print("Fitting clinical-only baseline model...")
    clin_scaler, clin_model, clin_cols_used = fit_clinical_model(train_df, CLINICAL_ONLY_FEATURES)
    print(f"  Clinical features used: {clin_cols_used}")

    # Helper: get probabilities for a given split
    def get_probas(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_hcm = hcm_scaler.transform(df[HCM_FEATURES].to_numpy(dtype=float))
        # The retained HCM model is trained with class 1 = HPV-positive.
        # This DCA uses the complementary action: identify HPV-negative cases.
        proba_hcm = 1.0 - hcm_model.predict_proba(x_hcm)[:, 1]

        # Clinical model: only use columns that were available at fit time
        clin_available = [f for f in clin_cols_used if f in df.columns]
        if clin_available:
            x_clin_raw = df[clin_available].to_numpy(dtype=float)
            # If some training columns are missing in this split, fall back gracefully
            if len(clin_available) == len(clin_cols_used):
                x_clin_sc = clin_scaler.transform(x_clin_raw)
            else:
                # Pad with zeros for missing columns (rare edge case)
                padded = np.zeros((len(df), len(clin_cols_used)))
                for j, col in enumerate(clin_cols_used):
                    if col in clin_available:
                        idx = clin_available.index(col)
                        padded[:, j] = x_clin_raw[:, idx]
                x_clin_sc = clin_scaler.transform(padded)
            proba_clin = 1.0 - clin_model.predict_proba(x_clin_sc)[:, 1]
        else:
            # Fallback: prevalence-based flat probability for HPV-negative.
            prevalence = float((train_df[TARGET] == 0).mean())
            proba_clin = np.full(len(df), prevalence)

        y_true = (df[TARGET].to_numpy(dtype=int) == 0).astype(int)
        return y_true, proba_hcm, proba_clin

    # ---- Compute DCA tables ----
    print("\nComputing DCA net benefit curves...")
    y_dev, p_hcm_dev, p_clin_dev = get_probas(train_df)
    y_test, p_hcm_test, p_clin_test = get_probas(test_df)
    y_chus, p_hcm_chus, p_clin_chus = get_probas(ext_df)

    dca_dev = build_dca_table(y_dev, p_hcm_dev, p_clin_dev, THRESHOLDS, "dev")
    dca_test = build_dca_table(y_test, p_hcm_test, p_clin_test, THRESHOLDS, "test")
    dca_chus = build_dca_table(y_chus, p_hcm_chus, p_clin_chus, THRESHOLDS, "chus")

    # ---- Print summary statistics ----
    for label, dca_df in [("Dev", dca_dev), ("Test", dca_test), ("CHUS", dca_chus)]:
        hcm_pos = (dca_df["nb_hcm_model"] > dca_df["nb_treat_all"]).sum()
        hcm_pos_clin = (dca_df["nb_hcm_model"] > dca_df["nb_clinical_only"]).sum()
        total = len(dca_df)
        print(
            f"  {label}: HCM > Treat-All at {hcm_pos}/{total} thresholds; "
            f"HCM > Clinical-only at {hcm_pos_clin}/{total} thresholds"
        )

    # ---- Save CSV tables ----
    print("\nSaving CSV tables...")
    dca_dev.to_csv(OUT_DIR / "T2_DCA_net_benefit_table_dev.csv", index=False)
    dca_test.to_csv(OUT_DIR / "T2_DCA_net_benefit_table_test.csv", index=False)
    dca_chus.to_csv(OUT_DIR / "T2_DCA_net_benefit_table_chus.csv", index=False)
    print("  Saved: T2_DCA_net_benefit_table_dev.csv")
    print("  Saved: T2_DCA_net_benefit_table_test.csv")
    print("  Saved: T2_DCA_net_benefit_table_chus.csv")

    # ---- Save plots ----
    print("\nSaving plots...")
    save_single_panel(
        dca_dev,
        title="Decision Curve Analysis - HCM HPV-Negative Action, Development Cohort (N=67)",
        out_path=OUT_DIR / "T2_DCA_net_benefit_dev.png",
    )
    save_single_panel(
        dca_test,
        title="Decision Curve Analysis - HCM HPV-Negative Action, Internal Test Cohort (N=20)",
        out_path=OUT_DIR / "T2_DCA_net_benefit_test.png",
    )
    save_single_panel(
        dca_chus,
        title="Decision Curve Analysis - HCM HPV-Negative Action, External Validation: CHUS (N=27)",
        out_path=OUT_DIR / "T2_DCA_net_benefit_chus.png",
    )
    save_combined_panel(
        dca_dev, dca_test, dca_chus,
        out_path=OUT_DIR / "T2_DCA_net_benefit_combined.png",
    )

    print("\n" + "=" * 72)
    print("BRANCH A3 COMPLETE")
    print("=" * 72)
    print(f"All outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
