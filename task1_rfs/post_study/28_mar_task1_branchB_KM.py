"""
28_mar_task1_branchB_KM.py

Task 1 Post-Study — Branch B: Survival Presentation (Kaplan-Meier)
Locked winner: PT_S3_1_768xCT_S3_8_235, N=12 (1 clinical + 7 PT + 4 CT)
Coach: ExtraSurvivalTrees(n_estimators=200, random_state=42)

Design:
  - PRIMARY: cohort-specific median split — each cohort split at its own median risk score.
    This is standard in radiomics KM papers. The C-index already demonstrates cross-cohort
    ranking; KM shows within-cohort risk stratification separately.
    Rationale: dev-derived threshold (10.36) does not transport to CHUP (median ~16.3)
    because CHUP's absolute risk score distribution is shifted relative to dev. Applying
    a fixed dev threshold pushes 32/35 CHUP patients into high-risk, making KM unstable.
  - SECONDARY (calibration check): dev-derived median threshold applied uniformly.
    Saved separately so both approaches are available for honest reporting.
  - RFS in days converted to months for readability (÷ 30.44)
  - KM curves: dev, CHUS, CHUP
  - Log-rank p-value annotated on each plot
  - Hazard ratio (HR) from univariate Cox on risk-group binary variable

Outputs (all saved to Mar_2026/28_mar_task1_post_study_outputs/branchB/):
  PRIMARY (cohort-specific median split):
  - km_dev.png, km_chus.png, km_chup.png
  - km_combined_3panel.png
  - km_summary_table.csv
  SECONDARY (dev threshold applied uniformly — calibration reference):
  - km_dev_devthresh.png, km_chus_devthresh.png, km_chup_devthresh.png
  - km_combined_3panel_devthresh.png
  - km_summary_table_devthresh.csv

Reproducibility check: recomputed CHUS and CHUP C-index must match locked values
  CHUS = 0.742857142857143  (tolerance 0.001)
  CHUP = 0.727586206896552  (tolerance 0.001)
Script aborts if either delta exceeds tolerance.
"""

import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Auto-install lifelines if missing
try:
    import lifelines
except ImportError:
    print("Installing lifelines...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lifelines", "-q"])

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import ExtraSurvivalTrees
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

# ============================================================
# CONFIG
# ============================================================
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "28_mar_task1_post_study_outputs" / "branchB"
OUT_DIR.mkdir(exist_ok=True, parents=True)

PT_DEV_FILE   = ROOT / "27_feb_PT_development.csv"
CT_DEV_FILE   = ROOT / "27_feb_CT_development.csv"
PT_EXT_FILE   = ROOT / "27_feb_PT_external.csv"
CT_EXT_FILE   = ROOT / "27_feb_CT_external.csv"
CLINICAL_FILE = ROOT.parent / "Feb_2026" / "25_feb_clinical_reduced_dataset" / "25_feb_Processed_clinical_reduced.csv"
PT_FEAT_FILE  = ROOT / "2_mar_finalist_outputs" / "PT_inter1_768_features_recheck.csv"
CT_FEAT_FILE  = ROOT / "2_mar_finalist_outputs" / "CT_inter8_235_features_recheck.csv"

SEED = 42
DAYS_TO_MONTHS = 1 / 30.44

# Locked winner feature set
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

# Plot style
COLOR_HIGH = "#d62728"   # red = high risk
COLOR_LOW  = "#1f77b4"   # blue = low risk

# ============================================================
# HELPERS
# ============================================================
def make_surv(event, time):
    return Surv.from_arrays(event=np.asarray(event, dtype=bool),
                            time=np.asarray(time, dtype=float))

def safe_ci(y, risk):
    try:
        return float(concordance_index_censored(y["event"], y["time"], risk)[0])
    except Exception:
        return float("nan")

def plot_km(df_cohort, risk_scores, threshold, cohort_name, out_path, ci_val=None):
    """
    Plot KM curve for one cohort.
    df_cohort: DataFrame with columns Relapse, RFS (days)
    risk_scores: 1D array aligned to df_cohort rows
    threshold: median dev risk score (applied to all cohorts)
    """
    group = (risk_scores >= threshold).astype(int)  # 1 = high risk, 0 = low risk
    n_high = int(group.sum())
    n_low  = int((group == 0).sum())

    duration_months = df_cohort["RFS"].values * DAYS_TO_MONTHS
    event            = df_cohort["Relapse"].values.astype(bool)

    # Log-rank test
    lr = logrank_test(
        duration_months[group == 1], duration_months[group == 0],
        event_observed_A=event[group == 1],
        event_observed_B=event[group == 0],
    )
    p_val = lr.p_value

    fig, ax = plt.subplots(figsize=(7, 5))

    for grp_val, label, color in [
        (1, f"High risk (n={n_high})", COLOR_HIGH),
        (0, f"Low risk  (n={n_low})",  COLOR_LOW),
    ]:
        mask = group == grp_val
        if mask.sum() == 0:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(duration_months[mask], event_observed=event[mask], label=label)
        kmf.plot_survival_function(ax=ax, ci_show=True, color=color, linewidth=2)

    # Annotate p-value, HR, and CI
    p_text = f"p = {p_val:.4f}" if p_val >= 0.0001 else "p < 0.0001"
    try:
        cox_df_inner = pd.DataFrame({
            "duration": df_cohort["RFS"].values * DAYS_TO_MONTHS,
            "event": df_cohort["Relapse"].values.astype(int),
            "high_risk": group,
        })
        cph_inner = CoxPHFitter()
        cph_inner.fit(cox_df_inner, duration_col="duration", event_col="event",
                      formula="high_risk", show_progress=False)
        hr_v = float(np.exp(cph_inner.summary.loc["high_risk", "coef"]))
        hr_lo_v = float(np.exp(cph_inner.summary.loc["high_risk", "coef lower 95%"]))
        hr_hi_v = float(np.exp(cph_inner.summary.loc["high_risk", "coef upper 95%"]))
        hr_text = f"HR={hr_v:.2f} ({hr_lo_v:.2f}–{hr_hi_v:.2f})"
    except Exception:
        hr_text = "HR=N/A"
    ci_text = f"C-index = {ci_val:.3f}" if ci_val is not None else ""
    annotation = f"{p_text}\n{hr_text}\n{ci_text}" if ci_text else f"{p_text}\n{hr_text}"
    ax.text(0.97, 0.97, annotation,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Time (months)", fontsize=11)
    ax.set_ylabel("Relapse-free survival probability", fontsize=11)
    ax.set_title(f"RPM — Kaplan-Meier Risk Stratification, {cohort_name} Cohort\n"
                 f"N=12 features (1 clinical + 7 PET + 4 CT), ExtraSurvivalTrees deployment model", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="lower left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close("all")
    return n_high, n_low, p_val

def compute_hr(df_cohort, risk_scores, threshold):
    """
    Univariate Cox HR for high vs low risk group.
    Returns HR, CI_lower, CI_upper, p_value.
    """
    group = (risk_scores >= threshold).astype(int)
    cox_df = pd.DataFrame({
        "duration": df_cohort["RFS"].values * DAYS_TO_MONTHS,
        "event":    df_cohort["Relapse"].values.astype(int),
        "high_risk": group,
    })
    try:
        cph = CoxPHFitter()
        cph.fit(cox_df, duration_col="duration", event_col="event",
                formula="high_risk", show_progress=False)
        summary = cph.summary
        hr      = float(np.exp(summary.loc["high_risk", "coef"]))
        hr_lo   = float(np.exp(summary.loc["high_risk", "coef lower 95%"]))
        hr_hi   = float(np.exp(summary.loc["high_risk", "coef upper 95%"]))
        p_cox   = float(summary.loc["high_risk", "p"])
    except Exception as e:
        print(f"  Cox HR failed: {e}")
        hr, hr_lo, hr_hi, p_cox = float("nan"), float("nan"), float("nan"), float("nan")
    return hr, hr_lo, hr_hi, p_cox

# ============================================================
# LOAD DATA  (mirrors recheck script merge logic exactly)
# ============================================================
print("Loading data...")

pt_feat_pool = pd.read_csv(PT_FEAT_FILE)["Feature"].tolist()
ct_feat_raw  = pd.read_csv(CT_FEAT_FILE)["Feature"].tolist()
pt_set       = set(pt_feat_pool)
ct_feat_pool = [f for f in ct_feat_raw if f not in pt_set]

clinical = pd.read_csv(CLINICAL_FILE).dropna(subset=["Relapse", "RFS"])

clin_dev  = clinical[clinical["Cohort"] == "Dev"][
    ["PatientID", "CenterID", "Relapse", "RFS"] + CLINICAL_FEATURES].copy()
clin_chus = clinical[clinical["CenterID"] == 3][
    ["PatientID", "Relapse", "RFS"] + CLINICAL_FEATURES].copy()
clin_chup = clinical[clinical["CenterID"] == 2][
    ["PatientID", "Relapse", "RFS"] + CLINICAL_FEATURES].copy()

pt_dev = pd.read_csv(PT_DEV_FILE)
ct_dev = pd.read_csv(CT_DEV_FILE)
pt_ext = pd.read_csv(PT_EXT_FILE)
ct_ext = pd.read_csv(CT_EXT_FILE)

rad_dev  = pt_dev[["PatientID"] + pt_feat_pool].merge(
               ct_dev[["PatientID"] + ct_feat_pool], on="PatientID", how="inner")
rad_ext  = pt_ext[["PatientID"] + pt_feat_pool].merge(
               ct_ext[["PatientID"] + ct_feat_pool], on="PatientID", how="inner")

rad_chus = rad_ext[rad_ext["PatientID"].str.startswith("CHUS")]
rad_chup = rad_ext[rad_ext["PatientID"].str.startswith("CHUP")]

dev_df  = clin_dev.merge(rad_dev,  on="PatientID", how="inner")
chus_df = clin_chus.merge(rad_chus, on="PatientID", how="inner")
chup_df = clin_chup.merge(rad_chup, on="PatientID", how="inner")

print(f"Dev={len(dev_df)} ({int(dev_df['Relapse'].sum())} events) | "
      f"CHUS={len(chus_df)} ({int(chus_df['Relapse'].sum())} events) | "
      f"CHUP={len(chup_df)} ({int(chup_df['Relapse'].sum())} events)")

X_dev  = dev_df[WINNER_FEATURES].values.astype(float)
y_dev  = make_surv(dev_df["Relapse"], dev_df["RFS"])
X_chus = chus_df[WINNER_FEATURES].values.astype(float)
y_chus = make_surv(chus_df["Relapse"], chus_df["RFS"])
X_chup = chup_df[WINNER_FEATURES].values.astype(float)
y_chup = make_surv(chup_df["Relapse"], chup_df["RFS"])

# ============================================================
# REFIT LOCKED MODEL
# ============================================================
print("\nRefitting locked EST model on full dev...")
scaler = StandardScaler()
X_dev_sc  = scaler.fit_transform(X_dev)
X_chus_sc = scaler.transform(X_chus)
X_chup_sc = scaler.transform(X_chup)

model = ExtraSurvivalTrees(n_estimators=200, random_state=SEED, n_jobs=-1)
model.fit(X_dev_sc, y_dev)

risk_dev  = model.predict(X_dev_sc)
risk_chus = model.predict(X_chus_sc)
risk_chup = model.predict(X_chup_sc)

ci_chus = safe_ci(y_chus, risk_chus)
ci_chup = safe_ci(y_chup, risk_chup)
print(f"Reproduced CHUS={ci_chus:.6f}  (locked={LOCKED_CHUS:.6f}  delta={abs(ci_chus - LOCKED_CHUS):.6f})")
print(f"Reproduced CHUP={ci_chup:.6f}  (locked={LOCKED_CHUP:.6f}  delta={abs(ci_chup - LOCKED_CHUP):.6f})")

if abs(ci_chus - LOCKED_CHUS) > REPRO_TOL or abs(ci_chup - LOCKED_CHUP) > REPRO_TOL:
    raise RuntimeError(
        f"Reproducibility check FAILED. "
        f"CHUS delta={abs(ci_chus - LOCKED_CHUS):.6f}, "
        f"CHUP delta={abs(ci_chup - LOCKED_CHUP):.6f}. "
        f"Check data files or feature list."
    )
print("Reproducibility check PASSED.")

# ============================================================
# THRESHOLDS
# ============================================================
threshold_dev = float(np.median(risk_dev))  # dev-derived, for calibration reference

# Cohort-specific medians (primary)
threshold_chus = float(np.median(risk_chus))
threshold_chup = float(np.median(risk_chup))

print(f"\nDev median threshold  = {threshold_dev:.6f}")
print(f"CHUS median threshold = {threshold_chus:.6f}")
print(f"CHUP median threshold = {threshold_chup:.6f}")
print(f"\nWith dev threshold applied uniformly:")
print(f"  Dev  high-risk: {(risk_dev  >= threshold_dev).sum()} / {len(risk_dev)}")
print(f"  CHUS high-risk: {(risk_chus >= threshold_dev).sum()} / {len(risk_chus)}")
print(f"  CHUP high-risk: {(risk_chup >= threshold_dev).sum()} / {len(risk_chup)}")
print(f"\nWith cohort-specific medians:")
print(f"  Dev  high-risk: {(risk_dev  >= threshold_dev).sum()} / {len(risk_dev)}")
print(f"  CHUS high-risk: {(risk_chus >= threshold_chus).sum()} / {len(risk_chus)}")
print(f"  CHUP high-risk: {(risk_chup >= threshold_chup).sum()} / {len(risk_chup)}")


def run_km_pass(cohort_specs, suffix, suptitle, out_3panel):
    """Run one full KM pass (individual plots + 3-panel + summary table)."""
    summary_rows = []
    print(f"\nPlotting KM curves [{suffix}]...")

    for df_c, risk_c, thresh_c, name, ci_c in cohort_specs:
        fname = f"T1_KM_km_{name.lower()}{suffix}.png"
        out_path = OUT_DIR / fname
        n_h, n_l, p_lr = plot_km(df_c, risk_c, thresh_c, name, out_path, ci_val=ci_c)
        hr, hr_lo, hr_hi, p_cox = compute_hr(df_c, risk_c, thresh_c)
        print(f"  {name}: n_high={n_h}, n_low={n_l}, logrank_p={p_lr:.4f}, "
              f"HR={hr:.2f} [{hr_lo:.2f}-{hr_hi:.2f}], cox_p={p_cox:.4f}")
        summary_rows.append({
            "cohort":      name,
            "n_total":     len(df_c),
            "n_events":    int(df_c["Relapse"].sum()),
            "n_high_risk": n_h,
            "n_low_risk":  n_l,
            "threshold":   round(thresh_c, 6),
            "logrank_p":   round(p_lr, 6),
            "HR":          round(hr, 4),
            "HR_CI_lower": round(hr_lo, 4),
            "HR_CI_upper": round(hr_hi, 4),
            "cox_p":       round(p_cox, 6),
            "c_index":     round(ci_c, 6) if ci_c is not None else None,
        })

    # 3-panel figure
    print(f"Plotting combined 3-panel KM figure [{suffix}]...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, (df_c, risk_c, thresh_c, name, ci_c) in zip(axes, cohort_specs):
        group = (risk_c >= thresh_c).astype(int)
        duration_months = df_c["RFS"].values * DAYS_TO_MONTHS
        event = df_c["Relapse"].values.astype(bool)
        lr = logrank_test(
            duration_months[group == 1], duration_months[group == 0],
            event_observed_A=event[group == 1],
            event_observed_B=event[group == 0],
        )
        p_val = lr.p_value
        for grp_val, label, color in [
            (1, f"High risk (n={int(group.sum())})",      COLOR_HIGH),
            (0, f"Low risk  (n={int((group==0).sum())})", COLOR_LOW),
        ]:
            mask = group == grp_val
            if mask.sum() == 0:
                continue
            kmf = KaplanMeierFitter()
            kmf.fit(duration_months[mask], event_observed=event[mask], label=label)
            kmf.plot_survival_function(ax=ax, ci_show=True, color=color, linewidth=2)
        p_text = f"p = {p_val:.4f}" if p_val >= 0.0001 else "p < 0.0001"
        try:
            _cox_df = pd.DataFrame({
                "duration": df_c["RFS"].values * DAYS_TO_MONTHS,
                "event": df_c["Relapse"].values.astype(int),
                "high_risk": group,
            })
            _cph = CoxPHFitter()
            _cph.fit(_cox_df, duration_col="duration", event_col="event",
                     formula="high_risk", show_progress=False)
            _hr = float(np.exp(_cph.summary.loc["high_risk", "coef"]))
            _lo = float(np.exp(_cph.summary.loc["high_risk", "coef lower 95%"]))
            _hi = float(np.exp(_cph.summary.loc["high_risk", "coef upper 95%"]))
            hr_text = f"HR={_hr:.2f} ({_lo:.2f}–{_hi:.2f})"
        except Exception:
            hr_text = "HR=N/A"
        ci_text = f"C-index = {ci_c:.3f}" if ci_c is not None else ""
        ann = f"{p_text}\n{hr_text}\n{ci_text}" if ci_text else f"{p_text}\n{hr_text}"
        ax.text(0.97, 0.97, ann, transform=ax.transAxes, ha="right", va="top",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("Time (months)", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc="lower left")
    axes[0].set_ylabel("Relapse-free survival probability", fontsize=10)
    fig.suptitle(suptitle, fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT_DIR / out_3panel, dpi=300, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved: {out_3panel}")

    # summary CSV
    csv_name = f"T1_KM_km_summary_table{suffix}.csv"
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / csv_name, index=False)
    print(f"  Saved: {csv_name}")
    return summary_rows


# ============================================================
# PRIMARY: cohort-specific median split
# ============================================================
primary_specs = [
    (dev_df,  risk_dev,  threshold_dev,  "Dev",  None),
    (chus_df, risk_chus, threshold_chus, "CHUS", ci_chus),
    (chup_df, risk_chup, threshold_chup, "CHUP", ci_chup),
]
primary_rows = run_km_pass(
    primary_specs,
    suffix="",
    suptitle=(
        "RPM — Kaplan-Meier Survival Curves, Cohort-Specific Median Risk Split\n"
        "N=12 features (1 clinical + 7 PET + 4 CT) | Development, CHUS, and CHUP cohorts"
    ),
    out_3panel="T1_KM_km_combined_3panel.png",
)

# ============================================================
# SECONDARY: dev threshold applied uniformly (calibration reference)
# ============================================================
devthresh_specs = [
    (dev_df,  risk_dev,  threshold_dev, "Dev",  None),
    (chus_df, risk_chus, threshold_dev, "CHUS", ci_chus),
    (chup_df, risk_chup, threshold_dev, "CHUP", ci_chup),
]
devthresh_rows = run_km_pass(
    devthresh_specs,
    suffix="_devthresh",
    suptitle=(
        "RPM — Kaplan-Meier Survival Curves, Dev-Derived Threshold Transport (Calibration Reference)\n"
        "N=12 features (1 clinical + 7 PET + 4 CT) | Development, CHUS, and CHUP cohorts"
    ),
    out_3panel="T1_KM_km_combined_3panel_devthresh.png",
)

# ============================================================
# PRINT SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("BRANCH B — KM SUMMARY (PRIMARY: cohort-specific median split)")
print("=" * 60)
print(f"\nThresholds — Dev:{threshold_dev:.4f}  CHUS:{threshold_chus:.4f}  CHUP:{threshold_chup:.4f}")
print(f"\n{'Cohort':<6}  {'N':>4}  {'Events':>6}  {'N_high':>6}  {'N_low':>5}  "
      f"{'logrank_p':>10}  {'HR':>5}  {'95% CI':>14}  {'C-index':>8}")
for row in primary_rows:
    ci_str = f"{row['c_index']:.3f}" if row['c_index'] is not None else "  N/A "
    print(f"{row['cohort']:<6}  {row['n_total']:>4}  {row['n_events']:>6}  "
          f"{row['n_high_risk']:>6}  {row['n_low_risk']:>5}  "
          f"{row['logrank_p']:>10.4f}  {row['HR']:>5.2f}  "
          f"[{row['HR_CI_lower']:.2f}-{row['HR_CI_upper']:.2f}]  {ci_str:>8}")

print(f"\nAll outputs saved to: {OUT_DIR}")
print("Branch B complete.")
