"""
28_mar_task1_branchA_SHAP.py

Task 1 Post-Study — Branch A: xAI / SHAP Interpretation
Locked winner: PT_S3_1_768xCT_S3_8_235, N=12 (1 clinical + 7 PT + 4 CT)
Coach: ExtraSurvivalTrees(n_estimators=200, random_state=42)

Outputs (all saved to Mar_2026/28_mar_task1_post_study_outputs/branchA/):
  - beeswarm_dev.png
  - beeswarm_chus.png
  - beeswarm_chup.png
  - bar_mean_abs_shap.png          (mean |SHAP| per feature, all 3 cohorts overlaid)
  - dependence_top4_dev.png        (2x2 grid, top 4 features by mean |SHAP|)
  - waterfall_high_risk_dev.png
  - waterfall_low_risk_dev.png
  - waterfall_median_risk_dev.png
  - waterfall_chus065.png          (CHUS-065 cross-task linker; only with --chus065 flag)
  - modality_region_contribution.csv
  - shap_summary_table.csv         (mean |SHAP| per feature x cohort)

Reproducibility check: recomputed CHUS and CHUP C-index must match locked values
  CHUS = 0.742857142857143  (tolerance 0.001)
  CHUP = 0.727586206896552  (tolerance 0.001)
Script aborts if either delta exceeds tolerance.

Usage:
  python 28_mar_task1_branchA_SHAP.py             # full run (slow, ~10-15 min)
  python 28_mar_task1_branchA_SHAP.py --chus065   # fast mode: only add CHUS-065 waterfall
                                                   # skips CHUP SHAP and all other plots
"""

import argparse
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Auto-install shap if missing
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
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import ExtraSurvivalTrees
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

parser = argparse.ArgumentParser(description="RPM Branch A SHAP")
parser.add_argument(
    "--chus065",
    action="store_true",
    help="Fast mode: only compute CHUS-065 SHAP waterfall (skips CHUP and all other plots)",
)
_ARGS = parser.parse_args()

# ============================================================
# CONFIG
# ============================================================
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "28_mar_task1_post_study_outputs" / "branchA"
OUT_DIR.mkdir(exist_ok=True, parents=True)

PT_DEV_FILE   = ROOT / "27_feb_PT_development.csv"
CT_DEV_FILE   = ROOT / "27_feb_CT_development.csv"
PT_EXT_FILE   = ROOT / "27_feb_PT_external.csv"
CT_EXT_FILE   = ROOT / "27_feb_CT_external.csv"
CLINICAL_FILE = ROOT.parent / "Feb_2026" / "25_feb_clinical_reduced_dataset" / "25_feb_Processed_clinical_reduced.csv"
PT_FEAT_FILE  = ROOT / "2_mar_finalist_outputs" / "PT_inter1_768_features_recheck.csv"
CT_FEAT_FILE  = ROOT / "2_mar_finalist_outputs" / "CT_inter8_235_features_recheck.csv"

SEED = 42

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

def feature_label(feat):
    if feat == "Gender_Male":
        return feat
    parts = feat.split("_")
    if len(parts) >= 4 and parts[0].startswith("GTV"):
        mod = modality_of(feat)
        region = parts[0]
        transform = parts[1].replace("wavelet-", "wav-")
        metric = parts[-1]
        return f"{mod} {region} {transform} {metric}"
    return "_".join(parts[-2:]) if len(parts) >= 2 else feat

def modality_of(feat):
    if feat in CLINICAL_FEATURES:
        return "Clinical"
    if feat in PT_WINNER:
        return "PET"
    return "CT"

def region_of(feat):
    if feat in CLINICAL_FEATURES:
        return "Clinical"
    if "GTVp" in feat:
        return "GTVp"
    if "GTVn" in feat:
        return "GTVn"
    return "Unknown"

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

pt_dev  = pd.read_csv(PT_DEV_FILE)
ct_dev  = pd.read_csv(CT_DEV_FILE)
pt_ext  = pd.read_csv(PT_EXT_FILE)
ct_ext  = pd.read_csv(CT_EXT_FILE)

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
# SHAP COMPUTATION
# ============================================================
# ExtraSurvivalTrees is not supported by TreeExplainer.
# Use PermutationExplainer with a kmeans-summarized background (100 clusters).
# predict() returns a float risk score, so this is a standard regression SHAP task.
print("\nComputing SHAP values (PermutationExplainer, sampled background n=100)...")
rng = np.random.default_rng(SEED)
bg_idx    = rng.choice(len(X_dev_sc), size=100, replace=False)
background = X_dev_sc[bg_idx]
explainer  = shap.PermutationExplainer(model.predict, background)

print("  Computing SHAP for dev...")
exp_dev  = explainer(X_dev_sc)
print("  Computing SHAP for CHUS...")
exp_chus = explainer(X_chus_sc)

if not _ARGS.chus065:
    print("  Computing SHAP for CHUP...")
    exp_chup = explainer(X_chup_sc)
    shap_chup = np.array(exp_chup.values, dtype=float)
else:
    exp_chup  = None
    shap_chup = None

shap_dev  = np.array(exp_dev.values,  dtype=float)
shap_chus = np.array(exp_chus.values, dtype=float)

short_labels = [feature_label(f) for f in WINNER_FEATURES]

# ============================================================
# HELPER: AXIS LABEL CLEANUP
# ============================================================
def clean_shap_axis(ax):
    """Remove redundant 'f(x)' or long feature paths from shap bar chart axes."""
    ax.set_xlabel("mean(|SHAP value|)", fontsize=10)

if not _ARGS.chus065:
    # ============================================================
    # PLOT 1-3: BEESWARM per cohort
    # ============================================================
    print("Plotting beeswarm plots...")
    for tag, sv, Xsc, n_ev in [
        ("dev",  shap_dev,  X_dev_sc,  int(y_dev["event"].sum())),
        ("chus", shap_chus, X_chus_sc, int(y_chus["event"].sum())),
        ("chup", shap_chup, X_chup_sc, int(y_chup["event"].sum())),
    ]:
        fig, ax = plt.subplots(figsize=(9, 6))
        shap.summary_plot(
            sv, Xsc,
            feature_names=short_labels,
            show=False, plot_type="dot",
            max_display=12,
        )
        plt.title(f"RPM — SHAP Beeswarm, {tag.upper()} Cohort (N={Xsc.shape[0]}, events={n_ev})", fontsize=11)
        plt.tight_layout()
        out_path = OUT_DIR / f"T1_SHAP_beeswarm_{tag}.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close("all")
        print(f"  Saved: {out_path.name}")

    # ============================================================
    # PLOT 4: Mean |SHAP| bar — all 3 cohorts overlaid
    # ============================================================
    print("Plotting mean |SHAP| bar chart...")
    mean_abs = {
        "Dev":  np.abs(shap_dev).mean(axis=0),
        "CHUS": np.abs(shap_chus).mean(axis=0),
        "CHUP": np.abs(shap_chup).mean(axis=0),
    }
    order = np.argsort(mean_abs["Dev"])[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(WINNER_FEATURES))
    w = 0.25
    colors = {"Dev": "#1f77b4", "CHUS": "#ff7f0e", "CHUP": "#2ca02c"}
    for i, (label, vals) in enumerate(mean_abs.items()):
        ax.bar(x + (i - 1) * w, vals[order], w, label=label, color=colors[label], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([short_labels[i] for i in order], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("mean(|SHAP value|)", fontsize=10)
    ax.set_title("RPM — Mean |SHAP| per Feature across Development, CHUS, and CHUP Cohorts", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out_path = OUT_DIR / "T1_SHAP_bar_mean_abs_shap.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved: {out_path.name}")

    # ============================================================
    # PLOT 5: Dependence plots — top 4 features by dev mean |SHAP|
    # ============================================================
    print("Plotting dependence plots (top 4 features)...")
    top4_idx = np.argsort(mean_abs["Dev"])[::-1][:4]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, fidx in zip(axes.flat, top4_idx):
        shap.dependence_plot(
            fidx,
            shap_dev,
            X_dev_sc,
            feature_names=short_labels,
            ax=ax,
            show=False,
        )
        ax.set_title(short_labels[fidx], fontsize=9)

    plt.suptitle("RPM — SHAP Dependence Plots for Top 4 Features (Development Cohort, standardised units)", fontsize=11)
    plt.tight_layout()
    out_path = OUT_DIR / "T1_SHAP_dependence_top4_dev.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved: {out_path.name}")

    # ============================================================
    # PLOT 6-8: Waterfall — high / low / median risk patients (dev)
    # ============================================================
    print("Plotting waterfall plots (high / low / median risk, dev)...")
    risk_order = np.argsort(risk_dev)
    idx_high   = risk_order[-1]
    idx_low    = risk_order[0]
    idx_median = risk_order[len(risk_order) // 2]

    # base_values come directly from the PermutationExplainer Explanation object
    base_val = float(np.mean(exp_dev.base_values)) if hasattr(exp_dev, "base_values") else float(explainer.expected_value)

    for tag, pidx in [("high_risk", idx_high), ("low_risk", idx_low), ("median_risk", idx_median)]:
        exp_obj = shap.Explanation(
            values=shap_dev[pidx],
            base_values=base_val,
            data=X_dev_sc[pidx],
            feature_names=short_labels,
        )
        fig, ax = plt.subplots(figsize=(9, 5))
        shap.waterfall_plot(exp_obj, max_display=12, show=False)
        for txt in plt.gca().texts:
            txt.set_color("black")
            txt.set_fontsize(8)
        plt.title(f"RPM — SHAP Waterfall, {tag.replace('_', ' ').title()} Patient (Dev, PatientID={dev_df.iloc[pidx]['PatientID']})", fontsize=10)
        plt.tight_layout()
        out_path = OUT_DIR / f"T1_SHAP_waterfall_{tag}_dev.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close("all")
        print(f"  Saved: {out_path.name}")

    # ============================================================
    # TABLE 1: SHAP summary table — mean |SHAP| per feature x cohort
    # ============================================================
    print("Saving SHAP summary table...")
    shap_table = pd.DataFrame({
        "feature":    WINNER_FEATURES,
        "short_label": short_labels,
        "modality":   [modality_of(f) for f in WINNER_FEATURES],
        "region":     [region_of(f)   for f in WINNER_FEATURES],
        "mean_abs_shap_dev":  mean_abs["Dev"],
        "mean_abs_shap_chus": mean_abs["CHUS"],
        "mean_abs_shap_chup": mean_abs["CHUP"],
    }).sort_values("mean_abs_shap_dev", ascending=False)

    out_path = OUT_DIR / "T1_SHAP_shap_summary_table.csv"
    shap_table.to_csv(out_path, index=False)
    print(f"  Saved: {out_path.name}")

    # ============================================================
    # TABLE 2: Modality / region contribution
    # ============================================================
    print("Saving modality/region contribution table...")
    rows = []
    for cohort, sv in [("Dev", shap_dev), ("CHUS", shap_chus), ("CHUP", shap_chup)]:
        abs_sv = np.abs(sv)
        total  = abs_sv.sum()
        for grp_col, grp_fn in [("modality", modality_of), ("region", region_of)]:
            for grp_val in (["PET", "CT", "Clinical"] if grp_col == "modality" else ["GTVp", "GTVn", "Clinical"]):
                feat_mask = np.array([grp_fn(f) == grp_val for f in WINNER_FEATURES])
                contrib = float(abs_sv[:, feat_mask].sum() / total * 100) if feat_mask.any() else 0.0
                rows.append({"cohort": cohort, grp_col: grp_val, "contribution_pct": round(contrib, 2)})

    contrib_df = pd.DataFrame(rows)
    out_path = OUT_DIR / "T1_SHAP_modality_region_contribution.csv"
    contrib_df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path.name}")

    # ============================================================
    # PRINT SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    print("BRANCH A — SHAP SUMMARY")
    print("=" * 60)
    print(f"\nReproduced CHUS={ci_chus:.6f}  CHUP={ci_chup:.6f}")
    print("\nTop 5 features by mean |SHAP| (dev):")
    for _, row in shap_table.head(5).iterrows():
        print(f"  {row['short_label']:40s}  {row['modality']:8s}  {row['region']:8s}  "
              f"dev={row['mean_abs_shap_dev']:.4f}  chus={row['mean_abs_shap_chus']:.4f}  chup={row['mean_abs_shap_chup']:.4f}")

    print("\nModality contribution (Dev):")
    mod_rows = contrib_df[(contrib_df["cohort"] == "Dev") & contrib_df["modality"].notna()]
    for _, row in mod_rows.iterrows():
        print(f"  {row['modality']:10s}  {row['contribution_pct']:.1f}%")

    print("\nRegion contribution (Dev):")
    reg_rows = contrib_df[(contrib_df["cohort"] == "Dev") & contrib_df["region"].notna()]
    for _, row in reg_rows.iterrows():
        print(f"  {row['region']:10s}  {row['contribution_pct']:.1f}%")

    print(f"\nAll outputs saved to: {OUT_DIR}")

# ============================================================
# PLOT 9: Waterfall — CHUS-065 cross-task linker (always runs)
# ============================================================
print("\nPlotting CHUS-065 waterfall (cross-task linker)...")
chus065_mask = chus_df["PatientID"] == "CHUS-065"
if not chus065_mask.any():
    print("  WARNING: CHUS-065 not found in CHUS cohort — skipping waterfall.")
else:
    chus065_idx = int(np.where(chus065_mask.values)[0][0])
    base_val_chus = (
        float(np.mean(exp_chus.base_values))
        if hasattr(exp_chus, "base_values")
        else float(explainer.expected_value)
    )
    exp_c065 = shap.Explanation(
        values=shap_chus[chus065_idx],
        base_values=base_val_chus,
        data=X_chus_sc[chus065_idx],
        feature_names=short_labels,
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    shap.waterfall_plot(exp_c065, max_display=12, show=False)
    for txt in plt.gca().texts:
        txt.set_color("black")
        txt.set_fontsize(8)
    plt.title("RPM — SHAP Waterfall, CHUS-065 (External Cross-Task Linker, CHUS)", fontsize=10)
    plt.tight_layout()
    out_path = OUT_DIR / "T1_SHAP_waterfall_chus065.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved: {out_path.name}")

print("Branch A complete.")
