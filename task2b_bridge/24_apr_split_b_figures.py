"""Regenerate T2B_B_interpretation.png as 3 separate PNGs.

Panels:
  T2B_B1_hpv_lr_coefficients.png  - HPV LR coefficients (B.1)
  T2B_B2_survival_permutation.png  - EST permutation importance (B.2)
  T2B_B3_hpv_shap.png             - SHAP mean |SHAP| HPV head (B.3)

Source CSVs (already exist in 10_apr_postudy_outputs/):
  T2B_B_lr_coefficients.csv
  T2B_B_permutation_importance.csv
  T2B_B_shap_hpv_head.csv
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent / "10_apr_postudy_outputs"

COLOR_MAP = {"PT": "#f58518", "CT": "#54a24b", "Clinical": "#4c78a8"}

FEATURE_LABELS = {
    "GTVp_wavelet-HHL_glrlm_ShortRunHighGrayLevelEmphasis": "GTVp wav-HHL GLRLM SRHGLE (B4)",
    "GTVp_original_firstorder_InterquartileRange":          "GTVp original IQR (B3)",
    "GTVp_wavelet-HLH_glrlm_ShortRunHighGrayLevelEmphasis": "GTVp wav-HLH GLRLM SRHGLE (B5)",
    "GTVp_log-sigma-1-mm-3D_firstorder_Range":              "GTVp LoG-1mm Range (B6)",
    "GTVp_wavelet-HLL_ngtdm_Complexity":                    "GTVp wav-HLL NGTDM Complexity (B7)",
    "GTVp_wavelet-LLH_glrlm_HighGrayLevelRunEmphasis":      "GTVp wav-LLH GLRLM HGRE (B8)",
    "Gender_Male":                                           "Gender Male (B1)",
    "GTVn_wavelet-LLH_firstorder_Mean":                     "GTVn wav-LLH Mean (B2)",
}


def label(feat: str) -> str:
    return FEATURE_LABELS.get(feat, feat)


# --- Panel B.1: HPV LR coefficients ---
lr = pd.read_csv(OUT_DIR / "T2B_B_lr_coefficients.csv")
lr["label"] = lr["Feature"].map(label)
lr_sorted = lr.sort_values("abs_coef", ascending=True)

fig, ax = plt.subplots(figsize=(7, 5))
colors = [("#f58518" if m == "PT" else "#54a24b" if m == "CT" else "#4c78a8")
          for m in lr_sorted["Modality"]]
bars = ax.barh(lr_sorted["label"], lr_sorted["Coefficient"], color=colors)
ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("LR coefficient (red=HPV−, blue=HPV+)", fontsize=10)
ax.set_title("B.1 HPV classification arm\nLR_L2_0.5 coefficients", fontsize=11)
# colour legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor="#f58518", label="PT"),
                   Patch(facecolor="#54a24b", label="CT"),
                   Patch(facecolor="#4c78a8", label="Clinical")]
ax.legend(handles=legend_elements, fontsize=9, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
out1 = OUT_DIR / "T2B_B1_hpv_lr_coefficients.png"
fig.savefig(out1, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out1}")


# --- Panel B.2: EST permutation importance ---
perm = pd.read_csv(OUT_DIR / "T2B_B_permutation_importance.csv")
perm["label"] = perm["Feature"].map(label)
perm_sorted = perm.sort_values("CI_drop_mean", ascending=True)

fig, ax = plt.subplots(figsize=(7, 5))
colors = [("#f58518" if m == "PT" else "#54a24b" if m == "CT" else "#4c78a8")
          for m in perm_sorted["Modality"]]
ax.barh(perm_sorted["label"], perm_sorted["CI_drop_mean"], color=colors)
ax.set_xlabel("Mean C-index drop (permutation, CHUS)", fontsize=10)
ax.set_title("B.2 Survival prediction arm\nEST permutation importance (CHUS)", fontsize=11)
legend_elements = [Patch(facecolor="#f58518", label="PT"),
                   Patch(facecolor="#54a24b", label="CT"),
                   Patch(facecolor="#4c78a8", label="Clinical")]
ax.legend(handles=legend_elements, fontsize=9, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
out2 = OUT_DIR / "T2B_B2_survival_permutation.png"
fig.savefig(out2, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out2}")


# --- Panel B.3: HPV SHAP mean |SHAP| ---
shap_df = pd.read_csv(OUT_DIR / "T2B_B_shap_hpv_head.csv")
shap_df["label"] = shap_df["Feature"].map(label)
shap_sorted = shap_df.sort_values("mean_abs_SHAP", ascending=True)

fig, ax = plt.subplots(figsize=(7, 5))
colors = [("#f58518" if m == "PT" else "#54a24b" if m == "CT" else "#4c78a8")
          for m in shap_sorted["Modality"]]
ax.barh(shap_sorted["label"], shap_sorted["mean_abs_SHAP"], color=colors)
ax.set_xlabel("Mean |SHAP| (linear SHAP, HPV head, train N=67)", fontsize=10)
ax.set_title("B.3 HPV classification arm\nMean |SHAP| attribution", fontsize=11)
legend_elements = [Patch(facecolor="#f58518", label="PT"),
                   Patch(facecolor="#54a24b", label="CT"),
                   Patch(facecolor="#4c78a8", label="Clinical")]
ax.legend(handles=legend_elements, fontsize=9, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
out3 = OUT_DIR / "T2B_B3_hpv_shap.png"
fig.savefig(out3, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out3}")

print("Done. Three separate B-panel figures written to", OUT_DIR)
