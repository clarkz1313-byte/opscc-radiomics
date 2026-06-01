"""
29_mar_task2_branchA_SHAP.py

Task 2 Post-Study - Branch A: xAI / SHAP Interpretation
Locked winner: T68357, N=16 (1 clinical + 8 PT + 7 CT)
Coach: LR_L2, C=0.3689386715674865

Outputs (saved to Mar_2026_task2/29_mar_task2_post_study_outputs/branchA/):
  - beeswarm_dev.png
  - beeswarm_chus.png
  - bar_mean_abs_shap.png
  - dependence_top4_dev.png
  - waterfall_high_hpv.png
  - waterfall_low_hpv.png
  - waterfall_hpvneg_correct.png
  - shap_summary_table.csv
  - modality_region_contribution.csv
  - coefficient_table.csv
  - cross_task_shap_comparison.csv

Smoke mode:
  python Mar_2026_task2/29_mar_task2_branchA_SHAP.py --smoke

Reproducibility guard:
  AUC_ext  = 0.7785714285714286  (tolerance 0.001)
  BA_ext   = 0.775               (tolerance 0.01)
  Spe_ext  = 1.0                 (tolerance 0.01)
  Threshold= 0.8719759300122685

Important implementation note:
  The Task 2 CT/PT split CSVs each contain the full column set. This script does not
  merge the full CT and PT tables directly. It selects CT features from the CT file,
  PT features from the PT file, and labels/clinical columns from the CT side, then
  merges only the selected PT feature columns by PatientID to avoid duplicate-column
  collisions.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIG
# ============================================================
ROOT = Path(__file__).resolve().parent
RAD_DIR = ROOT / "12_mar_task2_rad_data"
OUT_DIR = ROOT / "29_mar_task2_post_study_outputs" / "branchA"

CT_TRAIN_FILE = RAD_DIR / "13_mar_task2_CT_primary_train.csv"
CT_TEST_FILE = RAD_DIR / "13_mar_task2_CT_primary_test.csv"
CT_EXT_FILE = RAD_DIR / "12_mar_task2_CT_primary_ext.csv"
PT_TRAIN_FILE = RAD_DIR / "13_mar_task2_PT_primary_train.csv"
PT_TEST_FILE = RAD_DIR / "13_mar_task2_PT_primary_test.csv"
PT_EXT_FILE = RAD_DIR / "12_mar_task2_PT_primary_ext.csv"

TASK1_SHAP_SUMMARY_FILE = (
    ROOT.parent / "Mar_2026" / "28_mar_task1_post_study_outputs" / "branchA" / "T2_SHAP_shap_summary_table.csv"
)

SEED = 42

CLINICAL_FEATURES = ["Gender_Male"]
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
WINNER_FEATURES = CLINICAL_FEATURES + PT_WINNER + CT_WINNER

LOCKED_COACH_PARAMS = {"C": 0.3689386715674865}
LOCKED_AUC_EXT = 0.7785714285714286
LOCKED_BA_EXT = 0.775
LOCKED_SPE_EXT = 1.0
LOCKED_YOUDEN_EXT = 0.8719759300122685
REPRO_TOL_AUC = 0.001
REPRO_TOL_BA = 0.01
REPRO_TOL_SPE = 0.01

TASK1_DOC_FALLBACK = {
    "GTVp_wavelet-HHL_glrlm_ShortRunHighGrayLevelEmphasis": {"rank": "6", "mean_abs_shap_dev": "1.041"},
    "GTVp_original_firstorder_InterquartileRange": {"rank": ">12", "mean_abs_shap_dev": ""},
    "GTVn_wavelet-LHL_glszm_GrayLevelVariance": {"rank": ">12", "mean_abs_shap_dev": ""},
    "GTVn_logarithm_glszm_SmallAreaLowGrayLevelEmphasis": {"rank": ">12", "mean_abs_shap_dev": ""},
    "GTVn_wavelet-LLH_firstorder_Skewness": {"rank": ">12", "mean_abs_shap_dev": ""},
    "GTVp_wavelet-LLH_firstorder_Median": {"rank": ">12", "mean_abs_shap_dev": ""},
}


# ============================================================
# HELPERS
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 2 Branch A SHAP analysis for T68357.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run column/path checks, refit the locked model, and verify reproducibility only.",
    )
    return parser.parse_args()


def ensure_shap():
    try:
        import shap  # type: ignore

        return shap
    except ImportError:
        print("Installing shap...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "shap", "-q"])
        import shap  # type: ignore

        return shap


def modality_of(feature: str) -> str:
    if feature in CLINICAL_FEATURES:
        return "Clinical"
    if feature in PT_WINNER:
        return "PET"
    if feature in CT_WINNER:
        return "CT"
    return "Unknown"


def region_of(feature: str) -> str:
    if feature in CLINICAL_FEATURES:
        return "Clinical"
    if "GTVp" in feature:
        return "GTVp"
    if "GTVn" in feature:
        return "GTVn"
    return "Unknown"


def feature_label(feature: str) -> str:
    if feature == "Gender_Male":
        return feature
    parts = feature.split("_")
    if len(parts) >= 4 and parts[0].startswith("GTV"):
        mod = modality_of(feature)
        region = parts[0]
        transform = parts[1].replace("wavelet-", "wav-").replace("log-sigma-1-mm-3D", "LoG-1mm")
        metric = parts[-1]
        return f"{mod} {region} {transform} {metric}"
    return "_".join(parts[-2:]) if len(parts) >= 2 else feature


DISPLAY_FEATURES = [feature_label(f) for f in WINNER_FEATURES]


def build_split(ct_path: Path, pt_path: Path) -> pd.DataFrame:
    ct_df = pd.read_csv(ct_path)
    pt_df = pd.read_csv(pt_path)

    ct_keep = ["PatientID", "HPV_binary"] + CLINICAL_FEATURES + CT_WINNER
    pt_keep = ["PatientID"] + PT_WINNER

    merged = ct_df[ct_keep].merge(pt_df[pt_keep], on="PatientID", how="inner", validate="one_to_one")
    return merged.copy()


def verify_inputs() -> None:
    print("=" * 72)
    print("INPUT CHECKS")
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
        print(f"{label:10s} {'PASS' if ok else 'FAIL'}  {path}")
        all_ok &= ok
    if not all_ok:
        raise FileNotFoundError("One or more required input files are missing.")

    required_cols = ["PatientID", "HPV_binary", "Gender_Male"] + PT_WINNER + CT_WINNER
    for label, path in paths.items():
        cols = pd.read_csv(path, nrows=1).columns.tolist()
        missing = [col for col in required_cols if col not in cols]
        ok = len(missing) == 0
        print(f"{label:10s} {'PASS' if ok else 'FAIL'}  ncols={len(cols)}")
        print(f"  FIRST20_COLS: {' | '.join(cols[:20])}")
        if missing:
            print(f"  MISSING: {missing}")
        if not ok:
            raise ValueError(f"Missing required columns in {path}: {missing}")

    ct_cols = pd.read_csv(CT_TRAIN_FILE, nrows=1).columns.tolist()
    pt_cols = pd.read_csv(PT_TRAIN_FILE, nrows=1).columns.tolist()
    print(f"train_csv_columns_equal: {'PASS' if ct_cols == pt_cols else 'FAIL'}")
    if ct_cols != pt_cols:
        raise ValueError("CT/PT train CSV columns differ unexpectedly.")


def fit_scalers_and_model(train_df: pd.DataFrame) -> tuple[StandardScaler, StandardScaler, StandardScaler, LogisticRegression]:
    x_ct = train_df[CT_WINNER].to_numpy(dtype=float)
    x_pt = train_df[PT_WINNER].to_numpy(dtype=float)
    x_clin = train_df[CLINICAL_FEATURES].to_numpy(dtype=float)
    y = train_df["HPV_binary"].to_numpy(dtype=int)

    sc_ct = StandardScaler().fit(x_ct)
    sc_pt = StandardScaler().fit(x_pt)
    sc_clin = StandardScaler().fit(x_clin)

    x_train_sc = np.hstack([sc_clin.transform(x_clin), sc_pt.transform(x_pt), sc_ct.transform(x_ct)])

    model = LogisticRegression(
        C=LOCKED_COACH_PARAMS["C"],
        penalty="l2",
        solver="lbfgs",
        class_weight="balanced",
        max_iter=2000,
        random_state=SEED,
    )
    model.fit(x_train_sc, y)
    return sc_ct, sc_pt, sc_clin, model


def transform_split(
    df: pd.DataFrame, sc_ct: StandardScaler, sc_pt: StandardScaler, sc_clin: StandardScaler
) -> tuple[np.ndarray, np.ndarray]:
    x_ct = df[CT_WINNER].to_numpy(dtype=float)
    x_pt = df[PT_WINNER].to_numpy(dtype=float)
    x_clin = df[CLINICAL_FEATURES].to_numpy(dtype=float)
    x_sc = np.hstack([sc_clin.transform(x_clin), sc_pt.transform(x_pt), sc_ct.transform(x_ct)])
    y = df["HPV_binary"].to_numpy(dtype=int)
    return x_sc, y


def compute_binary_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    spe = tn / (tn + fp) if (tn + fp) else np.nan
    sen = tp / (tp + fn) if (tp + fn) else np.nan
    ba = balanced_accuracy_score(y_true, pred)
    return {"ba": float(ba), "spe": float(spe), "sen": float(sen), "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def run_reproducibility_check(model: LogisticRegression, x_ext_sc: np.ndarray, y_ext: np.ndarray) -> dict[str, float]:
    proba_ext = model.predict_proba(x_ext_sc)[:, 1]
    auc_ext = float(roc_auc_score(y_ext, proba_ext))
    metrics = compute_binary_metrics(y_ext, proba_ext, LOCKED_YOUDEN_EXT)

    delta_auc = abs(auc_ext - LOCKED_AUC_EXT)
    delta_ba = abs(metrics["ba"] - LOCKED_BA_EXT)
    delta_spe = abs(metrics["spe"] - LOCKED_SPE_EXT)

    print("\nReproducibility check:")
    print(f"  AUC_ext = {auc_ext:.6f}  locked={LOCKED_AUC_EXT:.6f}  delta={delta_auc:.6f}")
    print(f"  BA_ext  = {metrics['ba']:.6f}  locked={LOCKED_BA_EXT:.6f}  delta={delta_ba:.6f}")
    print(f"  Spe_ext = {metrics['spe']:.6f}  locked={LOCKED_SPE_EXT:.6f}  delta={delta_spe:.6f}")
    print(f"  Sen_ext = {metrics['sen']:.6f}  locked=0.550000")

    if delta_auc > REPRO_TOL_AUC or delta_ba > REPRO_TOL_BA or delta_spe > REPRO_TOL_SPE:
        raise RuntimeError(
            "Reproducibility check FAILED. "
            f"AUC delta={delta_auc:.6f}, BA delta={delta_ba:.6f}, Spe delta={delta_spe:.6f}."
        )

    print("Reproducibility check PASSED.")
    metrics["auc_ext"] = auc_ext
    return metrics


def make_explanation(shap, values: np.ndarray, base_values: np.ndarray, data: np.ndarray) -> "shap.Explanation":
    return shap.Explanation(
        values=values,
        base_values=base_values,
        data=data,
        feature_names=DISPLAY_FEATURES,
    )


def save_beeswarm(shap, explanation, title: str, out_path: Path) -> None:
    shap.summary_plot(
        explanation.values,
        explanation.data,
        feature_names=DISPLAY_FEATURES,
        plot_type="dot",
        max_display=len(WINNER_FEATURES),
        plot_size=(14, 9),
        show=False,
    )
    plt.gca().tick_params(axis="y", labelsize=11)
    plt.gca().tick_params(axis="x", labelsize=10)
    plt.title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close("all")


def save_mean_abs_bar(mean_abs_dev: np.ndarray, mean_abs_chus: np.ndarray, out_path: Path) -> None:
    order = np.argsort(mean_abs_dev)[::-1]
    labels = [feature_label(WINNER_FEATURES[idx]) for idx in order]
    n = len(order)
    y = np.arange(n)
    bar_h = 0.38

    fig, ax = plt.subplots(figsize=(12, 8))
    # Dev bars above centre, CHUS bars below — side-by-side to avoid overlap
    ax.barh(y - bar_h / 2, mean_abs_dev[order], height=bar_h, color="#2166ac", alpha=0.90, label="Dev (N=67)")
    ax.barh(y + bar_h / 2, mean_abs_chus[order], height=bar_h, color="#4daf4a", alpha=0.85, label="CHUS (N=27)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_title("HCM Mean |SHAP| per Feature — Dev vs CHUS")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close("all")


def save_dependence_grid(
    x_dev_raw: np.ndarray, shap_dev: np.ndarray, mean_abs_dev: np.ndarray, out_path: Path
) -> list[str]:
    top4_idx = np.argsort(mean_abs_dev)[::-1][:4]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for ax, feat_idx in zip(axes.flat, top4_idx):
        xvals = x_dev_raw[:, feat_idx]
        yvals = shap_dev[:, feat_idx]
        sc = ax.scatter(xvals, yvals, c=xvals, cmap="coolwarm", s=34, alpha=0.85, edgecolors="none")
        ax.set_title(DISPLAY_FEATURES[feat_idx], fontsize=10)
        ax.set_xlabel("Feature value", fontsize=9)
        ax.set_ylabel("SHAP value", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.03)
        cbar.ax.tick_params(labelsize=8)

    plt.suptitle("SHAP Dependence Plots - Top 4 Features (Dev)", fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close("all")
    return [WINNER_FEATURES[idx] for idx in top4_idx]


def save_waterfall(shap, explanation_row, title: str, out_path: Path) -> None:
    plt.figure(figsize=(12, 7.5))
    shap.plots.waterfall(explanation_row, max_display=len(WINNER_FEATURES), show=False)
    ax = plt.gca()
    for txt in ax.texts:
        txt.set_color("black")
        txt.set_fontsize(8)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=9)
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close("all")


def load_task1_shap_reference() -> pd.DataFrame:
    rows = []
    if TASK1_SHAP_SUMMARY_FILE.exists():
        task1_df = pd.read_csv(TASK1_SHAP_SUMMARY_FILE)
        task1_df = task1_df.sort_values("mean_abs_shap_dev", ascending=False).reset_index(drop=True)
        task1_df["task1_rank"] = np.arange(1, len(task1_df) + 1)
        for feature in PT_WINNER:
            match = task1_df[task1_df["feature"] == feature]
            if match.empty:
                rows.append(
                    {
                        "feature": feature,
                        "task1_rank": ">12",
                        "task1_mean_abs_shap_dev": "",
                        "task1_reference_source": "task1_csv_absent_from_top_table",
                    }
                )
            else:
                rank = int(match["task1_rank"].iloc[0])
                rows.append(
                    {
                        "feature": feature,
                        "task1_rank": str(rank) if rank <= 12 else ">12",
                        "task1_mean_abs_shap_dev": float(match["mean_abs_shap_dev"].iloc[0]),
                        "task1_reference_source": "task1_shap_summary_csv",
                    }
                )
        return pd.DataFrame(rows)

    for feature in PT_WINNER:
        fallback = TASK1_DOC_FALLBACK.get(feature, {"rank": ">12", "mean_abs_shap_dev": ""})
        rows.append(
            {
                "feature": feature,
                "task1_rank": fallback["rank"],
                "task1_mean_abs_shap_dev": fallback["mean_abs_shap_dev"],
                "task1_reference_source": "doc_fallback",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    verify_inputs()

    print("\nLoading Task 2 Branch A data...")
    train_df = build_split(CT_TRAIN_FILE, PT_TRAIN_FILE)
    test_df = build_split(CT_TEST_FILE, PT_TEST_FILE)
    chus_df = build_split(CT_EXT_FILE, PT_EXT_FILE)

    print(f"  train={len(train_df)}  test={len(test_df)}  chus={len(chus_df)}")
    print(
        f"  train HPV+: {int(train_df['HPV_binary'].sum())} / {len(train_df)}"
        f" | chus HPV+: {int(chus_df['HPV_binary'].sum())} / {len(chus_df)}"
    )

    sc_ct, sc_pt, sc_clin, model = fit_scalers_and_model(train_df)
    x_train_sc, y_train = transform_split(train_df, sc_ct, sc_pt, sc_clin)
    x_test_sc, y_test = transform_split(test_df, sc_ct, sc_pt, sc_clin)
    x_chus_sc, y_chus = transform_split(chus_df, sc_ct, sc_pt, sc_clin)
    _ = y_test  # loaded for smoke checks and future extension; not used in current outputs
    x_train_raw = train_df[WINNER_FEATURES].to_numpy(dtype=float)
    x_chus_raw = chus_df[WINNER_FEATURES].to_numpy(dtype=float)

    repro = run_reproducibility_check(model, x_chus_sc, y_chus)

    if args.smoke:
        print("\nSmoke mode complete. Full SHAP run not started.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    shap = ensure_shap()

    print("\nComputing SHAP values...")
    explainer = shap.LinearExplainer(model, x_train_sc)
    shap_dev = np.asarray(explainer.shap_values(x_train_sc), dtype=float)
    shap_chus = np.asarray(explainer.shap_values(x_chus_sc), dtype=float)

    base_dev = np.asarray(explainer.expected_value, dtype=float)
    if base_dev.ndim == 0:
        base_dev = np.full(len(train_df), float(base_dev))
        base_chus = np.full(len(chus_df), float(explainer.expected_value))
    else:
        base_dev = np.full(len(train_df), float(base_dev[0]))
        base_chus = np.full(len(chus_df), float(base_dev[0]))

    exp_dev = make_explanation(shap, shap_dev, base_dev, x_train_raw)
    exp_chus = make_explanation(shap, shap_chus, base_chus, x_chus_raw)

    print("Saving plots...")
    save_beeswarm(shap, exp_dev, "SHAP beeswarm - Task 2 dev cohort (n=67)", OUT_DIR / "T2_SHAP_beeswarm_dev.png")
    save_beeswarm(shap, exp_chus, "SHAP beeswarm - CHUS external cohort (n=27)", OUT_DIR / "T2_SHAP_beeswarm_chus.png")

    mean_abs_dev = np.abs(shap_dev).mean(axis=0)
    mean_abs_chus = np.abs(shap_chus).mean(axis=0)
    save_mean_abs_bar(mean_abs_dev, mean_abs_chus, OUT_DIR / "T2_SHAP_bar_mean_abs_shap.png")
    _ = save_dependence_grid(x_train_raw, shap_dev, mean_abs_dev, OUT_DIR / "T2_SHAP_dependence_top4_dev.png")

    proba_chus = model.predict_proba(x_chus_sc)[:, 1]
    pred_chus = (proba_chus >= LOCKED_YOUDEN_EXT).astype(int)

    # Case 1: most confidently HPV+ (true positive, highest probability)
    hpvpos_correct = np.where((y_chus == 1) & (pred_chus == 1))[0]
    if len(hpvpos_correct) == 0:
        raise RuntimeError("No correctly classified HPV-positive CHUS patient found for waterfall plot.")
    idx_high = int(hpvpos_correct[np.argmax(proba_chus[hpvpos_correct])])

    # Case 2: least confidently HPV+ among true positives (borderline correct HPV+)
    idx_low = int(hpvpos_correct[np.argmin(proba_chus[hpvpos_correct])])

    # Case 3: most confidently HPV- (true negative, lowest probability)
    hpvneg_correct = np.where((y_chus == 0) & (pred_chus == 0))[0]
    if len(hpvneg_correct) == 0:
        raise RuntimeError("No correctly classified HPV-negative CHUS patient found for waterfall plot.")
    idx_neg = int(hpvneg_correct[np.argmin(proba_chus[hpvneg_correct])])

    def prob_label(p: float) -> str:
        return f"P(HPV+)={p:.3f}"

    save_waterfall(
        shap,
        exp_chus[idx_high],
        f"SHAP Waterfall — Most confident HPV+ [{chus_df.iloc[idx_high]['PatientID']}, {prob_label(proba_chus[idx_high])}]",
        OUT_DIR / "T2_SHAP_waterfall_high_hpv.png",
    )
    save_waterfall(
        shap,
        exp_chus[idx_low],
        f"SHAP Waterfall — Borderline HPV+ (lowest confidence TP) [{chus_df.iloc[idx_low]['PatientID']}, {prob_label(proba_chus[idx_low])}]",
        OUT_DIR / "T2_SHAP_waterfall_low_hpv.png",
    )
    save_waterfall(
        shap,
        exp_chus[idx_neg],
        f"SHAP Waterfall — Most confident HPV- [{chus_df.iloc[idx_neg]['PatientID']}, {prob_label(proba_chus[idx_neg])}]",
        OUT_DIR / "T2_SHAP_waterfall_hpvneg_correct.png",
    )

    print("Saving CSV outputs...")
    shap_summary = pd.DataFrame(
        {
            "feature": WINNER_FEATURES,
            "modality": [modality_of(f) for f in WINNER_FEATURES],
            "region": [region_of(f) for f in WINNER_FEATURES],
            "mean_abs_shap_dev": mean_abs_dev,
            "mean_abs_shap_chus": mean_abs_chus,
        }
    )
    shap_summary["rank_dev"] = shap_summary["mean_abs_shap_dev"].rank(method="min", ascending=False).astype(int)
    shap_summary["rank_chus"] = shap_summary["mean_abs_shap_chus"].rank(method="min", ascending=False).astype(int)
    shap_summary = shap_summary.sort_values(["rank_dev", "feature"]).reset_index(drop=True)
    shap_summary.to_csv(OUT_DIR / "T2_SHAP_shap_summary_table.csv", index=False)

    def pct(mask: np.ndarray, values: np.ndarray) -> float:
        total = float(values.sum())
        return 0.0 if total == 0 else float(values[mask].sum() / total * 100.0)

    rows = []
    for cohort, values in [("dev", np.abs(shap_dev).mean(axis=0)), ("chus", np.abs(shap_chus).mean(axis=0))]:
        feat_arr = np.array(WINNER_FEATURES)
        rows.append(
            {
                "cohort": cohort,
                "PT_pct": pct(np.isin(feat_arr, PT_WINNER), values),
                "CT_pct": pct(np.isin(feat_arr, CT_WINNER), values),
                "Clinical_pct": pct(np.isin(feat_arr, CLINICAL_FEATURES), values),
                "GTVp_pct": pct(np.array([region_of(f) == "GTVp" for f in WINNER_FEATURES]), values),
                "GTVn_pct_CT": pct(np.array([(f in CT_WINNER) and ("GTVn" in f) for f in WINNER_FEATURES]), values),
                "GTVn_pct_PT": pct(np.array([(f in PT_WINNER) and ("GTVn" in f) for f in WINNER_FEATURES]), values),
            }
        )
    pd.DataFrame(rows).to_csv(OUT_DIR / "T2_SHAP_modality_region_contribution.csv", index=False)

    coef = model.coef_[0]
    coef_df = pd.DataFrame(
        {
            "feature": WINNER_FEATURES,
            "modality": [modality_of(f) for f in WINNER_FEATURES],
            "region": [region_of(f) for f in WINNER_FEATURES],
            "coef_raw": coef,
            "coef_standardised": coef,
            "direction": ["+" if val >= 0 else "-" for val in coef],
        }
    )
    coef_df.to_csv(OUT_DIR / "T2_SHAP_coefficient_table.csv", index=False)

    task1_ref = load_task1_shap_reference()
    task2_ref = shap_summary[shap_summary["feature"].isin(PT_WINNER)][
        ["feature", "rank_dev", "mean_abs_shap_dev"]
    ].rename(
        columns={
            "rank_dev": "task2_rank_dev",
            "mean_abs_shap_dev": "task2_mean_abs_shap_dev",
        }
    )
    cross_task = task2_ref.merge(task1_ref, on="feature", how="left").sort_values("task2_rank_dev")
    cross_task.to_csv(OUT_DIR / "T2_SHAP_cross_task_shap_comparison.csv", index=False)

    print("\nCross-task SHAP comparison:")
    print(
        cross_task[
            [
                "feature",
                "task2_rank_dev",
                "task2_mean_abs_shap_dev",
                "task1_rank",
                "task1_mean_abs_shap_dev",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 72)
    print("TASK 2 BRANCH A COMPLETE")
    print("=" * 72)
    print(f"Train/Test/CHUS sizes: {len(train_df)} / {len(test_df)} / {len(chus_df)}")
    print(
        f"Reproduced CHUS: AUC={repro['auc_ext']:.6f}, BA={repro['ba']:.6f}, "
        f"Spe={repro['spe']:.6f}, Sen={repro['sen']:.6f}"
    )
    print(f"Outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
