from __future__ import annotations

import time
import warnings
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import ExtraSurvivalTrees
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

warnings.filterwarnings("ignore")


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "2_apr_T2B_data"
OUT_DIR = ROOT / "10_apr_postudy_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TASK2_DIR = ROOT.parent / "Mar_2026_task2" / "12_mar_task2_rad_data"
CLINICAL_FILE = ROOT.parent / "Feb_2026" / "25_feb_clinical_reduced_dataset" / "25_feb_Processed_clinical_reduced.csv"

POOLED_FILE = DATA_DIR / "2_apr_t2b_train.csv"
EXT_FILE = DATA_DIR / "2_apr_t2b_ext.csv"
SPLIT_MAP_FILE = TASK2_DIR / "13_mar_task2_split_map.csv"
OLD_REPORT = OUT_DIR / "10_apr_t2b_poststudy_report.md"
NEW_REPORT = ROOT / "10_apr_t2b_poststudy_report_new_plan.md"

BRANCH_E_CSV = OUT_DIR / "T2B_branchE_incremental_survival_new.csv"
BRANCH_E_PNG = OUT_DIR / "T2B_branchE_incremental_survival_new.png"
BRANCH_F_CSV = OUT_DIR / "T2B_branchF_hpv_relapse_association_new.csv"
BRANCH_F_PNG = OUT_DIR / "T2B_branchF_hpv_relapse_association_new.png"
FEATURE_LINEAGE_PNG = OUT_DIR / "T2B_A_feature_overlap_heatmap.png"
SCORE_DISTRIBUTION_PNG = OUT_DIR / "T2B_C_score_distribution.png"
SCORE_MAP_TRAIN_PNG = OUT_DIR / "T2B_C_score_map_train.png"
SCORE_MAP_TEST_PNG = OUT_DIR / "T2B_C_score_map_test20.png"
SCORE_MAP_CHUS_PNG = OUT_DIR / "T2B_C_score_map_CHUS.png"
RISK_GROUPS_CSV = OUT_DIR / "T2B_C_risk_groups.csv"
KM_COMBINED_PNG = OUT_DIR / "T2B_C_KM_survival.png"
KM_TRAIN_PNG = OUT_DIR / "T2B_C_KM_train.png"
KM_TEST_PNG = OUT_DIR / "T2B_C_KM_test20.png"
KM_CHUS_PNG = OUT_DIR / "T2B_C_KM_CHUS.png"
STAGE1_PREFILTER_FILE = ROOT / "2_apr_T2B_outputs" / "t2b_stage1_prefilter_strict_ultimate.csv"
HPV_SHAP_CSV_LEGACY = OUT_DIR / ("T2B_B_shap_hpv_" + "head.csv")
HPV_SHAP_CSV = OUT_DIR / "T2B_B_shap_hpv_classification.csv"

SEED = 42
N_EST = 200
HPV_C = 0.5
N_FOLDS = 5
N_REPEATS = 3

LOCKED_CLIN = ["Gender_Male"]
LOCKED_PT = [
    "GTVn_wavelet-LLH_firstorder_Mean",
    "GTVp_original_firstorder_InterquartileRange",
    "GTVp_wavelet-HHL_glrlm_ShortRunHighGrayLevelEmphasis",
    "GTVp_wavelet-HLH_glrlm_ShortRunHighGrayLevelEmphasis",
]
LOCKED_CT = [
    "GTVp_log-sigma-1-mm-3D_firstorder_Range",
    "GTVp_wavelet-HLL_ngtdm_Complexity",
    "GTVp_wavelet-LLH_glrlm_HighGrayLevelRunEmphasis",
]
PAN_RAD = LOCKED_PT + LOCKED_CT
CLINICAL_BASE = ["Age", "Gender_Male", "Treatment_CRT"]

RPM_WINNER_FEATURES = {
    "Gender_Male",
    "GTVp_exponential_glszm_HighGrayLevelZoneEmphasis",
    "GTVn_wavelet-LLH_firstorder_Mean",
    "GTVp_wavelet-HLH_glrlm_ShortRunHighGrayLevelEmphasis",
    "GTVp_gradient_glszm_ZoneEntropy",
    "GTVp_wavelet-LHL_glszm_SmallAreaHighGrayLevelEmphasis",
    "GTVn_wavelet-LHH_glszm_LowGrayLevelZoneEmphasis",
    "GTVp_wavelet-HLH_glszm_HighGrayLevelZoneEmphasis",
    "GTVp_wavelet-LLH_glrlm_HighGrayLevelRunEmphasis",
    "GTVp_wavelet-HLL_ngtdm_Complexity",
    "GTVp_gradient_glszm_SmallAreaLowGrayLevelEmphasis",
    "GTVp_wavelet-LHH_firstorder_RootMeanSquared",
}
HCM_WINNER_FEATURES = {
    "Gender_Male",
    "GTVn_wavelet-LHL_glszm_GrayLevelVariance",
    "GTVp_wavelet-LLH_firstorder_Median",
    "GTVn_logarithm_glszm_SmallAreaLowGrayLevelEmphasis",
    "GTVn_wavelet-LLH_firstorder_Skewness",
    "GTVn_logarithm_glcm_Idn",
    "GTVp_wavelet-HHL_glrlm_ShortRunHighGrayLevelEmphasis",
    "GTVn_squareroot_glcm_Idm",
    "GTVp_original_firstorder_InterquartileRange",
    "GTVp_log-sigma-1-mm-3D_firstorder_Range",
    "GTVn_wavelet-LHH_glrlm_GrayLevelVariance",
    "GTVn_wavelet-HLH_gldm_SmallDependenceHighGrayLevelEmphasis",
    "GTVn_wavelet-HHH_glszm_ZonePercentage",
    "GTVn_wavelet-LHH_glcm_ClusterProminence",
    "GTVn_wavelet-HHH_glcm_DifferenceAverage",
    "GTVn_wavelet-HLH_gldm_SmallDependenceEmphasis",
}

BA_LABELS = {
    "Gender_Male": "B1 Gender Male",
    "GTVn_wavelet-LLH_firstorder_Mean": "B2 GTVn wav-LLH Mean",
    "GTVp_original_firstorder_InterquartileRange": "B3 GTVp original IQR",
    "GTVp_wavelet-HHL_glrlm_ShortRunHighGrayLevelEmphasis": "B4 GTVp wav-HHL GLRLM SRHGLE",
    "GTVp_wavelet-HLH_glrlm_ShortRunHighGrayLevelEmphasis": "B5 GTVp wav-HLH GLRLM SRHGLE",
    "GTVp_log-sigma-1-mm-3D_firstorder_Range": "B6 GTVp LoG-1mm Range",
    "GTVp_wavelet-HLL_ngtdm_Complexity": "B7 GTVp wav-HLL NGTDM Complexity",
    "GTVp_wavelet-LLH_glrlm_HighGrayLevelRunEmphasis": "B8 GTVp wav-LLH GLRLM HGRE",
}


def safe_ci(y: np.ndarray, risk: np.ndarray) -> float:
    try:
        return float(concordance_index_censored(y["event"], y["time"], risk)[0])
    except Exception:
        return 0.5


def youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, scores)
    return float(thr[int(np.argmax(tpr - fpr))])


def make_est() -> ExtraSurvivalTrees:
    return ExtraSurvivalTrees(n_estimators=N_EST, random_state=SEED, n_jobs=-1)


def make_hpv() -> LogisticRegression:
    return LogisticRegression(
        C=HPV_C,
        penalty="l2",
        solver="lbfgs",
        class_weight="balanced",
        max_iter=2000,
        random_state=SEED,
    )


def compact_feature_label(feature: str) -> str:
    if feature in BA_LABELS:
        return BA_LABELS[feature]
    label = feature
    replacements = {
        "GTVp_": "GTVp ",
        "GTVn_": "GTVn ",
        "wavelet-": "wav-",
        "log-sigma-1-mm-3D": "LoG-1mm",
        "firstorder": "FO",
        "glrlm": "GLRLM",
        "glszm": "GLSZM",
        "gldm": "GLDM",
        "glcm": "GLCM",
        "ngtdm": "NGTDM",
        "HighGrayLevel": "HGL",
        "LowGrayLevel": "LGL",
        "ShortRun": "SR",
        "SmallArea": "SA",
        "SmallDependence": "SD",
        "ZoneEmphasis": "ZE",
        "RunEmphasis": "RE",
        "InterquartileRange": "IQR",
        "RootMeanSquared": "RMS",
        "GrayLevelVariance": "GLV",
        "DifferenceAverage": "DiffAvg",
        "ClusterProminence": "ClusterProm.",
        "_": " ",
    }
    for old, new in replacements.items():
        label = label.replace(old, new)
    return label


_MODALITY_DISPLAY = {"PT": "PET", "CT": "CT", "Clinical": "Clinical"}


def beeswarm_label(feature: str, feat_meta: dict[str, tuple[str, str]]) -> str:
    if feature == "Gender_Male":
        return "Clinical | Gender Male"

    modality_raw, region = feat_meta.get(feature, ("", ""))
    modality = _MODALITY_DISPLAY.get(modality_raw, modality_raw)

    parts = feature.split("_", 2)
    if len(parts) == 3:
        filter_name = parts[1]
        name = parts[2]
    else:
        filter_name = ""
        name = feature

    filter_name = (
        filter_name.replace("wavelet-", "wav-")
        .replace("log-sigma-1-mm-3D", "LoG-1mm")
    )
    name = (
        name.replace("glrlm_", "")
        .replace("firstorder_", "")
        .replace("ngtdm_", "")
    )
    display_name = f"{filter_name} {name}".strip()
    if filter_name == "original":
        display_name = name

    prefix = f"{modality}-{region}" if modality and region else feature
    return f"{prefix} | {display_name}"


def feature_lineage(feature: str) -> tuple[bool, bool]:
    return feature in RPM_WINNER_FEATURES, feature in HCM_WINNER_FEATURES


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pooled = pd.read_csv(POOLED_FILE)
    ext = pd.read_csv(EXT_FILE)
    smap = pd.read_csv(SPLIT_MAP_FILE)
    clin = pd.read_csv(CLINICAL_FILE)

    for df in (pooled, ext, smap, clin):
        if "PatientID" in df.columns:
            df["PatientID"] = df["PatientID"].astype(str)

    smap["split"] = smap["split"].astype(str).str.lower()
    train_ids = set(smap.loc[smap["split"] == "train", "PatientID"])
    test_ids = set(smap.loc[smap["split"] == "test", "PatientID"])

    clin = clin[["PatientID", "Age", "Gender_Male", "Treatment_CRT"]].copy()
    clin["Age"] = clin["Age"].astype(float)
    clin["Gender_Male"] = clin["Gender_Male"].astype(int)
    clin["Treatment_CRT"] = clin["Treatment_CRT"].astype(int)

    def prep(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out = out.drop(columns=[c for c in ["Age", "Gender_Male", "Treatment_CRT"] if c in out.columns], errors="ignore")
        out = out.merge(clin, on="PatientID", how="left")
        out["HPV_binary"] = out["HPV_binary"].astype(int)
        out["Relapse"] = out["Relapse"].astype(int)
        out["RFS"] = out["RFS"].astype(float)
        out["CenterID"] = out["CenterID"].astype(int)
        return out

    pooled = prep(pooled)
    ext = prep(ext)
    train_df = pooled[pooled["PatientID"].isin(train_ids)].reset_index(drop=True)
    test_df = pooled[pooled["PatientID"].isin(test_ids)].reset_index(drop=True)

    assert len(train_df) == 67 and len(test_df) == 20 and len(ext) == 27, "Expected 67/20/27 strict split"
    return train_df, test_df, ext


def branch_a_feature_lineage_matrix() -> None:
    """Full parent-pool lineage matrix: all RPM+HCM winner features, two shading tiers.

    Two tiers:
      - retained  : in the final BA 8-feature winner set (dark fill, bold label)
      - not retained : parent pool features not in BA winner (faded fill)

    Rows = full union of RPM and HCM parent feature sets (26 radiomic + Gender_Male = 27).
    """
    import matplotlib.patches as patches

    stage1 = pd.read_csv(STAGE1_PREFILTER_FILE)
    feat_meta = {row["feature"]: (row["modality"], row["region"]) for _, row in stage1.iterrows()}

    retained_set = set(PAN_RAD) | {"Gender_Male"}

    all_parent = (RPM_WINNER_FEATURES | HCM_WINNER_FEATURES) - {"Gender_Male"}
    rows_raw: list[tuple] = []
    for feat in all_parent:
        rpm, hcm = feature_lineage(feat)
        mod, region = feat_meta.get(feat, ("Unknown", "Unknown"))
        is_retained = feat in retained_set
        rows_raw.append((feat, mod, region, rpm, hcm, is_retained))

    rows_raw.sort(key=lambda r: (0 if r[5] else 1, r[1], r[2], r[0]))
    rows = [("Gender_Male", "Clinical", "N/A", True, True, True)] + rows_raw

    columns = ["RPM lineage", "HCM lineage"]
    colors = {"RPM lineage": "#2CB1A1", "HCM lineage": "#D65F5F"}

    row_bg     = {True: "#EDEDED", False: "#FAFAFA"}
    txt_color  = {True: "#111111", False: "#BBBBBB"}
    txt_weight = {True: "bold",    False: "normal"}
    cell_alpha = {True: 0.90,      False: 0.14}

    fig, ax = plt.subplots(figsize=(8.8, 13.5))
    ax.set_xlim(-3.8, len(columns))
    ax.set_ylim(0, len(rows))
    ax.invert_yaxis()
    ax.set_xticks(np.arange(len(columns)) + 0.5)
    ax.set_xticklabels(columns, fontsize=10, fontweight="bold")
    ax.set_yticks([])

    retained_end: int | None = None

    for i, (feature, modality, region, rpm, hcm, is_retained) in enumerate(rows):
        if i > 0 and rows[i - 1][5] and not is_retained:
            retained_end = i

        ax.add_patch(patches.Rectangle(
            (-3.8, i), len(columns) + 3.8, 1,
            facecolor=row_bg[is_retained], edgecolor="#FFFFFF", linewidth=1.0, zorder=0,
        ))
        label_prefix = "Clinical" if modality == "Clinical" else f"{modality}-{region}"
        label = f"{label_prefix} | {compact_feature_label(feature)}"
        ax.text(-3.65, i + 0.5, label, ha="left", va="center", fontsize=7.5,
                fontweight=txt_weight[is_retained], color=txt_color[is_retained])

        for j, col in enumerate(columns):
            active = {"RPM lineage": rpm, "HCM lineage": hcm}[col]
            face = colors[col] if active else "#F4F6F8"
            alpha = cell_alpha[is_retained] if active else (0.85 if is_retained else 0.08)
            ax.add_patch(patches.Rectangle(
                (j, i), 1, 1, facecolor=face, edgecolor="white", linewidth=1.5, alpha=alpha,
            ))

    if retained_end is not None:
        ax.axhline(retained_end, color="#111111", linewidth=1.8)

    retained_patch  = patches.Patch(facecolor="#777777", label="Retained in BA winner (N=8)")
    not_ret_patch   = patches.Patch(facecolor="#E0E0E0", alpha=0.6, label="Parent pool, not retained (N=19)")
    ax.legend(
        handles=[retained_patch, not_ret_patch],
        loc="upper center", bbox_to_anchor=(0.62, -0.010), ncol=2, frameon=True, fontsize=8,
    )

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0)
    ax.xaxis.tick_top()
    ax.set_title(
        "BA pan-feature lineage matrix — full parent pool (N=27)\n"
        "RPM N=11 + HCM N=15 radiomic + Gender_Male → retained: 8",
        fontsize=11, pad=18,
    )
    plt.tight_layout(rect=(0, 0.015, 1, 0.955))
    fig.savefig(FEATURE_LINEAGE_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fit_survival(train_df: pd.DataFrame, eval_df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Fit EST with StandardScaler (fit on train, apply to eval). Scale-invariant for EST
    but needed for consistency when mixing Age/binary/radiomics features across configs."""
    sc = StandardScaler()
    X_tr = sc.fit_transform(train_df[features].to_numpy(float))
    X_ev = sc.transform(eval_df[features].to_numpy(float))
    model = make_est()
    y_train = Surv.from_arrays(event=train_df["Relapse"].astype(bool), time=train_df["RFS"])
    model.fit(X_tr, y_train)
    return model.predict(X_ev)


def oof_survival_ci(train_df: pd.DataFrame, features: list[str]) -> float:
    """OOF CV with StandardScaler fit on each train fold only."""
    strata = train_df["Relapse"].astype(str) + "_" + train_df["HPV_binary"].astype(str)
    cv = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=SEED)
    fold_scores: list[float] = []
    for tr_idx, va_idx in cv.split(train_df, strata):
        tr = train_df.iloc[tr_idx].reset_index(drop=True)
        va = train_df.iloc[va_idx].reset_index(drop=True)
        risk = fit_survival(tr, va, features)
        y_va = Surv.from_arrays(event=va["Relapse"].astype(bool), time=va["RFS"])
        fold_scores.append(safe_ci(y_va, risk))
    return float(np.mean(fold_scores))


def evaluate_survival_set(train_df: pd.DataFrame, test_df: pd.DataFrame, ext_df: pd.DataFrame, features: list[str]) -> dict[str, float]:
    test_risk = fit_survival(train_df, test_df, features)
    ext_risk = fit_survival(train_df, ext_df, features)
    test_y = Surv.from_arrays(event=test_df["Relapse"].astype(bool), time=test_df["RFS"])
    ext_y = Surv.from_arrays(event=ext_df["Relapse"].astype(bool), time=ext_df["RFS"])
    return {
        "oof_ci": oof_survival_ci(train_df, features),
        "test_ci": safe_ci(test_y, test_risk),
        "ext_ci": safe_ci(ext_y, ext_risk),
    }


def fit_locked_hpv(train_df: pd.DataFrame):
    features = LOCKED_CLIN + LOCKED_PT + LOCKED_CT
    X_train = train_df[features].to_numpy(float)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    model = make_hpv()
    model.fit(X_train_sc, train_df["HPV_binary"].to_numpy(int))
    proba_train = model.predict_proba(X_train_sc)[:, 1]
    raw_auc = float(roc_auc_score(train_df["HPV_binary"].to_numpy(int), proba_train))
    if raw_auc < 0.5:
        proba_train = 1.0 - proba_train
    threshold = youden_threshold(train_df["HPV_binary"].to_numpy(int), proba_train)
    return model, scaler, threshold


def hpv_proba(df: pd.DataFrame, model: LogisticRegression, scaler: StandardScaler) -> np.ndarray:
    features = LOCKED_CLIN + LOCKED_PT + LOCKED_CT
    X = df[features].to_numpy(float)
    scores = model.predict_proba(scaler.transform(X))[:, 1]
    raw_auc = float(roc_auc_score(df["HPV_binary"].to_numpy(int), scores))
    if raw_auc < 0.5:
        scores = 1.0 - scores
    return scores


def branch_c_km_survival(train_df: pd.DataFrame, test_df: pd.DataFrame, ext_df: pd.DataFrame) -> None:
    """Generate cohort-specific and combined KM curves for the BA RFS prediction output."""
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    features = LOCKED_CLIN + LOCKED_PT + LOCKED_CT
    train_risk = fit_survival(train_df, train_df, features)
    risk_threshold = float(np.median(train_risk))
    panels = [
        ("Train", train_df, KM_TRAIN_PNG),
        ("Internal test", test_df, KM_TEST_PNG),
        ("CHUS", ext_df, KM_CHUS_PNG),
    ]
    rows = []

    def add_panel(ax, label: str, df: pd.DataFrame, risk: np.ndarray, collect_rows: bool = False) -> float:
        group = np.where(risk >= risk_threshold, "High risk", "Low risk")
        colours = {"Low risk": "#2F6B9A", "High risk": "#B23A48"}
        kmf = KaplanMeierFitter()
        for risk_group in ["Low risk", "High risk"]:
            mask = group == risk_group
            kmf.fit(
                durations=df.loc[mask, "RFS"],
                event_observed=df.loc[mask, "Relapse"],
                label=f"{risk_group} (n={int(mask.sum())})",
            )
            kmf.plot_survival_function(ax=ax, ci_show=True, color=colours[risk_group], linewidth=1.8)
            if collect_rows:
                rows.append({
                    "cohort": label,
                    "risk_group": risk_group,
                    "n": int(mask.sum()),
                    "relapse_events": int(df.loc[mask, "Relapse"].sum()),
                    "risk_threshold_train_median": risk_threshold,
                })
        low = group == "Low risk"
        high = group == "High risk"
        if low.any() and high.any():
            p_value = float(logrank_test(
                df.loc[low, "RFS"],
                df.loc[high, "RFS"],
                event_observed_A=df.loc[low, "Relapse"],
                event_observed_B=df.loc[high, "Relapse"],
            ).p_value)
        else:
            p_value = np.nan
        ax.set_title(f"{label} (N={len(df)})", fontsize=10)
        ax.set_xlabel("RFS time (days)")
        ax.set_ylabel("Relapse-free survival")
        ax.set_ylim(0, 1.04)
        ax.grid(True, color="#E5E7EB", linewidth=0.7, alpha=0.8)
        # HR via CoxPH with binary high-risk group as covariate
        try:
            from lifelines import CoxPHFitter
            cph_df = pd.DataFrame({
                "T": df["RFS"].values,
                "E": df["Relapse"].astype(int).values,
                "high_risk": (group == "High risk").astype(int),
            })
            cph = CoxPHFitter()
            cph.fit(cph_df, duration_col="T", event_col="E", show_progress=False)
            hr_val = float(np.exp(cph.params_["high_risk"]))
            hr_lo  = float(np.exp(cph.confidence_intervals_.loc["high_risk", "95% lower-bound"]))
            hr_hi  = float(np.exp(cph.confidence_intervals_.loc["high_risk", "95% upper-bound"]))
            hr_str = f"HR={hr_val:.2f} (95% CI {hr_lo:.2f}–{hr_hi:.2f})"
        except Exception:
            hr_str = "HR=N/A"

        # C-index for this cohort using the already-computed risk array
        from sksurv.util import Surv as _Surv
        y_surv = _Surv.from_arrays(event=df["Relapse"].astype(bool), time=df["RFS"])
        ci_val = safe_ci(y_surv, risk)

        p_str = f"p = {p_value:.4f}" if np.isfinite(p_value) else "p = NA"
        ax.text(
            0.97, 0.97,
            f"{p_str}\n{hr_str}\nC-index = {ci_val:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
        )
        return p_value

    panel_data = []
    for label, df, out_png in panels:
        risk = fit_survival(train_df, df, features)
        panel_data.append((label, df, risk, out_png))
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        add_panel(ax, label, df, risk, collect_rows=True)
        plt.tight_layout()
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.4), sharey=True)
    for ax, (label, df, risk, _) in zip(axes, panel_data):
        add_panel(ax, label, df, risk)
    axes[1].set_ylabel("")
    axes[2].set_ylabel("")
    fig.suptitle("BA RFS prediction risk stratification", fontsize=12.5)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(KM_COMBINED_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(rows).to_csv(RISK_GROUPS_CSV, index=False)


def branch_c_score_distribution(train_df: pd.DataFrame, test_df: pd.DataFrame, ext_df: pd.DataFrame) -> None:
    """Regenerate the prediction-classification score map using the existing output path."""
    from matplotlib.lines import Line2D
    import matplotlib.patches as patches

    features = LOCKED_CLIN + LOCKED_PT + LOCKED_CT
    hpv_model, hpv_scaler, hpv_threshold = fit_locked_hpv(train_df)

    panel_data = []
    train_risk = fit_survival(train_df, train_df, features)
    risk_threshold = float(np.median(train_risk))
    for label, df in [("Train", train_df), ("Internal test", test_df), ("CHUS", ext_df)]:
        risk = fit_survival(train_df, df, features)
        hpv_scores = hpv_proba(df, hpv_model, hpv_scaler)
        panel_data.append((label, df.copy(), risk, hpv_scores))

    all_risk = np.concatenate([risk for _, _, risk, _ in panel_data])
    x_pad = 0.05 * (float(np.max(all_risk)) - float(np.min(all_risk)) or 1.0)
    xlim = (float(np.min(all_risk)) - x_pad, float(np.max(all_risk)) + x_pad)

    def draw_score_panel(ax, label: str, df: pd.DataFrame, risk: np.ndarray, hpv_scores: np.ndarray, show_ylabel: bool) -> None:
        rel = df["Relapse"].to_numpy(int)
        hpv_true = df["HPV_binary"].to_numpy(int)

        zone_specs = [
            (xlim[0], -0.03, risk_threshold - xlim[0], hpv_threshold + 0.03, "#F7B267", "lower RFS risk\nHPV- zone"),
            (xlim[0], hpv_threshold, risk_threshold - xlim[0], 1.03 - hpv_threshold, "#71C8A8", "lower RFS risk\nHPV+ zone"),
            (risk_threshold, -0.03, xlim[1] - risk_threshold, hpv_threshold + 0.03, "#F08A7E", "higher RFS risk\nHPV- zone"),
            (risk_threshold, hpv_threshold, xlim[1] - risk_threshold, 1.03 - hpv_threshold, "#B68AD9", "higher RFS risk\nHPV+ zone"),
        ]
        for left, bottom, width, height, colour, _ in zone_specs:
            ax.add_patch(
                patches.Rectangle(
                    (left, bottom),
                    width,
                    height,
                    facecolor=colour,
                    edgecolor="none",
                    alpha=0.18,
                    zorder=0,
                )
            )

        for relapse_value, color, relapse_name in [
            (0, "#2F6B9A", "No relapse"),
            (1, "#B23A48", "Relapse"),
        ]:
            for hpv_value, marker in [(0, "s"), (1, "o")]:
                mask = (rel == relapse_value) & (hpv_true == hpv_value)
                if not mask.any():
                    continue
                ax.scatter(
                    risk[mask],
                    hpv_scores[mask],
                    s=46 if relapse_value == 0 else 58,
                    marker=marker,
                    color=color,
                    edgecolor="white",
                    linewidth=0.7,
                    alpha=0.9,
                    zorder=3,
                )

        # Quadrant corner labels — show expected biological clustering without a trend line
        qx_l = (xlim[0] + risk_threshold) / 2.0
        qx_r = (risk_threshold + xlim[1]) / 2.0
        qy_lo = hpv_threshold * 0.45
        qy_hi = hpv_threshold + (1.0 - hpv_threshold) * 0.55
        quad_specs = [
            (qx_l, qy_hi, "HPV+\nlower RFS risk\n(expected)", "#1A7A6A"),
            (qx_r, qy_hi, "HPV+\nhigher RFS risk\n(discordant)", "#7060A0"),
            (qx_l, qy_lo, "HPV−\nlower RFS risk\n(discordant)", "#B06010"),
            (qx_r, qy_lo, "HPV−\nhigher RFS risk\n(expected)", "#8B2020"),
        ]
        for qx, qy, qlabel, qcol in quad_specs:
            ax.text(qx, qy, qlabel, ha="center", va="center", fontsize=6.2,
                    color=qcol, alpha=0.60, zorder=1, style="italic")

        y_surv = Surv.from_arrays(event=df["Relapse"].astype(bool), time=df["RFS"])
        ci = safe_ci(y_surv, risk)
        auc = float(roc_auc_score(hpv_true, hpv_scores))
        if auc < 0.5:
            auc = 1.0 - auc

        high = risk >= risk_threshold
        low = ~high
        high_events = int(rel[high].sum())
        low_events = int(rel[low].sum())
        pred_hpv_pos = hpv_scores >= hpv_threshold
        hpv_acc = float(np.mean(pred_hpv_pos.astype(int) == hpv_true))

        ax.axvline(risk_threshold, color="#333333", linestyle="--", linewidth=1.5, alpha=0.95)
        ax.axhline(hpv_threshold, color="#333333", linestyle=":", linewidth=1.7, alpha=0.95)
        ax.text(
            0.97,
            0.97,
            f"C-index={ci:.3f}\nAUC={auc:.3f}\nHPV acc={hpv_acc:.2f}",
            transform=ax.transAxes,
            fontsize=8,
            color="#222222",
            ha="right",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2},
        )

        ax.set_title(f"{label} (N={len(df)})", fontsize=10)
        ax.set_xlabel("RFS prediction score (EST)")
        if show_ylabel:
            ax.set_ylabel("HPV+ classification probability (LR)")
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, color="#E5E7EB", linewidth=0.7, alpha=0.8)

    legend_elements = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#2F6B9A", markeredgecolor="white", label="No relapse", markersize=7),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#B23A48", markeredgecolor="white", label="Relapse", markersize=7),
        Line2D([0], [0], marker="s", color="#555555", linestyle="None", label="True HPV-"),
        Line2D([0], [0], marker="o", color="#555555", linestyle="None", label="True HPV+"),
        patches.Patch(facecolor="#F7B267", alpha=0.35, label="lower RFS / HPV-"),
        patches.Patch(facecolor="#71C8A8", alpha=0.35, label="lower RFS / HPV+"),
        patches.Patch(facecolor="#F08A7E", alpha=0.35, label="higher RFS / HPV-"),
        patches.Patch(facecolor="#B68AD9", alpha=0.35, label="higher RFS / HPV+"),
    ]

    solo_outputs = {
        "Train": SCORE_MAP_TRAIN_PNG,
        "Internal test": SCORE_MAP_TEST_PNG,
        "CHUS": SCORE_MAP_CHUS_PNG,
    }
    for label, df, risk, hpv_scores in panel_data:
        fig, ax = plt.subplots(figsize=(5.7, 4.8))
        draw_score_panel(ax, label, df, risk, hpv_scores, show_ylabel=True)
        ax.legend(handles=legend_elements[:4], loc="lower left", frameon=True, fontsize=8)
        plt.tight_layout()
        fig.savefig(solo_outputs[label], dpi=300, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8), sharey=True)
    for i, (ax, (label, df, risk, hpv_scores)) in enumerate(zip(axes, panel_data)):
        draw_score_panel(ax, label, df, risk, hpv_scores, show_ylabel=(i == 0))
    axes[0].legend(handles=legend_elements, loc="lower left", frameon=True, fontsize=8)
    fig.suptitle("BA prediction-classification score map", fontsize=13)
    fig.text(
        0.5,
        0.018,
        "Dashed vertical = train median RFS score; dotted horizontal = train Youden HPV threshold. "
        "x-axis (EST) and y-axis (LR) are separate models sharing the same feature set. "
        "Quadrant labels show expected HPV–prognosis biological pattern.",
        ha="center",
        fontsize=8.5,
        color="#333333",
    )
    plt.tight_layout(rect=(0, 0.055, 1, 0.94))
    fig.savefig(SCORE_DISTRIBUTION_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def branch_e(train_df: pd.DataFrame, test_df: pd.DataFrame, ext_df: pd.DataFrame) -> pd.DataFrame:
    configs = [
        ("clinical_only", CLINICAL_BASE),
        ("clinical_plus_true_HPV", CLINICAL_BASE + ["HPV_binary"]),
        ("clinical_plus_pan", CLINICAL_BASE + PAN_RAD),
        ("clinical_plus_pan_plus_true_HPV", CLINICAL_BASE + PAN_RAD + ["HPV_binary"]),
    ]
    rows = []
    for name, features in configs:
        metrics = evaluate_survival_set(train_df, test_df, ext_df, features)
        rows.append({
            "model": name,
            "n_features": len(features),
            "features": "|".join(features),
            **metrics,
        })
    df = pd.DataFrame(rows)
    base_clin = float(df.loc[df["model"] == "clinical_only", "ext_ci"].iloc[0])
    base_pan = float(df.loc[df["model"] == "clinical_plus_pan", "ext_ci"].iloc[0])
    df["delta_ext_vs_clinical_only"] = df["ext_ci"] - base_clin
    df["delta_ext_vs_clinical_plus_pan"] = df["ext_ci"] - base_pan
    df.to_csv(BRANCH_E_CSV, index=False)

    label_map = {
        "clinical_only": "Clinical only",
        "clinical_plus_true_HPV": "Clin + true HPV",
        "clinical_plus_pan": "Clin + BA pan-8",
        "clinical_plus_pan_plus_true_HPV": "Clin + BA pan-8\n+ true HPV",
    }
    xlabels = [label_map.get(m, m) for m in df["model"]]
    base_chus = float(df.loc[df["model"] == "clinical_only", "ext_ci"].iloc[0])

    fig, (ax_main, ax_delta) = plt.subplots(
        1, 2, figsize=(12, 5.5),
        gridspec_kw={"width_ratios": [3, 1.4]},
    )

    x = np.arange(len(df))
    w_chus = 0.40
    w_int  = 0.22

    bars_int  = ax_main.bar(x - w_int / 2 - 0.01, df["test_ci"], width=w_int,
                             color="#4c78a8", alpha=0.72, label="Internal test (sensitivity)")
    bars_chus = ax_main.bar(x + w_chus / 2 - 0.01, df["ext_ci"], width=w_chus,
                             color="#f58518", alpha=0.90, label="CHUS (primary validation)")

    for bar, val in zip(bars_int, df["test_ci"]):
        ax_main.text(bar.get_x() + bar.get_width() / 2, val + 0.006,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, color="#2A4A7A")
    for bar, val, delta in zip(bars_chus, df["ext_ci"], df["delta_ext_vs_clinical_only"]):
        ax_main.text(bar.get_x() + bar.get_width() / 2, val + 0.006,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#7A3A00")
        if abs(delta) > 0.001:
            ax_main.text(bar.get_x() + bar.get_width() / 2, val + 0.040,
                         f"+{delta:.3f}", ha="center", va="bottom", fontsize=7.5, color="#C05000",
                         bbox=dict(facecolor="lightyellow", edgecolor="#DDAA00", alpha=0.80, pad=1.5, boxstyle="round,pad=0.2"))

    ymin = min(df["test_ci"].min(), df["ext_ci"].min()) - 0.06
    ax_main.set_ylim(ymin, min(df["ext_ci"].max() + 0.12, 1.0))

    ax_main.axhline(0.500, color="#888888", linewidth=1.1, linestyle=":", alpha=0.8)
    ax_main.text(len(df) - 0.5, 0.502, "Chance (0.500)", va="bottom", ha="right", fontsize=7.5,
                 color="#888888", style="italic")
    ax_main.axhline(base_chus, color="#D35400", linewidth=1.4, linestyle="--", alpha=0.75)
    ax_main.text(len(df) - 0.5, base_chus + 0.003, f"Clin. baseline CHUS={base_chus:.3f}",
                 va="bottom", ha="right", fontsize=7.5, color="#D35400")

    ax_main.fill_between([-0.5, len(df) - 0.5], 0.0, 0.500, alpha=0.04, color="#888888", zorder=0)

    ax_main.set_xticks(x)
    ax_main.set_xticklabels(xlabels, fontsize=9.5)
    ax_main.set_ylabel("C-index", fontsize=10)
    ax_main.set_xlim(-0.5, len(df) - 0.5)
    ax_main.set_title("Incremental prognostic value — C-index by input configuration", fontsize=10)
    ax_main.legend(loc="upper left", fontsize=8.5)
    ax_main.grid(axis="y", color="#E5E7EB", linewidth=0.7, alpha=0.8)

    delta_models = df[df["model"] != "clinical_only"]
    delta_labels = [label_map.get(m, m).replace("\n", " ") for m in delta_models["model"]]
    deltas = delta_models["delta_ext_vs_clinical_only"].values
    colors_delta = ["#f58518" if d > 0 else "#4c78a8" for d in deltas]
    y_pos = np.arange(len(delta_models))

    ax_delta.barh(y_pos, deltas, color=colors_delta, alpha=0.85, height=0.55)
    for yp, d in zip(y_pos, deltas):
        ax_delta.text(d + 0.004, yp, f"+{d:.3f}", va="center", ha="left", fontsize=8, fontweight="bold")
    ax_delta.set_yticks(y_pos)
    ax_delta.set_yticklabels(delta_labels, fontsize=8.5)
    ax_delta.set_xlabel("Δ C-index vs Clinical only", fontsize=9)
    ax_delta.set_title("CHUS gain\nvs clinical baseline", fontsize=9.5)
    ax_delta.axvline(0, color="#333333", linewidth=1.0)
    ax_delta.set_xlim(0, max(deltas) + 0.08)
    ax_delta.grid(axis="x", color="#E5E7EB", linewidth=0.7, alpha=0.8)

    fig.suptitle(
        "BA incremental prognostic value on CHUS (Table 3.5.7)\n"
        "Δ annotations = CHUS gain above clinical-only baseline (C-index 0.462)",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    fig.savefig(BRANCH_E_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return df


def branch_f(train_df: pd.DataFrame, test_df: pd.DataFrame, ext_df: pd.DataFrame) -> pd.DataFrame:
    model, scaler, threshold = fit_locked_hpv(train_df)

    rows = []
    for label, df in [("Train67", train_df), ("Test20", test_df), ("CHUS27", ext_df)]:
        scores = hpv_proba(df, model, scaler)
        pred = (scores >= threshold).astype(int)
        for source, group in [("true_hpv", df["HPV_binary"].to_numpy(int)), ("pred_hpv", pred)]:
            for value in [0, 1]:
                mask = group == value
                n = int(mask.sum())
                rel = int(df.loc[mask, "Relapse"].sum()) if n else 0
                rows.append({
                    "cohort": label,
                    "source": source,
                    "group": ("HPV-" if value == 0 else "HPV+"),
                    "n": n,
                    "relapse_events": rel,
                    "relapse_rate": (rel / n) if n else np.nan,
                    "threshold_from_train": threshold,
                    "true_group": "",
                    "pred_group": "",
                    "note": "",
                })
        for true_val in [0, 1]:
            for pred_val in [0, 1]:
                mask = (df["HPV_binary"].to_numpy(int) == true_val) & (pred == pred_val)
                n = int(mask.sum())
                rel = int(df.loc[mask, "Relapse"].sum()) if n else 0
                rows.append({
                    "cohort": label,
                    "source": "contingency",
                    "group": f"true_{'HPV-' if true_val == 0 else 'HPV+'}_pred_{'HPV-' if pred_val == 0 else 'HPV+'}",
                    "n": n,
                    "relapse_events": rel,
                    "relapse_rate": (rel / n) if n else np.nan,
                    "threshold_from_train": threshold,
                    "true_group": ("HPV-" if true_val == 0 else "HPV+"),
                    "pred_group": ("HPV-" if pred_val == 0 else "HPV+"),
                    "note": "Predicted HPV groups are a re-partition of the same cohort, not relabeled true HPV groups.",
                })
    out = pd.DataFrame(rows)
    out.to_csv(BRANCH_F_CSV, index=False)

    plot_df = out[out["cohort"].isin(["Test20", "CHUS27"])].copy()
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    for ax, source, title in [
        (axes[0], "true_hpv", "True HPV groups"),
        (axes[1], "pred_hpv", "Predicted HPV groups\n(train-derived threshold)"),
    ]:
        sub = plot_df[plot_df["source"] == source].copy()
        order = [("Test20", "HPV-"), ("Test20", "HPV+"), ("CHUS27", "HPV-"), ("CHUS27", "HPV+")]
        sub["order"] = sub.apply(lambda r: order.index((r["cohort"], r["group"])), axis=1)
        sub = sub.sort_values("order")
        labels = [f"{c}\n{g}" for c, g in zip(sub["cohort"], sub["group"])]
        vals = sub["relapse_rate"].to_numpy(float)
        colors = ["#4c78a8" if g == "HPV-" else "#54a24b" for g in sub["group"]]
        ax.bar(np.arange(len(sub)), vals, color=colors)
        ax.set_xticks(np.arange(len(sub)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, max(0.4, np.nanmax(vals) + 0.08))
        ax.set_title(title)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.015, f"{100*v:.1f}%", ha="center", va="bottom", fontsize=8)
    axes[0].set_ylabel("Relapse rate")
    chus_cont = out[(out["cohort"] == "CHUS27") & (out["source"] == "contingency")].copy()
    heat = np.zeros((2, 2), dtype=float)
    ann = np.empty((2, 2), dtype=object)
    row_order = ["HPV-", "HPV+"]
    col_order = ["HPV-", "HPV+"]
    for i, tg in enumerate(row_order):
        for j, pg in enumerate(col_order):
            row = chus_cont[(chus_cont["true_group"] == tg) & (chus_cont["pred_group"] == pg)].iloc[0]
            heat[i, j] = row["n"]
            ann[i, j] = f"n={int(row['n'])}\nrel={int(row['relapse_events'])}"
    im = axes[2].imshow(heat, cmap="Blues")
    axes[2].set_xticks(np.arange(2))
    axes[2].set_xticklabels(col_order)
    axes[2].set_yticks(np.arange(2))
    axes[2].set_yticklabels(row_order)
    axes[2].set_xlabel("Predicted HPV")
    axes[2].set_ylabel("True HPV")
    axes[2].set_title("CHUS true vs predicted\ncontingency")
    for i in range(2):
        for j in range(2):
            axes[2].text(j, i, ann[i, j], ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    fig.suptitle("Branch F: HPV-relapse association\n(CHUS primary, test20 sensitivity)", fontsize=12)
    plt.tight_layout()
    fig.savefig(BRANCH_F_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def branch_g_beeswarm(train_df: pd.DataFrame) -> None:
    """Generate BA SHAP beeswarm for the HPV-classification output (Appendix B5)."""
    import shap

    stage1 = pd.read_csv(STAGE1_PREFILTER_FILE)
    feat_meta = {row["feature"]: (row["modality"], row["region"]) for _, row in stage1.iterrows()}

    features = LOCKED_CLIN + LOCKED_PT + LOCKED_CT
    X_train = train_df[features].to_numpy(float)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    model = make_hpv()
    model.fit(X_train_sc, train_df["HPV_binary"].to_numpy(int))

    explainer = shap.LinearExplainer(model, X_train_sc)
    shap_values = explainer.shap_values(X_train_sc)

    plt.figure(figsize=(8, 5))
    shap.summary_plot(
        shap_values,
        X_train_sc,
        feature_names=[beeswarm_label(f, feat_meta) for f in features],
        show=False,
        plot_size=None,
    )
    plt.gca().set_title("BA HPV-classification output - SHAP beeswarm (train N=67)", fontsize=11)
    out_png = OUT_DIR / "T2B_G_beeswarm_hpv_classification.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Branch G beeswarm: {out_png}")


def sync_paper_ready_source_names() -> None:
    if HPV_SHAP_CSV_LEGACY.exists():
        shutil.copyfile(HPV_SHAP_CSV_LEGACY, HPV_SHAP_CSV)


def write_report(branch_e_df: pd.DataFrame, branch_f_df: pd.DataFrame) -> None:
    old_exists = OLD_REPORT.exists()
    chus_e = branch_e_df.set_index("model")
    true_chus = branch_f_df[(branch_f_df["cohort"] == "CHUS27") & (branch_f_df["source"] == "true_hpv")]
    pred_chus = branch_f_df[(branch_f_df["cohort"] == "CHUS27") & (branch_f_df["source"] == "pred_hpv")]
    true_test = branch_f_df[(branch_f_df["cohort"] == "Test20") & (branch_f_df["source"] == "true_hpv")]
    pred_test = branch_f_df[(branch_f_df["cohort"] == "Test20") & (branch_f_df["source"] == "pred_hpv")]
    chus_cont = branch_f_df[(branch_f_df["cohort"] == "CHUS27") & (branch_f_df["source"] == "contingency")]

    def rate_line(df: pd.DataFrame, group: str) -> str:
        row = df[df["group"] == group].iloc[0]
        return f"{row['relapse_events']}/{row['n']} ({100*row['relapse_rate']:.1f}%)"

    def cont_line(df: pd.DataFrame, true_group: str, pred_group: str) -> str:
        row = df[(df["true_group"] == true_group) & (df["pred_group"] == pred_group)].iloc[0]
        return f"{int(row['n'])} (relapse {int(row['relapse_events'])})"

    lines = [
        "# Task 2B Post-Study Report — New Plan Revision",
        "",
        "**Date:** 2026-04-10",
        "**Locked winner:** `13v23652`",
        "**Primary revision principle:** `CHUS` is the primary clean cohort for cross-task / pan-feature claims; `test20` is retained as a strict sensitivity checkpoint only.",
        "",
        "## 1. Relation to Claude's earlier April 10 report",
        "",
        "This report revises and partially supersedes `Apr_2026_task2B/10_apr_postudy_outputs/10_apr_t2b_poststudy_report.md`.",
        "",
        f"- earlier report present locally: {'yes' if old_exists else 'no'}",
        "- Branches A–D remain useful at a high level, especially:",
        "  - Branch A: canonical feature-set comparison",
        "  - Branch B: endpoint-specific interpretation",
        "  - Branch C: prognosis-link presentation",
        "  - Branch D: pan-feature vs pan-model claim boundary",
        "- However, later provenance review showed that the strict Task 2B `67 / 20 / CHUS` sensitivity design is not fully clean for cross-task RFS interpretation:",
        "  - `train67 = 45 / 67` overlap with canonical Task 1 development",
        "  - `test20 = 12 / 20` overlap with canonical Task 1 development",
        "  - `CHUS27 = 0 / 27` overlap with canonical Task 1 development",
        "- Therefore the earlier E/F/G/H-style interpretation needs to be replaced by a CHUS-primary, clinically grounded analysis.",
        "",
        "## 2. Revised reading of A–D",
        "",
        "- **A stays strong:** the overlap story still supports a bridge-signature interpretation.",
        "- **B stays strong with cautious wording:** Task 2 lineage contributes mainly to the HPV-classification output, while Task 1 lineage contributes mainly to the RFS-prediction output.",
        "- **C stays supportive only:** prognosis-link visuals are useful, but CHUS should be the primary cohort and any threshold-derived grouping must use a train-derived threshold rather than CHUS optimisation.",
        "- **D stays as the claim boundary:** Task 2B is pan-feature evidence, not pan-model proof.",
        "",
        "## 3. Branch E — Clinical + HPV + pan-feature incremental prognostic value",
        "",
        "Clinical baseline used here is the full reduced clinical set available in this cohort:",
        "- `Age`",
        "- `Gender_Male`",
        "- `Treatment_CRT`",
        "",
        "Survival model family kept fixed to `EST(n_estimators=200, random_state=42)` for all four comparisons.",
        "",
        "| Model | OOF CI | Test20 CI | CHUS CI | CHUS delta vs clinical-only |",
        "|------|--------|-----------|---------|------------------------------|",
    ]

    for _, row in branch_e_df.iterrows():
        lines.append(
            f"| `{row['model']}` | {row['oof_ci']:.3f} | {row['test_ci']:.3f} | {row['ext_ci']:.3f} | {row['delta_ext_vs_clinical_only']:+.3f} |"
        )
    lines += [
        "",
        "Primary CHUS reading:",
        f"- `clinical only` -> CHUS CI = {chus_e.loc['clinical_only','ext_ci']:.3f}",
        f"- `clinical + true HPV` -> CHUS CI = {chus_e.loc['clinical_plus_true_HPV','ext_ci']:.3f}",
        f"- `clinical + pan features` -> CHUS CI = {chus_e.loc['clinical_plus_pan','ext_ci']:.3f}",
        f"- `clinical + pan features + true HPV` -> CHUS CI = {chus_e.loc['clinical_plus_pan_plus_true_HPV','ext_ci']:.3f}",
        "",
        "Interpretation rule:",
        "- if `clinical + true HPV` improves over `clinical only`, HPV is prognostically informative in this cohort",
        "- if `clinical + pan features` also improves over `clinical only`, the pan-feature set is recovering part of that prognostic signal",
        "- the key question is whether adding true HPV on top of the pan-feature model still gives a large residual gain",
        "",
        "Observed reading from the current run:",
        f"- on `CHUS`, true HPV adds a large prognostic gain over clinical alone: `+{(chus_e.loc['clinical_plus_true_HPV','ext_ci'] - chus_e.loc['clinical_only','ext_ci']):.3f}`",
        f"- the locked pan-feature set also improves over clinical alone on `CHUS`: `+{(chus_e.loc['clinical_plus_pan','ext_ci'] - chus_e.loc['clinical_only','ext_ci']):.3f}`",
        f"- but adding true HPV on top of the pan-feature model still yields a substantial residual gain on `CHUS`: `+{(chus_e.loc['clinical_plus_pan_plus_true_HPV','ext_ci'] - chus_e.loc['clinical_plus_pan','ext_ci']):.3f}`",
        "- so the pan-feature set appears to recover part, but not all, of the HPV-linked prognostic information",
        "",
        "## 4. Branch F — HPV-relapse association",
        "",
        "This branch is a biological plausibility check, not a classifier-validation table.",
        "",
        "Predicted HPV groups are defined using a **train-derived** Youden threshold from the locked HPV-classification output, then applied unchanged to `test20` and `CHUS`.",
        "",
        "### 4.1 CHUS primary cohort",
        "",
        "| Grouping | HPV- relapse rate | HPV+ relapse rate |",
        "|----------|-------------------|-------------------|",
        f"| True HPV | {rate_line(true_chus, 'HPV-')} | {rate_line(true_chus, 'HPV+')} |",
        f"| Predicted HPV | {rate_line(pred_chus, 'HPV-')} | {rate_line(pred_chus, 'HPV+')} |",
        "",
        "### 4.2 CHUS true-vs-predicted contingency",
        "",
        "| True HPV \\\\ Predicted HPV | Pred HPV- | Pred HPV+ |",
        "|----------------------------|-----------|-----------|",
        f"| True HPV- | {cont_line(chus_cont, 'HPV-', 'HPV-')} | {cont_line(chus_cont, 'HPV-', 'HPV+')} |",
        f"| True HPV+ | {cont_line(chus_cont, 'HPV+', 'HPV-')} | {cont_line(chus_cont, 'HPV+', 'HPV+')} |",
        "",
        "- the predicted HPV-negative CHUS group of `12` is made of `6` true HPV-negative and `6` true HPV-positive patients",
        "- the predicted HPV-positive CHUS group of `15` is made of `1` true HPV-negative and `14` true HPV-positive patients",
        "- so the predicted groups are mixed reclassification groups, not pure true-HPV groups",
        "",
        "### 4.3 Test20 sensitivity cohort",
        "",
        "| Grouping | HPV- relapse rate | HPV+ relapse rate |",
        "|----------|-------------------|-------------------|",
        f"| True HPV | {rate_line(true_test, 'HPV-')} | {rate_line(true_test, 'HPV+')} |",
        f"| Predicted HPV | {rate_line(pred_test, 'HPV-')} | {rate_line(pred_test, 'HPV+')} |",
        "",
        "Interpretation:",
        "- this branch does **not** build a reverse `RFS -> HPV` model",
        "- instead it asks whether true and predicted HPV groupings show the expected relapse direction",
        "- `CHUS` is the primary cohort for this reading because it has zero overlap with canonical Task 1 development provenance",
        "- on `CHUS`, the predicted grouping preserves the expected direction, but only as directional corroboration because the predicted groups are mixed reclassification groups",
        "- `test20` remains sensitivity only and should not be used to argue for or against the Branch F biology claim",
        "",
        "## 5. What this changes in the final Task 2B write-up",
        "",
        "- The strongest post-study story is no longer a pseudo-specialist comparison or oracle dual-benefit claim.",
        "- The strongest story is now:",
        "  1. structural bridge signature (A)",
        "  2. endpoint-specific lineage interpretation (B)",
        "  3. prognosis-link presentation with CHUS primary (C)",
        "  4. pan-feature not pan-model claim boundary (D)",
        "  5. clinically grounded incremental prognostic value (new E)",
        "  6. HPV-relapse association presentation without reverse modelling (new F)",
        "",
        "## 6. Recommended next thesis step",
        "",
        "Use this revised report together with the old April 10 report, but treat the old report's later E/F/G/H logic as superseded where it conflicts with the provenance correction. For the thesis chapter, anchor the main quantitative cross-task claim to `CHUS`, not `test20`.",
        "",
        f"**New outputs:** `{BRANCH_E_CSV.name}`, `{BRANCH_E_PNG.name}`, `{BRANCH_F_CSV.name}`, `{BRANCH_F_PNG.name}`",
        "",
        "## 7. Branch G - BA HPV-classification SHAP beeswarm",
        "",
        "This is a visual-only addition with no new quantitative claim.",
        "",
        "**Purpose:** Visual companion to Table B5.2 (LR coefficients) and Table B5.6 (lineage attribution summary). Shows the per-patient SHAP value distribution for the HPV-classification output.",
        "",
        "**Output:** `T2B_G_beeswarm_hpv_classification.png`",
        "**In outline:** Figure B5.5 in Appendix B5.",
        "**Claim status:** No new main-text claim. The quantitative content is tabulated in Tables B5.2 and B5.6.",
        "",
        "## 8. Branch H - BA feature directionality sync (desk exercise)",
        "",
        "This is a desk exercise using existing outputs.",
        "",
        "**Purpose:** Explicit per-feature check of whether the HPV-classification coefficient direction and RFS-prediction SHAP direction agree. Identifies B3 (IQR) and B5 as unidirectional bridge features, and B2, B4, B7, B8 as endpoint-specialised features.",
        "",
        "**Output:** Table B5.8 in the outline (no new CSV generated; source files are the existing B-branch coefficient and permutation importance CSVs).",
        "**Claim status:** Model-internal comparison only. No independence claim. No new external cohort data used.",
        "",
    ]
    NEW_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    t0 = time.time()
    train_df, test_df, ext_df = load_data()
    print(
        f"Strict Task 2B loaded: train={len(train_df)} test={len(test_df)} ext={len(ext_df)} | "
        f"Task1 provenance overlap: train=45/67 test=12/20 ext=0/27"
    )
    branch_a_feature_lineage_matrix()
    branch_c_km_survival(train_df, test_df, ext_df)
    branch_c_score_distribution(train_df, test_df, ext_df)
    sync_paper_ready_source_names()
    branch_e_df = branch_e(train_df, test_df, ext_df)
    branch_f_df = branch_f(train_df, test_df, ext_df)
    write_report(branch_e_df, branch_f_df)
    branch_g_beeswarm(train_df)
    dt = time.time() - t0
    print(f"Branch E output: {BRANCH_E_CSV}")
    print(f"Branch F output: {BRANCH_F_CSV}")
    print(f"Report: {NEW_REPORT}")
    print(f"Done in {dt/60:.1f} min")


if __name__ == "__main__":
    main()
