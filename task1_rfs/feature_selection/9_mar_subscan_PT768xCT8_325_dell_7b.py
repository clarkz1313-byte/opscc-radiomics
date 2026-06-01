"""
9_mar_subscan_PT768xCT8_325_dell_7b.py

dell_7b: deterministic single-swap subscan around pc_3 No.16147 anchor
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "9_mar_subscan_outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

PT_DEV_FILE = ROOT / "27_feb_PT_development.csv"
CT_DEV_FILE = ROOT / "27_feb_CT_development.csv"
PT_EXT_FILE = ROOT / "27_feb_PT_external.csv"
CT_EXT_FILE = ROOT / "27_feb_CT_external.csv"

PT_FEATURES_FILE = ROOT / "2_mar_finalist_outputs" / "PT_inter1_768_features.csv"
CT_FEATURES_FILE = ROOT / "2_mar_finalist_outputs" / "CT_inter8_325_features.csv"
PC3_RESULTS_FILE = ROOT / "7_mar_PT768_results_compilation" / "8_mar_SC3_rad_PT768xCT8_325_LOCODEV_pc_3_all_results.csv"

CLINICAL_FILE = (
    ROOT.parent / "Feb_2026" / "25_feb_clinical_reduced_dataset" / "25_feb_Processed_clinical_reduced.csv"
)

ANCHOR_NO = 16147
ANCHOR_CT = "GTVp_wavelet-LLH_glrlm_HighGrayLevelRunEmphasis"
ANCHOR_CLIN = "Gender_Male"
ALPHA = 0.1
CHUP_TARGET = 0.689655
TOP_CT_K = 15


def make_surv(event: pd.Series, time: pd.Series):
    return Surv.from_arrays(event=event.astype(bool).values, time=time.astype(float).values)


def safe_ci(y, risk: np.ndarray) -> float:
    try:
        return float(concordance_index_censored(y["event"], y["time"], risk)[0])
    except Exception:
        return float("nan")


def loco_evt_rank_univar(X: np.ndarray, y, centers: np.ndarray, feat_names: List[str], min_events: int = 10) -> List[str]:
    uniq = np.unique(centers)
    scores = []
    for j in range(X.shape[1]):
        fold_vals = []
        for c in uniq:
            mask = centers == c
            e_c = int(y["event"][mask].sum())
            if mask.sum() < 5 or e_c < min_events:
                continue
            ci = safe_ci(y[mask], X[mask, j])
            if np.isnan(ci):
                ci = 0.5
            fold_vals.append(max(ci, 1 - ci))
        scores.append(np.mean(fold_vals) if fold_vals else 0.5)
    order = np.argsort(np.array(scores))[::-1]
    return [feat_names[i] for i in order]


def parse_feature_list(raw: str) -> List[str]:
    if pd.isna(raw):
        return []
    return [x.strip() for x in str(raw).split(";") if x.strip()]


def canonical_key(pt_feats: List[str], ct_feat: str) -> Tuple[str, ...]:
    return tuple(sorted(pt_feats) + [ct_feat])


def eval_combo(
    dev_df: pd.DataFrame,
    chus_df: pd.DataFrame,
    chup_df: pd.DataFrame,
    clin_feat: str,
    pt_feats: List[str],
    ct_feat: str,
    combo_name: str,
) -> Dict[str, object]:
    cols = [clin_feat] + pt_feats + [ct_feat]

    X_tr = dev_df[cols].values.astype(float)
    y_tr = make_surv(dev_df["Relapse"], dev_df["RFS"])
    X_chus = chus_df[cols].values.astype(float)
    y_chus = make_surv(chus_df["Relapse"], chus_df["RFS"])
    X_chup = chup_df[cols].values.astype(float)
    y_chup = make_surv(chup_df["Relapse"], chup_df["RFS"])

    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_chus_sc = sc.transform(X_chus)
    X_chup_sc = sc.transform(X_chup)

    model = CoxPHSurvivalAnalysis(alpha=ALPHA)
    model.fit(X_tr_sc, y_tr)

    pred_chus = model.predict(X_chus_sc)
    pred_chup = model.predict(X_chup_sc)

    ci_chus = safe_ci(y_chus, pred_chus)
    ci_chup = safe_ci(y_chup, pred_chup)

    return {
        "combo": combo_name,
        "N": len(cols),
        "n_clin": 1,
        "n_pt": len(pt_feats),
        "n_ct": 1,
        "clin_feature": clin_feat,
        "pt_features": "; ".join(pt_feats),
        "ct_feature": ct_feat,
        "ci_chus": ci_chus,
        "ci_chup": ci_chup,
        "alpha": ALPHA,
    }


def main() -> None:
    print("=" * 70)
    print("9_mar_subscan_PT768xCT8_325_dell_7b")
    print("Deterministic subscan around pc_3 No.16147 anchor (CoxPH alpha=0.1)")
    print("=" * 70)

    clinical = pd.read_csv(CLINICAL_FILE)
    clinical = clinical.dropna(subset=["Relapse", "RFS"])

    pt_feats = pd.read_csv(PT_FEATURES_FILE)["Feature"].tolist()
    ct_raw = pd.read_csv(CT_FEATURES_FILE)["Feature"].tolist()
    ct_feats = [f for f in ct_raw if f not in set(pt_feats)]

    pt_dev = pd.read_csv(PT_DEV_FILE)
    ct_dev = pd.read_csv(CT_DEV_FILE)
    pt_ext = pd.read_csv(PT_EXT_FILE)
    ct_ext = pd.read_csv(CT_EXT_FILE)

    rad_dev = pt_dev[["PatientID"] + pt_feats].merge(ct_dev[["PatientID"] + ct_feats], on="PatientID", how="inner")
    rad_ext = pt_ext[["PatientID"] + pt_feats].merge(ct_ext[["PatientID"] + ct_feats], on="PatientID", how="inner")

    clin_dev = clinical[clinical["Cohort"] == "Dev"][["PatientID", "CenterID", "Relapse", "RFS", ANCHOR_CLIN]].copy()
    dev_df = clin_dev.merge(rad_dev, on="PatientID", how="inner")

    chus_clin = clinical[clinical["CenterID"] == 3][["PatientID", "Relapse", "RFS", ANCHOR_CLIN]].copy()
    chup_clin = clinical[clinical["CenterID"] == 2][["PatientID", "Relapse", "RFS", ANCHOR_CLIN]].copy()

    chus_df = chus_clin.merge(rad_ext[rad_ext["PatientID"].str.startswith("CHUS")], on="PatientID", how="inner")
    chup_df = chup_clin.merge(rad_ext[rad_ext["PatientID"].str.startswith("CHUP")], on="PatientID", how="inner")

    y_dev = make_surv(dev_df["Relapse"], dev_df["RFS"])
    dev_centers = dev_df["CenterID"].values

    X_pt = dev_df[pt_feats].values.astype(float)
    X_ct = dev_df[ct_feats].values.astype(float)
    pt_rank = loco_evt_rank_univar(X_pt, y_dev, dev_centers, pt_feats, min_events=10)
    ct_rank = loco_evt_rank_univar(X_ct, y_dev, dev_centers, ct_feats, min_events=10)

    pc3 = pd.read_csv(PC3_RESULTS_FILE)
    row = pc3[pc3["No"] == ANCHOR_NO]
    if row.empty:
        raise RuntimeError(f"Anchor row No={ANCHOR_NO} not found in {PC3_RESULTS_FILE}")
    row0 = row.iloc[0]

    pt_col = "pt_features" if "pt_features" in row0.index else "pt_names"
    anchor_pt = parse_feature_list(row0[pt_col])
    if len(anchor_pt) != 8:
        raise RuntimeError(f"Expected 8 PT anchor features from row {ANCHOR_NO}, got {len(anchor_pt)}")

    print(f"Dev/CHUS/CHUP: {len(dev_df)}/{len(chus_df)}/{len(chup_df)}")
    print(f"Anchor PT count: {len(anchor_pt)} | Anchor CT: {ANCHOR_CT}")

    seen = set()
    rows: List[Dict[str, object]] = []

    def add_eval(pt_list: List[str], ct_feature: str, tag: str):
        key = canonical_key(pt_list, ct_feature)
        if key in seen:
            return
        seen.add(key)
        rows.append(eval_combo(dev_df, chus_df, chup_df, ANCHOR_CLIN, pt_list, ct_feature, tag))

    add_eval(anchor_pt, ANCHOR_CT, "baseline")

    # Scan 1: PT single swaps by position
    pt_candidates = [f for f in pt_rank if f not in set(anchor_pt)]
    for pos in range(len(anchor_pt)):
        for cand in pt_candidates:
            new_pt = list(anchor_pt)
            new_pt[pos] = cand
            add_eval(new_pt, ANCHOR_CT, f"scan1_ptswap_pos{pos+1}")

    # Scan 2: CT swaps top-15
    for ct_cand in ct_rank[:TOP_CT_K]:
        add_eval(list(anchor_pt), ct_cand, "scan2_ctswap")

    # Scan 3: N=9 drop one PT (lowest-ranked first)
    pt_rank_pos = {f: i for i, f in enumerate(pt_rank)}
    sorted_anchor_low_first = sorted(anchor_pt, key=lambda f: pt_rank_pos.get(f, 10**9), reverse=True)
    for f_drop in sorted_anchor_low_first:
        new_pt = [f for f in anchor_pt if f != f_drop]
        add_eval(new_pt, ANCHOR_CT, "scan3_drop1pt")

    # Scan 4: N=11 add one PT not in anchor
    for cand in pt_candidates:
        new_pt = list(anchor_pt) + [cand]
        add_eval(new_pt, ANCHOR_CT, "scan4_add1pt")

    out = pd.DataFrame(rows)
    out.insert(0, "No", np.arange(1, len(out) + 1))
    out = out.sort_values(["ci_chup", "ci_chus"], ascending=False).reset_index(drop=True)

    out_file = OUT_DIR / "9_mar_subscan_all_results.csv"
    out.to_csv(out_file, index=False)

    print(f"Saved: {out_file} ({len(out)} rows)")
    print("\nTop-10 by CHUP:")
    print(out.head(10)[["No", "combo", "N", "n_pt", "ci_chus", "ci_chup", "ct_feature"]].to_string(index=False))

    flagged = out[out["ci_chup"] > CHUP_TARGET]
    print(f"\nRows with CHUP > {CHUP_TARGET:.6f}: {len(flagged)}")
    if len(flagged) > 0:
        print(flagged.head(20)[["No", "combo", "N", "n_pt", "ci_chus", "ci_chup", "pt_features", "ct_feature"]].to_string(index=False))


if __name__ == "__main__":
    main()
