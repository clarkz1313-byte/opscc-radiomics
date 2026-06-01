# -*- coding: utf-8 -*-
"""
Re-extract PT_inter1_768 and CT_inter8_235 from Stage 4 row-1 pipelines.

Inputs:
  Mar_2026/2_mar_stage4_rad/27_feb_PT_Stage4_2v_ALLtrials_20260228_Processed.csv
  Mar_2026/2_mar_stage4_rad/28_feb_CT_Stage4_2v_ALLtrials_Unified_Processed.csv
  Mar_2026/27_feb_PT_development.csv
  Mar_2026/27_feb_CT_development.csv

Outputs:
  Mar_2026/2_mar_finalist_outputs/PT_inter1_768_features.csv
  Mar_2026/2_mar_finalist_outputs/CT_inter8_235_features.csv
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
MAR_DIR = SCRIPT_DIR.parent
THESIS_ROOT = MAR_DIR.parent
sys.path.insert(0, str(THESIS_ROOT))

from fs_utils import (
    lasso_cox_selection,
    permutation_importance_survival,
    xgboost_survival_selection,
)

STAGE4_DIR = MAR_DIR / "2_mar_stage4_rad"
OUTPUT_DIR = MAR_DIR / "2_mar_finalist_outputs"

PT_STAGE4_FILE = STAGE4_DIR / "27_feb_PT_Stage4_2v_ALLtrials_20260228_Processed.csv"
CT_STAGE4_FILE = STAGE4_DIR / "28_feb_CT_Stage4_2v_ALLtrials_Unified_Processed.csv"

PT_DEV_FILE = MAR_DIR / "27_feb_PT_development.csv"
CT_DEV_FILE = MAR_DIR / "27_feb_CT_development.csv"

PT_OUT_FILE = OUTPUT_DIR / "PT_inter1_768_features.csv"
CT_OUT_FILE = OUTPUT_DIR / "CT_inter8_235_features.csv"

SEED = 42


def _load_xy(dev_file: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(dev_file)
    feature_cols = [c for c in df.columns if c not in ("PatientID", "Relapse", "RFS")]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(df[feature_cols].median())
    y = pd.DataFrame(
        {
            "RFS_time": df["RFS"].values,
            "event": df["Relapse"].values.astype(bool),
        }
    )
    return X, y


def _build_rows(features: list[str], modality: str, inter_no: int, row: pd.Series) -> list[dict]:
    fea = int(row.get("Fea4", len(features))) if pd.notna(row.get("Fea4")) else len(features)
    return [
        {
            "Feature": f,
            "Modality": modality,
            "inter_no": inter_no,
            "intra_no": str(row["intra_no"]),
            "Pipeline": str(row.get("Pipeline", "")),
            "CV4": row.get("CV4"),
            "Fea": fea,
            "Ext_CHUS": row.get("Ext_CHUS"),
            "Ext_CHUP": row.get("Ext_CHUP"),
        }
        for f in features
    ]


def _extract_pt_row1() -> tuple[list[str], pd.Series]:
    pt_row = pd.read_csv(PT_STAGE4_FILE).iloc[0]
    X_pt, y_pt = _load_xy(PT_DEV_FILE)

    s1 = lasso_cox_selection(
        X_pt,
        y_pt,
        target_features=int(pt_row["params_lasso_target_features"]),
        n_alphas=int(pt_row["params_lasso_n_alphas"]),
    )
    s2 = permutation_importance_survival(
        X_pt[s1],
        y_pt,
        n_features=min(int(pt_row["params_permimp_n_features"]), len(s1)),
        n_estimators=int(pt_row["params_permimp_n_estimators"]),
        random_state=SEED,
    )
    s3 = xgboost_survival_selection(
        X_pt[s1][s2],
        y_pt,
        n_features=min(int(pt_row["params_xgb_n_features"]), len(s2)),
        n_estimators=int(pt_row["params_xgb_n_estimators"]),
        random_state=SEED,
    )
    return s3, pt_row


def _extract_ct_row1() -> tuple[list[str], pd.Series]:
    ct_row = pd.read_csv(CT_STAGE4_FILE).iloc[0]
    X_ct, y_ct = _load_xy(CT_DEV_FILE)

    s1 = lasso_cox_selection(
        X_ct,
        y_ct,
        target_features=int(ct_row["params_lasso_target_features"]),
        n_alphas=int(ct_row["params_lasso_n_alphas"]),
    )
    s2 = permutation_importance_survival(
        X_ct[s1],
        y_ct,
        n_features=min(int(ct_row["params_permimp_n_features"]), len(s1)),
        n_estimators=int(ct_row["params_permimp_n_estimators"]),
        random_state=SEED,
    )
    s3 = xgboost_survival_selection(
        X_ct[s1][s2],
        y_ct,
        n_features=min(int(ct_row["params_s3_xgb_n_features"]), len(s2)),
        n_estimators=int(ct_row["params_s3_xgb_n_estimators"]),
        random_state=SEED,
    )
    return s3, ct_row


def main() -> None:
    if not PT_STAGE4_FILE.exists() or not CT_STAGE4_FILE.exists():
        raise FileNotFoundError("Stage 4 input files are missing in Mar_2026/2_mar_stage4_rad")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pt_features, pt_row = _extract_pt_row1()
    ct_features, ct_row = _extract_ct_row1()

    pd.DataFrame(_build_rows(pt_features, "PT", 1, pt_row)).to_csv(PT_OUT_FILE, index=False)
    pd.DataFrame(_build_rows(ct_features, "CT", 8, ct_row)).to_csv(CT_OUT_FILE, index=False)

    print(f"Saved: {PT_OUT_FILE} ({len(pt_features)} features)")
    print(f"Saved: {CT_OUT_FILE} ({len(ct_features)} features)")
    print("Done.")


if __name__ == "__main__":
    main()
