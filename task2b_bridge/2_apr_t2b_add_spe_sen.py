"""
2_apr_t2b_add_spe_sen.py

Retroactively adds ext_spe and ext_sen columns to:
  - t2b_all_results_4.csv
  - t2b_all_results_5_merged.csv  (and the four pool-level CSVs)

For each row, the HPV model is refit on full training data using the stored
feature set and GM labels, then evaluated on the CHUS external cohort using the
Youden threshold.  All other columns are preserved unchanged.

Run:
    python Apr_2026_task2B/2_apr_t2b_add_spe_sen.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sksurv.util import Surv

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "2_apr_T2B_data"
OUT_DIR  = ROOT / "2_apr_T2B_outputs"

TRAIN_FILE = DATA_DIR / "2_apr_t2b_train.csv"
EXT_FILE   = DATA_DIR / "2_apr_t2b_ext.csv"

CLINICAL_FEATURES = ["Gender_Male"]
SEED = 42


# ---------------------------------------------------------------------------
# GM factories (matching v4/v5 exactly)
# ---------------------------------------------------------------------------

def _make_hpv(hpv_gm_label: str):
    if hpv_gm_label == "LR_L2_0.5":
        return LogisticRegression(C=0.5, penalty="l2", solver="lbfgs",
                                  class_weight="balanced", max_iter=2000, random_state=SEED)
    if hpv_gm_label == "LR_EN_1.0":
        return LogisticRegression(C=1.0, penalty="elasticnet", solver="saga",
                                  l1_ratio=0.5, class_weight="balanced",
                                  max_iter=5000, random_state=SEED)
    if hpv_gm_label == "SVM_L_001":
        base = LinearSVC(C=0.01, class_weight="balanced", max_iter=5000, random_state=SEED)
        return CalibratedClassifierCV(base, cv=3)
    # v3 default (no hpv_gm column) -- LR_L2 with search C from lr_C column
    raise ValueError(f"Unknown hpv_gm label: {hpv_gm_label!r}")


def _youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        fpr, tpr, thr = roc_curve(y_true, scores)
        return float(thr[int(np.argmax(tpr - fpr))])
    except Exception:
        return 0.5


def _scale_and_predict(train_df, ext_df, feat_pt: list[str], feat_ct: list[str],
                       hpv_gm_label: str) -> tuple[float, float, float]:
    """Returns (ext_spe, ext_sen, ext_ba_check).  Raises on failure."""
    all_feat = CLINICAL_FEATURES + feat_pt + feat_ct

    x_clin_tr = train_df[CLINICAL_FEATURES].to_numpy(dtype=float)
    x_pt_tr   = train_df[feat_pt].to_numpy(dtype=float)
    x_ct_tr   = train_df[feat_ct].to_numpy(dtype=float)

    x_clin_ex = ext_df[CLINICAL_FEATURES].to_numpy(dtype=float)
    x_pt_ex   = ext_df[feat_pt].to_numpy(dtype=float)
    x_ct_ex   = ext_df[feat_ct].to_numpy(dtype=float)

    sc_c = StandardScaler().fit(x_clin_tr)
    sc_p = StandardScaler().fit(x_pt_tr)
    sc_t = StandardScaler().fit(x_ct_tr)

    x_tr = np.hstack([sc_c.transform(x_clin_tr), sc_p.transform(x_pt_tr), sc_t.transform(x_ct_tr)])
    x_ex = np.hstack([sc_c.transform(x_clin_ex), sc_p.transform(x_pt_ex), sc_t.transform(x_ct_ex)])

    y_hpv_tr  = train_df["HPV_binary"].to_numpy(dtype=int)
    y_hpv_ext = ext_df["HPV_binary"].to_numpy(dtype=int)

    hm = _make_hpv(hpv_gm_label)
    hm.fit(x_tr, y_hpv_tr)
    proba = hm.predict_proba(x_ex)[:, 1]

    thresh = _youden_threshold(y_hpv_ext, proba)
    pred   = (proba >= thresh).astype(int)

    tn = int(((y_hpv_ext == 0) & (pred == 0)).sum())
    fp = int(((y_hpv_ext == 0) & (pred == 1)).sum())
    fn = int(((y_hpv_ext == 1) & (pred == 0)).sum())
    tp = int(((y_hpv_ext == 1) & (pred == 1)).sum())

    ext_spe = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    ext_sen = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")

    return ext_spe, ext_sen


def _add_metrics(df: pd.DataFrame, train_df: pd.DataFrame, ext_df: pd.DataFrame,
                 default_hpv_label: str | None = None) -> pd.DataFrame:
    """Add ext_spe / ext_sen columns.  Skips rows that already have them."""
    if "ext_spe" in df.columns and "ext_sen" in df.columns:
        print("  Already has ext_spe/ext_sen — skipping.")
        return df

    spe_vals, sen_vals = [], []
    for i, row in df.iterrows():
        feat_pt = [f for f in str(row["feat_pt"]).split("|") if f]
        feat_ct = [f for f in str(row["feat_ct"]).split("|") if f]

        if default_hpv_label is not None:
            hpv_label = default_hpv_label
        else:
            hpv_label = str(row["hpv_gm"])

        try:
            spe, sen = _scale_and_predict(train_df, ext_df, feat_pt, feat_ct, hpv_label)
        except Exception as e:
            print(f"  [WARN] row {i} failed: {e}")
            spe, sen = float("nan"), float("nan")

        spe_vals.append(spe)
        sen_vals.append(sen)

    out = df.copy()
    # Insert after ext_ba if present, else after ext_auc
    insert_after = "ext_ba" if "ext_ba" in out.columns else "ext_auc"
    pos = out.columns.get_loc(insert_after) + 1
    out.insert(pos,     "ext_spe", spe_vals)
    out.insert(pos + 1, "ext_sen", sen_vals)
    return out


def main() -> None:
    print("Loading staged data...")
    train_df = pd.read_csv(TRAIN_FILE)
    ext_df   = pd.read_csv(EXT_FILE)
    assert len(train_df) == 87 and len(ext_df) == 27

    # ---- v4 ----
    v4_path = OUT_DIR / "t2b_all_results_4.csv"
    print(f"\nProcessing {v4_path.name} ({sum(1 for _ in open(v4_path))-1} rows)...")
    df4 = pd.read_csv(v4_path)
    df4_out = _add_metrics(df4, train_df, ext_df)
    df4_out.to_csv(v4_path, index=False)
    print(f"  Saved {v4_path.name}")

    # ---- v5 merged ----
    v5_merged_path = OUT_DIR / "t2b_all_results_5_merged.csv"
    print(f"\nProcessing {v5_merged_path.name} ({sum(1 for _ in open(v5_merged_path))-1} rows)...")
    df5m = pd.read_csv(v5_merged_path)
    df5m_out = _add_metrics(df5m, train_df, ext_df)
    df5m_out.to_csv(v5_merged_path, index=False)
    print(f"  Saved {v5_merged_path.name}")

    # ---- v5 pool-level CSVs ----
    for pool_name in ["RFS_consensus", "UNIVAR_HPV", "UNIVAR_JOINT", "GBC_HPV"]:
        p = OUT_DIR / f"t2b_all_results_5_{pool_name}.csv"
        if not p.exists():
            print(f"  [SKIP] {p.name} not found")
            continue
        print(f"\nProcessing {p.name}...")
        dfp = pd.read_csv(p)
        dfp_out = _add_metrics(dfp, train_df, ext_df)
        dfp_out.to_csv(p, index=False)
        print(f"  Saved {p.name}")

    # ---- v6 pool-level CSVs (partial run — patch whatever exists) ----
    import glob
    v6_files = sorted(glob.glob(str(OUT_DIR / "t2b_all_results_6_*.csv")))
    for p_str in v6_files:
        p = Path(p_str)
        if "merged" in p.name:
            continue  # merged is rebuilt separately
        try:
            dfv6 = pd.read_csv(p)
            if dfv6.empty:
                print(f"  [SKIP] {p.name} is empty")
                continue
        except Exception:
            print(f"  [SKIP] {p.name} unreadable")
            continue
        if "ext_ci" not in dfv6.columns and "ext_auc" not in dfv6.columns:
            print(f"  [SKIP] {p.name} has no ext_* columns (no hard-floor survivors)")
            continue
        print(f"\nProcessing {p.name} ({len(dfv6)} rows)...")
        dfv6_out = _add_metrics(dfv6, train_df, ext_df)
        dfv6_out.to_csv(p, index=False)
        print(f"  Saved {p.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
