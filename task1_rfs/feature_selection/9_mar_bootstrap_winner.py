"""
9_mar_bootstrap_winner.py

Bootstrap validation of the confirmed winner pipeline:
  Gender_Male + 8 PT features + 1 CT feature
  GM: SVM (alpha=0.000143, from pc_7b No.21683)
  Coach: GBS (n_estimators=100, lr=0.1, max_depth=3, subsample=0.8)
  N_BOOT = 5000

Reports point estimates and 95% bootstrap CI for:
  - OOF (dev set, 5-fold stratified CV)
  - ci_chus (CHUS external, N=55)
  - ci_chup (CHUP external, N=35)

Two coach variants tested:
  A) GBS coach (heuristic params as used in pc_6b/pc_7b)
  B) SVM coach (same alpha as GM, for comparison)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

ROOT = Path(__file__).resolve().parent

PT_DEV_FILE  = ROOT / "27_feb_PT_development.csv"
CT_DEV_FILE  = ROOT / "27_feb_CT_development.csv"
PT_EXT_FILE  = ROOT / "27_feb_PT_external.csv"
CT_EXT_FILE  = ROOT / "27_feb_CT_external.csv"
CLINICAL_FILE = (
    ROOT.parent / "Feb_2026" / "25_feb_clinical_reduced_dataset"
    / "25_feb_Processed_clinical_reduced.csv"
)

SEED   = 42
N_BOOT = 5000
N_FOLDS = 5

WINNER_CLIN = ["Gender_Male"]
WINNER_PT   = [
    "GTVp_exponential_glszm_HighGrayLevelZoneEmphasis",
    "GTVn_wavelet-LLH_firstorder_Mean",
    "GTVp_wavelet-HLH_glrlm_ShortRunHighGrayLevelEmphasis",
    "GTVp_gradient_glszm_ZoneEntropy",
    "GTVp_wavelet-LHL_glszm_SmallAreaHighGrayLevelEmphasis",
    "GTVn_wavelet-LHH_glszm_LowGrayLevelZoneEmphasis",
    "GTVp_wavelet-HLH_glszm_HighGrayLevelZoneEmphasis",
    "GTVn_log-sigma-1-mm-3D_glszm_GrayLevelNonUniformity",
]
WINNER_CT   = ["GTVp_wavelet-LLH_glrlm_HighGrayLevelRunEmphasis"]
ALL_FEATS   = WINNER_CLIN + WINNER_PT + WINNER_CT

SVM_ALPHA   = 0.000143   # pc_7b No.21683
GBS_PARAMS  = dict(n_estimators=100, learning_rate=0.1, max_depth=3, subsample=0.8)


def make_surv(event, time):
    return Surv.from_arrays(event=np.asarray(event, dtype=bool),
                            time=np.asarray(time, dtype=float))


def safe_ci(y, risk):
    try:
        return float(concordance_index_censored(y["event"], y["time"], risk)[0])
    except Exception:
        return float("nan")


def oof_ci(X, y, model_fn, seed=SEED, n_folds=N_FOLDS):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    pred = np.zeros(len(X))
    ok   = np.zeros(len(X), dtype=bool)
    for tr, vl in skf.split(X, y["event"].astype(int)):
        sc = StandardScaler()
        try:
            m = model_fn().fit(sc.fit_transform(X[tr]), y[tr])
            pred[vl] = m.predict(sc.transform(X[vl]))
            ok[vl]   = True
        except Exception:
            continue
    if ok.sum() < len(X) * 0.5:
        return float("nan")
    return safe_ci(y[ok], pred[ok])


def bootstrap_ci_ext(y_ext, risk_ext, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n   = len(y_ext)
    cis = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        c   = safe_ci(y_ext[idx], risk_ext[idx])
        if not np.isnan(c):
            cis.append(c)
    a = np.array(cis)
    if len(a) == 0:
        return dict(mean=np.nan, std=np.nan, lo=np.nan, hi=np.nan, n=0)
    return dict(
        mean=float(a.mean()),
        std=float(a.std()),
        lo=float(np.percentile(a, 2.5)),
        hi=float(np.percentile(a, 97.5)),
        n=len(a),
    )


def run_pipeline(coach_name, coach_fn):
    print(f"\n{'='*60}")
    print(f"Coach: {coach_name}")
    print(f"{'='*60}")

    # --- load data ---
    clinical = pd.read_csv(CLINICAL_FILE).dropna(subset=["Relapse", "RFS"])

    pt_dev = pd.read_csv(PT_DEV_FILE)
    ct_dev = pd.read_csv(CT_DEV_FILE)
    pt_ext = pd.read_csv(PT_EXT_FILE)
    ct_ext = pd.read_csv(CT_EXT_FILE)

    rad_dev = pt_dev[["PatientID"] + WINNER_PT].merge(
        ct_dev[["PatientID"] + WINNER_CT], on="PatientID", how="inner"
    )
    rad_ext = pt_ext[["PatientID"] + WINNER_PT].merge(
        ct_ext[["PatientID"] + WINNER_CT], on="PatientID", how="inner"
    )

    clin_dev  = clinical[clinical["Cohort"] == "Dev"][
        ["PatientID", "Relapse", "RFS"] + WINNER_CLIN
    ]
    clin_chus = clinical[clinical["CenterID"] == 3][
        ["PatientID", "Relapse", "RFS"] + WINNER_CLIN
    ]
    clin_chup = clinical[clinical["CenterID"] == 2][
        ["PatientID", "Relapse", "RFS"] + WINNER_CLIN
    ]

    dev_df  = clin_dev.merge(rad_dev, on="PatientID", how="inner")
    chus_df = clin_chus.merge(
        rad_ext[rad_ext["PatientID"].str.startswith("CHUS")], on="PatientID", how="inner"
    )
    chup_df = clin_chup.merge(
        rad_ext[rad_ext["PatientID"].str.startswith("CHUP")], on="PatientID", how="inner"
    )

    print(f"Dev: {len(dev_df)} pts | CHUS: {len(chus_df)} pts | CHUP: {len(chup_df)} pts")
    print(f"Dev events: {int(dev_df['Relapse'].sum())} | "
          f"CHUS events: {int(chus_df['Relapse'].sum())} | "
          f"CHUP events: {int(chup_df['Relapse'].sum())}")

    X_dev  = dev_df[ALL_FEATS].values.astype(float)
    y_dev  = make_surv(dev_df["Relapse"], dev_df["RFS"])
    X_chus = chus_df[ALL_FEATS].values.astype(float)
    y_chus = make_surv(chus_df["Relapse"], chus_df["RFS"])
    X_chup = chup_df[ALL_FEATS].values.astype(float)
    y_chup = make_surv(chup_df["Relapse"], chup_df["RFS"])

    # --- OOF (point estimate only, bootstrapping OOF is expensive and less standard) ---
    print(f"\nComputing OOF ({N_FOLDS}-fold)...")
    oof = oof_ci(X_dev, y_dev, coach_fn)
    print(f"  OOF C-index: {oof:.6f}")

    # --- train final model on full dev ---
    sc_final = StandardScaler()
    X_dev_sc  = sc_final.fit_transform(X_dev)
    X_chus_sc = sc_final.transform(X_chus)
    X_chup_sc = sc_final.transform(X_chup)

    model = coach_fn()
    model.fit(X_dev_sc, y_dev)
    risk_chus = model.predict(X_chus_sc)
    risk_chup = model.predict(X_chup_sc)

    ci_chus_pt = safe_ci(y_chus, risk_chus)
    ci_chup_pt = safe_ci(y_chup, risk_chup)
    print(f"\nPoint estimates:")
    print(f"  ci_chus = {ci_chus_pt:.6f}")
    print(f"  ci_chup = {ci_chup_pt:.6f}")

    # --- bootstrap external CIs ---
    print(f"\nBootstrapping CHUS (N_BOOT={N_BOOT})...")
    b_chus = bootstrap_ci_ext(y_chus, risk_chus)
    print(f"  CHUS: mean={b_chus['mean']:.4f}  std={b_chus['std']:.4f}  "
          f"95% CI [{b_chus['lo']:.4f}, {b_chus['hi']:.4f}]  (n_valid={b_chus['n']})")

    print(f"Bootstrapping CHUP (N_BOOT={N_BOOT})...")
    b_chup = bootstrap_ci_ext(y_chup, risk_chup)
    print(f"  CHUP: mean={b_chup['mean']:.4f}  std={b_chup['std']:.4f}  "
          f"95% CI [{b_chup['lo']:.4f}, {b_chup['hi']:.4f}]  (n_valid={b_chup['n']})")

    # --- concordant pairs breakdown ---
    from sksurv.metrics import concordance_index_censored as cic
    chus_raw = cic(y_chus["event"], y_chus["time"], risk_chus)
    chup_raw = cic(y_chup["event"], y_chup["time"], risk_chup)
    print(f"\nConcordant pairs:")
    print(f"  CHUS: concordant={chus_raw[1]}, discordant={chus_raw[2]}, tied={chus_raw[3]+chus_raw[4]}, "
          f"total comparable={chus_raw[1]+chus_raw[2]}")
    print(f"  CHUP: concordant={chup_raw[1]}, discordant={chup_raw[2]}, tied={chup_raw[3]+chup_raw[4]}, "
          f"total comparable={chup_raw[1]+chup_raw[2]}")

    return dict(
        coach=coach_name,
        oof=oof,
        ci_chus=ci_chus_pt,
        ci_chup=ci_chup_pt,
        chus_boot_mean=b_chus["mean"], chus_boot_std=b_chus["std"],
        chus_95lo=b_chus["lo"], chus_95hi=b_chus["hi"],
        chup_boot_mean=b_chup["mean"], chup_boot_std=b_chup["std"],
        chup_95lo=b_chup["lo"], chup_95hi=b_chup["hi"],
    )


if __name__ == "__main__":
    print("Winner pipeline bootstrap — 9_mar_bootstrap_winner.py")
    print(f"Features: {len(ALL_FEATS)} total ({len(WINNER_CLIN)} clin + {len(WINNER_PT)} PT + {len(WINNER_CT)} CT)")
    print(f"N_BOOT = {N_BOOT}")

    results = []

    # Coach A: GBS heuristic (as used in pc_6b/pc_7b winning rows)
    results.append(run_pipeline(
        "GBS_heuristic",
        lambda: GradientBoostingSurvivalAnalysis(
            random_state=SEED, **GBS_PARAMS
        )
    ))

    # Coach B: SVM (same alpha as GM)
    results.append(run_pipeline(
        "SVM_alpha0.000143",
        lambda: FastSurvivalSVM(alpha=SVM_ALPHA, max_iter=1000, tol=1e-4, random_state=SEED)
    ))

    # Coach C: CoxPH (alpha=0.1, the EPV_CUT heuristic default)
    results.append(run_pipeline(
        "CoxPH_alpha0.1",
        lambda: CoxPHSurvivalAnalysis(alpha=0.1)
    ))

    print("\n\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    out = ROOT / "9_mar_bootstrap_winner_results.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")
