"""
6_mar_SC3_rad_PT768xCT8_325_PTforced_pc.py

Exhaustive SC3 with PT-forced modality split in GM Optuna objective.
PC machine version: TRIALS=2000, GM=SVM+CoxPH only, saves ALL trials x 4 coaches.

Problem identified from PT467/PT768 results:
  - Both runs failed: all rad features selected were GTVn/CT-dominant.
  - Root cause: unified rad ranking pools PT(21) + CT(39) features together.
    When n_rad is small, top features are CT/GTVn -- PT never selected.
  - PT817 succeeded because 5/6 rad features were PT.

Fix -- Modality-split GM objective (same as Dell traditional version):
  - Rankings are computed SEPARATELY for PT features and CT features.
  - Optuna now samples n_pt (PT count) and n_ct (CT count) independently.
  - Hard constraints:
      PT_MIN = 2   (at least 2 PT features always selected)
      CT_MIN = 1   (at least 1 CT feature always selected)
      n_pt + n_ct = n_rad = N - n_clin
  - n_pt sampled as integer in [PT_MIN, min(n_rad - CT_MIN, len(pt_features))]
  - n_ct = n_rad - n_pt

Exhaustive design (same as 6_mar_SC3_rad_PT467xCT8_325_exhaustive.py):
  - Saves EVERY completed Optuna trial x 4 coaches (no Stage 2 substitution scan)
  - No bootstrap (add later once winner identified)
  - OOF reused from GM trial for matching coach; NaN for non-GM coaches
  - Checkpoint saved after each ranker/GM combo

New output columns: n_pt, n_ct; feat_names tagged with "PT:" / "CT:" prefix.

Machine: PC | TRIALS=2000 | N_MIN=6, N_MAX=15 | PT_MIN=4, CT_MIN=1
GM: SVM + CoxPH (EST/GBS too slow at 2000 trials)
Coaches: EST, SVM, GBS, CoxPH

Rad pool: PT768 (35 PT feat) + CT8_325 (39 CT feat) = 60 rad features
Clinical: 3 features (Age, Gender_Male, Treatment_CRT)
Config: N_FOLDS=5, SEED=42, TRIALS=2000

Outputs: Mar_2026/6_mar_SC3_rad_PT768xCT8_325_PTforced_pc_outputs/

Usage:
  python Mar_2026/6_mar_SC3_rad_PT768xCT8_325_PTforced_pc.py
  SC3_TRIALS=1500 python Mar_2026/6_mar_SC3_rad_PT768xCT8_325_PTforced_pc.py
"""

import os
import subprocess
import sys
import warnings
warnings.filterwarnings("ignore")

try:
    import optuna
except ImportError:
    print("Installing optuna...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna", "-q"])
    import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.ensemble import ExtraSurvivalTrees, GradientBoostingSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

# ==========================================
# CONFIG
# ==========================================
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "6_mar_SC3_rad_PT768xCT8_325_PTforced_pc_outputs"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

FINALIST_DIR = ROOT / "2_mar_finalist_outputs"
PT_FEATURES_FILE = FINALIST_DIR / "PT_inter1_768_features.csv"   # 21 PT features
CT_FEATURES_FILE = FINALIST_DIR / "CT_inter8_325_features.csv"   # 39 CT features

PT_DEV_FILE = ROOT / "27_feb_PT_development.csv"
CT_DEV_FILE = ROOT / "27_feb_CT_development.csv"
PT_EXT_FILE = ROOT / "27_feb_PT_external.csv"
CT_EXT_FILE = ROOT / "27_feb_CT_external.csv"

CLINICAL_FILE = (
    ROOT.parent / "Feb_2026" / "25_feb_clinical_reduced_dataset"
    / "25_feb_Processed_clinical_reduced.csv"
)
CLINICAL_FEATURES = ["Age", "Gender_Male", "Treatment_CRT"]

SEED = 42
N_FOLDS = 5
W_PERF, W_STAB, STD_THRESHOLD = 0.7, 0.3, 0.08
N_MIN, N_MAX = 6, 15                 # min=6: n_clin(1)+PT_MIN(4)+CT_MIN(1)=6
CLIN_RATIO_MIN, CLIN_RATIO_MAX = 0.05, 0.50
PT_MIN = 4                           # always at least 4 PT features in rad slot
CT_MIN = 1                           # always at least 1 CT feature in rad slot
TRIALS = int(os.getenv("SC3_TRIALS", "2000"))

# GM: SVM + CoxPH only (EST/GBS ~10-15s/trial -> infeasible at 2000 trials)
# EST and GBS still appear as coaches (heuristic params, near-zero eval cost)
RANKING_METHODS = ["UNIVAR", "BORDA", "GBS", "StabLASSO"]
GM_MODEL_NAMES = ["SVM", "CoxPH"]
COACH_NAMES = ["EST", "SVM", "GBS", "CoxPH"]
COACH_HEURISTICS = {
    "EST":  {"n_estimators": 300, "max_depth": 12, "min_samples_split": 6,
             "min_samples_leaf": 4, "max_features": "sqrt"},
    "GBS":  {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "subsample": 0.8},
    "SVM":  {"alpha": 0.01},
    "CoxPH":{"alpha": 0.1},
}
N_STAB_BOOT = 50
LASSO_RANK_ALPHA = 1e-4
GBS_RANK_N_EST, GBS_RANK_LR, GBS_RANK_DEPTH, GBS_RANK_SUBSAMP = 100, 0.1, 3, 0.8

# ==========================================
# LOGGING
# ==========================================
LOG_FILE_PATH = OUTPUT_DIR / "6_mar_SC3_rad_PT768xCT8_325_PTforced_pc_log.md"


class _Tee:
    def __init__(self, *s):
        self.streams = s

    def write(self, d):
        for s in self.streams:
            s.write(d)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


_log_fh = open(LOG_FILE_PATH, "w", encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, _log_fh)
sys.stderr = _Tee(sys.__stderr__, _log_fh)

# ==========================================
# HELPERS
# ==========================================
def make_surv(e, t):
    return Surv.from_arrays(event=np.asarray(e, dtype=bool), time=np.asarray(t, dtype=float))


def safe_ci(y, r):
    try:
        return float(concordance_index_censored(y["event"], y["time"], r)[0])
    except Exception:
        return np.nan


def make_est(**p):
    return ExtraSurvivalTrees(**p, random_state=SEED, n_jobs=-1)


def make_svm(**p):
    return FastSurvivalSVM(**p, max_iter=1000, tol=1e-4, random_state=SEED)


def make_gbs(**p):
    return GradientBoostingSurvivalAnalysis(**p, random_state=SEED)


def make_coxph(**p):
    return CoxPHSurvivalAnalysis(**p)


def build_model_fn(n, p):
    return {"EST": make_est, "GBS": make_gbs, "SVM": make_svm, "CoxPH": make_coxph}[n](**p)


def model_oof(X, y, f_idx, fn):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    y_ev = y["event"].astype(int)
    oof_pred = np.zeros(len(X))
    oof_ok = np.zeros(len(X), dtype=bool)
    f_cs = []
    for tr_idx, v_idx in skf.split(X, y_ev):
        X_tr, X_v = X[tr_idx][:, f_idx], X[v_idx][:, f_idx]
        sc = StandardScaler()
        try:
            m = fn().fit(sc.fit_transform(X_tr), y[tr_idx])
            p = m.predict(sc.transform(X_v))
            oof_pred[v_idx], oof_ok[v_idx] = p, True
            c = safe_ci(y[v_idx], p)
            if not np.isnan(c):
                f_cs.append(c)
        except Exception:
            continue
    if oof_ok.sum() < len(X) * 0.5:
        return np.nan, np.nan
    return safe_ci(y[oof_ok], oof_pred[oof_ok]), float(np.std(f_cs)) if len(f_cs) > 1 else 1.0


def cox_rank(X, y, n, a=LASSO_RANK_ALPHA):
    c = None
    for _a in [a, a / 10, a / 100]:
        try:
            m = CoxnetSurvivalAnalysis(alphas=[_a], l1_ratio=1., fit_baseline_model=True,
                                       normalize=False, max_iter=10000).fit(X, y)
            c = np.asarray(m.coef_).ravel()
            break
        except Exception:
            c = None
            continue
    if c is None:
        c = np.zeros(len(n))
    i = np.abs(c)
    o = np.argsort(i)[::-1]
    return [n[j] for j in o], i[o]


def univar_rank(X, y, n):
    s = [max(c, 1 - c) if not np.isnan(c := safe_ci(y, X[:, i])) else 0.5
         for i in range(X.shape[1])]
    o = np.argsort(np.array(s))[::-1]
    return [n[i] for i in o], np.array(s)[o]


def borda_combine(rl, n):
    pm = {f: [r for l in rl for r, p in enumerate(l) if p == f] for f in n}
    ap = {f: (np.mean(pm[f]) if pm[f] else len(n)) for f in n}
    sf = sorted(n, key=lambda f: (ap[f], f))
    return sf, np.array([ap[f] for f in sf])


def gbs_rank(X, y, n):
    m = GradientBoostingSurvivalAnalysis(
        n_estimators=GBS_RANK_N_EST, learning_rate=GBS_RANK_LR,
        max_depth=GBS_RANK_DEPTH, subsample=GBS_RANK_SUBSAMP, random_state=SEED
    ).fit(X, y)
    i = m.feature_importances_
    o = np.argsort(i)[::-1]
    return [n[j] for j in o], i[o]


def stab_lasso_rank(X, y, n, nb=N_STAB_BOOT, a=LASSO_RANK_ALPHA):
    rng, f, ok = np.random.default_rng(SEED), np.zeros(len(n)), 0
    for b in range(nb):
        idx = rng.integers(0, len(X), size=len(X))
        for _a in [a, a / 10, a / 100]:
            try:
                m = CoxnetSurvivalAnalysis(alphas=[_a], l1_ratio=1., fit_baseline_model=True,
                                           normalize=False, max_iter=5000).fit(X[idx], y[idx])
                f += (np.abs(np.asarray(m.coef_).ravel()) > 0).astype(float)
                ok += 1
                break
            except Exception:
                continue
    f /= max(ok, 1)
    o = np.argsort(f)[::-1]
    return [n[i] for i in o], f[o]


# ==========================================
# 1. LOAD DATA
# ==========================================
print("=" * 70)
print("6_mar_SC3_rad_PT768xCT8_325_PTforced_pc")
print("Exhaustive SC3 + PT-forced modality split: n_pt and n_ct sampled independently")
print(f"PT_MIN={PT_MIN} (always >=2 PT features) | CT_MIN={CT_MIN}")
print("Rad pool: PT768 (35 PT feat) + CT8_325 (39 CT feat) = 60 rad features")
print("GM: SVM + CoxPH | Coaches: EST, SVM, GBS, CoxPH | Machine: PC")
print("No Stage 2 substitution | No bootstrap | Saves ALL trials x 4 coaches")
print("=" * 70)

clinical = pd.read_csv(CLINICAL_FILE)
clinical = clinical.dropna(subset=["Relapse", "RFS"]).copy()
clin_dev = clinical[clinical["Cohort"] == "Dev"][
    ["PatientID", "CenterID", "Cohort", "Relapse", "RFS"] + CLINICAL_FEATURES
].copy()
clin_chus = clinical[clinical["CenterID"] == 3][
    ["PatientID", "Relapse", "RFS"] + CLINICAL_FEATURES
].copy()
clin_chup = clinical[clinical["CenterID"] == 2][
    ["PatientID", "Relapse", "RFS"] + CLINICAL_FEATURES
].copy()
print(f"Clinical dev: {len(clin_dev)} patients, {clin_dev['Relapse'].sum():.0f} events")
print(f"Clinical CHUS: {len(clin_chus)} patients, {clin_chus['Relapse'].sum():.0f} events")
print(f"Clinical CHUP: {len(clin_chup)} patients, {clin_chup['Relapse'].sum():.0f} events")

pt_feat_list = pd.read_csv(PT_FEATURES_FILE)["Feature"].tolist()
ct_feat_list_raw = pd.read_csv(CT_FEATURES_FILE)["Feature"].tolist()
pt_feat_set = set(pt_feat_list)
ct_feat_list = [f for f in ct_feat_list_raw if f not in pt_feat_set]
n_dropped = len(ct_feat_list_raw) - len(ct_feat_list)
if n_dropped > 0:
    dropped = [f for f in ct_feat_list_raw if f in pt_feat_set]
    print(f"  Dropped {n_dropped} CT features duplicated in PT list: {dropped}")
rad_features = pt_feat_list + ct_feat_list
print(f"\nRadiomics pool: PT467={len(pt_feat_list)} + CT8_325={len(ct_feat_list)} "
      f"(raw {len(ct_feat_list_raw)}, {n_dropped} deduped) = {len(rad_features)} features")
print(f"PT features: {len(pt_feat_list)} | CT features: {len(ct_feat_list)}")

pt_dev = pd.read_csv(PT_DEV_FILE)
ct_dev = pd.read_csv(CT_DEV_FILE)
pt_ext = pd.read_csv(PT_EXT_FILE)
ct_ext = pd.read_csv(CT_EXT_FILE)

rad_dev = pt_dev[["PatientID"] + pt_feat_list].merge(
    ct_dev[["PatientID"] + ct_feat_list], on="PatientID", how="inner"
)
rad_ext = pt_ext[["PatientID"] + pt_feat_list].merge(
    ct_ext[["PatientID"] + ct_feat_list], on="PatientID", how="inner"
)
rad_chus = rad_ext[rad_ext["PatientID"].str.startswith("CHUS")]
rad_chup = rad_ext[rad_ext["PatientID"].str.startswith("CHUP")]

all_features = CLINICAL_FEATURES + rad_features
dev_merged = clin_dev.merge(rad_dev, on="PatientID", how="inner")
chus_merged = clin_chus.merge(rad_chus, on="PatientID", how="inner")
chup_merged = clin_chup.merge(rad_chup, on="PatientID", how="inner")

print(f"\nAfter merge:")
print(f"  Dev: {len(dev_merged)} pts, {dev_merged['Relapse'].sum():.0f} events")
print(f"  CHUS: {len(chus_merged)} pts, {chus_merged['Relapse'].sum():.0f} events")
print(f"  CHUP: {len(chup_merged)} pts, {chup_merged['Relapse'].sum():.0f} events")
print(f"  Total features: {len(all_features)} ({len(CLINICAL_FEATURES)} clin + {len(rad_features)} rad)")

for name, df in [("Dev", dev_merged), ("CHUS", chus_merged), ("CHUP", chup_merged)]:
    missing = df[all_features].isna().sum().sum()
    if missing > 0:
        print(f"WARNING: {missing} missing values in {name}")
    else:
        print(f"  {name}: 0 missing values")

X_train = dev_merged[all_features].values.astype(float)
y_train = make_surv(dev_merged["Relapse"], dev_merged["RFS"])
X_chus = chus_merged[all_features].values.astype(float)
y_chus = make_surv(chus_merged["Relapse"], chus_merged["RFS"])
X_chup = chup_merged[all_features].values.astype(float)
y_chup = make_surv(chup_merged["Relapse"], chup_merged["RFS"])

feat_idx_map = {f: i for i, f in enumerate(all_features)}
clin_feat = CLINICAL_FEATURES
pt_feat = pt_feat_list                 # PT features only
ct_feat = ct_feat_list                 # CT features only
clin_idx = [feat_idx_map[f] for f in clin_feat]
pt_idx   = [feat_idx_map[f] for f in pt_feat]
ct_idx   = [feat_idx_map[f] for f in ct_feat]
pt_feat_set_idx = set(pt_feat)
ct_feat_set_idx = set(ct_feat)

n_events_train = int(y_train["event"].sum())
print(f"\nEPV (train, {len(all_features)} feat): {n_events_train}/{len(all_features)} = {n_events_train/len(all_features):.2f}")
print(f"TRIALS per study: {TRIALS}")
expected_rows = len(RANKING_METHODS) * len(GM_MODEL_NAMES) * TRIALS * len(COACH_NAMES)
print(f"Expected output rows: {len(RANKING_METHODS)} rankers x {len(GM_MODEL_NAMES)} GM x {TRIALS} trials x {len(COACH_NAMES)} coaches = {expected_rows:,}")

# ==========================================
# 2. PRE-COMPUTE RANKINGS — separate PT and CT
# ==========================================
print("\n--- Pre-computing feature rankings (PT and CT ranked separately) ---")
g_sc = StandardScaler()
X_train_sc = g_sc.fit_transform(X_train)

clin_feat_arr = np.array(clin_feat)
pt_feat_arr   = np.array(pt_feat)
ct_feat_arr   = np.array(ct_feat)

rankings = {}
for m in ["UNIVAR", "GBS"]:
    fn = univar_rank if m == "UNIVAR" else gbs_rank
    rankings[m] = {
        "clin": fn(X_train_sc[:, clin_idx], y_train, clin_feat_arr)[0],
        "pt":   fn(X_train_sc[:, pt_idx],   y_train, pt_feat_arr)[0],
        "ct":   fn(X_train_sc[:, ct_idx],   y_train, ct_feat_arr)[0],
    }
    print(f"  {m} done")

rankings["StabLASSO"] = {
    "clin": stab_lasso_rank(X_train_sc[:, clin_idx], y_train, clin_feat_arr)[0],
    "pt":   stab_lasso_rank(X_train_sc[:, pt_idx],   y_train, pt_feat_arr)[0],
    "ct":   stab_lasso_rank(X_train_sc[:, ct_idx],   y_train, ct_feat_arr)[0],
}
print("  StabLASSO done")

rankings["BORDA"] = {
    "clin": borda_combine(
        [cox_rank(X_train_sc[:, clin_idx], y_train, clin_feat_arr)[0],
         rankings["UNIVAR"]["clin"]], clin_feat
    )[0],
    "pt": borda_combine(
        [cox_rank(X_train_sc[:, pt_idx], y_train, pt_feat_arr)[0],
         rankings["UNIVAR"]["pt"]], pt_feat
    )[0],
    "ct": borda_combine(
        [cox_rank(X_train_sc[:, ct_idx], y_train, ct_feat_arr)[0],
         rankings["UNIVAR"]["ct"]], ct_feat
    )[0],
}
print("  BORDA done")

# ==========================================
# 3. OPTUNA GM SEARCH + EXHAUSTIVE COACH EVAL (PT-forced split)
# ==========================================
print(f"\n--- Exhaustive search: {TRIALS} trials x {len(RANKING_METHODS)} rankers x "
      f"{len(GM_MODEL_NAMES)} GM x {len(COACH_NAMES)} coaches (PT-forced) ---")


def make_gm_objective(gm_name, clin_ridx, pt_ridx, ct_ridx):
    n_pt_pool = len(pt_ridx)
    n_ct_pool = len(ct_ridx)

    def objective(trial):
        N = trial.suggest_int("total_n", N_MIN, N_MAX)
        clin_ratio = trial.suggest_float("clin_ratio", CLIN_RATIO_MIN, CLIN_RATIO_MAX)
        n_clin = max(1, round(N * clin_ratio))
        n_clin = min(n_clin, len(clin_ridx), N - PT_MIN - CT_MIN)
        n_rad = N - n_clin

        # PT/CT split: sample n_pt independently; n_ct = remainder
        n_pt_max = min(n_rad - CT_MIN, n_pt_pool)
        n_pt_min = min(PT_MIN, n_pt_max)
        if n_pt_min > n_pt_max:
            n_pt_min = n_pt_max
        n_pt = trial.suggest_int("n_pt", n_pt_min, n_pt_max)
        n_ct = n_rad - n_pt
        n_ct = max(CT_MIN, min(n_ct, n_ct_pool))
        # Re-adjust n_pt if n_ct was clamped
        n_pt = n_rad - n_ct
        n_pt = max(PT_MIN, min(n_pt, n_pt_pool))

        feat_idx = clin_ridx[:n_clin] + pt_ridx[:n_pt] + ct_ridx[:n_ct]

        if gm_name == "SVM":
            model_fn = lambda: build_model_fn("SVM", {"alpha": trial.suggest_float("alpha_svm", 1e-4, 1., log=True)})
        elif gm_name == "CoxPH":
            model_fn = lambda: build_model_fn("CoxPH", {"alpha": trial.suggest_float("alpha_cox", 1e-3, 10., log=True)})
        else:
            raise ValueError(f"Unknown GM: {gm_name}")

        oof_ci, fold_std = model_oof(X_train, y_train, feat_idx, model_fn)
        if np.isnan(oof_ci):
            oof_ci, fold_std = 0.5, 1.0
        score = W_PERF * oof_ci + W_STAB * max(0., 1. - fold_std / STD_THRESHOLD)

        trial.set_user_attr("N", N)
        trial.set_user_attr("n_clin", n_clin)
        trial.set_user_attr("n_pt", n_pt)
        trial.set_user_attr("n_ct", n_ct)
        trial.set_user_attr("oof_ci", oof_ci)
        trial.set_user_attr("fold_std", fold_std)
        trial.set_user_attr("features", [all_features[i] for i in feat_idx])
        return score

    return objective


all_rows = []
global_row_no = 0

for method in RANKING_METHODS:
    clin_r = rankings[method]["clin"]
    pt_r   = rankings[method]["pt"]
    ct_r   = rankings[method]["ct"]
    clin_ridx = [feat_idx_map[f] for f in clin_r]
    pt_ridx   = [feat_idx_map[f] for f in pt_r]
    ct_ridx   = [feat_idx_map[f] for f in ct_r]

    for gm_name in GM_MODEL_NAMES:
        print(f"\n  Optuna: {method}/{gm_name} ({TRIALS} trials, PT-forced)...", flush=True)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEED)
        )
        # Seed enqueue near PT817 known-good: N=6, 1 clin, 3 PT, 2 CT
        if method == "BORDA" and gm_name == "SVM":
            study.enqueue_trial({"total_n": 6, "clin_ratio": 0.17, "n_pt": 3, "alpha_svm": 1e-3})
            study.enqueue_trial({"total_n": 6, "clin_ratio": 0.17, "n_pt": 4, "alpha_svm": 5e-4})
            study.enqueue_trial({"total_n": 6, "clin_ratio": 0.05, "n_pt": 3, "alpha_svm": 1e-4})
        elif gm_name == "SVM":
            study.enqueue_trial({"total_n": 6, "clin_ratio": 0.17, "n_pt": 3, "alpha_svm": 0.01})
        elif gm_name == "CoxPH":
            study.enqueue_trial({"total_n": 6, "clin_ratio": 0.17, "n_pt": 3, "alpha_cox": 0.1})
            study.enqueue_trial({"total_n": 5, "clin_ratio": 0.20, "n_pt": 3, "alpha_cox": 0.1})

        study.optimize(
            make_gm_objective(gm_name, clin_ridx, pt_ridx, ct_ridx),
            n_trials=TRIALS,
            show_progress_bar=False
        )

        bt = study.best_trial
        bt_feats = bt.user_attrs["features"]
        bt_n_pt = bt.user_attrs.get("n_pt", sum(1 for f in bt_feats if f in pt_feat_set_idx))
        bt_n_ct = bt.user_attrs.get("n_ct", sum(1 for f in bt_feats if f in ct_feat_set_idx))
        print(f"  Best: N={bt.user_attrs['N']} (PT={bt_n_pt} CT={bt_n_ct}) "
              f"| OOF={bt.user_attrs['oof_ci']:.4f} | trial #{bt.number}")

        # --- EXHAUSTIVE: evaluate ALL completed trials with all coaches ---
        completed = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE]
        print(f"  Evaluating {len(completed)} completed trials x {len(COACH_NAMES)} coaches...",
              flush=True)

        for t_idx, trial in enumerate(completed):
            feat_names = trial.user_attrs.get("features", [])
            if not feat_names:
                continue
            col_idx = [feat_idx_map[f] for f in feat_names if f in feat_idx_map]
            if len(col_idx) == 0:
                continue

            n_clin_t = trial.user_attrs.get("n_clin", 0)
            n_pt_t   = trial.user_attrs.get("n_pt", sum(1 for f in feat_names if f in pt_feat_set_idx))
            n_ct_t   = trial.user_attrs.get("n_ct", sum(1 for f in feat_names if f in ct_feat_set_idx))
            oof_ci_gm = trial.user_attrs.get("oof_ci", np.nan)
            fold_std_gm = trial.user_attrs.get("fold_std", np.nan)

            # GM params for this trial
            if gm_name == "SVM":
                gm_params = {"alpha": trial.params.get("alpha_svm", 0.01)}
            elif gm_name == "CoxPH":
                gm_params = {"alpha": trial.params.get("alpha_cox", 0.1)}
            else:
                gm_params = {}

            # Separate feat_names into pt and ct lists (preserving order from feat_names)
            pt_names_t = [f for f in feat_names if f in pt_feat_set_idx]
            ct_names_t = [f for f in feat_names if f in ct_feat_set_idx]
            clin_names_t = [f for f in feat_names if f not in pt_feat_set_idx and f not in ct_feat_set_idx]

            # Fit scaler once per trial feature set
            sc_final = StandardScaler()
            X_tr_sc = sc_final.fit_transform(X_train[:, col_idx])
            X_chus_sc = sc_final.transform(X_chus[:, col_idx])
            X_chup_sc = sc_final.transform(X_chup[:, col_idx])

            for coach_name in COACH_NAMES:
                coach_params = gm_params if coach_name == gm_name else COACH_HEURISTICS[coach_name]
                coach_fn = lambda cn=coach_name, cp=coach_params: build_model_fn(cn, cp)

                # OOF reused from GM trial (already computed during Optuna).
                # Non-GM coaches do NOT recompute OOF (would add days of runtime).
                if coach_name == gm_name:
                    coach_oof = oof_ci_gm
                    coach_fstd = fold_std_gm
                else:
                    coach_oof, coach_fstd = np.nan, np.nan

                try:
                    m = coach_fn().fit(X_tr_sc, y_train)
                    risk_tr = m.predict(X_tr_sc)
                    risk_chus = m.predict(X_chus_sc)
                    risk_chup = m.predict(X_chup_sc)
                except Exception:
                    risk_tr = np.zeros(len(X_tr_sc))
                    risk_chus = np.zeros(len(X_chus))
                    risk_chup = np.zeros(len(X_chup))

                global_row_no += 1
                all_rows.append({
                    "No": global_row_no,
                    "ranker": method,
                    "gm_model": gm_name,
                    "coach_model": coach_name,
                    "optuna_trial_no": trial.number,
                    "is_best_trial": (trial.number == bt.number),
                    "N": trial.user_attrs.get("N", len(feat_names)),
                    "n_clin": n_clin_t,
                    "n_pt": n_pt_t,
                    "n_ct": n_ct_t,
                    "n_rad": n_pt_t + n_ct_t,
                    "clin_ratio": trial.params.get("clin_ratio", np.nan),
                    "gm_oof_ci": oof_ci_gm,
                    "gm_fold_std": fold_std_gm,
                    "coach_oof_ci": coach_oof,
                    "coach_fold_std": coach_fstd,
                    "ci_train": safe_ci(y_train, risk_tr),
                    "ci_chus": safe_ci(y_chus, risk_chus),
                    "ci_chup": safe_ci(y_chup, risk_chup),
                    "gm_params": str(gm_params),
                    "coach_params": str(coach_params),
                    "clin_names": "; ".join(clin_names_t),
                    "pt_names": "; ".join(pt_names_t),
                    "ct_names": "; ".join(ct_names_t),
                    "feat_names": "; ".join(
                        clin_names_t +
                        [f"PT:{f}" for f in pt_names_t] +
                        [f"CT:{f}" for f in ct_names_t]
                    ),
                })

            if (t_idx + 1) % 200 == 0:
                print(f"    ... {t_idx + 1}/{len(completed)} trials processed", flush=True)

        print(f"  Done {method}/{gm_name}: {len(completed)} trials -> "
              f"{len(completed) * len(COACH_NAMES)} rows added")

        # Checkpoint: save intermediate results after each ranker/GM combo
        ckpt_df = pd.DataFrame(all_rows)
        ckpt_path = OUTPUT_DIR / "6_mar_SC3_rad_PT768xCT8_325_PTforced_pc_checkpoint.csv"
        ckpt_df.to_csv(ckpt_path, index=False)
        print(f"  Checkpoint saved: {len(ckpt_df)} rows so far")

# ==========================================
# 4. SAVE FINAL OUTPUT
# ==========================================
print("\n--- Saving final output ---")
results_df = pd.DataFrame(all_rows)

# Add trio metrics
results_df["trio_mean"] = (results_df["coach_oof_ci"] + results_df["ci_chus"] + results_df["ci_chup"]) / 3
results_df["trio_gap"] = (results_df[["coach_oof_ci", "ci_chus", "ci_chup"]].max(axis=1) -
                          results_df[["coach_oof_ci", "ci_chus", "ci_chup"]].min(axis=1))
results_df["chus_chup_mean"] = (results_df["ci_chus"] + results_df["ci_chup"]) / 2

out_path = OUTPUT_DIR / "6_mar_SC3_rad_PT768xCT8_325_PTforced_pc_all_results.csv"
results_df.to_csv(out_path, index=False)
print(f"Saved: {out_path.name} ({len(results_df):,} rows)")

# Top 20 by CHUS+CHUP mean (primary selection criterion)
top20_ext = results_df.nlargest(20, "chus_chup_mean")[
    ["No", "ranker", "gm_model", "coach_model", "optuna_trial_no", "is_best_trial",
     "N", "n_clin", "n_pt", "n_ct", "n_rad", "clin_ratio",
     "gm_oof_ci", "coach_oof_ci", "ci_chus", "ci_chup", "chus_chup_mean", "trio_mean", "trio_gap",
     "clin_names", "pt_names", "ct_names"]
]
print("\nTop 20 by (CHUS + CHUP) / 2:")
print(top20_ext.to_string(index=False))

top20_ext_path = OUTPUT_DIR / "6_mar_SC3_rad_PT768xCT8_325_PTforced_pc_top20_ext.csv"
top20_ext.to_csv(top20_ext_path, index=False)

# Top 20 by trio mean (OOF + CHUS + CHUP)
top20_trio = results_df.nlargest(20, "trio_mean")[
    ["No", "ranker", "gm_model", "coach_model", "optuna_trial_no", "is_best_trial",
     "N", "n_clin", "n_pt", "n_ct", "n_rad", "clin_ratio",
     "gm_oof_ci", "coach_oof_ci", "ci_chus", "ci_chup", "chus_chup_mean", "trio_mean", "trio_gap",
     "clin_names", "pt_names", "ct_names"]
]
top20_trio_path = OUTPUT_DIR / "6_mar_SC3_rad_PT768xCT8_325_PTforced_pc_top20_trio.csv"
top20_trio.to_csv(top20_trio_path, index=False)
print(f"\nSaved top20_ext and top20_trio CSVs")

# Summary stats
print(f"\nSummary:")
print(f"  Total rows: {len(results_df):,}")
print(f"  Unique feature sets: {results_df['feat_names'].nunique():,}")
print(f"  CHUS range: {results_df['ci_chus'].min():.4f} - {results_df['ci_chus'].max():.4f}")
print(f"  CHUP range: {results_df['ci_chup'].min():.4f} - {results_df['ci_chup'].max():.4f}")
print(f"  Best CHUS+CHUP mean: {results_df['chus_chup_mean'].max():.4f}")
best_row = results_df.loc[results_df['chus_chup_mean'].idxmax()]
print(f"  Best pipeline: {best_row['ranker']}/{best_row['gm_model']}/{best_row['coach_model']} "
      f"N={best_row['N']} ({best_row['n_clin']}c+{best_row['n_pt']}pt+{best_row['n_ct']}ct) "
      f"OOF={best_row['coach_oof_ci']:.4f} CHUS={best_row['ci_chus']:.4f} CHUP={best_row['ci_chup']:.4f}")
print(f"  Best features: {best_row['feat_names']}")

print("\n6_mar_SC3_rad_PT768xCT8_325_PTforced_pc finished.")
