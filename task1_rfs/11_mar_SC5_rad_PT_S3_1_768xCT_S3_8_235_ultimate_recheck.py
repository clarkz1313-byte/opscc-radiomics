"""
11_mar_SC3_rad_PT768xCT8_235_9_ultimate_recheck.py

Ultimate SC3 exhaustive tournament - convergence confirmation run.
PT768 x CT8_235 feature pools.

Purpose: Confirm that the final_8 strict winner (N=12, 1+7+4, EST coach,
CHUS=0.743, CHUP=0.728) is a structural attractor under both LOCO-based
and naive UNIVAR ranking. Also confirms 4-GM tournament fairness.

Design:
  - 5 rankers: FULL_LOCO_evt, FULL_LOCO_EPV_CUT, FULL_WLOCO_enon,
               FULL_WLOCO_epv_cut, UNIVAR (new)
  - 4 GMs: EST, SVM, GBS, CoxPH (expanded from final_8's 2)
  - 4 Coaches: EST, SVM, GBS, CoxPH (unchanged)
  - TRIALS=2000, N_BOOT=1000 per row
  - Exhaustive: all TRIALS x 4 coaches evaluated; no Stage 2 scan
  - Expected rows: 5 x 4 x 2000 x 4 = 160000

New seeds: N=12 (1+7+4) added as first structural anchor.
Old seeds: N=10 (1+8+1) retained for continuity.

Runtime: ~20 hours single machine; split across PC + Dell for parallel run.
"""

import subprocess
import sys
import time
import warnings
from pathlib import Path

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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import ExtraSurvivalTrees, GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.svm import FastSurvivalSVM
from sksurv.util import Surv

# ==========================================
# CONFIG
# ==========================================
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "11_mar_SC3_rad_PT768xCT8_235_9_ultimate_recheck_outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

ALL_CSV = OUT_DIR / "11_mar_SC3_rad_PT768xCT8_235_9_ultimate_recheck_all_results.csv"
CKPT_CSV = OUT_DIR / "11_mar_SC3_rad_PT768xCT8_235_9_ultimate_recheck_checkpoint.csv"
TOP20_STRICT_CSV = OUT_DIR / "11_mar_SC3_rad_PT768xCT8_235_9_ultimate_recheck_top20_strict.csv"
TOP20_TRIO_CSV = OUT_DIR / "11_mar_SC3_rad_PT768xCT8_235_9_ultimate_recheck_top20_trio.csv"
LOG_MD = OUT_DIR / "11_mar_SC3_rad_PT768xCT8_235_9_ultimate_recheck_log.md"

PT_DEV_FILE = ROOT / "27_feb_PT_development.csv"
CT_DEV_FILE = ROOT / "27_feb_CT_development.csv"
PT_EXT_FILE = ROOT / "27_feb_PT_external.csv"
CT_EXT_FILE = ROOT / "27_feb_CT_external.csv"
CLINICAL_FILE = ROOT.parent / "Feb_2026" / "25_feb_clinical_reduced_dataset" / "25_feb_Processed_clinical_reduced.csv"

FINALIST_DIR = ROOT / "2_mar_finalist_outputs"
PT_FEATURES_FILE = FINALIST_DIR / "PT_inter1_768_features_recheck.csv"
CT_FEATURES_FILE = FINALIST_DIR / "CT_inter8_235_features_recheck.csv"

SEED = 42
N_BOOT = 1000
TRIALS = 2000
N_FOLDS = 5
N_MIN = 6
N_MAX = 12
CLIN_RATIO_MIN = 0.05
CLIN_RATIO_MAX = 0.20
PT_MIN = 4
CT_MIN = 1
W_PERF = 0.7
W_STAB = 0.3
STD_THRESHOLD = 0.08
LOCO_LAMBDA = 0.0
LOCO_KAPPA = 5.0
EPV_CUT_EVENTS_MIN = 8
EPV_CUT_ENON_MIN = 300
EPV_CUT_RATIO_MIN = 0.0

RANKING_METHODS = [
    "FULL_LOCO_evt",
    "FULL_LOCO_EPV_CUT",
    "FULL_WLOCO_enon",
    "FULL_WLOCO_epv_cut",
    "UNIVAR",
]
GM_MODEL_NAMES = ["EST", "SVM", "GBS", "CoxPH"]
COACH_NAMES = ["EST", "SVM", "GBS", "CoxPH"]
CLINICAL_FEATURES = ["Age", "Gender_Male", "Treatment_CRT"]

# ==========================================
# LOGGING
# ==========================================
class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


_log_fh = open(LOG_MD, "w", encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, _log_fh)
sys.stderr = _Tee(sys.__stderr__, _log_fh)

# ==========================================
# HELPERS
# ==========================================
def make_surv(event, time):
    return Surv.from_arrays(event=np.asarray(event, dtype=bool), time=np.asarray(time, dtype=float))


def safe_ci(y, risk):
    try:
        return float(concordance_index_censored(y["event"], y["time"], risk)[0])
    except Exception:
        return np.nan


def bootstrap_ci_ext(y_ext, risk_ext, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y_ext)
    cis = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        c = safe_ci(y_ext[idx], risk_ext[idx])
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


def make_model(name, params):
    if name == "EST":
        return ExtraSurvivalTrees(**params, random_state=SEED, n_jobs=-1)
    if name == "SVM":
        return FastSurvivalSVM(**params, max_iter=1000, tol=1e-4, random_state=SEED)
    if name == "GBS":
        return GradientBoostingSurvivalAnalysis(**params, random_state=SEED)
    if name == "CoxPH":
        return CoxPHSurvivalAnalysis(**params)
    raise ValueError(f"Unknown model: {name}")


def model_oof(X, y, feat_idx, model_name, model_params):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    y_event = y["event"].astype(int)
    oof_pred = np.zeros(len(X), dtype=float)
    oof_ok = np.zeros(len(X), dtype=bool)
    fold_cis = []

    for tr, vl in skf.split(X, y_event):
        X_tr = X[tr][:, feat_idx]
        X_vl = X[vl][:, feat_idx]
        scaler = StandardScaler()
        try:
            m = make_model(model_name, model_params)
            m.fit(scaler.fit_transform(X_tr), y[tr])
            pred_vl = m.predict(scaler.transform(X_vl))
            oof_pred[vl] = pred_vl
            oof_ok[vl] = True
            c = safe_ci(y[vl], pred_vl)
            if not np.isnan(c):
                fold_cis.append(c)
        except Exception:
            continue

    if oof_ok.sum() < len(X) * 0.5:
        return np.nan, np.nan

    oof_ci = safe_ci(y[oof_ok], oof_pred[oof_ok])
    fold_std = float(np.std(fold_cis)) if len(fold_cis) > 0 else np.nan
    return oof_ci, fold_std


def univar_rank(X, y, feat_names):
    scores = []
    for i in range(X.shape[1]):
        c = safe_ci(y, X[:, i])
        scores.append(max(c, 1 - c) if not np.isnan(c) else 0.5)
    order = np.argsort(np.array(scores))[::-1]
    score_arr = np.array(scores)[order]
    return [feat_names[i] for i in order], score_arr


def split_ranked_pt_ct(ranked_features, pt_set, ct_set):
    pt_ranked = [f for f in ranked_features if f in pt_set]
    ct_ranked = [f for f in ranked_features if f in ct_set]
    return pt_ranked, ct_ranked


def loco_rank_mode(X_full, y_full, center_ids, feat_names, mode):
    centres = np.unique(center_ids)
    n_feats = X_full.shape[1]

    ci_matrix = np.full((len(centres), n_feats), np.nan)
    centre_weights = np.zeros(len(centres), dtype=float)

    for fold_i, held_centre in enumerate(centres):
        mask_val = center_ids == held_centre
        n_c = int(mask_val.sum())
        if n_c < 5:
            continue
        e_c = int(y_full["event"][mask_val].sum())
        ne_c = n_c - e_c
        epv_ratio = (e_c / n_c) if n_c > 0 else 0.0
        enon = float(e_c * ne_c)

        include = True
        use_shrinkage = False

        if mode == "loco_evt":
            include = e_c >= 10
            w_c = 1.0
            use_shrinkage = False
        elif mode == "loco_epv_cut":
            include = (
                e_c >= EPV_CUT_EVENTS_MIN
                and enon >= EPV_CUT_ENON_MIN
                and epv_ratio >= EPV_CUT_RATIO_MIN
            )
            w_c = 1.0
            use_shrinkage = False
        elif mode == "w_enon":
            w_c = max(1.0, enon)
            use_shrinkage = True
        elif mode == "w_epv_cut":
            include = (
                e_c >= EPV_CUT_EVENTS_MIN
                and enon >= EPV_CUT_ENON_MIN
                and epv_ratio >= EPV_CUT_RATIO_MIN
            )
            w_c = 1.0
            use_shrinkage = True
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if not include:
            continue

        centre_weights[fold_i] = w_c
        X_val = X_full[mask_val]
        y_val = y_full[mask_val]

        for fi in range(n_feats):
            c = safe_ci(y_val, X_val[:, fi])
            c = max(c, 1 - c) if not np.isnan(c) else 0.5
            if use_shrinkage:
                c = (e_c * c + LOCO_KAPPA * 0.5) / (e_c + LOCO_KAPPA)
            ci_matrix[fold_i, fi] = c

    valid = centre_weights > 0
    if not np.any(valid):
        return list(feat_names), np.full(len(feat_names), 0.5)

    M = np.where(np.isnan(ci_matrix[valid, :]), 0.5, ci_matrix[valid, :])
    W = centre_weights[valid]

    if mode in ("w_enon", "w_epv_cut"):
        mean_w = np.sum(M * W[:, None], axis=0) / float(np.sum(W))
        std_w = np.sqrt(np.sum(W[:, None] * (M - mean_w) ** 2, axis=0) / float(np.sum(W)))
        score = mean_w - LOCO_LAMBDA * std_w
    else:
        score = np.mean(M, axis=0)

    order = np.argsort(score)[::-1]
    return [feat_names[i] for i in order], score[order]


def enqueue_structural_seeds(study, gm_name):
    structures = [
        (12, 0.083, 7),
        (10, 0.10, 8),
        (10, 0.10, 7),
        (9, 0.111, 7),
        (11, 0.091, 9),
        (6, 0.167, 4),
    ]
    alpha_vals_svm = [0.001, 0.01, 0.05, 0.1, 0.3]
    alpha_vals_cox = [0.01, 0.1, 1.0, 3.0, 10.0]

    for n, c_ratio, n_pt in structures:
        n_clin = max(1, round(n * c_ratio))
        n_ct = (n - n_clin) - n_pt
        if n_ct < CT_MIN:
            continue

        seed_payload = {
            "total_n": int(n),
            "clin_ratio": float(c_ratio),
            "n_pt": int(n_pt),
        }

        if gm_name == "SVM":
            for alpha in alpha_vals_svm:
                payload = dict(seed_payload)
                payload["alpha"] = float(alpha)
                study.enqueue_trial(payload)
        elif gm_name == "CoxPH":
            for alpha in alpha_vals_cox:
                payload = dict(seed_payload)
                payload["alpha"] = float(alpha)
                study.enqueue_trial(payload)
        else:
            study.enqueue_trial(seed_payload)


# ==========================================
# LOAD DATA
# ==========================================
print("=" * 80)
print("11_mar_SC3_rad_PT768xCT8_235_9_ultimate_recheck")
print("Ultimate SC3 exhaustive tournament")
print(
    f"TRIALS={TRIALS} | N_BOOT={N_BOOT} | Rankers={len(RANKING_METHODS)} | "
    f"GMs={len(GM_MODEL_NAMES)} | Coaches={len(COACH_NAMES)}"
)
print(f"Expected rows: {TRIALS * len(RANKING_METHODS) * len(GM_MODEL_NAMES) * len(COACH_NAMES)}")
print(
    f"N=[{N_MIN},{N_MAX}] | clin_ratio=[{CLIN_RATIO_MIN},{CLIN_RATIO_MAX}] | "
    f"PT_MIN={PT_MIN} | CT_MIN={CT_MIN}"
)
print("=" * 80)

clinical = pd.read_csv(CLINICAL_FILE).dropna(subset=["Relapse", "RFS"])

clin_dev = clinical[clinical["Cohort"] == "Dev"][["PatientID", "CenterID", "Relapse", "RFS"] + CLINICAL_FEATURES].copy()
clin_chus = clinical[clinical["CenterID"] == 3][["PatientID", "Relapse", "RFS"] + CLINICAL_FEATURES].copy()
clin_chup = clinical[clinical["CenterID"] == 2][["PatientID", "Relapse", "RFS"] + CLINICAL_FEATURES].copy()

pt_feat = pd.read_csv(PT_FEATURES_FILE)["Feature"].tolist()
ct_raw = pd.read_csv(CT_FEATURES_FILE)["Feature"].tolist()
pt_set = set(pt_feat)
ct_feat = [f for f in ct_raw if f not in pt_set]
ct_set = set(ct_feat)

pt_dev = pd.read_csv(PT_DEV_FILE)
ct_dev = pd.read_csv(CT_DEV_FILE)
pt_ext = pd.read_csv(PT_EXT_FILE)
ct_ext = pd.read_csv(CT_EXT_FILE)

rad_dev = pt_dev[["PatientID"] + pt_feat].merge(ct_dev[["PatientID"] + ct_feat], on="PatientID", how="inner")
rad_ext = pt_ext[["PatientID"] + pt_feat].merge(ct_ext[["PatientID"] + ct_feat], on="PatientID", how="inner")
rad_chus = rad_ext[rad_ext["PatientID"].str.startswith("CHUS")]
rad_chup = rad_ext[rad_ext["PatientID"].str.startswith("CHUP")]

all_features = CLINICAL_FEATURES + pt_feat + ct_feat

dev_df = clin_dev.merge(rad_dev, on="PatientID", how="inner")
chus_df = clin_chus.merge(rad_chus, on="PatientID", how="inner")
chup_df = clin_chup.merge(rad_chup, on="PatientID", how="inner")

print(
    f"Dev={len(dev_df)} ({int(dev_df['Relapse'].sum())} events) | "
    f"CHUS={len(chus_df)} ({int(chus_df['Relapse'].sum())} events) | "
    f"CHUP={len(chup_df)} ({int(chup_df['Relapse'].sum())} events)"
)

X_train = dev_df[all_features].values.astype(float)
y_train = make_surv(dev_df["Relapse"], dev_df["RFS"])
X_chus = chus_df[all_features].values.astype(float)
y_chus = make_surv(chus_df["Relapse"], chus_df["RFS"])
X_chup = chup_df[all_features].values.astype(float)
y_chup = make_surv(chup_df["Relapse"], chup_df["RFS"])

feat_idx_map = {f: i for i, f in enumerate(all_features)}
clin_idx = [feat_idx_map[f] for f in CLINICAL_FEATURES]
pt_idx = [feat_idx_map[f] for f in pt_feat]
ct_idx = [feat_idx_map[f] for f in ct_feat]
dev_center_ids = dev_df["CenterID"].values

# ==========================================
# PRECOMPUTE RANKINGS
# ==========================================
print("\n--- Pre-computing rankers ---")
sc_global = StandardScaler()
X_train_sc = sc_global.fit_transform(X_train)

clin_ranked, _ = univar_rank(X_train_sc[:, clin_idx], y_train, CLINICAL_FEATURES)

ranker_mode_map = {
    "FULL_LOCO_evt": "loco_evt",
    "FULL_LOCO_EPV_CUT": "loco_epv_cut",
    "FULL_WLOCO_enon": "w_enon",
    "FULL_WLOCO_epv_cut": "w_epv_cut",
}

rad_names = pt_feat + ct_feat
rad_idx = pt_idx + ct_idx
rankings = {}
for rk in RANKING_METHODS:
    if rk == "UNIVAR":
        pt_ranked, _ = univar_rank(X_train_sc[:, pt_idx], y_train, pt_feat)
        ct_ranked, _ = univar_rank(X_train_sc[:, ct_idx], y_train, ct_feat)
    else:
        mode = ranker_mode_map[rk]
        ranked_rad, _ = loco_rank_mode(X_train_sc[:, rad_idx], y_train, dev_center_ids, rad_names, mode)
        pt_ranked, ct_ranked = split_ranked_pt_ct(ranked_rad, pt_set, ct_set)

    rankings[rk] = {"clin": clin_ranked, "pt": pt_ranked, "ct": ct_ranked}
    print(f"  {rk}: pt={len(pt_ranked)}, ct={len(ct_ranked)}")

# ==========================================
# EXHAUSTIVE OPTUNA
# ==========================================
print("\n--- Exhaustive search (all trials x all coaches) ---")
all_rows = []
trial_counter = 0

for ranker in RANKING_METHODS:
    r_clin = rankings[ranker]["clin"]
    r_pt = rankings[ranker]["pt"]
    r_ct = rankings[ranker]["ct"]
    r_clin_idx = [feat_idx_map[f] for f in r_clin]
    r_pt_idx = [feat_idx_map[f] for f in r_pt]
    r_ct_idx = [feat_idx_map[f] for f in r_ct]

    for gm in GM_MODEL_NAMES:
        block_start = time.time()
        block_row_start = len(all_rows)

        print(f"\nRanker={ranker} | GM={gm} | Running {TRIALS} trials...")
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
        enqueue_structural_seeds(study, gm)

        def objective(trial):
            N = trial.suggest_int("total_n", N_MIN, N_MAX)
            clin_ratio = trial.suggest_float("clin_ratio", CLIN_RATIO_MIN, CLIN_RATIO_MAX)

            n_clin = max(1, round(N * clin_ratio))
            n_clin = min(n_clin, len(r_clin_idx), N - PT_MIN - CT_MIN)
            n_rad = N - n_clin

            n_pt_max = min(n_rad - CT_MIN, len(r_pt_idx))
            n_pt_min = min(PT_MIN, n_pt_max)
            if n_pt_min > n_pt_max:
                n_pt_min = n_pt_max
            n_pt = trial.suggest_int("n_pt", n_pt_min, n_pt_max)

            n_ct = n_rad - n_pt
            if n_ct < CT_MIN:
                n_ct = CT_MIN
                n_pt = n_rad - n_ct
            n_ct = min(n_ct, len(r_ct_idx))
            n_pt = min(max(n_pt, PT_MIN), len(r_pt_idx))

            if gm == "SVM":
                alpha = trial.suggest_float("alpha", 1e-4, 1.0, log=True)
                gm_params = {"alpha": alpha}
            elif gm == "CoxPH":
                alpha = trial.suggest_float("alpha", 1e-3, 10.0, log=True)
                gm_params = {"alpha": alpha}
            elif gm == "EST":
                alpha = np.nan
                gm_params = {"n_estimators": 200}
            elif gm == "GBS":
                alpha = np.nan
                gm_params = {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 3,
                    "subsample": 0.8,
                }
            else:
                alpha = np.nan
                gm_params = {}

            feat_idx = r_clin_idx[:n_clin] + r_pt_idx[:n_pt] + r_ct_idx[:n_ct]

            oof_ci, fold_std = model_oof(X_train, y_train, feat_idx, gm, gm_params)
            if np.isnan(oof_ci):
                oof_ci = 0.5
            if np.isnan(fold_std):
                fold_std = STD_THRESHOLD

            score = W_PERF * oof_ci + W_STAB * max(0.0, 1.0 - fold_std / STD_THRESHOLD)

            trial.set_user_attr("N", int(N))
            trial.set_user_attr("n_clin", int(n_clin))
            trial.set_user_attr("n_pt", int(n_pt))
            trial.set_user_attr("n_ct", int(n_ct))
            trial.set_user_attr("alpha", float(alpha) if not np.isnan(alpha) else np.nan)
            trial.set_user_attr("oof", float(oof_ci))
            trial.set_user_attr("fold_std", float(fold_std))
            trial.set_user_attr("feat_clin", [all_features[i] for i in r_clin_idx[:n_clin]])
            trial.set_user_attr("feat_pt", [all_features[i] for i in r_pt_idx[:n_pt]])
            trial.set_user_attr("feat_ct", [all_features[i] for i in r_ct_idx[:n_ct]])

            if (trial.number + 1) % 100 == 0:
                print(f"  Trial {trial.number + 1}/{TRIALS} | OOF={oof_ci:.4f}")

            return score

        study.optimize(objective, n_trials=TRIALS, show_progress_bar=False)

        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        print(f"  Evaluating {len(completed)} completed trials x {len(COACH_NAMES)} coaches...")

        for t in completed:
            N = int(t.user_attrs["N"])
            n_clin = int(t.user_attrs["n_clin"])
            n_pt = int(t.user_attrs["n_pt"])
            n_ct = int(t.user_attrs["n_ct"])
            alpha = t.user_attrs["alpha"]
            alpha = float(alpha) if alpha is not None and not np.isnan(alpha) else np.nan
            oof = float(t.user_attrs["oof"])
            feat_clin = list(t.user_attrs["feat_clin"])
            feat_pt = list(t.user_attrs["feat_pt"])
            feat_ct = list(t.user_attrs["feat_ct"])

            feat_all = feat_clin + feat_pt + feat_ct
            col_idx = [feat_idx_map[f] for f in feat_all]

            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_train[:, col_idx])
            X_chus_sc = scaler.transform(X_chus[:, col_idx])
            X_chup_sc = scaler.transform(X_chup[:, col_idx])

            for coach in COACH_NAMES:
                trial_counter += 1

                if coach == "EST":
                    coach_params = {"n_estimators": 200}
                elif coach == "GBS":
                    coach_params = {
                        "n_estimators": 100,
                        "learning_rate": 0.1,
                        "max_depth": 3,
                        "subsample": 0.8,
                    }
                elif coach == "SVM":
                    coach_alpha = alpha if gm == "SVM" and not np.isnan(alpha) else 0.01
                    coach_params = {"alpha": float(coach_alpha)}
                elif coach == "CoxPH":
                    coach_alpha = alpha if gm == "CoxPH" and not np.isnan(alpha) else 0.1
                    coach_params = {"alpha": float(coach_alpha)}
                else:
                    coach_params = {}

                try:
                    model = make_model(coach, coach_params)
                    model.fit(X_tr_sc, y_train)
                    risk_chus = model.predict(X_chus_sc)
                    risk_chup = model.predict(X_chup_sc)
                except Exception:
                    risk_chus = np.full(len(X_chus_sc), np.nan)
                    risk_chup = np.full(len(X_chup_sc), np.nan)

                ci_chus = safe_ci(y_chus, risk_chus)
                ci_chup = safe_ci(y_chup, risk_chup)

                b_chus = bootstrap_ci_ext(y_chus, risk_chus, n_boot=N_BOOT, seed=SEED)
                b_chup = bootstrap_ci_ext(y_chup, risk_chup, n_boot=N_BOOT, seed=SEED)
                boot_n_valid = int(min(b_chus["n"], b_chup["n"]))

                row = {
                    "trial_no": trial_counter,
                    "ranker": ranker,
                    "gm": gm,
                    "coach": coach,
                    "N": N,
                    "n_clin": n_clin,
                    "n_pt": n_pt,
                    "n_ct": n_ct,
                    "alpha": alpha,
                    "gm_params": str(gm_params),
                    "coach_params": str(coach_params),
                    "feat_clin": "|".join(feat_clin),
                    "feat_pt": "|".join(feat_pt),
                    "feat_ct": "|".join(feat_ct),
                    "oof": oof,
                    "ci_chus": ci_chus,
                    "ci_chup": ci_chup,
                    "dual_ok": int((ci_chus >= 0.70) and (ci_chup >= 0.70)),
                    "boot_ci_chus_mean": b_chus["mean"],
                    "boot_ci_chus_std": b_chus["std"],
                    "boot_ci_chus_lo95": b_chus["lo"],
                    "boot_ci_chus_hi95": b_chus["hi"],
                    "boot_ci_chup_mean": b_chup["mean"],
                    "boot_ci_chup_std": b_chup["std"],
                    "boot_ci_chup_lo95": b_chup["lo"],
                    "boot_ci_chup_hi95": b_chup["hi"],
                    "boot_n_valid": boot_n_valid,
                }
                all_rows.append(row)

        block_df = pd.DataFrame(all_rows[block_row_start:])
        n_strict = int(((block_df["oof"] > 0.68) & (block_df["ci_chus"] >= 0.70) & (block_df["ci_chup"] >= 0.70)).sum()) if len(block_df) else 0
        n_near = int(((block_df["oof"] > 0.68) & (block_df["ci_chus"] >= 0.70) & (block_df["ci_chup"] >= 0.68)).sum()) if len(block_df) else 0
        best_chus = float(block_df["ci_chus"].max()) if len(block_df) else np.nan
        best_chup = float(block_df["ci_chup"].max()) if len(block_df) else np.nan

        ckpt_df = pd.DataFrame(all_rows)
        ckpt_df.to_csv(CKPT_CSV, index=False)

        elapsed = time.time() - block_start
        print(
            f"  Block summary | strict={n_strict} | near={n_near} | "
            f"best CHUS={best_chus:.4f} | best CHUP={best_chup:.4f} | elapsed={elapsed/60.0:.1f} min"
        )
        print(f"  Checkpoint saved: {CKPT_CSV.name} ({len(ckpt_df)} rows)")

# ==========================================
# SAVE FINAL OUTPUTS
# ==========================================
print("\n--- Saving final outputs ---")
all_df = pd.DataFrame(all_rows)
all_df.to_csv(ALL_CSV, index=False)

strict_df = all_df[all_df["dual_ok"] == 1].copy()
if len(strict_df) > 0:
    strict_df["chus_chup_sum"] = strict_df["ci_chus"] + strict_df["ci_chup"]
    top20_strict = strict_df.sort_values(["chus_chup_sum", "oof"], ascending=[False, False]).head(20)
else:
    top20_strict = strict_df

top20_strict.to_csv(TOP20_STRICT_CSV, index=False)

all_df["trio_mean"] = (all_df["oof"] + all_df["ci_chus"] + all_df["ci_chup"]) / 3.0
top20_trio = all_df.sort_values(["trio_mean", "oof"], ascending=[False, False]).head(20)
top20_trio.to_csv(TOP20_TRIO_CSV, index=False)

print(f"Saved all results: {ALL_CSV} ({len(all_df)} rows)")
print(f"Saved checkpoint: {CKPT_CSV}")
print(f"Saved top20 strict: {TOP20_STRICT_CSV}")
print(f"Saved top20 trio: {TOP20_TRIO_CSV}")
print("\n11_mar_SC3_rad_PT768xCT8_235_9_ultimate_recheck finished.")

