"""
LOCO-DEV voting: compare standard (all centres) vs events-threshold (>=10 events) ranking.
Reports where No.244 key features land and top PT/CT in each system.
Pre-filtering: reduce pool to top-K by a criterion so No.244 mean rank improves (fewer competitors).
Uses same data loading as SC3 scripts: clinical dev + PT/CT dev feature matrices.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

def safe_ci(relapse_arr, rfs_arr, pred):
    try:
        c = concordance_index_censored(
            relapse_arr.astype(bool), rfs_arr.astype(float), pred
        )[0]
        return max(c, 1 - c)
    except Exception:
        return np.nan

def main():
    ROOT = Path(__file__).resolve().parent
    FINALIST_DIR = ROOT / "2_mar_finalist_outputs"
    PT_FEATURES_FILE = FINALIST_DIR / "PT_inter1_768_features.csv"
    CT_FEATURES_FILE = FINALIST_DIR / "CT_inter8_325_features.csv"
    PT_DEV_FILE = ROOT / "27_feb_PT_development.csv"
    CT_DEV_FILE = ROOT / "27_feb_CT_development.csv"
    CLINICAL_FILE = ROOT.parent / "Feb_2026" / "25_feb_clinical_reduced_dataset" / "25_feb_Processed_clinical_reduced.csv"

    pt_feat_list = pd.read_csv(PT_FEATURES_FILE)["Feature"].tolist()
    ct_feat_list_raw = pd.read_csv(CT_FEATURES_FILE)["Feature"].tolist()
    pt_feat_set = set(pt_feat_list)
    ct_feat_list = [f for f in ct_feat_list_raw if f not in pt_feat_set]
    rad_features = pt_feat_list + ct_feat_list
    print(f"PT features: {len(pt_feat_list)}, CT after dedup: {len(ct_feat_list)}, total rad: {len(rad_features)}")

    clinical = pd.read_csv(CLINICAL_FILE).dropna(subset=["Relapse", "RFS"])
    clin_dev = clinical[clinical["Cohort"] == "Dev"][
        ["PatientID", "CenterID", "Relapse", "RFS"]
    ].copy()
    pt_dev = pd.read_csv(PT_DEV_FILE)[["PatientID"] + pt_feat_list]
    ct_dev = pd.read_csv(CT_DEV_FILE)[["PatientID"] + ct_feat_list]
    rad_dev = pt_dev.merge(ct_dev, on="PatientID", how="inner")
    m = clin_dev.merge(rad_dev, on="PatientID", how="inner").dropna(
        subset=rad_features + ["RFS", "Relapse"]
    )
    print(f"Dev after merge: N={len(m)}")

    def make_y(relapse, rfs):
        return np.array(
            [(bool(r), float(t)) for r, t in zip(relapse, rfs)],
            dtype=[("Relapse", bool), ("RFS", float)],
        )

    pt_cols = pt_feat_list
    ct_cols_clean = ct_feat_list
    all_rad = rad_features

    centres = sorted(m["CenterID"].unique())
    centre_n_events = {c: int(m[m["CenterID"] == c]["Relapse"].sum()) for c in centres}
    print("Centre event counts in merged dev:", centre_n_events, "\n")

    pt244 = [
        "GTVn_wavelet-HLH_glszm_GrayLevelVariance",
        "GTVp_wavelet-HHL_glcm_ClusterProminence",
        "GTVp_wavelet-HLH_glrlm_ShortRunHighGrayLevelEmphasis",
        "GTVn_square_glszm_GrayLevelNonUniformity",
    ]
    ct244 = ["GTVp_wavelet-LLH_glrlm_HighGrayLevelRunEmphasis"]
    all244 = pt244 + ct244

    def compute_loco_ranking(m, all_rad, centres, centre_n_events, min_events=0, label="", silent=False):
        eligible = [c for c in centres if centre_n_events.get(c, 0) >= min_events]
        if not silent:
            print(f"\n--- {label} ---")
            print(f"  Min events threshold: {min_events}")
            print(f"  Eligible centres: {[(c, centre_n_events[c]) for c in eligible]}")

        X_all = m[all_rad].values.astype(float)
        rel = m["Relapse"].values
        rfs_v = m["RFS"].values
        cids = m["CenterID"].values
        y_all = make_y(rel, rfs_v)

        scores = {}
        for feat_idx, feat in enumerate(all_rad):
            X1 = X_all[:, feat_idx : feat_idx + 1]
            loco_cis = []
            for c in eligible:
                mask_v = cids == c
                mask_t = ~mask_v
                if mask_v.sum() < 3:
                    continue
                sc = StandardScaler()
                Xtr = sc.fit_transform(X1[mask_t])
                Xvl = sc.transform(X1[mask_v])
                for alpha in [0.1, 1.0, 0.01]:
                    try:
                        model = CoxPHSurvivalAnalysis(alpha=alpha).fit(
                            Xtr, y_all[mask_t]
                        )
                        pred = model.predict(Xvl)
                        ci = safe_ci(rel[mask_v], rfs_v[mask_v], pred)
                        if not np.isnan(ci):
                            loco_cis.append(ci)
                            break
                    except Exception:
                        continue
            scores[feat] = np.mean(loco_cis) if loco_cis else np.nan

        ranked = sorted(
            scores.items(), key=lambda x: -(x[1] if not np.isnan(x[1]) else 0)
        )
        return ranked, scores

    ranked_std, scores_std = compute_loco_ranking(
        m, all_rad, centres, centre_n_events, min_events=1,
        label="STANDARD LOCO-DEV (all centres)",
    )
    ranked_evt, scores_evt = compute_loco_ranking(
        m, all_rad, centres, centre_n_events, min_events=10,
        label="EVENTS-THRESHOLD LOCO-DEV (>=10 events)",
    )

    # Univariate C-index on full dev (no LOCO) for pre-filter option
    def univar_scores_full_dev(m, all_rad):
        X = m[all_rad].values.astype(float)
        rel = m["Relapse"].values
        rfs_v = m["RFS"].values
        y_all = make_y(rel, rfs_v)
        sc = StandardScaler()
        X_sc = sc.fit_transform(X)
        scores = {}
        for i, feat in enumerate(all_rad):
            for alpha in [0.1, 1.0, 0.01]:
                try:
                    model = CoxPHSurvivalAnalysis(alpha=alpha).fit(
                        X_sc[:, i : i + 1], y_all
                    )
                    pred = model.predict(X_sc[:, i : i + 1])
                    ci = safe_ci(rel, rfs_v, pred)
                    if not np.isnan(ci):
                        scores[feat] = ci
                        break
                except Exception:
                    continue
            if feat not in scores:
                scores[feat] = np.nan
        ranked = sorted(
            scores.items(), key=lambda x: -(x[1] if not np.isnan(x[1]) else 0)
        )
        return ranked, scores

    print("\n--- UNIVAR (full dev, for pre-filter) ---")
    ranked_univar, scores_univar = univar_scores_full_dev(m, all_rad)

    # Variance threshold: how many features survive?
    X_rad = m[all_rad].values.astype(float)
    vt = VarianceThreshold(threshold=1e-5)
    vt.fit(X_rad)
    kept_mask = vt.get_support()
    all_rad_var = [all_rad[i] for i in range(len(all_rad)) if kept_mask[i]]
    n_dropped_var = len(all_rad) - len(all_rad_var)
    print(f"  VarianceThreshold(1e-5): {len(all_rad)} -> {len(all_rad_var)} features (dropped {n_dropped_var})")
    no244_in_var = sum(1 for f in all244 if f in all_rad_var)
    print(f"  No.244 in variance-kept pool: {no244_in_var}/5")

    def mean_rank_no244_in_pool(ranked_list, no244_set):
        """ranked_list = [(feat, score), ...] in rank order. Return mean 1-based rank of no244."""
        ranks = []
        for r, (f, _) in enumerate(ranked_list, 1):
            if f in no244_set:
                ranks.append(r)
        return np.mean(ranks) if ranks else np.nan, len(ranks)

    no244_set = set(all244)
    print("\n" + "=" * 80)
    print("PRE-FILTER: top-K reduces pool -> No.244 mean rank within pool (lower = better)")
    print("=" * 80)
    for K in [40, 30, 25, 20]:
        pool_std = [f for f, _ in ranked_std[:K]]
        pool_evt = [f for f, _ in ranked_evt[:K]]
        pool_univar = [f for f, _ in ranked_univar[:K]]
        mean_std, n_std = mean_rank_no244_in_pool(ranked_std[:K], no244_set)
        mean_evt, n_evt = mean_rank_no244_in_pool(ranked_evt[:K], no244_set)
        mean_uni, n_uni = mean_rank_no244_in_pool(ranked_univar[:K], no244_set)
        print(f"  K={K}: LOCO_std  n_no244={n_std}/5  mean_rank={mean_std:.1f}  |  "
              f"LOCO_evt  n_no244={n_evt}/5  mean_rank={mean_evt:.1f}  |  "
              f"UNIVAR    n_no244={n_uni}/5  mean_rank={mean_uni:.1f}")
    print("  (If n_no244<5, some No.244 features were dropped by the top-K filter.)")

    def rank_of_feature_in_list(ranked_list, feat):
        """1-based rank of feat in ranked_list, or None if not in list."""
        for r, (f, _) in enumerate(ranked_list, 1):
            if f == feat:
                return r
        return None

    print("\n" + "-" * 80)
    print("No.244 FEATURE RANKS *AFTER* PRE-FILTER (within filtered pool)")
    print("-" * 80)
    for label, ranked_list, score_dict, K in [
        ("UNIVAR top-40", ranked_univar, scores_univar, 40),
        ("UNIVAR top-30", ranked_univar, scores_univar, 30),
        ("LOCO_std top-40", ranked_std, scores_std, 40),
        ("LOCO_evt top-40", ranked_evt, scores_evt, 40),
    ]:
        pool = ranked_list[:K]
        pool_size = len(pool)
        print(f"\n  [{label}] pool size = {pool_size}")
        print(f"  {'No.244 feature':<52} {'rank':>8} {'score':>8}")
        print("  " + "-" * 70)
        for feat in all244:
            r = rank_of_feature_in_list(pool, feat)
            score = score_dict.get(feat, np.nan)
            if r is not None:
                print(f"  {feat[-50:]:<52} {r:>5}/{pool_size}  {score:>8.4f}")
            else:
                print(f"  {feat[-50:]:<52}  dropped  {score:>8.4f}")
        in_pool = sum(1 for f in all244 if rank_of_feature_in_list(pool, f) is not None)
        mean_r = mean_rank_no244_in_pool(pool, no244_set)[0]
        print(f"  -> No.244 in pool: {in_pool}/5, mean rank = {mean_r:.1f}")

    # Pre-filter by UNIVAR top-30, then RE-RANK that pool by LOCO (std and evt)
    pool_uni30 = [f for f, _ in ranked_univar[:30]]
    ranked_std_on30, scores_std_on30 = compute_loco_ranking(
        m, pool_uni30, centres, centre_n_events, min_events=1,
        label="LOCO_std on UNIVAR-top-30 pool (silent)", silent=True,
    )
    ranked_evt_on30, scores_evt_on30 = compute_loco_ranking(
        m, pool_uni30, centres, centre_n_events, min_events=10,
        label="LOCO_evt on UNIVAR-top-30 pool (silent)", silent=True,
    )
    print("\n" + "-" * 80)
    print("PRE-FILTER UNIVAR top-30, THEN re-rank by LOCO (ranks are LOCO within pool of 30)")
    print("-" * 80)
    for label, ranked_on30, score_dict in [
        ("UNIVAR top-30 -> LOCO_std", ranked_std_on30, scores_std_on30),
        ("UNIVAR top-30 -> LOCO_evt", ranked_evt_on30, scores_evt_on30),
    ]:
        pool_size = len(ranked_on30)
        print(f"\n  [{label}] pool size = {pool_size}")
        print(f"  {'No.244 feature':<52} {'rank':>8} {'score':>8}")
        print("  " + "-" * 70)
        for feat in all244:
            r = rank_of_feature_in_list(ranked_on30, feat)
            score = score_dict.get(feat, np.nan)
            if r is not None:
                print(f"  {feat[-50:]:<52} {r:>5}/{pool_size}  {score:>8.4f}")
            else:
                print(f"  {feat[-50:]:<52}  (not in pool)  {score:>8.4f}")
        in_pool = sum(1 for f in all244 if rank_of_feature_in_list(ranked_on30, f) is not None)
        mean_r = mean_rank_no244_in_pool(ranked_on30, no244_set)[0]
        print(f"  -> No.244 in pool: {in_pool}/5, mean rank (LOCO) = {mean_r:.1f}")

    # After variance filter: take top-K from *original* LOCO ranking but only among variance-kept
    if len(all_rad_var) < len(all_rad):
        var_set = set(all_rad_var)
        ranked_std_var_only = [(f, s) for f, s in ranked_std if f in var_set]
        for K in [30, 25, 20]:
            mean_v, n_v = mean_rank_no244_in_pool(
                ranked_std_var_only[:K], no244_set
            )
            print(f"  Var+LOCO_std K={K}: n_pool={min(K,len(ranked_std_var_only))}, n_no244={n_v}/5, mean_rank={mean_v:.1f}")
    print()

    print("\n" + "=" * 80)
    print("No.244 FEATURE RANKS IN EACH RANKING SYSTEM")
    print("=" * 80)

    pt_ranked_std = [(f, v) for f, v in ranked_std if f in pt_cols]
    ct_ranked_std = [(f, v) for f, v in ranked_std if f in ct_cols_clean]
    pt_ranked_evt = [(f, v) for f, v in ranked_evt if f in pt_cols]
    ct_ranked_evt = [(f, v) for f, v in ranked_evt if f in ct_cols_clean]

    pt_names_std = [f for f, _ in pt_ranked_std]
    ct_names_std = [f for f, _ in ct_ranked_std]
    pt_names_evt = [f for f, _ in pt_ranked_evt]
    ct_names_evt = [f for f, _ in ct_ranked_evt]

    print(f"\n{'Feature':<55} {'Std Score':>10} {'Std Rank(PT)':>13} {'Evt Score':>10} {'Evt Rank(PT)':>13} {'Change':>8}")
    print("-" * 110)
    for feat in pt244:
        s_std = scores_std.get(feat, np.nan)
        s_evt = scores_evt.get(feat, np.nan)
        r_std = pt_names_std.index(feat) + 1 if feat in pt_names_std else None
        r_evt = pt_names_evt.index(feat) + 1 if feat in pt_names_evt else None
        change = f"{r_std - r_evt:+d}" if isinstance(r_std, int) and isinstance(r_evt, int) else "N/A"
        short = feat[-50:]
        print(f"...{short:<52} {s_std:>10.4f} {str(r_std)+'/'+str(len(pt_names_std)):>13} {s_evt:>10.4f} {str(r_evt)+'/'+str(len(pt_names_evt)):>13} {change:>8}")

    print(f"\n{'CT Feature':<55} {'Std Score':>10} {'Std Rank(CT)':>13} {'Evt Score':>10} {'Evt Rank(CT)':>13} {'Change':>8}")
    print("-" * 110)
    for feat in ct244:
        s_std = scores_std.get(feat, np.nan)
        s_evt = scores_evt.get(feat, np.nan)
        r_std = ct_names_std.index(feat) + 1 if feat in ct_names_std else None
        r_evt = ct_names_evt.index(feat) + 1 if feat in ct_names_evt else None
        change = f"{r_std - r_evt:+d}" if isinstance(r_std, int) and isinstance(r_evt, int) else "N/A"
        short = feat[-50:]
        print(f"...{short:<52} {s_std:>10.4f} {str(r_std)+'/'+str(len(ct_names_std)):>13} {s_evt:>10.4f} {str(r_evt)+'/'+str(len(ct_names_evt)):>13} {change:>8}")

    print("\n=== TOP 15 PT features (Standard LOCO, all centres) ===")
    for i, (f, v) in enumerate(pt_ranked_std[:15], 1):
        marker = " <-- No.244" if f in pt244 else ""
        print(f"  {i:2d}. {v:.4f}  {f}{marker}")

    print("\n=== TOP 15 PT features (Events-Threshold LOCO, >=10 events) ===")
    for i, (f, v) in enumerate(pt_ranked_evt[:15], 1):
        marker = " <-- No.244" if f in pt244 else ""
        print(f"  {i:2d}. {v:.4f}  {f}{marker}")

    print("\n=== TOP 20 CT features (Standard LOCO) ===")
    for i, (f, v) in enumerate(ct_ranked_std[:20], 1):
        marker = " <-- No.244" if f in ct244 else ""
        print(f"  {i:2d}. {v:.4f}  {f}{marker}")

    print("\n=== TOP 20 CT features (Events-Threshold LOCO) ===")
    for i, (f, v) in enumerate(ct_ranked_evt[:20], 1):
        marker = " <-- No.244" if f in ct244 else ""
        print(f"  {i:2d}. {v:.4f}  {f}{marker}")

    return ranked_std, ranked_evt, scores_std, scores_evt

if __name__ == "__main__":
    main()
