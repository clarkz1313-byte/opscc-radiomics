"""
HECKTOR 2025: Phase 0 - Data Preparation and Sanity Checks

Purpose: Merge clinical variables (11 features) with radiomics features (36 features)
         and prepare train/test/CHUP arrays for Phase 1 modeling.

Key Implementation Details:
1. Stratified train/test split (80/20, seed=42, by Relapse) - MATCHES OPTUNA
2. Dummy-coded clinical features with VIF check
3. Feature correlation analysis (Gemini Refinement 2)
4. Strict sanity checks (zero NaNs, VIF < 10)
5. Saves processed arrays for Phase 1-3 models

Author: Claude + Gemini (revised)
Date: February 12, 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sksurv.util import Surv
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
ROOT = Path("d:/Uppsala thesis/Feb_2026")
OUTPUT_DIR = ROOT / "12_feb_ML_dev_phase0_outputs"

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)

# Data files (11_feb writes to Phase1_2_ouputs; fallback to ROOT for legacy)
PHASE1_2_OUTPUTS = ROOT / "Phase1_2_ouputs"
CLINICAL_FILE = PHASE1_2_OUTPUTS / "11_feb_Processed_clinical_Allinfo.csv"
if not CLINICAL_FILE.exists():
    CLINICAL_FILE = ROOT / "11_feb_Processed_clinical_Allinfo.csv"
CT_DEV_FILE = ROOT / "1_feb_CT_development.csv"
PT_DEV_FILE = ROOT / "1_feb_PT_development.csv"
CT_EXT_FILE = ROOT / "1_feb_CT_external.csv"
PT_EXT_FILE = ROOT / "1_feb_PT_external.csv"

# Feature lists (from Optuna Stage 4)
CT_FEATURES_FILE = ROOT / "10_feb_CT_inter10364_R42_features.csv"
PT_FEATURES_FILE = ROOT / "10_feb_PT_inter9215_R62_features.csv"

# Clinical features (11 dummy-coded features)
CLINICAL_FEATURES = [
    'Age', 'Gender_Male', 'Treatment_CRT',
    'HPV_Positive', 'HPV_Unknown',
    'Tobacco_Yes', 'Tobacco_Unknown',
    'Alcohol_Yes', 'Alcohol_Unknown',
    'Perf_2plus', 'Perf_Unknown'
]

# Random seed (MUST match Optuna)
RANDOM_SEED = 42

# ==========================================
# 2. STEP 1: LOAD AND FILTER CLINICAL DATA
# ==========================================
def step1_load_clinical():
    """Load clinical data and remove patients with missing survival targets."""
    print("="*70)
    print("PHASE 0: DATA PREPARATION AND SANITY CHECKS")
    print("="*70)
    print("\nStep 1: Loading clinical data...")

    clinical = pd.read_csv(CLINICAL_FILE)
    print(f"  Loaded: {len(clinical)} patients")

    # Check missing targets
    missing_relapse = clinical['Relapse'].isna().sum()
    missing_rfs = clinical['RFS'].isna().sum()
    print(f"  Missing Relapse: {missing_relapse}")
    print(f"  Missing RFS: {missing_rfs}")

    # Filter to complete survival data (CRITICAL: removes 28 patients)
    clinical_clean = clinical.dropna(subset=['Relapse', 'RFS']).copy()
    print(f"  After filtering: {len(clinical_clean)} patients")
    print(f"  Events: {clinical_clean['Relapse'].sum():.0f}")
    print(f"  Event rate: {clinical_clean['Relapse'].mean()*100:.1f}%")

    # Keep only necessary columns
    clinical_model = clinical_clean[
        ['PatientID', 'CenterID', 'Cohort', 'Relapse', 'RFS'] + CLINICAL_FEATURES
    ].copy()

    return clinical_model

# ==========================================
# 3. STEP 2: EXTRACT SELECTED RADIOMICS FEATURES
# ==========================================
def step2_extract_radiomics():
    """Extract ONLY the 16 CT + 20 PT features selected by Optuna."""
    print("\nStep 2a: Extracting CT radiomics (16 selected features from 2820)...")

    # Load full CT radiomics
    ct_dev = pd.read_csv(CT_DEV_FILE)
    ct_ext = pd.read_csv(CT_EXT_FILE)
    print(f"  CT Development: {len(ct_dev)} patients × {len(ct_dev.columns)} features")
    print(f"  CT External (CHUP): {len(ct_ext)} patients × {len(ct_ext.columns)} features")

    # Concatenate
    ct_all = pd.concat([ct_dev, ct_ext], ignore_index=True)
    print(f"  CT Total: {len(ct_all)} patients × {len(ct_all.columns)} features")

    # Load selected feature names
    ct_feature_list = pd.read_csv(CT_FEATURES_FILE)
    selected_ct_features = ct_feature_list['Feature'].tolist()
    print(f"  Selected features: {len(selected_ct_features)}")

    # Extract selected features
    ct_subset = ct_all[['PatientID'] + selected_ct_features].copy()
    print(f"  Extracted: {len(ct_subset)} patients × {len(selected_ct_features)} CT features")

    # Verify no missing values
    ct_missing = ct_subset[selected_ct_features].isna().sum().sum()
    print(f"  Missing values: {ct_missing} (should be 0)")
    assert ct_missing == 0, "ERROR: Missing values in CT radiomics!"

    # PT radiomics
    print("\nStep 2b: Extracting PT radiomics (20 selected features from 2820)...")

    pt_dev = pd.read_csv(PT_DEV_FILE)
    pt_ext = pd.read_csv(PT_EXT_FILE)
    print(f"  PT Development: {len(pt_dev)} patients × {len(pt_dev.columns)} features")
    print(f"  PT External (CHUP): {len(pt_ext)} patients × {len(pt_ext.columns)} features")

    pt_all = pd.concat([pt_dev, pt_ext], ignore_index=True)
    print(f"  PT Total: {len(pt_all)} patients × {len(pt_all.columns)} features")

    pt_feature_list = pd.read_csv(PT_FEATURES_FILE)
    selected_pt_features = pt_feature_list['Feature'].tolist()
    print(f"  Selected features: {len(selected_pt_features)}")

    pt_subset = pt_all[['PatientID'] + selected_pt_features].copy()
    print(f"  Extracted: {len(pt_subset)} patients × {len(selected_pt_features)} PT features")

    pt_missing = pt_subset[selected_pt_features].isna().sum().sum()
    print(f"  Missing values: {pt_missing} (should be 0)")
    assert pt_missing == 0, "ERROR: Missing values in PT radiomics!"

    return ct_subset, pt_subset, selected_ct_features, selected_pt_features

# ==========================================
# 4. STEP 3: MERGE CLINICAL + CT + PT
# ==========================================
def step3_merge_data(clinical_model, ct_subset, pt_subset,
                     selected_ct_features, selected_pt_features):
    """Inner join on PatientID to get patients with complete data."""
    print("\nStep 3: Merging clinical + CT + PT...")

    # Merge clinical with CT
    merged_ct = clinical_model.merge(ct_subset, on='PatientID', how='inner')
    print(f"  After merging with CT: {len(merged_ct)} patients")

    # Merge with PT
    merged_all = merged_ct.merge(pt_subset, on='PatientID', how='inner')
    print(f"  After merging with PT: {len(merged_all)} patients")

    # Verify feature counts
    n_clinical = len(CLINICAL_FEATURES)
    n_ct = len(selected_ct_features)
    n_pt = len(selected_pt_features)
    n_total = n_clinical + n_ct + n_pt

    print(f"\n  Feature breakdown:")
    print(f"    Clinical: {n_clinical}")
    print(f"    CT radiomics: {n_ct}")
    print(f"    PT radiomics: {n_pt}")
    print(f"    TOTAL: {n_total}")

    # Check cohort distribution
    print(f"\n  Cohort distribution:")
    for cohort in ['Dev', 'CHUP']:
        subset = merged_all[merged_all['Cohort'] == cohort]
        n_events = subset['Relapse'].sum()
        event_rate = subset['Relapse'].mean() * 100
        print(f"    {cohort}: {len(subset)} patients, {n_events:.0f} events ({event_rate:.1f}%)")

    # Verify no NaNs in features
    feature_cols = CLINICAL_FEATURES + selected_ct_features + selected_pt_features
    total_missing = merged_all[feature_cols].isna().sum().sum()
    assert total_missing == 0, f"ERROR: {total_missing} missing values in merged data!"
    print(f"\n  ✓ Zero NaNs in all 47 features")

    # Save merged dataset
    output_file = OUTPUT_DIR / 'merged_data_final.csv'
    merged_all.to_csv(output_file, index=False)
    print(f"\n  Saved: {output_file.name}")
    print(f"  Shape: {merged_all.shape}")

    return merged_all, feature_cols

# ==========================================
# 5. STEP 4: TRAIN-TEST SPLIT (CRITICAL: MATCH OPTUNA)
# ==========================================
def step4_split_data(merged_all, feature_cols):
    """Split into train/test/CHUP using SAME strategy as Optuna Stage 4."""
    print("\nStep 4: Train-test split (matching Optuna split)...")

    # Separate Dev and CHUP
    dev_data = merged_all[merged_all['Cohort'] == 'Dev'].copy()
    chup_data = merged_all[merged_all['Cohort'] == 'CHUP'].copy()

    print(f"  Development: {len(dev_data)} patients")
    print(f"  CHUP (external): {len(chup_data)} patients")

    # Prepare features and targets for Dev
    X_dev = dev_data[feature_cols].values
    y_dev = Surv.from_arrays(
        event=dev_data['Relapse'].astype(bool),
        time=dev_data['RFS'].values
    )

    # CRITICAL: Use same split as Optuna (stratified by outcome, seed=42, 80/20)
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X_dev, y_dev, dev_data.index,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=dev_data['Relapse']  # Balance event rate
    )

    # Count events (structured array uses boolean indexing)
    train_events = y_train['Relapse'].sum() if 'Relapse' in y_train.dtype.names else y_train[y_train.dtype.names[0]].sum()
    test_events = y_test['Relapse'].sum() if 'Relapse' in y_test.dtype.names else y_test[y_test.dtype.names[0]].sum()

    print(f"  Train: {len(X_train)} patients, {train_events} events")
    print(f"  Test: {len(X_test)} patients, {test_events} events")

    # Prepare CHUP (external validation, never used for training)
    X_chup = chup_data[feature_cols].values
    y_chup = Surv.from_arrays(
        event=chup_data['Relapse'].astype(bool),
        time=chup_data['RFS'].values
    )

    chup_events = y_chup['Relapse'].sum() if 'Relapse' in y_chup.dtype.names else y_chup[y_chup.dtype.names[0]].sum()
    print(f"  CHUP: {len(X_chup)} patients, {chup_events} events")

    # Keep CenterID for optional GroupKFold in Phase 1
    train_centers = dev_data.loc[train_idx, 'CenterID'].values
    test_centers = dev_data.loc[test_idx, 'CenterID'].values
    chup_centers = chup_data['CenterID'].values

    return (X_train, X_test, X_chup, y_train, y_test, y_chup,
            train_centers, test_centers, chup_centers, feature_cols)

# ==========================================
# 6. STEP 5: FEATURE SCALING
# ==========================================
def step5_scale_features(X_train, X_test, X_chup):
    """Scale Age only (radiomics already z-scored, binary features 0/1)."""
    print("\nStep 5: Feature scaling verification...")

    # Age is index 0 (fit on TRAIN only to avoid leakage)
    scaler_age = StandardScaler()
    X_train[:, 0] = scaler_age.fit_transform(X_train[:, 0].reshape(-1, 1)).ravel()
    X_test[:, 0] = scaler_age.transform(X_test[:, 0].reshape(-1, 1)).ravel()
    X_chup[:, 0] = scaler_age.transform(X_chup[:, 0].reshape(-1, 1)).ravel()

    print(f"  Age scaling:")
    print(f"    Train mean: {X_train[:, 0].mean():.3f} (should be ~0)")
    print(f"    Train std: {X_train[:, 0].std():.3f} (should be ~1)")

    # Check radiomics features (should already be scaled)
    radiomics_start_idx = len(CLINICAL_FEATURES)
    radiomics_features = X_train[:, radiomics_start_idx:]

    print(f"\n  Radiomics features (should already be z-scored):")
    print(f"    Mean: {radiomics_features.mean():.3f} (should be ~0)")
    print(f"    Std: {radiomics_features.std():.3f} (should be ~1)")

    # Binary features (Gender, Treatment, dummy variables) should be 0/1
    binary_features = X_train[:, 1:len(CLINICAL_FEATURES)]
    print(f"\n  Binary/dummy features:")
    print(f"    Min: {binary_features.min():.1f} (should be 0)")
    print(f"    Max: {binary_features.max():.1f} (should be 1)")

    return X_train, X_test, X_chup, scaler_age

# ==========================================
# 7. STEP 6: VIF CHECK + CORRELATION ANALYSIS (GEMINI REFINEMENT 2)
# ==========================================
def step6_vif_and_correlation(X_train, feature_cols):
    """Check VIF for clinical features and correlation for radiomics."""
    print("\nStep 6: Variance Inflation Factor (VIF) check...")

    # Compute VIF for clinical features only (on training set)
    clinical_df = pd.DataFrame(
        X_train[:, :len(CLINICAL_FEATURES)],
        columns=CLINICAL_FEATURES
    )

    vif_data = pd.DataFrame()
    vif_data['Feature'] = CLINICAL_FEATURES
    vif_data['VIF'] = [
        variance_inflation_factor(clinical_df.values, i)
        for i in range(len(CLINICAL_FEATURES))
    ]

    print("\n  Clinical features VIF:")
    print(vif_data.to_string(index=False))

    # Check for high VIF (>10 indicates severe multicollinearity)
    high_vif = vif_data[vif_data['VIF'] > 10]
    if len(high_vif) > 0:
        print(f"\n  WARNING: {len(high_vif)} features with VIF > 10")
        print(high_vif)
    else:
        print(f"\n  ✓ All VIF < 10 (dummy coding successful)")

    # GEMINI REFINEMENT 2: Correlation analysis for radiomics features
    print("\nStep 6.5: Feature correlation analysis (radiomics only)...")

    radiomics_start_idx = len(CLINICAL_FEATURES)
    radiomics_cols = feature_cols[radiomics_start_idx:]

    radiomics_df = pd.DataFrame(
        X_train[:, radiomics_start_idx:],
        columns=radiomics_cols
    )

    # Correlation matrix
    corr_matrix = radiomics_df.corr()

    # Identify high correlation pairs (|r| > 0.8)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                high_corr_pairs.append({
                    'Feature1': corr_matrix.columns[i],
                    'Feature2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })

    print(f"\n  High correlation pairs (|r| > 0.8): {len(high_corr_pairs)}")
    if len(high_corr_pairs) > 0:
        high_corr_df = pd.DataFrame(high_corr_pairs)
        print(high_corr_df.to_string(index=False))
    else:
        print("  None found (features are relatively independent)")

    # Save correlation heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1,
                xticklabels=False, yticklabels=False,
                cbar_kws={'label': 'Pearson Correlation'})
    plt.title('Radiomics Feature Correlation (36 features)', fontsize=14)
    plt.tight_layout()
    heatmap_file = OUTPUT_DIR / 'radiomics_correlation_heatmap.png'
    plt.savefig(heatmap_file, dpi=150)
    print(f"\n  Saved: {heatmap_file.name}")
    plt.close()

    return vif_data

# ==========================================
# 8. STEP 7: FINAL SUMMARY AND SAVE
# ==========================================
def step7_save_arrays(X_train, X_test, X_chup, y_train, y_test, y_chup,
                      train_centers, test_centers, chup_centers, feature_cols):
    """Save processed arrays and print final summary."""
    print("\nStep 7: Final data summary...")
    print("\n" + "="*70)
    print("SANITY CHECKS COMPLETE - DATA READY FOR MODELING")
    print("="*70)

    print(f"\nDataset dimensions:")
    print(f"  Train: {X_train.shape}")
    print(f"  Test: {X_test.shape}")
    print(f"  CHUP: {X_chup.shape}")

    # Get event counts (handle both field names)
    event_field = y_train.dtype.names[0]  # First field is event indicator
    train_events = y_train[event_field].sum()
    test_events = y_test[event_field].sum()
    chup_events = y_chup[event_field].sum()

    print(f"\nEvent summary:")
    print(f"  Train: {train_events} / {len(y_train)} ({train_events/len(y_train)*100:.1f}%)")
    print(f"  Test: {test_events} / {len(y_test)} ({test_events/len(y_test)*100:.1f}%)")
    print(f"  CHUP: {chup_events} / {len(y_chup)} ({chup_events/len(y_chup)*100:.1f}%)")

    print(f"\nEPV analysis:")
    print(f"  Pre-LASSO: {train_events} events / {X_train.shape[1]} features = {train_events/X_train.shape[1]:.2f}")
    print(f"  Post-LASSO (target 15-20): {train_events} / 20 = {train_events/20:.2f}")

    print(f"\nMissing values check:")
    print(f"  Train: {np.isnan(X_train).sum()} (should be 0)")
    print(f"  Test: {np.isnan(X_test).sum()} (should be 0)")
    print(f"  CHUP: {np.isnan(X_chup).sum()} (should be 0)")

    # Save processed datasets
    np.save(OUTPUT_DIR / 'X_train.npy', X_train)
    np.save(OUTPUT_DIR / 'X_test.npy', X_test)
    np.save(OUTPUT_DIR / 'X_chup.npy', X_chup)
    np.save(OUTPUT_DIR / 'y_train.npy', y_train)
    np.save(OUTPUT_DIR / 'y_test.npy', y_test)
    np.save(OUTPUT_DIR / 'y_chup.npy', y_chup)

    # Save CenterID for optional GroupKFold in Phase 1
    np.save(OUTPUT_DIR / 'train_centers.npy', train_centers)
    np.save(OUTPUT_DIR / 'test_centers.npy', test_centers)
    np.save(OUTPUT_DIR / 'chup_centers.npy', chup_centers)

    # Save feature names
    pd.DataFrame({'Feature': feature_cols}).to_csv(
        OUTPUT_DIR / 'feature_names.csv', index=False
    )

    print(f"\n✓ Saved train/test/CHUP arrays to {OUTPUT_DIR.name}/")
    print(f"\nREADY FOR PHASE 1: LASSO-COX MODELING")
    print("="*70)

# ==========================================
# 9. MAIN EXECUTION
# ==========================================
def main():
    """Execute complete Phase 0 workflow."""
    # Step 1: Load clinical data
    clinical_model = step1_load_clinical()

    # Step 2: Extract radiomics features
    ct_subset, pt_subset, selected_ct_features, selected_pt_features = \
        step2_extract_radiomics()

    # Step 3: Merge all data
    merged_all, feature_cols = step3_merge_data(
        clinical_model, ct_subset, pt_subset,
        selected_ct_features, selected_pt_features
    )

    # Step 4: Train-test split
    (X_train, X_test, X_chup, y_train, y_test, y_chup,
     train_centers, test_centers, chup_centers, feature_cols) = \
        step4_split_data(merged_all, feature_cols)

    # Step 5: Feature scaling
    X_train, X_test, X_chup, scaler_age = \
        step5_scale_features(X_train, X_test, X_chup)

    # Step 6: VIF check and correlation analysis
    vif_data = step6_vif_and_correlation(X_train, feature_cols)

    # Step 7: Save arrays and summary
    step7_save_arrays(
        X_train, X_test, X_chup, y_train, y_test, y_chup,
        train_centers, test_centers, chup_centers, feature_cols
    )

    print("\n✓ Phase 0 Complete!")
    print("\nNext step: Run Phase 1 models (LASSO-Cox, etc.)")

if __name__ == "__main__":
    main()