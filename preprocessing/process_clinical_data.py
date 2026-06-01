"""
Clinical Data Processing Script
================================
Date: February 11, 2026
Purpose: Process HECKTOR_2025_training_Allinfo.csv with agreed encoding scheme

Key Decisions (Feb 11, 2026):
1. Treatment: Single binary column (Treatment_CRT)
2. Performance Status: 3 groups (ECOG 0-1, 2+, Unknown)
3. Encoding: Dummy coding (drop reference category)
4. Strategy: Global Optimization + Ablation (single model)

Output: 11_feb_Processed_clinical_Allinfo.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths (results under Phase1_2_ouputs for easier cross-check)
ROOT = Path("d:/Uppsala thesis")
FEB_ROOT = ROOT / "Feb_2026"
PHASE1_2_OUTPUTS = FEB_ROOT / "Phase1_2_ouputs"
INPUT_FILE = ROOT / "HECKTOR_2025_training_Allinfo.csv"
OUTPUT_FILE = PHASE1_2_OUTPUTS / "11_feb_Processed_clinical_Allinfo.csv"

print("="*60)
print("HECKTOR 2025 Clinical Data Processing")
print("="*60)
print()

# Load data
print(f"Loading: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)
print(f"Total rows: {len(df)}")
print()

# Filter to Has_GTVn=1 (current cohort constraint)
df = df[df['Has_GTVn'] == 1].copy()
print(f"After Has_GTVn=1 filter: {len(df)} rows")
print()

# Identify cohorts
df['Cohort'] = df['CenterID'].apply(lambda x: 'CHUP' if x == 2 else 'Dev')
print(f"Dev cohort: {(df['Cohort']=='Dev').sum()} patients")
print(f"CHUP cohort: {(df['Cohort']=='CHUP').sum()} patients")
print()

# ============================================================================
# 1. PERFORMANCE STATUS: Harmonize and Group
# ============================================================================
print("Processing Performance Status...")

def harmonize_performance(val):
    """Harmonize KPS to ECOG and group to 0-1, 2+, Unknown"""
    if pd.isna(val):
        return 'Unknown'

    # Harmonize KPS to ECOG
    if val == 100 or val == 90:
        val = 0
    elif val == 80:
        val = 1

    # Group ECOG
    if val in [0, 1]:
        return '0-1'
    elif val in [2, 3, 4]:
        return '2+'
    else:
        return 'Unknown'

df['Perf_Grouped'] = df['Performance Status'].apply(harmonize_performance)

# Dummy coding: Reference = '0-1' (functional)
df['Perf_2plus'] = (df['Perf_Grouped'] == '2+').astype(int)
df['Perf_Unknown'] = (df['Perf_Grouped'] == 'Unknown').astype(int)

print(f"  ECOG 0-1 (reference): {(df['Perf_Grouped']=='0-1').sum()}")
print(f"  ECOG 2+ (Perf_2plus=1): {df['Perf_2plus'].sum()}")
print(f"  Unknown (Perf_Unknown=1): {df['Perf_Unknown'].sum()}")
print()

# ============================================================================
# 2. TOBACCO CONSUMPTION: Dummy Coding
# ============================================================================
print("Processing Tobacco Consumption...")

def encode_tobacco(val):
    """Dummy coding: Reference = No (0)"""
    if pd.isna(val):
        return 'Unknown'
    elif val == 1:
        return 'Yes'
    else:
        return 'No'

df['Tobacco_Grouped'] = df['Tobacco Consumption'].apply(encode_tobacco)

df['Tobacco_Yes'] = (df['Tobacco_Grouped'] == 'Yes').astype(int)
df['Tobacco_Unknown'] = (df['Tobacco_Grouped'] == 'Unknown').astype(int)

print(f"  No (reference): {(df['Tobacco_Grouped']=='No').sum()}")
print(f"  Yes (Tobacco_Yes=1): {df['Tobacco_Yes'].sum()}")
print(f"  Unknown (Tobacco_Unknown=1): {df['Tobacco_Unknown'].sum()}")
print()

# ============================================================================
# 3. ALCOHOL CONSUMPTION: Dummy Coding
# ============================================================================
print("Processing Alcohol Consumption...")

def encode_alcohol(val):
    """Dummy coding: Reference = No (0)"""
    if pd.isna(val):
        return 'Unknown'
    elif val == 1:
        return 'Yes'
    else:
        return 'No'

df['Alcohol_Grouped'] = df['Alcohol Consumption'].apply(encode_alcohol)

df['Alcohol_Yes'] = (df['Alcohol_Grouped'] == 'Yes').astype(int)
df['Alcohol_Unknown'] = (df['Alcohol_Grouped'] == 'Unknown').astype(int)

print(f"  No (reference): {(df['Alcohol_Grouped']=='No').sum()}")
print(f"  Yes (Alcohol_Yes=1): {df['Alcohol_Yes'].sum()}")
print(f"  Unknown (Alcohol_Unknown=1): {df['Alcohol_Unknown'].sum()}")
print()

# ============================================================================
# 4. HPV STATUS: Dummy Coding
# ============================================================================
print("Processing HPV Status...")

def encode_hpv(val):
    """Dummy coding: Reference = Negative (0)"""
    if pd.isna(val):
        return 'Unknown'
    elif val == 1:
        return 'Positive'
    else:
        return 'Negative'

df['HPV_Grouped'] = df['HPV Status'].apply(encode_hpv)

df['HPV_Positive'] = (df['HPV_Grouped'] == 'Positive').astype(int)
df['HPV_Unknown'] = (df['HPV_Grouped'] == 'Unknown').astype(int)

print(f"  Negative (reference): {(df['HPV_Grouped']=='Negative').sum()}")
print(f"  Positive (HPV_Positive=1): {df['HPV_Positive'].sum()}")
print(f"  Unknown (HPV_Unknown=1): {df['HPV_Unknown'].sum()}")
print()

# ============================================================================
# 5. TREATMENT: Binary (already 0/1)
# ============================================================================
print("Processing Treatment...")

df['Treatment_CRT'] = df['Treatment'].astype(int)

print(f"  RT (reference, Treatment_CRT=0): {(df['Treatment_CRT']==0).sum()}")
print(f"  CRT (Treatment_CRT=1): {(df['Treatment_CRT']==1).sum()}")
print()

# ============================================================================
# 6. GENDER: Binary (already 0/1)
# ============================================================================
print("Processing Gender...")

df['Gender_Male'] = df['Gender'].astype(int)

print(f"  Female (reference, Gender_Male=0): {(df['Gender_Male']==0).sum()}")
print(f"  Male (Gender_Male=1): {(df['Gender_Male']==1).sum()}")
print()

# ============================================================================
# 7. AGE: Keep as continuous (scaling done at train time)
# ============================================================================
print("Processing Age...")
print(f"  Range: {df['Age'].min():.0f} - {df['Age'].max():.0f} years")
print(f"  Mean: {df['Age'].mean():.1f} ± {df['Age'].std():.1f}")
print()

# ============================================================================
# 8. DROP M-STAGE (100% missing in CHUP)
# ============================================================================
print("Dropping M-stage (100% missing in CHUP)...")
print()

# ============================================================================
# FINAL OUTPUT COLUMNS
# ============================================================================
print("="*60)
print("FINAL FEATURE SET")
print("="*60)

# Select final columns
output_cols = [
    # Identifiers
    'PatientID', 'CenterID', 'Cohort',

    # Targets (Objective 1: Survival)
    'Relapse', 'RFS',

    # Target (Objective 2: HPV classification from radiomics)
    'HPV Status',  # Keep original for Obj 2

    # Metadata
    'Has_GTVn', 'Dosiomics_available',

    # Clinical Features (11 features for modeling)
    'Age',               # 1: Continuous
    'Gender_Male',       # 2: Binary
    'Treatment_CRT',     # 3: Binary
    'HPV_Positive',      # 4: Dummy (ref=Negative)
    'HPV_Unknown',       # 5: Dummy
    'Tobacco_Yes',       # 6: Dummy (ref=No)
    'Tobacco_Unknown',   # 7: Dummy
    'Alcohol_Yes',       # 8: Dummy (ref=No)
    'Alcohol_Unknown',   # 9: Dummy
    'Perf_2plus',        # 10: Dummy (ref=ECOG 0-1)
    'Perf_Unknown',      # 11: Dummy

    # Grouped versions (for reference)
    'Perf_Grouped', 'Tobacco_Grouped', 'Alcohol_Grouped', 'HPV_Grouped'
]

df_output = df[output_cols].copy()

print()
print("Clinical Features (for Cox model):")
print("  1. Age (continuous, StandardScaler at train time)")
print("  2. Gender_Male (binary)")
print("  3. Treatment_CRT (binary)")
print("  4. HPV_Positive (dummy, ref=Negative)")
print("  5. HPV_Unknown (dummy)")
print("  6. Tobacco_Yes (dummy, ref=No)")
print("  7. Tobacco_Unknown (dummy)")
print("  8. Alcohol_Yes (dummy, ref=No)")
print("  9. Alcohol_Unknown (dummy)")
print(" 10. Perf_2plus (dummy, ref=ECOG 0-1)")
print(" 11. Perf_Unknown (dummy)")
print()
print("Total Clinical Features: 11")
print("Combined with Radiomics (36): 47 features before LASSO")
print("Target after LASSO: 15-20 features (EPV = 5.4-7.1)")
print()

# Save
PHASE1_2_OUTPUTS.mkdir(exist_ok=True, parents=True)
print(f"Saving to: {OUTPUT_FILE}")
df_output.to_csv(OUTPUT_FILE, index=False)
print(f"Saved {len(df_output)} rows, {len(df_output.columns)} columns")
print()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("="*60)
print("SUMMARY BY COHORT")
print("="*60)

for cohort in ['Dev', 'CHUP']:
    subset = df_output[df_output['Cohort'] == cohort]
    print(f"\n{cohort} Cohort (n={len(subset)}):")
    print(f"  Events: {subset['Relapse'].sum():.0f} ({subset['Relapse'].sum()/len(subset)*100:.1f}%)")
    print(f"  Age: {subset['Age'].mean():.1f} ± {subset['Age'].std():.1f}")
    print(f"  Male: {subset['Gender_Male'].sum()} ({subset['Gender_Male'].sum()/len(subset)*100:.1f}%)")
    print(f"  CRT: {subset['Treatment_CRT'].sum()} ({subset['Treatment_CRT'].sum()/len(subset)*100:.1f}%)")
    print(f"  HPV+: {subset['HPV_Positive'].sum()} ({subset['HPV_Positive'].sum()/len(subset)*100:.1f}%)")
    print(f"  HPV Unknown: {subset['HPV_Unknown'].sum()} ({subset['HPV_Unknown'].sum()/len(subset)*100:.1f}%)")
    print(f"  Tobacco Yes: {subset['Tobacco_Yes'].sum()} ({subset['Tobacco_Yes'].sum()/len(subset)*100:.1f}%)")
    print(f"  Tobacco Unknown: {subset['Tobacco_Unknown'].sum()} ({subset['Tobacco_Unknown'].sum()/len(subset)*100:.1f}%)")
    print(f"  Alcohol Yes: {subset['Alcohol_Yes'].sum()} ({subset['Alcohol_Yes'].sum()/len(subset)*100:.1f}%)")
    print(f"  Alcohol Unknown: {subset['Alcohol_Unknown'].sum()} ({subset['Alcohol_Unknown'].sum()/len(subset)*100:.1f}%)")
    print(f"  Perf 2+: {subset['Perf_2plus'].sum()} ({subset['Perf_2plus'].sum()/len(subset)*100:.1f}%)")
    print(f"  Perf Unknown: {subset['Perf_Unknown'].sum()} ({subset['Perf_Unknown'].sum()/len(subset)*100:.1f}%)")
    print(f"  Dosiomics: {subset['Dosiomics_available'].sum()} ({subset['Dosiomics_available'].sum()/len(subset)*100:.1f}%)")

print()
print("="*60)
print("PROCESSING COMPLETE")
print("="*60)
