# -*- coding: utf-8 -*-
"""
Create PET External Validation Split - CHUS Holdout
====================================================
Hold out CHUS center (55 patients) for external validation
Development: remaining patients from 6 centers

Matches CT methodology from Dec_2025/24_dec_create_external_split.py
"""

import pandas as pd
from pathlib import Path

print("=" * 80)
print("Creating PET External Validation Split - CHUS Holdout")
print("=" * 80)

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
PET_FEATURES = PROJECT_ROOT / "MultiRegion_Features" / "PT_features_Combined_GTVp+GTVn.csv"
OUTCOMES = PROJECT_ROOT / "HECKTOR_2025_Training_Task_2.csv"

# Load PET features
print(f"\n[1] Loading PET features...")
pet_df = pd.read_csv(PET_FEATURES)
pet_df = pet_df.rename(columns={'CaseID': 'PatientID'})
print(f"    Patients with PET features: {len(pet_df)}")

# Load outcomes
print(f"\n[2] Loading outcomes...")
outcomes_df = pd.read_csv(OUTCOMES)
outcomes_df = outcomes_df[['PatientID', 'Relapse', 'RFS']]
print(f"    Patients with RFS data: {len(outcomes_df)}")

# Merge
print(f"\n[3] Merging...")
df = pet_df.merge(outcomes_df, on='PatientID', how='inner')
print(f"    Merged patients: {len(df)}")

# Check for complete cases
feature_cols = [c for c in df.columns if c not in ['PatientID', 'Relapse', 'RFS']]
missing_per_patient = df[feature_cols].isnull().sum(axis=1)
df = df[missing_per_patient == 0]
print(f"    Complete cases: {len(df)}")

# Extract center from PatientID
df['Center'] = df['PatientID'].str[:4]

# Configuration - match CT methodology
EXTERNAL_CENTER = 'CHUS'

# Split by center
development_df = df[df['Center'] != EXTERNAL_CENTER].copy()
external_df = df[df['Center'] == EXTERNAL_CENTER].copy()

print(f"\n[4] Split results:")
print(f"    Original: {len(df)} patients")
print(f"    Development (excluding {EXTERNAL_CENTER}): {len(development_df)} patients ({len(development_df)/len(df)*100:.1f}%)")
print(f"    External ({EXTERNAL_CENTER}): {len(external_df)} patients ({len(external_df)/len(df)*100:.1f}%)")

# Event statistics
print(f"\n[5] Event statistics:")
print(f"    Development: {development_df['Relapse'].sum()} events ({100*development_df['Relapse'].mean():.1f}%)")
print(f"    External: {external_df['Relapse'].sum()} events ({100*external_df['Relapse'].mean():.1f}%)")

# Verify no overlap
dev_ids = set(development_df['PatientID'])
ext_ids = set(external_df['PatientID'])
overlap = dev_ids.intersection(ext_ids)
if len(overlap) > 0:
    print(f"    ERROR: {len(overlap)} patients in both sets!")
    exit(1)
print(f"    Verified: No patient overlap")

# Center breakdown
print(f"\n[6] Center breakdown (Development):")
for center in sorted(development_df['Center'].unique()):
    n = len(development_df[development_df['Center'] == center])
    print(f"    {center}: {n} patients")

# Save (drop Center column before saving)
output_dev = SCRIPT_DIR / '16_jan_dataset_PET_development.csv'
output_ext = SCRIPT_DIR / '16_jan_dataset_PET_external.csv'

development_df.drop('Center', axis=1).to_csv(output_dev, index=False)
external_df.drop('Center', axis=1).to_csv(output_ext, index=False)

print(f"\n[7] Files created:")
print(f"    {output_dev.name} ({len(development_df)} patients)")
print(f"    {output_ext.name} ({len(external_df)} patients)")

# Feature count
n_gtvp = len([c for c in feature_cols if c.startswith('GTVp_')])
n_gtvn = len([c for c in feature_cols if c.startswith('GTVn_')])
print(f"\n[8] Feature breakdown:")
print(f"    GTVp features: {n_gtvp}")
print(f"    GTVn features: {n_gtvn}")
print(f"    Total features: {len(feature_cols)}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Development cohort: {len(development_df)} patients (CHUS excluded)")
print(f"External holdout: {len(external_df)} patients (CHUS only)")
print(f"Features: {len(feature_cols)}")
print(f"\nReady for Stage 1 feature selection!")
print("=" * 80)
