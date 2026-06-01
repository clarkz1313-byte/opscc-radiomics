# -*- coding: utf-8 -*-
"""
Create External Validation Split - CHUS Holdout
================================================
Hold out CHUS center (55 patients) for external validation
Development: 492 patients from remaining 6 centers
"""

import pandas as pd
from pathlib import Path

print("="*80)
print("Creating External Validation Split - CHUS Holdout")
print("="*80)

# Load Dataset C
input_file = '19_dec_dataset_C_complete_cases.csv'
if not Path(input_file).exists():
    print(f"ERROR: {input_file} not found")
    exit(1)

df = pd.read_csv(input_file)
df['Center'] = df['PatientID'].str[:4]

# Configuration
EXTERNAL_CENTER = 'CHUS'

# Split by center
development_df = df[df['Center'] != EXTERNAL_CENTER].copy()
external_df = df[df['Center'] == EXTERNAL_CENTER].copy()

print(f"\nOriginal: {len(df)} patients")
print(f"Development (excluding {EXTERNAL_CENTER}): {len(development_df)} patients ({len(development_df)/len(df)*100:.1f}%)")
print(f"External ({EXTERNAL_CENTER}): {len(external_df)} patients ({len(external_df)/len(df)*100:.1f}%)")

# Verify no overlap
dev_ids = set(development_df['PatientID'])
ext_ids = set(external_df['PatientID'])
overlap = dev_ids.intersection(ext_ids)
if len(overlap) > 0:
    print(f"ERROR: {len(overlap)} patients in both sets!")
    exit(1)

# Save
development_df.drop('Center', axis=1).to_csv('24_dec_dataset_C_development.csv', index=False)
external_df.drop('Center', axis=1).to_csv('24_dec_dataset_C_external.csv', index=False)

print("\nFiles created:")
print("  24_dec_dataset_C_development.csv")
print("  24_dec_dataset_C_external.csv")
print("\nVerified: No patient overlap")
print("="*80)
