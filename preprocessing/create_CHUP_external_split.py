# -*- coding: utf-8 -*-
"""
Create External Validation Split - CHUP Holdout
================================================
New cohort configuration for CHUP-external validation:
- Development: All centers EXCEPT CHUP (includes CHUS)
- External: CHUP only (35 patients per modality)

Replaces original CHUS-external split (24_dec_create_external_split.py).

Input datasets:
- CT: Dec_2025/19_dec_dataset_C_complete_cases.csv  (547 patients)
- PT: Jan_2026/15_jan_dataset_PET_complete_cases.csv (545 patients)

Output (Feb_2026/):
- 1_feb_CT_development.csv
- 1_feb_CT_external.csv
- 1_feb_PT_development.csv
- 1_feb_PT_external.csv
"""

import pandas as pd
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent  # D:\Uppsala thesis
OUTPUT_DIR = Path(__file__).parent       # Feb_2026

print("=" * 80)
print("Creating External Validation Split - CHUP Holdout")
print("=" * 80)

# ============================================================================
# Configuration
# ============================================================================

EXTERNAL_CENTER = 'CHUP'

DATASETS = {
    'CT': BASE_DIR / 'Dec_2025' / '19_dec_dataset_C_complete_cases.csv',
    'PT': BASE_DIR / 'Jan_2026' / '15_jan_dataset_PET_complete_cases.csv',
}

# ============================================================================
# Split each modality
# ============================================================================

summary = {}

for modality, input_path in DATASETS.items():
    print(f"\n{'=' * 80}")
    print(f"[{modality}] {input_path.name}")
    print(f"{'=' * 80}")

    if not input_path.exists():
        print(f"ERROR: {input_path} not found")
        sys.exit(1)

    df = pd.read_csv(input_path)
    df['Center'] = df['PatientID'].str[:4]

    # Split by center
    development_df = df[df['Center'] != EXTERNAL_CENTER].copy()
    external_df = df[df['Center'] == EXTERNAL_CENTER].copy()

    # --- Verify no missing outcomes ---
    dev_missing = development_df[['Relapse', 'RFS']].isna().sum()
    ext_missing = external_df[['Relapse', 'RFS']].isna().sum()

    if dev_missing.sum() > 0:
        print(f"  ERROR: Development has missing outcomes: {dev_missing.to_dict()}")
        sys.exit(1)
    if ext_missing.sum() > 0:
        print(f"  ERROR: External has missing outcomes: {ext_missing.to_dict()}")
        sys.exit(1)

    # --- Verify no patient overlap ---
    overlap = set(development_df['PatientID']) & set(external_df['PatientID'])
    if len(overlap) > 0:
        print(f"  ERROR: {len(overlap)} patients appear in both sets!")
        sys.exit(1)

    # --- Report ---
    dev_events = int(development_df['Relapse'].sum())
    ext_events = int(external_df['Relapse'].sum())

    print(f"\n  Total patients: {len(df)}")
    print(f"  Development (excl. {EXTERNAL_CENTER}): {len(development_df)} patients, {dev_events} events ({dev_events/len(development_df)*100:.1f}%)")
    print(f"  External ({EXTERNAL_CENTER}):          {len(external_df)} patients, {ext_events} events ({ext_events/len(external_df)*100:.1f}%)")

    print(f"\n  Development center breakdown:")
    for center in sorted(development_df['Center'].unique()):
        subset = development_df[development_df['Center'] == center]
        events = int(subset['Relapse'].sum())
        print(f"    {center}: {len(subset)} patients, {events} events")

    print(f"\n  Overlap check: PASSED")
    print(f"  Missing outcomes check: PASSED")

    # --- Save ---
    dev_out = OUTPUT_DIR / f'1_feb_{modality}_development.csv'
    ext_out = OUTPUT_DIR / f'1_feb_{modality}_external.csv'

    development_df.drop('Center', axis=1).to_csv(dev_out, index=False)
    external_df.drop('Center', axis=1).to_csv(ext_out, index=False)

    print(f"\n  Saved: {dev_out.name}")
    print(f"  Saved: {ext_out.name}")

    summary[modality] = {
        'dev_patients': len(development_df),
        'dev_events': dev_events,
        'ext_patients': len(external_df),
        'ext_events': ext_events,
    }

# ============================================================================
# Summary
# ============================================================================

print(f"\n{'=' * 80}")
print("SUMMARY: CHUP-External Split")
print(f"{'=' * 80}")
print(f"{'Modality':<10} {'Dev Patients':<15} {'Dev Events':<13} {'Ext Patients':<15} {'Ext Events'}")
print("-" * 68)
for modality, s in summary.items():
    print(f"{modality:<10} {s['dev_patients']:<15} {s['dev_events']:<13} {s['ext_patients']:<15} {s['ext_events']}")
print(f"\nOutput directory: {OUTPUT_DIR}")
print(f"Next step: Use these files as input for Stage 3 pipeline search")
print(f"{'=' * 80}")
