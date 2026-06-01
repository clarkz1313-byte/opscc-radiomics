# -*- coding: utf-8 -*-
"""
Create External Validation Split - CHUS + CHUP Holdout (No-Split Approach)
==========================================================================
Supervisor-directed configuration (27_feb_wrap.md):
- Development: All centers EXCEPT CHUS and CHUP (remaining centerIDs)
- External: CHUS + CHUP combined (55 + 35 = 90 patients)

Input datasets:
- CT: Dec_2025/19_dec_dataset_C_complete_cases.csv (547 patients)
- PT: Jan_2026/16_jan_dataset_PET_development.csv + 16_jan_dataset_PET_external.csv (545 patients)

Output (Mar_2026/):
- 27_feb_CT_development.csv
- 27_feb_CT_external.csv
- 27_feb_PT_development.csv
- 27_feb_PT_external.csv
"""

import pandas as pd
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent  # D:\Uppsala thesis
OUTPUT_DIR = BASE_DIR / 'Mar_2026'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Creating External Validation Split - CHUS + CHUP Holdout (No-Split)")
print("=" * 80)

# ============================================================================
# Configuration
# ============================================================================

EXTERNAL_CENTERS = ['CHUS', 'CHUP']

DATASETS = {
    'CT': BASE_DIR / 'Dec_2025' / '19_dec_dataset_C_complete_cases.csv',
    'PT': (BASE_DIR / 'Jan_2026' / '16_jan_dataset_PET_development.csv',
           BASE_DIR / 'Jan_2026' / '16_jan_dataset_PET_external.csv'),
}

# ============================================================================
# Split each modality
# ============================================================================

summary = {}

for modality, input_spec in DATASETS.items():
    if isinstance(input_spec, tuple):
        input_paths = input_spec
        for p in input_paths:
            if not p.exists():
                print(f"ERROR: {p} not found")
                sys.exit(1)
        df = pd.concat([pd.read_csv(p) for p in input_paths], ignore_index=True)
        print(f"\n{'=' * 80}")
        print(f"[{modality}] {input_paths[0].name} + {input_paths[1].name}")
        print(f"{'=' * 80}")
    else:
        input_path = input_spec
        if not input_path.exists():
            print(f"ERROR: {input_path} not found")
            sys.exit(1)
        df = pd.read_csv(input_path)
        print(f"\n{'=' * 80}")
        print(f"[{modality}] {input_path.name}")
        print(f"{'=' * 80}")
    df['Center'] = df['PatientID'].str[:4]

    # Split: Dev = all except CHUS/CHUP, External = CHUS + CHUP
    development_df = df[~df['Center'].isin(EXTERNAL_CENTERS)].copy()
    external_df = df[df['Center'].isin(EXTERNAL_CENTERS)].copy()

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
    print(f"  Development (excl. CHUS, CHUP): {len(development_df)} patients, {dev_events} events ({dev_events/len(development_df)*100:.1f}%)")
    print(f"  External (CHUS + CHUP):         {len(external_df)} patients, {ext_events} events ({ext_events/len(external_df)*100:.1f}%)")

    print(f"\n  Development center breakdown:")
    for center in sorted(development_df['Center'].unique()):
        subset = development_df[development_df['Center'] == center]
        events = int(subset['Relapse'].sum())
        print(f"    {center}: {len(subset)} patients, {events} events")

    print(f"\n  External center breakdown:")
    for center in sorted(external_df['Center'].unique()):
        subset = external_df[external_df['Center'] == center]
        events = int(subset['Relapse'].sum())
        print(f"    {center}: {len(subset)} patients, {events} events")

    print(f"\n  Overlap check: PASSED")
    print(f"  Missing outcomes check: PASSED")

    # --- Save ---
    dev_out = OUTPUT_DIR / f'27_feb_{modality}_development.csv'
    ext_out = OUTPUT_DIR / f'27_feb_{modality}_external.csv'

    development_df.drop('Center', axis=1).to_csv(dev_out, index=False)
    external_df.drop('Center', axis=1).to_csv(ext_out, index=False)

    print(f"\n  Saved: {dev_out}")
    print(f"  Saved: {ext_out}")

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
print("SUMMARY: CHUS+CHUP External Split (No-Split Approach)")
print(f"{'=' * 80}")
print(f"{'Modality':<10} {'Dev Patients':<15} {'Dev Events':<13} {'Ext Patients':<15} {'Ext Events'}")
print("-" * 68)
for modality, s in summary.items():
    print(f"{modality:<10} {s['dev_patients']:<15} {s['dev_events']:<13} {s['ext_patients']:<15} {s['ext_events']}")
print(f"\nOutput directory: {OUTPUT_DIR}")
print(f"Next step: Use these files as input for phase0 / Stage 3 pipeline")
print(f"{'=' * 80}")
