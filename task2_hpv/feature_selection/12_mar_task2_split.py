# -*- coding: utf-8 -*-
"""
Task 2: HPV Status Classification — Cohort Split
=================================================
Supervisor-confirmed configuration (6_mar_task2.md, 2026-03-12):

PRIMARY split (no-MDA A1):
    Dev:      CID1 (CHUM) + CID6 (HGJ) + CID2 (CHUP) + CID7 (HMR)
    External: CID3 (CHUS)
    MDA (CID5) excluded — scanner confound + label instability (1 HPV- in 322 pts)

SENSITIVITY split (no-MDA Option3):
    Dev:      CID1 (CHUM) + CID6 (HGJ) + CID3 (CHUS) + CID7 (HMR)
    External: CID2 (CHUP) — 15/15 balanced, hardest stress test

SENSITIVITY 2 (full-MDA A1):
    Dev:      CID1 (CHUM) + CID5 (MDA, full) + CID6 (HGJ) + CID2 (CHUP) + CID7 (HMR)
    External: CID3 (CHUS)
    Upper-bound dominance comparison only — not used for biological interpretation

HPV label encoding:
    HPV_Positive=1                        -> label 1 (HPV+)
    HPV_Positive=0 AND HPV_Unknown=0      -> label 0 (HPV-)
    HPV_Unknown=1                         -> NaN -> excluded from all supervised steps

Notes:
    - HMR (CID7, 2 pts, 0 HPV-) kept in dev for centre diversity;
      excluded from LOCO folds and HPV metrics
    - USZ (CID8) has 0 HPV-known patients; absent from Task 2 by data
    - Radiomics source: 2v split files (457+90 CT, 455+90 PT)
      covering all centres from original Task 1 split
    - Feature selection runs from scratch on Task 2 dev cohort (not Task 1 finalists)
    - All outputs written to Mar_2026_task2/

Inputs:
    Feb_2026/12_feb_merged_Clinical_data_only_final.csv
    Mar_2026/27_feb_CT_development.csv
    Mar_2026/27_feb_CT_external.csv
    Mar_2026/27_feb_PT_development.csv
    Mar_2026/27_feb_PT_external.csv

Outputs (Mar_2026_task2/):
    12_mar_task2_CT_primary_dev.csv       -- primary dev, CT, HPV-known only
    12_mar_task2_CT_primary_ext.csv       -- primary ext (CHUS), CT, HPV-known only
    12_mar_task2_PT_primary_dev.csv       -- primary dev, PT, HPV-known only
    12_mar_task2_PT_primary_ext.csv       -- primary ext (CHUS), PT, HPV-known only
    12_mar_task2_CT_sens1_dev.csv         -- sensitivity Option3 dev, CT
    12_mar_task2_CT_sens1_ext.csv         -- sensitivity Option3 ext (CHUP), CT
    12_mar_task2_PT_sens1_dev.csv         -- sensitivity Option3 dev, PT
    12_mar_task2_PT_sens1_ext.csv         -- sensitivity Option3 ext (CHUP), PT
    12_mar_task2_CT_sens2_fullMDA_dev.csv -- full-MDA dev, CT (upper-bound only)
    12_mar_task2_PT_sens2_fullMDA_dev.csv -- full-MDA dev, PT (upper-bound only)

Usage:
    cd "D:/Uppsala thesis"
    python Mar_2026_task2/12_mar_task2_split.py
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# Paths
# ============================================================================

SCRIPT_DIR   = Path(__file__).resolve().parent          # Mar_2026_task2/
PROJECT_ROOT = SCRIPT_DIR.parent                        # D:\Uppsala thesis
OUTPUT_DIR   = SCRIPT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATH_CLINICAL = PROJECT_ROOT / 'Feb_2026' / '12_feb_merged_Clinical_data_only_final.csv'
PATH_CT_DEV   = PROJECT_ROOT / 'Mar_2026' / '27_feb_CT_development.csv'
PATH_CT_EXT   = PROJECT_ROOT / 'Mar_2026' / '27_feb_CT_external.csv'
PATH_PT_DEV   = PROJECT_ROOT / 'Mar_2026' / '27_feb_PT_development.csv'
PATH_PT_EXT   = PROJECT_ROOT / 'Mar_2026' / '27_feb_PT_external.csv'

# ============================================================================
# Split configuration
# ============================================================================

# Centre prefix -> 4-char prefix in PatientID
# CID1=CHUM, CID2=CHUP, CID3=CHUS, CID5=MDA-, CID6=HGJ-, CID7=HMR-

PRIMARY_DEV_PREFIXES  = ['CHUM', 'HGJ-', 'CHUP', 'HMR-']   # no MDA, no CHUS
PRIMARY_EXT_PREFIX    = 'CHUS'

SENS1_DEV_PREFIXES    = ['CHUM', 'HGJ-', 'CHUS', 'HMR-']   # Option3: swap CHUP<->CHUS
SENS1_EXT_PREFIX      = 'CHUP'

SENS2_DEV_PREFIXES    = ['CHUM', 'MDA-', 'HGJ-', 'CHUP', 'HMR-']  # full MDA
SENS2_EXT_PREFIX      = 'CHUS'

CLINICAL_COLS = ['PatientID', 'Age', 'Gender_Male', 'Treatment_CRT', 'HPV_binary', 'prefix']

# ============================================================================
# Verify inputs
# ============================================================================

print("=" * 80)
print("Task 2: HPV Status Classification — Cohort Split")
print("=" * 80)

for p in [PATH_CLINICAL, PATH_CT_DEV, PATH_CT_EXT, PATH_PT_DEV, PATH_PT_EXT]:
    if not p.exists():
        print(f"[ERROR] File not found: {p}")
        sys.exit(1)
    print(f"[OK] {p.name}")

# ============================================================================
# Load and prepare clinical labels
# ============================================================================

print("\n--- Clinical labels ---")
clin = pd.read_csv(PATH_CLINICAL)
print(f"Clinical file: {len(clin)} patients (Has_GTVn=1 all)")

# HPV binary label
clin['HPV_binary'] = clin.apply(
    lambda r: 1 if r['HPV_Positive'] == 1
    else (0 if (r['HPV_Positive'] == 0 and r['HPV_Unknown'] == 0)
          else np.nan),
    axis=1
)
clin['prefix'] = clin['PatientID'].str[:4]

n_pos     = int((clin['HPV_binary'] == 1).sum())
n_neg     = int((clin['HPV_binary'] == 0).sum())
n_unknown = int(clin['HPV_binary'].isna().sum())
print(f"  HPV+:     {n_pos}")
print(f"  HPV-:     {n_neg}")
print(f"  Unknown:  {n_unknown} (excluded from all supervised steps)")

# ============================================================================
# Load radiomics — concatenate 2v dev + ext for each modality
# ============================================================================

print("\n--- Radiomics files ---")
ct_all = pd.concat([pd.read_csv(PATH_CT_DEV), pd.read_csv(PATH_CT_EXT)], ignore_index=True)
pt_all = pd.concat([pd.read_csv(PATH_PT_DEV), pd.read_csv(PATH_PT_EXT)], ignore_index=True)

ct_all['prefix'] = ct_all['PatientID'].str[:4]
pt_all['prefix'] = pt_all['PatientID'].str[:4]

print(f"  CT total: {len(ct_all)} patients, {ct_all.shape[1]-2} radiomics features")
print(f"  PT total: {len(pt_all)} patients, {pt_all.shape[1]-2} radiomics features")

# Centre breakdown across all 2v patients
for label, df in [('CT', ct_all), ('PT', pt_all)]:
    counts = df['prefix'].value_counts().sort_index()
    print(f"\n  {label} centre breakdown (all 2v patients):")
    for pfx, n in counts.items():
        print(f"    {pfx}: {n}")

# ============================================================================
# Core split function
# ============================================================================

def build_split(rad_df: pd.DataFrame,
                clin_df: pd.DataFrame,
                dev_prefixes: list,
                ext_prefix: str,
                split_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Subset radiomics to dev/ext prefixes, merge HPV labels + clinical features,
    keep HPV-known patients only, verify no overlap.
    Returns (dev_hpv, ext_hpv).
    """
    dev_rad = rad_df[rad_df['prefix'].isin(dev_prefixes)].copy()
    ext_rad = rad_df[rad_df['prefix'] == ext_prefix].copy()

    # Merge clinical (left join — patients without clinical are dropped via notna filter)
    dev = dev_rad.merge(clin_df[CLINICAL_COLS], on='PatientID', how='left', suffixes=('', '_clin'))
    ext = ext_rad.merge(clin_df[CLINICAL_COLS], on='PatientID', how='left', suffixes=('', '_clin'))

    # Remove duplicate prefix column from merge if it appeared
    for df in [dev, ext]:
        dup_cols = [c for c in df.columns if c.endswith('_clin')]
        if dup_cols:
            df.drop(columns=dup_cols, inplace=True)

    # Keep HPV-known only
    dev_hpv = dev[dev['HPV_binary'].notna()].copy()
    ext_hpv = ext[ext['HPV_binary'].notna()].copy()

    # Verify no patient overlap
    overlap = set(dev_hpv['PatientID']) & set(ext_hpv['PatientID'])
    if overlap:
        print(f"  [ERROR] {split_name}: {len(overlap)} patients in both dev and ext!")
        sys.exit(1)

    return dev_hpv, ext_hpv

# ============================================================================
# Report helper
# ============================================================================

def report_split(name: str, modality: str, dev: pd.DataFrame, ext: pd.DataFrame | None = None):
    def centre_lines(df):
        lines = []
        for pfx in sorted(df['prefix'].unique()):
            sub = df[df['prefix'] == pfx]
            n_p = int((sub['HPV_binary'] == 1).sum())
            n_n = int((sub['HPV_binary'] == 0).sum())
            lines.append(f"      {pfx}: {len(sub)} HPV-known ({n_p}+, {n_n}-)")
        return '\n'.join(lines)

    d_pos = int((dev['HPV_binary'] == 1).sum())
    d_neg = int((dev['HPV_binary'] == 0).sum())
    ratio = f"{d_pos/d_neg:.1f}:1" if d_neg > 0 else "inf"
    print(f"\n  [{modality}] {name}")
    print(f"    Dev: {len(dev)} HPV-known | HPV+: {d_pos} | HPV-: {d_neg} | ratio: {ratio} | HPV-%: {d_neg/len(dev)*100:.1f}%")
    print(centre_lines(dev))
    if ext is not None:
        e_pos = int((ext['HPV_binary'] == 1).sum())
        e_neg = int((ext['HPV_binary'] == 0).sum())
        print(f"    Ext: {len(ext)} HPV-known | HPV+: {e_pos} | HPV-: {e_neg}")
        print(centre_lines(ext))

# ============================================================================
# Build and save all splits
# ============================================================================

print("\n" + "=" * 80)
print("Building splits")
print("=" * 80)

splits = {}

for modality, rad_all in [('CT', ct_all), ('PT', pt_all)]:

    # --- Primary: no-MDA A1 ---
    dev, ext = build_split(rad_all, clin, PRIMARY_DEV_PREFIXES, PRIMARY_EXT_PREFIX, f'primary_{modality}')
    splits[f'{modality}_primary'] = (dev, ext)
    report_split('Primary no-MDA A1', modality, dev, ext)

    # --- Sensitivity 1: no-MDA Option3 ---
    dev1, ext1 = build_split(rad_all, clin, SENS1_DEV_PREFIXES, SENS1_EXT_PREFIX, f'sens1_{modality}')
    splits[f'{modality}_sens1'] = (dev1, ext1)
    report_split('Sensitivity 1 no-MDA Option3', modality, dev1, ext1)

    # --- Sensitivity 2: full-MDA A1 (upper-bound only) ---
    dev2, _ = build_split(rad_all, clin, SENS2_DEV_PREFIXES, SENS2_EXT_PREFIX, f'sens2_{modality}')
    # ext is same CHUS as primary — no need to re-save; reuse primary ext
    splits[f'{modality}_sens2'] = (dev2, ext)
    report_split('Sensitivity 2 full-MDA A1 (upper-bound only)', modality, dev2)

# ============================================================================
# EPV summary
# ============================================================================

print("\n" + "=" * 80)
print("EPV Summary (minority class = HPV-)")
print("=" * 80)
print(f"{'Split':<42} {'Dev HPV-':>9} {'EPV@3':>7} {'EPV@5':>7}")
print("-" * 68)

for modality in ['CT', 'PT']:
    for tag, label in [
        ('primary', 'Primary no-MDA A1'),
        ('sens1',   'Sensitivity 1 no-MDA Option3'),
        ('sens2',   'Sensitivity 2 full-MDA A1'),
    ]:
        dev, _ = splits[f'{modality}_{tag}']
        n_neg = int((dev['HPV_binary'] == 0).sum())
        epv3 = n_neg / 3
        epv5 = n_neg / 5
        flag = ' <-- primary' if tag == 'primary' else ''
        print(f"  {modality} {label:<38} {n_neg:>9} {epv3:>7.1f} {epv5:>7.1f}{flag}")

print()
print("  EPV >= 5 required for model with 5 features (hard ceiling).")
print("  EPV >= 3 minimum floor for sensitivity splits.")

# ============================================================================
# Save outputs
# ============================================================================

print("\n" + "=" * 80)
print("Saving outputs")
print("=" * 80)

save_map = {
    'CT_primary':  ('12_mar_task2_CT_primary_dev.csv',           '12_mar_task2_CT_primary_ext.csv'),
    'PT_primary':  ('12_mar_task2_PT_primary_dev.csv',           '12_mar_task2_PT_primary_ext.csv'),
    'CT_sens1':    ('12_mar_task2_CT_sens1_dev.csv',             '12_mar_task2_CT_sens1_ext.csv'),
    'PT_sens1':    ('12_mar_task2_PT_sens1_dev.csv',             '12_mar_task2_PT_sens1_ext.csv'),
    'CT_sens2':    ('12_mar_task2_CT_sens2_fullMDA_dev.csv',     None),   # ext = same as primary
    'PT_sens2':    ('12_mar_task2_PT_sens2_fullMDA_dev.csv',     None),
}

drop_cols = ['prefix']   # internal helper column; not needed downstream

for key, (dev_fname, ext_fname) in save_map.items():
    dev, ext = splits[key]

    dev_out = OUTPUT_DIR / dev_fname
    dev.drop(columns=[c for c in drop_cols if c in dev.columns]).to_csv(dev_out, index=False)
    print(f"  Saved: {dev_fname}  ({len(dev)} rows)")

    if ext_fname is not None:
        ext_out = OUTPUT_DIR / ext_fname
        ext.drop(columns=[c for c in drop_cols if c in ext.columns]).to_csv(ext_out, index=False)
        print(f"  Saved: {ext_fname}  ({len(ext)} rows)")

print(f"\n  Note: Sensitivity 2 ext = same CHUS cohort as primary ext.")
print(f"        Use 12_mar_task2_CT_primary_ext.csv / PT variant for full-MDA validation.")

# ============================================================================
# Final summary
# ============================================================================

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print("""
Split strategy (6_mar_task2.md):
  Primary    : Dev = CID1+CID6+CID2+CID7 (no MDA), Ext = CHUS
  Sensitivity 1 : Dev = CID1+CID6+CID3+CID7 (no MDA), Ext = CHUP (15/15 balanced)
  Sensitivity 2 : Dev = CID1+CID5+CID6+CID2+CID7 (full MDA), Ext = CHUS
                  Upper-bound comparison only — not for biological interpretation

Next step: 12_mar_task2_prefilter.py
  ICC + variance + correlation pre-filters on Task 2 dev cohort (primary CT/PT dev)
  Starting from 2818 raw radiomics features -> ~200-500 after pre-filtering
  Do NOT reuse Task 1 pre-filters (different cohort, different centre composition)
""")
print("=" * 80)
