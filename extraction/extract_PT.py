"""
Multi-Region PET (PT) Radiomics Feature Extraction for HECKTOR Dataset
Template for extracting PT features with same multi-region approach as CT

Author: Based on CT extraction pipeline
Date: December 16, 2024
"""

import os
import sys
from pathlib import Path
import logging
import warnings

# Suppress ITK version mismatch warnings (harmless)
warnings.filterwarnings('ignore', category=UserWarning, module='itk')
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '4'

# Import standard libraries FIRST
import numpy as np
import pandas as pd

# Try to import tqdm (optional)
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc="Processing"):
        print(f"{desc}...")
        return iterable

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Try importing dependencies
print("Checking dependencies...")

# 1. Try SimpleITK from system FIRST (avoid FAE version conflicts)
try:
    import SimpleITK as sitk
    print("[OK] SimpleITK loaded from system")
except ImportError as e:
    print(f"[ERROR] Cannot import SimpleITK: {e}")
    print("Trying FAE's SimpleITK...")
    try:
        fae_path = r"d:\Uppsala thesis\FAEv0.6.0\FAE"
        if fae_path not in sys.path:
            sys.path.insert(0, fae_path)
        import SimpleITK as sitk
        print("[OK] SimpleITK loaded from FAE")
    except ImportError:
        print("[ERROR] SimpleITK not found. Please install: pip install SimpleITK")
        sys.exit(1)

# 2. Try radiomics from system (should be installed via pip/conda)
try:
    from radiomics import featureextractor
    print("[OK] PyRadiomics loaded from system")
except ImportError as e:
    print(f"[ERROR] Cannot load PyRadiomics: {e}")
    print("\nPyRadiomics not found. Please install it:")
    print("  conda activate pyrad_env")
    print("  pip install pyradiomics --no-build-isolation")
    sys.exit(1)

print("\n" + "="*80)
print("All dependencies loaded successfully!")
print("="*80 + "\n")


def check_labels_in_mask(mask_path):
    """Check which labels exist in segmentation"""
    try:
        mask = sitk.ReadImage(str(mask_path))
        mask_array = sitk.GetArrayFromImage(mask)
        unique_labels = np.unique(mask_array)
        nonzero = unique_labels[unique_labels > 0]
        return sorted(nonzero.tolist())
    except Exception as e:
        logger.error(f"Error reading mask {mask_path}: {e}")
        return None


def extract_features_for_label(image_path, mask_path, label_value, param_file):
    """Extract radiomics features for a specific label"""
    try:
        extractor = featureextractor.RadiomicsFeatureExtractor(param_file)
        extractor.settings['label'] = label_value
        result = extractor.execute(str(image_path), str(mask_path), label=label_value)
        features = {k: v for k, v in result.items() if 'diagnostics' not in k}
        return features
    except Exception as e:
        logger.error(f"Extraction failed for label {label_value}: {e}")
        return None


def process_hecktor_task1_PT(data_root, param_file, output_folder):
    """
    Process HECKTOR Task 1 dataset for PET (PT) images

    File structure:
    - Task 1/CHUM-001/CHUM-001__PT.nii.gz  (PET image)
    - Task 1/CHUM-001/CHUM-001.nii.gz      (segmentation)
    """
    task_path = Path(data_root) / "Task 1"
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    if not task_path.exists():
        logger.error(f"Task 1 path not found: {task_path}")
        return

    case_folders = sorted([f for f in task_path.iterdir() if f.is_dir()])
    logger.info(f"Found {len(case_folders)} cases in Task 1")

    results = {
        'label1_GTVp': {'cases': [], 'features': [], 'feature_names': None},
        'label2_GTVn': {'cases': [], 'features': [], 'feature_names': None},
        'combined': {'cases': [], 'features': [], 'feature_names': None}
    }

    errors = []
    stats = {'both_labels': 0, 'label1_only': 0, 'label2_only': 0, 'errors': 0}

    for case_folder in tqdm(case_folders, desc="Extracting PT features"):
        case_name = case_folder.name

        # PET IMAGE (different from CT)
        pt_file = case_folder / f"{case_name}__PT.nii.gz"
        seg_file = case_folder / f"{case_name}.nii.gz"

        if not pt_file.exists():
            logger.warning(f"{case_name}: PT image not found - {pt_file.name}")
            errors.append(f"{case_name}: PT image not found")
            stats['errors'] += 1
            continue

        if not seg_file.exists():
            logger.warning(f"{case_name}: Segmentation not found - {seg_file.name}")
            errors.append(f"{case_name}: Segmentation not found")
            stats['errors'] += 1
            continue

        available_labels = check_labels_in_mask(seg_file)
        if available_labels is None:
            errors.append(f"{case_name}: Could not read segmentation")
            stats['errors'] += 1
            continue

        has_label1 = 1 in available_labels
        has_label2 = 2 in available_labels

        if has_label1 and has_label2:
            stats['both_labels'] += 1
        elif has_label1:
            stats['label1_only'] += 1
        elif has_label2:
            stats['label2_only'] += 1

        # Extract Label 1 (GTVp - primary tumor)
        features_l1 = None
        if has_label1:
            print(f"\n{'='*60}\nProcessing: {case_name} - Label 1 (GTVp) - PT\n{'='*60}")
            features_l1 = extract_features_for_label(pt_file, seg_file, 1, param_file)
            if features_l1:
                if results['label1_GTVp']['feature_names'] is None:
                    results['label1_GTVp']['feature_names'] = list(features_l1.keys())
                results['label1_GTVp']['cases'].append(case_name)
                results['label1_GTVp']['features'].append(list(features_l1.values()))

        # Extract Label 2 (GTVn - lymph nodes)
        features_l2 = None
        if has_label2:
            print(f"\n{'='*60}\nProcessing: {case_name} - Label 2 (GTVn) - PT\n{'='*60}")
            features_l2 = extract_features_for_label(pt_file, seg_file, 2, param_file)
            if features_l2:
                if results['label2_GTVn']['feature_names'] is None:
                    results['label2_GTVn']['feature_names'] = list(features_l2.keys())
                results['label2_GTVn']['cases'].append(case_name)
                results['label2_GTVn']['features'].append(list(features_l2.values()))

        # Combined features
        if has_label1 and has_label2 and features_l1 and features_l2:
            combined_feat_names = [f"GTVp_{k}" for k in features_l1.keys()] + \
                                 [f"GTVn_{k}" for k in features_l2.keys()]
            combined_feat_values = list(features_l1.values()) + list(features_l2.values())

            if results['combined']['feature_names'] is None:
                results['combined']['feature_names'] = combined_feat_names
            results['combined']['cases'].append(case_name)
            results['combined']['features'].append(combined_feat_values)

    # Save results
    print("\n" + "="*80)
    print("EXTRACTION COMPLETE - PET (PT)")
    print("="*80)
    print(f"\nStatistics:")
    print(f"  Cases with both GTVp + GTVn: {stats['both_labels']}")
    print(f"  Cases with GTVp only: {stats['label1_only']}")
    print(f"  Cases with GTVn only: {stats['label2_only']}")
    print(f"  Errors: {stats['errors']}")

    # Save Label 1 (GTVp)
    if results['label1_GTVp']['cases']:
        df = pd.DataFrame(
            results['label1_GTVp']['features'],
            columns=results['label1_GTVp']['feature_names'],
            index=results['label1_GTVp']['cases']
        )
        df.index.name = 'CaseID'
        output_file = output_folder / "PT_features_GTVp_Label1.csv"
        df.to_csv(output_file)
        print(f"\n[SAVED] GTVp (Primary Tumor) PT features: {output_file}")
        print(f"   Shape: {df.shape} ({df.shape[0]} cases x {df.shape[1]} features)")

    # Save Label 2 (GTVn)
    if results['label2_GTVn']['cases']:
        df = pd.DataFrame(
            results['label2_GTVn']['features'],
            columns=results['label2_GTVn']['feature_names'],
            index=results['label2_GTVn']['cases']
        )
        df.index.name = 'CaseID'
        output_file = output_folder / "PT_features_GTVn_Label2.csv"
        df.to_csv(output_file)
        print(f"[SAVED] GTVn (Lymph Nodes) PT features: {output_file}")
        print(f"   Shape: {df.shape} ({df.shape[0]} cases x {df.shape[1]} features)")

    # Save Combined
    if results['combined']['cases']:
        df = pd.DataFrame(
            results['combined']['features'],
            columns=results['combined']['feature_names'],
            index=results['combined']['cases']
        )
        df.index.name = 'CaseID'
        output_file = output_folder / "PT_features_Combined_GTVp+GTVn.csv"
        df.to_csv(output_file)
        print(f"[SAVED] Combined (GTVp+GTVn) PT features: {output_file}")
        print(f"   Shape: {df.shape} ({df.shape[0]} cases x {df.shape[1]} features)")

    # Save error log
    if errors:
        error_file = output_folder / "PT_extraction_errors.txt"
        with open(error_file, 'w') as f:
            f.write('\n'.join(errors))
        print(f"\n[WARNING] Error log saved: {error_file}")

    return results, stats


# Main execution
if __name__ == "__main__":
    print("="*80)
    print("HECKTOR Task 1 Multi-Region PET (PT) Feature Extraction")
    print("="*80)
    print("\nThis script extracts PET features from:")
    print("  - Label 1 (GTVp): Primary tumor")
    print("  - Label 2 (GTVn): Lymph nodes")
    print("  - Combined: Both regions (features concatenated)")
    print("="*80)

    # Configuration
    DATA_ROOT = r"d:\Uppsala thesis\HECKTOR 2025 Training Data"
    PARAM_FILE = r"d:\Uppsala thesis\Dec_2025\4 Dec radiomics_config_param.yaml"
    OUTPUT_FOLDER = r"d:\Uppsala thesis\MultiRegion_Features"

    # Extract PT features
    print("\n" + "="*80)
    print("Extracting PET (PT) features...")
    print("="*80)

    results, stats = process_hecktor_task1_PT(
        DATA_ROOT,
        PARAM_FILE,
        OUTPUT_FOLDER
    )

    print("\n" + "="*80)
    print("ALL DONE!")
    print("="*80)
    print(f"\nOutput folder: {OUTPUT_FOLDER}")
    print("\nGenerated files:")
    print("  1. PT_features_GTVp_Label1.csv      <- Primary tumor (PET)")
    print("  2. PT_features_GTVn_Label2.csv      <- Lymph nodes (PET)")
    print("  3. PT_features_Combined_GTVp+GTVn.csv <- Both regions combined (PET)")
