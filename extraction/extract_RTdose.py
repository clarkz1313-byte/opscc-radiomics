"""
RT Dose (Dosiomics) Feature Extraction for HECKTOR Dataset
Multi-region extraction from Task 2 RTDOSE maps using Task 1 segmentations

NOTE: Dose extraction differences from CT/PT:
- Different file structure (Task 2 instead of Task 1)
- Different file naming (__RTDOSE.nii.gz)
- Segmentation from Task 1 (not Task 2)
- Dose values (0-70 Gy) instead of HU/SUV

Author: Based on CT/PT extraction pipeline
Date: December 16, 2024 (Updated: January 8, 2026)
"""

import os
import sys
from pathlib import Path
import logging
import warnings

# Suppress ITK version mismatch warnings
warnings.filterwarnings('ignore', category=UserWarning, module='itk')
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '4'

# Import standard libraries
import numpy as np
import pandas as pd

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


def resample_mask_to_image(mask_image, reference_image):
    """
    Resample mask to match reference image's spatial properties
    
    Args:
        mask_image: SimpleITK Image (segmentation mask)
        reference_image: SimpleITK Image (RTDOSE image to match)
    
    Returns:
        Resampled mask image with same spatial properties as reference, or None if failed
    """
    try:
        # Get reference image properties
        reference_size = reference_image.GetSize()
        reference_spacing = reference_image.GetSpacing()
        reference_origin = reference_image.GetOrigin()
        reference_direction = reference_image.GetDirection()
        
        # Create resampler
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(reference_spacing)
        resampler.SetSize(reference_size)
        resampler.SetOutputDirection(reference_direction)
        resampler.SetOutputOrigin(reference_origin)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # Nearest neighbor for labels
        resampler.SetDefaultPixelValue(0)  # Background value
        
        # Resample mask
        resampled_mask = resampler.Execute(mask_image)
        
        # Validate: Check if resampled mask has any non-zero labels
        resampled_array = sitk.GetArrayFromImage(resampled_mask)
        unique_labels = np.unique(resampled_array)
        nonzero_labels = unique_labels[unique_labels > 0]
        
        if len(nonzero_labels) == 0:
            logger.warning("Resampled mask has no labels - images may not overlap physically")
            return None
        
        logger.info(f"Resampled mask contains labels: {sorted(nonzero_labels.tolist())}")
        return resampled_mask
    except Exception as e:
        logger.error(f"Resampling failed: {e}")
        return None


def extract_dose_features_for_label(dose_path, mask_path, label_value, param_file):
    """
    Extract dosiomics features from dose distribution for a specific label
    
    Args:
        dose_path: Path to RTDOSE.nii.gz file
        mask_path: Path to segmentation mask (from Task 1)
        label_value: Which label to extract (1=GTVp tumor, 2=GTVn nodes)
        param_file: PyRadiomics parameter YAML
    
    Returns:
        Dictionary of features (or None if failed)
    """
    try:
        # Load images
        dose_image = sitk.ReadImage(str(dose_path))
        mask_image = sitk.ReadImage(str(mask_path))
        
        # Check if spatial properties match
        dose_size = dose_image.GetSize()
        dose_spacing = dose_image.GetSpacing()
        dose_origin = dose_image.GetOrigin()
        dose_direction = dose_image.GetDirection()
        
        mask_size = mask_image.GetSize()
        mask_spacing = mask_image.GetSpacing()
        mask_origin = mask_image.GetOrigin()
        mask_direction = mask_image.GetDirection()
        
        # Check if resampling is needed
        needs_resample = (
            dose_size != mask_size or
            not np.allclose(dose_spacing, mask_spacing, rtol=1e-3) or
            not np.allclose(dose_origin, mask_origin, rtol=1e-3) or
            not np.allclose(dose_direction, mask_direction, rtol=1e-3)
        )
        
        temp_mask_file = None
        temp_dose_file = None
        if needs_resample:
            logger.info(f"Resampling mask to match RTDOSE image spatial properties...")
            logger.info(f"  RTDOSE: size={dose_size}, spacing={dose_spacing}, origin={dose_origin}")
            logger.info(f"  Mask:   size={mask_size}, spacing={mask_spacing}, origin={mask_origin}")
            
            # Try resampling mask to RTDOSE space first
            resampled_mask = resample_mask_to_image(mask_image, dose_image)
            
            if resampled_mask is None:
                # If resampling mask to RTDOSE fails (no overlap), try resampling RTDOSE to mask space
                logger.warning("Resampling mask to RTDOSE space failed - trying reverse resampling...")
                logger.info("Resampling RTDOSE image to match mask spatial properties...")
                
                # Resample RTDOSE to mask space (reuse same resampling logic)
                mask_size = mask_image.GetSize()
                mask_spacing = mask_image.GetSpacing()
                mask_origin = mask_image.GetOrigin()
                mask_direction = mask_image.GetDirection()
                
                resampler = sitk.ResampleImageFilter()
                resampler.SetOutputSpacing(mask_spacing)
                resampler.SetSize(mask_size)
                resampler.SetOutputDirection(mask_direction)
                resampler.SetOutputOrigin(mask_origin)
                resampler.SetInterpolator(sitk.sitkLinear)  # Linear for dose values
                resampler.SetDefaultPixelValue(0)
                
                resampled_dose = resampler.Execute(dose_image)
                
                # Validate resampled dose
                dose_array = sitk.GetArrayFromImage(resampled_dose)
                if np.all(dose_array == 0):
                    raise ValueError("Resampled RTDOSE is empty - images may not overlap physically")
                
                logger.info("Successfully resampled RTDOSE to mask space")
                
                # Save resampled RTDOSE to temporary file
                import tempfile
                temp_dose_file = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
                temp_dose_file.close()
                sitk.WriteImage(resampled_dose, temp_dose_file.name)
                dose_path = temp_dose_file.name
                # Use original mask (no resampling needed) - mask_path stays as original
            else:
                # Save resampled mask to temporary file
                import tempfile
                temp_mask_file = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
                temp_mask_file.close()
                sitk.WriteImage(resampled_mask, temp_mask_file.name)
                mask_path = temp_mask_file.name
                mask_image = resampled_mask
        
        # Extract features
        extractor = featureextractor.RadiomicsFeatureExtractor(param_file)
        extractor.settings['label'] = label_value

        # Dose-specific settings (optional - adjust as needed)
        # Dose values are typically in Gy (0-70 Gy for head & neck)
        # May need different binning than CT/PT (consider binWidth: 5 for dose)

        result = extractor.execute(str(dose_path), str(mask_path), label=label_value)
        
        # Clean up temporary files if created
        if temp_mask_file is not None:
            try:
                os.unlink(temp_mask_file.name)
            except Exception as cleanup_error:
                logger.warning(f"Could not delete temporary mask file: {cleanup_error}")
        
        if temp_dose_file is not None:
            try:
                os.unlink(temp_dose_file.name)
            except Exception as cleanup_error:
                logger.warning(f"Could not delete temporary dose file: {cleanup_error}")
        
        # Filter out diagnostic info
        features = {k: v for k, v in result.items() if 'diagnostics' not in k}
        return features
    except Exception as e:
        logger.error(f"Dose extraction failed for label {label_value}: {e}")
        return None


def process_hecktor_task2_dose(data_root, param_file, output_folder):
    """
    Process HECKTOR Task 2 for RT dose (dosiomics) features
    
    File structure:
    - Task 2/CHUM-001/CHUM-001__RTDOSE.nii.gz  (RT dose map)
    - Task 1/CHUM-001/CHUM-001.nii.gz           (segmentation mask - Labels 1 & 2)
    
    Uses Task 1 segmentations to extract dosiomics features from Task 2 RTDOSE maps
    """
    task2_path = Path(data_root) / "Task 2"
    task1_path = Path(data_root) / "Task 1"
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    if not task2_path.exists():
        logger.error(f"Task 2 path not found: {task2_path}")
        return

    if not task1_path.exists():
        logger.error(f"Task 1 path not found: {task1_path}")
        logger.error("Task 1 segmentations are required for RT dose extraction")
        return

    # Find all case folders in Task 2
    case_folders = sorted([f for f in task2_path.iterdir() if f.is_dir()])
    logger.info(f"Found {len(case_folders)} cases in Task 2")

    # Storage for results (multi-region like CT/PT)
    results = {
        'label1_GTVp': {'cases': [], 'features': [], 'feature_names': None},
        'label2_GTVn': {'cases': [], 'features': [], 'feature_names': None},
        'combined': {'cases': [], 'features': [], 'feature_names': None}
    }

    errors = []
    stats = {'both_labels': 0, 'label1_only': 0, 'label2_only': 0, 'errors': 0}

    # Process each case
    for case_folder in tqdm(case_folders, desc="Extracting RT dose features"):
        case_name = case_folder.name

        # RTDOSE file from Task 2
        dose_file = case_folder / f"{case_name}__RTDOSE.nii.gz"
        
        # Segmentation mask from Task 1 (required for masking)
        seg_file = task1_path / case_name / f"{case_name}.nii.gz"

        if not dose_file.exists():
            logger.warning(f"{case_name}: RTDOSE file not found - {dose_file.name}")
            errors.append(f"{case_name}: RTDOSE file not found")
            stats['errors'] += 1
            continue

        if not seg_file.exists():
            logger.warning(f"{case_name}: Segmentation not found in Task 1 - {seg_file.name}")
            errors.append(f"{case_name}: Segmentation not found")
            stats['errors'] += 1
            continue

        # Check what labels exist in segmentation
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
            print(f"\n{'='*60}\nProcessing: {case_name} - Label 1 (GTVp) - RT Dose\n{'='*60}")
            features_l1 = extract_dose_features_for_label(dose_file, seg_file, 1, param_file)
            if features_l1:
                if results['label1_GTVp']['feature_names'] is None:
                    results['label1_GTVp']['feature_names'] = list(features_l1.keys())
                results['label1_GTVp']['cases'].append(case_name)
                results['label1_GTVp']['features'].append(list(features_l1.values()))

        # Extract Label 2 (GTVn - lymph nodes)
        features_l2 = None
        if has_label2:
            print(f"\n{'='*60}\nProcessing: {case_name} - Label 2 (GTVn) - RT Dose\n{'='*60}")
            features_l2 = extract_dose_features_for_label(dose_file, seg_file, 2, param_file)
            if features_l2:
                if results['label2_GTVn']['feature_names'] is None:
                    results['label2_GTVn']['feature_names'] = list(features_l2.keys())
                results['label2_GTVn']['cases'].append(case_name)
                results['label2_GTVn']['features'].append(list(features_l2.values()))

        # Combined features (both regions analyzed separately, then concatenated)
        if has_label1 and has_label2 and features_l1 and features_l2:
            # Concatenate features from both regions
            combined_feat_names = [f"GTVp_{k}" for k in features_l1.keys()] + \
                                 [f"GTVn_{k}" for k in features_l2.keys()]
            combined_feat_values = list(features_l1.values()) + list(features_l2.values())

            if results['combined']['feature_names'] is None:
                results['combined']['feature_names'] = combined_feat_names
            results['combined']['cases'].append(case_name)
            results['combined']['features'].append(combined_feat_values)

    # Save results
    print("\n" + "="*80)
    print("EXTRACTION COMPLETE - RT Dose (Dosiomics)")
    print("="*80)
    print(f"\nStatistics (by segmentation labels in mask):")
    print(f"  Cases with both GTVp + GTVn: {stats['both_labels']}")
    print(f"  Cases with GTVp only: {stats['label1_only']}")
    print(f"  Cases with GTVn only: {stats['label2_only']}")
    print(f"  Errors: {stats['errors']}")
    n_comb = len(results['combined']['cases'])
    n_gtvp = len(results['label1_GTVp']['cases'])
    n_gtvn = len(results['label2_GTVn']['cases'])
    n_gtvp_only = n_gtvp - n_comb
    n_gtvn_only = n_gtvn - n_comb
    print(f"\nWritten to CSV (extraction succeeded):")
    print(f"  Combined (both): {n_comb}  |  GTVp CSV: {n_gtvp}  |  GTVn CSV: {n_gtvn}")
    print(f"  (GTVp-only rows: {n_gtvp_only}, GTVn-only rows: {n_gtvn_only}; union = {n_gtvp_only + n_gtvn_only + n_comb})")

    # Save Label 1 (GTVp - Primary Tumor)
    if results['label1_GTVp']['cases']:
        df = pd.DataFrame(
            results['label1_GTVp']['features'],
            columns=results['label1_GTVp']['feature_names'],
            index=results['label1_GTVp']['cases']
        )
        df.index.name = 'CaseID'
        output_file = output_folder / "Dose_features_GTVp_Label1.csv"
        df.to_csv(output_file)
        print(f"\n[SAVED] GTVp (Primary Tumor) RT dose features: {output_file}")
        print(f"   Shape: {df.shape} ({df.shape[0]} cases x {df.shape[1]} features)")

    # Save Label 2 (GTVn - Lymph Nodes)
    if results['label2_GTVn']['cases']:
        df = pd.DataFrame(
            results['label2_GTVn']['features'],
            columns=results['label2_GTVn']['feature_names'],
            index=results['label2_GTVn']['cases']
        )
        df.index.name = 'CaseID'
        output_file = output_folder / "Dose_features_GTVn_Label2.csv"
        df.to_csv(output_file)
        print(f"[SAVED] GTVn (Lymph Nodes) RT dose features: {output_file}")
        print(f"   Shape: {df.shape} ({df.shape[0]} cases x {df.shape[1]} features)")

    # Save Combined (GTVp + GTVn concatenated)
    if results['combined']['cases']:
        df = pd.DataFrame(
            results['combined']['features'],
            columns=results['combined']['feature_names'],
            index=results['combined']['cases']
        )
        df.index.name = 'CaseID'
        output_file = output_folder / "Dose_features_Combined_GTVp+GTVn.csv"
        df.to_csv(output_file)
        print(f"[SAVED] Combined (GTVp+GTVn) RT dose features: {output_file}")
        print(f"   Shape: {df.shape} ({df.shape[0]} cases x {df.shape[1]} features)")

    # Save error log
    if errors:
        error_file = output_folder / "Dose_extraction_errors.txt"
        with open(error_file, 'w') as f:
            f.write('\n'.join(errors))
        print(f"\n[WARNING] Error log saved: {error_file}")

    return results, stats


# Main execution
if __name__ == "__main__":
    print("="*80)
    print("HECKTOR Task 2 Multi-Region RT Dose (Dosiomics) Feature Extraction")
    print("="*80)
    print("\nThis script extracts RT dose features from:")
    print("  - Label 1 (GTVp): Primary tumor")
    print("  - Label 2 (GTVn): Lymph nodes")
    print("  - Combined: Both regions (features concatenated)")
    print("\nFile structure:")
    print("  - RTDOSE maps: Task 2/CASE-XXX/CASE-XXX__RTDOSE.nii.gz")
    print("  - Segmentations: Task 1/CASE-XXX/CASE-XXX.nii.gz")
    print("="*80)

    # Configuration
    DATA_ROOT = r"d:\Uppsala thesis\HECKTOR 2025 Training Data"
    PARAM_FILE = r"d:\Uppsala thesis\Dec_2025\4 Dec radiomics_config_param.yaml"
    OUTPUT_FOLDER = r"d:\Uppsala thesis\MultiRegion_Features"

    # Extract RT dose features
    print("\n" + "="*80)
    print("Extracting RT dose (dosiomics) features...")
    print("="*80)

    results, stats = process_hecktor_task2_dose(
        DATA_ROOT,
        PARAM_FILE,
        OUTPUT_FOLDER
    )

    print("\n" + "="*80)
    print("ALL DONE!")
    print("="*80)
    print(f"\nOutput folder: {OUTPUT_FOLDER}")
    print("\nGenerated files:")
    print("  1. Dose_features_GTVp_Label1.csv      <- Primary tumor (RT dose)")
    print("  2. Dose_features_GTVn_Label2.csv      <- Lymph nodes (RT dose)")
    print("  3. Dose_features_Combined_GTVp+GTVn.csv <- Both regions combined (RT dose)")
    print("\nNote: Consider adjusting binWidth in config file for dose values (0-70 Gy)")
