# OPSCC Radiomics Analysis Pipeline

A repo for Master thesis project in Precision Medicine — Uppsala University
In collaboration with AIbiomelab - College of Medicine, Taipei Medical University, Taiwan

## Overview

PET/CT radiomics pipeline for oropharyngeal squamous cell carcinoma (OPSCC):

- **Task 1 (RFS):** Recurrence-free survival prediction using multi-region PET/CT radiomics and clinical variables
- **Task 1C (Dose):** Exploratory dosiomics extension on the dose-eligible subset
- **Task 2 (HPV):** HPV status classification from baseline PET/CT radiomics
- **Task 2B (Bridge):** Pan-feature bridge analysis connecting Task 1 and Task 2 feature spaces

## Structure

| Folder | Content |
|--------|---------|
| `config/` | PyRadiomics extraction configuration |
| `extraction/` | Feature extraction scripts (GTVp, GTVn, RTdose) |
| `preprocessing/` | Clinical data processing and cohort split creation |
| `task1_rfs/` | Task 1 staged feature selection, combined search, post-study |
| `task1c_dose/` | Task 1C dosiomics scripts |
| `task2_hpv/` | Task 2 staged feature selection, recheck, post-study |
| `task2b_bridge/` | Task 2B bridge analysis development scripts |
| `utils/` | Shared feature selection and model utilities |

## Dependencies

```
python == 3.8
pyradiomics == 3.0.1
scikit-learn >= 1.3
scikit-survival >= 0.21
lifelines >= 0.30.0
shap >= 0.44
dice-ml >= 0.9
optuna >= 3.3
pandas >= 2.0
numpy >= 1.24
matplotlib >= 3.7
seaborn >= 0.12
scipy >= 1.11
xgboost == 3.0.5
joblib >= 1.3
pingouin >= 0.5.4
```

Install all at once:

```bash
pip install -r requirements.txt
```

## Data Availability

Patient imaging data and clinical data are not included in this repository. The dataset used in this study is the HECKTOR 2025 challenge dataset. Dataset information and access conditions are described at https://hecktor25.grand-challenge.org/dataset/. Data download requires registration and acceptance of the data-use agreement at https://hecktor25.grand-challenge.org/data-download/. Data preprocessing scripts and the PyRadiomics extraction configuration are provided in this repository for reproducibility.

## Ethical Statement

Patient data handling and analysis were conducted under the ethical permit and data access agreements described in the thesis.
