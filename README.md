# optuna-ensemble

[![Python](https://img.shields.io/badge/Python-3.x-3776AB)](README.md)
[![Optuna](https://img.shields.io/badge/Hyperparameter%20Tuning-Optuna-4c78a8)](README.md)
[![Ensemble](https://img.shields.io/badge/Strategy-Soft%20Voting-2ea043)](README.md)

This repository contains a feature-level ensemble pipeline for binary classification of speech-derived features, tuned with Optuna and evaluated with soft voting.

Each training script loads `.npy` feature matrices from a shared feature directory, tunes a classifier on the ADReSSo21 train split, predicts the test split, and exports patient-level results for later analysis in the notebook.

## Table of Contents

- [What This Project Does](#what-this-project-does)
- [Why This Project Is Useful](#why-this-project-is-useful)
- [Architecture Overview](#architecture-overview)
- [Getting Started](#getting-started)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)
- [Where To Get Help](#where-to-get-help)
- [Maintainers and Contributions](#maintainers-and-contributions)

## What This Project Does

`optuna-ensemble` compares several classifiers on the same feature bank and combines their predictions with soft voting.

The main scripts are:

- `classify_dt_new_OPTUNA.py`: Decision Tree ensemble pipeline.
- `classify_dt_RF_new_OPTUNA.py`: Random Forest ensemble pipeline.
- `classify_dt_SVC_new_OPTUNA.py`: Support Vector Classifier ensemble pipeline.
- `classify_dt_XGBoost_new_OPTUNA.py`: XGBoost ensemble pipeline.
- `classify_dt_CatBoost_new_OPTUNA.py`: CatBoost ensemble pipeline.
- `classify_dt_LightGBM_new_OPTUNA.py`: LightGBM ensemble pipeline.

Common behavior across the scripts:

- Load feature sets from `BASE_FEATURES_DIR`.
- Read train/test folders named `ADReSSo21_train` and `ADReSSo21_test`.
- Tune hyperparameters with Optuna, usually for 135 trials.
- Apply lightweight feature filtering for very high-dimensional inputs.
- Export per-patient predictions to `ensemble_results_*.csv`.
- Print accuracy and a classification report for `HC` vs `AD`.

## Why This Project Is Useful

This repository is useful if you want to:

- Compare multiple classical machine learning models on the same speech-feature pipeline.
- Reproduce a binary health classification experiment with Optuna-based tuning.
- Study how feature-level models can be combined with soft voting.
- Inspect results with a dedicated analysis notebook and saved CSV outputs.

## Architecture Overview

The repository is organized around three steps:

1. Feature loading: each script walks through a feature folder, reads `.npy` arrays, and infers labels from folder names or file names.
2. Model tuning: Optuna explores a compact hyperparameter space for the selected classifier.
3. Ensemble aggregation: all feature-specific models emit probabilities, which are averaged per patient to form the final prediction.

High-dimensional feature sets, such as `ComParE_2016_6k`, are reduced with a combination of `VarianceThreshold` and `SelectKBest` in several scripts so the tuning loop stays practical.

## Getting Started

### Prerequisites

- Python 3.9+ recommended
- Packages used by the scripts, including `numpy`, `pandas`, `scikit-learn`, `optuna`, and the model-specific libraries:
	- `xgboost`
	- `lightgbm`
	- `catboost`
	- `joblib`

### Configure the data path

All training scripts currently point `BASE_FEATURES_DIR` to:

`/home/dani/Documentos/PEP/admodel-master-version/data/features`

Update that constant in the script you want to run so it matches your local feature directory.

### Run one ensemble script

From the repository root:

```bash
python src/classify_dt_LightGBM_new_OPTUNA.py
```

You can swap in any of the other `classify_dt_*.py` files to run a different classifier.

## Usage Examples

### 1) Run the Random Forest pipeline

```bash
python src/classify_dt_RF_new_OPTUNA.py
```

This script tunes a Random Forest for each feature family and writes a result file such as `ensemble_results_RF.csv`.

### 2) Analyze the saved ensemble outputs

Open `analise_ensemble.ipynb` after generating the CSV files. The notebook loads the exported results, computes metrics, and plots comparisons between model families.

The notebook expects files such as:

- `ensemble_results_dt.csv`
- `ensemble_results_RF.csv`
- `ensemble_results_SVC.csv`
- `ensemble_results_XG.csv`
- `ensemble_results_CB.csv`
- `ensemble_results_GBM.csv`

## Project Structure

```text
.
├── README.md                               # Project overview and usage guide
├── output.png                              # Example graph of finalized results
├── nb/                                     # Analysis and visualization of results
│   └── analise_ensemble.ipynb              # Notebook for comparing exported ensemble results
├── src/                                    # Main pipelines (The processing 'Engine')
│   ├── classify_dt_new_OPTUNA.py           # Decision Tree + Optuna pipeline
│   ├── classify_dt_RF_new_OPTUNA.py        # Random Forest + Optuna pipeline
│   ├── classify_dt_SVC_new_OPTUNA.py       # SVC + Optuna pipeline
│   ├── classify_dt_XGBoost_new_OPTUNA.py   # XGBoost + Optuna pipeline
│   ├── classify_dt_CatBoost_new_OPTUNA.py  # CatBoost + Optuna pipeline
│   └── classify_dt_LightGBM_new_OPTUNA.py  # LightGBM + Optuna pipeline
├── logs/                                   # Logs from previous runs (for traceability)
│   ├── logs_OPTUNA.txt
│   ├── logs_RF_OPTUNA.txt
│   ├── logs_SVC_OPTUNA.txt
│   ├── logs_XG_OPTUNA.txt
│   ├── logs_CB_OPTUNA.txt
│   └── logs_GBM_OPTUNA.txt
└── ...
```

### File notes

- `classify_dt_new_OPTUNA.py` is the Decision Tree implementation and writes `ensemble_results_dt.csv`.
- `classify_dt_RF_new_OPTUNA.py` is the Random Forest implementation and writes `ensemble_results_RF.csv`.
- `classify_dt_SVC_new_OPTUNA.py` is the SVC implementation and writes `ensemble_results_SVC.csv`.
- `classify_dt_XGBoost_new_OPTUNA.py` is the XGBoost implementation and writes `ensemble_results_XG.csv`.
- `classify_dt_CatBoost_new_OPTUNA.py` is the CatBoost implementation and writes `ensemble_results_CB.csv`.
- `classify_dt_LightGBM_new_OPTUNA.py` is the LightGBM implementation and writes `ensemble_results_GBM.csv`.
- `analise_ensemble.ipynb` reads those outputs and turns them into summary tables and plots.
- `logs/` stores example execution logs that can help when comparing runs or troubleshooting.

## Where To Get Help

- Start with the script for the model you want to reproduce.
- Check the `logs/` directory if you need to compare your run against an earlier execution.
- Open `analise_ensemble.ipynb` to inspect the exported CSV files and validate the ensemble outputs.

## Maintainers

Maintainer:

- [@kydoa](https://github.com/kydoa)