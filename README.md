# Binding-affinity-prediction
# Protein-Ligand Binding Affinity Prediction Models

A comprehensive machine learning framework for predicting protein-ligand binding affinity using multiple model architectures with mutation-aware and ligand-context-aware approaches.

## Project Overview

This project implements a two-stage training architecture:
- **Stage A (Ligand Baseline)**: Predicts ligand-level mean affinity using molecular descriptors
- **Stage B (Mutation Residual)**: Predicts mutation effects using delta protein features

Models are trained with group-stratified splitting by canonical SMILES to prevent data leakage.

## Supported Models

### Primary Models (Ready to Use)
- **MLP** (`mlp_mutation_delta.py`) - Neural network with warm-start training
- **XGBoost** (`xgb_mutation_delta.py`) - Gradient boosting with early stopping
- **LightGBM** (`lgbm_mutation_delta.py`) - Fast gradient boosting
- **CatBoost** (`catboost_mutation_delta.py`) - Categorical boosting
- **Random Forest** (`random_forest_mutation_delta.py`) - Ensemble tree-based model with OOB estimation

### Ensemble & Advanced (Templates/Ready to Use)
- **Ensemble** (`ensemble_mutation_delta.py`) - Weighted ensemble of all models
- **GNN** (`gnn_mutation_delta.py`) - Graph neural networks (template)

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies
- Python 3.8+
- pandas, numpy, scikit-learn
- xgboost, lightgbm, catboost
- rdkit
- torch, torch-geometric (for GNN)

## Project Structure

```
.
├── README.md
├── requirements.txt
├── models/
│   ├── mlp_mutation_delta.py
│   ├── xgb_mutation_delta.py
│   ├── lgbm_mutation_delta.py
│   ├── catboost_mutation_delta.py
│   ├── gnn_mutation_delta.py
│   └── ensemble_mutation_delta.py
├── analysis/
│   └── shap_catboost_analysis.py
├── data/
│   └── data.csv
└── outputs/
    └── [model outputs and checkpoints]
```

## Data Format

Input CSV should contain:
- `SMILES`: SMILES string of the ligand
- `ActiveSiteSeq`: Protein active site sequence (amino acid letters)
- `Binding_affinity`: Target binding affinity value

## Usage

### Training Individual Models

```bash
# MLP model
python models/mlp_mutation_delta.py

# XGBoost model
python models/xgb_mutation_delta.py

# LightGBM model
python models/lgbm_mutation_delta.py

# CatBoost model
python models/catboost_mutation_delta.py

# GNN model
python models/gnn_mutation_delta.py
```

### Training Ensemble

```bash
python models/ensemble_mutation_delta.py
```

This will load all individual model bundles and train weighted ensemble.

### SHAP Analysis

```bash
python analysis/shap_catboost_analysis.py \
    --data_csv data/data.csv \
    --model_dir outputs/catboost_grouped_mutation_delta \
    --outdir outputs/shap_catboost_analysis
```

## Configuration

Each script has configurable parameters at the top:

```python
# Data paths
DATA_PATH = "data/data.csv"

# Output directory
OUTDIR = "outputs/[model_name]"

# Training parameters
TEST_SIZE = 0.10
SPLIT_SEED = 42
MODEL_SEED = 42
```

Adjust these before running for your specific use case.

## Output Files

Each model training script produces:

- `model_bundle.pkl` - Serialized model with preprocessing pipelines
- `report.json` - Training metadata and metrics
- `test_predictions.csv` - Predictions and decomposition for test set

## Model Architecture Details

### Stage A (Ligand Baseline)
- Features: Top-20 RDKit descriptors (selected via SHAP)
- Target: Ligand-level mean binding affinity
- Purpose: Captures intrinsic ligand properties

### Stage B (Mutation Residual)
- Features: Delta protein encoding (difference from reference sequence)
  - Changes: Position-specific mutation indicator
  - Properties: hydrophobicity, charge, aromaticity, polarity
- Target: Residual = actual - baseline prediction
- Purpose: Captures mutation-specific effects

## Protein Encoding

Amino acid properties used for delta encoding:
- **Hydrophobicity**: Kyte-Doolittle scale
- **Charge**: -1, 0, or +1
- **Aromaticity**: Binary (F, H, W, Y)
- **Polarity**: Binary (D, E, K, N, Q, R, S, T, W, Y)

## Reference Sequence

Automatically selected as the most frequent active site sequence in the dataset (WT proxy).

## Evaluation Metrics

All models are evaluated using:
- **R²**: Coefficient of determination
- **RMSE**: Root mean squared error
- **MAE**: Mean absolute error

## Group Stratified Splitting

To prevent data leakage:
- Splitting by canonical SMILES (LigandID)
- Same ligand appears only in train or test
- Prevents overfitting on ligand properties


