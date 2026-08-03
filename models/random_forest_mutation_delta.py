#!/usr/bin/env python3
"""
Mutation-aware Random Forest with two-stage architecture.

Stage A (Ligand Baseline):
    Features: Top-20 RDKit descriptors
    Target: Ligand-level mean binding affinity
    
Stage B (Mutation Residual):
    Features: Delta protein features (mutation-centric encoding)
    Target: Residual = actual affinity - baseline prediction

Key features:
    - Group split by LigandID (canonical SMILES) to prevent leakage
    - Protein encoded as DELTA vs reference active-site sequence
    - Random Forest with out-of-bag error estimation
    - Saves model_bundle.pkl for inference
"""

import os
import json
import pickle
import logging
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress RDKit warnings
RDLogger.DisableLog("rdApp.error")


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for Random Forest training."""
    
    # Paths
    DATA_PATH = "data/data.csv"
    LIGAND_IMPORTANCE_CSV = "outputs/shap_ligand_feature_importance.csv"
    OUTDIR = "outputs/random_forest_grouped_mutation_delta"
    
    # Feature selection
    TOP_N_DESCRIPTORS = 20
    
    # Data splitting
    TEST_SIZE = 0.10
    SPLIT_SEED = 42
    
    # Training
    MODEL_SEED = 42
    
    # Random Forest parameters
    N_ESTIMATORS = 200
    MAX_DEPTH = 20
    MIN_SAMPLES_SPLIT = 5
    MIN_SAMPLES_LEAF = 2
    N_JOBS = -1  # Use all CPU cores


# ============================================================================
# HELPERS
# ============================================================================

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics."""
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def clean_smiles(x: str) -> Optional[str]:
    """Clean and validate SMILES string."""
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    return s


def canonical_smiles_safe(smi: str) -> Optional[str]:
    """Convert SMILES to canonical form with error handling."""
    smi = clean_smiles(smi)
    if smi is None:
        return None
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return None


def load_top_descriptor_names(csv_path: str, top_n: int) -> List[str]:
    """Load top-N descriptor names from SHAP importance CSV."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Importance CSV not found: {csv_path}")
    
    imp = pd.read_csv(csv_path)
    required_cols = {"feature", "mean_abs_shap"}
    missing = required_cols - set(imp.columns)
    if missing:
        raise ValueError(f"CSV missing columns {missing}")
    
    imp = imp.dropna(subset=["feature", "mean_abs_shap"]).copy()
    imp["mean_abs_shap"] = pd.to_numeric(imp["mean_abs_shap"], errors="coerce")
    imp = imp.dropna(subset=["mean_abs_shap"])
    imp = imp.sort_values("mean_abs_shap", ascending=False)
    
    names = imp["feature"].astype(str).head(top_n).tolist()
    if not names:
        raise ValueError("No descriptor names found in importance CSV.")
    
    logger.info(f"Loaded {len(names)} top descriptors from {csv_path}")
    return names


def calc_selected_descriptors_safe(
    smiles: str,
    descriptor_names: List[str]
) -> Optional[Dict[str, float]]:
    """Calculate selected RDKit descriptors with error handling."""
    smiles = clean_smiles(smiles)
    if smiles is None:
        return None
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        out = {}
        for name in descriptor_names:
            func = getattr(Descriptors, name, None)
            if func is None:
                raise ValueError(f"Descriptor '{name}' not found in RDKit.")
            try:
                out[name] = float(func(mol))
            except Exception:
                out[name] = np.nan
        
        return out
    except Exception:
        return None


def median_impute(df_num: pd.DataFrame) -> pd.DataFrame:
    """Impute infinities and NaN values with median."""
    df_num = df_num.replace([np.inf, -np.inf], np.nan)
    return df_num.fillna(df_num.median(numeric_only=True))


# ============================================================================
# PROTEIN DELTA ENCODING
# ============================================================================

# Amino acid physicochemical properties
AA_PROPS = {
    "A": {"hydro":  1.8, "charge":  0, "arom": 0, "polar": 0},
    "C": {"hydro":  2.5, "charge":  0, "arom": 0, "polar": 0},
    "D": {"hydro": -3.5, "charge": -1, "arom": 0, "polar": 1},
    "E": {"hydro": -3.5, "charge": -1, "arom": 0, "polar": 1},
    "F": {"hydro":  2.8, "charge":  0, "arom": 1, "polar": 0},
    "G": {"hydro": -0.4, "charge":  0, "arom": 0, "polar": 0},
    "H": {"hydro": -3.2, "charge":  0, "arom": 1, "polar": 1},
    "I": {"hydro":  4.5, "charge":  0, "arom": 0, "polar": 0},
    "K": {"hydro": -3.9, "charge":  1, "arom": 0, "polar": 1},
    "L": {"hydro":  3.8, "charge":  0, "arom": 0, "polar": 0},
    "M": {"hydro":  1.9, "charge":  0, "arom": 0, "polar": 0},
    "N": {"hydro": -3.5, "charge":  0, "arom": 0, "polar": 1},
    "P": {"hydro": -1.6, "charge":  0, "arom": 0, "polar": 0},
    "Q": {"hydro": -3.5, "charge":  0, "arom": 0, "polar": 1},
    "R": {"hydro": -4.5, "charge":  1, "arom": 0, "polar": 1},
    "S": {"hydro": -0.8, "charge":  0, "arom": 0, "polar": 1},
    "T": {"hydro": -0.7, "charge":  0, "arom": 0, "polar": 1},
    "V": {"hydro":  4.2, "charge":  0, "arom": 0, "polar": 0},
    "W": {"hydro": -0.9, "charge":  0, "arom": 1, "polar": 1},
    "Y": {"hydro": -1.3, "charge":  0, "arom": 1, "polar": 1},
}
PROT_KEYS = ["hydro", "charge", "arom", "polar"]


def protein_delta_columns(seq_len: int) -> List[str]:
    """Generate column names for protein delta encoding."""
    cols = []
    for i in range(seq_len):
        cols.append(f"pos{i+1}_changed")
        for k in PROT_KEYS:
            cols.append(f"pos{i+1}_d{k}")
    return cols


def encode_delta_vs_reference(seq: str, ref_seq: str) -> np.ndarray:
    """Encode protein sequence as delta (difference) vs reference."""
    seq = str(seq).strip().upper()
    ref_seq = str(ref_seq).strip().upper()
    
    if len(seq) != len(ref_seq):
        raise ValueError(
            f"Sequence length mismatch: got {len(seq)}, "
            f"expected {len(ref_seq)}"
        )
    
    feats = []
    for i, (aa, ref_aa) in enumerate(zip(seq, ref_seq)):
        if aa not in AA_PROPS or ref_aa not in AA_PROPS:
            raise ValueError(
                f"Unknown amino acid at position {i+1}: {aa} or {ref_aa}"
            )
        
        feats.append(1.0 if aa != ref_aa else 0.0)
        
        for k in PROT_KEYS:
            delta = AA_PROPS[aa][k] - AA_PROPS[ref_aa][k]
            feats.append(float(delta))
    
    return np.asarray(feats, dtype=float)


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """Execute full training pipeline."""
    
    os.makedirs(Config.OUTDIR, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("Random Forest Mutation-Delta Training Pipeline")
    logger.info("=" * 70)
    
    # Load descriptor names
    top_desc_names = load_top_descriptor_names(
        Config.LIGAND_IMPORTANCE_CSV,
        Config.TOP_N_DESCRIPTORS
    )
    
    # Load data
    logger.info(f"Loading data from: {Config.DATA_PATH}")
    df = pd.read_csv(Config.DATA_PATH)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", na=False)].copy()
    
    required_cols = {"SMILES", "ActiveSiteSeq", "Binding_affinity"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    # Clean SMILES and create ligand groups
    df["SMILES"] = df["SMILES"].apply(clean_smiles)
    df = df.dropna(subset=list(required_cols)).reset_index(drop=True)
    
    df["LigandID"] = df["SMILES"].apply(canonical_smiles_safe)
    df = df.dropna(subset=["LigandID"]).reset_index(drop=True)
    
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Unique ligands: {df['LigandID'].nunique()}")
    logger.info(f"Unique sequences: {df['ActiveSiteSeq'].nunique()}")
    
    # Collapse duplicates
    df = df.groupby(
        ["LigandID", "SMILES", "ActiveSiteSeq"],
        as_index=False
    )["Binding_affinity"].mean()
    logger.info(f"After collapsing duplicates: {df.shape}")
    
    # Calculate ligand descriptors
    logger.info("Calculating ligand descriptors...")
    desc_series = df["SMILES"].apply(
        lambda s: calc_selected_descriptors_safe(s, top_desc_names)
    )
    ok = desc_series.notna()
    df = df.loc[ok].reset_index(drop=True)
    desc_series = desc_series.loc[ok].reset_index(drop=True)
    
    ligand_X = median_impute(
        pd.DataFrame(list(desc_series))
    ).astype(float)
    logger.info(f"Ligand features shape: {ligand_X.shape}")
    
    # Identify reference sequence
    ref_seq = df["ActiveSiteSeq"].value_counts().idxmax()
    logger.info(f"Reference sequence: {ref_seq}")
    
    # Encode protein features
    logger.info("Encoding protein delta features...")
    seqs = df["ActiveSiteSeq"].astype(str).tolist()
    seq_lens = {len(s.strip()) for s in seqs}
    if len(seq_lens) != 1:
        raise ValueError(f"ActiveSiteSeq lengths differ: {seq_lens}")
    seq_len = next(iter(seq_lens))
    
    mut_X = pd.DataFrame(
        np.vstack([encode_delta_vs_reference(s, ref_seq) for s in seqs]),
        columns=protein_delta_columns(seq_len),
    ).astype(float)
    logger.info(f"Protein features shape: {mut_X.shape}")
    
    # Prepare target and groups
    y = df["Binding_affinity"].astype(float).values
    groups = df["LigandID"].values
    
    # Group-stratified split
    logger.info(f"Splitting data (test_size={Config.TEST_SIZE})...")
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=Config.TEST_SIZE,
        random_state=Config.SPLIT_SEED
    )
    tr_idx, te_idx = next(gss.split(df, y, groups=groups))
    logger.info(f"Train: {len(tr_idx)}, Test: {len(te_idx)}")
    
    # ========================================================================
    # STAGE A: LIGAND BASELINE MODEL
    # ========================================================================
    logger.info("-" * 70)
    logger.info("STAGE A: Training ligand baseline model")
    logger.info("-" * 70)
    
    train_df = df.iloc[tr_idx].copy()
    train_df["y"] = y[tr_idx]
    
    lig_mean = train_df.groupby("LigandID", as_index=False)["y"].mean()
    
    train_lig = pd.concat(
        [train_df[["LigandID"]].reset_index(drop=True),
         ligand_X.iloc[tr_idx].reset_index(drop=True)],
        axis=1
    ).groupby("LigandID", as_index=False).first()
    
    base_train = lig_mean.merge(train_lig, on="LigandID", how="inner")
    
    X_base = base_train[top_desc_names].astype(float).values
    y_base = base_train["y"].astype(float).values
    
    logger.info(f"Stage A training set: {X_base.shape}")
    
    # Feature preprocessing
    base_selector = VarianceThreshold(0.0)
    Xb_sel = base_selector.fit_transform(X_base)
    
    logger.info(f"Features after variance threshold: {Xb_sel.shape}")
    
    # Train Random Forest (Stage A)
    base_model = RandomForestRegressor(
        n_estimators=Config.N_ESTIMATORS,
        max_depth=Config.MAX_DEPTH,
        min_samples_split=Config.MIN_SAMPLES_SPLIT,
        min_samples_leaf=Config.MIN_SAMPLES_LEAF,
        random_state=Config.MODEL_SEED,
        n_jobs=Config.N_JOBS,
        oob_score=True,  # Out-of-bag error estimation
    )
    
    logger.info("Training Stage A Random Forest...")
    base_model.fit(Xb_sel, y_base)
    
    oob_score_a = float(base_model.oob_score_) if hasattr(base_model, 'oob_score_') else 0.0
    logger.info(f"Stage A OOB R²: {oob_score_a:.6f}")
    
    # Predict baseline
    def baseline_pred_rows(row_idx):
        Xr = ligand_X.iloc[row_idx][top_desc_names].astype(float).values
        Xr_sel = base_selector.transform(Xr)
        return base_model.predict(Xr_sel)
    
    base_pred_tr = baseline_pred_rows(tr_idx)
    base_pred_te = baseline_pred_rows(te_idx)
    
    y_tr = y[tr_idx]
    y_te = y[te_idx]
    y_res_tr = y_tr - base_pred_tr
    
    # ========================================================================
    # STAGE B: MUTATION RESIDUAL MODEL
    # ========================================================================
    logger.info("-" * 70)
    logger.info("STAGE B: Training mutation residual model")
    logger.info("-" * 70)
    
    X_mut_tr = mut_X.iloc[tr_idx].values.astype(float)
    X_mut_te = mut_X.iloc[te_idx].values.astype(float)
    
    mut_selector = VarianceThreshold(0.0)
    Xm_tr_sel = mut_selector.fit_transform(X_mut_tr)
    Xm_te_sel = mut_selector.transform(X_mut_te)
    
    logger.info(f"Stage B training set: {Xm_tr_sel.shape}")
    logger.info(f"Features after variance threshold: {Xm_tr_sel.shape}")
    
    # Train Random Forest (Stage B)
    mut_model = RandomForestRegressor(
        n_estimators=Config.N_ESTIMATORS,
        max_depth=Config.MAX_DEPTH,
        min_samples_split=Config.MIN_SAMPLES_SPLIT,
        min_samples_leaf=Config.MIN_SAMPLES_LEAF,
        random_state=Config.MODEL_SEED,
        n_jobs=Config.N_JOBS,
        oob_score=True,
    )
    
    logger.info("Training Stage B Random Forest...")
    mut_model.fit(Xm_tr_sel, y_res_tr)
    
    oob_score_b = float(mut_model.oob_score_) if hasattr(mut_model, 'oob_score_') else 0.0
    logger.info(f"Stage B OOB R²: {oob_score_b:.6f}")
    
    # Predict residuals
    res_pred_te = mut_model.predict(Xm_te_sel)
    res_pred_tr = mut_model.predict(Xm_tr_sel)
    
    # Final predictions
    y_pred_te = base_pred_te + res_pred_te
    y_pred_tr = base_pred_tr + res_pred_tr
    
    # ========================================================================
    # EVALUATE & SAVE
    # ========================================================================
    logger.info("-" * 70)
    logger.info("Evaluation & Saving")
    logger.info("-" * 70)
    
    mets_train = evaluate(y_tr, y_pred_tr)
    mets_test = evaluate(y_te, y_pred_te)
    
    logger.info("Train metrics:")
    for key, val in mets_train.items():
        logger.info(f"  {key}: {val:.6f}")
    
    logger.info("Test metrics:")
    for key, val in mets_test.items():
        logger.info(f"  {key}: {val:.6f}")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        "SMILES": df.iloc[te_idx]["SMILES"].values,
        "ActiveSiteSeq": df.iloc[te_idx]["ActiveSiteSeq"].values,
        "LigandID": df.iloc[te_idx]["LigandID"].values,
        "y_test": y_te,
        "y_pred": y_pred_te,
        "pred_baseline_ligand": base_pred_te,
        "pred_residual_mutation": res_pred_te,
    })
    pred_path = os.path.join(Config.OUTDIR, "test_predictions.csv")
    predictions_df.to_csv(pred_path, index=False)
    logger.info(f"Saved predictions to: {pred_path}")
    
    # Save report
    report = {
        "stage_a_oob_r2": oob_score_a,
        "stage_b_oob_r2": oob_score_b,
        "ref_seq_used": ref_seq,
        "top_descriptor_names": top_desc_names,
        "split": {
            "test_size": Config.TEST_SIZE,
            "seed": Config.SPLIT_SEED,
            "grouped_by": "LigandID"
        },
        "metrics_train": mets_train,
        "metrics_test": mets_test,
        "n_rows_used": int(len(df)),
        "n_unique_ligands": int(df["LigandID"].nunique()),
        "n_unique_sequences": int(df["ActiveSiteSeq"].nunique()),
        "rf_config": {
            "n_estimators": Config.N_ESTIMATORS,
            "max_depth": Config.MAX_DEPTH,
            "min_samples_split": Config.MIN_SAMPLES_SPLIT,
            "min_samples_leaf": Config.MIN_SAMPLES_LEAF,
        },
    }
    report_path = os.path.join(Config.OUTDIR, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved report to: {report_path}")
    
    # Save model bundle
    bundle = {
        "ref_seq": ref_seq,
        "top_desc_names": top_desc_names,
        "base_selector": base_selector,
        "base_model": base_model,
        "mut_selector": mut_selector,
        "mut_model": mut_model,
        "report": report,
    }
    bundle_path = os.path.join(Config.OUTDIR, "model_bundle.pkl")
    with open(bundle_path, "wb") as f:
        pickle.dump(bundle, f)
    logger.info(f"Saved model bundle to: {bundle_path}")
    
    logger.info("=" * 70)
    logger.info("Training complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
