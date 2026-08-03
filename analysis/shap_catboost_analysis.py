#!/usr/bin/env python3
"""
SHAP Feature Importance Analysis for CatBoost Models.

Analyzes a trained CatBoost model using SHAP (SHapley Additive exPlanations)
to identify the most important features for binding affinity prediction.

Features:
    - Matches CatBoost training pipeline exactly
    - Computes SHAP values for feature importance
    - Extracts top-N most important features
    - Splits importance by ligand vs protein features
    - Saves CSV outputs for downstream use

Output Files:
    - shap_feature_importance.csv - All features ranked by importance
    - shap_ligand_feature_importance.csv - Top ligand features only
    - shap_protein_feature_importance.csv - Top protein features only
    - protein_position_importance.csv - Importance aggregated by position
    - top_features_summary.json - Summary of top features
"""

import os
import re
import json
import logging
import argparse
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import pickle

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors

try:
    import shap
except ImportError as e:
    raise ImportError(
        "shap is not installed. Install with:\n"
        "  pip install shap"
    ) from e


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
    """Configuration for SHAP analysis."""
    
    # Data and model paths
    DATA_PATH = "data/data.csv"
    MODEL_DIR = "outputs/catboost_grouped_mutation_delta"
    MODEL_BUNDLE_PATH = "outputs/catboost_grouped_mutation_delta/model_bundle.pkl"
    
    # SHAP analysis
    OUTDIR = "outputs/shap_analysis"
    SHAP_SAMPLE = 400  # How many rows to sample for SHAP computation
    TOP_N = 20  # Number of top features to extract
    RANDOM_SEED = 42


# ============================================================================
# FEATURE ENGINEERING (Match CatBoost training)
# ============================================================================

# Morgan fingerprint settings
FP_BITS = 2048
FP_RADIUS = 2

# Amino acid physicochemical properties
AA_PROPS = {
    "A": {"hydro": 1.8, "charge": 0, "arom": 0, "polar": 0},
    "C": {"hydro": 2.5, "charge": 0, "arom": 0, "polar": 0},
    "D": {"hydro": -3.5, "charge": -1, "arom": 0, "polar": 1},
    "E": {"hydro": -3.5, "charge": -1, "arom": 0, "polar": 1},
    "F": {"hydro": 2.8, "charge": 0, "arom": 1, "polar": 0},
    "G": {"hydro": -0.4, "charge": 0, "arom": 0, "polar": 0},
    "H": {"hydro": -3.2, "charge": 0, "arom": 1, "polar": 1},
    "I": {"hydro": 4.5, "charge": 0, "arom": 0, "polar": 0},
    "K": {"hydro": -3.9, "charge": 1, "arom": 0, "polar": 1},
    "L": {"hydro": 3.8, "charge": 0, "arom": 0, "polar": 0},
    "M": {"hydro": 1.9, "charge": 0, "arom": 0, "polar": 0},
    "N": {"hydro": -3.5, "charge": 0, "arom": 0, "polar": 1},
    "P": {"hydro": -1.6, "charge": 0, "arom": 0, "polar": 0},
    "Q": {"hydro": -3.5, "charge": 0, "arom": 0, "polar": 1},
    "R": {"hydro": -4.5, "charge": 1, "arom": 0, "polar": 1},
    "S": {"hydro": -0.8, "charge": 0, "arom": 0, "polar": 1},
    "T": {"hydro": -0.7, "charge": 0, "arom": 0, "polar": 1},
    "V": {"hydro": 4.2, "charge": 0, "arom": 0, "polar": 0},
    "W": {"hydro": -0.9, "charge": 0, "arom": 1, "polar": 1},
    "Y": {"hydro": -1.3, "charge": 0, "arom": 1, "polar": 1},
}
PROT_KEYS = ["hydro", "charge", "arom", "polar"]

RDKIT_DESC_NAMES = [
    "MolWt",
    "MolLogP",
    "NumHDonors",
    "NumHAcceptors",
    "TPSA",
    "NumRotatableBonds",
]


# ============================================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================================

def encode_active_site_seq(seq: str) -> np.ndarray:
    """
    Encode amino acid sequence as physicochemical properties.
    
    Args:
        seq: Amino acid sequence (uppercase)
        
    Returns:
        Feature array with shape (seq_len * 4,)
    """
    seq = str(seq).strip().upper()
    feats = []
    for aa in seq:
        if aa not in AA_PROPS:
            raise ValueError(f"Unknown AA '{aa}' in ActiveSiteSeq: {seq}")
        for k in PROT_KEYS:
            feats.append(AA_PROPS[aa][k])
    return np.asarray(feats, dtype=np.float32)


def protein_feature_columns(seq_len: int) -> List[str]:
    """Generate protein feature column names."""
    return [f"prot_pos{i+1}_{k}" for i in range(seq_len) for k in PROT_KEYS]


def morgan_fp(mol, n_bits: int = FP_BITS, radius: int = FP_RADIUS) -> np.ndarray:
    """
    Generate Morgan fingerprint for molecule.
    
    Args:
        mol: RDKit molecule
        n_bits: Number of bits
        radius: Radius of fingerprint
        
    Returns:
        Fingerprint as binary array
    """
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)


def rdkit_descriptors(mol) -> np.ndarray:
    """
    Calculate RDKit molecular descriptors.
    
    Args:
        mol: RDKit molecule
        
    Returns:
        Array of 6 descriptors
    """
    return np.array([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
    ], dtype=np.float32)


def ligand_feature_columns() -> List[str]:
    """Generate ligand feature column names."""
    fp_cols = [f"fp_{i}" for i in range(FP_BITS)]
    return fp_cols + RDKIT_DESC_NAMES


def featurize_compound(smiles: str, active_site_seq: str) -> Optional[np.ndarray]:
    """
    Convert SMILES and protein sequence to feature vector.
    
    Args:
        smiles: SMILES string
        active_site_seq: Protein sequence
        
    Returns:
        Feature vector or None if SMILES invalid
    """
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    
    fp = morgan_fp(mol)
    desc = rdkit_descriptors(mol)
    prot = encode_active_site_seq(active_site_seq)
    
    return np.concatenate([fp, desc, prot], axis=0)


def build_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Build feature matrix from SMILES and sequences.
    
    Args:
        df: DataFrame with SMILES, ActiveSiteSeq, Binding_affinity
        
    Returns:
        Tuple of (features DataFrame, target array)
    """
    required_cols = {"SMILES", "ActiveSiteSeq", "Binding_affinity"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )
    
    df = df.dropna(subset=list(required_cols)).reset_index(drop=True)
    
    # Verify all sequences have same length
    seqs = df["ActiveSiteSeq"].astype(str).tolist()
    seq_lens = {len(s.strip()) for s in seqs}
    if len(seq_lens) != 1:
        raise ValueError(f"ActiveSiteSeq lengths differ: {seq_lens}")
    
    seq_len = next(iter(seq_lens))
    
    logger.info(f"Building features from {len(df)} compounds...")
    logger.info(f"Sequence length: {seq_len}")
    
    # Featurize all compounds
    X_list = []
    y_list = []
    valid_count = 0
    
    for i, row in df.iterrows():
        vec = featurize_compound(row["SMILES"], row["ActiveSiteSeq"])
        if vec is None:
            continue
        
        X_list.append(vec)
        y_list.append(float(row["Binding_affinity"]))
        valid_count += 1
    
    if not X_list:
        raise ValueError("No valid molecules found after SMILES parsing")
    
    logger.info(f"Successfully featurized {valid_count} compounds")
    
    # Create feature DataFrame
    feature_names = ligand_feature_columns() + protein_feature_columns(seq_len)
    X = pd.DataFrame(np.vstack(X_list), columns=feature_names).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    
    logger.info(f"Feature matrix shape: {X.shape}")
    
    return X, y


# ============================================================================
# SHAP ANALYSIS
# ============================================================================

def mean_abs_shap(shap_values: np.ndarray) -> np.ndarray:
    """Compute mean absolute SHAP values across samples."""
    return np.abs(shap_values).mean(axis=0)


def compute_shap_importance(
    model,
    X_sample: np.ndarray,
    feature_names: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute SHAP values for model predictions.
    
    Args:
        model: Trained model
        X_sample: Feature matrix
        feature_names: List of feature names
        
    Returns:
        Tuple of (shap_values, mean_abs_shap_values)
    """
    logger.info(f"Computing SHAP values for {X_sample.shape[0]} samples...")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    # Handle extra columns (some CatBoost/SHAP combinations)
    if shap_values.ndim == 2 and shap_values.shape[1] == X_sample.shape[1] + 1:
        shap_values = shap_values[:, :-1]
    
    # Validate dimensions
    if shap_values.shape[1] != X_sample.shape[1]:
        raise ValueError(
            f"SHAP output shape mismatch: got {shap_values.shape[1]} features, "
            f"expected {X_sample.shape[1]}"
        )
    
    logger.info(f"SHAP values computed: shape {shap_values.shape}")
    
    imp = mean_abs_shap(shap_values)
    
    return shap_values, imp


# ============================================================================
# SAVING RESULTS
# ============================================================================

def save_importance_csv(
    feature_names: List[str],
    importance_values: np.ndarray,
    output_path: str,
    top_n: Optional[int] = None
) -> pd.DataFrame:
    """
    Save feature importance to CSV.
    
    Args:
        feature_names: List of feature names
        importance_values: Importance values
        output_path: Path to save CSV
        top_n: If specified, only save top N features
        
    Returns:
        DataFrame of importance values
    """
    df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": importance_values
    }).sort_values("mean_abs_shap", ascending=False)
    
    if top_n is not None:
        df = df.head(top_n)
    
    df.to_csv(output_path, index=False)
    logger.info(f"Saved importance to: {output_path}")
    
    return df


def split_ligand_protein_importance(
    feature_names: List[str],
    importance_values: np.ndarray
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split importance by ligand and protein features.
    
    Args:
        feature_names: List of all feature names
        importance_values: Importance for all features
        
    Returns:
        Tuple of (ligand_importance_df, protein_importance_df)
    """
    ligand_mask = [not name.startswith("prot_") for name in feature_names]
    protein_mask = [name.startswith("prot_") for name in feature_names]
    
    ligand_features = [name for name, mask in zip(feature_names, ligand_mask) if mask]
    ligand_importance = importance_values[ligand_mask]
    
    protein_features = [name for name, mask in zip(feature_names, protein_mask) if mask]
    protein_importance = importance_values[protein_mask]
    
    lig_df = pd.DataFrame({
        "feature": ligand_features,
        "mean_abs_shap": ligand_importance
    }).sort_values("mean_abs_shap", ascending=False)
    
    prot_df = pd.DataFrame({
        "feature": protein_features,
        "mean_abs_shap": protein_importance
    }).sort_values("mean_abs_shap", ascending=False)
    
    logger.info(f"Ligand features: {len(ligand_features)}")
    logger.info(f"Protein features: {len(protein_features)}")
    
    return lig_df, prot_df


def aggregate_by_protein_position(
    feature_names: List[str],
    importance_values: np.ndarray
) -> pd.DataFrame:
    """
    Aggregate protein feature importance by position.
    
    Args:
        feature_names: List of all feature names
        importance_values: Importance for all features
        
    Returns:
        DataFrame with importance aggregated by position
    """
    pos_to_vals = {}
    
    for name, importance in zip(feature_names, importance_values):
        if not name.startswith("prot_"):
            continue
        
        # Extract position from name like "prot_pos1_hydro"
        match = re.match(r"prot_pos(\d+)_", name)
        if not match:
            continue
        
        pos = int(match.group(1))
        pos_to_vals.setdefault(pos, []).append(importance)
    
    if not pos_to_vals:
        logger.warning("No protein positions found")
        return pd.DataFrame()
    
    pos_df = pd.DataFrame({
        "position": sorted(pos_to_vals.keys()),
        "mean_abs_shap_sum": [
            float(np.sum(pos_to_vals[p]))
            for p in sorted(pos_to_vals.keys())
        ],
        "mean_abs_shap_mean": [
            float(np.mean(pos_to_vals[p]))
            for p in sorted(pos_to_vals.keys())
        ],
    }).sort_values("mean_abs_shap_sum", ascending=False)
    
    logger.info(f"Aggregated to {len(pos_df)} protein positions")
    
    return pos_df


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Execute SHAP feature importance analysis."""
    
    parser = argparse.ArgumentParser(
        description="SHAP feature importance analysis for binding affinity prediction"
    )
    parser.add_argument("--data_csv", type=str, default=Config.DATA_PATH,
                        help="Path to input data CSV")
    parser.add_argument("--model_bundle", type=str, default=Config.MODEL_BUNDLE_PATH,
                        help="Path to model bundle pickle")
    parser.add_argument("--outdir", type=str, default=Config.OUTDIR,
                        help="Output directory for results")
    parser.add_argument("--shap_sample", type=int, default=Config.SHAP_SAMPLE,
                        help="Number of samples to use for SHAP analysis")
    parser.add_argument("--top_n", type=int, default=Config.TOP_N,
                        help="Number of top features to extract")
    parser.add_argument("--seed", type=int, default=Config.RANDOM_SEED,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("SHAP Feature Importance Analysis")
    logger.info("=" * 70)
    
    # ========================================================================
    # LOAD DATA & MODEL
    # ========================================================================
    logger.info("-" * 70)
    logger.info("Loading data and model...")
    logger.info("-" * 70)
    
    # Load data
    if not os.path.exists(args.data_csv):
        raise FileNotFoundError(f"Data file not found: {args.data_csv}")
    
    logger.info(f"Loading data from: {args.data_csv}")
    df = pd.read_csv(args.data_csv)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", na=False)].copy()
    
    # Build features
    X, y = build_feature_matrix(df)
    feature_names = X.columns.tolist()
    
    # Load model
    if not os.path.exists(args.model_bundle):
        raise FileNotFoundError(f"Model bundle not found: {args.model_bundle}")
    
    logger.info(f"Loading model from: {args.model_bundle}")
    with open(args.model_bundle, "rb") as f:
        bundle = pickle.load(f)
    
    # Extract model (handle different bundle formats)
    if "base_model" in bundle:
        # Two-stage model bundle
        model = bundle["base_model"]
        logger.info("Using base_model from two-stage bundle")
    elif "model" in bundle:
        # Simple model bundle
        model = bundle["model"]
        logger.info("Using model from bundle")
    else:
        raise KeyError(f"No model found in bundle. Keys: {list(bundle.keys())}")
    
    # ========================================================================
    # COMPUTE SHAP VALUES
    # ========================================================================
    logger.info("-" * 70)
    logger.info("Computing SHAP values...")
    logger.info("-" * 70)
    
    # Sample data for SHAP
    n_sample = min(args.shap_sample, len(X))
    rng = np.random.default_rng(args.seed)
    sample_idx = rng.choice(len(X), size=n_sample, replace=False)
    
    X_sample = X.iloc[sample_idx].values.astype(np.float32)
    logger.info(f"Sampling {n_sample} compounds for SHAP analysis")
    
    # Compute SHAP
    shap_values, importance = compute_shap_importance(model, X_sample, feature_names)
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    logger.info("-" * 70)
    logger.info("Saving results...")
    logger.info("-" * 70)
    
    # 1. Global importance (all features)
    all_importance_df = save_importance_csv(
        feature_names,
        importance,
        os.path.join(args.outdir, "shap_feature_importance.csv")
    )
    
    # 2. Top-N features
    top_importance_df = all_importance_df.head(args.top_n).copy()
    top_path = os.path.join(args.outdir, f"shap_top{args.top_n}_features.csv")
    top_importance_df.to_csv(top_path, index=False)
    logger.info(f"Saved top-{args.top_n} features to: {top_path}")
    
    # 3. Split by ligand/protein
    lig_df, prot_df = split_ligand_protein_importance(feature_names, importance)
    
    lig_path = os.path.join(args.outdir, "shap_ligand_feature_importance.csv")
    lig_df.to_csv(lig_path, index=False)
    logger.info(f"Saved ligand importance to: {lig_path}")
    
    prot_path = os.path.join(args.outdir, "shap_protein_feature_importance.csv")
    prot_df.to_csv(prot_path, index=False)
    logger.info(f"Saved protein importance to: {prot_path}")
    
    # 4. Aggregate by protein position
    pos_df = aggregate_by_protein_position(feature_names, importance)
    if not pos_df.empty:
        pos_path = os.path.join(args.outdir, "protein_position_importance.csv")
        pos_df.to_csv(pos_path, index=False)
        logger.info(f"Saved position importance to: {pos_path}")
    
    # 5. Summary JSON
    summary = {
        "analysis": "SHAP feature importance",
        "n_samples": int(n_sample),
        "n_features": int(len(feature_names)),
        "top_n_extracted": args.top_n,
        "top_features": top_importance_df["feature"].tolist()[:args.top_n],
        "top_ligand_features": lig_df.head(args.top_n)["feature"].tolist(),
        "top_protein_features": prot_df.head(args.top_n)["feature"].tolist(),
        "total_ligand_features": int(len(lig_df)),
        "total_protein_features": int(len(prot_df)),
    }
    
    summary_path = os.path.join(args.outdir, "shap_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to: {summary_path}")
    
    # ========================================================================
    # REPORT
    # ========================================================================
    logger.info("=" * 70)
    logger.info("SHAP Analysis Complete!")
    logger.info("=" * 70)
    
    logger.info(f"\nTop-{args.top_n} Most Important Features:")
    for i, row in top_importance_df.iterrows():
        logger.info(f"  {i+1:2d}. {row['feature']:30s} | SHAP: {row['mean_abs_shap']:.6f}")
    
    logger.info(f"\nOutput Files:")
    logger.info(f"  ✅ {lig_path}")
    logger.info(f"  ✅ {prot_path}")
    logger.info(f"  ✅ {top_path}")
    logger.info(f"  ✅ {os.path.join(args.outdir, 'shap_feature_importance.csv')}")
    if not pos_df.empty:
        logger.info(f"  ✅ {pos_path}")
    logger.info(f"  ✅ {summary_path}")


if __name__ == "__main__":
    main()
