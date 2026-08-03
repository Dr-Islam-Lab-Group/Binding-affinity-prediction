"""
Common utilities for protein-ligand binding affinity prediction models.

This module contains shared functions used across different model implementations.
"""

from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors


# ============================================================================
# AMINO ACID PROPERTIES
# ============================================================================

# Amino acid physicochemical properties (Kyte-Doolittle hydrophobicity + others)
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


# ============================================================================
# SMILES & DESCRIPTORS
# ============================================================================

def clean_smiles(x: str) -> Optional[str]:
    """
    Clean and validate SMILES string.
    
    Args:
        x: Input SMILES or similar string
        
    Returns:
        Cleaned SMILES string or None if invalid
    """
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    return s


def canonical_smiles_safe(smi: str) -> Optional[str]:
    """
    Convert SMILES to canonical form with error handling.
    
    Args:
        smi: Input SMILES string
        
    Returns:
        Canonical SMILES or None if cannot parse
    """
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


def calc_selected_descriptors_safe(
    smiles: str,
    descriptor_names: List[str]
) -> Optional[Dict[str, float]]:
    """
    Calculate selected RDKit descriptors with error handling.
    
    Args:
        smiles: SMILES string
        descriptor_names: List of descriptor names to calculate
        
    Returns:
        Dictionary of descriptor_name -> value, or None if molecule invalid
    """
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
    """
    Impute infinities and NaN values with column-wise median.
    
    Args:
        df_num: Numerical DataFrame
        
    Returns:
        Imputed DataFrame
    """
    df_num = df_num.replace([np.inf, -np.inf], np.nan)
    return df_num.fillna(df_num.median(numeric_only=True))


# ============================================================================
# PROTEIN ENCODING
# ============================================================================

def protein_delta_columns(seq_len: int) -> List[str]:
    """
    Generate column names for protein delta encoding.
    
    Args:
        seq_len: Length of protein sequence
        
    Returns:
        List of feature column names
    """
    cols = []
    for i in range(seq_len):
        cols.append(f"pos{i+1}_changed")
        for k in PROT_KEYS:
            cols.append(f"pos{i+1}_d{k}")
    return cols


def encode_delta_vs_reference(seq: str, ref_seq: str) -> np.ndarray:
    """
    Encode protein sequence as delta (difference) vs reference.
    
    For each position:
        1. Binary mutation indicator (1 if changed from ref, 0 otherwise)
        2. Delta for each property (current_property - ref_property)
    
    Properties:
        - hydro: Hydrophobicity (Kyte-Doolittle)
        - charge: Charge state (-1, 0, +1)
        - arom: Aromaticity (0 or 1)
        - polar: Polarity (0 or 1)
    
    Args:
        seq: Current protein sequence (uppercase)
        ref_seq: Reference protein sequence (uppercase)
        
    Returns:
        Feature vector of shape (seq_len * 5,)
        
    Raises:
        ValueError: If sequences have different lengths or unknown amino acids
    """
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
        
        # Mutation indicator
        feats.append(1.0 if aa != ref_aa else 0.0)
        
        # Delta properties
        for k in PROT_KEYS:
            delta = AA_PROPS[aa][k] - AA_PROPS[ref_aa][k]
            feats.append(float(delta))
    
    return np.asarray(feats, dtype=float)


def encode_absolute_properties(seq: str) -> np.ndarray:
    """
    Encode protein sequence as absolute amino acid properties (not delta).
    
    For each position, includes the 4 properties without reference comparison.
    
    Args:
        seq: Protein sequence (uppercase)
        
    Returns:
        Feature vector of shape (seq_len * 4,)
    """
    seq = str(seq).strip().upper()
    
    feats = []
    for i, aa in enumerate(seq):
        if aa not in AA_PROPS:
            raise ValueError(f"Unknown amino acid at position {i+1}: {aa}")
        
        for k in PROT_KEYS:
            feats.append(float(AA_PROPS[aa][k]))
    
    return np.asarray(feats, dtype=float)


# ============================================================================
# METRICS
# ============================================================================

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute root mean squared error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        RMSE value
    """
    from sklearn.metrics import mean_squared_error
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary containing R2, RMSE, MAE
    """
    from sklearn.metrics import r2_score, mean_absolute_error
    
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


# ============================================================================
# DATA LOADING & VALIDATION
# ============================================================================

def load_and_validate_data(
    csv_path: str,
    required_cols: set = {"SMILES", "ActiveSiteSeq", "Binding_affinity"}
) -> pd.DataFrame:
    """
    Load CSV data and validate required columns.
    
    Args:
        csv_path: Path to CSV file
        required_cols: Set of required column names
        
    Returns:
        Validated DataFrame
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns missing
    """
    if not pd.io.common.file_exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", na=False)].copy()
    
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )
    
    return df
