#!/usr/bin/env python3
"""
Mutation-aware MLP with two-stage architecture.

Stage A (Ligand Baseline):
    Features: Top-20 RDKit descriptors
    Target: Ligand-level mean binding affinity
    
Stage B (Mutation Residual):
    Features: Top-20 ligand descriptors + delta protein features (interaction-aware encoding)
    Target: Residual = actual affinity - baseline prediction

Key features:
    - Group split by LigandID (canonical SMILES) to prevent leakage
    - Protein encoded as DELTA vs explicit wild-type active-site sequence
    - Manual warm-start loop with early stopping via patience
    - Saves model_bundle.pkl for inference

Reference:
    Protein delta encoding captures mutation effects by computing
    differences in amino acid properties vs a reference sequence.
"""

import os
import json
import pickle
import random
import logging
import warnings
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.exceptions import ConvergenceWarning


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress RDKit warnings
if True:
    RDLogger.DisableLog("rdApp.error")
    warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for MLP training."""
    
    # Paths
    DATA_PATH = "data/data.csv"
    LIGAND_IMPORTANCE_CSV = "outputs/shap_ligand_feature_importance.csv"
    OUTDIR = "outputs/mlp_grouped_mutation_delta"
    
    # Feature selection
    TOP_N_DESCRIPTORS = 20

    # Explicit wild-type active-site sequence used for all delta encoding
    WT_ACTIVE_SITE_SEQ = "HLYGSFLCGSCGSHQMELANHDKQVHQ"
    
    # Data splitting
    TEST_SIZE = 0.10
    SPLIT_SEED = 42
    
    # Training
    MODEL_SEED = 42
    MAX_EPOCHS = 300
    PATIENCE = 30
    MIN_DELTA = 1e-6
    
    # Optional: Load best params from gridsearch
    BEST_PARAMS_JSON = "outputs/mlp_gridsearch_multilayer/gridsearch_results.json"
    USE_BEST_PARAMS_JSON = True


# ============================================================================
# HELPERS
# ============================================================================

def set_all_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


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



def build_delta_score_table(
    eval_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    wt_seq: str,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Build ligand-specific mutant-minus-WT delta-score pairs and metrics."""
    tmp = eval_df[["LigandID", "SMILES", "ActiveSiteSeq"]].copy().reset_index(drop=True)
    tmp["y_true"] = np.asarray(y_true, dtype=float)
    tmp["y_pred"] = np.asarray(y_pred, dtype=float)
    tmp["ActiveSiteSeq_norm"] = tmp["ActiveSiteSeq"].astype(str).str.strip().str.upper()
    wt_seq = str(wt_seq).strip().upper()

    rows = []
    for ligand_id, grp in tmp.groupby("LigandID", sort=False):
        wt = grp.loc[grp["ActiveSiteSeq_norm"] == wt_seq]
        if wt.empty:
            continue
        wt_row = wt.iloc[0]
        mutants = grp.loc[grp["ActiveSiteSeq_norm"] != wt_seq]
        for _, mut_row in mutants.iterrows():
            rows.append({
                "LigandID": ligand_id,
                "SMILES": mut_row["SMILES"],
                "MutantActiveSiteSeq": mut_row["ActiveSiteSeq_norm"],
                "WT_true": float(wt_row["y_true"]),
                "Mutant_true": float(mut_row["y_true"]),
                "WT_pred": float(wt_row["y_pred"]),
                "Mutant_pred": float(mut_row["y_pred"]),
                "delta_true_mut_minus_wt": float(mut_row["y_true"] - wt_row["y_true"]),
                "delta_pred_mut_minus_wt": float(mut_row["y_pred"] - wt_row["y_pred"]),
            })

    delta_df = pd.DataFrame(rows)
    if delta_df.empty:
        return delta_df, {"R2": float("nan"), "RMSE": float("nan"), "MAE": float("nan"), "n_pairs": 0}

    d_true = delta_df["delta_true_mut_minus_wt"].to_numpy()
    d_pred = delta_df["delta_pred_mut_minus_wt"].to_numpy()
    metrics = evaluate(d_true, d_pred)
    metrics["n_pairs"] = int(len(delta_df))
    return delta_df, metrics

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
        raise ValueError(f"CSV missing columns {missing}. Found: {list(imp.columns)}")
    
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
    """
    Encode protein sequence as delta (difference) vs reference.
    
    For each position:
        1. Binary mutation indicator (1 if changed from ref, 0 otherwise)
        2. Delta for each property (current_property - ref_property)
    
    Args:
        seq: Current protein sequence
        ref_seq: Reference sequence
        
    Returns:
        Feature vector of shape (seq_len * 5,)
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


# ============================================================================
# LOAD BEST PARAMETERS
# ============================================================================

def load_best_params() -> Dict:
    """Load best hyperparameters from gridsearch results."""
    if not Config.USE_BEST_PARAMS_JSON:
        return {}
    
    if not os.path.exists(Config.BEST_PARAMS_JSON):
        logger.warning(f"Best params JSON not found: {Config.BEST_PARAMS_JSON}")
        return {}
    
    try:
        with open(Config.BEST_PARAMS_JSON, "r") as f:
            j = json.load(f)
        
        best_params = j.get("best_params", {})
        cleaned = {}
        
        for k, v in best_params.items():
            if k.startswith("mlp__"):
                cleaned[k.replace("mlp__", "")] = v
            else:
                cleaned[k] = v
        
        # Remove parameters controlled by manual warm-start loop
        for key in ["max_iter", "early_stopping", "n_iter_no_change", "validation_fraction"]:
            cleaned.pop(key, None)
        
        logger.info(f"Loaded {len(cleaned)} best parameters")
        return cleaned
    
    except Exception as e:
        logger.error(f"Error loading best params: {e}")
        return {}


# ============================================================================
# TRAINING WITH EARLY STOPPING
# ============================================================================

def fit_mlp_with_patience(
    model: MLPRegressor,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    max_epochs: int = Config.MAX_EPOCHS,
    patience: int = Config.PATIENCE,
    min_delta: float = Config.MIN_DELTA,
) -> Tuple[MLPRegressor, List[float], List[float]]:
    """
    Train MLP with manual early stopping via patience.
    
    Args:
        model: MLPRegressor with warm_start=True
        X_tr, y_tr: Training data
        X_va, y_va: Validation data
        max_epochs: Maximum epochs
        patience: Early stopping patience
        min_delta: Minimum improvement threshold
        
    Returns:
        Tuple of (trained_model, train_losses, val_losses)
    """
    train_losses = []
    val_losses = []
    best_val = float("inf")
    best_state = None
    wait = 0
    
    for epoch in range(1, max_epochs + 1):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model.fit(X_tr, y_tr)
        
        pred_tr = model.predict(X_tr)
        pred_va = model.predict(X_va)
        
        loss_tr = rmse(y_tr, pred_tr)
        loss_va = rmse(y_va, pred_va)
        
        train_losses.append(loss_tr)
        val_losses.append(loss_va)
        
        if loss_va < best_val - min_delta:
            best_val = loss_va
            best_state = pickle.dumps(model)
            wait = 0
            
            if epoch % 20 == 0:
                logger.debug(
                    f"Epoch {epoch:3d}: train_loss={loss_tr:.6f}, "
                    f"val_loss={loss_va:.6f} (improved)"
                )
        else:
            wait += 1
            if wait >= patience:
                logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(patience {patience} reached)"
                )
                break
    
    if best_state is not None:
        model = pickle.loads(best_state)
    
    return model, train_losses, val_losses


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """Execute full training pipeline."""
    
    os.makedirs(Config.OUTDIR, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("MLP Mutation-Delta Training Pipeline")
    logger.info("=" * 70)
    
    # Set seeds
    set_all_seeds(Config.MODEL_SEED)
    logger.info(f"Random seeds set to: {Config.MODEL_SEED}")
    
    # Load configuration
    best_params = load_best_params()
    logger.info(f"Best params: {best_params}")
    
    # Load descriptor names
    top_desc_names = load_top_descriptor_names(
        Config.LIGAND_IMPORTANCE_CSV,
        Config.TOP_N_DESCRIPTORS
    )
    logger.info(f"Top-{len(top_desc_names)} descriptors loaded")
    
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
    
    # Use the explicitly defined wild-type active-site sequence as the delta reference
    ref_seq = Config.WT_ACTIVE_SITE_SEQ.strip().upper()
    observed_sequences = set(df["ActiveSiteSeq"].astype(str).str.strip().str.upper())
    if ref_seq not in observed_sequences:
        raise ValueError(
            "Configured WT_ACTIVE_SITE_SEQ was not found in ActiveSiteSeq. "
            "Check the sequence and dataset before training."
        )
    logger.info(f"Explicit WT reference sequence: {ref_seq}")
    
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
    base_scaler = StandardScaler()
    
    Xb_sel = base_selector.fit_transform(X_base)
    Xb_sc = base_scaler.fit_transform(Xb_sel)
    
    logger.info(f"Features after variance threshold: {Xb_sel.shape}")
    
    # Train-val split for early stopping
    n = len(y_base)
    idx = np.arange(n)
    np.random.shuffle(idx)
    cut = int(max(5, 0.85 * n))
    tr_b, va_b = idx[:cut], idx[cut:]
    
    base_mlp = MLPRegressor(
        warm_start=True,
        max_iter=1,
        shuffle=True,
        random_state=Config.MODEL_SEED,
        **best_params,
    )
    
    logger.info("Training Stage A model with early stopping...")
    base_mlp, base_tr_curve, base_va_curve = fit_mlp_with_patience(
        base_mlp,
        Xb_sc[tr_b], y_base[tr_b],
        Xb_sc[va_b], y_base[va_b],
    )
    
    logger.info(
        f"Stage A training complete. "
        f"Final val RMSE: {base_va_curve[-1]:.6f}"
    )
    
    # Predict baseline
    def baseline_pred_rows(row_idx):
        Xr = ligand_X.iloc[row_idx][top_desc_names].astype(float).values
        Xr_sel = base_selector.transform(Xr)
        Xr_sc = base_scaler.transform(Xr_sel)
        return base_mlp.predict(Xr_sc)
    
    base_pred_tr = baseline_pred_rows(tr_idx)
    base_pred_te = baseline_pred_rows(te_idx)
    
    y_tr = y[tr_idx]
    y_te = y[te_idx]
    y_res_tr = y_tr - base_pred_tr
    
    # ========================================================================
    # STAGE B: MUTATION RESIDUAL MODEL
    # ========================================================================
    logger.info("-" * 70)
    logger.info("STAGE B: Training interaction-aware ligand × mutation residual model")
    logger.info("-" * 70)
    
    # Interaction-aware Stage B: concatenate ligand descriptors with mutation-delta features.
    # This allows the mutation residual to depend on ligand identity/chemistry.
    X_lig_tr_b = ligand_X.iloc[tr_idx][top_desc_names].astype(float).values
    X_lig_te_b = ligand_X.iloc[te_idx][top_desc_names].astype(float).values
    X_mut_tr = mut_X.iloc[tr_idx].values.astype(float)
    X_mut_te = mut_X.iloc[te_idx].values.astype(float)
    X_stageb_tr = np.hstack([X_lig_tr_b, X_mut_tr])
    X_stageb_te = np.hstack([X_lig_te_b, X_mut_te])
    
    mut_selector = VarianceThreshold(0.0)
    mut_scaler = StandardScaler()
    
    Xm_sel_tr = mut_selector.fit_transform(X_stageb_tr)
    Xm_sel_te = mut_selector.transform(X_stageb_te)
    
    Xm_sc_tr = mut_scaler.fit_transform(Xm_sel_tr)
    Xm_sc_te = mut_scaler.transform(Xm_sel_te)
    
    logger.info(f"Stage B training set: {Xm_sel_tr.shape}")
    logger.info(f"Features after variance threshold: {Xm_sel_tr.shape}")
    
    # Train-val split
    ntr = len(y_res_tr)
    idx = np.arange(ntr)
    np.random.shuffle(idx)
    cut = int(max(20, 0.85 * ntr))
    tr_m, va_m = idx[:cut], idx[cut:]
    
    mut_mlp = MLPRegressor(
        warm_start=True,
        max_iter=1,
        shuffle=True,
        random_state=Config.MODEL_SEED,
        **best_params,
    )
    
    logger.info("Training Stage B model with early stopping...")
    mut_mlp, mut_tr_curve, mut_va_curve = fit_mlp_with_patience(
        mut_mlp,
        Xm_sc_tr[tr_m], y_res_tr[tr_m],
        Xm_sc_tr[va_m], y_res_tr[va_m],
    )
    
    logger.info(
        f"Stage B training complete. "
        f"Final val RMSE: {mut_va_curve[-1]:.6f}"
    )
    
    # Predict residuals
    res_pred_te = mut_mlp.predict(Xm_sc_te)
    res_pred_tr = mut_mlp.predict(Xm_sc_tr)
    
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

    # Reviewer-requested ligand-specific mutant-minus-WT delta-score evaluation
    delta_df, delta_metrics = build_delta_score_table(
        df.iloc[te_idx], y_te, y_pred_te, ref_seq
    )
    logger.info("Ligand-specific delta-score metrics (mutant - WT):")
    for key, val in delta_metrics.items():
        if key == "n_pairs":
            logger.info(f"  {key}: {val}")
        else:
            logger.info(f"  {key}: {val:.6f}")
    delta_path = os.path.join(Config.OUTDIR, "test_delta_scores.csv")
    delta_df.to_csv(delta_path, index=False)
    logger.info(f"Saved delta-score pairs to: {delta_path}")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        "SMILES": df.iloc[te_idx]["SMILES"].values,
        "ActiveSiteSeq": df.iloc[te_idx]["ActiveSiteSeq"].values,
        "LigandID": df.iloc[te_idx]["LigandID"].values,
        "y_test": y_te,
        "y_pred": y_pred_te,
        "pred_baseline_ligand": base_pred_te,
        "pred_residual_interaction": res_pred_te,
    })
    pred_path = os.path.join(Config.OUTDIR, "test_predictions.csv")
    predictions_df.to_csv(pred_path, index=False)
    logger.info(f"Saved predictions to: {pred_path}")
    
    # Save report
    report = {
        "stage_a_final_val_rmse": float(base_va_curve[-1]),
        "stage_b_final_val_rmse": float(mut_va_curve[-1]),
        "ref_seq_used": ref_seq,
        "top_descriptor_names": top_desc_names,
        "split": {
            "test_size": Config.TEST_SIZE,
            "seed": Config.SPLIT_SEED,
            "grouped_by": "LigandID"
        },
        "metrics_train": mets_train,
        "metrics_test": mets_test,
        "delta_score_metrics_mutant_minus_wt": delta_metrics,
        "stage_a_definition": "ligand-level mean binding-affinity baseline",
        "stage_b_definition": "interaction-aware residual using ligand descriptors + WT-referenced mutation-delta features",
        "n_rows_used": int(len(df)),
        "n_unique_ligands": int(df["LigandID"].nunique()),
        "n_unique_sequences": int(df["ActiveSiteSeq"].nunique()),
        "best_params_used": best_params,
    }
    report_path = os.path.join(Config.OUTDIR, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved report to: {report_path}")
    
    # Save model bundle
    bundle = {
        "ref_seq": ref_seq,
        "top_desc_names": top_desc_names,
        "stage_b_uses_ligand_features": True,
        "stage_b_ligand_feature_names": top_desc_names,
        "base_selector": base_selector,
        "base_scaler": base_scaler,
        "base_model": base_mlp,
        "mut_selector": mut_selector,
        "mut_scaler": mut_scaler,
        "mut_model": mut_mlp,
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
