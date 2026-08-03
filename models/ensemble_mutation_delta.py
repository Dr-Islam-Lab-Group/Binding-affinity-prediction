#!/usr/bin/env python3
"""
Weighted ensemble of multiple model architectures.

Combines predictions from MLP, XGBoost, LightGBM, and CatBoost models
using learned optimal weights for improved performance.

Architecture:
    - Load individual model bundles (from training scripts)
    - Compute optimal ensemble weights via grid search
    - Make ensemble predictions = w1*pred1 + w2*pred2 + w3*pred3 + w4*pred4
    - Evaluate ensemble performance

Key features:
    - Weight optimization on validation set
    - Weighted averaging of predictions
    - Saves ensemble bundle for inference
    - Comprehensive metrics and reporting
"""

import os
import json
import pickle
import logging
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for ensemble training."""
    
    # Individual model bundles (from training scripts)
    MODEL_BUNDLES = {
        "mlp": "outputs/mlp_grouped_mutation_delta/model_bundle.pkl",
        "xgb": "outputs/xgb_grouped_mutation_delta/model_bundle.pkl",
        "lgbm": "outputs/lgbm_grouped_mutation_delta/model_bundle.pkl",
        "catboost": "outputs/catboost_grouped_mutation_delta/model_bundle.pkl",
    }
    
    # Output directory
    OUTDIR = "outputs/ensemble_weighted_mutation_delta"
    
    # Weight optimization
    WEIGHT_STEP = 0.05  # Grid search step size (0.05 = 21 weights per model)
    SEED = 42
    
    # Data split for ensemble training
    TEST_SIZE = 0.10
    VAL_SIZE = 0.15  # Validation set for weight optimization


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


def load_model_bundle(bundle_path: str) -> Dict:
    """
    Load serialized model bundle.
    
    Args:
        bundle_path: Path to model_bundle.pkl
        
    Returns:
        Dictionary containing model components
        
    Raises:
        FileNotFoundError: If bundle not found
    """
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"Model bundle not found: {bundle_path}")
    
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)
    
    logger.info(f"Loaded model bundle: {bundle_path}")
    return bundle


def predict_from_bundle(
    bundle: Dict,
    X_ligand: np.ndarray,
    X_protein: np.ndarray
) -> np.ndarray:
    """
    Make predictions using a model bundle.
    
    Args:
        bundle: Model bundle with preprocessing pipelines and models
        X_ligand: Ligand features
        X_protein: Protein features
        
    Returns:
        Predictions array
    """
    # Stage A: Baseline prediction (ligand)
    X_lig_selected = bundle["base_selector"].transform(X_ligand)
    y_base = bundle["base_model"].predict(X_lig_selected)
    
    # Stage B: Residual prediction (protein)
    X_prot_selected = bundle["mut_selector"].transform(X_protein)
    y_residual = bundle["mut_model"].predict(X_prot_selected)
    
    # Combined prediction
    return y_base + y_residual


# ============================================================================
# WEIGHT OPTIMIZATION
# ============================================================================

def generate_weight_grid(step: float = 0.05) -> List[float]:
    """
    Generate grid of weights for search (0.0 to 1.0).
    
    Args:
        step: Step size (smaller = finer search, slower)
        
    Returns:
        List of weight values
    """
    return np.arange(0, 1 + step, step).tolist()


def optimize_ensemble_weights(
    predictions: Dict[str, np.ndarray],
    y_val: np.ndarray,
    weight_step: float = 0.05,
    model_names: List[str] = None
) -> Dict[str, float]:
    """
    Find optimal ensemble weights via grid search.
    
    Args:
        predictions: Dict of model_name -> predictions array
        y_val: Target values for validation set
        weight_step: Grid search step size
        model_names: List of model names (default: all in predictions)
        
    Returns:
        Dictionary of model_name -> optimal_weight
    """
    if model_names is None:
        model_names = list(predictions.keys())
    
    weight_grid = generate_weight_grid(weight_step)
    best_rmse = float("inf")
    best_weights = {name: 1.0 / len(model_names) for name in model_names}
    
    n_combinations = len(weight_grid) ** len(model_names)
    logger.info(f"Searching {n_combinations} weight combinations...")
    
    evaluated = 0
    
    # Grid search over weight combinations
    def search_recursive(model_idx: int, current_weights: Dict[str, float]):
        nonlocal best_rmse, best_weights, evaluated
        
        if model_idx == len(model_names):
            # All weights assigned, evaluate this combination
            ensemble_pred = np.zeros_like(y_val, dtype=float)
            weight_sum = sum(current_weights.values())
            
            for name in model_names:
                ensemble_pred += current_weights[name] * predictions[name]
            
            # Normalize by total weight
            ensemble_pred /= weight_sum if weight_sum > 0 else 1.0
            
            loss = rmse(y_val, ensemble_pred)
            
            if loss < best_rmse:
                best_rmse = loss
                best_weights = current_weights.copy()
                if evaluated % max(1, n_combinations // 10) == 0:
                    logger.info(f"  Evaluated {evaluated}: Best RMSE = {best_rmse:.6f}")
            
            evaluated += 1
            return
        
        model_name = model_names[model_idx]
        for weight in weight_grid:
            current_weights[model_name] = weight
            search_recursive(model_idx + 1, current_weights)
    
    search_recursive(0, {})
    
    # Normalize weights to sum to 1.0
    total = sum(best_weights.values())
    best_weights = {k: v / total for k, v in best_weights.items()}
    
    logger.info(f"Optimal weights found (RMSE: {best_rmse:.6f}):")
    for name, weight in best_weights.items():
        logger.info(f"  {name}: {weight:.4f}")
    
    return best_weights


# ============================================================================
# MAIN ENSEMBLE TRAINING
# ============================================================================

def main():
    """Execute ensemble training pipeline."""
    
    os.makedirs(Config.OUTDIR, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("Ensemble Weighted Averaging - Training Pipeline")
    logger.info("=" * 70)
    
    # ========================================================================
    # LOAD INDIVIDUAL MODELS
    # ========================================================================
    logger.info("-" * 70)
    logger.info("Loading individual model bundles...")
    logger.info("-" * 70)
    
    bundles = {}
    for model_name, bundle_path in Config.MODEL_BUNDLES.items():
        try:
            bundles[model_name] = load_model_bundle(bundle_path)
        except FileNotFoundError as e:
            logger.error(f"Failed to load {model_name}: {e}")
            logger.error("Ensure all individual models are trained first:")
            for name in Config.MODEL_BUNDLES.keys():
                logger.error(f"  python models/{name}_mutation_delta.py")
            raise
    
    logger.info(f"Loaded {len(bundles)} model bundles")
    
    # ========================================================================
    # PREPARE DATA FOR WEIGHT OPTIMIZATION
    # ========================================================================
    logger.info("-" * 70)
    logger.info("Preparing data for weight optimization...")
    logger.info("-" * 70)
    
    # For this simple ensemble approach, we'll use test predictions from all models
    # In production, you'd load raw data and generate predictions yourself
    
    model_names = list(bundles.keys())
    logger.info(f"Using models: {', '.join(model_names)}")
    
    # Get test predictions from each model's report
    test_predictions = {}
    y_test = None
    
    for model_name, bundle in bundles.items():
        # For simplicity in this template: using report metrics
        # In production: would load actual prediction arrays
        report = bundle.get("report", {})
        logger.info(f"{model_name} test R2: {report.get('metrics_test', {}).get('R2', 'N/A')}")
        
        # NOTE: In production, would load actual predictions from CSV
        # test_predictions[model_name] = load predictions...
    
    # ========================================================================
    # WEIGHT OPTIMIZATION (TEMPLATE)
    # ========================================================================
    logger.info("-" * 70)
    logger.info("Optimizing ensemble weights...")
    logger.info("-" * 70)
    
    # In a full implementation:
    # 1. Load actual prediction arrays from test_predictions.csv of each model
    # 2. Run weight optimization on validation set
    # 3. Evaluate ensemble on test set
    
    # For this template, use uniform weights
    optimal_weights = {name: 1.0 / len(model_names) for name in model_names}
    
    logger.info("Optimal ensemble weights:")
    for name, weight in optimal_weights.items():
        logger.info(f"  {name}: {weight:.4f}")
    
    # ========================================================================
    # SAVE ENSEMBLE BUNDLE
    # ========================================================================
    logger.info("-" * 70)
    logger.info("Saving ensemble bundle...")
    logger.info("-" * 70)
    
    ensemble_bundle = {
        "model_type": "ensemble",
        "model_names": model_names,
        "individual_bundles": bundles,
        "weights": optimal_weights,
        "report": {
            "ensemble_type": "weighted_average",
            "models": model_names,
            "weights": optimal_weights,
            "note": "This template uses uniform weights. Implement weight optimization for production.",
            "individual_model_metrics": {
                name: bundle.get("report", {}).get("metrics_test", {})
                for name, bundle in bundles.items()
            }
        }
    }
    
    bundle_path = os.path.join(Config.OUTDIR, "ensemble_model_bundle.pkl")
    with open(bundle_path, "wb") as f:
        pickle.dump(ensemble_bundle, f)
    logger.info(f"Saved ensemble bundle: {bundle_path}")
    
    # Save report
    report_path = os.path.join(Config.OUTDIR, "report.json")
    with open(report_path, "w") as f:
        json.dump(ensemble_bundle["report"], f, indent=2)
    logger.info(f"Saved report: {report_path}")
    
    logger.info("=" * 70)
    logger.info("Ensemble training complete!")
    logger.info("=" * 70)
    logger.info("\nNOTE: This is a template implementation.")
    logger.info("To implement full weight optimization:")
    logger.info("1. Load test predictions from each model's CSV")
    logger.info("2. Call optimize_ensemble_weights() function")
    logger.info("3. Evaluate final ensemble performance")


if __name__ == "__main__":
    main()
