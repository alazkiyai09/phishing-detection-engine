"""
Stratified cross-validation for classifier evaluation.

Ensures balanced class distribution across folds.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.model_selection import StratifiedKFold
import logging

from src.models.classical.models.base_classifier import BaseClassifier
from src.models.classical.evaluation.metrics import compute_metrics, aggregate_cv_results
from src.models.classical.config import get_config

logger = logging.getLogger(__name__)


def stratified_cv(
    model: BaseClassifier,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42
) -> Tuple[Dict[str, Tuple[float, float]], List[Dict]]:
    """
    Perform stratified k-fold cross-validation.

    Args:
        model: Classifier instance
        X: Features (n_samples, n_features)
        y: Labels (n_samples,)
        n_folds: Number of CV folds
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (aggregated_metrics, fold_results_list)
        - aggregated_metrics: Dict mapping metric names to (mean, std)
        - fold_results_list: List of metric dictionaries per fold
    """
    config = get_config()

    logger.info(f"Starting {n_folds}-fold stratified cross-validation")

    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_state
    )

    fold_results = []
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        logger.info(f"Fold {fold_idx}/{n_folds}")

        # Split data
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]

        # Clone model to avoid fitting on previous folds
        model_clone = _clone_model(model)

        # Train model
        model_clone.fit_with_timing(X_train_fold, y_train_fold)

        # Predict
        y_pred = model_clone.predict(X_val_fold)
        y_proba = model_clone.predict_proba(X_val_fold)

        # Compute metrics
        metrics = compute_metrics(y_val_fold, y_pred, y_proba)
        metrics["fold"] = fold_idx
        metrics["training_time"] = model_clone.training_time
        metrics["inference_time"] = model_clone.inference_time

        fold_results.append(metrics)
        fold_metrics.append(metrics)

        logger.info(f"Fold {fold_idx} - F1: {metrics['f1']:.4f}, AUPRC: {metrics['auprc']:.4f}")

    # Aggregate results
    aggregated = aggregate_cv_results(fold_results)

    logger.info("Cross-validation complete")
    logger.info(f"Mean F1: {aggregated['f1'][0]:.4f} ± {aggregated['f1'][1]:.4f}")
    logger.info(f"Mean AUPRC: {aggregated['auprc'][0]:.4f} ± {aggregated['auprc'][1]:.4f}")

    return aggregated, fold_results


def _clone_model(model: BaseClassifier) -> BaseClassifier:
    """
    Clone a model instance.

    Creates a new instance with the same hyperparameters.

    Args:
        model: Model to clone

    Returns:
        New model instance
    """
    model_class = model.__class__
    params = model.get_params()
    return model_class(**params)


def cross_validate_all_models(
    models: Dict[str, BaseClassifier],
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Cross-validate multiple models and compare results.

    Args:
        models: Dictionary mapping model names to model instances
        X: Features (n_samples, n_features)
        y: Labels (n_samples,)
        n_folds: Number of CV folds
        random_state: Random seed

    Returns:
        DataFrame with all models' CV results
    """
    config = get_config()
    all_results = []

    for model_name, model in models.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {model_name}")
        logger.info(f"{'='*60}")

        try:
            aggregated, _ = stratified_cv(
                model=model,
                X=X,
                y=y,
                n_folds=n_folds,
                random_state=random_state
            )

            # Format results
            for metric_name, (mean_val, std_val) in aggregated.items():
                if isinstance(mean_val, (int, float)):
                    all_results.append({
                        "model": model_name,
                        "metric": metric_name,
                        "mean": mean_val,
                        "std": std_val
                    })

        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            continue

    # Create summary DataFrame
    results_df = pd.DataFrame(all_results)

    # Pivot for easier comparison
    summary_df = results_df.pivot(
        index="metric",
        columns="model",
        values=["mean", "std"]
    )

    return results_df, summary_df
