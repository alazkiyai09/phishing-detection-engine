"""
Temporal evaluation for phishing detection.

Tests model generalization to newer phishing patterns.
"""

import numpy as np
from typing import Dict, Tuple
import logging

from src.models.classical.models.base_classifier import BaseClassifier
from src.models.classical.evaluation.metrics import compute_metrics
from src.models.classical.config import get_config

logger = logging.getLogger(__name__)


def temporal_evaluation(
    model: BaseClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate model with temporal train/test split.

    Trains on older data, tests on newer data to simulate
    real-world deployment where phishing tactics evolve.

    Args:
        model: Classifier instance
        X_train: Training features (older data)
        y_train: Training labels
        X_test: Test features (newer data)
        y_test: Test labels

    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Training model on historical data")
    model.fit_with_timing(X_train, y_train)

    logger.info("Evaluating on new data")
    y_pred = model.predict_with_timing(X_test)
    y_proba = model.predict_proba_with_timing(X_test)

    metrics = compute_metrics(y_test, y_pred, y_proba)
    metrics["training_time"] = model.training_time
    metrics["inference_time"] = model.inference_time

    logger.info(f"Temporal evaluation results:")
    logger.info(f"  F1: {metrics['f1']:.4f}")
    logger.info(f"  AUPRC: {metrics['auprc']:.4f}")
    logger.info(f"  AUROC: {metrics['auroc']:.4f}")
    logger.info(f"  FPR: {metrics['fpr']:.4f}")

    return metrics


def compare_temporal_vs_cv(
    model: BaseClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cv_metrics: Dict[str, Tuple[float, float]]
) -> Dict[str, Dict]:
    """
    Compare temporal split performance with cross-validation.

    Checks for performance degradation when testing on newer data.

    Args:
        model: Classifier instance
        X_train: Training features
        y_train: Training labels
        X_test: Test features (newer data)
        y_test: Test labels
        cv_metrics: Cross-validation metrics from stratified_cv

    Returns:
        Dictionary with comparison results
    """
    # Run temporal evaluation
    temporal_metrics = temporal_evaluation(
        model, X_train, y_train, X_test, y_test
    )

    # Compare key metrics
    comparison = {}

    key_metrics = ["f1", "auprc", "auroc", "recall", "fpr"]

    for metric in key_metrics:
        if metric in temporal_metrics and metric in cv_metrics:
            cv_mean, cv_std = cv_metrics[metric]
            temporal_val = temporal_metrics[metric]

            diff = temporal_val - cv_mean
            diff_pct = (diff / cv_mean) * 100 if cv_mean != 0 else 0

            comparison[metric] = {
                "cv_mean": cv_mean,
                "cv_std": cv_std,
                "temporal": temporal_val,
                "diff": diff,
                "diff_pct": diff_pct,
                "degraded": diff < 0
            }

    logger.info("\nTemporal vs CV Comparison:")
    for metric, vals in comparison.items():
        status = "DEGRADED" if vals["degraded"] else "IMPROVED"
        logger.info(f"  {metric}: {vals['temporal']:.4f} vs CV {vals['cv_mean']:.4f} "
                   f"({status}: {vals['diff_pct']:+.2f}%)")

    return comparison


def evaluate_all_models_temporal(
    models: Dict[str, BaseClassifier],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """
    Evaluate all models with temporal split.

    Args:
        models: Dictionary mapping model names to model instances
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels

    Returns:
        Tuple of (temporal_results, comparisons_with_cv)
    """
    temporal_results = {}
    comparisons = {}

    for model_name, model in models.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Temporal Evaluation: {model_name}")
        logger.info(f"{'='*60}")

        try:
            metrics = temporal_evaluation(
                model, X_train, y_train, X_test, y_test
            )
            temporal_results[model_name] = metrics

        except Exception as e:
            logger.error(f"Error in temporal evaluation for {model_name}: {e}")
            temporal_results[model_name] = {}

    return temporal_results, comparisons
