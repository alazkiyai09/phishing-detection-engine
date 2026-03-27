"""
Evaluation metrics for phishing detection benchmark.

Implements standard and financial-sector-specific metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)
import logging

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    average: str = "weighted"
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.

    Args:
        y_true: True labels (n_samples,)
        y_pred: Predicted labels (n_samples,)
        y_proba: Predicted probabilities (n_samples, n_classes)
        average: Averaging method for multi-class ('weighted', 'macro', 'micro')

    Returns:
        Dictionary of metric names and values
    """
    metrics = {}

    # Primary metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, average=average, zero_division=0)

    # Binary metrics (assuming class 1 is phishing)
    metrics["precision_binary"] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    metrics["recall_binary"] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    metrics["f1_binary"] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    # False positive rate (legitimate emails flagged as phishing)
    metrics["fpr"] = compute_fpr(y_true, y_pred)

    # AUPRC and AUROC
    auprc, auroc = compute_auprc_auroc(y_true, y_proba)
    metrics["auprc"] = auprc
    metrics["auroc"] = auroc

    return metrics


def compute_auprc_auroc(
    y_true: np.ndarray,
    y_proba: np.ndarray
) -> Tuple[float, float]:
    """
    Compute Area Under PR Curve and ROC Curve.

    Args:
        y_true: True labels (n_samples,)
        y_proba: Predicted probabilities (n_samples, n_classes)

    Returns:
        Tuple of (auprc, auroc)
    """
    # Get probability of positive class
    if y_proba.ndim > 1 and y_proba.shape[1] > 1:
        y_proba_positive = y_proba[:, 1]
    else:
        y_proba_positive = y_proba

    # AUROC
    try:
        auroc = roc_auc_score(y_true, y_proba_positive)
    except ValueError as e:
        logger.warning(f"Could not compute AUROC: {e}")
        auroc = 0.0

    # AUPRC - compute manually for more control
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_proba_positive)
        # Use trapezoidal rule to compute area
        auprc = np.trapz(precision, recall)
        # Normalize to [0, 1]
        auprc = max(0.0, min(1.0, auprc))
    except ValueError as e:
        logger.warning(f"Could not compute AUPRC: {e}")
        auprc = 0.0

    return auprc, auroc


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: Optional[list] = None
) -> pd.DataFrame:
    """
    Compute metrics per class.

    Args:
        y_true: True labels (n_samples,)
        y_pred: Predicted labels (n_samples,)
        y_proba: Predicted probabilities (n_samples, n_classes)
        class_names: Names for each class

    Returns:
        DataFrame with per-class metrics
    """
    if class_names is None:
        class_names = ["Class_0", "Class_1"]

    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    results = []

    for cls in unique_classes:
        cls_idx = int(cls)

        # Binary metrics for this class
        y_true_binary = (y_true == cls).astype(int)
        y_pred_binary = (y_pred == cls).astype(int)

        metrics = {
            "class": class_names[cls_idx] if cls_idx < len(class_names) else f"Class_{cls_idx}",
            "support": np.sum(y_true_binary),
            "precision": precision_score(y_true_binary, y_pred_binary, zero_division=0),
            "recall": recall_score(y_true_binary, y_pred_binary, zero_division=0),
            "f1": f1_score(y_true_binary, y_pred_binary, zero_division=0)
        }

        results.append(metrics)

    return pd.DataFrame(results)


def compute_fpr(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_label: int = 1
) -> float:
    """
    Compute False Positive Rate.

    FPR = FP / (FP + TN)
    For phishing: legitimate emails incorrectly flagged as phishing

    Args:
        y_true: True labels (n_samples,)
        y_pred: Predicted labels (n_samples,)
        pos_label: Label considered positive (phishing)

    Returns:
        False positive rate
    """
    # Handle edge cases more robustly
    try:
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
    except ValueError as e:
        logger.warning(f"Could not compute confusion matrix: {e}")
        # Edge case: all predictions are the same class
        unique_preds = np.unique(y_pred)
        if len(unique_preds) == 1 and unique_preds[0] == pos_label:
            # All predicted as positive - all legitimate are false positives
            n_legitimate = (y_true == 0).sum()
            n_fp = (y_true == 0).sum()  # All legitimate emails are FP
            return n_fp / n_legitimate if n_legitimate > 0 else 0.0
        else:
            # All predicted as negative - no false positives
            return 0.0

    if fp + tn == 0:
        return 0.0

    return fp / (fp + tn)


def compute_financial_sector_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    financial_mask: np.ndarray
) -> Dict[str, float]:
    """
    Compute metrics for financial phishing subset.

    Args:
        y_true: True labels (n_samples,)
        y_pred: Predicted labels (n_samples,)
        y_proba: Predicted probabilities (n_samples, n_classes)
        financial_mask: Boolean mask for financial phishing samples

    Returns:
        Dictionary of financial sector metrics
    """
    metrics = {}

    # Filter to financial phishing subset
    y_true_fin = y_true[financial_mask]
    y_pred_fin = y_pred[financial_mask]
    y_proba_fin = y_proba[financial_mask] if y_proba is not None else None

    if len(y_true_fin) == 0:
        logger.warning("No financial phishing samples found")
        return {
            "financial_recall": 0.0,
            "financial_precision": 0.0,
            "financial_f1": 0.0,
            "financial_count": 0
        }

    # Recall on financial phishing (critical: must be > 95%)
    metrics["financial_recall"] = recall_score(
        y_true_fin, y_pred_fin, pos_label=1, zero_division=0
    )
    metrics["financial_precision"] = precision_score(
        y_true_fin, y_pred_fin, pos_label=1, zero_division=0
    )
    metrics["financial_f1"] = f1_score(
        y_true_fin, y_pred_fin, pos_label=1, zero_division=0
    )
    metrics["financial_count"] = len(y_true_fin)

    # Check if financial sector requirements are met
    metrics["meets_recall_threshold"] = metrics["financial_recall"] >= 0.95

    return metrics


def aggregate_cv_results(
    cv_results: list
) -> Dict[str, Tuple[float, float]]:
    """
    Aggregate cross-validation results with mean and std.

    Args:
        cv_results: List of metric dictionaries from each fold

    Returns:
        Dictionary mapping metric names to (mean, std) tuples
    """
    if not cv_results:
        return {}

    # Get all metric keys from first fold
    metric_keys = cv_results[0].keys()

    aggregated = {}

    for key in metric_keys:
        values = [fold_result[key] for fold_result in cv_results if key in fold_result]

        if values:
            aggregated[key] = (np.mean(values), np.std(values))

    return aggregated


def format_cv_results(
    aggregated_metrics: Dict[str, Tuple[float, float]],
    model_name: str
) -> pd.DataFrame:
    """
    Format cross-validation results for reporting.

    Args:
        aggregated_metrics: Dictionary from aggregate_cv_results
        model_name: Name of the model

    Returns:
        DataFrame with formatted results
    """
    results = []

    for metric_name, (mean_val, std_val) in aggregated_metrics.items():
        results.append({
            "model": model_name,
            "metric": metric_name,
            "mean": mean_val,
            "std": std_val,
            "formatted": f"{mean_val:.4f} ± {std_val:.4f}"
        })

    return pd.DataFrame(results)
