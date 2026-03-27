"""
Error analysis for phishing detection.

Analyzes false positives and false negatives in detail.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import logging

from src.models.classical.models.base_classifier import BaseClassifier
from src.models.classical.config import get_config

logger = logging.getLogger(__name__)


def analyze_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    df: Optional[pd.DataFrame] = None
) -> Dict[str, Union[int, float, np.ndarray]]:
    """
    Comprehensive error analysis.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        df: Original DataFrame with metadata (optional)

    Returns:
        Dictionary with error analysis results:
        - n_fp, n_fn, n_tp, n_tn (int)
        - fp_rate, fn_rate (float)
        - fp_indices, fn_indices (np.ndarray)
        - fp_mean_proba, fn_mean_proba (float)
    """
    results = {}

    # Error types
    false_positives = (y_true == 0) & (y_pred == 1)  # Legitimate marked as phishing
    false_negatives = (y_true == 1) & (y_pred == 0)  # Phishing marked as legitimate
    true_positives = (y_true == 1) & (y_pred == 1)
    true_negatives = (y_true == 0) & (y_pred == 0)

    results["n_fp"] = int(false_positives.sum())
    results["n_fn"] = int(false_negatives.sum())
    results["n_tp"] = int(true_positives.sum())
    results["n_tn"] = int(true_negatives.sum())

    # Rates
    n_total = len(y_true)
    results["fp_rate"] = results["n_fp"] / n_total if n_total > 0 else 0.0
    results["fn_rate"] = results["n_fn"] / n_total if n_total > 0 else 0.0

    # Get probability scores for errors
    if y_proba is not None:
        if y_proba.ndim > 1 and y_proba.shape[1] > 1:
            phishing_proba = y_proba[:, 1]
        else:
            phishing_proba = y_proba

        results["fp_mean_proba"] = float(phishing_proba[false_positives].mean()) if false_positives.sum() > 0 else 0.0
        results["fn_mean_proba"] = float(phishing_proba[false_negatives].mean()) if false_negatives.sum() > 0 else 0.0

    # Get indices
    results["fp_indices"] = np.where(false_positives)[0]
    results["fn_indices"] = np.where(false_negatives)[0]

    logger.info(f"Error Analysis:")
    logger.info(f"  False Positives: {results['n_fp']} ({results['fp_rate']:.2%})")
    logger.info(f"  False Negatives: {results['n_fn']} ({results['fn_rate']:.2%})")

    return results


def analyze_false_negatives(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    df: Optional[pd.DataFrame] = None,
    financial_mask: Optional[np.ndarray] = None
) -> Dict[str, Union[int, float, np.ndarray, pd.DataFrame]]:
    """
    Analyze false negatives (missed phishing).

    Critical for security - missed phishing emails are dangerous.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        df: Original DataFrame with metadata
        financial_mask: Boolean mask for financial phishing subset

    Returns:
        Dictionary with FN analysis results
    """
    fn_mask = (y_true == 1) & (y_pred == 0)
    fn_indices = np.where(fn_mask)[0]

    analysis: Dict[str, Union[int, float, np.ndarray, pd.DataFrame]] = {
        "n_fn": len(fn_indices),
        "fn_indices": fn_indices
    }

    if len(fn_indices) == 0:
        logger.info("No false negatives found!")
        return analysis

    # Get probabilities for FN
    if y_proba is not None:
        if y_proba.ndim > 1 and y_proba.shape[1] > 1:
            phishing_proba = y_proba[:, 1]
        else:
            phishing_proba = y_proba

        fn_proba = phishing_proba[fn_indices]

        analysis["fn_proba_mean"] = float(fn_proba.mean())
        analysis["fn_proba_std"] = float(fn_proba.std())
        analysis["fn_proba_min"] = float(fn_proba.min())
        analysis["fn_proba_max"] = float(fn_proba.max())

        # Near-miss FNs (probability close to threshold)
        near_miss = fn_proba > 0.4  # Just below 0.5 threshold
        analysis["fn_near_miss"] = int(near_miss.sum())
        analysis["fn_far_miss"] = int((~near_miss).sum())

    # Check financial phishing FNs
    if financial_mask is not None:
        financial_fn = financial_mask[fn_indices].sum()
        analysis["fn_financial"] = int(financial_fn)
        analysis["fn_financial_pct"] = float(financial_fn / len(fn_indices) if len(fn_indices) > 0 else 0.0)

    # Get sample data if DataFrame provided
    if df is not None:
        fn_samples = df.iloc[fn_indices].copy()
        fn_samples["predicted_proba"] = y_proba[fn_indices, 1] if y_proba is not None else np.nan
        analysis["fn_samples"] = fn_samples

    logger.info(f"False Negative Analysis:")
    logger.info(f"  Total FN: {analysis['n_fn']}")
    if "fn_proba_mean" in analysis:
        logger.info(f"  Mean prob: {analysis['fn_proba_mean']:.3f}")
        logger.info(f"  Near-miss: {analysis['fn_near_miss']}, Far-miss: {analysis['fn_far_miss']}")
    if "fn_financial" in analysis:
        logger.info(f"  Financial phishing FN: {analysis['fn_financial']} ({analysis['fn_financial_pct']:.1%})")

    return analysis


def analyze_false_positives(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    df: Optional[pd.DataFrame] = None
) -> Dict[str, Union[int, float, np.ndarray, pd.DataFrame]]:
    """
    Analyze false positives (legitimate emails flagged as phishing).

    Critical for user experience - too many FPs reduce trust.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        df: Original DataFrame with metadata

    Returns:
        Dictionary with FP analysis results
    """
    fp_mask = (y_true == 0) & (y_pred == 1)
    fp_indices = np.where(fp_mask)[0]

    analysis: Dict[str, Union[int, float, np.ndarray, pd.DataFrame]] = {
        "n_fp": len(fp_indices),
        "fp_indices": fp_indices
    }

    if len(fp_indices) == 0:
        logger.info("No false positives found!")
        return analysis

    # Get probabilities for FP
    if y_proba is not None:
        if y_proba.ndim > 1 and y_proba.shape[1] > 1:
            phishing_proba = y_proba[:, 1]
        else:
            phishing_proba = y_proba

        fp_proba = phishing_proba[fp_indices]

        analysis["fp_proba_mean"] = float(fp_proba.mean())
        analysis["fp_proba_std"] = float(fp_proba.std())
        analysis["fp_proba_min"] = float(fp_proba.min())
        analysis["fp_proba_max"] = float(fp_proba.max())

        # High-confidence FPs (model very confident but wrong)
        high_conf_fp = fp_proba > 0.8
        analysis["fp_high_conf"] = int(high_conf_fp.sum())
        analysis["fp_low_conf"] = int((~high_conf_fp).sum())

    # Check against FPR threshold
    n_legitimate = (y_true == 0).sum()
    fpr = len(fp_indices) / n_legitimate if n_legitimate > 0 else 0.0
    analysis["fpr"] = fpr

    config = get_config()
    fpr_threshold = config.get("fpr_threshold", 0.01)
    analysis["meets_fpr_requirement"] = fpr <= fpr_threshold

    # Get sample data if DataFrame provided
    if df is not None:
        fp_samples = df.iloc[fp_indices].copy()
        fp_samples["predicted_proba"] = y_proba[fp_indices, 1] if y_proba is not None else np.nan
        analysis["fp_samples"] = fp_samples

    logger.info(f"False Positive Analysis:")
    logger.info(f"  Total FP: {analysis['n_fp']}")
    logger.info(f"  FPR: {fpr:.2%} (threshold: {fpr_threshold:.1%})")
    logger.info(f"  Meets requirement: {analysis['meets_fpr_requirement']}")
    if "fp_proba_mean" in analysis:
        logger.info(f"  Mean prob: {analysis['fp_proba_mean']:.3f}")
        logger.info(f"  High-conf FP: {analysis['fp_high_conf']}, Low-conf FP: {analysis['fp_low_conf']}")

    return analysis


def create_error_report(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    df: Optional[pd.DataFrame] = None,
    financial_mask: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Create comprehensive error report.

    Args:
        model_name: Name of the model
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        df: Original DataFrame
        financial_mask: Mask for financial phishing
        save_path: Path to save CSV report

    Returns:
        DataFrame with error report
    """
    logger.info(f"Creating error report for {model_name}")

    # Run all analyses
    error_analysis = analyze_errors(y_true, y_pred, y_proba, df)
    fn_analysis = analyze_false_negatives(y_true, y_pred, y_proba, df, financial_mask)
    fp_analysis = analyze_false_positives(y_true, y_pred, y_proba, df)

    # Create report
    report_data = {
        "model": model_name,
        "n_samples": len(y_true),
        "n_fn": fn_analysis["n_fn"],
        "n_fp": fp_analysis["n_fp"],
        "fpr": fp_analysis.get("fpr", 0),
        "meets_fpr_requirement": fp_analysis.get("meets_fpr_requirement", False),
    }

    # Add FN details
    if "fn_proba_mean" in fn_analysis:
        report_data["fn_mean_proba"] = fn_analysis["fn_proba_mean"]
        report_data["fn_near_miss"] = fn_analysis["fn_near_miss"]
        report_data["fn_far_miss"] = fn_analysis["fn_far_miss"]

    # Add financial FN details
    if "fn_financial" in fn_analysis:
        report_data["fn_financial"] = fn_analysis["fn_financial"]
        report_data["fn_financial_pct"] = fn_analysis["fn_financial_pct"]

    # Add FP details
    if "fp_proba_mean" in fp_analysis:
        report_data["fp_mean_proba"] = fp_analysis["fp_proba_mean"]
        report_data["fp_high_conf"] = fp_analysis["fp_high_conf"]
        report_data["fp_low_conf"] = fp_analysis["fp_low_conf"]

    report_df = pd.DataFrame([report_data])

    # Save report
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(save_path, index=False)
        logger.info(f"Saved error report to {save_path}")

    return report_df


def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot probability distribution for correct and incorrect predictions.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        model_name: Name of the model
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    config = get_config()

    if y_proba.ndim > 1 and y_proba.shape[1] > 1:
        phishing_proba = y_proba[:, 1]
    else:
        phishing_proba = y_proba

    # Create masks
    correct_mask = y_true == y_pred
    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot for legitimate class
    ax = axes[0]
    legitimate_correct = correct_mask & (y_true == 0)
    ax.hist(phishing_proba[legitimate_correct], bins=50, alpha=0.5, label="Correct (Legitimate)", color="blue")
    ax.hist(phishing_proba[fp_mask], bins=50, alpha=0.5, label="False Positive", color="red")
    ax.set_xlabel("Phishing Probability")
    ax.set_ylabel("Count")
    ax.set_title("Legitimate Emails")
    ax.legend()
    ax.set_xlim(0, 1)

    # Plot for phishing class
    ax = axes[1]
    phishing_correct = correct_mask & (y_true == 1)
    ax.hist(phishing_proba[phishing_correct], bins=50, alpha=0.5, label="Correct (Phishing)", color="green")
    ax.hist(phishing_proba[fn_mask], bins=50, alpha=0.5, label="False Negative", color="orange")
    ax.set_xlabel("Phishing Probability")
    ax.set_ylabel("Count")
    ax.set_title("Phishing Emails")
    ax.legend()
    ax.set_xlim(0, 1)

    fig.suptitle(f"{model_name} - Prediction Probability Distribution")
    plt.tight_layout()

    # Save figure
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=config.get("dpi", 300), bbox_inches='tight')
        logger.info(f"Saved error distribution plot to {save_path}")

    return fig
