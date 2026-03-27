"""
Edge case identification and analysis.

Documents difficult cases and model limitations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

from src.models.classical.config import get_config

logger = logging.getLogger(__name__)


def identify_edge_cases(
    df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, List]:
    """
    Identify potential edge cases in the dataset.

    Args:
        df: Original DataFrame
        X: Feature array
        y: Labels
        y_proba: Predicted probabilities

    Returns:
        Dictionary with edge case indices and descriptions
    """
    edge_cases = {}

    # 1. Very short emails (may lack features)
    if "body_length" in df.columns:
        short_emails = df["body_length"] < 50
        edge_cases["short_emails"] = {
            "indices": np.where(short_emails)[0].tolist(),
            "description": "Very short emails (< 50 chars) - may lack sufficient features",
            "count": short_emails.sum()
        }

    # 2. Missing features
    missing_features = np.isnan(X).sum(axis=1)
    high_missing = missing_features > (X.shape[1] * 0.3)  # > 30% missing
    edge_cases["high_missing_features"] = {
        "indices": np.where(high_missing)[0].tolist(),
        "description": "Samples with > 30% missing features",
        "count": high_missing.sum()
    }

    # 3. Low confidence predictions
    if y_proba is not None:
        if y_proba.ndim > 1 and y_proba.shape[1] > 1:
            phishing_proba = y_proba[:, 1]
        else:
            phishing_proba = y_proba

        # Predictions near decision boundary
        uncertain_mask = (phishing_proba > 0.4) & (phishing_proba < 0.6)
        edge_cases["uncertain_predictions"] = {
            "indices": np.where(uncertain_mask)[0].tolist(),
            "description": "Predictions with probability between 0.4 and 0.6",
            "count": uncertain_mask.sum()
        }

    # 4. Extreme feature values (potential outliers)
    from scipy import stats
    z_scores = np.abs(stats.zscore(X, nan_policy='omit'))
    outliers = np.any(z_scores > 3, axis=1)  # Beyond 3 std dev
    edge_cases["outliers"] = {
        "indices": np.where(outliers)[0].tolist(),
        "description": "Samples with features beyond 3 standard deviations",
        "count": outliers.sum()
    }

    # 5. Duplicate samples
    if "email_id" in df.columns:
        duplicates = df.duplicated(subset=["email_id"], keep=False)
        edge_cases["duplicates"] = {
            "indices": np.where(duplicates)[0].tolist(),
            "description": "Duplicate email IDs",
            "count": duplicates.sum()
        }

    # 6. Encoding issues (non-ASCII content)
    if "body" in df.columns:
        # Check for high ratio of non-ASCII characters
        def has_encoding_issues(text):
            if pd.isna(text):
                return False
            try:
                text.encode('ascii')
                return False
            except UnicodeEncodeError:
                return True

        encoding_issues = df["body"].apply(has_encoding_issues)
        edge_cases["encoding_issues"] = {
            "indices": np.where(encoding_issues)[0].tolist(),
            "description": "Emails with encoding/non-ASCII issues",
            "count": encoding_issues.sum()
        }

    # 7. Class imbalance edge cases
    if "phishing_type" in df.columns:
        # Check for rare phishing types
        type_counts = df["phishing_type"].value_counts()
        rare_types = type_counts[type_counts < 10].index.tolist()

        for rare_type in rare_types:
            rare_mask = df["phishing_type"] == rare_type
            edge_cases[f"rare_type_{rare_type}"] = {
                "indices": np.where(rare_mask)[0].tolist(),
                "description": f"Rare phishing type: {rare_type} (< 10 samples)",
                "count": rare_mask.sum()
            }

    # Log summary
    logger.info("Edge Cases Identified:")
    for case_name, case_info in edge_cases.items():
        logger.info(f"  {case_name}: {case_info['count']} samples")

    return edge_cases


def create_edge_case_report(
    df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Create detailed edge case report.

    Args:
        df: Original DataFrame
        X: Feature array
        y: Labels
        y_proba: Predicted probabilities
        save_path: Path to save report

    Returns:
        DataFrame with edge case information
    """
    logger.info("Creating edge case report")

    # Identify edge cases
    edge_cases = identify_edge_cases(df, X, y, y_proba)

    # Create summary report
    report_data = []

    for case_name, case_info in edge_cases.items():
        report_data.append({
            "edge_case_type": case_name,
            "count": case_info["count"],
            "percentage": (case_info["count"] / len(df)) * 100,
            "description": case_info["description"]
        })

    report_df = pd.DataFrame(report_data)
    report_df = report_df.sort_values("count", ascending=False).reset_index(drop=True)

    # Save report
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(save_path, index=False)
        logger.info(f"Saved edge case report to {save_path}")

    return report_df


def analyze_edge_case_performance(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    edge_cases: Dict[str, List]
) -> pd.DataFrame:
    """
    Analyze model performance on edge cases.

    Args:
        df: DataFrame
        y_true: True labels
        y_pred: Predicted labels
        edge_cases: Dictionary from identify_edge_cases

    Returns:
        DataFrame with performance metrics per edge case type
    """
    from sklearn.metrics import accuracy_score, f1_score

    results = []

    # Overall performance
    overall_acc = accuracy_score(y_true, y_pred)
    overall_f1 = f1_score(y_true, y_pred)

    results.append({
        "edge_case_type": "overall",
        "n_samples": len(y_true),
        "accuracy": overall_acc,
        "f1": overall_f1,
        "accuracy_diff": 0.0,
        "f1_diff": 0.0
    })

    # Performance per edge case type
    for case_name, case_info in edge_cases.items():
        indices = case_info["indices"]

        if len(indices) == 0:
            continue

        y_true_ec = y_true[indices]
        y_pred_ec = y_pred[indices]

        acc = accuracy_score(y_true_ec, y_pred_ec)
        f1 = f1_score(y_true_ec, y_pred_ec)

        results.append({
            "edge_case_type": case_name,
            "n_samples": len(indices),
            "accuracy": acc,
            "f1": f1,
            "accuracy_diff": acc - overall_acc,
            "f1_diff": f1 - overall_f1
        })

    report_df = pd.DataFrame(results)

    logger.info("Edge Case Performance:")
    for _, row in report_df.iterrows():
        logger.info(f"  {row['edge_case_type']}: "
                   f"Acc={row['accuracy']:.3f} ({row['accuracy_diff']:+.3f}), "
                   f"F1={row['f1']:.3f} ({row['f1_diff']:+.3f})")

    return report_df


def save_edge_case_samples(
    df: pd.DataFrame,
    edge_cases: Dict[str, List],
    output_dir: Path
) -> None:
    """
    Save sample data for each edge case type.

    Args:
        df: DataFrame
        edge_cases: Dictionary from identify_edge_cases
        output_dir: Directory to save samples
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for case_name, case_info in edge_cases.items():
        indices = case_info["indices"]

        if len(indices) == 0:
            continue

        # Get samples
        samples = df.iloc[indices].copy()

        # Add edge case metadata
        samples["edge_case_type"] = case_name

        # Save to CSV
        csv_path = output_dir / f"{case_name}_samples.csv"
        samples.to_csv(csv_path, index=False)
        logger.info(f"Saved {len(samples)} {case_name} samples to {csv_path}")
