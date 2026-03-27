"""
Confusion matrix with example emails.

Extracts representative samples from each confusion matrix cell.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import seaborn as sns
import logging

from src.models.classical.config import get_config

logger = logging.getLogger(__name__)


def get_confusion_examples(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    df: pd.DataFrame,
    n_examples: int = 5
) -> Dict[str, pd.DataFrame]:
    """
    Get example emails for each confusion matrix cell.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        df: DataFrame with email data
        n_examples: Number of examples per cell

    Returns:
        Dictionary mapping cell names to example DataFrames
    """
    examples = {}

    # True Negatives: Legitimate correctly classified as legitimate
    tn_mask = (y_true == 0) & (y_pred == 0)
    tn_indices = np.where(tn_mask)[0]
    if len(tn_indices) > 0:
        examples["true_negatives"] = df.iloc[tn_indices[:n_examples]].copy()

    # False Positives: Legitimate incorrectly classified as phishing
    fp_mask = (y_true == 0) & (y_pred == 1)
    fp_indices = np.where(fp_mask)[0]
    if len(fp_indices) > 0:
        examples["false_positives"] = df.iloc[fp_indices[:n_examples]].copy()

    # False Negatives: Phishing incorrectly classified as legitimate
    fn_mask = (y_true == 1) & (y_pred == 0)
    fn_indices = np.where(fn_mask)[0]
    if len(fn_indices) > 0:
        examples["false_negatives"] = df.iloc[fn_indices[:n_examples]].copy()

    # True Positives: Phishing correctly classified as phishing
    tp_mask = (y_true == 1) & (y_pred == 1)
    tp_indices = np.where(tp_mask)[0]
    if len(tp_indices) > 0:
        examples["true_positives"] = df.iloc[tp_indices[:n_examples]].copy()

    logger.info(f"Extracted examples from confusion matrix:")
    for cell_name, cell_df in examples.items():
        logger.info(f"  {cell_name}: {len(cell_df)} examples")

    return examples


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    class_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        class_names: Names for classes
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import confusion_matrix

    config = get_config()

    if class_names is None:
        class_names = config.get("class_names", ["Class 0", "Class 1"])

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot counts
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        ax=axes[0]
    )
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_title(f'{model_name} - Confusion Matrix (Counts)')

    # Plot percentages
    sns.heatmap(
        cm_normalized, annot=True, fmt='.2%', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        ax=axes[1]
    )
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_title(f'{model_name} - Confusion Matrix (Percentages)')

    plt.tight_layout()

    # Save figure
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=config.get("dpi", 300), bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {save_path}")

    return fig


def plot_confusion_matrix_with_examples(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    df: pd.DataFrame,
    model_name: str = "Model",
    n_examples: int = 3,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot confusion matrix annotated with example emails.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        df: DataFrame with email data
        model_name: Name of the model
        n_examples: Number of examples to show per cell
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import confusion_matrix

    config = get_config()

    # Get examples
    examples = get_confusion_examples(y_true, y_pred, df, n_examples)

    # Create figure
    fig = plt.figure(figsize=(16, 12))

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    ax = plt.subplot(2, 1, 1)

    class_names = config.get("class_names", ["Legitimate", "Phishing"])
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        ax=ax
    )
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title(f'{model_name} - Confusion Matrix')

    # Add example text
    example_text = ""
    for cell_type, cell_df in examples.items():
        example_text += f"\n{'='*60}\n"
        example_text += f"{cell_type.upper().replace('_', ' ')}\n"
        example_text += f"{'='*60}\n"

        for idx, row in cell_df.iterrows():
            # Get subject if available
            if "subject" in row:
                example_text += f"\nSubject: {row['subject']}\n"

            # Get body preview if available
            if "body" in row:
                body_preview = str(row['body'])[:200]
                example_text += f"Body: {body_preview}...\n"

            example_text += "-" * 40 + "\n"

    plt.subplot(2, 1, 2)
    plt.axis('off')
    plt.text(0, 1, example_text, fontsize=8, verticalalignment='top', fontfamily='monospace')
    plt.title('Example Emails from Confusion Matrix')

    plt.tight_layout()

    # Save figure
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=config.get("dpi", 300), bbox_inches='tight')
        logger.info(f"Saved confusion matrix with examples to {save_path}")

    return fig


def create_confusion_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    df: pd.DataFrame,
    model_name: str,
    output_dir: Path
) -> Dict[str, pd.DataFrame]:
    """
    Create comprehensive confusion matrix report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        df: DataFrame with email data
        model_name: Name of the model
        output_dir: Directory to save outputs

    Returns:
        Dictionary with example DataFrames
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating confusion report for {model_name}")

    # Get examples
    examples = get_confusion_examples(y_true, y_pred, df, n_examples=10)

    # Save examples to CSV
    for cell_name, cell_df in examples.items():
        csv_path = output_dir / f"{model_name}_{cell_name}_examples.csv"
        cell_df.to_csv(csv_path, index=False)
        logger.info(f"Saved {cell_name} examples to {csv_path}")

    # Plot confusion matrix
    cm_path = output_dir / f"{model_name}_confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, model_name, save_path=cm_path)

    # Plot confusion matrix with examples
    cm_examples_path = output_dir / f"{model_name}_confusion_matrix_with_examples.png"
    plot_confusion_matrix_with_examples(y_true, y_pred, df, model_name, save_path=cm_examples_path)

    return examples
