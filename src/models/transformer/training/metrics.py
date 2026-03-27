"""
Evaluation metrics for phishing detection.
Matches Day 2 metrics for fair comparison.
"""
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns


def compute_auprc(
    probs: np.ndarray,
    labels: np.ndarray
) -> float:
    """
    Compute Area Under Precision-Recall Curve.

    Args:
        probs: Predicted probabilities for positive class [n_samples]
        labels: Ground truth labels [n_samples]

    Returns:
        AUPRC score
    """
    precision, recall, _ = precision_recall_curve(labels, probs)
    return auc(recall, precision)


def compute_auroc(
    probs: np.ndarray,
    labels: np.ndarray
) -> float:
    """
    Compute Area Under ROC Curve.

    Args:
        probs: Predicted probabilities for positive class [n_samples]
        labels: Ground truth labels [n_samples]

    Returns:
        AUROC score
    """
    fpr, tpr, _ = roc_curve(labels, probs)
    return auc(fpr, tpr)


def compute_all_metrics(
    probs: np.ndarray,
    preds: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """
    Compute all metrics matching Day 2 benchmark.

    Args:
        probs: Predicted probabilities [n_samples]
        preds: Predicted labels [n_samples]
        labels: Ground truth labels [n_samples]

    Returns:
        Dictionary with all metrics
    """
    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, average='binary'),
        'recall': recall_score(labels, preds, average='binary'),
        'f1': f1_score(labels, preds, average='binary'),
        'auprc': compute_auprc(probs, labels),
        'auroc': compute_auroc(probs, labels),
        'fpr': compute_fpr_at_tpr(probs, labels, target_tpr=0.95)
    }


def compute_fpr_at_tpr(
    probs: np.ndarray,
    labels: np.ndarray,
    target_tpr: float = 0.95
) -> float:
    """
    Compute False Positive Rate at a target True Positive Rate using probability thresholds.

    Important for financial sector requirements (95% recall with <1% FPR).
    This finds the decision threshold where TPR >= target_tpr, then computes FPR at that threshold.

    Args:
        probs: Predicted probabilities for positive class [n_samples]
        labels: Ground truth labels [n_samples]
        target_tpr: Target recall (0.95 for financial sector)

    Returns:
        FPR at target TPR
    """
    import logging
    logger = logging.getLogger(__name__)

    # Edge case: single class in labels
    if len(np.unique(labels)) < 2:
        logger.warning("Only one class present in labels. Returning FPR=1.0")
        return 1.0

    # Sort by probability (descending) - most likely phishing first
    sorted_indices = np.argsort(-probs)
    sorted_labels = labels[sorted_indices]

    # Find threshold where TPR >= target_tpr
    total_pos = (labels == 1).sum()
    target_pos = int(target_tpr * total_pos)

    # Edge case: not enough positive samples
    if target_pos == 0:
        logger.warning(f"Only {total_pos} positive samples, target_tpr={target_tpr} gives target_pos=0")
        target_pos = 1

    # Count positives until we reach target
    cumulative_pos = 0
    threshold_idx = len(sorted_labels) - 1  # Default to all samples

    for i, label in enumerate(sorted_labels):
        if label == 1:
            cumulative_pos += 1
            if cumulative_pos >= target_pos:
                threshold_idx = i
                break

    # Get probability threshold
    threshold_prob = probs[sorted_indices[threshold_idx]]

    # Compute predictions at this threshold
    preds_at_threshold = (probs >= threshold_prob).astype(int)

    # Compute confusion matrix at this threshold
    try:
        cm = confusion_matrix(labels, preds_at_threshold)

        # Handle edge case: single-class prediction
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        elif cm.shape == (1, 1):
            # All predictions were the same class
            if labels[0] == 0:
                tn, fp, fn, tp = cm[0, 0], 0, 0, 0
            else:
                tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
        else:
            logger.warning(f"Unexpected confusion matrix shape: {cm.shape}")
            return 1.0

    except ValueError as e:
        logger.error(f"Error computing confusion matrix: {e}")
        return 1.0

    # Compute final metrics
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0

    logger.debug(f"FPR at {target_tpr:.0%} TPR: fpr={fpr:.4f}, tpr={tpr:.4f}, threshold={threshold_prob:.4f}")

    return fpr


def extract_attention_maps(
    attentions: torch.Tensor,
    tokens: List[str],
    layer: int = -1,
    head: int = 0
) -> np.ndarray:
    """
    Extract attention map for visualization.

    Args:
        attentions: Attention tensor from model [num_layers, batch, num_heads, seq_len, seq_len]
        tokens: List of tokens
        layer: Which layer to extract
        head: Which head to extract

    Returns:
        Attention matrix [seq_len, seq_len]
    """
    # Extract specific layer and head
    attention_map = attentions[layer, 0, head].cpu().numpy()

    return attention_map


def plot_attention_heatmap(
    attention_map: np.ndarray,
    tokens: List[str],
    save_path: Optional[str] = None,
    title: str = "Attention Weights"
) -> None:
    """
    Plot attention heatmap.

    Args:
        attention_map: Attention matrix [seq_len, seq_len]
        tokens: List of tokens
        save_path: Path to save figure
        title: Plot title
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        attention_map,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='viridis',
        cbar=True
    )
    plt.title(title)
    plt.xlabel('Keys')
    plt.ylabel('Queries')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Attention heatmap saved to {save_path}")

    plt.close()


def compute_calibration_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute calibration curve for confidence analysis.

    Args:
        probs: Predicted probabilities [n_samples]
        labels: Ground truth labels [n_samples]
        n_bins: Number of bins

    Returns:
        Tuple of (bin_confidences, bin_accuracies)
    """
    from sklearn.calibration import calibration_curve

    bin_confidences, bin_accuracies = calibration_curve(
        labels, probs, n_bins=n_bins, strategy='uniform'
    )

    return bin_confidences, bin_accuracies


def plot_calibration_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Plot calibration curve.

    Args:
        probs: Predicted probabilities [n_samples]
        labels: Ground truth labels [n_samples]
        save_path: Path to save figure
    """
    from sklearn.calibration import calibration_curve

    bin_confidences, bin_accuracies = calibration_curve(
        labels, probs, n_bins=10, strategy='uniform'
    )

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.plot(bin_confidences, bin_accuracies, 'o-', label='Model')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Calibration curve saved to {save_path}")

    plt.close()


def compute_per_class_metrics(
    probs: np.ndarray,
    preds: np.ndarray,
    labels: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics per class (safe vs phishing).

    Args:
        probs: Predicted probabilities [n_samples, 2]
        preds: Predicted labels [n_samples]
        labels: Ground truth labels [n_samples]

    Returns:
        Dictionary with per-class metrics
    """
    return {
        'safe': {
            'precision': precision_score(labels, preds, labels=[0], average='micro'),
            'recall': recall_score(labels, preds, labels=[0], average='micro'),
            'f1': f1_score(labels, preds, labels=[0], average='micro'),
        },
        'phishing': {
            'precision': precision_score(labels, preds, labels=[1], average='micro'),
            'recall': recall_score(labels, preds, labels=[1], average='micro'),
            'f1': f1_score(labels, preds, labels=[1], average='micro'),
        }
    }
