"""
Evaluation module for classifier benchmark.

Provides metrics, cross-validation, and temporal splitting.
"""

from src.models.classical.evaluation.metrics import (
    compute_metrics,
    compute_auprc_auroc,
    compute_per_class_metrics,
    compute_fpr
)
from src.models.classical.evaluation.cross_validation import stratified_cv
from src.models.classical.evaluation.temporal_split import temporal_evaluation

__all__ = [
    "compute_metrics",
    "compute_auprc_auroc",
    "compute_per_class_metrics",
    "compute_fpr",
    "stratified_cv",
    "temporal_evaluation",
]
