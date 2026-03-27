"""
Unit tests for evaluation metrics module.
"""

import pytest
import numpy as np
from src.evaluation.metrics import (
    compute_metrics,
    compute_auprc_auroc,
    compute_per_class_metrics,
    compute_fpr,
    aggregate_cv_results,
    format_cv_results
)


class TestComputeMetrics:
    """Tests for compute_metrics function."""

    def test_compute_metrics_basic(self):
        """Test basic metric computation."""
        y_true = np.array([0, 0, 1, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 1, 0, 0])
        y_proba = np.array([
            [0.9, 0.1],
            [0.8, 0.2],
            [0.3, 0.7],
            [0.2, 0.8],
            [0.6, 0.4],
            [0.9, 0.1]
        ])

        metrics = compute_metrics(y_true, y_pred, y_proba)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "auprc" in metrics
        assert "auroc" in metrics
        assert "fpr" in metrics

        # Check value ranges
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["auprc"] <= 1
        assert 0 <= metrics["auroc"] <= 1
        assert 0 <= metrics["fpr"] <= 1

    def test_compute_metrics_perfect_predictions(self):
        """Test with perfect predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([
            [0.9, 0.1],
            [0.8, 0.2],
            [0.1, 0.9],
            [0.2, 0.8]
        ])

        metrics = compute_metrics(y_true, y_pred, y_proba)

        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["fpr"] == 0.0


class TestComputeAUPRCAUROC:
    """Tests for compute_auprc_auroc function."""

    def test_auprc_auroc_ranges(self):
        """Test that AUPRC and AUROC are in valid ranges."""
        y_true = np.array([0, 0, 1, 1, 1, 0])
        y_proba = np.array([0.1, 0.2, 0.7, 0.8, 0.6, 0.3])

        auprc, auroc = compute_auprc_auroc(y_true, y_proba)

        assert 0 <= auprc <= 1
        assert 0 <= auroc <= 1

    def test_auprc_auroc_perfect(self):
        """Test with perfect separation."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.9, 0.8])

        auprc, auroc = compute_auprc_auroc(y_true, y_proba)

        # Should be close to 1.0
        assert auprc > 0.9
        assert auroc > 0.9


class TestComputePerClassMetrics:
    """Tests for compute_per_class_metrics function."""

    def test_per_class_metrics(self):
        """Test per-class metric computation."""
        y_true = np.array([0, 0, 1, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 1, 0, 0])
        y_proba = np.array([
            [0.9, 0.1],
            [0.8, 0.2],
            [0.3, 0.7],
            [0.2, 0.8],
            [0.6, 0.4],
            [0.9, 0.1]
        ])

        df = compute_per_class_metrics(y_true, y_pred, y_proba)

        assert "class" in df.columns
        assert "precision" in df.columns
        assert "recall" in df.columns
        assert "f1" in df.columns
        assert "support" in df.columns

        # Should have both classes
        assert len(df) == 2


class TestComputeFPR:
    """Tests for compute_fpr function."""

    def test_fpr_zero(self):
        """Test FPR with no false positives."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])

        fpr = compute_fpr(y_true, y_pred)
        assert fpr == 0.0

    def test_fpr_with_false_positives(self):
        """Test FPR with false positives."""
        # 2 legitimate, 1 is FP
        y_true = np.array([0, 0, 1])
        y_pred = np.array([1, 0, 1])

        fpr = compute_fpr(y_true, y_pred)
        assert fpr == 0.5  # 1 FP out of 2 negatives


class TestAggregateCVResults:
    """Tests for aggregate_cv_results function."""

    def test_aggregate_cv_results(self):
        """Test aggregation of CV results."""
        cv_results = [
            {"f1": 0.8, "auprc": 0.75},
            {"f1": 0.82, "auprc": 0.77},
            {"f1": 0.78, "auprc": 0.73},
        ]

        aggregated = aggregate_cv_results(cv_results)

        assert "f1" in aggregated
        assert "auprc" in aggregated

        # Check that we get (mean, std) tuples
        assert isinstance(aggregated["f1"], tuple)
        assert len(aggregated["f1"]) == 2

        mean, std = aggregated["f1"]
        assert abs(mean - 0.8) < 0.01  # Should be close to average

    def test_aggregate_cv_results_empty(self):
        """Test with empty results."""
        aggregated = aggregate_cv_results([])
        assert aggregated == {}


class TestFormatCVResults:
    """Tests for format_cv_results function."""

    def test_format_cv_results(self):
        """Test formatting of CV results."""
        aggregated = {
            "f1": (0.8, 0.02),
            "auprc": (0.75, 0.03)
        }

        df = format_cv_results(aggregated, "TestModel")

        assert "model" in df.columns
        assert "metric" in df.columns
        assert "mean" in df.columns
        assert "std" in df.columns
        assert "formatted" in df.columns

        assert len(df) == 2
        assert (df["model"] == "TestModel").all()
