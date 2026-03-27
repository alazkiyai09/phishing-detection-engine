"""
Unit tests for cross-validation module.
"""

import pytest
import numpy as np
from src.models.logistic_regression import LogisticRegressionClassifier
from src.evaluation.cross_validation import stratified_cv, _clone_model


class TestCloneModel:
    """Tests for _clone_model function."""

    def test_clone_model_creates_new_instance(self):
        """Test that clone creates a new model instance."""
        model = LogisticRegressionClassifier(C=1.0, penalty="l2")
        clone = _clone_model(model)

        assert clone is not model
        assert isinstance(clone, LogisticRegressionClassifier)

    def test_clone_model_preserves_params(self):
        """Test that clone preserves hyperparameters."""
        model = LogisticRegressionClassifier(C=2.5, penalty="l1")
        clone = _clone_model(model)

        assert clone.C == 2.5
        assert clone.penalty == "l1"


class TestStratifiedCV:
    """Tests for stratified_cv function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        return X, y

    def test_stratified_cv_returns_results(self, sample_data):
        """Test that stratified CV returns results."""
        X, y = sample_data
        model = LogisticRegressionClassifier()

        aggregated, fold_results = stratified_cv(
            model=model,
            X=X,
            y=y,
            n_folds=3,
            random_state=42
        )

        assert isinstance(aggregated, dict)
        assert isinstance(fold_results, list)
        assert len(fold_results) == 3

    def test_stratified_cv_aggregates_metrics(self, sample_data):
        """Test that CV properly aggregates metrics."""
        X, y = sample_data
        model = LogisticRegressionClassifier()

        aggregated, _ = stratified_cv(
            model=model,
            X=X,
            y=y,
            n_folds=3,
            random_state=42
        )

        # Check that we have aggregated metrics
        assert "f1" in aggregated
        assert "auprc" in aggregated
        assert "auroc" in aggregated

        # Check (mean, std) format
        assert isinstance(aggregated["f1"], tuple)
        assert len(aggregated["f1"]) == 2

    def test_stratified_cv_fold_results(self, sample_data):
        """Test that each fold has proper results."""
        X, y = sample_data
        model = LogisticRegressionClassifier()

        _, fold_results = stratified_cv(
            model=model,
            X=X,
            y=y,
            n_folds=3,
            random_state=42
        )

        for fold_result in fold_results:
            assert "fold" in fold_result
            assert "f1" in fold_result
            assert "training_time" in fold_result
            assert "inference_time" in fold_result

    def test_stratified_cv_reproducibility(self, sample_data):
        """Test that CV is reproducible with same random state."""
        X, y = sample_data
        model1 = LogisticRegressionClassifier()
        model2 = LogisticRegressionClassifier()

        aggregated1, _ = stratified_cv(
            model=model1, X=X, y=y, n_folds=3, random_state=42
        )
        aggregated2, _ = stratified_cv(
            model=model2, X=X, y=y, n_folds=3, random_state=42
        )

        # Results should be identical
        assert aggregated1["f1"][0] == aggregated2["f1"][0]
