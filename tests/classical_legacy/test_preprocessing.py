"""
Unit tests for preprocessing module.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.preprocessing import (
    separate_features_target,
    handle_missing_values,
    scale_features,
    get_phishing_subsets
)


class TestSeparateFeaturesTarget:
    """Tests for separate_features_target function."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame."""
        return pd.DataFrame({
            "label": [0, 1, 0, 1],
            "feature1": [1, 2, 3, 4],
            "feature2": [5, 6, 7, 8],
            "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]
        })

    def test_separate_features_target(self, sample_df):
        """Test separating features and target."""
        X, y = separate_features_target(sample_df, "label")

        assert X.shape == (4, 2)  # 4 samples, 2 features (date is metadata)
        assert len(y) == 4

    def test_separate_features_target_missing_label(self):
        """Test error when label column missing."""
        df = pd.DataFrame({"feature1": [1, 2, 3]})

        with pytest.raises(ValueError):
            separate_features_target(df, "label")


class TestHandleMissingValues:
    """Tests for handle_missing_values function."""

    @pytest.fixture
    def sample_data_with_missing(self):
        """Create sample data with missing values."""
        X_train = pd.DataFrame({
            "f1": [1, 2, np.nan, 4],
            "f2": [5, np.nan, 7, 8]
        })
        X_val = pd.DataFrame({
            "f1": [1, np.nan, 3],
            "f2": [5, 6, np.nan]
        })
        X_test = pd.DataFrame({
            "f1": [np.nan, 2, 3],
            "f2": [5, 6, 7]
        })
        return X_train, X_val, X_test

    def test_handle_missing_values(self, sample_data_with_missing):
        """Test missing value imputation."""
        X_train, X_val, X_test = sample_data_with_missing

        X_train_imp, X_val_imp, X_test_imp, imputer = handle_missing_values(
            X_train, X_val, X_test, strategy="median"
        )

        # Check no missing values remain
        assert not np.isnan(X_train_imp).any()
        assert not np.isnan(X_val_imp).any()
        assert not np.isnan(X_test_imp).any()

        # Check shapes preserved
        assert X_train_imp.shape == X_train.shape
        assert X_val_imp.shape == X_val.shape
        assert X_test_imp.shape == X_test.shape


class TestScaleFeatures:
    """Tests for scale_features function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        X_train = np.array([[1, 10], [2, 20], [3, 30]])
        X_val = np.array([[1.5, 15]])
        X_test = np.array([[2.5, 25]])
        return X_train, X_val, X_test

    def test_scale_features(self, sample_data):
        """Test feature scaling."""
        X_train, X_val, X_test = sample_data

        X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
            X_train, X_val, X_test, method="standard"
        )

        # Check training data is standardized
        assert abs(X_train_scaled.mean()) < 1e-10
        assert abs(X_train_scaled.std() - 1.0) < 1e-10

        # Check shapes preserved
        assert X_train_scaled.shape == X_train.shape
        assert X_val_scaled.shape == X_val.shape
        assert X_test_scaled.shape == X_test.shape


class TestGetPhishingSubsets:
    """Tests for get_phishing_subsets function."""

    @pytest.fixture
    def sample_df_with_types(self):
        """Create sample DataFrame with phishing types."""
        return pd.DataFrame({
            "label": [0, 1, 1, 1, 1, 0],
            "phishing_type": [None, "generic", "financial", "spear", "generic", None],
            "feature1": [1, 2, 3, 4, 5, 6]
        })

    def test_get_phishing_subsets(self, sample_df_with_types):
        """Test getting phishing type subsets."""
        subsets = get_phishing_subsets(sample_df_with_types, "phishing_type")

        # Check that we have all subsets
        assert "generic" in subsets
        assert "financial" in subsets
        assert "spear" in subsets
        assert "all_phishing" in subsets
        assert "legitimate" in subsets

        # Check subset sizes
        assert len(subsets["generic"]) == 2
        assert len(subsets["financial"]) == 1
        assert len(subsets["spear"]) == 1
        assert len(subsets["all_phishing"]) == 4
        assert len(subsets["legitimate"]) == 2
