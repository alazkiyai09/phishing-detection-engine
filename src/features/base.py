"""Base abstract class for feature extractors.

All feature extractors must inherit from BaseExtractor and implement
the required methods for sklearn-compatible pipeline integration.
"""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseExtractor(ABC):
    """Abstract base class for all feature extractors.

    Provides a consistent interface for feature extraction with support for:
    - sklearn pipeline integration (fit/transform pattern)
    - Feature name tracking for interpretability
    - Graceful error handling for malformed data

    Attributes:
        feature_names (list[str]): List of feature names this extractor produces.
        extraction_times (list[float]): Timestamps of extraction operations for performance tracking.
    """

    def __init__(self) -> None:
        """Initialize the base extractor."""
        self.feature_names: list[str] = []
        self.extraction_times: list[float] = []
        self._is_fitted = False

    @abstractmethod
    def fit(self, emails: pd.DataFrame) -> "BaseExtractor":
        """Fit the extractor to the email data.

        For stateless extractors, this simply validates input and stores
        metadata. For stateful extractors (e.g., those requiring vocabulary
        building), this performs necessary preprocessing.

        Args:
            emails: DataFrame containing email data. Expected columns:
                - 'body': Email body text (str)
                - 'headers': Email headers (dict or str)
                - 'subject': Email subject line (str)
                - 'from_addr': Sender email address (str)

        Returns:
            self: Returns the instance itself for method chaining.

        Raises:
            ValueError: If required columns are missing from emails DataFrame.
        """
        # Validate input structure
        self._validate_input(emails)
        self._is_fitted = True
        return self

    @abstractmethod
    def transform(self, emails: pd.DataFrame) -> pd.DataFrame:
        """Transform emails into feature matrix.

        Extracts features from each email and returns a DataFrame where
        each column is a feature and each row is an email.

        Args:
            emails: DataFrame containing email data with same structure as fit().

        Returns:
            DataFrame with shape (n_emails, n_features). All features normalized
            to [0, 1] range. Missing or invalid values should be filled with 0.

        Raises:
            RuntimeError: If transform is called before fit.
        """
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before transform. "
                "Call fit() first."
            )
        return pd.DataFrame()

    def fit_transform(self, emails: pd.DataFrame) -> pd.DataFrame:
        """Fit to data, then transform it.

        Convenience method equivalent to fit().transform() but potentially
        more optimized for some extractors.

        Args:
            emails: DataFrame containing email data.

        Returns:
            Transformed feature matrix as DataFrame.
        """
        self.fit(emails)
        return self.transform(emails)

    def get_feature_names(self) -> list[str]:
        """Get names of extracted features.

        Returns:
            List of feature names. Must match column names in transform() output.
        """
        return self.feature_names.copy()

    def _validate_input(self, emails: pd.DataFrame) -> None:
        """Validate input DataFrame structure.

        Args:
            emails: DataFrame to validate.

        Raises:
            ValueError: If required columns are missing.
        """
        required_cols = {"body", "headers", "subject", "from_addr"}
        missing_cols = required_cols - set(emails.columns)
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Expected: {required_cols}, Got: {set(emails.columns)}"
            )

    def _safe_extract(
        self, value: Any, default: float = 0.0, transform_fn=None
    ) -> float:
        """Safely extract a feature value with error handling.

        Wraps feature extraction logic in try/except to prevent crashes
        on malformed or unexpected data.

        Args:
            value: Raw value to extract from (can be any type).
            default: Default value to return on error (must be in [0, 1]).
            transform_fn: Optional function to transform the value before returning.

        Returns:
            Extracted feature value in [0, 1] range, or default on error.
        """
        try:
            if value is None or (isinstance(value, float) and (pd.isna(value) or value == float("inf"))):
                return default

            if transform_fn is not None:
                result = transform_fn(value)
            else:
                result = float(value)

            # Ensure result is in valid range
            if isinstance(result, (int, float)):
                return max(0.0, min(1.0, float(result)))
            return default

        except (ValueError, TypeError, AttributeError, ZeroDivisionError):
            return default

    def get_extraction_stats(self) -> dict[str, float]:
        """Get performance statistics for feature extraction.

        Returns:
            Dictionary with extraction timing metrics:
                - 'total_emails': Number of emails processed
                - 'avg_time_ms': Average extraction time per email
                - 'features_count': Number of features extracted
        """
        if not self.extraction_times:
            return {"total_emails": 0, "avg_time_ms": 0.0, "features_count": len(self.feature_names)}

        return {
            "total_emails": len(self.extraction_times),
            "avg_time_ms": sum(self.extraction_times) / len(self.extraction_times),
            "features_count": len(self.feature_names),
        }

    def __repr__(self) -> str:
        """String representation of the extractor."""
        return f"{self.__class__.__name__}(n_features={len(self.feature_names)})"
