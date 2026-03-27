"""Safe feature normalization utilities.

Ensures all features are normalized to [0, 1] range with robust handling
of NaN, Inf, and edge cases.
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class SafeMinMaxScaler:
    """Safe Min-Max scaler with robust error handling.

    Extends sklearn's MinMaxScaler with:
    - NaN/Inf handling (fills with 0)
    - Constant feature handling (avoids division by zero)
    - Output clipping to [0, 1] range
    - Preservation of feature names

    Attributes:
        scaler_: Internal sklearn MinMaxScaler instance.
        feature_names_in_: List of feature names seen during fit.
        n_features_in_: Number of features seen during fit.
    """

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)) -> None:
        """Initialize the scaler.

        Args:
            feature_range: Desired range of transformed data (default: [0, 1]).
        """
        self.feature_range = feature_range
        self.scaler_: Optional[MinMaxScaler] = None
        self.feature_names_in_: list[str] = []
        self.n_features_in_: int = 0
        self._is_fitted = False

    def fit(self, X: pd.DataFrame) -> "SafeMinMaxScaler":
        """Fit the scaler to the data.

        Args:
            X: DataFrame of features to fit scaler on.

        Returns:
            self: Fitted scaler instance.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a DataFrame, got {type(X)}")

        # Store feature metadata
        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = len(self.feature_names_in_)

        # Handle NaN and Inf before fitting
        X_clean = self._clean_data(X)

        # Initialize and fit sklearn scaler
        self.scaler_ = MinMaxScaler(feature_range=self.feature_range)
        self.scaler_.fit(X_clean)
        self._is_fitted = True

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted scaler.

        Args:
            X: DataFrame of features to transform.

        Returns:
            DataFrame with normalized features in [0, 1] range.

        Raises:
            RuntimeError: If transform is called before fit.
        """
        if not self._is_fitted:
            raise RuntimeError("SafeMinMaxScaler must be fitted before transform")

        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a DataFrame, got {type(X)}")

        # Handle NaN and Inf
        X_clean = self._clean_data(X)

        # Transform
        X_scaled = self.scaler_.transform(X_clean)

        # Clip to ensure [0, 1] range (handles floating point errors)
        X_scaled = np.clip(X_scaled, self.feature_range[0], self.feature_range[1])

        # Convert back to DataFrame
        result = pd.DataFrame(
            X_scaled,
            columns=X.columns,
            index=X.index,
        )

        return result

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit to data, then transform it.

        Args:
            X: DataFrame of features to fit and transform.

        Returns:
            DataFrame with normalized features in [0, 1] range.
        """
        self.fit(X)
        return self.transform(X)

    def _clean_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clean data by handling NaN and Inf values.

        Args:
            X: Input DataFrame.

        Returns:
            Cleaned DataFrame with NaN/Inf replaced by 0.
        """
        X_clean = X.copy()

        # Replace NaN with 0
        X_clean = X_clean.fillna(0)

        # Replace Inf with 0 (or column max/min if all finite)
        for col in X_clean.columns:
            # Handle positive infinity
            if np.isinf(X_clean[col]).any():
                finite_values = X_clean[col][np.isfinite(X_clean[col])]
                if len(finite_values) > 0:
                    col_max = finite_values.max()
                else:
                    col_max = 0
                X_clean.loc[np.isinf(X_clean[col]) & (X_clean[col] > 0), col] = col_max

                # Handle negative infinity
                if len(finite_values) > 0:
                    col_min = finite_values.min()
                else:
                    col_min = 0
                X_clean.loc[np.isinf(X_clean[col]) & (X_clean[col] < 0), col] = col_min

        return X_clean

    def get_feature_names_out(self) -> list[str]:
        """Get feature names after transformation.

        Returns:
            List of feature names (same as input).
        """
        if not self._is_fitted:
            raise RuntimeError("SafeMinMaxScaler must be fitted first")
        return self.feature_names_in_.copy()


def clip_to_unit_interval(series: pd.Series) -> pd.Series:
    """Clip a pandas Series to [0, 1] range.

    Utility function for individual feature clipping.

    Args:
        series: Input Series to clip.

    Returns:
        Clipped Series with values in [0, 1].
    """
    return series.clip(lower=0.0, upper=1.0)


def normalize_boolean(value: bool, default: float = 0.0) -> float:
    """Convert boolean to normalized float.

    Args:
        value: Boolean value to convert.
        default: Default value if conversion fails.

    Returns:
        1.0 if True, 0.0 if False, default if error.
    """
    try:
        return 1.0 if value else 0.0
    except (TypeError, ValueError):
        return default


def normalize_count(
    value: int, max_val: int, default: float = 0.0, clip: bool = True
) -> float:
    """Normalize count to [0, 1] by dividing by max_val.

    Args:
        value: Count value to normalize.
        max_val: Maximum expected value (for scaling).
        default: Default value if normalization fails.
        clip: Whether to clip result to [0, 1].

    Returns:
        Normalized value in [0, 1], or default on error.
    """
    try:
        if max_val <= 0:
            return default

        normalized = value / max_val
        if clip:
            normalized = max(0.0, min(1.0, normalized))
        return normalized
    except (TypeError, ZeroDivisionError, ValueError):
        return default


def normalize_length(
    text: str, max_length: int = 1000, default: float = 0.0, clip: bool = True
) -> float:
    """Normalize text length to [0, 1].

    Args:
        text: Text string to measure.
        max_length: Maximum expected length (for scaling).
        default: Default value if normalization fails.
        clip: Whether to clip result to [0, 1].

    Returns:
        Normalized length in [0, 1], or default on error.
    """
    try:
        length = len(text)
        return normalize_count(length, max_length, default, clip)
    except (TypeError, AttributeError):
        return default
