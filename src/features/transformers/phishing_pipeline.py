"""Main sklearn-compatible pipeline for phishing feature extraction.

Orchestrates all feature extractors into a single transformer.
"""

import time
from typing import List, Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ..feature_extractors.base import BaseExtractor
from ..feature_extractors.content_features import ContentFeatureExtractor
from ..feature_extractors.financial_features import FinancialFeatureExtractor
from ..feature_extractors.header_features import HeaderFeatureExtractor
from ..feature_extractors.linguistic_features import LinguisticFeatureExtractor
from ..feature_extractors.sender_features import SenderFeatureExtractor
from ..feature_extractors.structural_features import StructuralFeatureExtractor
from ..feature_extractors.url_features import URLFeatureExtractor
from .normalizer import SafeMinMaxScaler


class PhishingFeaturePipeline(BaseEstimator, TransformerMixin):
    """Main pipeline for phishing email feature extraction.

    Integrates all feature extractors into a sklearn-compatible transformer.

    Example:
        >>> pipeline = PhishingFeaturePipeline()
        >>> features = pipeline.fit_transform(emails_df)
        >>> feature_names = pipeline.get_feature_names()

    Attributes:
        extractors: List of feature extractors to apply.
        scaler: Scaler for normalizing features to [0, 1].
        feature_names_: Combined list of all feature names.
        fitted_: Whether pipeline has been fitted.
    """

    def __init__(
        self,
        extractors: Optional[List[BaseExtractor]] = None,
        normalize: bool = True,
    ) -> None:
        """Initialize the pipeline.

        Args:
            extractors: List of feature extractors. If None, uses default extractors.
            normalize: Whether to normalize features to [0, 1].
        """
        if extractors is None:
            # Default extractors (all feature categories)
            extractors = [
                URLFeatureExtractor(),
                HeaderFeatureExtractor(),
                SenderFeatureExtractor(),
                ContentFeatureExtractor(),
                StructuralFeatureExtractor(),
                LinguisticFeatureExtractor(),
                FinancialFeatureExtractor(),  # KEY DIFFERENTIATOR
            ]

        self.extractors = extractors
        self.normalize = normalize
        self.scaler = SafeMinMaxScaler() if normalize else None
        self.feature_names_: List[str] = []
        self.fitted_ = False

    def fit(self, emails: pd.DataFrame, y=None) -> "PhishingFeaturePipeline":
        """Fit all feature extractors to the email data.

        Args:
            emails: DataFrame containing email data with required columns:
                - 'body': Email body text (str)
                - 'headers': Email headers (dict or str)
                - 'subject': Email subject line (str)
                - 'from_addr': Sender email address (str)
                - 'body_html': HTML body (str, optional)
                - 'attachments': List of attachment metadata (optional)
            y: Ignored (present for sklearn compatibility).

        Returns:
            self: Fitted pipeline.

        Raises:
            ValueError: If required columns are missing.
        """
        # Validate input
        self._validate_input(emails)

        # Fit each extractor
        self.feature_names_ = []
        for extractor in self.extractors:
            extractor.fit(emails)
            self.feature_names_.extend(extractor.get_feature_names())

        # Fit scaler if normalization enabled
        if self.normalize and self.scaler is not None:
            # Extract features to fit scaler
            features_df = self._extract_all_features(emails)
            self.scaler.fit(features_df)

        self.fitted_ = True
        return self

    def transform(self, emails: pd.DataFrame) -> pd.DataFrame:
        """Transform emails into feature matrix.

        Args:
            emails: DataFrame containing email data (same structure as fit()).

        Returns:
            DataFrame with shape (n_emails, n_features). All features
            normalized to [0, 1] range if normalize=True.

        Raises:
            RuntimeError: If transform is called before fit.
        """
        if not self.fitted_:
            raise RuntimeError("PhishingFeaturePipeline must be fitted before transform")

        # Extract features from all extractors
        features_df = self._extract_all_features(emails)

        # Normalize if enabled
        if self.normalize and self.scaler is not None:
            features_df = self.scaler.transform(features_df)

        return features_df

    def fit_transform(
        self, emails: pd.DataFrame, y=None
    ) -> pd.DataFrame:
        """Fit to data, then transform it.

        Args:
            emails: DataFrame containing email data.
            y: Ignored (present for sklearn compatibility).

        Returns:
            Transformed feature matrix as DataFrame.
        """
        self.fit(emails, y)
        return self.transform(emails)

    def get_feature_names(self) -> List[str]:
        """Get names of all extracted features.

        Returns:
            List of feature names. Matches columns in transform() output.
        """
        return self.feature_names_.copy()

    def get_feature_names_out(self) -> List[str]:
        """Get feature names (sklearn convention).

        Returns:
            List of feature names.
        """
        return self.get_feature_names()

    def get_extraction_stats(self) -> dict[str, dict[str, float]]:
        """Get extraction statistics for each extractor.

        Returns:
            Dictionary mapping extractor names to their stats:
                - extractor_name: {
                    'total_emails': int,
                    'avg_time_ms': float,
                    'features_count': int
                  }
        """
        if not self.fitted_:
            return {}

        stats = {}
        for extractor in self.extractors:
            extractor_name = extractor.__class__.__name__
            stats[extractor_name] = extractor.get_extraction_stats()

        return stats

    def print_extraction_summary(self) -> None:
        """Print extraction statistics summary."""
        stats = self.get_extraction_stats()

        print("\n" + "=" * 60)
        print("PHISHING FEATURE PIPELINE - EXTRACTION SUMMARY")
        print("=" * 60)

        total_features = 0
        total_time = 0.0

        for extractor_name, extractor_stats in stats.items():
            features_count = extractor_stats["features_count"]
            avg_time = extractor_stats["avg_time_ms"]
            total_emails = extractor_stats["total_emails"]

            total_features += features_count
            total_time += avg_time

            print(f"\n{extractor_name}:")
            print(f"  Features: {features_count}")
            print(f"  Avg Time: {avg_time:.2f} ms/email")
            print(f"  Total Emails: {total_emails}")

        print("\n" + "-" * 60)
        print(f"TOTAL FEATURES: {total_features}")
        print(f"TOTAL AVG TIME: {total_time:.2f} ms/email")
        print("=" * 60 + "\n")

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
                f"Expected at least: {required_cols}, "
                f"Got: {set(emails.columns)}"
            )

    def _extract_all_features(self, emails: pd.DataFrame) -> pd.DataFrame:
        """Extract features from all extractors.

        Args:
            emails: DataFrame of emails.

        Returns:
            Combined DataFrame of all features.
        """
        all_features = []

        for extractor in self.extractors:
            start_time = time.time()
            features_df = extractor.transform(emails)
            all_features.append(features_df)

        # Concatenate all feature DataFrames
        combined = pd.concat(all_features, axis=1)

        return combined

    def __repr__(self) -> str:
        """String representation of the pipeline."""
        n_extractors = len(self.extractors)
        n_features = len(self.feature_names_) if self.fitted_ else 0
        return (
            f"PhishingFeaturePipeline("
            f"n_extractors={n_extractors}, "
            f"n_features={n_features}, "
            f"normalize={self.normalize})"
        )


def create_default_pipeline() -> PhishingFeaturePipeline:
    """Create pipeline with default extractors.

    Convenience function for creating a standard pipeline.

    Returns:
        PhishingFeaturePipeline with all default extractors.
    """
    return PhishingFeaturePipeline()


def create_custom_pipeline(
    include_url: bool = True,
    include_header: bool = True,
    include_sender: bool = True,
    include_content: bool = True,
    include_structural: bool = True,
    include_linguistic: bool = True,
    include_financial: bool = True,  # Always include for banking context
    normalize: bool = True,
) -> PhishingFeaturePipeline:
    """Create pipeline with custom extractor configuration.

    Args:
        include_url: Include URL features.
        include_header: Include header features.
        include_sender: Include sender features.
        include_content: Include content features.
        include_structural: Include structural features.
        include_linguistic: Include linguistic features.
        include_financial: Include financial features (recommended).
        normalize: Whether to normalize features to [0, 1].

    Returns:
        Configured PhishingFeaturePipeline.
    """
    extractors = []

    if include_url:
        extractors.append(URLFeatureExtractor())
    if include_header:
        extractors.append(HeaderFeatureExtractor())
    if include_sender:
        extractors.append(SenderFeatureExtractor())
    if include_content:
        extractors.append(ContentFeatureExtractor())
    if include_structural:
        extractors.append(StructuralFeatureExtractor())
    if include_linguistic:
        extractors.append(LinguisticFeatureExtractor())
    if include_financial:
        extractors.append(FinancialFeatureExtractor())

    if not extractors:
        raise ValueError("At least one feature extractor must be enabled")

    return PhishingFeaturePipeline(extractors=extractors, normalize=normalize)
