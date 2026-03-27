"""Unit tests for LinguisticFeatureExtractor."""

import pytest
import pandas as pd


class TestLinguisticFeatureExtractor:
    """Test suite for Linguistic feature extraction."""

    @pytest.fixture
    def extractor(self):
        from src.feature_extractors.linguistic_features import LinguisticFeatureExtractor
        return LinguisticFeatureExtractor()

    @pytest.fixture
    def sample_emails(self):
        return pd.DataFrame({
            "body": [
                "Click here NOW!!! Act immediately.",
                "This is a normal email with proper grammar.",
                "MISTAKES in spellling and grammar are here.",
                "WHAT ARE YOU DOING???"
            ],
            "headers": [{}] * 4,
            "subject": ["Test"] * 4,
            "from_addr": ["test@example.com"] * 4,
        })

    def test_exclamation_detection(self, extractor, sample_emails):
        """Test exclamation mark detection."""
        extractor.fit(sample_emails)
        result = extractor.transform(sample_emails)
        # First email has exclamation marks
        assert result.iloc[0]["exclamation_mark_count"] > 0

    def test_all_caps_detection(self, extractor, sample_emails):
        """Test ALL CAPS detection."""
        extractor.fit(sample_emails)
        result = extractor.transform(sample_emails)
        # Fourth email has ALL CAPS
        assert result.iloc[3]["all_caps_ratio"] > 0

    def test_all_values_in_range(self, extractor, sample_emails):
        """Test that all feature values are in [0, 1] range."""
        extractor.fit(sample_emails)
        result = extractor.transform(sample_emails)
        assert (result >= 0.0).all().all()
        assert (result <= 1.0).all().all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
