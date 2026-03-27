"""Unit tests for ContentFeatureExtractor."""

import pytest
import pandas as pd


class TestContentFeatureExtractor:
    """Test suite for Content feature extraction."""

    @pytest.fixture
    def extractor(self):
        from src.feature_extractors.content_features import ContentFeatureExtractor
        return ContentFeatureExtractor()

    @pytest.fixture
    def sample_emails(self):
        return pd.DataFrame({
            "body": [
                "URGENT: Verify your account now or it will be closed",
                "Click here to login immediately",
                "Your account will be suspended",
                "Normal email text"
            ],
            "headers": [{}] * 4,
            "subject": ["Test"] * 4,
            "from_addr": ["test@example.com"] * 4,
        })

    def test_urgency_detection(self, extractor, sample_emails):
        """Test urgency keyword detection."""
        extractor.fit(sample_emails)
        result = extractor.transform(sample_emails)
        assert result.iloc[0]["urgency_keyword_count"] > 0
        assert result.iloc[3]["urgency_keyword_count"] == 0.0

    def test_cta_detection(self, extractor, sample_emails):
        """Test call-to-action detection."""
        extractor.fit(sample_emails)
        result = extractor.transform(sample_emails)
        # "Click here" appears in second email
        assert result.iloc[1]["cta_button_count"] > 0

    def test_all_values_in_range(self, extractor, sample_emails):
        """Test that all feature values are in [0, 1] range."""
        extractor.fit(sample_emails)
        result = extractor.transform(sample_emails)
        assert (result >= 0.0).all().all()
        assert (result <= 1.0).all().all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
