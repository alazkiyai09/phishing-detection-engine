"""Unit tests for SenderFeatureExtractor."""

import pytest
import pandas as pd


class TestSenderFeatureExtractor:
    """Test suite for Sender feature extraction."""

    @pytest.fixture
    def extractor(self):
        from src.feature_extractors.sender_features import SenderFeatureExtractor
        return SenderFeatureExtractor()

    @pytest.fixture
    def sample_emails(self):
        return pd.DataFrame({
            "body": ["Test"] * 4,
            "headers": [{}] * 4,
            "subject": ["Test"] * 4,
            "from_addr": [
                "user@gmail.com",
                "support@chase-secure.xyz",
                "admin@company.com",
                "notification@bank.com"
            ],
        })

    def test_freemail_detection(self, extractor, sample_emails):
        """Test freemail detection."""
        extractor.fit(sample_emails)
        result = extractor.transform(sample_emails)
        # First email uses gmail.com (freemail)
        assert result.iloc[0]["is_freemail"] == 1.0

    def test_all_values_in_range(self, extractor, sample_emails):
        """Test that all feature values are in [0, 1] range."""
        extractor.fit(sample_emails)
        result = extractor.transform(sample_emails)
        assert (result >= 0.0).all().all()
        assert (result <= 1.0).all().all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
