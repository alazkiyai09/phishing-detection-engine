"""Unit tests for HeaderFeatureExtractor."""

import pytest
import pandas as pd


class TestHeaderFeatureExtractor:
    """Test suite for Header feature extraction."""

    @pytest.fixture
    def extractor(self):
        from src.feature_extractors.header_features import HeaderFeatureExtractor
        return HeaderFeatureExtractor()

    @pytest.fixture
    def sample_emails(self):
        return pd.DataFrame({
            "body": ["Test body"] * 4,
            "headers": [
                {"Authentication-Results": "spf=pass dkim=pass"},
                {"Authentication-Results": "spf=fail dkim=none"},
                {"Received": ["mta1", "mta2", "mta3"], "Received-SPF": "pass"},
                {}
            ],
            "subject": ["Test"] * 4,
            "from_addr": ["test@example.com"] * 4,
        })

    def test_fit(self, extractor, sample_emails):
        """Test fit method."""
        fitted = extractor.fit(sample_emails)
        assert fitted is extractor
        assert extractor._is_fitted

    def test_transform_before_fit_raises(self, extractor, sample_emails):
        """Test that transform before fit raises error."""
        with pytest.raises(RuntimeError, match="must be fitted"):
            extractor.transform(sample_emails)

    def test_spf_validation(self, extractor, sample_emails):
        """Test SPF validation features."""
        extractor.fit(sample_emails)
        result = extractor.transform(sample_emails)
        # First email has spf=pass in Authentication-Results
        assert result.iloc[0]["spf_pass"] == 1.0
        # Second email has spf=fail
        assert result.iloc[1]["spf_fail"] == 1.0

    def test_hop_count(self, extractor, sample_emails):
        """Test hop count feature."""
        extractor.fit(sample_emails)
        result = extractor.transform(sample_emails)
        # Third email has 3 Received headers (1 from list, 2 from count)
        assert result.iloc[2]["hop_count"] > 0

    def test_all_values_in_range(self, extractor, sample_emails):
        """Test that all feature values are in [0, 1] range."""
        extractor.fit(sample_emails)
        result = extractor.transform(sample_emails)
        assert (result >= 0.0).all().all()
        assert (result <= 1.0).all().all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
