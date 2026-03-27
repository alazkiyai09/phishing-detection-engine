"""Unit tests for URLFeatureExtractor."""

import pytest
import pandas as pd


class TestURLFeatureExtractor:
    """Test suite for URL feature extraction."""

    @pytest.fixture
    def extractor(self):
        """Create URLFeatureExtractor instance."""
        from src.feature_extractors.url_features import URLFeatureExtractor

        return URLFeatureExtractor()

    @pytest.fixture
    def sample_emails(self):
        """Create sample email data."""
        return pd.DataFrame(
            {
                "body": [
                    # Email with suspicious URLs
                    "Click here http://192.168.1.1/login to verify your account.",
                    # Email with normal URL
                    "Visit https://www.chase.com for more info.",
                    # Email with multiple URLs
                    "Link1: http://suspicious.xyz/login Link2: http://verify-account.top",
                    # Email with no URLs
                    "This is a plain text email with no links.",
                ],
                "headers": [{}] * 4,
                "subject": ["Test"] * 4,
                "from_addr": ["test@example.com"] * 4,
            }
        )

    def test_fit(self, extractor, sample_emails):
        """Test fit method."""
        fitted = extractor.fit(sample_emails)
        assert fitted is extractor
        assert extractor._is_fitted

    def test_transform_before_fit_raises(self, extractor, sample_emails):
        """Test that transform before fit raises error."""
        with pytest.raises(RuntimeError, match="must be fitted"):
            extractor.transform(sample_emails)

    def test_transform_returns_dataframe(self, extractor, sample_emails):
        """Test that transform returns DataFrame with correct shape."""
        extractor.fit(sample_emails)
        result = extractor.transform(sample_emails)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (4, 10)  # 4 emails, 10 features
        assert list(result.columns) == extractor.feature_names

    def test_url_count_feature(self, extractor, sample_emails):
        """Test url_count feature extraction."""
        extractor.fit(sample_emails)
        result = extractor.transform(sample_emails)

        # First email: 1 URL
        assert result.iloc[0]["url_count"] > 0

        # Fourth email: 0 URLs
        assert result.iloc[3]["url_count"] == 0.0

    def test_has_ip_url_feature(self, extractor, sample_emails):
        """Test has_ip_url feature extraction."""
        extractor.fit(sample_emails)
        result = extractor.transform(sample_emails)

        # First email has IP address
        assert result.iloc[0]["has_ip_url"] == 1.0

        # Second email has domain (not IP)
        assert result.iloc[1]["has_ip_url"] == 0.0

    def test_suspicious_tld_feature(self, extractor, sample_emails):
        """Test has_suspicious_tld feature extraction."""
        extractor.fit(sample_emails)
        result = extractor.transform(sample_emails)

        # Third email has .xyz and .top TLDs
        assert result.iloc[2]["has_suspicious_tld"] == 1.0

        # Second email has .com (not suspicious)
        assert result.iloc[1]["has_suspicious_tld"] == 0.0

    def test_all_values_in_range(self, extractor, sample_emails):
        """Test that all feature values are in [0, 1] range."""
        extractor.fit(sample_emails)
        result = extractor.transform(sample_emails)

        assert (result >= 0.0).all().all()
        assert (result <= 1.0).all().all()

    def test_get_feature_names(self, extractor, sample_emails):
        """Test get_feature_names method."""
        extractor.fit(sample_emails)
        names = extractor.get_feature_names()

        assert isinstance(names, list)
        assert len(names) == 10
        assert "url_count" in names
        assert "has_ip_url" in names

    def test_get_extraction_stats(self, extractor, sample_emails):
        """Test extraction statistics tracking."""
        extractor.fit(sample_emails)
        extractor.transform(sample_emails)

        stats = extractor.get_extraction_stats()

        assert "total_emails" in stats
        assert "avg_time_ms" in stats
        assert "features_count" in stats
        assert stats["total_emails"] == 4
        assert stats["features_count"] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
