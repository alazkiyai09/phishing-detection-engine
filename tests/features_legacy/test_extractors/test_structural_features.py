"""Unit tests for StructuralFeatureExtractor."""

import pytest
import pandas as pd


class TestStructuralFeatureExtractor:
    """Test suite for Structural feature extraction."""

    @pytest.fixture
    def extractor(self):
        from src.feature_extractors.structural_features import StructuralFeatureExtractor
        return StructuralFeatureExtractor()

    @pytest.fixture
    def sample_emails(self):
        return pd.DataFrame({
            "body": ["Test"] * 4,
            "headers": [{}] * 4,
            "subject": ["Test"] * 4,
            "from_addr": ["test@example.com"] * 4,
            "body_html": [
                "<p>Plain text</p>",
                "<div><img src='cid:image1'></div>",
                "<form action='submit'></form>",
                "<script>alert(1)</script><p>Test</p>"
            ],
            "attachments": [[], [], [{"name": "file.pdf"}], []],
        })

    def test_html_detection(self, extractor, sample_emails):
        """Test HTML content detection."""
        extractor.fit(sample_emails)
        result = extractor.transform(sample_emails)
        # First email has HTML with text content
        assert result.iloc[0]["html_text_ratio"] > 0
        # Note: Emails with only images or empty forms have html_text_ratio = 0

    def test_javascript_detection(self, extractor, sample_emails):
        """Test JavaScript detection."""
        extractor.fit(sample_emails)
        result = extractor.transform(sample_emails)
        # Fourth email has JavaScript
        assert result.iloc[3]["has_javascript"] == 1.0
        # First email doesn't
        assert result.iloc[0]["has_javascript"] == 0.0

    def test_form_detection(self, extractor, sample_emails):
        """Test form detection."""
        extractor.fit(sample_emails)
        result = extractor.transform(sample_emails)
        # Third email has a form
        assert result.iloc[2]["has_forms"] == 1.0

    def test_all_values_in_range(self, extractor, sample_emails):
        """Test that all feature values are in [0, 1] range."""
        extractor.fit(sample_emails)
        result = extractor.transform(sample_emails)
        assert (result >= 0.0).all().all()
        assert (result <= 1.0).all().all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
