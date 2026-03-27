"""Unit tests for PhishingFeaturePipeline."""

import pytest
import pandas as pd
import numpy as np


class TestPhishingFeaturePipeline:
    """Test suite for the main feature pipeline."""

    @pytest.fixture
    def sample_emails(self):
        return pd.DataFrame({
            "body": [
                "Verify your account at http://chase-secure.xyz",
                "Hello, here is your monthly statement.",
                "URGENT: wire transfer immediately",
                "Normal business email"
            ] * 3,
            "headers": [
                {"spf": "pass", "dkim": "pass", "dmarc": "pass"},
                {"spf": "fail", "dkim": "none", "dmarc": "fail"},
                {"Received": ["mta1", "mta2"]},
                {}
            ] * 3,
            "subject": [
                "Account Verification",
                "Monthly Statement",
                "Urgent Action Required",
                "Business Update"
            ] * 3,
            "from_addr": [
                "support@chase-secure.xyz",
                "noreply@bank.com",
                "alerts@wire-transfer.com",
                "contact@company.com"
            ] * 3,
            "body_html": ["<p>Email</p>"] * 12,
            "attachments": [[]] * 12,
        })

    def test_pipeline_fit_transform(self, sample_emails):
        """Test the complete pipeline."""
        from src.transformers.phishing_pipeline import PhishingFeaturePipeline

        pipeline = PhishingFeaturePipeline()
        features = pipeline.fit_transform(sample_emails)

        # Check that we got features
        assert features.shape[0] == 12  # 12 emails
        assert features.shape[1] > 50  # Should have 60+ features

        # Check all values are in [0, 1]
        assert (features >= 0.0).all().all()
        assert (features <= 1.0).all().all()

    def test_feature_names(self, sample_emails):
        """Test that feature names are generated correctly."""
        from src.transformers.phishing_pipeline import PhishingFeaturePipeline

        pipeline = PhishingFeaturePipeline()
        pipeline.fit(sample_emails)
        feature_names = pipeline.get_feature_names()

        # Check we have feature names
        assert len(feature_names) > 50
        assert "url_count" in feature_names
        assert "bank_impersonation_score" in feature_names
        assert "spf_pass" in feature_names

    def test_extraction_stats(self, sample_emails):
        """Test extraction statistics tracking."""
        from src.transformers.phishing_pipeline import PhishingFeaturePipeline

        pipeline = PhishingFeaturePipeline()
        pipeline.fit_transform(sample_emails)

        stats = pipeline.get_extraction_stats()

        # Check we have stats for each extractor
        assert len(stats) > 0
        for extractor_name, extractor_stats in stats.items():
            assert "total_emails" in extractor_stats
            assert "avg_time_ms" in extractor_stats
            assert "features_count" in extractor_stats
            # fit_transform calls both fit() and transform(), so count is 2x
            assert extractor_stats["total_emails"] == 24

    def test_custom_pipeline(self, sample_emails):
        """Test custom pipeline creation."""
        from src.transformers.phishing_pipeline import create_custom_pipeline

        pipeline = create_custom_pipeline(
            include_url=True,
            include_header=True,
            include_sender=True,
            include_content=True,
            include_structural=True,
            include_linguistic=False,
            include_financial=True,
            normalize=True
        )

        features = pipeline.fit_transform(sample_emails)
        assert features.shape[0] == 12


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
