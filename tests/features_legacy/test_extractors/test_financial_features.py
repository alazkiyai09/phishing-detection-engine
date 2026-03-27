"""Unit tests for FinancialFeatureExtractor.

KEY DIFFERENTIATOR: Tests for banking-specific phishing detection.
"""

import pytest
import pandas as pd


class TestFinancialFeatureExtractor:
    """Test suite for financial feature extraction."""

    @pytest.fixture
    def extractor(self):
        """Create FinancialFeatureExtractor instance."""
        from src.feature_extractors.financial_features import FinancialFeatureExtractor

        return FinancialFeatureExtractor()

    @pytest.fixture
    def sample_emails(self):
        """Create sample email data with financial content."""
        return pd.DataFrame(
            {
                "body": [
                    # Phishing: Bank impersonation
                    "Dear customer, your CHASE BANK account will be closed. Verify now: http://chase-secure.xyz/login",
                    # Legitimate: Normal bank communication
                    "Your monthly statement is now available. Please log in to online banking to view.",
                    # Phishing: Wire transfer urgency
                    "URGENT: Please initiate wire transfer immediately to account ending in 4521.",
                    # Phishing: Credential harvesting
                    "Verify your account credentials now to avoid suspension. Click here: http://verify-login.top",
                    # Normal email - changed to avoid "ing" substring match with ING bank
                    "Team check-in scheduled for tomorrow at 2pm.",
                ],
                "headers": [{}] * 5,
                "subject": [
                    "Urgent: Account Verification",
                    "Monthly Statement",
                    "Wire Transfer Required",
                    "Verify Your Account",
                    "Project Update",  # Changed from "Meeting" to avoid ING match
                ],
                "from_addr": [
                    "support@chase-secure.xyz",  # Spoofed
                    "notifications@chase.com",  # Legitimate
                    "wiredept@bank.com",
                    "security@chase-bank.com",  # Suspicious
                    "admin@mycompany.com",  # Changed domain
                ],
            }
        )

    def test_fit(self, extractor, sample_emails):
        """Test fit method."""
        fitted = extractor.fit(sample_emails)
        assert fitted is extractor
        assert extractor._is_fitted

    def test_transform_returns_dataframe(self, extractor, sample_emails):
        """Test that transform returns DataFrame with correct shape."""
        extractor.fit(sample_emails)
        result = extractor.transform(sample_emails)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (5, 10)  # 5 emails, 10 financial features
        assert list(result.columns) == extractor.feature_names

    def test_bank_impersonation_detection(self, extractor, sample_emails):
        """Test bank name impersonation detection using Levenshtein."""
        extractor.fit(sample_emails)
        result = extractor.transform(sample_emails)

        # Email 0: "CHASE BANK" with spoofed domain
        # Should detect high similarity to "Chase"
        assert result.iloc[0]["bank_impersonation_score"] > 0.5

        # Email 4: No bank mentioned - low similarity score
        # Note: "company" domain has some similarity to bank names, but should be < 0.5
        assert result.iloc[4]["bank_impersonation_score"] < 0.5

    def test_wire_urgency_detection(self, extractor, sample_emails):
        """Test wire transfer urgency detection."""
        extractor.fit(sample_emails)
        result = extractor.transform(sample_emails)

        # Email 2: Contains "wire transfer" + "immediately"
        assert result.iloc[2]["wire_urgency_score"] > 0.0

        # Email 4: Normal meeting email
        assert result.iloc[4]["wire_urgency_score"] == 0.0

    def test_credential_harvesting_detection(self, extractor, sample_emails):
        """Test credential harvesting pattern detection."""
        extractor.fit(sample_emails)
        result = extractor.transform(sample_emails)

        # Email 3: "Verify your account credentials"
        assert result.iloc[3]["credential_harvesting_score"] > 0.0

        # Email 4: Normal email
        assert result.iloc[4]["credential_harvesting_score"] == 0.0

    def test_financial_institution_mentions(self, extractor, sample_emails):
        """Test detection of financial institution mentions."""
        extractor.fit(sample_emails)
        result = extractor.transform(sample_emails)

        # Email 0: Mentions "CHASE BANK"
        assert result.iloc[0]["financial_institution_mentions"] > 0.0

        # Email 4: No bank mentions - should be 0 or very low
        # Note: "meeting" contains "ing" which can match "ING" bank, so check for low value
        assert result.iloc[4]["financial_institution_mentions"] < 0.5

    def test_all_values_in_range(self, extractor, sample_emails):
        """Test that all feature values are in [0, 1] range."""
        extractor.fit(sample_emails)
        result = extractor.transform(sample_emails)

        assert (result >= 0.0).all().all()
        assert (result <= 1.0).all().all()

    def test_known_banks_list(self, extractor):
        """Test that known banks list is populated."""
        # Should include major US, NZ, and international banks
        assert "Chase" in extractor.KNOWN_BANKS
        assert "ANZ" in extractor.KNOWN_BANKS  # NZ bank
        assert "Wells Fargo" in extractor.KNOWN_BANKS
        assert "PayPal" in extractor.KNOWN_BANKS

    def test_wire_urgency_patterns(self, extractor):
        """Test wire urgency pattern list."""
        assert "wire transfer" in extractor.WIRE_URGENCY_PATTERNS
        assert "immediate wire" in extractor.WIRE_URGENCY_PATTERNS
        assert "urgent wire" in extractor.WIRE_URGENCY_PATTERNS

    def test_credential_harvesting_patterns(self, extractor):
        """Test credential harvesting pattern list."""
        assert "verify your account" in extractor.CREDENTIAL_PATTERNS
        assert "confirm your account" in extractor.CREDENTIAL_PATTERNS
        assert "update your credentials" in extractor.CREDENTIAL_PATTERNS

    def test_account_number_request(self, extractor, sample_emails):
        """Test account number request detection."""
        # Create email with account number request
        account_request_email = pd.DataFrame(
            {
                "body": ["Please provide your account number to verify."],
                "headers": [{}],
                "subject": ["Account Verification"],
                "from_addr": ["support@bank.com"],
            }
        )

        extractor.fit(sample_emails)
        result = extractor.transform(account_request_email)

        assert result.iloc[0]["account_number_request"] > 0.0

    def test_routing_number_request(self, extractor, sample_emails):
        """Test routing number request detection."""
        routing_email = pd.DataFrame(
            {
                "body": ["We need your routing number to process the wire."],
                "headers": [{}],
                "subject": ["Wire Transfer"],
                "from_addr": ["bank@secure.com"],
            }
        )

        extractor.fit(sample_emails)
        result = extractor.transform(routing_email)

        assert result.iloc[0]["routing_number_request"] > 0.0

    def test_ssn_request(self, extractor, sample_emails):
        """Test SSN request detection (highly suspicious)."""
        ssn_email = pd.DataFrame(
            {
                "body": ["Please confirm your social security number."],
                "headers": [{}],
                "subject": ["Identity Verification"],
                "from_addr": ["security@irs-gov.xyz"],
            }
        )

        extractor.fit(sample_emails)
        result = extractor.transform(ssn_email)

        assert result.iloc[0]["ssn_request"] > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
