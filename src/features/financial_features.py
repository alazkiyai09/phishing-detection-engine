"""Financial-specific feature extraction for phishing detection.

KEY DIFFERENTIATOR: Specialized features for banking/financial phishing.

Extracts features specific to financial fraud:
- Bank name impersonation (Levenshtein distance)
- Wire transfer urgency detection
- Account credential harvesting
- Invoice/payment terminology
- Routing/account number requests
- Financial institution terminology
"""

import re
import time
from difflib import SequenceMatcher
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .base import BaseExtractor


class FinancialFeatureExtractor(BaseExtractor):
    """Extract financial-specific features for phishing detection.

    Features extracted:
        - bank_impersonation_score: Bank name similarity (normalized)
        - wire_urgency_score: Wire transfer urgency (normalized)
        - credential_harvesting_score: Credential request (normalized)
        - invoice_terminology_density: Invoice terms (normalized)
        - account_number_request: Account # requests (normalized)
        - routing_number_request: Routing # requests (normalized)
        - ssn_request: SSN requests (normalized)
        - payment_urgency_score: Payment urgency (normalized)
        - financial_institution_mentions: Bank mentions (normalized)
        - wire_transfer_keywords: Wire transfer terms (normalized)
    """

    # Major banks (US, NZ, International)
    KNOWN_BANKS = [
        # US Major Banks
        "Chase", "JPMorgan", "Bank of America", "Wells Fargo", "Citibank", "Citi",
        "Capital One", "TD Bank", "PNC Bank", "US Bank", "BB&T", "SunTrust",
        # NZ Banks
        "ANZ", "ANZ Bank", "ASB", "BNZ", "Westpac", "Kiwibank",
        "Co-operative Bank", "Heartland Bank", "SBS Bank",
        # International
        "HSBC", "Barclays", "Deutsche Bank", "UBS", "Credit Suisse",
        "BNP Paribas", "Societe Generale", "ING", "Rabobank",
        # Payment Services
        "PayPal", "Venmo", "Zelle", "Square", "Stripe", "TransferWise", "Wise",
    ]

    # Wire transfer urgency patterns
    WIRE_URGENCY_PATTERNS = [
        "wire transfer",
        "international wire",
        "immediate wire",
        "urgent wire",
        "wire payment",
        "outgoing wire",
        "incoming wire",
        "wire confirmation",
        "wire instructions",
    ]

    # Credential harvesting patterns
    CREDENTIAL_PATTERNS = [
        "verify your account",
        "confirm your account",
        "update your credentials",
        "verify your identity",
        "confirm your identity",
        "login to verify",
        "sign in to confirm",
        "update your information",
        "verify your information",
        "confirm your information",
        "click to verify",
        "verify now",
        "confirm now",
    ]

    # Invoice/payment terminology
    INVOICE_PATTERNS = [
        "invoice",
        "payment due",
        "overdue invoice",
        "past due",
        "outstanding balance",
        "payment required",
        "bill payment",
        "remittance",
        "payment notification",
        "invoice attached",
        "statement attached",
    ]

    # Account number request patterns
    ACCOUNT_NUMBER_PATTERNS = [
        "account number",
        "account #",
        "acct number",
        "account no",
        "your account number",
        "provide your account",
        "confirm your account",
    ]

    # Routing number patterns
    ROUTING_NUMBER_PATTERNS = [
        "routing number",
        "routing #",
        "aba number",
        "aba routing",
        "bank routing",
        "swift code",
        "iban",
        "sort code",
    ]

    # SSN patterns
    SSN_PATTERNS = [
        "social security",
        "ssn",
        "social security number",
        "tax id",
        "tax identification",
        "federal id",
    ]

    # Payment urgency patterns
    PAYMENT_URGENCY_PATTERNS = [
        "payment immediately",
        "payment now",
        "payment due immediately",
        "pay now",
        "make payment",
        "send payment",
        "process payment",
        "payment overdue",
        "avoid late fee",
        "service disconnection",
        "account cancellation",
    ]

    # Max values for normalization
    MAX_KEYWORD_COUNT = 10
    MAX_LEVENSHTEIN_DISTANCE = 5  # For bank name similarity

    def __init__(self) -> None:
        """Initialize financial feature extractor."""
        super().__init__()
        self.feature_names = [
            "bank_impersonation_score",
            "wire_urgency_score",
            "credential_harvesting_score",
            "invoice_terminology_density",
            "account_number_request",
            "routing_number_request",
            "ssn_request",
            "payment_urgency_score",
            "financial_institution_mentions",
            "wire_transfer_keywords",
        ]

        # Compile regex patterns
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficiency."""
        # Pattern to find bank-like names (capitalized words)
        self.bank_name_pattern = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b")

        # Pattern to find account numbers (xxx-xx-xxxx format)
        self.account_number_pattern = re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b")

    def fit(self, emails: pd.DataFrame) -> "FinancialFeatureExtractor":
        """Fit the financial extractor.

        Stateless extractor - validates input structure only.

        Args:
            emails: DataFrame with 'body' and 'subject' columns.

        Returns:
            self: Fitted extractor.
        """
        self._validate_input(emails)
        self._is_fitted = True
        return self

    def transform(self, emails: pd.DataFrame) -> pd.DataFrame:
        """Transform emails into financial-specific features.

        Args:
            emails: DataFrame with 'body' and 'subject' columns.

        Returns:
            DataFrame with financial features (n_emails, n_features).
        """
        if not self._is_fitted:
            raise RuntimeError("FinancialFeatureExtractor must be fitted before transform")

        results = []

        for idx, row in emails.iterrows():
            start_time = time.time()

            body = str(row.get("body", ""))
            subject = str(row.get("subject", ""))
            from_addr = str(row.get("from_addr", ""))

            # Combine all text
            text = f"{subject} {body}"

            features = self._extract_financial_features(text, from_addr)
            results.append(features)

            self.extraction_times.append((time.time() - start_time) * 1000)  # ms

        return pd.DataFrame(results, columns=self.feature_names)

    def _extract_financial_features(self, text: str, from_addr: str) -> dict[str, float]:
        """Extract all financial features.

        Args:
            text: Email text (subject + body).
            from_addr: From address.

        Returns:
            Dictionary of feature names to values in [0, 1].
        """
        if not text:
            return {name: 0.0 for name in self.feature_names}

        text_lower = text.lower()

        features = {
            "bank_impersonation_score": self._detect_bank_impersonation(text, from_addr),
            "wire_urgency_score": self._detect_wire_urgency(text_lower),
            "credential_harvesting_score": self._detect_credential_harvesting(
                text_lower
            ),
            "invoice_terminology_density": self._detect_invoice_terminology(text_lower),
            "account_number_request": self._detect_account_number_request(text_lower),
            "routing_number_request": self._detect_routing_number_request(text_lower),
            "ssn_request": self._detect_ssn_request(text_lower),
            "payment_urgency_score": self._detect_payment_urgency(text_lower),
            "financial_institution_mentions": self._count_bank_mentions(text_lower),
            "wire_transfer_keywords": self._count_wire_keywords(text_lower),
        }

        return features

    def _detect_bank_impersonation(self, text: str, from_addr: str) -> float:
        """Detect bank name impersonation using Levenshtein distance.

        Finds similar-to-known-bank names that might be typosquatting.

        Args:
            text: Email text.
            from_addr: From address (also check domain).

        Returns:
            Normalized similarity score in [0, 1].
        """
        max_similarity = 0.0

        # Extract potential bank names (capitalized multi-word phrases)
        potential_names = self.bank_name_pattern.findall(text)

        # Also check From address domain
        if "@" in from_addr:
            domain = from_addr.split("@")[-1].split(".")[0]
            potential_names.append(domain)

        for name in potential_names:
            name_lower = name.lower().strip()

            for known_bank in self.KNOWN_BANKS:
                known_bank_lower = known_bank.lower()

                # Check for substring match first (faster)
                if name_lower in known_bank_lower or known_bank_lower in name_lower:
                    similarity = 1.0
                else:
                    # Use SequenceMatcher for Levenshtein-like similarity
                    similarity = SequenceMatcher(None, name_lower, known_bank_lower).ratio()

                max_similarity = max(max_similarity, similarity)

        return max_similarity

    def _detect_wire_urgency(self, text: str) -> float:
        """Detect wire transfer urgency patterns.

        Args:
            text: Lowercase text.

        Returns:
            Normalized urgency score in [0, 1].
        """
        count = sum(1 for pattern in self.WIRE_URGENCY_PATTERNS if pattern in text)
        return min(1.0, count / 3)  # Max 3 occurrences for normalization

    def _detect_credential_harvesting(self, text: str) -> float:
        """Detect credential harvesting patterns.

        Args:
            text: Lowercase text.

        Returns:
            Normalized harvesting score in [0, 1].
        """
        count = sum(1 for pattern in self.CREDENTIAL_PATTERNS if pattern in text)
        return min(1.0, count / 5)  # Max 5 occurrences

    def _detect_invoice_terminology(self, text: str) -> float:
        """Detect invoice/payment terminology.

        Args:
            text: Lowercase text.

        Returns:
            Normalized terminology density in [0, 1].
        """
        count = sum(1 for pattern in self.INVOICE_PATTERNS if pattern in text)
        return min(1.0, count / 5)  # Max 5 occurrences

    def _detect_account_number_request(self, text: str) -> float:
        """Detect account number requests.

        Args:
            text: Lowercase text.

        Returns:
            Normalized request score in [0, 1].
        """
        count = sum(1 for pattern in self.ACCOUNT_NUMBER_PATTERNS if pattern in text)

        # Also look for actual account numbers (suspicious in email body)
        if self.account_number_pattern.search(text):
            count += 2  # Bonus weight

        return min(1.0, count / 3)

    def _detect_routing_number_request(self, text: str) -> float:
        """Detect routing number requests.

        Args:
            text: Lowercase text.

        Returns:
            Normalized request score in [0, 1].
        """
        count = sum(1 for pattern in self.ROUTING_NUMBER_PATTERNS if pattern in text)
        return min(1.0, count / 3)

    def _detect_ssn_request(self, text: str) -> float:
        """Detect SSN requests (highly suspicious).

        Args:
            text: Lowercase text.

        Returns:
            Normalized request score in [0, 1].
        """
        count = sum(1 for pattern in self.SSN_PATTERNS if pattern in text)
        return min(1.0, count / 2)  # Max 2 occurrences (very sensitive)

    def _detect_payment_urgency(self, text: str) -> float:
        """Detect payment urgency patterns.

        Args:
            text: Lowercase text.

        Returns:
            Normalized urgency score in [0, 1].
        """
        count = sum(1 for pattern in self.PAYMENT_URGENCY_PATTERNS if pattern in text)
        return min(1.0, count / 5)

    def _count_bank_mentions(self, text: str) -> float:
        """Count mentions of financial institutions.

        Args:
            text: Lowercase text.

        Returns:
            Normalized mention count in [0, 1].
        """
        count = 0
        for bank in self.KNOWN_BANKS:
            if bank.lower() in text:
                count += 1

        return min(1.0, count / 5)  # Max 5 mentions

    def _count_wire_keywords(self, text: str) -> float:
        """Count wire transfer related keywords.

        Args:
            text: Lowercase text.

        Returns:
            Normalized keyword count in [0, 1].
        """
        wire_keywords = [
            "wire",
            "transfer",
            "swift",
            "iban",
            "ach",
            "electronic funds",
            "eft",
            "bank transfer",
            "fund transfer",
        ]

        count = sum(1 for keyword in wire_keywords if keyword in text)
        return min(1.0, count / 5)
