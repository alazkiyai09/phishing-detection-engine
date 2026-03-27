"""Sender-based feature extraction for phishing detection.

Extracts features related to email sender characteristics:
- Freemail vs corporate domain detection
- Display name spoofing/tricks
- Domain reputation (simulated)
- Email address patterns
- Sender name mismatch
"""

import re
import time

import pandas as pd

from .base import BaseExtractor


class SenderFeatureExtractor(BaseExtractor):
    """Extract sender-based features for phishing detection.

    Features extracted:
        - is_freemail: Sender uses freemail service (0/1)
        - display_name_mismatch: Display name doesn't match email (0/1)
        - display_name_has_bank: Display name contains bank name (0/1)
        - has_numbers_in_domain: Domain contains numbers (0/1)
        - email_address_length: Email address length (normalized)
        - domain_length: Domain length (normalized)
        - sender_name_length: Sender display name length (normalized)
        - has_reply_to_path: Has Reply-To header (0/1)
        - suspicious_pattern: Email matches suspicious patterns (0/1)

    Note: domain_age_days was removed as it requires actual WHOIS lookup.
    """

    # Common freemail providers
    FREEMAIL_DOMAINS = {
        "gmail.com",
        "yahoo.com",
        "hotmail.com",
        "outlook.com",
        "aol.com",
        "icloud.com",
        "protonmail.com",
        "mail.com",
        "yandex.com",
        "gmail.co.in",  # Country-specific
        "yahoo.co.in",
        "hotmail.co.in",
    }

    # Max values for normalization
    MAX_EMAIL_LENGTH = 100
    MAX_DOMAIN_LENGTH = 50
    MAX_NAME_LENGTH = 100

    def __init__(self) -> None:
        """Initialize sender feature extractor."""
        super().__init__()
        self.feature_names = [
            "is_freemail",
            "display_name_mismatch",
            "display_name_has_bank",
            "has_numbers_in_domain",
            "email_address_length",
            "domain_length",
            "sender_name_length",
            "has_reply_to_path",
            "suspicious_pattern",
        ]

        # Regex patterns
        self.digits_pattern = re.compile(r"\d")
        self.suspicious_pattern = re.compile(
            r".*(support|security|admin|noreply|no-reply|service|help).*@(gmail|yahoo|hotmail|outlook)\.com",
            re.IGNORECASE,
        )

    def fit(self, emails: pd.DataFrame) -> "SenderFeatureExtractor":
        """Fit the sender extractor.

        Stateless extractor - validates input structure only.

        Args:
            emails: DataFrame with 'from_addr' column.

        Returns:
            self: Fitted extractor.
        """
        self._validate_input(emails)
        self._is_fitted = True
        return self

    def transform(self, emails: pd.DataFrame) -> pd.DataFrame:
        """Transform emails into sender-based features.

        Args:
            emails: DataFrame with 'from_addr' column.

        Returns:
            DataFrame with sender features (n_emails, n_features).
        """
        if not self._is_fitted:
            raise RuntimeError("SenderFeatureExtractor must be fitted before transform")

        results = []

        for idx, row in emails.iterrows():
            start_time = time.time()

            from_addr = str(row.get("from_addr", ""))
            headers = row.get("headers", {})

            features = self._extract_sender_features(from_addr, headers)
            results.append(features)

            self.extraction_times.append((time.time() - start_time) * 1000)  # ms

        return pd.DataFrame(results, columns=self.feature_names)

    def _extract_sender_features(
        self, from_addr: str, headers
    ) -> dict[str, float]:
        """Extract all sender features.

        Args:
            from_addr: From email address.
            headers: Email headers dict.

        Returns:
            Dictionary of feature names to values in [0, 1].
        """
        if not from_addr or from_addr == "":
            return {name: 0.0 for name in self.feature_names}

        # Parse email address
        display_name = self._extract_display_name(from_addr)
        email_addr = self._extract_email_addr(from_addr)
        domain = self._extract_domain(email_addr)

        features = {
            "is_freemail": self._is_freemail(domain),
            "display_name_mismatch": self._check_display_name_mismatch(
                display_name, email_addr, domain
            ),
            "display_name_has_bank": self._display_name_has_bank(display_name),
            "has_numbers_in_domain": self._has_numbers_in_domain(domain),
            "email_address_length": self._normalize_length(
                email_addr, self.MAX_EMAIL_LENGTH
            ),
            "domain_length": self._normalize_length(domain, self.MAX_DOMAIN_LENGTH),
            "sender_name_length": self._normalize_length(
                display_name, self.MAX_NAME_LENGTH
            ),
            "has_reply_to_path": self._has_reply_to(headers),
            "suspicious_pattern": self._has_suspicious_pattern(email_addr),
        }

        return features

    def _extract_display_name(self, from_addr: str) -> str:
        """Extract display name from From address.

        Args:
            from_addr: Full From address (e.g., "John Doe <jdoe@example.com>").

        Returns:
            Display name portion.
        """
        if "<" in from_addr and ">" in from_addr:
            display = from_addr.split("<")[0].strip().strip('"')
            return display
        return ""

    def _extract_email_addr(self, from_addr: str) -> str:
        """Extract email address from From field.

        Args:
            from_addr: Full From address.

        Returns:
            Email address portion.
        """
        if "<" in from_addr and ">" in from_addr:
            email = from_addr.split("<")[1].split(">")[0].strip()
            return email.lower()
        # No angle brackets, might be just email
        if "@" in from_addr:
            return from_addr.strip().lower()
        return ""

    def _extract_domain(self, email_addr: str) -> str:
        """Extract domain from email address.

        Args:
            email_addr: Email address.

        Returns:
            Domain portion.
        """
        if "@" in email_addr:
            return email_addr.split("@")[-1].strip().lower()
        return ""

    def _is_freemail(self, domain: str) -> float:
        """Check if domain is a freemail provider.

        Freemail accounts are commonly used in phishing.

        Args:
            domain: Domain to check.

        Returns:
            1.0 if freemail, 0.0 otherwise.
        """
        if not domain:
            return 0.0

        if domain.lower() in self.FREEMAIL_DOMAINS:
            return 1.0
        return 0.0

    def _check_display_name_mismatch(
        self, display_name: str, email_addr: str, domain: str
    ) -> float:
        """Check if display name mismatches email address.

        Phishing often uses display names that don't match the email.

        Args:
            display_name: Display name.
            email_addr: Email address.
            domain: Email domain.

        Returns:
            1.0 if mismatch detected, 0.0 otherwise.
        """
        if not display_name or not email_addr:
            return 0.0

        # Extract local part of email (before @)
        local_part = email_addr.split("@")[0] if "@" in email_addr else ""

        # Check if display name is contained in local part or vice versa
        display_lower = display_name.lower()
        local_lower = local_part.lower()

        # Simple check: does display name match local part?
        if display_lower in local_lower or local_lower in display_lower:
            return 0.0

        # Check if display name contains domain
        if domain and domain.lower() in display_lower:
            return 0.0

        # Mismatch detected
        return 1.0

    def _display_name_has_bank(self, display_name: str) -> float:
        """Check if display name contains bank-related terms.

        Args:
            display_name: Display name to check.

        Returns:
            1.0 if bank terms found, 0.0 otherwise.
        """
        if not display_name:
            return 0.0

        bank_terms = [
            "bank",
            "chase",
            "wells fargo",
            "citi",
            "citibank",
            "american express",
            "amex",
            "capital one",
            "paypal",
            "venmo",
            "zelle",
            "wire",
            "transfer",
        ]

        display_lower = display_name.lower()
        for term in bank_terms:
            if term in display_lower:
                return 1.0

        return 0.0

    def _has_numbers_in_domain(self, domain: str) -> float:
        """Check if domain contains numbers.

        Numbers in domains can indicate typosquatting or obfuscation.

        Args:
            domain: Domain to check.

        Returns:
            1.0 if numbers found, 0.0 otherwise.
        """
        if not domain:
            return 0.0

        if self.digits_pattern.search(domain):
            return 1.0
        return 0.0

    def _normalize_length(self, text: str, max_length: int) -> float:
        """Normalize text length to [0, 1].

        Args:
            text: Text to measure.
            max_length: Maximum expected length.

        Returns:
            Normalized length in [0, 1].
        """
        if not text:
            return 0.0

        return min(1.0, len(text) / max_length)

    def _has_reply_to(self, headers) -> float:
        """Check if Reply-To header is present.

        Args:
            headers: Headers dict.

        Returns:
            1.0 if present, 0.0 otherwise.
        """
        if isinstance(headers, dict) and "Reply-To" in headers:
            return 1.0
        return 0.0

    def _has_suspicious_pattern(self, email_addr: str) -> float:
        """Check if email matches suspicious patterns.

        Common phishing patterns:
        - support@freemail.com
        - security@freemail.com
        - admin@freemail.com

        Args:
            email_addr: Email address to check.

        Returns:
            1.0 if suspicious, 0.0 otherwise.
        """
        if not email_addr:
            return 0.0

        if self.suspicious_pattern.match(email_addr):
            return 1.0
        return 0.0
