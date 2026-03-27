"""Email header feature extraction for authentication analysis.

Extracts features from email headers related to authentication and routing:
- SPF (Sender Policy Framework) validation status
- DKIM (DomainKeys Identified Mail) signature presence/validity
- DMARC (Domain-based Message Authentication) status
- Hop count (number of mail servers)
- Reply-To mismatch
- Priority flags
- Message ID analysis
"""

import re
import time
from typing import Dict

import pandas as pd

from .base import BaseExtractor


class HeaderFeatureExtractor(BaseExtractor):
    """Extract email header features for phishing detection.

    Features extracted:
        - spf_pass: SPF validation passed (0/1)
        - spf_fail: SPF validation failed (0/1)
        - dkim_present: DKIM signature present (0/1)
        - dkim_valid: DKIM signature valid (0/1)
        - dmarc_pass: DMARC validation passed (0/1)
        - dmarc_fail: DMARC validation failed (0/1)
        - hop_count: Number of mail servers in path (normalized)
        - reply_to_mismatch: Reply-To differs from From (0/1)
        - has_priority_flag: Email marked high priority (0/1)
        - has_authentication_results: Authentication headers present (0/1)
    """

    # Max values for normalization
    MAX_HOPS = 10  # More than 10 hops is unusual

    def __init__(self) -> None:
        """Initialize header feature extractor."""
        super().__init__()
        self.feature_names = [
            "spf_pass",
            "spf_fail",
            "dkim_present",
            "dkim_valid",
            "dmarc_pass",
            "dmarc_fail",
            "hop_count",
            "reply_to_mismatch",
            "has_priority_flag",
            "has_authentication_results",
        ]

        # Regex patterns for header parsing
        self.priority_pattern = re.compile(
            r"X-Priority\s*:\s*\s*[1-2]|Priority\s*:\s*urgent|Importance\s*:\s*high",
            re.IGNORECASE,
        )
        self.spf_pass_pattern = re.compile(r"spf=(pass|none)", re.IGNORECASE)
        self.spf_fail_pattern = re.compile(r"spf=(fail|softfail|permerror|temperror)", re.IGNORECASE)
        self.dmarc_pass_pattern = re.compile(r"dmarc=(pass|none)", re.IGNORECASE)
        self.dmarc_fail_pattern = re.compile(
            r"dmarc=(fail|softfail|permerror|temperror)", re.IGNORECASE
        )

    def fit(self, emails: pd.DataFrame) -> "HeaderFeatureExtractor":
        """Fit the header extractor.

        Stateless extractor - validates input structure only.

        Args:
            emails: DataFrame with 'headers' column.

        Returns:
            self: Fitted extractor.
        """
        self._validate_input(emails)
        self._is_fitted = True
        return self

    def transform(self, emails: pd.DataFrame) -> pd.DataFrame:
        """Transform emails into header-based features.

        Args:
            emails: DataFrame with 'headers' column.

        Returns:
            DataFrame with header features (n_emails, n_features).
        """
        if not self._is_fitted:
            raise RuntimeError("HeaderFeatureExtractor must be fitted before transform")

        results = []

        for idx, row in emails.iterrows():
            start_time = time.time()

            # Get headers - handle both dict and string formats
            headers = row.get("headers", {})
            from_addr = str(row.get("from_addr", ""))

            features = self._extract_header_features(headers, from_addr)
            results.append(features)

            self.extraction_times.append((time.time() - start_time) * 1000)  # ms

        return pd.DataFrame(results, columns=self.feature_names)

    def _extract_header_features(
        self, headers: Dict[str, str] | str, from_addr: str
    ) -> dict[str, float]:
        """Extract all header features from email headers.

        Args:
            headers: Email headers (dict or string).
            from_addr: From address string.

        Returns:
            Dictionary of feature names to values in [0, 1].
        """
        # Convert to dict if string
        if isinstance(headers, str):
            headers = self._parse_headers_string(headers)

        if not headers or not isinstance(headers, dict):
            return {name: 0.0 for name in self.feature_names}

        features = {
            "spf_pass": self._check_spf_pass(headers),
            "spf_fail": self._check_spf_fail(headers),
            "dkim_present": self._check_dkim_present(headers),
            "dkim_valid": self._check_dkim_valid(headers),
            "dmarc_pass": self._check_dmarc_pass(headers),
            "dmarc_fail": self._check_dmarc_fail(headers),
            "hop_count": self._count_hops(headers),
            "reply_to_mismatch": self._check_reply_to_mismatch(headers, from_addr),
            "has_priority_flag": self._check_priority_flag(headers),
            "has_authentication_results": self._has_auth_results(headers),
        }

        return features

    def _parse_headers_string(self, headers_str: str) -> Dict[str, str]:
        """Parse headers from string format.

        Args:
            headers_str: Headers as string.

        Returns:
            Dictionary of header names to values.
        """
        headers = {}
        if not headers_str:
            return headers

        try:
            for line in headers_str.split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    headers[key.strip()] = value.strip()
        except Exception:
            pass

        return headers

    def _check_spf_pass(self, headers: Dict[str, str]) -> float:
        """Check if SPF validation passed.

        Args:
            headers: Email headers.

        Returns:
            1.0 if SPF pass, 0.0 otherwise.
        """
        # Check Authentication-Results header
        auth_results = headers.get("Authentication-Results", "")
        if auth_results and self.spf_pass_pattern.search(auth_results):
            return 1.0

        # Check Received-SPF header
        received_spf = headers.get("Received-SPF", "")
        if received_spf and "pass" in received_spf.lower():
            return 1.0

        return 0.0

    def _check_spf_fail(self, headers: Dict[str, str]) -> float:
        """Check if SPF validation failed.

        Args:
            headers: Email headers.

        Returns:
            1.0 if SPF fail, 0.0 otherwise.
        """
        # Check Authentication-Results header
        auth_results = headers.get("Authentication-Results", "")
        if auth_results and self.spf_fail_pattern.search(auth_results):
            return 1.0

        # Check Received-SPF header
        received_spf = headers.get("Received-SPF", "")
        if received_spf and any(
            fail in received_spf.lower() for fail in ["fail", "softfail", "error"]
        ):
            return 1.0

        return 0.0

    def _check_dkim_present(self, headers: Dict[str, str]) -> float:
        """Check if DKIM signature is present.

        Args:
            headers: Email headers.

        Returns:
            1.0 if DKIM signature present, 0.0 otherwise.
        """
        if "DKIM-Signature" in headers:
            return 1.0
        return 0.0

    def _check_dkim_valid(self, headers: Dict[str, str]) -> float:
        """Check if DKIM signature is valid.

        Args:
            headers: Email headers.

        Returns:
            1.0 if DKIM valid, 0.0 otherwise.
        """
        auth_results = headers.get("Authentication-Results", "")
        if auth_results:
            # Check for dkim=pass
            if re.search(r"dkim\s*=\s*pass", auth_results, re.IGNORECASE):
                return 1.0

        return 0.0

    def _check_dmarc_pass(self, headers: Dict[str, str]) -> float:
        """Check if DMARC validation passed.

        Args:
            headers: Email headers.

        Returns:
            1.0 if DMARC pass, 0.0 otherwise.
        """
        auth_results = headers.get("Authentication-Results", "")
        if auth_results and self.dmarc_pass_pattern.search(auth_results):
            return 1.0

        return 0.0

    def _check_dmarc_fail(self, headers: Dict[str, str]) -> float:
        """Check if DMARC validation failed.

        Args:
            headers: Email headers.

        Returns:
            1.0 if DMARC fail, 0.0 otherwise.
        """
        auth_results = headers.get("Authentication-Results", "")
        if auth_results and self.dmarc_fail_pattern.search(auth_results):
            return 1.0

        return 0.0

    def _count_hops(self, headers: Dict[str, str]) -> float:
        """Count number of mail servers (Received headers).

        More hops can indicate email forwarding or routing anomalies.

        Args:
            headers: Email headers.

        Returns:
            Normalized hop count in [0, 1].
        """
        # Count Received headers (each represents a hop)
        received_count = 0
        for key in headers.keys():
            if key.strip().lower() == "received":
                received_count += 1

        # Normalize to [0, 1]
        return min(1.0, received_count / self.MAX_HOPS)

    def _check_reply_to_mismatch(self, headers: Dict[str, str], from_addr: str) -> float:
        """Check if Reply-To address differs from From address.

        Mismatch is a common phishing tactic.

        Args:
            headers: Email headers.
            from_addr: From address string.

        Returns:
            1.0 if mismatch detected, 0.0 otherwise.
        """
        reply_to = headers.get("Reply-To", "")
        if not reply_to or not from_addr:
            return 0.0

        # Extract domains for comparison
        try:
            reply_to_domain = self._extract_domain(reply_to)
            from_domain = self._extract_domain(from_addr)

            if not reply_to_domain or not from_domain:
                return 0.0

            # Compare domains
            if reply_to_domain.lower() != from_domain.lower():
                return 1.0
        except Exception:
            pass

        return 0.0

    def _extract_domain(self, email_addr: str) -> str:
        """Extract domain from email address.

        Args:
            email_addr: Email address string.

        Returns:
            Domain portion.
        """
        if "@" in email_addr:
            return email_addr.split("@")[-1].strip().lower()
        return ""

    def _check_priority_flag(self, headers: Dict[str, str]) -> float:
        """Check if email is marked as high priority/urgent.

        Phishing emails often use urgency to compel action.

        Args:
            headers: Email headers.

        Returns:
            1.0 if high priority, 0.0 otherwise.
        """
        # Check X-Priority header
        x_priority = headers.get("X-Priority", "")
        if x_priority and x_priority.strip() in ["1", "1 (Highest)", "2", "2 (High)"]:
            return 1.0

        # Check Priority header
        priority = headers.get("Priority", "")
        if priority and "urgent" in priority.lower():
            return 1.0

        # Check Importance header
        importance = headers.get("Importance", "")
        if importance and "high" in importance.lower():
            return 1.0

        return 0.0

    def _has_auth_results(self, headers: Dict[str, str]) -> float:
        """Check if Authentication-Results header is present.

        Presence indicates authentication was performed.

        Args:
            headers: Email headers.

        Returns:
            1.0 if present, 0.0 otherwise.
        """
        if "Authentication-Results" in headers:
            return 1.0
        return 0.0
