"""URL-based feature extraction for phishing detection.

Extracts features from URLs found in email bodies that are indicative of phishing:
- Domain age (newly registered domains are suspicious)
- IP address-based URLs (no domain name)
- URL length (phishing URLs are often longer)
- Special characters and obfuscation
- Suspicious TLDs
- Subdomain count
- HTTPS presence
"""

import re
import time
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from .base import BaseExtractor


class URLFeatureExtractor(BaseExtractor):
    """Extract URL-based features for phishing detection.

    Features extracted:
        - url_count: Number of URLs in email (normalized)
        - has_ip_url: Contains IP address in URL (0/1)
        - avg_url_length: Average URL length (normalized)
        - max_url_length: Maximum URL length (normalized)
        - has_suspicious_tld: Uses suspicious TLD (0/1)
        - has_https: Contains HTTPS URL (0/1)
        - avg_subdomain_count: Average subdomains per URL (normalized)
        - has_url_shortener: Uses URL shortener (0/1)
        - special_char_ratio: Ratio of special chars in URLs (normalized)
        - has_port_specified: URL contains port number (0/1)
    """

    # Suspicious TLDs commonly used in phishing
    SUSPICIOUS_TLDS = {
        ".xyz",
        ".top",
        ".zip",
        ".tk",
        ".ml",
        ".ga",
        ".cf",
        ".gq",
        ".pw",
        ".cc",
        ".club",
        ".online",
        ".site",
        ".icu",
    }

    # URL shorteners (can hide malicious URLs)
    URL_SHORTENERS = {
        "bit.ly",
        "tinyurl.com",
        "goo.gl",
        "t.co",
        "ow.ly",
        "is.gd",
        "buff.ly",
        "adf.ly",
        "bit.do",
        "mcaf.ee",
    }

    # Max values for normalization
    MAX_URL_COUNT = 20  # More than 20 URLs is definitely suspicious
    MAX_URL_LENGTH = 500  # Characters
    MAX_SUBDOMAINS = 5

    def __init__(self) -> None:
        """Initialize URL feature extractor."""
        super().__init__()
        self.feature_names = [
            "url_count",
            "has_ip_url",
            "avg_url_length",
            "max_url_length",
            "has_suspicious_tld",
            "has_https",
            "avg_subdomain_count",
            "has_url_shortener",
            "special_char_ratio",
            "has_port_specified",
        ]

        # Regex patterns
        self.url_pattern = re.compile(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        )
        self.ip_pattern = re.compile(
            r"http[s]?://(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?"
        )
        self.port_pattern = re.compile(r"http[s]?://[^:]+:\d+")

    def fit(self, emails: pd.DataFrame) -> "URLFeatureExtractor":
        """Fit the URL extractor.

        Stateless extractor - validates input structure only.

        Args:
            emails: DataFrame with 'body' column.

        Returns:
            self: Fitted extractor.
        """
        self._validate_input(emails)
        self._is_fitted = True
        return self

    def transform(self, emails: pd.DataFrame) -> pd.DataFrame:
        """Transform emails into URL-based features.

        Args:
            emails: DataFrame with 'body' column.

        Returns:
            DataFrame with URL features (n_emails, n_features).
        """
        if not self._is_fitted:
            raise RuntimeError("URLFeatureExtractor must be fitted before transform")

        results = []

        for _, row in emails.iterrows():
            start_time = time.time()
            body = str(row.get("body", ""))

            features = self._extract_url_features(body)
            results.append(features)

            self.extraction_times.append((time.time() - start_time) * 1000)  # ms

        return pd.DataFrame(results, columns=self.feature_names)

    def _extract_url_features(self, body: str) -> dict[str, float]:
        """Extract all URL features from email body.

        Args:
            body: Email body text.

        Returns:
            Dictionary of feature names to values in [0, 1].
        """
        urls = self._extract_urls(body)

        if not urls:
            return {name: 0.0 for name in self.feature_names}

        features = {
            "url_count": self._normalize_count(len(urls), self.MAX_URL_COUNT),
            "has_ip_url": self._has_ip_url(urls),
            "avg_url_length": self._avg_url_length(urls),
            "max_url_length": self._max_url_length(urls),
            "has_suspicious_tld": self._has_suspicious_tld(urls),
            "has_https": self._has_https(urls),
            "avg_subdomain_count": self._avg_subdomain_count(urls),
            "has_url_shortener": self._has_url_shortener(urls),
            "special_char_ratio": self._special_char_ratio(urls),
            "has_port_specified": self._has_port_specified(urls),
        }

        return features

    def _extract_urls(self, text: str) -> list[str]:
        """Extract all URLs from text.

        Args:
            text: Text to search.

        Returns:
            List of URLs found.
        """
        if not text:
            return []

        urls = self.url_pattern.findall(text)
        return list(set(urls))  # Remove duplicates

    def _has_ip_url(self, urls: list[str]) -> float:
        """Check if any URL uses IP address instead of domain.

        IP-based URLs are highly suspicious as they bypass domain reputation.

        Args:
            urls: List of URLs to check.

        Returns:
            1.0 if IP URL found, 0.0 otherwise.
        """
        for url in urls:
            if self.ip_pattern.search(url):
                return 1.0
        return 0.0

    def _avg_url_length(self, urls: list[str]) -> float:
        """Calculate average URL length (normalized).

        Phishing URLs are often longer due to obfuscation.

        Args:
            urls: List of URLs.

        Returns:
            Normalized average length in [0, 1].
        """
        if not urls:
            return 0.0

        # Filter out None or empty strings that might have been added
        valid_urls = [u for u in urls if u and isinstance(u, str)]

        if not valid_urls:  # Check again after filtering
            return 0.0

        avg_len = np.mean([len(url) for url in valid_urls])
        return min(1.0, avg_len / self.MAX_URL_LENGTH)

    def _max_url_length(self, urls: list[str]) -> float:
        """Calculate maximum URL length (normalized).

        Args:
            urls: List of URLs.

        Returns:
            Normalized max length in [0, 1].
        """
        if not urls:
            return 0.0

        max_len = max(len(url) for url in urls)
        return min(1.0, max_len / self.MAX_URL_LENGTH)

    def _has_suspicious_tld(self, urls: list[str]) -> float:
        """Check if any URL uses suspicious TLD.

        Args:
            urls: List of URLs to check.

        Returns:
            1.0 if suspicious TLD found, 0.0 otherwise.
        """
        for url in urls:
            try:
                parsed = urlparse(url)
                domain = parsed.netloc.lower()

                # Check if TLD is suspicious
                for tld in self.SUSPICIOUS_TLDS:
                    if domain.endswith(tld):
                        return 1.0
            except Exception:
                continue

        return 0.0

    def _has_https(self, urls: list[str]) -> float:
        """Check if any URL uses HTTPS.

        Note: HTTPS alone doesn't indicate legitimacy - phishing sites
        also use HTTPS. This feature is used in combination with others.

        Args:
            urls: List of URLs to check.

        Returns:
            1.0 if HTTPS URL found, 0.0 otherwise.
        """
        for url in urls:
            if url.startswith("https://"):
                return 1.0
        return 0.0

    def _avg_subdomain_count(self, urls: list[str]) -> float:
        """Calculate average subdomain count (normalized).

        Phishing URLs often use many subdomains for obfuscation.

        Args:
            urls: List of URLs.

        Returns:
            Normalized average subdomain count in [0, 1].
        """
        if not urls:
            return 0.0

        subdomain_counts = []
        for url in urls:
            try:
                parsed = urlparse(url)
                netloc = parsed.netloc

                # Count dots (subdomains + domain)
                dot_count = netloc.count(".")

                # Subtract 2 for domain.tld, remainder is subdomains
                subdomains = max(0, dot_count - 2)
                subdomain_counts.append(subdomains)

            except Exception:
                continue

        if not subdomain_counts:
            return 0.0

        avg_subdomains = np.mean(subdomain_counts)
        return min(1.0, avg_subdomains / self.MAX_SUBDOMAINS)

    def _has_url_shortener(self, urls: list[str]) -> float:
        """Check if any URL is a shortened URL.

        Shortened URLs can hide malicious destinations.

        Args:
            urls: List of URLs to check.

        Returns:
            1.0 if URL shortener found, 0.0 otherwise.
        """
        for url in urls:
            try:
                parsed = urlparse(url)
                domain = parsed.netloc.lower()

                for shortener in self.URL_SHORTENERS:
                    if domain == shortener or domain.endswith(f".{shortener}"):
                        return 1.0
            except Exception:
                continue

        return 0.0

    def _special_char_ratio(self, urls: list[str]) -> float:
        """Calculate ratio of special characters in URLs.

        Phishing URLs often use special characters for obfuscation.

        Args:
            urls: List of URLs.

        Returns:
            Normalized special character ratio in [0, 1].
        """
        if not urls:
            return 0.0

        total_chars = 0
        special_chars = 0

        # Special chars often used in phishing URLs
        special_set = set("@-._~:/?#[]@!$&'()*+,;=")

        for url in urls:
            total_chars += len(url)
            special_chars += sum(1 for c in url if c in special_set)

        if total_chars == 0:
            return 0.0

        ratio = special_chars / total_chars
        # Most URLs have < 20% special chars, so normalize by that
        return min(1.0, ratio / 0.2)

    def _has_port_specified(self, urls: list[str]) -> float:
        """Check if any URL specifies a non-standard port.

        Specifying ports is unusual in legitimate emails.

        Args:
            urls: List of URLs to check.

        Returns:
            1.0 if port specified, 0.0 otherwise.
        """
        for url in urls:
            if self.port_pattern.search(url):
                return 1.0
        return 0.0

    def _normalize_count(self, count: int, max_val: int) -> float:
        """Normalize count to [0, 1].

        Args:
            count: Count value.
            max_val: Maximum expected value.

        Returns:
            Normalized value in [0, 1].
        """
        if max_val <= 0:
            return 0.0
        return min(1.0, count / max_val)
