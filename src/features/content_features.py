"""Content-based feature extraction for phishing detection.

Extracts features from email content text:
- Urgency keywords (immediate action required)
- Call-to-action density
- Pressure tactics
- Financial terminology
- Threat language
- Link and button text analysis
"""

import re
import time
from typing import List

import numpy as np
import pandas as pd

from .base import BaseExtractor


class ContentFeatureExtractor(BaseExtractor):
    """Extract content-based features for phishing detection.

    Features extracted:
        - urgency_keyword_count: Urgency terms frequency (normalized)
        - cta_button_count: Call-to-action phrases count (normalized)
        - threat_language_count: Threatening phrases count (normalized)
        - financial_term_count: Financial terms frequency (normalized)
        - immediate_action_count: "Immediate" action requests (normalized)
        - verification_request_count: "Verify your account" phrases (normalized)
        - click_here_count: "Click here" frequency (normalized)
        - password_request_count: Password/password-related requests (normalized)
        - account_suspended_count: Account suspension warnings (normalized)

    Note: URL features are extracted by URLFeatureExtractor to avoid redundancy.
    """

    # Keyword dictionaries for feature extraction
    URGENCY_KEYWORDS = [
        "urgent",
        "immediately",
        "asap",
        "right away",
        "at once",
        "without delay",
        "hurry",
        "expires",
        "deadline",
        "time sensitive",
        "act now",
        "do not wait",
        "before it's too late",
        "limited time",
    ]

    CTA_PHRASES = [
        "click here",
        "click the link",
        "follow the link",
        "use the link",
        "visit the link",
        "open the link",
        "select the link",
        "press the link",
        "tap the link",
        "download now",
        "install now",
        "update now",
        "verify now",
        "confirm now",
    ]

    THREAT_LANGUAGE = [
        "account will be closed",
        "account suspended",
        "account terminated",
        "service suspended",
        "access denied",
        "legal action",
        "immediate attention",
        "serious consequences",
        "your account is on hold",
        "unusual activity",
        "suspicious activity",
        "security breach",
        "compromised account",
        "locked out",
        "permanently deleted",
    ]

    FINANCIAL_TERMS = [
        "bank account",
        "credit card",
        "debit card",
        "routing number",
        "account number",
        "social security",
        "ssn",
        "wire transfer",
        "payment",
        "invoice",
        "transaction",
        "balance",
        "statement",
        "deposit",
        "withdrawal",
    ]

    VERIFICATION_PHRASES = [
        "verify your account",
        "verify your identity",
        "confirm your account",
        "confirm your identity",
        "validate your account",
        "update your information",
        "update your details",
        "verify your email",
        "confirm your email",
    ]

    PASSWORD_REQUESTS = [
        "password",
        "pin",
        "security question",
        "mother's maiden name",
        "login credentials",
        "sign in credentials",
        "user id",
        "username",
        "otp",
        "one time password",
        "authentication",
    ]

    # Max values for normalization
    MAX_KEYWORD_COUNT = 10

    def __init__(self) -> None:
        """Initialize content feature extractor."""
        super().__init__()
        self.feature_names = [
            "urgency_keyword_count",
            "cta_button_count",
            "threat_language_count",
            "financial_term_count",
            "immediate_action_count",
            "verification_request_count",
            "click_here_count",
            "password_request_count",
            "account_suspended_count",
        ]

    def fit(self, emails: pd.DataFrame) -> "ContentFeatureExtractor":
        """Fit the content extractor.

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
        """Transform emails into content-based features.

        Args:
            emails: DataFrame with 'body' and 'subject' columns.

        Returns:
            DataFrame with content features (n_emails, n_features).
        """
        if not self._is_fitted:
            raise RuntimeError("ContentFeatureExtractor must be fitted before transform")

        results = []

        for idx, row in emails.iterrows():
            start_time = time.time()

            body = str(row.get("body", ""))
            subject = str(row.get("subject", ""))

            # Combine subject and body for analysis
            text = f"{subject} {body}"

            features = self._extract_content_features(text)
            results.append(features)

            self.extraction_times.append((time.time() - start_time) * 1000)  # ms

        return pd.DataFrame(results, columns=self.feature_names)

    def _extract_content_features(self, text: str) -> dict[str, float]:
        """Extract all content features.

        Args:
            text: Full text (subject + body).

        Returns:
            Dictionary of feature names to values in [0, 1].
        """
        if not text:
            return {name: 0.0 for name in self.feature_names}

        text_lower = text.lower()

        features = {
            "urgency_keyword_count": self._count_keywords(
                text_lower, self.URGENCY_KEYWORDS
            ),
            "cta_button_count": self._count_keywords(text_lower, self.CTA_PHRASES),
            "threat_language_count": self._count_keywords(
                text_lower, self.THREAT_LANGUAGE
            ),
            "financial_term_count": self._count_keywords(
                text_lower, self.FINANCIAL_TERMS
            ),
            "immediate_action_count": self._count_immediate_actions(text_lower),
            "verification_request_count": self._count_keywords(
                text_lower, self.VERIFICATION_PHRASES
            ),
            "click_here_count": self._count_phrase_occurrences(
                text_lower, "click here"
            ),
            "password_request_count": self._count_keywords(
                text_lower, self.PASSWORD_REQUESTS
            ),
            "account_suspended_count": self._count_account_suspended(text_lower),
        }

        return features

    def _count_keywords(self, text: str, keywords: List[str]) -> float:
        """Count keyword occurrences in text.

        Args:
            text: Text to search.
            keywords: List of keywords to count.

        Returns:
            Normalized count in [0, 1].
        """
        count = 0
        for keyword in keywords:
            # Count all occurrences
            count += text.count(keyword.lower())

        return min(1.0, count / self.MAX_KEYWORD_COUNT)

    def _count_immediate_actions(self, text: str) -> float:
        """Count immediate action requests.

        Args:
            text: Text to search.

        Returns:
            Normalized count in [0, 1].
        """
        immediate_phrases = [
            "immediately",
            "right away",
            "at once",
            "as soon as possible",
            "asap",
            "within 24 hours",
            "within 48 hours",
            "today",
            "now",
        ]

        count = 0
        for phrase in immediate_phrases:
            count += text.count(phrase.lower())

        return min(1.0, count / self.MAX_KEYWORD_COUNT)

    def _count_phrase_occurrences(self, text: str, phrase: str) -> float:
        """Count occurrences of a specific phrase.

        Args:
            text: Text to search.
            phrase: Phrase to count.

        Returns:
            Normalized count in [0, 1].
        """
        count = text.count(phrase.lower())
        return min(1.0, count / 5)  # Max 5 occurrences for normalization

    def _count_account_suspended(self, text: str) -> float:
        """Count account suspension warnings.

        Args:
            text: Text to search.

        Returns:
            Normalized count in [0, 1].
        """
        suspension_phrases = [
            "account will be suspended",
            "account has been suspended",
            "account suspended",
            "service will be suspended",
            "account will be closed",
            "account will be terminated",
            "your account is on hold",
            "account locked",
        ]

        count = 0
        for phrase in suspension_phrases:
            count += text.count(phrase.lower())

        return min(1.0, count / 3)  # Max 3 occurrences
