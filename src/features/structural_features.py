"""Structural feature extraction for phishing detection.

Extracts features related to email structure and format:
- HTML to text ratio
- Attachment presence and count
- Embedded image count
- External resource references
- Form presence
- JavaScript presence
- Email size
"""

import re
import time
from typing import Dict, List

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from .base import BaseExtractor


class StructuralFeatureExtractor(BaseExtractor):
    """Extract structural features for phishing detection.

    Features extracted:
        - html_text_ratio: HTML to text content ratio (normalized)
        - has_attachments: Contains attachments (0/1)
        - attachment_count: Number of attachments (normalized)
        - has_executable_attachment: Contains executable file (0/1)
        - has_office_attachment: Contains Office doc (0/1)
        - embedded_image_count: Number of embedded images (normalized)
        - external_image_count: Number of external images (normalized)
        - has_forms: Contains HTML forms (0/1)
        - has_javascript: Contains JavaScript (0/1)
        - email_size_kb: Email size in KB (normalized)
    """

    # Executable file extensions
    EXECUTABLE_EXTS = {".exe", ".scr", ".bat", ".com", ".pif", ".vbs", ".js", ".jar"}

    # Office document extensions
    OFFICE_EXTS = {
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".docm",
        ".xlsm",
        ".pptm",
    }

    # Max values for normalization
    MAX_ATTACHMENTS = 10
    MAX_IMAGES = 20
    MAX_EMAIL_SIZE_KB = 500  # 500 KB

    def __init__(self) -> None:
        """Initialize structural feature extractor."""
        super().__init__()
        self.feature_names = [
            "html_text_ratio",
            "has_attachments",
            "attachment_count",
            "has_executable_attachment",
            "has_office_attachment",
            "embedded_image_count",
            "external_image_count",
            "has_forms",
            "has_javascript",
            "email_size_kb",
        ]

        # Regex patterns
        self.form_pattern = re.compile(r"<form[^>]*>", re.IGNORECASE)
        self.script_pattern = re.compile(r"<script[^>]*>", re.IGNORECASE)
        self.external_img_pattern = re.compile(r'<img[^>]+src=["\']http', re.IGNORECASE)

    def fit(self, emails: pd.DataFrame) -> "StructuralFeatureExtractor":
        """Fit the structural extractor.

        Stateless extractor - validates input structure only.

        Args:
            emails: DataFrame with 'body', 'body_html', 'attachments' columns.

        Returns:
            self: Fitted extractor.
        """
        self._validate_input(emails)
        self._is_fitted = True
        return self

    def transform(self, emails: pd.DataFrame) -> pd.DataFrame:
        """Transform emails into structural features.

        Args:
            emails: DataFrame with required columns.

        Returns:
            DataFrame with structural features (n_emails, n_features).
        """
        if not self._is_fitted:
            raise RuntimeError("StructuralFeatureExtractor must be fitted before transform")

        results = []

        for idx, row in emails.iterrows():
            start_time = time.time()

            body = str(row.get("body", ""))
            body_html = str(row.get("body_html", ""))
            attachments = row.get("attachments", [])

            features = self._extract_structural_features(body, body_html, attachments)
            results.append(features)

            self.extraction_times.append((time.time() - start_time) * 1000)  # ms

        return pd.DataFrame(results, columns=self.feature_names)

    def _extract_structural_features(
        self, body: str, body_html: str, attachments: List[Dict]
    ) -> dict[str, float]:
        """Extract all structural features.

        Args:
            body: Plain text body.
            body_html: HTML body.
            attachments: List of attachment metadata.

        Returns:
            Dictionary of feature names to values in [0, 1].
        """
        features = {
            "html_text_ratio": self._calculate_html_text_ratio(body, body_html),
            "has_attachments": self._has_attachments(attachments),
            "attachment_count": self._count_attachments(attachments),
            "has_executable_attachment": self._has_executable(attachments),
            "has_office_attachment": self._has_office_doc(attachments),
            "embedded_image_count": self._count_embedded_images(body_html),
            "external_image_count": self._count_external_images(body_html),
            "has_forms": self._has_forms(body_html),
            "has_javascript": self._has_javascript(body_html),
            "email_size_kb": self._calculate_email_size(body, body_html, attachments),
        }

        return features

    def _calculate_html_text_ratio(self, body: str, body_html: str) -> float:
        """Calculate HTML to text ratio.

        Phishing emails often have disproportionate HTML to text content.

        Args:
            body: Plain text content.
            body_html: HTML content.

        Returns:
            Normalized ratio in [0, 1].
        """
        if not body_html:
            return 0.0

        # Extract text from HTML
        try:
            soup = BeautifulSoup(body_html, "lxml")
            html_text = soup.get_text(separator=" ", strip=True)
            html_text_len = len(html_text)
        except Exception:
            html_text_len = 0

        text_len = len(body)

        if text_len == 0 and html_text_len == 0:
            return 0.0

        if text_len == 0:
            # All HTML, no text
            return 1.0

        # Calculate ratio: HTML length / (HTML length + text length)
        # Normalize to [0, 1]
        ratio = html_text_len / (html_text_len + text_len)

        return min(1.0, ratio)

    def _has_attachments(self, attachments: List[Dict]) -> float:
        """Check if email has attachments.

        Args:
            attachments: List of attachment metadata.

        Returns:
            1.0 if attachments present, 0.0 otherwise.
        """
        if attachments and len(attachments) > 0:
            return 1.0
        return 0.0

    def _count_attachments(self, attachments: List[Dict]) -> float:
        """Count number of attachments.

        Args:
            attachments: List of attachment metadata.

        Returns:
            Normalized count in [0, 1].
        """
        if not attachments:
            return 0.0

        count = len(attachments)
        return min(1.0, count / self.MAX_ATTACHMENTS)

    def _has_executable(self, attachments: List[Dict]) -> float:
        """Check if email has executable attachment.

        Args:
            attachments: List of attachment metadata.

        Returns:
            1.0 if executable found, 0.0 otherwise.
        """
        if not attachments:
            return 0.0

        for attachment in attachments:
            filename = attachment.get("filename", "").lower()

            # Check extension
            for ext in self.EXECUTABLE_EXTS:
                if filename.endswith(ext):
                    return 1.0

            # Check MIME type
            content_type = attachment.get("content_type", "").lower()
            if "executable" in content_type or "application/x-executable" in content_type:
                return 1.0

        return 0.0

    def _has_office_doc(self, attachments: List[Dict]) -> float:
        """Check if email has Office document attachment.

        Office docs can contain macros (malicious).

        Args:
            attachments: List of attachment metadata.

        Returns:
            1.0 if Office doc found, 0.0 otherwise.
        """
        if not attachments:
            return 0.0

        for attachment in attachments:
            filename = attachment.get("filename", "").lower()

            # Check extension
            for ext in self.OFFICE_EXTS:
                if filename.endswith(ext):
                    return 1.0

            # Check MIME type
            content_type = attachment.get("content_type", "").lower()
            if any(
                office_type in content_type
                for office_type in [
                    "msword",
                    "ms-excel",
                    "ms-powerpoint",
                    "officedocument",
                ]
            ):
                return 1.0

        return 0.0

    def _count_embedded_images(self, body_html: str) -> float:
        """Count embedded images (base64 or CID).

        Args:
            body_html: HTML content.

        Returns:
            Normalized count in [0, 1].
        """
        if not body_html:
            return 0.0

        try:
            soup = BeautifulSoup(body_html, "lxml")
            imgs = soup.find_all("img")

            # Count embedded (non-external) images
            embedded_count = 0
            for img in imgs:
                src = img.get("src", "")
                # Embedded images use cid: or base64 data:
                if src.startswith("cid:") or src.startswith("data:image"):
                    embedded_count += 1

            return min(1.0, embedded_count / self.MAX_IMAGES)

        except Exception:
            return 0.0

    def _count_external_images(self, body_html: str) -> float:
        """Count external image references.

        External images can be tracking pixels.

        Args:
            body_html: HTML content.

        Returns:
            Normalized count in [0, 1].
        """
        if not body_html:
            return 0.0

        try:
            matches = self.external_img_pattern.findall(body_html)
            count = len(matches)

            return min(1.0, count / self.MAX_IMAGES)

        except Exception:
            return 0.0

    def _has_forms(self, body_html: str) -> float:
        """Check if HTML contains forms.

        Forms in emails are suspicious (phishing for credentials).

        Args:
            body_html: HTML content.

        Returns:
            1.0 if forms present, 0.0 otherwise.
        """
        if not body_html:
            return 0.0

        if self.form_pattern.search(body_html):
            return 1.0
        return 0.0

    def _has_javascript(self, body_html: str) -> float:
        """Check if HTML contains JavaScript.

        JavaScript in emails is highly suspicious.

        Args:
            body_html: HTML content.

        Returns:
            1.0 if JavaScript present, 0.0 otherwise.
        """
        if not body_html:
            return 0.0

        if self.script_pattern.search(body_html):
            return 1.0
        return 0.0

    def _calculate_email_size(
        self, body: str, body_html: str, attachments: List[Dict]
    ) -> float:
        """Calculate email size in KB.

        Args:
            body: Plain text body.
            body_html: HTML body.
            attachments: List of attachment metadata.

        Returns:
            Normalized size in [0, 1].
        """
        # Calculate body size
        body_size = len(body.encode("utf-8", errors="ignore"))
        html_size = len(body_html.encode("utf-8", errors="ignore"))

        # Calculate attachment size
        attachment_size = 0
        if attachments:
            for attachment in attachments:
                attachment_size += attachment.get("size", 0)

        # Total size in bytes
        total_bytes = body_size + html_size + attachment_size

        # Convert to KB
        size_kb = total_bytes / 1024

        # Normalize
        return min(1.0, size_kb / self.MAX_EMAIL_SIZE_KB)
