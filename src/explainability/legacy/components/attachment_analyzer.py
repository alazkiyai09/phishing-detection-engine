"""
Attachment component analyzer.

Analyzes email attachments for suspicious patterns including:
- Dangerous file types
- Macro presence
- Double extensions
- Suspicious filenames
"""

from typing import List, Set, Optional
import os
import re

from src.explainability.legacy.utils.data_structures import Attachment, AttachmentExplanation, EmailData


class AttachmentAnalyzer:
    """
    Analyze email attachments for suspicious patterns.

    Attachments are typically checked last by humans.
    """

    # Dangerous file types
    DANGEROUS_EXTENSIONS = {
        # Executables
        '.exe', '.scr', '.bat', '.cmd', '.pif', '.com',
        # Scripts
        '.js', '.vbs', '.vb', '.ps1', '.sh',
        # Documents with macros
        '.doc', '.docm', '.xls', '.xlsm', '.ppt', '.pptm',
        # Archives (can hide malware)
        '.zip', '.rar', '.7z', '.tar', '.gz',
        # Other suspicious
        '.msi', '.dll', '.sys', '.cpl'
    }

    # Safe file types (that typically don't contain malware)
    SAFE_EXTENSIONS = {
        '.jpg', '.jpeg', '.png', '.gif', '.pdf', '.txt',
        '.docx', '.xlsx', '.pptx', '.mp3', '.mp4'
    }

    # Suspicious filename patterns
    SUSPICIOUS_PATTERNS = [
        r'invoice\d*',
        r'receipt\d*',
        r'invoice_\d{4,}',  # invoice_2023, etc.
        r'document_\d+',
        r'scan_\d+',
        r'photo_\d+',
        r'order_\d+',
        r'shipment_\d+',
        r'payment_\d+',
        r'transaction_\d+'
    ]

    # Common words in malicious attachment names
    SUSPICIOUS_KEYWORDS = [
        'invoice', 'receipt', 'payment', 'order', 'shipment',
        'document', 'scan', 'photo', 'urgent', 'important',
        'confidential', 'secret', 'verify', 'confirm'
    ]

    # Double extension patterns
    DOUBLE_EXTENSION_PATTERN = r'\.\w+\.\w+$'

    def __init__(self, strict_mode: bool = False):
        """
        Initialize attachment analyzer.

        Args:
            strict_mode: If True, more conservative in marking as safe
        """
        self.strict_mode = strict_mode

    def analyze(self, email: EmailData) -> AttachmentExplanation:
        """
        Analyze attachments in email.

        Args:
            email: Email data to analyze

        Returns:
            AttachmentExplanation with analysis results
        """
        attachments = email.attachments
        reasons = []
        dangerous_attachments = []
        is_suspicious = False

        # Analyze each attachment
        for attachment in attachments:
            attachment_analysis = self._analyze_single_attachment(attachment)

            if attachment_analysis['is_dangerous']:
                is_suspicious = True
                dangerous_attachments.append(attachment_analysis)
                reasons.append(attachment_analysis['reason'])

        # Calculate confidence
        confidence = self._calculate_confidence(is_suspicious, len(dangerous_attachments))

        return AttachmentExplanation(
            is_suspicious=is_suspicious,
            confidence=confidence,
            reasons=reasons,
            dangerous_attachments=dangerous_attachments
        )

    def _analyze_single_attachment(self, attachment: Attachment) -> dict:
        """
        Analyze a single attachment.

        Returns:
            Dict with analysis results
        """
        is_dangerous = False
        reasons = []

        filename = attachment.filename.lower()
        file_ext = os.path.splitext(filename)[1].lower()

        # Check for dangerous file type
        if file_ext in self.DANGEROUS_EXTENSIONS:
            is_dangerous = True
            reasons.append(f'Dangerous file type: {file_ext}')

        # Check for macros
        if attachment.has_macros:
            is_dangerous = True
            reasons.append('Contains macros (can execute code)')

        # Check for double extension
        if self._has_double_extension(attachment.filename):
            is_dangerous = True
            reasons.append('Double extension (hides real file type)')

        # Check for suspicious filename pattern
        if self._has_suspicious_pattern(attachment.filename):
            is_dangerous = True
            reasons.append('Suspicious filename pattern')

        # Check for suspicious keywords
        suspicious_keywords = self._get_suspicious_keywords(attachment.filename)
        if suspicious_keywords:
            is_dangerous = True
            reasons.append(f'Suspicious keywords: {", ".join(suspicious_keywords)}')

        # Check for very long filenames (obfuscation)
        if len(attachment.filename) > 100:
            is_dangerous = True
            reasons.append('Unusually long filename (potential obfuscation)')

        # Check for excessive spaces or special characters
        if self._has_obfuscated_name(attachment.filename):
            is_dangerous = True
            reasons.append('Filename contains obfuscation characters')

        # Check size (suspicious if very small but executable-looking)
        if attachment.size_bytes < 1000 and file_ext in {'.exe', '.scr', '.bat'}:
            is_dangerous = True
            reasons.append('Suspicious: Very small executable file')

        # Build reason string
        reason_str = '; '.join(reasons) if reasons else 'Suspicious attachment'

        return {
            'filename': attachment.filename,
            'file_type': attachment.file_type,
            'is_dangerous': is_dangerous,
            'reason': reason_str
        }

    def _has_double_extension(self, filename: str) -> bool:
        """Check for double extension (e.g., file.doc.exe)."""
        return bool(re.search(self.DOUBLE_EXTENSION_PATTERN, filename.lower()))

    def _has_suspicious_pattern(self, filename: str) -> bool:
        """Check for suspicious filename patterns."""
        filename_lower = filename.lower()

        for pattern in self.SUSPICIOUS_PATTERNS:
            if re.search(pattern, filename_lower):
                return True

        return False

    def _get_suspicious_keywords(self, filename: str) -> List[str]:
        """Get suspicious keywords found in filename."""
        filename_lower = filename.lower()

        found = [kw for kw in self.SUSPICIOUS_KEYWORDS if kw in filename_lower]

        return found

    def _has_obfuscated_name(self, filename: str) -> bool:
        """Check for obfuscation characters in filename."""
        # Check for excessive spaces
        if '  ' in filename:
            return True

        # Check for special characters (excluding common ones)
        common_chars = {'.', '-', '_', ' ', '(', ')', '@', '#', '~'}
        special_chars = [c for c in filename if not c.isalnum() and c not in common_chars]

        if len(special_chars) > 3:
            return True

        return False

    def _calculate_confidence(self, is_suspicious: bool, num_dangerous: int) -> float:
        """
        Calculate confidence score.

        Args:
            is_suspicious: Whether any attachments are dangerous
            num_dangerous: Number of dangerous attachments

        Returns:
            Confidence score (0.0 to 1.0)
        """
        if not is_suspicious:
            return 0.90  # High confidence when no dangerous attachments

        # Base confidence increases with number of dangerous attachments
        base = 0.70
        increase = min(num_dangerous * 0.10, 0.20)

        return min(base + increase, 0.95)

    def analyze_multiple(self, emails: List[EmailData]) -> List[AttachmentExplanation]:
        """
        Analyze attachments for multiple emails.

        Args:
            emails: List of emails to analyze

        Returns:
            List of attachment explanations
        """
        return [self.analyze(email) for email in emails]
