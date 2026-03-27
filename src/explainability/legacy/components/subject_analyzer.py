"""
Subject component analyzer.

Analyzes email subject line for suspicious patterns including:
- Urgency keywords
- Unusual formatting (ALL CAPS, excessive punctuation)
- Pressure language
- Suspicious phrases
"""

from typing import List
import re

from src.explainability.legacy.utils.data_structures import SubjectExplanation, EmailData
from src.explainability.legacy.utils.text_processing import (
    detect_urgency_keywords,
    detect_pressure_language
)


class SubjectAnalyzer:
    """
    Analyze email subject line for suspicious patterns.

    Subject is typically checked second by humans after sender.
    """

    # Urgency keywords
    URGENCY_KEYWORDS = [
        'urgent', 'immediately', 'asap', 'right away', 'at once',
        'deadline', 'expires', 'expiring', 'limited time',
        'act now', 'don\'t wait', 'hurry', 'time sensitive',
        'final notice', 'last chance', 'account suspended',
        'verify immediately', 'confirm now', 'payment overdue'
    ]

    # Pressure phrases
    PRESSURE_PHRASES = [
        'you must', 'required to', 'mandatory', 'compulsory',
        'or else', 'otherwise', 'failure to', 'legal action',
        'immediate attention', 'serious consequences'
    ]

    # Suspicious phrases
    SUSPICIOUS_PHRASES = [
        'you have won', 'congratulations', 'claim your prize',
        'your account', 'verify your identity', 'confirm your account',
        'unusual sign-in', 'suspicious activity', 'security alert',
        'password expires', 'update your information'
    ]

    # Formatting issues to detect
    FORMATTING_PATTERNS = {
        'all_caps': r'^[A-Z\s\W]+$',
        'excessive_punctuation': r'!{2,}|\?{2,}',
        'excessive_spaces': r' {3,}',
        'starts_with_re': r'^(re|fw|fwd):\s*',
        'missing_spaces': r'[.,!?][A-Z]'
    }

    def __init__(self, strict_mode: bool = False):
        """
        Initialize subject analyzer.

        Args:
            strict_mode: If True, more conservative in marking as safe
        """
        self.strict_mode = strict_mode

    def analyze(self, email: EmailData) -> SubjectExplanation:
        """
        Analyze email subject line.

        Args:
            email: Email data to analyze

        Returns:
            SubjectExplanation with analysis results
        """
        subject = email.subject
        reasons = []
        is_suspicious = False
        urgency_keywords = []
        unusual_formatting = []

        # Check for urgency keywords
        urgency_keywords = detect_urgency_keywords(subject)
        if len(urgency_keywords) > 0:
            is_suspicious = True
            reasons.append(f"Contains urgency words: {', '.join(urgency_keywords)}")

        # Check for pressure language
        pressure_found = detect_pressure_language(subject)
        if pressure_found:
            is_suspicious = True
            reasons.append(f"Uses pressure language: {', '.join(pressure_found)}")

        # Check for suspicious phrases
        suspicious_found = self._detect_suspicious_phrases(subject)
        if suspicious_found:
            is_suspicious = True
            reasons.append(f"Contains suspicious phrases: {', '.join(suspicious_found)}")

        # Check for unusual formatting
        unusual_formatting = self._detect_formatting_issues(subject)
        if unusual_formatting:
            is_suspicious = True
            reasons.append(f"Unusual formatting: {', '.join(unusual_formatting)}")

        # Check for excessive capitalization
        if self._has_excessive_caps(subject):
            is_suspicious = True
            reasons.append("Excessive use of capital letters")

        # Check for character substitution (e.g., $ for S)
        if self._has_char_substitution(subject):
            is_suspicious = True
            reasons.append("Suspicious character substitutions")

        # Calculate confidence
        confidence = self._calculate_confidence(
            is_suspicious,
            len(urgency_keywords),
            len(unusual_formatting)
        )

        return SubjectExplanation(
            is_suspicious=is_suspicious,
            confidence=confidence,
            reasons=reasons,
            urgency_keywords=urgency_keywords,
            unusual_formatting=unusual_formatting
        )

    def _detect_suspicious_phrases(self, subject: str) -> List[str]:
        """Detect suspicious phrases in subject."""
        subject_lower = subject.lower()
        found = []

        for phrase in self.SUSPICIOUS_PHRASES:
            if phrase in subject_lower:
                found.append(phrase)

        return found

    def _detect_formatting_issues(self, subject: str) -> List[str]:
        """Detect formatting issues in subject."""
        issues = []

        for issue_name, pattern in self.FORMATTING_PATTERNS.items():
            if re.search(pattern, subject, re.IGNORECASE):
                # Convert issue_name to human-readable
                human_readable = issue_name.replace('_', ' ')
                issues.append(human_readable)

        return issues

    def _has_excessive_caps(self, subject: str) -> bool:
        """Check if subject has excessive capitalization."""
        # Remove spaces and check
        without_spaces = subject.replace(' ', '')

        if len(without_spaces) < 5:
            return False

        caps_ratio = sum(c.isupper() for c in without_spaces) / len(without_spaces)
        return caps_ratio > 0.7

    def _has_char_substitution(self, subject: str) -> bool:
        """Check for character substitution attempts."""
        substitutions = {
            '$': 's', '@': 'a', '0': 'o', '1': 'i', '3': 'e',
            '!': 'i', '|': 'l', '7': 't'
        }

        for char, replacement in substitutions.items():
            if char in subject.lower():
                # Check if it's substituting a letter
                return True

        return False

    def _calculate_confidence(
        self,
        is_suspicious: bool,
        num_urgency: int,
        num_formatting: int
    ) -> float:
        """
        Calculate confidence score.

        Args:
            is_suspicious: Whether subject is flagged
            num_urgency: Number of urgency keywords
            num_formatting: Number of formatting issues

        Returns:
            Confidence score (0.0 to 1.0)
        """
        if not is_suspicious:
            return 0.80

        # Base confidence
        base = 0.55

        # Urgency adds more confidence than formatting
        urgency_bonus = min(num_urgency * 0.10, 0.25)
        formatting_bonus = min(num_formatting * 0.05, 0.15)

        return min(base + urgency_bonus + formatting_bonus, 0.95)

    def analyze_multiple(self, emails: List[EmailData]) -> List[SubjectExplanation]:
        """
        Analyze subjects for multiple emails.

        Args:
            emails: List of emails to analyze

        Returns:
            List of subject explanations
        """
        return [self.analyze(email) for email in emails]
