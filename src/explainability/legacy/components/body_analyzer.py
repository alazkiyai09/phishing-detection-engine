"""
Body component analyzer.

Analyzes email body content for suspicious patterns including:
- Social engineering tactics
- Grammar and quality issues
- Pressure language
- Request for sensitive information
"""

from typing import List
import re

from src.explainability.legacy.utils.data_structures import BodyExplanation, EmailData
from src.explainability.legacy.utils.text_processing import (
    detect_social_engineering,
    detect_pressure_language,
    detect_grammar_issues
)


class BodyAnalyzer:
    """
    Analyze email body for suspicious patterns.

    Body content is typically checked after subject by humans.
    """

    # Sensitive information indicators
    SENSITIVE_REQUESTS = [
        'password', 'social security', 'credit card', 'bank account',
        'ssn', 'cvv', 'pin', 'login details', 'verify identity',
        'confirm account', 'update payment', 'provide information'
    ]

    # Action requests (typical in phishing)
    ACTION_REQUESTS = [
        'click here', 'click the link', 'visit website', 'download attachment',
        'open attachment', 'enable macros', 'run macro', 'verify now',
        'confirm now', 'update now', 'click immediately'
    ]

    # Threatening language
    THREATENING_LANGUAGE = [
        'account will be', 'account suspended', 'legal action',
        'lawsuit', 'police', 'fbi', 'authorities', 'prosecution',
        'lose access', 'service terminated', 'immediate action'
    ]

    # Financial keywords
    FINANCIAL_KEYWORDS = [
        'invoice', 'payment', 'refund', 'transfer', 'wire',
        'bank', 'credit card', 'debit', 'transaction', 'deposit'
    ]

    # Typical grammar issues in phishing
    GRAMMAR_PATTERNS = {
        'missing_articles': r'\b(a|an|the)\s+(is|are|was|were)\s+(?:noun|verb|adjective)',
        'wrong_tense': r'\bwill\b.*\bed\b',
        'subject_verb_disagreement': None,  # Complex, detected via heuristics
        'exclamation_overuse': r'!{3,}',
    }

    def __init__(self, strict_mode: bool = False):
        """
        Initialize body analyzer.

        Args:
            strict_mode: If True, more conservative in marking as safe
        """
        self.strict_mode = strict_mode

    def analyze(self, email: EmailData) -> BodyExplanation:
        """
        Analyze email body content.

        Args:
            email: Email data to analyze

        Returns:
            BodyExplanation with analysis results
        """
        body = email.body
        reasons = []
        is_suspicious = False

        # Detect social engineering tactics
        social_engineering_tactics = detect_social_engineering(body)
        if social_engineering_tactics:
            is_suspicious = True
            reasons.append(f"Uses social engineering: {len(social_engineering_tactics)} tactics detected")

        # Detect pressure language
        pressure_language = detect_pressure_language(body)
        if pressure_language:
            is_suspicious = True
            reasons.append(f"Uses pressure language: {', '.join(pressure_language)}")

        # Detect requests for sensitive information
        sensitive_requests = self._detect_sensitive_requests(body)
        if sensitive_requests:
            is_suspicious = True
            reasons.append(f"Requests sensitive information: {', '.join(sensitive_requests)}")

        # Detect action requests
        action_requests = self._detect_action_requests(body)
        if action_requests:
            is_suspicious = True
            reasons.append("Contains suspicious action requests")

        # Detect threatening language
        threats = self._detect_threats(body)
        if threats:
            is_suspicious = True
            reasons.append(f"Uses threatening language: {', '.join(threats)}")

        # Detect grammar issues
        grammar_issues = detect_grammar_issues(body)
        if len(grammar_issues) > 1:  # More than 1 issue is suspicious
            is_suspicious = True
            reasons.append(f"Quality issues detected: {', '.join(grammar_issues)}")

        # Check for generic greetings
        if self._has_generic_greeting(body):
            is_suspicious = True
            reasons.append("Uses generic greeting (typical of mass phishing)")

        # Check for personalized information mismatch
        if self._has_personalization_mismatch(email):
            is_suspicious = True
            reasons.append("Personalization inconsistencies detected")

        # Calculate confidence
        confidence = self._calculate_confidence(
            is_suspicious,
            len(social_engineering_tactics),
            len(grammar_issues),
            len(sensitive_requests)
        )

        return BodyExplanation(
            is_suspicious=is_suspicious,
            confidence=confidence,
            reasons=reasons,
            social_engineering_tactics=social_engineering_tactics,
            grammar_issues=grammar_issues,
            pressure_language=pressure_language
        )

    def _detect_sensitive_requests(self, body: str) -> List[str]:
        """Detect requests for sensitive information."""
        body_lower = body.lower()
        found = []

        for keyword in self.SENSITIVE_REQUESTS:
            if keyword in body_lower:
                found.append(keyword)

        return found

    def _detect_action_requests(self, body: str) -> List[str]:
        """Detect suspicious action requests."""
        body_lower = body.lower()
        found = []

        for request in self.ACTION_REQUESTS:
            if request in body_lower:
                found.append(request)

        return found

    def _detect_threats(self, body: str) -> List[str]:
        """Detect threatening language."""
        body_lower = body.lower()
        found = []

        for threat in self.THREATENING_LANGUAGE:
            if threat in body_lower:
                found.append(threat)

        return found

    def _has_generic_greeting(self, body: str) -> bool:
        """Check for generic greetings."""
        generic_greetings = [
            'dear customer', 'dear valued customer', 'dear client',
            'dear user', 'dear account holder', 'dear sir/madam',
            'greetings', 'hello customer', 'attention account holder'
        ]

        body_lower = body.lower()
        return any(greeting in body_lower for greeting in generic_greetings)

    def _has_personalization_mismatch(self, email: EmailData) -> bool:
        """
        Check for personalization inconsistencies.

        For example, greeting doesn't match recipient name.
        """
        # If we have recipient information, check if it's used
        if email.recipients and len(email.recipients) > 0:
            recipient_name = email.recipients[0].display_name

            if recipient_name:
                # Check if name appears in body
                body_lower = email.body.lower()
                name_lower = recipient_name.lower()

                # Check for generic greeting instead of personalized
                generic_greetings = ['dear customer', 'dear client', 'dear user']
                has_generic = any(greeting in body_lower for greeting in generic_greetings)

                # If we have recipient name but use generic greeting
                if has_generic and name_lower not in body_lower:
                    return True

        return False

    def _calculate_confidence(
        self,
        is_suspicious: bool,
        num_tactics: int,
        num_grammar: int,
        num_sensitive: int
    ) -> float:
        """
        Calculate confidence score.

        Args:
            is_suspicious: Whether body is flagged
            num_tactics: Number of social engineering tactics
            num_grammar: Number of grammar issues
            num_sensitive: Number of sensitive requests

        Returns:
            Confidence score (0.0 to 1.0)
        """
        if not is_suspicious:
            return 0.75

        # Base confidence
        base = 0.50

        # Social engineering is strong indicator
        tactics_bonus = min(num_tactics * 0.08, 0.25)

        # Sensitive requests are also strong indicator
        sensitive_bonus = min(num_sensitive * 0.10, 0.20)

        # Grammar issues add some confidence
        grammar_bonus = min(num_grammar * 0.03, 0.10)

        return min(base + tactics_bonus + sensitive_bonus + grammar_bonus, 0.95)

    def analyze_multiple(self, emails: List[EmailData]) -> List[BodyExplanation]:
        """
        Analyze bodies for multiple emails.

        Args:
            emails: List of emails to analyze

        Returns:
            List of body explanations
        """
        return [self.analyze(email) for email in emails]
