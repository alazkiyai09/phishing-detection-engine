"""
Sender component analyzer.

Analyzes email sender for suspicious patterns including:
- Domain reputation
- Display name spoofing
- Lookalike domains
- Email mismatch patterns
"""

from typing import List, Set, Optional
from urllib.parse import urlparse
import re

from src.explainability.legacy.utils.data_structures import (
    EmailAddress,
    SenderExplanation,
    EmailData
)
from src.explainability.legacy.utils.text_processing import (
    extract_domain_from_email,
    check_lookalike_domain
)


class SenderAnalyzer:
    """
    Analyze email sender for suspicious patterns.

    Follows cognitive processing order - sender is checked first by humans.
    """

    # Known legitimate domains (in production, load from database)
    LEGITIMATE_DOMAINS = {
        'gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com',
        'apple.com', 'amazon.com', 'microsoft.com', 'google.com',
        'facebook.com', 'linkedin.com', 'twitter.com',
        'chase.com', 'bankofamerica.com', 'wellsfargo.com',
        'citibank.com', 'capitalone.com'
    }

    # Suspicious TLDs
    SUSPICIOUS_TLDS = {
        '.xyz', '.top', '.zip', '.mov', '.tk', '.ml', '.ga',
        '.cf', '.gq', '.cc', '.pw'
    }

    # Common free email providers
    FREE_EMAIL_PROVIDERS = {
        'gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com',
        'aol.com', 'icloud.com', 'protonmail.com', 'mail.com'
    }

    def __init__(
        self,
        legitimate_domains: Optional[Set[str]] = None,
        strict_mode: bool = False
    ):
        """
        Initialize sender analyzer.

        Args:
            legitimate_domains: Set of known legitimate domains
            strict_mode: If True, more conservative in marking as safe
        """
        self.legitimate_domains = legitimate_domains or self.LEGITIMATE_DOMAINS
        self.strict_mode = strict_mode

    def analyze(self, email: EmailData) -> SenderExplanation:
        """
        Analyze email sender.

        Args:
            email: Email data to analyze

        Returns:
            SenderExplanation with analysis results
        """
        sender = email.sender
        reasons = []
        is_suspicious = False

        # Check for display name mismatch
        display_name_mismatch = self._check_display_name_mismatch(sender)
        if display_name_mismatch:
            is_suspicious = True
            reasons.append("Display name doesn't match email address")

        # Check for lookalike domain
        lookalike_domain = check_lookalike_domain(
            extract_domain_from_email(sender.email),
            self.legitimate_domains
        )
        if lookalike_domain:
            is_suspicious = True
            reasons.append("Email domain looks similar to well-known company")

        # Check domain reputation
        domain_reputation = self._check_domain_reputation(sender)
        if domain_reputation == "poor":
            is_suspicious = True
            reasons.append("Email domain has poor reputation")

        # Check for suspicious TLD
        if self._has_suspicious_tld(sender):
            is_suspicious = True
            reasons.append("Email domain uses uncommon extension")

        # Check for numeric domains
        if self._has_numeric_domain(sender):
            is_suspicious = True
            reasons.append("Email domain contains unusual numbers")

        # Check for role-based addresses in business context
        if self._is_suspicious_role_address(sender):
            is_suspicious = True
            reasons.append("Suspicious role-based email address")

        # Calculate confidence
        confidence = self._calculate_confidence(is_suspicious, len(reasons))

        return SenderExplanation(
            is_suspicious=is_suspicious,
            confidence=confidence,
            reasons=reasons,
            domain_reputation=domain_reputation,
            display_name_mismatch=display_name_mismatch,
            lookalike_domain=lookalike_domain
        )

    def _check_display_name_mismatch(self, sender: EmailAddress) -> bool:
        """Check if display name doesn't match email address."""
        if not sender.display_name:
            return False

        # Extract name from display name
        display_name = sender.display_name.lower().strip()
        email_local = sender.email.split('@')[0].lower()

        # Remove common separators
        email_local_clean = re.sub(r'[._-]', '', email_local)

        # Check if display name is contained in email or vice versa
        return (
            display_name not in email_local_clean and
            email_local_clean not in display_name
        )

    def _check_domain_reputation(self, sender: EmailAddress) -> str:
        """
        Check domain reputation.

        Returns: 'good', 'unknown', or 'poor'
        """
        domain = extract_domain_from_email(sender.email).lower()

        # Known legitimate domains
        if domain in self.legitimate_domains:
            return "good"

        # Known free email providers
        if domain in self.FREE_EMAIL_PROVIDERS:
            return "good"

        # Suspicious TLDs
        if any(domain.endswith(tld) for tld in self.SUSPICIOUS_TLDS):
            return "poor"

        # Recently registered (check length - very short or very long domains suspicious)
        if len(domain) < 6 or len(domain) > 30:
            return "poor"

        return "unknown"

    def _has_suspicious_tld(self, sender: EmailAddress) -> bool:
        """Check if domain has suspicious TLD."""
        domain = extract_domain_from_email(sender.email).lower()
        return any(domain.endswith(tld) for tld in self.SUSPICIOUS_TLDS)

    def _has_numeric_domain(self, sender: EmailAddress) -> bool:
        """Check if domain has unusual numeric patterns."""
        domain = extract_domain_from_email(sender.email).split('@')[0]

        # Check for excessive numbers
        numbers = sum(c.isdigit() for c in domain)
        return numbers > 3

    def _is_suspicious_role_address(self, sender: EmailAddress) -> bool:
        """Check for suspicious role-based addresses."""
        local_part = sender.email.split('@')[0].lower()

        suspicious_roles = {
            'admin', 'administrator', 'support', 'info',
            'service', 'help', 'contact', 'billing'
        }

        # Role addresses from free providers are suspicious
        domain = extract_domain_from_email(sender.email)
        if domain in self.FREE_EMAIL_PROVIDERS:
            return local_part in suspicious_roles

        return False

    def _calculate_confidence(self, is_suspicious: bool, num_reasons: int) -> float:
        """
        Calculate confidence score for sender analysis.

        Args:
            is_suspicious: Whether sender is flagged as suspicious
            num_reasons: Number of suspicious reasons found

        Returns:
            Confidence score (0.0 to 1.0)
        """
        if not is_suspicious:
            return 0.85  # High confidence in safe assessment

        # More reasons = higher confidence in suspicious assessment
        base_confidence = 0.60
        confidence_increase = min(num_reasons * 0.10, 0.30)

        return min(base_confidence + confidence_increase, 0.95)

    def analyze_multiple(self, emails: List[EmailData]) -> List[SenderExplanation]:
        """
        Analyze senders for multiple emails.

        Args:
            emails: List of emails to analyze

        Returns:
            List of sender explanations
        """
        return [self.analyze(email) for email in emails]
