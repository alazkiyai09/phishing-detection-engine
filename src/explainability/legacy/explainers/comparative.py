"""
Comparative explainer.

Compares email against known phishing campaigns to identify
similarities and patterns.
"""

from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher
import re

from src.explainability.legacy.utils.data_structures import (
    EmailData,
    ComparativeExplanation
)


class ComparativeExplainer:
    """
    Comparative explainer for phishing detection.

    Compares emails against known phishing campaigns.
    """

    def __init__(
        self,
        known_campaigns: Optional[List[Dict[str, Any]]] = None,
        similarity_threshold: float = 0.3
    ):
        """
        Initialize comparative explainer.

        Args:
            known_campaigns: List of known phishing campaigns
            similarity_threshold: Minimum similarity to consider as match
        """
        self.known_campaigns = known_campaigns or self._get_default_campaigns()
        self.similarity_threshold = similarity_threshold

    def _get_default_campaigns(self) -> List[Dict[str, Any]]:
        """Get default known phishing campaigns."""
        return [
            {
                'name': 'Netflix Account Verification',
                'characteristics': ['verify account', 'netflix', 'update payment', 'click link'],
                'target_brands': ['netflix', 'netflix.com'],
                'typical_sender_patterns': ['@.*\.com$', 'support@', 'netflix@'],
                'urls_pattern': 'netflix.*login|verify.*account'
            },
            {
                'name': 'Bank of America Security Alert',
                'characteristics': ['unusual activity', 'security alert', 'bank of america', 'verify identity'],
                'target_brands': ['bank of america', 'bankofamerica.com'],
                'typical_sender_patterns': ['@.*\.com$', 'security@', 'alert@'],
                'urls_pattern': 'bankofamerica.*login|secure.*sign'
            },
            {
                'name': 'IRS Tax Refund Scam',
                'characteristics': ['tax refund', 'irs', 'direct deposit', 'verify information'],
                'target_brands': ['irs', 'irs.gov'],
                'typical_sender_patterns': ['@.*\.gov$', 'irs@', 'refund@'],
                'urls_pattern': 'irs.*refund|tax.*direct'
            },
            {
                'name': 'Microsoft Office 365 Expiring',
                'characteristics': ['office 365', 'expiring', 'renew subscription', 'payment due'],
                'target_brands': ['microsoft', 'microsoft.com', 'office 365'],
                'typical_sender_patterns': ['@.*\.com$', 'microsoft@', 'noreply@'],
                'urls_pattern': 'microsoft.*renew|office.*subscription'
            },
            {
                'name': 'Amazon Order Confirmation',
                'characteristics': ['order confirmation', 'amazon', 'cannot ship', 'verify address'],
                'target_brands': ['amazon', 'amazon.com'],
                'typical_sender_patterns': ['@.*\.com$', 'amazon@', 'auto-confirm@'],
                'urls_pattern': 'amazon.*order|shipment.*verify'
            },
            {
                'name': 'PayPal Account Limited',
                'characteristics': ['account limited', 'paypal', 'restore access', 'verify information'],
                'target_brands': ['paypal', 'paypal.com'],
                'typical_sender_patterns': ['@.*\.com$', 'paypal@', 'service@'],
                'urls_pattern': 'paypal.*limited|account.*restore'
            },
            {
                'name': 'CEO Fraud Business Email Compromise',
                'characteristics': ['urgent transfer', 'wire transfer', 'confidential', 'ceo request'],
                'target_brands': [],  # No specific brand
                'typical_sender_patterns': ['ceo@', 'executive@', 'president@'],
                'urls_pattern': 'wire.*transfer|urgent.*payment'
            },
            {
                'name': 'LinkedIn Job Offer Scam',
                'characteristics': ['job offer', 'linkedin', 'apply now', 'high salary'],
                'target_brands': ['linkedin', 'linkedin.com'],
                'typical_sender_patterns': ['@.*\.com$', 'recruiting@', 'jobs@', 'linkedin@'],
                'urls_pattern': 'linkedin.*apply|job.*offer'
            }
        ]

    def explain(self, email: EmailData) -> ComparativeExplanation:
        """
        Compare email against known campaigns.

        Args:
            email: Email to analyze

        Returns:
            ComparativeExplanation with campaign matches
        """
        similarities = []
        shared_characteristics = []

        for campaign in self.known_campaigns:
            similarity_score = self._compute_similarity(email, campaign)

            if similarity_score >= self.similarity_threshold:
                similarities.append({
                    'name': campaign['name'],
                    'similarity': similarity_score,
                    'characteristics': campaign['characteristics']
                })

        # Sort by similarity
        similarities.sort(key=lambda x: x['similarity'], reverse=True)

        # Extract shared characteristics
        if similarities:
            top_campaign = similarities[0]
            shared_characteristics = top_campaign['characteristics']

        return ComparativeExplanation(
            similar_campaigns=[s['name'] for s in similarities],
            similarity_scores=[s['similarity'] for s in similarities],
            shared_characteristics=shared_characteristics
        )

    def _compute_similarity(self, email: EmailData, campaign: Dict[str, Any]) -> float:
        """
        Compute similarity between email and campaign.

        Args:
            email: Email to analyze
            campaign: Campaign information

        Returns:
            Similarity score (0.0 to 1.0)
        """
        similarity = 0.0

        # Check sender domain against brands
        sender_domain = email.sender.email.split('@')[1].lower()
        for brand in campaign.get('target_brands', []):
            if brand.lower() in sender_domain:
                similarity += 0.2

        # Check subject for campaign characteristics
        subject_lower = email.subject.lower()
        for char in campaign.get('characteristics', []):
            if char.lower() in subject_lower:
                similarity += 0.15

        # Check body for campaign characteristics
        body_lower = email.body.lower()
        body_matches = sum(
            1 for char in campaign.get('characteristics', [])
            if char.lower() in body_lower
        )
        similarity += min(body_matches * 0.10, 0.30)

        # Check URLs for campaign patterns
        if email.urls:
            for url_obj in email.urls:
                url_lower = url_obj.original.lower()
                urls_pattern = campaign.get('urls_pattern', '')
                if urls_pattern and re.search(urls_pattern, url_lower):
                    similarity += 0.15

        return min(similarity, 1.0)

    def get_most_similar_campaign(
        self,
        email: EmailData
    ) -> Optional[Dict[str, Any]]:
        """
        Get the most similar campaign for email.

        Args:
            email: Email to analyze

        Returns:
            Most similar campaign dict or None
        """
        explanation = self.explain(email)

        if explanation.similar_campaigns:
            best_idx = explanation.similarity_scores.index(max(explanation.similarity_scores))

            return {
                'name': explanation.similar_campaigns[best_idx],
                'similarity': explanation.similarity_scores[best_idx],
                'shared_characteristics': explanation.shared_characteristics
            }

        return None

    def explain_multiple(
        self,
        emails: List[EmailData]
    ) -> List[ComparativeExplanation]:
        """
        Compare multiple emails against known campaigns.

        Args:
            emails: List of emails to analyze

        Returns:
            List of ComparativeExplanation objects
        """
        return [self.explain(email) for email in emails]


class SimpleComparativeExplainer:
    """
    Simplified comparative explainer with basic pattern matching.
    """

    def __init__(self):
        """Initialize simple comparative explainer."""
        # Simple suspicious patterns
        self.suspicious_patterns = {
            'account_verification': ['verify', 'confirm', 'account suspended', 'unusual activity'],
            'payment_scams': ['invoice', 'wire transfer', 'payment due', 'immediate payment'],
            'urgency_tactics': ['urgent', 'immediately', 'expires soon', 'act now'],
            'brand_impersonation': ['netflix', 'paypal', 'amazon', 'bank of america', 'irs']
        }

    def explain(self, email: EmailData) -> ComparativeExplanation:
        """
        Compare email against simple patterns.

        Args:
            email: Email to analyze

        Returns:
            ComparativeExplanation with pattern matches
        """
        text = f"{email.subject} {email.body}".lower()

        matched_patterns = []
        similarity_scores = []

        for pattern_name, keywords in self.suspicious_patterns.items():
            matches = sum(1 for kw in keywords if kw in text)

            if matches > 0:
                matched_patterns.append(f"{pattern_name}: {matches} matches")
                similarity_scores.append(min(matches * 0.2, 0.8))

        if not similarity_scores:
            similarity_scores = [0.0]

        return ComparativeExplanation(
            similar_campaigns=matched_patterns or ['No known pattern'],
            similarity_scores=similarity_scores,
            shared_characteristics=[]
        )
