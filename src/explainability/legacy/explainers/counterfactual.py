"""
Counterfactual explainer.

Generates "what-if" scenarios showing minimal changes needed
to change the prediction.
"""

from typing import List, Optional, Dict, Any
import copy
import re

from src.explainability.legacy.utils.data_structures import (
    EmailData,
    ModelOutput,
    EmailAddress,
    CounterfactualExplanation,
    EmailCategory
)


class CounterfactualExplainer:
    """
    Counterfactual explainer for phishing detection.

    Shows what minimal changes would make an email safe (or phishing).
    """

    def __init__(
        self,
        model: Any,
        max_counterfactuals: int = 3,
        feature_importance: Optional[Dict[str, float]] = None
    ):
        """
        Initialize counterfactual explainer.

        Args:
            model: Model to query for predictions
            max_counterfactuals: Maximum number of counterfactuals to generate
            feature_importance: Optional feature importance to guide search
        """
        self.model = model
        self.max_counterfactuals = max_counterfactuals
        self.feature_importance = feature_importance or {}

    def generate_counterfactuals(
        self,
        email: EmailData,
        original_prediction: ModelOutput,
        num_cf: Optional[int] = None
    ) -> List[CounterfactualExplanation]:
        """
        Generate counterfactual examples.

        Args:
            email: Original email
            original_prediction: Original model prediction
            num_cf: Number of counterfactuals to generate

        Returns:
            List of counterfactual explanations
        """
        num_cf = num_cf or self.max_counterfactuals
        counterfactuals = []

        # Strategy 1: Fix suspicious sender
        if email.sender.is_suspicious:
            cf = self._fix_sender(email, original_prediction)
            if cf:
                counterfactuals.append(cf)

        # Strategy 2: Remove urgency from subject
        if any(word in email.subject.lower() for word in ['urgent', 'immediate', 'asap']):
            cf = self._fix_subject_urgency(email, original_prediction)
            if cf:
                counterfactuals.append(cf)

        # Strategy 3: Remove suspicious URLs
        suspicious_urls = [url for url in email.urls if url.is_suspicious]
        if suspicious_urls:
            cf = self._fix_urls(email, original_prediction)
            if cf:
                counterfactuals.append(cf)

        # Strategy 4: Remove dangerous attachments
        dangerous_atts = [att for att in email.attachments if att.is_dangerous]
        if dangerous_atts:
            cf = self._fix_attachments(email, original_prediction)
            if cf:
                counterfactuals.append(cf)

        # Strategy 5: Remove sensitive requests from body
        if any(word in email.body.lower() for word in ['password', 'verify', 'confirm']):
            cf = self._fix_body_sensitive(email, original_prediction)
            if cf:
                counterfactuals.append(cf)

        return counterfactuals[:num_cf]

    def _fix_sender(
        self,
        original_email: EmailData,
        original_prediction: ModelOutput
    ) -> Optional[CounterfactualExplanation]:
        """Generate counterfactual with fixed sender."""
        modified_email = copy.deepcopy(original_email)

        # Fix sender
        original_sender_email = modified_email.sender.email
        modified_email.sender.email = "legitimate@example.com"
        modified_email.sender.is_suspicious = False

        # Get new prediction
        new_prediction = self._predict(modified_email)

        return CounterfactualExplanation(
            original_email=original_email,
            modified_email=modified_email,
            changed_features={
                'sender_email': (original_sender_email, modified_email.sender.email)
            },
            predicted_label_change=(
                original_prediction.predicted_label,
                new_prediction.predicted_label
            ),
            confidence_change=(
                original_prediction.confidence,
                new_prediction.confidence
            )
        )

    def _fix_subject_urgency(
        self,
        original_email: EmailData,
        original_prediction: ModelOutput
    ) -> Optional[CounterfactualExplanation]:
        """Generate counterfactual with urgency removed from subject."""
        modified_email = copy.deepcopy(original_email)

        # Remove urgency words
        original_subject = modified_email.subject
        modified_email.subject = self._remove_urgency_words(original_subject)

        # Get new prediction
        new_prediction = self._predict(modified_email)

        return CounterfactualExplanation(
            original_email=original_email,
            modified_email=modified_email,
            changed_features={
                'subject': (original_subject, modified_email.subject)
            },
            predicted_label_change=(
                original_prediction.predicted_label,
                new_prediction.predicted_label
            ),
            confidence_change=(
                original_prediction.confidence,
                new_prediction.confidence
            )
        )

    def _fix_urls(
        self,
        original_email: EmailData,
        original_prediction: ModelOutput
    ) -> Optional[CounterfactualExplanation]:
        """Generate counterfactual with URLs fixed."""
        modified_email = copy.deepcopy(original_email)

        # Remove suspicious URLs
        original_urls = [url.original for url in modified_email.urls]
        modified_email.urls = [url for url in modified_email.urls if not url.is_suspicious]

        # Get new prediction
        new_prediction = self._predict(modified_email)

        return CounterfactualExplanation(
            original_email=original_email,
            modified_email=modified_email,
            changed_features={
                'urls': (len(original_urls), len(modified_email.urls))
            },
            predicted_label_change=(
                original_prediction.predicted_label,
                new_prediction.predicted_label
            ),
            confidence_change=(
                original_prediction.confidence,
                new_prediction.confidence
            )
        )

    def _fix_attachments(
        self,
        original_email: EmailData,
        original_prediction: ModelOutput
    ) -> Optional[CounterfactualExplanation]:
        """Generate counterfactual with attachments removed."""
        modified_email = copy.deepcopy(original_email)

        # Remove dangerous attachments
        original_atts = [att.filename for att in modified_email.attachments]
        modified_email.attachments = [
            att for att in modified_email.attachments
            if not att.is_dangerous
        ]

        # Get new prediction
        new_prediction = self._predict(modified_email)

        return CounterfactualExplanation(
            original_email=original_email,
            modified_email=modified_email,
            changed_features={
                'attachments': (len(original_atts), len(modified_email.attachments))
            },
            predicted_label_change=(
                original_prediction.predicted_label,
                new_prediction.predicted_label
            ),
            confidence_change=(
                original_prediction.confidence,
                new_prediction.confidence
            )
        )

    def _fix_body_sensitive(
        self,
        original_email: EmailData,
        original_prediction: ModelOutput
    ) -> Optional[CounterfactualExplanation]:
        """Generate counterfactual with sensitive requests removed."""
        modified_email = copy.deepcopy(original_email)

        # Remove sensitive request phrases
        original_body = modified_email.body
        modified_email.body = self._remove_sensitive_phrases(original_body)

        # Get new prediction
        new_prediction = self._predict(modified_email)

        return CounterfactualExplanation(
            original_email=original_email,
            modified_email=modified_email,
            changed_features={
                'body_content': ('original', 'sanitized')
            },
            predicted_label_change=(
                original_prediction.predicted_label,
                new_prediction.predicted_label
            ),
            confidence_change=(
                original_prediction.confidence,
                new_prediction.confidence
            )
        )

    def _remove_urgency_words(self, text: str) -> str:
        """Remove urgency words from text."""
        urgency_words = ['urgent', 'immediately', 'asap', 'right away', 'hurry']

        result = text
        for word in urgency_words:
            result = re.sub(word, '', result, flags=re.IGNORECASE)

        return result.strip()

    def _remove_sensitive_phrases(self, text: str) -> str:
        """Remove sensitive request phrases from text."""
        sensitive_phrases = [
            'enter your password',
            'verify your account',
            'confirm your identity',
            'provide your'
        ]

        result = text
        for phrase in sensitive_phrases:
            result = re.sub(phrase, '[REMOVED]', result, flags=re.IGNORECASE)

        return result

    def _predict(self, email: EmailData) -> ModelOutput:
        """
        Get model prediction for email.

        In production, this would call the actual model.
        For now, returns simulated prediction.
        """
        if self.model is not None:
            try:
                # Try to call model
                prediction = self.model.predict(email)
                if isinstance(prediction, ModelOutput):
                    return prediction
            except Exception:
                pass

        # Fallback: heuristic prediction
        is_phishing = (
            email.sender.is_suspicious or
            any(url.is_suspicious for url in email.urls) or
            any(att.is_dangerous for att in email.attachments)
        )

        confidence = 0.85 if is_phishing else 0.15

        return ModelOutput(
            predicted_label=EmailCategory.PHISHING if is_phishing else EmailCategory.SAFE,
            confidence=confidence
        )


class SimpleCounterfactualExplainer:
    """
    Simplified counterfactual explainer that doesn't require a model.

    Generates hypothetical counterfactuals based on heuristics.
    """

    def generate_counterfactuals(
        self,
        email: EmailData,
        original_prediction: ModelOutput,
        num_cf: int = 3
    ) -> List[CounterfactualExplanation]:
        """
        Generate hypothetical counterfactuals.

        Args:
            email: Original email
            original_prediction: Original prediction
            num_cf: Number of counterfactuals

        Returns:
            List of counterfactual explanations
        """
        counterfactuals = []

        # Simulate counterfactuals
        changes = [
            ("sender_email", email.sender.email, "legitimate@bank.com"),
            ("subject", email.subject, "Your Monthly Statement"),
            ("URLs", len(email.urls), 0),
        ]

        for i, (feature, old_val, new_val) in enumerate(changes[:num_cf]):
            # Simulate prediction change
            new_label = EmailCategory.SAFE
            new_confidence = 0.15

            cf = CounterfactualExplanation(
                original_email=email,
                modified_email=email,  # Same email for simplicity
                changed_features={feature: (old_val, new_val)},
                predicted_label_change=(original_prediction.predicted_label, new_label),
                confidence_change=(original_prediction.confidence, new_confidence)
            )
            counterfactuals.append(cf)

        return counterfactuals
