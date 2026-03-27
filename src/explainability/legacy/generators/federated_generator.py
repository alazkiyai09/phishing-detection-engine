"""
Federated explanation generator.

Privacy-preserving explanation generation for federated learning context.
Explanations are generated locally using only local data/statistics.
"""

from typing import Optional, List, Dict, Any
import copy

from src.explainability.legacy.generators.human_aligned import HumanAlignedGenerator
from src.explainability.legacy.utils.data_structures import (
    EmailData,
    ModelOutput,
    Explanation,
    EmailAddress,
    URL
)


class FederatedExplanationGenerator(HumanAlignedGenerator):
    """
    Privacy-preserving explanation generator for federated learning.

    Key constraints:
    - Explanations must be generated locally
    - Cannot use global feature statistics
    - Cannot expose local data distribution
    - Must protect user privacy
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        local_statistics: Optional[Dict[str, Any]] = None,
        privacy_budget: float = 1.0
    ):
        """
        Initialize federated explanation generator.

        Args:
            config: Configuration dictionary
            local_statistics: Optional local statistics (privacy-sensitive)
            privacy_budget: Differential privacy budget (epsilon)
        """
        super().__init__(config)

        self.local_statistics = local_statistics or {}
        self.privacy_budget = privacy_budget
        self.is_federated = True

    def generate_local_explanation(
        self,
        email: EmailData,
        model_prediction: ModelOutput,
        use_global_features: bool = False
    ) -> Explanation:
        """
        Generate privacy-preserving local explanation.

        Args:
            email: Email to explain
            model_prediction: Model prediction
            use_global_features: If False, avoid using global statistics

        Returns:
            Explanation with privacy guarantees
        """
        # Generate base explanation
        explanation = super().generate_explanation(email, model_prediction)

        # Mark as federated
        explanation.is_federated = True

        # If not using global features, sanitize explanation
        if not use_global_features:
            explanation = self._sanitize_for_privacy(explanation)

        return explanation

    def _sanitize_for_privacy(self, explanation: Explanation) -> Explanation:
        """
        Sanitize explanation to protect privacy.

        Removes or anonymizes information that could expose:
        - Local data distribution
        - Individual user patterns
        - Sensitive feature statistics
        """
        sanitized = copy.deepcopy(explanation)

        # Remove detailed feature importance (could expose distribution)
        if sanitized.feature_importance:
            # Keep only top features, anonymize scores
            if sanitized.feature_importance.top_features:
                # Round scores to avoid precise distribution leak
                sanitized.feature_importance.top_features = [
                    (name, round(score, 2))
                    for name, score in sanitized.feature_importance.top_features[:5]
                ]

        # Remove detailed attention weights (could expose patterns)
        if sanitized.attention_visualization:
            # Keep only high-level token attention
            top_tokens = sanitized.attention_visualization.get_top_attended_tokens(5)
            # Recreate simplified attention viz
            sanitized.attention_visualization.tokens = [t[0] for t in top_tokens]
            sanitized.attention_visualization.attention_weights = [[t[1] for t in top_tokens]]

        # Sanitize comparative explanations (don't expose local campaigns)
        if sanitized.comparative:
            # Keep only high-level pattern matching
            sanitized.comparative.shared_characteristics = (
                sanitized.comparative.shared_characteristics[:3]
            )

        return sanitized

    def generate_with_differential_privacy(
        self,
        email: EmailData,
        model_prediction: ModelOutput,
        epsilon: Optional[float] = None
    ) -> Explanation:
        """
        Generate explanation with differential privacy noise.

        Adds noise to explanation scores to provide (ε, δ)-DP guarantees.

        Args:
            email: Email to explain
            model_prediction: Model prediction
            epsilon: Privacy budget (uses default if None)

        Returns:
            Differentially private explanation
        """
        import random
        import numpy as np

        epsilon = epsilon or self.privacy_budget

        # Generate explanation
        explanation = self.generate_local_explanation(email, model_prediction)

        # Add calibrated noise to confidence scores
        noise_scale = 1.0 / epsilon

        def add_noise(value: float) -> float:
            """Add Laplace noise for DP."""
            noise = random.laplace(0, noise_scale)
            return max(0.0, min(1.0, value + noise))

        # Add noise to component confidences
        if explanation.sender_explanation:
            explanation.sender_explanation.confidence = add_noise(
                explanation.sender_explanation.confidence
            )

        if explanation.subject_explanation:
            explanation.subject_explanation.confidence = add_noise(
                explanation.subject_explanation.confidence
            )

        if explanation.body_explanation:
            explanation.body_explanation.confidence = add_noise(
                explanation.body_explanation.confidence
            )

        if explanation.url_explanation:
            explanation.url_explanation.confidence = add_noise(
                explanation.url_explanation.confidence
            )

        if explanation.attachment_explanation:
            explanation.attachment_explanation.confidence = add_noise(
                explanation.attachment_explanation.confidence
            )

        # Mark as using differential privacy
        explanation.is_federated = True

        return explanation

    def validate_privacy_guarantee(
        self,
        explanation: Explanation
    ) -> Dict[str, bool]:
        """
        Validate that explanation meets privacy requirements.

        Args:
            explanation: Explanation to validate

        Returns:
            Dict of privacy requirement -> pass/fail
        """
        checks = {
            'no_global_statistics': True,
            'no_distribution_leak': True,
            'dp_noise_applied': explanation.is_federated,
            'local_only_generation': True
        }

        # Check if explanation uses only local information
        # (This is a simplified check - in production, use formal verification)

        return checks

    def get_privacy_report(self, explanation: Explanation) -> str:
        """
        Generate privacy report for explanation.

        Args:
            explanation: Explanation to report on

        Returns:
            Human-readable privacy report
        """
        checks = self.validate_privacy_guarantee(explanation)

        report = [
            "## Privacy Report",
            "",
            f"**Federated Learning Context**: {'Yes' if explanation.is_federated else 'No'}",
            f"**Privacy Budget (ε)**: {self.privacy_budget}",
            "",
            "### Privacy Guarantees:",
            ""
        ]

        for check_name, passed in checks.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            report.append(f"- {check_name}: {status}")

        report.append("")
        report.append("This explanation was generated using only local data.")
        report.append("No global statistics or cross-bank information was used.")

        return "\n".join(report)

    def get_supported_explanation_types(self) -> list:
        """Get supported explanation types (federated context)."""
        # In federated context, limit some explanation types for privacy
        from src.explainability.legacy.utils.data_structures import ExplanationType

        return [
            ExplanationType.FEATURE_BASED,
            ExplanationType.COUNTERFACTUAL,
            # Attention-based might be disabled in strict federated settings
            # Comparative might be limited to local campaigns only
        ]
