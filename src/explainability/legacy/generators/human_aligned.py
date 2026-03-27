"""
Human-aligned explanation generator.

Orchestrates multiple explainers and component analyzers to produce
comprehensive explanations following cognitive processing order:
sender → subject → body → URLs → attachments.
"""

from typing import Optional, List, Dict, Any
import time

from src.explainability.legacy.generators.base_generator import BaseExplanationGenerator
from src.explainability.legacy.utils.data_structures import (
    EmailData,
    ModelOutput,
    Explanation,
    ExplanationType,
    SenderExplanation,
    SubjectExplanation,
    BodyExplanation,
    URLExplanation,
    AttachmentExplanation
)
from src.explainability.legacy.components.sender_analyzer import SenderAnalyzer
from src.explainability.legacy.components.subject_analyzer import SubjectAnalyzer
from src.explainability.legacy.components.body_analyzer import BodyAnalyzer
from src.explainability.legacy.components.url_analyzer import URLAnalyzer
from src.explainability.legacy.components.attachment_analyzer import AttachmentAnalyzer


class HumanAlignedGenerator(BaseExplanationGenerator):
    """
    Human-aligned explanation generator.

    Generates explanations that follow human cognitive processing order
    and provide actionable, non-technical guidance.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        use_feature_importance: bool = True,
        use_attention: bool = True,
        use_counterfactuals: bool = True,
        use_comparisons: bool = True
    ):
        """
        Initialize human-aligned generator.

        Args:
            config: Configuration dictionary
            use_feature_importance: Whether to include feature-based explanations
            use_attention: Whether to include attention-based explanations
            use_counterfactuals: Whether to include counterfactual explanations
            use_comparisons: Whether to include comparative explanations
        """
        super().__init__(config)

        self.use_feature_importance = use_feature_importance
        self.use_attention = use_attention
        self.use_counterfactuals = use_counterfactuals
        self.use_comparisons = use_comparisons

        # Initialize component analyzers (cognitive order)
        self.sender_analyzer = SenderAnalyzer()
        self.subject_analyzer = SubjectAnalyzer()
        self.body_analyzer = BodyAnalyzer()
        self.url_analyzer = URLAnalyzer()
        self.attachment_analyzer = AttachmentAnalyzer()

        # Initialize explainers (lazy loading)
        self._feature_explainer = None
        self._attention_explainer = None
        self._counterfactual_explainer = None
        self._comparative_explainer = None

        # Track which explanation types are supported
        self._update_supported_types()

    def generate_explanation(
        self,
        email: EmailData,
        model_prediction: ModelOutput,
        attention_weights: Optional[Any] = None,
        **kwargs
    ) -> Explanation:
        """
        Generate comprehensive explanation in cognitive order.

        Args:
            email: Email to explain
            model_prediction: Model prediction
            attention_weights: Optional pre-computed attention weights
            **kwargs: Additional parameters

        Returns:
            Complete Explanation object
        """
        # Validate input
        self.validate_input(email, model_prediction)

        # Create explanation object
        explanation = Explanation(
            email=email,
            model_prediction=model_prediction
        )

        # Component analysis (cognitive order: sender → subject → body → URLs → attachments)
        explanation.sender_explanation = self._analyze_sender(email)
        explanation.subject_explanation = self._analyze_subject(email)
        explanation.body_explanation = self._analyze_body(email)
        explanation.url_explanation = self._analyze_urls(email)
        explanation.attachment_explanation = self._analyze_attachments(email)

        # Advanced explanations (if enabled)
        if self.use_feature_importance:
            explanation.feature_importance = self._get_feature_importance(email, model_prediction)

        if self.use_attention and attention_weights is not None:
            explanation.attention_visualization = self._get_attention_visualization(email, attention_weights)

        if self.use_counterfactuals:
            explanation.counterfactuals = self._get_counterfactuals(email, model_prediction)

        if self.use_comparisons:
            explanation.comparative = self._get_comparative(email)

        # Track explanation types
        explanation.explanation_types = self._get_explanation_types()

        return explanation

    def _analyze_sender(self, email: EmailData) -> SenderExplanation:
        """Analyze email sender."""
        return self.sender_analyzer.analyze(email)

    def _analyze_subject(self, email: EmailData) -> SubjectExplanation:
        """Analyze email subject."""
        return self.subject_analyzer.analyze(email)

    def _analyze_body(self, email: EmailData) -> BodyExplanation:
        """Analyze email body."""
        return self.body_analyzer.analyze(email)

    def _analyze_urls(self, email: EmailData) -> URLExplanation:
        """Analyze email URLs."""
        return self.url_analyzer.analyze(email)

    def _analyze_attachments(self, email: EmailData) -> AttachmentExplanation:
        """Analyze email attachments."""
        return self.attachment_analyzer.analyze(email)

    def _get_feature_importance(
        self,
        email: EmailData,
        model_prediction: ModelOutput
    ):
        """Get feature-based explanation."""
        if self._feature_explainer is None:
            from src.explainability.legacy.explainers.feature_based import SimpleFeatureExplainer
            self._feature_explainer = SimpleFeatureExplainer()

        return self._feature_explainer.explain(email, model_prediction)

    def _get_attention_visualization(self, email: EmailData, attention_weights: Any):
        """Get attention-based explanation."""
        if self._attention_explainer is None:
            from src.explainability.legacy.explainers.attention_based import SimpleAttentionExplainer
            self._attention_explainer = SimpleAttentionExplainer()

        return self._attention_explainer.explain(email)

    def _get_counterfactuals(
        self,
        email: EmailData,
        model_prediction: ModelOutput
    ) -> List:
        """Get counterfactual explanations."""
        if self._counterfactual_explainer is None:
            from src.explainability.legacy.explainers.counterfactual import SimpleCounterfactualExplainer
            self._counterfactual_explainer = SimpleCounterfactualExplainer()

        return self._counterfactual_explainer.generate_counterfactuals(
            email,
            model_prediction,
            num_cf=3
        )

    def _get_comparative(self, email: EmailData):
        """Get comparative explanation."""
        if self._comparative_explainer is None:
            from src.explainability.legacy.explainers.comparative import SimpleComparativeExplainer
            self._comparative_explainer = SimpleComparativeExplainer()

        return self._comparative_explainer.explain(email)

    def _get_explanation_types(self) -> List[ExplanationType]:
        """Get list of explanation types being used."""
        types = []

        if self.use_feature_importance:
            types.append(ExplanationType.FEATURE_BASED)

        if self.use_attention:
            types.append(ExplanationType.ATTENTION_BASED)

        if self.use_counterfactuals:
            types.append(ExplanationType.COUNTERFACTUAL)

        if self.use_comparisons:
            types.append(ExplanationType.COMPARATIVE)

        return types

    def _update_supported_types(self):
        """Update supported explanation types."""
        self.supported_types = self._get_explanation_types()

    def get_supported_explanation_types(self) -> list:
        """Get list of supported explanation types."""
        return self._get_explanation_types()

    def generate_batch(
        self,
        emails: List[EmailData],
        model_predictions: List[ModelOutput]
    ) -> List[Explanation]:
        """
        Generate explanations for multiple emails.

        Args:
            emails: List of emails to explain
            model_predictions: List of corresponding predictions

        Returns:
            List of Explanation objects
        """
        if len(emails) != len(model_predictions):
            raise ValueError("Number of emails must match number of predictions")

        return [
            self.generate_explanation(email, pred)
            for email, pred in zip(emails, model_predictions)
        ]

    def set_legitimate_domains(self, domains: set):
        """Update legitimate domains for component analyzers."""
        self.sender_analyzer.legitimate_domains = domains
        self.url_analyzer.legitimate_domains = domains

    def set_strict_mode(self, strict: bool):
        """Update strict mode for all analyzers."""
        self.sender_analyzer.strict_mode = strict
        self.subject_analyzer.strict_mode = strict
        self.body_analyzer.strict_mode = strict
        self.url_analyzer.strict_mode = strict
        self.attachment_analyzer.strict_mode = strict
