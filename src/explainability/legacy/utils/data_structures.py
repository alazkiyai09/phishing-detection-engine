"""
Core data structures for the explanation system.

Defines all data types used across the system.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum


class EmailCategory(Enum):
    """Email classification categories."""
    SAFE = "safe"
    PHISHING = "phishing"
    SUSPICIOUS = "suspicious"


class ExplanationType(Enum):
    """Types of explanations."""
    FEATURE_BASED = "feature_based"
    ATTENTION_BASED = "attention_based"
    COUNTERFACTUAL = "counterfactual"
    COMPARATIVE = "comparative"


@dataclass
class EmailAddress:
    """Email address with display name."""
    display_name: Optional[str]
    email: str
    is_suspicious: bool = False
    suspicion_reasons: List[str] = field(default_factory=list)


@dataclass
class URL:
    """URL with safety information."""
    original: str
    domain: str
    path: Optional[str] = None
    has_https: bool = False
    domain_age_days: Optional[int] = None
    is_suspicious: bool = False
    suspicion_reasons: List[str] = field(default_factory=list)


@dataclass
class Attachment:
    """Email attachment with risk information."""
    filename: str
    file_type: str
    size_bytes: int
    has_macros: bool = False
    is_dangerous: bool = False
    risk_reasons: List[str] = field(default_factory=list)


@dataclass
class EmailData:
    """Complete email data for explanation."""
    # Basic email structure
    sender: EmailAddress
    recipients: List[EmailAddress]
    subject: str
    body: str
    urls: List[URL] = field(default_factory=list)
    attachments: List[Attachment] = field(default_factory=list)

    # Metadata
    timestamp: Optional[str] = None
    email_id: Optional[str] = None
    category: EmailCategory = EmailCategory.SAFE

    # Additional headers
    reply_to: Optional[EmailAddress] = None
    cc: List[EmailAddress] = field(default_factory=list)
    bcc: List[EmailAddress] = field(default_factory=list)

    def __post_init__(self):
        """Validate email data."""
        if not self.sender or not self.sender.email:
            raise ValueError("Email must have a valid sender")


@dataclass
class ModelOutput:
    """Model prediction output."""
    predicted_label: EmailCategory
    confidence: float  # 0.0 to 1.0
    probabilities: Dict[str, float] = field(default_factory=dict)  # Label -> prob

    # Optional: Model-specific information
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    logits: Optional[Any] = None

    def __post_init__(self):
        """Validate model output."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass
class SenderExplanation:
    """Explanation for sender analysis."""
    is_suspicious: bool
    confidence: float
    reasons: List[str] = field(default_factory=list)
    domain_reputation: Optional[str] = None  # "good", "unknown", "poor"
    display_name_mismatch: bool = False
    lookalike_domain: bool = False


@dataclass
class SubjectExplanation:
    """Explanation for subject analysis."""
    is_suspicious: bool
    confidence: float
    reasons: List[str] = field(default_factory=list)
    urgency_keywords: List[str] = field(default_factory=list)
    unusual_formatting: List[str] = field(default_factory=list)


@dataclass
class BodyExplanation:
    """Explanation for body analysis."""
    is_suspicious: bool
    confidence: float
    reasons: List[str] = field(default_factory=list)
    social_engineering_tactics: List[str] = field(default_factory=list)
    grammar_issues: List[str] = field(default_factory=list)
    pressure_language: List[str] = field(default_factory=list)


@dataclass
class URLExplanation:
    """Explanation for URL analysis."""
    is_suspicious: bool
    confidence: float
    reasons: List[str] = field(default_factory=list)
    suspicious_urls: List[Dict[str, Any]] = field(default_factory=list)
    safe_urls: List[str] = field(default_factory=list)


@dataclass
class AttachmentExplanation:
    """Explanation for attachment analysis."""
    is_suspicious: bool
    confidence: float
    reasons: List[str] = field(default_factory=list)
    dangerous_attachments: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class FeatureImportance:
    """Feature importance from SHAP or similar."""
    feature_names: List[str]
    importance_scores: List[float]
    top_features: List[tuple] = field(default_factory=list)  # (feature, score)

    def __post_init__(self):
        """Compute top features."""
        if not self.top_features:
            features = list(zip(self.feature_names, self.importance_scores))
            self.top_features = sorted(features, key=lambda x: abs(x[1]), reverse=True)


@dataclass
class AttentionVisualization:
    """Attention weights from transformer model."""
    tokens: List[str]
    attention_weights: List[List[float]]  # Layer x Head x Token x Token
    layer_indices: List[int] = field(default_factory=list)

    def get_top_attended_tokens(self, top_k: int = 5) -> List[tuple]:
        """Get top-k attended tokens."""
        # Average attention across layers and heads
        avg_attention = [
            sum(layer_weights[i] for layer_weights in self.attention_weights)
            for i in range(len(self.tokens))
        ]
        token_attention = list(zip(self.tokens, avg_attention))
        return sorted(token_attention, key=lambda x: x[1], reverse=True)[:top_k]


@dataclass
class CounterfactualExplanation:
    """Counterfactual explanation."""
    original_email: EmailData
    modified_email: EmailData
    changed_features: Dict[str, tuple]  # feature -> (original, new)
    predicted_label_change: tuple  # (original_label, new_label)
    confidence_change: tuple  # (original_conf, new_conf)

    def get_summary(self) -> str:
        """Get human-readable summary."""
        changes = []
        for feature, (old, new) in self.changed_features.items():
            changes.append(f"{feature}: '{old}' → '{new}'")

        return (
            f"If these changes were made: {', '.join(changes)},\n"
            f"the prediction would change from {self.predicted_label_change[0]} "
            f"to {self.predicted_label_change[1]} "
            f"(confidence: {self.confidence_change[0]:.2%} → {self.confidence_change[1]:.2%})"
        )


@dataclass
class ComparativeExplanation:
    """Comparative explanation against known campaigns."""
    similar_campaigns: List[Dict[str, Any]] = field(default_factory=list)
    similarity_scores: List[float] = field(default_factory=list)
    shared_characteristics: List[str] = field(default_factory=list)

    def get_summary(self) -> str:
        """Get human-readable summary."""
        if not self.similar_campaigns:
            return "This email does not match known phishing campaigns."

        top_campaign = self.similar_campaigns[0]
        return (
            f"This email is {self.similarity_scores[0]:.0%} similar to "
            f"known phishing campaign '{top_campaign.get('name', 'Unknown')}'. "
            f"Shared characteristics: {', '.join(self.shared_characteristics)}."
        )


@dataclass
class Explanation:
    """
    Complete explanation for an email prediction.

    Follows cognitive processing order: sender → subject → body → URLs → attachments.
    """
    # Core prediction
    email: EmailData
    model_prediction: ModelOutput

    # Component explanations (in cognitive order)
    sender_explanation: Optional[SenderExplanation] = None
    subject_explanation: Optional[SubjectExplanation] = None
    body_explanation: Optional[BodyExplanation] = None
    url_explanation: Optional[URLExplanation] = None
    attachment_explanation: Optional[AttachmentExplanation] = None

    # Advanced explanations
    feature_importance: Optional[FeatureImportance] = None
    attention_visualization: Optional[AttentionVisualization] = None
    counterfactuals: List[CounterfactualExplanation] = field(default_factory=list)
    comparative: Optional[ComparativeExplanation] = None

    # Metadata
    explanation_types: List[ExplanationType] = field(default_factory=list)
    generation_time_ms: float = 0.0
    is_federated: bool = False

    def get_summary(self) -> str:
        """Get brief summary of explanation."""
        suspicious_parts = []

        if self.sender_explanation and self.sender_explanation.is_suspicious:
            suspicious_parts.append("sender")
        if self.subject_explanation and self.subject_explanation.is_suspicious:
            suspicious_parts.append("subject line")
        if self.body_explanation and self.body_explanation.is_suspicious:
            suspicious_parts.append("body content")
        if self.url_explanation and self.url_explanation.is_suspicious:
            suspicious_parts.append("URLs")
        if self.attachment_explanation and self.attachment_explanation.is_suspicious:
            suspicious_parts.append("attachments")

        if not suspicious_parts:
            return f"Email appears safe (confidence: {self.model_prediction.confidence:.1%})"

        return (
            f"Email flagged as {self.model_prediction.predicted_label.value} "
            f"(confidence: {self.model_prediction.confidence:.1%}) "
            f"due to suspicious {', '.join(suspicious_parts)}."
        )
