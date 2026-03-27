"""
Feature-based explainer using SHAP values.

Computes feature importance for email classification predictions.
"""

from typing import Optional, List, Dict, Any
import numpy as np

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from src.explainability.legacy.utils.data_structures import (
    EmailData,
    ModelOutput,
    FeatureImportance
)


class FeatureBasedExplainer:
    """
    Feature-based explainer using SHAP (SHapley Additive exPlanations).

    SHAP values show how much each feature contributed to the prediction.
    """

    def __init__(
        self,
        model: Any,
        background_data: Optional[np.ndarray] = None,
        use_kernel_shap: bool = True
    ):
        """
        Initialize feature-based explainer.

        Args:
            model: The model to explain
            background_data: Background dataset for SHAP
            use_kernel_shap: If True, use KernelSHAP (slower but model-agnostic)
        """
        self.model = model
        self.background_data = background_data
        self.use_kernel_shap = use_kernel_shap

        if not SHAP_AVAILABLE:
            raise ImportError(
                "SHAP is not installed. Install with: pip install shap"
            )

        self.explainer = None
        self._initialize_explainer()

    def _initialize_explainer(self):
        """Initialize SHAP explainer."""
        if self.background_data is not None:
            if self.use_kernel_shap:
                # KernelSHAP - model-agnostic but slower
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba,
                    self.background_data
                )
            else:
                # TreeExplainer - faster for tree models
                try:
                    self.explainer = shap.TreeExplainer(self.model)
                except Exception:
                    # Fall back to KernelSHAP
                    self.explainer = shap.KernelExplainer(
                        self.model.predict_proba,
                        self.background_data
                    )

    def explain(
        self,
        email: EmailData,
        feature_names: Optional[List[str]] = None
    ) -> FeatureImportance:
        """
        Compute SHAP values for email features.

        Args:
            email: Email to explain
            feature_names: List of feature names

        Returns:
            FeatureImportance with SHAP values
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Provide background_data.")

        # Extract features from email
        features = self._extract_features(email)

        # Compute SHAP values
        shap_values = self.explainer.shap_values(features.reshape(1, -1))

        # Handle binary classification (SHAP returns array)
        if isinstance(shap_values, list):
            # Binary classification - use positive class
            shap_values = shap_values[1]

        # Flatten if needed
        shap_values = np.array(shap_values).flatten()

        # Default feature names
        if feature_names is None:
            feature_names = self._get_default_feature_names(len(shap_values))

        return FeatureImportance(
            feature_names=feature_names,
            importance_scores=shap_values.tolist()
        )

    def _extract_features(self, email: EmailData) -> np.ndarray:
        """
        Extract feature vector from email.

        Args:
            email: Email data

        Returns:
            Feature vector as numpy array
        """
        features = []

        # Sender features
        features.append(1 if email.sender.is_suspicious else 0)
        features.append(len(email.sender.email))
        features.append(1 if '@' in email.sender.email else 0)

        # Subject features
        features.append(len(email.subject))
        features.append(email.subject.count('!'))
        features.append(1 if email.subject.isupper() else 0)

        # Body features
        features.append(len(email.body))
        features.append(email.body.count('!'))
        features.append(email.body.count('?'))

        # URL features
        features.append(len(email.urls))
        features.append(sum(1 for url in email.urls if not url.has_https))

        # Attachment features
        features.append(len(email.attachments))
        features.append(sum(1 for att in email.attachments if att.is_dangerous))

        return np.array(features, dtype=np.float32)

    def _get_default_feature_names(self, n_features: int) -> List[str]:
        """Get default feature names."""
        names = [
            'sender_suspicious', 'sender_email_length', 'has_at_symbol',
            'subject_length', 'subject_exclamations', 'subject_all_caps',
            'body_length', 'body_exclamations', 'body_questions',
            'num_urls', 'num_http_urls',
            'num_attachments', 'num_dangerous_attachments'
        ]

        # Extend if needed
        while len(names) < n_features:
            names.append(f'feature_{len(names)}')

        return names[:n_features]

    def explain_multiple(
        self,
        emails: List[EmailData],
        feature_names: Optional[List[str]] = None
    ) -> List[FeatureImportance]:
        """
        Compute SHAP values for multiple emails.

        Args:
            emails: List of emails to explain
            feature_names: List of feature names

        Returns:
            List of FeatureImportance objects
        """
        return [self.explain(email, feature_names) for email in emails]


# Alternative: Simplified feature explainer without SHAP dependency
class SimpleFeatureExplainer:
    """
    Simplified feature explainer that doesn't require SHAP.

    Uses heuristic-based feature importance instead of SHAP values.
    """

    def __init__(self, model: Optional[Any] = None):
        """
        Initialize simple feature explainer.

        Args:
            model: Model (optional, for heuristic weighting)
        """
        self.model = model

    def explain(
        self,
        email: EmailData,
        model_prediction: Optional[ModelOutput] = None
    ) -> FeatureImportance:
        """
        Compute heuristic feature importance.

        Args:
            email: Email to explain
            model_prediction: Model prediction (optional)

        Returns:
            FeatureImportance with heuristic scores
        """
        feature_names = []
        importance_scores = []

        # Sender features
        if email.sender.is_suspicious:
            feature_names.append("suspicious_sender")
            importance_scores.append(0.8)

        if email.sender.lookalike_domain:
            feature_names.append("lookalike_domain")
            importance_scores.append(0.9)

        # Subject features
        if any(word in email.subject.lower() for word in ['urgent', 'immediate', 'asap']):
            feature_names.append("urgency_in_subject")
            importance_scores.append(0.6)

        if email.subject.count('!') > 2:
            feature_names.append("excessive_punctuation_subject")
            importance_scores.append(0.5)

        # Body features
        if any(word in email.body.lower() for word in ['password', 'verify', 'confirm']):
            feature_names.append("sensitive_request_body")
            importance_scores.append(0.85)

        if email.body.count('!') > 5:
            feature_names.append("excessive_punctuation_body")
            importance_scores.append(0.4)

        # URL features
        suspicious_urls = sum(1 for url in email.urls if url.is_suspicious)
        if suspicious_urls > 0:
            feature_names.append(f"suspicious_urls_{suspicious_urls}")
            importance_scores.append(0.95 * suspicious_urls)

        # Attachment features
        dangerous_attachments = sum(1 for att in email.attachments if att.is_dangerous)
        if dangerous_attachments > 0:
            feature_names.append(f"dangerous_attachments_{dangerous_attachments}")
            importance_scores.append(0.90 * dangerous_attachments)

        # If no features, add default
        if not feature_names:
            feature_names = ["no_suspicious_features"]
            importance_scores = [0.0]

        return FeatureImportance(
            feature_names=feature_names,
            importance_scores=importance_scores
        )
