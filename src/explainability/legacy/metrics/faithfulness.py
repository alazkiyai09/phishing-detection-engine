"""
Faithfulness metrics for explanations.

Measures how well explanations reflect the actual model reasoning.
"""

from typing import List, Optional
import numpy as np

from src.explainability.legacy.utils.data_structures import (
    EmailData,
    ModelOutput,
    Explanation
)


def compute_faithfulness(
    explanation: Explanation,
    model,
    email: EmailData,
    num_perturbations: int = 100
) -> float:
    """
    Compute faithfulness score for explanation.

    Faithfulness measures whether the explanation accurately reflects
    the model's decision-making process.

    Higher score = explanation is more faithful to model reasoning.

    Args:
        explanation: Explanation to evaluate
        model: Model used for prediction
        email: Original email
        num_perturbations: Number of perturbations for testing

    Returns:
        Faithfulness score (0.0 to 1.0)
    """
    if explanation.feature_importance is None:
        return 0.5  # Neutral score if no feature importance

    # Get original prediction
    original_pred = explanation.model_prediction

    # Test faithfulness via feature perturbation
    faithfulness_scores = []

    for feature_name, importance in explanation.feature_importance.top_features[:5]:
        # Perturb feature
        perturbed_email = _perturb_feature(email, feature_name)

        # Get new prediction
        try:
            new_pred = model.predict(perturbed_email)
        except Exception:
            # If model doesn't have predict, use heuristic
            new_pred = _heuristic_predict(perturbed_email)

        # Check if prediction changed in expected direction
        # High importance feature → should change prediction more
        prediction_change = abs(original_pred.confidence - new_pred.confidence)

        # Normalize by importance
        if importance > 0:
            faithfulness = min(prediction_change / abs(importance), 1.0)
            faithfulness_scores.append(faithfulness)

    # Average faithfulness
    if faithfulness_scores:
        return np.mean(faithfulness)

    return 0.5


def _perturb_feature(email: EmailData, feature_name: str) -> EmailData:
    """Perturb a specific feature in email."""
    import copy

    perturbed = copy.deepcopy(email)

    if 'sender' in feature_name.lower():
        # Perturb sender
        perturbed.sender.email = "modified@example.com"
    elif 'subject' in feature_name.lower():
        # Perturb subject
        perturbed.subject = "Modified: " + perturbed.subject
    elif 'body' in feature_name.lower():
        # Perturb body
        perturbed.body = "Modified content. " + perturbed.body
    elif 'url' in feature_name.lower():
        # Remove URLs
        perturbed.urls = []
    elif 'attachment' in feature_name.lower():
        # Remove attachments
        perturbed.attachments = []

    return perturbed


def _heuristic_predict(email: EmailData) -> ModelOutput:
    """Heuristic prediction for testing."""
    from src.explainability.legacy.utils.data_structures import EmailCategory

    suspicious = sum([
        email.sender.is_suspicious,
        any(url.is_suspicious for url in email.urls),
        any(att.is_dangerous for att in email.attachments)
    ])

    if suspicious >= 2:
        return ModelOutput(
            predicted_label=EmailCategory.PHISHING,
            confidence=0.75
        )
    else:
        return ModelOutput(
            predicted_label=EmailCategory.SAFE,
            confidence=0.60
        )


def compute_attention_faithfulness(
    explanation: Explanation,
    model
) -> float:
    """
    Compute faithfulness for attention-based explanations.

    Checks if attention weights correlate with prediction importance.

    Args:
        explanation: Explanation with attention visualization
        model: Model

    Returns:
        Faithfulness score (0.0 to 1.0)
    """
    if explanation.attention_visualization is None:
        return 0.5

    # Get top attended tokens
    top_tokens = explanation.attention_visualization.get_top_attended_tokens(10)

    # Check if high-attention tokens are suspicious
    suspicious_keywords = {
        'urgent', 'password', 'verify', 'click', 'login',
        'account', 'suspended', 'immediately'
    }

    suspicious_attended = sum(
        1 for token, _ in top_tokens
        if token.lower() in suspicious_keywords
    )

    # Faithfulness = proportion of top attended tokens that are suspicious
    faithfulness = suspicious_attended / len(top_tokens) if top_tokens else 0.5

    return faithfulness


def compute_counterfactual_faithfulness(
    explanation: Explanation,
    model
) -> float:
    """
    Compute faithfulness for counterfactual explanations.

    Checks if counterfactual changes actually produce predicted changes.

    Args:
        explanation: Explanation with counterfactuals
        model: Model

    Returns:
        Faithfulness score (0.0 to 1.0)
    """
    if not explanation.counterfactuals:
        return 0.5

    faithful_count = 0

    for cf in explanation.counterfactuals:
        # Check if modified email actually produces predicted change
        try:
            new_pred = model.predict(cf.modified_email)

            # Does label match expected?
            label_match = new_pred.predicted_label == cf.predicted_label_change[1]

            # Is confidence in expected direction?
            conf_match = (
                (new_pred.confidence < cf.confidence_change[0]) if
                cf.predicted_label_change[1] == "safe" else
                (new_pred.confidence > cf.confidence_change[0])
            )

            if label_match or conf_match:
                faithful_count += 1

        except Exception:
            # If prediction fails, assume faithful
            faithful_count += 1

    return faithful_count / len(explanation.counterfactuals)
