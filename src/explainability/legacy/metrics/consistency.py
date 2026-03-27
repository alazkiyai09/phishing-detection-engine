"""
Consistency metrics for explanations.

Measures whether similar inputs receive similar explanations.
"""

from typing import List, Tuple
import numpy as np
from difflib import SequenceMatcher

from src.explainability.legacy.utils.data_structures import (
    EmailData,
    ModelOutput,
    Explanation
)
from src.explainability.legacy.generators.base_generator import BaseExplanationGenerator


def compute_consistency(
    generator: BaseExplanationGenerator,
    emails: List[EmailData],
    predictions: List[ModelOutput],
    threshold: float = 0.8
) -> float:
    """
    Compute consistency score for explanations.

    Consistency measures whether similar emails receive similar explanations.

    Args:
        generator: Explanation generator to evaluate
        emails: List of emails to compare
        predictions: Corresponding predictions
        threshold: Similarity threshold for considering emails as "similar"

    Returns:
        Consistency score (0.0 to 1.0)
    """
    if len(emails) < 2:
        return 1.0

    # Generate explanations for all emails
    explanations = [
        generator.generate_explanation(email, pred)
        for email, pred in zip(emails, predictions)
    ]

    # Compare all pairs
    consistency_scores = []

    for i in range(len(explanations)):
        for j in range(i + 1, len(explanations)):
            # Compute email similarity
            email_sim = _compute_email_similarity(emails[i], emails[j])

            # If emails are similar, explanations should be similar
            if email_sim >= threshold:
                explanation_sim = _compute_explanation_similarity(
                    explanations[i],
                    explanations[j]
                )

                # Consistency = explanation similarity for similar emails
                consistency_scores.append(explanation_sim)

    if not consistency_scores:
        return 1.0  # No similar emails found

    return np.mean(consistency_scores)


def _compute_email_similarity(email1: EmailData, email2: EmailData) -> float:
    """Compute similarity between two emails."""
    scores = []

    # Subject similarity
    subject_sim = SequenceMatcher(None, email1.subject, email2.subject).ratio()
    scores.append(subject_sim)

    # Sender similarity
    sender_sim = 1.0 if email1.sender.email == email2.sender.email else 0.0
    scores.append(sender_sim)

    # Body similarity (sample)
    body_sample1 = email1.body[:500]
    body_sample2 = email2.body[:500]
    body_sim = SequenceMatcher(None, body_sample1, body_sample2).ratio()
    scores.append(body_sim)

    # URL count similarity
    url_count_sim = 1.0 - abs(len(email1.urls) - len(email2.urls)) / max(len(email1.urls), len(email2.urls), 1)
    scores.append(url_count_sim)

    return np.mean(scores)


def _compute_explanation_similarity(exp1: Explanation, exp2: Explanation) -> float:
    """Compute similarity between two explanations."""
    scores = []

    # Prediction similarity
    pred_match = 1.0 if exp1.model_prediction.predicted_label == exp2.model_prediction.predicted_label else 0.0
    scores.append(pred_match)

    # Component similarity
    components = [
        ('sender', exp1.sender_explanation, exp2.sender_explanation),
        ('subject', exp1.subject_explanation, exp2.subject_explanation),
        ('body', exp1.body_explanation, exp2.body_explanation),
        ('url', exp1.url_explanation, exp2.url_explanation),
        ('attachment', exp1.attachment_explanation, exp2.attachment_explanation),
    ]

    for name, comp1, comp2 in components:
        if comp1 is not None and comp2 is not None:
            # Both have component - check agreement
            agreement = 1.0 if comp1.is_suspicious == comp2.is_suspicious else 0.0
            scores.append(agreement)
        elif comp1 is None and comp2 is None:
            # Both missing component - neutral
            scores.append(1.0)
        else:
            # One missing - penalize
            scores.append(0.5)

    return np.mean(scores)


def compute_stability(
    generator: BaseExplanationGenerator,
    email: EmailData,
    prediction: ModelOutput,
    num_perturbations: int = 10
) -> float:
    """
    Compute stability score for explanations.

    Stability measures whether explanations are robust to small input changes.

    Args:
        generator: Explanation generator to evaluate
        email: Email to test
        prediction: Prediction for email
        num_perturbations: Number of small perturbations to test

    Returns:
        Stability score (0.0 to 1.0)
    """
    # Generate explanation for original email
    original_explanation = generator.generate_explanation(email, prediction)

    # Generate explanations for perturbed emails
    similarities = []

    for _ in range(num_perturbations):
        # Create small perturbation
        perturbed_email = _small_perturbation(email)
        perturbed_pred = _perturb_prediction(prediction)

        # Generate explanation
        perturbed_explanation = generator.generate_explanation(perturbed_email, perturbed_pred)

        # Compute similarity
        sim = _compute_explanation_similarity(original_explanation, perturbed_explanation)
        similarities.append(sim)

    return np.mean(similarities)


def _small_perturbation(email: EmailData) -> EmailData:
    """Create small perturbation of email."""
    import copy

    perturbed = copy.deepcopy(email)

    # Add/remove a character in subject
    if len(perturbed.subject) > 5:
        perturbed.subject = perturbed.subject[:-1] + "."

    # Add space in body
    perturbed.body = perturbed.body.replace("  ", " ", 1)

    return perturbed


def _perturb_prediction(pred: ModelOutput) -> ModelOutput:
    """Perturb prediction slightly."""
    import copy

    perturbed = copy.deepcopy(pred)
    # Small confidence change
    perturbed.confidence = max(0.0, min(1.0, perturbed.confidence + np.random.uniform(-0.02, 0.02)))

    return perturbed


def compute_completeness(explanation: Explanation) -> float:
    """
    Compute completeness score for explanation.

    Completeness measures whether explanation covers all important components.

    Args:
        explanation: Explanation to evaluate

    Returns:
        Completeness score (0.0 to 1.0)
    """
    required_components = [
        'sender_explanation',
        'subject_explanation',
        'body_explanation',
        'url_explanation',
        'attachment_explanation'
    ]

    scores = []

    for component in required_components:
        comp_value = getattr(explanation, component, None)

        if comp_value is not None:
            # Component exists - check quality
            scores.append(1.0)
        else:
            # Component missing - check if it's applicable
            # E.g., no URLs → url_explanation can be None
            is_applicable = True

            if component == 'url_explanation' and len(explanation.email.urls) == 0:
                is_applicable = False
            elif component == 'attachment_explanation' and len(explanation.email.attachments) == 0:
                is_applicable = False

            if is_applicable:
                scores.append(0.0)  # Should have this component
            else:
                scores.append(1.0)  # OK to skip

    # Bonus for advanced explanations
    if explanation.feature_importance is not None:
        scores.append(0.1)
    if explanation.attention_visualization is not None:
        scores.append(0.1)
    if explanation.counterfactuals:
        scores.append(0.1)
    if explanation.comparative is not None:
        scores.append(0.1)

    return min(np.mean(scores), 1.0)
