"""
Explanation algorithms for phishing detection.

This module contains various explanation methods:
- Feature-based: SHAP values
- Attention-based: Transformer attention visualization
- Counterfactual: What-if scenarios
- Comparative: Similar to known campaigns
"""

from src.explainability.legacy.explainers.feature_based import FeatureBasedExplainer
from src.explainability.legacy.explainers.attention_based import AttentionBasedExplainer
from src.explainability.legacy.explainers.counterfactual import CounterfactualExplainer
from src.explainability.legacy.explainers.comparative import ComparativeExplainer

__all__ = [
    "FeatureBasedExplainer",
    "AttentionBasedExplainer",
    "CounterfactualExplainer",
    "ComparativeExplainer",
]
