"""
Model interpretation module.

Provides feature importance, SHAP values, and visualizations.
"""

from src.models.classical.interpretation.feature_importance import (
    get_native_importance,
    compute_shap_values,
    plot_feature_importance
)
from src.models.classical.interpretation.partial_dependence import (
    compute_partial_dependence,
    plot_partial_dependence,
    get_top_features
)
from src.models.classical.interpretation.decision_boundary import (
    plot_decision_boundary_2d,
    project_pca_2d
)

__all__ = [
    "get_native_importance",
    "compute_shap_values",
    "plot_feature_importance",
    "compute_partial_dependence",
    "plot_partial_dependence",
    "get_top_features",
    "plot_decision_boundary_2d",
    "project_pca_2d",
]
