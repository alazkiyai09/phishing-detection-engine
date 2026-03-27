"""Analysis modules for feature importance and correlation."""

from .importance import (
    compute_mutual_information,
    compute_shap_values,
    get_feature_importance_from_shap,
    plot_feature_importance,
    plot_shap_summary,
    compare_feature_importance,
    rank_features_by_importance,
    print_feature_ranking,
)
from .correlation import (
    compute_correlation_matrix,
    find_highly_correlated_features,
    plot_correlation_matrix,
    plot_correlation_clustermap,
    identify_redundant_features,
    remove_redundant_features,
    analyze_feature_correlation,
    print_correlation_summary,
)

__all__ = [
    "compute_mutual_information",
    "compute_shap_values",
    "get_feature_importance_from_shap",
    "plot_feature_importance",
    "plot_shap_summary",
    "compare_feature_importance",
    "rank_features_by_importance",
    "print_feature_ranking",
    "compute_correlation_matrix",
    "find_highly_correlated_features",
    "plot_correlation_matrix",
    "plot_correlation_clustermap",
    "identify_redundant_features",
    "remove_redundant_features",
    "analyze_feature_correlation",
    "print_correlation_summary",
]
