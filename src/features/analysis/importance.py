"""Feature importance analysis using SHAP and mutual information.

Provides tools for understanding which features are most discriminative
for phishing detection.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score


def compute_mutual_information(
    X: pd.DataFrame, y: pd.Series, random_state: int = 42
) -> pd.Series:
    """Compute mutual information between each feature and target.

    Mutual information measures the dependency between variables.
    Higher values indicate more predictive features.

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Target vector (n_samples,).
        random_state: Random seed for reproducibility.

    Returns:
        Series of mutual information scores, indexed by feature name.
        Sorted in descending order.

    Raises:
        ValueError: If X and y have incompatible shapes.
    """
    if len(X) != len(y):
        raise ValueError(
            f"X and y must have same length: {len(X)} != {len(y)}"
        )

    mi_scores = mutual_info_classif(X, y, random_state=random_state)

    mi_series = pd.Series(mi_scores, index=X.columns)
    mi_series = mi_series.sort_values(ascending=False)

    return mi_series


def compute_shap_values(
    model, X: pd.DataFrame, max_samples: Optional[int] = 100
) -> pd.DataFrame:
    """Compute SHAP values for feature importance.

    SHAP (SHapley Additive exPlanations) provides game-theoretic
    feature importance scores.

    Args:
        model: Fitted sklearn-compatible model.
        X: Feature matrix to explain.
        max_samples: Max samples to explain (for speed).

    Returns:
        DataFrame of SHAP values (n_samples, n_features).
    """
    # Limit samples for performance
    if max_samples and len(X) > max_samples:
        X_sample = X.sample(n=max_samples, random_state=42)
    else:
        X_sample = X

    # Initialize SHAP explainer
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        # Fallback to kernel explainer for non-tree models
        explainer = shap.KernelExplainer(model.predict_proba, X_sample)

    # Compute SHAP values
    shap_values = explainer.shap_values(X_sample)

    # Handle binary classification (returns list of arrays)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use positive class

    # Convert to DataFrame
    shap_df = pd.DataFrame(shap_values, columns=X.columns, index=X_sample.index)

    return shap_df


def get_feature_importance_from_shap(shap_df: pd.DataFrame) -> pd.Series:
    """Get overall feature importance from SHAP values.

    Uses mean absolute SHAP value as importance metric.

    Args:
        shap_df: DataFrame of SHAP values from compute_shap_values().

    Returns:
        Series of feature importance scores, sorted descending.
    """
    importance = np.abs(shap_df).mean(axis=0)
    importance = importance.sort_values(ascending=False)

    return importance


def plot_feature_importance(
    importance_scores: pd.Series,
    title: str = "Feature Importance",
    top_n: int = 20,
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None,
) -> None:
    """Plot feature importance scores.

    Args:
        importance_scores: Series of importance scores (from MI or SHAP).
        title: Plot title.
        top_n: Number of top features to show.
        figsize: Figure size (width, height).
        save_path: Optional path to save figure.
    """
    # Get top features
    top_features = importance_scores.head(top_n)

    # Create plot
    plt.figure(figsize=figsize)
    sns.barplot(x=top_features.values, y=top_features.index, palette="viridis")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Importance Score", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    plt.show()


def plot_shap_summary(
    shap_df: pd.DataFrame,
    X: pd.DataFrame,
    max_features: int = 20,
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None,
) -> None:
    """Plot SHAP summary plot.

    Shows feature importance and direction of impact.

    Args:
        shap_df: DataFrame of SHAP values.
        X: Feature matrix (for feature values).
        max_features: Max features to show.
        figsize: Figure size.
        save_path: Optional path to save figure.
    """
    # Limit features for readability
    if len(shap_df.columns) > max_features:
        # Select top features by importance
        importance = np.abs(shap_df).mean(axis=0).sort_values(ascending=False)
        top_features = importance.head(max_features).index.tolist()
        shap_df = shap_df[top_features]
        X = X[top_features]

    # Create summary plot
    plt.figure(figsize=figsize)
    shap.summary_plot(
        shap_df.values, X.values, feature_names=shap_df.columns, show=False
    )
    plt.title("SHAP Summary Plot", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    plt.show()


def compare_feature_importance(
    mi_scores: pd.Series,
    shap_scores: pd.Series,
    top_n: int = 20,
    figsize: tuple = (12, 8),
    save_path: Optional[str] = None,
) -> None:
    """Compare feature importance from multiple methods.

    Args:
        mi_scores: Mutual information scores.
        shap_scores: SHAP importance scores.
        top_n: Number of top features to compare.
        figsize: Figure size.
        save_path: Optional path to save figure.
    """
    # Get top features from both methods
    top_mi = mi_scores.head(top_n)
    top_shap = shap_scores.head(top_n)

    # Normalize scores for comparison
    mi_norm = (top_mi - top_mi.min()) / (top_mi.max() - top_mi.min())
    shap_norm = (top_shap - top_shap.min()) / (top_shap.max() - top_shap.min())

    # Combine into DataFrame
    comparison = pd.DataFrame({"Mutual Information": mi_norm, "SHAP": pd.Series(dtype=float)})
    for feat in top_shap.index:
        if feat in mi_norm.index:
            comparison.loc[feat, "SHAP"] = shap_norm[feat]

    comparison = comparison.dropna()

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    comparison.sort_values("Mutual Information").plot(
        kind="barh", ax=ax, color=["#2ecc71", "#3498db"]
    )
    ax.set_title("Feature Importance Comparison: MI vs SHAP", fontsize=14, fontweight="bold")
    ax.set_xlabel("Normalized Importance Score", fontsize=12)
    ax.legend(loc="best")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    plt.show()


def rank_features_by_importance(
    X: pd.DataFrame,
    y: pd.Series,
    model=None,
    method: str = "mutual_info",
    top_n: int = 20,
) -> pd.DataFrame:
    """Rank features by importance using specified method.

    Args:
        X: Feature matrix.
        y: Target vector.
        model: Fitted model (required for SHAP).
        method: Importance method ('mutual_info' or 'shap').
        top_n: Number of top features to return.

    Returns:
        DataFrame with feature rankings.
    """
    if method == "mutual_info":
        scores = compute_mutual_information(X, y)
    elif method == "shap":
        if model is None:
            raise ValueError("Model must be provided for SHAP importance")
        shap_df = compute_shap_values(model, X)
        scores = get_feature_importance_from_shap(shap_df)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Create ranking DataFrame
    ranking = pd.DataFrame(
        {"feature": scores.index, "importance": scores.values, "rank": range(1, len(scores) + 1)}
    )

    return ranking.head(top_n)


def print_feature_ranking(ranking_df: pd.DataFrame, title: str = "Feature Ranking") -> None:
    """Print feature ranking table.

    Args:
        ranking_df: DataFrame from rank_features_by_importance().
        title: Table title.
    """
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}\n")

    for idx, row in ranking_df.iterrows():
        print(f"#{row['rank']:2d} | {row['feature']:40s} | {row['importance']:.4f}")

    print(f"{'=' * 60}\n")
