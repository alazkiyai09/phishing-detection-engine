"""Correlation analysis for feature redundancy detection.

Identifies highly correlated features that can be removed to reduce
dimensionality and improve model performance.
"""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform


def compute_correlation_matrix(
    X: pd.DataFrame, method: str = "pearson"
) -> pd.DataFrame:
    """Compute correlation matrix for features.

    Args:
        X: Feature matrix (n_samples, n_features).
        method: Correlation method ('pearson', 'spearman', 'kendall').

    Returns:
        Correlation matrix DataFrame (n_features, n_features).
    """
    return X.corr(method=method)


def find_highly_correlated_features(
    corr_matrix: pd.DataFrame, threshold: float = 0.9
) -> List[Tuple[str, str, float]]:
    """Find feature pairs with correlation above threshold.

    Args:
        corr_matrix: Correlation matrix from compute_correlation_matrix().
        threshold: Correlation threshold (0-1).

    Returns:
        List of tuples (feature1, feature2, correlation) for pairs
        exceeding threshold.
    """
    # Get upper triangle (excluding diagonal)
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Find pairs above threshold
    high_corr_pairs = []
    for col1 in corr_matrix.columns:
        for col2 in corr_matrix.columns:
            if col1 != col2:
                corr_value = upper_triangle.loc[col1, col2]
                if not pd.isna(corr_value) and abs(corr_value) >= threshold:
                    high_corr_pairs.append((col1, col2, corr_value))

    # Sort by correlation strength
    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    return high_corr_pairs


def plot_correlation_matrix(
    corr_matrix: pd.DataFrame,
    figsize: tuple = (16, 12),
    cmap: str = "RdBu_r",
    save_path: Optional[str] = None,
    title: str = "Feature Correlation Matrix",
) -> None:
    """Plot correlation matrix heatmap.

    Args:
        corr_matrix: Correlation matrix.
        figsize: Figure size (width, height).
        cmap: Colormap name.
        save_path: Optional path to save figure.
        title: Plot title.
    """
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Plot
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    plt.show()


def plot_correlation_clustermap(
    X: pd.DataFrame,
    method: str = "pearson",
    figsize: tuple = (16, 12),
    cmap: str = "RdBu_r",
    save_path: Optional[str] = None,
    title: str = "Feature Correlation Clustermap",
) -> None:
    """Plot clustered correlation heatmap.

    Groups correlated features together using hierarchical clustering.

    Args:
        X: Feature matrix.
        method: Correlation method.
        figsize: Figure size.
        cmap: Colormap name.
        save_path: Optional path to save figure.
        title: Plot title.
    """
    # Compute correlation
    corr = X.corr(method=method)

    # Perform hierarchical clustering
    linkage = hierarchy.linkage(
        squareform(1 - np.abs(corr)), method="average"
    )

    # Plot clustermap
    g = sns.clustermap(
        corr,
        row_linkage=linkage,
        col_linkage=linkage,
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        figsize=figsize,
        linewidths=0.5,
        cbar_pos=(0.02, 0.8, 0.05, 0.18),
    )
    g.fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    plt.show()


def identify_redundant_features(
    X: pd.DataFrame,
    threshold: float = 0.9,
    method: str = "pearson",
    print_report: bool = True,
) -> List[str]:
    """Identify redundant features based on correlation.

    For each highly correlated pair, removes the feature with lower
    average correlation to all other features.

    Args:
        X: Feature matrix.
        threshold: Correlation threshold.
        method: Correlation method.
        print_report: Whether to print redundancy report.

    Returns:
        List of redundant feature names to remove.
    """
    # Compute correlation
    corr = X.corr(method=method)

    # Find highly correlated pairs
    high_corr_pairs = find_highly_correlated_features(corr, threshold)

    if not high_corr_pairs:
        if print_report:
            print(f"\nNo features found with correlation >= {threshold}")
        return []

    # Identify features to remove
    features_to_remove = set()
    removal_reasons = []

    for feat1, feat2, corr_value in high_corr_pairs:
        # Skip if already marked for removal
        if feat1 in features_to_remove or feat2 in features_to_remove:
            continue

        # Compute average correlation for each feature
        avg_corr1 = np.abs(corr[feat1]).mean()
        avg_corr2 = np.abs(corr[feat2]).mean()

        # Remove feature with lower average correlation
        if avg_corr1 < avg_corr2:
            features_to_remove.add(feat1)
            removal_reasons.append((feat1, feat2, corr_value, avg_corr1, avg_corr2))
        else:
            features_to_remove.add(feat2)
            removal_reasons.append((feat2, feat1, corr_value, avg_corr2, avg_corr1))

    # Print report
    if print_report:
        print(f"\n{'=' * 70}")
        print(f" REDUNDANT FEATURES DETECTION (threshold >= {threshold})")
        print(f"{'=' * 70}\n")

        print(f"Found {len(removal_reasons)} redundant feature pairs:\n")

        for i, (remove, keep, corr, avg_rem, avg_keep) in enumerate(
            removal_reasons, 1
        ):
            print(
                f"{i}. REMOVE: {remove:40s} (avg corr: {avg_rem:.3f})"
            )
            print(f"   KEEP:   {keep:40s} (avg corr: {avg_keep:.3f})")
            print(f"   Correlation: {corr:.3f}\n")

        print("-" * 70)
        print(f"\nTotal features to remove: {len(features_to_remove)}")
        print(f"Remaining features: {len(X.columns) - len(features_to_remove)}")
        print(f"{'=' * 70}\n")

    return list(features_to_remove)


def remove_redundant_features(
    X: pd.DataFrame, threshold: float = 0.9, method: str = "pearson"
) -> Tuple[pd.DataFrame, List[str]]:
    """Remove redundant features from dataset.

    Args:
        X: Feature matrix.
        threshold: Correlation threshold.
        method: Correlation method.

    Returns:
        Tuple of (reduced feature matrix, list of removed features).
    """
    features_to_remove = identify_redundant_features(
        X, threshold=threshold, method=method, print_report=True
    )

    if not features_to_remove:
        return X, []

    X_reduced = X.drop(columns=features_to_remove)

    return X_reduced, features_to_remove


def analyze_feature_correlation(
    X: pd.DataFrame,
    threshold: float = 0.9,
    plot: bool = True,
    plot_clustered: bool = False,
    save_dir: Optional[str] = None,
) -> dict:
    """Perform comprehensive correlation analysis.

    Args:
        X: Feature matrix.
        threshold: Correlation threshold for redundancy detection.
        plot: Whether to generate plots.
        plot_clustered: Whether to generate clustered plot.
        save_dir: Optional directory to save plots.

    Returns:
        Dictionary containing:
            - 'correlation_matrix': Full correlation matrix
            - 'redundant_features': List of redundant features
            - 'high_corr_pairs': List of highly correlated pairs
    """
    results = {}

    # Compute correlation matrix
    corr = compute_correlation_matrix(X)
    results["correlation_matrix"] = corr

    # Find highly correlated pairs
    high_corr_pairs = find_highly_correlated_features(corr, threshold)
    results["high_corr_pairs"] = high_corr_pairs

    # Identify redundant features
    redundant = identify_redundant_features(X, threshold, print_report=True)
    results["redundant_features"] = redundant

    # Generate plots
    if plot:
        save_path = f"{save_dir}/correlation_matrix.png" if save_dir else None
        plot_correlation_matrix(corr, save_path=save_path)

    if plot_clustered:
        save_path = f"{save_dir}/correlation_clustermap.png" if save_dir else None
        plot_correlation_clustermap(X, save_path=save_path)

    return results


def print_correlation_summary(high_corr_pairs: List[Tuple[str, str, float]]) -> None:
    """Print summary of highly correlated feature pairs.

    Args:
        high_corr_pairs: List from find_highly_correlated_features().
    """
    if not high_corr_pairs:
        print("\nNo highly correlated features found.")
        return

    print(f"\n{'=' * 70}")
    print(f" HIGHLY CORRELATED FEATURE PAIRS")
    print(f"{'=' * 70}\n")

    print(f"Found {len(high_corr_pairs)} pairs:\n")

    for i, (feat1, feat2, corr) in enumerate(high_corr_pairs, 1):
        print(f"{i:2d}. {feat1:40s} <-> {feat2:40s}")
        print(f"    Correlation: {corr:.3f}\n")

    print(f"{'=' * 70}\n")
