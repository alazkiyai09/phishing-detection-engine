"""
Decision boundary visualization.

Projects features to 2D using PCA for visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from sklearn.decomposition import PCA
from pathlib import Path
import logging

from src.models.classical.models.base_classifier import BaseClassifier
from src.models.classical.config import get_config

logger = logging.getLogger(__name__)


def project_pca_2d(
    X: np.ndarray,
    n_components: int = 2,
    random_state: int = 42
) -> Tuple[np.ndarray, PCA]:
    """
    Project features to 2D using PCA.

    Args:
        X: Features (n_samples, n_features)
        n_components: Number of PCA components
        random_state: Random seed

    Returns:
        Tuple of (projected_features, fitted_pca)
    """
    logger.info(f"Projecting {X.shape[1]} features to {n_components}D using PCA")

    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)

    explained_var = pca.explained_variance_ratio_
    logger.info(f"Explained variance: {explained_var.sum():.2%} "
                f"(PC1: {explained_var[0]:.2%}, PC2: {explained_var[1]:.2%})")

    return X_pca, pca


def plot_decision_boundary_2d(
    model: BaseClassifier,
    X: np.ndarray,
    y: np.ndarray,
    model_name: str = "Model",
    resolution: int = 100,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot 2D decision boundary using PCA projection.

    Args:
        model: Fitted classifier
        X: Features (n_samples, n_features)
        y: Labels (n_samples,)
        model_name: Name of the model
        resolution: Grid resolution for decision boundary
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    config = get_config()

    # Project to 2D
    X_pca, pca = project_pca_2d(X)

    # Create mesh grid
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )

    # Create figure
    fig, ax = plt.subplots(figsize=config.get("figsize", (12, 8)))

    # Predict on mesh grid
    try:
        # We need to project back to original space for prediction
        # This is approximate since PCA is lossy
        from sklearn.preprocessing import StandardScaler

        # Use mean of original features for other dimensions
        X_mean = X.mean(axis=0)

        # Create grid points in PCA space and project back
        grid_pca = np.c_[xx.ravel(), yy.ravel()]

        # Approximate inverse transform (add zeros for other components)
        if pca.n_components < pca.n_features_:
            zeros = np.zeros((len(grid_pca), pca.n_features_ - pca.n_components))
            grid_full = np.hstack([grid_pca, zeros])
        else:
            grid_full = grid_pca

        # Project back to original space
        grid_original = pca.inverse_transform(grid_full)

        # Get predictions
        Z = model.predict(grid_original)
        Z = Z.reshape(xx.shape)

        # Plot decision boundary
        ax.contourf(xx, yy, Z, alpha=0.3, levels=20, cmap=plt.cm.RdYlBu)

    except Exception as e:
        logger.warning(f"Could not plot decision boundary: {e}")

    # Plot data points
    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=y,
        cmap=plt.cm.RdYlBu,
        alpha=0.6,
        edgecolors='k',
        s=50
    )

    # Add labels
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title(f"{model_name} - Decision Boundary (PCA Projection)")

    # Add colorbar
    plt.colorbar(scatter, ax=ax, label="Class")

    plt.tight_layout()

    # Save figure
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=config.get("dpi", 300), bbox_inches='tight')
        logger.info(f"Saved decision boundary plot to {save_path}")

    return fig


def plot_all_decision_boundaries(
    models: Dict[str, BaseClassifier],
    X: np.ndarray,
    y: np.ndarray,
    output_dir: Path
) -> None:
    """
    Create decision boundary plots for all models.

    Args:
        models: Dictionary mapping model names to fitted models
        X: Features
        y: Labels
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model in models.items():
        logger.info(f"\nCreating decision boundary for {model_name}")

        try:
            save_path = output_dir / f"{model_name}_decision_boundary.png"
            plot_decision_boundary_2d(
                model=model,
                X=X,
                y=y,
                model_name=model_name,
                save_path=save_path
            )
        except Exception as e:
            logger.error(f"Error creating decision boundary for {model_name}: {e}")


def plot_prediction_confidence_2d(
    model: BaseClassifier,
    X: np.ndarray,
    y: np.ndarray,
    model_name: str = "Model",
    resolution: int = 100,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot 2D prediction confidence using PCA projection.

    Shows probability contours instead of hard decision boundary.

    Args:
        model: Fitted classifier
        X: Features (n_samples, n_features)
        y: Labels (n_samples,)
        model_name: Name of the model
        resolution: Grid resolution
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    config = get_config()

    # Project to 2D
    X_pca, pca = project_pca_2d(X)

    # Create mesh grid
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )

    # Create figure
    fig, ax = plt.subplots(figsize=config.get("figsize", (12, 8)))

    # Predict probabilities on mesh grid
    try:
        from sklearn.preprocessing import StandardScaler

        X_mean = X.mean(axis=0)
        grid_pca = np.c_[xx.ravel(), yy.ravel()]

        if pca.n_components < pca.n_features_:
            zeros = np.zeros((len(grid_pca), pca.n_features_ - pca.n_components))
            grid_full = np.hstack([grid_pca, zeros])
        else:
            grid_full = grid_pca

        grid_original = pca.inverse_transform(grid_full)

        # Get probability of positive class
        Z_proba = model.predict_proba(grid_original)

        if Z_proba.ndim > 1 and Z_proba.shape[1] > 1:
            Z_proba = Z_proba[:, 1]

        Z_proba = Z_proba.reshape(xx.shape)

        # Plot probability contours
        contour = ax.contourf(xx, yy, Z_proba, levels=20, cmap=plt.cm.RdYlBu, alpha=0.6)
        ax.contour(xx, yy, Z_proba, levels=[0.5], colors='black', linewidths=2)

        plt.colorbar(contour, ax=ax, label="Phishing Probability")

    except Exception as e:
        logger.warning(f"Could not plot probability contours: {e}")

    # Plot data points
    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=y,
        cmap=plt.cm.RdYlBu,
        alpha=0.8,
        edgecolors='k',
        s=50
    )

    # Add labels
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title(f"{model_name} - Prediction Confidence (PCA Projection)")

    plt.tight_layout()

    # Save figure
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=config.get("dpi", 300), bbox_inches='tight')
        logger.info(f"Saved prediction confidence plot to {save_path}")

    return fig
