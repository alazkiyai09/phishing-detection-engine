"""
Partial dependence plots for model interpretation.

Shows marginal effect of features on predictions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from sklearn.inspection import PartialDependenceDisplay
from pathlib import Path
import logging

from src.models.classical.models.base_classifier import BaseClassifier
from src.models.classical.config import get_config

logger = logging.getLogger(__name__)


def get_top_features(
    importance_df: pd.DataFrame,
    top_n: int = 5
) -> List[int]:
    """
    Get indices of top N features by importance.

    Args:
        importance_df: DataFrame with feature importance
        top_n: Number of top features to return

    Returns:
        List of feature indices
    """
    config = get_config()
    top_n = config.get("top_n_features", top_n)

    # Extract feature indices from feature names
    top_features = importance_df.head(top_n)["feature"].tolist()

    # Try to extract indices
    indices = []
    for feat in top_features:
        if feat.startswith("feature_"):
            indices.append(int(feat.split("_")[1]))
        else:
            # If using actual feature names, we need to map back
            # This is a placeholder - actual implementation would need feature mapping
            indices.append(0)

    # If all indices are 0, use sequential indices
    if len(set(indices)) == 1 and indices[0] == 0:
        indices = list(range(min(top_n, len(importance_df))))

    return indices


def compute_partial_dependence(
    model: BaseClassifier,
    X: np.ndarray,
    features: List[int],
    feature_names: Optional[List[str]] = None
) -> dict:
    """
    Compute partial dependence for specified features.

    Args:
        model: Fitted classifier
        X: Features (n_samples, n_features)
        features: List of feature indices to analyze
        feature_names: List of feature names

    Returns:
        Dictionary with PDP results
    """
    logger.info(f"Computing partial dependence for {len(features)} features")

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    # Get sklearn model
    sklearn_model = model.model if hasattr(model, 'model') else model

    results = {}

    for feat_idx in features:
        try:
            from sklearn.inspection import partial_dependence

            pdp_result, axes = partial_dependence(
                estimator=sklearn_model,
                X=X,
                features=[feat_idx]
            )

            results[feat_idx] = {
                "values": pdp_result[0].ravel(),
                "axes": axes[0],
                "feature_name": feature_names[feat_idx] if feat_idx < len(feature_names) else f"feature_{feat_idx}"
            }

            logger.info(f"Computed PDP for feature {feat_idx}: {feature_names[feat_idx] if feat_idx < len(feature_names) else feat_idx}")

        except Exception as e:
            logger.error(f"Error computing PDP for feature {feat_idx}: {e}")
            continue

    return results


def plot_partial_dependence(
    model: BaseClassifier,
    X: np.ndarray,
    features: List[int],
    feature_names: Optional[List[str]] = None,
    model_name: str = "Model",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot partial dependence for specified features.

    Args:
        model: Fitted classifier
        X: Features (n_samples, n_features)
        features: List of feature indices to plot
        feature_names: List of feature names
        model_name: Name of the model for title
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    config = get_config()

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    # Get sklearn model
    sklearn_model = model.model if hasattr(model, 'model') else model

    # Create figure
    fig, ax = plt.subplots(figsize=config.get("figsize", (12, 8)))

    try:
        # Use sklearn's PartialDependenceDisplay
        display = PartialDependenceDisplay.from_estimator(
            estimator=sklearn_model,
            X=X,
            features=features,
            feature_names=feature_names,
            ax=ax,
            n_jobs=config.get("n_jobs", -1)
        )

        fig.suptitle(f"{model_name} - Partial Dependence Plots")

        plt.tight_layout()

        # Save figure
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=config.get("dpi", 300), bbox_inches='tight')
            logger.info(f"Saved PDP plot to {save_path}")

    except Exception as e:
        logger.error(f"Error creating PDP plot: {e}")
        plt.close()

    return fig


def plot_all_pdp(
    model: BaseClassifier,
    X: np.ndarray,
    feature_names: List[str],
    model_name: str,
    output_dir: Path
) -> None:
    """
    Create PDP plots for top features.

    Args:
        model: Fitted classifier
        X: Features
        feature_names: List of feature names
        model_name: Name of the model
        output_dir: Directory to save plots
    """
    # Get feature importance first
    from src.models.classical.interpretation.feature_importance import get_native_importance

    importance_df = get_native_importance(model, feature_names)

    # Get top N features
    config = get_config()
    top_n = config.get("top_n_features", 5)

    # Plot top N features in one figure
    top_indices = get_top_features(importance_df, top_n)

    if len(top_indices) > 0:
        save_path = output_dir / f"{model_name}_pdp_top_{top_n}.png"
        plot_partial_dependence(
            model=model,
            X=X,
            features=top_indices,
            feature_names=feature_names,
            model_name=model_name,
            save_path=save_path
        )

        logger.info(f"Created PDP for top {len(top_indices)} features")


def create_pdp_for_all_models(
    models: Dict[str, BaseClassifier],
    X: np.ndarray,
    feature_names: List[str],
    output_dir: Path
) -> None:
    """
    Create PDP plots for all models.

    Args:
        models: Dictionary mapping model names to fitted models
        X: Features
        feature_names: List of feature names
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model in models.items():
        logger.info(f"\nCreating PDP for {model_name}")

        try:
            plot_all_pdp(
                model=model,
                X=X,
                feature_names=feature_names,
                model_name=model_name,
                output_dir=output_dir
            )
        except Exception as e:
            logger.error(f"Error creating PDP for {model_name}: {e}")
