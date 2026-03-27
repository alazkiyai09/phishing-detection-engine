"""
Feature importance analysis for model interpretation.

Computes native importance and SHAP values.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union
import shap
from pathlib import Path
import logging

from src.models.classical.models.base_classifier import BaseClassifier
from src.models.classical.config import get_config

logger = logging.getLogger(__name__)


def get_native_importance(
    model: BaseClassifier,
    feature_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Get native feature importance from model.

    Args:
        model: Fitted classifier
        feature_names: List of feature names

    Returns:
        DataFrame with feature names and importance scores
    """
    importance = model.get_feature_importance()

    if len(importance) == 0:
        logger.warning("Model has no native feature importance")
        return pd.DataFrame()

    # Normalize to sum to 1
    if importance.sum() > 0:
        importance = importance / importance.sum()

    # Create DataFrame
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importance))]

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    })

    # Sort by importance
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)

    return df


def compute_shap_values(
    model: BaseClassifier,
    X: np.ndarray,
    background_samples: int = 100,
    feature_names: Optional[List[str]] = None
) -> Dict[str, Union[np.ndarray, shap.Explainer]]:
    """
    Compute SHAP values for model interpretation.

    Args:
        model: Fitted classifier
        X: Features to explain (n_samples, n_features)
        background_samples: Number of background samples for TreeExplainer
        feature_names: List of feature names

    Returns:
        Dictionary with SHAP values and explainer object
    """
    config = get_config()
    background_samples = config.get("shap_background_samples", background_samples)

    # Set random seed for reproducibility
    random_seed = config.get("random_state", 42)
    np.random.seed(random_seed)

    logger.info(f"Computing SHAP values with {background_samples} background samples (seed={random_seed})")

    try:
        # Sample background data for TreeExplainer (faster than KernelExplainer)
        background_idx = np.random.choice(
            len(X),
            min(background_samples, len(X)),
            replace=False
        )
        X_background = X[background_idx]

        # Create explainer based on model type
        model_type = model.__class__.__name__.lower()

        if "xgboost" in model_type or "xgb" in model_type:
            import xgboost as xgb
            explainer = shap.TreeExplainer(model.model, data=X_background)
        elif "lightgbm" in model_type or "lgb" in model_type:
            explainer = shap.TreeExplainer(model.model, data=X_background)
        elif "catboost" in model_type:
            explainer = shap.TreeExplainer(model.model, data=X_background)
        elif "random" in model_type or "forest" in model_type:
            explainer = shap.TreeExplainer(model.model, data=X_background)
        elif "gbdt" in model_type or "gradient" in model_type:
            explainer = shap.TreeExplainer(model.model, data=X_background)
        elif "logistic" in model_type or "svm" in model_type:
            # Use LinearExplainer for linear models
            explainer = shap.LinearExplainer(model.model, X_background)
        else:
            # Fallback to KernelExplainer (slower but general)
            explainer = shap.KernelExplainer(model.predict_proba, X_background)

        # Compute SHAP values for a subset of samples
        max_samples = min(500, len(X))
        X_explain = X[:max_samples]

        shap_values = explainer.shap_values(X_explain)

        # Handle multi-class output
        if isinstance(shap_values, list):
            # Binary classification: use positive class
            shap_values = shap_values[1]

        logger.info(f"SHAP values computed for {max_samples} samples")

        return {
            "shap_values": shap_values,
            "explainer": explainer,
            "X_explain": X_explain
        }

    except Exception as e:
        logger.error(f"Error computing SHAP values: {e}")
        return {
            "shap_values": None,
            "explainer": None,
            "error": str(e)
        }


def get_shap_feature_importance(
    shap_values: np.ndarray,
    feature_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Get feature importance from SHAP values.

    Uses mean absolute SHAP value as importance score.

    Args:
        shap_values: SHAP values (n_samples, n_features)
        feature_names: List of feature names

    Returns:
        DataFrame with feature names and SHAP importance
    """
    if shap_values is None:
        return pd.DataFrame()

    # Compute mean absolute SHAP value per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Normalize to sum to 1
    if mean_abs_shap.sum() > 0:
        mean_abs_shap = mean_abs_shap / mean_abs_shap.sum()

    # Create DataFrame
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(mean_abs_shap))]

    df = pd.DataFrame({
        "feature": feature_names,
        "shap_importance": mean_abs_shap
    })

    # Sort by importance
    df = df.sort_values("shap_importance", ascending=False).reset_index(drop=True)

    return df


def plot_feature_importance(
    importance_df: pd.DataFrame,
    model_name: str,
    importance_type: str = "native",
    top_n: int = 20,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot feature importance.

    Args:
        importance_df: DataFrame from get_native_importance or get_shap_feature_importance
        model_name: Name of the model
        importance_type: Type of importance ('native' or 'shap')
        top_n: Number of top features to display
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    config = get_config()
    figsize = config.get("figsize", (12, 8))

    # Get importance column name
    importance_col = "importance" if importance_type == "native" else "shap_importance"

    # Get top features
    top_df = importance_df.head(top_n).copy()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot horizontal bar chart
    y_pos = np.arange(len(top_df))
    ax.barh(y_pos, top_df[importance_col], align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_df["feature"])
    ax.invert_yaxis()
    ax.set_xlabel(f"{importance_type.capitalize()} Importance")
    ax.set_title(f"{model_name} - Top {top_n} Features ({importance_type.capitalize()} Importance)")

    plt.tight_layout()

    # Save figure
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=config.get("dpi", 300), bbox_inches='tight')
        logger.info(f"Saved feature importance plot to {save_path}")

    return fig


def compare_feature_importance(
    models: Dict[str, BaseClassifier],
    X: np.ndarray,
    feature_names: List[str],
    output_dir: Path
) -> Dict[str, pd.DataFrame]:
    """
    Compare feature importance across all models.

    Args:
        models: Dictionary mapping model names to fitted models
        X: Features for SHAP computation
        feature_names: List of feature names
        output_dir: Directory to save plots

    Returns:
        Dictionary mapping model names to importance DataFrames
    """
    all_importance = {}

    for model_name, model in models.items():
        logger.info(f"\nComputing feature importance for {model_name}")

        # Native importance
        native_importance = get_native_importance(model, feature_names)
        all_importance[f"{model_name}_native"] = native_importance

        # Save native importance plot
        save_path = output_dir / f"{model_name}_native_importance.png"
        plot_feature_importance(
            native_importance,
            model_name,
            importance_type="native",
            save_path=save_path
        )

        # SHAP importance
        shap_results = compute_shap_values(model, X, feature_names=feature_names)

        if shap_results["shap_values"] is not None:
            shap_importance = get_shap_feature_importance(
                shap_results["shap_values"],
                feature_names
            )
            all_importance[f"{model_name}_shap"] = shap_importance

            # Save SHAP importance plot
            save_path = output_dir / f"{model_name}_shap_importance.png"
            plot_feature_importance(
                shap_importance,
                model_name,
                importance_type="shap",
                save_path=save_path
            )

            # Save SHAP summary plot
            plt.figure()
            shap.summary_plot(
                shap_results["shap_values"],
                shap_results["X_explain"],
                feature_names=feature_names,
                show=False
            )
            summary_path = output_dir / f"{model_name}_shap_summary.png"
            plt.savefig(summary_path, dpi=config.get("dpi", 300), bbox_inches='tight')
            plt.close()
            logger.info(f"Saved SHAP summary plot to {summary_path}")

    return all_importance
