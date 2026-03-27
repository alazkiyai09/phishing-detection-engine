"""
Hyperparameter optimization using Optuna.

Automated hyperparameter search with pruning for efficiency.
"""

import optuna
import numpy as np
from typing import Dict, Any, Type, Callable, Optional
import logging
import warnings

from src.models.classical.models.base_classifier import BaseClassifier
from src.models.classical.evaluation.metrics import compute_metrics
from src.models.classical.config import get_config

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
logger = logging.getLogger(__name__)


# Define search spaces for each model
SEARCH_SPACES = {
    "logistic_regression": {
        "C": lambda trial: trial.suggest_float("C", 0.001, 100.0, log=True),
        "penalty": lambda trial: trial.suggest_categorical("penalty", ["l1", "l2"]),
        "solver": lambda trial: trial.suggest_categorical("solver", ["liblinear", "saga"]),
    },
    "random_forest": {
        "n_estimators": lambda trial: trial.suggest_int("n_estimators", 50, 500),
        "max_depth": lambda trial: trial.suggest_int("max_depth", 5, 30),
        "min_samples_split": lambda trial: trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": lambda trial: trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": lambda trial: trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.7, 1.0]),
    },
    "xgboost": {
        "n_estimators": lambda trial: trial.suggest_int("n_estimators", 50, 1000),
        "max_depth": lambda trial: trial.suggest_int("max_depth", 3, 12),
        "learning_rate": lambda trial: trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": lambda trial: trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": lambda trial: trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": lambda trial: trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": lambda trial: trial.suggest_float("reg_lambda", 0.0, 10.0),
    },
    "lightgbm": {
        "n_estimators": lambda trial: trial.suggest_int("n_estimators", 50, 1000),
        "max_depth": lambda trial: trial.suggest_int("max_depth", 3, 15),
        "num_leaves": lambda trial: trial.suggest_int("num_leaves", 20, 150),
        "learning_rate": lambda trial: trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": lambda trial: trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": lambda trial: trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": lambda trial: trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": lambda trial: trial.suggest_float("reg_lambda", 0.0, 10.0),
    },
    "catboost": {
        "iterations": lambda trial: trial.suggest_int("iterations", 50, 1000),
        "depth": lambda trial: trial.suggest_int("depth", 4, 10),
        "learning_rate": lambda trial: trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": lambda trial: trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
    },
    "svm": {
        "C": lambda trial: trial.suggest_float("C", 0.01, 100.0, log=True),
        "gamma": lambda trial: trial.suggest_categorical("gamma", ["scale", "auto", 0.001, 0.01, 0.1, 1.0]),
    },
    "gbdt": {
        "n_estimators": lambda trial: trial.suggest_int("n_estimators", 50, 500),
        "max_depth": lambda trial: trial.suggest_int("max_depth", 3, 10),
        "learning_rate": lambda trial: trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": lambda trial: trial.suggest_float("subsample", 0.5, 1.0),
        "min_samples_split": lambda trial: trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": lambda trial: trial.suggest_int("min_samples_leaf", 1, 10),
    },
}


def get_search_space(model_name: str) -> Dict[str, Callable]:
    """
    Get search space for a model.

    Args:
        model_name: Name of the model

    Returns:
        Dictionary mapping parameter names to Optuna suggest functions
    """
    if model_name not in SEARCH_SPACES:
        raise ValueError(f"No search space defined for model: {model_name}")

    return SEARCH_SPACES[model_name]


def optimize_hyperparams(
    model_cls: Type[BaseClassifier],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_name: str,
    n_trials: int = 50,
    timeout: Optional[int] = None,
    random_state: int = 42,
    direction: str = "maximize",
    metric: str = "f1"
) -> Dict[str, Any]:
    """
    Optimize hyperparameters using Optuna.

    Args:
        model_cls: Model class (not instance)
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        model_name: Name of the model (for search space lookup)
        n_trials: Number of optimization trials
        timeout: Timeout in seconds (None = no limit)
        random_state: Random seed
        direction: Optimization direction ('maximize' or 'minimize')
        metric: Metric to optimize

    Returns:
        Dictionary with best parameters and study results
    """
    config = get_config()
    search_space = get_search_space(model_name)

    logger.info(f"Starting hyperparameter optimization for {model_name}")
    logger.info(f"Trials: {n_trials}, Metric: {metric}, Direction: {direction}")

    def objective(trial: optuna.Trial) -> float:
        """Objective function for optimization."""

        # Sample hyperparameters
        params = {}
        for param_name, suggest_fn in search_space.items():
            params[param_name] = suggest_fn(trial)

        # Add fixed parameters
        params["random_state"] = random_state

        # Create model with sampled parameters
        try:
            model = model_cls(**params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)

            # Compute metrics
            metrics = compute_metrics(y_val, y_pred, y_proba)
            score = metrics.get(metric, 0.0)

            # Handle NaN
            if np.isnan(score) or np.isinf(score):
                return -1.0 if direction == "maximize" else 1.0

            return score

        except Exception as e:
            logger.warning(f"Trial failed with params {params}: {e}")
            return -1.0 if direction == "maximize" else 1.0

    # Create study
    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=random_state)
    )

    # Optimize
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )

    # Get best parameters
    best_params = study.best_params.copy()
    best_params["random_state"] = random_state

    # For class_weight, add it if not in search space
    if "class_weight" not in best_params:
        best_params["class_weight"] = "balanced"

    logger.info(f"Best {metric}: {study.best_value:.4f}")
    logger.info(f"Best parameters: {best_params}")

    return {
        "best_params": best_params,
        "best_value": study.best_value,
        "n_trials": len(study.trials),
        "study": study
    }


def get_best_params(study: optuna.Study) -> Dict[str, Any]:
    """
    Extract best parameters from Optuna study.

    Args:
        study: Completed Optuna study

    Returns:
        Dictionary of best hyperparameters
    """
    return study.best_params


def tune_all_models(
    models: Dict[str, Type[BaseClassifier]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    random_state: int = 42
) -> Dict[str, Dict[str, Any]]:
    """
    Tune hyperparameters for all models.

    Args:
        models: Dictionary mapping model names to model classes
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        n_trials: Number of trials per model
        random_state: Random seed

    Returns:
        Dictionary mapping model names to tuning results
    """
    config = get_config()

    # Override from config if specified
    n_trials = config.get("n_trials", n_trials)

    all_results = {}

    for model_name, model_cls in models.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Tuning: {model_name}")
        logger.info(f"{'='*60}")

        try:
            results = optimize_hyperparams(
                model_cls=model_cls,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                model_name=model_name,
                n_trials=n_trials,
                random_state=random_state
            )

            all_results[model_name] = results

        except Exception as e:
            logger.error(f"Error tuning {model_name}: {e}")
            all_results[model_name] = {
                "best_params": {},
                "best_value": 0.0,
                "error": str(e)
            }

    return all_results
