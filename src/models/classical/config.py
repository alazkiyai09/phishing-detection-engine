"""
Central configuration for Phishing Classifier Benchmark.

Ensures reproducibility and consistent parameters across all experiments.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional


def get_config() -> Dict[str, Any]:
    """
    Get central configuration dictionary.

    Returns:
        dict: Configuration parameters for the benchmark
    """
    # Use relative paths for portability
    BASE_DIR = Path(__file__).parent.parent

    return {
        # Paths (portable relative paths)
        "base_dir": BASE_DIR,
        "data_dir": BASE_DIR / "data",
        "results_dir": BASE_DIR / "results",
        "notebooks_dir": BASE_DIR / "notebooks",

        # Feature pipeline from Day 1 (relative path or env var)
        "day1_features_path": Path(os.getenv(
            "DAY1_FEATURES_PATH",
            str(BASE_DIR.parent / "phishing_email_analysis")
        )),

        # Reproducibility
        "random_state": 42,
        "n_jobs": -1,  # Use all available cores

        # Cross-validation
        "n_folds": 5,
        "cv_strategy": "stratified",

        # Train/Val/Test split
        "test_size": 0.2,
        "val_size": 0.2,  # From remaining training data
        "temporal_col": "date",  # Column for temporal split

        # Hyperparameter tuning
        "n_trials": 50,
        "timeout": 3600,  # 1 hour per study
        "n_jobs_optuna": 1,  # Optuna requires 1 for proper pruning

        # Evaluation thresholds
        "fpr_threshold": 0.01,  # 1% FPR requirement for financial sector
        "recall_threshold": 0.95,  # 95% recall on financial phishing

        # Computation budget
        "max_iter_lr": 1000,
        "max_trees_rf": 500,
        "max_rounds_xgb": 1000,
        "max_rounds_lgb": 1000,
        "max_rounds_cat": 1000,
        "max_iter_svm": 1000,

        # Feature importance
        "top_n_features": 5,  # For partial dependence plots
        "shap_background_samples": 100,

        # Visualization
        "figsize": (12, 8),
        "dpi": 300,

        # Phishing categories (for subset analysis)
        "phishing_type_col": "phishing_type",  # Column in dataset
        "phishing_types": ["generic", "financial", "spear"],

        # Model names for reporting
        "model_names": {
            "logistic_regression": "Logistic Regression",
            "random_forest": "Random Forest",
            "xgboost": "XGBoost",
            "lightgbm": "LightGBM",
            "catboost": "CatBoost",
            "svm": "SVM (RBF)",
            "gbdt": "Gradient Boosted Trees"
        },

        # Class labels
        "class_names": ["Legitimate", "Phishing"],
        "class_labels": {0: "Legitimate", 1: "Phishing"},

        # Metrics to track
        "primary_metrics": ["accuracy", "precision", "recall", "f1", "auprc", "auroc"],
        "per_class_metrics": ["precision", "recall", "f1"],

        # Compute tracking
        "track_time": True,
        "track_memory": False,  # Set to True if memory profiling needed
    }


# Convenience function to get individual config values
def get_config_value(key: str, default: Any = None) -> Any:
    """
    Get a specific configuration value.

    Args:
        key: Configuration key
        default: Default value if key not found

    Returns:
        Configuration value or default
    """
    config = get_config()
    return config.get(key, default)
