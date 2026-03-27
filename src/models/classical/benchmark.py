"""
Main benchmark orchestration script.

Runs the complete evaluation pipeline for all 7 classical ML classifiers.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import json
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

from src.models.classical.config import get_config
from src.models.classical.preprocessing import (
    load_features,
    create_splits,
    prepare_data_for_model,
    separate_features_target
)
from src.models.classical.models import (
    LogisticRegressionClassifier,
    RandomForestClassifier,
    XGBoostClassifier,
    LightGBMClassifier,
    CatBoostClassifier,
    SVMClassifier,
    GBDTReferenceClassifier
)
from src.models.classical.evaluation import (
    stratified_cv,
    temporal_evaluation,
    compute_metrics,
    aggregate_cv_results,
    format_cv_results
)
from src.models.classical.tuning import tune_all_models
from src.models.classical.interpretation import (
    compare_feature_importance,
    create_pdp_for_all_models,
    plot_all_decision_boundaries
)
from src.models.classical.analysis import (
    create_error_report,
    create_confusion_report,
    create_edge_case_report,
    analyze_edge_case_performance,
    save_edge_case_samples
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model class mapping
MODEL_CLASSES = {
    "logistic_regression": LogisticRegressionClassifier,
    "random_forest": RandomForestClassifier,
    "xgboost": XGBoostClassifier,
    "lightgbm": LightGBMClassifier,
    "catboost": CatBoostClassifier,
    "svm": SVMClassifier,
    "gbdt": GBDTReferenceClassifier
}


def run_benchmark(
    features_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    tune_hyperparams: bool = True,
    run_interpretation: bool = True,
    run_analysis: bool = True
) -> Dict[str, any]:
    """
    Run complete benchmark evaluation.

    Args:
        features_path: Path to features CSV (default: from config)
        output_dir: Directory for outputs (default: from config)
        tune_hyperparams: Whether to run hyperparameter tuning
        run_interpretation: Whether to run interpretation analysis
        run_analysis: Whether to run error analysis

    Returns:
        Dictionary with all benchmark results
    """
    config = get_config()

    # Setup paths
    if features_path is None:
        features_path = config["day1_features_path"]

    if output_dir is None:
        output_dir = config["results_dir"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (output_dir / "models").mkdir(exist_ok=True)
    (output_dir / "interpretation").mkdir(exist_ok=True)
    (output_dir / "analysis").mkdir(exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)

    logger.info("="*80)
    logger.info("PHISHING CLASSIFIER BENCHMARK - CLASSICAL ML")
    logger.info("="*80)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")

    # Phase 1: Load and prepare data
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: DATA LOADING AND PREPARATION")
    logger.info("="*80)

    df = load_features(features_path)
    train_df, val_df, test_df = create_splits(df)

    # Prepare features and labels
    feature_names = None
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = prepare_data_for_model(
        train_df, val_df, test_df, scale=False
    )

    logger.info(f"Feature shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    # Phase 2: Hyperparameter tuning (optional)
    best_params = {}
    if tune_hyperparams:
        logger.info("\n" + "="*80)
        logger.info("PHASE 2: HYPERPARAMETER TUNING")
        logger.info("="*80)

        tuning_results = tune_all_models(
            models=MODEL_CLASSES,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            n_trials=config["n_trials"],
            random_state=config["random_state"]
        )

        # Save best parameters
        for model_name, results in tuning_results.items():
            if "best_params" in results:
                best_params[model_name] = results["best_params"]

        # Save tuning results
        tuning_path = output_dir / "tuning_results.json"
        with open(tuning_path, 'w') as f:
            json.dump(tuning_results, f, indent=2, default=str)
        logger.info(f"Saved tuning results to {tuning_path}")

    # Phase 3: Model evaluation with cross-validation
    logger.info("\n" + "="*80)
    logger.info("PHASE 3: CROSS-VALIDATION EVALUATION")
    logger.info("="*80)

    cv_results_all = {}
    models_fitted = {}

    for model_name, model_class in MODEL_CLASSES.items():
        logger.info(f"\nEvaluating: {model_name}")

        # Initialize model with best params from tuning or defaults
        if model_name in best_params:
            model = model_class(**best_params[model_name])
            logger.info(f"Using tuned hyperparameters")
        else:
            model = model_class()
            logger.info(f"Using default hyperparameters")

        # Run cross-validation
        try:
            aggregated, fold_results = stratified_cv(
                model=model,
                X=np.vstack([X_train, X_val]),  # Combine train+val for CV
                y=np.concatenate([y_train, y_val]),
                n_folds=config["n_folds"],
                random_state=config["random_state"]
            )

            cv_results_all[model_name] = {
                "aggregated": aggregated,
                "fold_results": fold_results
            }

            # Fit on full training data for later phases
            model.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))
            models_fitted[model_name] = model

        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            continue

    # Phase 4: Temporal evaluation
    logger.info("\n" + "="*80)
    logger.info("PHASE 4: TEMPORAL EVALUATION")
    logger.info("="*80)

    temporal_results = {}

    for model_name, model in models_fitted.items():
        try:
            metrics = temporal_evaluation(
                model=model,
                X_train=np.vstack([X_train, X_val]),
                y_train=np.concatenate([y_train, y_val]),
                X_test=X_test,
                y_test=y_test
            )
            temporal_results[model_name] = metrics

        except Exception as e:
            logger.error(f"Error in temporal evaluation for {model_name}: {e}")

    # Phase 5: Model interpretation (optional)
    interpretation_results = {}
    if run_interpretation:
        logger.info("\n" + "="*80)
        logger.info("PHASE 5: MODEL INTERPRETATION")
        logger.info("="*80)

        interpretation_dir = output_dir / "interpretation"

        # Feature importance
        logger.info("Computing feature importance...")
        importance_results = compare_feature_importance(
            models=models_fitted,
            X=X_test,
            feature_names=feature_names,
            output_dir=interpretation_dir
        )
        interpretation_results["importance"] = importance_results

        # Partial dependence plots
        logger.info("Creating partial dependence plots...")
        create_pdp_for_all_models(
            models=models_fitted,
            X=X_test,
            feature_names=feature_names,
            output_dir=interpretation_dir
        )

        # Decision boundaries
        logger.info("Creating decision boundary plots...")
        plot_all_decision_boundaries(
            models=models_fitted,
            X=X_test,
            y=y_test,
            output_dir=interpretation_dir
        )

    # Phase 6: Error analysis (optional)
    analysis_results = {}
    if run_analysis:
        logger.info("\n" + "="*80)
        logger.info("PHASE 6: ERROR ANALYSIS")
        logger.info("="*80)

        analysis_dir = output_dir / "analysis"

        for model_name, model in models_fitted.items():
            logger.info(f"\nAnalyzing errors for {model_name}")

            # Get predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)

            # Get financial mask if phishing_type column exists
            financial_mask = None
            if "phishing_type" in test_df.columns:
                financial_mask = (test_df["phishing_type"] == "financial").values
                logger.info(f"Found {financial_mask.sum()} financial phishing samples in test set")

            # Create error report
            error_report = create_error_report(
                model_name=model_name,
                y_true=y_test,
                y_pred=y_pred,
                y_proba=y_proba,
                df=test_df,
                financial_mask=financial_mask,
                save_path=analysis_dir / f"{model_name}_error_report.csv"
            )

            # Create confusion report with examples
            confusion_examples = create_confusion_report(
                y_true=y_test,
                y_pred=y_pred,
                df=test_df,
                model_name=model_name,
                output_dir=analysis_dir
            )

            analysis_results[model_name] = {
                "error_report": error_report,
                "confusion_examples": confusion_examples
            }

        # Edge case analysis
        logger.info("\nIdentifying edge cases...")
        from src.models.classical.analysis.edge_cases import identify_edge_cases

        edge_cases = identify_edge_cases(
            df=test_df,
            X=X_test,
            y=y_test,
            y_proba=None
        )

        # Save edge case report
        edge_case_report = create_edge_case_report(
            df=test_df,
            X=X_test,
            y=y_test,
            save_path=analysis_dir / "edge_cases_report.csv"
        )

        # Analyze edge case performance
        for model_name, model in models_fitted.items():
            y_pred = model.predict(X_test)

            edge_perf = analyze_edge_case_performance(
                df=test_df,
                y_true=y_test,
                y_pred=y_pred,
                edge_cases=edge_cases
            )

            edge_perf_path = analysis_dir / f"{model_name}_edge_case_performance.csv"
            edge_perf.to_csv(edge_perf_path, index=False)

        # Save edge case samples
        save_edge_case_samples(
            df=test_df,
            edge_cases=edge_cases,
            output_dir=analysis_dir / "edge_case_samples"
        )

    # Phase 7: Create summary report
    logger.info("\n" + "="*80)
    logger.info("PHASE 7: SUMMARY REPORT")
    logger.info("="*80)

    summary = create_summary_report(
        cv_results=cv_results_all,
        temporal_results=temporal_results,
        config=config,
        output_dir=output_dir
    )

    logger.info("\nBenchmark complete!")
    logger.info(f"Results saved to: {output_dir}")

    return {
        "cv_results": cv_results_all,
        "temporal_results": temporal_results,
        "interpretation": interpretation_results,
        "analysis": analysis_results,
        "summary": summary
    }


def create_summary_report(
    cv_results: Dict,
    temporal_results: Dict,
    config: Dict,
    output_dir: Path
) -> pd.DataFrame:
    """
    Create summary report with all results.

    Args:
        cv_results: Cross-validation results
        temporal_results: Temporal evaluation results
        config: Configuration dictionary
        output_dir: Output directory

    Returns:
        Summary DataFrame
    """
    summary_data = []

    for model_name in config["model_names"].keys():
        if model_name not in cv_results:
            continue

        cv_metrics = cv_results[model_name]["aggregated"]
        temporal_metrics = temporal_results.get(model_name, {})

        row = {
            "model": config["model_names"][model_name],
        }

        # Add CV metrics (mean ± std)
        for metric_name, (mean_val, std_val) in cv_metrics.items():
            if isinstance(mean_val, (int, float)):
                row[f"cv_{metric_name}_mean"] = mean_val
                row[f"cv_{metric_name}_std"] = std_val

        # Add temporal metrics
        for metric_name, value in temporal_metrics.items():
            if isinstance(value, (int, float)):
                row[f"temporal_{metric_name}"] = value

        # Add compute stats
        if "training_time" in temporal_metrics:
            row["training_time"] = temporal_metrics["training_time"]
        if "inference_time" in temporal_metrics:
            row["inference_time"] = temporal_metrics["inference_time"]

        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)

    # Save summary
    summary_path = output_dir / "benchmark_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved summary to {summary_path}")

    # Create formatted table for README
    create_results_table(summary_df, output_dir)

    return summary_df


def create_results_table(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create formatted results table for README.

    Args:
        summary_df: Summary DataFrame
        output_dir: Output directory
    """
    # Select key metrics
    key_metrics = ["model", "cv_f1_mean", "cv_f1_std", "cv_auprc_mean", "cv_auprc_std",
                   "cv_auroc_mean", "cv_auroc_std", "cv_fpr_mean", "cv_fpr_std"]

    table_df = summary_df[[col for col in key_metrics if col in summary_df.columns]].copy()

    # Format as markdown table
    table_path = output_dir / "results_table.md"

    with open(table_path, 'w') as f:
        f.write("# Benchmark Results\n\n")
        f.write("## Model Performance Summary\n\n")

        # Header
        f.write("| Model | F1 (mean±std) | AUPRC (mean±std) | AUROC (mean±std) | FPR (mean±std) |\n")
        f.write("|-------|---------------|------------------|------------------|----------------|\n")

        # Rows
        for _, row in table_df.iterrows():
            model_name = row["model"]
            f1 = f"{row['cv_f1_mean']:.4f} ± {row['cv_f1_std']:.4f}"
            auprc = f"{row['cv_auprc_mean']:.4f} ± {row['cv_auprc_std']:.4f}"
            auroc = f"{row['cv_auroc_mean']:.4f} ± {row['cv_auroc_std']:.4f}"
            fpr = f"{row['cv_fpr_mean']:.4f} ± {row['cv_fpr_std']:.4f}"

            f.write(f"| {model_name} | {f1} | {auprc} | {auroc} | {fpr} |\n")

    logger.info(f"Saved results table to {table_path}")


if __name__ == "__main__":
    results = run_benchmark()
