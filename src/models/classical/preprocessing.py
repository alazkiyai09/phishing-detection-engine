"""
Preprocessing pipeline for phishing classifier benchmark.

Loads features from Day 1 pipeline and prepares train/validation/test splits
for fair model comparison.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import logging

from src.models.classical.config import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_features(dataset_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load features from Day 1 pipeline or generated CSV.

    Args:
        dataset_path: Path to features CSV. If None, uses default from config.

    Returns:
        DataFrame with features and labels
    """
    config = get_config()

    if dataset_path is None:
        # Check for sample features first
        sample_path = config["data_dir"] / "sample_features.csv"
        if sample_path.exists():
            dataset_path = sample_path
            logger.info(f"Using sample features from {dataset_path}")
        else:
            # Fall back to config path
            dataset_path = config["day1_features_path"]

    # Handle directory vs file path
    if dataset_path.is_dir():
        # It's the Day 1 project directory
        logger.info(f"Day 1 project directory: {dataset_path}")
        raise FileNotFoundError(
            f"Path {dataset_path} is a directory. "
            f"Please run 'python generate_sample_features.py' first to create sample features, "
            f"or provide a path to a CSV file with extracted features."
        )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Features file not found at {dataset_path}. "
            f"Run 'python generate_sample_features.py' to create sample features."
        )

    logger.info(f"Loading features from {dataset_path}")
    df = pd.read_csv(dataset_path)
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")

    return df


def separate_features_target(
    df: pd.DataFrame,
    target_col: str = "label"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target from dataset.

    Args:
        df: Input DataFrame
        target_col: Name of target column

    Returns:
        Tuple of (features DataFrame, target Series)
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Store metadata columns for analysis
    metadata_cols = []
    potential_metadata = ["date", "phishing_type", "email_id", "subject", "body"]
    for col in potential_metadata:
        if col in X.columns:
            metadata_cols.append(col)

    X_features = X.drop(columns=metadata_cols)

    logger.info(f"Features shape: {X_features.shape}, Target shape: {y.shape}")
    logger.info(f"Class distribution:\n{y.value_counts()}")

    return X_features, y


def handle_missing_values(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    strategy: str = "median"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, SimpleImputer]:
    """
    Handle missing values using imputation.

    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features
        strategy: Imputation strategy ('mean', 'median', 'most_frequent', 'constant')

    Returns:
        Tuple of imputed arrays and fitted imputer
    """
    logger.info(f"Imputing missing values with strategy: {strategy}")

    imputer = SimpleImputer(strategy=strategy)
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)
    X_test_imputed = imputer.transform(X_test)

    missing_before = X_train.isna().sum().sum()
    logger.info(f"Missing values before imputation: {missing_before}")

    return X_train_imputed, X_val_imputed, X_test_imputed, imputer


def scale_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    method: str = "standard"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Scale features for models that require normalization.

    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features
        method: Scaling method ('standard', 'minmax', 'robust')

    Returns:
        Tuple of scaled arrays and fitted scaler
    """
    logger.info(f"Scaling features with method: {method}")

    if method == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError(f"Scaling method '{method}' not implemented")

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    logger.info(f"Feature scaling complete. Train mean: {X_train_scaled.mean():.4f}, std: {X_train_scaled.std():.4f}")

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def create_splits(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.2,
    temporal_col: Optional[str] = None,
    target_col: str = "label",
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create train/validation/test splits.

    Supports both random and temporal splitting strategies.

    Args:
        df: Input DataFrame
        test_size: Fraction of data for test set
        val_size: Fraction of training data for validation
        temporal_col: Column name for temporal sorting (if None, uses random split)
        target_col: Name of target column
        random_state: Random seed

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info(f"Creating splits - test_size={test_size}, val_size={val_size}")

    if temporal_col and temporal_col in df.columns:
        # Temporal split: oldest data for training, newest for testing
        logger.info(f"Using temporal split with column: {temporal_col}")

        # Verify temporal column is valid
        if df[temporal_col].isna().any():
            raise ValueError(f"Temporal column '{temporal_col}' contains NaN values")

        # Sort by date
        df_sorted = df.sort_values(temporal_col).reset_index(drop=True)

        n_test = int(len(df_sorted) * test_size)
        n_val = int(len(df_sorted) * (1 - test_size) * val_size)

        test_df = df_sorted.iloc[-n_test:]
        train_val_df = df_sorted.iloc[:-n_test]
        val_df = train_val_df.iloc[-n_val:]
        train_df = train_val_df.iloc[:-n_val]

        # Validate no temporal overlap
        train_max = train_df[temporal_col].max()
        val_min = val_df[temporal_col].min()
        val_max = val_df[temporal_col].max()
        test_min = test_df[temporal_col].min()

        if train_max >= val_min:
            logger.warning(f"Train/Val temporal overlap: train_max={train_max} >= val_min={val_min}")
        if val_max >= test_min:
            logger.warning(f"Val/Test temporal overlap: val_max={val_max} >= test_min={test_min}")

        logger.info(f"Temporal split date ranges:")
        logger.info(f"  Train: {train_df[temporal_col].min()} to {train_df[temporal_col].max()}")
        logger.info(f"  Val:   {val_df[temporal_col].min()} to {val_df[temporal_col].max()}")
        logger.info(f"  Test:  {test_df[temporal_col].min()} to {test_df[temporal_col].max()}")

    else:
        # Stratified random split
        logger.info("Using stratified random split")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # First split: separate test set
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )

        # Second split: separate train and validation from remaining
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_size,
            stratify=y_train_val,
            random_state=random_state
        )

        # Recombine with labels
        train_df = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
        val_df = pd.concat([X_val, y_val], axis=1).reset_index(drop=True)
        test_df = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)

    # Log split sizes and class distributions
    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        logger.info(f"{split_name}: n={len(split_df)}, "
                   f"class_dist={split_df[target_col].value_counts().to_dict()}")

    return train_df, val_df, test_df


def prepare_data_for_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "label",
    scale: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Complete preprocessing pipeline for a single model.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        target_col: Name of target column
        scale: Whether to apply feature scaling

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, feature_names)
    """
    # Separate features and targets
    X_train_df, y_train = separate_features_target(train_df, target_col)
    X_val_df, y_val = separate_features_target(val_df, target_col)
    X_test_df, y_test = separate_features_target(test_df, target_col)

    feature_names = X_train_df.columns.tolist()

    # Handle missing values
    X_train, X_val, X_test, _ = handle_missing_values(
        X_train_df, X_val_df, X_test_df
    )

    # Scale features if requested
    if scale:
        X_train, X_val, X_test, _ = scale_features(
            X_train, X_val, X_test
        )

    # Convert targets to numpy arrays
    y_train = y_train.values
    y_val = y_val.values
    y_test = y_test.values

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names


def get_phishing_subsets(
    df: pd.DataFrame,
    phishing_type_col: str
) -> Dict[str, pd.DataFrame]:
    """
    Get subsets of data by phishing type for detailed analysis.

    Args:
        df: Input DataFrame
        phishing_type_col: Column containing phishing type labels

    Returns:
        Dictionary mapping phishing types to DataFrames
    """
    config = get_config()
    subsets = {}

    for ptype in config["phishing_types"]:
        subset = df[df[phishing_type_col] == ptype]
        subsets[ptype] = subset
        logger.info(f"{ptype.capitalize()} phishing: n={len(subset)}")

    # Also get generic phishing subset (all phishing)
    subsets["all_phishing"] = df[df["label"] == 1]
    subsets["legitimate"] = df[df["label"] == 0]

    return subsets
