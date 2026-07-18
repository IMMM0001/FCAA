"""
Data loading and preprocessing utilities.

Provides functions to load datasets, generate synthetic data for testing,
and perform train/test splitting with proper scaling.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression, make_friedman1


def generate_synthetic_data(
    n_samples: int = 200,
    n_features: int = 50,
    n_informative: int = 10,
    noise: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a synthetic regression dataset with many irrelevant features.

    Simulates the small-sample, high-dimensional scenario typical in
    materials informatics.

    Parameters
    ----------
    n_samples : int
        Number of samples (keep small to simulate data scarcity).
    n_features : int
        Total number of features (many are noise).
    n_informative : int
        Number of truly informative features.
    noise : float
        Gaussian noise standard deviation.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
        Train/test split of features and targets.
    """
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=random_state,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def generate_friedman_data(
    n_samples: int = 200,
    n_features: int = 50,
    noise: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a Friedman #1 dataset — a nonlinear regression benchmark.

    Only 5 features are actually relevant; the rest are noise.
    Good for testing feature selection quality.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_features : int
        Total features (first 5 are informative).
    noise : float
        Noise level.
    random_state : int
        Random seed.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X, y = make_friedman1(
        n_samples=n_samples, n_features=n_features, noise=noise, random_state=random_state
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def load_csv_data(
    filepath: str,
    target_column: str = "target",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a dataset from CSV file.

    Parameters
    ----------
    filepath : str
        Path to CSV file.
    target_column : str
        Name of the target/response column.
    test_size : float
        Fraction for test split.
    random_state : int
        Random seed.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    df = pd.read_csv(filepath)
    y = df[target_column].values
    X = df.drop(columns=[target_column]).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def get_cv_folds(
    X: np.ndarray, y: np.ndarray, n_folds: int = 5, random_state: int = 42
):
    """
    Create cross-validation fold splits.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    n_folds : int
        Number of CV folds.
    random_state : int
        Random seed.

    Yields
    ------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (X_train_fold, X_val_fold, y_train_fold, y_val_fold) for each fold.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for train_idx, val_idx in kf.split(X):
        yield X[train_idx], X[val_idx], y[train_idx], y[val_idx]
