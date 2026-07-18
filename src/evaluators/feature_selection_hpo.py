"""
Feature Selection + Hyperparameter Optimization (FS+HPO) Evaluator.

Combines feature selection (binary mask via threshold) and ML model
hyperparameter tuning into a single multi-objective optimization problem.

Objective 1 (f1): Cross-validated RMSE → minimize
Objective 2 (f2): Fraction of selected features → minimize
                   f2 = N_selected / N_total
"""

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

from .base import BaseEvaluator
from .models import get_model_wrapper, ModelWrapper
from ..utils.encoding import SolutionEncoding
from ..utils.metrics import rmse


def _rmse_scorer(estimator, X, y):
    """Custom RMSE scorer for cross_val_score."""
    from sklearn.metrics import mean_squared_error
    y_pred = estimator.predict(X)
    return -np.sqrt(mean_squared_error(y, y_pred))  # Negative for maximization


class FeatureSelectionHPOEvaluator(BaseEvaluator):
    """
    Multi-objective evaluator for simultaneous feature selection and HPO.

    Parameters
    ----------
    X_train : np.ndarray, shape (N_samples, N_features)
        Training feature matrix.
    y_train : np.ndarray, shape (N_samples,)
        Training target values.
    model_name : str
        Model identifier: 'svr', 'krr', 'random_forest', 'mlp'.
    cv_folds : int
        Number of cross-validation folds (default 5).
    feature_threshold : float
        Threshold for converting continuous values to binary mask (default 0.5).
    n_jobs : int
        Parallel jobs for CV (-1 = all cores).
    """
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_name: str = "svr",
        cv_folds: int = 5,
        feature_threshold: float = 0.5,
        n_jobs: int = -1,
    ):
        super().__init__(n_objectives=2)
        self.X_train = X_train
        self.y_train = y_train
        self.n_samples, self.n_features = X_train.shape
        self.cv_folds = cv_folds
        self.feature_threshold = feature_threshold
        self.n_jobs = n_jobs

        # Load model wrapper
        self.model_wrapper: ModelWrapper = get_model_wrapper(model_name)
        self.hp_bounds = self.model_wrapper.hyperparameter_bounds()
        self.model_name = model_name

        # Initialize solution encoding
        self.encoding = SolutionEncoding(self.n_features, self.hp_bounds)

        # Cache for already-evaluated solutions (optional)
        self._cache: Dict[bytes, np.ndarray] = {}

    def get_dimension(self) -> int:
        return self.encoding.total_dimension

    def decode_solution(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Decode a solution vector."""
        mask, hparams, indices = self.encoding.decode(x)
        return mask, hparams

    def evaluate(self, population: np.ndarray) -> np.ndarray:
        """
        Evaluate a population of solutions.

        For each solution:
        1. Decode feature mask (threshold at 0.5)
        2. Decode hyperparameters
        3. Train model with selected features + hparams
        4. Compute cross-validated RMSE
        5. Return [RMSE, feature_ratio]

        Parameters
        ----------
        population : np.ndarray, shape (N, D)

        Returns
        -------
        fitnesses : np.ndarray, shape (N, 2)
            Column 0: RMSE (minimize), Column 1: feature_ratio (minimize)
        """
        n = population.shape[0]
        fitnesses = np.zeros((n, 2))

        for i in range(n):
            fitnesses[i] = self._evaluate_single(population[i])

        return fitnesses

    def _evaluate_single(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate a single solution vector.

        Edge case: if no features are selected (all mask = False),
        return a large penalty RMSE.

        Returns
        -------
        np.ndarray, shape (2,) : [RMSE, feature_ratio]
        """
        mask, hparams, selected_indices = self.encoding.decode(x)
        n_selected = int(mask.sum())

        # Feature ratio (objective 2)
        feature_ratio = n_selected / self.n_features

        # Edge case: no features selected → large penalty
        if n_selected == 0:
            # Penalty: RMSE = std(y) * 2 (worse than mean predictor)
            penalty_rmse = float(np.std(self.y_train) * 2.0)
            return np.array([penalty_rmse, feature_ratio])

        # Select features
        X_selected = self.X_train[:, selected_indices]

        # Build model
        try:
            model = self.model_wrapper.build_model(**hparams)
        except Exception:
            # Invalid hyperparameter combination → penalty
            return np.array([float(np.std(self.y_train) * 2.0), feature_ratio])

        # Cross-validated RMSE
        try:
            cv_rmse = self._cross_validate(model, X_selected, self.y_train)
        except Exception:
            cv_rmse = float(np.std(self.y_train) * 2.0)

        # Clamp RMSE to reasonable range
        cv_rmse = float(np.clip(cv_rmse, 0.0, np.std(self.y_train) * 5.0))

        return np.array([cv_rmse, feature_ratio])

    def _cross_validate(self, model, X: np.ndarray, y: np.ndarray) -> float:
        """
        Perform k-fold cross-validation and return mean RMSE.

        For very small datasets (n < cv_folds * 2), automatically
        reduces the number of folds.
        """
        n_samples = X.shape[0]
        effective_folds = min(self.cv_folds, n_samples // 2)
        effective_folds = max(2, effective_folds)  # At least 2 folds

        if n_samples < 3:
            # Too few samples: train on all, return training RMSE
            model.fit(X, y)
            y_pred = model.predict(X)
            return rmse(y, y_pred)

        scores = cross_val_score(
            model,
            X,
            y,
            cv=effective_folds,
            scoring="neg_root_mean_squared_error",
            n_jobs=self.n_jobs,
            error_score=np.nan,
        )
        # neg_root_mean_squared_error → positive RMSE
        valid_scores = scores[~np.isnan(scores)]
        if len(valid_scores) == 0:
            return float(np.std(y) * 2.0)
        return float(-np.mean(valid_scores))

    def get_dimension_info(self) -> Dict:
        """Return information about the problem encoding."""
        return {
            **self.encoding.get_dimension_info(),
            "model": self.model_name,
            "n_samples": self.n_samples,
            "cv_folds": self.cv_folds,
        }
