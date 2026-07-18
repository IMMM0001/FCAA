"""
Abstract base class for multi-objective fitness evaluators.

Defines the interface that all problem-specific evaluators must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np


class BaseEvaluator(ABC):
    """
    Abstract evaluator for multi-objective optimization.

    An evaluator takes a population of solution vectors and returns
    a fitness matrix of shape (N, M) where N = population size and
    M = number of objectives.

    For the feature-selection + HPO problem:
    - Objective 1: RMSE (minimize)
    - Objective 2: Feature ratio (minimize)
    """

    def __init__(self, n_objectives: int = 2):
        self.n_objectives = n_objectives
        self.n_evaluations = 0  # Track number of fitness evaluations

    @abstractmethod
    def evaluate(self, population: np.ndarray) -> np.ndarray:
        """
        Evaluate a population of solution vectors.

        Parameters
        ----------
        population : np.ndarray, shape (N, D)
            Solution vectors in [0, 1] space.

        Returns
        -------
        fitnesses : np.ndarray, shape (N, M)
            Objective values. All objectives are MINIMIZED.
        """
        pass

    def __call__(self, population: np.ndarray) -> np.ndarray:
        """Allow evaluator to be called as a function."""
        self.n_evaluations += len(population)
        return self.evaluate(population)

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the total solution vector dimension."""
        pass

    @abstractmethod
    def decode_solution(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Decode a single solution vector.

        Parameters
        ----------
        x : np.ndarray, shape (D,)

        Returns
        -------
        feature_mask : np.ndarray, shape (N_features,), dtype bool
        hyperparameters : dict
        """
        pass
