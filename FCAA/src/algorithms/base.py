"""
Abstract base class for all multi-objective optimization algorithms.

Provides the common interface that FCAA, NSGA-II, and MOPSO implement.
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


class BaseOptimizer(ABC):
    """
    Abstract base class for multi-objective optimizers.

    Parameters
    ----------
    dimension : int
        Dimensionality of the search space (solution vector length).
    pop_size : int
        Population size.
    max_generations : int
        Maximum number of generations/iterations.
    fitness_fn : callable
        Function that evaluates a population and returns fitness values.
        Signature: fitness_fn(population: np.ndarray) -> np.ndarray of shape (N, M)
    lower_bound : float, optional
        Lower bound for each dimension (default 0.0).
    upper_bound : float, optional
        Upper bound for each dimension (default 1.0).
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        dimension: int,
        pop_size: int,
        max_generations: int,
        fitness_fn: Callable[[np.ndarray], np.ndarray],
        lower_bound: float = 0.0,
        upper_bound: float = 1.0,
        seed: Optional[int] = 42,
    ):
        self.dimension = dimension
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.fitness_fn = fitness_fn
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.rng = np.random.default_rng(seed)

        # History tracking
        self.fitness_history: List[np.ndarray] = []
        self.population: Optional[np.ndarray] = None
        self.fitnesses: Optional[np.ndarray] = None

    @abstractmethod
    def optimize(
        self, verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the optimization and return the final Pareto front.

        Parameters
        ----------
        verbose : bool
            Print progress information.

        Returns
        -------
        pareto_population : np.ndarray, shape (K, D)
            Pareto-optimal solutions.
        pareto_fitnesses : np.ndarray, shape (K, M)
            Objective values of the Pareto-optimal solutions.
        """
        pass

    def _initialize_population(self) -> np.ndarray:
        """Generate initial random population."""
        return self.rng.uniform(
            self.lower_bound,
            self.upper_bound,
            size=(self.pop_size, self.dimension),
        )

    def _clip(self, x: np.ndarray) -> np.ndarray:
        """Clip solution to bounds."""
        return np.clip(x, self.lower_bound, self.upper_bound)

    def get_history(self) -> Dict[str, List]:
        """Return optimization history for analysis."""
        return {
            "fitness_history": self.fitness_history,
        }
