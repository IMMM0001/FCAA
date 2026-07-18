"""
MOPSO (Multi-Objective Particle Swarm Optimization).

Standard MOPSO implementation for comparison against FCAA.
Uses an external archive with crowding-distance-based truncation
and adaptive mutation for diversity maintenance.

Reference: Coello Coello & Lechuga, "MOPSO: A Proposal for Multiple
Objective Particle Swarm Optimization", CEC 2002.
"""

from typing import Callable, List, Optional, Tuple

import numpy as np

from .base import BaseOptimizer
from .multi_objective import (
    non_dominated_sort,
    crowding_distance,
    get_pareto_front,
)


class MOPSOOptimizer(BaseOptimizer):
    """
    Multi-Objective Particle Swarm Optimization.

    Parameters
    ----------
    dimension : int
        Solution vector dimensionality.
    pop_size : int
        Population (swarm) size.
    max_generations : int
        Maximum iterations.
    fitness_fn : callable
        Evaluates population → fitness array (N, M).
    archive_size : int
        Maximum size of the external Pareto archive (default = pop_size).
    inertia : float
        Inertia weight for velocity update (default 0.5).
    cognitive : float
        Cognitive (personal best) coefficient (default 1.5).
    social : float
        Social (global best) coefficient (default 1.5).
    mutation_rate : float
        Mutation probability for diversity (default 0.1).
    lower_bound : float
        Search space lower bound.
    upper_bound : float
        Search space upper bound.
    seed : int, optional
        Random seed.
    """

    def __init__(
        self,
        dimension: int,
        pop_size: int,
        max_generations: int,
        fitness_fn: Callable[[np.ndarray], np.ndarray],
        archive_size: Optional[int] = None,
        inertia: float = 0.5,
        cognitive: float = 1.5,
        social: float = 1.5,
        mutation_rate: float = 0.1,
        lower_bound: float = 0.0,
        upper_bound: float = 1.0,
        seed: Optional[int] = 42,
    ):
        super().__init__(
            dimension=dimension,
            pop_size=pop_size,
            max_generations=max_generations,
            fitness_fn=fitness_fn,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            seed=seed,
        )
        self.archive_size = archive_size or pop_size
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.mutation_rate = mutation_rate

        # Swarm state
        self.velocities: Optional[np.ndarray] = None
        self.personal_best: Optional[np.ndarray] = None
        self.personal_best_fitness: Optional[np.ndarray] = None
        self.archive: Optional[np.ndarray] = None
        self.archive_fitness: Optional[np.ndarray] = None

    def optimize(
        self, verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run MOPSO optimization."""
        # Initialize swarm
        self.population = self._initialize_population()
        self.fitnesses = self.fitness_fn(self.population)
        self.fitness_history = [self.fitnesses.copy()]

        # Initialize velocities
        v_max = 0.2 * (self.upper_bound - self.lower_bound)
        self.velocities = self.rng.uniform(
            -v_max, v_max, size=(self.pop_size, self.dimension)
        )

        # Initialize personal bests
        self.personal_best = self.population.copy()
        self.personal_best_fitness = self.fitnesses.copy()

        # Initialize external archive with non-dominated solutions
        self._update_archive(self.population, self.fitnesses)

        for gen in range(self.max_generations):
            # Adaptive inertia (linearly decreasing)
            w = self.inertia * (1.0 - gen / self.max_generations)

            for i in range(self.pop_size):
                # Select global best from archive
                if self.archive is not None and len(self.archive) > 0:
                    leader = self._select_leader()
                else:
                    leader = self.personal_best[i]

                # Velocity update
                r1 = self.rng.random(self.dimension)
                r2 = self.rng.random(self.dimension)

                cognitive_vel = (
                    self.cognitive * r1 * (self.personal_best[i] - self.population[i])
                )
                social_vel = (
                    self.social * r2 * (leader - self.population[i])
                )

                self.velocities[i] = (
                    w * self.velocities[i] + cognitive_vel + social_vel
                )

                # Clamp velocity
                self.velocities[i] = np.clip(self.velocities[i], -v_max, v_max)

                # Position update
                self.population[i] = self.population[i] + self.velocities[i]
                self.population[i] = self._clip(self.population[i])

                # Mutation for diversity
                if self.rng.random() < self.mutation_rate:
                    dim = int(self.rng.integers(0, self.dimension))
                    self.population[i, dim] = self.rng.uniform(
                        self.lower_bound, self.upper_bound
                    )

            # Evaluate population
            self.fitnesses = self.fitness_fn(self.population)

            # Update personal bests
            for i in range(self.pop_size):
                if self._dominates(self.fitnesses[i], self.personal_best_fitness[i]):
                    self.personal_best[i] = self.population[i].copy()
                    self.personal_best_fitness[i] = self.fitnesses[i].copy()

            # Update archive
            self._update_archive(self.population, self.fitnesses)

            self.fitness_history.append(self.archive_fitness.copy()
                                        if self.archive_fitness is not None
                                        else self.fitnesses.copy())

            if verbose and (gen + 1) % max(1, self.max_generations // 10) == 0:
                n_archive = len(self.archive_fitness) if self.archive_fitness is not None else 0
                print(
                    f"  Gen {gen + 1:4d}/{self.max_generations} | "
                    f"Archive size: {n_archive}"
                )

        # Return archive as final Pareto front
        if self.archive is not None and len(self.archive) > 0:
            final_pop, final_fit = self.archive, self.archive_fitness
        else:
            final_pop, final_fit = get_pareto_front(
                self.population, self.fitnesses
            )

        if verbose:
            print(f"\nMOPSO finished. Pareto front size: {len(final_fit)}")

        return final_pop, final_fit

    def _update_archive(self, population: np.ndarray, fitnesses: np.ndarray):
        """
        Update the external archive with new non-dominated solutions.

        Strategy:
        1. Add all new solutions to the archive
        2. Remove dominated solutions
        3. If archive exceeds max size, truncate by crowding distance
        """
        if self.archive is None:
            # First update: find Pareto front and add
            pareto_mask = self._find_pareto_mask(fitnesses)
            self.archive = population[pareto_mask].copy()
            self.archive_fitness = fitnesses[pareto_mask].copy()
            return

        # Combine archive with new population
        combined_pop = np.vstack([self.archive, population])
        combined_fit = np.vstack([self.archive_fitness, fitnesses])

        # Keep only non-dominated
        pareto_mask = self._find_pareto_mask(combined_fit)
        new_archive_pop = combined_pop[pareto_mask]
        new_archive_fit = combined_fit[pareto_mask]

        # Truncate if needed
        if len(new_archive_pop) > self.archive_size:
            cd = crowding_distance(new_archive_fit)
            keep_idx = np.argsort(-cd)[:self.archive_size]
            self.archive = new_archive_pop[keep_idx]
            self.archive_fitness = new_archive_fit[keep_idx]
        else:
            self.archive = new_archive_pop
            self.archive_fitness = new_archive_fit

    def _select_leader(self) -> np.ndarray:
        """
        Select a global best from the archive.

        Strategy: roulette wheel based on crowding distance —
        solutions in less crowded regions are more likely to be chosen.
        """
        cd = crowding_distance(self.archive_fitness)
        # Handle inf values and edge case where all are inf
        finite_mask = ~np.isinf(cd)
        if finite_mask.any():
            max_finite = np.max(cd[finite_mask])
            cd_finite = np.where(np.isinf(cd), max_finite * 2.0, cd)
        else:
            # All solutions are boundary — uniform selection
            cd_finite = np.ones(len(cd))
        cd_finite = np.maximum(cd_finite, 1e-10)
        probs = cd_finite / cd_finite.sum()
        idx = int(self.rng.choice(len(self.archive), p=probs))
        return self.archive[idx]

    @staticmethod
    def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
        """Check if a dominates b (minimization)."""
        return bool(np.all(a <= b) and np.any(a < b))

    @staticmethod
    def _find_pareto_mask(fitnesses: np.ndarray) -> np.ndarray:
        """Find boolean mask of non-dominated solutions."""
        n = fitnesses.shape[0]
        mask = np.ones(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i == j or not mask[i]:
                    continue
                if MOPSOOptimizer._dominates(fitnesses[j], fitnesses[i]):
                    mask[i] = False
                    break
        return mask
