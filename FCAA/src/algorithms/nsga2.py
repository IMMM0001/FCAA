"""
NSGA-II (Non-dominated Sorting Genetic Algorithm II).

Standard real-coded NSGA-II implementation for comparison against FCAA.
Uses SBX crossover and polynomial mutation.

Reference: Deb et al., "A Fast and Elitist Multiobjective Genetic
Algorithm: NSGA-II", IEEE TEC, 2002.
"""

from typing import Callable, List, Optional, Tuple

import numpy as np

from .base import BaseOptimizer
from .operators import sbx_crossover, polynomial_mutation
from .multi_objective import select_survivors, get_pareto_front, non_dominated_sort


class NSGA2Optimizer(BaseOptimizer):
    """
    NSGA-II with real-coded GA operators.

    Parameters
    ----------
    dimension : int
        Solution vector dimensionality.
    pop_size : int
        Population size (should be even).
    max_generations : int
        Maximum generations.
    fitness_fn : callable
        Evaluates population → fitness array (N, M).
    crossover_prob : float
        SBX crossover probability (default 0.9).
    mutation_prob : float
        Polynomial mutation probability (per gene, default 1/D).
    eta_crossover : float
        SBX distribution index (default 20.0).
    eta_mutation : float
        Polynomial mutation distribution index (default 20.0).
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
        crossover_prob: float = 0.9,
        mutation_prob: Optional[float] = None,
        eta_crossover: float = 20.0,
        eta_mutation: float = 20.0,
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
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob or (1.0 / dimension)
        self.eta_crossover = eta_crossover
        self.eta_mutation = eta_mutation

    def optimize(
        self, verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run NSGA-II optimization."""
        # Initialize population
        self.population = self._initialize_population()
        self.fitnesses = self.fitness_fn(self.population)
        self.fitness_history = [self.fitnesses.copy()]

        # Ensure even population size for pairwise crossover
        if self.pop_size % 2 != 0:
            self.pop_size += 1
            extra = self._initialize_population()
            extra = extra[:1]
            self.population = np.vstack([self.population, extra])
            extra_fit = self.fitness_fn(extra)
            self.fitnesses = np.vstack([self.fitnesses, extra_fit])

        for gen in range(self.max_generations):
            # ---- Tournament selection + SBX crossover + mutation ----
            offspring = self._create_offspring()

            # Evaluate offspring
            offspring_fitnesses = self.fitness_fn(offspring)

            # ---- Mu + Lambda selection ----
            combined_pop = np.vstack([self.population, offspring])
            combined_fitnesses = np.vstack([self.fitnesses, offspring_fitnesses])

            self.population, self.fitnesses, _ = select_survivors(
                combined_pop, combined_fitnesses, self.pop_size
            )

            self.fitness_history.append(self.fitnesses.copy())

            if verbose and (gen + 1) % max(1, self.max_generations // 10) == 0:
                pareto_pop, pareto_fit = get_pareto_front(
                    self.population, self.fitnesses
                )
                print(
                    f"  Gen {gen + 1:4d}/{self.max_generations} | "
                    f"Pareto size: {len(pareto_fit)} | "
                    f"RMSE range: [{pareto_fit[:, 0].min():.4f}, "
                    f"{pareto_fit[:, 0].max():.4f}]"
                )

        pareto_population, pareto_fitnesses = get_pareto_front(
            self.population, self.fitnesses
        )

        if verbose:
            print(f"\nNSGA-II finished. Pareto front size: {len(pareto_fitnesses)}")

        return pareto_population, pareto_fitnesses

    def _create_offspring(self) -> np.ndarray:
        """Generate offspring via tournament selection, SBX, and mutation."""
        n = len(self.population)
        offspring = np.zeros_like(self.population)

        for i in range(0, n, 2):
            # Binary tournament selection
            parent1_idx = self._tournament_select()
            parent2_idx = self._tournament_select()

            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]

            # SBX crossover
            if self.rng.random() < self.crossover_prob:
                child1, child2 = sbx_crossover(
                    parent1, parent2, eta=self.eta_crossover, rng=self.rng
                )
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Polynomial mutation
            child1 = polynomial_mutation(
                child1, eta_mut=self.eta_mutation,
                mutation_rate=self.mutation_prob, rng=self.rng
            )
            child2 = polynomial_mutation(
                child2, eta_mut=self.eta_mutation,
                mutation_rate=self.mutation_prob, rng=self.rng
            )

            # Clip to bounds
            child1 = self._clip(child1)
            child2 = self._clip(child2)

            offspring[i] = child1
            if i + 1 < n:
                offspring[i + 1] = child2

        return offspring

    def _tournament_select(self) -> int:
        """Binary tournament selection using Pareto rank + crowding."""
        k1 = int(self.rng.integers(0, len(self.population)))
        k2 = int(self.rng.integers(0, len(self.population)))

        f1, f2 = self.fitnesses[k1], self.fitnesses[k2]

        # Determine dominance
        if np.all(f1 <= f2) and np.any(f1 < f2):
            return k1
        elif np.all(f2 <= f1) and np.any(f2 < f1):
            return k2
        else:
            # Non-dominated: randomly pick
            return int(k1 if self.rng.random() < 0.5 else k2)
