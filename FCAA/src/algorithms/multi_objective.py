"""
Multi-objective selection mechanisms based on NSGA-II.

Implements:
- Non-dominated sorting: partitions population into Pareto fronts
- Crowding distance: measures solution density for diversity preservation
- Survivor selection: fills next generation front-by-front
"""

from typing import List, Tuple

import numpy as np


def non_dominated_sort(
    fitnesses: np.ndarray,
) -> List[np.ndarray]:
    """
    Non-dominated sorting (NSGA-II algorithm).

    Partitions the population into successive Pareto fronts.
    All objectives are minimized.

    Reference: Deb et al., "A Fast and Elitist Multiobjective Genetic
    Algorithm: NSGA-II", IEEE TEC, 2002.

    Parameters
    ----------
    fitnesses : np.ndarray, shape (N, M)
        Objective values. Each row is one solution, each column one objective.
        All objectives are assumed to be MINIMIZED.

    Returns
    -------
    fronts : list of np.ndarray
        Each element is a 1D array of indices belonging to that front.
        fronts[0] = Pareto front (rank 0), fronts[1] = rank 1, etc.
    """
    n = fitnesses.shape[0]

    # domination_counts[i] = number of solutions that dominate i
    domination_counts = np.zeros(n, dtype=int)
    # dominated_sets[i] = list of solutions that i dominates
    dominated_sets = [[] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Check if i dominates j: i <= j in all AND i < j in at least one
            if _dominates(fitnesses[i], fitnesses[j]):
                dominated_sets[i].append(j)
            elif _dominates(fitnesses[j], fitnesses[i]):
                domination_counts[i] += 1

    fronts = []
    # Front 0: solutions with domination_count == 0
    current_front = np.where(domination_counts == 0)[0]

    while len(current_front) > 0:
        fronts.append(current_front)
        next_front = []
        for i in current_front:
            for j in dominated_sets[i]:
                domination_counts[j] -= 1
                if domination_counts[j] == 0:
                    next_front.append(j)
        current_front = np.array(next_front, dtype=int)

    return fronts


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Check if a dominates b (minimization)."""
    return bool(np.all(a <= b) and np.any(a < b))


def crowding_distance(
    fitnesses: np.ndarray,
) -> np.ndarray:
    """
    Compute crowding distance for each solution in a front.

    Crowding distance measures how isolated a solution is — larger values
    mean fewer neighbors, which is preferred for diversity.

    Parameters
    ----------
    fitnesses : np.ndarray, shape (N, M)
        Objective values of solutions in the SAME front.

    Returns
    -------
    np.ndarray, shape (N,)
        Crowding distance for each solution. Boundary solutions get
        infinite distance to ensure they're always kept.
    """
    n, n_obj = fitnesses.shape

    if n <= 2:
        # All solutions are on the boundary
        return np.full(n, np.inf)

    distances = np.zeros(n)

    for obj in range(n_obj):
        # Sort by this objective
        sorted_idx = np.argsort(fitnesses[:, obj])
        sorted_f = fitnesses[sorted_idx, obj]

        f_min, f_max = sorted_f[0], sorted_f[-1]
        if f_max - f_min < 1e-12:
            continue  # All values equal, skip

        # Boundaries get infinite distance
        distances[sorted_idx[0]] = np.inf
        distances[sorted_idx[-1]] = np.inf

        # Interior points: normalized neighbor distance
        for i in range(1, n - 1):
            distances[sorted_idx[i]] += (
                sorted_f[i + 1] - sorted_f[i - 1]
            ) / (f_max - f_min)

    return distances


def select_survivors(
    population: np.ndarray,
    fitnesses: np.ndarray,
    pop_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    NSGA-II survivor selection.

    Fills the next generation front-by-front. If a front doesn't fit
    entirely, selects the least crowded solutions from that front.

    Parameters
    ----------
    population : np.ndarray, shape (2N or N, D)
        Combined parent + offspring population (mu + lambda).
    fitnesses : np.ndarray, shape (2N or N, M)
        Objective values corresponding to population.
    pop_size : int
        Target population size for next generation.

    Returns
    -------
    survivors : np.ndarray, shape (pop_size, D)
        Selected survivors.
    survivor_fitnesses : np.ndarray, shape (pop_size, M)
        Fitnesses of survivors.
    ranks : np.ndarray, shape (pop_size,)
        Pareto front rank of each survivor (0 = best).
    """
    fronts = non_dominated_sort(fitnesses)

    survivors = []
    survivor_fitnesses = []
    survivor_ranks = []

    for rank, front in enumerate(fronts):
        if len(survivors) + len(front) <= pop_size:
            # Entire front fits
            survivors.extend(front)
            survivor_ranks.extend([rank] * len(front))
        else:
            # Partial front: select by crowding distance
            remaining = pop_size - len(survivors)
            front_fitnesses = fitnesses[front]
            cd = crowding_distance(front_fitnesses)
            # Select `remaining` solutions with highest crowding distance
            sorted_by_cd = front[np.argsort(-cd)]  # descending
            selected = sorted_by_cd[:remaining]
            survivors.extend(selected)
            survivor_ranks.extend([rank] * len(selected))
            break

    survivors = np.array(survivors, dtype=int)
    return (
        population[survivors],
        fitnesses[survivors],
        np.array(survivor_ranks),
    )


def get_pareto_front(
    population: np.ndarray,
    fitnesses: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the Pareto-optimal (rank-0) solutions.

    Parameters
    ----------
    population : np.ndarray
    fitnesses : np.ndarray

    Returns
    -------
    pareto_population, pareto_fitnesses
    """
    fronts = non_dominated_sort(fitnesses)
    if len(fronts) == 0:
        return np.array([]), np.array([])
    pareto_idx = fronts[0]
    return population[pareto_idx], fitnesses[pareto_idx]
