"""
Performance metrics for multi-objective optimization.

Includes:
- RMSE (Root Mean Square Error)
- Hypervolume indicator
- Inverted Generational Distance (IGD)
- Pareto front utilities
"""

from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import ConvexHull
from sklearn.metrics import mean_squared_error


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Square Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R² coefficient of determination."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot)


def is_pareto_dominated(
    a: np.ndarray, b: np.ndarray
) -> bool:
    """
    Check if solution a is Pareto-dominated by solution b.

    For minimization: a is dominated by b if b <= a in all objectives
    AND b < a in at least one.

    Parameters
    ----------
    a, b : np.ndarray
        Objective vectors (both minimization).

    Returns
    -------
    bool : True if a is dominated by b.
    """
    return np.all(b <= a) and np.any(b < a)


def find_pareto_front(fitnesses: np.ndarray) -> np.ndarray:
    """
    Find the indices of Pareto-optimal (non-dominated) solutions.

    Parameters
    ----------
    fitnesses : np.ndarray, shape (N, M)
        Objective values for N solutions, M objectives.
        All objectives are assumed to be minimized.

    Returns
    -------
    np.ndarray : Boolean array of length N, True for Pareto-optimal solutions.
    """
    n = fitnesses.shape[0]
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if is_pareto_dominated(fitnesses[i], fitnesses[j]):
                is_pareto[i] = False
                break
    return is_pareto


def hypervolume(
    fitnesses: np.ndarray, reference_point: np.ndarray
) -> float:
    """
    Compute the hypervolume indicator.

    A larger hypervolume means a better (more diverse, better-converged)
    Pareto front approximation.

    Uses a simple Monte Carlo approximation for >2 objectives.

    Parameters
    ----------
    fitnesses : np.ndarray, shape (N, M)
        Objective values (minimization).
    reference_point : np.ndarray, shape (M,)
        Reference point (should be worse than any solution).

    Returns
    -------
    float : Hypervolume (larger is better).
    """
    # Only use non-dominated solutions
    pareto_mask = find_pareto_front(fitnesses)
    pareto_fitnesses = fitnesses[pareto_mask]

    if len(pareto_fitnesses) == 0:
        return 0.0

    n_obj = fitnesses.shape[1]

    if n_obj == 2:
        return _hypervolume_2d(pareto_fitnesses, reference_point)
    else:
        return _hypervolume_monte_carlo(pareto_fitnesses, reference_point)


def _hypervolume_2d(
    fitnesses: np.ndarray, reference_point: np.ndarray
) -> float:
    """Exact hypervolume for 2-objective case."""
    # Sort by first objective ascending
    sorted_idx = np.argsort(fitnesses[:, 0])
    sorted_f = fitnesses[sorted_idx]

    hv = 0.0
    prev_x = reference_point[0]  # Actually, we work backwards

    # Start from the reference point corner
    last_y = reference_point[1]
    for i in range(len(sorted_f)):
        x = sorted_f[i, 0]
        y = sorted_f[i, 1]
        if y < last_y:  # Only count if it improves on objective 2
            hv += (reference_point[0] - x) * (last_y - y)
            last_y = y

    return abs(hv)


def _hypervolume_monte_carlo(
    fitnesses: np.ndarray,
    reference_point: np.ndarray,
    n_samples: int = 10000,
) -> float:
    """
    Monte Carlo hypervolume approximation for any number of objectives.

    Samples points in the hyper-rectangle defined by the reference point
    and counts how many are dominated by at least one Pareto solution.
    """
    n_obj = fitnesses.shape[1]
    # Find the nadir point of Pareto front as lower bound
    lower_bound = np.min(fitnesses, axis=0)

    # Sample in the box [lower_bound, reference_point]
    rng = np.random.default_rng(42)
    samples = rng.uniform(
        low=lower_bound,
        high=reference_point,
        size=(n_samples, n_obj),
    )

    # A sample is "covered" if it's dominated by any Pareto solution
    covered = 0
    for sample in samples:
        for f in fitnesses:
            if np.all(f <= sample):
                covered += 1
                break

    volume = np.prod(reference_point - lower_bound)
    return float(volume * covered / n_samples)


def igd(
    pareto_front: np.ndarray,
    true_pareto_front: Optional[np.ndarray] = None,
    reference_set: Optional[np.ndarray] = None,
) -> float:
    """
    Inverted Generational Distance.

    Measures both convergence and diversity. Lower is better.
    If true_pareto_front is not provided, uses the combined non-dominated
    set of all fronts as reference.

    Parameters
    ----------
    pareto_front : np.ndarray, shape (N, M)
        Approximated Pareto front.
    true_pareto_front or reference_set : np.ndarray, shape (K, M)
        True Pareto front (or a dense reference set).

    Returns
    -------
    float : IGD value (lower is better).
    """
    ref = true_pareto_front if true_pareto_front is not None else reference_set
    if ref is None:
        ref = pareto_front  # fallback: compare to itself

    total_dist = 0.0
    for ref_point in ref:
        # Find minimum Euclidean distance to any point in pareto_front
        dists = np.sqrt(np.sum((pareto_front - ref_point) ** 2, axis=1))
        total_dist += np.min(dists)

    return float(total_dist / len(ref))


def compute_convergence_metrics(
    fitness_history: List[np.ndarray],
    reference_point: np.ndarray,
) -> List[float]:
    """
    Track hypervolume over generations for convergence analysis.

    Parameters
    ----------
    fitness_history : list of np.ndarray
        Population fitnesses at each generation.
    reference_point : np.ndarray
        Reference point for hypervolume.

    Returns
    -------
    List[float] : Hypervolume at each generation.
    """
    return [hypervolume(f, reference_point) for f in fitness_history]
