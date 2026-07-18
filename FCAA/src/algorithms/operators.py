"""
FCAA core operators: Cauchy mutation and Gaussian random walk.

These are the two asymmetric update mechanisms that define the
Fiddler Crab Asymmetric Algorithm (FCAA):

- Major claw (exploration): Cauchy mutation with heavy tails
  helps escape local optima by taking occasional large jumps.
- Minor claw (exploitation): Gaussian random walk with shrinking
  variance for fine-grained local search.

Key improvements over the baseline:
1. Adaptive Gaussian sigma: sigma(t) = sigma_0 * (1 - t/T)^2
   → quadratic decay ensures fine convergence in late stages
2. Dynamic claw split ratio: shifts from exploration-dominant
   (early) to exploitation-dominant (late)
"""

from typing import Optional, Tuple

import numpy as np


def cauchy_mutation(
    x: np.ndarray,
    x_best: np.ndarray,
    alpha: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Major claw update — Cauchy mutation for global exploration.

    Formula: X_new = X_old + α · C(0, 1) · (X_best - X_old)

    The Cauchy distribution has heavy tails, enabling occasional
    large jumps that help escape local optima.

    Parameters
    ----------
    x : np.ndarray, shape (D,)
        Current solution vector.
    x_best : np.ndarray, shape (D,)
        Best-known solution (leader).
    alpha : float
        Step size scaling factor.
    rng : np.random.Generator, optional

    Returns
    -------
    np.ndarray, shape (D,)
        Updated solution vector.
    """
    if rng is None:
        rng = np.random.default_rng()
    cauchy_noise = rng.standard_cauchy(size=len(x))
    return x + alpha * cauchy_noise * (x_best - x)


def gaussian_walk(
    x: np.ndarray,
    sigma: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Minor claw update — Gaussian random walk for local exploitation.

    Formula: X_new = X_old + N(0, sigma) · X_old   (multiplicative noise)
    OR:      X_new = X_old + N(0, sigma)            (additive noise)

    Uses a MIXED strategy:
    - 50% multiplicative (scales with current value, good for params in [0,1])
    - 50% additive (constant noise floor, prevents stagnation near zero)

    Parameters
    ----------
    x : np.ndarray, shape (D,)
        Current solution vector.
    sigma : float
        Gaussian standard deviation (should DECAY over generations).
    rng : np.random.Generator, optional

    Returns
    -------
    np.ndarray, shape (D,)
        Updated solution vector.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(x)
    gaussian_noise = rng.normal(0, sigma, size=n)

    # Mixed multiplicative + additive noise
    # Multiplicative: proportional to current value → effective for mid-range values
    # Additive: constant noise floor → prevents stagnation when x ≈ 0
    mix_mask = rng.random(n) < 0.5
    delta = np.where(mix_mask, gaussian_noise * x, gaussian_noise)

    return x + delta


def elite_gaussian_refinement(
    x: np.ndarray,
    x_elite: np.ndarray,
    sigma: float = 0.02,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Elite-guided local refinement — pulls solutions toward elite with
    very small Gaussian perturbation.

    Formula: X_new = X_old + sigma · N(0,1) · (X_elite - X_old)

    This is a fine-grained local search that specifically targets
    hyperparameter precision, complementing the standard minor claw update.

    Parameters
    ----------
    x : np.ndarray, shape (D,)
        Current solution.
    x_elite : np.ndarray, shape (D,)
        Elite solution to learn from.
    sigma : float
        Very small step size (e.g., 0.01–0.05).
    rng : np.random.Generator, optional

    Returns
    -------
    np.ndarray, shape (D,)
        Refined solution.
    """
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0, sigma, size=len(x))
    return x + noise * (x_elite - x)


def fcaa_update(
    population: np.ndarray,
    best_idx: int,
    alpha: float = 1.0,
    sigma: float = 0.1,
    claw_ratio: float = 0.5,
    elite_indices: Optional[np.ndarray] = None,
    elite_sigma: float = 0.02,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Apply FCAA asymmetric update to the entire population.

    Key improvements over baseline:
    1. Dynamic claw_ratio: fraction of dims assigned to major claw.
       - High (0.7–0.8) early → exploration-dominant
       - Low (0.2–0.3) late → exploitation-dominant
    2. sigma (Gaussian std) decays quadratically over generations
    3. Elite refinement pass for top individuals

    Parameters
    ----------
    population : np.ndarray, shape (N, D)
        Current population.
    best_idx : int
        Index of the best individual (leader) to follow.
    alpha : float
        Cauchy mutation scale (should decay slowly).
    sigma : float
        Gaussian walk standard deviation (should decay QUADRATICALLY).
    claw_ratio : float
        Fraction of dimensions assigned to major claw [0.0, 1.0].
        Higher = more exploration, lower = more exploitation.
    elite_indices : np.ndarray, optional
        Indices of elite individuals for refinement pass.
    elite_sigma : float
        Very small sigma for elite refinement.
    rng : np.random.Generator, optional

    Returns
    -------
    np.ndarray, shape (N, D)
        Updated population (clipped to [0, 1]).
    """
    if rng is None:
        rng = np.random.default_rng()

    n, d = population.shape
    x_best = population[best_idx]
    new_population = population.copy()

    # Number of major claw dimensions (at least 1, at most d-1)
    n_major = max(1, min(d - 1, int(d * claw_ratio)))
    n_minor = d - n_major

    for i in range(n):
        # Randomly split dimensions according to claw_ratio
        perm = rng.permutation(d)
        major_dims = perm[:n_major]
        minor_dims = perm[n_major:]

        # ---- Major claw: Cauchy mutation toward best ----
        cauchy_noise = rng.standard_cauchy(size=len(major_dims))
        new_population[i, major_dims] = (
            population[i, major_dims]
            + alpha * cauchy_noise * (x_best[major_dims] - population[i, major_dims])
        )

        # ---- Minor claw: HYBRID Gaussian strategy ----
        # Split minor dims further: 70% GUIDED (toward leader), 30% DIVERSE (random walk)
        n_minor_dims = len(minor_dims)
        n_guided = max(1, int(n_minor_dims * 0.7))
        n_diverse = n_minor_dims - n_guided

        minor_perm = rng.permutation(n_minor_dims)
        guided_dims = minor_dims[minor_perm[:n_guided]]
        diverse_dims = minor_dims[minor_perm[n_guided:]]

        # Guided: small Gaussian step TOWARD the leader
        # X_new = X_old + sigma · N(0,1) · (X_best - X_old)
        # Same form as Cauchy but with Gaussian (thin-tailed) noise
        if len(guided_dims) > 0:
            guided_noise = rng.normal(0, sigma, size=len(guided_dims))
            new_population[i, guided_dims] = (
                population[i, guided_dims]
                + guided_noise * (x_best[guided_dims] - population[i, guided_dims])
            )

        # Diverse: pure Gaussian walk for diversity maintenance
        # Mixed multiplicative + additive
        if len(diverse_dims) > 0:
            diverse_noise = rng.normal(0, sigma, size=len(diverse_dims))
            mix_mask = rng.random(len(diverse_dims)) < 0.5
            delta = np.where(
                mix_mask,
                diverse_noise * population[i, diverse_dims],
                diverse_noise,
            )
            new_population[i, diverse_dims] = population[i, diverse_dims] + delta

        # Boundary handling: clip to [0, 1]
        new_population[i] = np.clip(new_population[i], 0.0, 1.0)

    # ---- Elite refinement pass ----
    # For top individuals, do an additional fine-grained local search
    # toward the best solution to improve hyperparameter precision
    if elite_indices is not None and len(elite_indices) > 0:
        for idx in elite_indices:
            if idx == best_idx:
                continue  # Don't refine the leader with itself
            refined = elite_gaussian_refinement(
                new_population[idx], x_best, sigma=elite_sigma, rng=rng
            )
            new_population[idx] = np.clip(refined, 0.0, 1.0)

    return new_population


def sbx_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    eta: float = 20.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulated Binary Crossover (SBX) — used by NSGA-II.

    Parameters
    ----------
    parent1, parent2 : np.ndarray, shape (D,)
    eta : float
        Distribution index (larger = offspring closer to parents).
    rng : np.random.Generator, optional

    Returns
    -------
    child1, child2 : np.ndarray
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(parent1)
    child1 = np.zeros(n)
    child2 = np.zeros(n)

    for i in range(n):
        if rng.random() <= 0.5:
            if abs(parent2[i] - parent1[i]) > 1e-14:
                if parent1[i] < parent2[i]:
                    y1, y2 = parent1[i], parent2[i]
                else:
                    y1, y2 = parent2[i], parent1[i]

                yl, yu = 0.0, 1.0
                rand = rng.random()

                beta_q = 1.0 + (2.0 * (y1 - yl) / (y2 - y1))
                alpha_q = 2.0 - beta_q ** (-(eta + 1.0))

                if rand <= 1.0 / alpha_q:
                    betaq = (rand * alpha_q) ** (1.0 / (eta + 1.0))
                else:
                    betaq = (1.0 / (2.0 - rand * alpha_q)) ** (1.0 / (eta + 1.0))

                c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
                beta_q = 1.0 + (2.0 * (yu - y2) / (y2 - y1))
                alpha_q = 2.0 - beta_q ** (-(eta + 1.0))

                if rand <= 1.0 / alpha_q:
                    betaq = (rand * alpha_q) ** (1.0 / (eta + 1.0))
                else:
                    betaq = (1.0 / (2.0 - rand * alpha_q)) ** (1.0 / (eta + 1.0))

                c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))

                c1 = np.clip(c1, yl, yu)
                c2 = np.clip(c2, yl, yu)

                if rng.random() <= 0.5:
                    child1[i], child2[i] = c2, c1
                else:
                    child1[i], child2[i] = c1, c2
            else:
                child1[i] = parent1[i]
                child2[i] = parent2[i]
        else:
            child1[i] = parent1[i]
            child2[i] = parent2[i]

    return child1, child2


def polynomial_mutation(
    x: np.ndarray,
    eta_mut: float = 20.0,
    mutation_rate: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Polynomial mutation — used by NSGA-II.

    Parameters
    ----------
    x : np.ndarray, shape (D,)
    eta_mut : float
        Distribution index for mutation.
    mutation_rate : float, optional
        Per-gene mutation probability. Defaults to 1/D.
    rng : np.random.Generator, optional

    Returns
    -------
    np.ndarray
        Mutated individual.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(x)
    if mutation_rate is None:
        mutation_rate = 1.0 / n

    mutated = x.copy()
    yl, yu = 0.0, 1.0

    for i in range(n):
        if rng.random() <= mutation_rate:
            y = x[i]
            delta1 = (y - yl) / (yu - yl)
            delta2 = (yu - y) / (yu - yl)
            rand = rng.random()

            if rand <= 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta_mut + 1.0)
                delta_q = val ** (1.0 / (eta_mut + 1.0)) - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta_mut + 1.0)
                delta_q = 1.0 - val ** (1.0 / (eta_mut + 1.0))

            y = y + delta_q * (yu - yl)
            mutated[i] = np.clip(y, yl, yu)

    return mutated
