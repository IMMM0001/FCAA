"""
Fiddler Crab Asymmetric Algorithm (FCAA) — Multi-Objective Version (v2).

FCAA is a nature-inspired metaheuristic based on the asymmetric claw
morphology of fiddler crabs:

- Major claw (large): Used for coarse, wide-ranging exploration via
  Cauchy mutation with heavy-tailed jumps.
- Minor claw (small): Used for fine-grained local exploitation via
  Gaussian random walk with QUADRATICALLY DECAYING variance.

Key improvements over v1 (addressing review weaknesses):
1. ADAPTIVE GAUSSIAN SIGMA: sigma(t) = sigma_0 * (1 - t/T)^2
   → quadratic decay ensures precise hyperparameter convergence
2. DYNAMIC CLAW SPLIT RATIO: shifts from 80% major (exploration)
   to 20% major (exploitation) over generations
3. ELITE REFINEMENT: top-k solutions get an extra fine-grained
   Gaussian pull toward the best solution
4. ADAPTIVE LEADER SELECTION: early → diverse (random Pareto),
   late → greedy (best RMSE)
"""

from typing import Callable, List, Optional, Tuple

import numpy as np

from .base import BaseOptimizer
from .operators import fcaa_update
from .multi_objective import (
    select_survivors,
    get_pareto_front,
    non_dominated_sort,
    crowding_distance,
)


class FCAAOptimizer(BaseOptimizer):
    """
    Multi-Objective Fiddler Crab Asymmetric Algorithm (v2).

    Parameters
    ----------
    dimension : int
        Solution vector dimensionality.
    pop_size : int
        Population size.
    max_generations : int
        Maximum generations.
    fitness_fn : callable
        Evaluates population → fitness array (N, M), all minimized.
    alpha_init : float
        Initial Cauchy mutation scale (default 1.0).
    sigma_init : float
        Initial Gaussian walk std (default 0.15).
    sigma_final : float
        Final Gaussian walk std floor (default 0.005).
    claw_ratio_init : float
        Initial major claw fraction (default 0.75 → 75% exploration).
    claw_ratio_final : float
        Final major claw fraction (default 0.20 → 20% exploration).
    elite_ratio : float
        Fraction of population getting elite refinement (default 0.2).
    elite_sigma : float
        Gaussian std for elite refinement (very small, default 0.02).
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
        alpha_init: float = 1.0,
        sigma_init: float = 0.15,
        sigma_final: float = 0.005,
        claw_ratio_init: float = 0.75,
        claw_ratio_final: float = 0.20,
        elite_ratio: float = 0.2,
        elite_sigma: float = 0.02,
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
        self.alpha_init = alpha_init
        self.sigma_init = sigma_init
        self.sigma_final = sigma_final
        self.claw_ratio_init = claw_ratio_init
        self.claw_ratio_final = claw_ratio_final
        self.elite_ratio = elite_ratio
        self.elite_sigma = elite_sigma

        # Compute elite count
        self.n_elite = max(1, int(pop_size * elite_ratio))

    def optimize(
        self, verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run FCAA v2 optimization with adaptive scheduling.

        Scheduling summary:
        - Early (t=0):  alpha=1.0, sigma=0.15, claw_ratio=0.75
          → Heavy exploration, coarse feature selection
        - Mid  (t=T/2): alpha=0.75, sigma≈0.04, claw_ratio≈0.48
          → Balanced exploration/exploitation
        - Late (t=T):   alpha=0.5, sigma=0.005, claw_ratio=0.20
          → Heavy exploitation, precise hyperparameter tuning

        Returns
        -------
        pareto_population : np.ndarray
            Pareto-optimal solution vectors.
        pareto_fitnesses : np.ndarray
            Objective values of Pareto front.
        """
        T = self.max_generations

        # Initialize population
        self.population = self._initialize_population()
        self.fitnesses = self.fitness_fn(self.population)
        self.fitness_history = [self.fitnesses.copy()]

        for gen in range(T):
            progress = gen / max(T - 1, 1)  # t/T ∈ [0, 1]

            # ── Adaptive parameter schedules ──
            # Alpha (Cauchy scale): linear decay from alpha_init → 0.5*alpha_init
            alpha_gen = self.alpha_init * (1.0 - 0.5 * progress)

            # Sigma (Gaussian std): QUADRATIC decay
            # sigma(t) = sigma_0 * (1 - t/T)^2, with a floor of sigma_final
            sigma_gen = self.sigma_init * (1.0 - progress) ** 2
            sigma_gen = max(sigma_gen, self.sigma_final)

            # Claw ratio: LINEAR transition from exploration → exploitation
            claw_ratio = self.claw_ratio_init + progress * (
                self.claw_ratio_final - self.claw_ratio_init
            )

            # ── Leader selection ──
            # Early: random Pareto (diversity) → Late: best RMSE (convergence)
            best_idx = self._select_leader(progress)

            # ── Elite indices for refinement ──
            # Top-k solutions by Pareto rank + crowding
            elite_indices = self._select_elites()

            # ── FCAA asymmetric update ──
            offspring = fcaa_update(
                self.population,
                best_idx=best_idx,
                alpha=alpha_gen,
                sigma=sigma_gen,
                claw_ratio=claw_ratio,
                elite_indices=elite_indices,
                elite_sigma=self.elite_sigma,
                rng=self.rng,
            )

            # Evaluate offspring
            offspring_fitnesses = self.fitness_fn(offspring)

            # ── Mu + Lambda selection ──
            combined_pop = np.vstack([self.population, offspring])
            combined_fitnesses = np.vstack([self.fitnesses, offspring_fitnesses])

            self.population, self.fitnesses, _ = select_survivors(
                combined_pop, combined_fitnesses, self.pop_size
            )

            self.fitness_history.append(self.fitnesses.copy())

            if verbose and (gen + 1) % max(1, T // 10) == 0:
                pareto_pop, pareto_fit = get_pareto_front(
                    self.population, self.fitnesses
                )
                print(
                    f"  Gen {gen + 1:4d}/{T} | "
                    f"α={alpha_gen:.3f} σ={sigma_gen:.4f} "
                    f"claw={claw_ratio:.2f} | "
                    f"Pareto: {len(pareto_fit)} | "
                    f"RMSE: [{pareto_fit[:, 0].min():.2f}, "
                    f"{pareto_fit[:, 0].max():.2f}]"
                )

        pareto_population, pareto_fitnesses = get_pareto_front(
            self.population, self.fitnesses
        )

        if verbose:
            print(f"\nFCAA v2 finished. Pareto front size: {len(pareto_fitnesses)}")

        return pareto_population, pareto_fitnesses

    def _select_leader(self, progress: float) -> int:
        """
        Adaptive leader selection.

        Early (low progress): Favor diversity — random Pareto front member.
        Late (high progress): Favor convergence — best RMSE solution.

        Uses a soft transition: P(best_RMSE) = progress^2
        (quadratic shift toward greedy selection in late stages)
        """
        fronts = non_dominated_sort(self.fitnesses)
        pareto_front = fronts[0]

        if len(pareto_front) == 1:
            return int(pareto_front[0])

        # Probability of choosing the best-RMSE leader
        p_greedy = progress ** 2  # Quadratic: accelerates toward greedy in late stages

        if self.rng.random() < p_greedy:
            # Greedy: best RMSE in Pareto front
            pf_fitnesses = self.fitnesses[pareto_front]
            best_idx_in_pf = int(np.argmin(pf_fitnesses[:, 0]))
            return int(pareto_front[best_idx_in_pf])
        else:
            # Diversity: weighted by crowding distance
            pf_fitnesses = self.fitnesses[pareto_front]
            cd = crowding_distance(pf_fitnesses)
            # Handle inf values
            finite_mask = ~np.isinf(cd)
            if finite_mask.any():
                max_finite = np.max(cd[finite_mask])
                cd_fixed = np.where(np.isinf(cd), max_finite * 2.0, cd)
            else:
                cd_fixed = np.ones(len(cd))
            cd_fixed = np.maximum(cd_fixed, 1e-10)
            probs = cd_fixed / cd_fixed.sum()
            chosen = int(self.rng.choice(len(pareto_front), p=probs))
            return int(pareto_front[chosen])

    def _select_elites(self) -> np.ndarray:
        """
        Select elite individuals for the refinement pass.

        Strategy: top `n_elite` individuals by:
        1. First by Pareto rank
        2. Then by RMSE within each rank
        """
        fronts = non_dominated_sort(self.fitnesses)
        elites = []
        for front in fronts:
            if len(elites) >= self.n_elite:
                break
            # Within this front, prefer lower RMSE
            front_fitnesses = self.fitnesses[front]
            sorted_by_rmse = front[np.argsort(front_fitnesses[:, 0])]
            remaining = self.n_elite - len(elites)
            elites.extend(sorted_by_rmse[:remaining].tolist())

        return np.array(elites[:self.n_elite], dtype=int)
