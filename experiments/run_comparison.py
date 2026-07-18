#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MO-FCAA Comparison Experiment: FCAA vs NSGA-II vs MOPSO.

Runs all three multi-objective optimizers on the simultaneous
feature selection + hyperparameter tuning problem, using real
materials science datasets (Boron-based and Carbon-based HECs).

Produces:
- Pareto front comparison plot
- Convergence curve (hypervolume over generations)
- Console summary of best solutions found
"""

import sys
import os
import time
import warnings
from pathlib import Path

# Fix Unicode encoding issues on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

from src.evaluators.feature_selection_hpo import FeatureSelectionHPOEvaluator
from src.algorithms.multi_objective import get_pareto_front, non_dominated_sort, crowding_distance
from src.utils.metrics import hypervolume

# Shared utilities (data loading, optimizer factory) — extracted to common.py
from experiments.common import load_and_preprocess_data, create_optimizer

warnings.filterwarnings("ignore")

# ─── Configuration ───────────────────────────────────────────────
CONFIG = {
    # Dataset settings
    "data_file": "B_data.xlsx",          # Dataset filename in data/
    "target_column": "Tm of MD (K)",     # Target column name
    "drop_columns": ["Systems"],          # Non-feature columns to drop
    "test_size": 0.2,
    "poly_degree": 3,                    # Polynomial feature expansion degree
    "add_noise_features": 20,            # Additional noise features as distractors

    # Model settings
    "model": "svr",                      # 'svr', 'krr', 'random_forest', 'mlp'
    "cv_folds": 5,

    # Optimization settings
    "pop_size": 80,
    "max_generations": 200,
    "n_runs": 3,                         # Independent runs for statistical analysis

    # FCAA v2-specific (adaptive scheduling)
    "fcaa_alpha_init": 1.0,
    "fcaa_sigma_init": 0.15,
    "fcaa_sigma_final": 0.002,
    "fcaa_claw_ratio_init": 0.80,
    "fcaa_claw_ratio_final": 0.15,
    "fcaa_elite_ratio": 0.3,
    "fcaa_elite_sigma": 0.015,

    # NSGA-II-specific
    "nsga2_crossover_prob": 0.9,
    "nsga2_eta_crossover": 20.0,
    "nsga2_eta_mutation": 20.0,

    # MOPSO-specific
    "mopso_archive_size": 50,
    "mopso_inertia": 0.5,
    "mopso_cognitive": 1.5,
    "mopso_social": 1.5,
    "mopso_mutation_rate": 0.1,

    # Output
    "seed": 42,
    "figures_dir": "../results/figures",
    "logs_dir": "../results/logs",
    "dpi": 150,
}

# ─── Run Single Algorithm ─────────────────────────────────────────
def run_algorithm(algo_name, evaluator, config, run_id=0):
    """
    Run a single optimization algorithm and return results.

    Uses create_optimizer() from experiments.common for algorithm
    construction, keeping this wrapper focused on logging + timing.
    """
    print(f"\n{'='*60}")
    print(f"Running {algo_name} (run {run_id + 1}/{config['n_runs']})")
    print(f"  Dimension: {evaluator.get_dimension()}")
    print(f"  Population: {config['pop_size']}, Generations: {config['max_generations']}")
    print(f"{'='*60}")

    optimizer = create_optimizer(algo_name, evaluator, config, run_id)

    t0 = time.perf_counter()
    pareto_pop, pareto_fit = optimizer.optimize(verbose=True)
    elapsed = time.perf_counter() - t0

    print(f"  Time: {elapsed:.1f}s, Pareto size: {len(pareto_fit)}")

    return {
        "name": algo_name,
        "run": run_id,
        "pareto_population": pareto_pop,
        "pareto_fitnesses": pareto_fit,
        "fitness_history": optimizer.fitness_history,
        "time": elapsed,
        "n_evaluations": evaluator.n_evaluations,
    }


# ─── Plotting ─────────────────────────────────────────────────────
def plot_pareto_fronts(all_results, config):
    """
    Plot Pareto fronts from all algorithms on a single figure.

    X-axis: Feature Ratio (lower better)
    Y-axis: RMSE (lower better)
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = {"FCAA": "#E63946", "NSGA-II": "#457B9D", "MOPSO": "#2A9D8F"}
    markers = {"FCAA": "o", "NSGA-II": "s", "MOPSO": "^"}

    for algo_name, runs in all_results.items():
        color = colors.get(algo_name, "gray")
        marker = markers.get(algo_name, "x")

        # Collect all Pareto fitnesses across runs
        all_pareto_fit = []
        for run_data in runs:
            pf = run_data["pareto_fitnesses"]
            if len(pf) > 0:
                all_pareto_fit.append(pf)

        if not all_pareto_fit:
            continue

        # Combine and find overall Pareto front
        combined = np.vstack(all_pareto_fit)
        from src.utils.metrics import find_pareto_front
        pareto_mask = find_pareto_front(combined)
        overall_pareto = combined[pareto_mask]

        # Sort by feature ratio for clean line
        sorted_idx = np.argsort(overall_pareto[:, 1])
        overall_pareto = overall_pareto[sorted_idx]

        # Plot individual run points (faded)
        for i, pf in enumerate(all_pareto_fit):
            alpha_val = 0.3 if len(all_pareto_fit) > 1 else 0.5
            ax.scatter(
                pf[:, 1] * 100, pf[:, 0],
                color=color, marker=marker, alpha=alpha_val,
                s=30, edgecolors="none",
                label=f"{algo_name} (run {i+1})" if i == 0 else "",
            )

        # Plot overall Pareto front (bold line + markers)
        ax.plot(
            overall_pareto[:, 1] * 100, overall_pareto[:, 0],
            color=color, linewidth=2.5, alpha=0.9, linestyle="-",
            label=f"{algo_name} Pareto Front",
        )
        ax.scatter(
            overall_pareto[:, 1] * 100, overall_pareto[:, 0],
            color=color, marker=marker, s=80, edgecolors="white",
            linewidth=1.5, zorder=10,
        )

    ax.set_xlabel("Feature Retention Ratio (%)", fontsize=13)
    ax.set_ylabel("Cross-Validated RMSE", fontsize=13)
    ax.set_title(
        f"Pareto Front Comparison: FCAA vs NSGA-II vs MOPSO\n"
        f"({config['model'].upper()} on {config['data_file'].replace('.xlsx','')})",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    # Auto-scale x-axis to actual data range
    all_ratios_plot = np.concatenate([r["pareto_fitnesses"][:, 1] for runs in all_results.values()
                                      for r in runs if len(r["pareto_fitnesses"]) > 0])
    x_max_plot = max(np.max(all_ratios_plot) * 100 * 1.15, 25)
    ax.set_xlim(0, x_max_plot)

    plt.tight_layout()
    return fig


def plot_convergence(all_results, config):
    """
    Plot hypervolume convergence over generations with error bands.

    Hypervolume is computed relative to a common reference point.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Find global reference point across all algorithms
    all_fitnesses = []
    all_generations = 0
    for runs in all_results.values():
        for run_data in runs:
            for gen_fit in run_data["fitness_history"]:
                if gen_fit is not None and len(gen_fit) > 0:
                    all_fitnesses.append(gen_fit)
            all_generations = max(
                all_generations, len(run_data["fitness_history"])
            )

    if not all_fitnesses:
        print("Warning: No fitness data for convergence plot")
        plt.close(fig)
        return None

    combined = np.vstack(all_fitnesses)
    ref_point = np.array([
        np.max(combined[:, 0]) * 1.1,
        1.05,  # Feature ratio max is 1.0
    ])

    colors = {"FCAA": "#E63946", "NSGA-II": "#457B9D", "MOPSO": "#2A9D8F"}

    for algo_name, runs in all_results.items():
        color = colors.get(algo_name, "gray")

        # Align histories to same length
        max_gen = max(len(r["fitness_history"]) for r in runs)
        hv_matrix = []

        for run_data in runs:
            hv_values = []
            for gen_fit in run_data["fitness_history"]:
                if gen_fit is not None and len(gen_fit) > 0:
                    hv = hypervolume(gen_fit, ref_point)
                    hv_values.append(hv)
            # Pad with last value if shorter
            while len(hv_values) < max_gen:
                hv_values.append(hv_values[-1] if hv_values else 0)
            hv_matrix.append(hv_values)

        hv_matrix = np.array(hv_matrix)

        # Mean and std
        generations = np.arange(1, max_gen + 1)
        hv_mean = np.mean(hv_matrix, axis=0)
        hv_std = np.std(hv_matrix, axis=0)

        ax.plot(generations, hv_mean, color=color, linewidth=2,
                label=algo_name)
        ax.fill_between(
            generations,
            hv_mean - hv_std,
            hv_mean + hv_std,
            color=color, alpha=0.15,
        )

    ax.set_xlabel("Generation", fontsize=13)
    ax.set_ylabel("Hypervolume", fontsize=13)
    ax.set_title(
        "Convergence Comparison (Hypervolume Indicator)\n"
        f"{config['model'].upper()} on {config['data_file'].replace('.xlsx','')}",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    return fig


def print_best_solutions(all_results, X_train, y_train, config):
    """Print the best solution details for each algorithm."""
    from src.utils.encoding import SolutionEncoding
    from src.evaluators.models import get_model_wrapper

    hp_bounds = get_model_wrapper(config["model"]).hyperparameter_bounds()
    encoding = SolutionEncoding(X_train.shape[1], hp_bounds)

    for algo_name, runs in all_results.items():
        best_rmse = np.inf
        best_solution = None
        for run_data in runs:
            pf = run_data["pareto_fitnesses"]
            pp = run_data["pareto_population"]
            if len(pf) == 0:
                continue
            best_idx = np.argmin(pf[:, 0])
            if pf[best_idx, 0] < best_rmse:
                best_rmse = pf[best_idx, 0]
                best_solution = pp[best_idx]

        if best_solution is not None:
            mask, hparams, indices = encoding.decode(best_solution)
            print(f"\n{algo_name} -- Best Solution:")
            print(f"  RMSE: {best_rmse:.4f}")
            print(f"  Features selected: {mask.sum()}/{X_train.shape[1]} "
                  f"({mask.sum()/X_train.shape[1]*100:.1f}%)")
            print(f"  Hyperparameters: {hparams}")


# ─── Main Experiment ──────────────────────────────────────────────
def main():
    config = CONFIG.copy()

    # Ensure output directories exist
    figures_dir = Path(__file__).resolve().parent.parent / "results" / "figures"
    logs_dir = Path(__file__).resolve().parent.parent / "results" / "logs"
    figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MO-FCAA: Feature Selection + Hyperparameter Optimization")
    print("Fiddler Crab Asymmetric Algorithm vs Baselines")
    print("=" * 70)

    # ── Load Data ──
    X_train, X_test, y_train, y_test = load_and_preprocess_data(config)

    # ── Show problem encoding ──
    temp_eval = FeatureSelectionHPOEvaluator(
        X_train=X_train, y_train=y_train,
        model_name=config["model"], cv_folds=config["cv_folds"],
    )
    dim_info = temp_eval.get_dimension_info()
    print(f"\nProblem encoding:")
    print(f"  Total dimensions: {dim_info['total_dimension']}")
    print(f"  Feature dimensions: {dim_info['n_features']}")
    print(f"  Hyperparameter dimensions: {dim_info['n_hyperparams']}")
    print(f"  Hyperparameters: {list(dim_info['hp_bounds'].keys())}")
    del temp_eval

    # ── Run All Algorithms ──
    algorithms = ["FCAA", "NSGA-II", "MOPSO"]
    all_results = {algo: [] for algo in algorithms}

    for algo_name in algorithms:
        for run_id in range(config["n_runs"]):
            # Fresh evaluator for each run (resets counter and cache)
            evaluator = FeatureSelectionHPOEvaluator(
                X_train=X_train, y_train=y_train,
                model_name=config["model"], cv_folds=config["cv_folds"],
            )
            result = run_algorithm(algo_name, evaluator, config, run_id)
            all_results[algo_name].append(result)

    # ── Print Summary ──
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for algo_name in algorithms:
        runs = all_results[algo_name]
        pareto_sizes = [len(r["pareto_fitnesses"]) for r in runs]
        best_rmses = []
        best_feat_ratios = []
        for r in runs:
            pf = r["pareto_fitnesses"]
            if len(pf) > 0:
                best_rmses.append(np.min(pf[:, 0]))
                best_feat_ratios.append(np.min(pf[:, 1]))

        times = [r["time"] for r in runs]

        print(f"\n{algo_name}:")
        print(f"  Avg Pareto size: {np.mean(pareto_sizes):.0f} ± {np.std(pareto_sizes):.0f}")
        print(f"  Best RMSE:       {np.min(best_rmses):.4f} (avg best: {np.mean(best_rmses):.4f} ± {np.std(best_rmses):.4f})")
        print(f"  Min features:    {np.min(best_feat_ratios)*100:.1f}%")
        print(f"  Avg time:        {np.mean(times):.1f}s")

    # ── Best Solutions Detail ──
    print_best_solutions(all_results, X_train, y_train, config)

    # ── Generate Plots ──
    print("\nGenerating figures...")

    # 1. Pareto Front Comparison
    fig1 = plot_pareto_fronts(all_results, config)
    path1 = figures_dir / "pareto_front_comparison.png"
    fig1.savefig(path1, dpi=config["dpi"], bbox_inches="tight")
    fig1.savefig(figures_dir / "pareto_front_comparison.pdf", bbox_inches="tight")
    plt.close(fig1)
    print(f"  [OK] Pareto front: {path1}")

    # 2. Convergence Curves
    fig2 = plot_convergence(all_results, config)
    if fig2 is not None:
        path2 = figures_dir / "convergence_comparison.png"
        fig2.savefig(path2, dpi=config["dpi"], bbox_inches="tight")
        fig2.savefig(figures_dir / "convergence_comparison.pdf", bbox_inches="tight")
        plt.close(fig2)
        print(f"  [OK] Convergence: {path2}")

    # 3. Save numerical results
    results_csv = logs_dir / "summary_results.csv"
    rows = []
    for algo_name in algorithms:
        for run_data in all_results[algo_name]:
            pf = run_data["pareto_fitnesses"]
            if len(pf) > 0:
                rows.append({
                    "algorithm": algo_name,
                    "run": run_data["run"] + 1,
                    "pareto_size": len(pf),
                    "best_rmse": np.min(pf[:, 0]),
                    "min_feature_ratio": np.min(pf[:, 1]),
                    "time_seconds": run_data["time"],
                    "evaluations": run_data["n_evaluations"],
                })
    pd.DataFrame(rows).to_csv(results_csv, index=False)
    print(f"  [OK] Results CSV: {results_csv}")

    print("\n" + "=" * 70)
    print("Experiment complete! Check results/figures/ for outputs.")
    print("=" * 70)


if __name__ == "__main__":
    main()
