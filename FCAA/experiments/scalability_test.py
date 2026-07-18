#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MO-FCAA Computational Scalability Analysis
===========================================
Controlled-variable experiment measuring how runtime scales with:

1. Population Size:  N ∈ [20, 50, 100, 150, 200]
2. Feature Dimension: D ∈ [25, 50, 100, 200, 400] (via synthetic data)

For each configuration, we measure:
  - Total wall-clock time
  - Time per generation
  - Time per fitness evaluation

Output:
  - Scaling line plots (runtime vs N, runtime vs D)
  - Summary CSV
  - Publication-quality figures (PNG + PDF)

Usage:
    python experiments/scalability_test.py
    python experiments/scalability_test.py --quick
"""

import sys
import os
import time
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add src to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.algorithms.fcaa import FCAAOptimizer
from src.algorithms.nsga2 import NSGA2Optimizer
from src.algorithms.mopso import MOPSOOptimizer
from src.evaluators.feature_selection_hpo import FeatureSelectionHPOEvaluator
from src.utils.data import generate_synthetic_data

warnings.filterwarnings("ignore")

# ── Style ──
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "figure.dpi": 150,
    "savefig.dpi": 150,
})

# ==============================================================================
# Config
# ==============================================================================

BASE_CONFIG = {
    "seed": 42,
    "max_generations": 100,
    "cv_folds": 5,
    "model": "svr",
    "n_runs": 3,  # Runs per configuration for averaging
}

# Experiment 1: Vary population size (fixed dimension)
POP_SIZE_VALUES = [20, 50, 100, 150, 200]
FIXED_N_FEATURES = 50
FIXED_N_SAMPLES = 200

# Experiment 2: Vary feature dimension (fixed population)
DIM_VALUES = [25, 50, 100, 200, 400]
FIXED_POP_SIZE = 50

# Algorithms to test
ALGORITHMS = ["FCAA", "NSGA-II", "MOPSO"]


# ==============================================================================
# Data Generation
# ==============================================================================

def make_dataset(n_features: int, n_samples: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic dataset with controlled dimensionality.
    Uses make_regression for consistent small-sample scenario.
    """
    n_informative = min(n_features, 10)
    X_train, X_test, y_train, y_test = generate_synthetic_data(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=0.1,
        random_state=BASE_CONFIG["seed"],
    )
    return X_train, y_train


# ==============================================================================
# Scaling Measurement
# ==============================================================================

def measure_single_config(
    X_train: np.ndarray,
    y_train: np.ndarray,
    algo_name: str,
    pop_size: int,
    max_generations: int,
    seed: int = 42,
) -> Dict:
    """
    Run one algorithm on one configuration and return timing metrics.
    """
    evaluator = FeatureSelectionHPOEvaluator(
        X_train=X_train, y_train=y_train,
        model_name=BASE_CONFIG["model"],
        cv_folds=BASE_CONFIG["cv_folds"],
    )

    n_features = X_train.shape[1]
    dimension = evaluator.get_dimension()

    if algo_name == "FCAA":
        optimizer = FCAAOptimizer(
            dimension=dimension, pop_size=pop_size,
            max_generations=max_generations, fitness_fn=evaluator,
            seed=seed,
        )
    elif algo_name == "NSGA-II":
        optimizer = NSGA2Optimizer(
            dimension=dimension, pop_size=pop_size,
            max_generations=max_generations, fitness_fn=evaluator,
            seed=seed,
        )
    else:
        optimizer = MOPSOOptimizer(
            dimension=dimension, pop_size=pop_size,
            max_generations=max_generations, fitness_fn=evaluator,
            seed=seed,
        )

    t0 = time.perf_counter()
    pareto_pop, pareto_fit = optimizer.optimize(verbose=False)
    total_time = time.perf_counter() - t0

    n_evals = evaluator.n_evaluations

    return {
        "algorithm": algo_name,
        "pop_size": pop_size,
        "n_features": n_features,
        "n_samples": X_train.shape[0],
        "dimension": dimension,
        "max_generations": max_generations,
        "total_time_s": total_time,
        "time_per_gen_s": total_time / max_generations if max_generations > 0 else np.nan,
        "time_per_eval_ms": (total_time / n_evals * 1000) if n_evals > 0 else np.nan,
        "n_evaluations": n_evals,
        "pareto_size": len(pareto_fit),
        "best_rmse": float(np.min(pareto_fit[:, 0])) if len(pareto_fit) > 0 else np.nan,
    }


# ==============================================================================
# Experiment Runners
# ==============================================================================

def run_population_scaling(config: Dict, output_dir: Path) -> pd.DataFrame:
    """
    Experiment 1: Vary population size with fixed feature dimension.
    """
    print("=" * 68)
    print("Experiment 1: Population Size Scaling")
    print("=" * 68)
    print(f"  Feature dim: {FIXED_N_FEATURES}, Samples: {FIXED_N_SAMPLES}")
    print(f"  Pop sizes:   {POP_SIZE_VALUES}")
    print(f"  Algorithms:  {ALGORITHMS}")
    print(f"  Runs/conf:   {config['n_runs']}")
    print("=" * 68)

    X_train, y_train = make_dataset(n_features=FIXED_N_FEATURES, n_samples=FIXED_N_SAMPLES)
    print(f"\n  Actual dataset: {X_train.shape[1]} features, {X_train.shape[0]} samples\n")

    total = len(POP_SIZE_VALUES) * len(ALGORITHMS) * config["n_runs"]
    pbar = tqdm(total=total, desc="Pop Scaling", unit="run", ncols=100)
    records = []

    for pop_size in POP_SIZE_VALUES:
        for algo_name in ALGORITHMS:
            times = []
            for run_id in range(config["n_runs"]):
                seed = config["seed"] + run_id
                result = measure_single_config(
                    X_train, y_train, algo_name,
                    pop_size=pop_size,
                    max_generations=config["max_generations"],
                    seed=seed,
                )
                result["run_id"] = run_id
                result["experiment"] = "population_scaling"
                records.append(result)
                times.append(result["total_time_s"])
                pbar.update(1)

            # Log summary
            avg_t = np.mean(times)
            pbar.write(f"  {algo_name:8s} | N={pop_size:3d} | "
                       f"time={avg_t:.1f}s (±{np.std(times):.1f}s)")

    pbar.close()

    df = pd.DataFrame(records)
    df.to_csv(output_dir / "scalability_population.csv", index=False)
    print(f"\n[OK] Population scaling results → {output_dir / 'scalability_population.csv'}")
    return df


def run_dimension_scaling(config: Dict, output_dir: Path) -> pd.DataFrame:
    """
    Experiment 2: Vary feature dimension with fixed population size.
    """
    print("\n" + "=" * 68)
    print("Experiment 2: Feature Dimension Scaling")
    print("=" * 68)
    print(f"  Population:  {FIXED_POP_SIZE}")
    print(f"  Dimensions:  {DIM_VALUES}")
    print(f"  Algorithms:  {ALGORITHMS}")
    print(f"  Runs/conf:   {config['n_runs']}")
    print("=" * 68)

    total = len(DIM_VALUES) * len(ALGORITHMS) * config["n_runs"]
    pbar = tqdm(total=total, desc="Dim Scaling", unit="run", ncols=100)
    records = []

    for n_features in DIM_VALUES:
        X_train, y_train = make_dataset(n_features=n_features, n_samples=FIXED_N_SAMPLES)

        for algo_name in ALGORITHMS:
            times = []
            for run_id in range(config["n_runs"]):
                seed = config["seed"] + run_id
                result = measure_single_config(
                    X_train, y_train, algo_name,
                    pop_size=FIXED_POP_SIZE,
                    max_generations=config["max_generations"],
                    seed=seed,
                )
                result["run_id"] = run_id
                result["experiment"] = "dimension_scaling"
                records.append(result)
                times.append(result["total_time_s"])
                pbar.update(1)

            avg_t = np.mean(times)
            pbar.write(f"  {algo_name:8s} | D={n_features:3d} | "
                       f"time={avg_t:.1f}s (±{np.std(times):.1f}s)")

    pbar.close()

    df = pd.DataFrame(records)
    df.to_csv(output_dir / "scalability_dimension.csv", index=False)
    print(f"\n[OK] Dimension scaling results → {output_dir / 'scalability_dimension.csv'}")
    return df


# ==============================================================================
# Visualization
# ==============================================================================

def plot_scaling_results(
    df_pop: pd.DataFrame,
    df_dim: pd.DataFrame,
    output_dir: Path,
):
    """
    Generate a 2-panel figure: (a) Runtime vs Pop Size, (b) Runtime vs Dimension.
    """
    colors = {"FCAA": "#E63946", "NSGA-II": "#457B9D", "MOPSO": "#2A9D8F"}
    markers = {"FCAA": "o", "NSGA-II": "s", "MOPSO": "^"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # ── Panel (a): Population Size Scaling ──
    for algo_name in ALGORITHMS:
        algo_df = df_pop[df_pop["algorithm"] == algo_name]
        agg = algo_df.groupby("pop_size")["total_time_s"].agg(["mean", "std"]).reset_index()
        ax1.errorbar(
            agg["pop_size"], agg["mean"], yerr=agg["std"],
            color=colors[algo_name], marker=markers[algo_name],
            markersize=10, linewidth=2.5, capsize=5,
            markeredgecolor="white", markeredgewidth=1.5,
            label=algo_name,
        )

    ax1.set_xlabel("Population Size (N)", fontsize=13)
    ax1.set_ylabel("Total Runtime (s)", fontsize=13)
    ax1.set_title(
        f"Runtime vs Population Size\n"
        f"(D={FIXED_N_FEATURES} features, {FIXED_N_SAMPLES} samples, {BASE_CONFIG['max_generations']} gens)",
        fontsize=12, fontweight="bold",
    )
    ax1.legend(fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle="--")

    # ── Panel (b): Feature Dimension Scaling ──
    for algo_name in ALGORITHMS:
        algo_df = df_dim[df_dim["algorithm"] == algo_name]
        agg = algo_df.groupby("n_features")["total_time_s"].agg(["mean", "std"]).reset_index()
        ax2.errorbar(
            agg["n_features"], agg["mean"], yerr=agg["std"],
            color=colors[algo_name], marker=markers[algo_name],
            markersize=10, linewidth=2.5, capsize=5,
            markeredgecolor="white", markeredgewidth=1.5,
            label=algo_name,
        )

    ax2.set_xlabel("Feature Dimension (D)", fontsize=13)
    ax2.set_ylabel("Total Runtime (s)", fontsize=13)
    ax2.set_title(
        f"Runtime vs Feature Dimension\n"
        f"(N={FIXED_POP_SIZE}, {FIXED_N_SAMPLES} samples, {BASE_CONFIG['max_generations']} gens)",
        fontsize=12, fontweight="bold",
    )
    ax2.legend(fontsize=11, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.set_xscale("log", base=2)

    fig.suptitle(
        "MO-FCAA Computational Scalability Analysis",
        fontsize=15, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    fig.savefig(output_dir / "scalability_analysis.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / "scalability_analysis.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Scaling plot → {output_dir / 'scalability_analysis.png'}")


def plot_per_eval_cost(
    df_pop: pd.DataFrame,
    df_dim: pd.DataFrame,
    output_dir: Path,
):
    """
    Generate a supplementary figure: time-per-evaluation vs scale.
    """
    colors = {"FCAA": "#E63946", "NSGA-II": "#457B9D", "MOPSO": "#2A9D8F"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for algo_name in ALGORITHMS:
        algo_df = df_pop[df_pop["algorithm"] == algo_name]
        agg = algo_df.groupby("pop_size")["time_per_eval_ms"].agg(["mean", "std"]).reset_index()
        ax1.plot(agg["pop_size"], agg["mean"], color=colors[algo_name],
                marker="o", markersize=8, linewidth=2.5, label=algo_name)

    ax1.set_xlabel("Population Size (N)", fontsize=13)
    ax1.set_ylabel("Time per Evaluation (ms)", fontsize=13)
    ax1.set_title("Per-Evaluation Cost vs Population Size", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle="--")

    for algo_name in ALGORITHMS:
        algo_df = df_dim[df_dim["algorithm"] == algo_name]
        agg = algo_df.groupby("n_features")["time_per_eval_ms"].agg(["mean", "std"]).reset_index()
        ax2.plot(agg["n_features"], agg["mean"], color=colors[algo_name],
                marker="s", markersize=8, linewidth=2.5, label=algo_name)

    ax2.set_xlabel("Feature Dimension (D)", fontsize=13)
    ax2.set_ylabel("Time per Evaluation (ms)", fontsize=13)
    ax2.set_title("Per-Evaluation Cost vs Feature Dimension", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.set_xscale("log", base=2)

    fig.suptitle("Per-Fitness-Evaluation Cost Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    fig.savefig(output_dir / "per_eval_cost.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / "per_eval_cost.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Per-eval cost plot → {output_dir / 'per_eval_cost.png'}")


# ==============================================================================
# Print Summary
# ==============================================================================

def print_scaling_summary(df_pop: pd.DataFrame, df_dim: pd.DataFrame):
    """Print a formatted scaling analysis summary."""
    print("\n" + "=" * 68)
    print("SCALABILITY SUMMARY")
    print("=" * 68)

    if df_pop is not None and len(df_pop) > 0 and "algorithm" in df_pop.columns:
        print("\nPopulation Scaling (fixed D={}):".format(FIXED_N_FEATURES))
        for algo in ALGORITHMS:
            adf = df_pop[df_pop["algorithm"] == algo]
            for n in POP_SIZE_VALUES:
                ndf = adf[adf["pop_size"] == n]
                if len(ndf) > 0:
                    t = ndf["total_time_s"]
                    std_val = t.std() if t.std() == t.std() else 0.0  # Handle NaN
                    print(f"  {algo:8s} N={n:3d}: {t.mean():6.1f}s ± {std_val:5.1f}s")

    if df_dim is not None and len(df_dim) > 0 and "algorithm" in df_dim.columns:
        print("\nDimension Scaling (fixed N={}):".format(FIXED_POP_SIZE))
        for algo in ALGORITHMS:
            adf = df_dim[df_dim["algorithm"] == algo]
            for d in DIM_VALUES:
                ddf = adf[adf["n_features"] == d]
                if len(ddf) > 0:
                    t = ddf["total_time_s"]
                    std_val = t.std() if t.std() == t.std() else 0.0
                    print(f"  {algo:8s} D={d:3d}: {t.mean():6.1f}s ± {std_val:5.1f}s")

    print("=" * 68)


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="MO-FCAA Computational Scalability Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n-runs", type=int, default=3,
                        help="Runs per config for averaging (default: 3)")
    parser.add_argument("--quick", action="store_true",
                        help="Fast mode: 1 run, 50 gens, reduced grid")
    parser.add_argument("--output-dir", type=str, default="results/scalability",
                        help="Output directory")
    parser.add_argument("--skip-dimension", action="store_true",
                        help="Skip dimension scaling (faster)")
    parser.add_argument("--skip-population", action="store_true",
                        help="Skip population scaling (faster)")
    return parser.parse_args()


def main():
    args = parse_args()
    config = BASE_CONFIG.copy()

    global POP_SIZE_VALUES, DIM_VALUES

    if args.quick:
        config["n_runs"] = 1
        config["max_generations"] = 50
        POP_SIZE_VALUES = [20, 50, 100, 200]
        DIM_VALUES = [25, 50, 100, 200]
    else:
        config["n_runs"] = args.n_runs

    output_dir = _PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df_pop, df_dim = None, None

    if not args.skip_population:
        df_pop = run_population_scaling(config, output_dir)

    if not args.skip_dimension:
        df_dim = run_dimension_scaling(config, output_dir)

    if df_pop is not None and df_dim is not None:
        plot_scaling_results(df_pop, df_dim, output_dir)
        plot_per_eval_cost(df_pop, df_dim, output_dir)
        print_scaling_summary(df_pop, df_dim)
    elif df_pop is not None:
        print_scaling_summary(df_pop, pd.DataFrame())
        # Single-panel plot for population only
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = {"FCAA": "#E63946", "NSGA-II": "#457B9D", "MOPSO": "#2A9D8F"}
        for algo_name in ALGORITHMS:
            algo_df = df_pop[df_pop["algorithm"] == algo_name]
            agg = algo_df.groupby("pop_size")["total_time_s"].agg(["mean", "std"]).reset_index()
            ax.errorbar(agg["pop_size"], agg["mean"], yerr=agg["std"],
                       color=colors[algo_name], marker="o", markersize=10,
                       linewidth=2.5, capsize=5)
        ax.set_xlabel("Population Size")
        ax.set_ylabel("Total Runtime (s)")
        ax.legend(ALGORITHMS)
        fig.savefig(output_dir / "scalability_population.png", dpi=150)
        plt.close(fig)
    elif df_dim is not None:
        print_scaling_summary(pd.DataFrame(), df_dim)

    print(f"\n[OK] All results saved to {output_dir}")


if __name__ == "__main__":
    main()
