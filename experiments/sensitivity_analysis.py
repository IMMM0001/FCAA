#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MO-FCAA Parameter Sensitivity Analysis
=======================================
Grid-search over core FCAA adaptive parameters and visualize
their impact on optimization performance via heatmaps.

Parameters explored:
  - Claw Ratio Init (ρ):  [0.6, 0.7, 0.8, 0.9, 1.0]
  - Cauchy Scale Init (α): [0.5, 0.75, 1.0, 1.25, 1.5]
  - Gaussian Sigma Init (σ): [0.05, 0.10, 0.15, 0.20, 0.25]

Output:
  - Heatmap grids for each parameter pair vs Hypervolume / Best RMSE
  - Summary CSV with all grid-search results
  - Publication-quality figures (PNG + PDF)

Usage:
    python experiments/sensitivity_analysis.py
    python experiments/sensitivity_analysis.py --n-runs 5 --quick
"""

import sys
import os
import time
import json
import argparse
import warnings
from pathlib import Path
from itertools import product
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterSciNotation
import seaborn as sns
from tqdm import tqdm

# Add src to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluators.feature_selection_hpo import FeatureSelectionHPOEvaluator
from src.algorithms.fcaa import FCAAOptimizer
from src.utils.metrics import hypervolume

from experiments.common import load_and_preprocess_data

warnings.filterwarnings("ignore")

# ── Style ──
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 150,
})

# ==============================================================================
# Default Config
# ==============================================================================

BASE_CONFIG = {
    # Dataset
    "data_file": "B_data.xlsx",
    "target_column": "Tm of MD (K)",
    "drop_columns": ["Systems"],
    "test_size": 0.2,
    "poly_degree": 3,
    "add_noise_features": 20,
    "seed": 42,

    # Optimization (reduced for grid search speed)
    "pop_size": 50,
    "max_generations": 100,
    "cv_folds": 5,
    "n_runs": 3,  # Runs per grid point for averaging

    # FCAA defaults (parameter under test will override)
    "fcaa_alpha_init": 1.0,
    "fcaa_sigma_init": 0.15,
    "fcaa_sigma_final": 0.002,
    "fcaa_claw_ratio_init": 0.80,
    "fcaa_claw_ratio_final": 0.15,
    "fcaa_elite_ratio": 0.3,
    "fcaa_elite_sigma": 0.015,
}

# Grid search space
PARAM_GRID = {
    "claw_ratio_init": [0.6, 0.7, 0.8, 0.9, 1.0],
    "alpha_init": [0.5, 0.75, 1.0, 1.25, 1.5],
    "sigma_init": [0.05, 0.10, 0.15, 0.20, 0.25],
}

PARAM_LABELS = {
    "claw_ratio_init": "Initial Claw Ratio (ρ)",
    "alpha_init": "Initial Cauchy Scale (α)",
    "sigma_init": "Initial Gaussian Sigma (σ)",
}


# ==============================================================================
# Core Sensitivity Runner
# ==============================================================================

def run_single_grid_point(
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_name: str,
    param_value: float,
    config: Dict[str, Any],
    run_id: int = 0,
) -> Dict[str, Any]:
    """
    Run FCAA with a specific parameter value and return metrics.
    """
    local_config = config.copy()
    # Map param_name to FCAA constructor kwarg
    fcaa_key = {
        "claw_ratio_init": "fcaa_claw_ratio_init",
        "alpha_init": "fcaa_alpha_init",
        "sigma_init": "fcaa_sigma_init",
    }.get(param_name, f"fcaa_{param_name}")
    local_config[fcaa_key] = param_value

    evaluator = FeatureSelectionHPOEvaluator(
        X_train=X_train, y_train=y_train,
        model_name=config.get("model", "svr"),
        cv_folds=config["cv_folds"],
    )

    optimizer = FCAAOptimizer(
        dimension=evaluator.get_dimension(),
        pop_size=local_config["pop_size"],
        max_generations=local_config["max_generations"],
        fitness_fn=evaluator,
        alpha_init=local_config["fcaa_alpha_init"],
        sigma_init=local_config["fcaa_sigma_init"],
        sigma_final=local_config["fcaa_sigma_final"],
        claw_ratio_init=local_config["fcaa_claw_ratio_init"],
        claw_ratio_final=local_config["fcaa_claw_ratio_final"],
        elite_ratio=local_config["fcaa_elite_ratio"],
        elite_sigma=local_config["fcaa_elite_sigma"],
        seed=local_config["seed"] + run_id,
    )

    t0 = time.perf_counter()
    pareto_pop, pareto_fit = optimizer.optimize(verbose=False)
    elapsed = time.perf_counter() - t0

    if len(pareto_fit) == 0:
        return {
            "best_rmse": np.nan, "mean_rmse": np.nan,
            "pareto_size": 0, "hypervolume": np.nan,
            "time_seconds": elapsed,
        }

    rmse_col = pareto_fit[:, 0]
    ref_point = np.array([np.max(rmse_col) * 1.1, 1.05])
    hv = hypervolume(pareto_fit, ref_point)

    return {
        "best_rmse": float(np.min(rmse_col)),
        "mean_rmse": float(np.mean(rmse_col)),
        "pareto_size": int(len(pareto_fit)),
        "hypervolume": float(hv),
        "time_seconds": float(elapsed),
        "n_evaluations": int(evaluator.n_evaluations),
    }


def run_sensitivity_grid(
    config: Dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    """Run the full grid search over all parameter combinations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("FCAA v2 — Parameter Sensitivity Analysis (Grid Search)")
    print("=" * 68)

    # Load data once
    ds_config = {**config}
    X_train, X_test, y_train, y_test = load_and_preprocess_data(ds_config)
    print(f"  Features: {X_train.shape[1]}, Samples: {X_train.shape[0]}")

    # Total work
    total_points = len(PARAM_GRID["claw_ratio_init"]) * len(PARAM_GRID["alpha_init"]) * len(PARAM_GRID["sigma_init"])
    total_runs = total_points * config["n_runs"]
    print(f"  Grid points: {total_points} × {config['n_runs']} runs = {total_runs} total")
    print(f"  Output dir: {output_dir.resolve()}")
    print("=" * 68)

    records = []
    global_pbar = tqdm(total=total_runs, desc="Grid Search", unit="run", ncols=100)

    for claw, alpha, sigma in product(
        PARAM_GRID["claw_ratio_init"],
        PARAM_GRID["alpha_init"],
        PARAM_GRID["sigma_init"],
    ):
        metrics_accum = []
        for run_id in range(config["n_runs"]):
            metrics = run_single_grid_point(
                X_train, y_train,
                param_name=None,  # all params overridden
                param_value=None,
                config={
                    **config,
                    "fcaa_claw_ratio_init": claw,
                    "fcaa_alpha_init": alpha,
                    "fcaa_sigma_init": sigma,
                },
                run_id=run_id,
            )
            metrics["claw_ratio_init"] = claw
            metrics["alpha_init"] = alpha
            metrics["sigma_init"] = sigma
            metrics["run_id"] = run_id
            records.append(metrics)
            metrics_accum.append(metrics)
            global_pbar.update(1)

    global_pbar.close()

    df = pd.DataFrame(records)

    # Save CSV
    csv_path = output_dir / "sensitivity_grid_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] Grid results → {csv_path}")

    # Generate heatmaps
    _generate_heatmaps(df, output_dir)

    # Generate interaction plot
    _generate_interaction_plot(df, output_dir)

    return df


# ==============================================================================
# Heatmap Generation
# ==============================================================================

def _generate_heatmaps(df: pd.DataFrame, output_dir: Path):
    """
    Generate 2D heatmaps for each parameter pair.

    For each pair (p1, p2):
      1. Average the metric over the third parameter and runs
      2. Plot heatmap of avg metric vs (p1, p2)
    """
    # Aggregate: mean over runs + over the third parameter
    agg_df = df.groupby(["claw_ratio_init", "alpha_init", "sigma_init"]).agg(
        best_rmse_mean=("best_rmse", "mean"),
        hypervolume_mean=("hypervolume", "mean"),
        pareto_size_mean=("pareto_size", "mean"),
    ).reset_index()

    param_pairs = [
        ("claw_ratio_init", "alpha_init", "sigma_init", "σ_init"),
        ("claw_ratio_init", "sigma_init", "alpha_init", "α_init"),
        ("alpha_init", "sigma_init", "claw_ratio_init", "ρ_init"),
    ]

    metrics = [
        ("best_rmse_mean", "Best RMSE", "RdYlBu_r", True),
        ("hypervolume_mean", "Hypervolume", "RdYlBu", False),
    ]

    for p1, p2, p3, p3_label in param_pairs:
        # Average over p3
        pivot = agg_df.groupby([p1, p2])[["best_rmse_mean", "hypervolume_mean"]].mean().reset_index()

        for metric_col, metric_label, cmap, lower_better in metrics:
            fig, ax = plt.subplots(figsize=(8, 6.5))

            heatmap_data = pivot.pivot(index=p2, columns=p1, values=metric_col)

            # Ensure correct order
            heatmap_data = heatmap_data.reindex(
                index=sorted(heatmap_data.index),
                columns=sorted(heatmap_data.columns),
            )

            if lower_better:
                # Find best (lowest RMSE) for annotation
                best_val = heatmap_data.min().min()
                best_label = "★ Best"
            else:
                best_val = heatmap_data.max().max()
                best_label = "★ Best"

            sns.heatmap(
                heatmap_data,
                annot=True, fmt=".2f", cmap=cmap,
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": metric_label, "shrink": 0.82},
                ax=ax, square=True,
            )

            # Mark best cell
            best_pos = np.unravel_index(
                np.argmin(heatmap_data.values) if lower_better else np.argmax(heatmap_data.values),
                heatmap_data.values.shape,
            )
            ax.add_patch(plt.Rectangle(
                (best_pos[1], best_pos[0]), 1, 1,
                fill=False, edgecolor="#E63946", linewidth=3, linestyle="--",
            ))

            ax.set_title(
                f"Sensitivity: {PARAM_LABELS[p1]} × {PARAM_LABELS[p2]}\n"
                f"(Averaged over {PARAM_LABELS[p3]}, {len(df['run_id'].unique())} runs/grid-point)",
                fontsize=12, fontweight="bold",
            )
            ax.set_xlabel(PARAM_LABELS[p1], fontsize=11)
            ax.set_ylabel(PARAM_LABELS[p2], fontsize=11)

            plt.tight_layout()
            fname = f"heatmap_{p1}_vs_{p2}_{metric_col}"
            fig.savefig(output_dir / f"{fname}.png", dpi=150, bbox_inches="tight")
            fig.savefig(output_dir / f"{fname}.pdf", bbox_inches="tight")
            plt.close(fig)
            print(f"  [OK] Heatmap: {output_dir / fname}.png")


def _generate_interaction_plot(df: pd.DataFrame, output_dir: Path):
    """
    Generate a single comprehensive figure: 3×2 grid of line plots
    showing each parameter's marginal effect on RMSE and Hypervolume.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    params = ["claw_ratio_init", "alpha_init", "sigma_init"]
    metrics = [
        ("best_rmse", "Best RMSE", True),   # lower is better
        ("hypervolume", "Hypervolume", False),  # higher is better
    ]

    for row, (metric_col, metric_label, lower_better) in enumerate(metrics):
        for col, param in enumerate(params):
            ax = axes[row][col]

            # Aggregate over the OTHER 2 params
            other_params = [p for p in params if p != param]
            agg = df.groupby([param] + other_params)[metric_col].mean().reset_index()
            # Further average over other_params
            marginal = agg.groupby(param)[metric_col].agg(["mean", "std"]).reset_index()

            x = marginal[param]
            y = marginal["mean"]
            err = marginal["std"]

            color = "#E63946" if lower_better else "#457B9D"
            ax.errorbar(x, y, yerr=err, color=color, linewidth=2.5, marker="o",
                       markersize=10, capsize=5, markeredgecolor="white",
                       markeredgewidth=1.5, label="Mean ± Std")

            # Mark optimum
            opt_idx = np.argmin(y) if lower_better else np.argmax(y)
            ax.scatter([x.iloc[opt_idx]], [y.iloc[opt_idx]],
                      marker="*", s=300, c="#E63946", edgecolors="white",
                      linewidth=1, zorder=10,
                      label=f"Optimum: {x.iloc[opt_idx]}")

            ax.set_xlabel(PARAM_LABELS[param], fontsize=11)
            ax.set_ylabel(metric_label, fontsize=11)
            ax.set_title(f"{PARAM_LABELS[param]} → {metric_label}", fontsize=11, fontweight="bold")
            ax.legend(fontsize=8, framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle="--")

    fig.suptitle(
        "FCAA v2 Parameter Sensitivity: Marginal Effects\n"
        f"(Averaged across {len(df)} runs, Error bars = ±1 Std)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fig.savefig(output_dir / "interaction_plot.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / "interaction_plot.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Interaction plot: {output_dir / 'interaction_plot.png'}")


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="FCAA v2 Parameter Sensitivity Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n-runs", type=int, default=3,
                        help="Runs per grid point (default: 3)")
    parser.add_argument("--quick", action="store_true",
                        help="Fast mode: 1 run, 50 gens, 125 combos")
    parser.add_argument("--output-dir", type=str, default="results/sensitivity",
                        help="Output directory")
    parser.add_argument("--model", type=str, default="svr",
                        help="ML model (default: svr)")
    parser.add_argument("--dataset", type=str, default="B_data",
                        help="Dataset name")
    return parser.parse_args()


def main():
    args = parse_args()
    config = BASE_CONFIG.copy()

    if args.quick:
        config["n_runs"] = 1
        config["max_generations"] = 50
        config["pop_size"] = 40
        # Reduce grid size for quick test
        global PARAM_GRID
        PARAM_GRID = {
            "claw_ratio_init": [0.6, 0.8, 1.0],
            "alpha_init": [0.5, 1.0, 1.5],
            "sigma_init": [0.05, 0.15, 0.25],
        }
    else:
        config["n_runs"] = args.n_runs

    config["model"] = args.model
    config["data_file"] = f"{args.dataset}.xlsx" if not args.dataset.endswith(".xlsx") else args.dataset

    output_dir = _PROJECT_ROOT / args.output_dir

    run_sensitivity_grid(config, output_dir)


if __name__ == "__main__":
    main()
