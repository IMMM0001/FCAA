#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MO-FCAA 30-Run Batch Experiment Script
======================================
Automated pipeline for running MO-FCAA and all baseline algorithms
(NGSA-II, MOPSO) for N independent runs with different random seeds.

Output structure (per run):
    results/30runs/
    ├── summary_30runs.csv              # Global summary
    ├── {algorithm}/
    │   └── {dataset}/
    │       └── {model}/
    │           ├── run_00_scalars.json  # RMSE, HV, time, etc.
    │           ├── run_00_arrays.npz    # Pareto pop + fitnesses
    │           ├── run_00_history.npy   # Fitness history sample
    │           └── ...

Designed for the Major Revision response to address:
  - Statistical rigor (30 runs → Wilcoxon, Friedman tests)
  - Algorithm generalization (multi-model, multi-dataset)
  - Reproducibility (seeded, structured, versioned output)

Usage:
    python experiments/run_experiments.py --n-runs 30 --dataset B_data --model svr
    python experiments/run_experiments.py --n-runs 30 --all-datasets --all-models
    python experiments/run_experiments.py --n-runs 5 --quick  # fast smoke test
"""

import sys
import os
import time
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# Fix Unicode encoding issues on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add src to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.evaluators.feature_selection_hpo import FeatureSelectionHPOEvaluator
from experiments.common import (
    load_and_preprocess_data,
    create_optimizer,
    compute_scalar_metrics,
    save_run_results,
)

warnings.filterwarnings("ignore")


# ==============================================================================
# Default Configuration
# ==============================================================================

# Dataset registry — maps short names to their configs
DATASET_REGISTRY = {
    "B_data": {
        "name": "B_data",
        "data_file": "B_data.xlsx",
        "target_column": "Tm of MD (K)",
        "drop_columns": ["Systems"],
    },
    "C_data": {
        "name": "C_data",
        "data_file": "C_data.xlsx",
        "target_column": "Tm of MD (K)",
        "drop_columns": ["Systems"],
    },
    "C_tm_forFS": {
        "name": "C_tm_forFS",
        "data_file": "C_tm_forFS.xlsx",
        "target_column": "Tm of MD (K)",
        "drop_columns": ["Systems"],
    },
}

# Model registry
MODEL_REGISTRY = ["svr", "krr", "random_forest", "mlp"]

# Algorithm registry
ALGORITHM_REGISTRY = ["FCAA", "NSGA-II", "MOPSO"]

# Default experiment config
DEFAULT_CONFIG = {
    # ── Reproducibility ──
    "seed": 42,
    "n_runs": 30,

    # ── Data preprocessing ──
    "test_size": 0.2,
    "poly_degree": 3,
    "add_noise_features": 20,

    # ── Optimization ──
    "pop_size": 80,
    "max_generations": 200,
    "cv_folds": 5,

    # ── FCAA v2-specific ──
    "fcaa_alpha_init": 1.0,
    "fcaa_sigma_init": 0.15,
    "fcaa_sigma_final": 0.002,
    "fcaa_claw_ratio_init": 0.80,
    "fcaa_claw_ratio_final": 0.15,
    "fcaa_elite_ratio": 0.3,
    "fcaa_elite_sigma": 0.015,

    # ── NSGA-II-specific ──
    "nsga2_crossover_prob": 0.9,
    "nsga2_eta_crossover": 20.0,
    "nsga2_eta_mutation": 20.0,

    # ── MOPSO-specific ──
    "mopso_archive_size": 50,
    "mopso_inertia": 0.5,
    "mopso_cognitive": 1.5,
    "mopso_social": 1.5,
    "mopso_mutation_rate": 0.1,

    # ── Output ──
    "output_dir": "results/30runs",
}


# ==============================================================================
# Core Experiment Runner
# ==============================================================================

def run_batch(
    config: Dict[str, Any],
    datasets: List[str],
    models: List[str],
    algorithms: List[str],
    n_runs: int,
    output_dir: Path,
    resume: bool = True,
) -> pd.DataFrame:
    """
    Run the full batch experiment across datasets, models, and algorithms.

    Parameters
    ----------
    config : dict
        Global configuration (DEFAULT_CONFIG + CLI overrides).
    datasets : list[str]
        Dataset names from DATASET_REGISTRY.
    models : list[str]
        Model names from MODEL_REGISTRY.
    algorithms : list[str]
        Algorithm names from ALGORITHM_REGISTRY.
    n_runs : int
        Number of independent runs per combination.
    output_dir : Path
        Root output directory for results.
    resume : bool
        If True, skip runs whose output files already exist.

    Returns
    -------
    summary_df : pd.DataFrame
        Combined summary of all completed runs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Compute total work ──
    total_combos = len(datasets) * len(models) * len(algorithms)
    total_runs = total_combos * n_runs

    print("=" * 72)
    print("MO-FCAA 30-Run Batch Experiment")
    print("=" * 72)
    print(f"  Datasets:   {datasets}")
    print(f"  Models:     {models}")
    print(f"  Algorithms: {algorithms}")
    print(f"  Runs each:  {n_runs}")
    print(f"  Total runs: {total_runs}")
    print(f"  Output:     {output_dir.resolve()}")
    print(f"  Resume:     {resume}")
    print("=" * 72)

    # ── Global progress bar ──
    global_pbar = tqdm(total=total_runs, desc="Total Progress", unit="run",
                       position=0, ncols=100)

    run_records = []  # For summary CSV
    start_time = datetime.now()

    for dataset_name in datasets:
        ds_cfg = DATASET_REGISTRY[dataset_name]

        # ── Load & preprocess data (once per dataset) ──
        print(f"\n{'─'*72}")
        print(f"Dataset: {dataset_name}")
        print(f"{'─'*72}")

        ds_config = {**config, **ds_cfg, "seed": config["seed"]}
        try:
            X_train, X_test, y_train, y_test = load_and_preprocess_data(ds_config)
        except Exception as e:
            print(f"  [ERROR] Failed to load {dataset_name}: {e}")
            global_pbar.update(n_runs * len(models) * len(algorithms))
            continue

        for model_name in models:
            # ── Pre-compute encoding info ──
            try:
                temp_eval = FeatureSelectionHPOEvaluator(
                    X_train=X_train, y_train=y_train,
                    model_name=model_name,
                    cv_folds=config["cv_folds"],
                )
                dim_info = temp_eval.get_dimension_info()
                print(f"\n  Model: {model_name}")
                print(f"    Dimension: {dim_info['total_dimension']} "
                      f"(features={dim_info['n_features']}, "
                      f"hparams={dim_info['n_hyperparams']})")
                del temp_eval
            except Exception as e:
                print(f"    [ERROR] Failed to init evaluator for {model_name}: {e}")
                global_pbar.update(n_runs * len(algorithms))
                continue

            for algo_name in algorithms:
                run_output_dir = output_dir / algo_name / dataset_name / model_name
                run_output_dir.mkdir(parents=True, exist_ok=True)

                # Describe progress
                desc = f"{algo_name:8s} | {dataset_name:10s} | {model_name:15s}"
                algo_pbar = tqdm(
                    total=n_runs, desc=desc, unit="run",
                    position=1, leave=False, ncols=100,
                )

                for run_id in range(n_runs):
                    seed = config["seed"] + run_id

                    # ── Skip existing runs when resuming ──
                    scalars_path = run_output_dir / f"run_{run_id:02d}_scalars.json"
                    if resume and scalars_path.exists():
                        try:
                            with open(scalars_path, "r") as f:
                                record = json.load(f)
                            record["dataset"] = dataset_name
                            record["model"] = model_name
                            record["algorithm"] = algo_name
                            run_records.append(record)
                            algo_pbar.update(1)
                            global_pbar.update(1)
                            continue
                        except Exception:
                            pass  # Corrupt file → re-run

                    # ── Run one optimization ──
                    try:
                        evaluator = FeatureSelectionHPOEvaluator(
                            X_train=X_train, y_train=y_train,
                            model_name=model_name,
                            cv_folds=config["cv_folds"],
                        )

                        t0 = time.perf_counter()
                        optimizer = create_optimizer(algo_name, evaluator, config, run_id)
                        pareto_pop, pareto_fit = optimizer.optimize(verbose=False)
                        elapsed = time.perf_counter() - t0

                        # Build result dict
                        result = {
                            "name": algo_name,
                            "run": run_id,
                            "seed": seed,
                            "pareto_population": pareto_pop,
                            "pareto_fitnesses": pareto_fit,
                            "fitness_history": optimizer.fitness_history,
                            "time": elapsed,
                            "n_evaluations": evaluator.n_evaluations,
                        }

                        # Save to disk
                        save_run_results(result, run_output_dir, run_id, save_arrays=True)

                        # Collect metrics for summary
                        metrics = compute_scalar_metrics(
                            pareto_fit, optimizer.fitness_history, elapsed, evaluator.n_evaluations
                        )
                        metrics["algorithm"] = algo_name
                        metrics["dataset"] = dataset_name
                        metrics["model"] = model_name
                        metrics["run_id"] = run_id
                        metrics["seed"] = seed
                        run_records.append(metrics)

                    except Exception as e:
                        tqdm.write(f"  [ERROR] {algo_name}/{dataset_name}/{model_name} "
                                   f"run {run_id}: {e}")
                        # Record failed run with error marker
                        run_records.append({
                            "algorithm": algo_name,
                            "dataset": dataset_name,
                            "model": model_name,
                            "run_id": run_id,
                            "seed": seed,
                            "best_rmse": None,
                            "pareto_size": 0,
                            "error": str(e),
                        })

                    algo_pbar.update(1)
                    global_pbar.update(1)

                algo_pbar.close()

    global_pbar.close()

    # ── Save Summary CSV ──
    summary_df = pd.DataFrame(run_records)
    summary_path = output_dir / "summary_30runs.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[OK] Summary saved to {summary_path}")

    # ── Print quick overview ──
    _print_summary(summary_df, start_time)

    return summary_df


def _print_summary(df: pd.DataFrame, start_time: datetime):
    """Print a compact summary of the batch experiment results."""
    elapsed = datetime.now() - start_time

    print(f"\n{'='*72}")
    print(f"BATCH EXPERIMENT COMPLETE")
    print(f"  Total wall time: {elapsed}")
    print(f"  Total runs:      {len(df)}")

    if "error" in df.columns:
        errors = df[df["error"].notna()]
        if len(errors) > 0:
            print(f"  Failed runs:     {len(errors)}")
            for _, row in errors.iterrows():
                print(f"    - {row['algorithm']}/{row['dataset']}/{row['model']} "
                      f"run {row['run_id']}: {row['error']}")

    if "best_rmse" in df.columns and df["best_rmse"].notna().any():
        valid = df[df["best_rmse"].notna()]
        print(f"\n  Best RMSE by Algorithm (across all datasets/models):")
        for algo in sorted(valid["algorithm"].unique()):
            algo_df = valid[valid["algorithm"] == algo]
            if len(algo_df) == 0:
                continue
            best = algo_df.loc[algo_df["best_rmse"].idxmin()]
            print(f"    {algo:10s}: {best['best_rmse']:.4f} "
                  f"({best['dataset']}/{best['model']}, run {best['run_id']})")

    print(f"{'='*72}\n")


# ==============================================================================
# Quick-Test Mode
# ==============================================================================

def run_quick_test(config: Dict[str, Any], output_dir: Path):
    """
    Fast smoke test: 5 runs × 50 generations × 1 dataset × 1 model.
    Validates the pipeline end-to-end before committing to a full 30-run job.
    """
    print("=" * 72)
    print("MO-FCAA Quick Smoke Test (reduced params)")
    print("=" * 72)

    test_config = {
        **config,
        "n_runs": 5,
        "max_generations": 50,
        "pop_size": 40,
    }

    df = run_batch(
        config=test_config,
        datasets=["B_data"],
        models=["svr"],
        algorithms=["FCAA", "NSGA-II", "MOPSO"],
        n_runs=5,
        output_dir=output_dir,
        resume=False,
    )
    return df


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="MO-FCAA 30-Run Batch Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full 30-run experiment on B_data with SVR only
  python experiments/run_experiments.py --n-runs 30 --dataset B_data --model svr

  # 30 runs across all datasets and all models (very long!)
  python experiments/run_experiments.py --n-runs 30 --all-datasets --all-models

  # Quick smoke test (5 runs, reduced gens)
  python experiments/run_experiments.py --quick

  # Resume interrupted run (skip existing output files)
  python experiments/run_experiments.py --n-runs 30 --resume
        """,
    )

    # ── Experiment size ──
    parser.add_argument("--n-runs", type=int, default=30,
                        help="Number of independent runs per combination (default: 30)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test: 5 runs × 50 gens × 1 dataset × 1 model")

    # ── Dataset selection ──
    ds_group = parser.add_mutually_exclusive_group()
    ds_group.add_argument("--dataset", type=str, default="B_data",
                          choices=list(DATASET_REGISTRY.keys()),
                          help="Single dataset (default: B_data)")
    ds_group.add_argument("--all-datasets", action="store_true",
                          help="Run on all available datasets")

    # ── Model selection ──
    mdl_group = parser.add_mutually_exclusive_group()
    mdl_group.add_argument("--model", type=str, default="svr",
                           choices=MODEL_REGISTRY,
                           help="Single ML model (default: svr)")
    mdl_group.add_argument("--all-models", action="store_true",
                           help="Run on all available models")

    # ── Algorithm selection ──
    alg_group = parser.add_mutually_exclusive_group()
    alg_group.add_argument("--algorithm", type=str, default=None,
                           choices=ALGORITHM_REGISTRY,
                           help="Single algorithm (default: all three)")
    alg_group.add_argument("--algorithms", type=str, nargs="+",
                           default=None,
                           help="Space-separated list of algorithms, e.g. --algorithms FCAA NSGA-II")

    # ── Optimization params ──
    parser.add_argument("--pop-size", type=int, default=None,
                        help="Population size (default: 80)")
    parser.add_argument("--max-generations", type=int, default=None,
                        help="Max generations (default: 200)")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Cross-validation folds (default: 5)")
    parser.add_argument("--poly-degree", type=int, default=3,
                        help="Polynomial expansion degree (default: 3)")

    # ── I/O ──
    parser.add_argument("--output-dir", type=str, default="results/30runs",
                        help="Output directory (default: results/30runs)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (default: 42)")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Do NOT skip existing runs — re-run everything")
    parser.set_defaults(resume=True)

    return parser.parse_args()


def main():
    args = parse_args()
    config = DEFAULT_CONFIG.copy()

    # ── Apply CLI overrides ──
    config["seed"] = args.seed
    config["n_runs"] = args.n_runs
    config["cv_folds"] = args.cv_folds
    config["poly_degree"] = args.poly_degree

    if args.pop_size is not None:
        config["pop_size"] = args.pop_size
    if args.max_generations is not None:
        config["max_generations"] = args.max_generations

    output_dir = _PROJECT_ROOT / args.output_dir

    # ── Quick test mode ──
    if args.quick:
        run_quick_test(config, output_dir)
        return

    # ── Resolve dataset list ──
    if args.all_datasets:
        datasets = list(DATASET_REGISTRY.keys())
    else:
        datasets = [args.dataset]

    # ── Resolve model list ──
    if args.all_models:
        models = list(MODEL_REGISTRY)
    else:
        models = [args.model]

    # ── Resolve algorithm list ──
    if args.algorithms:
        algorithms = args.algorithms
    elif args.algorithm:
        algorithms = [args.algorithm]
    else:
        algorithms = list(ALGORITHM_REGISTRY)  # Default: all three

    # Validate
    for algo in algorithms:
        if algo not in ALGORITHM_REGISTRY:
            print(f"[ERROR] Unknown algorithm: '{algo}'. Available: {ALGORITHM_REGISTRY}")
            sys.exit(1)
    for ds in datasets:
        if ds not in DATASET_REGISTRY:
            print(f"[ERROR] Unknown dataset: '{ds}'. Available: {list(DATASET_REGISTRY.keys())}")
            sys.exit(1)
    for mdl in models:
        if mdl not in MODEL_REGISTRY:
            print(f"[ERROR] Unknown model: '{mdl}'. Available: {MODEL_REGISTRY}")
            sys.exit(1)

    # ── Run batch ──
    run_batch(
        config=config,
        datasets=datasets,
        models=models,
        algorithms=algorithms,
        n_runs=args.n_runs,
        output_dir=output_dir,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
