#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared utilities for MO-FCAA experiment scripts.

Provides reusable functions for data loading, optimizer construction,
and result serialization — shared by run_comparison.py, run_experiments.py,
and future experiment scripts.

╔══════════════════════════════════════════════════════════════════════╗
║  HOW TO ADD A NEW BASELINE ALGORITHM (e.g., NSGA-III, MOEA/D):      ║
║                                                                      ║
║  1. Create your algorithm class in src/algorithms/ (e.g., nsga3.py) ║
║     - MUST extend BaseOptimizer                                     ║
║     - MUST implement optimize() → (population, fitnesses)           ║
║     - MUST accept: dimension, pop_size, max_generations,            ║
║       fitness_fn, lower_bound, upper_bound, seed                    ║
║                                                                      ║
║  2. Add an entry to OPTIMIZER_REGISTRY below with:                  ║
║     - optimizer_class: your class                                   ║
║     - extra_params: dict of default kwargs passed to constructor    ║
║                                                                      ║
║  3. That's it — create_optimizer(), run_experiments.py, and         ║
║     stats_analyzer.py will automatically work with the new algo     ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split

# Ensure src/ is on the import path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.algorithms.fcaa import FCAAOptimizer
from src.algorithms.nsga2 import NSGA2Optimizer
from src.algorithms.mopso import MOPSOOptimizer
from src.evaluators.feature_selection_hpo import FeatureSelectionHPOEvaluator
from src.algorithms.multi_objective import get_pareto_front
from src.utils.metrics import hypervolume, find_pareto_front


# ==============================================================================
# 1. Data Loading & Preprocessing
# ==============================================================================

def load_and_preprocess_data(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a materials science dataset and preprocess for feature selection + HPO.

    Pipeline:
    1. Load Excel file from data/
    2. Drop non-feature columns (e.g. 'Systems') and target
    3. Train/test split
    4. Polynomial feature expansion
    5. Add Gaussian noise features as distractors
    6. Standardize (zero mean, unit variance)

    Parameters
    ----------
    config : dict
        Required keys:
        - data_file: str           — filename in data/
        - target_column: str       — target column name
        - drop_columns: list[str]  — non-feature columns to drop
        - seed: int                — random seed for split & noise
        - test_size: float         — test fraction (default 0.2)
        - poly_degree: int         — polynomial expansion degree (1 = none)
        - add_noise_features: int  — number of noise distractors (0 = none)

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
    """
    data_dir = _PROJECT_ROOT / "data"
    filepath = data_dir / config["data_file"]

    print(f"Loading dataset: {filepath}")
    df = pd.read_excel(filepath)

    # Separate features and target
    drop_cols = list(config.get("drop_columns", [])) + [config["target_column"]]
    drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop_cols).select_dtypes(include=[np.number]).values
    y = df[config["target_column"]].values

    print(f"  Original features: {X.shape[1]}, Samples: {X.shape[0]}")

    # Train/test split
    test_size = config.get("test_size", 0.2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=config.get("seed", 42)
    )

    # Polynomial feature expansion
    poly_degree = config.get("poly_degree", 1)
    if poly_degree > 1:
        poly = PolynomialFeatures(
            degree=poly_degree,
            include_bias=False,
            interaction_only=False,
        )
        X_train = poly.fit_transform(X_train)
        X_test = poly.transform(X_test)
        print(f"  After poly features (degree={poly_degree}): {X_train.shape[1]}")

    # Add noise features as distractors
    n_noise = config.get("add_noise_features", 0)
    if n_noise > 0:
        rng = np.random.default_rng(config.get("seed", 42))
        noise_train = rng.normal(0, 1, size=(X_train.shape[0], n_noise))
        noise_test = rng.normal(0, 1, size=(X_test.shape[0], n_noise))
        X_train = np.hstack([X_train, noise_train])
        X_test = np.hstack([X_test, noise_test])
        print(f"  After adding {n_noise} noise features: {X_train.shape[1]}")

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"  Final: X_train={X_train.shape}, X_test={X_test.shape}\n")

    return X_train, X_test, y_train, y_test


# ==============================================================================
# 2. Optimizer Registry & Factory
# ==============================================================================

# Standardized registry of available algorithms.
# To add a new baseline (e.g. NSGA-III, MOEA/D):
#   1. Create your optimizer class extending BaseOptimizer
#   2. Add an entry here with the class and its default extra params
OPTIMIZER_REGISTRY: Dict[str, Dict[str, Any]] = {
    "FCAA": {
        "class": FCAAOptimizer,
        "extra_params": {
            "alpha_init": 1.0,
            "sigma_init": 0.15,
            "sigma_final": 0.002,
            "claw_ratio_init": 0.80,
            "claw_ratio_final": 0.15,
            "elite_ratio": 0.3,
            "elite_sigma": 0.015,
        },
    },
    "NSGA-II": {
        "class": NSGA2Optimizer,
        "extra_params": {
            "crossover_prob": 0.9,
            "eta_crossover": 20.0,
            "eta_mutation": 20.0,
        },
    },
    "MOPSO": {
        "class": MOPSOOptimizer,
        "extra_params": {
            "archive_size": 50,
            "inertia": 0.5,
            "cognitive": 1.5,
            "social": 1.5,
            "mutation_rate": 0.1,
        },
    },
    # ── Reserved slots for future baselines ──
    # "NSGA-III": {
    #     "class": NSGA3Optimizer,
    #     "extra_params": {
    #         "n_divisions": 12,
    #         "crossover_prob": 0.9,
    #         "eta_crossover": 20.0,
    #         "eta_mutation": 20.0,
    #     },
    # },
    # "MOEA/D": {
    #     "class": MOEADOptimizer,
    #     "extra_params": {
    #         "n_neighbors": 20,
    #         "crossover_prob": 0.9,
    #         "eta_mutation": 20.0,
    #         "replacement_prob": 0.01,
    #     },
    # },
}

# Shorthand access
ALGORITHM_NAMES = list(OPTIMIZER_REGISTRY.keys())


def create_optimizer(
    algo_name: str,
    evaluator: FeatureSelectionHPOEvaluator,
    config: Dict[str, Any],
    run_id: int = 0,
):
    """
    Create an optimizer instance for a given algorithm.

    Does NOT modify any algorithm core logic — purely a construction helper.

    Parameters
    ----------
    algo_name : str
        One of 'FCAA', 'NSGA-II', 'MOPSO'.
    evaluator : FeatureSelectionHPOEvaluator
        Fresh evaluator instance (resets counter & cache per run).
    config : dict
        Configuration dictionary with algorithm-specific parameters.
    run_id : int
        Run index (0-based); seed = config['seed'] + run_id.

    Returns
    -------
    optimizer : BaseOptimizer subclass
    """
    if algo_name not in OPTIMIZER_REGISTRY:
        raise ValueError(
            f"Unknown algorithm: '{algo_name}'. "
            f"Available: {ALGORITHM_NAMES}"
        )

    entry = OPTIMIZER_REGISTRY[algo_name]
    algo_class = entry["class"]
    extra_params = dict(entry["extra_params"])  # copy defaults

    # Allow config overrides for each algorithm's extra params
    prefix_map = {
        "FCAA": "fcaa_",
        "NSGA-II": "nsga2_",
        "MOPSO": "mopso_",
    }
    prefix = prefix_map.get(algo_name, algo_name.lower().replace("-", "_") + "_")
    for key in list(extra_params.keys()):
        config_key = prefix + key
        if config_key in config:
            extra_params[key] = config[config_key]

    # Core params (required by all BaseOptimizer subclasses)
    core_params = {
        "dimension": evaluator.get_dimension(),
        "pop_size": config["pop_size"],
        "max_generations": config["max_generations"],
        "fitness_fn": evaluator,
        "lower_bound": 0.0,
        "upper_bound": 1.0,
        "seed": config.get("seed", 42) + run_id,
    }

    return algo_class(**core_params, **extra_params)


# ==============================================================================
# 3. Metric Computation
# ==============================================================================

def compute_scalar_metrics(
    pareto_fitnesses: np.ndarray,
    fitness_history: List[np.ndarray],
    elapsed_time: float,
    n_evaluations: int,
) -> Dict[str, Any]:
    """
    Compute summary scalar metrics from a completed optimization run.

    Parameters
    ----------
    pareto_fitnesses : np.ndarray, shape (K, 2)
        Final Pareto front fitnesses: [RMSE, feature_ratio].
    fitness_history : list of np.ndarray
        Population fitnesses at each generation.
    elapsed_time : float
        Wall-clock time in seconds.
    n_evaluations : int
        Total number of fitness evaluations.

    Returns
    -------
    metrics : dict
        Scalars suitable for JSON serialization.
    """
    if len(pareto_fitnesses) == 0:
        return {
            "best_rmse": None,
            "mean_rmse": None,
            "median_rmse": None,
            "worst_rmse": None,
            "best_feature_ratio": None,
            "mean_feature_ratio": None,
            "pareto_size": 0,
            "hypervolume": None,
            "time_seconds": float(elapsed_time),
            "n_evaluations": int(n_evaluations),
            "warning": "Empty Pareto front",
        }

    rmse_col = pareto_fitnesses[:, 0]
    feat_col = pareto_fitnesses[:, 1]

    metrics = {
        "best_rmse": float(np.min(rmse_col)),
        "mean_rmse": float(np.mean(rmse_col)),
        "median_rmse": float(np.median(rmse_col)),
        "worst_rmse": float(np.max(rmse_col)),
        "best_feature_ratio": float(np.min(feat_col)),
        "mean_feature_ratio": float(np.mean(feat_col)),
        "pareto_size": int(len(pareto_fitnesses)),
        "time_seconds": float(elapsed_time),
        "n_evaluations": int(n_evaluations),
    }

    # Hypervolume (reference point: 10% worse than max RMSE, feature ratio max 1.05)
    try:
        ref_point = np.array([np.max(rmse_col) * 1.1, 1.05])
        metrics["hypervolume"] = float(hypervolume(pareto_fitnesses, ref_point))
    except Exception:
        metrics["hypervolume"] = None

    # Hypervolume convergence curve
    try:
        ref_point_hv = np.array([np.max(rmse_col) * 1.1, 1.05])
        hv_history = []
        for gen_fit in fitness_history:
            if gen_fit is not None and len(gen_fit) > 0:
                hv_history.append(float(hypervolume(gen_fit, ref_point_hv)))
            else:
                hv_history.append(None)
        metrics["hypervolume_history"] = hv_history
    except Exception:
        metrics["hypervolume_history"] = []

    return metrics


def sample_fitness_history(
    fitness_history: List[np.ndarray],
    max_samples: int = 50,
) -> List[np.ndarray]:
    """
    Downsample fitness history for compact NPZ storage.

    Uniformly samples at most `max_samples` snapshots from the history,
    always including the first and last generations.

    Parameters
    ----------
    fitness_history : list of np.ndarray
        Full fitness history (one array per generation).
    max_samples : int
        Maximum number of snapshots to keep.

    Returns
    -------
    list of np.ndarray
        Subsampled history.
    """
    n = len(fitness_history)
    if n <= max_samples:
        return fitness_history

    indices = np.linspace(0, n - 1, max_samples, dtype=int)
    return [fitness_history[i] for i in indices]


# ==============================================================================
# 4. Run Result Collector
# ==============================================================================

def run_single_algorithm(
    algo_name: str,
    evaluator: FeatureSelectionHPOEvaluator,
    config: Dict[str, Any],
    run_id: int = 0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run a single optimization algorithm and collect results.

    Parameters
    ----------
    algo_name : str
        Algorithm name.
    evaluator : FeatureSelectionHPOEvaluator
        Fresh evaluator instance.
    config : dict
        Global configuration.
    run_id : int
        Run index for seed offset.
    verbose : bool
        Print progress during optimization.

    Returns
    -------
    result : dict
        Keys: name, run, pareto_population, pareto_fitnesses,
              fitness_history, time, n_evaluations
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running {algo_name} (run {run_id + 1})")
        print(f"  Dimension: {evaluator.get_dimension()}")
        print(f"  Population: {config['pop_size']}, Generations: {config['max_generations']}")
        print(f"{'='*60}")

    optimizer = create_optimizer(algo_name, evaluator, config, run_id)

    t0 = time.perf_counter()
    pareto_pop, pareto_fit = optimizer.optimize(verbose=verbose)
    elapsed = time.perf_counter() - t0

    if verbose:
        print(f"  Time: {elapsed:.1f}s, Pareto size: {len(pareto_fit)}")

    return {
        "name": algo_name,
        "run": run_id,
        "seed": config.get("seed", 42) + run_id,
        "pareto_population": pareto_pop,
        "pareto_fitnesses": pareto_fit,
        "fitness_history": optimizer.fitness_history,
        "time": elapsed,
        "n_evaluations": evaluator.n_evaluations,
    }


# ==============================================================================
# 5. I/O Helpers
# ==============================================================================

def save_run_results(
    result: Dict[str, Any],
    output_dir: Path,
    run_id: int,
    save_arrays: bool = True,
) -> Tuple[Path, Optional[Path]]:
    """
    Save a single run's results as JSON (scalars) and NPZ (arrays).

    Parameters
    ----------
    result : dict
        From run_single_algorithm().
    output_dir : Path
        Directory to save into (created if needed).
    run_id : int
        Run index (used in filename).
    save_arrays : bool
        If True, also save Pareto arrays and fitness history as NPZ.

    Returns
    -------
    (json_path, npz_path_or_None)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute scalar metrics
    metrics = compute_scalar_metrics(
        result["pareto_fitnesses"],
        result["fitness_history"],
        result["time"],
        result["n_evaluations"],
    )
    # Attach metadata
    metrics["algorithm"] = result["name"]
    metrics["run_id"] = run_id
    metrics["seed"] = result["seed"]

    # Save scalars as JSON
    json_path = output_dir / f"run_{run_id:02d}_scalars.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=_json_default)

    # Save arrays as NPZ
    npz_path = None
    if save_arrays:
        npz_path = output_dir / f"run_{run_id:02d}_arrays.npz"
        history_sample = sample_fitness_history(result["fitness_history"], max_samples=50)
        # Pad history to uniform shape for NPZ
        np.savez_compressed(
            npz_path,
            pareto_population=result["pareto_population"],
            pareto_fitnesses=result["pareto_fitnesses"],
            fitness_history_final=result["fitness_history"][-1]
            if result["fitness_history"] else np.array([]),
            allow_pickle=False,
        )
        # Save history separately as NPY (variable-length list)
        history_npy_path = output_dir / f"run_{run_id:02d}_history.npy"
        np.save(history_npy_path, np.array(history_sample, dtype=object), allow_pickle=True)

    return json_path, npz_path


def _json_default(obj):
    """Handle numpy types in JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def load_run_scalars(json_path: Path) -> Dict[str, Any]:
    """Load scalar metrics from a run JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_all_runs(
    base_dir: Path,
) -> pd.DataFrame:
    """
    Walk the 30runs output directory and collect all scalar JSONs into a DataFrame.

    Parameters
    ----------
    base_dir : Path
        Root of the 30runs output (e.g., results/30runs/).

    Returns
    -------
    df : pd.DataFrame
        One row per run, with columns: algorithm, dataset, model, run_id, seed,
        best_rmse, pareto_size, hypervolume, time_seconds, ...
    """
    records = []
    for json_file in sorted(base_dir.rglob("*_scalars.json")):
        try:
            data = load_run_scalars(json_file)
            # Infer dataset and model from path structure:
            #   results/30runs/{algo}/{dataset}/{model}/run_XX_scalars.json
            parts = json_file.relative_to(base_dir).parts
            if len(parts) >= 4:
                data["algorithm"] = parts[0]
                data["dataset"] = parts[1]
                data["model"] = parts[2]
            records.append(data)
        except Exception as e:
            print(f"  [WARN] Failed to load {json_file}: {e}")

    return pd.DataFrame(records)
