#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick test: verify FCAA pipeline end-to-end with reduced params."""

import sys, os, time, warnings
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split

from src.algorithms.fcaa import FCAAOptimizer
from src.algorithms.nsga2 import NSGA2Optimizer
from src.algorithms.mopso import MOPSOOptimizer
from src.evaluators.feature_selection_hpo import FeatureSelectionHPOEvaluator
from src.algorithms.multi_objective import get_pareto_front
from src.utils.metrics import hypervolume, find_pareto_front
from src.utils.encoding import SolutionEncoding
from src.evaluators.models import get_model_wrapper

warnings.filterwarnings("ignore")

# ─── Config ───
DATA_FILE = "B_data.xlsx"
TARGET = "Tm of MD (K)"
MODEL = "svr"
POP_SIZE = 80
MAX_GEN = 200
N_RUNS = 1
POLY_DEGREE = 3
NOISE_FEAT = 20
CV_FOLDS = 5
SEED = 42

# FCAA v2 parameters
FCAA_ALPHA_INIT = 1.0
FCAA_SIGMA_INIT = 0.15
FCAA_SIGMA_FINAL = 0.002
FCAA_CLAW_RATIO_INIT = 0.80
FCAA_CLAW_RATIO_FINAL = 0.15
FCAA_ELITE_RATIO = 0.3
FCAA_ELITE_SIGMA = 0.015

# ─── Load Data ───
print("=" * 60)
print("MO-FCAA Quick Test")
print("=" * 60)

data_dir = Path(__file__).resolve().parent.parent / "data"
df = pd.read_excel(data_dir / DATA_FILE)
drop_cols = ["Systems", TARGET]
drop_cols = [c for c in drop_cols if c in df.columns]
X = df.drop(columns=drop_cols).select_dtypes(include=[np.number]).values
y = df[TARGET].values
print(f"Original: {X.shape[1]} features, {X.shape[0]} samples")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# Feature expansion
poly = PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)
X_train = poly.fit_transform(X_train)
X_test = poly.transform(X_test)

# Add noise distractors
rng = np.random.default_rng(SEED)
X_train = np.hstack([X_train, rng.normal(0, 1, (X_train.shape[0], NOISE_FEAT))])
X_test = np.hstack([X_test, rng.normal(0, 1, (X_test.shape[0], NOISE_FEAT))])

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(f"After expansion: {X_train.shape[1]} features")

# ─── Run Algorithms ───
results = {}
for algo_name in ["FCAA", "NSGA-II", "MOPSO"]:
    print(f"\n{'='*50}")
    print(f"Running {algo_name}")
    print(f"{'='*50}")

    evaluator = FeatureSelectionHPOEvaluator(
        X_train, y_train, model_name=MODEL, cv_folds=CV_FOLDS, n_jobs=4
    )

    t0 = time.perf_counter()
    if algo_name == "FCAA":
        opt = FCAAOptimizer(
            dimension=evaluator.get_dimension(),
            pop_size=POP_SIZE, max_generations=MAX_GEN,
            fitness_fn=evaluator,
            alpha_init=FCAA_ALPHA_INIT,
            sigma_init=FCAA_SIGMA_INIT,
            sigma_final=FCAA_SIGMA_FINAL,
            claw_ratio_init=FCAA_CLAW_RATIO_INIT,
            claw_ratio_final=FCAA_CLAW_RATIO_FINAL,
            elite_ratio=FCAA_ELITE_RATIO,
            elite_sigma=FCAA_ELITE_SIGMA,
            seed=SEED,
        )
    elif algo_name == "NSGA-II":
        opt = NSGA2Optimizer(dimension=evaluator.get_dimension(),
                           pop_size=POP_SIZE, max_generations=MAX_GEN,
                           fitness_fn=evaluator, seed=SEED)
    else:
        opt = MOPSOOptimizer(dimension=evaluator.get_dimension(),
                           pop_size=POP_SIZE, max_generations=MAX_GEN,
                           fitness_fn=evaluator, seed=SEED)

    pareto_pop, pareto_fit = opt.optimize(verbose=True)
    elapsed = time.perf_counter() - t0

    results[algo_name] = {
        "pareto_pop": pareto_pop,
        "pareto_fit": pareto_fit,
        "fitness_history": opt.fitness_history,
        "time": elapsed,
    }

    if len(pareto_fit) > 0:
        print(f"  Time: {elapsed:.1f}s, Pareto: {len(pareto_fit)}, "
              f"Best RMSE: {pareto_fit[:,0].min():.4f}, "
              f"Min features: {pareto_fit[:,1].min():.1%}")

# ─── Summary ───
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for algo_name in ["FCAA", "NSGA-II", "MOPSO"]:
    r = results[algo_name]
    pf = r["pareto_fit"]
    if len(pf) > 0:
        print(f"{algo_name}: Best RMSE={pf[:,0].min():.4f}, "
              f"Pareto size={len(pf)}, Time={r['time']:.1f}s")

# Best solutions detail
hp_bounds = get_model_wrapper(MODEL).hyperparameter_bounds()
encoding = SolutionEncoding(X_train.shape[1], hp_bounds)
for algo_name in ["FCAA", "NSGA-II", "MOPSO"]:
    pf = results[algo_name]["pareto_fit"]
    pp = results[algo_name]["pareto_pop"]
    if len(pf) == 0:
        continue
    best = np.argmin(pf[:, 0])
    mask, hparams, _ = encoding.decode(pp[best])
    print(f"\n{algo_name} best: RMSE={pf[best,0]:.4f}, "
          f"Features={mask.sum()}/{X_train.shape[1]} ({mask.sum()/X_train.shape[1]:.0%})")
    for k, v in hparams.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6g}")
        else:
            print(f"  {k}: {v}")

# ─── Plots ───
out_dir = Path(__file__).resolve().parent.parent / "results" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Pareto Front
fig, ax = plt.subplots(figsize=(10, 7))
colors = {"FCAA": "#E63946", "NSGA-II": "#457B9D", "MOPSO": "#2A9D8F"}
markers = {"FCAA": "o", "NSGA-II": "s", "MOPSO": "^"}

for algo_name in ["FCAA", "NSGA-II", "MOPSO"]:
    pf = results[algo_name]["pareto_fit"]
    if len(pf) == 0:
        continue
    # Get overall Pareto
    mask = find_pareto_front(pf)
    pareto = pf[mask]
    sorted_idx = np.argsort(pareto[:, 1])
    pareto = pareto[sorted_idx]

    ax.plot(pareto[:, 1] * 100, pareto[:, 0], color=colors[algo_name],
            linewidth=2.5, marker=markers[algo_name], markersize=10,
            markeredgecolor="white", markeredgewidth=1.5,
            label=f"{algo_name} (n={len(pareto)})", zorder=10)

ax.set_xlabel("Feature Retention Ratio (%)", fontsize=13)
ax.set_ylabel("Cross-Validated RMSE (K)", fontsize=13)
ax.set_title(f"Pareto Front: FCAA vs NSGA-II vs MOPSO\n({MODEL.upper()} on Boron HEC, {X_train.shape[1]} features)",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
# Auto-scale x-axis to actual data range with 5% margin
all_ratios = np.concatenate([results[a]["pareto_fit"][:, 1] for a in ["FCAA", "NSGA-II", "MOPSO"]
                             if len(results[a]["pareto_fit"]) > 0])
x_max = np.max(all_ratios) * 100 * 1.15  # 15% margin
x_max = max(x_max, 25)  # At least 25%
ax.set_xlim(0, x_max)
plt.tight_layout()
fig.savefig(out_dir / "pareto_front_comparison.png", dpi=150, bbox_inches="tight")
fig.savefig(out_dir / "pareto_front_comparison.pdf", bbox_inches="tight")
plt.close(fig)
print(f"\n[OK] Pareto front saved to {out_dir}")

# 2. Convergence
fig, ax = plt.subplots(figsize=(10, 6))
ref_point = np.array([np.max(np.vstack([r["pareto_fit"] for r in results.values()
                      if len(r["pareto_fit"]) > 0])[:, 0]) * 1.2, 1.1])

for algo_name in ["FCAA", "NSGA-II", "MOPSO"]:
    history = results[algo_name]["fitness_history"]
    hv_vals = []
    for gen_fit in history:
        if gen_fit is not None and len(gen_fit) > 0:
            hv_vals.append(hypervolume(gen_fit, ref_point))
    if hv_vals:
        ax.plot(range(1, len(hv_vals) + 1), hv_vals, color=colors[algo_name],
                linewidth=2, label=algo_name)

ax.set_xlabel("Generation", fontsize=13)
ax.set_ylabel("Hypervolume", fontsize=13)
ax.set_title(f"Convergence Comparison ({MODEL.upper()} on Boron HEC)", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(out_dir / "convergence_comparison.png", dpi=150, bbox_inches="tight")
fig.savefig(out_dir / "convergence_comparison.pdf", bbox_inches="tight")
plt.close(fig)
print("[OK] Convergence saved to", out_dir)

# ─── CSV ───
logs_dir = Path(__file__).resolve().parent.parent / "results" / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
rows = []
for algo_name in ["FCAA", "NSGA-II", "MOPSO"]:
    pf = results[algo_name]["pareto_fit"]
    if len(pf) > 0:
        rows.append({
            "algorithm": algo_name,
            "pareto_size": len(pf),
            "best_rmse": float(np.min(pf[:, 0])),
            "min_feature_ratio": float(np.min(pf[:, 1])),
            "time_seconds": results[algo_name]["time"],
        })
pd.DataFrame(rows).to_csv(logs_dir / "summary_results.csv", index=False)
print("[OK] CSV saved to", logs_dir)

print("\n" + "=" * 60)
print("Test complete! All outputs generated successfully.")
print("=" * 60)
