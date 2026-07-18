#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistical Analysis Script for MO-FCAA 30-Run Experiments
==========================================================
Reads the structured output of run_experiments.py and produces:

1. Descriptive Statistics: Mean, Std, Median, 95% CI per algorithm
2. Wilcoxon Signed-Rank Test: Pairwise comparison (FCAA vs baselines)
3. Friedman Test: Omnibus test across all algorithms
4. Nemenyi Post-Hoc Test: Critical Difference (CD) diagram data
5. Publication-Quality Figures:
   - Boxplot of Best RMSE by algorithm
   - Critical Difference diagram

Output: Markdown report + LaTeX-ready tables + PNG/PDF figures.

Usage:
    python experiments/stats_analyzer.py --results-dir results/30runs
    python experiments/stats_analyzer.py --results-dir results/30runs --metric best_rmse
"""

import sys
import os
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Style ──
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

# ==============================================================================
# 1. Data Loading
# ==============================================================================

def load_results(results_dir: Path, metric: str = "best_rmse") -> pd.DataFrame:
    """
    Load all run scalar JSONs and return a tidy DataFrame.

    Parameters
    ----------
    results_dir : Path
        Root directory containing algorithm/dataset/model/run_*_scalars.json.
    metric : str
        Primary metric column to extract.

    Returns
    -------
    df : pd.DataFrame
        Columns: algorithm, dataset, model, run_id, seed, {metric}, pareto_size, ...
    """
    records = []
    for json_path in sorted(results_dir.rglob("*_scalars.json")):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  [WARN] Skipping {json_path}: {e}")
            continue

        # Infer metadata from path structure
        parts = json_path.relative_to(results_dir).parts
        if len(parts) >= 4:
            data["algorithm"] = parts[0]
            data["dataset"] = parts[1]
            data["model"] = parts[2]

        # Extract filename run_id
        fname = json_path.stem  # e.g., "run_00_scalars"
        try:
            data["run_id"] = int(fname.split("_")[1])
        except (IndexError, ValueError):
            data["run_id"] = data.get("run_id", -1)

        records.append(data)

    if not records:
        raise FileNotFoundError(f"No *_scalars.json files found under {results_dir}")

    df = pd.DataFrame(records)

    # Ensure metric column exists
    if metric not in df.columns:
        available = [c for c in df.columns if c not in ("hypervolume_history",)]
        raise KeyError(f"Metric '{metric}' not found. Available: {available}")

    return df


# ==============================================================================
# 2. Descriptive Statistics
# ==============================================================================

def compute_descriptive_stats(
    df: pd.DataFrame,
    metric: str = "best_rmse",
    group_col: str = "algorithm",
) -> pd.DataFrame:
    """
    Compute descriptive statistics per algorithm.

    Returns DataFrame with columns:
    Algorithm, N, Mean, Std, Median, Min, Max, CI_95_Lower, CI_95_Upper
    """
    results = []
    for name, group in df.groupby(group_col):
        values = group[metric].dropna().values
        n = len(values)
        if n < 2:
            results.append({
                "Algorithm": name, "N": n,
                "Mean": np.nan, "Std": np.nan, "Median": np.nan,
                "Min": np.nan, "Max": np.nan,
                "CI_95_Lower": np.nan, "CI_95_Upper": np.nan,
            })
            continue

        mean = np.mean(values)
        std = np.std(values, ddof=1)
        sem = std / np.sqrt(n)
        ci = stats.t.interval(0.95, df=n - 1, loc=mean, scale=sem)

        results.append({
            "Algorithm": name,
            "N": n,
            "Mean": mean,
            "Std": std,
            "Median": np.median(values),
            "Min": np.min(values),
            "Max": np.max(values),
            "CI_95_Lower": ci[0],
            "CI_95_Upper": ci[1],
        })

    return pd.DataFrame(results)


def format_descriptive_table(stats_df: pd.DataFrame, metric_name: str = "Best RMSE") -> str:
    """Format descriptive statistics as a Markdown/LaTeX table."""
    lines = []
    lines.append(f"## Descriptive Statistics: {metric_name}")
    lines.append("")
    lines.append("| Algorithm | N | Mean | Std | Median | 95% CI | Min | Max |")
    lines.append("|-----------|----|------|-----|--------|--------|-----|-----|")

    for _, row in stats_df.iterrows():
        ci_str = f"[{row['CI_95_Lower']:.4f}, {row['CI_95_Upper']:.4f}]"
        lines.append(
            f"| {row['Algorithm']} | {int(row['N'])} | "
            f"{row['Mean']:.4f} | {row['Std']:.4f} | {row['Median']:.4f} | "
            f"{ci_str} | {row['Min']:.4f} | {row['Max']:.4f} |"
        )

    return "\n".join(lines)


# ==============================================================================
# 3. Wilcoxon Signed-Rank Test (Paired)
# ==============================================================================

def run_wilcoxon_tests(
    df: pd.DataFrame,
    metric: str = "best_rmse",
    target_algo: str = "FCAA",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Paired Wilcoxon signed-rank test: target vs each baseline.

    Uses the 30 paired observations (same dataset × model × run_id).
    """
    target_data = df[df["algorithm"] == target_algo].copy()
    baselines = [a for a in df["algorithm"].unique() if a != target_algo]

    results = []
    for baseline in baselines:
        baseline_data = df[df["algorithm"] == baseline].copy()

        # Merge on dataset, model, run_id to pair observations
        merged = target_data.merge(
            baseline_data,
            on=["dataset", "model", "run_id"],
            suffixes=("_target", "_baseline"),
        )

        target_vals = merged[f"{metric}_target"].dropna().values
        baseline_vals = merged[f"{metric}_baseline"].dropna().values

        if len(target_vals) < 5:
            results.append({
                "Comparison": f"{target_algo} vs {baseline}",
                "N_pairs": len(target_vals),
                "Statistic": np.nan, "p_value": np.nan,
                "Significant": "N/A",
                "Effect_Size_r": np.nan,
            })
            continue

        # Wilcoxon signed-rank test
        try:
            # For zero-handling: use exact or normal approximation
            diff = target_vals - baseline_vals
            # Remove exact zeros (wilcoxon with zero_method='zsplit')
            non_zero = diff != 0
            if non_zero.sum() < 5:
                stat, p = np.nan, np.nan
            else:
                result = stats.wilcoxon(target_vals, baseline_vals, alternative="two-sided")
                stat, p = result.statistic, result.pvalue
        except Exception:
            stat, p = np.nan, np.nan

        # Effect size r = Z / sqrt(N)
        try:
            z_stat = stats.norm.ppf(min(p / 2, 1 - 1e-15))  # approximate Z from p
            # Better: directly compute from Wilcoxon
            z_stat = stats.norm.ppf(1 - p / 2) if not np.isnan(p) else np.nan
            effect_r = abs(z_stat) / np.sqrt(len(target_vals)) if not np.isnan(z_stat) else np.nan
        except Exception:
            effect_r = np.nan

        sig = "Yes *" if (not np.isnan(p) and p < alpha) else "No"

        results.append({
            "Comparison": f"{target_algo} vs {baseline}",
            "N_pairs": len(target_vals),
            "Statistic": stat,
            "p_value": p,
            "Significant": sig,
            "Effect_Size_r": effect_r,
        })

    return pd.DataFrame(results)


def format_wilcoxon_table(wilcoxon_df: pd.DataFrame, alpha: float = 0.05) -> str:
    """Format Wilcoxon results as Markdown."""
    lines = []
    lines.append(f"## Wilcoxon Signed-Rank Test (α = {alpha})")
    lines.append("")
    lines.append("| Comparison | N | W Statistic | p-value | Significant | Effect Size (r) |")
    lines.append("|------------|---|-------------|---------|-------------|-----------------|")

    for _, row in wilcoxon_df.iterrows():
        p_str = f"{row['p_value']:.6f}" if not np.isnan(row['p_value']) else "N/A"
        r_str = f"{row['Effect_Size_r']:.4f}" if not np.isnan(row['Effect_Size_r']) else "N/A"
        lines.append(
            f"| {row['Comparison']} | {int(row['N_pairs'])} | "
            f"{row['Statistic']:.1f} | {p_str} | {row['Significant']} | {r_str} |"
        )

    lines.append("")
    lines.append("*Interpretation:* p < 0.05 indicates a statistically significant difference. "
                  "Effect size r ≈ |Z|/√N; r < 0.3 (small), 0.3–0.5 (medium), > 0.5 (large).")
    return "\n".join(lines)


# ==============================================================================
# 4. Friedman Test + Nemenyi Post-Hoc
# ==============================================================================

def run_friedman_test(
    df: pd.DataFrame,
    metric: str = "best_rmse",
    alpha: float = 0.05,
) -> Dict:
    """
    Friedman test: non-parametric equivalent of repeated-measures ANOVA.

    Requires: each algorithm evaluated on the same set of "problems"
    (dataset × model combinations), each with the same number of runs.
    """
    algorithms = sorted(df["algorithm"].unique())
    if len(algorithms) < 2:
        return {"error": "Need at least 2 algorithms for Friedman test."}

    # Pivot: each row = one complete observation (dataset × model × run_id)
    # columns = algorithm, values = metric
    pivot = df.pivot_table(
        index=["dataset", "model", "run_id"],
        columns="algorithm",
        values=metric,
    ).dropna()

    if len(pivot) < 5:
        return {"error": f"Not enough paired observations ({len(pivot)}). Need ≥ 5."}

    # Friedman test
    samples = [pivot[algo].values for algo in algorithms]
    try:
        stat, p = stats.friedmanchisquare(*samples)
    except Exception as e:
        return {"error": str(e)}

    result = {
        "N_problems": len(pivot),
        "N_algorithms": len(algorithms),
        "chi2_statistic": stat,
        "p_value": p,
        "significant": p < alpha,
        "algorithms": algorithms,
    }

    # ── Nemenyi post-hoc test ──
    if p < alpha and len(algorithms) >= 2:
        result["nemenyi"] = _nemenyi_test(samples, algorithms, alpha, pivot)
    else:
        result["nemenyi"] = None

    return result


def _nemenyi_test(
    samples: List[np.ndarray],
    algorithms: List[str],
    alpha: float,
    pivot: pd.DataFrame,
) -> Dict:
    """
    Nemenyi post-hoc test for Friedman.

    Critical Difference: CD = q_alpha * sqrt(k*(k+1) / (6*N))
    where q_alpha is the studentized range statistic, k = #algorithms, N = #problems.
    """
    k = len(algorithms)
    N = len(pivot)

    # Average rank per algorithm
    ranks = np.zeros((N, k))
    for i in range(N):
        # Rank within this problem (lower = better)
        row_values = np.array([s[i] for s in samples])
        ranks[i] = stats.rankdata(row_values)

    avg_ranks = np.mean(ranks, axis=0)

    # Critical difference
    # q_alpha for k groups, df=inf, alpha=0.05
    from scipy.stats import studentized_range
    q_alpha = studentized_range.ppf(1 - alpha, k, 1e6)  # ~q(0.95, k, inf)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * N))

    # Pairwise rank differences
    pairwise = []
    for i in range(k):
        for j in range(i + 1, k):
            diff = abs(avg_ranks[i] - avg_ranks[j])
            sig = diff > cd
            pairwise.append({
                "algo_1": algorithms[i],
                "algo_2": algorithms[j],
                "rank_diff": diff,
                "significant": sig,
            })

    return {
        "average_ranks": {algo: float(r) for algo, r in zip(algorithms, avg_ranks)},
        "critical_difference": float(cd),
        "q_alpha": float(q_alpha),
        "pairwise": pairwise,
    }


def format_friedman_table(friedman_result: Dict) -> str:
    """Format Friedman + Nemenyi results as Markdown."""
    lines = []

    if "error" in friedman_result:
        lines.append(f"## Friedman Test: ERROR — {friedman_result['error']}")
        return "\n".join(lines)

    lines.append("## Friedman Test (Omnibus)")
    lines.append("")
    lines.append(f"- **N** (problems): {friedman_result['N_problems']}")
    lines.append(f"- **k** (algorithms): {friedman_result['N_algorithms']}")
    lines.append(f"- **χ² statistic**: {friedman_result['chi2_statistic']:.4f}")
    lines.append(f"- **p-value**: {friedman_result['p_value']:.6f}")
    lines.append(f"- **Significant**: {'Yes *' if friedman_result['significant'] else 'No'}")

    if friedman_result["nemenyi"]:
        nem = friedman_result["nemenyi"]
        lines.append("")
        lines.append("### Nemenyi Post-Hoc Test")
        lines.append("")
        lines.append(f"**Critical Difference (CD)**: {nem['critical_difference']:.4f}")
        lines.append("")
        lines.append("**Average Ranks** (lower = better):")
        for algo in friedman_result["algorithms"]:
            lines.append(f"  - {algo}: {nem['average_ranks'][algo]:.4f}")
        lines.append("")
        lines.append("**Pairwise Comparisons:**")
        lines.append("| Pair | Rank Difference | > CD? | Significant? |")
        lines.append("|------|-----------------|-------|--------------|")
        for pw in nem["pairwise"]:
            lines.append(
                f"| {pw['algo_1']} vs {pw['algo_2']} | "
                f"{pw['rank_diff']:.4f} | {pw['rank_diff']:.4f} > {nem['critical_difference']:.4f} "
                f"= {pw['rank_diff'] > nem['critical_difference']} | "
                f"{'Yes *' if pw['significant'] else 'No'} |"
            )

    return "\n".join(lines)


# ==============================================================================
# 5. Boxplot
# ==============================================================================

def plot_boxplot(
    df: pd.DataFrame,
    metric: str = "best_rmse",
    metric_label: str = "Best RMSE",
    output_path: Optional[Path] = None,
    palette: Optional[Dict[str, str]] = None,
    figsize: Tuple[float, float] = (8, 6),
) -> plt.Figure:
    """Generate a publication-quality boxplot comparing algorithms."""
    if palette is None:
        palette = {"FCAA": "#E63946", "NSGA-II": "#457B9D", "MOPSO": "#2A9D8F"}

    algorithms = sorted(df["algorithm"].unique(),
                        key=lambda a: df[df["algorithm"] == a][metric].mean())

    fig, ax = plt.subplots(figsize=figsize)

    data_by_algo = [df[df["algorithm"] == a][metric].dropna().values for a in algorithms]
    colors = [palette.get(a, "#888888") for a in algorithms]

    bp = ax.boxplot(
        data_by_algo,
        labels=algorithms,
        patch_artist=True,
        widths=0.5,
        showfliers=True,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="#333333", markersize=8),
        medianprops=dict(color="#333333", linewidth=2.5),
        flierprops=dict(marker="o", markerfacecolor="gray", markersize=5, alpha=0.5),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Overlay individual data points with jitter
    for i, algo in enumerate(algorithms):
        vals = df[df["algorithm"] == algo][metric].dropna().values
        jitter = np.random.default_rng(42).normal(0, 0.04, size=len(vals))
        ax.scatter(
            np.full(len(vals), i + 1) + jitter, vals,
            color=colors[i], alpha=0.5, s=30, edgecolors="white", linewidth=0.5, zorder=5,
        )

    ax.set_ylabel(metric_label, fontsize=13)
    ax.set_title(f"Algorithm Comparison: {metric_label}\n(30 Independent Runs)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")

    # Add mean + std annotation
    for i, algo in enumerate(algorithms):
        vals = df[df["algorithm"] == algo][metric].dropna().values
        mean, std = np.mean(vals), np.std(vals, ddof=1)
        ax.annotate(
            f"μ={mean:.2f}\nσ={std:.2f}",
            xy=(i + 1, ax.get_ylim()[1]),
            fontsize=9, ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
        print(f"  [OK] Boxplot: {output_path}")

    return fig


# ==============================================================================
# 6. Critical Difference Diagram
# ==============================================================================

def plot_cd_diagram(
    friedman_result: Dict,
    output_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (10, 3),
) -> Optional[plt.Figure]:
    """
    Draw a Critical Difference diagram (Demsar 2006 style).

    Algorithms connected by a bar are NOT significantly different.
    """
    if friedman_result.get("nemenyi") is None:
        print("  [SKIP] CD diagram: no Nemenyi data (Friedman not significant or error)")
        return None

    nem = friedman_result["nemenyi"]
    cd = nem["critical_difference"]
    avg_ranks = nem["average_ranks"]

    # Sort algorithms by average rank (lower = better → left side)
    sorted_algos = sorted(avg_ranks.items(), key=lambda x: x[1])
    algo_names = [a for a, _ in sorted_algos]
    ranks = [r for _, r in sorted_algos]

    fig, ax = plt.subplots(figsize=figsize)

    # Draw axis line
    min_rank = min(ranks) - 0.5
    max_rank = max(ranks) + 0.5
    ax.set_xlim(min_rank, max_rank)
    ax.set_ylim(0, 2)

    # CD bar reference
    ax.axhline(y=1.3, xmin=0.05, xmax=0.95, color="#333333", linewidth=1)
    ax.annotate(
        f"CD = {cd:.3f}",
        xy=((min_rank + max_rank) / 2, 1.35),
        fontsize=12, ha="center", fontweight="bold",
    )

    # Plot algorithm markers on rank axis
    colors = {"FCAA": "#E63946", "NSGA-II": "#457B9D", "MOPSO": "#2A9D8F"}
    for i, (name, rank) in enumerate(zip(algo_names, ranks)):
        color = colors.get(name, "#888888")
        ax.plot(rank, 1.0, marker="o", markersize=14, color=color, markeredgecolor="white",
                markeredgewidth=1.5, zorder=10)
        # Label above or below
        offset = 0.15 if i % 2 == 0 else -0.25
        ax.annotate(
            f"{name}\n({rank:.2f})",
            xy=(rank, 1.0 + offset),
            fontsize=11, ha="center", va="center" if offset < 0 else "center",
            fontweight="bold",
        )

    # Draw CD bars connecting non-significantly-different groups
    pairwise = nem["pairwise"]
    y_pos = 1.0
    for pw in pairwise:
        if not pw["significant"]:
            r1 = avg_ranks[pw["algo_1"]]
            r2 = avg_ranks[pw["algo_2"]]
            ax.plot([r1, r2], [y_pos, y_pos], color="#666666", linewidth=3, alpha=0.6)

    ax.set_yticks([])
    ax.set_xlabel("Average Rank (lower = better)", fontsize=13)
    ax.set_title(f"Critical Difference Diagram (Nemenyi Post-Hoc, α=0.05)", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
        print(f"  [OK] CD Diagram: {output_path}")

    return fig


# ==============================================================================
# 7. Convergence Analysis
# ==============================================================================

def plot_convergence_bands(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (10, 6),
) -> plt.Figure:
    """
    Plot hypervolume convergence curves with 95% CI bands across 30 runs.
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = {"FCAA": "#E63946", "NSGA-II": "#457B9D", "MOPSO": "#2A9D8F"}

    for algo_name in sorted(df["algorithm"].unique()):
        algo_df = df[df["algorithm"] == algo_name]
        hv_histories = algo_df["hypervolume_history"].dropna().values

        if len(hv_histories) == 0:
            continue

        # Pad to uniform length
        max_len = max(len(h) for h in hv_histories)
        padded = []
        for h in hv_histories:
            if len(h) < max_len:
                padded.append(list(h) + [h[-1]] * (max_len - len(h)))
            else:
                padded.append(list(h))
        hv_matrix = np.array(padded)

        generations = np.arange(1, max_len + 1)
        hv_mean = np.mean(hv_matrix, axis=0)
        hv_std = np.std(hv_matrix, axis=0)
        n = hv_matrix.shape[0]
        ci = 1.96 * hv_std / np.sqrt(n)

        ax.plot(generations, hv_mean, color=colors.get(algo_name, "#888888"),
                linewidth=2.5, label=algo_name)
        ax.fill_between(generations, hv_mean - ci, hv_mean + ci,
                        color=colors.get(algo_name, "#888888"), alpha=0.15)

    ax.set_xlabel("Generation", fontsize=13)
    ax.set_ylabel("Hypervolume", fontsize=13)
    ax.set_title("Convergence Curves (95% CI, 30 Independent Runs)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
        print(f"  [OK] Convergence plot: {output_path}")

    return fig


# ==============================================================================
# 8. Main Report Generator
# ==============================================================================

def generate_report(
    df: pd.DataFrame,
    metric: str = "best_rmse",
    metric_label: str = "Best RMSE",
    output_dir: Path = None,
    alpha: float = 0.05,
) -> str:
    """Run all analyses and produce a Markdown report."""
    if output_dir is None:
        output_dir = Path("results/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    sections = []

    # Header
    sections.append("# MO-FCAA Statistical Analysis Report")
    sections.append(f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    sections.append(f"**Metric**: {metric_label}")
    sections.append(f"**Significance level**: α = {alpha}")
    sections.append(f"**Total runs analyzed**: {len(df)}")
    sections.append("")

    # 1. Descriptive Statistics
    desc_stats = compute_descriptive_stats(df, metric)
    sections.append(format_descriptive_table(desc_stats, metric_label))
    sections.append("")

    # 2. Wilcoxon
    wilcoxon_df = run_wilcoxon_tests(df, metric, alpha=alpha)
    sections.append(format_wilcoxon_table(wilcoxon_df, alpha))
    sections.append("")

    # 3. Friedman + Nemenyi
    friedman_result = run_friedman_test(df, metric, alpha)
    sections.append(format_friedman_table(friedman_result))
    sections.append("")

    # 4. Generate figures
    # Boxplot
    fig_box = plot_boxplot(
        df, metric, metric_label,
        output_path=output_dir / "boxplot_comparison.png",
    )
    plt.close(fig_box)
    sections.append(f"![Boxplot](boxplot_comparison.png)")
    sections.append("")

    # CD diagram
    fig_cd = plot_cd_diagram(
        friedman_result,
        output_path=output_dir / "cd_diagram.png",
    )
    if fig_cd is not None:
        plt.close(fig_cd)
        sections.append(f"![CD Diagram](cd_diagram.png)")
        sections.append("")

    # Convergence
    fig_conv = plot_convergence_bands(
        df,
        output_path=output_dir / "convergence_bands.png",
    )
    plt.close(fig_conv)
    sections.append(f"![Convergence Bands](convergence_bands.png)")
    sections.append("")

    # ── Export tables as CSV for LaTeX ──
    desc_stats.to_csv(output_dir / "descriptive_stats.csv", index=False)
    wilcoxon_df.to_csv(output_dir / "wilcoxon_results.csv", index=False)
    print(f"  [OK] Descriptive stats → {output_dir / 'descriptive_stats.csv'}")
    print(f"  [OK] Wilcoxon results → {output_dir / 'wilcoxon_results.csv'}")

    report = "\n".join(sections)

    report_path = output_dir / "analysis_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n{'='*60}")
    print(f"Report saved to: {report_path}")
    print(f"{'='*60}")

    return report


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="MO-FCAA Statistical Analysis (30-Run Results)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-dir", type=str, default="results/30runs",
                        help="Directory containing 30-run experiment output")
    parser.add_argument("--output-dir", type=str, default="results/analysis",
                        help="Output directory for report and figures")
    parser.add_argument("--metric", type=str, default="best_rmse",
                        help="Primary metric to analyze (default: best_rmse)")
    parser.add_argument("--metric-label", type=str, default="Best RMSE",
                        help="Display label for the metric")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level (default: 0.05)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Filter to a specific dataset")
    parser.add_argument("--model", type=str, default=None,
                        help="Filter to a specific ML model")
    return parser.parse_args()


def main():
    args = parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"[ERROR] Results directory not found: {results_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MO-FCAA Statistical Analysis")
    print("=" * 60)
    print(f"  Results dir: {results_dir}")
    print(f"  Output dir:  {output_dir}")
    print(f"  Metric:      {args.metric}")
    print(f"  Alpha:       {args.alpha}")

    # Load data
    df = load_results(results_dir, metric=args.metric)

    # Apply filters
    if args.dataset:
        df = df[df["dataset"] == args.dataset]
        print(f"  Filter dataset: {args.dataset} → {len(df)} runs")
    if args.model:
        df = df[df["model"] == args.model]
        print(f"  Filter model:   {args.model} → {len(df)} runs")

    print(f"  Total runs:  {len(df)}")
    print(f"  Algorithms:  {sorted(df['algorithm'].unique())}")
    print(f"  Datasets:    {sorted(df['dataset'].unique())}")
    print(f"  Models:      {sorted(df['model'].unique())}")
    print()

    if len(df) == 0:
        print("[ERROR] No data after filtering.")
        sys.exit(1)

    # Generate report
    generate_report(
        df,
        metric=args.metric,
        metric_label=args.metric_label,
        output_dir=output_dir,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()
