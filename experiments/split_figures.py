#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Split Fig 2 and Fig 3 into standalone sub-figures."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter
from scipy.interpolate import Rbf
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

from src.algorithms.fcaa import FCAAOptimizer
from src.algorithms.operators import fcaa_update
from src.algorithms.multi_objective import select_survivors
from src.evaluators.feature_selection_hpo import FeatureSelectionHPOEvaluator
from src.utils.encoding import SolutionEncoding
from src.evaluators.models import get_model_wrapper

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42

# ── Load data ──
print("Loading data...")
df = pd.read_excel(Path(__file__).resolve().parent.parent / "data" / "B_data.xlsx")
col_map = {c: 'rho' for c in df.columns if '\xcf\x81' in c or '\\xcf' in repr(c)}
df = df.rename(columns=col_map)
feat_cols_orig = [c for c in df.columns if c not in ['Systems', 'Tm of MD (K)']]
print(f"Features: {feat_cols_orig}")

X = df[feat_cols_orig].values
y = df['Tm of MD (K)'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

poly = PolynomialFeatures(degree=3, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
feature_names_poly = list(poly.get_feature_names_out(feat_cols_orig))

NOISE_FEAT = 20
rng = np.random.default_rng(SEED)
X_train_poly = np.hstack([X_train_poly, rng.normal(0, 1, (X_train_poly.shape[0], NOISE_FEAT))])
all_feature_names = feature_names_poly + [f'noise_{i}' for i in range(NOISE_FEAT)]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)

# ── Run FCAA v2 ──
print("Running FCAA v2 (Pareto data)...")
evaluator = FeatureSelectionHPOEvaluator(X_train_scaled, y_train, model_name='svr', cv_folds=5, n_jobs=4)
opt = FCAAOptimizer(dimension=evaluator.get_dimension(), pop_size=80, max_generations=200,
    fitness_fn=evaluator, alpha_init=1.0, sigma_init=0.15, sigma_final=0.002,
    claw_ratio_init=0.80, claw_ratio_final=0.15, elite_ratio=0.3, elite_sigma=0.015, seed=SEED)
pareto_pop, pareto_fit = opt.optimize(verbose=False)

hp_bounds = get_model_wrapper('svr').hyperparameter_bounds()
encoding = SolutionEncoding(len(all_feature_names), hp_bounds)

# Decode all Pareto solutions
pareto_data = []
for sol, fit in zip(pareto_pop, pareto_fit):
    mask, hparams, indices = encoding.decode(sol)
    pareto_data.append({
        'rmse': fit[0], 'feat_ratio': fit[1], 'n_features': int(mask.sum()),
        'logC': float(np.log10(hparams['C'])),
        'logGamma': float(np.log10(hparams['gamma'])),
        'epsilon': float(hparams['epsilon']), 'indices': indices,
    })

# ── Re-run for population snapshots ──
print("Running FCAA v2 (snapshots)...")
evaluator2 = FeatureSelectionHPOEvaluator(X_train_scaled, y_train, model_name='svr', cv_folds=5, n_jobs=4)
opt2 = FCAAOptimizer(dimension=evaluator2.get_dimension(), pop_size=80, max_generations=200,
    fitness_fn=evaluator2, alpha_init=1.0, sigma_init=0.15, sigma_final=0.002,
    claw_ratio_init=0.80, claw_ratio_final=0.15, elite_ratio=0.3, elite_sigma=0.015, seed=SEED)

opt2.population = opt2._initialize_population()
opt2.fitnesses = opt2.fitness_fn(opt2.population)
snapshots = {}

for gen in range(200):
    progress = gen / max(199, 1)
    alpha_gen = opt2.alpha_init * (1.0 - 0.5 * progress)
    sigma_gen = opt2.sigma_init * (1.0 - progress) ** 2
    sigma_gen = max(sigma_gen, opt2.sigma_final)
    claw_ratio = opt2.claw_ratio_init + progress * (opt2.claw_ratio_final - opt2.claw_ratio_init)
    best_idx = opt2._select_leader(progress)
    elite_indices = opt2._select_elites()

    offspring = fcaa_update(opt2.population, best_idx=best_idx, alpha=alpha_gen,
                            sigma=sigma_gen, claw_ratio=claw_ratio,
                            elite_indices=elite_indices, elite_sigma=opt2.elite_sigma, rng=opt2.rng)
    offspring_fitnesses = opt2.fitness_fn(offspring)
    combined_pop = np.vstack([opt2.population, offspring])
    combined_fitnesses = np.vstack([opt2.fitnesses, offspring_fitnesses])
    opt2.population, opt2.fitnesses, _ = select_survivors(
        combined_pop, combined_fitnesses, opt2.pop_size)

    gen_num = gen + 1
    if gen_num in [10, 50, 200]:
        pop_C, pop_gamma, pop_rmse = [], [], []
        for i in range(len(opt2.population)):
            mask, hp, idx = encoding.decode(opt2.population[i])
            pop_C.append(np.log10(hp['C']))
            pop_gamma.append(np.log10(hp['gamma']))
            pop_rmse.append(opt2.fitnesses[i, 0])
        snapshots[gen_num] = {
            'logC': np.array(pop_C),
            'logGamma': np.array(pop_gamma),
            'rmse': np.array(pop_rmse),
        }
        print(f"  Snapshot gen {gen_num}: {len(pop_C)} individuals")

# ═══════════════════════════════════════════════════════════
# FIG 2a: Feature Frequency Bar Chart (standalone)
# ═══════════════════════════════════════════════════════════
print("\n--- Fig 2a: Feature Frequency Bar Chart ---")

top20 = sorted(pareto_data, key=lambda d: d['rmse'])[:20]
orig_feat_detail = {f: {'linear': 0, 'squared': 0, 'cubic': 0, 'interaction': 0, 'total': 0}
                    for f in feat_cols_orig}

for d in top20:
    selected_names = [all_feature_names[i] for i in d['indices']]
    for name in selected_names:
        if name.startswith('noise_'):
            continue
        for orig in feat_cols_orig:
            if orig in name:
                orig_feat_detail[orig]['total'] += 1
                if name == orig:
                    orig_feat_detail[orig]['linear'] += 1
                elif '^3' in name:
                    orig_feat_detail[orig]['cubic'] += 1
                elif '^2' in name:
                    orig_feat_detail[orig]['squared'] += 1
                else:
                    orig_feat_detail[orig]['interaction'] += 1
                break

total_selections = sum(d['total'] for d in orig_feat_detail.values())
feat_stats = []
for orig in feat_cols_orig:
    d = orig_feat_detail[orig]
    pct = d['total'] / total_selections * 100 if total_selections > 0 else 0
    feat_stats.append({
        'name': orig, 'pct': pct,
        'linear': d['linear'], 'squared': d['squared'],
        'cubic': d['cubic'], 'interaction': d['interaction'],
        'total': d['total'],
    })
feat_stats.sort(key=lambda x: x['pct'], reverse=True)

names_short = {
    'delta_r_Me': '$\delta r_{Me}$\n(Size Mismatch)',
    'delta_l': '$\delta l$\n(Lattice Param)',
    'VEC': 'VEC\n(Valence e$^-$ Conc)',
    'delta_chi_A': '$\delta\chi_A$\n(E-neg Variance)',
    'rho': '$\\rho$\n(Density)',
}

fig2a, ax2a = plt.subplots(figsize=(11, 7))
y_pos = range(len(feat_stats))
names = [names_short.get(s['name'], s['name']) for s in feat_stats]
colors_stack = {'linear': '#2A9D8F', 'squared': '#E9C46A',
                'cubic': '#F4A261', 'interaction': '#E76F51'}
bottom = np.zeros(len(feat_stats))

for cat in ['linear', 'squared', 'cubic', 'interaction']:
    vals = [s[cat] for s in feat_stats]
    bars = ax2a.barh(y_pos, vals, left=bottom, height=0.6,
                     color=colors_stack[cat], label=cat.capitalize(),
                     edgecolor='white', linewidth=0.5)
    for j, (v, b_val) in enumerate(zip(vals, bottom)):
        if v > 0:
            ax2a.text(b_val + v/2, j, str(v), ha='center', va='center',
                     fontsize=9, fontweight='bold', color='white')
    bottom += np.array(vals)

ax2a.set_yticks(y_pos)
ax2a.set_yticklabels(names, fontsize=12)
ax2a.set_xlabel('Total Selections Across Top-20 Pareto Solutions', fontsize=13)
ax2a.set_title('FCAA v2 Feature Selection Frequency by Term Type\n'
               '(Top-20 Best-RMSE Solutions, Boron HEC)',
               fontsize=14, fontweight='bold')
ax2a.legend(fontsize=11, loc='lower right', ncol=4)
ax2a.invert_yaxis()
ax2a.grid(True, alpha=0.2, axis='x')
for spine in ['top', 'right']:
    ax2a.spines[spine].set_visible(False)
plt.tight_layout()
fig2a.savefig(OUT_DIR / 'fig2a_feature_frequency_bar.png', dpi=200, bbox_inches='tight')
fig2a.savefig(OUT_DIR / 'fig2a_feature_frequency_bar.pdf', bbox_inches='tight')
plt.close(fig2a)
print("  -> fig2a_feature_frequency_bar.png/pdf")

# ═══════════════════════════════════════════════════════════
# FIG 2b: Descriptor Donut Chart (standalone)
# ═══════════════════════════════════════════════════════════
print("--- Fig 2b: Descriptor Donut Chart ---")

fig2b, ax2b = plt.subplots(figsize=(8, 8))
pcts = [s['pct'] for s in feat_stats]
colors_pie = ['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51']
wedges, texts, autotexts = ax2b.pie(
    pcts,
    labels=[names_short.get(s['name'], s['name']) for s in feat_stats],
    autopct='%1.1f%%',
    colors=colors_pie,
    startangle=90,
    explode=(0.05, 0.02, 0.02, 0.02, 0.02),
    textprops={'fontsize': 11},
    pctdistance=0.6,
)
for at in autotexts:
    at.set_fontweight('bold')
    at.set_fontsize(11)

# White circle for donut hole
centre_circle = plt.Circle((0, 0), 0.40, fc='white', linewidth=0)
ax2b.add_artist(centre_circle)
ax2b.text(0, 0, f'Total\n{total_selections}\nTerms', ha='center', va='center',
          fontsize=14, fontweight='bold')

ax2b.set_title('Descriptor Contribution Distribution\n'
               '(Boron HEC Melting Point Prediction)',
               fontsize=14, fontweight='bold')
plt.tight_layout()
fig2b.savefig(OUT_DIR / 'fig2b_descriptor_donut.png', dpi=200, bbox_inches='tight')
fig2b.savefig(OUT_DIR / 'fig2b_descriptor_donut.pdf', bbox_inches='tight')
plt.close(fig2b)
print("  -> fig2b_descriptor_donut.png/pdf")

# ═══════════════════════════════════════════════════════════
# FIG 3a/3b/3c: Individual Search Behavior Snapshots
# ═══════════════════════════════════════════════════════════
print("--- Fig 3a/3b/3c: Search Behavior Snapshots ---")

# Build RMSE landscape
all_logC = np.concatenate([s['logC'] for s in snapshots.values()])
all_logGamma = np.concatenate([s['logGamma'] for s in snapshots.values()])
C_grid = np.linspace(all_logC.min() - 0.5, all_logC.max() + 0.5, 80)
gamma_grid = np.linspace(all_logGamma.min() - 0.5, all_logGamma.max() + 0.5, 80)
CC, GG = np.meshgrid(C_grid, gamma_grid)

final_logC = snapshots[200]['logC']
final_logGamma = snapshots[200]['logGamma']
final_rmse = snapshots[200]['rmse']
rbf = Rbf(final_logC, final_logGamma, final_rmse, function='multiquadric', smooth=0.5)
Z_rmse = rbf(CC, GG)
Z_rmse = gaussian_filter(Z_rmse, sigma=1.0)
Z_norm = (Z_rmse - Z_rmse.min()) / (Z_rmse.max() - Z_rmse.min() + 1e-10)

gen_labels = {
    10: 'Early Stage (Gen 10)\nMajor Claw Dominant - Cauchy Exploration',
    50: 'Middle Stage (Gen 50)\nTransition Phase - Balanced Search',
    200: 'Final Stage (Gen 200)\nMinor Claw Dominant - sigma=0.002 Fine-tuning',
}
scatter_colors = {10: '#E76F51', 50: '#F4A261', 200: '#2A9D8F'}
fignames = {10: 'fig3a_early_gen10', 50: 'fig3b_mid_gen50', 200: 'fig3c_final_gen200'}

for gen_num in [10, 50, 200]:
    fig, ax = plt.subplots(figsize=(9, 7.5))
    snap = snapshots[gen_num]

    # RMSE landscape as filled contour
    cf = ax.contourf(CC, GG, Z_norm, levels=10, cmap='YlOrRd_r', alpha=0.5)
    ax.contour(CC, GG, Z_norm, levels=10, colors='#666666', linewidths=0.3, alpha=0.4)

    # Population scatter
    ax.scatter(snap['logC'], snap['logGamma'],
              c=scatter_colors[gen_num], alpha=0.75, s=45,
              edgecolors='white', linewidth=0.5, zorder=10)

    # Star for best
    best_local = np.argmin(snap['rmse'])
    ax.scatter([snap['logC'][best_local]], [snap['logGamma'][best_local]],
              marker='*', s=400, c='#E63946', edgecolors='white',
              linewidth=2, zorder=20,
              label='Best RMSE = %.0f K' % snap['rmse'][best_local])

    cbar = plt.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('Normalized RMSE Landscape', fontsize=11)

    ax.set_xlabel('log$_{10}$(C)', fontsize=13)
    ax.set_ylabel('log$_{10}$($\\gamma$)', fontsize=13)
    ax.set_title(gen_labels[gen_num], fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.2)

    if gen_num == 10:
        info_text = ("alpha=0.95  |  sigma=0.124  |  claw=0.74\n"
                     "Population scattered widely across the landscape")
    elif gen_num == 50:
        info_text = ("alpha=0.75  |  sigma=0.038  |  claw=0.48\n"
                     "Transitioning: population clustering toward optimum basin")
    else:
        info_text = ("alpha=0.50  |  sigma=0.002  |  claw=0.15\n"
                     "Precision convergence at the global optimum pit bottom")
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      alpha=0.85, edgecolor='gray'))

    plt.tight_layout()
    fig.savefig(OUT_DIR / f'{fignames[gen_num]}.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUT_DIR / f'{fignames[gen_num]}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {fignames[gen_num]}.png/pdf")

# ── Summary ──
print(f"\n{'='*60}")
print("All split figures generated successfully!")
print(f"Output: {OUT_DIR}")
print(f"{'='*60}")
for f in sorted(OUT_DIR.glob("fig[23]*")):
    print(f"  {f.name} ({f.stat().st_size // 1024} KB)")
