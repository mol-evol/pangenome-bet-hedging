#!/usr/bin/env python3
"""
================================================================================
SUPPLEMENTARY FIGURES S6–S10
================================================================================

Generates the supplementary panels that complement the main-text empirical
figures (Figures 4 and 5 in v40 numbering).

Each main-text figure cherry-picks key panels from the individual analysis
scripts.  The remaining panels appear in the SI:

  S6  Decoupling (2 panels):
      A — Jaccard distance distributions (same vs different source)
      B — Heatmap of mean Jaccard distances between source categories

  S7  Gene Classification (2 panels):
      A — Frequency profiles across isolation sources
      B — Cross-phylogroup validation (B2 → D)

  S8  Niche Insurance (2 panels):
      A — Home vs away frequency scatter
      B — Retention by home niche (Kruskal-Wallis symmetry)

  S9  Model Discrimination (2 panels):
      A — Joint scatter (ε² vs R²) with migration/bet-hedging clouds
      B — Blood-home vs Feces-home median retention

  S10 Fitness Landscape (1 panel):
      A — Expected fitness vs switching rate with carrying cost

Data: Horesh et al. (2021) E. coli, phylogroup B2.

USAGE
-----
  python supplementary_figures_s6_s10.py [--output-dir DIR] [--format FMT]
================================================================================
"""

import os
import sys
import argparse
import numpy as np

# Path setup
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.join(_SCRIPT_DIR, '..')
sys.path.insert(0, _CODE_DIR)
sys.path.insert(0, _SCRIPT_DIR)

from shared.plotting import (setup_plotting, save_figure, add_panel_label,
                              print_header, COLORS)
import matplotlib.pyplot as plt

# Reuse panel functions from the individual analysis scripts
from decoupling_analysis import (load_horesh_data,
                                  panel_a_jaccard_distributions,
                                  panel_c_source_heatmap)
from gene_classification import (load_classification_data,
                                  panel_c_frequency_profiles,
                                  panel_d_cross_validation)
from niche_insurance_analysis import (compute_niche_away_frequencies,
                                       panel_b_home_vs_away,
                                       panel_c_retention_by_home)
from niche_simulation import (load_real_data, run_simulations,
                               generate_migration_freqs,
                               generate_bethedging_freqs)
from fitness_landscape import (compute_strategy_portfolios,
                                compute_fitness_matrix,
                                expected_fitness_vs_switching,
                                STRATEGY_COLORS, STRATEGY_STYLES)


# ==============================================================================
# Figure S6: Decoupling Analysis — supplementary panels
# ==============================================================================

def make_figure_s6(output_dir, fmt):
    """Panels not shown in main Figure 4 (old 5): Jaccard + heatmap."""
    print('\n--- Figure S6: Decoupling (supplementary panels) ---')
    data = load_horesh_data()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    panel_a_jaccard_distributions(axes[0], data)
    add_panel_label(axes[0], 'A')
    panel_c_source_heatmap(axes[1], data)
    add_panel_label(axes[1], 'B')

    fig.tight_layout()
    save_figure(fig, 'supplementary_s6_decoupling', output_dir=output_dir,
                fmt=fmt)
    plt.close(fig)
    print('  Saved supplementary_s6_decoupling')


# ==============================================================================
# Figure S7: Gene Classification — supplementary panels
# ==============================================================================

def make_figure_s7(output_dir, fmt):
    """Panels not shown in main Figure 4 (old 5): freq profiles + cross-val."""
    print('\n--- Figure S7: Gene Classification (supplementary panels) ---')
    data = load_classification_data()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    panel_c_frequency_profiles(axes[0], data)
    add_panel_label(axes[0], 'A')
    panel_d_cross_validation(axes[1], data)
    add_panel_label(axes[1], 'B')

    fig.tight_layout()
    save_figure(fig, 'supplementary_s7_classification', output_dir=output_dir,
                fmt=fmt)
    plt.close(fig)
    print('  Saved supplementary_s7_classification')


# ==============================================================================
# Figure S8: Niche Insurance — supplementary panels
# ==============================================================================

def make_figure_s8(output_dir, fmt):
    """Panels not shown in main Figure 5 (old 6): scatter + symmetry box."""
    print('\n--- Figure S8: Niche Insurance (supplementary panels) ---')
    cls_data = load_classification_data()
    away_data = compute_niche_away_frequencies(cls_data)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    panel_b_home_vs_away(axes[0], away_data)
    add_panel_label(axes[0], 'A')
    panel_c_retention_by_home(axes[1], away_data)
    add_panel_label(axes[1], 'B')

    fig.tight_layout()
    save_figure(fig, 'supplementary_s8_insurance', output_dir=output_dir,
                fmt=fmt)
    plt.close(fig)
    print('  Saved supplementary_s8_insurance')


# ==============================================================================
# Figure S9: Model Discrimination — supplementary panels
# ==============================================================================

def make_figure_s9(output_dir, fmt, n_sims=500):
    """Panels not shown in main Figure 5 (old 6): joint scatter + niche sym."""
    print(f'\n--- Figure S9: Model Discrimination (supplementary panels, '
          f'{n_sims} sims) ---')
    real = load_real_data()

    # Fit Beta for bet-hedging model
    mu = real['retention'].mean()
    var = real['retention'].var() * 0.80
    common = mu * (1 - mu) / var - 1
    alpha = mu * common
    beta_param = (1 - mu) * common

    print('  Running migration simulations...')
    mig = run_simulations(
        generate_migration_freqs,
        {'home_freqs': real['home_freqs'],
         'home_niches': real['home_niches'],
         'v_values': real['v_values']},
        n_sims)

    print('  Running bet-hedging simulations...')
    bh = run_simulations(
        generate_bethedging_freqs,
        {'home_freqs': real['home_freqs'],
         'home_niches': real['home_niches'],
         'alpha': alpha,
         'beta_param': beta_param},
        n_sims)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel A: joint scatter (ε² vs R²)
    ax = axes[0]
    ax.scatter(mig['epsilon_sq'], mig['r_sq'], s=12, alpha=0.4,
               color=COLORS['red'], label='Migration')
    ax.scatter(bh['epsilon_sq'], bh['r_sq'], s=12, alpha=0.4,
               color=COLORS['blue'], label='Bet-hedging')
    ax.scatter(real['epsilon_sq'], real['r_sq'], s=250, marker='*',
               color='black', zorder=10, label='Real data')
    ax.set_xlabel('ε² (symmetry)')
    ax.set_ylabel('R² (slope)')
    ax.set_title('Joint comparison')
    ax.legend(fontsize=7)
    add_panel_label(ax, 'A')

    # Panel B: Blood-home vs Feces-home median retention
    ax = axes[1]
    ax.scatter(mig['blood_median'], mig['feces_median'], s=12, alpha=0.4,
               color=COLORS['red'], label='Migration')
    ax.scatter(bh['blood_median'], bh['feces_median'], s=12, alpha=0.4,
               color=COLORS['blue'], label='Bet-hedging')
    ax.scatter(real['niche_medians']['Blood'], real['niche_medians']['Feces'],
               s=250, marker='*', color='black', zorder=10, label='Real data')
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], '--', color=COLORS['grey'], alpha=0.5,
            label='Symmetric')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('Blood-home median retention')
    ax.set_ylabel('Feces-home median retention')
    ax.set_title('Retention symmetry: Blood vs Feces home')
    ax.legend(fontsize=7)
    add_panel_label(ax, 'B')

    fig.tight_layout()
    save_figure(fig, 'supplementary_s9_simulation', output_dir=output_dir,
                fmt=fmt)
    plt.close(fig)
    print('  Saved supplementary_s9_simulation')


# ==============================================================================
# Figure S10: Fitness Landscape — supplementary panel
# ==============================================================================

def make_figure_s10(output_dir, fmt):
    """Panel not shown in main Figure 5 (old 6): fitness with carrying cost."""
    print('\n--- Figure S10: Fitness Landscape (supplementary panel) ---')
    cls_data = load_classification_data()
    info = compute_strategy_portfolios(cls_data)

    sources = info['sources']
    portfolios = info['portfolios']
    source_freqs = info['source_freqs']
    source_sizes = info['source_sizes']

    cost = 0.10
    p_range = np.linspace(0.01, 0.99, 200)

    F = compute_fitness_matrix(portfolios, source_freqs, sources, cost=cost)
    expected_c = expected_fitness_vs_switching(F, sources, source_sizes,
                                               p_range)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    for strat_name in expected_c:
        ax.plot(p_range, expected_c[strat_name],
                color=STRATEGY_COLORS[strat_name],
                linestyle=STRATEGY_STYLES[strat_name], linewidth=2,
                label=strat_name.replace('\n', ' '))
    ax.set_xlabel('Environmental switching probability (p)')
    ax.set_ylabel('Expected log fitness')
    ax.set_title(f'Fitness landscape (carrying cost = {cost:.0%})')
    ax.legend(fontsize=8)
    add_panel_label(ax, 'A')

    fig.tight_layout()
    save_figure(fig, 'supplementary_s10_fitness', output_dir=output_dir,
                fmt=fmt)
    plt.close(fig)
    print('  Saved supplementary_s10_fitness')


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Supplementary Figures S6–S10')
    parser.add_argument('--output-dir', default='.', help='Output directory')
    parser.add_argument('--format', default='png')
    parser.add_argument('--n-sims', type=int, default=500,
                        help='Simulations for S9 (default 500)')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--figures', nargs='+',
                        choices=['S6', 'S7', 'S8', 'S9', 'S10', 'all'],
                        default=['all'],
                        help='Which supplementary figures to generate')
    args = parser.parse_args()

    setup_plotting()
    print_header('SUPPLEMENTARY FIGURES S6–S10',
                 {'Source': 'Horesh et al. (2021) E. coli',
                  'Sims': f'{args.n_sims} (for S9)'})

    which = set(args.figures)
    run_all = 'all' in which

    if run_all or 'S6' in which:
        make_figure_s6(args.output_dir, args.format)
    if run_all or 'S7' in which:
        make_figure_s7(args.output_dir, args.format)
    if run_all or 'S8' in which:
        make_figure_s8(args.output_dir, args.format)
    if run_all or 'S9' in which:
        make_figure_s9(args.output_dir, args.format, args.n_sims)
    if run_all or 'S10' in which:
        make_figure_s10(args.output_dir, args.format)

    print('\nAll requested supplementary figures complete.')


if __name__ == '__main__':
    main()
