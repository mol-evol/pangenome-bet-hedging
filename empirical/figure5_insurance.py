#!/usr/bin/env python3
"""
================================================================================
FIGURE 5: Niche Insurance, Model Discrimination, and Fitness Landscape
================================================================================

Main-text figure for the paper. 6 panels (2×3):

  A — Retention histogram (niche genes carried at ~63% in away niches)
  B — V vs retention scatter (flat slope, R²=0.076, rejects migration)
  C — Simulation R² distributions (real at 64th %ile of bet-hedging)
  D — Simulation ε² distributions (symmetry test)
  E — Fitness vs switching rate (three strategies)
  F — Observed strategy advantage over both extremes

Data: Horesh et al. (2021) E. coli, phylogroup B2.

USAGE
-----
  cd decoupling_analysis
  python figure5_insurance.py [--n-sims 500] [--output-dir DIR] [--format FMT]
================================================================================
"""

import os
import sys
import argparse
import numpy as np
from scipy.stats import linregress

# Path setup
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.join(_SCRIPT_DIR, '..')
_PROJECT_ROOT = os.path.join(_CODE_DIR, '..')
sys.path.insert(0, _CODE_DIR)       # for shared.*
sys.path.insert(0, _SCRIPT_DIR)     # for sibling empirical scripts

from shared.plotting import (setup_plotting, save_figure, add_panel_label,
                              print_header, COLORS)
import matplotlib.pyplot as plt

# Data functions from existing scripts (now in same directory)
from gene_classification import load_classification_data
from niche_insurance_analysis import compute_niche_away_frequencies
from niche_simulation import (load_real_data, run_simulations,
                               generate_migration_freqs,
                               generate_bethedging_freqs)
from fitness_landscape import (compute_strategy_portfolios,
                                compute_fitness_matrix,
                                expected_fitness_vs_switching,
                                find_crossover, STRATEGY_COLORS,
                                STRATEGY_STYLES)


# ==============================================================================
# Panels
# ==============================================================================

def draw_panel_a(ax, away_data):
    """Retention histogram — streamlined for main text."""
    retention = away_data['retention']

    ax.hist(retention, bins=50, color=COLORS['blue'], alpha=0.7,
            edgecolor='white', linewidth=0.5)

    ax.axvline(1.0, color=COLORS['red'], linestyle='-', linewidth=1.5,
               label='No depletion')
    median_ret = np.median(retention)
    ax.axvline(median_ret, color=COLORS['orange'], linestyle='--',
               linewidth=1.5, label=f'Median = {median_ret:.2f}')

    pct_above_50 = (retention >= 0.50).mean() * 100

    ax.set_xlabel('Away frequency / home frequency')
    ax.set_ylabel('Number of niche genes')
    ax.set_title('Retention of niche genes in away niches')
    ax.legend(fontsize=7, loc='upper left')

    txt = (f'n = {len(retention):,} genes\n'
           f'Median = {median_ret:.2f}\n'
           f'≥50% retained: {pct_above_50:.0f}%')
    ax.text(0.97, 0.97, txt, transform=ax.transAxes, fontsize=7,
            va='top', ha='right', bbox=dict(boxstyle='round,pad=0.3',
            facecolor='white', alpha=0.8))


def draw_panel_b(ax, away_data):
    """V vs retention scatter — the decisive test."""
    v = away_data['cramers_v']
    retention = away_data['retention']
    overall = away_data['overall_freq']

    sc = ax.scatter(v, retention, s=3, alpha=0.3, c=overall,
                    cmap='viridis', edgecolors='none', rasterized=True)

    # Decile medians
    n_bins = 10
    sorted_idx = np.argsort(v)
    bin_size = len(v) // n_bins
    bx, by = [], []
    for b in range(n_bins):
        start = b * bin_size
        end = start + bin_size if b < n_bins - 1 else len(v)
        idx = sorted_idx[start:end]
        bx.append(np.median(v[idx]))
        by.append(np.median(retention[idx]))
    ax.plot(bx, by, '-o', color=COLORS['red'], markersize=4,
            linewidth=1.5, label='Decile medians', zorder=5)

    # OLS regression
    valid = v > 0
    slope, intercept, r_val, p_val, _ = linregress(v[valid], retention[valid])
    r_sq = r_val ** 2
    x_line = np.linspace(v.min(), v.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, '--',
            color=COLORS['orange'], linewidth=1.5)

    ax.set_xlabel("Cramér's V (niche effect size)")
    ax.set_ylabel('Retention (away / home)')
    ax.set_title('Effect size vs retention')

    txt = (f'R² = {r_sq:.3f}\n'
           f'slope = {slope:.3f}\n'
           f'Migration predicts R² ≈ 0.41')
    ax.text(0.97, 0.97, txt, transform=ax.transAxes, fontsize=7,
            va='top', ha='right', bbox=dict(boxstyle='round,pad=0.3',
            facecolor='white', alpha=0.8))

    cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label('Overall B2 frequency', fontsize=7)
    cbar.ax.tick_params(labelsize=6)


def draw_panel_c(ax, real, mig, bh):
    """R² distributions — the decisive discriminator."""
    ax.hist(mig['r_sq'], bins=30, alpha=0.6, color=COLORS['red'],
            label='Migration model', density=True)
    ax.hist(bh['r_sq'], bins=30, alpha=0.6, color=COLORS['blue'],
            label='Bet-hedging model', density=True)
    ax.axvline(real['r_sq'], color='black', lw=2, ls='--',
               label=f'Real data (R² = {real["r_sq"]:.3f})')

    # Percentile ranks
    m_pct = (mig['r_sq'] <= real['r_sq']).mean() * 100
    b_pct = (bh['r_sq'] <= real['r_sq']).mean() * 100

    ax.set_xlabel('R² (V–retention slope)')
    ax.set_ylabel('Density')
    ax.set_title('Model discrimination: slope test')
    ax.legend(fontsize=7)

    txt = (f'Real at {m_pct:.0f}th %ile of migration\n'
           f'Real at {b_pct:.0f}th %ile of bet-hedging')
    ax.text(0.97, 0.97, txt, transform=ax.transAxes, fontsize=7,
            va='top', ha='right', bbox=dict(boxstyle='round,pad=0.3',
            facecolor='white', alpha=0.8))


def draw_panel_d(ax, real, mig, bh):
    """ε² distributions — symmetry test."""
    ax.hist(mig['epsilon_sq'], bins=30, alpha=0.6, color=COLORS['red'],
            label='Migration model', density=True)
    ax.hist(bh['epsilon_sq'], bins=30, alpha=0.6, color=COLORS['blue'],
            label='Bet-hedging model', density=True)
    ax.axvline(real['epsilon_sq'], color='black', lw=2, ls='--',
               label=f'Real data (ε² = {real["epsilon_sq"]:.3f})')

    m_pct = (mig['epsilon_sq'] <= real['epsilon_sq']).mean() * 100
    b_pct = (bh['epsilon_sq'] <= real['epsilon_sq']).mean() * 100

    ax.set_xlabel('ε² (symmetry test)')
    ax.set_ylabel('Density')
    ax.set_title('Model discrimination: symmetry test')
    ax.legend(fontsize=7)

    txt = (f'Real at {m_pct:.0f}th %ile of migration\n'
           f'Real at {b_pct:.0f}th %ile of bet-hedging')
    ax.text(0.97, 0.97, txt, transform=ax.transAxes, fontsize=7,
            va='top', ha='right', bbox=dict(boxstyle='round,pad=0.3',
            facecolor='white', alpha=0.8))


def draw_panel_e(ax, p_range, expected_c0):
    """Fitness vs switching rate — three strategies."""
    for strat_name, ew in expected_c0.items():
        ax.plot(p_range, ew,
                color=STRATEGY_COLORS[strat_name],
                linestyle=STRATEGY_STYLES[strat_name],
                linewidth=2.0, label=strat_name.replace('\n', ' '))

    niche_key = 'Pure niche\n(all-in)'
    bet_key = 'Pure bet-hedging\n(insurance)'
    xover = find_crossover(p_range, expected_c0[niche_key],
                           expected_c0[bet_key])
    if xover is not None:
        ax.axvline(xover, color='grey', linestyle=':', linewidth=1, alpha=0.7)
        ax.text(xover + 0.02, ax.get_ylim()[0] +
                0.6 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                f'p = {xover:.2f}', fontsize=8, color='grey')

    ax.set_xlabel('Switching probability (p)')
    ax.set_ylabel('Expected fitness (Σ q·f)')
    ax.set_title('Fitness vs environmental switching')
    ax.legend(fontsize=6, loc='best')


def draw_panel_f(ax, p_range, expected_c0):
    """Observed strategy advantage over both extremes."""
    niche_key = 'Pure niche\n(all-in)'
    bet_key = 'Pure bet-hedging\n(insurance)'
    obs_key = 'Observed\nE. coli'

    obs = expected_c0[obs_key]
    niche = expected_c0[niche_key]
    bet = expected_c0[bet_key]

    adv_vs_niche = (obs - niche) / obs * 100
    adv_vs_bet = (obs - bet) / obs * 100

    ax.fill_between(p_range, 0, adv_vs_niche, alpha=0.3,
                    color=COLORS['red'], label='vs pure niche')
    ax.plot(p_range, adv_vs_niche, color=COLORS['red'], linewidth=2)

    ax.fill_between(p_range, 0, adv_vs_bet, alpha=0.3,
                    color=COLORS['blue'], label='vs pure bet-hedging')
    ax.plot(p_range, adv_vs_bet, color=COLORS['blue'], linewidth=2)

    ax.axhline(0, color='black', linewidth=0.5)

    xover = find_crossover(p_range, bet, obs)
    if xover is not None:
        ax.axvline(xover, color='grey', linestyle=':', linewidth=1, alpha=0.7)
        ax.text(xover + 0.02, adv_vs_niche.max() * 0.4,
                f'p = {xover:.2f}', fontsize=8, color='grey')

    ax.set_xlabel('Switching probability (p)')
    ax.set_ylabel('Advantage of observed strategy (%)')
    ax.set_title('Observed E. coli advantage')
    ax.legend(fontsize=7, loc='right')


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Figure 6: Niche Insurance')
    parser.add_argument('--n-sims', type=int, default=500)
    parser.add_argument('--output-dir', default='.', help='Output directory')
    parser.add_argument('--format', default='png')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    setup_plotting()
    print_header('FIGURE 5: Niche Insurance + Models + Fitness')

    # ---- Row 1: Niche insurance panels ----
    print('\n[1/5] Loading niche insurance data...')
    data = load_classification_data()
    away_data = compute_niche_away_frequencies(data)
    print(f'  {away_data["n_niche"]} niche genes of {away_data["n_total"]} total')

    # ---- Row 1-2: Simulation data ----
    print('\n[2/5] Loading real data + running simulations...')
    real = load_real_data()
    n_genes = len(real['home_freqs'])

    # Fit Beta distribution
    mu = real['retention'].mean()
    var = real['retention'].var() * 0.80
    common = mu * (1 - mu) / var - 1
    alpha = mu * common
    beta_param = (1 - mu) * common
    print(f'  {n_genes} niche genes, Beta(α={alpha:.2f}, β={beta_param:.2f})')

    print(f'  Running {args.n_sims} migration sims...')
    mig = run_simulations(
        generate_migration_freqs,
        {'home_freqs': real['home_freqs'],
         'home_niches': real['home_niches'],
         'v_values': real['v_values']},
        args.n_sims)

    print(f'  Running {args.n_sims} bet-hedging sims...')
    bh = run_simulations(
        generate_bethedging_freqs,
        {'home_freqs': real['home_freqs'],
         'home_niches': real['home_niches'],
         'alpha': alpha,
         'beta_param': beta_param},
        args.n_sims)

    # ---- Row 2: Fitness landscape ----
    print('\n[3/5] Computing fitness landscape...')
    info = compute_strategy_portfolios(data, phylogroup='B2')
    sources = info['sources']
    portfolios = info['portfolios']
    source_freqs = info['source_freqs']
    source_sizes = info['source_sizes']
    p_range = np.linspace(0, 1, 500)

    fitness_c0 = compute_fitness_matrix(portfolios, source_freqs, sources, cost=0.0)
    expected_c0 = expected_fitness_vs_switching(fitness_c0, sources,
                                                source_sizes, p_range)

    # ---- Build figure ----
    print('\n[4/5] Generating 6-panel figure...')
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    draw_panel_a(axes[0, 0], away_data)
    add_panel_label(axes[0, 0], 'A')

    draw_panel_b(axes[0, 1], away_data)
    add_panel_label(axes[0, 1], 'B')

    draw_panel_c(axes[0, 2], real, mig, bh)
    add_panel_label(axes[0, 2], 'C')

    draw_panel_d(axes[1, 0], real, mig, bh)
    add_panel_label(axes[1, 0], 'D')

    draw_panel_e(axes[1, 1], p_range, expected_c0)
    add_panel_label(axes[1, 1], 'E')

    draw_panel_f(axes[1, 2], p_range, expected_c0)
    add_panel_label(axes[1, 2], 'F')

    fig.tight_layout(w_pad=3.0, h_pad=2.5)
    saved = save_figure(fig, 'figure5_insurance', output_dir=args.output_dir,
                        fmt=args.format)
    for p in saved:
        print(f'    → {p}')

    if args.show:
        plt.show()
    else:
        plt.close(fig)

    print('\n[5/5] Done.')


if __name__ == '__main__':
    main()
