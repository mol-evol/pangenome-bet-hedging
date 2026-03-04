#!/usr/bin/env python3
"""
================================================================================
FITNESS LANDSCAPE: PURE NICHE vs PURE BET-HEDGING vs OBSERVED E. COLI
================================================================================

PURPOSE
-------
Compare expected fitness under three pangenome management strategies as a
function of environmental switching rate:

  1. PURE NICHE (all-in): carry only genes optimised for current niche;
     purge genes whose home is a different body site.
  2. PURE BET-HEDGING: carry every gene at its cross-niche mean frequency,
     regardless of current environment.
  3. OBSERVED E. COLI: actual observed per-niche frequencies, which show
     niche enrichment plus ~63% retention of niche genes in away niches.

MODEL
-----
For a population using strategy S in niche X:

  fitness(S, X) = Σ_g  q_S(g) · f_X(g)  −  c · Σ_g q_S(g)

where:
  q_S(g) = frequency of gene g under strategy S
  f_X(g) = gene g's equilibrium frequency in niche X (proxy for selective value)
  c       = per-gene carrying cost

Long-term expected fitness for a lineage in home niche H:

  E[W] = (1 − p) · fitness(S, H) + p · mean_{Y≠H}[ fitness(S, Y) ]

where p = probability of being in a non-home niche.

FIGURE
------
3-panel (1 row × 3 columns):
  A: Fitness vs switching rate (three strategies, no carrying cost)
  B: Fitness vs switching rate (with moderate carrying cost c=0.05)
  C: Critical switching rate (crossover) vs carrying cost

DATA
----
Horesh et al. (2021), phylogroup B2. Uses per-gene per-source frequencies
for all 4,862 accessory genes (not just niche genes).

USAGE
-----
  cd decoupling_analysis/fitness_comparison
  python fitness_landscape.py

================================================================================
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Path setup
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.join(_SCRIPT_DIR, '..')
_PROJECT_ROOT = os.path.join(_CODE_DIR, '..')
sys.path.insert(0, _CODE_DIR)       # for shared.*
sys.path.insert(0, _SCRIPT_DIR)     # for sibling empirical scripts

from shared.plotting import setup_plotting, save_figure, add_panel_label, \
    print_header, COLORS
from gene_classification import (load_classification_data, per_gene_chi_squared,
                                  compute_fdr, classify_genes)


# ==============================================================================
# Core computation
# ==============================================================================

def compute_strategy_portfolios(data, phylogroup='B2'):
    """Compute gene frequency portfolios for three strategies.

    Returns
    -------
    dict with:
        sources : list of str (e.g. ['Blood', 'Feces', 'Urine'])
        source_freqs : dict {source: (n_genes,) array}  — per-gene freq in each niche
        n_genes : int
        home_niche : (n_genes,) array of str — home niche for each gene
        portfolios : dict of strategy -> dict of niche -> (n_genes,) array
        source_sizes : dict {source: int}  — n genomes per source
    """
    result = per_gene_chi_squared(data, phylogroup=phylogroup)
    sources = result['sources_used']
    source_freqs = result['source_freqs']   # {source: (n_genes,) array}
    n_genes = len(result['p_values'])

    # Source sample sizes
    mask = data['phylogroup'] == phylogroup
    iso = data['isolation_source'][mask]
    source_sizes = {src: (iso == src).sum() for src in sources}

    # Home niche = source with highest frequency for each gene
    freq_matrix = np.column_stack([source_freqs[src] for src in sources])
    home_idx = freq_matrix.argmax(axis=1)
    home_niche = np.array([sources[i] for i in home_idx])

    # Mean frequency across niches (unweighted — each niche counts equally)
    f_mean = freq_matrix.mean(axis=1)

    # --- Strategy portfolios ---
    # Each strategy maps: niche -> (n_genes,) array of gene frequencies

    portfolios = {}

    # 1. PURE NICHE (all-in on current niche):
    #    carry gene g at f_H(g) if home(g) = current niche, else frequency = 0
    #    (purge genes not optimised for this niche)
    pure_niche = {}
    for src in sources:
        q = np.zeros(n_genes)
        mask_home = home_niche == src
        q[mask_home] = source_freqs[src][mask_home]
        pure_niche[src] = q
    portfolios['Pure niche\n(all-in)'] = pure_niche

    # 2. PURE BET-HEDGING:
    #    carry gene g at mean frequency, same in every niche
    pure_bet = {src: f_mean.copy() for src in sources}
    portfolios['Pure bet-hedging\n(insurance)'] = pure_bet

    # 3. OBSERVED E. COLI:
    #    carry gene g at its actual observed frequency in the current niche
    observed = {src: source_freqs[src].copy() for src in sources}
    portfolios['Observed\nE. coli'] = observed

    return {
        'sources': sources,
        'source_freqs': source_freqs,
        'n_genes': n_genes,
        'home_niche': home_niche,
        'portfolios': portfolios,
        'source_sizes': source_sizes,
        'f_mean': f_mean,
    }


def compute_fitness_matrix(portfolios, source_freqs, sources, cost=0.0):
    """Compute fitness(strategy, home, current_niche) for all combinations.

    fitness(S, X) = Σ_g q_S(g) * f_X(g) - c * Σ_g q_S(g)

    Returns dict: strategy_name -> (n_homes, n_niches) array
    where rows = home niche, cols = current niche.
    """
    fitness = {}
    for strat_name, strat_portfolios in portfolios.items():
        mat = np.zeros((len(sources), len(sources)))
        for h_idx, home in enumerate(sources):
            # Portfolio is what you carry, determined by HOME niche for
            # pure-niche and observed, same everywhere for bet-hedging
            q = strat_portfolios[home]
            for x_idx, current in enumerate(sources):
                f_x = source_freqs[current]
                benefit = np.sum(q * f_x)
                carrying = cost * np.sum(q)
                mat[h_idx, x_idx] = benefit - carrying
        fitness[strat_name] = mat
    return fitness


def expected_fitness_vs_switching(fitness_matrix, sources, source_sizes,
                                  p_range):
    """Compute expected fitness vs switching rate, weighted across home niches.

    For home niche H at switching rate p:
      E[W|H] = (1-p) * W(H,H) + p * mean_{Y≠H}[W(H,Y)]

    Weighted average across homes by population size.

    Returns dict: strategy_name -> (len(p_range),) array
    """
    total = sum(source_sizes.values())
    weights = np.array([source_sizes[src] / total for src in sources])

    expected = {}
    for strat_name, mat in fitness_matrix.items():
        ew = np.zeros(len(p_range))
        for p_idx, p in enumerate(p_range):
            per_home = np.zeros(len(sources))
            for h_idx in range(len(sources)):
                w_home = mat[h_idx, h_idx]
                away_indices = [j for j in range(len(sources)) if j != h_idx]
                w_away = np.mean([mat[h_idx, j] for j in away_indices])
                per_home[h_idx] = (1 - p) * w_home + p * w_away
            ew[p_idx] = np.sum(weights * per_home)
        expected[strat_name] = ew
    return expected


def find_crossover(p_range, fitness_a, fitness_b):
    """Find switching rate where fitness_b first exceeds fitness_a."""
    diff = fitness_b - fitness_a
    crossings = np.where(np.diff(np.sign(diff)))[0]
    if len(crossings) == 0:
        return None
    # Linear interpolation
    i = crossings[0]
    p0, p1 = p_range[i], p_range[i + 1]
    d0, d1 = diff[i], diff[i + 1]
    return p0 + (p1 - p0) * (-d0) / (d1 - d0)


# ==============================================================================
# Plotting
# ==============================================================================

STRATEGY_COLORS = {
    'Pure niche\n(all-in)': COLORS['red'],
    'Pure bet-hedging\n(insurance)': COLORS['blue'],
    'Observed\nE. coli': COLORS['green'],
}

STRATEGY_STYLES = {
    'Pure niche\n(all-in)': '--',
    'Pure bet-hedging\n(insurance)': ':',
    'Observed\nE. coli': '-',
}


def panel_a(ax, p_range, expected_c0, sources):
    """Fitness vs switching rate, no carrying cost."""
    for strat_name, ew in expected_c0.items():
        ax.plot(p_range, ew,
                color=STRATEGY_COLORS[strat_name],
                linestyle=STRATEGY_STYLES[strat_name],
                linewidth=2.0, label=strat_name)

    # Crossover
    niche_key = 'Pure niche\n(all-in)'
    bet_key = 'Pure bet-hedging\n(insurance)'
    obs_key = 'Observed\nE. coli'
    xover = find_crossover(p_range, expected_c0[niche_key],
                           expected_c0[bet_key])
    if xover is not None:
        ax.axvline(xover, color='grey', linestyle=':', linewidth=1,
                   alpha=0.7)
        ax.text(xover + 0.02, ax.get_ylim()[0] +
                0.6 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                f'Crossover\np = {xover:.2f}', fontsize=8, color='grey')

    ax.set_xlabel('Switching probability (p)')
    ax.set_ylabel('Expected fitness (Σ q·f)')
    ax.set_title('No carrying cost (c = 0)', fontsize=10)
    ax.legend(fontsize=7, loc='best')


def panel_b(ax, p_range, expected_c, cost, sources):
    """Fitness vs switching rate, with carrying cost."""
    niche_key = 'Pure niche\n(all-in)'
    bet_key = 'Pure bet-hedging\n(insurance)'
    obs_key = 'Observed\nE. coli'

    for strat_name, ew in expected_c.items():
        ax.plot(p_range, ew,
                color=STRATEGY_COLORS[strat_name],
                linestyle=STRATEGY_STYLES[strat_name],
                linewidth=2.0, label=strat_name)

    xover = find_crossover(p_range, expected_c[niche_key],
                           expected_c[bet_key])
    if xover is not None:
        ax.axvline(xover, color='grey', linestyle=':', linewidth=1,
                   alpha=0.7)
        ax.text(xover + 0.02, ax.get_ylim()[0] +
                0.6 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                f'Crossover\np = {xover:.2f}', fontsize=8, color='grey')

    ax.set_xlabel('Switching probability (p)')
    ax.set_ylabel(f'Expected net fitness (Σ q·f − {cost}·Σq)')
    ax.set_title(f'Carrying cost c = {cost}', fontsize=10)
    ax.legend(fontsize=7, loc='best')


def panel_c(ax, p_range, expected_c0):
    """Fitness advantage of observed E. coli over both extremes."""
    niche_key = 'Pure niche\n(all-in)'
    bet_key = 'Pure bet-hedging\n(insurance)'
    obs_key = 'Observed\nE. coli'

    obs = expected_c0[obs_key]
    niche = expected_c0[niche_key]
    bet = expected_c0[bet_key]

    # % advantage of observed over niche
    adv_vs_niche = (obs - niche) / obs * 100
    # % advantage of observed over bet-hedging (negative = bet wins)
    adv_vs_bet = (obs - bet) / obs * 100

    ax.fill_between(p_range, 0, adv_vs_niche, alpha=0.3,
                    color=COLORS['red'], label='vs pure niche')
    ax.plot(p_range, adv_vs_niche, color=COLORS['red'], linewidth=2)

    ax.fill_between(p_range, 0, adv_vs_bet, alpha=0.3,
                    color=COLORS['blue'], label='vs pure bet-hedging')
    ax.plot(p_range, adv_vs_bet, color=COLORS['blue'], linewidth=2)

    ax.axhline(0, color='black', linewidth=0.5)

    # Mark the observed-vs-bet crossover
    xover = find_crossover(p_range, bet, obs)
    if xover is not None:
        ax.axvline(xover, color='grey', linestyle=':', linewidth=1, alpha=0.7)
        ax.text(xover + 0.02, adv_vs_niche.max() * 0.4,
                f'p = {xover:.2f}', fontsize=8, color='grey')

    ax.set_xlabel('Switching probability (p)')
    ax.set_ylabel('Fitness advantage of observed E. coli (%)')
    ax.set_title('Observed strategy advantage', fontsize=10)
    ax.legend(fontsize=7, loc='right')


def plot_fitness_landscape(info, output_dir='.', fmt='png'):
    """Main plotting function."""
    setup_plotting()

    sources = info['sources']
    portfolios = info['portfolios']
    source_freqs = info['source_freqs']
    source_sizes = info['source_sizes']

    p_range = np.linspace(0, 1, 500)

    # --- Panel A: no carrying cost ---
    fitness_c0 = compute_fitness_matrix(portfolios, source_freqs, sources,
                                         cost=0.0)
    expected_c0 = expected_fitness_vs_switching(fitness_c0, sources,
                                                source_sizes, p_range)

    # --- Panel B: moderate carrying cost ---
    cost_b = 0.05
    fitness_cb = compute_fitness_matrix(portfolios, source_freqs, sources,
                                         cost=cost_b)
    expected_cb = expected_fitness_vs_switching(fitness_cb, sources,
                                                source_sizes, p_range)

    # --- Figure ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    panel_a(axes[0], p_range, expected_c0, sources)
    add_panel_label(axes[0], 'A')

    panel_b(axes[1], p_range, expected_cb, cost_b, sources)
    add_panel_label(axes[1], 'B')

    panel_c(axes[2], p_range, expected_c0)
    add_panel_label(axes[2], 'C')

    plt.tight_layout()
    save_figure(fig, 'fitness_landscape', output_dir=output_dir, fmt=fmt)
    plt.close(fig)

    return {
        'p_range': p_range,
        'expected_c0': expected_c0,
        'expected_cb': expected_cb,
        'cost_b': cost_b,
        'fitness_c0': fitness_c0,
    }


# ==============================================================================
# Print summary
# ==============================================================================

def print_summary(info, plot_data):
    """Print key results to stdout."""
    sources = info['sources']
    source_sizes = info['source_sizes']
    n_genes = info['n_genes']
    p_range = plot_data['p_range']

    niche_key = 'Pure niche\n(all-in)'
    bet_key = 'Pure bet-hedging\n(insurance)'
    obs_key = 'Observed\nE. coli'

    print_header('FITNESS LANDSCAPE: STRATEGY COMPARISON')

    print(f'\nData: {n_genes} accessory genes in B2')
    print(f'Sources: {", ".join(f"{s} (n={source_sizes[s]})" for s in sources)}')

    # Fitness at key switching rates (no cost)
    print(f'\n--- No carrying cost (c = 0) ---')
    print(f'{"Strategy":<28} {"p=0":>10} {"p=0.25":>10} {"p=0.50":>10} {"p=1.0":>10}')
    for strat in [niche_key, bet_key, obs_key]:
        ew = plot_data['expected_c0'][strat]
        vals = [ew[0], ew[124], ew[249], ew[-1]]
        label = strat.replace('\n', ' ')
        print(f'{label:<28} {vals[0]:>10.1f} {vals[1]:>10.1f} '
              f'{vals[2]:>10.1f} {vals[3]:>10.1f}')

    # Crossover (no cost)
    xover_c0 = find_crossover(p_range, plot_data['expected_c0'][niche_key],
                               plot_data['expected_c0'][bet_key])
    if xover_c0:
        print(f'\nCrossover (niche vs bet-hedging, c=0): p* = {xover_c0:.3f}')
    else:
        # Check which dominates
        if plot_data['expected_c0'][bet_key][0] > plot_data['expected_c0'][niche_key][0]:
            print(f'\nBet-hedging dominates at ALL switching rates (c=0)')
        else:
            print(f'\nNiche dominates at ALL switching rates (c=0)')

    # Crossover (with cost)
    cost_b = plot_data['cost_b']
    xover_cb = find_crossover(p_range, plot_data['expected_cb'][niche_key],
                               plot_data['expected_cb'][bet_key])
    if xover_cb:
        print(f'Crossover (niche vs bet-hedging, c={cost_b}): p* = {xover_cb:.3f}')

    # Observed vs alternatives
    print(f'\n--- Observed E. coli advantage ---')
    for p_val, p_label in [(0.0, 'p=0'), (0.25, 'p=0.25'), (0.5, 'p=0.5')]:
        idx = int(p_val * 499)
        obs_w = plot_data['expected_c0'][obs_key][idx]
        niche_w = plot_data['expected_c0'][niche_key][idx]
        bet_w = plot_data['expected_c0'][bet_key][idx]
        print(f'  {p_label}: observed vs niche = {(obs_w - niche_w)/obs_w*100:+.1f}%, '
              f'observed vs bet-hedging = {(obs_w - bet_w)/obs_w*100:+.1f}%')

    # Fitness matrices (no cost)
    print(f'\n--- Fitness matrix (c=0): current niche → ---')
    for strat in [niche_key, bet_key, obs_key]:
        label = strat.replace('\n', ' ')
        print(f'\n  {label}:')
        mat = plot_data['fitness_c0'][strat]
        print(f'  {"Home↓ / Current→":<18} {sources[0]:>10} {sources[1]:>10} {sources[2]:>10}')
        for h_idx, home in enumerate(sources):
            print(f'  {home:<18} {mat[h_idx,0]:>10.1f} {mat[h_idx,1]:>10.1f} {mat[h_idx,2]:>10.1f}')


# ==============================================================================
# Main
# ==============================================================================

def main():
    print_header('LOADING DATA')
    data = load_classification_data()

    print_header('COMPUTING STRATEGY PORTFOLIOS')
    info = compute_strategy_portfolios(data, phylogroup='B2')
    print(f'  {info["n_genes"]} accessory genes')
    print(f'  Sources: {info["sources"]}')
    print(f'  Sample sizes: {info["source_sizes"]}')

    # Count genes per home niche
    for src in info['sources']:
        n = (info['home_niche'] == src).sum()
        print(f'  Genes with home = {src}: {n}')

    print_header('PLOTTING')
    output_dir = os.path.dirname(os.path.abspath(__file__))
    plot_data = plot_fitness_landscape(info, output_dir=output_dir)

    print_summary(info, plot_data)

    print(f'\n✓ Figure saved to {output_dir}/fitness_landscape.png')


if __name__ == '__main__':
    main()
