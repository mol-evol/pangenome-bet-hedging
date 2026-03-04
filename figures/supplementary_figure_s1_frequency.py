#!/usr/bin/env python3
"""
================================================================================
SUPPLEMENTARY FIGURE S1: OPTIMAL POPULATION DIVERSITY (BET-HEDGING)
================================================================================

At the POPULATION level, maintaining a MIX of carriers and non-carriers
maximises geometric mean fitness. This is the classic bet-hedging result.

Key insight: Intermediate carrier frequency f* outperforms both extremes
(all carriers or all non-carriers) by reducing fitness variance.

This is DIFFERENT from gene-level maintenance (Figure 2):
- Here we ask: given a gene exists, what POPULATION COMPOSITION is optimal?
- Answer: mixed populations beat monomorphic ones

USAGE:
    python supplementary_figure_s1_frequency.py
    python supplementary_figure_s1_frequency.py --quick
    python supplementary_figure_s1_frequency.py -s 0.4 -c 0.15 -p 0.25

================================================================================
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ── path setup ───────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared.plotting import COLORS, setup_plotting, save_figure, print_header, add_panel_label
from shared.params import THEORY_PARAMS as PARAMS

BASENAME = 'supplementary_s1_frequency'


# =============================================================================
# SIMULATION
# =============================================================================

def geometric_mean(fitnesses):
    """Calculate geometric mean using log transform for stability."""
    return np.exp(np.mean(np.log(np.maximum(fitnesses, 1e-10))))


def simulate(s=0.3, c=0.1, p=0.3, n_generations=2000, n_replicates=100):
    """
    Simulate population-level bet-hedging.

    For each carrier frequency f, compute geometric mean fitness across
    n_generations in a fluctuating environment.

    The population fitness in each generation is:
    - Favourable environment (prob p): W = 1 + f*s (carriers contribute)
    - Unfavourable (prob 1-p): W = 1 - f*c (carriers are costly)
    """
    f_values = np.linspace(0, 1, 41)
    results = []

    print(f"  Generations: {n_generations:,}")
    print(f"  Replicates:  {n_replicates}")
    print(f"  Testing {len(f_values)} carrier frequencies")
    print()

    for idx, f in enumerate(f_values):
        if idx % max(1, len(f_values) // 20) == 0:
            pct = 100 * idx / len(f_values)
            bar_length = 40
            filled = int(bar_length * idx / len(f_values))
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f'\r  Simulating: |{bar}| {pct:5.1f}% ({idx}/{len(f_values)})',
                  end='', flush=True)

        geo_means = []
        for _ in range(n_replicates):
            fitnesses = []
            for _ in range(n_generations):
                if np.random.random() < p:
                    W = 1 + f * s  # Population mean fitness
                else:
                    W = 1 - f * c
                fitnesses.append(W)
            geo_means.append(geometric_mean(np.array(fitnesses)))

        # Theoretical geometric mean
        W_fav = 1 + f * s
        W_unfav = 1 - f * c
        theoretical = W_fav**p * W_unfav**(1-p)

        results.append({
            'f': f,
            'sim_mean': np.mean(geo_means),
            'sim_std': np.std(geo_means),
            'theoretical': theoretical
        })

    print(f'\r  Simulating: |{"█" * 40}| 100.0% ({len(f_values)}/{len(f_values)})')
    return results


# =============================================================================
# FIGURE CREATION
# =============================================================================

def create_figure(results, s, c, p):
    """
    Create single-panel figure showing geometric mean fitness vs carrier frequency.

    Demonstrates that intermediate f maximises geometric mean fitness.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    f_vals = [r['f'] for r in results]
    sim_means = [r['sim_mean'] for r in results]
    sim_stds = [r['sim_std'] for r in results]
    theoretical = [r['theoretical'] for r in results]

    opt_idx = np.argmax(sim_means)
    opt_f = f_vals[opt_idx]
    opt_fitness = sim_means[opt_idx]

    # Confidence band
    ax.fill_between(f_vals,
                    np.array(sim_means) - np.array(sim_stds),
                    np.array(sim_means) + np.array(sim_stds),
                    alpha=0.3, color=COLORS['blue'])

    # Simulation
    ax.plot(f_vals, sim_means, 'o-', color=COLORS['blue'], markersize=4,
            label='Simulation (mean ± SD)')

    # Theory
    ax.plot(f_vals, theoretical, '--', color=COLORS['orange'], lw=2, label='Theory')

    # Baseline
    ax.axhline(y=1.0, color='grey', linestyle=':', alpha=0.5, label='Neutral (W=1)')

    # Optimum
    ax.axvline(x=opt_f, color=COLORS['green'], linestyle=':', alpha=0.7)
    ax.scatter([opt_f], [opt_fitness], s=120, color=COLORS['green'], zorder=5,
               marker='*', label=f'Optimum (f* = {opt_f:.2f})')

    # Monomorphic endpoints
    ax.scatter([0, 1], [sim_means[0], sim_means[-1]], s=60, color=COLORS['red'],
               marker='x', zorder=5, linewidths=2, label='Monomorphic')

    ax.set_xlabel('Carrier frequency (f)')
    ax.set_ylabel('Geometric mean fitness')
    ax.set_xlim(-0.02, 1.02)
    ax.legend(loc='lower center', frameon=True, fancybox=False, edgecolor='grey')

    ax.text(0.98, 0.98, f's = {s}, c = {c}, p = {p}',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='grey', alpha=0.8))

    plt.tight_layout()
    return fig


def print_summary(results, s, c, p):
    """Print summary statistics."""
    print()
    print("-" * 70)
    print("RESULTS SUMMARY")
    print("-" * 70)
    print()

    f_vals = [r['f'] for r in results]
    sim_means = [r['sim_mean'] for r in results]

    opt_idx = np.argmax(sim_means)
    opt_f = f_vals[opt_idx]

    print(f"Parameters: s = {s}, c = {c}, p = {p}")
    print()
    print(f"{'Carrier freq':<15} {'Geo Mean Fitness':<20} {'vs Baseline':<15}")
    print("-" * 50)

    for i in [0, opt_idx, -1]:
        r = results[i]
        diff = (r['sim_mean'] - 1) * 100
        sign = "+" if diff >= 0 else ""
        label = ""
        if i == 0:
            label = " (all non-carriers)"
        elif i == opt_idx:
            label = " <- OPTIMAL"
        else:
            label = " (all carriers)"
        print(f"{r['f']:<15.2f} {r['sim_mean']:<20.4f} {sign}{diff:<14.2f}%{label}")

    print()
    print(f"Optimal f* = {opt_f:.2f}")
    print()

    # Advantage of optimal over endpoints
    adv_over_zero = (results[opt_idx]['sim_mean'] - results[0]['sim_mean']) * 100
    adv_over_one = (results[opt_idx]['sim_mean'] - results[-1]['sim_mean']) * 100

    print(f"Advantage over f=0 (no carriers): +{adv_over_zero:.2f}%")
    print(f"Advantage over f=1 (all carriers): +{adv_over_one:.2f}%")
    print()
    print("Key insight: Mixed populations outperform monomorphic ones by")
    print("reducing fitness variance across environmental states.")
    print()


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Supplementary Figure S1: Optimal Population Diversity (Bet-Hedging)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('-s', '--selective-benefit', type=float, default=PARAMS['s'],
                        help=f"Selective benefit when gene needed (default: {PARAMS['s']})")
    parser.add_argument('-c', '--carriage-cost', type=float, default=0.1,
                        help='Fitness cost of carrying gene (default: 0.1)')
    parser.add_argument('-p', '--env-frequency', type=float, default=0.3,
                        help='Probability environment favours gene (default: 0.3)')
    parser.add_argument('--generations', '-g', type=int, default=2000,
                        help='Number of generations (default: 2000)')
    parser.add_argument('--replicates', '-r', type=int, default=100,
                        help='Number of replicates (default: 100)')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                        help='Output directory (default: ../output)')
    parser.add_argument('--format', type=str,
                        choices=['png', 'pdf', 'svg', 'all'], default='png',
                        help='Output format (default: png)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: reduced generations/replicates')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--show', action='store_true',
                        help='Display figure interactively')
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    setup_plotting()

    output_dir = args.output_dir or os.path.join(os.path.dirname(__file__), '..', 'output')

    s = args.selective_benefit
    c = args.carriage_cost
    p = args.env_frequency

    print_header('SUPPLEMENTARY FIGURE S1: OPTIMAL POPULATION DIVERSITY (BET-HEDGING)', {
        'Selective benefit (s)': s,
        'Carriage cost (c)': c,
        'Environmental frequency (p)': p,
    })

    # Adjust for quick mode
    n_gen = args.generations // 5 if args.quick else args.generations
    n_rep = args.replicates // 5 if args.quick else args.replicates

    if args.quick:
        print("  [QUICK MODE - reduced parameters]")
        print()

    # Run simulation
    results = simulate(s=s, c=c, p=p, n_generations=n_gen, n_replicates=n_rep)

    # Create figure
    print()
    print("  Creating figure...")
    fig = create_figure(results, s, c, p)

    # Save
    saved = save_figure(fig, BASENAME, output_dir, args.format)
    print()
    print("  Saved:")
    for path in saved:
        print(f"    → {path}")

    # Summary
    print_summary(results, s, c, p)

    if args.show:
        plt.show()
    else:
        plt.close(fig)

    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print()


if __name__ == '__main__':
    main()
