#!/usr/bin/env python3
"""
FIGURE 1: THE COST OF VARIANCE

Strategies with identical arithmetic mean but different variance have
dramatically different long-term outcomes.  Geometric mean predicts the winner.

Usage:
    python figure1_variance.py
    python figure1_variance.py --quick
    python figure1_variance.py --generations 2000 --replicates 1000
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ── path setup ───────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared.plotting import COLORS, setup_plotting, save_figure, print_header, print_progress, add_panel_label

BASENAME = 'figure1_variance'


# ── simulation ───────────────────────────────────────────────────────────────

def simulate(n_generations=1000, n_replicates=500):
    """Simulate three strategies with identical arithmetic mean, different variance."""
    strategies = {
        'Constant (σ²=0)':       {'good': 1.0, 'bad': 1.0, 'color': COLORS['blue']},
        'Moderate (σ²=0.04)':    {'good': 1.2, 'bad': 0.8, 'color': COLORS['green']},
        'High variance (σ²=0.25)': {'good': 1.5, 'bad': 0.5, 'color': COLORS['red']},
    }
    results = {name: {'log_pops': [], 'trajectories': []} for name in strategies}

    print(f"  Generations: {n_generations:,}")
    print(f"  Replicates:  {n_replicates:,}")
    print()

    for rep in range(n_replicates):
        if rep % max(1, n_replicates // 20) == 0:
            print_progress(rep, n_replicates, 'Simulating')

        environments = np.random.choice(['good', 'bad'], size=n_generations)

        for name, params in strategies.items():
            log_pop = 0
            traj = [0]
            for env in environments:
                log_pop += np.log10(params[env])
                traj.append(log_pop)
            results[name]['log_pops'].append(log_pop)
            if rep < 50:
                results[name]['trajectories'].append(traj)

    print_progress(n_replicates, n_replicates, 'Simulating')
    return results, strategies, n_generations


# ── figure ───────────────────────────────────────────────────────────────────

def create_figure(results, strategies, n_generations):
    """Panel A: trajectories.  Panel B: violin plots with theoretical predictions."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    # Panel A
    ax = axes[0]
    generations = np.arange(len(results['Constant (σ²=0)']['trajectories'][0]))
    for name, params in strategies.items():
        for i, traj in enumerate(results[name]['trajectories'][:20]):
            alpha = 0.3 if i > 0 else 0.8
            lw = 0.5 if i > 0 else 1.2
            label = name if i == 0 else None
            ax.plot(generations, traj, color=params['color'],
                    alpha=alpha, lw=lw, label=label)
    ax.set_xlabel('Generations')
    ax.set_ylabel(r'$\log_{10}$(relative population size)')
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='grey')
    add_panel_label(ax, 'A')

    # Panel B
    ax = axes[1]
    positions = [1, 2, 3]
    for i, (name, params) in enumerate(strategies.items()):
        log_pops = results[name]['log_pops']
        parts = ax.violinplot([log_pops], positions=[positions[i]],
                              showmeans=True, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor(params['color'])
            pc.set_alpha(0.7)
        parts['cmeans'].set_color('black')

        geo_mean = np.sqrt(params['good'] * params['bad'])
        theoretical_log = n_generations * np.log10(geo_mean)
        ax.scatter([positions[i]], [theoretical_log], marker='_',
                   s=100, color='black', zorder=5, linewidths=2)

    ax.set_xticks(positions)
    ax.set_xticklabels(['Constant\n(σ²=0)', 'Moderate\n(σ²=0.04)',
                        'High\n(σ²=0.25)'])
    ax.set_ylabel(r'Final $\log_{10}$(population)')
    ax.axhline(y=0, color='grey', linestyle='--', alpha=0.5)
    add_panel_label(ax, 'B')

    ax.text(0.98, 0.02, 'Black marks = theoretical\n(geometric mean prediction)',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=7,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='grey', alpha=0.8))

    plt.tight_layout()
    return fig


def print_summary(results, strategies, n_generations):
    print()
    print("-" * 70)
    print("RESULTS SUMMARY")
    print("-" * 70)
    print()
    print(f"{'Strategy':<25} {'Arith Mean':<12} {'Geo Mean':<12} {'Final log10':<15}")
    print("-" * 65)
    for name, params in strategies.items():
        arith_mean = (params['good'] + params['bad']) / 2
        geo_mean = np.sqrt(params['good'] * params['bad'])
        actual_mean = np.mean(results[name]['log_pops'])
        print(f"{name:<25} {arith_mean:<12.2f} {geo_mean:<12.4f} {actual_mean:<15.1f}")
    print()


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Figure 1: The Cost of Variance')
    parser.add_argument('--generations', '-g', type=int, default=1000)
    parser.add_argument('--replicates', '-r', type=int, default=500)
    parser.add_argument('--output-dir', '-o', type=str, default=None)
    parser.add_argument('--format', type=str, choices=['png', 'pdf', 'svg', 'all'], default='png')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--show', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    setup_plotting()

    output_dir = args.output_dir or os.path.join(os.path.dirname(__file__), '..', 'output')

    print_header('FIGURE 1: THE COST OF VARIANCE')

    n_gen = args.generations // 5 if args.quick else args.generations
    n_rep = args.replicates // 5 if args.quick else args.replicates
    if args.quick:
        print("  [QUICK MODE]\n")

    results, strategies, n_generations = simulate(n_gen, n_rep)

    print("\n  Creating figure...")
    fig = create_figure(results, strategies, n_generations)

    saved = save_figure(fig, BASENAME, output_dir, args.format)
    print("\n  Saved:")
    for p in saved:
        print(f"    → {p}")

    print_summary(results, strategies, n_generations)

    if args.show:
        plt.show()
    else:
        plt.close(fig)

    print("COMPLETE\n")


if __name__ == '__main__':
    main()
