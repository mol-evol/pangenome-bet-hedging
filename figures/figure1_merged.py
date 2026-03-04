#!/usr/bin/env python3
"""
FIGURE 1 (MERGED): THE COST OF VARIANCE AND THE BET-HEDGING MECHANISM

A unified figure demonstrating why bet-hedging works:

Top row (A-B):
  Panel A: Population trajectories under different variance strategies
  Panel B: Final population sizes showing the variance cost

Bottom row (C-E):
  Panel C: Equilibrium gene frequency vs HGT rate (the insurance policy)
  Panel D: Mean fitness vs HGT rate (the insurance premium)
  Panel E: Gene frequency trajectories with/without HGT (the mechanism)

This merged figure shows the complete story: strategies with identical arithmetic
mean but different variance have dramatically different outcomes (variance cost),
and HGT implements the bet-hedging mechanism to maintain genetic diversity as
insurance against environmental uncertainty.

Usage:
    python figure1_merged.py
    python figure1_merged.py --quick
    python figure1_merged.py --generations 2000 --replicates 1000
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── path setup ───────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared.plotting import COLORS, setup_plotting, save_figure, print_header, print_progress, add_panel_label
from shared.params import THEORY_PARAMS as PARAMS
from figure7_mechanism import simulate_equilibrium, run_simulations, simulate_trajectory

BASENAME = 'figure1_merged'


# ── VARIANCE SIMULATION (Figure 1 original) ──────────────────────────────────

def simulate_variance(n_generations=1000, n_replicates=500):
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
            print_progress(rep, n_replicates, 'Variance simulations')

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

    print_progress(n_replicates, n_replicates, 'Variance simulations')
    return results, strategies, n_generations


# ── MECHANISM SIMULATION (Figure 7 panels) ───────────────────────────────────

def run_mechanism_simulations(p, s, c, delta, n_generations, n_populations):
    """Run simulations for mechanism panels (A, B, C from figure7)."""
    # HGT rates to test (0 to 10x the standard h)
    h_base = PARAMS['h']
    h_values = [0] + [h_base * i for i in range(1, 11)]

    print(f"  Generations: {n_generations:,}")
    print(f"  Replicates:  {n_populations:,}")
    print(f"  Testing {len(h_values)} HGT rates")
    print()

    results = []
    for i, h in enumerate(h_values):
        print_progress(i, len(h_values), 'Mechanism simulations')

        result_list = run_simulations(p, s, c, [h], delta, n_generations, n_populations)
        results.append(result_list[0])

    print_progress(len(h_values), len(h_values), 'Mechanism simulations')
    return results, h_base


# ── figure ───────────────────────────────────────────────────────────────────

def create_figure(variance_results, variance_strategies, variance_n_gen,
                  mechanism_results, h_base, p, s, c, delta):
    """
    Create merged 5-panel figure using gridspec.

    Top row: 2 panels (A, B) - variance cost
    Bottom row: 3 panels (C, D, E) - bet-hedging mechanism
    """
    fig = plt.figure(figsize=(11, 7))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3,
                           height_ratios=[1, 1])

    # ─── PANEL A: Population trajectories (variance) ───
    ax = fig.add_subplot(gs[0, :1])  # Top-left

    results = variance_results
    strategies = variance_strategies
    n_generations = variance_n_gen

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
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='grey', fontsize=8)
    add_panel_label(ax, 'A')

    # ─── PANEL B: Final population sizes (variance) ───
    ax = fig.add_subplot(gs[0, 1:])  # Top-right

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

    # ─── PANEL C: Gene frequency vs HGT rate ───
    ax = fig.add_subplot(gs[1, 0])

    h_vals = [r['h'] for r in mechanism_results]
    mean_freqs = [r['mean_freq'] for r in mechanism_results]

    ax.plot(h_vals, mean_freqs, 'o-', color=COLORS['green'], lw=2,
            markersize=7, markeredgecolor='white', markeredgewidth=1)

    # Shade the "gene lost" zone
    ax.axhspan(0, 0.01, alpha=0.15, color=COLORS['red'], zorder=0)
    ax.text(0.005, 0.025, 'Gene lost', fontsize=7, color=COLORS['red'],
            alpha=0.8, ha='center', style='italic')

    ax.set_xlabel('HGT rate (h)', fontsize=9)
    ax.set_ylabel('Equilibrium gene frequency', fontsize=9)
    ax.set_xlim(-0.0005, 0.0105)
    ax.set_ylim(0, 0.55)
    add_panel_label(ax, 'C')

    # ─── PANEL D: Mean fitness vs HGT rate ───
    ax = fig.add_subplot(gs[1, 1])

    mean_fitnesses = [r['mean_fitness'] for r in mechanism_results]

    # Only plot where gene frequency is meaningful (gene not lost)
    h_maintained = [h_vals[i] for i in range(len(h_vals)) if mean_freqs[i] > 0.01]
    fit_maintained = [mean_fitnesses[i] for i in range(len(h_vals)) if mean_freqs[i] > 0.01]

    if h_maintained:
        ax.plot(h_maintained, fit_maintained, 's-', color=COLORS['blue'], lw=2,
                markersize=7, markeredgecolor='white', markeredgewidth=1)

    # Reference line at 1.0
    ax.axhline(y=1.0, color=COLORS['grey'], linestyle=':', lw=1, alpha=0.7)
    ax.text(0.008, 1.0002, 'No cost', fontsize=7, color=COLORS['grey'], ha='right')

    ax.set_xlabel('HGT rate (h)', fontsize=9)
    ax.set_ylabel('Mean fitness', fontsize=9)
    ax.set_xlim(-0.0005, 0.0105)
    # Zoom y-axis to show the cost clearly
    if fit_maintained:
        ymin = min(fit_maintained) - 0.002
        ax.set_ylim(ymin, 1.001)
    add_panel_label(ax, 'D')

    # ─── PANEL E: Gene frequency trajectories ───
    ax = fig.add_subplot(gs[1, 2])

    print("  Simulating trajectories for Panel E...")

    n_trajectories = 20
    h_maintain = PARAMS['h']

    # h=0 trajectories (gene will be lost)
    for _ in range(n_trajectories):
        traj = simulate_trajectory(p, s, c, h=0, delta=delta, n_generations=500)
        ax.plot(traj, color=COLORS['red'], alpha=0.3, lw=0.6)

    # h>0 trajectories (gene will be maintained)
    for _ in range(n_trajectories):
        traj = simulate_trajectory(p, s, c, h=h_maintain, delta=delta, n_generations=500)
        ax.plot(traj, color=COLORS['blue'], alpha=0.3, lw=0.6)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COLORS['red'], lw=2, label='h = 0 (no HGT)'),
        Line2D([0], [0], color=COLORS['blue'], lw=2, label=f'h = {h_maintain:.4f} (with HGT)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=7, frameon=True)

    ax.set_xlabel('Generation', fontsize=9)
    ax.set_ylabel('Gene frequency', fontsize=9)
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 1.0)
    add_panel_label(ax, 'E')

    return fig


def print_summary_variance(results, strategies, n_generations):
    print()
    print("-" * 70)
    print("VARIANCE COST RESULTS")
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


def print_summary_mechanism(results):
    print()
    print("-" * 70)
    print("BET-HEDGING MECHANISM RESULTS")
    print("-" * 70)
    print()
    print(f"{'HGT rate':<12} {'Gene Freq':<14} {'Mean Fitness':<14}")
    print("-" * 40)

    for r in results:
        print(f"{r['h']:<12.4f} {r['mean_freq']:<14.4f} {r['mean_fitness']:<14.4f}")
    print()


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Figure 1 (Merged): Variance Cost and Bet-Hedging Mechanism')

    # Variance simulation parameters
    parser.add_argument('--generations', '-g', type=int, default=1000,
                        help='Generations for variance simulation (default: 1000)')
    parser.add_argument('--replicates', '-r', type=int, default=500,
                        help='Replicates for variance simulation (default: 500)')

    # Mechanism parameters (from THEORY_PARAMS)
    parser.add_argument('-p', '--prob-beneficial', type=float, default=PARAMS['p'],
                        help=f"Probability gene is beneficial (default: {PARAMS['p']})")
    parser.add_argument('-s', '--selective-benefit', type=float, default=PARAMS['s'],
                        help=f"Selective benefit when gene beneficial (default: {PARAMS['s']})")
    parser.add_argument('-c', '--carriage-cost', type=float, default=PARAMS['c'],
                        help=f"Fitness cost of carrying gene (default: {PARAMS['c']})")
    parser.add_argument('--loss-rate', type=float, default=PARAMS['delta'],
                        help=f"Gene loss rate (default: {PARAMS['delta']})")
    parser.add_argument('--mechanism-generations', type=int, default=2000,
                        help='Generations per mechanism simulation (default: 2000)')
    parser.add_argument('--mechanism-populations', type=int, default=50,
                        help='Number of replicate mechanism simulations (default: 50)')

    # Output parameters
    parser.add_argument('--output-dir', '-o', type=str, default=None)
    parser.add_argument('--format', type=str, choices=['png', 'pdf', 'svg', 'all'], default='png')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: reduced replicates for both simulations')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--show', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    setup_plotting()

    output_dir = args.output_dir or os.path.join(os.path.dirname(__file__), '..', 'output')

    print_header('FIGURE 1 (MERGED): VARIANCE COST AND BET-HEDGING MECHANISM')

    # Adjust parameters for quick mode
    n_gen_var = args.generations // 5 if args.quick else args.generations
    n_rep_var = args.replicates // 5 if args.quick else args.replicates
    n_gen_mech = args.mechanism_generations // 5 if args.quick else args.mechanism_generations
    n_pop_mech = args.mechanism_populations // 5 if args.quick else args.mechanism_populations

    if args.quick:
        print("  [QUICK MODE]\n")

    # Run variance simulation
    print("Running variance cost simulation...")
    variance_results, variance_strategies, variance_n_gen = simulate_variance(n_gen_var, n_rep_var)

    # Run mechanism simulation
    print("\nRunning bet-hedging mechanism simulation...")
    mechanism_results, h_base = run_mechanism_simulations(
        args.prob_beneficial, args.selective_benefit, args.carriage_cost,
        args.loss_rate, n_gen_mech, n_pop_mech
    )

    # Create figure
    print("\n  Creating figure...")
    fig = create_figure(variance_results, variance_strategies, variance_n_gen,
                        mechanism_results, h_base,
                        args.prob_beneficial, args.selective_benefit, args.carriage_cost,
                        args.loss_rate)

    # Save
    saved = save_figure(fig, BASENAME, output_dir, args.format)
    print("\n  Saved:")
    for p in saved:
        print(f"    → {p}")

    # Summary
    print_summary_variance(variance_results, variance_strategies, variance_n_gen)
    print_summary_mechanism(mechanism_results)

    if args.show:
        plt.show()
    else:
        plt.close(fig)

    print("COMPLETE\n")


if __name__ == '__main__':
    main()
