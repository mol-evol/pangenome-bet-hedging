#!/usr/bin/env python3
"""
================================================================================
MECHANISM SIMULATIONS
================================================================================

Component module providing HGT/bet-hedging simulation functions.
Imported by figure1_variance_mechanism.py (panels C-E).

The mechanism:
- HGT maintains costly genes at low frequency (the insurance policy)
- This costs fitness today (the premium)
- This provides insurance for uncertain future environments

Key insight: Bet-hedging means sacrificing short-term (arithmetic mean) fitness
for long-term (geometric mean) success. HGT is the mechanism that maintains
the gene reservoir needed for this strategy.

Panel D tests robustness across environmental switching rates (autocorrelation).
Fast switching (τ=1) favors bet-hedging most; slow switching reduces the advantage.

USAGE:
    python mechanism_simulations.py
    python mechanism_simulations.py --quick
    python mechanism_simulations.py --format svg

================================================================================
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ── path setup ───────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared.plotting import COLORS, setup_plotting, save_figure, print_header, print_progress, add_panel_label
from shared.params import THEORY_PARAMS as PARAMS

BASENAME = 'mechanism_simulations'


# =============================================================================
# SIMULATION
# =============================================================================

def simulate_equilibrium(p, s, c, h, delta, n_generations):
    """
    Simulate gene frequency dynamics to find equilibrium.

    No extinction logic - just track how gene frequency and fitness evolve
    under selection, loss, and HGT.

    Returns:
        final_freq: equilibrium gene frequency
        mean_fitness: mean fitness over simulation
    """
    f = 0.5  # Start at 50% gene frequency
    fitness_sum = 0

    # Burn-in period to reach equilibrium
    burn_in = n_generations // 3

    for gen in range(n_generations):
        # Environment: gene beneficial with probability p, costly otherwise
        beneficial = np.random.random() < p

        if beneficial:
            w_carrier = 1 + s
            w_noncarrier = 1 - s
        else:
            w_carrier = 1 - c
            w_noncarrier = 1

        # Population mean fitness
        w_pop = f * w_carrier + (1 - f) * w_noncarrier

        # Only count fitness after burn-in
        if gen >= burn_in:
            fitness_sum += w_pop

        # Selection changes frequency
        if w_pop > 0:
            f = f * w_carrier / w_pop

        # Gene loss
        f = f * (1 - delta)

        # HGT: non-carriers acquire gene
        f = f + (1 - f) * h

        # Bounds (prevent numerical issues)
        f = max(1e-10, min(1 - 1e-10, f))

    mean_fit = fitness_sum / (n_generations - burn_in)
    return {
        'final_freq': f,
        'mean_fitness': mean_fit
    }


def run_simulations(p, s, c, h_values, delta, n_generations, n_populations):
    """
    Run simulations across different HGT rates.
    """
    results = []

    for h in h_values:
        freq_sum = 0
        fitness_sum = 0

        for _ in range(n_populations):
            result = simulate_equilibrium(p, s, c, h, delta, n_generations)
            freq_sum += result['final_freq']
            fitness_sum += result['mean_fitness']

        results.append({
            'h': h,
            'mean_freq': freq_sum / n_populations,
            'mean_fitness': fitness_sum / n_populations
        })

    return results


def simulate_trajectory(p, s, c, h, delta, n_generations):
    """
    Simulate a single trajectory of gene frequency over time.
    Returns the full frequency history.
    """
    f = 0.5
    freq_history = [f]

    for gen in range(n_generations):
        beneficial = np.random.random() < p

        if beneficial:
            w_carrier = 1 + s
            w_noncarrier = 1 - s
        else:
            w_carrier = 1 - c
            w_noncarrier = 1

        w_pop = f * w_carrier + (1 - f) * w_noncarrier
        if w_pop > 0:
            f = f * w_carrier / w_pop

        f = f * (1 - delta)
        f = f + (1 - f) * h
        f = max(1e-10, min(1 - 1e-10, f))

        freq_history.append(f)

    return freq_history


def simulate_with_autocorrelation(p, s, c, h, delta, n_generations, tau):
    """
    Simulate gene frequency dynamics with autocorrelated environments.

    Instead of i.i.d. environmental sampling each generation, the environment
    follows a Markov process with autocorrelation time tau.

    Parameters:
        tau: Environmental autocorrelation time (generations)
             tau=1 is i.i.d. (current model), tau=inf is static
             switch_prob = 1/tau each generation

    Returns:
        dict with final_freq and geometric_mean_fitness
    """
    f = 0.5
    burn_in = n_generations // 3
    log_fitness_sum = 0

    # Initialize environment state based on long-run probability p
    env_beneficial = np.random.random() < p

    # Switching probability per generation
    switch_prob = 1.0 / tau if tau > 0 else 0

    for gen in range(n_generations):
        # Markov environmental dynamics
        if tau == 1:
            # Special case: i.i.d. sampling (original model)
            env_beneficial = np.random.random() < p
        elif np.random.random() < switch_prob:
            # Environment switches - draw new state based on long-run frequency
            env_beneficial = np.random.random() < p
        # else: environment stays the same

        if env_beneficial:
            w_carrier = 1 + s
            w_noncarrier = 1 - s
        else:
            w_carrier = 1 - c
            w_noncarrier = 1

        # Population mean fitness
        w_pop = f * w_carrier + (1 - f) * w_noncarrier

        # Track log fitness for geometric mean (after burn-in)
        if gen >= burn_in and w_pop > 0:
            log_fitness_sum += np.log(w_pop)

        # Selection
        if w_pop > 0:
            f = f * w_carrier / w_pop

        # Gene loss
        f = f * (1 - delta)

        # HGT
        f = f + (1 - f) * h

        f = max(1e-10, min(1 - 1e-10, f))

    geo_mean_fitness = np.exp(log_fitness_sum / (n_generations - burn_in))

    return {
        'final_freq': f,
        'geo_mean_fitness': geo_mean_fitness
    }


def run_switching_rate_analysis(p, s, c, h, delta, n_generations, n_replicates, tau_values):
    """
    Run simulations across different environmental switching rates.

    Returns results comparing HGT vs no-HGT across tau values.
    """
    results = []

    for tau in tau_values:
        # With HGT
        freq_hgt = []
        geo_fit_hgt = []
        for _ in range(n_replicates):
            res = simulate_with_autocorrelation(p, s, c, h, delta, n_generations, tau)
            freq_hgt.append(res['final_freq'])
            geo_fit_hgt.append(res['geo_mean_fitness'])

        # Without HGT
        freq_no_hgt = []
        geo_fit_no_hgt = []
        for _ in range(n_replicates):
            res = simulate_with_autocorrelation(p, s, c, 0, delta, n_generations, tau)
            freq_no_hgt.append(res['final_freq'])
            geo_fit_no_hgt.append(res['geo_mean_fitness'])

        results.append({
            'tau': tau,
            'freq_hgt': np.mean(freq_hgt),
            'freq_no_hgt': np.mean(freq_no_hgt),
            'geo_fit_hgt': np.mean(geo_fit_hgt),
            'geo_fit_no_hgt': np.mean(geo_fit_no_hgt),
            'bet_hedging_advantage': np.mean(geo_fit_hgt) - np.mean(geo_fit_no_hgt)
        })

    return results


# =============================================================================
# FIGURE CREATION
# =============================================================================

def create_figure(results, p, s, c, delta, n_generations, h, switching_results=None):
    """
    Create four-panel figure demonstrating the bet-hedging mechanism.

    Panel A: Gene frequency vs HGT rate (the insurance policy)
    Panel B: Mean fitness vs HGT rate (the cost of insurance)
    Panel C: Gene frequency trajectories (the dynamics)
    Panel D: Bet-hedging advantage vs environmental switching rate
    """
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))

    h_vals = [r['h'] for r in results]
    mean_freqs = [r['mean_freq'] for r in results]
    mean_fitnesses = [r['mean_fitness'] for r in results]

    # --- Panel A: Gene frequency vs HGT rate ---
    ax = axes[0, 0]

    ax.plot(h_vals, mean_freqs, 'o-', color=COLORS['green'], lw=2,
            markersize=8, markeredgecolor='white', markeredgewidth=1)

    # Shade the "gene lost" zone
    ax.axhspan(0, 0.01, alpha=0.15, color=COLORS['red'], zorder=0)
    ax.text(0.005, 0.025, 'Gene lost\n(no insurance)', fontsize=8, color=COLORS['red'],
            alpha=0.8, ha='center', style='italic')

    ax.set_xlabel('HGT rate (h)')
    ax.set_ylabel('Equilibrium gene frequency')
    ax.set_xlim(-0.0005, 0.0105)
    ax.set_ylim(0, 0.55)
    ax.set_title('HGT maintains the insurance policy', fontsize=10)
    add_panel_label(ax, 'A')

    # --- Panel B: Mean fitness vs HGT rate ---
    ax = axes[0, 1]

    # Only plot where gene frequency is meaningful (gene not lost)
    h_maintained = [h_vals[i] for i in range(len(h_vals)) if mean_freqs[i] > 0.01]
    fit_maintained = [mean_fitnesses[i] for i in range(len(h_vals)) if mean_freqs[i] > 0.01]

    if h_maintained:
        ax.plot(h_maintained, fit_maintained, 's-', color=COLORS['blue'], lw=2,
                markersize=8, markeredgecolor='white', markeredgewidth=1)

    # Reference line at 1.0
    ax.axhline(y=1.0, color=COLORS['grey'], linestyle=':', lw=1, alpha=0.7)
    ax.text(0.008, 1.0002, 'No cost', fontsize=8, color=COLORS['grey'], ha='right')

    ax.set_xlabel('HGT rate (h)')
    ax.set_ylabel('Mean fitness')
    ax.set_xlim(-0.0005, 0.0105)
    # Zoom y-axis to show the cost clearly
    if fit_maintained:
        ymin = min(fit_maintained) - 0.002
        ax.set_ylim(ymin, 1.001)
    ax.set_title('Insurance has a fitness cost', fontsize=10)
    add_panel_label(ax, 'B')

    # --- Panel C: Gene frequency trajectories ---
    ax = axes[1, 0]

    print("  Simulating trajectories for Panel C...")

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

    ax.set_xlabel('Generation')
    ax.set_ylabel('Gene frequency')
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 1.0)
    ax.set_title('Without HGT, the gene is lost', fontsize=10)
    add_panel_label(ax, 'C')

    # --- Panel D: Gene frequency vs switching rate ---
    ax = axes[1, 1]

    if switching_results is not None:
        tau_vals = [r['tau'] for r in switching_results]
        freq_hgt = [r['freq_hgt'] for r in switching_results]
        freq_no_hgt = [r['freq_no_hgt'] for r in switching_results]

        # Plot gene frequency with HGT
        ax.semilogx(tau_vals, freq_hgt, 'o-', color=COLORS['blue'], lw=2,
                   markersize=8, markeredgecolor='white', markeredgewidth=1,
                   label='With HGT')

        # Plot without HGT (should be ~0)
        ax.semilogx(tau_vals, freq_no_hgt, 's--', color=COLORS['red'], lw=1.5,
                   markersize=6, markeredgecolor='white', markeredgewidth=1,
                   alpha=0.7, label='Without HGT')

        # Mark the i.i.d. case (tau=1)
        ax.axvline(x=1, color=COLORS['grey'], linestyle=':', lw=1, alpha=0.7)
        ax.text(1.15, max(freq_hgt) * 0.95, 'i.i.d.\n(τ=1)', fontsize=7,
                color=COLORS['grey'], ha='left', va='top')

        # Shade the "gene lost" zone
        ax.axhspan(0, 0.01, alpha=0.15, color=COLORS['red'], zorder=0)

        ax.set_xlabel('Environmental autocorrelation time (τ, generations)')
        ax.set_ylabel('Equilibrium gene frequency')
        ax.set_title('HGT maintains genes across switching rates', fontsize=10)
        ax.set_ylim(0, max(0.3, max(freq_hgt) * 1.2))
        ax.legend(loc='upper right', fontsize=7, frameon=True)

        # Add annotation
        ax.text(0.95, 0.15, 'Gene maintained\nacross all τ',
                transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
                color=COLORS['blue'], style='italic')

    else:
        ax.text(0.5, 0.5, 'Switching rate analysis\nnot run', transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color=COLORS['grey'])

    add_panel_label(ax, 'D')

    plt.tight_layout()
    return fig


def print_summary(results, switching_results=None):
    """Print summary of results."""
    print()
    print("-" * 70)
    print("RESULTS SUMMARY")
    print("-" * 70)
    print()
    print(f"{'HGT rate':<12} {'Gene Freq':<14} {'Mean Fitness':<14}")
    print("-" * 40)

    for r in results:
        print(f"{r['h']:<12.4f} {r['mean_freq']:<14.4f} {r['mean_fitness']:<14.4f}")

    if switching_results:
        print()
        print("-" * 70)
        print("SWITCHING RATE ANALYSIS")
        print("-" * 70)
        print()
        print(f"{'τ (gens)':<12} {'Freq (HGT)':<14} {'Freq (no HGT)':<14} {'Advantage':<14}")
        print("-" * 55)

        for r in switching_results:
            print(f"{r['tau']:<12} {r['freq_hgt']:<14.4f} {r['freq_no_hgt']:<14.4f} {r['bet_hedging_advantage']:<14.6f}")

    print()
    print("=" * 70)
    print("KEY FINDING: THE BET-HEDGING MECHANISM")
    print("=" * 70)
    print()
    print("Without HGT (h=0): Gene is lost → no insurance against environmental change")
    print()
    print("With HGT (h>0):    Gene is maintained at low frequency")
    print("                   → Costs fitness today (the premium)")
    print("                   → Provides insurance for tomorrow")
    print()
    print("ROBUSTNESS: Bet-hedging advantage is greatest under fast environmental")
    print("            switching (τ ≈ 1) and diminishes under slow switching,")
    print("            where individual lineages can track the environment.")
    print()
    print("This IS bet-hedging: sacrificing short-term fitness for long-term insurance.")
    print("HGT is the mechanism that maintains the pangenomic reservoir.")
    print()


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Figure 7: The Bet-Hedging Mechanism',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('-p', '--prob-beneficial', type=float, default=PARAMS['p'],
                        help=f"Probability gene is beneficial (default: {PARAMS['p']})")
    parser.add_argument('-s', '--selective-benefit', type=float, default=PARAMS['s'],
                        help=f"Selective benefit when gene beneficial (default: {PARAMS['s']})")
    parser.add_argument('-c', '--carriage-cost', type=float, default=PARAMS['c'],
                        help=f"Fitness cost of carrying gene (default: {PARAMS['c']})")
    parser.add_argument('--loss-rate', type=float, default=PARAMS['delta'],
                        help=f"Gene loss rate (default: {PARAMS['delta']})")
    parser.add_argument('--generations', '-g', type=int, default=2000,
                        help='Generations per simulation (default: 2000)')
    parser.add_argument('--populations', '-n', type=int, default=50,
                        help='Number of replicate simulations (default: 50)')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                        help='Output directory (default: ../output)')
    parser.add_argument('--format', type=str,
                        choices=['png', 'pdf', 'svg', 'all'], default='png',
                        help='Output format (default: png)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: fewer replicates')
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

    p = args.prob_beneficial
    s = args.selective_benefit
    c = args.carriage_cost
    delta = args.loss_rate

    # HGT rates to test (0 to 10x the standard h)
    h_base = PARAMS['h']
    h_values = [0] + [h_base * i for i in range(1, 11)]

    n_generations = args.generations
    n_populations = args.populations // 5 if args.quick else args.populations

    print_header('FIGURE 7: THE BET-HEDGING MECHANISM', {
        'Probability gene beneficial (p)': p,
        'Selective benefit (s)': s,
        'Carriage cost (c)': c,
        'Gene loss rate (δ)': delta,
    })

    if args.quick:
        print("  [QUICK MODE - reduced replicates]")
        print()

    print(f"  Generations per simulation: {n_generations:,}")
    print(f"  Number of replicates:       {n_populations}")
    print(f"  Testing {len(h_values)} HGT rates")
    print()

    # Run simulations
    print("Running simulations...")
    results = []
    for i, h in enumerate(h_values):
        print_progress(i, len(h_values), 'Simulating')

        result_list = run_simulations(p, s, c, [h], delta, n_generations, n_populations)
        results.append(result_list[0])

    print_progress(len(h_values), len(h_values), 'Simulating')

    # Run switching rate analysis for Panel D
    print()
    print("Running switching rate analysis (Panel D)...")
    tau_values = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    n_reps_switching = n_populations // 2 if args.quick else n_populations

    switching_results = []
    for i, tau in enumerate(tau_values):
        print_progress(i, len(tau_values), 'Switching rates')
        res = run_switching_rate_analysis(p, s, c, h_base, delta, n_generations,
                                          n_reps_switching, [tau])
        switching_results.extend(res)
    print_progress(len(tau_values), len(tau_values), 'Switching rates')

    # Create figure
    print()
    print("  Creating figure...")
    fig = create_figure(results, p, s, c, delta, n_generations, h_base, switching_results)

    # Save
    saved = save_figure(fig, BASENAME, output_dir, args.format)
    print()
    print("  Saved:")
    for path in saved:
        print(f"    → {path}")

    # Summary
    print_summary(results, switching_results)

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
