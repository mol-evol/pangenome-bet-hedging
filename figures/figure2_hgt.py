#!/usr/bin/env python3
"""
================================================================================
FIGURE 2: GENE MAINTENANCE VIA HGT-SELECTION BALANCE
================================================================================

Shows how horizontal gene transfer (HGT) maintains genes at LOW frequencies
when selection alone would drive them to extinction.

Key insight: The threshold p* = c/(s+c) determines:
- Above p*: selection maintains gene at HIGH frequency (core gene)
- Below p*: gene would be lost, but HGT maintains it at LOW frequency (accessory)

This is a GENE-LEVEL mechanism, distinct from population-level bet-hedging.

USAGE:
    python figure2_hgt.py
    python figure2_hgt.py --quick
    python figure2_hgt.py -s 0.2 -c 0.01 --generations 10000

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

BASENAME = 'figure2_hgt'


# =============================================================================
# ANALYTICAL THEORY
# =============================================================================

def analytical_equilibrium(p, s, c, h, delta=0.001):
    """
    Analytical equilibrium frequency for HGT-selection balance.

    At equilibrium, gains from HGT balance losses from selection and deletion.

    Parameters:
        p: probability environment favours the gene
        s: selective benefit when gene is needed
        c: fitness cost when gene is not needed
        h: HGT rate (probability non-carrier acquires gene)
        delta: gene loss rate

    Returns:
        Equilibrium frequency (bounded to [0, 1])
    """
    s_net = p * s - (1 - p) * c

    if s_net < 0:
        # Gene deleterious on average: LOW equilibrium maintained by HGT
        f_eq = h / (abs(s_net) + delta + h)
    else:
        # Gene beneficial on average: HIGH equilibrium
        f_eq = 1 - delta / (s_net + delta + h)

    return max(0, min(1, f_eq))


# =============================================================================
# SIMULATION
# =============================================================================

def simulate_single(p, s, c, h, delta, n_generations=5000, n_replicates=10):
    """
    Simulate equilibrium frequency for a single parameter set.

    Returns mean and std of equilibrium frequency across replicates.
    """
    eq_freqs = []
    for _ in range(n_replicates):
        f = 0.1
        freq_history = []
        for _ in range(n_generations):
            # Selection
            if np.random.random() < p:
                w_carrier = 1 + s
            else:
                w_carrier = 1 - c
            w_noncarrier = 1

            mean_w = f * w_carrier + (1 - f) * w_noncarrier
            if mean_w > 0:
                f = f * w_carrier / mean_w

            # Gene loss
            f = f * (1 - delta)

            # HGT
            f = f + (1 - f) * h

            f = max(0.0001, min(0.9999, f))
            freq_history.append(f)

        eq_freqs.append(np.mean(freq_history[-1000:]))

    return np.mean(eq_freqs), np.std(eq_freqs)


def simulate(s=0.1, c=0.005, h=0.001, delta=0.001,
            n_generations=5000, n_replicates=20):
    """
    Simulate gene frequency dynamics with and without HGT.

    Compares equilibrium frequencies across different environmental frequencies p.
    """
    p_threshold = c / (s + c)

    # Test across range of environmental frequencies - focused around threshold
    p_values = np.array([0.01, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05,
                         0.055, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15])

    print(f"  Generations: {n_generations:,}")
    print(f"  Replicates:  {n_replicates}")
    print(f"  Testing {len(p_values)} values of p")
    print()

    results_with_hgt = []
    results_without_hgt = []

    for idx, p in enumerate(p_values):
        print_progress(idx, len(p_values), 'Simulating')

        eq_freqs_hgt = []
        eq_freqs_no_hgt = []

        for _ in range(n_replicates):
            # WITH HGT
            f = 0.1
            freq_history = []
            for _ in range(n_generations):
                # Selection
                if np.random.random() < p:
                    w_carrier = 1 + s
                else:
                    w_carrier = 1 - c
                w_noncarrier = 1

                mean_w = f * w_carrier + (1 - f) * w_noncarrier
                if mean_w > 0:
                    f = f * w_carrier / mean_w

                # Gene loss
                f = f * (1 - delta)

                # HGT: non-carriers acquire gene
                f = f + (1 - f) * h

                f = max(0.0001, min(0.9999, f))
                freq_history.append(f)

            eq_freqs_hgt.append(np.mean(freq_history[-1000:]))

            # WITHOUT HGT
            f = 0.1
            freq_history = []
            for _ in range(n_generations):
                if np.random.random() < p:
                    w_carrier = 1 + s
                else:
                    w_carrier = 1 - c
                w_noncarrier = 1

                mean_w = f * w_carrier + (1 - f) * w_noncarrier
                if mean_w > 0:
                    f = f * w_carrier / mean_w

                f = f * (1 - delta)
                f = max(0.0001, min(0.9999, f))
                freq_history.append(f)

            eq_freqs_no_hgt.append(np.mean(freq_history[-1000:]))

        results_with_hgt.append({
            'p': p,
            'eq_freq': np.mean(eq_freqs_hgt),
            'eq_std': np.std(eq_freqs_hgt),
            'analytical': analytical_equilibrium(p, s, c, h, delta)
        })

        results_without_hgt.append({
            'p': p,
            'eq_freq': np.mean(eq_freqs_no_hgt),
            'eq_std': np.std(eq_freqs_no_hgt)
        })

    print_progress(len(p_values), len(p_values), 'Simulating')

    return results_with_hgt, results_without_hgt, p_threshold


# =============================================================================
# FIGURE CREATION
# =============================================================================

def create_figure(results_hgt, results_no_hgt, p_threshold, s, c, h, delta,
                  n_generations=5000, n_replicates=10, quick=False):
    """
    Create four-panel figure.

    Panel A: Equilibrium frequency vs p, comparing with/without HGT
    Panel B: Phase diagram across (s, c) parameter space
    Panel C: Effect of carriage cost for different p values (SIMULATED)
    Panel D: Gene persistence depends on HGT rate (heterogeneous h)
    """
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    axes = axes.flatten()  # Makes indexing easier: axes[0], axes[1], axes[2], axes[3]

    # --- Panel A: Equilibrium frequency vs p ---
    ax = axes[0]

    p_vals = [r['p'] for r in results_hgt]
    eq_hgt = [r['eq_freq'] for r in results_hgt]
    eq_hgt_std = [r['eq_std'] for r in results_hgt]
    eq_no_hgt = [r['eq_freq'] for r in results_no_hgt]
    eq_no_hgt_std = [r['eq_std'] for r in results_no_hgt]

    # Soft background shading for regions (draw first, behind everything)
    ax.axhspan(0, 0.15, alpha=0.06, color='#2196F3', zorder=0)
    ax.axhspan(0.85, 1.0, alpha=0.06, color='#4CAF50', zorder=0)

    # Threshold line - subtle but clear
    ax.axvline(x=p_threshold, color='#616161', linestyle='--', lw=1.2,
               alpha=0.8, zorder=1)
    ax.text(p_threshold + 0.003, 0.52, f'p* = {p_threshold:.3f}',
            fontsize=8, color='#616161', rotation=90, va='center')

    # Simulation: with HGT - cleaner markers
    ax.errorbar(p_vals, eq_hgt, yerr=eq_hgt_std, fmt='o', color='#1976D2',
                capsize=2, capthick=1, markersize=7, markeredgecolor='white',
                markeredgewidth=0.8, label='With HGT', zorder=4, elinewidth=1)

    # Simulation: without HGT - distinct style
    ax.errorbar(p_vals, eq_no_hgt, yerr=eq_no_hgt_std, fmt='D', color='#E53935',
                capsize=2, capthick=1, markersize=5, markeredgecolor='white',
                markeredgewidth=0.8, alpha=0.85, label='Without HGT', zorder=3, elinewidth=1)

    # Region labels - cleaner positioning
    ax.text(0.008, 0.07, 'Rare', fontsize=9, color='#1976D2', alpha=0.7, style='italic')
    ax.text(0.008, 0.92, 'Core', fontsize=9, color='#388E3C', alpha=0.7, style='italic')

    ax.set_xlabel('Probability gene is beneficial (p)')
    ax.set_ylabel('Equilibrium frequency')
    ax.set_xlim(0, 0.16)
    ax.set_ylim(0, 1.0)
    ax.legend(loc='center right', frameon=True, fancybox=False,
              edgecolor='#BDBDBD', fontsize=8, framealpha=0.95)
    add_panel_label(ax, 'A')

    # --- Panel B: Phase diagram across (s, c) ---
    ax = axes[1]

    p_fixed = 0.03  # Gene beneficial only 3% of the time

    s_values = np.linspace(0.02, 0.3, 50)
    c_values = np.linspace(0.001, 0.01, 50)
    S, C = np.meshgrid(s_values, c_values)

    Eq_freq = np.zeros_like(S)
    for i in range(len(c_values)):
        for j in range(len(s_values)):
            Eq_freq[i, j] = analytical_equilibrium(p_fixed, S[i, j], C[i, j], h, delta)

    levels = np.linspace(0, 1.0, 21)
    contourf = ax.contourf(S, C * 100, Eq_freq, levels=levels, cmap='YlOrRd', alpha=0.9)

    contour_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    contours = ax.contour(S, C * 100, Eq_freq, levels=contour_levels,
                          colors='grey', linewidths=0.8)
    ax.clabel(contours, inline=True, fontsize=7, fmt='%.1f')

    # Add threshold boundary line: p* = p, i.e., c/(s+c) = p_fixed
    # Solving: c = p_fixed * s / (1 - p_fixed)
    # Above this line: p < p*, gene is below threshold (HGT-maintained)
    # Below this line: p > p*, gene is above threshold (selection-maintained)
    s_line = np.linspace(0.02, 0.3, 100)
    c_threshold_line = p_fixed * s_line / (1 - p_fixed) * 100  # convert to %
    ax.plot(s_line, c_threshold_line, '-', color='white', lw=3.5, alpha=1.0, zorder=4)
    ax.plot(s_line, c_threshold_line, '-', color='#1a1a1a', lw=2, alpha=0.9, zorder=5)

    cbar = plt.colorbar(contourf, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label('Equilibrium freq.', fontsize=9)

    ax.set_xlabel('Selective benefit (s)')
    ax.set_ylabel('Carriage cost (c, %)')
    ax.set_title(f'p = {p_fixed} (rarely needed)', fontsize=9)
    ax.set_xlim(0.02, 0.3)
    ax.set_ylim(0.1, 1.0)
    add_panel_label(ax, 'B')

    # --- Panel C: Equilibrium vs cost for different p (SIMULATED) ---
    ax = axes[2]

    n_c_points = 10 if quick else 20
    c_range = np.linspace(0.001, 0.01, n_c_points)  # 0.1% to 1%, consistent with Panel B
    p_examples = [0.02, 0.05, 0.10, 0.20]
    colors_p = [COLORS['red'], COLORS['orange'], COLORS['purple'], COLORS['green']]

    print("  Simulating Panel C...")
    total_sims = len(p_examples) * n_c_points
    sim_count = 0

    for p_val, col in zip(p_examples, colors_p):
        eq_means = []
        eq_stds = []
        for c_val in c_range:
            mean_eq, std_eq = simulate_single(p_val, s, c_val, h, delta,
                                              n_generations=n_generations,
                                              n_replicates=n_replicates)
            eq_means.append(mean_eq)
            eq_stds.append(std_eq)
            sim_count += 1
            print_progress(sim_count, total_sims, 'Panel C')
        ax.plot(c_range * 100, eq_means, '-', color=col, lw=2, label=f'p = {p_val}')
        ax.fill_between(c_range * 100,
                        np.array(eq_means) - np.array(eq_stds),
                        np.array(eq_means) + np.array(eq_stds),
                        color=col, alpha=0.15)

    ax.axhline(y=0.1, color=COLORS['grey'], linestyle=':', lw=1, alpha=0.5)
    ax.axhline(y=0.9, color=COLORS['grey'], linestyle=':', lw=1, alpha=0.5)

    ax.set_xlabel('Carriage cost (c, %)')
    ax.set_ylabel('Equilibrium frequency')
    ax.set_xlim(0.1, 1.0)
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='grey', fontsize=8)
    add_panel_label(ax, 'C')

    # --- Panel D: Gene persistence depends on HGT rate ---
    ax = axes[3]

    # Equilibrium for a net deleterious gene (p=0) as function of h
    def eq_vs_h(h_val, cost=0.01, loss=0.001):
        if h_val == 0:
            return 0
        return h_val / (cost + loss + h_val)

    h_range = np.logspace(-8, -1, 200)
    f_eq_vs_h = [eq_vs_h(hv, cost=c, loss=delta) for hv in h_range]

    ax.semilogx(h_range, f_eq_vs_h, '-', color=COLORS['blue'], lw=2)

    # Mark representative HGT rates (labels in figure legend, not on plot)
    # Green: ~10^-2 (typical of conjugative plasmids)
    # Orange: ~10^-3 (typical of ICEs)
    # Purple: ~10^-5 (typical of chromosomal genes)
    # Red: ~10^-8 (typical of phylogenetically distant genes)
    gene_categories = [
        (1e-2, COLORS['green']),
        (1e-3, COLORS['orange']),
        (1e-5, COLORS['purple']),
        (1e-8, COLORS['red']),
    ]

    for h_cat, color in gene_categories:
        f_cat = eq_vs_h(h_cat, cost=c, loss=delta)
        ax.plot(h_cat, f_cat, 'o', color=color, markersize=10,
                markeredgecolor='white', markeredgewidth=1.2, zorder=5)

    # Detection limit line
    ax.axhline(y=0.01, color=COLORS['grey'], linestyle=':', lw=1, alpha=0.7)
    ax.text(1e-7, 0.025, 'Detection limit (~1%)', fontsize=7, color=COLORS['grey'])

    ax.set_xlabel('HGT rate (h)')
    ax.set_ylabel('Equilibrium frequency')
    ax.set_ylim(0, 0.75)
    ax.set_xlim(1e-8, 1e-1)
    add_panel_label(ax, 'D')

    plt.tight_layout()
    return fig


def print_summary(results_hgt, p_threshold):
    """Print summary statistics."""
    print()
    print("-" * 70)
    print("RESULTS SUMMARY")
    print("-" * 70)
    print()
    print(f"Threshold p* = {p_threshold:.4f}")
    print()
    print(f"{'p':<8} {'Eq. Freq':<12} {'Category':<15} {'Theory':<12}")
    print("-" * 50)

    for r in results_hgt:
        if r['eq_freq'] < 0.1:
            cat = "RARE (HGT)"
        elif r['eq_freq'] > 0.9:
            cat = "CORE (selection)"
        else:
            cat = "ACCESSORY"
        print(f"{r['p']:<8.2f} {r['eq_freq']:<12.3f} {cat:<15} {r['analytical']:<12.3f}")

    print()
    print("Key insight: Genes with p < p* would be lost without HGT.")
    print("HGT maintains them at low but stable frequencies.")
    print()


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Figure 2: Gene Maintenance via HGT-Selection Balance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('-s', '--selective-benefit', type=float, default=PARAMS['s'],
                        help=f"Selective benefit when gene needed (default: {PARAMS['s']})")
    parser.add_argument('-c', '--carriage-cost', type=float, default=PARAMS['c'],
                        help=f"Fitness cost of carrying gene (default: {PARAMS['c']})")
    parser.add_argument('--hgt-rate', type=float, default=PARAMS['h'],
                        help=f"HGT rate (default: {PARAMS['h']})")
    parser.add_argument('--loss-rate', type=float, default=PARAMS['delta'],
                        help=f"Gene loss rate (default: {PARAMS['delta']})")
    parser.add_argument('--generations', '-g', type=int, default=5000,
                        help='Number of generations (default: 5000)')
    parser.add_argument('--replicates', '-r', type=int, default=20,
                        help='Number of replicates (default: 20)')
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
    h = args.hgt_rate
    delta = args.loss_rate

    print_header('FIGURE 2: GENE MAINTENANCE VIA HGT-SELECTION BALANCE', {
        'Selective benefit (s)': s,
        'Carriage cost (c)': c,
        'HGT rate (h)': h,
        'Gene loss rate (δ)': delta,
    })

    # Adjust for quick mode
    n_gen = args.generations // 5 if args.quick else args.generations
    n_rep = args.replicates // 4 if args.quick else args.replicates

    if args.quick:
        print("  [QUICK MODE - reduced parameters]")
        print()

    # Run simulation
    results_hgt, results_no_hgt, p_threshold = simulate(
        s=s, c=c, h=h, delta=delta,
        n_generations=n_gen, n_replicates=n_rep
    )

    # Create figure
    print()
    print("  Creating figure...")
    fig = create_figure(results_hgt, results_no_hgt, p_threshold, s, c, h, delta,
                        n_generations=n_gen, n_replicates=n_rep, quick=args.quick)

    # Save
    saved = save_figure(fig, BASENAME, output_dir, args.format)
    print()
    print("  Saved:")
    for path in saved:
        print(f"    → {path}")

    # Summary
    print_summary(results_hgt, p_threshold)

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
