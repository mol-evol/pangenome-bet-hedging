#!/usr/bin/env python3
"""
================================================================================
FIGURE 3: U-SHAPED GENE FREQUENCY DISTRIBUTION
================================================================================

Shows how heterogeneous gene parameters produce the characteristic U-shape
observed in bacterial pangenomes.

Key insight: Each gene has its own threshold p*_i = c_i/(s_i + c_i)
- Genes with p > p*: maintained at HIGH frequency by selection (core genes)
- Genes with p < p*: maintained at LOW frequency by HGT (accessory genes)
- Few genes at intermediate frequencies → U-shaped distribution

USAGE:
    python figure3_ushape.py
    python figure3_ushape.py --quick
    python figure3_ushape.py --genes 5000 --generations 3000

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

BASENAME = 'figure3_ushape'


# =============================================================================
# SIMULATION
# =============================================================================

def simulate(n_genes=2000, n_generations=2000, n_replicates=30,
             h=0.0005, delta=0.001):
    """
    Simulate gene frequency dynamics for many genes with heterogeneous parameters.

    Each gene has its own (s, c, p) drawn from distributions.
    Returns equilibrium frequencies for all genes.
    """
    # Parameter distributions
    s_mean, s_std = 0.25, 0.15
    c_mean, c_std = 0.04, 0.03
    p_alpha, p_beta = 0.5, 2.0  # Right-skewed: most genes rarely needed

    print(f"  Generations:  {n_generations:,}")
    print(f"  Replicates:   {n_replicates}")
    print()

    # Generate parameters for all genes
    s_values = np.clip(np.random.normal(s_mean, s_std, n_genes), 0.05, 0.8)
    c_values = np.clip(np.random.normal(c_mean, c_std, n_genes), 0.005, 0.2)
    p_values = np.random.beta(p_alpha, p_beta, n_genes)

    # Calculate each gene's threshold and cost:benefit ratio
    thresholds = c_values / (s_values + c_values)
    cs_ratios = c_values / s_values
    above_threshold = p_values > thresholds

    final_frequencies = []

    for gene_idx in range(n_genes):
        if gene_idx % max(1, n_genes // 40) == 0:
            print_progress(gene_idx, n_genes, 'Simulating')

        s_g, c_g, p_g = s_values[gene_idx], c_values[gene_idx], p_values[gene_idx]

        gene_freqs = []
        for _ in range(n_replicates):
            f = 0.5
            for _ in range(n_generations):
                # Selection
                if np.random.random() < p_g:
                    w_carrier, w_noncarrier = 1 + s_g, 1
                else:
                    w_carrier, w_noncarrier = 1 - c_g, 1

                mean_w = f * w_carrier + (1 - f) * w_noncarrier
                if mean_w > 0:
                    f = f * w_carrier / mean_w

                # Gene loss
                f = f * (1 - delta)

                # HGT
                f = f + (1 - f) * h

                f = max(0.0001, min(0.9999, f))

            gene_freqs.append(f)

        final_frequencies.append(np.mean(gene_freqs))

    print_progress(n_genes, n_genes, 'Simulating')

    return {
        's_values': s_values,
        'c_values': c_values,
        'p_values': p_values,
        'thresholds': thresholds,
        'cs_ratios': cs_ratios,
        'above_threshold': above_threshold,
        'final_frequencies': np.array(final_frequencies)
    }


# =============================================================================
# FIGURE CREATION
# =============================================================================

def create_figure(results):
    """
    Create three-panel figure.

    Panel A: Distribution of cost:benefit ratios (c/s)
    Panel B: Distribution of selection direction thresholds (p*)
    Panel C: U-shaped equilibrium frequency distribution
    """
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

    f_finals = results['final_frequencies']
    thresholds = results['thresholds']
    cs_ratios = results['cs_ratios']

    # --- Panel A: Distribution of cost:benefit ratios ---
    ax = axes[0]

    ax.hist(cs_ratios, bins=30, color=COLORS['blue'], alpha=0.7,
            edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Cost:benefit ratio (c/s)')
    ax.set_ylabel('Number of genes')
    ax.set_title('Distribution of cost:benefit ratios')
    add_panel_label(ax, 'A')

    # --- Panel B: Distribution of selection direction thresholds ---
    ax = axes[1]

    ax.hist(thresholds, bins=30, color=COLORS['green'], alpha=0.7,
            edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Selection direction threshold (p*)')
    ax.set_ylabel('Number of genes')
    ax.set_title('Distribution of selection direction thresholds')
    add_panel_label(ax, 'B')

    # --- Panel C: U-shaped equilibrium frequency distribution ---
    ax = axes[2]

    bins = np.linspace(0, 1, 21)
    n, bins_out, patches = ax.hist(f_finals, bins=bins, alpha=0.7,
                                    edgecolor='black', linewidth=0.5)

    # Color bars by frequency category
    for i, patch in enumerate(patches):
        bin_center = (bins_out[i] + bins_out[i+1]) / 2
        if bin_center < 0.1:
            patch.set_facecolor(COLORS['blue'])
        elif bin_center > 0.9:
            patch.set_facecolor(COLORS['green'])
        else:
            patch.set_facecolor(COLORS['orange'])

    ax.set_xlabel('Gene frequency (f)')
    ax.set_ylabel('Number of genes')
    ax.set_title('Equilibrium frequency distribution')

    # Stats and legend
    rare = np.mean(f_finals < 0.1) * 100
    intermediate = np.mean((f_finals >= 0.1) & (f_finals <= 0.9)) * 100
    fixed = np.mean(f_finals > 0.9) * 100

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['blue'], label=f'Rare: {rare:.0f}%'),
        Patch(facecolor=COLORS['orange'], label=f'Intermediate: {intermediate:.0f}%'),
        Patch(facecolor=COLORS['green'], label=f'Fixed: {fixed:.0f}%')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True,
              fancybox=False, edgecolor='grey', fontsize=7)
    add_panel_label(ax, 'C')

    plt.tight_layout()
    return fig


def print_summary(results):
    """Print summary statistics."""
    print()
    print("-" * 70)
    print("RESULTS SUMMARY")
    print("-" * 70)
    print()

    f_finals = results['final_frequencies']
    thresholds = results['thresholds']
    cs_ratios = results['cs_ratios']

    n_genes = len(f_finals)
    rare = np.sum(f_finals < 0.1)
    intermediate = np.sum((f_finals >= 0.1) & (f_finals <= 0.9))
    core = np.sum(f_finals > 0.9)

    print(f"Total genes simulated: {n_genes}")
    print()
    print(f"{'Category':<20} {'Count':<10} {'Percentage':<15}")
    print("-" * 45)
    print(f"{'Rare (f < 0.1)':<20} {rare:<10} {100*rare/n_genes:.1f}%")
    print(f"{'Intermediate':<20} {intermediate:<10} {100*intermediate/n_genes:.1f}%")
    print(f"{'Core (f > 0.9)':<20} {core:<10} {100*core/n_genes:.1f}%")
    print()
    print(f"Cost:benefit ratio (c/s):")
    print(f"  Median: {np.median(cs_ratios):.3f}")
    print(f"  Range:  [{np.min(cs_ratios):.3f}, {np.max(cs_ratios):.3f}]")
    print()
    print(f"Selection direction threshold (p*):")
    print(f"  Median: {np.median(thresholds):.3f}")
    print(f"  Range:  [{np.min(thresholds):.3f}, {np.max(thresholds):.3f}]")
    print()
    print("Key insight: U-shape emerges from heterogeneous gene parameters.")
    print("Most genes are either rare (HGT-maintained) or core (selection-maintained).")
    print()


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Figure 3: U-Shaped Gene Frequency Distribution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--genes', '-n', type=int, default=2000,
                        help='Number of genes to simulate (default: 2000)')
    parser.add_argument('--generations', '-g', type=int, default=2000,
                        help='Number of generations (default: 2000)')
    parser.add_argument('--replicates', '-r', type=int, default=30,
                        help='Replicates per gene (default: 30)')
    parser.add_argument('--hgt-rate', type=float, default=PARAMS['h'],
                        help=f"HGT rate (default: {PARAMS['h']})")
    parser.add_argument('--loss-rate', type=float, default=PARAMS['delta'],
                        help=f"Gene loss rate (default: {PARAMS['delta']})")
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                        help='Output directory (default: ../output)')
    parser.add_argument('--format', type=str,
                        choices=['png', 'pdf', 'svg', 'all'], default='png',
                        help='Output format (default: png)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: reduced genes/generations/replicates')
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

    h = args.hgt_rate
    delta = args.loss_rate

    # Adjust for quick mode
    n_genes = args.genes // 10 if args.quick else args.genes
    n_gen = args.generations // 5 if args.quick else args.generations
    n_rep = args.replicates // 5 if args.quick else args.replicates

    print_header('FIGURE 3: U-SHAPED GENE FREQUENCY DISTRIBUTION', {
        'Number of genes': n_genes,
        'HGT rate (h)': h,
        'Gene loss rate (δ)': delta,
    })

    if args.quick:
        print("  [QUICK MODE - reduced parameters]")
        print()

    # Run simulation
    results = simulate(
        n_genes=n_genes,
        n_generations=n_gen,
        n_replicates=n_rep,
        h=h,
        delta=delta
    )

    # Create figure
    print()
    print("  Creating figure...")
    fig = create_figure(results)

    # Save
    saved = save_figure(fig, BASENAME, output_dir, args.format)
    print()
    print("  Saved:")
    for path in saved:
        print(f"    → {path}")

    # Summary
    print_summary(results)

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
