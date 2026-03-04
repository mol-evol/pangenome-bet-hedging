#!/usr/bin/env python3
"""
================================================================================
FIGURE 3 MERGED: U-SHAPED DISTRIBUTION + COMPLEXITY THRESHOLD
================================================================================

Combines two key perspectives on pangenome structure:

TOP ROW (U-shaped distribution, 3 panels):
  Shows how heterogeneous gene parameters produce the characteristic U-shape
  observed in bacterial pangenomes.
  - Panel A: Distribution of cost:benefit ratios (c/s)
  - Panel B: Distribution of selection direction thresholds (p*)
  - Panel C: U-shaped equilibrium frequency distribution

BOTTOM ROW (Complexity threshold, 3 panels):
  Shows that as environmental complexity E increases, distributed pangenomes
  become OBLIGATE above E_crit = s/c.
  - Panel D: Fitness heatmap showing complexity threshold
  - Panel E: Multi-line m* collapse across parameter ratios
  - Panel F: Phase diagram showing coverage advantage

USAGE:
    python figure3_merged.py
    python figure3_merged.py --quick
    python figure3_merged.py --genes 5000 --generations 3000

================================================================================
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ── path setup ───────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared.plotting import COLORS, setup_plotting, save_figure, print_header, print_progress, add_panel_label
from shared.params import THEORY_PARAMS as PARAMS

BASENAME = 'figure3_merged'


# =============================================================================
# U-SHAPED DISTRIBUTION SIMULATION
# =============================================================================

def simulate_ushape(n_genes=2000, n_generations=2000, n_replicates=30,
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
            print_progress(gene_idx, n_genes, 'Simulating U-shape')

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

    print_progress(n_genes, n_genes, 'Simulating U-shape')

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
# COMPLEXITY THRESHOLD SIMULATION (Functions from figure4_complexity.py)
# =============================================================================

def geometric_mean(fitnesses):
    """Calculate geometric mean using log transform for stability."""
    return np.exp(np.mean(np.log(np.maximum(fitnesses, 1e-10))))


def geo_mean_single(E, m, s, c):
    """
    Analytical geometric mean fitness for single-genome strategy.

    Each generation, one of E environments is needed (uniform).
    Individual carries genes 0..m-1, so gene matches with prob min(m,E)/E.
    Fitness when match: 1 + s - c*m.  When no match: 1 - c*m.
    Geometric mean = exp(E[ln W]).
    """
    m_eff = min(m, E)
    W_match = 1 + s - c * m
    W_miss = 1 - c * m
    if W_match <= 0 or W_miss <= 0:
        return 0.0
    p_match = m_eff / E
    return np.exp(p_match * np.log(W_match) + (1 - p_match) * np.log(W_miss))


def geo_mean_distributed(E, m, k, s, c):
    """
    Analytical geometric mean fitness for distributed pangenome strategy.

    Population maintains m categories, each individual carries k.
    Each generation, one of E environments is needed (uniform).
    If gene is in pangenome (prob min(m,E)/E): population fitness =
        (k/m)(1+s-ck) + (1-k/m)(1-ck) = 1-ck + sk/m.
    If gene is NOT in pangenome (prob max(0,E-m)/E): W = 1-ck.
    Geometric mean = exp(E[ln W_pop]).
    """
    m_eff = min(m, E)
    W_covered = 1 - c * k + s * k / m
    W_uncovered = 1 - c * k
    if W_covered <= 0 or W_uncovered <= 0:
        return 0.0
    p_covered = m_eff / E
    if p_covered >= 1.0:
        return W_covered  # Constant every generation, geo mean = value
    return np.exp(p_covered * np.log(W_covered)
                  + (1 - p_covered) * np.log(W_uncovered))


def find_best_single(E, s, c, n_gen=500, n_rep=30):
    """Find optimal m for single genome strategy.

    Uses analytical geometric mean (exact) for optimization.
    Searches m from 1 to E+5, allowing the collapse above E_crit
    to emerge naturally.
    """
    best_m, best_fit = 1, 0
    for m in range(1, E + 5):
        fit = geo_mean_single(E, m, s, c)
        if fit > best_fit:
            best_fit, best_m = fit, m
    return best_m, best_fit


def find_best_distributed(E, s, c, n_gen=500, n_rep=30):
    """Find optimal (m, k) for distributed strategy.

    Uses analytical geometric mean (exact) for optimization.
    Searches m from 1 to 2*E and k from 1 to m, so the result
    m* ≈ E is genuinely discovered, not assumed.
    """
    best_m, best_k, best_fit = 1, 1, 0
    for m in range(1, 2 * E + 1):
        for k in range(1, m + 1):
            fit = geo_mean_distributed(E, m, k, s, c)
            if fit > best_fit:
                best_fit, best_m, best_k = fit, m, k
    return best_m, best_k, best_fit


def simulate_complexity(s=0.3, c=0.02, n_gen=500, n_rep=30):
    """
    Simulate complexity threshold data for three panels.

    Panel D: Heatmap of single-genome fitness across (m, E) space.
    Panel E: Phase diagram showing coverage advantage of distributed
    Panel F: Multi-line m* curves.
    """
    E_crit = s / c

    # Panel D: full heatmap — E from 1..30, m from 1..35
    print("  Running Panel D: fitness heatmap...")
    E_range = np.arange(1, 31)
    m_range = np.arange(1, 36)
    fitness_grid = np.zeros((len(E_range), len(m_range)))
    best_m_single = np.zeros(len(E_range), dtype=int)

    for i, E in enumerate(E_range):
        best_fit = -1
        for j, m in enumerate(m_range):
            f = geo_mean_single(E, m, s, c)
            fitness_grid[i, j] = f
            if f > best_fit:
                best_fit = f
                best_m_single[i] = m

    heatmap_data = {
        'E_range': E_range,
        'm_range': m_range,
        'fitness_grid': fitness_grid,
        'best_m': best_m_single,
    }

    # Panel E: m* vs E for multiple s/c ratios
    print("  Running Panel E: multi-line m* curves...")
    E_line = np.arange(1, 41)
    sc_ratios = [5, 10, 15, 20, 30]
    c_fixed = 0.02
    multiline_data = {}
    for ratio in sc_ratios:
        s_val = ratio * c_fixed
        best_ms = [find_best_single(E, s_val, c_fixed)[0] for E in E_line]
        multiline_data[ratio] = best_ms
    multiline_data['E_range'] = E_line
    multiline_data['sc_ratios'] = sc_ratios

    # Panel F: phase diagram (coverage advantage)
    print("  Running Panel F: phase diagram...")

    E_phase = np.arange(1, 31)
    ratio_phase = np.arange(1, 16)
    c_fixed = 0.02

    coverage_diff = np.zeros((len(ratio_phase), len(E_phase)))

    for i, ratio in enumerate(ratio_phase):
        s_panel = ratio * c_fixed
        for j, E in enumerate(E_phase):
            m_s, _ = find_best_single(E, s_panel, c_fixed, n_gen, n_rep)
            m_d, _, _ = find_best_distributed(E, s_panel, c_fixed, n_gen, n_rep)
            coverage_diff[i, j] = m_d / E - m_s / E

    print("  Phase diagram complete.")

    return E_crit, heatmap_data, multiline_data, {
        'E_values': E_phase,
        'ratio_values': ratio_phase,
        'coverage_diff': coverage_diff
    }


# =============================================================================
# FIGURE CREATION
# =============================================================================

def create_figure(ushape_results, complexity_data):
    """
    Create six-panel merged figure (2 rows × 3 columns).

    Top row (U-shaped distribution):
      Panel A: Distribution of cost:benefit ratios (c/s)
      Panel B: Distribution of selection direction thresholds (p*)
      Panel C: U-shaped equilibrium frequency distribution

    Bottom row (Complexity threshold):
      Panel D: Fitness heatmap (single-genome strategy across E and m)
      Panel E: Multi-line m* collapse
      Panel F: Phase diagram (coverage advantage)
    """
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))

    # =========================================================================
    # TOP ROW: U-SHAPED DISTRIBUTION
    # =========================================================================

    f_finals = ushape_results['final_frequencies']
    thresholds = ushape_results['thresholds']
    cs_ratios = ushape_results['cs_ratios']

    # --- Panel A: Distribution of cost:benefit ratios ---
    ax = axes[0, 0]

    ax.hist(cs_ratios, bins=30, color=COLORS['blue'], alpha=0.7,
            edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Cost:benefit ratio (c/s)')
    ax.set_ylabel('Number of genes')
    ax.set_title('Distribution of cost:benefit ratios')
    add_panel_label(ax, 'A')

    # --- Panel B: Distribution of selection direction thresholds ---
    ax = axes[0, 1]

    ax.hist(thresholds, bins=30, color=COLORS['green'], alpha=0.7,
            edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Selection direction threshold (p*)')
    ax.set_ylabel('Number of genes')
    ax.set_title('Distribution of selection direction thresholds')
    add_panel_label(ax, 'B')

    # --- Panel C: U-shaped equilibrium frequency distribution ---
    ax = axes[0, 2]

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

    # =========================================================================
    # BOTTOM ROW: COMPLEXITY THRESHOLD
    # =========================================================================

    E_crit, heatmap_data, multiline_data, phase_data = complexity_data

    # --- Panel D: Fitness heatmap (E on x-axis, m on y-axis) ---
    ax = axes[1, 0]

    E_range = heatmap_data['E_range']
    m_range = heatmap_data['m_range']
    grid = heatmap_data['fitness_grid']
    best_m = heatmap_data['best_m']

    # Normalise each row by its peak fitness
    grid_norm = grid.copy()
    for i in range(len(E_range)):
        row_max = grid[i].max()
        if row_max > 0:
            grid_norm[i] = grid[i] / row_max
    grid_norm[grid <= 0] = np.nan

    im = ax.pcolormesh(E_range - 0.5, m_range - 0.5, grid_norm.T,
                       cmap='magma', vmin=0, vmax=1.0, shading='auto')

    ax.plot(E_range, best_m, '-', color='white', lw=3, zorder=4)
    ax.plot(E_range, best_m, '-', color='#44AA99', lw=1.8, zorder=5,
            label='Optimal $m^*$')

    ax.axvline(x=E_crit, color='#44AA99', linestyle='--', lw=1.5,
               alpha=0.9, zorder=5, label=f'$E_{{crit}} = {E_crit:.0f}$')

    diag = np.arange(1, 31)
    ax.plot(diag, diag, ':', color='white', alpha=0.5, lw=1, label='$m = E$')

    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('Geometric mean fitness\n(fraction of row optimum)', fontsize=7)

    ax.set_xlabel('Environmental complexity ($E$)')
    ax.set_ylabel('Adaptive categories\nper individual ($m$)')
    ax.legend(loc='upper left', fontsize=6, frameon=True,
              edgecolor='#cccccc', framealpha=0.9, facecolor='white')
    ax.set_xlim(0.5, 30.5)
    ax.set_ylim(0.5, 35)
    add_panel_label(ax, 'D')

    # --- Panel E: Multi-line m* vs E ---
    ax = axes[1, 1]

    colors_lines = ['#332288', '#44AA99', '#DDCC77', '#CC6677', '#AA4499']
    E_line = multiline_data['E_range']
    sc_ratios = multiline_data['sc_ratios']

    for idx, ratio in enumerate(sc_ratios):
        best_ms = multiline_data[ratio]
        ax.plot(E_line, best_ms, '-', color=colors_lines[idx], lw=2,
                label=f'$s/c$ = {ratio}')

    ax.plot(E_line, E_line, '--', color='#999999', lw=1, alpha=0.6,
            label='$m^* = E$')

    ax.set_xlabel('Environmental complexity ($E$)')
    ax.set_ylabel('Optimal categories ($m^*$)')
    ax.legend(loc='upper left', fontsize=6, frameon=True,
              edgecolor='#cccccc', framealpha=0.9)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 40)
    add_panel_label(ax, 'E')

    # --- Panel F: Phase diagram (coverage advantage) ---
    ax = axes[1, 2]

    colors_cmap = ['#F7F7F7', '#D1E5F0', '#92C5DE', '#4393C3', '#2166AC', '#053061']
    cmap = LinearSegmentedColormap.from_list('dist_advantage', colors_cmap)

    E_phase = phase_data['E_values']
    ratio_phase = phase_data['ratio_values']
    cov_diff = phase_data['coverage_diff']

    im2 = ax.pcolormesh(E_phase - 0.5, ratio_phase - 0.5, cov_diff,
                        cmap=cmap, vmin=0, vmax=1.0, shading='auto')

    E_theory = np.linspace(0, max(E_phase) + 2, 100)
    ax.plot(E_theory, E_theory, '--', color='#333333', lw=1.8,
            label='$E_{crit} = s/c$')

    cbar2 = plt.colorbar(im2, ax=ax, shrink=0.85, pad=0.02)
    cbar2.set_label('Coverage advantage', fontsize=8)
    cbar2.set_ticks([0, 0.5, 1.0])
    cbar2.set_ticklabels(['None', '', 'Full'])

    ax.set_xlabel('Environmental complexity ($E$)')
    ax.set_ylabel('Benefit-to-cost ratio ($s/c$)')
    ax.set_xlim(0.5, max(E_phase) + 0.5)
    ax.set_ylim(0.5, max(ratio_phase) + 0.5)
    ax.legend(loc='upper left', fontsize=7)

    ax.text(4, max(ratio_phase) * 0.85, 'Both strategies\nviable',
            fontsize=7, ha='center', color='#666666')
    ax.text(max(E_phase) * 0.7, max(ratio_phase) * 0.25, 'Distributed\nobligate',
            fontsize=7, ha='center', color='white')

    add_panel_label(ax, 'F')

    plt.tight_layout()
    return fig


def print_summary(ushape_results):
    """Print summary statistics."""
    print()
    print("-" * 70)
    print("RESULTS SUMMARY")
    print("-" * 70)
    print()

    f_finals = ushape_results['final_frequencies']
    thresholds = ushape_results['thresholds']
    cs_ratios = ushape_results['cs_ratios']

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
        description='Figure 3 Merged: U-Shaped Distribution + Complexity Threshold',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--genes', '-n', type=int, default=2000,
                        help='Number of genes to simulate (default: 2000)')
    parser.add_argument('--generations', '-g', type=int, default=2000,
                        help='Number of generations for U-shape (default: 2000)')
    parser.add_argument('--replicates', '-r', type=int, default=30,
                        help='Replicates per gene (default: 30)')
    parser.add_argument('--hgt-rate', type=float, default=PARAMS['h'],
                        help=f"HGT rate (default: {PARAMS['h']})")
    parser.add_argument('--loss-rate', type=float, default=PARAMS['delta'],
                        help=f"Gene loss rate (default: {PARAMS['delta']})")
    parser.add_argument('-s', '--selective-benefit', type=float, default=PARAMS['s'],
                        help=f"Selective benefit (default: {PARAMS['s']})")
    parser.add_argument('-c', '--carriage-cost', type=float, default=PARAMS['c'],
                        help=f"Carriage cost per gene (default: {PARAMS['c']})")
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
    s = args.selective_benefit
    c = args.carriage_cost

    # Adjust for quick mode
    n_genes = args.genes // 10 if args.quick else args.genes
    n_gen = args.generations // 5 if args.quick else args.generations
    n_rep = args.replicates // 5 if args.quick else args.replicates

    print_header('FIGURE 3 MERGED: U-SHAPED DISTRIBUTION + COMPLEXITY THRESHOLD', {
        'Number of genes': n_genes,
        'HGT rate (h)': h,
        'Gene loss rate (δ)': delta,
        'Selective benefit (s)': s,
        'Carriage cost (c)': c,
    })

    if args.quick:
        print("  [QUICK MODE - reduced parameters]")
        print()

    # Run U-shape simulation
    print()
    print("=" * 70)
    print("SIMULATING U-SHAPED DISTRIBUTION (PANELS A-C)")
    print("=" * 70)
    print()
    ushape_results = simulate_ushape(
        n_genes=n_genes,
        n_generations=n_gen,
        n_replicates=n_rep,
        h=h,
        delta=delta
    )

    # Run complexity threshold simulation
    print()
    print("=" * 70)
    print("SIMULATING COMPLEXITY THRESHOLD (PANELS D-F)")
    print("=" * 70)
    print()
    complexity_data = simulate_complexity(
        s=s, c=c, n_gen=n_gen, n_rep=n_rep
    )

    # Create figure
    print()
    print("  Creating merged figure...")
    fig = create_figure(ushape_results, complexity_data)

    # Save
    saved = save_figure(fig, BASENAME, output_dir, args.format)
    print()
    print("  Saved:")
    for path in saved:
        print(f"    → {path}")

    # Summary
    print_summary(ushape_results)

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
