#!/usr/bin/env python3
"""
================================================================================
FIGURE 4: ENVIRONMENTAL COMPLEXITY AND ADAPTIVE CATEGORIES
================================================================================

The KEY theoretical result: as environmental complexity E increases,
distributed pangenomes become OBLIGATE above E_crit = s/c.

Single genomes cannot maintain full coverage when E > E_crit because
the cumulative cost of carrying all genes exceeds fitness benefits.

USAGE:
    python figure4_complexity.py
    python figure4_complexity.py --quick
    python figure4_complexity.py -s 0.4 -c 0.03

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

BASENAME = 'figure4_complexity'


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def geometric_mean(fitnesses):
    """Calculate geometric mean using log transform for stability."""
    return np.exp(np.mean(np.log(np.maximum(fitnesses, 1e-10))))


# =============================================================================
# SIMULATION
# =============================================================================

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


def sim_single_genome(E, m, s, c, n_gen=500, n_rep=30):
    """
    Simulation of single-genome strategy (for validation).
    Vectorized: generates all environments at once.
    """
    required = np.random.randint(0, E, size=(n_rep, n_gen))
    has_gene = required < m
    W = np.where(has_gene, 1 + s - c * m, 1 - c * m)
    W = np.maximum(W, 0.001)
    geo_means = np.exp(np.mean(np.log(W), axis=1))
    return np.mean(geo_means), np.std(geo_means)


def sim_distributed(E, m, k, s, c, n_gen=500, n_rep=30):
    """
    Simulation of distributed pangenome strategy (for validation).
    Vectorized: generates all environments at once.
    """
    required = np.random.randint(0, E, size=(n_rep, n_gen))
    covered = required < m

    W_covered = 1 - c * k + s * k / m
    W_uncovered = 1 - c * k

    W = np.where(covered, W_covered, W_uncovered)
    W = np.maximum(W, 0.001)
    geo_means = np.exp(np.mean(np.log(W), axis=1))
    return np.mean(geo_means), np.std(geo_means)


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


def simulate(s=0.3, c=0.02, n_gen=500, n_rep=30):
    """
    Panel A: Heatmap of single-genome fitness across (m, E) space.
    Shows the fitness ridge at m=E collapsing above E_crit.

    Panel B: Phase diagram showing coverage advantage of distributed
    pangenomes across parameter space (m* discovered by optimization).
    """
    E_crit = s / c

    # Panel A: full heatmap — E from 1..30, m from 1..35
    print("  Running Panel A: fitness heatmap...")
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

    # Summary table for print_summary
    E_values = [2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 32, 48]
    single_results = []
    dist_results = []
    print("  Running optimization across E values...")
    for E in E_values:
        print(f"    E = {E}...")
        best_m_s, best_fit_s = find_best_single(E, s, c, n_gen, n_rep)
        single_results.append({'E': E, 'm': best_m_s, 'k': best_m_s, 'fitness': best_fit_s})
        best_m_d, best_k_d, best_fit_d = find_best_distributed(E, s, c, n_gen, n_rep)
        dist_results.append({'E': E, 'm': best_m_d, 'k': best_k_d, 'fitness': best_fit_d})

    # Panel B: m* vs E for multiple s/c ratios
    print()
    print("  Running Panel B: multi-line m* curves...")
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

    # Panel C: phase diagram (coverage advantage)
    print()
    print("  Running Panel C: phase diagram...")

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

    return single_results, dist_results, E_crit, heatmap_data, multiline_data, {
        'E_values': E_phase,
        'ratio_values': ratio_phase,
        'coverage_diff': coverage_diff
    }


# =============================================================================
# FIGURE CREATION
# =============================================================================

def create_figure(single_res, dist_res, E_crit, s, c,
                  heatmap_data, multiline_data, phase_data):
    """
    Create three-panel figure.

    Panel A: Heatmap — single-genome fitness as function of (m, E)
             for one s/c ratio. Shows the fitness ridge collapsing.
    Panel B: Multi-line — optimal m* vs E for several s/c ratios.
             Shows the collapse is universal.
    Panel C: Phase diagram — coverage advantage of distributed
             pangenomes across parameter space.
    """
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

    # --- Panel A: Fitness heatmap (E on x-axis, m on y-axis) ---
    ax = axes[0]

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
    add_panel_label(ax, 'A')

    # --- Panel B: Multi-line m* vs E ---
    ax = axes[1]

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
    add_panel_label(ax, 'B')

    # --- Panel C: Phase diagram (coverage advantage) ---
    ax = axes[2]

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

    add_panel_label(ax, 'C')

    plt.tight_layout()
    return fig


def print_summary(single_res, dist_res, E_crit, s, c):
    """Print summary statistics."""
    print()
    print("-" * 70)
    print("RESULTS SUMMARY")
    print("-" * 70)
    print()
    print(f"Parameters: s = {s}, c = {c}")
    print(f"Critical complexity: E_crit = s/c = {E_crit:.1f}")
    print()
    print(f"{'E':<6} {'Single m*':<12} {'Coverage':<12} {'Dist m*':<12} {'Dist k*':<10}")
    print("-" * 55)

    for sr, dr in zip(single_res, dist_res):
        cov = sr['m'] / sr['E']
        print(f"{sr['E']:<6} {sr['m']:<12} {cov:.0%}{'':<8} {dr['m']:<12} {dr['k']}")

    print()

    # Find where coverage drops
    for sr in single_res:
        cov = sr['m'] / sr['E']
        if cov < 1.0:
            print(f"Single genome loses full coverage at E = {sr['E']}")
            break

    print()
    print("Key insight: Above E_crit, single genomes cannot maintain coverage")
    print("because cumulative carriage costs exceed fitness benefits.")
    print("Distributed pangenomes become obligate in complex environments.")
    print()


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Figure 4: Environmental Complexity and Adaptive Categories',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('-s', '--selective-benefit', type=float, default=PARAMS['s'],
                        help=f"Selective benefit (default: {PARAMS['s']})")
    parser.add_argument('-c', '--carriage-cost', type=float, default=PARAMS['c'],
                        help=f"Carriage cost per gene (default: {PARAMS['c']})")
    parser.add_argument('--generations', '-g', type=int, default=500,
                        help='Number of generations (default: 500)')
    parser.add_argument('--replicates', '-r', type=int, default=50,
                        help='Number of replicates (default: 50)')
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

    print_header('FIGURE 4: ENVIRONMENTAL COMPLEXITY AND ADAPTIVE CATEGORIES', {
        'Selective benefit (s)': s,
        'Carriage cost (c)': c,
    })

    # Adjust for quick mode
    n_gen = args.generations // 5 if args.quick else args.generations
    n_rep = args.replicates // 5 if args.quick else args.replicates

    if args.quick:
        print("  [QUICK MODE - reduced parameters]")
        print()

    # Run simulation
    single_res, dist_res, E_crit, heatmap_data, multiline_data, phase_data = simulate(
        s=s, c=c, n_gen=n_gen, n_rep=n_rep
    )

    # Create figure
    print()
    print("  Creating figure...")
    fig = create_figure(single_res, dist_res, E_crit, s, c,
                        heatmap_data, multiline_data, phase_data)

    # Save
    saved = save_figure(fig, BASENAME, output_dir, args.format)
    print()
    print("  Saved:")
    for path in saved:
        print(f"    → {path}")

    # Summary
    print_summary(single_res, dist_res, E_crit, s, c)

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
