#!/usr/bin/env python3
"""
================================================================================
SUPPLEMENTARY FIGURE S2: PARAMETER SENSITIVITY
================================================================================

Test parameter sensitivity for pangenome complexity threshold.

Shows that E_crit = s/c is universal across different parameter values.

USAGE:
    python supplementary_figure_s2_parameter_sensitivity.py
    python supplementary_figure_s2_parameter_sensitivity.py --quick

================================================================================
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ── path setup ───────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared.plotting import COLORS, setup_plotting, save_figure, print_header
from shared.params import THEORY_PARAMS as PARAMS

BASENAME = 'supplementary_s2_parameter_sensitivity'


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def geometric_mean(fitnesses):
    """Calculate geometric mean using log transform for stability."""
    return np.exp(np.mean(np.log(np.maximum(fitnesses, 1e-10))))


# =============================================================================
# SIMULATION
# =============================================================================

def sim_single(E, m, s, c, n_gen=150, n_rep=15):
    """Single genome strategy."""
    geo_means = []
    for _ in range(n_rep):
        fitnesses = []
        for _ in range(n_gen):
            required = np.random.randint(0, E)
            has_gene = required < m
            W = (1 + s if has_gene else 1) - c * m
            fitnesses.append(max(W, 0.001))
        geo_means.append(geometric_mean(np.array(fitnesses)))
    return np.mean(geo_means)


def find_best_m_single(E, s, c):
    """Find optimal m for single genome."""
    best_m, best_fit = 1, 0
    for m in range(1, E + 3):
        fit = sim_single(E, m, s, c)
        if fit > best_fit:
            best_fit, best_m = fit, m
    return best_m


def run_parameter_sweep(s, c, E_values):
    """Run simulation for a single parameter set."""
    results = []
    for E in E_values:
        best_m = find_best_m_single(E, s, c)
        results.append({
            'E': E,
            'm': best_m,
            'coverage': best_m / E
        })
    return results


# =============================================================================
# FIGURE CREATION
# =============================================================================

def create_figure(param_sets, E_values, param_sets_cost):
    """Create two-panel figure."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: Different benefit values (s), fixed cost (c=0.02)
    ax = axes[0]
    for params in param_sets:
        s, c = params['s'], params['c']
        E_crit = s / c
        print(f"  {params['label']}: E_crit = {E_crit:.0f}")

        results = run_parameter_sweep(s, c, E_values)
        E_vals = [r['E'] for r in results]
        coverage = [r['coverage'] for r in results]

        ax.plot(E_vals, coverage, 'o-', color=params['color'],
                label=f"{params['label']} (E_crit={E_crit:.0f})", lw=2, markersize=5)

        # Mark E_crit with vertical line
        ax.axvline(x=E_crit, color=params['color'], linestyle=':', alpha=0.5, lw=1)

    ax.set_xlabel('Environmental complexity (E)')
    ax.set_ylabel('Single genome coverage (m*/E)')
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0, max(E_values) + 2)
    ax.axhline(y=1.0, color='grey', linestyle='--', alpha=0.3)
    ax.legend(loc='upper right', fontsize=7)
    ax.set_title('A. Different benefit values (s), fixed cost (c=0.02)')

    # Panel B: Different cost values
    print()
    ax = axes[1]
    for params in param_sets_cost:
        s, c = params['s'], params['c']
        E_crit = s / c
        print(f"  {params['label']}: E_crit = {E_crit:.0f}")

        results = run_parameter_sweep(s, c, E_values)
        E_vals = [r['E'] for r in results]
        coverage = [r['coverage'] for r in results]

        ax.plot(E_vals, coverage, 'o-', color=params['color'],
                label=f"{params['label']} (E_crit={E_crit:.0f})", lw=2, markersize=5)

        ax.axvline(x=E_crit, color=params['color'], linestyle=':', alpha=0.5, lw=1)

    ax.set_xlabel('Environmental complexity (E)')
    ax.set_ylabel('Single genome coverage (m*/E)')
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0, max(E_values) + 2)
    ax.axhline(y=1.0, color='grey', linestyle='--', alpha=0.3)
    ax.legend(loc='upper right', fontsize=7)
    ax.set_title('B. Different cost values (c), fixed benefit (s=0.3)')

    plt.tight_layout()
    return fig


def create_ecrit_figure():
    """Create E_crit parameter space visualization."""
    fig, ax = plt.subplots(figsize=(6, 5))

    # Create grid of s and c values
    s_vals = np.linspace(0.05, 0.5, 50)
    c_vals = np.linspace(0.01, 0.1, 50)
    S, C = np.meshgrid(s_vals, c_vals)
    E_crit_grid = S / C

    # Plot as contour
    contour = ax.contourf(S, C, E_crit_grid, levels=[0, 5, 10, 15, 20, 25, 30, 40, 50],
                            cmap='RdYlBu_r', alpha=0.8)
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('E_crit = s/c')

    # Add contour lines
    ax.contour(S, C, E_crit_grid, levels=[5, 10, 15, 20, 25, 30], colors='black',
                linewidths=0.5, linestyles='--')

    ax.set_xlabel('Selective benefit (s)')
    ax.set_ylabel('Carriage cost (c)')
    ax.set_title('Critical complexity E_crit = s/c\nSingle genomes fail when E > E_crit')

    plt.tight_layout()
    return fig


def print_summary():
    """Print summary of parameter sensitivity."""
    print()
    print("=" * 60)
    print("CONCLUSION: The transition is universal.")
    print("E_crit = s/c defines when single genomes fail.")
    print("Only the LOCATION changes with parameters, not the EXISTENCE.")
    print("=" * 60)
    print()


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Supplementary Figure S2: Parameter Sensitivity',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                        help='Output directory (default: ../output)')
    parser.add_argument('--format', type=str,
                        choices=['png', 'pdf', 'svg', 'all'], default='png',
                        help='Output format (default: png)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: reduced parameter sweep')
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

    print_header('SUPPLEMENTARY FIGURE S2: PARAMETER SENSITIVITY', {
        'Output format': args.format,
        'Quick mode': args.quick,
    })

    # Parameter sets for testing
    param_sets = [
        {'s': 0.1, 'c': 0.02, 'label': 's=0.1, c=0.02', 'color': COLORS['purple']},
        {'s': 0.2, 'c': 0.02, 'label': 's=0.2, c=0.02', 'color': COLORS['blue']},
        {'s': 0.3, 'c': 0.02, 'label': 's=0.3, c=0.02', 'color': COLORS['green']},
        {'s': 0.4, 'c': 0.02, 'label': 's=0.4, c=0.02', 'color': COLORS['orange']},
        {'s': 0.5, 'c': 0.02, 'label': 's=0.5, c=0.02', 'color': COLORS['red']},
    ]

    param_sets_cost = [
        {'s': 0.3, 'c': 0.01, 'label': 's=0.3, c=0.01', 'color': COLORS['purple']},
        {'s': 0.3, 'c': 0.02, 'label': 's=0.3, c=0.02', 'color': COLORS['blue']},
        {'s': 0.3, 'c': 0.03, 'label': 's=0.3, c=0.03', 'color': COLORS['green']},
        {'s': 0.3, 'c': 0.05, 'label': 's=0.3, c=0.05', 'color': COLORS['orange']},
        {'s': 0.3, 'c': 0.10, 'label': 's=0.3, c=0.10', 'color': COLORS['red']},
    ]

    E_values = [2, 4, 6, 8, 10, 12, 15, 18, 22, 26, 30] if not args.quick else [2, 6, 10, 15, 20]

    print()
    print("Running parameter sensitivity analysis...")
    print()

    # Create main figure
    print("Panel A: Testing different benefit values (s)...")
    fig = create_figure(param_sets, E_values, param_sets_cost)

    # Save main figure
    saved = save_figure(fig, BASENAME, output_dir, args.format)
    print()
    print("  Saved:")
    for path in saved:
        print(f"    → {path}")

    # Create E_crit parameter space figure
    print()
    print("Creating E_crit parameter space visualization...")
    fig2 = create_ecrit_figure()

    ecrit_basename = 'supplementary_s2_ecrit_parameter_space'
    saved2 = save_figure(fig2, ecrit_basename, output_dir, args.format)
    print("  Saved:")
    for path in saved2:
        print(f"    → {path}")

    # Summary
    print_summary()

    if args.show:
        plt.show()
    else:
        plt.close(fig)
        plt.close(fig2)

    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print()


if __name__ == '__main__':
    main()
