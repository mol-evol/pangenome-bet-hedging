#!/usr/bin/env python3
"""
================================================================================
SUPPLEMENTARY FIGURE S3: HGT RATE OPTIMISATION
================================================================================

Two-gene model showing that the optimal HGT rate peaks at intermediate
environmental switching rates (autocorrelation ρ).

Panel A: Geometric mean fitness vs HGT rate for different ρ values.
Panel B: Optimal HGT rate vs environmental switching rate (1-ρ).

Matches Simulation 6 in SI S2:
  - 40 replicates × 12,000 generations
  - Two genes: gene A beneficial in env A, gene B in env B
  - Environmental autocorrelation ρ varied from 0.5 to 0.99
  - HGT rate varied from 0 to 0.1 per generation
  - HGT from external reservoir at fixed frequency (0.4)

USAGE:
    python supplementary_figure_s3_hgt_optimization.py
    python supplementary_figure_s3_hgt_optimization.py --quick

================================================================================
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared.plotting import COLORS, setup_plotting, save_figure, add_panel_label

BASENAME = 'supplementary_s3_hgt_optimization'


# =============================================================================
# SIMULATION
# =============================================================================

def simulate_two_gene_hgt(s, c, rho, h_rate, n_gen, n_reps, reservoir_freq=0.4):
    """
    Simulate a two-gene model with autocorrelated environments.
    Vectorized: all replicates run in parallel via numpy arrays.

    Each generation:
      1. Environment switches with probability (1-ρ)
      2. Selection: gene matching environment gives benefit s,
         non-matching gene has cost c
      3. HGT: gene frequency moves toward reservoir at rate h

    Returns geometric mean fitness across replicates.
    """
    # All replicates in parallel — shape (n_reps,)
    freq_a = np.full(n_reps, reservoir_freq)
    freq_b = np.full(n_reps, reservoir_freq)
    env = np.zeros(n_reps, dtype=int)  # 0 = env A, 1 = env B
    log_fitness_sum = np.zeros(n_reps)

    for _ in range(n_gen):
        # Environment switching (Markov with autocorrelation ρ)
        switches = np.random.random(n_reps) > rho
        env = np.where(switches, 1 - env, env)

        # Masks for which replicates are in env A vs B
        in_a = (env == 0)
        in_b = ~in_a

        # Population fitness
        w = np.ones(n_reps)
        w[in_a] = (1 + s * freq_a[in_a]) * (1 - c * freq_b[in_a])
        w[in_b] = (1 + s * freq_b[in_b]) * (1 - c * freq_a[in_b])
        log_fitness_sum += np.log(np.maximum(w, 1e-15))

        # Selection on gene frequencies
        new_freq_a = freq_a.copy()
        new_freq_b = freq_b.copy()

        # Env A: gene A selected for, gene B selected against
        new_freq_a[in_a] = freq_a[in_a] * (1 + s) / (1 + s * freq_a[in_a])
        new_freq_b[in_a] = freq_b[in_a] * (1 - c) / np.maximum(1 - c * freq_b[in_a], 1e-15)

        # Env B: gene B selected for, gene A selected against
        new_freq_b[in_b] = freq_b[in_b] * (1 + s) / (1 + s * freq_b[in_b])
        new_freq_a[in_b] = freq_a[in_b] * (1 - c) / np.maximum(1 - c * freq_a[in_b], 1e-15)

        freq_a = new_freq_a
        freq_b = new_freq_b

        # HGT: move toward reservoir frequency
        freq_a += h_rate * (reservoir_freq - freq_a)
        freq_b += h_rate * (reservoir_freq - freq_b)

        # Clamp
        np.clip(freq_a, 0, 1, out=freq_a)
        np.clip(freq_b, 0, 1, out=freq_b)

    geo_means = np.exp(log_fitness_sum / n_gen)
    return np.mean(geo_means), np.std(geo_means)


def run_simulation(quick=False):
    """Run the full parameter sweep."""
    if quick:
        n_gen = 3000
        n_reps = 20
        n_h_values = 30
        rho_values = [0.5, 0.8, 0.9, 0.95, 0.98, 0.99]
    else:
        n_gen = 12000
        n_reps = 40
        n_h_values = 50
        rho_values = [0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99]

    s = 0.10
    c = 0.10
    h_values = np.linspace(0, 0.1, n_h_values)

    print(f"  Parameters: s={s}, c={c}")
    print(f"  Generations: {n_gen:,}, Replicates: {n_reps}")
    print(f"  HGT rates: {n_h_values} values from 0 to 0.1")
    print(f"  ρ values: {rho_values}")
    print()

    results = {}
    total = len(rho_values) * n_h_values
    done = 0

    for rho in rho_values:
        results[rho] = {'h_values': h_values, 'means': [], 'stds': []}
        for h in h_values:
            mean, std = simulate_two_gene_hgt(s, c, rho, h, n_gen, n_reps)
            results[rho]['means'].append(mean)
            results[rho]['stds'].append(std)
            done += 1
            if done % max(1, total // 20) == 0:
                print(f'\r  Progress: {100*done/total:.0f}%', end='', flush=True)

        results[rho]['means'] = np.array(results[rho]['means'])
        results[rho]['stds'] = np.array(results[rho]['stds'])

    print('\r  Progress: 100%     ')
    return results, h_values, rho_values


# =============================================================================
# PLOTTING
# =============================================================================

def plot_figure(results, h_values, rho_values, output_dir='.', fmt='png'):
    """Create the two-panel figure."""
    setup_plotting()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Color map for ρ values
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=min(rho_values), vmax=max(rho_values))

    # --- Panel A: Fitness vs HGT rate for each ρ ---
    ax = axes[0]
    for rho in rho_values:
        color = cmap(norm(rho))
        ax.plot(h_values, results[rho]['means'],
                color=color, linewidth=1.5,
                label=f'ρ = {rho}')

    ax.set_xlabel('HGT rate (h)')
    ax.set_ylabel('Geometric mean fitness')
    ax.legend(fontsize=7, loc='lower right', framealpha=0.9)
    add_panel_label(ax, 'A')

    # --- Panel B: Optimal HGT rate vs switching rate ---
    ax = axes[1]
    switching_rates = []
    optimal_h = []

    for rho in rho_values:
        best_idx = np.argmax(results[rho]['means'])
        switching_rates.append(1 - rho)
        optimal_h.append(h_values[best_idx])

    ax.plot(switching_rates, optimal_h, 'o-',
            color=COLORS['blue'], linewidth=2, markersize=6)
    ax.set_xlabel('Environmental switching rate (1 − ρ)')
    ax.set_ylabel('Optimal HGT rate')
    ax.set_xscale('log')

    add_panel_label(ax, 'B')

    fig.tight_layout()
    save_figure(fig, BASENAME, output_dir=output_dir, fmt=fmt)
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Supplementary Figure S3: HGT Rate Optimisation')
    parser.add_argument('--output-dir', default='.', help='Output directory')
    parser.add_argument('--format', default='png', choices=['png', 'pdf', 'svg', 'all'])
    parser.add_argument('--quick', action='store_true', help='Quick run with fewer replicates')
    args = parser.parse_args()

    print("=" * 60)
    print("SUPPLEMENTARY FIGURE S3: HGT RATE OPTIMISATION")
    print("=" * 60)
    print()

    results, h_values, rho_values = run_simulation(quick=args.quick)
    plot_figure(results, h_values, rho_values,
                output_dir=args.output_dir, fmt=args.format)

    # Print summary
    print("\n  Summary of optimal HGT rates:")
    for rho in rho_values:
        best_idx = np.argmax(results[rho]['means'])
        print(f"    ρ = {rho:.2f}: optimal h = {h_values[best_idx]:.4f}, "
              f"fitness = {results[rho]['means'][best_idx]:.6f}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
