#!/usr/bin/env python3
"""
================================================================================
SIMULATION TEST: BET-HEDGING vs MIGRATION MODELS
================================================================================

PURPOSE
-------
The niche insurance analysis found that niche genes retain ~63% of their home
frequency in away niches, with symmetric retention across home niches
(epsilon-squared = 0.037) and a near-flat V-retention slope (R² = 0.076).

This simulation tests whether these patterns better match a migration model
or a bet-hedging model by generating datasets under each and comparing
summary statistics to the real data.

MODELS
------
Model 1 — Migration-selection balance:
  Away freq = m_{H->A} / (m_{H->A} + k * V) * home_freq
  where m_{H->A} are asymmetric migration rates and k*V is selection.
  Predicts: asymmetric retention, V-dependent depletion.

Model 2 — Bet-hedging (insurance carriage):
  Away freq = retention * home_freq
  where retention ~ Beta(alpha, beta), independent of V and home niche.
  Predicts: symmetric retention, V-independent carriage.

Both models use REAL niche gene parameters (home frequencies, V values,
home niche assignments) as input. Only away-frequency generation differs.

FIGURE
------
4-panel (2 x 2):
  A: epsilon-squared distributions (symmetry test)
  B: R-squared distributions (slope test)
  C: Joint epsilon-squared vs R-squared scatter
  D: Blood-home vs Feces-home median retention (symmetry visual)

USAGE
-----
  python niche_simulation.py [--n-sims N] [--output-dir DIR] [--show]

================================================================================
"""

import os
import sys
import argparse
import numpy as np
from scipy.stats import kruskal, linregress

# Path setup
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.join(_SCRIPT_DIR, '..')
_PROJECT_ROOT = os.path.join(_CODE_DIR, '..')
sys.path.insert(0, _CODE_DIR)       # for shared.*
sys.path.insert(0, _SCRIPT_DIR)     # for sibling empirical scripts

from shared.plotting import (setup_plotting, save_figure, add_panel_label,
                              print_header, COLORS)
import matplotlib.pyplot as plt

from gene_classification import load_classification_data
from niche_insurance_analysis import compute_niche_away_frequencies


# ==============================================================================
# Constants
# ==============================================================================

SOURCES = ['Blood', 'Feces', 'Urine']
SAMPLE_SIZES = {'Blood': 1066, 'Feces': 273, 'Urine': 366}
NS = np.array([SAMPLE_SIZES[s] for s in SOURCES], dtype=float)
TOTAL_N = NS.sum()

# Migration rates — biologically motivated, asymmetric
#   Feces->Blood: common (bacteremia from gut translocation)
#   Blood->Feces: rare
#   Feces->Urine: very common (UPEC originate from gut)
#   Urine->Feces: rare
#   Blood->Urine: moderate (ExPEC)
#   Urine->Blood: moderate (urosepsis)
MIGRATION_RATES = {
    ('Blood', 'Feces'): 0.04,
    ('Blood', 'Urine'): 0.15,
    ('Feces', 'Blood'): 0.25,
    ('Feces', 'Urine'): 0.30,
    ('Urine', 'Blood'): 0.18,
    ('Urine', 'Feces'): 0.06,
}
SELECTION_SCALE = 0.40   # k: selection coefficient = k * V


# ==============================================================================
# Real data
# ==============================================================================

def load_real_data():
    """Load real niche gene parameters and compute summary stats."""
    data = load_classification_data()
    ad = compute_niche_away_frequencies(data)

    home_freqs = ad['home_freq']
    home_niches = ad['home_source']
    v_values = ad['cramers_v']
    retention = ad['retention']

    # Summary stats from real data
    groups = [retention[home_niches == s] for s in SOURCES
              if (home_niches == s).sum() > 0]
    H, _ = kruskal(*groups)
    eps_sq = H / (len(retention) - 1)

    valid = home_freqs > 0
    slope, _, r_val, _, _ = linregress(v_values[valid], retention[valid])
    r_sq = r_val ** 2

    niche_meds = {}
    for s in SOURCES:
        m = home_niches == s
        if m.sum() > 0:
            niche_meds[s] = np.median(retention[m])

    return {
        'home_freqs': home_freqs,
        'home_niches': home_niches,
        'v_values': v_values,
        'retention': retention,
        'median_retention': np.median(retention),
        'epsilon_sq': eps_sq,
        'r_sq': r_sq,
        'slope': slope,
        'niche_medians': niche_meds,
    }


# ==============================================================================
# Model generators
# ==============================================================================

def generate_migration_freqs(home_freqs, home_niches, v_values):
    """Model 1: migration-selection balance (vectorised)."""
    n = len(home_freqs)
    freqs = {s: np.zeros(n) for s in SOURCES}

    for away in SOURCES:
        for home in SOURCES:
            mask = home_niches == home
            if not mask.any():
                continue
            if away == home:
                freqs[away][mask] = home_freqs[mask]
            else:
                m = MIGRATION_RATES[(home, away)]
                s = SELECTION_SCALE * v_values[mask]
                ret = m / (m + s)
                freqs[away][mask] = ret * home_freqs[mask]
    return freqs


def generate_bethedging_freqs(home_freqs, home_niches, alpha, beta_param):
    """Model 2: bet-hedging / insurance (vectorised).

    Each gene gets an independent retention draw for each away niche,
    all from the same Beta distribution, independent of V and home niche.
    """
    n = len(home_freqs)
    freqs = {s: np.zeros(n) for s in SOURCES}

    # Independent retention for each gene x away-niche
    all_ret = np.random.beta(alpha, beta_param, size=(n, len(SOURCES)))

    for j, src in enumerate(SOURCES):
        home_mask = home_niches == src
        freqs[src][home_mask] = home_freqs[home_mask]
        away_mask = ~home_mask
        freqs[src][away_mask] = all_ret[away_mask, j] * home_freqs[away_mask]
    return freqs


# ==============================================================================
# Simulation engine
# ==============================================================================

def simulate_one(true_freqs):
    """Sample PA matrix, compute summary stats.

    Returns dict with: median_retention, epsilon_sq, r_sq, slope,
                        blood_median, feces_median
    """
    n_genes = len(true_freqs[SOURCES[0]])

    # Sample observed counts (Binomial)
    count_matrix = np.column_stack([
        np.random.binomial(int(NS[j]), np.clip(true_freqs[s], 0, 1))
        for j, s in enumerate(SOURCES)
    ])
    freq_matrix = count_matrix / NS[np.newaxis, :]

    # Home niche = highest observed frequency
    home_idx = freq_matrix.argmax(axis=1)
    home_freq = freq_matrix[np.arange(n_genes), home_idx]
    home_source = np.array([SOURCES[i] for i in home_idx])

    # Max away frequency
    away_matrix = freq_matrix.copy()
    away_matrix[np.arange(n_genes), home_idx] = -np.inf
    max_away = away_matrix.max(axis=1)

    # Retention
    retention = np.where(home_freq > 0, max_away / home_freq, 0.0)

    # Vectorised Cramer's V
    row_present = count_matrix.sum(axis=1).astype(float)
    chi2 = np.zeros(n_genes)
    for j in range(3):
        e_p = row_present * NS[j] / TOTAL_N
        e_a = (TOTAL_N - row_present) * NS[j] / TOTAL_N
        o_p = count_matrix[:, j].astype(float)
        o_a = NS[j] - o_p
        safe = e_p > 0
        chi2[safe] += (o_p[safe] - e_p[safe])**2 / e_p[safe]
        safe = e_a > 0
        chi2[safe] += (o_a[safe] - e_a[safe])**2 / e_a[safe]
    v = np.sqrt(chi2 / TOTAL_N)

    # --- Summary stats ---
    med_ret = np.median(retention)

    # Symmetry (epsilon-squared)
    groups = [retention[home_source == s] for s in SOURCES
              if (home_source == s).sum() > 0]
    if len(groups) >= 2:
        H, _ = kruskal(*groups)
        eps_sq = H / (n_genes - 1)
    else:
        eps_sq = 0.0

    # Slope R-squared
    valid = (home_freq > 0) & (v > 0)
    if valid.sum() > 10:
        sl, _, r_val, _, _ = linregress(v[valid], retention[valid])
        r_sq = r_val ** 2
    else:
        r_sq = 0.0
        sl = 0.0

    # Per-niche medians
    bm = home_source == 'Blood'
    fm = home_source == 'Feces'
    blood_med = np.median(retention[bm]) if bm.sum() > 0 else np.nan
    feces_med = np.median(retention[fm]) if fm.sum() > 0 else np.nan

    return {
        'median_retention': med_ret,
        'epsilon_sq': eps_sq,
        'r_sq': r_sq,
        'slope': sl,
        'blood_median': blood_med,
        'feces_median': feces_med,
    }


def run_simulations(gen_func, gen_kwargs, n_sims):
    """Run n_sims simulations, return arrays of summary stats."""
    keys = ['median_retention', 'epsilon_sq', 'r_sq', 'slope',
            'blood_median', 'feces_median']
    results = {k: [] for k in keys}

    for _ in range(n_sims):
        freqs = gen_func(**gen_kwargs)
        stats = simulate_one(freqs)
        for k in keys:
            results[k].append(stats[k])

    return {k: np.array(v) for k, v in results.items()}


# ==============================================================================
# Plotting
# ==============================================================================

def plot_comparison(real, mig, bh, output_dir, fmt):
    """4-panel comparison figure."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # --- Panel A: epsilon-squared ---
    ax = axes[0, 0]
    ax.hist(mig['epsilon_sq'], bins=30, alpha=0.6, color=COLORS['red'],
            label='Migration', density=True)
    ax.hist(bh['epsilon_sq'], bins=30, alpha=0.6, color=COLORS['blue'],
            label='Bet-hedging', density=True)
    ax.axvline(real['epsilon_sq'], color='black', lw=2, ls='--',
               label=f'Real (ε² = {real["epsilon_sq"]:.3f})')
    ax.set_xlabel('ε² (symmetry test)')
    ax.set_ylabel('Density')
    ax.set_title('Symmetry of retention across home niches')
    ax.legend(fontsize=7)
    add_panel_label(ax, 'A')

    # --- Panel B: R-squared ---
    ax = axes[0, 1]
    ax.hist(mig['r_sq'], bins=30, alpha=0.6, color=COLORS['red'],
            label='Migration', density=True)
    ax.hist(bh['r_sq'], bins=30, alpha=0.6, color=COLORS['blue'],
            label='Bet-hedging', density=True)
    ax.axvline(real['r_sq'], color='black', lw=2, ls='--',
               label=f'Real (R² = {real["r_sq"]:.3f})')
    ax.set_xlabel('R² (V–retention slope)')
    ax.set_ylabel('Density')
    ax.set_title('Niche effect size vs retention')
    ax.legend(fontsize=7)
    add_panel_label(ax, 'B')

    # --- Panel C: joint scatter ---
    ax = axes[1, 0]
    ax.scatter(mig['epsilon_sq'], mig['r_sq'], s=12, alpha=0.4,
               color=COLORS['red'], label='Migration')
    ax.scatter(bh['epsilon_sq'], bh['r_sq'], s=12, alpha=0.4,
               color=COLORS['blue'], label='Bet-hedging')
    ax.scatter(real['epsilon_sq'], real['r_sq'], s=250, marker='*',
               color='black', zorder=10, label='Real data')
    ax.set_xlabel('ε² (symmetry)')
    ax.set_ylabel('R² (slope)')
    ax.set_title('Joint comparison')
    ax.legend(fontsize=7)
    add_panel_label(ax, 'C')

    # --- Panel D: Blood-home vs Feces-home median retention ---
    ax = axes[1, 1]
    ax.scatter(mig['blood_median'], mig['feces_median'], s=12, alpha=0.4,
               color=COLORS['red'], label='Migration')
    ax.scatter(bh['blood_median'], bh['feces_median'], s=12, alpha=0.4,
               color=COLORS['blue'], label='Bet-hedging')
    ax.scatter(real['niche_medians']['Blood'], real['niche_medians']['Feces'],
               s=250, marker='*', color='black', zorder=10, label='Real data')
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], '--', color=COLORS['grey'], alpha=0.5,
            label='Symmetric')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('Blood-home median retention')
    ax.set_ylabel('Feces-home median retention')
    ax.set_title('Retention symmetry: Blood vs Feces home')
    ax.legend(fontsize=7)
    add_panel_label(ax, 'D')

    fig.tight_layout()
    save_figure(fig, 'niche_simulation', output_dir=output_dir, fmt=fmt)
    plt.close(fig)


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Simulation: bet-hedging vs migration models')
    parser.add_argument('--n-sims', type=int, default=500)
    parser.add_argument('--output-dir', default='.', help='Output directory')
    parser.add_argument('--format', default='png')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    setup_plotting()
    print_header('Simulation: Bet-Hedging vs Migration',
                 {'Models': '(1) migration-selection balance, '
                            '(2) insurance carriage',
                  'Simulations': f'{args.n_sims} per model'})

    # --- Load real data ---
    print('\n[1/4] Loading real data...')
    real = load_real_data()
    n_genes = len(real['home_freqs'])
    print(f'  {n_genes} niche genes')
    print(f'  Real ε²  = {real["epsilon_sq"]:.4f}')
    print(f'  Real R²  = {real["r_sq"]:.4f}')
    print(f'  Real median retention = {real["median_retention"]:.3f}')

    # Fit Beta for bet-hedging model (method of moments)
    mu = real['retention'].mean()
    var = real['retention'].var()
    # Use 80% of observed variance (the rest is sampling noise)
    var_bio = var * 0.80
    common = mu * (1 - mu) / var_bio - 1
    alpha = mu * common
    beta_param = (1 - mu) * common
    print(f'  Beta fit: α = {alpha:.2f}, β = {beta_param:.2f}')

    # --- Migration model ---
    print(f'\n[2/4] Simulating migration model ({args.n_sims} runs)...')
    mig = run_simulations(
        generate_migration_freqs,
        {'home_freqs': real['home_freqs'],
         'home_niches': real['home_niches'],
         'v_values': real['v_values']},
        args.n_sims)
    print(f'  Median ε² = {np.median(mig["epsilon_sq"]):.4f}')
    print(f'  Median R² = {np.median(mig["r_sq"]):.4f}')
    print(f'  Median retention = {np.median(mig["median_retention"]):.3f}')

    # --- Bet-hedging model ---
    print(f'\n[3/4] Simulating bet-hedging model ({args.n_sims} runs)...')
    bh = run_simulations(
        generate_bethedging_freqs,
        {'home_freqs': real['home_freqs'],
         'home_niches': real['home_niches'],
         'alpha': alpha,
         'beta_param': beta_param},
        args.n_sims)
    print(f'  Median ε² = {np.median(bh["epsilon_sq"]):.4f}')
    print(f'  Median R² = {np.median(bh["r_sq"]):.4f}')
    print(f'  Median retention = {np.median(bh["median_retention"]):.3f}')

    # --- Plot ---
    print('\n[4/4] Generating comparison figure...')
    plot_comparison(real, mig, bh, args.output_dir, args.format)

    # --- Summary ---
    print('\n' + '=' * 60)
    print('SIMULATION RESULTS')
    print('=' * 60)

    for stat, label in [('epsilon_sq', 'SYMMETRY (ε²)'),
                         ('r_sq', 'SLOPE (R²)')]:
        real_val = real[stat]
        m_vals = mig[stat]
        b_vals = bh[stat]
        m_lo, m_hi = np.percentile(m_vals, [5, 95])
        b_lo, b_hi = np.percentile(b_vals, [5, 95])
        m_pct = (m_vals <= real_val).mean() * 100
        b_pct = (b_vals <= real_val).mean() * 100

        print(f'\n{label}:')
        print(f'  Real data:         {real_val:.4f}')
        print(f'  Migration model:   {np.median(m_vals):.4f} '
              f'[{m_lo:.4f}, {m_hi:.4f}] 90% CI')
        print(f'  Bet-hedging model: {np.median(b_vals):.4f} '
              f'[{b_lo:.4f}, {b_hi:.4f}] 90% CI')
        print(f'  Real at {m_pct:.1f}th pct of migration model')
        print(f'  Real at {b_pct:.1f}th pct of bet-hedging model')

        if m_pct < 5 or m_pct > 95:
            print(f'  → Real OUTSIDE migration 90% CI')
        else:
            print(f'  → Real inside migration 90% CI')

        if 5 <= b_pct <= 95:
            print(f'  → Real inside bet-hedging 90% CI')
        else:
            print(f'  → Real OUTSIDE bet-hedging 90% CI')

    print(f'\nNICHE SYMMETRY (Blood-home vs Feces-home):')
    print(f'  Real:       Blood = {real["niche_medians"]["Blood"]:.3f}, '
          f'Feces = {real["niche_medians"]["Feces"]:.3f}')
    print(f'  Migration:  Blood = {np.median(mig["blood_median"]):.3f}, '
          f'Feces = {np.median(mig["feces_median"]):.3f}')
    print(f'  Bet-hedging:Blood = {np.median(bh["blood_median"]):.3f}, '
          f'Feces = {np.median(bh["feces_median"]):.3f}')

    # Migration model parameters used
    print(f'\n--- Migration model parameters ---')
    for (h, a), rate in sorted(MIGRATION_RATES.items()):
        print(f'  m({h} → {a}) = {rate}')
    print(f'  k (selection scale) = {SELECTION_SCALE}')

    print('\n' + '=' * 60)


if __name__ == '__main__':
    main()
