#!/usr/bin/env python3
"""
================================================================================
SUPPLEMENTARY FIGURE S11: Sensitivity to Isolation-Source Mislabelling
================================================================================

Tests robustness of the V-retention R² result to label noise.

Two scenarios:
  A-C: Blood→Feces relabelling (ExPEC critique)
  B-D: Symmetric relabelling (all directions)

Top row (A-B):  R² of Cramér's V vs retention
Bottom row (C-D): Number of niche genes surviving classification

USAGE:
    python supplementary_figure_s11_contamination.py --output-dir output
    python supplementary_figure_s11_contamination.py --n-reps 50  # more replicates

================================================================================
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import linregress

from gene_classification import (
    load_classification_data, per_gene_chi_squared, compute_fdr, classify_genes
)
from niche_insurance_analysis import compute_niche_away_frequencies
from shared.plotting import COLORS, setup_plotting, save_figure, add_panel_label

# Contamination fractions to test
FRACS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]

# Reference R² values from simulations (SI S9)
R2_MIGRATION = 0.41
R2_BETHEDGING = 0.07


def run_one_replicate(data, corrupted_source):
    """Re-run classification + retention with corrupted labels."""
    cdata = dict(data)
    cdata['isolation_source'] = corrupted_source

    result = per_gene_chi_squared(cdata, phylogroup='B2')
    q = compute_fdr(result['p_values'])
    v = result['cramers_v']
    labels = classify_genes(q, v)

    n_niche = (labels == 'niche').sum()
    if n_niche < 10:
        return None

    away = compute_niche_away_frequencies(cdata, phylogroup='B2')
    ret = away['retention']
    v_away = away['cramers_v']
    slope, _, r_val, _, _ = linregress(v_away, ret)

    return {'r2': r_val ** 2, 'slope': slope, 'n_niche': n_niche}


def run_scenario(data, corrupt_fn, n_reps, rng):
    """Run all contamination fractions for one scenario."""
    r2_mean, r2_lo, r2_hi = [], [], []
    niche_mean, niche_lo, niche_hi = [], [], []

    for frac in FRACS:
        r2s, niches = [], []
        for _ in range(n_reps):
            corrupted = corrupt_fn(data, frac, rng)
            res = run_one_replicate(data, corrupted)
            if res:
                r2s.append(res['r2'])
                niches.append(res['n_niche'])
        if len(r2s) == 0:
            r2_mean.append(np.nan)
            r2_lo.append(np.nan)
            r2_hi.append(np.nan)
            niche_mean.append(0)
            niche_lo.append(0)
            niche_hi.append(0)
            print(f'    {frac:>5.0%}: no valid replicates (too few niche genes)')
        else:
            r2_mean.append(np.mean(r2s))
            r2_lo.append(np.percentile(r2s, 5))
            r2_hi.append(np.percentile(r2s, 95))
            niche_mean.append(np.mean(niches))
            niche_lo.append(np.percentile(niches, 5))
            niche_hi.append(np.percentile(niches, 95))
            print(f'    {frac:>5.0%}: R²={np.mean(r2s):.4f}, niche={np.mean(niches):.0f}')

    return (r2_mean, r2_lo, r2_hi, niche_mean, niche_lo, niche_hi)


def make_blood_to_feces_fn(b2_blood_idx):
    """Return a corruption function for blood→feces relabelling."""
    n_b2_blood = len(b2_blood_idx)

    def corrupt(data, frac, rng):
        corrupted = data['isolation_source'].copy()
        n_flip = int(frac * n_b2_blood)
        if n_flip > 0:
            flip_idx = rng.choice(b2_blood_idx, size=n_flip, replace=False)
            corrupted[flip_idx] = 'Feces'
        return corrupted

    return corrupt


def make_symmetric_fn(source_idx, sources):
    """Return a corruption function for symmetric all-direction relabelling."""
    def corrupt(data, frac, rng):
        corrupted = data['isolation_source'].copy()
        for s in sources:
            idx = source_idx[s]
            n_flip = int(frac * len(idx))
            if n_flip > 0:
                other = [x for x in sources if x != s]
                flip_idx = rng.choice(idx, size=n_flip, replace=False)
                corrupted[flip_idx] = rng.choice(other, size=n_flip)
        return corrupted

    return corrupt


def plot_figure(res_bf, res_sym, baseline_niche):
    """Create 2x2 figure."""
    setup_plotting()
    fig, axes = plt.subplots(2, 2, figsize=(7, 6))
    fracs_pct = [f * 100 for f in FRACS]

    blue = COLORS.get('blue', '#4477AA')
    red = COLORS.get('red', '#EE6677')
    green = COLORS.get('green', '#228833')
    grey = '#888888'

    scenarios = [
        (0, 'Blood \u2192 Feces', res_bf),
        (1, 'Symmetric (all directions)', res_sym),
    ]

    for col, title, (r2m, r2lo, r2hi, nm, nlo, nhi) in scenarios:
        # Top row: R²
        ax = axes[0, col]
        ax.fill_between(fracs_pct, r2lo, r2hi, alpha=0.15, color=blue)
        ax.plot(fracs_pct, r2m, 'o-', color=blue, linewidth=1.5,
                markersize=4, zorder=5)
        ax.axhline(R2_MIGRATION, color=red, linestyle='--', linewidth=1.2,
                   label='Migration prediction')
        ax.axhline(R2_BETHEDGING, color=green, linestyle='--', linewidth=1.2,
                   label='Bet-hedging prediction')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_ylim(-0.02, 0.52)
        ax.set_xlim(-2, 62)
        if col == 0:
            ax.set_ylabel('R\u00B2 (V vs retention)', fontsize=9)
            ax.legend(fontsize=7, loc='upper right')
        ax.tick_params(labelsize=8)

        # Bottom row: niche gene count
        ax2 = axes[1, col]
        ax2.fill_between(fracs_pct, nlo, nhi, alpha=0.15, color=blue)
        ax2.plot(fracs_pct, nm, 'o-', color=blue, linewidth=1.5,
                 markersize=4, zorder=5)
        ax2.axhline(baseline_niche, color=grey, linestyle=':', linewidth=1,
                    label=f'Baseline ({baseline_niche:,})')
        ax2.set_xlabel('Label noise (%)', fontsize=9)
        if col == 0:
            ax2.set_ylabel('Niche genes classified', fontsize=9)
            ax2.legend(fontsize=7, loc='lower left')
        ax2.set_xlim(-2, 62)
        ax2.set_ylim(0, baseline_niche * 1.15)
        ax2.tick_params(labelsize=8)

    for i, label in enumerate(['A', 'B', 'C', 'D']):
        add_panel_label(axes.flat[i], label)

    fig.tight_layout(h_pad=2.0, w_pad=2.0)
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Figure S11: Contamination sensitivity analysis')
    parser.add_argument('--output-dir', default='.', help='Output directory')
    parser.add_argument('--format', default='png')
    parser.add_argument('--n-reps', type=int, default=20,
                        help='Replicates per contamination level (default: 20)')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    print()
    print('=' * 70)
    print('SUPPLEMENTARY FIGURE S11: Contamination Sensitivity')
    print('=' * 70)
    print()

    print('[1/4] Loading data...')
    data = load_classification_data()

    sources = ['Blood', 'Feces', 'Urine']
    b2_mask = data['phylogroup'] == 'B2'
    source_idx = {}
    for s in sources:
        source_idx[s] = np.where(b2_mask & (data['isolation_source'] == s))[0]
        print(f'  B2 {s}: {len(source_idx[s])}')

    rng = np.random.default_rng(42)

    # Baseline niche count
    result = per_gene_chi_squared(data, phylogroup='B2')
    q = compute_fdr(result['p_values'])
    v = result['cramers_v']
    labels = classify_genes(q, v)
    baseline_niche = int((labels == 'niche').sum())
    print(f'  Baseline niche genes: {baseline_niche}')

    print(f'\n[2/4] Blood \u2192 Feces scenario ({args.n_reps} reps)...')
    blood_fn = make_blood_to_feces_fn(source_idx['Blood'])
    res_bf = run_scenario(data, blood_fn, args.n_reps, rng)

    print(f'\n[3/4] Symmetric scenario ({args.n_reps} reps)...')
    sym_fn = make_symmetric_fn(source_idx, sources)
    res_sym = run_scenario(data, sym_fn, args.n_reps, rng)

    print('\n[4/4] Generating figure...')
    fig = plot_figure(res_bf, res_sym, baseline_niche)

    save_figure(fig, 'supplementary_s11_contamination',
                output_dir=args.output_dir, fmt=args.format)

    if args.show:
        plt.show()

    print('\nDone.')


if __name__ == '__main__':
    main()
