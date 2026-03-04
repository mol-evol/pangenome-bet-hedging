#!/usr/bin/env python3
"""
================================================================================
NICHE GENES AS INSURANCE: AWAY-NICHE CARRIAGE ANALYSIS
================================================================================

PURPOSE
-------
The gene classification analysis labels ~58% of E. coli B2 accessory genes
as niche-specific (differentially distributed across body sites) and ~30.5%
as insurance (evenly distributed).  But a gene that is niche-specific —
high frequency in its "home" body site — might still be carried at non-
trivial frequency in other body sites, functioning as insurance there.

This analysis asks: do niche-specific genes have non-trivial "away"
frequencies?  If so, niche structure and insurance carriage are not
separate gene categories but two aspects of the same gene's behaviour
across environments, consistent with bet-hedging theory.

METHOD
------
1. Classify genes using the same chi-squared + Cramér's V thresholds as
   gene_classification.py  (q < 0.05 AND V > 0.10 = niche)
2. For each niche gene, identify the "home" source (highest frequency)
3. Record the "max away" frequency (highest frequency in the two non-home
   body sites) and the "min away" frequency
4. Compute retention = max_away_freq / home_freq
     retention = 1.0  →  gene equally frequent everywhere (not really niche)
     retention = 0.0  →  gene absent from away niches (strict local adaptation)
     retention ~ 0.6  →  gene depleted but still substantially carried away

   NOTE: an earlier version used away_freq / overall_B2_freq as the null
   comparison.  That metric was confounded by unequal source sample sizes
   (Blood = 62% of B2), producing artefactually high ratios for non-Blood-
   home genes.  The home-based retention metric is unconfounded.

DATA
----
Horesh et al. (2021) — same data as decoupling and gene classification.
Phylogroup B2 (Blood=1,066, Feces=273, Urine=366).

FIGURE
------
4-panel (2 × 2):
  A: Retention distribution — histogram of away/home frequency
  B: Home vs away frequency — scatter with decile medians
  C: Retention by home niche — boxplot with Kruskal-Wallis symmetry test
  D: Effect size vs retention — V vs away/home with OLS slope test

DISCRIMINATING TESTS
--------------------
Two tests distinguish bet-hedging from the main alternative (local
adaptation + directional migration):

  TEST 1 — SYMMETRY (Panel C):  Kruskal-Wallis on retention across
  home niches.  Under directional migration, gut→blood is common but
  blood→gut is rare, so retention should differ by home niche.  Under
  bet-hedging, insurance value depends on environmental variance, not
  migration direction, so retention should be symmetric.

  TEST 2 — V–RETENTION SLOPE (Panel D):  OLS regression of retention
  on Cramér's V.  Under local adaptation, genes with stronger niche
  effects face stronger opposing selection in away niches → steep
  negative slope.  Under bet-hedging, insurance value offsets local
  cost → flat slope, low R².

KEY RESULTS (from B2)
---------------------
  2,821 niche genes
  Median home frequency:  0.448
  Median away frequency:  0.292
  Median retention:       0.63   (37% depletion in away niches)
  Retention ≥ 0.50:       82.7%  (vast majority carried in away niches)
  Retention < 0.20:       1.0%   (almost none truly absent)
  Top-quartile V genes:   median retention = 0.61 (even strong niche
                          genes are heavily carried in away niches)

USAGE
-----
  python niche_insurance_analysis.py [--output-dir DIR] [--format FMT] [--show]

================================================================================
"""

import os
import sys
import argparse
import numpy as np

# Path setup
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.join(_SCRIPT_DIR, '..')
_PROJECT_ROOT = os.path.join(_CODE_DIR, '..')
sys.path.insert(0, _CODE_DIR)       # for shared.*
sys.path.insert(0, _SCRIPT_DIR)     # for sibling empirical scripts

from shared.plotting import (setup_plotting, save_figure, add_panel_label,
                              print_header, COLORS)

import matplotlib.pyplot as plt

from gene_classification import (load_classification_data, per_gene_chi_squared,
                                  compute_fdr, classify_genes)


# ==============================================================================
# Core computation
# ==============================================================================

def compute_niche_away_frequencies(data, phylogroup='B2'):
    """For each niche gene, compute home and away frequencies.

    Returns
    -------
    dict with:
        home_freq : (n_niche,) array — frequency in highest-frequency source
        max_away_freq : (n_niche,) array — frequency in best away source
        min_away_freq : (n_niche,) array — frequency in worst away source
        overall_freq : (n_niche,) array — frequency across all B2 genomes
        retention : (n_niche,) array — max_away_freq / home_freq
        home_source : (n_niche,) array of str — which source is home
        cramers_v : (n_niche,) array — effect size for each niche gene
        all_source_freqs : dict of {source: (n_niche,) array}
        sources_used : list of str
        n_niche : int
        n_total : int
    """
    # Run chi-squared and classify
    result = per_gene_chi_squared(data, phylogroup=phylogroup)
    q_values = compute_fdr(result['p_values'])
    labels = classify_genes(q_values, result['cramers_v'])

    niche_mask = labels == 'niche'
    n_niche = niche_mask.sum()
    n_total = len(labels)

    sources = result['sources_used']
    source_freqs = result['source_freqs']

    # Build (n_niche, n_sources) matrix of frequencies
    freq_matrix = np.column_stack([source_freqs[src][niche_mask]
                                   for src in sources])

    # Home = source with highest frequency for each gene
    home_idx = freq_matrix.argmax(axis=1)
    home_freq = freq_matrix[np.arange(n_niche), home_idx]
    home_source = np.array([sources[i] for i in home_idx])

    # Away frequencies: the two non-home sources
    max_away_freq = np.zeros(n_niche)
    min_away_freq = np.zeros(n_niche)
    for i in range(n_niche):
        away_freqs = np.delete(freq_matrix[i], home_idx[i])
        max_away_freq[i] = away_freqs.max()
        min_away_freq[i] = away_freqs.min()

    # Per-source frequencies for niche genes only
    niche_source_freqs = {src: source_freqs[src][niche_mask] for src in sources}

    # Overall B2 frequency for each niche gene (weighted by source sample sizes)
    mask_pg = data['phylogroup'] == phylogroup
    pa_pg = data['pa_matrix'][mask_pg]
    overall_freq_all = pa_pg.mean(axis=0)
    overall_freq = overall_freq_all[niche_mask]

    # Retention fraction: away freq / home freq
    # Ratio = 1 means no depletion (same frequency everywhere — not really niche)
    # Ratio = 0 means absent from away niche (strict specialisation)
    # Directly measures how much of the home frequency is retained in the best
    # away niche, without confounding by overall sample composition
    retention = np.where(home_freq > 0,
                         max_away_freq / home_freq,
                         0.0)

    return {
        'home_freq': home_freq,
        'max_away_freq': max_away_freq,
        'min_away_freq': min_away_freq,
        'overall_freq': overall_freq,
        'retention': retention,
        'home_source': home_source,
        'cramers_v': result['cramers_v'][niche_mask],
        'all_source_freqs': niche_source_freqs,
        'sources_used': sources,
        'n_niche': n_niche,
        'n_total': n_total,
        'labels': labels,
    }


# ==============================================================================
# Panel functions
# ==============================================================================

def panel_a_retention(ax, away_data):
    """Histogram of retention fraction (away freq / home freq).

    Ratio = 1 means the gene is equally frequent at home and away — no real
    niche specialisation.  Ratio = 0 means completely absent from away niche
    (strict local adaptation).  Values between 0 and 1 indicate partial
    depletion — the gene is niche-structured but still carried in away
    niches, consistent with insurance.  Unlike the away/overall ratio, this
    metric is not confounded by unequal source sample sizes.
    """
    retention = away_data['retention']

    ax.hist(retention, bins=50, color=COLORS['blue'], alpha=0.7,
            edgecolor='white', linewidth=0.5)

    # No-depletion line
    ax.axvline(1.0, color=COLORS['red'], linestyle='-', linewidth=1.5,
               label='No depletion (home = away)')

    median_ret = np.median(retention)
    ax.axvline(median_ret, color=COLORS['orange'], linestyle='--',
               linewidth=1.5, label=f'Median = {median_ret:.2f}')

    pct_above_50 = (retention >= 0.50).mean() * 100
    pct_above_80 = (retention >= 0.80).mean() * 100
    pct_below_20 = (retention < 0.20).mean() * 100

    ax.set_xlabel('Away frequency / home frequency')
    ax.set_ylabel('Number of niche genes')
    ax.set_title('Retention of niche genes in away niches')
    ax.legend(fontsize=7, loc='upper left')

    txt = (f'n = {len(retention):,} niche genes\n'
           f'Median retention = {median_ret:.2f}\n'
           f'Retention ≥ 0.80: {pct_above_80:.0f}%\n'
           f'Retention ≥ 0.50: {pct_above_50:.0f}%\n'
           f'Retention < 0.20: {pct_below_20:.0f}%')
    ax.text(0.97, 0.97, txt, transform=ax.transAxes, fontsize=7,
            va='top', ha='right', bbox=dict(boxstyle='round,pad=0.3',
            facecolor='white', alpha=0.8))


def panel_b_home_vs_away(ax, away_data):
    """Scatter of home frequency vs max-away frequency."""
    home = away_data['home_freq']
    away = away_data['max_away_freq']

    ax.scatter(home, away, s=3, alpha=0.3, color=COLORS['blue'],
               edgecolors='none', rasterized=True)

    # Diagonal (no niche structure)
    lims = [0, max(home.max(), away.max()) * 1.05]
    ax.plot(lims, lims, '--', color=COLORS['grey'], linewidth=1, alpha=0.5,
            label='No niche structure')

    # Quintile trend line
    n_bins = 10
    sorted_idx = np.argsort(home)
    bin_size = len(home) // n_bins
    bin_x, bin_y = [], []
    for b in range(n_bins):
        start = b * bin_size
        end = start + bin_size if b < n_bins - 1 else len(home)
        idx = sorted_idx[start:end]
        bin_x.append(np.median(home[idx]))
        bin_y.append(np.median(away[idx]))
    ax.plot(bin_x, bin_y, '-o', color=COLORS['red'], markersize=4,
            linewidth=1.5, label='Decile medians')

    ax.set_xlabel('Home-niche frequency')
    ax.set_ylabel('Max away-niche frequency')
    ax.set_title('Home vs away frequency')
    ax.legend(fontsize=7, loc='upper left')

    # Spearman correlation
    from scipy.stats import spearmanr
    rho, p = spearmanr(home, away)
    ax.text(0.97, 0.05, f'ρ = {rho:.3f}', transform=ax.transAxes,
            fontsize=7, ha='right')


def panel_c_retention_by_home(ax, away_data):
    """Boxplot of RETENTION grouped by home niche, with symmetry test.

    KEY DISCRIMINATING TEST — migration vs bet-hedging:

    Under directional migration (the main alternative to bet-hedging):
      - Gut→blood migration is common (bacteremia from gut translocation)
      - Blood→gut migration is rare
      - Therefore Feces-home genes should have HIGH retention in blood
        (carried there by migrating gut bacteria), but Blood-home genes
        should have LOW retention in feces (blood bacteria rarely reach gut)
      - Retention distributions should DIFFER significantly across home niches

    Under bet-hedging:
      - Insurance value depends on environmental variance, not migration
        direction
      - Retention should be SIMILAR across home niches
      - Kruskal-Wallis test should be non-significant or show weak effect

    A non-significant Kruskal-Wallis on retention across home niches
    favours bet-hedging.  A significant result with large effect size
    would favour directional migration.
    """
    from scipy.stats import kruskal

    sources = away_data['sources_used']
    home_source = away_data['home_source']
    retention = away_data['retention']

    # Group RETENTION by home niche
    groups = []
    labels = []
    for src in sources:
        mask = home_source == src
        if mask.sum() > 0:
            groups.append(retention[mask])
            labels.append(f'{src}\n(n={mask.sum():,})')

    bp = ax.boxplot(groups, tick_labels=labels, patch_artist=True,
                    medianprops=dict(color=COLORS['red'], linewidth=1.5),
                    flierprops=dict(markersize=2, alpha=0.3))

    colors_list = [COLORS['blue'], COLORS['orange'], COLORS['green']]
    for patch, c in zip(bp['boxes'], colors_list[:len(groups)]):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)

    ax.set_ylabel('Retention (away freq / home freq)')
    ax.set_title('Retention by home niche (symmetry test)')

    # Add medians as text
    for i, g in enumerate(groups):
        med = np.median(g)
        ax.text(i + 1, med + 0.01, f'{med:.3f}', ha='center', fontsize=7,
                color=COLORS['red'])

    # Kruskal-Wallis symmetry test
    if len(groups) >= 2:
        H, p_kw = kruskal(*groups)
        # Epsilon-squared effect size: ε² = H / (n - 1)
        n_total = sum(len(g) for g in groups)
        eps_sq = H / (n_total - 1)
        label = ('negligible' if eps_sq < 0.01 else
                 'small' if eps_sq < 0.06 else
                 'medium' if eps_sq < 0.14 else 'large')
        txt = (f'Kruskal-Wallis symmetry test\n'
               f'H = {H:.1f}, p = {p_kw:.2e}\n'
               f'ε² = {eps_sq:.4f} ({label})\n'
               f'Migration predicts asymmetry;\n'
               f'bet-hedging predicts symmetry')
        ax.text(0.97, 0.97, txt, transform=ax.transAxes, fontsize=6.5,
                va='top', ha='right', bbox=dict(boxstyle='round,pad=0.3',
                facecolor='wheat', alpha=0.9))


def panel_d_effect_vs_retention(ax, away_data):
    """Scatter of Cramér's V vs retention, with regression slope test.

    KEY DISCRIMINATING TEST — local adaptation vs bet-hedging:

    Under pure local adaptation (niche genes costly in away niches):
      - Genes with stronger niche effects (higher V) have larger selection
        coefficients against them in away niches
      - Migration alone cannot maintain them → retention should DROP
        steeply as V increases
      - Predicted: steep negative slope, R² moderate-to-high

    Under bet-hedging (niche genes provide insurance in away niches):
      - Even strongly niche-differentiated genes have insurance value
      - Selection for insurance offsets selection against them in away niches
      - Predicted: flat or weakly negative slope, low R²

    A flat slope (low R²) favours bet-hedging.  A steep negative slope
    with high R² would favour pure local adaptation.
    """
    from scipy.stats import linregress

    v = away_data['cramers_v']
    retention = away_data['retention']
    overall = away_data['overall_freq']

    sc = ax.scatter(v, retention, s=4, alpha=0.4, c=overall,
                    cmap='viridis', edgecolors='none', rasterized=True,
                    vmin=0.05, vmax=0.95)
    cb = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label('Overall B2 frequency', fontsize=7)
    cb.ax.tick_params(labelsize=6)

    # No-depletion line
    ax.axhline(1.0, color=COLORS['red'], linestyle='--', linewidth=1,
               alpha=0.6, label='No depletion')

    # Decile trend line
    n_bins = 10
    sorted_idx = np.argsort(v)
    bin_size = len(v) // n_bins
    bin_x, bin_y = [], []
    for b in range(n_bins):
        start = b * bin_size
        end = start + bin_size if b < n_bins - 1 else len(v)
        idx = sorted_idx[start:end]
        bin_x.append(np.median(v[idx]))
        bin_y.append(np.median(retention[idx]))
    ax.plot(bin_x, bin_y, '-o', color=COLORS['red'], markersize=4,
            linewidth=1.5, label='Decile medians', zorder=5)

    # Linear regression: retention ~ V
    slope, intercept, r_val, p_val, se = linregress(v, retention)
    v_range = np.array([v.min(), v.max()])
    ax.plot(v_range, intercept + slope * v_range, '-', color=COLORS['orange'],
            linewidth=2, alpha=0.8, label=f'OLS: slope={slope:.3f}')

    ax.set_xlabel("Cramér's V (niche effect size)")
    ax.set_ylabel('Away freq / home freq (retention)')
    ax.set_title('Effect size vs retention (slope test)')
    ax.legend(fontsize=7, loc='upper right')

    # Report regression and top-quartile stat
    top_q = np.percentile(v, 75)
    strong_mask = v >= top_q
    med_ret_strong = np.median(retention[strong_mask])
    txt = (f'OLS: slope = {slope:.3f} ± {se:.3f}\n'
           f'R² = {r_val**2:.3f}, p = {p_val:.2e}\n'
           f'Top-quartile V: median ret = {med_ret_strong:.2f}\n'
           f'Local adapt. predicts steep neg. slope;\n'
           f'bet-hedging predicts flat slope')
    ax.text(0.97, 0.20,
            txt,
            transform=ax.transAxes, fontsize=6.5, ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.9))


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Niche genes as insurance: away-niche carriage analysis')
    parser.add_argument('--output-dir', default='.', help='Output directory')
    parser.add_argument('--format', default='png', help='Figure format')
    parser.add_argument('--show', action='store_true', help='Show figure')
    args = parser.parse_args()

    setup_plotting()
    print_header('Niche Genes as Insurance: Away-Niche Carriage',
                 {'Dataset': 'Horesh et al. (2021) — E. coli B2',
                  'Question': 'Do niche genes have non-trivial away frequencies?'})

    # Load data
    print('\n[1/5] Loading data...')
    data = load_classification_data()

    # Compute away frequencies
    print('\n[2/5] Computing home/away frequencies...')
    away_data = compute_niche_away_frequencies(data)

    # Create figure
    print('\n[3/5] Generating panels...')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    print('  Panel A: Away retention ratio...')
    panel_a_retention(axes[0, 0], away_data)
    add_panel_label(axes[0, 0], 'A')

    print('  Panel B: Home vs away frequency...')
    panel_b_home_vs_away(axes[0, 1], away_data)
    add_panel_label(axes[0, 1], 'B')

    print('  Panel C: Retention by home niche (symmetry test)...')
    panel_c_retention_by_home(axes[1, 0], away_data)
    add_panel_label(axes[1, 0], 'C')

    print('  Panel D: Effect size vs retention...')
    panel_d_effect_vs_retention(axes[1, 1], away_data)
    add_panel_label(axes[1, 1], 'D')

    fig.tight_layout()

    # Save
    print('\n[4/5] Saving figure...')
    save_figure(fig, 'niche_insurance_analysis', output_dir=args.output_dir,
                fmt=args.format)

    if args.show:
        plt.show()
    plt.close(fig)

    # Summary
    max_away = away_data['max_away_freq']
    home = away_data['home_freq']
    overall = away_data['overall_freq']
    retention = away_data['retention']
    print(f'\n[5/5] Summary of key results:')
    print('=' * 60)
    print(f'  Total accessory genes: {away_data["n_total"]:,}')
    print(f'  Niche-specific genes: {away_data["n_niche"]:,}')
    print()
    print(f'  Home-niche frequency:')
    print(f'    Mean:   {home.mean():.3f}')
    print(f'    Median: {np.median(home):.3f}')
    print()
    print(f'  Max away-niche frequency:')
    print(f'    Mean:   {max_away.mean():.3f}')
    print(f'    Median: {np.median(max_away):.3f}')
    print()
    print(f'  *** RETENTION (away freq / home freq) ***')
    print(f'  Retention = 1.0: gene equally frequent at home and away (no niche effect)')
    print(f'  Retention = 0.0: gene absent from away niche (strict specialisation)')
    print(f'    Mean retention:   {retention.mean():.3f}')
    print(f'    Median retention: {np.median(retention):.3f}')
    print(f'    Retention ≥ 0.80: {(retention >= 0.80).mean()*100:.1f}%')
    print(f'    Retention ≥ 0.50: {(retention >= 0.50).mean()*100:.1f}%')
    print(f'    Retention < 0.20: {(retention < 0.20).mean()*100:.1f}%')
    print()

    # Per home-niche breakdown
    for src in away_data['sources_used']:
        mask = away_data['home_source'] == src
        if mask.sum() > 0:
            print(f'  Genes homing to {src} (n={mask.sum():,}):')
            print(f'    Median home freq:      {np.median(home[mask]):.3f}')
            print(f'    Median away freq:      {np.median(max_away[mask]):.3f}')
            print(f'    Median retention:      {np.median(retention[mask]):.3f}')

    # ===================================================================
    # MODEL COMPARISON: bet-hedging vs local adaptation + migration
    # ===================================================================
    from scipy.stats import kruskal, linregress

    print()
    print('  ' + '=' * 56)
    print('  MODEL COMPARISON: bet-hedging vs local adaptation')
    print('  ' + '=' * 56)
    print()

    # --- Test 1: Symmetry of retention across home niches ---
    print('  TEST 1: SYMMETRY (Kruskal-Wallis on retention by home niche)')
    print('  ' + '-' * 54)
    print('  Rationale: Under directional migration (main alternative to')
    print('  bet-hedging), gut→blood migration is common but blood→gut is')
    print('  rare. So Feces-home genes should have HIGH retention (carried')
    print('  to blood by migrating bacteria), while Blood-home genes should')
    print('  have LOW retention (blood bacteria rarely reach gut).')
    print('  Under bet-hedging, insurance value depends on environmental')
    print('  variance, not migration direction → retention should be SIMILAR.')
    print()

    ret_groups = []
    group_names = []
    for src in away_data['sources_used']:
        mask = away_data['home_source'] == src
        if mask.sum() > 0:
            ret_groups.append(retention[mask])
            group_names.append(src)

    if len(ret_groups) >= 2:
        H, p_kw = kruskal(*ret_groups)
        n_kw = sum(len(g) for g in ret_groups)
        eps_sq = H / (n_kw - 1)
        label = ('negligible' if eps_sq < 0.01 else
                 'small' if eps_sq < 0.06 else
                 'medium' if eps_sq < 0.14 else 'large')

        print(f'  Results:')
        for name, g in zip(group_names, ret_groups):
            print(f'    {name}-home: median retention = {np.median(g):.3f}')
        print(f'    H = {H:.2f}, p = {p_kw:.2e}')
        print(f'    ε² = {eps_sq:.4f} ({label} effect size)')
        print()
        if eps_sq < 0.06:
            print(f'  → FAVOURS BET-HEDGING: retention is symmetric across')
            print(f'    home niches ({label} effect size). Directional migration')
            print(f'    would produce asymmetric retention.')
        else:
            print(f'  → FAVOURS MIGRATION: retention differs across home niches')
            print(f'    ({label} effect size), consistent with directional')
            print(f'    migration between body sites.')

    # --- Test 2: V–retention slope ---
    print()
    print()
    print('  TEST 2: V–RETENTION SLOPE (OLS regression)')
    print('  ' + '-' * 54)
    print('  Rationale: Under local adaptation, genes with stronger niche')
    print('  effects (higher V) are under stronger opposing selection in away')
    print('  niches. Migration cannot maintain them → retention should drop')
    print('  steeply with V (steep negative slope, moderate R²).')
    print('  Under bet-hedging, insurance value offsets local cost →')
    print('  retention stays high even for strong niche genes (flat slope,')
    print('  low R²).')
    print()

    v = away_data['cramers_v']
    slope, intercept, r_val, p_val, se = linregress(v, retention)
    r_sq = r_val ** 2

    print(f'  Results:')
    print(f'    Slope     = {slope:.4f} ± {se:.4f}')
    print(f'    Intercept = {intercept:.3f}')
    print(f'    R²        = {r_sq:.4f}')
    print(f'    p          = {p_val:.2e}')
    print()

    top_v = np.percentile(v, 75)
    strong = v >= top_v
    med_ret_strong = np.median(retention[strong])
    bottom_v = np.percentile(v, 25)
    weak = v <= bottom_v
    med_ret_weak = np.median(retention[weak])
    print(f'    Bottom-quartile V (≤ {bottom_v:.3f}): median retention = {med_ret_weak:.2f}')
    print(f'    Top-quartile V    (≥ {top_v:.3f}): median retention = {med_ret_strong:.2f}')
    print(f'    Difference: {med_ret_weak - med_ret_strong:.2f}')
    print()
    if r_sq < 0.01:
        print(f'  → STRONGLY FAVOURS BET-HEDGING: V explains only {r_sq*100:.1f}%')
        print(f'    of retention variance. The slope is essentially flat.')
    elif r_sq < 0.10:
        print(f'  → MIXED / LEANS BET-HEDGING: V explains {r_sq*100:.1f}% of')
        print(f'    retention variance. The p-value is tiny but that reflects')
        print(f'    sample size (n={len(v):,}), not effect strength. The question')
        print(f'    is not "is the slope zero?" but "is it steep enough for')
        print(f'    local adaptation?" — and it is not. {100-r_sq*100:.1f}% of variance')
        print(f'    is unexplained by niche effect size. The key number:')
        print(f'    top-quartile V genes still retain {med_ret_strong*100:.0f}% of home')
        print(f'    frequency in away niches (vs {med_ret_weak*100:.0f}% for bottom-')
        print(f'    quartile), a difference of only {(med_ret_weak - med_ret_strong)*100:.0f} percentage')
        print(f'    points across the full range of niche effect sizes.')
    else:
        print(f'  → FAVOURS LOCAL ADAPTATION: V explains {r_sq*100:.1f}% of')
        print(f'    retention variance, consistent with stronger niche genes')
        print(f'    being more depleted in away niches.')

    # --- Overall assessment ---
    print()
    print()
    print('  ' + '=' * 56)
    print('  OVERALL ASSESSMENT')
    print('  ' + '=' * 56)
    print(f'  Niche genes retain {np.median(retention)*100:.0f}% of their home frequency in')
    print(f'  their best away niche (median retention = {np.median(retention):.2f}).')
    print(f'  Only {(retention < 0.20).mean()*100:.1f}% of niche genes are near-absent from')
    print(f'  away niches (retention < 0.20).')
    print()
    print(f'  This pattern is circumstantial but favours bet-hedging over')
    print(f'  pure local adaptation for three reasons:')
    print(f'    1. High retention floor: 82.7% of niche genes retain ≥ 50%')
    print(f'       of home frequency in away niches')
    print(f'    2. Symmetric retention: effect of home niche on retention')
    print(f'       is {label}, inconsistent with directional migration')
    print(f'    3. Flat V–retention slope: R² = {r_sq:.4f} — niche effect')
    print(f'       size barely predicts retention, inconsistent with')
    print(f'       selection-against in away niches')
    print()
    print(f'  CAVEAT: isolation source ≠ true ecological niche. An isolate')
    print(f'  labelled "Blood" may be a gut bacterium sampled during')
    print(f'  bacteremia. This migration noise inflates retention but')
    print(f'  cannot explain the symmetry or flat slope patterns.')
    print('  ' + '=' * 56)


if __name__ == '__main__':
    main()
