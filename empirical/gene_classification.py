#!/usr/bin/env python3
"""
================================================================================
GENE CLASSIFICATION: NICHE-SPECIFIC vs INSURANCE GENES
================================================================================

PURPOSE
-------
Classify individual accessory genes as niche-specific, insurance (backup),
or ambiguous using within-phylogroup chi-squared tests.

The decoupling analysis (Test 4) found ~15% of gene content variance explained
by isolation source — significant coupling but far from complete. This implies
a MIXED MODEL: some genes are niche-specific (frequency differs by body site)
while others are insurance/backup genes (maintained stochastically regardless
of environment).

METHOD
------
1. Per-gene chi-squared test within phylogroup B2 (n≈1,705 genomes)
   - For each gene: 2×3 contingency table (present/absent × Blood/Feces/Urine)
   - Chi-squared test → p-value; Cramér's V → effect size
2. BH-FDR correction for multiple testing
3. Storey (2002) π₀ estimation → fraction of true nulls (insurance genes)
4. A priori classification thresholds:
   - Niche: q < 0.05 AND V > 0.10
   - Insurance: q > 0.05 OR V < 0.05
   - Ambiguous: everything else
5. Cross-validation in phylogroup D

DATA
----
Horesh et al. (2021) — same data as decoupling analysis.

FIGURE
------
4-panel (2 × 2):
  A: Volcano plot (Cramér's V vs -log10 q)
  B: P-value histogram with Storey π₀ line
  C: Frequency profiles for niche vs insurance genes
  D: Cross-validation scatter (B2 vs D)

USAGE
-----
  python gene_classification.py [--output-dir DIR] [--format FMT] [--show]

================================================================================
"""

import os
import sys
import argparse
import numpy as np
from collections import Counter

# Path setup
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.join(_SCRIPT_DIR, '..')
_PROJECT_ROOT = os.path.join(_CODE_DIR, '..')
sys.path.insert(0, _CODE_DIR)       # for shared.*
sys.path.insert(0, _SCRIPT_DIR)     # for sibling empirical scripts

from shared.plotting import (setup_plotting, save_figure, add_panel_label,
                              print_header, COLORS)

import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, false_discovery_control
from scipy.interpolate import UnivariateSpline


# ==============================================================================
# Data loading
# ==============================================================================

def load_classification_data():
    """Load Horesh data via decoupling_analysis and verify gene_names.

    Returns the same dict as load_horesh_data() which now includes 'gene_names'.
    """
    from decoupling_analysis import load_horesh_data
    data = load_horesh_data()
    assert 'gene_names' in data, (
        "gene_names not in data dict — update load_horesh_data()")
    return data


# ==============================================================================
# Core statistics
# ==============================================================================

def per_gene_chi_squared(data, phylogroup='B2'):
    """Chi-squared test for each accessory gene within a phylogroup.

    For each gene, builds a 2×k contingency table (present/absent × sources)
    and computes chi-squared test + Cramér's V.

    Parameters
    ----------
    data : dict from load_classification_data()
    phylogroup : str

    Returns
    -------
    dict with:
        p_values : (n_genes,) array
        cramers_v : (n_genes,) array
        chi2_stats : (n_genes,) array
        source_freqs : dict of {source: (n_genes,) array}
        n_genomes : int
        sources_used : list of str
    """
    mask = data['phylogroup'] == phylogroup
    pa = data['pa_matrix'][mask]       # (n_genomes_in_pg, n_genes)
    sources = data['isolation_source'][mask]
    n_genomes_pg = pa.shape[0]
    n_genes = pa.shape[1]

    # Get unique sources with sufficient counts
    source_counts = Counter(sources)
    sources_used = sorted([s for s, c in source_counts.items() if c >= 10])
    k = len(sources_used)

    # Build source masks
    source_masks = {}
    for src in sources_used:
        source_masks[src] = (sources == src)

    # Per-source gene frequencies
    source_freqs = {}
    for src in sources_used:
        src_pa = pa[source_masks[src]]
        source_freqs[src] = src_pa.mean(axis=0)

    # Per-gene chi-squared
    p_values = np.ones(n_genes)
    chi2_stats = np.zeros(n_genes)
    cramers_v = np.zeros(n_genes)

    for g in range(n_genes):
        # Build 2×k contingency table
        table = np.zeros((2, k), dtype=int)
        for j, src in enumerate(sources_used):
            src_mask = source_masks[src]
            gene_vals = pa[src_mask, g]
            table[1, j] = gene_vals.sum()       # present
            table[0, j] = len(gene_vals) - table[1, j]  # absent

        # Check: skip if gene is monomorphic within this phylogroup
        if table[0].sum() == 0 or table[1].sum() == 0:
            p_values[g] = 1.0
            chi2_stats[g] = 0.0
            cramers_v[g] = 0.0
            continue

        # Check for zero columns (source with no genomes — shouldn't happen)
        col_sums = table.sum(axis=0)
        if np.any(col_sums == 0):
            p_values[g] = 1.0
            chi2_stats[g] = 0.0
            cramers_v[g] = 0.0
            continue

        # Chi-squared test
        try:
            chi2, p, dof, expected = chi2_contingency(table)
            p_values[g] = p
            chi2_stats[g] = chi2

            # Cramér's V for 2×k table: V = sqrt(chi2 / (n * min(r-1, k-1)))
            n_total = table.sum()
            min_dim = min(table.shape[0] - 1, table.shape[1] - 1)
            if min_dim > 0 and n_total > 0:
                cramers_v[g] = np.sqrt(chi2 / (n_total * min_dim))
            else:
                cramers_v[g] = 0.0
        except ValueError:
            # Degenerate table
            p_values[g] = 1.0
            chi2_stats[g] = 0.0
            cramers_v[g] = 0.0

    return {
        'p_values': p_values,
        'chi2_stats': chi2_stats,
        'cramers_v': cramers_v,
        'source_freqs': source_freqs,
        'n_genomes': n_genomes_pg,
        'sources_used': sources_used,
        'phylogroup': phylogroup,
    }


def compute_fdr(p_values):
    """BH-FDR correction using scipy.

    Returns q-values (adjusted p-values).
    """
    # scipy.stats.false_discovery_control returns adjusted p-values
    # using Benjamini-Hochberg method
    q = false_discovery_control(p_values, method='bh')
    return q


def estimate_pi0(p_values, lambda_grid=None):
    """Storey (2002) estimation of π₀ (fraction of true null hypotheses).

    Handles discrete p-value distributions (common with chi-squared tests
    on sparse contingency tables) by separating the p=1.0 spike from the
    continuous portion. Genes with p=1.0 have zero test statistic (identical
    frequency across sources) — these are definitionally null/insurance.
    The Storey method is then applied to the continuous p < 1.0 portion,
    and the two estimates are combined.

    π₀(λ) = #{p > λ} / (m × (1 - λ))

    Parameters
    ----------
    p_values : array
    lambda_grid : array, optional
        Grid of λ values to evaluate. Default: [0.05, 0.10, ..., 0.90]

    Returns
    -------
    dict with pi0, lambda_grid, pi0_lambda, n_exact_null, pi0_continuous
    """
    if lambda_grid is None:
        lambda_grid = np.arange(0.05, 0.91, 0.05)

    m = len(p_values)

    # Handle discrete spike: genes with p = exactly 1.0 have zero test
    # statistic and are definitionally null (insurance genes)
    n_exact_null = int(np.sum(p_values >= 0.9999))
    p_continuous = p_values[p_values < 0.9999]
    m_continuous = len(p_continuous)

    # Apply Storey to the continuous portion
    pi0_lambda = np.zeros(len(lambda_grid))

    if m_continuous > 100:
        for i, lam in enumerate(lambda_grid):
            n_above = np.sum(p_continuous > lam)
            pi0_lambda[i] = n_above / (m_continuous * (1 - lam))

        pi0_lambda = np.clip(pi0_lambda, 0.001, 0.999)

        try:
            valid = np.isfinite(pi0_lambda)
            if valid.sum() >= 4:
                spline = UnivariateSpline(lambda_grid[valid],
                                           pi0_lambda[valid],
                                           k=3, s=len(lambda_grid) * 0.05)
                pi0_continuous = float(spline(lambda_grid[-1]))
            else:
                pi0_continuous = float(np.mean(
                    pi0_lambda[lambda_grid >= 0.5]))
        except Exception:
            pi0_continuous = float(np.mean(pi0_lambda[lambda_grid >= 0.5]))

        pi0_continuous = float(np.clip(pi0_continuous, 0.01, 0.99))
    else:
        pi0_continuous = 0.5  # fallback

    # Combine: total insurance = exact nulls + estimated nulls in rest
    n_insurance_continuous = int(pi0_continuous * m_continuous)
    n_insurance_total = n_exact_null + n_insurance_continuous
    pi0 = float(np.clip(n_insurance_total / m, 0.01, 0.99))

    return {
        'pi0': pi0,
        'lambda_grid': lambda_grid,
        'pi0_lambda': pi0_lambda,
        'n_exact_null': n_exact_null,
        'pi0_continuous': pi0_continuous,
    }


def classify_genes(q_values, cramers_v, q_thresh=0.05, v_niche=0.10,
                    v_insurance=0.05):
    """Classify genes as niche-specific, insurance, or ambiguous.

    A priori thresholds (stated before seeing results):
      - Niche: q < q_thresh AND V > v_niche
      - Insurance: q > q_thresh OR V < v_insurance
      - Ambiguous: everything else

    Parameters
    ----------
    q_values : array of BH-adjusted p-values
    cramers_v : array of Cramér's V effect sizes
    q_thresh : float (default 0.05)
    v_niche : float (default 0.10)
    v_insurance : float (default 0.05)

    Returns
    -------
    labels : array of str ('niche', 'insurance', 'ambiguous')
    """
    n = len(q_values)
    labels = np.full(n, 'ambiguous', dtype='U10')

    # Niche: significant AND large effect
    niche_mask = (q_values < q_thresh) & (cramers_v > v_niche)
    labels[niche_mask] = 'niche'

    # Insurance: not significant OR tiny effect
    insurance_mask = (q_values > q_thresh) | (cramers_v < v_insurance)
    # Don't overwrite niche genes
    insurance_mask = insurance_mask & ~niche_mask
    labels[insurance_mask] = 'insurance'

    return labels


def cross_validate(data, phylogroup_train='B2', phylogroup_test='D'):
    """Run per-gene chi-squared in two phylogroups, compare classifications.

    Returns
    -------
    dict with:
        v_train, v_test : Cramér's V arrays
        labels_train, labels_test : classification labels
        concordance : fraction of genes with same label
        kappa : Cohen's kappa
        n_classifiable : genes classifiable in both
    """
    # Run chi-squared in both phylogroups
    result_train = per_gene_chi_squared(data, phylogroup=phylogroup_train)
    result_test = per_gene_chi_squared(data, phylogroup=phylogroup_test)

    # FDR correction
    q_train = compute_fdr(result_train['p_values'])
    q_test = compute_fdr(result_test['p_values'])

    # Classify in each
    labels_train = classify_genes(q_train, result_train['cramers_v'])
    labels_test = classify_genes(q_test, result_test['cramers_v'])

    # Concordance (simple agreement)
    agree = np.sum(labels_train == labels_test)
    n_total = len(labels_train)
    concordance = agree / n_total

    # Cohen's kappa
    # Categories: niche, insurance, ambiguous
    cats = ['niche', 'insurance', 'ambiguous']
    n_cats = len(cats)
    confusion = np.zeros((n_cats, n_cats), dtype=int)
    cat_to_idx = {c: i for i, c in enumerate(cats)}
    for lt, lte in zip(labels_train, labels_test):
        i = cat_to_idx.get(lt, 2)
        j = cat_to_idx.get(lte, 2)
        confusion[i, j] += 1

    p_o = concordance  # observed agreement
    # Expected agreement under chance
    row_sums = confusion.sum(axis=1) / n_total
    col_sums = confusion.sum(axis=0) / n_total
    p_e = np.sum(row_sums * col_sums)

    if p_e < 1.0:
        kappa = (p_o - p_e) / (1 - p_e)
    else:
        kappa = 1.0

    return {
        'v_train': result_train['cramers_v'],
        'v_test': result_test['cramers_v'],
        'q_train': q_train,
        'q_test': q_test,
        'labels_train': labels_train,
        'labels_test': labels_test,
        'concordance': concordance,
        'kappa': kappa,
        'confusion': confusion,
        'categories': cats,
        'n_total': n_total,
        'phylogroup_train': phylogroup_train,
        'phylogroup_test': phylogroup_test,
        'train_result': result_train,
        'test_result': result_test,
    }


# ==============================================================================
# Panel functions
# ==============================================================================

def panel_a_volcano(ax, data):
    """Panel A: Volcano plot — Cramér's V vs -log10(q-value).

    Color by classification: red = niche, blue = insurance, grey = ambiguous.
    Horizontal dashed line at q = 0.05, vertical at V = 0.10.
    """
    print('  Panel A: Volcano plot...')

    result = per_gene_chi_squared(data, phylogroup='B2')
    q = compute_fdr(result['p_values'])
    v = result['cramers_v']
    labels = classify_genes(q, v)

    neg_log_q = -np.log10(np.clip(q, 1e-300, 1))

    # Classification thresholds
    q_line = -np.log10(0.05)   # ~1.3
    v_niche = 0.10
    v_insurance = 0.05

    # Determine axis limits first so shading fills the full panel
    v_max = max(v.max() * 1.08, 0.65)
    y_max = max(neg_log_q.max() * 1.08, 10)

    # ── Background shading for classification zones ──────────────────
    from matplotlib.patches import Rectangle

    # Insurance: q > 0.05 (below q-line, any V) OR V < 0.05 (any q)
    # → bottom strip + left strip
    ax.add_patch(Rectangle((0, 0), v_max, q_line,
                            facecolor=COLORS['blue'], alpha=0.08,
                            edgecolor='none', zorder=0))
    ax.add_patch(Rectangle((0, q_line), v_insurance, y_max - q_line,
                            facecolor=COLORS['blue'], alpha=0.08,
                            edgecolor='none', zorder=0))

    # Ambiguous: q < 0.05 AND 0.05 ≤ V ≤ 0.10
    ax.add_patch(Rectangle((v_insurance, q_line),
                            v_niche - v_insurance, y_max - q_line,
                            facecolor=COLORS['grey'], alpha=0.08,
                            edgecolor='none', zorder=0))

    # Niche: q < 0.05 AND V > 0.10
    ax.add_patch(Rectangle((v_niche, q_line),
                            v_max - v_niche, y_max - q_line,
                            facecolor=COLORS['red'], alpha=0.08,
                            edgecolor='none', zorder=0))

    # Zone labels
    ax.text(v_max * 0.55, y_max * 0.92, 'Niche', fontsize=9,
            color=COLORS['red'], alpha=0.5, fontweight='bold',
            ha='center', va='top', zorder=1)
    ax.text(0.025, y_max * 0.55, 'Insurance', fontsize=8,
            color=COLORS['blue'], alpha=0.5, fontweight='bold',
            ha='center', va='center', rotation=90, zorder=1)
    ax.text((v_insurance + v_niche) / 2, y_max * 0.55, 'Ambig.',
            fontsize=7, color=COLORS['grey'], alpha=0.5,
            fontweight='bold', ha='center', va='center',
            rotation=90, zorder=1)

    # ── Plot points ──────────────────────────────────────────────────
    n_niche = np.sum(labels == 'niche')
    n_ins = np.sum(labels == 'insurance')
    n_amb = np.sum(labels == 'ambiguous')

    for cls, color, zorder, alpha, ms in [
        ('insurance', COLORS['blue'], 2, 0.3, 3),
        ('ambiguous', COLORS['grey'], 3, 0.4, 3),
        ('niche', COLORS['red'], 4, 0.6, 4),
    ]:
        mask = labels == cls
        n_cls = mask.sum()
        ax.scatter(v[mask], neg_log_q[mask], c=color, s=ms, alpha=alpha,
                   edgecolors='none', zorder=zorder,
                   label=f'{cls.capitalize()} (n={n_cls:,})')

    # ── Threshold lines ──────────────────────────────────────────────
    ax.axhline(q_line, color=COLORS['grey'], linestyle='--', linewidth=0.8,
               alpha=0.5, zorder=1)
    ax.axvline(v_niche, color=COLORS['grey'], linestyle='--', linewidth=0.8,
               alpha=0.5, zorder=1)
    ax.axvline(v_insurance, color=COLORS['grey'], linestyle=':', linewidth=0.6,
               alpha=0.4, zorder=1)

    ax.set_xlim(0, v_max)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Cramér's V (effect size)")
    ax.set_ylabel(r'$-\log_{10}$(q-value)')
    ax.set_title('Gene-environment association (B2)')

    ax.legend(fontsize=7, loc='upper right', markerscale=3,
              framealpha=0.9, edgecolor=COLORS['grey'])

    data['_panel_a_results'] = {
        'n_niche': int(n_niche),
        'n_insurance': int(n_ins),
        'n_ambiguous': int(n_amb),
        'n_total': len(labels),
        'result': result,
        'q_values': q,
        'labels': labels,
    }


def panel_b_pvalue_histogram(ax, data):
    """Panel B: P-value histogram with Storey π₀ line.

    Uses a broken y-axis so both the massive spike near p=0 and the
    flat null portion (where π₀ is estimated) are clearly visible.

    The caller passes a single axes object; this function replaces it
    with two vertically stacked subplots sharing the same x-axis.
    """
    print('  Panel B: P-value histogram...')

    # Use cached result from panel A if available, otherwise compute
    if '_panel_a_results' in data:
        result = data['_panel_a_results']['result']
    else:
        result = per_gene_chi_squared(data, phylogroup='B2')

    p = result['p_values']
    pi0_result = estimate_pi0(p)
    pi0 = pi0_result['pi0']
    m = len(p)

    # ── Create broken y-axis via inset axes ──────────────────────────
    # We'll draw on the original ax (bottom = zoomed-in) and create
    # an inset for the top (full-range) portion.
    fig = ax.get_figure()
    pos = ax.get_position()

    # Clear the original axes — we'll draw two new ones
    ax.set_visible(False)

    # Bottom axes: zoomed to show the flat null region + π₀ line
    gap = 0.02  # gap between the two axes
    bottom_frac = 0.55  # fraction of height for the bottom panel
    h_bottom = (pos.height - gap) * bottom_frac
    h_top = (pos.height - gap) * (1 - bottom_frac)

    ax_bot = fig.add_axes([pos.x0, pos.y0, pos.width, h_bottom])
    ax_top = fig.add_axes([pos.x0, pos.y0 + h_bottom + gap,
                           pos.width, h_top])

    # ── Compute histogram (shared bin edges) ─────────────────────────
    n_bins = 25
    counts, edges = np.histogram(p, bins=n_bins)
    width = edges[1] - edges[0]
    centres = (edges[:-1] + edges[1:]) / 2

    expected_per_bin = m * pi0 / n_bins

    # Colour: orange for bins near zero with excess counts
    bar_colors = []
    for i, count in enumerate(counts):
        if count > expected_per_bin * 1.5 and edges[i] < 0.15:
            bar_colors.append(COLORS['orange'])
        else:
            bar_colors.append(COLORS['blue'])

    # ── Draw on both axes ────────────────────────────────────────────
    for a in (ax_top, ax_bot):
        a.bar(centres, counts, width=width * 0.92, color=bar_colors,
              alpha=0.75, edgecolor='white', linewidth=0.5)
        a.axhline(expected_per_bin, color=COLORS['red'], linestyle='--',
                  linewidth=1.5)

    # ── Set y-limits ─────────────────────────────────────────────────
    # Top: show the tall spikes
    top_max = max(counts) * 1.15
    ax_top.set_ylim(200, top_max)
    # Bottom: show the flat null region clearly
    ax_bot.set_ylim(0, 150)

    # ── Broken-axis styling ──────────────────────────────────────────
    ax_top.spines['bottom'].set_visible(False)
    ax_bot.spines['top'].set_visible(False)
    ax_top.tick_params(bottom=False, labelbottom=False)

    # Diagonal break marks
    d = 0.015
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False,
                  linewidth=0.8)
    ax_top.plot((-d, +d), (-d * 2, +d * 2), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d * 2, +d * 2), **kwargs)

    kwargs.update(transform=ax_bot.transAxes)
    ax_bot.plot((-d, +d), (1 - d * 2, 1 + d * 2), **kwargs)
    ax_bot.plot((1 - d, 1 + d), (1 - d * 2, 1 + d * 2), **kwargs)

    # Remove top/right spines to match other panels
    for a in (ax_top, ax_bot):
        a.spines['right'].set_visible(False)
    ax_top.spines['top'].set_visible(False)

    # ── Labels and annotations ───────────────────────────────────────
    ax_bot.set_xlabel('p-value (per-gene chi-squared)')
    ax_bot.set_ylabel('Count')
    # Shift ylabel to span both axes
    ax_bot.yaxis.set_label_coords(-0.1, 1.0 + gap / pos.height)
    ax_top.set_title('P-value distribution (B2)')

    # π₀ label on the bottom panel where the line is visible
    ax_bot.text(0.55, expected_per_bin + 3,
                r'$\hat\pi_0$' + f' = {pi0:.2f}',
                fontsize=8, color=COLORS['red'], va='bottom')

    # Summary annotation
    n_niche_est = int(m * (1 - pi0))
    n_ins_est = int(m * pi0)
    ax_bot.text(0.97, 0.92,
                r'$\hat\pi_0$' + f' = {pi0:.3f}\n'
                f'Insurance: ~{n_ins_est:,} ({pi0*100:.0f}%)\n'
                f'Niche: ~{n_niche_est:,} ({(1-pi0)*100:.0f}%)',
                transform=ax_bot.transAxes, fontsize=7, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=COLORS['grey'], alpha=0.9),
                family='monospace')

    data['_panel_b_results'] = {
        'pi0': pi0,
        'pi0_result': pi0_result,
        'n_insurance_est': n_ins_est,
        'n_niche_est': n_niche_est,
    }

    # Return the top axes so the caller can add a panel label
    return ax_top


def panel_c_frequency_profiles(ax, data):
    """Panel C: Gene frequency profiles for niche vs insurance genes.

    Show per-source frequencies for top niche genes (highest V) and
    random insurance genes. Niche genes should show uneven bars;
    insurance genes should show even bars.
    """
    print('  Panel C: Frequency profiles...')

    # Use cached results
    if '_panel_a_results' in data:
        result = data['_panel_a_results']['result']
        labels = data['_panel_a_results']['labels']
        v = result['cramers_v']
    else:
        result = per_gene_chi_squared(data, phylogroup='B2')
        q = compute_fdr(result['p_values'])
        v = result['cramers_v']
        labels = classify_genes(q, v)

    sources = result['sources_used']
    source_freqs = result['source_freqs']
    gene_names = data['gene_names']

    # Select top 5 niche genes (highest V) and 5 random insurance genes
    niche_idx = np.where(labels == 'niche')[0]
    ins_idx = np.where(labels == 'insurance')[0]

    n_show = 5
    if len(niche_idx) >= n_show:
        top_niche = niche_idx[np.argsort(v[niche_idx])[-n_show:]][::-1]
    else:
        top_niche = niche_idx

    rng = np.random.default_rng(42)
    if len(ins_idx) >= n_show:
        # Pick insurance genes with moderate frequency (not too rare/common)
        freqs = data['gene_frequencies'][ins_idx]
        moderate = ins_idx[(freqs > 0.2) & (freqs < 0.8)]
        if len(moderate) >= n_show:
            sample_ins = rng.choice(moderate, size=n_show, replace=False)
        else:
            sample_ins = rng.choice(ins_idx, size=min(n_show, len(ins_idx)),
                                     replace=False)
    else:
        sample_ins = ins_idx

    # Combine: niche genes first, then insurance
    gene_indices = np.concatenate([top_niche, sample_ins])
    n_genes_show = len(gene_indices)

    # Grouped bar chart
    x = np.arange(n_genes_show)
    width = 0.25
    source_colors = {
        'Blood': COLORS['red'],
        'Feces': COLORS['green'],
        'Urine': COLORS['orange'],
    }

    for i, src in enumerate(sources):
        freqs_src = [source_freqs[src][gi] for gi in gene_indices]
        color = source_colors.get(src, COLORS['grey'])
        ax.bar(x + i * width, freqs_src, width, label=src, color=color,
               alpha=0.8, edgecolor='white', linewidth=0.3)

    # Labels
    gene_labels = []
    for gi in gene_indices:
        name = gene_names[gi]
        # Truncate long names
        if len(name) > 12:
            name = name[:10] + '..'
        gene_labels.append(name)

    ax.set_xticks(x + width)
    ax.set_xticklabels(gene_labels, fontsize=5, rotation=45, ha='right')
    ax.set_ylabel('Gene frequency')
    ax.set_title('Frequency profiles: niche vs insurance')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_ylim(0, 1.05)

    # Divider between niche and insurance
    if len(top_niche) > 0 and len(sample_ins) > 0:
        divider_x = len(top_niche) - 0.5 + width
        ax.axvline(divider_x, color='black', linestyle=':', linewidth=0.8)
        ax.text(divider_x - 0.3, 1.02, 'Niche', fontsize=6, ha='right',
                transform=ax.get_xaxis_transform())
        ax.text(divider_x + 0.3, 1.02, 'Insurance', fontsize=6, ha='left',
                transform=ax.get_xaxis_transform())


def panel_d_cross_validation(ax, data):
    """Panel D: Cross-phylogroup validation (B2 vs D).

    Scatter: Cramér's V in B2 (x) vs D (y) for all genes.
    Color by B2 classification. Diagonal = perfect concordance.
    """
    print('  Panel D: Cross-validation (B2 vs D)...')

    cv = cross_validate(data, phylogroup_train='B2', phylogroup_test='D')

    v_b2 = cv['v_train']
    v_d = cv['v_test']
    labels_b2 = cv['labels_train']

    # Plot each class
    for cls, color, alpha, ms in [
        ('insurance', COLORS['blue'], 0.2, 3),
        ('ambiguous', COLORS['grey'], 0.3, 3),
        ('niche', COLORS['red'], 0.7, 5),
    ]:
        mask = labels_b2 == cls
        ax.scatter(v_b2[mask], v_d[mask], c=color, s=ms, alpha=alpha,
                   edgecolors='none',
                   label=f'{cls.capitalize()} in B2')

    # Diagonal
    max_val = max(v_b2.max(), v_d.max())
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Cramér's V in B2")
    ax.set_ylabel("Cramér's V in D")
    ax.set_title('Cross-phylogroup validation')
    ax.legend(fontsize=6, loc='upper left', markerscale=3)

    # Spearman correlation between V values
    from scipy.stats import spearmanr
    rho, p_rho = spearmanr(v_b2, v_d)

    ax.text(0.95, 0.05,
            f'Concordance: {cv["concordance"]*100:.1f}%\n'
            f'Cohen κ = {cv["kappa"]:.3f}\n'
            f'Spearman ρ(V) = {rho:.3f}\n'
            f'n = {cv["n_total"]:,} genes',
            transform=ax.transAxes, fontsize=7, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=COLORS['grey'], alpha=0.9),
            family='monospace')

    data['_panel_d_results'] = {
        'concordance': cv['concordance'],
        'kappa': cv['kappa'],
        'rho_v': rho,
        'p_rho': p_rho,
        'confusion': cv['confusion'],
        'categories': cv['categories'],
    }


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Gene classification: niche-specific vs insurance genes')
    parser.add_argument('--output-dir', default='.', help='Output directory')
    parser.add_argument('--format', default='png',
                        choices=['png', 'pdf', 'svg', 'all'],
                        help='Figure format')
    parser.add_argument('--show', action='store_true', help='Show figure')
    args = parser.parse_args()

    print_header('Gene Classification: Niche vs Insurance Genes', {
        'Dataset': 'Horesh et al. (2021) — E. coli accessory genes',
        'Method': 'Per-gene chi-squared within phylogroup B2',
        'Niche threshold': 'BH q < 0.05 AND Cramér V > 0.10',
        'Insurance threshold': 'q > 0.05 OR V < 0.05',
        'Cross-validation': 'Phylogroup D',
    })

    setup_plotting()

    # Load data
    print('\n[1/5] Loading data...')
    data = load_classification_data()
    print(f'  Loaded: {data["n_genomes"]} genomes, '
          f'{data["n_accessory_genes"]} accessory genes, '
          f'{len(data["gene_names"])} gene names')

    # Create figure
    print('\n[2/5] Creating figure...')
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))
    fig.suptitle(
        'Gene Classification: Niche-Specific vs Insurance Genes\n'
        '(Per-gene chi-squared within E. coli phylogroup B2)',
        fontsize=12, fontweight='bold', y=0.98)

    # Generate panels
    print('\n[3/5] Generating panels...')
    panel_a_volcano(axes[0, 0], data)
    add_panel_label(axes[0, 0], 'A')

    panel_b_pvalue_histogram(axes[0, 1], data)
    add_panel_label(axes[0, 1], 'B')

    panel_c_frequency_profiles(axes[1, 0], data)
    add_panel_label(axes[1, 0], 'C')

    panel_d_cross_validation(axes[1, 1], data)
    add_panel_label(axes[1, 1], 'D')

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save
    print('\n[4/5] Saving figure...')
    save_figure(fig, 'gene_classification', output_dir=args.output_dir,
                fmt=args.format)

    if args.show:
        plt.show()
    else:
        plt.close(fig)

    # Summary
    print('\n[5/5] Summary of key results:')
    print('=' * 60)

    if '_panel_a_results' in data:
        r = data['_panel_a_results']
        print(f'\n  Panel A (Gene Classification):')
        print(f'    Total accessory genes: {r["n_total"]:,}')
        print(f'    Niche-specific: {r["n_niche"]:,} ({r["n_niche"]/r["n_total"]*100:.1f}%)')
        print(f'    Insurance/backup: {r["n_insurance"]:,} ({r["n_insurance"]/r["n_total"]*100:.1f}%)')
        print(f'    Ambiguous: {r["n_ambiguous"]:,} ({r["n_ambiguous"]/r["n_total"]*100:.1f}%)')

    if '_panel_b_results' in data:
        r = data['_panel_b_results']
        print(f'\n  *** Panel B (KEY RESULT — Storey π₀) ***')
        print(f'    π₀ = {r["pi0"]:.3f}')
        print(f'    Estimated insurance genes: {r["n_insurance_est"]:,} ({r["pi0"]*100:.1f}%)')
        print(f'    Estimated niche genes: {r["n_niche_est"]:,} ({(1-r["pi0"])*100:.1f}%)')

    if '_panel_d_results' in data:
        r = data['_panel_d_results']
        print(f'\n  Panel D (Cross-validation B2 → D):')
        print(f'    Concordance: {r["concordance"]*100:.1f}%')
        print(f'    Cohen κ = {r["kappa"]:.3f}')
        print(f'    Spearman ρ(V) = {r["rho_v"]:.3f}')

    print('\n' + '=' * 60)


if __name__ == '__main__':
    main()
