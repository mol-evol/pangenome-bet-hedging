#!/usr/bin/env python3
"""
================================================================================
VARIANCE ANALYSIS: SELECTION CONSTRAINS THE SECOND MOMENT OF GENE CONTENT
================================================================================

Prediction: The core bet-hedging prediction is that pangenomes reduce variance
in fitness outcomes: W_geo ≈ W_arith − σ²_W / (2·W_arith). The frozen
selection analysis (Panels A–F) shows that accessory genes are under purifying
selection — a first-moment result. This analysis tests whether selection also
constrains the VARIANCE (second moment) of gene content heterogeneity.

Bet-hedging does NOT predict uniform gene content. It predicts CONTROLLED
diversity — different genomes carrying different genes. What bet-hedging
DOES predict is that this variability is constrained by selection intensity.

Data source:
  Douglas & Shapiro (2024) DOI: 10.1038/s41559-023-02268-6
  655 prokaryotic species (after filtering for valid SD values), each with:
    - SD and mean of singleton gene counts (subsampled to 9 genomes)
    - SD and mean of singleton pseudogene counts (neutral reference)
    - dN/dS (core-gene selection proxy for Ne)
    - Genomic fluidity (gene and pseudogene)
    - Taxonomic class (GTDB)

Four-panel figure:
  Panel A: Mean constraint vs variance constraint (fluidity ratio vs CV ratio)
  Panel B: CV of gene singletons tracks selection intensity (CV vs dN/dS)
  Panel C: Taylor's law — tighter mean-variance scaling for genes
  Panel D: Within-class replication of the CV–selection gradient

Usage:
    python variance_analysis.py
    python variance_analysis.py --show
    python variance_analysis.py --output-dir results --format pdf
================================================================================
"""

import sys
import os
import argparse
import numpy as np

# Path setup
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.join(_SCRIPT_DIR, '..')
_PROJECT_ROOT = os.path.join(_CODE_DIR, '..')
sys.path.insert(0, _CODE_DIR)       # for shared.*
sys.path.insert(0, _SCRIPT_DIR)     # for sibling empirical scripts

from shared.params import PARAMS
from shared.plotting import (setup_plotting, COLORS, save_figure,
                              add_panel_label, print_header)
import matplotlib.pyplot as plt


# ==============================================================================
# Argument parsing
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Variance Analysis: Selection Constrains Gene Content Variability')
    parser.add_argument('--output-dir', default='.', help='Output directory')
    parser.add_argument('--format', default='png',
                        choices=['png', 'pdf', 'svg', 'all'],
                        help='Output format (default: png)')
    parser.add_argument('--show', action='store_true',
                        help='Display the figure interactively')
    return parser.parse_args()


# ==============================================================================
# Data loading — real data only, no fabrication
# ==============================================================================

def load_real_data():
    """Load species-level data from Douglas & Shapiro (2024), including SD columns.

    Reads pangenome_and_related_metrics.tsv.gz: filters to 655 species with
    valid SD values for both gene and pseudogene singletons and non-zero means.

    Computes CV (coefficient of variation) for gene and pseudogene singletons.

    Returns
    -------
    dict with keys:
        n_species : int (655)
        species_names : np.ndarray of str
        dnds : np.ndarray — core-gene dN/dS (Ne proxy)
        genomic_fluidity : np.ndarray — gene fluidity
        pseudogene_fluidity : np.ndarray — pseudogene fluidity
        mean_singletons_gene : np.ndarray — mean singleton count (genes, per 9)
        sd_singletons_gene : np.ndarray — SD singleton count (genes, per 9)
        mean_singletons_pseudo : np.ndarray — mean singleton count (pseudogenes, per 9)
        sd_singletons_pseudo : np.ndarray — SD singleton count (pseudogenes, per 9)
        cv_gene : np.ndarray — CV of gene singletons
        cv_pseudo : np.ndarray — CV of pseudogene singletons
        cv_ratio : np.ndarray — CV_gene / CV_pseudo
        taxonomic_class : np.ndarray of str
    """
    import gzip
    import csv

    data_dir = os.path.join(_CODE_DIR, 'data', 'douglas_shapiro')
    metrics_path = os.path.join(data_dir, 'pangenome_and_related_metrics.tsv.gz')

    if not os.path.exists(metrics_path):
        raise FileNotFoundError(
            f"Data file not found: {metrics_path}\n"
            "Download Douglas & Shapiro (2024) data from:\n"
            "  Zenodo: https://zenodo.org/records/8326664"
        )

    print(f'  Loading {metrics_path}...')

    species_names = []
    dnds = []
    genomic_fluidity = []
    pseudogene_fluidity = []
    mean_singletons_gene = []
    sd_singletons_gene = []
    mean_singletons_pseudo = []
    sd_singletons_pseudo = []
    taxonomic_class = []

    with gzip.open(metrics_path, 'rt') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            try:
                d_dnds = float(row['dnds'])
                d_dn = float(row['dn'])
                d_ds = float(row['ds'])
                # Skip species with invalid dN/dS, dN or dS
                if d_dnds <= 0 or d_dn <= 0 or d_ds <= 0:
                    continue

                # SD columns — skip if missing or invalid
                sd_gene_str = row.get('sd_num_singletons_per9', '')
                sd_pseudo_str = row.get('sd_num_singletons_pseudo_per9', '')
                if not sd_gene_str or not sd_pseudo_str:
                    continue
                d_sd_gene = float(sd_gene_str)
                d_sd_pseudo = float(sd_pseudo_str)
                if d_sd_gene <= 0 or d_sd_pseudo <= 0:
                    continue

                # Mean singletons — need non-zero for CV
                d_mean_gene = float(row['mean_num_singletons_per9'])
                d_mean_pseudo = float(row['mean_num_singletons_pseudo_per9'])
                if d_mean_gene <= 0 or d_mean_pseudo <= 0:
                    continue

                species_names.append(row[''])
                dnds.append(d_dnds)
                genomic_fluidity.append(float(row['genomic_fluidity']))
                pseudogene_fluidity.append(float(row['pseudogene_genomic_fluidity']))
                mean_singletons_gene.append(d_mean_gene)
                sd_singletons_gene.append(d_sd_gene)
                mean_singletons_pseudo.append(d_mean_pseudo)
                sd_singletons_pseudo.append(d_sd_pseudo)
                taxonomic_class.append(row['class'])

            except (ValueError, KeyError, TypeError):
                continue

    # Convert to arrays
    mean_sg = np.array(mean_singletons_gene)
    sd_sg = np.array(sd_singletons_gene)
    mean_sp = np.array(mean_singletons_pseudo)
    sd_sp = np.array(sd_singletons_pseudo)

    # Compute CVs
    cv_gene = sd_sg / mean_sg
    cv_pseudo = sd_sp / mean_sp
    cv_ratio = cv_gene / cv_pseudo

    n_species = len(species_names)
    print(f'  Loaded {n_species} species with valid CV data')

    return {
        'n_species': n_species,
        'species_names': np.array(species_names),
        'dnds': np.array(dnds),
        'genomic_fluidity': np.array(genomic_fluidity),
        'pseudogene_fluidity': np.array(pseudogene_fluidity),
        'mean_singletons_gene': mean_sg,
        'sd_singletons_gene': sd_sg,
        'mean_singletons_pseudo': mean_sp,
        'sd_singletons_pseudo': sd_sp,
        'cv_gene': cv_gene,
        'cv_pseudo': cv_pseudo,
        'cv_ratio': cv_ratio,
        'taxonomic_class': np.array(taxonomic_class),
    }


# ==============================================================================
# Panel functions
# ==============================================================================

def panel_a_mean_vs_variance(ax, data):
    """Panel A: Mean constraint vs variance constraint (joint visualization).

    X-axis: fluidity ratio (gene/pseudo) — measures first-moment constraint.
    Y-axis: CV ratio (gene/pseudo) — measures second-moment constraint.
    Coloured by dN/dS.

    Species below x=1 have mean gene content constrained below neutral (known).
    Species above y=1 have MORE variable gene content than neutral (new).
    Stronger selection (bluer) should cluster closer to y=1.
    """
    print('  Panel A: Mean constraint vs variance constraint...')

    from scipy.stats import spearmanr

    flu_ratio = data['genomic_fluidity'] / data['pseudogene_fluidity']
    cv_ratio = data['cv_ratio']
    dnds = data['dnds']

    # Scatter coloured by dN/dS
    sc = ax.scatter(flu_ratio, cv_ratio, s=12, alpha=0.5, c=dnds,
                    cmap='viridis', edgecolors='none', zorder=3,
                    vmin=0, vmax=0.5)

    # Reference lines
    ax.axhline(y=1, color=COLORS['red'], linestyle=':', linewidth=0.8,
               alpha=0.6, label='Neutral expectation')
    ax.axvline(x=1, color=COLORS['red'], linestyle=':', linewidth=0.8,
               alpha=0.6)

    # Annotate quadrants
    ax.text(0.05, 0.95, 'Mean constrained\nVariance elevated',
            transform=ax.transAxes, fontsize=6, va='top', ha='left',
            color=COLORS['grey'], style='italic')

    # Correlation
    rho, p = spearmanr(cv_ratio, dnds)

    ax.set_xlabel('Fluidity ratio (gene / pseudo)')
    ax.set_ylabel('CV ratio (gene / pseudo)')
    ax.set_title('Selection constrains both moments')

    ax.text(0.95, 0.05,
            f'n = {data["n_species"]} species\n'
            f'Median CV ratio: {np.median(cv_ratio):.2f}\n'
            f'CV ratio > 1: {np.mean(cv_ratio > 1)*100:.0f}%\n'
            f'CV ratio vs dN/dS: ρ = {rho:.3f}',
            transform=ax.transAxes, fontsize=6.5, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=COLORS['grey'], alpha=0.9))

    # Colorbar
    cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('dN/dS (core genes)', fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # Store results
    data['_panel_a_results'] = {
        'median_flu_ratio': float(np.median(flu_ratio)),
        'median_cv_ratio': float(np.median(cv_ratio)),
        'frac_cv_above_1': float(np.mean(cv_ratio > 1)),
        'rho_cv_dnds': rho, 'p_cv_dnds': p,
    }


def panel_b_cv_vs_selection(ax, data):
    """Panel B: CV of gene singletons tracks selection intensity.

    Both gene and pseudogene CVs correlate with dN/dS, but the gene
    correlation is stronger — showing that selection constrains gene
    content variability BEYOND the neutral baseline.
    """
    print('  Panel B: CV vs selection intensity...')

    from scipy.stats import spearmanr

    dnds = data['dnds']
    cv_gene = data['cv_gene']
    cv_pseudo = data['cv_pseudo']

    # Scatter: genes (blue) and pseudogenes (orange)
    ax.scatter(dnds, cv_gene, s=10, alpha=0.3, color=COLORS['blue'],
               edgecolors='none', zorder=3, label='Genes')
    ax.scatter(dnds, cv_pseudo, s=10, alpha=0.3, color=COLORS['orange'],
               edgecolors='none', zorder=2, label='Pseudogenes')

    # Lowess-style trend: bin by dN/dS quintiles and plot medians
    quintile_edges = np.percentile(dnds, [0, 20, 40, 60, 80, 100])
    for cv_arr, color in [(cv_gene, COLORS['blue']),
                          (cv_pseudo, COLORS['orange'])]:
        bin_centres = []
        bin_medians = []
        for i in range(len(quintile_edges) - 1):
            mask = (dnds >= quintile_edges[i]) & (dnds < quintile_edges[i+1])
            if i == len(quintile_edges) - 2:  # include upper bound
                mask = (dnds >= quintile_edges[i]) & (dnds <= quintile_edges[i+1])
            if mask.sum() > 0:
                bin_centres.append(np.median(dnds[mask]))
                bin_medians.append(np.median(cv_arr[mask]))
        ax.plot(bin_centres, bin_medians, '-o', color=color, linewidth=2,
                markersize=5, zorder=5, alpha=0.9)

    # Correlations
    rho_gene, p_gene = spearmanr(dnds, cv_gene)
    rho_pseudo, p_pseudo = spearmanr(dnds, cv_pseudo)

    ax.set_xlabel('dN/dS (core genes)')
    ax.set_ylabel('CV of singleton counts')
    ax.set_title('Weaker selection → more variable gene content')
    ax.legend(fontsize=7, loc='upper left', framealpha=0.9)

    ax.text(0.95, 0.05,
            f'Gene CV vs dN/dS:\n'
            f'  ρ = {rho_gene:.3f} (p = {p_gene:.1e})\n'
            f'Pseudo CV vs dN/dS:\n'
            f'  ρ = {rho_pseudo:.3f} (p = {p_pseudo:.1e})\n'
            f'Δρ = {rho_gene - rho_pseudo:.3f}',
            transform=ax.transAxes, fontsize=6.5, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=COLORS['grey'], alpha=0.9),
            family='monospace')

    # Store results
    data['_panel_b_results'] = {
        'rho_gene': rho_gene, 'p_gene': p_gene,
        'rho_pseudo': rho_pseudo, 'p_pseudo': p_pseudo,
        'delta_rho': rho_gene - rho_pseudo,
    }


def panel_c_taylors_law(ax, data):
    """Panel C: Taylor's law — tighter mean-variance scaling for genes.

    Taylor's law: SD ~ Mean^b (i.e., log(SD) = a + b*log(Mean)).
    Exponent b < 1 means sub-proportional scaling (variance grows slower
    than mean). Genes should have a LOWER exponent than pseudogenes,
    showing that selection constrains the mean-variance relationship.

    Reference exponents: b = 0.5 (Poisson), b = 1.0 (proportional).
    """
    print("  Panel C: Taylor's law (mean-variance scaling)...")

    from scipy.stats import linregress

    mean_g = data['mean_singletons_gene']
    sd_g = data['sd_singletons_gene']
    mean_p = data['mean_singletons_pseudo']
    sd_p = data['sd_singletons_pseudo']

    log_mean_g = np.log10(mean_g)
    log_sd_g = np.log10(sd_g)
    log_mean_p = np.log10(mean_p)
    log_sd_p = np.log10(sd_p)

    # Fit Taylor's law for genes
    slope_g, inter_g, r_g, p_g, se_g = linregress(log_mean_g, log_sd_g)
    r2_g = r_g**2

    # Fit Taylor's law for pseudogenes
    slope_p, inter_p, r_p, p_p, se_p = linregress(log_mean_p, log_sd_p)
    r2_p = r_p**2

    # Bootstrap CI on slope difference
    rng = np.random.default_rng(42)
    n = len(mean_g)
    n_boot = 5000
    boot_diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bg, _, _, _, _ = linregress(log_mean_g[idx], log_sd_g[idx])
        bp, _, _, _, _ = linregress(log_mean_p[idx], log_sd_p[idx])
        boot_diffs[i] = bg - bp
    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])

    # Plot: genes and pseudogenes
    ax.scatter(log_mean_g, log_sd_g, s=8, alpha=0.3, color=COLORS['blue'],
               edgecolors='none', zorder=2, label='Genes')
    ax.scatter(log_mean_p, log_sd_p, s=8, alpha=0.3, color=COLORS['orange'],
               edgecolors='none', zorder=2, label='Pseudogenes')

    # Regression lines
    x_range = np.array([min(log_mean_g.min(), log_mean_p.min()) - 0.1,
                         max(log_mean_g.max(), log_mean_p.max()) + 0.1])
    ax.plot(x_range, inter_g + slope_g * x_range, color=COLORS['blue'],
            linewidth=2, zorder=4)
    ax.plot(x_range, inter_p + slope_p * x_range, color=COLORS['orange'],
            linewidth=2, zorder=4)

    # Reference lines
    # Poisson: b = 0.5 (dashed grey)
    mid_y = np.mean([log_sd_g.mean(), log_sd_p.mean()])
    mid_x = np.mean([log_mean_g.mean(), log_mean_p.mean()])
    ax.plot(x_range, mid_y + 0.5 * (x_range - mid_x),
            '--', color=COLORS['grey'], linewidth=0.8, alpha=0.5,
            label='Poisson (b=0.5)')

    ax.set_xlabel(r'$\log_{10}$(mean singleton count)')
    ax.set_ylabel(r'$\log_{10}$(SD singleton count)')
    ax.set_title("Taylor's law: genes more tightly scaled")
    ax.legend(fontsize=6, loc='upper left', framealpha=0.9)

    ax.text(0.95, 0.05,
            f'Gene exponent b = {slope_g:.3f} (R² = {r2_g:.3f})\n'
            f'Pseudo exponent b = {slope_p:.3f} (R² = {r2_p:.3f})\n'
            f'Δb = {slope_g - slope_p:.3f}\n'
            f'Bootstrap 95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]',
            transform=ax.transAxes, fontsize=6.5, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=COLORS['grey'], alpha=0.9),
            family='monospace')

    # Store results
    data['_panel_c_results'] = {
        'slope_gene': slope_g, 'r2_gene': r2_g, 'se_gene': se_g,
        'slope_pseudo': slope_p, 'r2_pseudo': r2_p, 'se_pseudo': se_p,
        'slope_diff': slope_g - slope_p,
        'boot_ci_lo': ci_lo, 'boot_ci_hi': ci_hi,
    }


def panel_d_within_class_cv(ax, data):
    """Panel D: Within-class replication of the CV–selection gradient.

    The positive correlation between CV_gene and dN/dS should replicate
    independently in all major taxonomic classes, ruling out phylogenetic
    confounding.
    """
    print('  Panel D: Within-class replication of CV vs dN/dS...')

    from scipy.stats import spearmanr

    classes = data['taxonomic_class']
    cv_gene = data['cv_gene']
    dnds = data['dnds']

    # Find classes with ≥ 30 species
    unique_classes, counts = np.unique(classes, return_counts=True)
    big_classes = unique_classes[counts >= 30]

    # Sort by count (largest first)
    class_counts = {c: counts[unique_classes == c][0] for c in big_classes}
    big_classes = sorted(big_classes, key=lambda c: -class_counts[c])

    results = {}
    y_positions = []
    rho_values = []
    ci_los = []
    ci_his = []
    class_labels = []

    rng = np.random.default_rng(42)

    for i, cls in enumerate(big_classes):
        mask = classes == cls
        n_cls = int(mask.sum())
        cv = cv_gene[mask]
        d = dnds[mask]

        rho, p = spearmanr(cv, d)

        # Bootstrap 95% CI
        n_boot = 5000
        boot_rhos = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n_cls, size=n_cls)
            boot_rhos[b], _ = spearmanr(cv[idx], d[idx])
        ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5])

        short_name = cls.replace('c__', '')
        class_labels.append(f'{short_name}\n(n={n_cls})')
        y_positions.append(i)
        rho_values.append(rho)
        ci_los.append(ci_lo)
        ci_his.append(ci_hi)

        results[cls] = {'rho': rho, 'p': p, 'n': n_cls,
                        'ci_lo': ci_lo, 'ci_hi': ci_hi}

    y_pos = np.array(y_positions)
    rho_arr = np.array(rho_values)
    ci_lo_arr = np.array(ci_los)
    ci_hi_arr = np.array(ci_his)

    # Horizontal point-and-whisker plot
    ax.hlines(y_pos, ci_lo_arr, ci_hi_arr, color=COLORS['blue'], linewidth=2,
              alpha=0.6, zorder=2)
    ax.scatter(rho_arr, y_pos, s=50, color=COLORS['blue'], zorder=3,
               edgecolors='white', linewidth=0.5)

    ax.axvline(x=0, color='black', linewidth=0.5, linestyle=':', zorder=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_labels, fontsize=6.5)
    ax.set_xlabel('Spearman ρ (CV_gene vs dN/dS)')
    ax.set_title('CV–selection signal replicates within classes')
    ax.invert_yaxis()

    # Annotate p-values
    for i, cls in enumerate(big_classes):
        r = results[cls]
        ax.text(r['ci_hi'] + 0.02, i, f'p={r["p"]:.1e}', va='center',
                fontsize=5.5)

    data['_panel_d_results'] = results


# ==============================================================================
# Main
# ==============================================================================

def main():
    args = parse_args()

    print_header('Variance Analysis: Selection Constrains Gene Content Variability', {
        'Data source': 'Douglas & Shapiro (2024)',
        'DOI': '10.1038/s41559-023-02268-6',
    })

    setup_plotting()

    # Load real data — no synthetic, no fabrication
    print('\n  Loading real data...')
    data = load_real_data()

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))

    panel_a_mean_vs_variance(axes[0, 0], data)
    add_panel_label(axes[0, 0], 'A')

    panel_b_cv_vs_selection(axes[0, 1], data)
    add_panel_label(axes[0, 1], 'B')

    panel_c_taylors_law(axes[1, 0], data)
    add_panel_label(axes[1, 0], 'C')

    panel_d_within_class_cv(axes[1, 1], data)
    add_panel_label(axes[1, 1], 'D')

    fig.tight_layout(w_pad=3.0, h_pad=3.0)

    save_figure(fig, 'supplementary_s5_variance', output_dir=args.output_dir,
                fmt=args.format)

    if args.show:
        plt.show()
    else:
        plt.close(fig)

    # ── Summary ──────────────────────────────────────────────────────────
    from scipy.stats import spearmanr, wilcoxon

    print('\n' + '=' * 70)
    print('  SUMMARY')
    print('=' * 70)
    print(f'  Species analysed: {data["n_species"]}')
    print(f'  Data: Douglas & Shapiro (2024)')

    # --- CV comparison ---
    cv_gene_mean = np.mean(data['cv_gene'])
    cv_pseudo_mean = np.mean(data['cv_pseudo'])
    cv_gene_med = np.median(data['cv_gene'])
    cv_pseudo_med = np.median(data['cv_pseudo'])
    frac_greater = np.mean(data['cv_gene'] > data['cv_pseudo'])
    stat_w, p_w = wilcoxon(data['cv_gene'], data['cv_pseudo'],
                            alternative='greater')

    print(f'\n  Coefficient of variation (singleton counts):')
    print(f'    CV_gene:   mean = {cv_gene_mean:.3f}, median = {cv_gene_med:.3f}')
    print(f'    CV_pseudo: mean = {cv_pseudo_mean:.3f}, median = {cv_pseudo_med:.3f}')
    print(f'    CV_gene > CV_pseudo: {frac_greater*100:.1f}% of species')
    print(f'    Wilcoxon (gene > pseudo): W = {stat_w:.0f}, p = {p_w:.2e}')

    cv_ratio_med = np.median(data['cv_ratio'])
    cv_ratio_mean = np.mean(data['cv_ratio'])
    frac_above_1 = np.mean(data['cv_ratio'] > 1)
    print(f'\n  CV ratio (gene/pseudo):')
    print(f'    Mean:   {cv_ratio_mean:.3f}')
    print(f'    Median: {cv_ratio_med:.3f}')
    print(f'    CV ratio > 1: {frac_above_1*100:.1f}% of species')

    # --- Panel A ---
    a_res = data.get('_panel_a_results', {})
    if a_res:
        print(f'\n  Panel A: Mean vs variance constraint')
        print(f'    Median fluidity ratio (gene/pseudo): {a_res["median_flu_ratio"]:.3f}')
        print(f'    Median CV ratio (gene/pseudo): {a_res["median_cv_ratio"]:.3f}')
        print(f'    CV ratio vs dN/dS: rho = {a_res["rho_cv_dnds"]:.3f}, '
              f'p = {a_res["p_cv_dnds"]:.2e}')

    # --- Panel B ---
    b_res = data.get('_panel_b_results', {})
    if b_res:
        print(f'\n  Panel B: CV vs selection intensity')
        print(f'    CV_gene vs dN/dS:   rho = {b_res["rho_gene"]:.3f} (p = {b_res["p_gene"]:.2e})')
        print(f'    CV_pseudo vs dN/dS: rho = {b_res["rho_pseudo"]:.3f} (p = {b_res["p_pseudo"]:.2e})')
        print(f'    Differential:       Δρ = {b_res["delta_rho"]:.3f}')

    # --- Panel C ---
    c_res = data.get('_panel_c_results', {})
    if c_res:
        print(f'\n  Panel C: Taylor\'s law (mean-variance scaling)')
        print(f'    Gene exponent:   b = {c_res["slope_gene"]:.3f} '
              f'(R² = {c_res["r2_gene"]:.3f}, SE = {c_res["se_gene"]:.3f})')
        print(f'    Pseudo exponent: b = {c_res["slope_pseudo"]:.3f} '
              f'(R² = {c_res["r2_pseudo"]:.3f}, SE = {c_res["se_pseudo"]:.3f})')
        print(f'    Δb = {c_res["slope_diff"]:.3f}')
        print(f'    Bootstrap 95% CI: [{c_res["boot_ci_lo"]:.3f}, {c_res["boot_ci_hi"]:.3f}]')

    # --- Panel D ---
    d_res = data.get('_panel_d_results', {})
    if d_res:
        print(f'\n  Panel D: Within-class replication (CV_gene vs dN/dS)')
        for cls in sorted(d_res, key=lambda c: -d_res[c]['n']):
            r = d_res[cls]
            short = cls.replace('c__', '')
            print(f'    {short} (n={r["n"]}): '
                  f'rho = {r["rho"]:.3f} [{r["ci_lo"]:.3f}, {r["ci_hi"]:.3f}], '
                  f'p = {r["p"]:.2e}')

    print('=' * 70)


if __name__ == '__main__':
    main()
