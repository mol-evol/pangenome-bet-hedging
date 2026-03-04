#!/usr/bin/env python3
"""
================================================================================
SELECTION ANALYSIS: PANGENOME GENES ARE UNDER PURIFYING SELECTION
================================================================================

Prediction: If accessory genes are maintained by bet-hedging (functional insurance),
they should show signatures of purifying selection. Pseudogenes provide a neutral
reference: they evolve without selective constraint, so any measure of constraint
should differ between intact genes and pseudogenes.

Data source:
  Douglas & Shapiro (2024) DOI: 10.1038/s41559-023-02268-6
  670 prokaryotic species, each with:
    - Genome-wide dN/dS, dN, and dS for intact genes
    - Gene singleton rate (% gene families found in only 1 of 9 subsampled genomes)
    - Pseudogene singleton rate (same metric for pseudogenes — neutral reference)
    - Genomic fluidity (proportion of genes unique to each genome pair)
    - Pseudogene fluidity (same metric for pseudogenes — neutral reference)
    - si_sp: singleton index ratio (gene singleton rate / pseudogene singleton rate)
    - Taxonomic class (GTDB), genome size, pseudogene counts

Five-panel figure:
  Panel A: Gene vs pseudogene singleton rates (paired comparison across 670 species)
  Panel B: dN/dS vs genomic fluidity (selection strength predicts pangenome openness)
  Panel C: Core dN/dS (Ne proxy) vs si_sp (accessory selection) as independent predictors
  Panel D: Partial correlations distinguishing bet-hedging from selfish DNA
  Panel E: Within-class replication of the dN/dS–fluidity correlation

Download instructions:
  Douglas & Shapiro (2024) data from Zenodo: https://zenodo.org/records/8326664
  Key file: pangenome_and_related_metrics.tsv.gz

Usage:
    python selection_analysis.py
    python selection_analysis.py --show
    python selection_analysis.py --output-dir results --format pdf
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
        description='Test 7: Accessory Genes Show Signatures of Selection')
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
    """Load species-level selection data from Douglas & Shapiro (2024).

    Reads pangenome_and_related_metrics.tsv.gz: 670 species, each with
    genome-wide dN/dS, singleton rates (gene and pseudogene), genomic
    fluidity (gene and pseudogene), and the si_sp singleton index.

    All values are directly from the published data. Nothing is simulated,
    generated, or fabricated.

    Returns
    -------
    dict with keys:
        n_species : int
        species_names : np.ndarray of str
        dnds : np.ndarray (670,) — genome-wide dN/dS for intact genes
        dn : np.ndarray (670,) — nonsynonymous divergence
        ds : np.ndarray (670,) — synonymous divergence
        genomic_fluidity : np.ndarray (670,) — gene fluidity
        pseudogene_fluidity : np.ndarray (670,) — pseudogene fluidity
        gene_singleton_pct : np.ndarray (670,) — gene singleton % (9-genome subsample)
        pseudo_singleton_pct : np.ndarray (670,) — pseudogene singleton %
        si_sp : np.ndarray (670,) — singleton index ratio
        mean_num_genes : np.ndarray (670,) — mean gene count per genome
        mean_num_pseudo : np.ndarray (670,) — mean pseudogene count per genome
        taxonomic_class : np.ndarray of str
        genome_size : np.ndarray (670,) — mean genome size in bp
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
    dn = []
    ds = []
    genomic_fluidity = []
    pseudogene_fluidity = []
    gene_singleton_pct = []
    pseudo_singleton_pct = []
    si_sp = []
    mean_num_genes = []
    mean_num_pseudo = []
    taxonomic_class = []
    genome_size = []

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

                species_names.append(row[''])
                dnds.append(d_dnds)
                dn.append(d_dn)
                ds.append(d_ds)
                genomic_fluidity.append(float(row['genomic_fluidity']))
                pseudogene_fluidity.append(float(row['pseudogene_genomic_fluidity']))
                gene_singleton_pct.append(float(row['mean_percent_singletons_per9']))
                pseudo_singleton_pct.append(float(row['mean_percent_singletons_pseudo_per9']))
                si_sp.append(float(row['si_sp']))
                mean_num_genes.append(float(row['mean_num_genes']))
                mean_num_pseudo.append(float(row['mean_num_pseudo']))
                taxonomic_class.append(row['class'])
                genome_size.append(float(row['genome_size']))
            except (ValueError, KeyError, TypeError):
                continue

    n_species = len(species_names)
    print(f'  Loaded {n_species} species from Douglas & Shapiro (2024)')

    return {
        'n_species': n_species,
        'species_names': np.array(species_names),
        'dnds': np.array(dnds),
        'dn': np.array(dn),
        'ds': np.array(ds),
        'genomic_fluidity': np.array(genomic_fluidity),
        'pseudogene_fluidity': np.array(pseudogene_fluidity),
        'gene_singleton_pct': np.array(gene_singleton_pct),
        'pseudo_singleton_pct': np.array(pseudo_singleton_pct),
        'si_sp': np.array(si_sp),
        'mean_num_genes': np.array(mean_num_genes),
        'mean_num_pseudo': np.array(mean_num_pseudo),
        'taxonomic_class': np.array(taxonomic_class),
        'genome_size': np.array(genome_size),
    }


# ==============================================================================
# Panel functions
# ==============================================================================

def panel_a_singleton_comparison(ax, data):
    """Panel A: Gene vs pseudogene singleton rates across 670 species.

    Each species has a gene singleton rate and a pseudogene singleton rate
    (both measured by subsampling to 9 genomes). Pseudogenes are the neutral
    reference — if genes were neutral, their singleton rate would match.
    Instead, genes have ~8x fewer singletons, showing pervasive selection.
    """
    print('  Panel A: Gene vs pseudogene singleton rates...')

    gene_pct = data['gene_singleton_pct']
    pseudo_pct = data['pseudo_singleton_pct']
    n = data['n_species']

    # Paired scatter: each dot is one species
    ax.scatter(pseudo_pct, gene_pct, s=12, alpha=0.4, color=COLORS['blue'],
               edgecolors='none', zorder=3)

    # Diagonal (neutral expectation: gene rate = pseudogene rate)
    max_val = max(pseudo_pct.max(), gene_pct.max()) * 1.05
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=0.8, alpha=0.5,
            label='Neutral expectation', zorder=1)

    # Summary statistics
    gene_mean = np.mean(gene_pct)
    pseudo_mean = np.mean(pseudo_pct)
    ratio = pseudo_mean / gene_mean

    ax.set_xlabel('Pseudogene singleton rate (%)')
    ax.set_ylabel('Gene singleton rate (%)')
    ax.set_title('Genes have fewer singletons than pseudogenes')

    # Annotate
    ax.text(0.95, 0.05,
            f'n = {n} species\n'
            f'Mean gene: {gene_mean:.1f}%\n'
            f'Mean pseudo: {pseudo_mean:.1f}%\n'
            f'Ratio: {ratio:.1f}x',
            transform=ax.transAxes, fontsize=7, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=COLORS['grey'], alpha=0.9))

    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)


def outlier_robust_correlation(dnds, fluidity):
    """Compute full and outlier-trimmed Spearman correlations with bootstrap CI.

    Outliers are defined by IQR method on dN/dS: values above Q3 + 1.5*IQR.

    Returns
    -------
    dict with keys: rho_full, p_full, n_full,
                    rho_trimmed, p_trimmed, n_trimmed, n_outliers,
                    upper_fence, bootstrap_ci_lo, bootstrap_ci_hi
    """
    from scipy.stats import spearmanr

    # Full correlation
    rho_full, p_full = spearmanr(dnds, fluidity)

    # IQR-based outlier removal on dN/dS
    q1 = np.percentile(dnds, 25)
    q3 = np.percentile(dnds, 75)
    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr
    mask = dnds <= upper_fence

    rho_trimmed, p_trimmed = spearmanr(dnds[mask], fluidity[mask])
    n_trimmed = int(mask.sum())
    n_outliers = int((~mask).sum())

    # Bootstrap 95% CI on full data
    rng = np.random.default_rng(42)
    n = len(dnds)
    n_boot = 10000
    boot_rhos = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_rhos[i], _ = spearmanr(dnds[idx], fluidity[idx])
    ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5])

    return {
        'rho_full': rho_full, 'p_full': p_full, 'n_full': len(dnds),
        'rho_trimmed': rho_trimmed, 'p_trimmed': p_trimmed,
        'n_trimmed': n_trimmed, 'n_outliers': n_outliers,
        'upper_fence': upper_fence,
        'bootstrap_ci_lo': ci_lo, 'bootstrap_ci_hi': ci_hi,
    }


def panel_c_dnds_vs_fluidity(ax, data):
    """Panel C: dN/dS vs genomic fluidity across 670 species.

    If pangenome openness is shaped by selection, then species with
    stronger purifying selection (lower dN/dS) should have less open
    pangenomes (lower fluidity). This tests whether selection intensity
    predicts pangenome architecture.

    Reports both full-data and outlier-trimmed Spearman correlations,
    plus bootstrap 95% CI, because high-dN/dS outliers inflate the
    full correlation.
    """
    print('  Panel B: dN/dS vs genomic fluidity...')

    dnds = data['dnds']
    fluidity = data['genomic_fluidity']
    n = data['n_species']

    # Compute robust correlation statistics
    corr = outlier_robust_correlation(dnds, fluidity)

    # Scatter coloured by si_sp (singleton index)
    sc = ax.scatter(dnds, fluidity, s=12, alpha=0.5, c=data['si_sp'],
                    cmap='viridis_r', edgecolors='none', zorder=3,
                    vmin=0, vmax=1)

    ax.set_xlabel('dN/dS (intact genes)')
    ax.set_ylabel('Genomic fluidity')
    ax.set_title('Weaker selection → more open pangenome')

    # Reference lines
    ax.axvline(x=1.0, color=COLORS['red'], linestyle=':', linewidth=0.8,
               alpha=0.5)
    ax.text(1.02, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 0.3,
            'Neutral', fontsize=6, color=COLORS['red'], va='top')

    # Annotation: report both full and trimmed correlations
    ax.text(0.95, 0.95,
            f'n = {n} species\n'
            f'Spearman ρ = {corr["rho_full"]:.3f} (p = {corr["p_full"]:.1e})\n'
            f'Trimmed ρ = {corr["rho_trimmed"]:.3f} '
            f'(n={corr["n_trimmed"]}, {corr["n_outliers"]} outliers removed)\n'
            f'Bootstrap 95% CI: [{corr["bootstrap_ci_lo"]:.3f}, {corr["bootstrap_ci_hi"]:.3f}]',
            transform=ax.transAxes, fontsize=6.5, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=COLORS['grey'], alpha=0.9))

    # Colorbar
    cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('$s_i/s_p$ (singleton index)', fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # Store correlation results in data dict for summary access
    data['_panel_c_corr'] = corr


def panel_d_core_vs_accessory_selection(ax, data):
    """Panel D: Core dN/dS (Ne proxy) vs si_sp (accessory selection) as
    independent predictors of pangenome openness.

    Douglas & Shapiro's dN/dS is computed on core genes only (ubiquitous
    single-copy). It is therefore a proxy for effective population size (Ne),
    not a direct measure of selection on accessory genes. Their si_sp
    (gene singleton rate / pseudogene singleton rate), by contrast, directly
    measures selection on the rarest accessory genes relative to neutral
    pseudogene turnover.

    The bet-hedging model predicts that selection on accessory genes (si_sp)
    should predict pangenome openness INDEPENDENTLY of Ne (core dN/dS).
    If pangenome openness were driven entirely by Ne, si_sp should add
    nothing after controlling for core dN/dS.

    New analysis: partial correlations and variance decomposition separating
    core-gene selection (Ne proxy) from accessory-gene selection (si_sp)
    have not been reported.
    """
    print('  Panel C: Core vs accessory selection as predictors of fluidity...')

    from scipy.stats import spearmanr, rankdata

    dnds = data['dnds']       # core genes only → Ne proxy
    si_sp = data['si_sp']     # accessory gene selection relative to pseudogenes
    fluidity = data['genomic_fluidity']

    # Raw correlations with fluidity
    rho_core, p_core = spearmanr(dnds, fluidity)
    rho_si, p_si = spearmanr(si_sp, fluidity)

    # Partial correlations
    rho_si_given_core, p_si_given_core = partial_spearman(
        si_sp, fluidity, dnds)
    rho_core_given_si, p_core_given_si = partial_spearman(
        dnds, fluidity, si_sp)

    # Variance decomposition (rank-based R²)
    rf = rankdata(fluidity)
    rd = rankdata(dnds)
    rs = rankdata(si_sp)
    ss_tot = np.sum((rf - rf.mean())**2)

    # Core dN/dS alone
    coef_c = np.polyfit(rd, rf, 1)
    r2_core = 1 - np.sum((rf - np.polyval(coef_c, rd))**2) / ss_tot

    # si_sp alone
    coef_s = np.polyfit(rs, rf, 1)
    r2_si = 1 - np.sum((rf - np.polyval(coef_s, rs))**2) / ss_tot

    # Both together
    X = np.column_stack([rd, rs, np.ones(len(rd))])
    coef_both = np.linalg.lstsq(X, rf, rcond=None)[0]
    pred_both = X @ coef_both
    r2_both = 1 - np.sum((rf - pred_both)**2) / ss_tot

    # Unique contributions
    delta_si = r2_both - r2_core    # unique si_sp contribution
    delta_core = r2_both - r2_si    # unique core dN/dS contribution
    shared = r2_both - delta_si - delta_core  # shared variance

    # --- Plot: grouped bar chart ---
    categories = ['Raw ρ\nwith fluidity', 'Partial ρ\n(controlling other)']
    x_pos = np.array([0, 1.2])
    width = 0.35

    # Core dN/dS bars (blue)
    core_vals = [rho_core, rho_core_given_si]
    bars_core = ax.bar(x_pos - width/2, core_vals, width,
                       color=COLORS['blue'], alpha=0.7, label='Core dN/dS (Ne proxy)',
                       edgecolor='white', linewidth=0.5, zorder=3)

    # si_sp bars (orange)
    si_vals = [rho_si, rho_si_given_core]
    bars_si = ax.bar(x_pos + width/2, si_vals, width,
                     color=COLORS['orange'], alpha=0.7, label='$s_i/s_p$ (accessory selection)',
                     edgecolor='white', linewidth=0.5, zorder=3)

    ax.axhline(y=0, color='black', linewidth=0.5, zorder=1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, fontsize=7)
    ax.set_ylabel('Spearman ρ with fluidity')
    ax.set_title('Accessory selection predicts openness\nbeyond $N_e$')

    # p-value annotations
    p_vals_core = [p_core, p_core_given_si]
    p_vals_si = [p_si, p_si_given_core]
    for i in range(2):
        # Core bar annotation
        y_c = core_vals[i]
        ax.text(x_pos[i] - width/2, y_c - 0.03, f'ρ={y_c:.2f}',
                ha='center', va='top', fontsize=5.5)
        # si_sp bar annotation
        y_s = si_vals[i]
        ax.text(x_pos[i] + width/2, y_s + 0.02, f'ρ={y_s:.2f}',
                ha='center', va='bottom', fontsize=5.5)

    ax.legend(fontsize=6, loc='lower left', framealpha=0.9)
    ax.set_ylim(-0.5, 0.65)

    # Variance annotation box
    ax.text(0.98, 0.05,
            f'Rank R² decomposition:\n'
            f'  Core dN/dS alone: {r2_core:.3f}\n'
            f'  $s_i/s_p$ alone: {r2_si:.3f}\n'
            f'  Both: {r2_both:.3f}\n'
            f'  Unique $s_i/s_p$: +{delta_si:.3f}\n'
            f'  Unique core: +{delta_core:.3f}',
            transform=ax.transAxes, fontsize=5.5, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=COLORS['grey'], alpha=0.9),
            family='monospace')

    # Store results
    data['_panel_d_results'] = {
        'rho_core_raw': rho_core, 'p_core_raw': p_core,
        'rho_si_raw': rho_si, 'p_si_raw': p_si,
        'rho_si_given_core': rho_si_given_core, 'p_si_given_core': p_si_given_core,
        'rho_core_given_si': rho_core_given_si, 'p_core_given_si': p_core_given_si,
        'r2_core': r2_core, 'r2_si': r2_si, 'r2_both': r2_both,
        'delta_si': delta_si, 'delta_core': delta_core,
    }


def partial_spearman(x, y, z):
    """Spearman partial correlation of x and y controlling for z.

    Rank-based: residualize ranks of x and y on ranks of z,
    then correlate residuals.
    """
    from scipy.stats import spearmanr, rankdata

    rx = rankdata(x)
    ry = rankdata(y)
    rz = rankdata(z)

    # Residualize both on z
    cx = np.polyfit(rz, rx, 1)
    cy = np.polyfit(rz, ry, 1)
    rx_resid = rx - np.polyval(cx, rz)
    ry_resid = ry - np.polyval(cy, rz)

    return spearmanr(rx_resid, ry_resid)


def panel_e_partial_correlations(ax, data):
    """Panel E: Partial correlations distinguishing bet-hedging from selfish DNA.

    The selfish DNA hypothesis predicts that pangenome openness is driven by
    selfish element (pseudogene) dynamics. If true, controlling for pseudogene
    burden should eliminate the dN/dS–fluidity signal.

    The bet-hedging hypothesis predicts that selection intensity drives pangenome
    openness independently of pseudogene dynamics.

    New analysis: partial correlations controlling for pseudogene burden and
    neutral divergence (dS).
    """
    print('  Panel D: Partial correlations (bet-hedging vs selfish DNA)...')

    from scipy.stats import spearmanr

    dnds = data['dnds']
    fluidity = data['genomic_fluidity']
    ds = data['ds']
    pseudo_ratio = data['mean_num_pseudo'] / data['mean_num_genes']

    # Four correlations
    labels = []
    rhos = []
    errs = []  # bootstrap SE

    rng = np.random.default_rng(42)
    n = len(dnds)
    n_boot = 5000

    # 1. Raw dN/dS vs fluidity
    rho_raw, p_raw = spearmanr(dnds, fluidity)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i], _ = spearmanr(dnds[idx], fluidity[idx])
    labels.append('dN/dS vs fluidity\n(raw)')
    rhos.append(rho_raw)
    errs.append(np.std(boot))

    # 2. dN/dS vs fluidity | pseudogene burden
    rho_pseudo, p_pseudo = partial_spearman(dnds, fluidity, pseudo_ratio)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i], _ = partial_spearman(dnds[idx], fluidity[idx], pseudo_ratio[idx])
    labels.append('dN/dS vs fluidity\n| pseudo burden')
    rhos.append(rho_pseudo)
    errs.append(np.std(boot))

    # 3. dN/dS vs fluidity | dS
    rho_ds, p_ds = partial_spearman(dnds, fluidity, ds)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i], _ = partial_spearman(dnds[idx], fluidity[idx], ds[idx])
    labels.append('dN/dS vs fluidity\n| $d_S$')
    rhos.append(rho_ds)
    errs.append(np.std(boot))

    # 4. Pseudo/gene ratio vs fluidity | dN/dS
    rho_rev, p_rev = partial_spearman(pseudo_ratio, fluidity, dnds)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i], _ = partial_spearman(pseudo_ratio[idx], fluidity[idx], dnds[idx])
    labels.append('Pseudo burden\nvs fluidity | dN/dS')
    rhos.append(rho_rev)
    errs.append(np.std(boot))

    # Bar chart
    x_pos = np.arange(len(labels))
    bar_colors = [COLORS['blue'], COLORS['blue'], COLORS['blue'], COLORS['orange']]
    bars = ax.bar(x_pos, rhos, yerr=[1.96 * e for e in errs],
                  color=bar_colors, alpha=0.7, capsize=4, edgecolor='white',
                  linewidth=0.5, zorder=3)

    ax.axhline(y=0, color='black', linewidth=0.5, zorder=1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel('Spearman ρ')
    ax.set_title('Selection signal robust to controls')

    # p-value annotations
    p_vals = [p_raw, p_pseudo, p_ds, p_rev]
    for i, (rho, p) in enumerate(zip(rhos, p_vals)):
        y_off = 0.02 if rho >= 0 else -0.04
        ax.text(i, rho + y_off + (1.96 * errs[i] if rho >= 0 else -1.96 * errs[i]),
                f'ρ={rho:.3f}\np={p:.1e}', ha='center', va='bottom' if rho >= 0 else 'top',
                fontsize=5.5)

    ax.set_ylim(-0.55, 0.35)

    # Store results
    data['_panel_e_results'] = {
        'rho_raw': rho_raw, 'p_raw': p_raw,
        'rho_partial_pseudo': rho_pseudo, 'p_partial_pseudo': p_pseudo,
        'rho_partial_ds': rho_ds, 'p_partial_ds': p_ds,
        'rho_pseudo_vs_flu_given_dnds': rho_rev, 'p_pseudo_vs_flu_given_dnds': p_rev,
    }


def panel_f_within_class(ax, data):
    """Panel F: dN/dS–fluidity correlation within each major taxonomic class.

    If the overall correlation were driven by phylogenetic non-independence
    (e.g., one clade with both low dN/dS and high fluidity), it would
    disappear within classes. Instead, it replicates independently in
    all six classes with ≥30 species.

    New analysis: Douglas & Shapiro do not stratify by class.
    """
    print('  Panel E: Within-class replication of dN/dS vs fluidity...')

    from scipy.stats import spearmanr

    classes = data['taxonomic_class']
    dnds = data['dnds']
    fluidity = data['genomic_fluidity']

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
        d = dnds[mask]
        f = fluidity[mask]

        rho, p = spearmanr(d, f)

        # Bootstrap 95% CI
        n_boot = 5000
        boot_rhos = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n_cls, size=n_cls)
            boot_rhos[b], _ = spearmanr(d[idx], f[idx])
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
    ax.set_xlabel('Spearman ρ (dN/dS vs fluidity)')
    ax.set_title('Signal replicates within classes')
    ax.invert_yaxis()

    # Annotate p-values
    for i, cls in enumerate(big_classes):
        r = results[cls]
        ax.text(r['ci_hi'] + 0.02, i, f'p={r["p"]:.1e}', va='center',
                fontsize=5.5)

    data['_panel_f_results'] = results


# ==============================================================================
# Main
# ==============================================================================

def main():
    args = parse_args()

    print_header('Selection Analysis: Pangenome Genes Under Purifying Selection', {
        'Data source': 'Douglas & Shapiro (2024)',
        'DOI': '10.1038/s41559-023-02268-6',
    })

    setup_plotting()

    # Load real data — no synthetic, no fabrication
    print('\n  Loading real data...')
    data = load_real_data()

    # Create 5-panel figure: 2 on top, 3 on bottom
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))

    # Row 1: Baseline + correlation (+ empty slot)
    panel_a_singleton_comparison(axes[0, 0], data)
    add_panel_label(axes[0, 0], 'A')

    panel_c_dnds_vs_fluidity(axes[0, 1], data)
    add_panel_label(axes[0, 1], 'B')

    panel_d_core_vs_accessory_selection(axes[0, 2], data)
    add_panel_label(axes[0, 2], 'C')

    # Row 2: New analyses
    panel_e_partial_correlations(axes[1, 0], data)
    add_panel_label(axes[1, 0], 'D')

    panel_f_within_class(axes[1, 1], data)
    add_panel_label(axes[1, 1], 'E')

    # Hide empty panel
    axes[1, 2].set_visible(False)

    fig.tight_layout(w_pad=3.0, h_pad=3.0)

    save_figure(fig, 'supplementary_s4_selection', output_dir=args.output_dir,
                fmt=args.format)

    if args.show:
        plt.show()
    else:
        plt.close(fig)

    # ── Summary ──────────────────────────────────────────────────────────
    from scipy.stats import wilcoxon, spearmanr, mannwhitneyu

    print('\n' + '=' * 70)
    print('  SUMMARY')
    print('=' * 70)
    print(f'  Species analysed: {data["n_species"]}')
    print(f'  Data: Douglas & Shapiro (2024)')

    # --- Singleton comparison ---
    gene_mean = np.mean(data['gene_singleton_pct'])
    pseudo_mean = np.mean(data['pseudo_singleton_pct'])
    ratio = pseudo_mean / gene_mean
    print(f'\n  Singleton rates (subsampled to 9 genomes):')
    print(f'    Gene singleton rate:       {gene_mean:.2f}%')
    print(f'    Pseudogene singleton rate: {pseudo_mean:.2f}%')
    print(f'    Ratio (pseudo/gene):       {ratio:.1f}x')

    stat_s, p_s = wilcoxon(data['gene_singleton_pct'],
                           data['pseudo_singleton_pct'],
                           alternative='less')
    print(f'    Wilcoxon signed-rank (gene < pseudo): W = {stat_s:.0f}, p = {p_s:.2e}')

    frac_lower = np.mean(data['gene_singleton_pct'] < data['pseudo_singleton_pct'])
    print(f'    Species where gene < pseudo: {frac_lower*100:.1f}%')

    # --- Fluidity comparison ---
    gene_flu_mean = np.mean(data['genomic_fluidity'])
    pseudo_flu_mean = np.mean(data['pseudogene_fluidity'])
    print(f'\n  Genomic fluidity:')
    print(f'    Gene fluidity:       {gene_flu_mean:.4f}')
    print(f'    Pseudogene fluidity: {pseudo_flu_mean:.4f}')
    print(f'    Ratio (pseudo/gene): {pseudo_flu_mean/gene_flu_mean:.2f}x')

    stat_f, p_f = wilcoxon(data['genomic_fluidity'],
                           data['pseudogene_fluidity'],
                           alternative='less')
    print(f'    Wilcoxon signed-rank (gene < pseudo): W = {stat_f:.0f}, p = {p_f:.2e}')

    frac_lower_f = np.mean(data['genomic_fluidity'] < data['pseudogene_fluidity'])
    print(f'    Species where gene < pseudo: {frac_lower_f*100:.1f}%')

    # --- dN/dS ---
    print(f'\n  dN/dS across {data["n_species"]} species:')
    print(f'    Mean:   {np.mean(data["dnds"]):.3f}')
    print(f'    Median: {np.median(data["dnds"]):.3f}')
    print(f'    Range:  {np.min(data["dnds"]):.3f} -- {np.max(data["dnds"]):.3f}')
    frac_purifying = np.mean(data['dnds'] < 1.0)
    frac_strong = np.mean(data['dnds'] < 0.3)
    print(f'    dN/dS < 1.0 (purifying):   {frac_purifying*100:.1f}% of species')
    print(f'    dN/dS < 0.3 (strong purif): {frac_strong*100:.1f}% of species')

    # --- dN/dS vs fluidity correlation (with outlier robustness) ---
    corr = data.get('_panel_c_corr')
    if corr is None:
        corr = outlier_robust_correlation(data['dnds'], data['genomic_fluidity'])
    print(f'\n  dN/dS vs genomic fluidity:')
    print(f'    Spearman rho = {corr["rho_full"]:.3f}, p = {corr["p_full"]:.2e}')
    print(f'    Outlier-trimmed (IQR, n={corr["n_trimmed"]}): '
          f'rho = {corr["rho_trimmed"]:.3f}, p = {corr["p_trimmed"]:.2e}')
    print(f'    Outliers removed: {corr["n_outliers"]} '
          f'(dN/dS > {corr["upper_fence"]:.3f})')
    print(f'    Bootstrap 95% CI: [{corr["bootstrap_ci_lo"]:.3f}, '
          f'{corr["bootstrap_ci_hi"]:.3f}]')

    # --- si_sp summary ---
    si_mean = np.mean(data['si_sp'])
    si_median = np.median(data['si_sp'])
    frac_below_1 = np.mean(data['si_sp'] < 1.0)
    print(f'\n  Singleton index (si_sp = gene_singleton / pseudo_singleton):')
    print(f'    Mean:   {si_mean:.3f}')
    print(f'    Median: {si_median:.3f}')
    print(f'    si_sp < 1.0: {frac_below_1*100:.1f}% of species')

    # --- Panel D: Core vs accessory selection ---
    d_res = data.get('_panel_d_results', {})
    if d_res:
        print(f'\n  Core dN/dS (Ne proxy) vs si_sp (accessory selection):')
        print(f'    Note: dN/dS is computed on CORE genes only (ubiquitous single-copy)')
        print(f'    Note: si_sp measures selection on the RAREST accessory genes')
        print(f'    Raw correlations with fluidity:')
        print(f'      Core dN/dS: rho = {d_res["rho_core_raw"]:.3f}, p = {d_res["p_core_raw"]:.2e}')
        print(f'      si_sp:      rho = {d_res["rho_si_raw"]:.3f}, p = {d_res["p_si_raw"]:.2e}')
        print(f'    Partial correlations:')
        print(f'      si_sp | core dN/dS:  rho = {d_res["rho_si_given_core"]:.3f}, '
              f'p = {d_res["p_si_given_core"]:.2e}')
        print(f'      Core dN/dS | si_sp:  rho = {d_res["rho_core_given_si"]:.3f}, '
              f'p = {d_res["p_core_given_si"]:.2e}')
        print(f'    Rank-based R² decomposition:')
        print(f'      Core dN/dS alone: {d_res["r2_core"]:.3f}')
        print(f'      si_sp alone:      {d_res["r2_si"]:.3f}')
        print(f'      Both together:    {d_res["r2_both"]:.3f}')
        print(f'      Unique si_sp:     +{d_res["delta_si"]:.3f} '
              f'({d_res["delta_si"]*100:.1f}% additional variance)')
        print(f'      Unique core:      +{d_res["delta_core"]:.3f} '
              f'({d_res["delta_core"]*100:.1f}% additional variance)')

    # --- Panel E: Partial correlations ---
    e_res = data.get('_panel_e_results', {})
    if e_res:
        print(f'\n  Partial correlations (distinguishing bet-hedging from selfish DNA):')
        print(f'    dN/dS vs fluidity (raw):           '
              f'rho = {e_res["rho_raw"]:.3f}, p = {e_res["p_raw"]:.2e}')
        print(f'    dN/dS vs fluidity | pseudo burden:  '
              f'rho = {e_res["rho_partial_pseudo"]:.3f}, p = {e_res["p_partial_pseudo"]:.2e}')
        print(f'    dN/dS vs fluidity | dS:             '
              f'rho = {e_res["rho_partial_ds"]:.3f}, p = {e_res["p_partial_ds"]:.2e}')
        print(f'    Pseudo burden vs fluidity | dN/dS:  '
              f'rho = {e_res["rho_pseudo_vs_flu_given_dnds"]:.3f}, '
              f'p = {e_res["p_pseudo_vs_flu_given_dnds"]:.2e}')

    # --- Panel F: Within-class replication ---
    f_res = data.get('_panel_f_results', {})
    if f_res:
        print(f'\n  Within-class replication (dN/dS vs fluidity):')
        for cls in sorted(f_res, key=lambda c: -f_res[c]['n']):
            r = f_res[cls]
            short = cls.replace('c__', '')
            print(f'    {short} (n={r["n"]}): '
                  f'rho = {r["rho"]:.3f} [{r["ci_lo"]:.3f}, {r["ci_hi"]:.3f}], '
                  f'p = {r["p"]:.2e}')

    print('=' * 70)


if __name__ == '__main__':
    main()
