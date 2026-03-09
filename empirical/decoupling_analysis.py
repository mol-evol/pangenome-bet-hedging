#!/usr/bin/env python3
"""
================================================================================
GENE-ENVIRONMENT DECOUPLING ANALYSIS (Test 4)
================================================================================

PURPOSE
-------
Test whether accessory gene content is COUPLED to isolation environment
(local adaptation) or DECOUPLED from it (bet-hedging).

This is the key discriminating test between the two hypotheses:
  - Bet-hedging: gene content is stochastic insurance → genomes from the
    SAME environment still differ in accessory genes → DECOUPLED
  - Local adaptation: gene content tracks niche → genomes from the
    SAME environment carry similar accessory genes → COUPLED

DATA
----
Horesh et al. (2021) — 10,146 E. coli genomes with:
  - Gene presence/absence matrix (Roary pangenome)
  - Isolation source metadata (blood, faeces, urine)
  - Phylogroup assignments

Figshare: https://microbiology.figshare.com/articles/dataset/13270073
Reference: DOI: 10.1099/mgen.0.000499

FIGURE
------
4-panel (2 × 2):
  A: Jaccard distance distributions (same vs different source)
  B: PERMANOVA variance decomposition
  C: Heatmap of mean Jaccard between source categories
  D: Effect size across phylogroups (forest plot)

A PRIORI THRESHOLDS (stated before seeing results):
  - <1% variance by isolation source = "decoupled" (bet-hedging)
  - 1-5% = ambiguous
  - >5% = "coupled" (local adaptation)

USAGE
-----
  python decoupling_analysis.py [--output-dir DIR] [--format FMT] [--show]

================================================================================
"""

import os
import sys
import argparse
import csv
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
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mannwhitneyu


# ==============================================================================
# Data loading — real data only, no fabrication
# ==============================================================================

def load_horesh_data():
    """Load Horesh et al. (2021) E. coli gene presence/absence + metadata.

    Reads:
      1. F1_genome_metadata.csv — isolation source, phylogroup, genome ID
      2. F4_complete_presence_absence.csv — binary gene PA matrix

    Filters:
      - Removes genomes with unknown isolation source or phylogroup
      - Keeps only body-site sources with ≥50 genomes (Blood, Feces, Urine)
      - Filters genes to accessory range (5-95% frequency)

    Returns dict with numpy arrays for analysis.
    """
    data_dir = os.path.join(_CODE_DIR, 'data', 'horesh')

    # ------------------------------------------------------------------
    # 1. Load metadata
    # ------------------------------------------------------------------
    print('  [1/4] Loading metadata...')
    meta_path = os.path.join(data_dir, 'F1_genome_metadata.csv')
    metadata = {}  # genome_id -> {'source': ..., 'phylogroup': ...}

    # Map isolation source to standardised categories
    source_map = {
        'Blood': 'Blood',
        'Feces': 'Feces',
        'Urine': 'Urine',
    }

    with open(meta_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            source = row['Isolation'].strip()
            phylogroup = row['Phylogroup'].strip()

            # Only keep genomes with known source and phylogroup
            if source not in source_map:
                continue
            if phylogroup in ('Not Determined', '', 'Unknown'):
                continue

            # The PA matrix uses 'name_in_presence_absence' as genome ID
            pa_name = row.get('name_in_presence_absence', '').strip()
            if not pa_name or pa_name == 'NA':
                continue  # genome not in PA matrix

            metadata[pa_name] = {
                'source': source_map[source],
                'phylogroup': phylogroup,
            }

    print(f'    {len(metadata)} genomes with valid source + phylogroup')

    # ------------------------------------------------------------------
    # 2. Load gene presence/absence matrix
    # ------------------------------------------------------------------
    print('  [2/4] Loading gene presence/absence matrix (this may take a moment)...')
    pa_path = os.path.join(data_dir, 'F4_complete_presence_absence.csv')

    # First pass: read header to get genome column names
    with open(pa_path, 'r') as f:
        header = f.readline().strip().split(',')

    # The header format depends on how Roary outputs CSV.
    # Typically: Gene,genome1,genome2,...
    # First column is gene name, rest are genomes
    # But Roary's gene_presence_absence.csv has more metadata columns.
    # Let's detect the format.

    # Check if this is Roary format (has extra columns like Non-unique Gene name, etc.)
    # or a simple binary matrix
    # Read first data line to understand
    with open(pa_path, 'r') as f:
        header_line = f.readline().strip()
        first_data = f.readline().strip()

    header_fields = header_line.split(',')

    # Detect format: if first few values in data line are non-numeric text,
    # this is Roary format with gene annotation columns
    first_values = first_data.split(',')

    # Find where numeric (0/1) data starts
    # Roary CSV: Gene, Non-unique Gene name, Annotation, No. isolates,
    #            No. sequences, Avg sequences per isolate, Genome Fragment,
    #            Order within Fragment, Accessory Fragment, Accessory Order with Fragment,
    #            QC, Min group size nuc, Max group size nuc, Avg group size nuc,
    #            then genome columns...
    # Simple CSV: Gene, genome1, genome2, ...

    # Try to detect: check if header[1] looks like a genome name
    # If it contains typical ID patterns (ESC_, esc_, GCA_) → simple format
    # Otherwise → Roary format

    # Check a few columns for 0/1 pattern
    numeric_start = None
    for i in range(1, min(20, len(first_values))):
        val = first_values[i].strip().strip('"')
        if val in ('0', '1', ''):
            if numeric_start is None:
                numeric_start = i
        else:
            numeric_start = None  # Reset — need contiguous 0/1 columns

    if numeric_start is None:
        # Fallback: look for the first genome column by matching to metadata
        for i, col in enumerate(header_fields):
            col_clean = col.strip().strip('"')
            if col_clean in metadata:
                numeric_start = i
                break

    if numeric_start is None:
        # Last resort: assume Roary format with 14 annotation columns
        numeric_start = 14

    print(f'    Genome columns start at index {numeric_start}')

    # Get genome column names (from header)
    genome_columns = [h.strip().strip('"') for h in header_fields[numeric_start:]]

    # Find which columns match our filtered metadata
    col_indices = []  # indices into the CSV columns (relative to numeric_start)
    col_genomes = []  # genome names in order
    for idx, genome in enumerate(genome_columns):
        if genome in metadata:
            col_indices.append(idx)
            col_genomes.append(genome)

    print(f'    {len(col_genomes)} of {len(genome_columns)} PA columns match metadata')

    if len(col_genomes) < 1000:
        print(f'    WARNING: Only {len(col_genomes)} matches — checking alternate ID format...')
        # Try lowercase matching
        meta_lower = {k.lower(): k for k in metadata}
        col_indices = []
        col_genomes = []
        for idx, genome in enumerate(genome_columns):
            g_lower = genome.lower()
            if g_lower in meta_lower:
                col_indices.append(idx)
                col_genomes.append(meta_lower[g_lower])
            elif genome in metadata:
                col_indices.append(idx)
                col_genomes.append(genome)
        print(f'    After case-insensitive match: {len(col_genomes)}')

    # Second pass: read PA data for matched genomes
    print('  [3/4] Reading PA matrix for matched genomes...')
    gene_names = []
    pa_rows = []
    n_genomes = len(col_genomes)

    # Pre-compute absolute indices for speed
    abs_indices = [numeric_start + ci for ci in col_indices]

    with open(pa_path, 'r') as f:
        f.readline()  # skip header
        for line_num, line in enumerate(f):
            fields = line.strip().split(',')
            gene_name = fields[0].strip().strip('"')

            # Skip metadata rows (e.g. "Lineage" row in Horesh data)
            if gene_name.lower() in ('lineage', 'phylogroup', 'source',
                                       'isolation', 'st', ''):
                continue

            # Extract values for matched columns
            row = np.zeros(n_genomes, dtype=np.int8)
            valid = True
            for out_idx, abs_idx in enumerate(abs_indices):
                if abs_idx < len(fields):
                    val = fields[abs_idx].strip().strip('"')
                    if val == '1':
                        row[out_idx] = 1
                    elif val == '0' or val == '':
                        row[out_idx] = 0
                    else:
                        # Non-binary value — skip this gene
                        valid = False
                        break

            if valid:
                gene_names.append(gene_name)
                pa_rows.append(row)

            if (line_num + 1) % 10000 == 0:
                sys.stdout.write(f'\r    Read {line_num + 1} genes...')
                sys.stdout.flush()

    print(f'\r    Read {len(gene_names)} total genes for {n_genomes} genomes')

    # Stack into matrix (genes × genomes), then transpose to (genomes × genes)
    pa_matrix = np.array(pa_rows, dtype=np.int8).T  # now (genomes, genes)

    # ------------------------------------------------------------------
    # 3. Filter to accessory genes (5-95% frequency)
    # ------------------------------------------------------------------
    print('  [4/4] Filtering to accessory genes (5-95% frequency)...')
    gene_freqs = pa_matrix.mean(axis=0)  # frequency per gene
    accessory_mask = (gene_freqs >= 0.05) & (gene_freqs <= 0.95)
    pa_accessory = pa_matrix[:, accessory_mask]
    freq_accessory = gene_freqs[accessory_mask]
    gene_names_accessory = np.array(gene_names)[accessory_mask]

    print(f'    {accessory_mask.sum()} accessory genes (of {len(gene_freqs)} total)')

    # Build output arrays
    sources = np.array([metadata[g]['source'] for g in col_genomes])
    phylogroups = np.array([metadata[g]['phylogroup'] for g in col_genomes])

    source_counts = dict(Counter(sources))
    phylogroup_counts = dict(Counter(phylogroups))

    return {
        'n_genomes': n_genomes,
        'n_accessory_genes': int(accessory_mask.sum()),
        'genome_ids': np.array(col_genomes),
        'isolation_source': sources,
        'phylogroup': phylogroups,
        'pa_matrix': pa_accessory,
        'gene_frequencies': freq_accessory,
        'gene_names': gene_names_accessory,
        'source_counts': source_counts,
        'phylogroup_counts': phylogroup_counts,
    }


# ==============================================================================
# Statistical utilities
# ==============================================================================

def compute_jaccard_within_phylogroup(data, phylogroup, max_genomes=5000):
    """Compute pairwise Jaccard distances within a phylogroup.

    Parameters
    ----------
    data : dict from load_horesh_data()
    phylogroup : str
    max_genomes : int
        Subsample if phylogroup has more genomes (default 5000 = effectively all)

    Returns
    -------
    dict with 'distances' (condensed), 'source_i', 'source_j' (pair labels),
         'dm_square' (square distance matrix), 'sources' (per-genome)
    """
    mask = data['phylogroup'] == phylogroup
    pa = data['pa_matrix'][mask]
    sources = data['isolation_source'][mask]

    # Subsample if too large
    n = pa.shape[0]
    if n > max_genomes:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=max_genomes, replace=False)
        pa = pa[idx]
        sources = sources[idx]
        n = max_genomes

    # Compute pairwise Jaccard distances
    dists = pdist(pa.astype(float), metric='jaccard')

    # Build pair source labels
    source_i = []
    source_j = []
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            source_i.append(sources[i])
            source_j.append(sources[j])
            k += 1

    dm_square = squareform(dists)

    return {
        'distances': dists,
        'source_i': np.array(source_i),
        'source_j': np.array(source_j),
        'dm_square': dm_square,
        'sources': sources,
        'n': n,
    }


def compare_same_vs_different(distances, source_i, source_j):
    """Compare Jaccard distances between same-source and different-source pairs.

    Returns dict with U, p, effect_size (rank-biserial), medians.
    """
    same_mask = source_i == source_j
    diff_mask = ~same_mask

    same_dists = distances[same_mask]
    diff_dists = distances[diff_mask]

    if len(same_dists) < 10 or len(diff_dists) < 10:
        return {'U': np.nan, 'p': np.nan, 'effect_size': np.nan,
                'median_same': np.nan, 'median_diff': np.nan,
                'n_same': len(same_dists), 'n_diff': len(diff_dists)}

    U, p = mannwhitneyu(same_dists, diff_dists, alternative='two-sided')

    # Rank-biserial effect size: r = 1 - (2U)/(n1*n2)
    n1, n2 = len(same_dists), len(diff_dists)
    r_rb = 1 - (2 * U) / (n1 * n2)

    return {
        'U': U,
        'p': p,
        'effect_size': r_rb,
        'median_same': np.median(same_dists),
        'median_diff': np.median(diff_dists),
        'n_same': n1,
        'n_diff': n2,
    }


def permanova_one_factor(dm_square, groups, n_perms=999):
    """One-factor PERMANOVA: partition variance in distance matrix by groups.

    Anderson (2001) method.

    Parameters
    ----------
    dm_square : (n, n) distance matrix
    groups : (n,) group labels
    n_perms : int

    Returns
    -------
    dict with F, p, R2
    """
    n = len(groups)
    unique_groups = np.unique(groups)

    # Compute SS_total: sum of squared distances / n
    ss_total = np.sum(dm_square ** 2) / (2 * n)

    # Compute SS_within (sum over groups)
    ss_within = 0.0
    for g in unique_groups:
        mask = groups == g
        n_g = mask.sum()
        if n_g > 1:
            dm_g = dm_square[np.ix_(mask, mask)]
            ss_within += np.sum(dm_g ** 2) / (2 * n_g)

    ss_between = ss_total - ss_within

    # Degrees of freedom
    a = len(unique_groups)
    df_between = a - 1
    df_within = n - a

    if df_between == 0 or df_within == 0:
        return {'F': np.nan, 'p': np.nan, 'R2': np.nan}

    # Pseudo-F
    f_obs = (ss_between / df_between) / (ss_within / df_within)
    r2 = ss_between / ss_total

    # Permutation test
    rng = np.random.default_rng(42)
    n_ge = 0
    for _ in range(n_perms):
        perm_groups = rng.permutation(groups)
        ss_within_perm = 0.0
        for g in unique_groups:
            mask = perm_groups == g
            n_g = mask.sum()
            if n_g > 1:
                dm_g = dm_square[np.ix_(mask, mask)]
                ss_within_perm += np.sum(dm_g ** 2) / (2 * n_g)
        ss_between_perm = ss_total - ss_within_perm
        f_perm = (ss_between_perm / df_between) / (ss_within_perm / df_within)
        if f_perm >= f_obs:
            n_ge += 1

    p = (n_ge + 1) / (n_perms + 1)

    return {'F': f_obs, 'p': p, 'R2': r2}


def run_global_permanova(data, max_per_group=5000, n_perms=999):
    """Run two PERMANOVAs: one for phylogroup effect, one for source effect.

    Uses all genomes by default (max_per_group=5000 effectively no cap).
    """
    # Subsample: take max_per_group genomes per phylogroup (effectively all)
    rng = np.random.default_rng(42)
    indices = []
    for pg in np.unique(data['phylogroup']):
        pg_idx = np.where(data['phylogroup'] == pg)[0]
        if len(pg_idx) > max_per_group:
            pg_idx = rng.choice(pg_idx, size=max_per_group, replace=False)
        indices.extend(pg_idx)

    indices = np.array(sorted(indices))
    pa = data['pa_matrix'][indices].astype(float)
    sources = data['isolation_source'][indices]
    phylogroups = data['phylogroup'][indices]

    # Compute distance matrix
    dists = pdist(pa, metric='jaccard')
    dm = squareform(dists)

    # PERMANOVA for phylogroup
    r_pg = permanova_one_factor(dm, phylogroups, n_perms=n_perms)

    # PERMANOVA for source
    r_src = permanova_one_factor(dm, sources, n_perms=n_perms)

    return {
        'R2_phylogroup': r_pg['R2'],
        'F_phylogroup': r_pg['F'],
        'p_phylogroup': r_pg['p'],
        'R2_source': r_src['R2'],
        'F_source': r_src['F'],
        'p_source': r_src['p'],
        'n_genomes': len(indices),
    }


def compute_effect_per_phylogroup(data, max_genomes=5000, n_perms=999):
    """Compute environment effect size within each phylogroup separately."""
    results = []
    for pg in sorted(data['phylogroup_counts'].keys()):
        n_pg = data['phylogroup_counts'][pg]
        # Need at least 2 source categories with 10+ genomes each
        mask = data['phylogroup'] == pg
        sources_in_pg = data['isolation_source'][mask]
        source_counts = Counter(sources_in_pg)
        usable = sum(1 for c in source_counts.values() if c >= 10)

        if usable < 2 or n_pg < 50:
            results.append({
                'phylogroup': pg, 'n': n_pg, 'computable': False,
                'R2': np.nan, 'F': np.nan, 'p': np.nan,
                'effect_size': np.nan, 'ci_low': np.nan, 'ci_high': np.nan,
            })
            continue

        # Compute Jaccard and effect
        jac = compute_jaccard_within_phylogroup(data, pg,
                                                 max_genomes=max_genomes)
        perm = permanova_one_factor(jac['dm_square'], jac['sources'],
                                     n_perms=n_perms)
        comp = compare_same_vs_different(
            jac['distances'], jac['source_i'], jac['source_j'])

        # Bootstrap CI for effect size
        rng = np.random.default_rng(42)
        boot_effects = []
        same_mask = jac['source_i'] == jac['source_j']
        diff_mask = ~same_mask
        same_d = jac['distances'][same_mask]
        diff_d = jac['distances'][diff_mask]
        for _ in range(200):
            s_boot = rng.choice(same_d, size=len(same_d), replace=True)
            d_boot = rng.choice(diff_d, size=len(diff_d), replace=True)
            u_boot, _ = mannwhitneyu(s_boot, d_boot, alternative='two-sided')
            r_boot = 1 - (2 * u_boot) / (len(s_boot) * len(d_boot))
            boot_effects.append(r_boot)

        ci_low, ci_high = np.percentile(boot_effects, [2.5, 97.5])

        results.append({
            'phylogroup': pg, 'n': jac['n'], 'computable': True,
            'R2': perm['R2'], 'F': perm['F'], 'p': perm['p'],
            'effect_size': comp['effect_size'],
            'ci_low': ci_low, 'ci_high': ci_high,
        })

    return results


# ==============================================================================
# Panel functions
# ==============================================================================

def panel_a_jaccard_distributions(ax, data):
    """Panel A: Jaccard distance distributions — same vs different source.

    Within the largest phylogroup, compare distributions of pairwise
    Jaccard distances between genomes from the same vs different
    isolation sources.
    """
    print('  Panel A: Jaccard distance distributions...')

    counts = data['phylogroup_counts']
    largest = max(counts, key=counts.get)

    jac = compute_jaccard_within_phylogroup(data, largest, max_genomes=500)
    comp = compare_same_vs_different(
        jac['distances'], jac['source_i'], jac['source_j'])

    same_mask = jac['source_i'] == jac['source_j']
    same_d = jac['distances'][same_mask]
    diff_d = jac['distances'][~same_mask]

    # Histograms
    bins = np.linspace(
        min(same_d.min(), diff_d.min()),
        max(same_d.max(), diff_d.max()),
        50)
    ax.hist(same_d, bins=bins, density=True, alpha=0.6, color=COLORS['blue'],
            label=f'Same source (n={comp["n_same"]:,})', edgecolor='none')
    ax.hist(diff_d, bins=bins, density=True, alpha=0.6, color=COLORS['orange'],
            label=f'Different source (n={comp["n_diff"]:,})', edgecolor='none')

    # Median lines
    ax.axvline(comp['median_same'], color=COLORS['blue'], linestyle='--',
               linewidth=1.5, alpha=0.8)
    ax.axvline(comp['median_diff'], color=COLORS['orange'], linestyle='--',
               linewidth=1.5, alpha=0.8)

    ax.set_xlabel('Jaccard distance (accessory gene content)')
    ax.set_ylabel('Density')
    ax.set_title(f'Gene content similarity within {largest} (n={jac["n"]})')
    ax.legend(fontsize=7, loc='upper left')

    p_str = f'{comp["p"]:.1e}' if comp["p"] < 0.001 else f'{comp["p"]:.3f}'
    ax.text(0.95, 0.95,
            f'Mann-Whitney p = {p_str}\n'
            f'Effect (rank-biserial) = {comp["effect_size"]:.4f}\n'
            f'Median same = {comp["median_same"]:.4f}\n'
            f'Median diff = {comp["median_diff"]:.4f}',
            transform=ax.transAxes, fontsize=7, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=COLORS['grey'], alpha=0.9),
            family='monospace')

    data['_panel_a_results'] = {
        'phylogroup': largest,
        'n': jac['n'],
        **comp,
    }


def panel_b_permanova_decomposition(ax, data):
    """Panel B: PERMANOVA variance decomposition.

    Show how much of the variance in gene content (Jaccard distances)
    is explained by phylogroup vs isolation source.
    """
    print('  Panel B: PERMANOVA variance decomposition...')

    result = run_global_permanova(data, max_per_group=5000, n_perms=999)

    # Bar chart
    components = ['Phylogroup', 'Isolation\nsource', 'Residual']
    r2_pg = result['R2_phylogroup']
    r2_src = result['R2_source']
    # Residual: approximate (since these are marginal, not nested)
    r2_resid = max(0, 1.0 - r2_pg - r2_src)
    values = [r2_pg * 100, r2_src * 100, r2_resid * 100]
    colors = [COLORS['green'], COLORS['orange'], COLORS['grey']]

    bars = ax.bar(components, values, color=colors, edgecolor='white',
                  linewidth=0.5, width=0.6)

    # Labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=8,
                fontweight='bold')

    ax.set_ylabel('Variance explained (%)')
    ax.set_title('PERMANOVA: sources of gene content variation')
    ax.set_ylim(0, max(values) * 1.3)

    p_pg_str = f'{result["p_phylogroup"]:.3f}'
    p_src_str = f'{result["p_source"]:.3f}'
    ax.text(0.95, 0.95,
            f'n = {result["n_genomes"]:,} genomes\n'
            f'Phylogroup: F = {result["F_phylogroup"]:.1f}, p = {p_pg_str}\n'
            f'Source: F = {result["F_source"]:.1f}, p = {p_src_str}',
            transform=ax.transAxes, fontsize=7, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=COLORS['grey'], alpha=0.9),
            family='monospace')

    data['_panel_b_results'] = result


def panel_c_source_heatmap(ax, data):
    """Panel C: Heatmap of mean Jaccard distance between source categories.

    Within the largest phylogroup, compute mean Jaccard distance between
    each pair of isolation source categories. Diagonal = within-source.
    """
    print('  Panel C: Source heatmap...')

    counts = data['phylogroup_counts']
    largest = max(counts, key=counts.get)

    jac = compute_jaccard_within_phylogroup(data, largest, max_genomes=500)

    # Get unique sources in this phylogroup
    sources = sorted(set(jac['sources']))
    n_src = len(sources)
    src_to_idx = {s: i for i, s in enumerate(sources)}

    # Build mean distance matrix
    mean_dm = np.zeros((n_src, n_src))
    count_dm = np.zeros((n_src, n_src))

    for k, (si, sj) in enumerate(zip(jac['source_i'], jac['source_j'])):
        i, j = src_to_idx[si], src_to_idx[sj]
        d = jac['distances'][k]
        mean_dm[i, j] += d
        mean_dm[j, i] += d
        count_dm[i, j] += 1
        count_dm[j, i] += 1

    # Self-distances (diagonal)
    # These are already covered by pairs within the same source

    # Normalise
    count_dm[count_dm == 0] = 1  # avoid division by zero
    mean_dm /= count_dm

    im = ax.imshow(mean_dm, cmap='viridis', aspect='auto')
    ax.set_xticks(range(n_src))
    ax.set_xticklabels(sources, fontsize=8)
    ax.set_yticks(range(n_src))
    ax.set_yticklabels(sources, fontsize=8)

    # Annotate cells
    for i in range(n_src):
        for j in range(n_src):
            ax.text(j, i, f'{mean_dm[i, j]:.4f}', ha='center', va='center',
                    fontsize=7, color='white' if mean_dm[i, j] > np.median(mean_dm) else 'black')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Mean Jaccard distance', fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    ax.set_title(f'Mean gene distance by source ({largest})')

    # Check if diagonal is lower than off-diagonal
    diag = np.diag(mean_dm)
    off_diag = mean_dm[~np.eye(n_src, dtype=bool)]
    diag_mean = np.mean(diag) if len(diag) > 0 else np.nan
    offdiag_mean = np.mean(off_diag) if len(off_diag) > 0 else np.nan

    data['_panel_c_results'] = {
        'phylogroup': largest,
        'sources': sources,
        'mean_dm': mean_dm,
        'diag_mean': diag_mean,
        'offdiag_mean': offdiag_mean,
    }


def panel_d_effect_across_phylogroups(ax, data):
    """Panel D: Effect size across phylogroups (forest plot).

    Show rank-biserial effect size with 95% bootstrap CI for each
    phylogroup where the analysis is computable.
    """
    print('  Panel D: Effect size across phylogroups...')

    results = compute_effect_per_phylogroup(data, max_genomes=5000, n_perms=999)

    # Filter to computable
    computable = [r for r in results if r['computable']]
    computable.sort(key=lambda r: r['n'], reverse=True)

    if not computable:
        ax.text(0.5, 0.5, 'No phylogroups with sufficient data',
                transform=ax.transAxes, ha='center', va='center')
        data['_panel_d_results'] = {'computable': []}
        return

    labels = [f"{r['phylogroup']} (n={r['n']:,})" for r in computable]
    effects = [r['effect_size'] for r in computable]
    ci_lows = [r['ci_low'] for r in computable]
    ci_highs = [r['ci_high'] for r in computable]

    y_pos = list(range(len(computable)))

    # Plot
    ax.axvline(0, color=COLORS['grey'], linestyle=':', linewidth=0.8)
    for i, (eff, lo, hi, r) in enumerate(zip(effects, ci_lows, ci_highs,
                                               computable)):
        color = COLORS['blue'] if abs(eff) < 0.05 else COLORS['orange']
        ax.plot([lo, hi], [i, i], color=color, linewidth=2.5,
                solid_capstyle='round')
        ax.plot(eff, i, 'o', color=color, markersize=8, zorder=5,
                markeredgecolor='white', markeredgewidth=0.5)

        # Inline annotation to the right of each CI
        p_str = f'{r["p"]:.3f}' if r["p"] >= 0.001 else f'{r["p"]:.1e}'
        ax.annotate(f'R²={r["R2"]:.3f}, p={p_str}',
                    xy=(hi + 0.005, i), fontsize=6.5, va='center',
                    family='monospace', color=COLORS['grey'])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Rank-biserial effect size')
    ax.set_title('Environment effect per phylogroup')
    ax.invert_yaxis()

    # Tighten x-axis: pad beyond data range
    all_vals = ci_lows + ci_highs + effects
    x_min = min(all_vals) - 0.02
    x_max = max(all_vals) + 0.12  # room for annotations
    ax.set_xlim(max(-0.05, x_min), x_max)

    data['_panel_d_results'] = {'computable': computable}


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Gene-environment decoupling analysis (Test 4)')
    parser.add_argument('--output-dir', default='.', help='Output directory')
    parser.add_argument('--format', default='png',
                        choices=['png', 'pdf', 'svg', 'all'],
                        help='Figure format')
    parser.add_argument('--show', action='store_true', help='Show figure')
    args = parser.parse_args()

    print_header('Gene-Environment Decoupling Analysis (Test 4)', {
        'Dataset': 'Horesh et al. (2021) — 10,146 E. coli genomes',
        'Test': 'Is accessory gene content coupled to isolation environment?',
        'Bet-hedging prediction': 'Decoupled (<1% variance explained)',
        'Local adaptation prediction': 'Coupled (>5% variance explained)',
    })

    setup_plotting()

    # Load data
    print('\n[1/5] Loading data...')
    data = load_horesh_data()
    print(f'  Loaded: {data["n_genomes"]} genomes, '
          f'{data["n_accessory_genes"]} accessory genes')
    print(f'  Sources: {data["source_counts"]}')
    print(f'  Phylogroups: {data["phylogroup_counts"]}')

    # Create figure
    print('\n[2/5] Creating panels...')
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))
    fig.suptitle(
        'Test 4: Gene-Environment Decoupling in E. coli\n'
        '(Horesh et al. 2021 — accessory gene content vs isolation source)',
        fontsize=12, fontweight='bold', y=0.98)

    print('\n[3/5] Generating panels...')
    panel_a_jaccard_distributions(axes[0, 0], data)
    add_panel_label(axes[0, 0], 'A')

    panel_b_permanova_decomposition(axes[0, 1], data)
    add_panel_label(axes[0, 1], 'B')

    panel_c_source_heatmap(axes[1, 0], data)
    add_panel_label(axes[1, 0], 'C')

    panel_d_effect_across_phylogroups(axes[1, 1], data)
    add_panel_label(axes[1, 1], 'D')

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save
    print('\n[4/5] Saving figure...')
    save_figure(fig, 'decoupling_analysis', output_dir=args.output_dir,
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
        print(f'  Panel A (Jaccard distributions, {r["phylogroup"]}):')
        print(f'    Median same-source = {r["median_same"]:.4f}')
        print(f'    Median diff-source = {r["median_diff"]:.4f}')
        print(f'    Mann-Whitney p = {r["p"]:.4f}')
        print(f'    Effect (rank-biserial) = {r["effect_size"]:.4f}')

    if '_panel_b_results' in data:
        r = data['_panel_b_results']
        print(f'\n  *** PANEL B (KEY RESULT — PERMANOVA) ***')
        print(f'  Phylogroup explains {r["R2_phylogroup"]*100:.1f}% '
              f'(F={r["F_phylogroup"]:.1f}, p={r["p_phylogroup"]:.3f})')
        print(f'  Isolation source explains {r["R2_source"]*100:.2f}% '
              f'(F={r["F_source"]:.2f}, p={r["p_source"]:.3f})')
        pct = r['R2_source'] * 100
        if pct < 1:
            print(f'  → DECOUPLED (supports bet-hedging)')
        elif pct < 5:
            print(f'  → AMBIGUOUS')
        else:
            print(f'  → COUPLED (supports local adaptation)')

    if '_panel_c_results' in data:
        r = data['_panel_c_results']
        print(f'\n  Panel C (heatmap, {r["phylogroup"]}):')
        print(f'    Mean within-source Jaccard = {r["diag_mean"]:.4f}')
        print(f'    Mean between-source Jaccard = {r["offdiag_mean"]:.4f}')

    if '_panel_d_results' in data:
        r = data['_panel_d_results']
        print(f'\n  Panel D (effect per phylogroup):')
        for pg in r['computable']:
            print(f'    {pg["phylogroup"]}: R²={pg["R2"]:.4f}, '
                  f'effect={pg["effect_size"]:.4f}, p={pg["p"]:.3f}')

    print('\n' + '=' * 60)


if __name__ == '__main__':
    main()
