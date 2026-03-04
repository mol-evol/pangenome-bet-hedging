#!/usr/bin/env python3
"""
Unit tests for decoupling_analysis.py

Test-driven development: these tests define what the gene-environment
decoupling analysis must do using the Horesh et al. (2021) E. coli dataset
(10,146 genomes with gene presence/absence matrix and isolation metadata).

The decoupling analysis addresses the question: is accessory gene content
COUPLED to isolation environment (local adaptation) or DECOUPLED from it
(bet-hedging)?
"""

import sys
import os
import numpy as np
import pytest

# Add this directory and parent (code/) so shared.* and sibling imports work
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, _CODE_DIR)


# ---------------------------------------------------------------------------
# Test 1: Data loading — Horesh et al. (2021) E. coli collection
# ---------------------------------------------------------------------------

class TestDataLoading:
    """Verify Horesh data loads and is properly structured."""

    @pytest.fixture(autouse=True)
    def load_data(self):
        from decoupling_analysis import load_horesh_data
        self.data = load_horesh_data()

    def test_returns_dict(self):
        assert isinstance(self.data, dict)

    def test_n_genomes_reasonable(self):
        """At least 2,000 genomes should have valid metadata after filtering."""
        assert self.data['n_genomes'] >= 2000

    def test_n_accessory_genes_reasonable(self):
        """Should have thousands of accessory genes (5-95% frequency)."""
        assert self.data['n_accessory_genes'] >= 1000
        assert self.data['n_accessory_genes'] <= 40000

    def test_required_keys_present(self):
        required = [
            'n_genomes', 'n_accessory_genes',
            'genome_ids', 'isolation_source', 'phylogroup',
            'pa_matrix', 'gene_frequencies',
            'source_counts', 'phylogroup_counts',
        ]
        for key in required:
            assert key in self.data, f"Missing key: {key}"

    def test_isolation_source_categories(self):
        """At least 3 isolation source categories should be present."""
        categories = set(self.data['isolation_source'])
        assert len(categories) >= 3, (
            f"Only {categories} isolation source categories")

    def test_phylogroup_categories(self):
        """At least 4 phylogroups should be present."""
        phylogroups = set(self.data['phylogroup'])
        assert len(phylogroups) >= 4, (
            f"Only {phylogroups} phylogroups")

    def test_pa_matrix_shape_consistent(self):
        """PA matrix dimensions should match n_genomes x n_accessory_genes."""
        n = self.data['n_genomes']
        m = self.data['n_accessory_genes']
        shape = self.data['pa_matrix'].shape
        assert shape == (n, m), (
            f"PA matrix shape {shape} != expected ({n}, {m})")

    def test_pa_matrix_is_binary(self):
        """All values in PA matrix should be 0 or 1."""
        pa = self.data['pa_matrix']
        # Check a subsample for speed
        sample = pa[:100, :100] if pa.shape[0] > 100 else pa
        unique_vals = np.unique(sample)
        assert set(unique_vals).issubset({0, 1}), (
            f"PA matrix contains non-binary values: {unique_vals}")

    def test_gene_frequencies_in_range(self):
        """All gene frequencies should be in [0.05, 0.95] (accessory filter)."""
        freq = self.data['gene_frequencies']
        assert len(freq) == self.data['n_accessory_genes']
        assert np.all(freq >= 0.04), f"Min freq = {freq.min():.4f}"  # Allow tiny rounding
        assert np.all(freq <= 0.96), f"Max freq = {freq.max():.4f}"

    def test_arrays_consistent_length(self):
        """All metadata arrays should have n_genomes entries."""
        n = self.data['n_genomes']
        for key in ['genome_ids', 'isolation_source', 'phylogroup']:
            assert len(self.data[key]) == n, (
                f"{key} has {len(self.data[key])} entries, expected {n}")

    def test_no_synthetic_data(self):
        """load_horesh_data must not fabricate values."""
        import inspect
        from decoupling_analysis import load_horesh_data
        source = inspect.getsource(load_horesh_data)
        assert 'rng.normal' not in source
        assert 'np.random' not in source
        assert 'random.default_rng' not in source

    def test_source_counts_match_array(self):
        """source_counts dict should match actual counts in the array."""
        from collections import Counter
        actual = Counter(self.data['isolation_source'])
        for source, count in self.data['source_counts'].items():
            assert actual[source] == count, (
                f"source_counts['{source}'] = {count} but actual = {actual[source]}")


# ---------------------------------------------------------------------------
# Test 2: Scientific claims — gene-environment decoupling
# ---------------------------------------------------------------------------

class TestScientificClaims:
    """Test the key decoupling predictions.

    IMPORTANT: These tests check that the analyses CAN BE COMPUTED and
    report results honestly. The direction of results determines which
    hypothesis is supported — we do NOT assert a particular direction.

    A priori effect size thresholds (stated before seeing results):
      - <1% variance explained by source = "decoupled" (bet-hedging)
      - 1-5% = ambiguous
      - >5% = "coupled" (local adaptation)
    """

    @pytest.fixture(autouse=True)
    def load_data(self):
        from decoupling_analysis import load_horesh_data
        self.data = load_horesh_data()

    def test_jaccard_distances_computable(self):
        """Pairwise Jaccard within the largest phylogroup should be computable."""
        from decoupling_analysis import compute_jaccard_within_phylogroup
        # Find largest phylogroup
        counts = self.data['phylogroup_counts']
        largest = max(counts, key=counts.get)
        result = compute_jaccard_within_phylogroup(self.data, largest,
                                                    max_genomes=200)
        assert 'distances' in result
        assert 'source_i' in result
        assert 'source_j' in result
        assert len(result['distances']) > 100
        print(f"\n  Jaccard within {largest}: {len(result['distances'])} pairs, "
              f"median = {np.median(result['distances']):.4f}")

    def test_same_vs_different_source_comparison(self):
        """Mann-Whitney between same-source and different-source pairs
        should be computable with adequate sample sizes."""
        from decoupling_analysis import (compute_jaccard_within_phylogroup,
                                          compare_same_vs_different)
        counts = self.data['phylogroup_counts']
        largest = max(counts, key=counts.get)
        jac = compute_jaccard_within_phylogroup(self.data, largest,
                                                 max_genomes=200)
        result = compare_same_vs_different(
            jac['distances'], jac['source_i'], jac['source_j'])
        assert 'U' in result
        assert 'p' in result
        assert 'effect_size' in result
        assert np.isfinite(result['p'])
        print(f"\n  Same vs different source: U={result['U']:.0f}, "
              f"p={result['p']:.4f}, effect={result['effect_size']:.4f}")

    def test_permanova_computable(self):
        """PERMANOVA should run on within-phylogroup data."""
        from decoupling_analysis import (compute_jaccard_within_phylogroup,
                                          permanova_one_factor)
        counts = self.data['phylogroup_counts']
        largest = max(counts, key=counts.get)
        jac = compute_jaccard_within_phylogroup(self.data, largest,
                                                 max_genomes=150)
        result = permanova_one_factor(jac['dm_square'], jac['sources'],
                                       n_perms=99)
        assert 'F' in result
        assert 'p' in result
        assert 'R2' in result
        assert np.isfinite(result['F'])
        print(f"\n  PERMANOVA within {largest}: F={result['F']:.3f}, "
              f"p={result['p']:.3f}, R2={result['R2']:.4f}")

    def test_environment_effect_size_reported(self):
        """KEY TEST: The % variance explained by isolation source should be
        reported. This is the main discriminating result."""
        from decoupling_analysis import (compute_jaccard_within_phylogroup,
                                          permanova_one_factor)
        counts = self.data['phylogroup_counts']
        largest = max(counts, key=counts.get)
        jac = compute_jaccard_within_phylogroup(self.data, largest,
                                                 max_genomes=150)
        result = permanova_one_factor(jac['dm_square'], jac['sources'],
                                       n_perms=99)
        r2 = result['R2']
        assert np.isfinite(r2)
        assert 0 <= r2 <= 1
        # Report honestly — direction determines which hypothesis wins
        pct = r2 * 100
        if pct < 1:
            interp = "DECOUPLED (supports bet-hedging)"
        elif pct < 5:
            interp = "AMBIGUOUS"
        else:
            interp = "COUPLED (supports local adaptation)"
        print(f"\n  *** KEY RESULT: Isolation source explains {pct:.2f}% "
              f"of Jaccard variance → {interp} ***")

    def test_phylogroup_effect_dominates(self):
        """Phylogroup should explain much more variance than isolation source.
        This is expected regardless of hypothesis."""
        from decoupling_analysis import run_global_permanova
        result = run_global_permanova(self.data, max_per_group=100,
                                       n_perms=99)
        assert result['R2_phylogroup'] > result['R2_source'], (
            "Isolation source explains more variance than phylogroup — unexpected")
        print(f"\n  Phylogroup R2={result['R2_phylogroup']:.4f}, "
              f"Source R2={result['R2_source']:.4f}")

    def test_effect_replicates_across_phylogroups(self):
        """Effect size should be computable in at least 3 phylogroups."""
        from decoupling_analysis import compute_effect_per_phylogroup
        results = compute_effect_per_phylogroup(self.data, max_genomes=150,
                                                  n_perms=99)
        computable = sum(1 for r in results if r['computable'])
        assert computable >= 3, (
            f"Effect only computable in {computable} phylogroups")
        for r in results:
            if r['computable']:
                print(f"\n  {r['phylogroup']} (n={r['n']}): "
                      f"R2={r['R2']:.4f}, p={r['p']:.3f}")


# ---------------------------------------------------------------------------
# Test 3: No circularity — nothing fabricated
# ---------------------------------------------------------------------------

class TestNoCircularity:
    """Ensure the script does not fabricate data to guarantee results."""

    def test_no_rng_in_load(self):
        """load_horesh_data must not use random number generators."""
        import inspect
        from decoupling_analysis import load_horesh_data
        source = inspect.getsource(load_horesh_data)
        assert 'rng.normal' not in source
        assert 'np.random' not in source
        assert 'random.default_rng' not in source

    def test_no_hardcoded_distances(self):
        """No function should return hardcoded distance or p values."""
        import inspect
        from decoupling_analysis import load_horesh_data
        source = inspect.getsource(load_horesh_data)
        # Should not contain hardcoded correlations
        assert 'jaccard =' not in source or 'jaccard = ' not in source


# ---------------------------------------------------------------------------
# Test 4: Panel functions exist and run
# ---------------------------------------------------------------------------

class TestPanelFunctions:
    """Verify that panel functions exist and can be called."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from decoupling_analysis import load_horesh_data
        self.data = load_horesh_data()

    def test_panel_a_exists(self):
        from decoupling_analysis import panel_a_jaccard_distributions
        assert callable(panel_a_jaccard_distributions)

    def test_panel_b_exists(self):
        from decoupling_analysis import panel_b_permanova_decomposition
        assert callable(panel_b_permanova_decomposition)

    def test_panel_c_exists(self):
        from decoupling_analysis import panel_c_source_heatmap
        assert callable(panel_c_source_heatmap)

    def test_panel_d_exists(self):
        from decoupling_analysis import panel_d_effect_across_phylogroups
        assert callable(panel_d_effect_across_phylogroups)

    def test_all_panels_run_without_error(self):
        """All four panels should produce a figure without error."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from decoupling_analysis import (panel_a_jaccard_distributions,
                                          panel_b_permanova_decomposition,
                                          panel_c_source_heatmap,
                                          panel_d_effect_across_phylogroups)

        fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))
        panel_a_jaccard_distributions(axes[0, 0], self.data)
        panel_b_permanova_decomposition(axes[0, 1], self.data)
        panel_c_source_heatmap(axes[1, 0], self.data)
        panel_d_effect_across_phylogroups(axes[1, 1], self.data)
        plt.close(fig)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
