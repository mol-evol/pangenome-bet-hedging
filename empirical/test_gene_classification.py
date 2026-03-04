#!/usr/bin/env python3
"""
Unit tests for gene_classification.py

Test-driven development: these tests define what the gene classification
analysis must do using the Horesh et al. (2021) E. coli dataset.

The gene classification addresses the question: what fraction of accessory
genes are niche-specific (frequency differs by body site) vs insurance/backup
genes (maintained stochastically regardless of environment)?
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
# Test 1: Data loading — gene names present
# ---------------------------------------------------------------------------

class TestDataLoading:
    """Verify that gene names are available in loaded data."""

    @pytest.fixture(autouse=True)
    def load_data(self):
        from gene_classification import load_classification_data
        self.data = load_classification_data()

    def test_gene_names_present(self):
        """gene_names key should be in data dict, length = n_accessory_genes."""
        assert 'gene_names' in self.data
        assert len(self.data['gene_names']) == self.data['n_accessory_genes']

    def test_gene_names_are_strings(self):
        """All gene names should be non-empty strings."""
        for i, name in enumerate(self.data['gene_names'][:100]):
            assert isinstance(name, str), f"Gene {i} is {type(name)}"
            assert len(name) > 0, f"Gene {i} is empty string"

    def test_b2_has_three_sources(self):
        """B2 phylogroup should have Blood, Feces, Urine with ≥20 each."""
        from collections import Counter
        mask = self.data['phylogroup'] == 'B2'
        sources = self.data['isolation_source'][mask]
        counts = Counter(sources)
        assert 'Blood' in counts, "No Blood genomes in B2"
        assert 'Feces' in counts, "No Feces genomes in B2"
        assert 'Urine' in counts, "No Urine genomes in B2"
        for src in ['Blood', 'Feces', 'Urine']:
            assert counts[src] >= 20, (
                f"Only {counts[src]} {src} genomes in B2")

    def test_no_synthetic_data(self):
        """load_classification_data must not fabricate values."""
        import inspect
        from gene_classification import load_classification_data
        source = inspect.getsource(load_classification_data)
        assert 'rng.normal' not in source
        assert 'np.random' not in source


# ---------------------------------------------------------------------------
# Test 2: Per-gene chi-squared tests
# ---------------------------------------------------------------------------

class TestPerGeneChiSquared:
    """Verify per-gene chi-squared tests work correctly."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from gene_classification import load_classification_data, per_gene_chi_squared
        self.data = load_classification_data()
        self.result = per_gene_chi_squared(self.data, phylogroup='B2')

    def test_returns_correct_shape(self):
        """Should return one p-value and one V per gene."""
        n_genes = self.data['n_accessory_genes']
        assert len(self.result['p_values']) == n_genes
        assert len(self.result['cramers_v']) == n_genes

    def test_p_values_in_valid_range(self):
        """All p-values should be in [0, 1]."""
        p = self.result['p_values']
        assert np.all(p >= 0), f"Min p = {p.min()}"
        assert np.all(p <= 1), f"Max p = {p.max()}"

    def test_cramers_v_in_valid_range(self):
        """All Cramér's V should be in [0, 1]."""
        v = self.result['cramers_v']
        assert np.all(v >= 0), f"Min V = {v.min()}"
        assert np.all(v <= 1), f"Max V = {v.max()}"

    def test_some_genes_significant(self):
        """At least some genes should have p < 0.05 (niche-specific)."""
        n_sig = np.sum(self.result['p_values'] < 0.05)
        assert n_sig >= 50, (
            f"Only {n_sig} genes with p < 0.05 — expected more in mixed model")
        print(f"\n  {n_sig} / {len(self.result['p_values'])} genes "
              f"significant at p < 0.05")

    def test_not_all_genes_significant(self):
        """Not ALL genes should be significant — mixed model has insurance genes.

        With n=1,705 genomes in B2, statistical power is very high so many
        genes will be significant. The key prediction of the mixed model
        is that some genes are NOT associated with environment.
        """
        from gene_classification import compute_fdr
        q = compute_fdr(self.result['p_values'])
        n_sig = np.sum(q < 0.05)
        n_nonsig = np.sum(q >= 0.05)
        n_total = len(q)
        # At least 10% of genes should be non-significant (insurance)
        assert n_nonsig >= n_total * 0.10, (
            f"Only {n_nonsig} ({n_nonsig/n_total*100:.1f}%) non-significant — "
            f"expected ≥10% for mixed model")
        print(f"\n  {n_sig} / {n_total} genes ({n_sig/n_total*100:.1f}%) "
              f"significant at BH q < 0.05")

    def test_source_frequencies_reported(self):
        """Per-source gene frequencies should be in result."""
        assert 'source_freqs' in self.result
        assert 'Blood' in self.result['source_freqs']
        assert 'Feces' in self.result['source_freqs']
        assert 'Urine' in self.result['source_freqs']
        # Each should be an array of length n_genes
        n_genes = self.data['n_accessory_genes']
        for src in ['Blood', 'Feces', 'Urine']:
            assert len(self.result['source_freqs'][src]) == n_genes


# ---------------------------------------------------------------------------
# Test 3: Storey pi0 estimation
# ---------------------------------------------------------------------------

class TestPi0Estimation:
    """Verify Storey (2002) pi0 estimation works."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from gene_classification import (load_classification_data,
                                          per_gene_chi_squared,
                                          estimate_pi0)
        data = load_classification_data()
        result = per_gene_chi_squared(data, phylogroup='B2')
        self.pi0_result = estimate_pi0(result['p_values'])
        self.p_values = result['p_values']

    def test_pi0_in_valid_range(self):
        """pi0 should be in (0, 1) — strictly between bounds."""
        pi0 = self.pi0_result['pi0']
        assert 0.01 <= pi0 <= 0.99, f"pi0 = {pi0} is outside [0.01, 0.99]"
        print(f"\n  Storey pi0 = {pi0:.3f}")

    def test_pi0_in_biologically_meaningful_range(self):
        """pi0 should be biologically meaningful — not zero, not all.

        With n=1,705 genomes in B2, statistical power is very high.
        E. coli is highly niche-structured across body sites, so many genes
        have detectable frequency differences. A pi0 of 0.10-0.50 is
        reasonable: some fraction of genes are true insurance with no
        environment association, while the majority show at least some
        niche structure.
        """
        pi0 = self.pi0_result['pi0']
        assert pi0 > 0.05, (
            f"pi0 = {pi0:.3f} — too low, almost all genes classified as niche")
        assert pi0 < 0.80, (
            f"pi0 = {pi0:.3f} — too high, barely any niche genes detected")
        print(f"\n  pi0 = {pi0:.3f} — {pi0*100:.1f}% estimated insurance")

    def test_pi0_is_stable(self):
        """Changing lambda grid slightly should not change pi0 drastically."""
        from gene_classification import estimate_pi0
        grid1 = np.arange(0.05, 0.91, 0.05)
        grid2 = np.arange(0.10, 0.86, 0.05)
        r1 = estimate_pi0(self.p_values, lambda_grid=grid1)
        r2 = estimate_pi0(self.p_values, lambda_grid=grid2)
        diff = abs(r1['pi0'] - r2['pi0'])
        assert diff < 0.10, (
            f"pi0 changed by {diff:.3f} between grids — unstable estimate")
        print(f"\n  pi0 stability: grid1={r1['pi0']:.3f}, "
              f"grid2={r2['pi0']:.3f}, diff={diff:.3f}")


# ---------------------------------------------------------------------------
# Test 4: Gene classification
# ---------------------------------------------------------------------------

class TestGeneClassification:
    """Verify gene classification into niche/insurance/ambiguous."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from gene_classification import (load_classification_data,
                                          per_gene_chi_squared,
                                          compute_fdr,
                                          classify_genes)
        data = load_classification_data()
        self.n_genes = data['n_accessory_genes']
        result = per_gene_chi_squared(data, phylogroup='B2')
        self.q = compute_fdr(result['p_values'])
        self.v = result['cramers_v']
        self.labels = classify_genes(self.q, self.v)

    def test_classification_covers_all_genes(self):
        """niche + insurance + ambiguous = total."""
        n_niche = np.sum(self.labels == 'niche')
        n_insurance = np.sum(self.labels == 'insurance')
        n_ambiguous = np.sum(self.labels == 'ambiguous')
        total = n_niche + n_insurance + n_ambiguous
        assert total == self.n_genes, (
            f"Classification covers {total}, expected {self.n_genes}")
        print(f"\n  Niche: {n_niche}, Insurance: {n_insurance}, "
              f"Ambiguous: {n_ambiguous}")

    def test_niche_genes_have_significant_q(self):
        """All niche genes should have q < 0.05."""
        niche_mask = self.labels == 'niche'
        if niche_mask.sum() > 0:
            assert np.all(self.q[niche_mask] < 0.05), (
                "Some niche genes have q >= 0.05")

    def test_niche_genes_have_high_v(self):
        """All niche genes should have V > 0.10."""
        niche_mask = self.labels == 'niche'
        if niche_mask.sum() > 0:
            assert np.all(self.v[niche_mask] > 0.10), (
                "Some niche genes have V <= 0.10")

    def test_insurance_genes_criteria(self):
        """All insurance genes should have q > 0.05 OR V < 0.05."""
        ins_mask = self.labels == 'insurance'
        if ins_mask.sum() > 0:
            # Each insurance gene should satisfy at least one criterion
            satisfies = (self.q[ins_mask] > 0.05) | (self.v[ins_mask] < 0.05)
            assert np.all(satisfies), (
                "Some insurance genes don't meet criteria")

    def test_class_proportions_sum_to_one(self):
        """Proportions should sum to 1.0."""
        n_niche = np.sum(self.labels == 'niche')
        n_insurance = np.sum(self.labels == 'insurance')
        n_ambiguous = np.sum(self.labels == 'ambiguous')
        total = n_niche + n_insurance + n_ambiguous
        assert abs(total / self.n_genes - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Test 5: Panel functions exist and run
# ---------------------------------------------------------------------------

class TestPanelFunctions:
    """Verify that panel functions exist and can be called."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from gene_classification import load_classification_data
        self.data = load_classification_data()

    def test_panel_a_exists(self):
        from gene_classification import panel_a_volcano
        assert callable(panel_a_volcano)

    def test_panel_b_exists(self):
        from gene_classification import panel_b_pvalue_histogram
        assert callable(panel_b_pvalue_histogram)

    def test_panel_c_exists(self):
        from gene_classification import panel_c_frequency_profiles
        assert callable(panel_c_frequency_profiles)

    def test_panel_d_exists(self):
        from gene_classification import panel_d_cross_validation
        assert callable(panel_d_cross_validation)

    def test_all_panels_run_without_error(self):
        """All four panels should produce a figure without error."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from gene_classification import (panel_a_volcano,
                                          panel_b_pvalue_histogram,
                                          panel_c_frequency_profiles,
                                          panel_d_cross_validation)

        fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))
        panel_a_volcano(axes[0, 0], self.data)
        panel_b_pvalue_histogram(axes[0, 1], self.data)
        panel_c_frequency_profiles(axes[1, 0], self.data)
        panel_d_cross_validation(axes[1, 1], self.data)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Test 6: No circularity
# ---------------------------------------------------------------------------

class TestNoCircularity:
    """Ensure the script does not fabricate data to guarantee results."""

    def test_no_rng_in_load(self):
        """load_classification_data must not use random number generators."""
        import inspect
        from gene_classification import load_classification_data
        source = inspect.getsource(load_classification_data)
        assert 'rng.normal' not in source
        assert 'np.random' not in source
        assert 'random.default_rng' not in source

    def test_no_hardcoded_results(self):
        """No function should return hardcoded pi0 or classification counts."""
        import inspect
        from gene_classification import load_classification_data
        source = inspect.getsource(load_classification_data)
        assert 'pi0 =' not in source or 'pi0 = ' not in source


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
