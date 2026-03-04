#!/usr/bin/env python3
"""
Unit tests for variance_analysis.py

Test-driven development: these tests define what the variance analysis
must do using only real data from Douglas & Shapiro (2024).
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
# Test 1: Data loading returns the right structure
# ---------------------------------------------------------------------------

class TestDataLoading:
    """Verify load_real_data() returns correct structure from the real file."""

    @pytest.fixture(autouse=True)
    def load_data(self):
        from variance_analysis import load_real_data
        self.data = load_real_data()

    def test_returns_dict(self):
        assert isinstance(self.data, dict)

    def test_n_species_is_655(self):
        """After filtering for valid SD values and non-zero means, 655 species remain."""
        assert self.data['n_species'] == 655

    def test_required_keys_present(self):
        required = [
            'n_species',
            'species_names',
            'dnds',
            'genomic_fluidity',
            'pseudogene_fluidity',
            'mean_singletons_gene',        # mean singleton count (genes)
            'sd_singletons_gene',           # SD singleton count (genes)
            'mean_singletons_pseudo',       # mean singleton count (pseudogenes)
            'sd_singletons_pseudo',         # SD singleton count (pseudogenes)
            'cv_gene',                      # CV of gene singletons
            'cv_pseudo',                    # CV of pseudogene singletons
            'cv_ratio',                     # CV_gene / CV_pseudo
            'taxonomic_class',
        ]
        for key in required:
            assert key in self.data, f"Missing key: {key}"

    def test_all_arrays_have_655_entries(self):
        n = 655
        for key in ['dnds', 'genomic_fluidity', 'pseudogene_fluidity',
                     'mean_singletons_gene', 'sd_singletons_gene',
                     'mean_singletons_pseudo', 'sd_singletons_pseudo',
                     'cv_gene', 'cv_pseudo', 'cv_ratio']:
            assert len(self.data[key]) == n, (
                f"{key} has {len(self.data[key])} entries, expected {n}")

    def test_sd_values_are_positive(self):
        assert np.all(self.data['sd_singletons_gene'] > 0)
        assert np.all(self.data['sd_singletons_pseudo'] > 0)

    def test_cv_values_are_positive(self):
        assert np.all(self.data['cv_gene'] > 0)
        assert np.all(self.data['cv_pseudo'] > 0)
        assert np.all(self.data['cv_ratio'] > 0)

    def test_no_synthetic_data_generated(self):
        """There must be no fabricated/simulated values anywhere."""
        import inspect
        from variance_analysis import load_real_data
        source = inspect.getsource(load_real_data)
        assert 'rng.normal' not in source
        assert 'np.random' not in source
        assert 'random.default_rng' not in source


# ---------------------------------------------------------------------------
# Test 2: Scientific claims
# ---------------------------------------------------------------------------

class TestScientificClaims:
    """Verify the key scientific results using genuine Douglas & Shapiro data."""

    @pytest.fixture(autouse=True)
    def load_data(self):
        from variance_analysis import load_real_data
        self.data = load_real_data()

    def test_cv_gene_greater_than_cv_pseudo(self):
        """Gene singletons are more variable than pseudogene singletons
        (bet-hedging maintains functional diversity across genomes)."""
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(self.data['cv_gene'], self.data['cv_pseudo'],
                           alternative='greater')
        assert p < 0.001, f"CV_gene not significantly > CV_pseudo: p={p:.4f}"
        # Also check majority direction
        frac_greater = np.mean(self.data['cv_gene'] > self.data['cv_pseudo'])
        assert frac_greater > 0.70, (
            f"Only {frac_greater*100:.1f}% of species have CV_gene > CV_pseudo")

    def test_cv_gene_correlates_with_dnds(self):
        """CV of gene singletons should correlate positively with dN/dS:
        weaker selection -> less constrained variability."""
        from scipy.stats import spearmanr
        rho, p = spearmanr(self.data['cv_gene'], self.data['dnds'])
        assert rho > 0.4, f"CV_gene vs dN/dS rho should be > 0.4, got {rho:.3f}"
        assert p < 0.001, f"CV_gene vs dN/dS not significant: p={p:.4f}"

    def test_cv_gene_dnds_stronger_than_cv_pseudo_dnds(self):
        """The CV-dN/dS correlation should be stronger for genes than pseudogenes,
        showing that selection constrains gene CV beyond the neutral baseline."""
        from scipy.stats import spearmanr
        rho_gene, _ = spearmanr(self.data['cv_gene'], self.data['dnds'])
        rho_pseudo, _ = spearmanr(self.data['cv_pseudo'], self.data['dnds'])
        assert rho_gene > rho_pseudo, (
            f"Gene rho ({rho_gene:.3f}) should exceed pseudo rho ({rho_pseudo:.3f})")

    def test_taylor_gene_exponent_below_one(self):
        """Taylor's law exponent for genes should be < 1 (sub-proportional scaling)."""
        from scipy.stats import linregress
        log_mean = np.log10(self.data['mean_singletons_gene'])
        log_sd = np.log10(self.data['sd_singletons_gene'])
        slope, _, _, _, _ = linregress(log_mean, log_sd)
        assert slope < 1.0, f"Gene Taylor exponent {slope:.3f} should be < 1.0"
        assert slope > 0.5, f"Gene Taylor exponent {slope:.3f} implausibly low"

    def test_taylor_gene_exponent_less_than_pseudo(self):
        """Gene Taylor exponent should be lower than pseudogene exponent
        (tighter mean-variance scaling under selection)."""
        from scipy.stats import linregress
        log_mean_g = np.log10(self.data['mean_singletons_gene'])
        log_sd_g = np.log10(self.data['sd_singletons_gene'])
        slope_g, _, _, _, _ = linregress(log_mean_g, log_sd_g)

        log_mean_p = np.log10(self.data['mean_singletons_pseudo'])
        log_sd_p = np.log10(self.data['sd_singletons_pseudo'])
        slope_p, _, _, _, _ = linregress(log_mean_p, log_sd_p)

        assert slope_g < slope_p, (
            f"Gene exponent ({slope_g:.3f}) should be < pseudo ({slope_p:.3f})")

    def test_cv_selection_replicates_in_gammaproteobacteria(self):
        """The largest class should independently show the CV-dN/dS correlation."""
        from scipy.stats import spearmanr
        mask = self.data['taxonomic_class'] == 'c__Gammaproteobacteria'
        assert mask.sum() > 100, "Too few Gammaproteobacteria"
        rho, p = spearmanr(self.data['cv_gene'][mask],
                           self.data['dnds'][mask])
        assert rho > 0, f"Within-class rho should be positive, got {rho:.3f}"
        assert p < 0.01, f"Within-class correlation not significant: p={p:.4f}"


# ---------------------------------------------------------------------------
# Test 3: Panel functions exist and run
# ---------------------------------------------------------------------------

class TestPanelFunctions:
    """Verify that panel functions exist and can be called."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from variance_analysis import load_real_data
        self.data = load_real_data()

    def test_panel_a_exists(self):
        from variance_analysis import panel_a_mean_vs_variance
        assert callable(panel_a_mean_vs_variance)

    def test_panel_b_exists(self):
        from variance_analysis import panel_b_cv_vs_selection
        assert callable(panel_b_cv_vs_selection)

    def test_panel_c_exists(self):
        from variance_analysis import panel_c_taylors_law
        assert callable(panel_c_taylors_law)

    def test_panel_d_exists(self):
        from variance_analysis import panel_d_within_class_cv
        assert callable(panel_d_within_class_cv)

    def test_all_panels_run_without_error(self):
        """All four panels should produce a figure without error."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from variance_analysis import (panel_a_mean_vs_variance,
                                        panel_b_cv_vs_selection,
                                        panel_c_taylors_law,
                                        panel_d_within_class_cv)

        fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))
        panel_a_mean_vs_variance(axes[0, 0], self.data)
        panel_b_cv_vs_selection(axes[0, 1], self.data)
        panel_c_taylors_law(axes[1, 0], self.data)
        panel_d_within_class_cv(axes[1, 1], self.data)
        plt.close(fig)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
