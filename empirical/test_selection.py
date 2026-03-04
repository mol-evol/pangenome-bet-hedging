#!/usr/bin/env python3
"""
Unit tests for selection_analysis.py

Test-driven development: these tests define what the analysis script
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
        from selection_analysis import load_real_data
        self.data = load_real_data()

    def test_returns_dict(self):
        assert isinstance(self.data, dict)

    def test_n_species_is_670(self):
        """Douglas & Shapiro 2024 has 670 species."""
        assert self.data['n_species'] == 670

    def test_required_keys_present(self):
        required = [
            'n_species',
            'species_names',
            'dnds',                    # species-level dN/dS (intact genes)
            'dn',                      # nonsynonymous divergence
            'ds',                      # synonymous divergence
            'genomic_fluidity',        # pangenome openness
            'pseudogene_fluidity',     # neutral fluidity reference
            'gene_singleton_pct',      # % singleton intact genes (subsampled to 9)
            'pseudo_singleton_pct',    # % singleton pseudogenes (subsampled to 9)
            'si_sp',                   # singleton ratio
            'mean_num_genes',          # mean gene count per genome
            'mean_num_pseudo',         # mean pseudogene count per genome
            'taxonomic_class',         # GTDB class
            'genome_size',             # mean genome size in bp
        ]
        for key in required:
            assert key in self.data, f"Missing key: {key}"

    def test_all_arrays_have_670_entries(self):
        n = 670
        for key in ['dnds', 'dn', 'ds', 'genomic_fluidity', 'pseudogene_fluidity',
                     'gene_singleton_pct', 'pseudo_singleton_pct', 'si_sp',
                     'mean_num_genes', 'mean_num_pseudo', 'genome_size']:
            assert len(self.data[key]) == n, f"{key} has {len(self.data[key])} entries, expected {n}"

    def test_no_synthetic_data_generated(self):
        """There must be no fabricated/simulated values anywhere."""
        # The old script generated pseudogene dN/dS from rng.normal(1.0, 0.15).
        # The new data dict should NOT contain any key like 'dnds_pseudo'
        # that was drawn from a random distribution.
        assert 'dnds_pseudo' not in self.data
        # Also no 'categories' key splitting species into fake gene categories
        assert 'categories' not in self.data

    def test_dnds_values_are_positive(self):
        assert np.all(self.data['dnds'] > 0)

    def test_dn_values_are_positive(self):
        assert np.all(self.data['dn'] > 0)

    def test_ds_values_are_positive(self):
        assert np.all(self.data['ds'] > 0)

    def test_singleton_pcts_are_percentages(self):
        """Singleton rates should be in 0-100 range (percentages)."""
        assert np.all(self.data['gene_singleton_pct'] >= 0)
        assert np.all(self.data['gene_singleton_pct'] <= 100)
        assert np.all(self.data['pseudo_singleton_pct'] >= 0)
        assert np.all(self.data['pseudo_singleton_pct'] <= 100)

    def test_fluidity_values_are_fractions(self):
        """Genomic fluidity is a proportion in [0, 1]."""
        assert np.all(self.data['genomic_fluidity'] >= 0)
        assert np.all(self.data['genomic_fluidity'] <= 1)


# ---------------------------------------------------------------------------
# Test 2: The core scientific claims use only real data
# ---------------------------------------------------------------------------

class TestScientificClaims:
    """Verify the key comparisons use only genuine Douglas & Shapiro data."""

    @pytest.fixture(autouse=True)
    def load_data(self):
        from selection_analysis import load_real_data
        self.data = load_real_data()

    def test_gene_singletons_lower_than_pseudo_singletons(self):
        """Key result: genes have fewer singletons than pseudogenes (selection)."""
        gene_mean = np.mean(self.data['gene_singleton_pct'])
        pseudo_mean = np.mean(self.data['pseudo_singleton_pct'])
        assert gene_mean < pseudo_mean, (
            f"Gene singleton rate ({gene_mean:.2f}%) should be lower than "
            f"pseudogene rate ({pseudo_mean:.2f}%)"
        )

    def test_selection_ratio_substantial(self):
        """Pseudogene/gene singleton ratio should be > 2x."""
        gene_mean = np.mean(self.data['gene_singleton_pct'])
        pseudo_mean = np.mean(self.data['pseudo_singleton_pct'])
        ratio = pseudo_mean / gene_mean
        assert ratio > 2.0, f"Selection ratio {ratio:.1f}x is too low"

    def test_most_species_under_purifying_selection(self):
        """Most species should have dN/dS < 1 (purifying selection)."""
        frac_purifying = np.mean(self.data['dnds'] < 1.0)
        assert frac_purifying > 0.90, (
            f"Only {frac_purifying*100:.1f}% species have dN/dS < 1"
        )

    def test_pseudogene_fluidity_higher_than_gene_fluidity(self):
        """Pseudogene fluidity should exceed gene fluidity (neutral > selected)."""
        gene_mean = np.mean(self.data['genomic_fluidity'])
        pseudo_mean = np.mean(self.data['pseudogene_fluidity'])
        assert pseudo_mean > gene_mean, (
            f"Pseudogene fluidity ({pseudo_mean:.3f}) should exceed "
            f"gene fluidity ({gene_mean:.3f})"
        )

    def test_si_sp_mostly_below_one(self):
        """si_sp (gene/pseudo singleton ratio) should be < 1 for most species,
        indicating selection reduces gene singletons relative to pseudogenes."""
        frac_below_one = np.mean(self.data['si_sp'] < 1.0)
        assert frac_below_one > 0.90, (
            f"Only {frac_below_one*100:.1f}% species have si_sp < 1"
        )

    def test_si_sp_predicts_fluidity_beyond_core_dnds(self):
        """si_sp (accessory gene selection) should predict fluidity even after
        controlling for core dN/dS (Ne proxy). This separates accessory gene
        selection from population-size effects."""
        from selection_analysis import partial_spearman
        si_sp = self.data['si_sp']
        fluidity = self.data['genomic_fluidity']
        dnds = self.data['dnds']
        rho, p = partial_spearman(si_sp, fluidity, dnds)
        assert rho > 0, f"Partial rho(si_sp, fluidity | dN/dS) should be positive, got {rho:.3f}"
        assert p < 0.001, f"Partial correlation not significant: p={p:.4f}"

    def test_dnds_fluidity_robust_to_pseudo_burden(self):
        """dN/dS vs fluidity should remain significant after controlling
        for pseudogene burden (rejects selfish DNA hypothesis)."""
        from selection_analysis import partial_spearman
        dnds = self.data['dnds']
        fluidity = self.data['genomic_fluidity']
        pseudo_ratio = self.data['mean_num_pseudo'] / self.data['mean_num_genes']
        rho, p = partial_spearman(dnds, fluidity, pseudo_ratio)
        assert rho < 0, f"Partial rho should be negative, got {rho:.3f}"
        assert p < 0.001, f"Partial correlation not significant: p={p:.4f}"

    def test_dnds_fluidity_holds_within_gammaproteobacteria(self):
        """The largest class should independently show the correlation."""
        from scipy.stats import spearmanr
        mask = self.data['taxonomic_class'] == 'c__Gammaproteobacteria'
        assert mask.sum() > 100, "Too few Gammaproteobacteria"
        rho, p = spearmanr(self.data['dnds'][mask],
                           self.data['genomic_fluidity'][mask])
        assert rho < 0, f"Within-class rho should be negative, got {rho:.3f}"
        assert p < 0.01, f"Within-class correlation not significant: p={p:.4f}"


# ---------------------------------------------------------------------------
# Test 3: No circularity — nothing is fabricated
# ---------------------------------------------------------------------------

class TestNoCircularity:
    """Ensure the script does not fabricate data to guarantee results."""

    def test_no_rng_normal_in_load(self):
        """load_real_data must not use np.random to generate values."""
        import inspect
        from selection_analysis import load_real_data
        source = inspect.getsource(load_real_data)
        assert 'rng.normal' not in source, "load_real_data uses rng.normal (fabrication)"
        assert 'np.random' not in source, "load_real_data uses np.random (fabrication)"
        assert 'random.default_rng' not in source, "load_real_data uses RNG (fabrication)"

    def test_no_hardcoded_dnds_generation(self):
        """No function should generate fake dN/dS distributions."""
        import inspect
        from selection_analysis import load_real_data
        source = inspect.getsource(load_real_data)
        # Check for the specific circularity pattern from the old script
        assert 'normal(1.0' not in source, "Contains hardcoded neutral dN/dS generation"
        assert 'clip(dnds' not in source, "Contains clipping of generated dN/dS"

    def test_no_fluidity_tercile_relabelling(self):
        """Species should not be relabelled as 'core' or 'accessory' genes."""
        import inspect
        from selection_analysis import load_real_data
        source = inspect.getsource(load_real_data)
        assert 'tercile' not in source.lower(), "Script splits by fluidity terciles"
        assert 'core_mask' not in source, "Script creates fake core/accessory from species"


# ---------------------------------------------------------------------------
# Test 4: Panel functions exist and accept the right data
# ---------------------------------------------------------------------------

class TestPanelFunctions:
    """Verify that panel functions exist and can be called."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from selection_analysis import load_real_data
        self.data = load_real_data()

    def test_panel_a_exists(self):
        from selection_analysis import panel_a_singleton_comparison
        assert callable(panel_a_singleton_comparison)

    def test_panel_b_exists(self):
        from selection_analysis import panel_c_dnds_vs_fluidity
        assert callable(panel_c_dnds_vs_fluidity)

    def test_panel_c_exists(self):
        from selection_analysis import panel_d_core_vs_accessory_selection
        assert callable(panel_d_core_vs_accessory_selection)

    def test_panel_d_exists(self):
        from selection_analysis import panel_e_partial_correlations
        assert callable(panel_e_partial_correlations)

    def test_panel_e_exists(self):
        from selection_analysis import panel_f_within_class
        assert callable(panel_f_within_class)

    def test_all_panels_run_without_error(self):
        """All five panels should produce a figure without error."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from selection_analysis import (panel_a_singleton_comparison,
                                         panel_c_dnds_vs_fluidity,
                                         panel_d_core_vs_accessory_selection,
                                         panel_e_partial_correlations,
                                         panel_f_within_class)

        fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
        panel_a_singleton_comparison(axes[0, 0], self.data)
        panel_c_dnds_vs_fluidity(axes[0, 1], self.data)
        panel_d_core_vs_accessory_selection(axes[0, 2], self.data)
        panel_e_partial_correlations(axes[1, 0], self.data)
        panel_f_within_class(axes[1, 1], self.data)
        plt.close(fig)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
