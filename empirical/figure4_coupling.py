#!/usr/bin/env python3
"""
================================================================================
FIGURE 4: Environment Coupling and Gene Classification
================================================================================

Main-text figure for the paper. 4 panels (2×2):

  A — PERMANOVA variance decomposition (15.5% source, 59.1% phylogroup)
  B — Forest plot of effect size across phylogroups (B2, D, F all >5%)
  C — Volcano plot (58% niche, 30.5% insurance, 11.4% ambiguous)
  D — P-value histogram with Storey π₀ estimate

Data: Horesh et al. (2021) E. coli, 2,579 genomes.

USAGE
-----
  cd decoupling_analysis
  python figure4_coupling.py [--output-dir DIR] [--format FMT]
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

# Data functions (now all in same directory)
from decoupling_analysis import (load_horesh_data,
                                  panel_b_permanova_decomposition,
                                  panel_d_effect_across_phylogroups)
from gene_classification import (load_classification_data,
                                  panel_a_volcano,
                                  panel_b_pvalue_histogram)


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Figure 5: Environment Coupling')
    parser.add_argument('--output-dir', default='.', help='Output directory')
    parser.add_argument('--format', default='png')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    setup_plotting()
    print_header('FIGURE 4: Environment Coupling + Gene Classification')

    # Load data
    print('\n[1/3] Loading data...')
    horesh_data = load_horesh_data()
    class_data = load_classification_data()

    # Build figure
    print('\n[2/3] Generating 4-panel figure...')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel A: PERMANOVA
    panel_b_permanova_decomposition(axes[0, 0], horesh_data)
    add_panel_label(axes[0, 0], 'A')

    # Panel B: Forest plot
    panel_d_effect_across_phylogroups(axes[0, 1], horesh_data)
    add_panel_label(axes[0, 1], 'B')

    # Panel C: Volcano plot
    panel_a_volcano(axes[1, 0], class_data)
    add_panel_label(axes[1, 0], 'C')

    # Panel D: P-value histogram
    panel_b_pvalue_histogram(axes[1, 1], class_data)
    add_panel_label(axes[1, 1], 'D')

    fig.tight_layout(w_pad=3.0, h_pad=2.5)
    saved = save_figure(fig, 'figure4_coupling', output_dir=args.output_dir,
                        fmt=args.format)
    for p in saved:
        print(f'    → {p}')

    if args.show:
        plt.show()
    else:
        plt.close(fig)

    print('\n[3/3] Done.')


if __name__ == '__main__':
    main()
