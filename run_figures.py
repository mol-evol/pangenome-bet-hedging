#!/usr/bin/env python3
"""
================================================================================
RUN BET-HEDGING FIGURES
================================================================================

Wrapper to run all or selected figure programs.

Figure numbering follows main text v42:
  Figure 1: Variance + mechanism (figure1_variance_mechanism.py)
  Figure 2: HGT equilibrium (figure2_hgt.py)
  Figure 3: U-shape + complexity (figure3_ushape_complexity.py)
  Figure 4: E. coli coupling (empirical/figure4_coupling.py)
  Figure 5: Niche insurance (empirical/figure5_insurance.py)

USAGE:
    python run_figures.py                    # Run all main figures
    python run_figures.py --figures 1 3      # Run specific figures
    python run_figures.py --quick            # Quick mode for all
    python run_figures.py --figures 2 --show # Run figure 2 and display
    python run_figures.py --supplementary    # Run supplementary figures
    python run_figures.py --empirical        # Run empirical figures (4-5)
    python run_figures.py --all              # Run everything

================================================================================
"""

import argparse
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, 'figures')
EMPIRICAL_DIR = os.path.join(SCRIPT_DIR, 'empirical')

FIGURES = {
    1: {'script': os.path.join(FIGURES_DIR, 'figure1_variance_mechanism.py'),
        'name': 'Variance + Mechanism (5-panel, A–E)'},
    2: {'script': os.path.join(FIGURES_DIR, 'figure2_hgt.py'),
        'name': 'HGT Equilibrium (4-panel)'},
    3: {'script': os.path.join(FIGURES_DIR, 'figure3_ushape_complexity.py'),
        'name': 'U-Shape + Complexity (6-panel, A–F)'},
    4: {'script': os.path.join(EMPIRICAL_DIR, 'figure4_coupling.py'),
        'name': 'Environment Coupling & Classification (4-panel)'},
    5: {'script': os.path.join(EMPIRICAL_DIR, 'figure5_insurance.py'),
        'name': 'Niche Insurance (6-panel)'},
}

SUPPLEMENTARY = {
    'S1': {'script': os.path.join(FIGURES_DIR, 'supplementary_figure_s1_frequency.py'),
           'name': 'Optimal Population Diversity'},
    'S2': {'script': os.path.join(FIGURES_DIR, 'supplementary_figure_s2_parameter_sensitivity.py'),
           'name': 'Parameter Sensitivity'},
    'S3': {'script': os.path.join(FIGURES_DIR, 'supplementary_figure_s3_hgt_optimization.py'),
           'name': 'HGT Rate Optimisation'},
    'S4': {'script': os.path.join(EMPIRICAL_DIR, 'selection_analysis.py'),
           'name': 'Selection Reanalysis'},
    'S5': {'script': os.path.join(EMPIRICAL_DIR, 'variance_analysis.py'),
           'name': 'Variance Reanalysis'},
    'S6-S10': {'script': os.path.join(EMPIRICAL_DIR, 'supplementary_figures_s6_s10.py'),
               'name': 'Empirical Supplementary (S6–S10)'},
    'S11': {'script': os.path.join(EMPIRICAL_DIR, 'supplementary_figure_s11_contamination.py'),
            'name': 'Contamination Sensitivity'},
}

EMPIRICAL = {}  # Figures 4-5 are in FIGURES dict above


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run bet-hedging figure programs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--figures', '-f', nargs='+', type=int,
                        choices=[1, 2, 3, 4, 5], default=[1, 2, 3])
    parser.add_argument('--supplementary', '-s', action='store_true')
    parser.add_argument('--empirical', '-e', action='store_true')
    parser.add_argument('--all', '-a', action='store_true')
    parser.add_argument('--quick', '-q', action='store_true')
    parser.add_argument('--output-dir', '-o', type=str, default=None)
    parser.add_argument('--format', type=str,
                        choices=['png', 'pdf', 'svg', 'all'], default='png')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def run_one(script, args, output_dir):
    cmd = [sys.executable, script,
           '--output-dir', output_dir,
           '--format', args.format]
    # Only figure scripts accept --seed and --quick; empirical scripts do not
    is_figure = os.path.join('figures', '') in script
    if is_figure:
        cmd.extend(['--seed', str(args.seed)])
    if args.quick and is_figure:
        cmd.append('--quick')
    if args.show:
        cmd.append('--show')
    return cmd


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.join(SCRIPT_DIR, 'output')

    print()
    print("=" * 70)
    print("BET-HEDGING FIGURES — BATCH RUNNER")
    print("=" * 70)
    print(f"  Output:  {output_dir}")
    print(f"  Format:  {args.format}")
    print(f"  Quick:   {args.quick}")
    print()

    tasks = []
    fig_nums = list(FIGURES.keys()) if args.all else args.figures
    for fig_num in fig_nums:
        info = FIGURES[fig_num]
        tasks.append((f"Figure {fig_num}", info['script'], info['name']))

    if args.supplementary or args.all:
        for key, info in SUPPLEMENTARY.items():
            tasks.append((f"Figure {key}", info['script'], info['name']))

    if args.dry_run:
        print("DRY RUN:")
        for label, script, name in tasks:
            cmd = run_one(script, args, output_dir)
            print(f"  {label} ({name}): {' '.join(cmd)}")
        return

    results = []
    total_start = time.time()

    for label, script, name in tasks:
        cmd = run_one(script, args, output_dir)
        print("=" * 70)
        print(f"{label.upper()}: {name.upper()}")
        print("=" * 70)
        start = time.time()
        result = subprocess.run(cmd)
        elapsed = time.time() - start
        results.append({'label': label, 'name': name,
                        'success': result.returncode == 0, 'time': elapsed})
        print()

    total_elapsed = time.time() - total_start

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Figure':<12} {'Name':<40} {'Status':<8} {'Time':<8}")
    print("-" * 68)
    for r in results:
        status = "OK" if r['success'] else "FAILED"
        print(f"{r['label']:<12} {r['name']:<40} {status:<8} {r['time']:.1f}s")
    print("-" * 68)
    print(f"{'Total':<52} {'':<8} {total_elapsed:.1f}s")
    print()

    failed = [r for r in results if not r['success']]
    if failed:
        print(f"WARNING: {len(failed)} figure(s) failed")
        sys.exit(1)
    else:
        print("All figures completed successfully!")


if __name__ == '__main__':
    main()
