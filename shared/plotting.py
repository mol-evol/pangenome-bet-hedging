"""
================================================================================
SHARED PLOTTING UTILITIES
================================================================================

Single source of truth for colours, matplotlib setup, figure saving,
and progress display across all figure and analysis scripts.

Usage:
    from shared.plotting import COLORS, setup_plotting, save_figure
    from shared.plotting import print_header, print_progress, add_panel_label
================================================================================
"""

import os
import matplotlib as mpl
import matplotlib.pyplot as plt


# ── Colour palette (Paul Tol's bright scheme) ────────────────────────────────

COLORS = {
    'blue': '#0077BB',
    'orange': '#EE7733',
    'green': '#009988',
    'red': '#CC3311',
    'purple': '#AA3377',
    'grey': '#888888',
    'light_blue': '#33BBEE',
    'light_orange': '#EE9966',
}


# ── Matplotlib configuration ─────────────────────────────────────────────────

def setup_plotting():
    """Configure matplotlib for publication-quality figures.

    Style: seaborn-whitegrid, Arial/Helvetica, top/right spines removed.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams.update({
        'figure.dpi': 150,
        'savefig.dpi': 600,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


# ── Figure saving ────────────────────────────────────────────────────────────

def save_figure(fig, basename, output_dir='.', fmt='png'):
    """Save figure in one or more formats.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    basename : str
        File stem without extension (e.g. 'figure1_variance').
    output_dir : str
        Directory to write into (created if needed).
    fmt : str
        'png', 'pdf', 'svg', or 'all'.

    Returns
    -------
    list[str] : paths of saved files.
    """
    formats = ['png', 'pdf', 'svg'] if fmt == 'all' else [fmt]
    os.makedirs(output_dir, exist_ok=True)

    saved = []
    for f in formats:
        path = os.path.join(output_dir, f'{basename}.{f}')
        fig.savefig(path, dpi=600, bbox_inches='tight', facecolor='white')
        saved.append(path)
    return saved


# ── Console helpers ──────────────────────────────────────────────────────────

def print_header(title, params=None):
    """Print a formatted header block.

    Parameters
    ----------
    title : str
        E.g. 'FIGURE 1: THE COST OF VARIANCE'
    params : dict or None
        Key-value pairs to display below the title.
    """
    print()
    print('=' * 70)
    print(title)
    print('=' * 70)
    if params:
        print()
        for k, v in params.items():
            print(f'  {k}: {v}')
    print()


def print_progress(current, total, prefix='Progress'):
    """Print an in-place progress bar."""
    pct = 100 * current / total
    bar_length = 40
    filled = int(bar_length * current / total)
    bar = '█' * filled + '░' * (bar_length - filled)
    print(f'\r  {prefix}: |{bar}| {pct:5.1f}% ({current}/{total})',
          end='', flush=True)
    if current == total:
        print()


def add_panel_label(ax, label, x=-0.12, y=1.05):
    """Add a bold panel label (A, B, C …) to an axes."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=12, fontweight='bold')
