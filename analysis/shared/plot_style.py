# analysis/shared/plot_style.py
"""PI Lab-inspired figure styling with Tol colorblind-safe palette.

Visual style: warm cream backgrounds, bold outlines, monospace labels.
Color safety: Paul Tol's colorblind-safe qualitative scheme.
Layout: CVPR single-column (3.25in) and double-column (6.875in).

References:
  - PI Lab style: https://www.pi.website/blog/pi05, MEM paper (2025)
  - Tol colors: https://personal.sron.nl/~pault/data/colourschemes.pdf
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl

# ── CVPR column widths ──────────────────────────────────────────────────────
SINGLE_COL = 3.25
DOUBLE_COL = 6.875

# ── Tol colorblind-safe palette (mapped to PI Lab warmth) ───────────────────
# Primary palette: Tol "bright" qualitative scheme
COLORS_5CLASS = {
    'Object Transfer': '#4477AA',   # Tol blue
    'Task Operation':  '#CC6677',   # Tol rose
    'Stationary':      '#117733',   # Tol green
    'Locomotion':      '#DDCC77',   # Tol yellow (golden, PI-like)
    'Search':          '#AA4499',   # Tol purple
}

COLORS_4CLASS = {
    'Manipulation': '#4477AA',   # blue
    'Stationary':   '#117733',   # green
    'Locomotion':   '#DDCC77',   # yellow
    'Search':       '#AA4499',   # purple
}

COLORS_3CLASS = {
    'Manipulation': '#4477AA',   # blue
    'Passive':      '#117733',   # green
    'Locomotion':   '#DDCC77',   # yellow
}

COLORS_TIER = {1: '#117733', 2: '#DDCC77', 3: '#CC6677', 4: '#BBBBBB'}

COLORS_SOURCE = {'gold': '#117733', 'propagated': '#88CCEE', 'llm': '#BBBBBB'}

# PI Lab accent colors
PI_CREAM = '#FFFFFF'
PI_GOLD = '#DDCC77'
PI_DARK = '#2D2D2D'

# ── Bar chart defaults (PI Lab style) ───────────────────────────────────────
BAR_DEFAULTS = {
    'edgecolor': PI_DARK,
    'linewidth': 1.2,
}

ERRORBAR_DEFAULTS = {
    'capsize': 4,
    'capthick': 1.5,
    'ecolor': PI_DARK,
    'elinewidth': 1.2,
}


def apply_style():
    """Apply PI Lab-inspired publication-quality matplotlib style."""
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        try:
            plt.style.use('seaborn-whitegrid')
        except OSError:
            pass
    mpl.rcParams.update({
        # Typography — sans-serif for publication figures
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 9,
        'axes.titlesize': 9,
        'axes.titleweight': 'semibold',
        'axes.labelsize': 8,
        'axes.labelweight': 'normal',
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 6.5,
        # DPI
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.08,
        # Background — PI Lab cream
        'figure.facecolor': '#FFFFFF',
        'axes.facecolor': '#FFFFFF',
        'savefig.facecolor': '#FFFFFF',
        # Grid — very subtle (PI style)
        'axes.grid': True,
        'grid.alpha': 0.15,
        'grid.linewidth': 0.5,
        # Spines — left and bottom only, bold
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 1.2,
        # Legend
        'legend.frameon': True,
        'legend.fancybox': False,
        'legend.edgecolor': '#CCCCCC',
        'legend.framealpha': 0.9,
    })


def save_figure(fig, path_stem, formats=('pdf', 'png')):
    """Save figure in multiple formats. path_stem has no extension."""
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(path_stem.with_suffix(f'.{fmt}'), facecolor=fig.get_facecolor())


def get_5class_colors(classes):
    """Return list of colors for given class names (5-class)."""
    return [COLORS_5CLASS.get(c, '#999999') for c in classes]


def get_tier_colors(tiers):
    """Return list of colors for given tier values."""
    return [COLORS_TIER.get(t, '#999999') for t in tiers]


def styled_bar(ax, x, heights, **kwargs):
    """Create a PI Lab-styled bar chart with bold outlines."""
    merged = {**BAR_DEFAULTS, **kwargs}
    # Extract error-bar-only kwargs that ax.bar doesn't accept directly;
    # pass them via error_kw so matplotlib routes them to the errorbar artist.
    _EB_ONLY = {'capthick', 'elinewidth', 'ecolor'}
    error_kw = merged.pop('error_kw', {})
    for k in list(merged):
        if k in _EB_ONLY:
            error_kw[k] = merged.pop(k)
    if error_kw:
        merged['error_kw'] = error_kw
    return ax.bar(x, heights, **merged)


def styled_barh(ax, y, widths, **kwargs):
    """Create a PI Lab-styled horizontal bar chart."""
    merged = {**BAR_DEFAULTS, **kwargs}
    return ax.barh(y, widths, **merged)
