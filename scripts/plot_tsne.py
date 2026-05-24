"""Generate publication-quality t-SNE visualizations from extracted embeddings.

Produces SEPARATE high-resolution figures for scenario and action coloring,
each with dense scatter plots, labeled centroids with distinct markers,
and a clean white background -- matching the quality of the original
HAR_Lab_Initiative_AI visualize_embeddings_fast.py figures.

Target: CVPR 2026 Sense of Space Workshop
"""
import argparse
import inspect
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for server
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path


# ---------------------------------------------------------------------------
# Color palettes  -- tab10-derived, high-saturation for dense scatter
# ---------------------------------------------------------------------------
# 8-scenario palette — Paul Tol "bright" + "vibrant" (colorblind-safe, max distinguishable)
# Each color is from a different hue family — no two are close
COLORS_SCENARIO = {
    0: '#EE6677',  # Carpentry       — rose/red
    1: '#228833',  # Cleaning        — green
    2: '#4477AA',  # Cooking         — blue
    3: '#CCBB44',  # Desk Work       — yellow
    4: '#66CCEE',  # Mechanical Repair — cyan
    5: '#AA3377',  # Playing Instrument — magenta
    6: '#BBBBBB',  # Gardening       — gray
    7: '#EE8866',  # Walking Outdoors — orange
}
SCENARIO_NAMES = {
    0: 'Carpentry',
    1: 'Cleaning',
    2: 'Cooking',
    3: 'Desk Work',
    4: 'Mech. Repair',
    5: 'Play Instr.',
    6: 'Gardening',
    7: 'Walk Outdoors',
}

# 5-action palette — Tol "muted" subset (colorblind-safe, distinct from scenario palette)
# Chosen to be visually different from scenario colors above
COLORS_ACTION = {
    0: '#332288',  # Object Transfer — indigo (dark, distinct)
    1: '#CC6677',  # Task Operation  — dusty rose
    2: '#44AA99',  # Stationary      — teal
    3: '#DDCC77',  # Locomotion      — sand/gold
    4: '#882255',  # Search          — wine/burgundy
}
ACTION_NAMES = {
    0: 'Object Transfer',
    1: 'Task Operation',
    2: 'Stationary',
    3: 'Locomotion',
    4: 'Search',
}

# Distinct centroid markers (one per class, easily distinguishable)
CENTROID_MARKERS = ['X', 'P', '*', 'D', 'H', '8', 'p', 's', '^', 'v']
SCATTER_MARKERS = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']


def apply_style():
    """Clean, publication-ready matplotlib style."""
    plt.rcParams.update({
        # Typography
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
        'font.size': 13,
        'axes.titlesize': 16,
        'axes.titleweight': 'bold',
        'axes.labelsize': 13,
        'axes.labelweight': 'bold',
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,

        # Background -- pure white
        'figure.facecolor': '#FFFFFF',
        'axes.facecolor': '#FFFFFF',
        'savefig.facecolor': '#FFFFFF',

        # Grid -- subtle dashed
        'axes.grid': True,
        'grid.alpha': 0.2,
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,

        # Spines
        'axes.spines.top': True,
        'axes.spines.right': True,
        'axes.linewidth': 0.8,

        # DPI
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })


def plot_single(coords, labels, color_map, name_map, title, out_path_base,
                marker_list=None):
    """Plot a single large t-SNE figure with centroids and labeled markers.

    Dense, vivid style: large opaque dots that fill the embedding space,
    matching the visual density of the original HAR_Lab_Initiative_AI figures.
    """
    if marker_list is None:
        marker_list = SCATTER_MARKERS

    fig, ax = plt.subplots(figsize=(12, 9))

    unique_labels = sorted(np.unique(labels))
    # Filter out negative labels (unlabeled)
    unique_labels = [l for l in unique_labels if l >= 0]

    # Sort classes largest-first so smaller/minority classes render on top
    class_counts = {lid: (labels == lid).sum() for lid in unique_labels}
    sorted_labels = sorted(unique_labels, key=lambda l: class_counts[l], reverse=True)

    for draw_i, lid in enumerate(sorted_labels):
        # Recover the canonical index for color/marker consistency
        i = unique_labels.index(lid)
        mask = labels == lid
        color = color_map.get(int(lid), '#999999')
        name = name_map.get(int(lid), f'Class {lid}')

        # Fixed high-contrast settings for dense scatter
        alpha = 0.55
        point_size = 20

        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=color,
            s=point_size,
            alpha=alpha,
            label=name,
            marker=marker_list[i % len(marker_list)],
            edgecolors='face',       # tinted edge reinforces color
            linewidths=0.18,
            rasterized=True,
            zorder=2 + draw_i,       # minority classes on top
        )

        # Centroid with large distinct marker
        cx = coords[mask, 0].mean()
        cy = coords[mask, 1].mean()
        ax.scatter(
            cx, cy,
            marker=CENTROID_MARKERS[i % len(CENTROID_MARKERS)],
            s=300,
            c=color,
            edgecolors='black',
            linewidths=2.5,
            zorder=100,
            alpha=1.0,
        )
        # Annotate centroid
        ax.annotate(
            name,
            (cx, cy),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color,
                      alpha=0.7, edgecolor='black'),
            zorder=101,
        )

    # Unlabeled points (action == -1) as faint gray background
    unlabeled_mask = labels < 0
    if unlabeled_mask.sum() > 0:
        ax.scatter(
            coords[unlabeled_mask, 0], coords[unlabeled_mask, 1],
            c='#E0E0E0', s=4, alpha=0.10,
            edgecolors='none', rasterized=True,
        )

    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('t-SNE Component 1', fontsize=13, fontweight='bold')
    ax.set_ylabel('t-SNE Component 2', fontsize=13, fontweight='bold')
    ax.set_axisbelow(True)

    # Legend below plot area (horizontal), matching reference style
    n_classes = len(unique_labels)
    legend = ax.legend(
        bbox_to_anchor=(0.5, -0.12),
        loc='upper center',
        ncol=min(n_classes, 5),     # 5 for action, up to 5 for scenario row
        fontsize=10,
        frameon=False,
        markerscale=2.5,            # bigger legend dots
    )

    plt.tight_layout()

    for fmt in ['pdf', 'png']:
        fig.savefig(f'{out_path_base}.{fmt}', dpi=300, bbox_inches='tight',
                    facecolor='white')
    print(f"  Saved {out_path_base}.{{pdf,png}}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Run t-SNE on extracted embeddings and produce high-quality figures."
    )
    parser.add_argument('--input', required=True,
                        help='Path to embeddings .npz file')
    parser.add_argument('--output-dir', default='figures/tsne',
                        help='Output directory for figures')
    parser.add_argument('--perplexity', type=int, default=30,
                        help='t-SNE perplexity')
    parser.add_argument('--n-iter', type=int, default=1000,
                        help='t-SNE iterations')
    parser.add_argument('--max-samples', type=int, default=0,
                        help='Subsample to this many points (0 = ALL points)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--tag', type=str, default='',
                        help='Filename tag (e.g., "b03", "v3b03")')
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    data = np.load(args.input)
    h_cls = data['h_cls']
    scenarios = data['scenario_labels']
    actions = data['action_labels']

    print(f"Loaded {len(h_cls)} embeddings, dim={h_cls.shape[1]}")
    print(f"  Scenarios present: {sorted(np.unique(scenarios))}")
    print(f"  Actions present:   {sorted(np.unique(actions))}")

    # Subsample only if explicitly requested (default=0 means use all)
    if args.max_samples > 0 and len(h_cls) > args.max_samples:
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(len(h_cls), args.max_samples, replace=False)
        h_cls = h_cls[idx]
        scenarios = scenarios[idx]
        actions = actions[idx]
        print(f"Subsampled to {len(h_cls)} points")
    else:
        print(f"Using ALL {len(h_cls)} points (no subsampling)")

    # ------------------------------------------------------------------
    # Run t-SNE
    # ------------------------------------------------------------------
    print(f"Running t-SNE (perplexity={args.perplexity}, n_iter={args.n_iter}) "
          f"on {len(h_cls)} points...")

    # sklearn >= 1.6 renamed n_iter -> max_iter; handle both
    _tsne_params = inspect.signature(TSNE.__init__).parameters
    iter_kwarg = 'max_iter' if 'max_iter' in _tsne_params else 'n_iter'
    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        random_state=args.seed,
        learning_rate='auto',
        init='pca',
        **{iter_kwarg: args.n_iter},
    )
    coords = tsne.fit_transform(h_cls)
    print(f"t-SNE done. KL divergence: {tsne.kl_divergence_:.4f}")

    # ------------------------------------------------------------------
    # Apply style and create output directory
    # ------------------------------------------------------------------
    apply_style()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""

    # ------------------------------------------------------------------
    # Figure 1: Colored by Scenario  (single panel, large)
    # ------------------------------------------------------------------
    print("Generating scenario figure...")
    plot_single(
        coords, scenarios,
        COLORS_SCENARIO, SCENARIO_NAMES,
        title='Scenario Embeddings (t-SNE)',
        out_path_base=str(out_dir / f'fig_tsne_scenario{tag}'),
    )

    # ------------------------------------------------------------------
    # Figure 2: Colored by Action  (single panel, large)
    # ------------------------------------------------------------------
    print("Generating action figure...")
    plot_single(
        coords, actions,
        COLORS_ACTION, ACTION_NAMES,
        title='Action Embeddings (t-SNE)',
        out_path_base=str(out_dir / f'fig_tsne_action{tag}'),
    )

    # ------------------------------------------------------------------
    # Figure 3: Compact dual-panel for paper body (CVPR column width)
    # ------------------------------------------------------------------
    print("Generating dual-panel figure...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # -- Dual-panel scatter settings: dense and vivid --
    DUAL_ALPHA = 0.50
    DUAL_SIZE = 14

    # Panel (a): Scenario — plot largest classes first, minority on top
    scenario_ids = sorted(np.unique(scenarios))
    scenario_counts = {sid: (scenarios == sid).sum() for sid in scenario_ids}
    scenario_draw_order = sorted(scenario_ids, key=lambda s: scenario_counts[s], reverse=True)

    for draw_j, sid in enumerate(scenario_draw_order):
        i = scenario_ids.index(sid)   # canonical index for color/marker
        mask = scenarios == sid
        ax1.scatter(
            coords[mask, 0], coords[mask, 1],
            c=COLORS_SCENARIO.get(int(sid), '#999999'),
            s=DUAL_SIZE, alpha=DUAL_ALPHA,
            label=SCENARIO_NAMES.get(int(sid), f'Scenario {sid}'),
            marker=SCATTER_MARKERS[i % len(SCATTER_MARKERS)],
            edgecolors='face', linewidths=0.1,
            rasterized=True,
            zorder=2 + draw_j,
        )
        # Centroid
        cx, cy = coords[mask, 0].mean(), coords[mask, 1].mean()
        ax1.scatter(cx, cy, marker=CENTROID_MARKERS[i % len(CENTROID_MARKERS)],
                    s=200, c=COLORS_SCENARIO.get(int(sid), '#999999'),
                    edgecolors='black', linewidths=2, zorder=100)
        ax1.annotate(SCENARIO_NAMES.get(int(sid), ''), (cx, cy),
                     xytext=(8, 8), textcoords='offset points',
                     fontsize=7, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.2',
                               facecolor=COLORS_SCENARIO.get(int(sid), '#999'),
                               alpha=0.7, edgecolor='black'),
                     zorder=101)

    ax1.set_title('(a) Colored by Scenario', fontsize=13, fontweight='bold')
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    ax1.set_xticks([])
    ax1.set_yticks([])
    leg1 = ax1.legend(
        bbox_to_anchor=(0.5, -0.12),
        loc='upper center',
        ncol=4,
        fontsize=8,
        frameon=False,
        markerscale=3,
        handletextpad=0.3,
        borderpad=0.4,
    )

    # Panel (b): Action — same largest-first draw order
    valid_action_ids = sorted([a for a in np.unique(actions) if a >= 0])
    action_counts = {aid: (actions == aid).sum() for aid in valid_action_ids}
    action_draw_order = sorted(valid_action_ids, key=lambda a: action_counts[a], reverse=True)

    for draw_j, aid in enumerate(action_draw_order):
        i = valid_action_ids.index(aid)
        mask = actions == aid
        ax2.scatter(
            coords[mask, 0], coords[mask, 1],
            c=COLORS_ACTION.get(int(aid), '#999999'),
            s=DUAL_SIZE, alpha=DUAL_ALPHA,
            label=ACTION_NAMES.get(int(aid), f'Action {aid}'),
            marker=SCATTER_MARKERS[i % len(SCATTER_MARKERS)],
            edgecolors='face', linewidths=0.1,
            rasterized=True,
            zorder=2 + draw_j,
        )
        cx, cy = coords[mask, 0].mean(), coords[mask, 1].mean()
        ax2.scatter(cx, cy, marker=CENTROID_MARKERS[i % len(CENTROID_MARKERS)],
                    s=200, c=COLORS_ACTION.get(int(aid), '#999999'),
                    edgecolors='black', linewidths=2, zorder=100)
        ax2.annotate(ACTION_NAMES.get(int(aid), ''), (cx, cy),
                     xytext=(8, 8), textcoords='offset points',
                     fontsize=7, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.2',
                               facecolor=COLORS_ACTION.get(int(aid), '#999'),
                               alpha=0.7, edgecolor='black'),
                     zorder=101)

    # Unlabeled as faint gray
    unlabeled_mask = actions < 0
    if unlabeled_mask.sum() > 0:
        ax2.scatter(coords[unlabeled_mask, 0], coords[unlabeled_mask, 1],
                    c='#E0E0E0', s=3, alpha=0.10,
                    edgecolors='none', rasterized=True)

    ax2.set_title('(b) Colored by Action', fontsize=13, fontweight='bold')
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    ax2.set_xticks([])
    ax2.set_yticks([])
    leg2 = ax2.legend(
        bbox_to_anchor=(0.5, -0.12),
        loc='upper center',
        ncol=5,
        fontsize=8,
        frameon=False,
        markerscale=3,
        handletextpad=0.3,
        borderpad=0.4,
    )

    plt.tight_layout(w_pad=2.0)
    for fmt in ['pdf', 'png']:
        fig.savefig(str(out_dir / f'fig_tsne_dual{tag}.{fmt}'),
                    dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved dual-panel to {out_dir}/fig_tsne_dual{tag}.{{pdf,png}}")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Save t-SNE coordinates for reproducibility
    # ------------------------------------------------------------------
    np.savez(
        out_dir / f'tsne_coords{tag}.npz',
        coords=coords,
        scenario_labels=scenarios,
        action_labels=actions,
        perplexity=args.perplexity,
        kl_divergence=tsne.kl_divergence_,
    )
    print(f"Saved t-SNE coordinates to {out_dir}/tsne_coords{tag}.npz")
    print("Done.")


if __name__ == '__main__':
    main()
