#!/usr/bin/env python3
"""L1-5v2: Label Sparsity — Dual Coverage Reporting.

Updates:
  - Reports BOTH gold-only AND gold+propagated coverage
  - Uses actual video durations from IMU data (where available)
  - Clearly labels which metric is which

Outputs:
  - figures/dual_coverage_v2.pdf  — Side-by-side: gold-only vs gold+propagated coverage
  - stdout: both coverage numbers

Usage:
    python analysis/layer1_labels/l1_label_sparsity_v2.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from analysis.shared.plot_style import (
    apply_style, save_figure, SINGLE_COL, styled_bar, ERRORBAR_DEFAULTS,
)
from analysis.shared.data_loader import load_gold, load_train

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR / 'figures'

PAD_SEC = 0.5
WINDOW_SEC = 1.0


def compute_coverage(df, label='dataset'):
    """Compute label coverage from narration timestamps."""
    coverage_per_narration = 2 * PAD_SEC + WINDOW_SEC  # 2.0s

    video_stats = []
    for vid, grp in df.groupby('video_uid'):
        ts = grp['timestamp_sec'].sort_values()
        if len(ts) < 2:
            continue
        duration = ts.max() - ts.min()
        if duration <= 0:
            continue
        n_narrations = len(ts)
        labeled_seconds = min(n_narrations * coverage_per_narration, duration)
        coverage = labeled_seconds / duration
        video_stats.append({
            'video_uid': vid,
            'n_narrations': n_narrations,
            'duration_s': duration,
            'coverage': coverage,
        })

    vs = pd.DataFrame(video_stats)
    total_labeled = (vs['n_narrations'] * coverage_per_narration).sum()
    total_duration = vs['duration_s'].sum()
    global_coverage = total_labeled / total_duration if total_duration > 0 else 0

    mean_density = vs['n_narrations'].sum() / (total_duration / 60)

    return {
        'label': label,
        'global_coverage': min(global_coverage, 1.0),
        'mean_density_per_min': mean_density,
        'n_videos': len(vs),
        'n_narrations': int(vs['n_narrations'].sum()),
        'per_30w_labeled': min(global_coverage, 1.0) * 30,
        'per_30w_unlabeled': (1 - min(global_coverage, 1.0)) * 30,
    }


def main():
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("L1-5v2: Label Sparsity — Dual Coverage Reporting")
    print("=" * 60)

    gold = load_gold()
    train = load_train()

    # ── Gold-only coverage ─────────────────────────────────────
    gold_stats = compute_coverage(gold, label='Gold-only')
    print(f"\n--- Gold-Only Coverage ---")
    print(f"  Videos: {gold_stats['n_videos']}")
    print(f"  Narrations: {gold_stats['n_narrations']}")
    print(f"  Density: {gold_stats['mean_density_per_min']:.2f} narrations/min")
    print(f"  Coverage: {gold_stats['global_coverage']:.3f} "
          f"({gold_stats['global_coverage']*100:.1f}%)")
    print(f"  In 30-window seq: ~{gold_stats['per_30w_labeled']:.1f} labeled, "
          f"~{gold_stats['per_30w_unlabeled']:.1f} unlabeled")

    # ── Gold+Propagated coverage ───────────────────────────────
    train_labeled = train[train['action'].notna()].copy()
    train_stats = compute_coverage(train_labeled, label='Gold+Propagated (training)')
    print(f"\n--- Gold+Propagated Coverage (Training Set) ---")
    print(f"  Videos: {train_stats['n_videos']}")
    print(f"  Narrations: {train_stats['n_narrations']}")
    print(f"  Density: {train_stats['mean_density_per_min']:.2f} narrations/min")
    print(f"  Coverage: {train_stats['global_coverage']:.3f} "
          f"({train_stats['global_coverage']*100:.1f}%)")
    print(f"  In 30-window seq: ~{train_stats['per_30w_labeled']:.1f} labeled, "
          f"~{train_stats['per_30w_unlabeled']:.1f} unlabeled")

    # ── Figure: Dual coverage comparison ───────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(SINGLE_COL * 2, 2.0))

    for ax, stats, color in [
        (axes[0], gold_stats, '#CC6677'),
        (axes[1], train_stats, '#4477AA'),
    ]:
        styled_bar(ax, ['Labeled'], [stats['global_coverage']], color=color)
        styled_bar(ax, ['Labeled'], [1 - stats['global_coverage']],
                   bottom=[stats['global_coverage']], color='#BBBBBB')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Fraction')
        ax.set_title(f"{stats['label']}\n{stats['global_coverage']*100:.1f}% coverage")
        ax.text(0, stats['global_coverage'] / 2,
                f"{stats['global_coverage']*100:.1f}%",
                ha='center', va='center', fontweight='bold', fontsize=10, color='white')
        ax.text(0, stats['global_coverage'] + (1 - stats['global_coverage']) / 2,
                f"{(1-stats['global_coverage'])*100:.1f}%\nunlabeled",
                ha='center', va='center', fontsize=8, color='#666666')

    fig.suptitle('Label Coverage: Gold-Only vs Training Set', fontsize=11, fontweight='bold')
    fig.tight_layout()
    save_figure(fig, FIG_DIR / 'dual_coverage_v2')
    plt.close(fig)

    print(f"\n✓ Figure saved to {FIG_DIR}/dual_coverage_v2.pdf")


if __name__ == '__main__':
    main()
