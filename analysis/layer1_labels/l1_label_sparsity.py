#!/usr/bin/env python3
"""L1-5: Label Sparsity & Temporal Density Analysis.

Research question: How sparse are action labels? What fraction of IMU
windows actually receive supervision? What is the temporal gap distribution?

Outputs:
  - figures/narration_density_histogram.pdf  — Narrations per minute distribution
  - figures/temporal_gap_distribution.pdf    — CDF of inter-narration gaps
  - figures/window_coverage_estimate.pdf     — Estimated labeled vs unlabeled windows
  - figures/action_transition_matrix.pdf     — Action-to-action transition probabilities
  - stdout: sparsity statistics

Usage:
    python analysis/layer1_labels/l1_label_sparsity.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from analysis.shared.plot_style import (
    apply_style, save_figure, SINGLE_COL, DOUBLE_COL, COLORS_5CLASS,
)
from analysis.shared.data_loader import load_gold, load_train, ACTION_ORDER

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR / 'figures'


def main():
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("L1-5: Label Sparsity & Temporal Density Analysis")
    print("=" * 60)

    gold = load_gold()
    train = load_train()

    # -- 1. Narration density per video --------------------------------------
    print("\n--- Narration Density ---")
    video_stats = []
    for vid, grp in gold.groupby('video_uid'):
        ts = grp['timestamp_sec'].sort_values()
        duration = ts.max() - ts.min() if len(ts) > 1 else 60
        density = len(ts) / (duration / 60) if duration > 0 else 0
        gaps = ts.diff().dropna()
        video_stats.append({
            'video_uid': vid, 'n_narrations': len(ts),
            'duration_min': duration / 60, 'density_per_min': density,
            'median_gap_s': gaps.median() if len(gaps) > 0 else np.nan,
            'scenario': grp['scenario'].iloc[0],
        })
    vs = pd.DataFrame(video_stats)

    print(f"Videos: {len(vs)}")
    print(f"Mean density: {vs['density_per_min'].mean():.2f} narrations/min")
    print(f"Median density: {vs['density_per_min'].median():.2f} narrations/min")
    print(f"Median inter-narration gap: {vs['median_gap_s'].median():.1f}s")

    # Figure 1: Density histogram
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.2))
    ax.hist(vs['density_per_min'].clip(upper=30), bins=40, color='#4477AA',
            edgecolor='white', linewidth=0.5, alpha=0.8)
    ax.axvline(vs['density_per_min'].median(), color='#EE6677', linestyle='--',
               linewidth=1.5, label=f"Median: {vs['density_per_min'].median():.1f}/min")
    ax.set_xlabel('Narrations per Minute')
    ax.set_ylabel('# Videos')
    ax.set_title('Narration Density Distribution')
    ax.legend(frameon=True, fancybox=False)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / 'narration_density_histogram')
    plt.close(fig)

    # -- 2. Inter-narration gap distribution ---------------------------------
    print("\n--- Inter-Narration Gap Distribution ---")
    all_gaps = []
    for vid, grp in gold.groupby('video_uid'):
        ts = grp['timestamp_sec'].sort_values()
        gaps = ts.diff().dropna().values
        all_gaps.extend(gaps)
    all_gaps = np.array(all_gaps)
    all_gaps = all_gaps[all_gaps > 0]

    percentiles = [25, 50, 75, 90, 95]
    for p in percentiles:
        val = np.percentile(all_gaps, p)
        print(f"  P{p}: {val:.1f}s")
    print(f"  Mean: {all_gaps.mean():.1f}s")
    print(f"  Gaps < 2s: {(all_gaps < 2).mean()*100:.1f}%")
    print(f"  Gaps > 30s: {(all_gaps > 30).mean()*100:.1f}%")

    # Figure 2: Gap CDF
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.2))
    sorted_gaps = np.sort(all_gaps)
    cdf = np.arange(1, len(sorted_gaps) + 1) / len(sorted_gaps)
    ax.plot(sorted_gaps, cdf, color='#4477AA', linewidth=1.5)
    ax.axhline(0.5, color='#BBBBBB', linestyle=':', linewidth=0.8)
    ax.axvline(np.median(all_gaps), color='#EE6677', linestyle='--', linewidth=1,
               label=f"Median: {np.median(all_gaps):.1f}s")
    ax.set_xlabel('Inter-Narration Gap (seconds)')
    ax.set_ylabel('CDF')
    ax.set_title('Temporal Gap Distribution')
    ax.set_xlim(0, 60)
    ax.legend(frameon=True, fancybox=False)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / 'temporal_gap_distribution')
    plt.close(fig)

    # -- 3. Window coverage estimate -----------------------------------------
    print("\n--- Window Coverage Estimate ---")
    # With +/-0.5s padding, each narration covers ~2s = 2 windows (at 1s window size)
    # Estimated coverage
    pad = 0.5  # action_label_pad
    window_size = 1.0  # 50 samples @ 50Hz
    coverage_per_narration = 2 * pad + window_size  # = 2.0s
    total_labeled_seconds = len(gold) * coverage_per_narration
    total_video_seconds = vs['duration_min'].sum() * 60
    coverage_rate = total_labeled_seconds / total_video_seconds if total_video_seconds > 0 else 0

    print(f"Estimated label coverage: {coverage_rate:.3f} ({coverage_rate*100:.1f}%)")
    print(f"Estimated unlabeled windows: {(1-coverage_rate)*100:.1f}%")
    print(f"In a 30-window sequence: ~{30*coverage_rate:.1f} labeled, "
          f"~{30*(1-coverage_rate):.1f} unlabeled (-1)")

    # Figure 3: Coverage bar
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 1.5))
    ax.barh(['Windows'], [coverage_rate], color='#228833', label='Labeled')
    ax.barh(['Windows'], [1 - coverage_rate], left=[coverage_rate],
            color='#BBBBBB', label='Unlabeled (-1)')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Fraction')
    ax.set_title(f'Estimated Window Label Coverage ({coverage_rate*100:.0f}% labeled)')
    ax.legend(frameon=True, fancybox=False, fontsize=8, loc='center right')
    fig.tight_layout()
    save_figure(fig, FIG_DIR / 'window_coverage_estimate')
    plt.close(fig)

    # -- 4. Action transition matrix -----------------------------------------
    print("\n--- Action Transition Matrix ---")
    transitions = pd.DataFrame(0, index=ACTION_ORDER, columns=ACTION_ORDER)
    for vid, grp in gold.groupby('video_uid'):
        actions = grp.sort_values('timestamp_sec')['action'].values
        for i in range(len(actions) - 1):
            if actions[i] in ACTION_ORDER and actions[i+1] in ACTION_ORDER:
                transitions.loc[actions[i], actions[i+1]] += 1

    # Normalize rows
    trans_norm = transitions.div(transitions.sum(axis=1), axis=0).fillna(0)
    print(trans_norm.round(3).to_string())

    # Self-transition rates
    print("\nSelf-transition rates:")
    for cls in ACTION_ORDER:
        print(f"  {cls:20s}: {trans_norm.loc[cls, cls]:.3f}")

    # Figure 4: Transition heatmap
    fig, ax = plt.subplots(figsize=(SINGLE_COL + 0.5, SINGLE_COL + 0.2))
    sns.heatmap(trans_norm, annot=True, fmt='.2f', cmap='Blues',
                linewidths=0.5, ax=ax, vmin=0, vmax=0.6,
                cbar_kws={'shrink': 0.8})
    ax.set_xlabel('Next Action')
    ax.set_ylabel('Current Action')
    ax.set_title('Action Transition Probabilities')
    fig.tight_layout()
    save_figure(fig, FIG_DIR / 'action_transition_matrix')
    plt.close(fig)

    print(f"\nFigures saved to {FIG_DIR}/")


if __name__ == '__main__':
    main()
