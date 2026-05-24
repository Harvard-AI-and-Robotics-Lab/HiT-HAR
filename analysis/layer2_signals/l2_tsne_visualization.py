#!/usr/bin/env python3
"""L2-3: t-SNE/UMAP Visualization of IMU Features.

Research question: What does the IMU feature space look like?
Are action classes visually clustered or interleaved?

Inputs:
  - analysis/layer2_signals/outputs/gold_imu_features.csv (from L2-1)

Outputs:
  - figures/tsne_5class.pdf         — t-SNE colored by 5-class action
  - figures/tsne_4class.pdf         — t-SNE colored by 4-class
  - figures/tsne_3class.pdf         — t-SNE colored by 3-class
  - figures/tsne_tier.pdf           — t-SNE colored by quality tier
  - figures/tsne_confusable.pdf     — t-SNE highlighting OT vs TaskOp overlap

Usage:
    python analysis/layer2_signals/l2_tsne_visualization.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from analysis.shared.plot_style import (
    apply_style, save_figure, SINGLE_COL, DOUBLE_COL,
    COLORS_5CLASS, COLORS_4CLASS, COLORS_3CLASS, COLORS_TIER,
)
from analysis.shared.data_loader import (
    ACTION_ORDER, ACTION_ORDER_4CLASS, ACTION_ORDER_3CLASS,
    map_to_4class, map_to_3class,
)

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR / 'figures'
FEATURE_CSV = SCRIPT_DIR / 'outputs' / 'gold_imu_features.csv'

META_COLS = {'video_uid', 'timestamp_sec', 'action', 'scenario', 'tier', 'confidence'}


def main():
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("L2-3: t-SNE Visualization of IMU Features")
    print("=" * 60)

    df = pd.read_csv(FEATURE_CSV)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    df = df.dropna(subset=feature_cols)

    # Subsample for t-SNE (max 8000 points for speed)
    max_n = 8000
    if len(df) > max_n:
        df_sub = df.groupby('action', group_keys=False).apply(
            lambda x: x.sample(min(len(x), max_n // 5), random_state=42)
        )
    else:
        df_sub = df
    print(f"Using {len(df_sub)} samples for t-SNE")

    # Standardize features
    X = StandardScaler().fit_transform(df_sub[feature_cols].values)

    # Run t-SNE
    print("Running t-SNE (perplexity=30, n_iter=1000)...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000,
                random_state=42, init='pca', learning_rate='auto')
    embedding = tsne.fit_transform(X)
    df_sub = df_sub.copy()
    df_sub['tsne_x'] = embedding[:, 0]
    df_sub['tsne_y'] = embedding[:, 1]

    # -- Figure 1: 5-class --
    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.6, DOUBLE_COL * 0.5))
    for cls in ACTION_ORDER:
        mask = df_sub['action'] == cls
        ax.scatter(df_sub.loc[mask, 'tsne_x'], df_sub.loc[mask, 'tsne_y'],
                   c=COLORS_5CLASS[cls], s=8, alpha=0.5, label=cls, edgecolors='none')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('IMU Feature Space (5-Class)')
    ax.legend(markerscale=3, frameon=True, fancybox=False, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    save_figure(fig, FIG_DIR / 'tsne_5class')
    plt.close(fig)

    # -- Figure 2: 4-class --
    df_sub['action_4'] = df_sub['action'].map(map_to_4class)
    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.6, DOUBLE_COL * 0.5))
    for cls in ACTION_ORDER_4CLASS:
        mask = df_sub['action_4'] == cls
        ax.scatter(df_sub.loc[mask, 'tsne_x'], df_sub.loc[mask, 'tsne_y'],
                   c=COLORS_4CLASS[cls], s=8, alpha=0.5, label=cls, edgecolors='none')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('IMU Feature Space (4-Class)')
    ax.legend(markerscale=3, frameon=True, fancybox=False, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    save_figure(fig, FIG_DIR / 'tsne_4class')
    plt.close(fig)

    # -- Figure 3: 3-class --
    df_sub['action_3'] = df_sub['action'].map(map_to_3class)
    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.6, DOUBLE_COL * 0.5))
    for cls in ACTION_ORDER_3CLASS:
        mask = df_sub['action_3'] == cls
        ax.scatter(df_sub.loc[mask, 'tsne_x'], df_sub.loc[mask, 'tsne_y'],
                   c=COLORS_3CLASS[cls], s=8, alpha=0.5, label=cls, edgecolors='none')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('IMU Feature Space (3-Class)')
    ax.legend(markerscale=3, frameon=True, fancybox=False, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    save_figure(fig, FIG_DIR / 'tsne_3class')
    plt.close(fig)

    # -- Figure 4: Tier overlay --
    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.6, DOUBLE_COL * 0.5))
    for tier in [4, 3, 2, 1]:  # Plot T1 on top
        mask = df_sub['tier'] == tier
        if mask.sum() == 0:
            continue
        ax.scatter(df_sub.loc[mask, 'tsne_x'], df_sub.loc[mask, 'tsne_y'],
                   c=COLORS_TIER[tier], s=8, alpha=0.5,
                   label=f'Tier {tier}', edgecolors='none')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('IMU Feature Space Colored by Quality Tier')
    ax.legend(markerscale=3, frameon=True, fancybox=False, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    save_figure(fig, FIG_DIR / 'tsne_tier')
    plt.close(fig)

    # -- Figure 5: Confusable pair highlight --
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.4))

    # OT vs TaskOp
    ax = axes[0]
    for cls in ACTION_ORDER:
        mask = df_sub['action'] == cls
        color = COLORS_5CLASS[cls] if cls in ('Object Transfer', 'Task Operation') else '#DDDDDD'
        alpha = 0.6 if cls in ('Object Transfer', 'Task Operation') else 0.1
        zorder = 2 if cls in ('Object Transfer', 'Task Operation') else 1
        ax.scatter(df_sub.loc[mask, 'tsne_x'], df_sub.loc[mask, 'tsne_y'],
                   c=color, s=8, alpha=alpha, zorder=zorder,
                   label=cls if cls in ('Object Transfer', 'Task Operation') else None,
                   edgecolors='none')
    ax.set_title('OT vs Task Operation')
    ax.legend(markerscale=3, frameon=True, fancybox=False, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])

    # Search vs Stationary
    ax = axes[1]
    for cls in ACTION_ORDER:
        mask = df_sub['action'] == cls
        color = COLORS_5CLASS[cls] if cls in ('Search', 'Stationary') else '#DDDDDD'
        alpha = 0.6 if cls in ('Search', 'Stationary') else 0.1
        zorder = 2 if cls in ('Search', 'Stationary') else 1
        ax.scatter(df_sub.loc[mask, 'tsne_x'], df_sub.loc[mask, 'tsne_y'],
                   c=color, s=8, alpha=alpha, zorder=zorder,
                   label=cls if cls in ('Search', 'Stationary') else None,
                   edgecolors='none')
    ax.set_title('Search vs Stationary')
    ax.legend(markerscale=3, frameon=True, fancybox=False, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.suptitle('Confusable Class Pairs in IMU Feature Space', fontsize=10)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / 'tsne_confusable')
    plt.close(fig)

    print(f"\nFigures saved to {FIG_DIR}/")


if __name__ == '__main__':
    main()
