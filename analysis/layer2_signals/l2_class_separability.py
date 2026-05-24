#!/usr/bin/env python3
"""L2-2: Class Pair Separability in IMU Feature Space.

Research question: How separable are the 5 action classes in IMU feature
space? Computes Bhattacharyya distance and Fisher discriminant ratio for
all class pairs under 5-class, 4-class, and 3-class taxonomies.

Inputs:
  - analysis/layer2_signals/outputs/gold_imu_features.csv (from L2-1)

Outputs:
  - figures/separability_heatmap_5class.pdf   — Bhattacharyya distance matrix
  - figures/separability_comparison.pdf       — 5 vs 4 vs 3 class mean separability
  - figures/fisher_ratio_by_feature.pdf       — Top discriminative features per pair
  - stdout: separability metrics

Usage:
    python analysis/layer2_signals/l2_class_separability.py
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
from analysis.shared.plot_style import apply_style, save_figure, SINGLE_COL, DOUBLE_COL
from analysis.shared.data_loader import (
    ACTION_ORDER, ACTION_ORDER_4CLASS, ACTION_ORDER_3CLASS,
    map_to_4class, map_to_3class,
)

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR / 'figures'
FEATURE_CSV = SCRIPT_DIR / 'outputs' / 'gold_imu_features.csv'

# Feature columns (exclude metadata)
META_COLS = {'video_uid', 'timestamp_sec', 'action', 'scenario', 'tier', 'confidence'}


def bhattacharyya_distance(mu1, cov1, mu2, cov2):
    """Compute Bhattacharyya distance between two multivariate Gaussians.

    Higher = more separable. 0 = identical distributions.
    """
    cov_avg = (cov1 + cov2) / 2
    diff = mu1 - mu2

    try:
        sign, logdet_avg = np.linalg.slogdet(cov_avg)
        _, logdet1 = np.linalg.slogdet(cov1)
        _, logdet2 = np.linalg.slogdet(cov2)

        if sign <= 0:
            return 0.0

        cov_avg_inv = np.linalg.pinv(cov_avg)
        term1 = 0.125 * diff @ cov_avg_inv @ diff
        term2 = 0.5 * (logdet_avg - 0.5 * (logdet1 + logdet2))
        return float(term1 + term2)
    except np.linalg.LinAlgError:
        return 0.0


def fisher_discriminant_ratio(x1, x2):
    """Compute per-feature Fisher discriminant ratio.

    FDR = (mu1 - mu2)^2 / (var1 + var2). Higher = more discriminative.
    """
    mu1, mu2 = x1.mean(axis=0), x2.mean(axis=0)
    var1, var2 = x1.var(axis=0), x2.var(axis=0)
    denom = var1 + var2
    denom[denom == 0] = 1e-10
    return (mu1 - mu2) ** 2 / denom


def compute_separability_matrix(df, feature_cols, class_col, class_order):
    """Compute pairwise Bhattacharyya distance matrix."""
    n = len(class_order)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            c1 = df[df[class_col] == class_order[i]][feature_cols].dropna()
            c2 = df[df[class_col] == class_order[j]][feature_cols].dropna()

            if len(c1) < 10 or len(c2) < 10:
                continue

            mu1, mu2 = c1.mean().values, c2.mean().values
            cov1, cov2 = np.cov(c1.values, rowvar=False), np.cov(c2.values, rowvar=False)

            # Regularize covariance
            reg = 1e-6 * np.eye(len(feature_cols))
            bd = bhattacharyya_distance(mu1, cov1 + reg, mu2, cov2 + reg)
            matrix[i, j] = bd
            matrix[j, i] = bd

    return pd.DataFrame(matrix, index=class_order, columns=class_order)


def main():
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("L2-2: Class Pair Separability in IMU Feature Space")
    print("=" * 60)

    df = pd.read_csv(FEATURE_CSV)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    # Drop rows with NaN features
    df = df.dropna(subset=feature_cols)
    print(f"Samples: {len(df)}, Features: {len(feature_cols)}")

    # -- 1. 5-class separability --
    print("\n--- 5-Class Bhattacharyya Distance Matrix ---")
    sep_5 = compute_separability_matrix(df, feature_cols, 'action', ACTION_ORDER)
    print(sep_5.round(3).to_string())

    # Mean pairwise separability
    upper = sep_5.values[np.triu_indices(5, k=1)]
    print(f"\nMean pairwise BD (5-class): {upper.mean():.4f}")
    print(f"Min pair: {upper.min():.4f}, Max pair: {upper.max():.4f}")

    # Figure 1: 5-class heatmap
    fig, ax = plt.subplots(figsize=(SINGLE_COL + 0.5, SINGLE_COL + 0.2))
    mask = np.eye(5, dtype=bool)
    sns.heatmap(sep_5, annot=True, fmt='.3f', cmap='YlGnBu', mask=mask,
                linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8, 'label': 'Bhattacharyya Distance'})
    ax.set_title('IMU Signal Separability (5-Class)')
    fig.tight_layout()
    save_figure(fig, FIG_DIR / 'separability_heatmap_5class')
    plt.close(fig)

    # -- 2. 4-class and 3-class comparison --
    print("\n--- Taxonomy Separability Comparison ---")
    df_4 = df.copy()
    df_4['action_4'] = df_4['action'].map(map_to_4class)
    sep_4 = compute_separability_matrix(df_4, feature_cols, 'action_4', ACTION_ORDER_4CLASS)

    df_3 = df.copy()
    df_3['action_3'] = df_3['action'].map(map_to_3class)
    sep_3 = compute_separability_matrix(df_3, feature_cols, 'action_3', ACTION_ORDER_3CLASS)

    upper_4 = sep_4.values[np.triu_indices(4, k=1)]
    upper_3 = sep_3.values[np.triu_indices(3, k=1)]

    print(f"Mean pairwise BD — 5-class: {upper.mean():.4f}, "
          f"4-class: {upper_4.mean():.4f}, 3-class: {upper_3.mean():.4f}")
    print(f"Min pairwise BD — 5-class: {upper.min():.4f}, "
          f"4-class: {upper_4.min():.4f}, 3-class: {upper_3.min():.4f}")

    # Figure 2: Comparison bar chart
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.2))
    taxonomies = ['5-class', '4-class', '3-class']
    means = [upper.mean(), upper_4.mean(), upper_3.mean()]
    mins = [upper.min(), upper_4.min(), upper_3.min()]
    x = np.arange(len(taxonomies))
    width = 0.35
    ax.bar(x - width/2, means, width, label='Mean BD', color='#4477AA')
    ax.bar(x + width/2, mins, width, label='Min BD (hardest pair)', color='#EE6677')
    ax.set_xticks(x)
    ax.set_xticklabels(taxonomies)
    ax.set_ylabel('Bhattacharyya Distance')
    ax.set_title('IMU Separability by Taxonomy')
    ax.legend(frameon=True, fancybox=False)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / 'separability_comparison')
    plt.close(fig)

    # -- 3. Per-feature Fisher discriminant ratio --
    print("\n--- Top Discriminative Features (OT vs TaskOp) ---")
    ot = df[df['action'] == 'Object Transfer'][feature_cols].dropna().values
    to = df[df['action'] == 'Task Operation'][feature_cols].dropna().values
    fdr = fisher_discriminant_ratio(ot, to)
    fdr_series = pd.Series(fdr, index=feature_cols).sort_values(ascending=False)
    print("Top 10 features:")
    for feat, val in fdr_series.head(10).items():
        print(f"  {feat:30s}: FDR={val:.4f}")

    # Figure 3: Top features
    fig, ax = plt.subplots(figsize=(SINGLE_COL + 0.5, 3.0))
    top_n = 15
    top_feats = fdr_series.head(top_n)
    ax.barh(top_feats.index[::-1], top_feats.values[::-1],
            color='#4477AA', edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Fisher Discriminant Ratio')
    ax.set_title('Top Discriminative Features\n(Object Transfer vs Task Operation)')
    fig.tight_layout()
    save_figure(fig, FIG_DIR / 'fisher_ratio_by_feature')
    plt.close(fig)

    print(f"\nFigures saved to {FIG_DIR}/")


if __name__ == '__main__':
    main()
