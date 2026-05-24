#!/usr/bin/env python3
"""L2-2v2: Class Pair Separability — MMD + Normality Test + Permutation P-Values.

Updates:
  - Adds MMD (nonparametric) as primary separability metric
  - Runs Mardia normality test to validate/reject BD's Gaussian assumption
  - Permutation test for each pair → p-values with significance stars
  - Retains BD as secondary metric for cross-paper comparability

Outputs:
  - figures/separability_mmd_5class_v2.pdf       — MMD matrix with significance stars
  - figures/separability_normality_v2.pdf         — Normality test results per class
  - figures/separability_dual_comparison_v2.pdf   — BD vs MMD side-by-side
  - figures/separability_taxonomy_v2.pdf          — 5/4/3 class MMD comparison
  - stdout: full metrics with p-values

Usage:
    python analysis/layer2_signals/l2_class_separability_v2.py
"""
import json
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
    apply_style, save_figure, SINGLE_COL, DOUBLE_COL, PI_CREAM,
    COLORS_5CLASS, get_5class_colors, styled_bar,
)
from analysis.shared.data_loader import (
    ACTION_ORDER, ACTION_ORDER_4CLASS, ACTION_ORDER_3CLASS,
    map_to_4class, map_to_3class,
)
from analysis.shared.stats_utils import (
    compute_mmd, permutation_test_mmd_grouped, mardia_normality_test,
    global_median_bandwidth, bonferroni_correct,
)

# Import original BD for comparison
from analysis.layer2_signals.l2_class_separability import bhattacharyya_distance

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR / 'figures'
FEATURE_CSV = SCRIPT_DIR / 'outputs' / 'gold_imu_features.csv'
META_COLS = {'video_uid', 'timestamp_sec', 'action', 'scenario', 'tier', 'confidence'}

# Subsample per class for MMD speed (permutation test is O(n²))
MAX_SAMPLES_PER_CLASS = 2000


def compute_separability_matrix_mmd(df, feature_cols, class_col, class_order,
                                     groups=None, n_permutations=1000, seed=42):
    """Compute pairwise MMD matrix with group-aware permutation p-values.

    Args:
        df: DataFrame with features and class labels
        feature_cols: list of feature column names
        class_col: column name for class labels
        class_order: ordered list of class names
        groups: column name for group labels (e.g., 'video_uid') for
                group-aware permutation test. If None, falls back to
                ungrouped permutation.
        n_permutations: number of permutations for significance test
        seed: random seed
    Returns:
        (mmd_df, pval_df) with Bonferroni-corrected p-values
    """
    n = len(class_order)
    mmd_matrix = np.zeros((n, n))
    pval_matrix = np.ones((n, n))

    # Compute global bandwidth once from all data for comparability
    all_features = df[feature_cols].dropna().values
    bandwidth = global_median_bandwidth(all_features)

    raw_pvals = []  # collect (i, j, p) for Bonferroni correction

    for i in range(n):
        for j in range(i + 1, n):
            mask_i = df[class_col] == class_order[i]
            mask_j = df[class_col] == class_order[j]
            c1 = df[mask_i][feature_cols].dropna().values
            c2 = df[mask_j][feature_cols].dropna().values

            if len(c1) < 10 or len(c2) < 10:
                continue

            # Subsample for speed
            rng = np.random.RandomState(seed)
            if groups is not None:
                g1 = df[mask_i][groups].values
                g2 = df[mask_j][groups].values
            if len(c1) > MAX_SAMPLES_PER_CLASS:
                idx = rng.choice(len(c1), MAX_SAMPLES_PER_CLASS, replace=False)
                c1 = c1[idx]
                if groups is not None:
                    g1 = g1[idx]
            if len(c2) > MAX_SAMPLES_PER_CLASS:
                idx = rng.choice(len(c2), MAX_SAMPLES_PER_CLASS, replace=False)
                c2 = c2[idx]
                if groups is not None:
                    g2 = g2[idx]

            if groups is not None:
                mmd_val, p_val = permutation_test_mmd_grouped(
                    c1, c2, g1, g2,
                    n_permutations=n_permutations,
                    bandwidth=bandwidth, seed=seed,
                )
            else:
                # Fallback: ungrouped (compute_mmd + manual permutation)
                from analysis.shared.stats_utils import permutation_test_mmd
                mmd_val, p_val = permutation_test_mmd(
                    c1, c2, n_permutations=n_permutations,
                    bandwidth=bandwidth, seed=seed,
                )

            mmd_matrix[i, j] = mmd_val
            mmd_matrix[j, i] = mmd_val
            raw_pvals.append((i, j, p_val))

    # Apply Bonferroni correction to all pairwise p-values
    if raw_pvals:
        corrected = bonferroni_correct([p for _, _, p in raw_pvals])
        for (i, j, _), p_corr in zip(raw_pvals, corrected):
            pval_matrix[i, j] = p_corr
            pval_matrix[j, i] = p_corr

    mmd_df = pd.DataFrame(mmd_matrix, index=class_order, columns=class_order)
    pval_df = pd.DataFrame(pval_matrix, index=class_order, columns=class_order)
    return mmd_df, pval_df


def significance_stars(p):
    """Convert Bonferroni-corrected p-value to significance stars."""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return 'n.s.'


def main():
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("L2-2v2: Class Separability — MMD + Normality + Permutation")
    print("=" * 60)

    df = pd.read_csv(FEATURE_CSV)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    df = df.dropna(subset=feature_cols)
    print(f"Samples: {len(df)}, Features: {len(feature_cols)}")

    # ── 1. Mardia normality test per class ──────────────────────
    print("\n--- Mardia Multivariate Normality Test ---")
    normality_results = {}
    for cls in ACTION_ORDER:
        X = df[df['action'] == cls][feature_cols].values
        if len(X) < 50:
            continue
        # Subsample for Mardia test (O(n²) in memory)
        if len(X) > 1000:
            rng = np.random.RandomState(42)
            X = X[rng.choice(len(X), 1000, replace=False)]
        result = mardia_normality_test(X)
        normality_results[cls] = result
        status = "NORMAL" if result['is_normal'] else "NOT NORMAL"
        print(f"  {cls:20s}: skew_p={result['skewness_pvalue']:.4f}, "
              f"kurt_p={result['kurtosis_pvalue']:.4f} → {status}")

    any_normal = any(r['is_normal'] for r in normality_results.values())
    all_normal = all(r['is_normal'] for r in normality_results.values())
    print(f"\n  Conclusion: {'All classes normal → BD valid' if all_normal else 'NOT all normal → MMD is primary metric'}")

    # Figure 1: Normality test results
    fig, ax = plt.subplots(figsize=(SINGLE_COL + 0.5, 2.2))
    classes = list(normality_results.keys())
    skew_pvals = [normality_results[c]['skewness_pvalue'] for c in classes]
    kurt_pvals = [normality_results[c]['kurtosis_pvalue'] for c in classes]
    x = np.arange(len(classes))
    width = 0.35
    styled_bar(ax, x - width/2, skew_pvals, width=width, label='Skewness p',
               color='#88CCEE')
    styled_bar(ax, x + width/2, kurt_pvals, width=width, label='Kurtosis p',
               color='#CC6677')
    ax.axhline(0.05, color='#2D2D2D', linestyle='--', linewidth=1, alpha=0.7, label='α=0.05')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(' ', '\n') for c in classes], fontsize=7)
    ax.set_ylabel('p-value')
    ax.set_title('Mardia Multivariate Normality Test')
    ax.legend(frameon=True, fancybox=False, fontsize=7)
    ax.set_yscale('log')
    fig.tight_layout()
    save_figure(fig, FIG_DIR / 'separability_normality_v2')
    plt.close(fig)

    # ── 2. MMD matrix with permutation p-values ────────────────
    print("\n--- 5-Class MMD Matrix (with permutation test) ---")
    # Use video_uid groups for group-aware permutation if available
    groups_col = 'video_uid' if 'video_uid' in df.columns else None
    mmd_5, pval_5 = compute_separability_matrix_mmd(
        df, feature_cols, 'action', ACTION_ORDER,
        groups=groups_col, n_permutations=1000, seed=42,
    )
    # Print with significance stars
    for i, a in enumerate(ACTION_ORDER):
        for j, b in enumerate(ACTION_ORDER):
            if i < j:
                stars = significance_stars(pval_5.loc[a, b])
                print(f"  {a:20s} ↔ {b:20s}: MMD²={mmd_5.loc[a, b]:.4f} "
                      f"p={pval_5.loc[a, b]:.4f} {stars}")

    # Figure 2: MMD heatmap with significance annotations
    fig, ax = plt.subplots(figsize=(SINGLE_COL + 0.8, SINGLE_COL + 0.3))
    mask = np.eye(5, dtype=bool)
    # Build annotation labels with stars
    annot_labels = []
    for i in range(5):
        row = []
        for j in range(5):
            if i == j:
                row.append('')
            else:
                stars = significance_stars(pval_5.iloc[i, j])
                row.append(f'{mmd_5.iloc[i, j]:.3f}\n{stars}')
        annot_labels.append(row)
    annot_array = np.array(annot_labels)

    sns.heatmap(mmd_5, annot=annot_array, fmt='', cmap='YlOrRd', mask=mask,
                linewidths=0.5, ax=ax,
                cbar_kws={'shrink': 0.8, 'label': 'MMD²'})
    ax.set_title('IMU Signal Separability — MMD² (5-Class)')
    fig.tight_layout()
    save_figure(fig, FIG_DIR / 'separability_mmd_5class_v2')
    plt.close(fig)

    # ── 3. BD vs MMD side-by-side ──────────────────────────────
    print("\n--- BD vs MMD Comparison ---")
    # Compute BD matrix for comparison
    from analysis.layer2_signals.l2_class_separability import compute_separability_matrix
    bd_5 = compute_separability_matrix(df, feature_cols, 'action', ACTION_ORDER)

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, SINGLE_COL + 0.2))

    sns.heatmap(bd_5, annot=True, fmt='.3f', cmap='YlGnBu', mask=mask,
                linewidths=0.5, ax=axes[0], cbar_kws={'shrink': 0.7})
    axes[0].set_title('Bhattacharyya Distance')

    sns.heatmap(mmd_5, annot=True, fmt='.3f', cmap='YlOrRd', mask=mask,
                linewidths=0.5, ax=axes[1], cbar_kws={'shrink': 0.7})
    axes[1].set_title('MMD² (Nonparametric)')

    fig.suptitle('Separability: Parametric vs Nonparametric', fontsize=11, fontweight='bold')
    fig.tight_layout()
    save_figure(fig, FIG_DIR / 'separability_dual_comparison_v2')
    plt.close(fig)

    # ── 4. Taxonomy comparison (MMD) ───────────────────────────
    print("\n--- Taxonomy MMD Comparison ---")
    df_4 = df.copy()
    df_4['action_4'] = df_4['action'].map(map_to_4class)
    mmd_4, _ = compute_separability_matrix_mmd(
        df_4, feature_cols, 'action_4', ACTION_ORDER_4CLASS,
        groups=groups_col, n_permutations=200, seed=42,
    )

    df_3 = df.copy()
    df_3['action_3'] = df_3['action'].map(map_to_3class)
    mmd_3, _ = compute_separability_matrix_mmd(
        df_3, feature_cols, 'action_3', ACTION_ORDER_3CLASS,
        groups=groups_col, n_permutations=200, seed=42,
    )

    upper_5 = mmd_5.values[np.triu_indices(5, k=1)]
    upper_4 = mmd_4.values[np.triu_indices(4, k=1)]
    upper_3 = mmd_3.values[np.triu_indices(3, k=1)]

    print(f"  Mean MMD² — 5-class: {upper_5.mean():.4f}, "
          f"4-class: {upper_4.mean():.4f}, 3-class: {upper_3.mean():.4f}")
    print(f"  Min MMD²  — 5-class: {upper_5.min():.4f}, "
          f"4-class: {upper_4.min():.4f}, 3-class: {upper_3.min():.4f}")

    # Figure 4: Taxonomy comparison
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.5))
    taxonomies = ['5-class', '4-class', '3-class']
    means = [upper_5.mean(), upper_4.mean(), upper_3.mean()]
    mins = [upper_5.min(), upper_4.min(), upper_3.min()]
    x = np.arange(3)
    width = 0.35
    styled_bar(ax, x - width/2, means, width=width, label='Mean MMD²', color='#4477AA')
    styled_bar(ax, x + width/2, mins, width=width, label='Min MMD² (hardest pair)', color='#CC6677')
    ax.set_xticks(x)
    ax.set_xticklabels(taxonomies)
    ax.set_ylabel('MMD²')
    ax.set_title('IMU Separability by Taxonomy (MMD)')
    ax.legend(frameon=True, fancybox=False)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / 'separability_taxonomy_v2')
    plt.close(fig)

    # ── 5. Emit JSON summary ─────────────────────────────────
    out_dir = SCRIPT_DIR / 'outputs'
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        'class_labels_5': list(ACTION_ORDER),
        'class_labels_4': list(ACTION_ORDER_4CLASS),
        'class_labels_3': list(ACTION_ORDER_3CLASS),
        'mmd_matrix_5class': mmd_5.values.tolist(),
        'pvalue_matrix_5class': pval_5.values.tolist(),
        'bhattacharyya_matrix_5class': bd_5.values.tolist(),
        'normality_results': {
            cls: {
                'skewness_pvalue': float(res['skewness_pvalue']),
                'kurtosis_pvalue': float(res['kurtosis_pvalue']),
                'is_normal': bool(res['is_normal']),
            }
            for cls, res in normality_results.items()
        },
        'taxonomy_mmd_summary': {
            '5-class': {
                'mean_mmd': float(upper_5.mean()),
                'min_mmd': float(upper_5.min()),
                'max_mmd': float(upper_5.max()),
            },
            '4-class': {
                'mean_mmd': float(upper_4.mean()),
                'min_mmd': float(upper_4.min()),
                'max_mmd': float(upper_4.max()),
            },
            '3-class': {
                'mean_mmd': float(upper_3.mean()),
                'min_mmd': float(upper_3.min()),
                'max_mmd': float(upper_3.max()),
            },
        },
    }

    json_path = out_dir / 'separability_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ JSON summary saved to {json_path}")

    print(f"\n✓ Figures saved to {FIG_DIR}/")


if __name__ == '__main__':
    main()
