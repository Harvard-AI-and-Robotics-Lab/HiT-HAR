#!/usr/bin/env python3
"""L1-3: Human Annotation Quality Audit.

Research question: What do annotator behavior signals reveal about
systematic label uncertainty? How reliable is each quality tier?

Outputs:
  - figures/secondary_action_pairs.pdf     — Top primary+secondary action pairings
  - figures/bad_correction_flow.pdf        — Sankey/alluvial: LLM->Human correction flows
  - figures/annotator_comparison.pdf       — Per-annotator Gold%, Bad%, secondary rate
  - figures/tier_distribution.pdf          — Tier 1-4 distribution with confidence
  - figures/lead_time_by_verdict.pdf       — Annotation lead time: Gold vs Bad vs secondary
  - stdout: quality signal summary

Usage:
    python analysis/layer1_labels/l1_annotation_quality.py
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
    apply_style, save_figure, SINGLE_COL, DOUBLE_COL,
    COLORS_5CLASS, COLORS_TIER, get_5class_colors,
)
from analysis.shared.data_loader import (
    load_r1_raw, load_r2_raw, load_tier_assignments, ACTION_ORDER,
)

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR / 'figures'


def compute_secondary_stats(df):
    """Compute statistics about secondary action annotations.

    Args:
        df: DataFrame with 'action', 'secondary_action', 'verdict' columns.
    Returns:
        dict with total_with_secondary, secondary_rate, top_pairs.
    """
    has_sec = df['secondary_action'].fillna('').str.strip().ne('')
    total = has_sec.sum()
    rate = total / len(df) if len(df) > 0 else 0

    # Top primary+secondary pairs
    sec_df = df[has_sec].copy()
    if len(sec_df) > 0:
        pairs = sec_df.groupby(['action', 'secondary_action']).size().reset_index(name='count')
        pairs = pairs.sort_values('count', ascending=False)
        top_pairs = {(row['action'], row['secondary_action']): row['count']
                     for _, row in pairs.head(10).iterrows()}
    else:
        top_pairs = {}

    return {
        'total_with_secondary': int(total),
        'secondary_rate': rate,
        'top_pairs': top_pairs,
    }


def main():
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("L1-3: Human Annotation Quality Audit")
    print("=" * 60)

    r1 = load_r1_raw()
    r2 = load_r2_raw()
    tiers = load_tier_assignments()

    # -- 1. Secondary action analysis (R2 only) ------------------------------
    print("\n--- Secondary Action Analysis (R2) ---")
    r2_gold = r2[r2['verdict'] == 'Gold'].copy()
    sec_stats = compute_secondary_stats(r2_gold)
    print(f"Gold samples with secondary action: {sec_stats['total_with_secondary']}"
          f" / {len(r2_gold)} ({sec_stats['secondary_rate']*100:.1f}%)")
    print("\nTop primary -> secondary pairs:")
    for (prim, sec), count in list(sec_stats['top_pairs'].items())[:7]:
        print(f"  {prim:20s} -> {sec:20s}: {count}")

    # Figure 1: Secondary action heatmap
    has_sec = r2_gold['secondary_action'].fillna('').str.strip().ne('')
    sec_df = r2_gold[has_sec].copy()
    sec_df['secondary_action'] = sec_df['secondary_action'].replace(
        {'Essential Operation': 'Task Operation'})
    sec_matrix = pd.crosstab(sec_df['action'], sec_df['secondary_action'])
    sec_matrix = sec_matrix.reindex(index=ACTION_ORDER, columns=ACTION_ORDER, fill_value=0)

    fig, ax = plt.subplots(figsize=(SINGLE_COL + 0.5, SINGLE_COL))
    sns.heatmap(sec_matrix, annot=True, fmt='d', cmap='Blues',
                linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_xlabel('Secondary Choice')
    ax.set_ylabel('Primary Label')
    ax.set_title('Annotator Secondary Action Choices (R2)')
    fig.tight_layout()
    save_figure(fig, FIG_DIR / 'secondary_action_pairs')
    plt.close(fig)

    # -- 2. Per-annotator comparison (R2) ------------------------------------
    print("\n--- Per-Annotator Comparison (R2) ---")
    annotator_stats = []
    for ann_id, grp in r2.groupby('annotator'):
        n = len(grp)
        gold_pct = (grp['verdict'] == 'Gold').mean()
        bad_pct = (grp['verdict'] == 'Bad').mean()
        has_sec = grp['secondary_action'].fillna('').str.strip().ne('').sum()
        sec_pct = has_sec / n
        median_lt = grp['lead_time'].median() if 'lead_time' in grp.columns else 0
        annotator_stats.append({
            'annotator': int(ann_id), 'n': n, 'gold_pct': gold_pct,
            'bad_pct': bad_pct, 'secondary_pct': sec_pct, 'median_lead_time': median_lt,
        })
        print(f"  Annotator {int(ann_id)}: n={n}, Gold={gold_pct:.1%}, "
              f"Bad={bad_pct:.1%}, Secondary={sec_pct:.1%}, "
              f"Median lead time={median_lt:.1f}s")

    ann_df = pd.DataFrame(annotator_stats)

    # Figure 2: Annotator comparison grouped bar
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.0))
    x = np.arange(len(ann_df))
    width = 0.25
    ax.bar(x - width, ann_df['gold_pct'], width, label='Gold %', color='#228833')
    ax.bar(x, ann_df['bad_pct'], width, label='Bad %', color='#EE6677')
    ax.bar(x + width, ann_df['secondary_pct'], width, label='Secondary %', color='#4477AA')
    ax.set_xticks(x)
    ax.set_xticklabels([f"Ann. {a}" for a in ann_df['annotator']])
    ax.set_ylabel('Rate')
    ax.set_title('Annotator Behavior Comparison (R2)')
    ax.legend(frameon=True, fancybox=False, fontsize=7)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / 'annotator_comparison')
    plt.close(fig)

    # -- 3. Tier distribution ------------------------------------------------
    print("\n--- Tier Distribution ---")
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.5))

    # Left: tier counts
    ax = axes[0]
    tier_counts = tiers['tier'].value_counts().sort_index()
    colors = [COLORS_TIER[t] for t in tier_counts.index]
    bars = ax.bar(tier_counts.index, tier_counts.values, color=colors,
                  edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Quality Tier')
    ax.set_ylabel('# Samples')
    ax.set_title('Annotation Quality Tier Distribution')
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(['T1\nHigh', 'T2\nModerate', 'T3\nCorrected', 'T4\nExclude'])
    for bar, count in zip(bars, tier_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f'{count:,}', ha='center', va='bottom', fontsize=8)
    for t in tier_counts.index:
        n = tier_counts[t]
        pct = n / len(tiers) * 100
        conf = tiers[tiers['tier'] == t]['confidence'].mean()
        print(f"  Tier {t}: {n:>6} ({pct:.1f}%) avg_confidence={conf:.2f}")

    # Right: tier by action class
    ax = axes[1]
    tier_by_class = pd.crosstab(tiers['action'], tiers['tier'], normalize='index')
    tier_by_class = tier_by_class.reindex(ACTION_ORDER)
    tier_by_class.plot(kind='barh', stacked=True, ax=ax,
                       color=[COLORS_TIER[c] for c in tier_by_class.columns],
                       edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Proportion')
    ax.set_title('Quality Tier by Action Class')
    ax.legend(title='Tier', labels=['T1', 'T2', 'T3', 'T4'],
              frameon=True, fancybox=False, fontsize=7)
    ax.invert_yaxis()
    fig.tight_layout()
    save_figure(fig, FIG_DIR / 'tier_distribution')
    plt.close(fig)

    # -- 4. Lead time analysis -----------------------------------------------
    print("\n--- Lead Time by Verdict (R2) ---")
    if 'lead_time' in r2.columns:
        fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.2))

        categories = []
        lead_times = []
        for verdict in ['Gold', 'Bad']:
            subset = r2[r2['verdict'] == verdict]
            categories.append(verdict)
            lead_times.append(subset['lead_time'].dropna().values)

        has_sec_mask = r2['secondary_action'].fillna('').str.strip().ne('')
        sec_subset = r2[has_sec_mask & (r2['verdict'] == 'Gold')]
        categories.append('Gold +\nSecondary')
        lead_times.append(sec_subset['lead_time'].dropna().values)

        no_sec_subset = r2[~has_sec_mask & (r2['verdict'] == 'Gold')]
        categories.append('Gold\nonly')
        lead_times.append(no_sec_subset['lead_time'].dropna().values)

        bp = ax.boxplot(lead_times, tick_labels=categories, vert=True, widths=0.6,
                        patch_artist=True, showfliers=False,
                        medianprops={'color': 'black', 'linewidth': 1.5})
        box_colors = ['#228833', '#EE6677', '#4477AA', '#CCBB44']
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        for i, lt in enumerate(lead_times):
            median = np.median(lt) if len(lt) > 0 else 0
            print(f"  {categories[i].replace(chr(10), ' '):15s}: "
                  f"median={median:.1f}s, n={len(lt)}")

        ax.set_ylabel('Lead Time (seconds)')
        ax.set_title('Annotation Lead Time by Verdict')
        ax.set_ylim(0, min(50, ax.get_ylim()[1]))
        fig.tight_layout()
        save_figure(fig, FIG_DIR / 'lead_time_by_verdict')
        plt.close(fig)

    print(f"\nFigures saved to {FIG_DIR}/")


if __name__ == '__main__':
    main()
