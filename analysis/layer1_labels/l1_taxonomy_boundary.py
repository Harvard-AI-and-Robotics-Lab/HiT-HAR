#!/usr/bin/env python3
"""L1-4: Taxonomy Boundary & Verb Ambiguity Analysis.

Research question: How well does the 5-class taxonomy separate at the
label/verb level? What fraction of training data lies in the ambiguous zone?
How does 4-class and 3-class compare?

Outputs:
  - figures/verb_class_overlap.pdf            — Heatmap: top ambiguous verbs x classes
  - figures/ambiguity_rate_by_taxonomy.pdf     — Bar chart: 5 vs 4 vs 3 class ambiguity
  - figures/taxonomy_comparison_sankey.pdf     — 5->4->3 class mapping visualization
  - figures/scenario_action_mi.pdf            — Scenario x Action mutual information
  - stdout: ambiguity statistics

Usage:
    python analysis/layer1_labels/l1_taxonomy_boundary.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import entropy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from analysis.shared.plot_style import (
    apply_style, save_figure, SINGLE_COL, DOUBLE_COL, COLORS_5CLASS,
)
from analysis.shared.data_loader import (
    load_gold, ACTION_ORDER, ACTION_ORDER_4CLASS, ACTION_ORDER_3CLASS,
    SCENARIO_ORDER, AMBIGUOUS_VERBS, map_to_4class, map_to_3class,
    extract_verb,
)

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR / 'figures'


def compute_verb_class_matrix(df):
    """Build verb x action class count matrix."""
    return pd.crosstab(df['verb'], df['action'])


def compute_ambiguity_rate(df, min_classes=2):
    """Fraction of samples whose verb appears in >= min_classes classes."""
    verb_nclasses = df.groupby('verb')['action'].nunique()
    ambiguous_verbs = set(verb_nclasses[verb_nclasses >= min_classes].index)
    return df['verb'].isin(ambiguous_verbs).mean()


def mutual_information(df, col_x, col_y):
    """Compute mutual information I(X;Y) in bits."""
    ct = pd.crosstab(df[col_x], df[col_y])
    p_xy = ct.values / ct.values.sum()
    p_x = p_xy.sum(axis=1, keepdims=True)
    p_y = p_xy.sum(axis=0, keepdims=True)

    # Avoid log(0)
    nonzero = p_xy > 0
    mi = np.sum(p_xy[nonzero] * np.log2(p_xy[nonzero] / (p_x * p_y)[nonzero]))
    h_y = entropy(p_y.flatten(), base=2)
    return mi, h_y


def main():
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("L1-4: Taxonomy Boundary & Verb Ambiguity Analysis")
    print("=" * 60)

    gold = load_gold()
    # gold already has 'verb' column from load_gold()

    # -- 1. Verb x Class overlap ---------------------------------------------
    print("\n--- Verb-Class Overlap (5-class) ---")
    verb_matrix = compute_verb_class_matrix(gold)
    verb_nclasses = (verb_matrix > 0).sum(axis=1)

    print(f"Total unique verbs: {len(verb_matrix)}")
    for n in range(1, 6):
        count = (verb_nclasses == n).sum()
        print(f"  Verbs in {n} class(es): {count}")

    ambig_rate_5 = compute_ambiguity_rate(gold, min_classes=2)
    print(f"\nAmbiguity rate (verb in 2+ classes): {ambig_rate_5:.3f} "
          f"({ambig_rate_5*100:.1f}% of samples)")

    # Top ambiguous verbs
    top_ambig = verb_nclasses[verb_nclasses >= 3].index
    top_verb_matrix = verb_matrix.loc[top_ambig].reindex(columns=ACTION_ORDER)
    top_verb_matrix = top_verb_matrix.loc[top_verb_matrix.sum(axis=1).nlargest(20).index]

    # Figure 1: Verb-class heatmap
    fig, ax = plt.subplots(figsize=(SINGLE_COL + 1.0, 4.0))
    sns.heatmap(top_verb_matrix, annot=True, fmt='d', cmap='YlOrRd',
                linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_xlabel('Action Class')
    ax.set_ylabel('Verb')
    ax.set_title(f'Top Ambiguous Verbs (>=3 classes, n={len(top_ambig)})')
    fig.tight_layout()
    save_figure(fig, FIG_DIR / 'verb_class_overlap')
    plt.close(fig)

    # -- 2. Taxonomy comparison: ambiguity rate ------------------------------
    print("\n--- Taxonomy Comparison: Ambiguity Rates ---")

    gold_4 = gold.copy()
    gold_4['action'] = gold_4['action'].map(map_to_4class)
    ambig_rate_4 = compute_ambiguity_rate(gold_4, min_classes=2)

    gold_3 = gold.copy()
    gold_3['action'] = gold_3['action'].map(map_to_3class)
    ambig_rate_3 = compute_ambiguity_rate(gold_3, min_classes=2)

    print(f"  5-class: {ambig_rate_5:.3f} ({ambig_rate_5*100:.1f}%)")
    print(f"  4-class: {ambig_rate_4:.3f} ({ambig_rate_4*100:.1f}%)")
    print(f"  3-class: {ambig_rate_3:.3f} ({ambig_rate_3*100:.1f}%)")

    # Figure 2: Ambiguity rate comparison
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.0))
    taxonomies = ['5-class\n(Original)', '4-class\n(Merge OT+TO)', '3-class\n(+Merge Stat+Search)']
    rates = [ambig_rate_5, ambig_rate_4, ambig_rate_3]
    colors = ['#EE6677', '#CCBB44', '#228833']
    bars = ax.bar(taxonomies, rates, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_ylabel('Ambiguity Rate')
    ax.set_title('Verb Ambiguity Rate by Taxonomy Granularity')
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{rate:.1%}', ha='center', va='bottom', fontsize=8)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / 'ambiguity_rate_by_taxonomy')
    plt.close(fig)

    # -- 3. Scenario x Action mutual information -----------------------------
    print("\n--- Scenario x Action Mutual Information ---")
    mi_5, h_5 = mutual_information(gold, 'scenario', 'action')
    mi_4, h_4 = mutual_information(gold_4, 'scenario', 'action')
    mi_3, h_3 = mutual_information(gold_3, 'scenario', 'action')

    print(f"  5-class: MI={mi_5:.4f} bits, H(Action)={h_5:.4f} bits, "
          f"MI/H={mi_5/h_5*100:.1f}%")
    print(f"  4-class: MI={mi_4:.4f} bits, H(Action)={h_4:.4f} bits, "
          f"MI/H={mi_4/h_4*100:.1f}%")
    print(f"  3-class: MI={mi_3:.4f} bits, H(Action)={h_3:.4f} bits, "
          f"MI/H={mi_3/h_3*100:.1f}%")

    # Figure 3: MI comparison
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.0))
    mi_pcts = [mi_5/h_5*100, mi_4/h_4*100, mi_3/h_3*100]
    ax.bar(taxonomies, mi_pcts, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_ylabel('MI / H(Action) (%)')
    ax.set_title('Scenario->Action Information Gain')
    for i, pct in enumerate(mi_pcts):
        ax.text(i, pct + 0.3, f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / 'scenario_action_mi')
    plt.close(fig)

    print(f"\nFigures saved to {FIG_DIR}/")


if __name__ == '__main__':
    main()
