#!/usr/bin/env python3
"""L1-4v2: Taxonomy Boundary — Entropy-Weighted Verb Ambiguity.

Updates:
  - Entropy-weighted ambiguity instead of binary "verb in 2+ classes"
  - Filters non-verb tokens (person/man/woman/child/lady + stopwords)
  - Reports both raw rate and entropy-weighted rate
  - Dual comparison: 5 vs 4 vs 3 class

Outputs:
  - figures/verb_entropy_distribution_v2.pdf     — Distribution of per-verb entropy
  - figures/ambiguity_entropy_by_taxonomy_v2.pdf  — Entropy-weighted rate: 5/4/3 class
  - figures/top_ambiguous_verbs_v2.pdf           — Top verbs ranked by entropy
  - stdout: ambiguity statistics

Usage:
    python analysis/layer1_labels/l1_taxonomy_boundary_v2.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from analysis.shared.plot_style import (
    apply_style, save_figure, SINGLE_COL, DOUBLE_COL,
    styled_bar, styled_barh,
)
from analysis.shared.data_loader import (
    load_gold, ACTION_ORDER, ACTION_ORDER_4CLASS, ACTION_ORDER_3CLASS,
    map_to_4class, map_to_3class, extract_verb,
)

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR / 'figures'

# Non-verb tokens to filter (subjects, not actions)
NON_VERB_TOKENS = {
    'man', 'woman', 'person', 'child', 'lady', 'guy', 'boy', 'girl',
    'someone', 'he', 'she', 'they', 'it', 'the', 'a', 'an', 'is', 'are',
}


def compute_entropy_ambiguity(df, max_entropy=None):
    """Compute entropy-weighted verb ambiguity.

    For each verb v: H(v) = -Σ p(c|v) log₂ p(c|v)
    Sample-level ambiguity = mean H(verb_i) / H_max

    Args:
        df: DataFrame with 'verb' and 'action' columns
        max_entropy: maximum possible entropy (log₂(n_classes)). Auto-computed if None.
    Returns:
        (weighted_rate, per_verb_entropy_dict)
    """
    n_classes = df['action'].nunique()
    if max_entropy is None:
        max_entropy = np.log2(n_classes) if n_classes > 1 else 1.0

    # Per-verb class distribution entropy
    verb_entropy = {}
    for verb, grp in df.groupby('verb'):
        class_counts = grp['action'].value_counts().values
        probs = class_counts / class_counts.sum()
        h = scipy_entropy(probs, base=2)
        verb_entropy[verb] = float(h)

    # Map each sample's verb to its entropy
    sample_entropies = df['verb'].map(verb_entropy).fillna(0)
    weighted_rate = float((sample_entropies / max_entropy).mean())

    return weighted_rate, verb_entropy


def main():
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("L1-4v2: Taxonomy Boundary — Entropy-Weighted Ambiguity")
    print("=" * 60)

    gold = load_gold()

    # Filter non-verb tokens
    gold_filtered = gold[~gold['verb'].isin(NON_VERB_TOKENS)].copy()
    n_filtered = len(gold) - len(gold_filtered)
    print(f"Filtered {n_filtered} samples with non-verb tokens "
          f"({n_filtered/len(gold)*100:.1f}%)")
    print(f"Remaining: {len(gold_filtered)} samples, "
          f"{gold_filtered['verb'].nunique()} unique verbs")

    # ── 1. 5-class entropy ambiguity ───────────────────────────
    print("\n--- 5-Class Entropy-Weighted Ambiguity ---")
    rate_5, entropy_5 = compute_entropy_ambiguity(gold_filtered)
    raw_rate_5 = (gold_filtered.groupby('verb')['action'].nunique() >= 2).mean()
    print(f"  Raw ambiguity (verb in 2+ classes): {raw_rate_5:.3f} ({raw_rate_5*100:.1f}%)")
    print(f"  Entropy-weighted ambiguity:         {rate_5:.3f} ({rate_5*100:.1f}%)")

    # ── 2. 4-class and 3-class comparison ──────────────────────
    print("\n--- Taxonomy Comparison ---")
    gold_4 = gold_filtered.copy()
    gold_4['action'] = gold_4['action'].map(map_to_4class)
    rate_4, entropy_4 = compute_entropy_ambiguity(gold_4)
    raw_rate_4 = (gold_4.groupby('verb')['action'].nunique() >= 2).mean()

    gold_3 = gold_filtered.copy()
    gold_3['action'] = gold_3['action'].map(map_to_3class)
    rate_3, entropy_3 = compute_entropy_ambiguity(gold_3)
    raw_rate_3 = (gold_3.groupby('verb')['action'].nunique() >= 2).mean()

    print(f"  5-class: raw={raw_rate_5:.3f}, entropy={rate_5:.3f}")
    print(f"  4-class: raw={raw_rate_4:.3f}, entropy={rate_4:.3f}")
    print(f"  3-class: raw={raw_rate_3:.3f}, entropy={rate_3:.3f}")

    # Figure 1: Entropy distribution
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.2))
    entropies = list(entropy_5.values())
    ax.hist(entropies, bins=30, color='#4477AA', edgecolor='#2D2D2D',
            linewidth=0.8, alpha=0.8)
    ax.axvline(np.mean(entropies), color='#CC6677', linestyle='--',
               linewidth=1.5, label=f'Mean: {np.mean(entropies):.2f}')
    ax.set_xlabel('Per-Verb Class Entropy (bits)')
    ax.set_ylabel('# Verbs')
    ax.set_title('Verb Class Entropy Distribution (5-Class)')
    ax.legend(frameon=True, fancybox=False)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / 'verb_entropy_distribution_v2')
    plt.close(fig)

    # Figure 2: Raw vs entropy-weighted comparison
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.5))
    taxonomies = ['5-class', '4-class', '3-class']
    x = np.arange(3)
    width = 0.35
    styled_bar(ax, x - width/2, [raw_rate_5, raw_rate_4, raw_rate_3],
               width=width, label='Raw (verb in 2+ classes)', color='#CC6677')
    styled_bar(ax, x + width/2, [rate_5, rate_4, rate_3],
               width=width, label='Entropy-weighted', color='#4477AA')
    ax.set_xticks(x)
    ax.set_xticklabels(taxonomies)
    ax.set_ylabel('Ambiguity Rate')
    ax.set_title('Verb Ambiguity: Raw vs Entropy-Weighted')
    ax.legend(frameon=True, fancybox=False, fontsize=7)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / 'ambiguity_entropy_by_taxonomy_v2')
    plt.close(fig)

    # Figure 3: Top ambiguous verbs by entropy
    fig, ax = plt.subplots(figsize=(SINGLE_COL + 0.5, 3.5))
    sorted_verbs = sorted(entropy_5.items(), key=lambda x: -x[1])[:20]
    verb_names = [v for v, _ in sorted_verbs][::-1]
    verb_vals = [e for _, e in sorted_verbs][::-1]
    styled_barh(ax, verb_names, verb_vals, color='#4477AA')
    ax.set_xlabel('Class Entropy (bits)')
    ax.set_title('Top 20 Ambiguous Verbs by Entropy')
    fig.tight_layout()
    save_figure(fig, FIG_DIR / 'top_ambiguous_verbs_v2')
    plt.close(fig)

    print(f"\n✓ Figures saved to {FIG_DIR}/")


if __name__ == '__main__':
    main()
