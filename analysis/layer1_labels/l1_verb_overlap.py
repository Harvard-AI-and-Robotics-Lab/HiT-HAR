#!/usr/bin/env python3
"""L1: Verb-Level Class Overlap Detection.

For each class pair (focus on OT↔TO), identifies:
  1. Shared verbs (appear in both classes) — these are taxonomy-ambiguous
  2. Per-verb class distribution and entropy
  3. Exclusive verbs (reliable class indicators)
  4. Overlap rate: what % of samples use shared verbs?

Outputs:
  - figures/verb_overlap_ot_to.pdf — stacked bar of top shared verbs
  - figures/verb_overlap_all_pairs.pdf — overlap rate for all 10 class pairs
  - outputs/verb_overlap.json — per-verb ambiguity scores
  - stdout: summary tables

Usage:
    python analysis/layer1_labels/l1_verb_overlap.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from analysis.shared.data_loader import load_gold, ACTION_ORDER
from analysis.shared.plot_style import apply_style, save_figure, DOUBLE_COL

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # [FIX-1]
FIG_DIR = SCRIPT_DIR / 'figures'
OUT_DIR = SCRIPT_DIR / 'outputs'

NON_VERB_TOKENS = {
    'man', 'woman', 'person', 'child', 'lady', 'guy', 'boy', 'girl',
    'someone', 'he', 'she', 'they', 'it', 'the', 'a', 'an', 'is', 'are',
}


def load_cleaned_gold():
    """[FIX-1] Load gold with same cleaning as training pipeline.

    Applies: status=='Gold', remove #O narrations, remove multi-label
    narrations, remove invalid actions. Matches clean_labels.py logic.
    """
    gold = load_gold()
    gold = gold[gold['status'] == 'Gold'].copy()

    # Remove #O (other person's actions) — same as clean_labels.py:138
    mask_O = gold['narration_text'].str.contains(r'#O\b', na=False)
    gold = gold[~mask_O].copy()

    # Remove multi-label narrations — same as clean_labels.py:195
    nunique = gold.groupby('narr_norm')['action'].nunique()
    multi_narrs = set(nunique[nunique > 1].index)
    if multi_narrs:
        gold = gold[~gold['narr_norm'].isin(multi_narrs)].copy()

    # Remove invalid actions
    gold = gold[~gold['action'].isin(['Error', 'Invalid', 'Unknown', ''])].copy()
    return gold


def compute_pair_verb_overlap(gold, class_a, class_b):
    """Compute verb overlap stats between two classes."""
    pair = gold[gold['action'].isin([class_a, class_b])].copy()
    pair = pair[~pair['verb'].isin(NON_VERB_TOKENS)]
    pair = pair.dropna(subset=['verb'])
    pair = pair[pair['verb'].str.len() > 1]

    verb_stats = []
    for verb, grp in pair.groupby('verb'):
        counts = grp['action'].value_counts()
        n_a = int(counts.get(class_a, 0))
        n_b = int(counts.get(class_b, 0))
        total = n_a + n_b
        probs = np.array([n_a, n_b]) / total
        h = float(scipy_entropy(probs, base=2))  # max=1.0 for binary

        verb_stats.append({
            'verb': verb,
            'n_total': total,
            'n_a': n_a, 'n_b': n_b,
            'pct_a': n_a / total, 'pct_b': n_b / total,
            'entropy': h,
            'is_shared': n_a > 0 and n_b > 0,
            'purity': max(n_a, n_b) / total,
        })

    df = pd.DataFrame(verb_stats).sort_values('n_total', ascending=False)
    shared = df[df['is_shared']]
    shared_samples = shared['n_total'].sum()
    total_samples = df['n_total'].sum()

    return df, {
        'class_a': class_a, 'class_b': class_b,
        'total_verbs': len(df),
        'shared_verbs': len(shared),
        'shared_sample_count': int(shared_samples),
        'total_sample_count': int(total_samples),
        'overlap_rate': float(shared_samples / total_samples) if total_samples > 0 else 0,
        # [FIX-2] Correct pandas boolean mask → .sum()
        'exclusive_a': int(((df['n_a'] > 0) & (df['n_b'] == 0)).sum()),
        'exclusive_b': int(((df['n_b'] > 0) & (df['n_a'] == 0)).sum()),
        'mean_shared_entropy': float(shared['entropy'].mean()) if len(shared) > 0 else 0,
    }


def main():
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gold = load_cleaned_gold()  # [FIX-1] Cleaned gold, not raw
    print(f"Loaded {len(gold)} cleaned gold samples, {gold['verb'].nunique()} unique verbs")

    # ── Focus pair: OT ↔ TO ──
    class_a, class_b = 'Object Transfer', 'Task Operation'
    verb_df, pair_summary = compute_pair_verb_overlap(gold, class_a, class_b)
    shared = verb_df[verb_df['is_shared']].copy()

    print(f"\n{'='*60}")
    print(f"OT ↔ TO Verb Overlap")
    print(f"{'='*60}")
    print(f"Total verbs: {pair_summary['total_verbs']}")
    print(f"Shared verbs: {pair_summary['shared_verbs']} "
          f"({pair_summary['shared_verbs']/pair_summary['total_verbs']*100:.1f}%)")
    print(f"Samples on shared verbs: {pair_summary['shared_sample_count']}/"
          f"{pair_summary['total_sample_count']} "
          f"({pair_summary['overlap_rate']*100:.1f}%)")
    print(f"Mean entropy of shared verbs: {pair_summary['mean_shared_entropy']:.3f}")

    print(f"\nTop 15 shared verbs (by frequency):")
    for _, row in shared.head(15).iterrows():
        print(f"  '{row['verb']}': n={row['n_total']}, "
              f"OT={row['pct_a']:.0%} TO={row['pct_b']:.0%}, "
              f"H={row['entropy']:.3f}")

    # Exclusive verbs (clean indicators)
    excl_a = verb_df[(verb_df['n_a'] > 0) & (verb_df['n_b'] == 0)].head(10)
    excl_b = verb_df[(verb_df['n_b'] > 0) & (verb_df['n_a'] == 0)].head(10)
    print(f"\nTop OT-exclusive: {', '.join(excl_a['verb'].tolist())}")
    print(f"Top TO-exclusive: {', '.join(excl_b['verb'].tolist())}")

    # ── All 10 class pairs: overlap rate ──
    all_pairs = []
    for i, ca in enumerate(ACTION_ORDER):
        for j, cb in enumerate(ACTION_ORDER):
            if j <= i:
                continue
            _, ps = compute_pair_verb_overlap(gold, ca, cb)
            all_pairs.append(ps)
            print(f"\n{ca} ↔ {cb}: overlap={ps['overlap_rate']*100:.1f}%, "
                  f"shared={ps['shared_verbs']}/{ps['total_verbs']}")

    # ── Figure 1: OT↔TO shared verbs stacked bar ──
    fig, ax = plt.subplots(figsize=(6.5, 4))
    top = shared.head(15).sort_values('n_total', ascending=True)
    y = range(len(top))
    ax.barh(y, top['n_a'].values, color='#4477AA', label='Object Transfer')
    ax.barh(y, top['n_b'].values, left=top['n_a'].values,
            color='#CC6677', label='Task Operation')
    ax.set_yticks(y)
    ax.set_yticklabels(top['verb'].values, fontsize=7)
    ax.set_xlabel('Narration Count', fontsize=8)
    ax.set_title('Top Shared Verbs: OT vs TO Distribution', fontsize=9, fontweight='bold')
    ax.legend(fontsize=7)
    for i, (_, row) in enumerate(top.iterrows()):
        ax.text(row['n_total'] + 3, i, f'H={row["entropy"]:.2f}',
                va='center', fontsize=5.5, color='#666666')
    fig.tight_layout()
    save_figure(fig, FIG_DIR / 'verb_overlap_ot_to')
    plt.close(fig)

    # ── Figure 2: All-pairs overlap rate bar ──
    fig, ax = plt.subplots(figsize=(7, 3.5))
    pairs_sorted = sorted(all_pairs, key=lambda x: x['overlap_rate'], reverse=True)
    pair_labels = [f"{p['class_a'][:6]}↔{p['class_b'][:6]}" for p in pairs_sorted]
    rates = [p['overlap_rate'] * 100 for p in pairs_sorted]
    colors = ['#CC6677' if r > 50 else '#DDCC77' if r > 30 else '#4477AA' for r in rates]

    ax.bar(range(len(rates)), rates, color=colors, edgecolor='#333333', linewidth=0.5)
    ax.set_xticks(range(len(rates)))
    ax.set_xticklabels(pair_labels, fontsize=6, rotation=45, ha='right')
    ax.set_ylabel('Verb Overlap Rate (%)', fontsize=8)
    ax.set_title('Narration Verb Overlap by Class Pair', fontsize=9, fontweight='bold')
    fig.tight_layout()
    save_figure(fig, FIG_DIR / 'verb_overlap_all_pairs')
    plt.close(fig)

    # ── JSON output ──
    output = {
        'ot_to_detail': pair_summary,
        'ot_to_top_shared': shared.head(30)[
            ['verb', 'n_total', 'n_a', 'n_b', 'pct_a', 'pct_b', 'entropy', 'purity']
        ].to_dict('records'),
        'ot_to_exclusive_a': excl_a['verb'].tolist(),
        'ot_to_exclusive_b': excl_b['verb'].tolist(),
        'all_pairs': all_pairs,
    }
    with open(OUT_DIR / 'verb_overlap.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n✓ JSON → {OUT_DIR / 'verb_overlap.json'}")
    print(f"✓ Figures → {FIG_DIR}/verb_overlap_*.pdf")


if __name__ == '__main__':
    main()
