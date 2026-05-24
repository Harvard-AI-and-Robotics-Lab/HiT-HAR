#!/usr/bin/env python3
"""L1-2: Label Conflict Source Analysis.

Research question: Where do label conflicts come from — LLM errors or
fundamental taxonomy ambiguity? Which conflicts are fixable?

Outputs:
  - figures/conflict_source_breakdown.pdf  — Pie/bar: taxonomy ambiguity vs LLM error
  - figures/conflict_pairs_ranked.pdf      — Horizontal bar: conflict pair frequency
  - figures/timestamp_conflict_heatmap.pdf — Which class pairs share timestamps
  - stdout: conflict counts and classification

Usage:
    python analysis/layer1_labels/l1_conflict_analysis.py
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
    load_gold, load_llm, load_train, ACTION_ORDER, normalize_narration,
)

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR / 'figures'

# Known physically-indistinguishable pairs (from comprehensive diagnosis)
TAXONOMY_AMBIGUITY_PAIRS = {
    frozenset({'Object Transfer', 'Task Operation'}),
    frozenset({'Search', 'Stationary'}),
    frozenset({'Object Transfer', 'Stationary'}),
    frozenset({'Stationary', 'Task Operation'}),
}


def find_timestamp_conflicts(df):
    """Find rows where same (video_uid, timestamp_sec) has different actions."""
    grouped = df.groupby(['video_uid', 'timestamp_sec'])['action'].nunique()
    conflict_keys = grouped[grouped > 1].index
    conflicts = df.set_index(['video_uid', 'timestamp_sec']).loc[conflict_keys].reset_index()
    return conflicts


def classify_conflict_source(pair):
    """Classify a conflict pair as taxonomy_ambiguity or llm_error.

    Args:
        pair: tuple of two action class names
    Returns:
        'taxonomy_ambiguity' if the pair is known to be physically indistinguishable,
        'llm_error' otherwise.
    """
    if frozenset(pair) in TAXONOMY_AMBIGUITY_PAIRS:
        return 'taxonomy_ambiguity'
    return 'llm_error'


def main():
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("L1-2: Label Conflict Source Analysis")
    print("=" * 60)

    gold = load_gold()
    train = load_train()

    # -- 1. Multi-label narration conflicts ----------------------------------
    print("\n--- Multi-Label Narration Conflicts (Gold) ---")
    narr_actions = gold.groupby('narr_norm')['action'].nunique()
    multi_label = narr_actions[narr_actions > 1]
    print(f"Unique narrations with 2+ actions: {len(multi_label)} / {len(narr_actions)} "
          f"({len(multi_label)/len(narr_actions)*100:.2f}%)")

    # Get the actual conflict pairs
    conflict_narrs = gold[gold['narr_norm'].isin(multi_label.index)]
    conflict_pairs = {}
    for narr, grp in conflict_narrs.groupby('narr_norm'):
        actions = sorted(grp['action'].unique())
        if len(actions) == 2:
            key = f"{actions[0]} <-> {actions[1]}"
            conflict_pairs[key] = conflict_pairs.get(key, 0) + 1

    print("\nConflict pair frequency:")
    for pair, count in sorted(conflict_pairs.items(), key=lambda x: -x[1])[:10]:
        # Classify source
        parts = pair.split(' <-> ')
        source = classify_conflict_source(tuple(parts))
        print(f"  {pair}: {count} narrations [{source}]")

    # -- 2. Same-timestamp conflicts in training data ------------------------
    print("\n--- Same-Timestamp Conflicts (Training Set) ---")
    ts_conflicts = find_timestamp_conflicts(train)
    n_conflict_rows = len(ts_conflicts)
    n_conflict_pairs = ts_conflicts.groupby(['video_uid', 'timestamp_sec']).ngroups
    print(f"Conflict rows: {n_conflict_rows} ({n_conflict_rows/len(train)*100:.2f}% of training)")
    print(f"Conflict (video, timestamp) pairs: {n_conflict_pairs}")

    # Build conflict pair matrix
    ts_pair_counts = {}
    for (vid, ts), grp in ts_conflicts.groupby(['video_uid', 'timestamp_sec']):
        actions = sorted(grp['action'].unique())
        if len(actions) == 2:
            key = f"{actions[0]} <-> {actions[1]}"
            ts_pair_counts[key] = ts_pair_counts.get(key, 0) + 1

    print("\nTimestamp conflict pairs:")
    for pair, count in sorted(ts_pair_counts.items(), key=lambda x: -x[1])[:10]:
        parts = pair.split(' <-> ')
        source = classify_conflict_source(tuple(parts))
        print(f"  {pair}: {count} [{source}]")

    # -- 3. Source attribution summary ---------------------------------------
    print("\n--- Conflict Source Attribution ---")
    total_ambiguity = 0
    total_llm_error = 0
    for pair, count in conflict_pairs.items():
        parts = pair.split(' <-> ')
        if classify_conflict_source(tuple(parts)) == 'taxonomy_ambiguity':
            total_ambiguity += count
        else:
            total_llm_error += count

    total = total_ambiguity + total_llm_error
    print(f"Taxonomy ambiguity: {total_ambiguity} ({total_ambiguity/total*100:.1f}%)")
    print(f"LLM error:          {total_llm_error} ({total_llm_error/total*100:.1f}%)")

    # Figure 1: Conflict source breakdown
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.5))

    # Left: pie chart
    ax = axes[0]
    ax.pie([total_ambiguity, total_llm_error],
           labels=['Taxonomy\nAmbiguity', 'LLM Error'],
           colors=['#EE6677', '#4477AA'],
           autopct='%1.1f%%', startangle=90, textprops={'fontsize': 8})
    ax.set_title('Conflict Source Attribution')

    # Right: conflict pairs ranked
    ax = axes[1]
    sorted_pairs = sorted(conflict_pairs.items(), key=lambda x: -x[1])[:8]
    pair_names = [p for p, _ in sorted_pairs]
    pair_counts = [c for _, c in sorted_pairs]
    pair_colors = ['#EE6677' if classify_conflict_source(tuple(p.split(' <-> '))) == 'taxonomy_ambiguity'
                   else '#4477AA' for p in pair_names]
    ax.barh(pair_names, pair_counts, color=pair_colors, edgecolor='white', linewidth=0.5)
    ax.set_xlabel('# Conflicting Narrations')
    ax.set_title('Top Label Conflict Pairs')
    ax.invert_yaxis()
    fig.tight_layout()
    save_figure(fig, FIG_DIR / 'conflict_source_breakdown')
    plt.close(fig)

    print(f"\nFigures saved to {FIG_DIR}/")


if __name__ == '__main__':
    main()
