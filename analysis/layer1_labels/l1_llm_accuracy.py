#!/usr/bin/env python3
"""L1-1: LLM Labeling Pipeline Accuracy Analysis.

Research question: How accurate is the LLM labeling pipeline compared to
human gold annotations? Which classes and scenarios does it struggle with?

Outputs:
  - figures/llm_per_class_agreement.pdf     — Per-class LLM accuracy bar chart
  - figures/llm_correction_matrix.pdf       — Heatmap: LLM label → human correction
  - figures/llm_per_scenario_accuracy.pdf   — Per-scenario LLM accuracy
  - stdout: summary statistics

Usage:
    python analysis/layer1_labels/l1_llm_accuracy.py
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
    COLORS_5CLASS, get_5class_colors,
)
from analysis.shared.data_loader import (
    load_gold, load_llm, load_r1_raw, load_r2_raw,
    ACTION_ORDER, SCENARIO_ORDER, normalize_narration,
)

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR / 'figures'


def compute_agreement(gold_df, llm_df, on='narr_norm'):
    """Compute per-class agreement rate between gold and LLM labels.

    Joins on `on` column, compares `action` columns.
    Returns dict with overall_agreement and per_class dict.
    """
    # Deduplicate: for each unique narr_norm, take majority action
    gold_actions = gold_df.groupby(on)['action'].agg(
        lambda x: x.value_counts().index[0]
    ).reset_index().rename(columns={'action': 'gold_action'})

    llm_actions = llm_df.groupby(on)['action'].agg(
        lambda x: x.value_counts().index[0]
    ).reset_index().rename(columns={'action': 'llm_action'})

    merged = gold_actions.merge(llm_actions, on=on, how='inner')
    merged['agree'] = merged['gold_action'] == merged['llm_action']

    overall = merged['agree'].mean()
    per_class = {}
    for cls in ACTION_ORDER:
        subset = merged[merged['gold_action'] == cls]
        if len(subset) > 0:
            per_class[cls] = subset['agree'].mean()

    return {
        'overall_agreement': overall,
        'per_class': per_class,
        'n_matched': len(merged),
        'merged': merged,
    }


def build_correction_matrix(disagreements):
    """Build confusion matrix: rows=LLM label, cols=gold (human) label.

    Args:
        disagreements: DataFrame with 'gold_action' and 'llm_action' columns.
    Returns:
        DataFrame confusion matrix (LLM x Gold).
    """
    ct = pd.crosstab(
        disagreements['llm_action'], disagreements['gold_action'],
        rownames=['LLM Label'], colnames=['Human Gold Label'],
    )
    # Reindex to full action order
    ct = ct.reindex(index=ACTION_ORDER, columns=ACTION_ORDER, fill_value=0)
    return ct


def main():
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("L1-1: LLM Labeling Pipeline Accuracy Analysis")
    print("=" * 60)

    # -- Load data -----------------------------------------------------------
    gold = load_gold()
    llm = load_llm()
    r1 = load_r1_raw()
    r2 = load_r2_raw()

    print(f"\nGold: {len(gold)} rows, LLM: {len(llm)} rows")
    print(f"R1 raw: {len(r1)} rows, R2 raw: {len(r2)} rows")

    # -- 1. Narration-level agreement ----------------------------------------
    print("\n--- Narration-level Agreement (Gold vs LLM) ---")
    result = compute_agreement(gold, llm, on='narr_norm')
    print(f"Matched narrations: {result['n_matched']}")
    print(f"Overall agreement: {result['overall_agreement']:.3f}")
    print("\nPer-class agreement:")
    for cls in ACTION_ORDER:
        rate = result['per_class'].get(cls, 0)
        print(f"  {cls:20s}: {rate:.3f}")

    # Figure 1: Per-class agreement bar chart
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.2))
    classes = ACTION_ORDER
    rates = [result['per_class'].get(c, 0) for c in classes]
    colors = get_5class_colors(classes)
    bars = ax.barh(classes, rates, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel('Agreement Rate')
    ax.set_title('LLM vs Human Gold: Per-Class Agreement')
    ax.invert_yaxis()
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f'{rate:.2f}', va='center', fontsize=8)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / 'llm_per_class_agreement')
    plt.close(fig)

    # -- 2. Correction matrix from Bad annotations ---------------------------
    print("\n--- Bad Correction Matrix (R1+R2 combined) ---")
    bad_r1 = r1[r1['verdict'] == 'Bad'].copy()
    bad_r2 = r2[r2['verdict'] == 'Bad'].copy()

    corrections = []
    for df_bad in [bad_r1, bad_r2]:
        mask = df_bad['corrected_action'].str.strip().ne('')
        subset = df_bad[mask].copy()
        subset['corrected_action'] = subset['corrected_action'].replace(
            {'Essential Operation': 'Task Operation'})
        corrections.append(subset[['action', 'corrected_action']].rename(
            columns={'action': 'llm_action', 'corrected_action': 'gold_action'}))
    all_corrections = pd.concat(corrections, ignore_index=True)
    # Filter to valid 5-class only
    valid = set(ACTION_ORDER)
    all_corrections = all_corrections[
        all_corrections['llm_action'].isin(valid) &
        all_corrections['gold_action'].isin(valid)
    ]

    cm = build_correction_matrix(all_corrections)
    print(f"Total Bad corrections with valid corrected_action: {len(all_corrections)}")
    print("\nCorrection matrix (LLM original -> Human corrected):")
    print(cm.to_string())

    # Symmetric confusion: top pairs
    print("\n--- Top Symmetric Confusion Pairs ---")
    pairs = {}
    for i, a in enumerate(ACTION_ORDER):
        for j, b in enumerate(ACTION_ORDER):
            if i < j:
                sym = cm.loc[a, b] + cm.loc[b, a]
                if sym > 0:
                    pairs[f'{a} <-> {b}'] = sym
    for pair, count in sorted(pairs.items(), key=lambda x: -x[1])[:7]:
        print(f"  {pair}: {count}")

    # Figure 2: Correction heatmap
    fig, ax = plt.subplots(figsize=(SINGLE_COL + 0.8, SINGLE_COL + 0.3))
    # Mask diagonal
    mask = np.eye(len(ACTION_ORDER), dtype=bool)
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', mask=mask,
                linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_xlabel('Human Corrected Label')
    ax.set_ylabel('LLM Original Label')
    ax.set_title('LLM Error Correction Matrix')
    fig.tight_layout()
    save_figure(fig, FIG_DIR / 'llm_correction_matrix')
    plt.close(fig)

    # -- 3. Per-scenario LLM accuracy ---------------------------------------
    print("\n--- Per-Scenario LLM Accuracy ---")
    merged = result['merged']
    # Need scenario info -- join from gold
    gold_scenario = gold[['narr_norm', 'scenario']].drop_duplicates('narr_norm')
    merged_s = merged.merge(gold_scenario, on='narr_norm', how='left')

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.5))
    scenario_acc = []
    for s in SCENARIO_ORDER:
        sub = merged_s[merged_s['scenario'] == s]
        acc = sub['agree'].mean() if len(sub) > 0 else 0
        scenario_acc.append(acc)
        print(f"  {s:22s}: {acc:.3f} (n={len(sub)})")

    ax.barh(SCENARIO_ORDER, scenario_acc, color='#4477AA', edgecolor='white', linewidth=0.5)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel('Agreement Rate')
    ax.set_title('LLM Accuracy by Scenario')
    ax.invert_yaxis()
    for i, (s, acc) in enumerate(zip(SCENARIO_ORDER, scenario_acc)):
        ax.text(acc + 0.02, i, f'{acc:.2f}', va='center', fontsize=8)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / 'llm_per_scenario_accuracy')
    plt.close(fig)

    # -- 4. Per-class Bad rate (R1 vs R2) ------------------------------------
    print("\n--- Per-Class Bad Rate (R1 vs R2) ---")
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.5))
    x = np.arange(len(ACTION_ORDER))
    width = 0.35

    bad_rates_r1, bad_rates_r2 = [], []
    for cls in ACTION_ORDER:
        r1_cls = r1[r1['action'] == cls]
        r1_bad = (r1_cls['verdict'] == 'Bad').sum()
        bad_rates_r1.append(r1_bad / len(r1_cls) if len(r1_cls) > 0 else 0)

        r2_cls = r2[r2['action'] == cls]
        r2_bad = (r2_cls['verdict'] == 'Bad').sum()
        bad_rates_r2.append(r2_bad / len(r2_cls) if len(r2_cls) > 0 else 0)
        print(f"  {cls:20s}: R1={bad_rates_r1[-1]:.3f} R2={bad_rates_r2[-1]:.3f}")

    ax.bar(x - width/2, bad_rates_r1, width, label='R1', color='#4477AA')
    ax.bar(x + width/2, bad_rates_r2, width, label='R2', color='#EE6677')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(' ', '\n') for c in ACTION_ORDER], fontsize=7)
    ax.set_ylabel('Bad Rate')
    ax.set_title('LLM Error Rate by Class (R1 vs R2)')
    ax.legend(frameon=True, fancybox=False)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / 'llm_bad_rate_by_class')
    plt.close(fig)

    print(f"\nFigures saved to {FIG_DIR}/")


if __name__ == '__main__':
    main()
