#!/usr/bin/env python3
"""L1: Sample-Level Ambiguity Scoring.

[FIX-3] Assigns each sample an ambiguity_score in [0, 1] based on two components:
  - verb_entropy: how ambiguous is this sample's verb across classes? (0-1, normalized)
  - tier_score: Tier 4 → excluded, Tier 3 → 0.7, Tier 2 → 0.3, Tier 1 → 0.0

  ambiguity_score = 0.6 * verb_entropy + 0.4 * tier_score

[FIX-6] Also propagates scores to propagated (non-gold) labels via narr_norm join,
since training data is 84% propagated rows.

Outputs:
  - outputs/ambiguous_samples.csv — cleaned gold + propagated with ambiguity_score
  - figures/ambiguity_distribution.pdf — distribution of scores
  - figures/ambiguity_by_class.pdf — per-class ambiguity distribution
  - outputs/ambiguity_summary.json — statistics and recommended thresholds

Usage:
    python analysis/layer1_labels/l1_sample_ambiguity.py
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
from analysis.shared.data_loader import load_gold, ACTION_ORDER, normalize_narration
from analysis.shared.plot_style import apply_style, save_figure

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
FIG_DIR = SCRIPT_DIR / 'figures'
OUT_DIR = SCRIPT_DIR / 'outputs'

# [FIX-4] Canonical tier source
TIER_CSV = PROJECT_ROOT / 'data' / 'processed' / 'tier_assignments.csv'
TRAIN_CSV = PROJECT_ROOT / 'data' / 'processed' / 'train.csv'

NON_VERB_TOKENS = {
    'man', 'woman', 'person', 'child', 'lady', 'guy', 'boy', 'girl',
    'someone', 'he', 'she', 'they', 'it', 'the', 'a', 'an', 'is', 'are',
}

# [FIX-3] Removed pair_overlap — it was a class prior, not sample-level
# [FIX-4] Tier 4 excluded from training, so don't score them
TIER_SCORES = {1: 0.0, 2: 0.3, 3: 0.7}  # Tier 4 excluded by pipeline


def load_cleaned_gold():
    """Load gold with same cleaning as training pipeline."""
    gold = load_gold()
    gold = gold[gold['status'] == 'Gold'].copy()
    mask_O = gold['narration_text'].str.contains(r'#O\b', na=False)
    gold = gold[~mask_O].copy()
    nunique = gold.groupby('narr_norm')['action'].nunique()
    multi_narrs = set(nunique[nunique > 1].index)
    if multi_narrs:
        gold = gold[~gold['narr_norm'].isin(multi_narrs)].copy()
    gold = gold[~gold['action'].isin(['Error', 'Invalid', 'Unknown', ''])].copy()
    return gold


def compute_verb_entropies(gold):
    """Compute per-verb entropy across ALL 5 classes."""
    valid = gold[~gold['verb'].isin(NON_VERB_TOKENS)].dropna(subset=['verb'])
    valid = valid[valid['verb'].str.len() > 1]
    max_h = np.log2(len(ACTION_ORDER))
    verb_entropy = {}
    for verb, grp in valid.groupby('verb'):
        counts = grp['action'].value_counts()
        probs = np.array([counts.get(c, 0) for c in ACTION_ORDER]) / len(grp)
        h = scipy_entropy(probs, base=2)
        verb_entropy[verb] = h / max_h
    return verb_entropy


def main():
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gold = load_cleaned_gold()
    print(f"Loaded {len(gold)} cleaned gold samples")

    # ── Component 1: Verb entropy ──
    verb_entropies = compute_verb_entropies(gold)
    gold['verb_entropy'] = gold['verb'].map(verb_entropies).fillna(0)

    # ── Component 2: Tier score [FIX-4] from canonical tier_assignments.csv ──
    if TIER_CSV.exists():
        tier_df = pd.read_csv(TIER_CSV)
        gold = gold.merge(tier_df[['video_uid', 'timestamp_sec', 'tier']].drop_duplicates(),
                          on=['video_uid', 'timestamp_sec'], how='left')
        print(f"  Tier matched: {gold['tier'].notna().sum()}/{len(gold)}")
    else:
        print(f"  WARNING: {TIER_CSV} not found, skipping tier component")
        gold['tier'] = np.nan

    gold['tier_score'] = gold['tier'].map(TIER_SCORES).fillna(0.3)  # Default to moderate

    # Exclude Tier 4 (pipeline already excludes them from training)
    n_tier4 = (gold['tier'] == 4).sum()
    gold = gold[gold['tier'] != 4].copy()
    print(f"  Excluded {n_tier4} Tier 4 samples (pipeline excludes them)")

    # ── Combined score [FIX-3] — two components only ──
    gold['ambiguity_score'] = (
        0.6 * gold['verb_entropy'] +
        0.4 * gold['tier_score']
    ).clip(0, 1)

    # ── Stats ──
    print(f"\n{'='*60}")
    print(f"Ambiguity Score Distribution")
    print(f"{'='*60}")
    print(f"Mean: {gold['ambiguity_score'].mean():.3f}")
    print(f"Median: {gold['ambiguity_score'].median():.3f}")
    print(f"Std: {gold['ambiguity_score'].std():.3f}")

    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        n_above = (gold['ambiguity_score'] >= threshold).sum()
        pct = n_above / len(gold) * 100
        print(f"  score >= {threshold}: {n_above} ({pct:.1f}%)")

    print(f"\nPer-class mean ambiguity:")
    for cls in ACTION_ORDER:
        cls_data = gold[gold['action'] == cls]
        print(f"  {cls:<20}: {cls_data['ambiguity_score'].mean():.3f} (n={len(cls_data)})")

    # ── Threshold recommendation ──
    # Use 0.5 as default: keeps ~85%+ of data while removing worst ambiguities
    recommended_threshold = 0.5
    n_keep = (gold['ambiguity_score'] < recommended_threshold).sum()
    n_flag = (gold['ambiguity_score'] >= recommended_threshold).sum()

    print(f"\n{'='*60}")
    print(f"RECOMMENDATION: threshold={recommended_threshold}")
    print(f"  Keep: {n_keep} ({n_keep/len(gold)*100:.1f}%)")
    print(f"  Flag/downweight: {n_flag} ({n_flag/len(gold)*100:.1f}%)")
    print(f"{'='*60}")

    # ── Save gold CSV ──
    out_cols = ['video_uid', 'timestamp_sec', 'narration_text', 'narr_norm',
                'scenario', 'action', 'verb', 'tier',
                'verb_entropy', 'tier_score', 'ambiguity_score']
    out_df = gold[[c for c in out_cols if c in gold.columns]]
    out_df.to_csv(OUT_DIR / 'ambiguous_samples_gold.csv', index=False)
    print(f"\n✓ Gold CSV → {OUT_DIR / 'ambiguous_samples_gold.csv'} ({len(out_df)} rows)")

    # ── [FIX-6] Propagate to training data via narr_norm join ──
    # Training data is 84% propagated — ambiguity must propagate too
    if TRAIN_CSV.exists():
        train_df = pd.read_csv(TRAIN_CSV)
        # Normalize narration for join
        if 'narr_norm' not in train_df.columns and 'narration_text' in train_df.columns:
            train_df['narr_norm'] = train_df['narration_text'].apply(normalize_narration)

        # Build verb→ambiguity lookup from gold
        verb_amb = gold.groupby('narr_norm')['ambiguity_score'].mean().to_dict()
        train_df['ambiguity_score'] = train_df['narr_norm'].map(verb_amb)

        # For unmatched propagated rows, use verb-level entropy as fallback
        if 'narration_text' in train_df.columns:
            from analysis.shared.data_loader import extract_verb
            train_df['_verb'] = train_df['narration_text'].apply(extract_verb)
            unmatched = train_df['ambiguity_score'].isna()
            train_df.loc[unmatched, 'ambiguity_score'] = (
                train_df.loc[unmatched, '_verb'].map(verb_entropies).fillna(0) * 0.6
            )
            train_df.drop(columns=['_verb'], inplace=True)

        train_df['ambiguity_score'] = train_df['ambiguity_score'].fillna(0).clip(0, 1)

        train_out = OUT_DIR / 'ambiguous_samples.csv'
        train_df.to_csv(train_out, index=False)
        n_flagged_train = (train_df['ambiguity_score'] >= recommended_threshold).sum()
        print(f"✓ Training CSV → {train_out} ({len(train_df)} rows, "
              f"{n_flagged_train} flagged at >={recommended_threshold})")
    else:
        print(f"  WARNING: {TRAIN_CSV} not found, skipping propagation")

    # ── Figure 1: Score distribution ──
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(gold['ambiguity_score'], bins=50, color='#4477AA', edgecolor='#333333',
            linewidth=0.3, alpha=0.8)
    ax.axvline(x=recommended_threshold, color='#CC6677', linestyle='--', linewidth=1.5,
               label=f'Threshold={recommended_threshold}')
    ax.set_xlabel('Ambiguity Score', fontsize=8)
    ax.set_ylabel('Count', fontsize=8)
    ax.set_title('Sample Ambiguity Score Distribution', fontsize=9, fontweight='bold')
    ax.legend(fontsize=7)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / 'ambiguity_distribution')
    plt.close(fig)

    # ── Figure 2: Per-class box plot ──
    fig, ax = plt.subplots(figsize=(5, 3.5))
    class_data = [gold[gold['action'] == cls]['ambiguity_score'].values
                  for cls in ACTION_ORDER]
    short_labels = ['ObjTr', 'TaskOp', 'Stat', 'Loco', 'Search']
    bp = ax.boxplot(class_data, labels=short_labels, patch_artist=True,
                    showfliers=False, medianprops={'color': '#333333'})
    colors = ['#4477AA', '#CC6677', '#117733', '#DDCC77', '#AA4499']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.axhline(y=recommended_threshold, color='#CC6677', linestyle='--',
               linewidth=1, alpha=0.7)
    ax.set_ylabel('Ambiguity Score', fontsize=8)
    ax.set_title('Ambiguity Score by Action Class', fontsize=9, fontweight='bold')
    fig.tight_layout()
    save_figure(fig, FIG_DIR / 'ambiguity_by_class')
    plt.close(fig)

    # ── JSON summary ──
    summary = {
        'total_samples': len(gold),
        'mean_score': float(gold['ambiguity_score'].mean()),
        'thresholds': {},
        'per_class': {},
        'recommended_threshold': recommended_threshold,
        'n_flagged': int(n_flag),
        'pct_flagged': float(n_flag / len(gold) * 100),
        'components': {
            'verb_entropy_weight': 0.6,
            'tier_score_weight': 0.4,
            # [FIX-3] pair_overlap removed — was a class prior, not sample-level
        },
    }
    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        n = int((gold['ambiguity_score'] >= t).sum())
        summary['thresholds'][str(t)] = {'n_flagged': n, 'pct': float(n/len(gold)*100)}
    for cls in ACTION_ORDER:
        cd = gold[gold['action'] == cls]['ambiguity_score']
        summary['per_class'][cls] = {
            'mean': float(cd.mean()), 'median': float(cd.median()),
            'n': len(cd), 'n_above_threshold': int((cd >= recommended_threshold).sum()),
        }

    with open(OUT_DIR / 'ambiguity_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ JSON → {OUT_DIR / 'ambiguity_summary.json'}")
    print(f"✓ Figures → {FIG_DIR}/ambiguity_*.pdf")


if __name__ == '__main__':
    main()
