#!/usr/bin/env python3
"""L2-5: Propagated Label IMU Consistency Check.

Research question: Do propagated labels (same narration text, different
videos) have consistent IMU patterns? Identifies unreliable propagations.

This analysis loads training data (gold + propagated), extracts IMU features
for propagated samples, and compares them against gold samples with the same
normalized narration.

Inputs:
  - data/processed/train.csv
  - IMU NPZ files (server)

Outputs:
  - figures/propagation_similarity_dist.pdf  — Distribution of cosine similarity
  - figures/propagation_quality_by_class.pdf — Per-class propagation reliability
  - outputs/unreliable_propagations.csv      — List of low-similarity propagated samples
  - stdout: consistency statistics

Usage:
    python analysis/layer2_signals/l2_propagation_consistency.py \
        --imu-dir data/processed_ego4d
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from analysis.shared.plot_style import (
    apply_style, save_figure, SINGLE_COL, DOUBLE_COL, COLORS_5CLASS,
    get_5class_colors,
)
from analysis.shared.data_loader import (
    load_train, normalize_narration, ACTION_ORDER,
)
from analysis.layer2_signals.l2_feature_extraction import extract_window_features

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR / 'figures'
OUT_DIR = SCRIPT_DIR / 'outputs'

PAD_SEC = 0.5
WINDOW_SAMPLES = 50


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imu-dir', type=str, required=True)
    parser.add_argument('--max-narrations', type=int, default=500,
                        help='Max narration groups to analyze (for speed)')
    args = parser.parse_args()

    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    imu_dir = Path(args.imu_dir)

    print("=" * 60)
    print("L2-5: Propagated Label IMU Consistency Check")
    print("=" * 60)

    train = load_train()
    train['narr_norm'] = train['narration_text'].apply(normalize_narration)

    gold = train[train['source'] == 'gold'].copy()
    propagated = train[train['source'] == 'propagated'].copy()
    print(f"Gold: {len(gold)}, Propagated: {len(propagated)}")

    # Find narrations that have both gold and propagated samples
    gold_narrs = set(gold['narr_norm'].unique())
    prop_narrs = set(propagated['narr_norm'].unique())
    shared_narrs = gold_narrs & prop_narrs
    print(f"Shared narrations (gold & propagated): {len(shared_narrs)}")

    # Sample for speed
    shared_list = sorted(shared_narrs)[:args.max_narrations]

    # Extract features and compute similarity
    similarities = []
    scaler = StandardScaler()

    # Cache loaded NPZ files
    npz_cache = {}

    def get_traj(vid):
        if vid not in npz_cache:
            # NPZ files stored as {video_uid}/seq.npz
            path = imu_dir / vid / 'seq.npz'
            if not path.exists():
                path = imu_dir / f'{vid}.npz'  # fallback
            if path.exists():
                data = np.load(path)
                npz_cache[vid] = (data['traj'], data['timestamp'])
            else:
                npz_cache[vid] = None
        return npz_cache[vid]

    def extract_for_row(row):
        result = get_traj(row['video_uid'])
        if result is None:
            return None
        traj, timestamps = result
        ts = row['timestamp_sec']
        mask = (timestamps >= ts - PAD_SEC) & (timestamps <= ts + PAD_SEC)
        window = traj[mask]
        if len(window) < 5:
            return None
        if len(window) > WINDOW_SAMPLES:
            center = len(window) // 2
            start = max(0, center - WINDOW_SAMPLES // 2)
            window = window[start:start + WINDOW_SAMPLES]
        feats = extract_window_features(window)
        return np.array([v for v in feats.values()])

    for narr in tqdm(shared_list, desc='Checking propagation consistency'):
        gold_rows = gold[gold['narr_norm'] == narr]
        prop_rows = propagated[propagated['narr_norm'] == narr]

        # Extract features for gold samples
        gold_feats = []
        for _, row in gold_rows.iterrows():
            f = extract_for_row(row)
            if f is not None:
                gold_feats.append(f)

        if len(gold_feats) == 0:
            continue

        gold_feat_array = np.array(gold_feats)
        gold_centroid = gold_feat_array.mean(axis=0)

        # Compare each propagated sample to gold centroid
        for _, row in prop_rows.iterrows():
            f = extract_for_row(row)
            if f is None:
                continue

            # Handle NaN
            valid = ~(np.isnan(f) | np.isnan(gold_centroid))
            if valid.sum() < 5:
                continue

            cos_sim = cosine_similarity(
                f[valid].reshape(1, -1),
                gold_centroid[valid].reshape(1, -1)
            )[0, 0]

            similarities.append({
                'narr_norm': narr,
                'video_uid': row['video_uid'],
                'timestamp_sec': row['timestamp_sec'],
                'action': row['action'],
                'cosine_similarity': cos_sim,
                'n_gold_refs': len(gold_feats),
            })

    sim_df = pd.DataFrame(similarities)
    print(f"\nAnalyzed {len(sim_df)} propagated samples across {len(shared_list)} narrations")

    if len(sim_df) == 0:
        print("No similarities computed. Check IMU data availability.")
        return

    # Statistics
    print(f"\nCosine similarity distribution:")
    for p in [10, 25, 50, 75, 90]:
        print(f"  P{p}: {sim_df['cosine_similarity'].quantile(p/100):.3f}")
    print(f"  Mean: {sim_df['cosine_similarity'].mean():.3f}")

    low_sim = sim_df[sim_df['cosine_similarity'] < 0.5]
    print(f"\nUnreliable propagations (cosine < 0.5): {len(low_sim)} "
          f"({len(low_sim)/len(sim_df)*100:.1f}%)")

    # Save unreliable list
    low_sim.to_csv(OUT_DIR / 'unreliable_propagations.csv', index=False)

    # -- Figure 1: Similarity distribution --
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.2))
    ax.hist(sim_df['cosine_similarity'], bins=50, color='#4477AA',
            edgecolor='white', linewidth=0.5, alpha=0.8)
    ax.axvline(0.5, color='#EE6677', linestyle='--', linewidth=1.5,
               label='Threshold (0.5)')
    ax.axvline(sim_df['cosine_similarity'].median(), color='#228833',
               linestyle='--', linewidth=1.5,
               label=f"Median ({sim_df['cosine_similarity'].median():.2f})")
    ax.set_xlabel('Cosine Similarity (propagated vs gold centroid)')
    ax.set_ylabel('Count')
    ax.set_title('Propagated Label IMU Consistency')
    ax.legend(frameon=True, fancybox=False, fontsize=7)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / 'propagation_similarity_dist')
    plt.close(fig)

    # -- Figure 2: Per-class reliability --
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.2))
    class_means = []
    class_unreliable = []
    for cls in ACTION_ORDER:
        sub = sim_df[sim_df['action'] == cls]
        class_means.append(sub['cosine_similarity'].mean() if len(sub) > 0 else 0)
        class_unreliable.append(
            (sub['cosine_similarity'] < 0.5).mean() * 100 if len(sub) > 0 else 0)

    x = np.arange(len(ACTION_ORDER))
    ax.bar(x, class_means, color=get_5class_colors(ACTION_ORDER),
           edgecolor='white', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(' ', '\n') for c in ACTION_ORDER], fontsize=7)
    ax.set_ylabel('Mean Cosine Similarity')
    ax.set_title('Propagation Reliability by Class')
    ax.set_ylim(0, 1.0)

    # Add % unreliable text
    for i, pct in enumerate(class_unreliable):
        ax.text(i, class_means[i] + 0.03, f'{pct:.0f}%\nunrel.',
                ha='center', va='bottom', fontsize=6, color='#EE6677')
    fig.tight_layout()
    save_figure(fig, FIG_DIR / 'propagation_quality_by_class')
    plt.close(fig)

    print(f"\nFigures saved to {FIG_DIR}/")
    print(f"Unreliable propagations saved to {OUT_DIR / 'unreliable_propagations.csv'}")


if __name__ == '__main__':
    main()
