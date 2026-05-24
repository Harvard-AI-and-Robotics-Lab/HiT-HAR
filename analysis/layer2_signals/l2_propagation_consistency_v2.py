#!/usr/bin/env python3
"""L2-5v2: Propagation Consistency — Full Evaluation + Random Sampling.

Updates:
  - Evaluates ALL shared narrations (not just first 500 sorted)
  - Random stratified sampling when --max-narrations is used
  - Threshold sweep (0.3, 0.4, 0.5, 0.6, 0.7) instead of single 0.5
  - Reports per-class breakdown

Outputs:
  - figures/propagation_similarity_dist_v2.pdf   — Distribution with threshold sweep
  - figures/propagation_threshold_sweep_v2.pdf   — % unreliable vs threshold
  - figures/propagation_per_class_v2.pdf         — Per-class reliability
  - outputs/unreliable_propagations_v2.csv       — Full list at threshold=0.5
  - stdout: statistics

Usage:
    python analysis/layer2_signals/l2_propagation_consistency_v2.py \
        --imu-dir data/processed_ego4d \
        --max-narrations 0
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
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from analysis.shared.plot_style import (
    apply_style, save_figure, SINGLE_COL, DOUBLE_COL,
    get_5class_colors, styled_bar, styled_barh,
)
from analysis.shared.data_loader import load_train, normalize_narration, ACTION_ORDER
from analysis.layer2_signals.l2_feature_extraction import extract_window_features

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR / 'figures'
OUT_DIR = SCRIPT_DIR / 'outputs'
PAD_SEC = 0.5
WINDOW_SAMPLES = 50
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imu-dir', type=str, required=True)
    parser.add_argument('--max-narrations', type=int, default=0,
                        help='Max narration groups (0=all). Uses random stratified sampling.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    imu_dir = Path(args.imu_dir)

    print("=" * 60)
    print("L2-5v2: Propagation Consistency — Full Eval + Random Sampling")
    print("=" * 60)

    train = load_train()
    train['narr_norm'] = train['narration_text'].apply(normalize_narration)

    gold = train[train['source'] == 'gold'].copy()
    propagated = train[train['source'] == 'propagated'].copy()
    print(f"Gold: {len(gold)}, Propagated: {len(propagated)}")

    shared_narrs = sorted(set(gold['narr_norm'].unique()) & set(propagated['narr_norm'].unique()))
    print(f"Shared narrations (gold ∩ propagated): {len(shared_narrs)}")

    # Random stratified sampling (NOT sorted[:N])
    if args.max_narrations > 0 and args.max_narrations < len(shared_narrs):
        rng = np.random.RandomState(args.seed)
        shared_narrs = list(rng.choice(shared_narrs, args.max_narrations, replace=False))
        print(f"Random-sampled {args.max_narrations} narrations (seed={args.seed})")

    # NPZ cache
    npz_cache = {}

    def get_traj(vid):
        if vid not in npz_cache:
            path = imu_dir / vid / 'seq.npz'
            if not path.exists():
                path = imu_dir / f'{vid}.npz'
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

    # Extract and compare
    similarities = []
    for narr in tqdm(shared_narrs, desc='Checking propagation consistency'):
        gold_rows = gold[gold['narr_norm'] == narr]
        prop_rows = propagated[propagated['narr_norm'] == narr]

        gold_feats = []
        for _, row in gold_rows.iterrows():
            f = extract_for_row(row)
            if f is not None:
                gold_feats.append(f)

        if len(gold_feats) == 0:
            continue

        gold_centroid = np.nanmean(gold_feats, axis=0)

        for _, row in prop_rows.iterrows():
            f = extract_for_row(row)
            if f is None:
                continue
            valid = ~(np.isnan(f) | np.isnan(gold_centroid))
            if valid.sum() < 5:
                continue
            cos_sim = cosine_similarity(
                f[valid].reshape(1, -1), gold_centroid[valid].reshape(1, -1)
            )[0, 0]
            similarities.append({
                'narr_norm': narr, 'video_uid': row['video_uid'],
                'timestamp_sec': row['timestamp_sec'], 'action': row['action'],
                'cosine_similarity': cos_sim, 'n_gold_refs': len(gold_feats),
            })

    sim_df = pd.DataFrame(similarities)
    print(f"\nAnalyzed {len(sim_df)} propagated samples across {len(shared_narrs)} narrations")

    if len(sim_df) == 0:
        print("No similarities computed.")
        return

    # Statistics
    print(f"\nCosine similarity distribution:")
    for p in [10, 25, 50, 75, 90]:
        print(f"  P{p}: {sim_df['cosine_similarity'].quantile(p/100):.3f}")

    # Threshold sweep
    print(f"\n--- Threshold Sweep ---")
    threshold_results = []
    for t in THRESHOLDS:
        n_unreliable = (sim_df['cosine_similarity'] < t).sum()
        pct = n_unreliable / len(sim_df) * 100
        threshold_results.append({'threshold': t, 'n_unreliable': n_unreliable, 'pct': pct})
        print(f"  threshold={t:.1f}: {n_unreliable} unreliable ({pct:.1f}%)")

    # Save unreliable at 0.5
    unreliable = sim_df[sim_df['cosine_similarity'] < 0.5]
    unreliable.to_csv(OUT_DIR / 'unreliable_propagations_v2.csv', index=False)

    # Figure 1: Distribution with threshold lines
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.5))
    ax.hist(sim_df['cosine_similarity'], bins=50, color='#4477AA',
            edgecolor='#2D2D2D', linewidth=0.8, alpha=0.8)
    for t, color in zip([0.3, 0.5, 0.7], ['#AA4499', '#CC6677', '#DDCC77']):
        ax.axvline(t, color=color, linestyle='--', linewidth=1.2, label=f'θ={t}')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Count')
    ax.set_title('Propagated vs Gold IMU Consistency')
    ax.legend(frameon=True, fancybox=False, fontsize=7)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / 'propagation_similarity_dist_v2')
    plt.close(fig)

    # Figure 2: Threshold sweep
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.0))
    tr = pd.DataFrame(threshold_results)
    styled_bar(ax, [str(t) for t in tr['threshold']], tr['pct'], color='#CC6677')
    ax.set_xlabel('Cosine Similarity Threshold')
    ax.set_ylabel('% Unreliable')
    ax.set_title('Unreliable Propagations vs Threshold')
    fig.tight_layout()
    save_figure(fig, FIG_DIR / 'propagation_threshold_sweep_v2')
    plt.close(fig)

    # Figure 3: Per-class
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.2))
    class_means = [sim_df[sim_df['action'] == c]['cosine_similarity'].mean()
                   if len(sim_df[sim_df['action'] == c]) > 0 else 0
                   for c in ACTION_ORDER]
    styled_bar(ax, range(len(ACTION_ORDER)), class_means,
               color=get_5class_colors(ACTION_ORDER))
    ax.set_xticks(range(len(ACTION_ORDER)))
    ax.set_xticklabels([c.replace(' ', '\n') for c in ACTION_ORDER], fontsize=7)
    ax.set_ylabel('Mean Cosine Similarity')
    ax.set_title('Propagation Reliability by Class')
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / 'propagation_per_class_v2')
    plt.close(fig)

    print(f"\n✓ Figures saved to {FIG_DIR}/")


if __name__ == '__main__':
    main()
