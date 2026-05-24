#!/usr/bin/env python3
"""L2: Per-Scenario Class Overlap Quantification.

For each scenario, quantifies how much OT and TO overlap using:
  1. Narration-level: % of samples with shared verbs
  2. IMU feature-level: KNN-based overlap score (how often does a sample's
     nearest neighbor belong to the OTHER class?)

This produces a per-scenario "overlap score" that can inform:
  - Which scenarios to downweight during training
  - Paper discussion of scenario-dependent taxonomy quality

Outputs:
  - figures/scenario_overlap.pdf — per-scenario overlap bar chart
  - outputs/scenario_overlap.json — per-scenario overlap metrics
  - stdout: ranked scenario table

Usage:
    python analysis/layer2_signals/l2_scenario_overlap.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from analysis.shared.data_loader import load_gold, SCENARIO_ORDER
from analysis.shared.plot_style import apply_style, save_figure

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR / 'figures'
OUT_DIR = SCRIPT_DIR / 'outputs'
FEATURE_CSV = OUT_DIR / 'gold_imu_features.csv'
META_COLS = {'video_uid', 'timestamp_sec', 'action', 'scenario', 'tier', 'confidence'}

NON_VERB_TOKENS = {
    'man', 'woman', 'person', 'child', 'lady', 'guy', 'boy', 'girl',
    'someone', 'he', 'she', 'they', 'it', 'the', 'a', 'an', 'is', 'are',
}


def knn_cross_class_rate(X, y, k=5):
    """[FIX-5] LOO-CV KNN cross-class rate with class-balance-adjusted baseline.

    For each sample, uses leave-one-out: finds k nearest neighbors EXCLUDING self,
    computes fraction belonging to a different class.

    Returns: per-sample cross_rates, mean rate, and expected rate under random mixing
    (which depends on class balance, not a fixed 50%).
    """
    from sklearn.neighbors import NearestNeighbors

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # LOO: fit on all, query k+1 (first is self), skip self
    nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
    nn.fit(X_scaled)
    distances, indices = nn.kneighbors(X_scaled)

    # Skip self (index 0)
    neighbor_labels = y[indices[:, 1:]]
    cross_rates = np.mean(neighbor_labels != y.reshape(-1, 1), axis=1)

    # [FIX-5] Class-balance-adjusted baseline: expected cross-class rate under random mixing
    # P(different class) = 1 - sum(p_i^2) where p_i is class proportion
    unique, counts = np.unique(y, return_counts=True)
    props = counts / counts.sum()
    random_baseline = 1.0 - np.sum(props ** 2)

    return cross_rates, float(np.mean(cross_rates)), float(random_baseline)


def main():
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load IMU features
    df = pd.read_csv(FEATURE_CSV)
    feature_cols = [c for c in df.columns if c not in META_COLS]

    # Load gold for verb analysis
    gold = load_gold()
    gold = gold[gold['status'] == 'Gold']

    results = {}

    print(f"{'Scenario':<22} {'n_OT':>5} {'n_TO':>5} {'VerbOvlp':>9} {'KNN_Ovlp':>9}")
    print('-' * 60)

    for scenario in SCENARIO_ORDER:
        # IMU feature-level overlap
        sc_feat = df[(df['scenario'] == scenario) &
                     (df['action'].isin(['Object Transfer', 'Task Operation']))]
        n_ot = len(sc_feat[sc_feat['action'] == 'Object Transfer'])
        n_to = len(sc_feat[sc_feat['action'] == 'Task Operation'])

        knn_overlap = None
        knn_baseline = None
        if n_ot >= 10 and n_to >= 10:
            X = sc_feat[feature_cols].values
            y = (sc_feat['action'] == 'Task Operation').values.astype(int)
            # [FIX-5] LOO-CV + class-balance baseline
            _, knn_overlap, knn_baseline = knn_cross_class_rate(X, y, k=5)

        # Verb-level overlap
        sc_gold = gold[(gold['scenario'] == scenario) &
                       (gold['action'].isin(['Object Transfer', 'Task Operation']))]
        sc_gold = sc_gold[~sc_gold['verb'].isin(NON_VERB_TOKENS)]
        sc_gold = sc_gold.dropna(subset=['verb'])

        shared_verbs = set()
        if len(sc_gold) > 0:
            ot_verbs = set(sc_gold[sc_gold['action'] == 'Object Transfer']['verb'])
            to_verbs = set(sc_gold[sc_gold['action'] == 'Task Operation']['verb'])
            shared_verbs = ot_verbs & to_verbs
            on_shared = sc_gold[sc_gold['verb'].isin(shared_verbs)]
            verb_overlap = len(on_shared) / len(sc_gold) if len(sc_gold) > 0 else 0
        else:
            verb_overlap = 0

        results[scenario] = {
            'n_ot': n_ot, 'n_to': n_to,
            'verb_overlap_rate': float(verb_overlap),
            'knn_cross_class_rate': knn_overlap,
            'knn_random_baseline': knn_baseline,  # [FIX-5] Class-balance adjusted
            'n_shared_verbs': len(shared_verbs),
            'shared_verbs': sorted(shared_verbs)[:10],
        }

        knn_str = f'{knn_overlap:.3f}' if knn_overlap is not None else 'N/A'
        print(f"{scenario:<22} {n_ot:>5} {n_to:>5} {verb_overlap:>8.1%} {knn_str:>9}")

    # ── Figure: Per-scenario overlap (dual metric) ──
    valid = [(s, r) for s, r in results.items()
             if r['knn_cross_class_rate'] is not None]
    valid.sort(key=lambda x: x[1]['knn_cross_class_rate'], reverse=True)

    scenarios = [s for s, _ in valid]
    knn_rates = [r['knn_cross_class_rate'] for _, r in valid]
    verb_rates = [r['verb_overlap_rate'] for _, r in valid]

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    x = np.arange(len(scenarios))
    w = 0.35
    ax.bar(x - w/2, [v * 100 for v in verb_rates], w, color='#4477AA',
           label='Verb Overlap %', edgecolor='#333333', linewidth=0.5)
    ax.bar(x + w/2, [k * 100 for k in knn_rates], w, color='#CC6677',
           label='KNN Cross-Class %', edgecolor='#333333', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace(' ', '\n') if len(s) > 8 else s
                        for s in scenarios], fontsize=6.5)
    ax.set_ylabel('Overlap Rate (%)', fontsize=8)
    ax.set_title('OT↔TO Overlap by Scenario (Verb + IMU Features)', fontsize=9,
                 fontweight='bold')
    ax.legend(fontsize=7)
    # [FIX-5] Per-scenario class-balance baseline (not hardcoded 50%)
    baselines = [r['knn_random_baseline'] * 100 for _, r in valid
                 if r.get('knn_random_baseline') is not None]
    if baselines:
        mean_bl = np.mean(baselines)
        ax.axhline(y=mean_bl, color='#999999', linestyle=':', linewidth=0.8,
                   label=f'Random baseline (avg={mean_bl:.0f}%)')
    fig.tight_layout()
    save_figure(fig, FIG_DIR / 'scenario_overlap')
    plt.close(fig)

    # ── JSON ──
    summary = {
        'results': results,
        'ranked_by_knn_overlap': [s for s, _ in valid],
        'worst_scenario': valid[0][0] if valid else None,
        'best_scenario': valid[-1][0] if valid else None,
    }
    with open(OUT_DIR / 'scenario_overlap.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ JSON → {OUT_DIR / 'scenario_overlap.json'}")
    print(f"✓ Figure → {FIG_DIR}/scenario_overlap.pdf")


if __name__ == '__main__':
    main()
