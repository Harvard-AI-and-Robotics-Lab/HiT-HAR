#!/usr/bin/env python3
"""L2-4v2: Physical F1 Ceiling — GroupKFold + Bootstrap CI.

Updates:
  - GroupKFold(video_uid) instead of StratifiedKFold (no video leakage)
  - Bootstrap 95% CI instead of ±std
  - Clearly labeled as "per-window statistical feature ceiling"

Outputs:
  - figures/ceiling_by_taxonomy_v2.pdf    — Ceiling with 95% CI error bars
  - figures/ceiling_per_class_v2.pdf      — Per-class ceiling vs model
  - stdout: ceiling table with CI

Usage:
    python analysis/layer2_signals/l2_ceiling_estimation_v2.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from analysis.shared.plot_style import (
    apply_style, save_figure, SINGLE_COL, DOUBLE_COL,
    COLORS_5CLASS, get_5class_colors, styled_bar, ERRORBAR_DEFAULTS,
)
from analysis.shared.data_loader import (
    ACTION_ORDER, ACTION_ORDER_4CLASS, ACTION_ORDER_3CLASS,
    map_to_4class, map_to_3class,
)
from analysis.shared.stats_utils import bootstrap_ci

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR / 'figures'
FEATURE_CSV = SCRIPT_DIR / 'outputs' / 'gold_imu_features.csv'
META_COLS = {'video_uid', 'timestamp_sec', 'action', 'scenario', 'tier', 'confidence'}

CURRENT_MODEL_F1 = {
    '5-class': {'macro': 0.40, 'per_class': {
        'Object Transfer': 0.493, 'Task Operation': 0.458,
        'Stationary': 0.276, 'Locomotion': 0.529, 'Search': 0.232,
    }},
    '4-class': {'macro': 0.41},
}


def evaluate_ceiling_grouped(X, y, groups, class_names, n_splits=5):
    """Evaluate KNN and LDA ceiling with GroupKFold + bootstrap CI.

    Args:
        X: (n, d) feature matrix
        y: (n,) string labels
        groups: (n,) group identifiers (video_uid)
        class_names: ordered class names
        n_splits: number of CV folds
    Returns:
        dict of method → {macro_f1_mean, ci_lower, ci_upper, per_class_f1}
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    scaler = StandardScaler()

    gkf = GroupKFold(n_splits=n_splits)
    results = {}

    for name, clf in [('KNN-5', KNeighborsClassifier(n_neighbors=5)),
                      ('KNN-11', KNeighborsClassifier(n_neighbors=11)),
                      ('LDA', LinearDiscriminantAnalysis())]:
        fold_f1s = []
        fold_per_class = []

        for train_idx, test_idx in gkf.split(X, y_enc, groups):
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            y_train, y_test = y_enc[train_idx], y_enc[test_idx]

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            macro = f1_score(y_test, y_pred, average='macro')
            per_cls = f1_score(y_test, y_pred, average=None,
                               labels=range(len(class_names)))
            fold_f1s.append(macro)
            fold_per_class.append(per_cls)

        fold_f1s = np.array(fold_f1s)
        ci_lo, ci_hi = bootstrap_ci(fold_f1s, confidence=0.95, n_bootstrap=1000)

        results[name] = {
            'macro_f1_mean': float(fold_f1s.mean()),
            'macro_f1_std': float(fold_f1s.std()),
            'ci_lower': ci_lo,
            'ci_upper': ci_hi,
            'per_class_f1': np.mean(fold_per_class, axis=0),
            'class_names': class_names,
        }

    return results


def main():
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("L2-4v2: Physical F1 Ceiling — GroupKFold + Bootstrap CI")
    print("=" * 60)

    df = pd.read_csv(FEATURE_CSV)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    df = df.dropna(subset=feature_cols)
    X = df[feature_cols].values
    groups = df['video_uid'].values

    # ── 5-class ─────────────────────────────────────────────────
    print("\n--- 5-Class Ceiling (GroupKFold, no video leakage) ---")
    y_5 = df['action'].values
    results_5 = evaluate_ceiling_grouped(X, y_5, groups, ACTION_ORDER)
    for name, res in results_5.items():
        print(f"  {name}: F1={res['macro_f1_mean']:.3f} "
              f"95%CI=[{res['ci_lower']:.3f}, {res['ci_upper']:.3f}]")
        for i, cls in enumerate(ACTION_ORDER):
            print(f"    {cls:20s}: {res['per_class_f1'][i]:.3f}")

    # ── 4-class ─────────────────────────────────────────────────
    print("\n--- 4-Class Ceiling (GroupKFold) ---")
    y_4 = np.array([map_to_4class(a) for a in df['action'].values])
    results_4 = evaluate_ceiling_grouped(X, y_4, groups, ACTION_ORDER_4CLASS)
    for name, res in results_4.items():
        print(f"  {name}: F1={res['macro_f1_mean']:.3f} "
              f"95%CI=[{res['ci_lower']:.3f}, {res['ci_upper']:.3f}]")

    # ── 3-class ─────────────────────────────────────────────────
    print("\n--- 3-Class Ceiling (GroupKFold) ---")
    y_3 = np.array([map_to_3class(a) for a in df['action'].values])
    results_3 = evaluate_ceiling_grouped(X, y_3, groups, ACTION_ORDER_3CLASS)
    for name, res in results_3.items():
        print(f"  {name}: F1={res['macro_f1_mean']:.3f} "
              f"95%CI=[{res['ci_lower']:.3f}, {res['ci_upper']:.3f}]")

    # ── Figure 1: Ceiling by taxonomy with CI ──────────────────
    fig, ax = plt.subplots(figsize=(SINGLE_COL + 0.5, 2.8))
    best = 'KNN-5'
    taxonomies = ['5-class', '4-class', '3-class']
    ceilings = [results_5[best]['macro_f1_mean'],
                results_4[best]['macro_f1_mean'],
                results_3[best]['macro_f1_mean']]
    ci_lows = [results_5[best]['ci_lower'],
               results_4[best]['ci_lower'],
               results_3[best]['ci_lower']]
    ci_highs = [results_5[best]['ci_upper'],
                results_4[best]['ci_upper'],
                results_3[best]['ci_upper']]
    yerr_low = [c - l for c, l in zip(ceilings, ci_lows)]
    yerr_high = [h - c for c, h in zip(ceilings, ci_highs)]

    current = [0.40, 0.41, 0]

    x = np.arange(3)
    width = 0.35
    styled_bar(ax, x - width/2, ceilings, width=width,
               label=f'KNN-5 Ceiling (GroupKFold)', color='#4477AA',
               yerr=[yerr_low, yerr_high], **ERRORBAR_DEFAULTS)
    styled_bar(ax, x + width/2, current, width=width,
               label='Current Deep Model', color='#DDCC77')
    ax.set_xticks(x)
    ax.set_xticklabels(taxonomies)
    ax.set_ylabel('Macro F1')
    ax.set_title('Per-Window Feature Ceiling vs Deep Model')
    ax.legend(frameon=True, fancybox=False, fontsize=7)
    ax.set_ylim(0, 0.6)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / 'ceiling_by_taxonomy_v2')
    plt.close(fig)

    # ── Figure 2: Per-class ceiling vs model (5-class) ─────────
    fig, ax = plt.subplots(figsize=(SINGLE_COL + 0.5, 2.5))
    per_cls_ceiling = results_5[best]['per_class_f1']
    per_cls_current = [CURRENT_MODEL_F1['5-class']['per_class'].get(c, 0) for c in ACTION_ORDER]
    x = np.arange(len(ACTION_ORDER))

    styled_bar(ax, x - width/2, per_cls_ceiling, width=width,
               label='KNN-5 Ceiling', color=get_5class_colors(ACTION_ORDER), alpha=0.7)
    styled_bar(ax, x + width/2, per_cls_current, width=width,
               label='Deep Model', color=get_5class_colors(ACTION_ORDER), alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(' ', '\n') for c in ACTION_ORDER], fontsize=7)
    ax.set_ylabel('F1 Score')
    ax.set_title('Per-Class: Deep Model vs Feature Ceiling')
    ax.legend(frameon=True, fancybox=False, fontsize=7)
    ax.set_ylim(0, 0.8)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / 'ceiling_per_class_v2')
    plt.close(fig)

    # ── Summary table ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CEILING SUMMARY (GroupKFold, 95% CI)")
    print("=" * 70)
    print(f"{'Taxonomy':12s} | {'KNN-5 [95% CI]':22s} | {'KNN-11 [95% CI]':22s} | {'Current':8s}")
    print("-" * 70)
    for tax, res in [('5-class', results_5), ('4-class', results_4), ('3-class', results_3)]:
        k5 = res['KNN-5']
        k11 = res['KNN-11']
        k5_str = f"{k5['macro_f1_mean']:.3f} [{k5['ci_lower']:.3f},{k5['ci_upper']:.3f}]"
        k11_str = f"{k11['macro_f1_mean']:.3f} [{k11['ci_lower']:.3f},{k11['ci_upper']:.3f}]"
        cur = CURRENT_MODEL_F1.get(tax, {}).get('macro', 'N/A')
        print(f"{tax:12s} | {k5_str:22s} | {k11_str:22s} | {cur}")

    # ── Emit JSON summary ────────────────────────────────────
    out_dir = SCRIPT_DIR / 'outputs'
    out_dir.mkdir(parents=True, exist_ok=True)

    def _serialize_results(res_dict, class_names):
        """Convert results dict to JSON-safe format."""
        out = {}
        for method, res in res_dict.items():
            out[method] = {
                'macro_f1_mean': float(res['macro_f1_mean']),
                'macro_f1_std': float(res['macro_f1_std']),
                'ci_lower': float(res['ci_lower']),
                'ci_upper': float(res['ci_upper']),
                'per_class_f1': {
                    cls: float(res['per_class_f1'][i])
                    for i, cls in enumerate(class_names)
                },
            }
        return out

    summary = {
        '5-class': _serialize_results(results_5, ACTION_ORDER),
        '4-class': _serialize_results(results_4, ACTION_ORDER_4CLASS),
        '3-class': _serialize_results(results_3, ACTION_ORDER_3CLASS),
        'current_model_f1': CURRENT_MODEL_F1,
    }

    json_path = out_dir / 'ceiling_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ JSON summary saved to {json_path}")

    print(f"\n✓ Figures saved to {FIG_DIR}/")


if __name__ == '__main__':
    main()
