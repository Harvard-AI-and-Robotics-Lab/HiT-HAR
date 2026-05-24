#!/usr/bin/env python3
"""L2-4: Physical F1 Ceiling Estimation.

Research question: What is the best possible macro-F1 achievable from head
IMU features alone, for each taxonomy? Uses KNN and LDA as non-parametric
upper bounds with cross-validation.

Inputs:
  - analysis/layer2_signals/outputs/gold_imu_features.csv (from L2-1)

Outputs:
  - figures/ceiling_by_taxonomy.pdf       — Bar: KNN/LDA ceiling for 5/4/3 class
  - figures/ceiling_per_class.pdf         — Per-class F1 ceiling (5-class)
  - figures/ceiling_vs_current.pdf        — Current model F1 vs ceiling
  - stdout: ceiling estimates with CI

Usage:
    python analysis/layer2_signals/l2_ceiling_estimation.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from analysis.shared.plot_style import (
    apply_style, save_figure, SINGLE_COL, DOUBLE_COL, COLORS_5CLASS,
    get_5class_colors,
)
from analysis.shared.data_loader import (
    ACTION_ORDER, ACTION_ORDER_4CLASS, ACTION_ORDER_3CLASS,
    map_to_4class, map_to_3class,
)

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR / 'figures'
FEATURE_CSV = SCRIPT_DIR / 'outputs' / 'gold_imu_features.csv'
META_COLS = {'video_uid', 'timestamp_sec', 'action', 'scenario', 'tier', 'confidence'}

# Current model best F1 (from WandB, for comparison)
CURRENT_MODEL_F1 = {
    '5-class': {'macro': 0.40, 'per_class': {
        'Object Transfer': 0.493, 'Task Operation': 0.458,
        'Stationary': 0.276, 'Locomotion': 0.529, 'Search': 0.232,
    }},
    '4-class': {'macro': 0.41},
}


def evaluate_ceiling(X, y, class_names, n_splits=5):
    """Evaluate KNN and LDA ceiling with stratified CV.

    Returns dict with macro_f1 mean/std and per-class F1 for each method.
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    scaler = StandardScaler()

    results = {}
    for name, clf in [('KNN-5', KNeighborsClassifier(n_neighbors=5)),
                      ('KNN-11', KNeighborsClassifier(n_neighbors=11)),
                      ('LDA', LinearDiscriminantAnalysis())]:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_f1s = []
        fold_per_class = []

        for train_idx, test_idx in skf.split(X, y_enc):
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            y_train, y_test = y_enc[train_idx], y_enc[test_idx]

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            macro = f1_score(y_test, y_pred, average='macro')
            per_cls = f1_score(y_test, y_pred, average=None, labels=range(len(class_names)))
            fold_f1s.append(macro)
            fold_per_class.append(per_cls)

        results[name] = {
            'macro_f1_mean': np.mean(fold_f1s),
            'macro_f1_std': np.std(fold_f1s),
            'per_class_f1': np.mean(fold_per_class, axis=0),
            'class_names': class_names,
        }

    return results


def main():
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("L2-4: Physical F1 Ceiling Estimation")
    print("=" * 60)

    df = pd.read_csv(FEATURE_CSV)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    df = df.dropna(subset=feature_cols)
    X = df[feature_cols].values

    # -- 1. 5-class ceiling --
    print("\n--- 5-Class Ceiling ---")
    y_5 = df['action'].values
    results_5 = evaluate_ceiling(X, y_5, ACTION_ORDER)
    for name, res in results_5.items():
        print(f"  {name}: macro-F1 = {res['macro_f1_mean']:.3f} +/- {res['macro_f1_std']:.3f}")
        for i, cls in enumerate(ACTION_ORDER):
            print(f"    {cls:20s}: {res['per_class_f1'][i]:.3f}")

    # -- 2. 4-class ceiling --
    print("\n--- 4-Class Ceiling ---")
    y_4 = df['action'].map(map_to_4class).values
    results_4 = evaluate_ceiling(X, y_4, ACTION_ORDER_4CLASS)
    for name, res in results_4.items():
        print(f"  {name}: macro-F1 = {res['macro_f1_mean']:.3f} +/- {res['macro_f1_std']:.3f}")

    # -- 3. 3-class ceiling --
    print("\n--- 3-Class Ceiling ---")
    y_3 = df['action'].map(map_to_3class).values
    results_3 = evaluate_ceiling(X, y_3, ACTION_ORDER_3CLASS)
    for name, res in results_3.items():
        print(f"  {name}: macro-F1 = {res['macro_f1_mean']:.3f} +/- {res['macro_f1_std']:.3f}")

    # -- Figure 1: Ceiling by taxonomy --
    fig, ax = plt.subplots(figsize=(SINGLE_COL + 0.5, 2.5))
    taxonomies = ['5-class', '4-class', '3-class']
    best_method = 'KNN-5'  # Use best method
    ceilings = [results_5[best_method]['macro_f1_mean'],
                results_4[best_method]['macro_f1_mean'],
                results_3[best_method]['macro_f1_mean']]
    ceiling_stds = [results_5[best_method]['macro_f1_std'],
                    results_4[best_method]['macro_f1_std'],
                    results_3[best_method]['macro_f1_std']]
    current = [0.40, 0.41, None]  # Current model F1

    x = np.arange(len(taxonomies))
    width = 0.35
    bars1 = ax.bar(x - width/2, ceilings, width, yerr=ceiling_stds,
                   label=f'KNN Ceiling', color='#4477AA', capsize=3)
    current_vals = [c if c is not None else 0 for c in current]
    bars2 = ax.bar(x + width/2, current_vals, width,
                   label='Current Model', color='#EE6677')
    ax.set_xticks(x)
    ax.set_xticklabels(taxonomies)
    ax.set_ylabel('Macro F1')
    ax.set_title('Model Performance vs Feature-Space Ceiling')
    ax.legend(frameon=True, fancybox=False)
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / 'ceiling_by_taxonomy')
    plt.close(fig)

    # -- Figure 2: Per-class ceiling (5-class) --
    fig, ax = plt.subplots(figsize=(SINGLE_COL + 0.5, 2.5))
    per_cls_ceiling = results_5[best_method]['per_class_f1']
    per_cls_current = [CURRENT_MODEL_F1['5-class']['per_class'].get(c, 0) for c in ACTION_ORDER]
    x = np.arange(len(ACTION_ORDER))

    bars1 = ax.bar(x - width/2, per_cls_ceiling, width,
                   label='KNN Ceiling', color=get_5class_colors(ACTION_ORDER), alpha=0.7)
    bars2 = ax.bar(x + width/2, per_cls_current, width,
                   label='Current Model', color=get_5class_colors(ACTION_ORDER), alpha=0.3,
                   hatch='//')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(' ', '\n') for c in ACTION_ORDER], fontsize=7)
    ax.set_ylabel('F1 Score')
    ax.set_title('Per-Class: Model vs Ceiling (5-Class)')
    ax.legend(frameon=True, fancybox=False, fontsize=7)
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / 'ceiling_per_class')
    plt.close(fig)

    # -- Summary table --
    print("\n" + "=" * 60)
    print("CEILING SUMMARY")
    print("=" * 60)
    print(f"{'Taxonomy':12s} | {'KNN-5':12s} | {'KNN-11':12s} | {'LDA':12s} | {'Current':8s}")
    print("-" * 60)
    for tax, res in [('5-class', results_5), ('4-class', results_4), ('3-class', results_3)]:
        knn5 = f"{res['KNN-5']['macro_f1_mean']:.3f}+/-{res['KNN-5']['macro_f1_std']:.3f}"
        knn11 = f"{res['KNN-11']['macro_f1_mean']:.3f}+/-{res['KNN-11']['macro_f1_std']:.3f}"
        lda = f"{res['LDA']['macro_f1_mean']:.3f}+/-{res['LDA']['macro_f1_std']:.3f}"
        cur = CURRENT_MODEL_F1.get(tax, {}).get('macro', 'N/A')
        print(f"{tax:12s} | {knn5:12s} | {knn11:12s} | {lda:12s} | {cur}")

    print(f"\nFigures saved to {FIG_DIR}/")


if __name__ == '__main__':
    main()
