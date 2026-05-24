# tests/test_v2_analyses.py
"""Tests for v2 analysis functions."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestTaxonomyV2:
    def test_entropy_weighted_ambiguity(self):
        from analysis.layer1_labels.l1_taxonomy_boundary_v2 import compute_entropy_ambiguity

        df = pd.DataFrame({
            'verb': ['picks', 'picks', 'picks', 'picks', 'walks', 'walks', 'looks'],
            'action': ['OT', 'OT', 'OT', 'TO', 'Loco', 'Loco', 'Search'],
        })
        # 'picks': 3 OT, 1 TO → H = -(0.75 log 0.75 + 0.25 log 0.25) ≈ 0.811
        # 'walks': 2 Loco → H = 0
        # 'looks': 1 Search → H = 0
        rate, details = compute_entropy_ambiguity(df)
        assert details['picks'] > 0.5  # High entropy
        assert details['walks'] == 0.0  # Zero entropy
        assert rate > 0  # Weighted rate should be positive


class TestSeparabilityV2:
    def test_compute_separability_matrix_mmd(self):
        from analysis.layer2_signals.l2_class_separability_v2 import compute_separability_matrix_mmd

        rng = np.random.RandomState(42)
        df = pd.DataFrame(
            np.vstack([rng.randn(50, 3), rng.randn(50, 3) + 3]),
            columns=['f1', 'f2', 'f3'],
        )
        df['action'] = ['A'] * 50 + ['B'] * 50

        mat, pval = compute_separability_matrix_mmd(df, ['f1', 'f2', 'f3'], 'action', ['A', 'B'])
        assert mat.shape == (2, 2)
        assert mat.loc['A', 'B'] > 0  # Different distributions
        assert mat.loc['A', 'A'] == 0  # Same class
        assert pval.loc['A', 'B'] < 0.05  # Should be significant


class TestCeilingV2:
    def test_groupkfold_no_video_leakage(self):
        from analysis.layer2_signals.l2_ceiling_estimation_v2 import evaluate_ceiling_grouped

        rng = np.random.RandomState(42)
        n = 200
        X = rng.randn(n, 5)
        y = np.array(['A'] * 100 + ['B'] * 100)
        groups = np.array([f'v{i // 10}' for i in range(n)])  # 20 groups

        results = evaluate_ceiling_grouped(X, y, groups, ['A', 'B'], n_splits=5)
        assert 'KNN-5' in results
        assert 'ci_lower' in results['KNN-5']
        assert 'ci_upper' in results['KNN-5']
        assert results['KNN-5']['ci_lower'] <= results['KNN-5']['macro_f1_mean']
        assert results['KNN-5']['macro_f1_mean'] <= results['KNN-5']['ci_upper']
