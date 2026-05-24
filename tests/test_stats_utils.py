# tests/test_stats_utils.py
"""Tests for statistical utility functions."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.shared.stats_utils import (
    compute_mmd, permutation_test_mmd, mardia_normality_test, bootstrap_ci,
)


class TestMMD:
    def test_identical_distributions_near_zero(self):
        rng = np.random.RandomState(42)
        X = rng.randn(200, 5)
        Y = rng.randn(200, 5)
        mmd = compute_mmd(X, Y)
        # Same distribution → MMD should be small
        assert mmd < 0.5

    def test_different_distributions_positive(self):
        rng = np.random.RandomState(42)
        X = rng.randn(200, 5)
        Y = rng.randn(200, 5) + 3.0  # Shifted by 3
        mmd = compute_mmd(X, Y)
        assert mmd > 0.5

    def test_mmd_is_symmetric(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        Y = rng.randn(100, 5) + 1.0
        assert abs(compute_mmd(X, Y) - compute_mmd(Y, X)) < 1e-10


class TestPermutationTest:
    def test_same_distribution_high_pvalue(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        Y = rng.randn(100, 5)
        mmd, p = permutation_test_mmd(X, Y, n_permutations=200, seed=42)
        assert p > 0.01  # Should not reject H0

    def test_different_distribution_low_pvalue(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        Y = rng.randn(100, 5) + 3.0
        mmd, p = permutation_test_mmd(X, Y, n_permutations=200, seed=42)
        assert p < 0.05  # Should reject H0


class TestMardiaNormality:
    def test_normal_data_not_rejected(self):
        rng = np.random.RandomState(42)
        X = rng.randn(500, 3)
        result = mardia_normality_test(X)
        assert 'skewness_pvalue' in result
        assert 'kurtosis_pvalue' in result
        # Normal data: p-values should be > 0.01
        assert result['skewness_pvalue'] > 0.01

    def test_nonnormal_data_rejected(self):
        rng = np.random.RandomState(42)
        X = np.exp(rng.randn(500, 3))  # Log-normal = not normal
        result = mardia_normality_test(X)
        assert result['skewness_pvalue'] < 0.05


class TestGroupAwarePermutation:
    def test_group_permutation_shuffles_groups(self):
        from analysis.shared.stats_utils import permutation_test_mmd_grouped

        rng = np.random.RandomState(42)
        # 10 groups per class (need enough groups for permutation power)
        n_groups = 10
        n_per_group = 20
        X = np.vstack([rng.randn(n_per_group, 3) for _ in range(n_groups)])
        Y = np.vstack([rng.randn(n_per_group, 3) + 3 for _ in range(n_groups)])
        groups_x = np.repeat(np.arange(n_groups), n_per_group)
        groups_y = np.repeat(np.arange(n_groups, 2 * n_groups), n_per_group)

        mmd, p = permutation_test_mmd_grouped(X, Y, groups_x, groups_y,
                                               n_permutations=200, seed=42)
        assert p < 0.05  # Should reject

    def test_bonferroni_correction(self):
        from analysis.shared.stats_utils import bonferroni_correct

        p_values = [0.01, 0.03, 0.06, 0.001]
        corrected = bonferroni_correct(p_values)
        assert corrected[0] == pytest.approx(0.04)  # 0.01 * 4
        assert corrected[2] == pytest.approx(0.24)  # 0.06 * 4
        assert corrected[3] == pytest.approx(0.004)  # 0.001 * 4
        # Capped at 1.0
        assert all(p <= 1.0 for p in corrected)


class TestBootstrapCI:
    def test_ci_contains_mean(self):
        rng = np.random.RandomState(42)
        values = rng.randn(50) + 5.0
        lo, hi = bootstrap_ci(values, n_bootstrap=500, seed=42)
        mean = np.mean(values)
        assert lo < mean < hi

    def test_ci_width(self):
        rng = np.random.RandomState(42)
        values = rng.randn(50)
        lo, hi = bootstrap_ci(values, confidence=0.95, n_bootstrap=500, seed=42)
        assert hi - lo > 0  # CI has positive width
        assert hi - lo < 2.0  # But not absurdly wide for N(0,1)
