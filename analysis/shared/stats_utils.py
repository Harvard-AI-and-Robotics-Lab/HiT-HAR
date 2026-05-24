# analysis/shared/stats_utils.py
"""Statistical utilities for rigorous empirical analysis.

Provides nonparametric separability metrics, significance testing,
normality checks, and confidence interval estimation.

References:
  - MMD: Gretton et al. (2012), JMLR 13
  - Mardia normality: Mardia (1970), Biometrika 57
  - Bootstrap CI: Efron & Tibshirani (1993)
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import chi2, norm


def _median_bandwidth(X, Y):
    """Compute median heuristic bandwidth for RBF kernel."""
    # Subsample for speed if large
    n = min(500, len(X), len(Y))
    rng = np.random.RandomState(0)
    Xs = X[rng.choice(len(X), n, replace=False)]
    Ys = Y[rng.choice(len(Y), n, replace=False)]
    dists = cdist(Xs, Ys, metric='sqeuclidean').flatten()
    median_dist = np.median(dists)
    return max(median_dist, 1e-10)


def compute_mmd(X, Y, bandwidth=None):
    """Compute Maximum Mean Discrepancy with RBF kernel.

    MMD²(P,Q) = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]

    Args:
        X: (n, d) samples from P
        Y: (m, d) samples from Q
        bandwidth: RBF bandwidth σ². If None, uses median heuristic.
    Returns:
        float: MMD² (unbiased estimator)
    """
    if bandwidth is None:
        bandwidth = _median_bandwidth(X, Y)

    def rbf(A, B):
        return np.exp(-cdist(A, B, 'sqeuclidean') / (2 * bandwidth))

    n, m = len(X), len(Y)

    Kxx = rbf(X, X)
    Kyy = rbf(Y, Y)
    Kxy = rbf(X, Y)

    # Unbiased estimator (exclude diagonal for Kxx, Kyy)
    np.fill_diagonal(Kxx, 0)
    np.fill_diagonal(Kyy, 0)

    mmd_sq = (Kxx.sum() / (n * (n - 1))
              - 2 * Kxy.sum() / (n * m)
              + Kyy.sum() / (m * (m - 1)))

    return float(mmd_sq)


def permutation_test_mmd(X, Y, n_permutations=1000, bandwidth=None, seed=42):
    """Permutation test for MMD: H0 = same distribution.

    Args:
        X: (n, d) samples from class 1
        Y: (m, d) samples from class 2
        n_permutations: number of permutations
        bandwidth: RBF bandwidth. If None, computed once from data.
        seed: random seed
    Returns:
        (mmd_observed, p_value)
    """
    if bandwidth is None:
        bandwidth = _median_bandwidth(X, Y)

    observed_mmd = compute_mmd(X, Y, bandwidth=bandwidth)

    combined = np.vstack([X, Y])
    n = len(X)
    rng = np.random.RandomState(seed)

    null_mmds = np.zeros(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(len(combined))
        X_perm = combined[perm[:n]]
        Y_perm = combined[perm[n:]]
        null_mmds[i] = compute_mmd(X_perm, Y_perm, bandwidth=bandwidth)

    p_value = (np.sum(null_mmds >= observed_mmd) + 1) / (n_permutations + 1)
    return float(observed_mmd), float(p_value)


def mardia_normality_test(X):
    """Mardia's multivariate normality test (skewness + kurtosis).

    Args:
        X: (n, p) data matrix
    Returns:
        dict with skewness_stat, skewness_pvalue, kurtosis_stat, kurtosis_pvalue, is_normal
    """
    n, p = X.shape
    X_centered = X - X.mean(axis=0)
    S = np.cov(X_centered, rowvar=False)

    try:
        S_inv = np.linalg.pinv(S)
    except np.linalg.LinAlgError:
        return {'skewness_stat': np.nan, 'skewness_pvalue': 0,
                'kurtosis_stat': np.nan, 'kurtosis_pvalue': 0, 'is_normal': False}

    # D matrix: D_ij = (x_i - mu)^T S^{-1} (x_j - mu)
    D = X_centered @ S_inv @ X_centered.T

    # Skewness: b1p = (1/n²) Σ_ij D_ij³
    b1p = (D ** 3).sum() / (n * n)
    skewness_stat = n * b1p / 6
    skewness_df = p * (p + 1) * (p + 2) / 6
    skewness_pvalue = 1 - chi2.cdf(skewness_stat, skewness_df)

    # Kurtosis: b2p = (1/n) Σ_i D_ii²
    b2p = (np.diag(D) ** 2).sum() / n
    kurtosis_expected = p * (p + 2)
    kurtosis_var = 8 * p * (p + 2) / n
    kurtosis_stat = (b2p - kurtosis_expected) / np.sqrt(max(kurtosis_var, 1e-10))
    kurtosis_pvalue = 2 * (1 - norm.cdf(abs(kurtosis_stat)))

    is_normal = (skewness_pvalue > 0.05) and (kurtosis_pvalue > 0.05)

    return {
        'skewness_stat': float(skewness_stat),
        'skewness_pvalue': float(skewness_pvalue),
        'kurtosis_stat': float(kurtosis_stat),
        'kurtosis_pvalue': float(kurtosis_pvalue),
        'is_normal': is_normal,
    }


def bootstrap_ci(values, confidence=0.95, n_bootstrap=1000, seed=42):
    """Bootstrap percentile confidence interval.

    Args:
        values: 1D array of observations (e.g., fold-level F1 scores)
        confidence: confidence level (default 0.95)
        n_bootstrap: number of bootstrap resamples
        seed: random seed
    Returns:
        (lower, upper) confidence interval
    """
    values = np.asarray(values)
    rng = np.random.RandomState(seed)
    boot_means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_means[i] = np.mean(sample)

    alpha = 1 - confidence
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def compute_mmd_global_bandwidth(X, Y, bandwidth):
    """Compute MMD with a pre-specified global bandwidth (not per-pair median)."""
    return compute_mmd(X, Y, bandwidth=bandwidth)


def global_median_bandwidth(all_data):
    """Compute a single global bandwidth from all data points.

    Use this once, then pass to all pairwise MMD computations for comparability.
    """
    n = min(2000, len(all_data))
    rng = np.random.RandomState(0)
    sample = all_data[rng.choice(len(all_data), n, replace=False)]
    dists = cdist(sample, sample, metric='sqeuclidean')
    # Use upper triangle only
    upper = dists[np.triu_indices(n, k=1)]
    return max(float(np.median(upper)), 1e-10)


def permutation_test_mmd_grouped(X, Y, groups_x, groups_y,
                                  n_permutations=500, bandwidth=None, seed=42):
    """Group-aware permutation test for MMD.

    Instead of shuffling individual samples, shuffles entire video groups
    between the two classes to preserve within-group correlation.

    Args:
        X: (n, d) samples from class 1
        Y: (m, d) samples from class 2
        groups_x: (n,) group labels for X (e.g., video_uid)
        groups_y: (m,) group labels for Y
        n_permutations: number of permutations
        bandwidth: RBF bandwidth. If None, uses median heuristic.
        seed: random seed
    Returns:
        (mmd_observed, p_value)
    """
    if bandwidth is None:
        bandwidth = _median_bandwidth(X, Y)

    observed_mmd = compute_mmd(X, Y, bandwidth=bandwidth)

    # Build group->indices mapping for combined data
    combined = np.vstack([X, Y])
    all_groups = np.concatenate([groups_x, groups_y])
    unique_groups = np.unique(all_groups)

    # Map each group to its sample indices
    group_to_idx = {}
    for g in unique_groups:
        group_to_idx[g] = np.where(all_groups == g)[0]

    n_x_groups = len(np.unique(groups_x))
    rng = np.random.RandomState(seed)

    null_mmds = np.zeros(n_permutations)
    for i in range(n_permutations):
        # Shuffle group assignments
        perm_groups = rng.permutation(unique_groups)
        x_groups = perm_groups[:n_x_groups]
        y_groups = perm_groups[n_x_groups:]

        x_idx = np.concatenate([group_to_idx[g] for g in x_groups])
        y_idx = np.concatenate([group_to_idx[g] for g in y_groups])

        if len(x_idx) < 5 or len(y_idx) < 5:
            null_mmds[i] = 0
            continue

        null_mmds[i] = compute_mmd(combined[x_idx], combined[y_idx],
                                    bandwidth=bandwidth)

    p_value = (np.sum(null_mmds >= observed_mmd) + 1) / (n_permutations + 1)
    return float(observed_mmd), float(p_value)


def bonferroni_correct(p_values):
    """Apply Bonferroni correction to a list of p-values.

    Args:
        p_values: list or array of raw p-values
    Returns:
        list of corrected p-values (capped at 1.0)
    """
    n = len(p_values)
    return [min(p * n, 1.0) for p in p_values]
