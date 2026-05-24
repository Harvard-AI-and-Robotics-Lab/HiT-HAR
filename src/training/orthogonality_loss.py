"""Orthogonality and collapse prevention losses for dual-space decomposition.

DSN-style Frobenius orthogonality (Bousmalis et al. 2016) + variance penalty + alpha entropy.
"""

import torch
import torch.nn.functional as F


def orthogonality_loss(z_a, z_b):
    """DSN-style Frobenius norm of cross-correlation matrix, batch-normalized.

    L_orth = ||z_a^T z_b||_F^2 / B

    Args:
        z_a: (B, D1) — first subspace embeddings
        z_b: (B, D2) — second subspace embeddings

    Returns:
        Scalar loss (>= 0). Lower means more orthogonal.
    """
    B = z_a.size(0)
    cross = torch.mm(z_a.t(), z_b)  # (D1, D2)
    return (cross ** 2).sum() / B


def variance_penalty(z, target_std=0.5):
    """Penalize dimensions with std below target to prevent collapse.

    L_var = mean(relu(target_std - std_per_dim))

    Args:
        z: (B, D) — embeddings
        target_std: minimum desired std per dimension

    Returns:
        Scalar loss.
    """
    if z.size(0) < 2:
        return torch.tensor(0.0, device=z.device, requires_grad=True)
    std = z.std(dim=0)
    return F.relu(target_std - std).mean()


def alpha_entropy_regularizer(alpha, eps=1e-7):
    """Entropy regularizer to prevent alpha collapse to 0 or 1.

    Returns NEGATIVE entropy (minimizing this maximizes entropy).

    Args:
        alpha: (num_classes,) — gate values in [0, 1]

    Returns:
        Scalar penalty (lower entropy -> higher penalty).
    """
    entropy = -(alpha * torch.log(alpha + eps) + (1 - alpha) * torch.log(1 - alpha + eps))
    return -entropy.mean()
