# src/training/language_loss.py
"""Language alignment losses for LanIMU.

L_language = w_proto * L_proto + w_inst * L_inst + w_vicreg * L_vicreg

- L_proto: CLS embedding vs class text prototypes (cross-entropy on cosine sim)
- L_inst: per-window embedding vs narration text embedding (confidence-weighted)
- L_vicreg: variance + covariance anti-collapse regularizer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeContrastiveLoss(nn.Module):
    """Cross-entropy on cosine similarity between embeddings and class prototypes.

    For each sequence, the CLS embedding should be close to the prototype of its
    dominant action class and far from other prototypes.
    """

    def __init__(self, embedding_dim=128, num_classes=5, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.num_classes = num_classes

    def forward(self, h_cls, prototypes, action_labels):
        """
        Args:
            h_cls: (B, D) — L2-normalized sequence embeddings
            prototypes: (num_classes, D) — L2-normalized text prototypes
            action_labels: (B,) — dominant action class per sequence [0..4]

        Returns:
            scalar loss
        """
        # Cosine similarity: (B, num_classes)
        sim = torch.mm(h_cls, prototypes.t()) / self.temperature

        # Filter out invalid labels
        valid = (action_labels >= 0) & (action_labels < self.num_classes)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=h_cls.device, requires_grad=True)

        sim = sim[valid]
        labels = action_labels[valid]

        return F.cross_entropy(sim, labels)


def vicreg_regularizer(embeddings, var_weight=1.0, cov_weight=0.04):
    """VICReg variance + covariance regularizer to prevent embedding collapse.

    Args:
        embeddings: (N, D) — batch of embeddings
        var_weight: weight for variance term
        cov_weight: weight for covariance term

    Returns:
        scalar regularization loss
    """
    N, D = embeddings.shape
    if N < 2:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    # Variance: encourage std > 1 per dimension
    std = embeddings.std(dim=0)
    var_loss = F.relu(1.0 - std).mean()

    # Covariance: penalize off-diagonal correlations
    embeddings_centered = embeddings - embeddings.mean(dim=0)
    cov = (embeddings_centered.T @ embeddings_centered) / (N - 1)
    # Zero out diagonal
    off_diag = cov - torch.diag(cov.diag())
    cov_loss = (off_diag ** 2).sum() / D

    return var_weight * var_loss + cov_weight * cov_loss


class LanguageAlignmentLoss(nn.Module):
    """Language alignment loss — backward compatible.

    In dual-space mode: uses z_lang, drops VICReg.
    In legacy mode: uses h_cls with optional VICReg (unchanged behavior).
    """

    def __init__(self, embedding_dim=128, num_classes=5,
                 temperature=0.1, w_proto=0.6, w_vicreg=0.1,
                 lang_subspace_dim=None):
        super().__init__()
        proto_dim = lang_subspace_dim if lang_subspace_dim else embedding_dim
        self.proto_loss = PrototypeContrastiveLoss(
            embedding_dim=proto_dim,
            num_classes=num_classes,
            temperature=temperature,
        )
        self.w_proto = w_proto
        self.w_vicreg = w_vicreg
        self.use_dual_space = lang_subspace_dim is not None

    def forward(self, model_output, prototypes, batch):
        """Auto-detects dual-space (z_lang present) vs legacy (h_cls only)."""
        # Use z_lang if dual-space, else fall back to h_cls (backward compat)
        if 'z_lang' in model_output:
            embeddings = F.normalize(model_output['z_lang'], dim=1)
        else:
            embeddings = F.normalize(model_output['h_cls'], dim=1)

        device = embeddings.device

        # Dominant action per sequence (mode of valid labels)
        action_labels = batch['action_labels'].to(device)
        dominant_actions = []
        for i in range(action_labels.shape[0]):
            valid = action_labels[i][action_labels[i] >= 0]
            if len(valid) > 0:
                dominant_actions.append(valid.mode().values.item())
            else:
                dominant_actions.append(-1)
        dominant_actions = torch.tensor(dominant_actions, device=device)

        # L_proto
        loss_proto = self.proto_loss(embeddings, prototypes, dominant_actions)

        loss_dict = {
            'loss_lang_proto': loss_proto.item(),
        }

        if self.use_dual_space:
            # Dual-space mode: no VICReg (confirmed dead weight)
            total = loss_proto
        else:
            # Legacy mode: keep VICReg for backward compatibility
            loss_vicreg = vicreg_regularizer(embeddings)
            total = self.w_proto * loss_proto + self.w_vicreg * loss_vicreg
            loss_dict['loss_lang_vicreg'] = loss_vicreg.item()

        loss_dict['loss_language'] = total.item()
        return total, loss_dict
