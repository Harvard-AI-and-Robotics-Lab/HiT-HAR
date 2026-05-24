# tests/test_language_loss.py
"""Tests for language alignment components."""
import torch
import torch.nn.functional as F
import pytest


class TestTextEncoder:
    def test_class_prototypes_shape(self):
        from src.models.text_encoder import TextAnchorEncoder
        encoder = TextAnchorEncoder(projection_dim=128)
        prototypes = encoder.get_class_prototypes()
        assert prototypes.shape == (5, 128), f"Expected (5, 128), got {prototypes.shape}"

    def test_prototypes_normalized(self):
        from src.models.text_encoder import TextAnchorEncoder
        encoder = TextAnchorEncoder(projection_dim=128)
        prototypes = encoder.get_class_prototypes()
        norms = torch.norm(prototypes, dim=1)
        assert torch.allclose(norms, torch.ones(5), atol=0.01), \
            f"Prototypes not L2-normalized: norms={norms}"

    def test_narration_encoding(self):
        from src.models.text_encoder import TextAnchorEncoder
        encoder = TextAnchorEncoder(projection_dim=128)
        texts = ["C picks up the cup", "C walks down the hallway"]
        embeds = encoder.encode_narrations(texts)
        assert embeds.shape == (2, 128)

    def test_projection_trainable(self):
        from src.models.text_encoder import TextAnchorEncoder
        encoder = TextAnchorEncoder(projection_dim=128)
        # Text encoder should be frozen
        for p in encoder.text_model.parameters():
            assert not p.requires_grad
        # Projection should be trainable
        for p in encoder.projection.parameters():
            assert p.requires_grad


class TestLanguageLoss:
    def test_prototype_loss_shape(self):
        from src.training.language_loss import PrototypeContrastiveLoss
        loss_fn = PrototypeContrastiveLoss(embedding_dim=128, num_classes=5)
        # Simulate: 4 sequences, each with CLS embedding
        h_cls = torch.randn(4, 128)
        h_cls = F.normalize(h_cls, dim=1)
        # Prototypes
        prototypes = torch.randn(5, 128)
        prototypes = F.normalize(prototypes, dim=1)
        # Action labels for the dominant action per sequence
        labels = torch.tensor([0, 2, 1, 4])

        loss = loss_fn(h_cls, prototypes, labels)
        assert loss.shape == (), f"Expected scalar, got {loss.shape}"
        assert loss.item() > 0

    def test_vicreg_prevents_collapse(self):
        from src.training.language_loss import vicreg_regularizer
        # Collapsed embeddings (all same)
        collapsed = torch.ones(32, 128)
        loss_collapsed = vicreg_regularizer(collapsed)
        # Diverse embeddings
        diverse = torch.randn(32, 128)
        loss_diverse = vicreg_regularizer(diverse)
        assert loss_collapsed > loss_diverse, \
            "VICReg should penalize collapsed embeddings more"
