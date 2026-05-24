"""Tests for class-wise scenario-conditioned gate and coarse auxiliary head (v3 features)."""
import torch
import pytest
import sys
import os

# Ensure project root is on sys.path so 'src.models' resolves.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.hit_har import (
    ContextAwareActionHead, HiTHAR, build_model,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(class_wise_gate=False, coarse_head=False, **overrides):
    """Return a minimal flat config dict for HiTHAR."""
    cfg = {
        'in_channels': 8,
        'embedding_dim': 128,
        'num_actions': 5,
        'num_scenarios': 8,
        'seq_len': 10,  # short for fast tests
        'hla_layers': 1,
        'hla_nhead': 4,
        'hla_ff': 256,
        'hla_dropout': 0.0,
        'lle_dropout': 0.0,
        'class_wise_gate': class_wise_gate,
        'coarse_head': coarse_head,
    }
    cfg.update(overrides)
    return cfg


def _make_nested_config(class_wise_gate=False, coarse_head=False):
    """Return a nested YAML-style config for build_model()."""
    return {
        'lle': {'in_channels': 8, 'embedding_dim': 128, 'dropout': 0.0},
        'hla': {
            'num_classes': 8, 'seq_len': 10, 'num_layers': 1,
            'nhead': 4, 'dim_feedforward': 256, 'dropout': 0.0,
        },
        'num_actions': 5,
        'class_wise_gate': class_wise_gate,
        'coarse_head': coarse_head,
    }


B, S, W, C = 2, 10, 50, 8  # batch, seq_len, window, channels
D = 128
NUM_CLASSES = 5
NUM_SCENARIOS = 8


# ---------------------------------------------------------------------------
# 1. Class-wise gate tests
# ---------------------------------------------------------------------------

class TestClassWiseGate:

    def test_gate_shape_is_per_class(self):
        """class_wise_gate=True -> gate blending should produce (B, S, C) output."""
        model = HiTHAR(_make_config(class_wise_gate=True))
        model.eval()
        x = torch.randn(B, S, W, C)
        with torch.no_grad():
            result = model(x)
        logits = result['action_logits']
        assert logits.shape == (B, S, NUM_CLASSES), (
            f"Expected (B,S,C)=({B},{S},{NUM_CLASSES}), got {logits.shape}"
        )

    def test_class_wise_gate_head_standalone(self):
        """Directly verify the ContextAwareActionHead produces per-class gate."""
        head = ContextAwareActionHead(
            embedding_dim=D, num_classes=NUM_CLASSES,
            class_wise_gate=True, num_scenarios=NUM_SCENARIOS,
        )
        head.eval()
        e_t = torch.randn(B, S, D)
        h_t = torch.randn(B, S, D)
        h_cls = torch.randn(B, D)
        scenario_logits = torch.randn(B, NUM_SCENARIOS)
        with torch.no_grad():
            out = head(e_t, h_t, h_cls=h_cls, scenario_logits=scenario_logits)
        assert out.shape == (B, S, NUM_CLASSES)

    def test_backward_compatible_scalar_gate(self):
        """class_wise_gate=False (default) -> identical to original scalar gate."""
        model = HiTHAR(_make_config(class_wise_gate=False))
        model.eval()
        x = torch.randn(B, S, W, C)
        with torch.no_grad():
            result = model(x)
        logits = result['action_logits']
        assert logits.shape == (B, S, NUM_CLASSES)
        # Gate MLP final layer should output 1, not num_classes
        gate_last = model.action_head.gate[-2]  # Linear before Sigmoid
        assert gate_last.out_features == 1, (
            f"Scalar gate should output 1, got {gate_last.out_features}"
        )

    def test_scalar_gate_ignores_extra_args(self):
        """Scalar-gate forward should work even if h_cls/scenario_logits are passed."""
        head = ContextAwareActionHead(
            embedding_dim=D, num_classes=NUM_CLASSES,
            class_wise_gate=False,
        )
        head.eval()
        e_t = torch.randn(B, S, D)
        h_t = torch.randn(B, S, D)
        with torch.no_grad():
            out = head(e_t, h_t, h_cls=torch.randn(B, D),
                       scenario_logits=torch.randn(B, NUM_SCENARIOS))
        assert out.shape == (B, S, NUM_CLASSES)


# ---------------------------------------------------------------------------
# 2. Coarse auxiliary head tests
# ---------------------------------------------------------------------------

class TestCoarseHead:

    def test_coarse_output_shape(self):
        """coarse_head=True -> result has 'coarse_logits' with shape (B, S, 3)."""
        model = HiTHAR(_make_config(coarse_head=True))
        model.eval()
        x = torch.randn(B, S, W, C)
        with torch.no_grad():
            result = model(x)
        assert 'coarse_logits' in result, "Missing 'coarse_logits' key"
        assert result['coarse_logits'].shape == (B, S, 3), (
            f"Expected (B,S,3)=({B},{S},3), got {result['coarse_logits'].shape}"
        )

    def test_coarse_fine_to_coarse_mapping(self):
        """fine_to_coarse buffer should have correct mapping."""
        model = HiTHAR(_make_config(coarse_head=True))
        expected = torch.tensor([0, 0, 1, 2, 1])
        assert 'fine_to_coarse' in {n for n, _ in model.named_buffers()}, (
            "fine_to_coarse should be a registered buffer"
        )
        assert torch.equal(model.fine_to_coarse, expected), (
            f"Mapping mismatch: {model.fine_to_coarse} != {expected}"
        )

    def test_coarse_absent_when_disabled(self):
        """coarse_head=False -> no coarse_logits in result."""
        model = HiTHAR(_make_config(coarse_head=False))
        model.eval()
        x = torch.randn(B, S, W, C)
        with torch.no_grad():
            result = model(x)
        assert 'coarse_logits' not in result
        assert model.coarse_head is None


# ---------------------------------------------------------------------------
# 3. build_model() config forwarding
# ---------------------------------------------------------------------------

class TestBuildModel:

    def test_build_model_forwards_class_wise_gate(self):
        """build_model with class_wise_gate=True should create the right head."""
        config = _make_nested_config(class_wise_gate=True)
        model = build_model(config)
        assert model.class_wise_gate is True
        assert model.action_head.class_wise_gate is True
        # Gate final linear should output num_classes=5
        gate_last = model.action_head.gate[-2]
        assert gate_last.out_features == NUM_CLASSES, (
            f"Class-wise gate should output {NUM_CLASSES}, got {gate_last.out_features}"
        )

    def test_build_model_forwards_coarse_head(self):
        """build_model with coarse_head=True should create coarse Linear(D, 3)."""
        config = _make_nested_config(coarse_head=True)
        model = build_model(config)
        assert model.coarse_head is not None
        assert model.coarse_head.out_features == 3

    def test_build_model_defaults_off(self):
        """build_model with no explicit flags should leave both features off."""
        config = _make_nested_config()
        model = build_model(config)
        assert model.class_wise_gate is False
        assert model.coarse_head is None

    def test_build_model_both_features(self):
        """build_model with both features produces correct forward pass."""
        config = _make_nested_config(class_wise_gate=True, coarse_head=True)
        model = build_model(config)
        model.eval()
        x = torch.randn(B, S, W, C)
        with torch.no_grad():
            result = model(x)
        assert result['action_logits'].shape == (B, S, NUM_CLASSES)
        assert result['coarse_logits'].shape == (B, S, 3)
        assert torch.equal(result['fine_to_coarse'], torch.tensor([0, 0, 1, 2, 1]))
