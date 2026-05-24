"""Tests for learnable beta modes in HiTHARLoss."""

import pytest
import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.losses_hit_har import HiTHARLoss


def _make_batch(B=4, S=10, num_actions=5, num_scenarios=8, device='cpu'):
    """Create a fake batch for testing."""
    return {
        'scenario_label': torch.randint(0, num_scenarios, (B,)),
        'action_labels': torch.randint(0, num_actions, (B, S)),
    }


def _make_model_output(B=4, S=10, num_actions=5, num_scenarios=8, device='cpu'):
    """Create fake model output for testing."""
    return {
        'scenario_logits': torch.randn(B, num_scenarios, requires_grad=True),
        'action_logits': torch.randn(B, S, num_actions, requires_grad=True),
    }


class TestFixedBetaBackwardCompat:
    """Test that fixed beta mode is unchanged."""

    def test_fixed_beta_default(self):
        """Default config: fixed beta = 0.7, no learnable params."""
        config = {'beta_task': 0.7, 'num_actions': 5}
        loss_fn = HiTHARLoss(config)
        assert loss_fn.beta_task == 0.7
        assert not loss_fn.learnable_beta
        assert not loss_fn.per_class_beta
        # No learnable parameters in loss
        assert len(list(loss_fn.parameters())) == 0

    def test_fixed_beta_forward(self):
        """Fixed beta produces correct loss and no beta in loss_dict."""
        config = {'beta_task': 1.0, 'num_actions': 5}
        loss_fn = HiTHARLoss(config)
        model_out = _make_model_output()
        batch = _make_batch()
        total_loss, loss_dict = loss_fn(model_out, batch)
        assert 'total_loss' in loss_dict
        assert 'loss_scenario' in loss_dict
        assert 'loss_action' in loss_dict
        assert 'beta_value' not in loss_dict
        assert 'beta_class_0' not in loss_dict

    def test_fixed_beta_no_grad_on_beta(self):
        """Fixed beta: no gradient flows through beta (it's a plain float)."""
        config = {'beta_task': 0.5, 'num_actions': 5}
        loss_fn = HiTHARLoss(config)
        model_out = _make_model_output()
        batch = _make_batch()
        total_loss, _ = loss_fn(model_out, batch)
        total_loss.backward()
        # No parameters in the loss module
        assert len(list(loss_fn.parameters())) == 0


class TestGlobalLearnableBeta:
    """Test global learnable beta mode."""

    def test_learnable_beta_creates_parameter(self):
        """learnable_beta=True creates a single nn.Parameter."""
        config = {'learnable_beta': True, 'beta_task': 0.5, 'num_actions': 5}
        loss_fn = HiTHARLoss(config)
        params = list(loss_fn.parameters())
        assert len(params) == 1
        assert params[0].shape == ()  # scalar

    def test_learnable_beta_initial_value(self):
        """Initial sigmoid(param) should equal the configured beta_task."""
        config = {'learnable_beta': True, 'beta_task': 0.3, 'num_actions': 5}
        loss_fn = HiTHARLoss(config)
        beta = torch.sigmoid(loss_fn.beta_task_param)
        assert abs(beta.item() - 0.3) < 0.01

    def test_learnable_beta_logs_value(self):
        """loss_dict should contain 'beta_value'."""
        config = {'learnable_beta': True, 'beta_task': 0.5, 'num_actions': 5}
        loss_fn = HiTHARLoss(config)
        model_out = _make_model_output()
        batch = _make_batch()
        _, loss_dict = loss_fn(model_out, batch)
        assert 'beta_value' in loss_dict
        assert 0.0 <= loss_dict['beta_value'] <= 1.0

    def test_learnable_beta_gradient_flows(self):
        """Gradient should flow to beta_task_param."""
        config = {'learnable_beta': True, 'beta_task': 0.5, 'num_actions': 5}
        loss_fn = HiTHARLoss(config)
        model_out = _make_model_output()
        batch = _make_batch()
        total_loss, _ = loss_fn(model_out, batch)
        total_loss.backward()
        assert loss_fn.beta_task_param.grad is not None
        assert loss_fn.beta_task_param.grad.abs() > 0

    def test_learnable_beta_constrained_to_01(self):
        """Beta value should always be in [0, 1] regardless of param value."""
        config = {'learnable_beta': True, 'beta_task': 0.5, 'num_actions': 5}
        loss_fn = HiTHARLoss(config)
        # Force extreme parameter values
        with torch.no_grad():
            loss_fn.beta_task_param.fill_(100.0)
        model_out = _make_model_output()
        batch = _make_batch()
        _, loss_dict = loss_fn(model_out, batch)
        assert 0.0 <= loss_dict['beta_value'] <= 1.0

        with torch.no_grad():
            loss_fn.beta_task_param.fill_(-100.0)
        _, loss_dict = loss_fn(model_out, batch)
        assert 0.0 <= loss_dict['beta_value'] <= 1.0


class TestPerClassLearnableBeta:
    """Test per-class learnable beta mode."""

    def test_per_class_creates_parameters(self):
        """per_class_beta=True creates parameter of shape (num_actions,)."""
        config = {'per_class_beta': True, 'beta_task': 0.5, 'num_actions': 5}
        loss_fn = HiTHARLoss(config)
        params = list(loss_fn.parameters())
        assert len(params) == 1
        assert params[0].shape == (5,)

    def test_per_class_4class(self):
        """4-class variant creates 4 beta parameters."""
        config = {'per_class_beta': True, 'beta_task': 0.5, 'num_actions': 4}
        loss_fn = HiTHARLoss(config)
        assert loss_fn.beta_per_class.shape == (4,)

    def test_per_class_initial_values(self):
        """Initial sigmoid(params) should all equal configured beta_task."""
        config = {'per_class_beta': True, 'beta_task': 0.7, 'num_actions': 5}
        loss_fn = HiTHARLoss(config)
        betas = torch.sigmoid(loss_fn.beta_per_class)
        for b in betas:
            assert abs(b.item() - 0.7) < 0.01

    def test_per_class_logs_all_betas(self):
        """loss_dict should contain beta_class_0 through beta_class_4."""
        config = {'per_class_beta': True, 'beta_task': 0.5, 'num_actions': 5}
        loss_fn = HiTHARLoss(config)
        model_out = _make_model_output()
        batch = _make_batch()
        _, loss_dict = loss_fn(model_out, batch)
        for c in range(5):
            assert f'beta_class_{c}' in loss_dict
            assert 0.0 <= loss_dict[f'beta_class_{c}'] <= 1.0
        assert 'beta_mean' in loss_dict

    def test_per_class_gradient_flows(self):
        """Gradient should flow to all per-class beta parameters."""
        config = {'per_class_beta': True, 'beta_task': 0.5, 'num_actions': 5}
        loss_fn = HiTHARLoss(config)
        model_out = _make_model_output()
        batch = _make_batch()
        total_loss, _ = loss_fn(model_out, batch)
        total_loss.backward()
        assert loss_fn.beta_per_class.grad is not None
        # At least some classes should have nonzero gradient
        assert (loss_fn.beta_per_class.grad.abs() > 0).any()

    def test_per_class_different_betas_different_loss(self):
        """Manually setting different per-class betas should change loss."""
        config = {'per_class_beta': True, 'beta_task': 0.5, 'num_actions': 5}
        loss_fn = HiTHARLoss(config)
        model_out = _make_model_output()
        batch = _make_batch()

        torch.manual_seed(42)
        _, loss_dict_1 = loss_fn(model_out, batch)
        loss_1 = loss_dict_1['total_loss']

        # Change betas drastically
        with torch.no_grad():
            loss_fn.beta_per_class[0] = 5.0   # sigmoid ~1.0
            loss_fn.beta_per_class[1] = -5.0  # sigmoid ~0.0

        _, loss_dict_2 = loss_fn(model_out, batch)
        loss_2 = loss_dict_2['total_loss']

        # Losses should differ after changing betas
        assert abs(loss_1 - loss_2) > 1e-4


class TestMutualExclusion:
    """Test that learnable_beta and per_class_beta are handled correctly."""

    def test_per_class_takes_priority(self):
        """If both are set, per_class_beta should be used (it's checked first)."""
        config = {'learnable_beta': True, 'per_class_beta': True,
                  'beta_task': 0.5, 'num_actions': 5}
        loss_fn = HiTHARLoss(config)
        assert hasattr(loss_fn, 'beta_per_class')
        assert not hasattr(loss_fn, 'beta_task_param')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
