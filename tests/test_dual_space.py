import torch
import torch.nn.functional as F
import pytest

def test_orthogonality_loss_near_zero_for_orthogonal():
    """Orthogonality loss should be near 0 for actually orthogonal subspaces."""
    from src.training.orthogonality_loss import orthogonality_loss
    B, D = 128, 64
    # QR on (B, 2D) gives Q with orthonormal columns; split into two (B, D) blocks
    M = torch.randn(B, D * 2)
    Q, _ = torch.linalg.qr(M)
    z_lang = Q[:, :D].clone().requires_grad_(True)    # (B, D)
    z_sensor = Q[:, D:].clone().requires_grad_(True)  # (B, D)
    loss = orthogonality_loss(z_lang, z_sensor)
    assert loss.shape == ()
    assert loss.item() < 0.01, f"Loss should be near zero for orthogonal inputs, got {loss.item()}"
    assert loss.requires_grad

def test_orthogonality_loss_high_for_identical():
    """Orthogonality loss should be high when subspaces are identical."""
    from src.training.orthogonality_loss import orthogonality_loss
    B = 32
    z = torch.randn(B, 64)
    loss_same = orthogonality_loss(z, z.clone())
    loss_diff = orthogonality_loss(z, torch.randn(B, 64))
    assert loss_same > loss_diff

def test_variance_penalty():
    """Variance penalty should penalize collapsed dimensions."""
    from src.training.orthogonality_loss import variance_penalty
    z_collapsed = torch.ones(32, 64) * 0.5
    z_healthy = torch.randn(32, 64)
    loss_collapsed = variance_penalty(z_collapsed)
    loss_healthy = variance_penalty(z_healthy)
    assert loss_collapsed > loss_healthy

def test_alpha_entropy_regularizer():
    """Entropy regularizer should prevent alpha collapse."""
    from src.training.orthogonality_loss import alpha_entropy_regularizer
    alpha_balanced = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])
    alpha_collapsed = torch.tensor([0.01, 0.01, 0.01, 0.01, 0.01])
    loss_balanced = alpha_entropy_regularizer(alpha_balanced)
    loss_collapsed = alpha_entropy_regularizer(alpha_collapsed)
    assert loss_collapsed > loss_balanced


def _make_dualspace_config():
    """Minimal config for testing dual-space model."""
    return {
        'lle': {'in_channels': 8, 'embedding_dim': 128, 'dropout': 0.2},
        'hla': {'num_layers': 1, 'num_classes': 8, 'seq_len': 10,
                'nhead': 4, 'dim_feedforward': 256, 'dropout': 0.1},
        'num_actions': 5,
        'class_wise_gate': True,
        'coarse_head': False,
        'multiscale': False,
        'language_distill': True,
        'dual_space': True,
        'lang_subspace_dim': 64,
        'sensor_subspace_dim': 64,
        'observability_gate': False,
    }

def test_dualspace_forward_output_keys():
    """DualSpace model should output z_lang, z_sensor, z_combined."""
    from src.models.hit_har import build_model
    config = _make_dualspace_config()
    model = build_model(config)
    x = torch.randn(2, 10, 50, 8)
    with torch.no_grad():
        out = model(x)
    assert 'z_lang' in out, "Missing z_lang in model output"
    assert 'z_sensor' in out, "Missing z_sensor in model output"
    assert 'z_combined' in out, "Missing z_combined in model output"
    assert out['z_lang'].shape == (2, 64), f"z_lang shape wrong: {out['z_lang'].shape}"
    assert out['z_sensor'].shape == (2, 64), f"z_sensor shape wrong: {out['z_sensor'].shape}"
    assert out['z_combined'].shape == (2, 128), f"z_combined shape wrong: {out['z_combined'].shape}"

def test_dualspace_scenario_uses_h_cls():
    """Scenario head should use h_cls (pre-split), not z_combined."""
    from src.models.hit_har import build_model
    config = _make_dualspace_config()
    model = build_model(config)
    x = torch.randn(2, 10, 50, 8)
    with torch.no_grad():
        out = model(x)
    assert out['scenario_logits'].shape == (2, 8)

def test_dualspace_text_prototypes_dim():
    """Text prototypes should match lang_subspace_dim (64), not embedding_dim (128)."""
    from src.models.hit_har import build_model
    config = _make_dualspace_config()
    model = build_model(config)
    x = torch.randn(2, 10, 50, 8)
    with torch.no_grad():
        out = model(x)
    assert 'text_prototypes' in out
    assert out['text_prototypes'].shape[1] == 64, (
        f"Text prototype dim should be 64, got {out['text_prototypes'].shape[1]}"
    )

def test_dualspace_dim_assertion():
    """Should raise AssertionError if lang_dim + sensor_dim != embedding_dim."""
    from src.models.hit_har import build_model
    config = _make_dualspace_config()
    config['lang_subspace_dim'] = 80
    config['sensor_subspace_dim'] = 80  # 80+80=160 != 128
    with pytest.raises(AssertionError):
        build_model(config)

def test_observability_gate_output():
    """Model with observability_gate=True should output obs_alpha per class."""
    from src.models.hit_har import build_model
    config = _make_dualspace_config()
    config['observability_gate'] = True
    model = build_model(config)
    x = torch.randn(2, 10, 50, 8)
    with torch.no_grad():
        out = model(x)
    assert 'obs_alpha' in out, "Missing obs_alpha in output"
    assert out['obs_alpha'].shape == (5,), f"obs_alpha shape wrong: {out['obs_alpha'].shape}"
    assert (out['obs_alpha'] >= 0).all() and (out['obs_alpha'] <= 1).all()

def test_hit_har_loss_with_dual_space():
    """HiTHARLoss should compute orthogonality and variance losses when dual_space outputs present."""
    from src.training.losses_hit_har import HiTHARLoss
    config = {
        'beta_task': 1.0,
        'lambda_lang': 0.15,
        'lambda_orth': 0.05,
        'lambda_var': 0.1,
        'embedding_dim': 128,
        'num_actions': 5,
        'dual_space': True,
        'lang_subspace_dim': 64,
    }
    criterion = HiTHARLoss(config)

    model_output = {
        'scenario_logits': torch.randn(4, 8, requires_grad=True),
        'action_logits': torch.randn(4, 10, 5, requires_grad=True),
        'h_cls': torch.randn(4, 128, requires_grad=True),
        'z_lang': torch.randn(4, 64, requires_grad=True),
        'z_sensor': torch.randn(4, 64, requires_grad=True),
        'text_prototypes': F.normalize(torch.randn(5, 64), dim=1),
        'e_t': torch.randn(4, 10, 128),
    }
    batch = {
        'scenario_label': torch.randint(0, 8, (4,)),
        'action_labels': torch.randint(0, 5, (4, 10)),
    }

    loss, loss_dict = criterion(model_output, batch)
    assert 'loss_orth' in loss_dict, "Missing orthogonality loss in loss_dict"
    assert 'loss_var_lang' in loss_dict, "Missing variance penalty in loss_dict"
    assert loss.requires_grad


def test_full_training_step_dual_space():
    """Full forward + backward pass should work without errors."""
    from src.models.hit_har import build_model
    from src.training.losses_hit_har import HiTHARLoss

    config = _make_dualspace_config()
    config['multiscale'] = False  # Faster for test
    model = build_model(config)

    loss_config = {
        'beta_task': 1.0,
        'lambda_lang': 0.15,
        'lambda_orth': 0.05,
        'lambda_var': 0.1,
        'embedding_dim': 128,
        'num_actions': 5,
        'dual_space': True,
        'lang_subspace_dim': 64,
        'action_weights': [1.0, 1.0, 1.0, 1.0, 1.0],
    }
    criterion = HiTHARLoss(loss_config)

    x = torch.randn(4, 10, 50, 8)
    batch = {
        'inputs': x,
        'scenario_label': torch.randint(0, 8, (4,)),
        'action_labels': torch.randint(0, 5, (4, 10)),
    }

    model.train()
    out = model(x)
    loss, loss_dict = criterion(out, batch)

    # Backward should work
    loss.backward()

    # Check gradients flow to both projection heads
    assert model.proj_lang[0].weight.grad is not None, "No gradient to proj_lang"
    assert model.proj_sensor[0].weight.grad is not None, "No gradient to proj_sensor"

    # Check gradient also flows to HLA (backbone)
    hla_param = next(model.hla.parameters())
    assert hla_param.grad is not None, "No gradient to HLA backbone"

    print(f"Loss: {loss.item():.4f}")
    print(f"Loss dict keys: {list(loss_dict.keys())}")


def test_full_training_step_with_gate():
    """Full forward + backward with observability gate."""
    from src.models.hit_har import build_model
    from src.training.losses_hit_har import HiTHARLoss

    config = _make_dualspace_config()
    config['observability_gate'] = True
    config['multiscale'] = False
    model = build_model(config)

    loss_config = {
        'beta_task': 1.0,
        'lambda_lang': 0.15,
        'lambda_orth': 0.05,
        'lambda_var': 0.1,
        'embedding_dim': 128,
        'num_actions': 5,
        'dual_space': True,
        'lang_subspace_dim': 64,
        'action_weights': [1.0, 1.0, 1.0, 1.0, 1.0],
    }
    criterion = HiTHARLoss(loss_config)

    x = torch.randn(4, 10, 50, 8)
    batch = {
        'inputs': x,
        'scenario_label': torch.randint(0, 8, (4,)),
        'action_labels': torch.randint(0, 5, (4, 10)),
    }

    model.train()
    out = model(x)
    loss, loss_dict = criterion(out, batch)
    loss.backward()

    # Alpha should have gradient
    assert model.obs_gate_logits.grad is not None, "No gradient to obs_gate_logits"
    # Alpha values should be logged
    assert 'alpha_class_0' in loss_dict


# ---- Learnable Beta Tests ----

def test_learnable_beta_global():
    """Global learnable beta should be an nn.Parameter and appear in loss_dict."""
    from src.training.losses_hit_har import HiTHARLoss
    config = {
        'beta_task': 0.5,
        'learnable_beta': True,
        'per_class_beta': False,
        'num_actions': 5,
    }
    criterion = HiTHARLoss(config)
    # Check beta_task_param is a parameter
    params = list(criterion.parameters())
    assert len(params) >= 1, "Learnable beta should register at least one parameter"
    assert hasattr(criterion, 'beta_task_param'), "Should have beta_task_param attribute"
    assert criterion.beta_task_param.requires_grad

    # Forward pass: check beta_value is logged
    model_output = {
        'scenario_logits': torch.randn(4, 8),
        'action_logits': torch.randn(4, 10, 5),
    }
    batch = {
        'scenario_label': torch.randint(0, 8, (4,)),
        'action_labels': torch.randint(0, 5, (4, 10)),
    }
    loss, loss_dict = criterion(model_output, batch)
    assert 'beta_value' in loss_dict, "loss_dict should contain 'beta_value'"
    assert 0.0 <= loss_dict['beta_value'] <= 1.0, "beta_value should be in [0, 1]"
    assert loss.requires_grad


def test_learnable_beta_per_class():
    """Per-class learnable beta should have shape (num_actions,) and log per-class values."""
    from src.training.losses_hit_har import HiTHARLoss
    config = {
        'beta_task': 0.5,
        'learnable_beta': False,
        'per_class_beta': True,
        'num_actions': 5,
    }
    criterion = HiTHARLoss(config)
    params = list(criterion.parameters())
    assert any(p.shape == (5,) for p in params), "Per-class beta should have shape (5,)"
    assert hasattr(criterion, 'beta_per_class'), "Should have beta_per_class attribute"

    # Forward pass: check per-class betas are logged
    model_output = {
        'scenario_logits': torch.randn(4, 8),
        'action_logits': torch.randn(4, 10, 5),
    }
    batch = {
        'scenario_label': torch.randint(0, 8, (4,)),
        'action_labels': torch.randint(0, 5, (4, 10)),
    }
    loss, loss_dict = criterion(model_output, batch)
    for i in range(5):
        assert f'beta_class_{i}' in loss_dict, f"loss_dict should contain 'beta_class_{i}'"
        assert 0.0 <= loss_dict[f'beta_class_{i}'] <= 1.0
    assert 'beta_mean' in loss_dict
    assert loss.requires_grad


def test_learnable_beta_gradient_flow():
    """Gradient should flow through learnable beta to allow optimization."""
    from src.training.losses_hit_har import HiTHARLoss
    config = {
        'beta_task': 0.5,
        'learnable_beta': True,
        'per_class_beta': False,
        'num_actions': 5,
    }
    criterion = HiTHARLoss(config)

    model_output = {
        'scenario_logits': torch.randn(4, 8, requires_grad=True),
        'action_logits': torch.randn(4, 10, 5, requires_grad=True),
    }
    batch = {
        'scenario_label': torch.randint(0, 8, (4,)),
        'action_labels': torch.randint(0, 5, (4, 10)),
    }
    loss, _ = criterion(model_output, batch)
    loss.backward()
    assert criterion.beta_task_param.grad is not None, "Beta param should receive gradient"


def test_learnable_beta_per_class_gradient_flow():
    """Gradient should flow through per-class beta parameters."""
    from src.training.losses_hit_har import HiTHARLoss
    config = {
        'beta_task': 0.5,
        'per_class_beta': True,
        'num_actions': 5,
    }
    criterion = HiTHARLoss(config)

    model_output = {
        'scenario_logits': torch.randn(4, 8, requires_grad=True),
        'action_logits': torch.randn(4, 10, 5, requires_grad=True),
    }
    batch = {
        'scenario_label': torch.randint(0, 8, (4,)),
        'action_labels': torch.randint(0, 5, (4, 10)),
    }
    loss, _ = criterion(model_output, batch)
    loss.backward()
    assert criterion.beta_per_class.grad is not None, "Per-class beta should receive gradient"


def test_fixed_beta_no_learnable_params():
    """Fixed beta mode should have no learnable parameters in criterion."""
    from src.training.losses_hit_har import HiTHARLoss
    config = {
        'beta_task': 1.0,
        'learnable_beta': False,
        'per_class_beta': False,
        'num_actions': 5,
    }
    criterion = HiTHARLoss(config)
    params = [p for p in criterion.parameters() if p.requires_grad]
    assert len(params) == 0, f"Fixed beta should have 0 learnable params, got {len(params)}"
