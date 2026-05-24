"""
HiT-HAR Model Architecture (v4 SCUQD) — Standalone IMU Pipeline.

Components:
  - LLE v4: Stem + Multi-Dilation CNN (3 blocks) + SE + BiGRU + Attention Pooling
  - HLA v4: 3-layer Pre-LN Transformer with CLS token
  - Context-Aware Action Head: gated fusion of local + contextual predictions

Per FINAL_DESIGN_CN.md §3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcite(nn.Module):
    """Channel attention via squeeze-and-excitation."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        # x: (B, C, T)
        b, c, t = x.shape
        se = x.mean(dim=2)  # (B, C)
        se = F.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se)).view(b, c, 1)
        return x * se


class MultiDilationBlock(nn.Module):
    """Parallel multi-dilation convolution block with SE attention."""

    def __init__(self, in_ch, branch_ch, kernel_size, dilations, out_ch, se_reduction=8):
        super().__init__()
        self.branches = nn.ModuleList()
        for d in dilations:
            padding = (kernel_size - 1) * d // 2
            self.branches.append(
                nn.Conv1d(in_ch, branch_ch, kernel_size, padding=padding, dilation=d)
            )
        concat_ch = branch_ch * len(dilations)
        self.se = SqueezeExcite(concat_ch, reduction=se_reduction)
        self.pointwise = nn.Conv1d(concat_ch, out_ch, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        branches = [F.gelu(b(x)) for b in self.branches]
        x = torch.cat(branches, dim=1)
        x = self.se(x)
        x = self.bn(F.gelu(self.pointwise(x)))
        return x


class AttentionPooling(nn.Module):
    """Learned attention pooling over temporal dimension."""

    def __init__(self, input_dim, attn_dim=64):
        super().__init__()
        self.W_a = nn.Linear(input_dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, h):
        # h: (B, T, D)
        scores = self.w(torch.tanh(self.W_a(h)))  # (B, T, 1)
        alpha = F.softmax(scores, dim=1)  # (B, T, 1)
        return (alpha * h).sum(dim=1)  # (B, D)


class LLEv4(nn.Module):
    """Low-Level Encoder v4.

    Architecture per FINAL_DESIGN_CN.md §3.2:
      Stem: Conv1d(8, 64, k=5) → BN → GELU
      Block 1: 3×Conv1d(64, 32, k=5, d=[1,2,4]) → Concat(96) → SE → Conv1d(96,128,k=1)
      Block 2: 3×Conv1d(128, 64, k=5, d=[1,2,4]) → Concat(192) → SE → Conv1d(192,128,k=1)
      Block 3: 3×Conv1d(128, 64, k=3, d=[1,2,4]) → Concat(192) → SE → Conv1d(192,128,k=1)
      BiGRU(128→96×2=192, 1 layer) → AttentionPooling → FC(192→128) → e_t
    """

    def __init__(self, in_channels=8, embedding_dim=128, dropout=0.2):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )

        # Multi-dilation blocks
        dilations = [1, 2, 4]
        self.block1 = MultiDilationBlock(64, 32, kernel_size=5, dilations=dilations, out_ch=128)
        self.block2 = MultiDilationBlock(128, 64, kernel_size=5, dilations=dilations, out_ch=128)
        self.block3 = MultiDilationBlock(128, 64, kernel_size=3, dilations=dilations, out_ch=128)

        # BiGRU + Attention Pooling
        self.gru = nn.GRU(128, 96, num_layers=1, batch_first=True, bidirectional=True)
        self.attn_pool = AttentionPooling(192, attn_dim=64)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(192, embedding_dim)

    def forward(self, x):
        # x: (B, 50, C) → (B, C, 50)
        x = x.transpose(1, 2)

        # CNN blocks
        x = self.stem(x)       # (B, 64, 50)
        x = self.block1(x)     # (B, 128, 50)
        x = self.block2(x)     # (B, 128, 50)
        x = self.block3(x)     # (B, 128, 50)

        # BiGRU
        x = x.transpose(1, 2)  # (B, 50, 128)
        x, _ = self.gru(x)     # (B, 50, 192)

        # Attention pooling
        x = self.attn_pool(x)  # (B, 192)
        x = self.dropout(x)

        # Project to embedding dim
        e = self.fc(x)          # (B, 128)
        return e


class HLAv4(nn.Module):
    """High-Level Aggregator v4.

    3-layer Pre-LN Transformer with CLS token.
    Returns both h_cls (sequence embedding) and h_t (per-window contextual embeddings).

    Per FINAL_DESIGN_CN.md §3.3.
    """

    def __init__(self, d_model=128, nhead=4, num_layers=3, dim_feedforward=512,
                 dropout=0.1, seq_len=30):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: (B, seq_len, d_model)
        B, S, D = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([cls_tokens, x], dim=1)           # (B, S+1, D)
        x = x + self.pos_embed[:, :S + 1, :]

        x = self.encoder(x)   # (B, S+1, D)
        x = self.norm(x)

        h_cls = x[:, 0, :]    # (B, D) — sequence-level
        h_t = x[:, 1:, :]     # (B, S, D) — per-window contextual

        return h_cls, h_t


class ContextAwareActionHead(nn.Module):
    """Gated fusion of local (LLE) and contextual (HLA) action predictions.

    Per FINAL_DESIGN_CN.md §3.4:
      a_loc = W_loc · e_t          (local from LLE)
      a_ctx = W_ctx · h_t          (contextual from HLA)
      g = sigmoid(MLP([e_t; h_t])) (gate)
      a = (1-g) · a_loc + g · a_ctx

    When class_wise_gate=True, the gate produces per-class weights (B, S, C)
    using scenario-conditioned input [e_t; h_t; h_cls_broadcast; softmax(scenario_logits)].
    When class_wise_gate=False (default), behavior is identical to original scalar gate.
    """

    def __init__(self, embedding_dim=128, num_classes=5,
                 class_wise_gate=False, num_scenarios=8):
        super().__init__()
        self.class_wise_gate = class_wise_gate
        self.num_classes = num_classes
        self.W_loc = nn.Linear(embedding_dim, num_classes)
        self.W_ctx = nn.Linear(embedding_dim, num_classes)

        if class_wise_gate:
            # Gate input: [e_t; h_t; h_cls_broadcast; softmax(scenario_logits)]
            gate_input_dim = 3 * embedding_dim + num_scenarios
            self.gate = nn.Sequential(
                nn.Linear(gate_input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes),
                nn.Sigmoid(),
            )
        else:
            # Original scalar gate: [e_t; h_t] → (B, S, 1)
            self.gate = nn.Sequential(
                nn.Linear(embedding_dim * 2, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

    def forward(self, e_t, h_t, h_cls=None, scenario_logits=None):
        """
        Args:
            e_t: (B, S, D) — LLE embeddings per window
            h_t: (B, S, D) — HLA contextual embeddings per window
            h_cls: (B, D) — HLA CLS embedding (only used when class_wise_gate=True)
            scenario_logits: (B, num_scenarios) — scenario predictions (only used when class_wise_gate=True)
        Returns:
            action_logits: (B, S, num_classes)
        """
        a_loc = self.W_loc(e_t)   # (B, S, C)
        a_ctx = self.W_ctx(h_t)   # (B, S, C)

        if self.class_wise_gate:
            B, S, D = e_t.shape
            h_cls_broadcast = h_cls.unsqueeze(1).expand(-1, S, -1)  # (B, S, D)
            scenario_probs = F.softmax(scenario_logits, dim=-1)      # (B, num_scenarios)
            scenario_broadcast = scenario_probs.unsqueeze(1).expand(-1, S, -1)  # (B, S, num_scenarios)
            gate_input = torch.cat([e_t, h_t, h_cls_broadcast, scenario_broadcast], dim=-1)
            g = self.gate(gate_input)  # (B, S, num_classes)
        else:
            g = self.gate(torch.cat([e_t, h_t], dim=-1))  # (B, S, 1)

        return (1 - g) * a_loc + g * a_ctx


class HiTHAR(nn.Module):
    """Full HiT-HAR model — standalone IMU pipeline (no distillation)."""

    def __init__(self, config):
        super().__init__()
        # Core dims
        in_channels = config.get('in_channels', 8)
        embedding_dim = config.get('embedding_dim', 128)
        num_actions = config.get('num_actions', 5)
        num_scenarios = config.get('num_scenarios', 8)
        seq_len = config.get('seq_len', 30)

        # HLA config
        hla_layers = config.get('hla_layers', 3)
        hla_nhead = config.get('hla_nhead', 4)
        hla_ff = config.get('hla_ff', 512)
        hla_dropout = config.get('hla_dropout', 0.1)

        # LLE dropout
        lle_dropout = config.get('lle_dropout', 0.2)

        # New feature flags
        self.class_wise_gate = config.get('class_wise_gate', False)
        use_coarse_head = config.get('coarse_head', False)

        # Student backbone
        self.lle = LLEv4(in_channels=in_channels, embedding_dim=embedding_dim, dropout=lle_dropout)
        self.hla = HLAv4(
            d_model=embedding_dim,
            nhead=hla_nhead,
            num_layers=hla_layers,
            dim_feedforward=hla_ff,
            dropout=hla_dropout,
            seq_len=seq_len,
        )

        # Heads
        self.action_head = ContextAwareActionHead(
            embedding_dim, num_actions,
            class_wise_gate=self.class_wise_gate,
            num_scenarios=num_scenarios,
        )
        self.scenario_head = nn.Linear(embedding_dim, num_scenarios)

        # Coarse auxiliary head
        # Mapping: OT→0(Manipulation), TaskOp→0(Manipulation), Stat→1(Passive),
        #          Loco→2(Locomotion), Search→1(Passive)
        if use_coarse_head:
            self.coarse_head = nn.Linear(embedding_dim, 3)
            self.register_buffer('fine_to_coarse', torch.tensor([0, 0, 1, 2, 1]))
        else:
            self.coarse_head = None

        self.seq_len = seq_len

    def forward(self, x):
        """
        Args:
            x: (B, S, 50, C) — IMU sequences

        Returns:
            dict with keys:
                scenario_logits: (B, num_scenarios)
                action_logits: (B, S, num_actions)
                e_t: (B, S, D) — LLE embeddings (for loss computation)
                h_cls: (B, D) — HLA CLS embedding
                h_t: (B, S, D) — HLA per-window embeddings
                coarse_logits: (B, S, 3) — only when coarse_head is active
                fine_to_coarse: (5,) — only when coarse_head is active
        """
        B, S, W, C = x.shape

        # LLE: process each window
        x_flat = x.view(B * S, W, C)       # (B*S, 50, C)
        e_flat = self.lle(x_flat)            # (B*S, D)
        e_t = e_flat.view(B, S, -1)          # (B, S, D)

        # HLA: aggregate sequence
        h_cls, h_t = self.hla(e_t)           # (B, D), (B, S, D)

        # Predictions
        scenario_logits = self.scenario_head(h_cls)  # (B, num_scenarios)

        if self.class_wise_gate:
            action_logits = self.action_head(e_t, h_t, h_cls=h_cls,
                                             scenario_logits=scenario_logits)
        else:
            action_logits = self.action_head(e_t, h_t)  # (B, S, num_actions)

        result = {
            'scenario_logits': scenario_logits,
            'action_logits': action_logits,
            'e_t': e_t,
            'h_cls': h_cls,
            'h_t': h_t,
        }

        # Coarse auxiliary head
        if self.coarse_head is not None:
            result['coarse_logits'] = self.coarse_head(h_t)  # (B, S, 3)
            result['fine_to_coarse'] = self.fine_to_coarse

        return result

    def count_parameters(self):
        """Count model parameters."""
        return sum(p.numel() for p in self.parameters())


class LanIMUHAR(HiTHAR):
    """LanIMU variant — Language-Anchored Multi-Scale IMU HAR.

    Extends HiTHAR with:
      - Optional multi-scale HLA (replaces standard HLA)
      - Language alignment via TextAnchorEncoder (frozen text encoder + projection)
    """

    def __init__(self, model_config, raw_config):
        """
        Args:
            model_config: flat dict for HiTHAR (embedding_dim, hla_layers, etc.)
            raw_config: full nested YAML config (for multiscale/language options)
        """
        super().__init__(model_config)

        embedding_dim = model_config.get('embedding_dim', 128)

        # Replace HLA with multi-scale if configured
        if raw_config.get('multiscale', False):
            from src.models.multiscale_hla import MultiScaleHLA
            scales = raw_config.get('scales', [1, 2, 4])
            self.hla = MultiScaleHLA(
                d_model=embedding_dim,
                nhead=model_config.get('hla_nhead', 4),
                num_layers=model_config.get('hla_layers', 3),
                dim_feedforward=model_config.get('hla_ff', 512),
                dropout=model_config.get('hla_dropout', 0.1),
                seq_len=model_config.get('seq_len', 30),
                scales=scales,
            )

        # Language alignment encoder (optional)
        self.text_encoder = None
        if raw_config.get('language_distill', False):
            from src.models.text_encoder import TextAnchorEncoder
            text_model = raw_config.get('text_model', 'all-mpnet-base-v2')
            class_names = raw_config.get('class_names', None)
            self.text_encoder = TextAnchorEncoder(
                projection_dim=embedding_dim,
                text_model_name=text_model,
                class_names=class_names,
            )

    def forward(self, x):
        result = super().forward(x)
        if self.text_encoder is not None:
            result['text_prototypes'] = self.text_encoder.get_class_prototypes()
        return result


class DualSpaceLanIMUHAR(LanIMUHAR):
    """Dual-Space variant — separates language-alignable and sensor-intrinsic subspaces.

    Extends LanIMUHAR with:
      - proj_lang: MLP(D -> lang_dim) — language contrastive loss applied here only
      - proj_sensor: MLP(D -> sensor_dim) — classification gradients only
      - z_combined = [z_lang; z_sensor] feeds action head
      - Scenario head stays on h_cls (pre-split)
      - Optional observability gate: per-class alpha_c blending of action logits
    """

    def __init__(self, model_config, raw_config):
        # Temporarily disable language in parent to avoid double SBERT load
        raw_config_parent = dict(raw_config)
        raw_config_parent['language_distill'] = False
        super().__init__(model_config, raw_config_parent)

        embedding_dim = model_config.get('embedding_dim', 128)
        lang_dim = raw_config.get('lang_subspace_dim', 64)
        sensor_dim = raw_config.get('sensor_subspace_dim', 64)
        num_actions = model_config.get('num_actions', 5)

        # Enforce that combined dim equals embedding_dim
        assert lang_dim + sensor_dim == embedding_dim, (
            f"lang_subspace_dim ({lang_dim}) + sensor_subspace_dim ({sensor_dim}) "
            f"must equal embedding_dim ({embedding_dim})"
        )

        # Create text encoder with correct projection dim (only loaded once)
        if raw_config.get('language_distill', False):
            from src.models.text_encoder import TextAnchorEncoder
            text_model = raw_config.get('text_model', 'all-mpnet-base-v2')
            class_names = raw_config.get('class_names', None)
            self.text_encoder = TextAnchorEncoder(
                projection_dim=lang_dim,
                text_model_name=text_model,
                class_names=class_names,
            )

        # Dual-space projection heads
        self.proj_lang = nn.Sequential(
            nn.Linear(embedding_dim, lang_dim),
            nn.LayerNorm(lang_dim),
        )
        self.proj_sensor = nn.Sequential(
            nn.Linear(embedding_dim, sensor_dim),
            nn.LayerNorm(sensor_dim),
        )

        # Observability gate (optional ablation)
        self.use_obs_gate = raw_config.get('observability_gate', False)
        if self.use_obs_gate:
            # Per-class scalar alpha_c in [0,1], initialized at 0.5 (logit=0)
            self.obs_gate_logits = nn.Parameter(torch.zeros(num_actions))

        # Store dims for loss computation
        self.lang_dim = lang_dim
        self.sensor_dim = sensor_dim

    def forward(self, x):
        B, S, W, C = x.shape

        # LLE: process each window
        x_flat = x.view(B * S, W, C)
        e_flat = self.lle(x_flat)
        e_t = e_flat.view(B, S, -1)  # (B, S, D=128)

        # HLA: aggregate sequence
        h_cls, h_t = self.hla(e_t)  # (B, D=128), (B, S, D=128)

        # Scenario head on h_cls (PRE-split, not affected by decomposition)
        scenario_logits = self.scenario_head(h_cls)  # (B, num_scenarios)

        # Dual-space decomposition
        z_lang = self.proj_lang(h_cls)      # (B, lang_dim=64)
        z_sensor = self.proj_sensor(h_cls)  # (B, sensor_dim=64)
        z_combined = torch.cat([z_lang, z_sensor], dim=1)  # (B, D=128)

        # Action head: z_combined is same dim as original h_cls (128)
        if self.class_wise_gate:
            action_logits = self.action_head(
                e_t, h_t, h_cls=z_combined,
                scenario_logits=scenario_logits,
            )
        else:
            action_logits = self.action_head(e_t, h_t)

        # Observability gate modifies action logits (not embeddings)
        if self.use_obs_gate:
            alpha = torch.sigmoid(self.obs_gate_logits)  # (num_actions,)
            z_lang_padded = torch.cat([z_lang, torch.zeros_like(z_sensor)], dim=1)
            z_sensor_padded = torch.cat([torch.zeros_like(z_lang), z_sensor], dim=1)
            if self.class_wise_gate:
                logits_lang = self.action_head(e_t, h_t, h_cls=z_lang_padded,
                                                scenario_logits=scenario_logits)
                logits_sensor = self.action_head(e_t, h_t, h_cls=z_sensor_padded,
                                                  scenario_logits=scenario_logits)
            else:
                logits_lang = logits_sensor = action_logits
            action_logits = alpha * logits_lang + (1 - alpha) * logits_sensor

        result = {
            'scenario_logits': scenario_logits,
            'action_logits': action_logits,
            'e_t': e_t,
            'h_cls': h_cls,
            'h_t': h_t,
            'z_lang': z_lang,
            'z_sensor': z_sensor,
            'z_combined': z_combined,
        }

        # Text prototypes (64-dim, matching z_lang)
        if self.text_encoder is not None:
            result['text_prototypes'] = self.text_encoder.get_class_prototypes()

        # Observability gate values (for logging)
        if self.use_obs_gate:
            result['obs_alpha'] = torch.sigmoid(self.obs_gate_logits)

        # Coarse head (on h_t, unchanged)
        if self.coarse_head is not None:
            result['coarse_logits'] = self.coarse_head(h_t)
            result['fine_to_coarse'] = self.fine_to_coarse

        return result


def build_model(config) -> HiTHAR:
    """Build HiT-HAR model from config dict.

    Flattens nested YAML config into flat model_config for HiTHAR.
    Uses DualSpaceLanIMUHAR if dual_space + language_distill is enabled.
    Uses LanIMUHAR if language_distill or multiscale is enabled.
    """
    model_config = {
        'in_channels': config['lle']['in_channels'],
        'embedding_dim': config['lle']['embedding_dim'],
        'num_actions': config.get('num_actions', 5),
        'num_scenarios': config['hla']['num_classes'],
        'seq_len': config['hla']['seq_len'],
        'hla_layers': config['hla']['num_layers'],
        'hla_nhead': config['hla']['nhead'],
        'hla_ff': config['hla'].get('dim_feedforward', 512),
        'hla_dropout': config['hla'].get('dropout', 0.1),
        'lle_dropout': config['lle'].get('dropout', 0.2),
        'class_wise_gate': config.get('class_wise_gate', False),
        'coarse_head': config.get('coarse_head', False),
    }

    use_language = config.get('language_distill', False)
    use_multiscale = config.get('multiscale', False)
    use_dual_space = config.get('dual_space', False)

    if use_dual_space and use_language:
        return DualSpaceLanIMUHAR(model_config, raw_config=config)
    elif use_language or use_multiscale:
        return LanIMUHAR(model_config, raw_config=config)
    return HiTHAR(model_config)
