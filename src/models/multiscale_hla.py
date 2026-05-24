"""Multi-Scale HLA — Feature Pyramid Temporal Aggregation.

Creates multiple temporal scales from LLE output via average pooling,
processes each through a shared Transformer, and fuses back to original resolution.

Scales: 1s (original), 2s (AvgPool k=2), 4s (AvgPool k=4)
No data pipeline changes needed — operates on LLE output tensors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleHLA(nn.Module):
    """Multi-scale HLA with Feature Pyramid Fusion.

    Architecture:
        Input: (B, S=30, D=128) — LLE window embeddings

        For each scale s in [1, 2, 4]:
            1. AvgPool1d(kernel=s) → (B, S//s, D)
            2. Add CLS token → (B, S//s+1, D)
            3. Add positional encoding
            4. Shared Transformer encoder → (B, S//s+1, D)

        Fusion:
            - Upsample coarser scales back to S via interpolation
            - Concatenate all scales → (B, S, D*n_scales)
            - Linear fusion → (B, S, D)

        Output: h_cls (B, D), h_t (B, S, D)
    """

    def __init__(self, d_model, nhead, num_layers, dim_feedforward,
                 dropout=0.1, seq_len=30, scales=None):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.scales = scales or [1, 2, 4]

        # Shared transformer encoder (weight sharing across scales)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)

        # Per-scale CLS tokens and positional embeddings
        self.cls_tokens = nn.ParameterDict()
        self.pos_embeds = nn.ParameterDict()
        for s in self.scales:
            scale_len = seq_len // s
            self.cls_tokens[str(s)] = nn.Parameter(torch.zeros(1, 1, d_model))
            self.pos_embeds[str(s)] = nn.Parameter(torch.zeros(1, scale_len + 1, d_model))
            nn.init.trunc_normal_(self.cls_tokens[str(s)], std=0.02)
            nn.init.trunc_normal_(self.pos_embeds[str(s)], std=0.02)

        # Fusion: concat scales → project back to d_model
        if len(self.scales) > 1:
            self.fusion = nn.Sequential(
                nn.Linear(d_model * len(self.scales), d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
        else:
            self.fusion = nn.Identity()

    def _process_scale(self, e_t, scale):
        """Process one temporal scale.

        Args:
            e_t: (B, S, D)
            scale: pooling factor (1, 2, or 4)

        Returns:
            cls_out: (B, D) — CLS token output
            seq_out: (B, S//scale, D) — per-window outputs
        """
        B, S, D = e_t.shape

        if scale > 1:
            # AvgPool along temporal dimension
            x = e_t.transpose(1, 2)  # (B, D, S)
            x = F.avg_pool1d(x, kernel_size=scale, stride=scale)  # (B, D, S//scale)
            x = x.transpose(1, 2)  # (B, S//scale, D)
        else:
            x = e_t

        # Add CLS token
        cls_token = self.cls_tokens[str(scale)].expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # (B, S//scale+1, D)

        # Add positional embedding
        x = x + self.pos_embeds[str(scale)][:, :x.size(1), :]

        # Shared transformer
        x = self.transformer(x)
        x = self.final_norm(x)

        cls_out = x[:, 0, :]   # (B, D)
        seq_out = x[:, 1:, :]  # (B, S//scale, D)

        return cls_out, seq_out

    def forward(self, e_t):
        """
        Args:
            e_t: (B, S, D) — LLE window embeddings

        Returns:
            h_cls: (B, D) — sequence-level embedding
            h_t: (B, S, D) — per-window contextual embeddings
        """
        B, S, D = e_t.shape
        scale_outputs = []
        cls_outputs = []

        for scale in self.scales:
            cls_out, seq_out = self._process_scale(e_t, scale)
            cls_outputs.append(cls_out)

            # Upsample back to original resolution if needed
            if scale > 1:
                # Interpolate: (B, S//scale, D) → (B, S, D)
                seq_up = seq_out.transpose(1, 2)  # (B, D, S//scale)
                seq_up = F.interpolate(seq_up, size=S, mode='linear',
                                       align_corners=False)
                seq_out = seq_up.transpose(1, 2)  # (B, S, D)

            scale_outputs.append(seq_out)

        # Fuse CLS tokens (mean across scales)
        h_cls = torch.stack(cls_outputs, dim=0).mean(dim=0)  # (B, D)

        # Fuse per-window features
        if len(self.scales) > 1:
            concat = torch.cat(scale_outputs, dim=2)  # (B, S, D*n_scales)
            h_t = self.fusion(concat)                  # (B, S, D)
        else:
            h_t = scale_outputs[0]

        return h_cls, h_t
