"""Tests for multi-scale HLA module."""
import torch
import pytest


class TestMultiScaleHLA:
    def test_output_shape(self):
        from src.models.multiscale_hla import MultiScaleHLA
        hla = MultiScaleHLA(d_model=128, nhead=4, num_layers=3,
                            dim_feedforward=512, seq_len=30)
        e_t = torch.randn(4, 30, 128)
        h_cls, h_t = hla(e_t)
        assert h_cls.shape == (4, 128), f"h_cls shape: {h_cls.shape}"
        assert h_t.shape == (4, 30, 128), f"h_t shape: {h_t.shape}"

    def test_multi_scale_pools(self):
        from src.models.multiscale_hla import MultiScaleHLA
        hla = MultiScaleHLA(d_model=128, nhead=4, num_layers=3,
                            dim_feedforward=512, seq_len=30,
                            scales=[1, 2, 4])
        e_t = torch.randn(2, 30, 128)
        h_cls, h_t = hla(e_t)
        # Should still output original resolution
        assert h_t.shape == (2, 30, 128)

    def test_single_scale_matches_original(self):
        """With scales=[1], should behave like original HLA."""
        from src.models.multiscale_hla import MultiScaleHLA
        hla = MultiScaleHLA(d_model=128, nhead=4, num_layers=3,
                            dim_feedforward=512, seq_len=30,
                            scales=[1])
        e_t = torch.randn(2, 30, 128)
        h_cls, h_t = hla(e_t)
        assert h_cls.shape == (2, 128)
        assert h_t.shape == (2, 30, 128)
