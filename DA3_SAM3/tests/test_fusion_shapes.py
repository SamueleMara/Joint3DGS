from __future__ import annotations

import torch

from dynamic_recon.fusion.model import DynamicFusionNet


def test_fusion_model_shapes() -> None:
    model = DynamicFusionNet(in_channels=9, base_channels=8)
    out = model(torch.zeros((2, 9, 16, 16), dtype=torch.float32))
    assert out["dynamic_logit"].shape == (2, 1, 16, 16)
    assert out["boundary_logit"].shape == (2, 1, 16, 16)
