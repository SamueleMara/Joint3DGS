from __future__ import annotations

import torch

from dynamic_recon.geometry.visibility import inside_image_mask, occlusion_mask, positive_depth_mask


def test_visibility_masks() -> None:
    uv = torch.tensor([[[0.0, 0.0], [5.0, 5.0]]], dtype=torch.float32)
    assert inside_image_mask(uv, 4, 4).tolist() == [[True, False]]
    z = torch.tensor([[1.0, -1.0]], dtype=torch.float32)
    assert positive_depth_mask(z).tolist() == [[True, False]]
    pred = torch.tensor([[2.0]], dtype=torch.float32)
    obs = torch.tensor([[1.0]], dtype=torch.float32)
    assert bool(occlusion_mask(pred, obs, 0.05, 0.1)[0, 0])
