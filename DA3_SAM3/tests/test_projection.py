from __future__ import annotations

import torch

from dynamic_recon.geometry.projection import backproject, project


def test_backproject_then_project_identity() -> None:
    depth = torch.ones((2, 2), dtype=torch.float32) * 2.0
    k = torch.tensor([[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]], dtype=torch.float32)
    e = torch.eye(4, dtype=torch.float32)
    world = backproject(depth, k, e)
    uv, z = project(world, k, e)
    expected = torch.tensor([[[0.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]]], dtype=torch.float32)
    assert torch.allclose(uv, expected, atol=1.0e-4)
    assert torch.allclose(z, depth)
