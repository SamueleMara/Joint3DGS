from __future__ import annotations

import torch

from dynamic_recon.config.schema import GeometryConfig
from dynamic_recon.geometry.dynamic_prior import PairResiduals, build_dynamic_prior


def test_dynamic_prior_monotonicity() -> None:
    cfg = GeometryConfig()
    low = PairResiduals(
        torch.zeros((2, 2, 2)),
        torch.zeros((2, 2)),
        torch.ones((2, 2)) * 0.1,
        torch.ones((2, 2)) * 0.1,
        torch.ones((2, 2)) * 0.1,
        torch.ones((2, 2)) * 0.1,
        torch.ones((2, 2)) * 0.1,
        torch.ones((2, 2)) * 0.1,
        torch.ones((2, 2)),
    )
    high = PairResiduals(
        torch.zeros((2, 2, 2)),
        torch.zeros((2, 2)),
        torch.ones((2, 2)) * 10.0,
        torch.ones((2, 2)) * 10.0,
        torch.ones((2, 2)) * 10.0,
        torch.ones((2, 2)) * 10.0,
        torch.ones((2, 2)) * 10.0,
        torch.ones((2, 2)) * 10.0,
        torch.ones((2, 2)),
    )
    assert torch.mean(build_dynamic_prior(high, cfg)) > torch.mean(build_dynamic_prior(low, cfg))
