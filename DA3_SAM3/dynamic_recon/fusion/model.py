from __future__ import annotations

import torch
from torch import nn


class DynamicFusionNet(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.GroupNorm(4, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.GroupNorm(4, base_channels),
            nn.SiLU(),
        )
        self.dynamic_head = nn.Conv2d(base_channels, 1, 1)
        self.boundary_head = nn.Conv2d(base_channels, 1, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.encoder(x)
        return {"dynamic_logit": self.dynamic_head(feat), "boundary_logit": self.boundary_head(feat)}
