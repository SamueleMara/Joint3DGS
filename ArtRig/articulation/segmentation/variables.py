from __future__ import annotations

import torch
from torch import nn


class SegmentationVariables(nn.Module):
    def __init__(self, num_points: int, num_steps: int):
        super().__init__()
        if num_points <= 0:
            raise ValueError("num_points must be > 0")
        if num_steps <= 0:
            raise ValueError("num_steps must be > 0")

        self.logits = nn.Parameter(torch.zeros(num_points))
        self.xi_part0 = nn.Parameter(torch.zeros(num_steps, 6))
        self.xi_part1 = nn.Parameter(torch.zeros(num_steps, 6))

    def probs(self) -> torch.Tensor:
        return torch.sigmoid(self.logits)
