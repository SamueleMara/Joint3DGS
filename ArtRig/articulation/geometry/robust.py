from __future__ import annotations

import torch



def huber(x: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    ax = torch.abs(x)
    quad = 0.5 * ax * ax
    lin = delta * (ax - 0.5 * delta)
    return torch.where(ax <= delta, quad, lin)
