from __future__ import annotations

import torch



def pairwise_distance_delta(anchor_points: torch.Tensor, points_t: torch.Tensor) -> torch.Tensor:
    """Absolute pairwise-distance change between anchor and current frame.

    Args:
        anchor_points: [N,3]
        points_t: [N,3]
    Returns:
        delta: [N,N]
    """
    d0 = torch.cdist(anchor_points, anchor_points)
    dt = torch.cdist(points_t, points_t)
    return torch.abs(dt - d0)
