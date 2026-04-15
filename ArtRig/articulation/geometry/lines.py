from __future__ import annotations

import torch



def normalize_direction(u: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if u.shape[-1] != 3:
        raise ValueError("u must end with dim 3")
    return u / torch.linalg.norm(u, dim=-1, keepdim=True).clamp_min(eps)



def point_to_line_distance(points: torch.Tensor, line_point: torch.Tensor, line_dir: torch.Tensor) -> torch.Tensor:
    if points.shape[-1] != 3:
        raise ValueError("points must end with dim 3")
    u = normalize_direction(line_dir)
    rel = points - line_point
    proj = (rel * u).sum(dim=-1, keepdim=True) * u
    perp = rel - proj
    return torch.linalg.norm(perp, dim=-1)
