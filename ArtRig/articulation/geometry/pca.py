from __future__ import annotations

import torch



def principal_axis(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be [N,3]")
    centered = points - points.mean(dim=0, keepdim=True)
    cov = centered.transpose(0, 1) @ centered / max(points.shape[0] - 1, 1)
    vals, vecs = torch.linalg.eigh(cov)
    idx = torch.argmax(vals)
    return vecs[:, idx], vals[idx]
