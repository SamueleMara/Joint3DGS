from __future__ import annotations

import torch



def _kmeans2_torch(x: torch.Tensor, iters: int = 50) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("x must be [P,C]")
    p = x.shape[0]
    if p < 2:
        return torch.zeros(p, dtype=torch.long, device=x.device)

    # deterministic endpoints in feature norm order for reproducibility
    norm = torch.linalg.norm(x, dim=1)
    c0 = x[norm.argmin()]
    c1 = x[norm.argmax()]

    for _ in range(iters):
        d0 = torch.sum((x - c0) ** 2, dim=1)
        d1 = torch.sum((x - c1) ** 2, dim=1)
        labels = (d1 < d0).long()

        if (labels == 0).sum() == 0 or (labels == 1).sum() == 0:
            break

        c0_new = x[labels == 0].mean(dim=0)
        c1_new = x[labels == 1].mean(dim=0)
        if torch.allclose(c0, c0_new, atol=1e-6) and torch.allclose(c1, c1_new, atol=1e-6):
            c0, c1 = c0_new, c1_new
            break
        c0, c1 = c0_new, c1_new

    return labels



def initialize_two_part_logits(
    point_features: torch.Tensor,
    low_logit: float = -2.0,
    high_logit: float = 2.0,
) -> torch.Tensor:
    labels = _kmeans2_torch(point_features)
    logits = torch.where(
        labels > 0,
        torch.full_like(labels, float(high_logit), dtype=point_features.dtype),
        torch.full_like(labels, float(low_logit), dtype=point_features.dtype),
    )
    return logits
