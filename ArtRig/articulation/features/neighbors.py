from __future__ import annotations

import torch



def knn_indices(features: torch.Tensor, k: int, exclude_self: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """k-NN in feature space using pairwise squared distances.

    Returns:
        idx: [P,K]
        dist2: [P,K]
    """
    if features.ndim != 2:
        raise ValueError("features must be [P,C]")
    p = features.shape[0]
    if k <= 0:
        raise ValueError("k must be > 0")
    if p == 0:
        raise ValueError("features must contain at least one point")

    k_eff = min(k + (1 if exclude_self else 0), p)
    d2 = torch.cdist(features, features, p=2.0) ** 2
    vals, idx = torch.topk(d2, k=k_eff, largest=False)

    if exclude_self:
        idx = idx[:, 1:]
        vals = vals[:, 1:]

    if idx.shape[1] > k:
        idx = idx[:, :k]
        vals = vals[:, :k]

    return idx, vals
