from __future__ import annotations

import torch
import torch.nn.functional as F

from articulation.data.dataclasses import FeatureGraph
from articulation.features.neighbors import knn_indices



def build_feature_graph(
    features: torch.Tensor,
    xy: torch.Tensor | None = None,
    num_neighbors: int = 16,
    spatial_gate_px: float | None = None,
    sigma: float = 0.25,
) -> FeatureGraph:
    """Build KNN graph with optional image-space gating.

    Args:
        features: [P,C] feature vectors
        xy: optional [P,2] anchor-frame coordinates for spatial gating
    """
    if features.ndim != 2:
        raise ValueError("features must be [P,C]")
    if xy is not None and (xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] != features.shape[0]):
        raise ValueError("xy must be [P,2] aligned with features")

    f = F.normalize(features, dim=1)
    idx, _ = knn_indices(f, k=num_neighbors, exclude_self=True)

    nbr_feat = f[idx]  # [P,K,C]
    center = f.unsqueeze(1)
    cos = (nbr_feat * center).sum(dim=-1).clamp(-1.0, 1.0)

    weights = torch.exp((cos - 1.0) / max(sigma, 1e-6))

    if spatial_gate_px is not None and xy is not None:
        nbr_xy = xy[idx]
        dist = torch.linalg.norm(nbr_xy - xy.unsqueeze(1), dim=-1)
        gate = (dist <= float(spatial_gate_px)).float()
        weights = weights * gate

    denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    weights = weights / denom
    return FeatureGraph(nn_idx=idx, nn_weight=weights)
