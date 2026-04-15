from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from articulation.data.dataclasses import RelativeMotionResult
from articulation.external.joint_clue_adapter import JointClueEstimator



def farthest_point_sampling(points: torch.Tensor, k: int) -> torch.Tensor:
    n = points.shape[0]
    if n == 0:
        return torch.zeros(0, dtype=torch.long, device=points.device)
    k = min(k, n)
    idx = torch.zeros(k, dtype=torch.long, device=points.device)
    dist = torch.full((n,), float("inf"), device=points.device)
    current = torch.randint(0, n, (1,), device=points.device).item()

    for i in range(k):
        idx[i] = current
        d = torch.linalg.norm(points - points[current], dim=1)
        dist = torch.minimum(dist, d)
        current = int(torch.argmax(dist).item())

    return idx


@dataclass
class PointwiseInitOutput:
    sampled_indices: torch.Tensor
    clues: list[dict]



def build_pointwise_initialization(
    rel: RelativeMotionResult,
    estimator: JointClueEstimator,
    num_point_samples: int = 256,
    strategy: str = "fps",
) -> PointwiseInitOutput:
    points = rel.canonical_points
    if strategy == "fps":
        idx = farthest_point_sampling(points, num_point_samples)
    elif strategy == "random":
        n = points.shape[0]
        k = min(num_point_samples, n)
        idx = torch.randperm(n, device=points.device)[:k]
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")

    clues: list[dict] = []
    for pi in idx.tolist():
        traj = rel.moving_points_rel[pi]
        valid = rel.valid[pi]
        if valid.sum() < 3:
            continue
        xyz = traj[valid].detach().cpu().numpy().astype(np.float32)
        clue = estimator.infer(xyz)
        clue["point_index"] = pi
        clues.append(clue)

    return PointwiseInitOutput(sampled_indices=idx, clues=clues)
