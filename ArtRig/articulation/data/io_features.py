from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F



def load_feature_tensor(path: str | Path) -> torch.Tensor:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pt":
        feat = torch.load(path, map_location="cpu")
        if not isinstance(feat, torch.Tensor):
            raise TypeError(".pt feature file must store a torch.Tensor")
        return feat.float()

    if suffix == ".npy":
        return torch.from_numpy(np.load(path)).float()

    if suffix == ".npz":
        data = np.load(path)
        key = "feat" if "feat" in data else list(data.keys())[0]
        return torch.from_numpy(data[key]).float()

    raise ValueError(f"Unsupported feature file suffix: {suffix}")



def sample_features_at_xy(feat_map: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    if feat_map.ndim != 3:
        raise ValueError(f"feat_map must be [C,H,W], got {feat_map.shape}")
    if xy.ndim != 2 or xy.shape[-1] != 2:
        raise ValueError(f"xy must be [P,2], got {xy.shape}")

    c, h, w = feat_map.shape
    if h <= 1 or w <= 1:
        raise ValueError("Feature map must have H,W > 1 for grid sampling")

    x = xy[:, 0].clamp(0, w - 1)
    y = xy[:, 1].clamp(0, h - 1)
    gx = (x / (w - 1)) * 2.0 - 1.0
    gy = (y / (h - 1)) * 2.0 - 1.0

    grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)
    feat = feat_map.unsqueeze(0)
    sampled = F.grid_sample(feat, grid, mode="bilinear", align_corners=True)
    sampled = sampled.squeeze(0).squeeze(-1).transpose(0, 1).contiguous()
    return sampled
