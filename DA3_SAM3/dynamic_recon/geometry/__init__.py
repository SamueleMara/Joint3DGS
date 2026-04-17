"""Torch-based camera geometry utilities plus compatibility helpers."""

from pathlib import Path

import numpy as np
import torch
import trimesh

from .camera import c2w_to_w2c, compose_extrinsics, from_homogeneous, invert_w2c, to_homogeneous, w2c_3x4_to_4x4, w2c_to_c2w
from .dynamic_prior import PairResiduals, build_dynamic_prior, build_sequence_dynamic_prior
from .projection import backproject, pixel_grid, project, reproject_static, sample_at_coords
from .residuals import compute_pair_residuals
from .visibility import forward_backward_consistency, inside_image_mask, occlusion_mask, positive_depth_mask

__all__ = [
    "PairResiduals",
    "backproject",
    "build_dynamic_prior",
    "build_sequence_dynamic_prior",
    "c2w_to_w2c",
    "compose_extrinsics",
    "compute_pair_residuals",
    "forward_backward_consistency",
    "from_homogeneous",
    "inside_image_mask",
    "invert_w2c",
    "occlusion_mask",
    "pixel_grid",
    "positive_depth_mask",
    "project",
    "reproject_static",
    "save_point_cloud",
    "sample_at_coords",
    "to_homogeneous",
    "w2c_3x4_to_4x4",
    "w2c_to_c2w",
    "fuse_point_cloud",
]


def fuse_point_cloud(
    depth: np.ndarray,
    confidence: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics_w2c: np.ndarray,
    min_confidence: float,
    min_depth: float,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    all_points: list[np.ndarray] = []
    all_conf: list[np.ndarray] = []
    for idx in range(depth.shape[0]):
        d = torch.from_numpy(depth[idx]).float()
        k = torch.from_numpy(intrinsics[idx]).float()
        e = torch.from_numpy(extrinsics_w2c[idx]).float()
        world = backproject(d, k, e).detach().cpu().numpy()
        mask = (confidence[idx] >= min_confidence) & (depth[idx] >= min_depth) & (depth[idx] <= max_depth)
        if np.any(mask):
            all_points.append(world[mask])
            all_conf.append(confidence[idx][mask])
    if not all_points:
        return np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=float)
    return np.concatenate(all_points, axis=0), np.concatenate(all_conf, axis=0)


def save_point_cloud(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    if points.size == 0:
        cloud = trimesh.PointCloud(vertices=np.zeros((0, 3), dtype=float))
    else:
        kwargs = {"vertices": points}
        if colors is not None:
            kwargs["colors"] = colors
        cloud = trimesh.PointCloud(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    cloud.export(path)
