from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch

from articulation.data.dataclasses import JointResult, RelativeMotionResult, SegmentationResult, TrackBatch



def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p



def save_segmentation_mask_preview(masks_per_frame: torch.Tensor, out_path: str | Path, frame_idx: int = 0) -> None:
    if masks_per_frame.ndim != 4 or masks_per_frame.shape[1] != 2:
        raise ValueError("masks_per_frame must be [T,2,H,W]")
    m0 = masks_per_frame[frame_idx, 0].detach().cpu().numpy()
    m1 = masks_per_frame[frame_idx, 1].detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(m0, cmap="Reds")
    axes[0].set_title("Part 0")
    axes[0].axis("off")
    axes[1].imshow(m1, cmap="Blues")
    axes[1].set_title("Part 1")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(_ensure_dir(out_path), dpi=150)
    plt.close(fig)



def save_point_label_overlay(
    tracks: TrackBatch,
    seg: SegmentationResult,
    out_path: str | Path,
    frame_idx: int = 0,
) -> None:
    xy = tracks.xy[:, frame_idx, :].detach().cpu().numpy()
    w = seg.point_probs.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=w, cmap="coolwarm", s=10)
    ax.set_title("Tracked points colored by P(part=1)")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(_ensure_dir(out_path), dpi=150)
    plt.close(fig)



def save_cog_trajectories(
    tracks: TrackBatch,
    seg: SegmentationResult,
    out_path: str | Path,
) -> None:
    w = seg.point_probs.detach()
    xyz = tracks.xyz.detach()
    valid = tracks.valid.detach().float()

    w1 = w[:, None] * valid
    w0 = (1.0 - w)[:, None] * valid

    sum0 = w0.sum(dim=0).clamp_min(1e-6)
    sum1 = w1.sum(dim=0).clamp_min(1e-6)

    c0 = (w0[..., None] * xyz).sum(dim=0) / sum0[:, None]
    c1 = (w1[..., None] * xyz).sum(dim=0) / sum1[:, None]

    c0 = c0.cpu().numpy()
    c1 = c1.cpu().numpy()

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(c0[:, 0], c0[:, 1], c0[:, 2], color="red", label="CoG part0")
    ax.plot(c1[:, 0], c1[:, 1], c1[:, 2], color="blue", label="CoG part1")
    ax.set_title("CoG trajectories")
    ax.legend()
    fig.tight_layout()
    fig.savefig(_ensure_dir(out_path), dpi=150)
    plt.close(fig)



def save_moving_points_ref(
    rel: RelativeMotionResult,
    out_path: str | Path,
    max_points: int = 512,
) -> None:
    pts = rel.moving_points_rel.detach().cpu().numpy()
    n = min(pts.shape[0], max_points)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    for i in range(n):
        ax.plot(pts[i, :, 0], pts[i, :, 1], pts[i, :, 2], alpha=0.5)
    ax.set_title("Moving-part trajectories in reference frame")
    fig.tight_layout()
    fig.savefig(_ensure_dir(out_path), dpi=150)
    plt.close(fig)



def save_axis_3d(
    points: torch.Tensor,
    axis_dir: torch.Tensor,
    axis_point: torch.Tensor | None,
    out_path: str | Path,
    length: float = 0.2,
) -> None:
    pts = points.detach().cpu().numpy()
    axis = axis_dir.detach().cpu().numpy()
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    if axis_point is None:
        ap = pts.mean(axis=0)
    else:
        ap = axis_point.detach().cpu().numpy()

    p0 = ap - length * axis
    p1 = ap + length * axis

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=4, alpha=0.4)
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color="green", linewidth=2)
    ax.set_title("Axis visualization")
    fig.tight_layout()
    fig.savefig(_ensure_dir(out_path), dpi=150)
    plt.close(fig)

    # Optional Open3D export for 3D inspection
    try:  # pragma: no cover
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector(np.stack([p0, p1], axis=0).astype(np.float64))
        line.lines = o3d.utility.Vector2iVector(np.array([[0, 1]], dtype=np.int32))

        ply_path = Path(out_path).with_suffix(".ply")
        o3d.io.write_point_cloud(str(ply_path), pcd)
        o3d.io.write_line_set(str(ply_path.with_name(ply_path.stem + "_axis.ply")), line)
    except Exception:
        pass



def save_model_fit_comparison(joint: JointResult, out_path: str | Path) -> None:
    names = [c.model_name for c in joint.candidates]
    losses = [c.loss for c in joint.candidates]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, losses, color="gray")
    ax.set_title("Per-model fit loss")
    ax.set_ylabel("Loss")
    fig.tight_layout()
    fig.savefig(_ensure_dir(out_path), dpi=150)
    plt.close(fig)



def save_state_vs_time(state: torch.Tensor, out_path: str | Path) -> None:
    s = state.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(len(s)), s, color="black")
    ax.set_title("Joint state vs time")
    ax.set_xlabel("Frame")
    ax.set_ylabel("State")
    fig.tight_layout()
    fig.savefig(_ensure_dir(out_path), dpi=150)
    plt.close(fig)
