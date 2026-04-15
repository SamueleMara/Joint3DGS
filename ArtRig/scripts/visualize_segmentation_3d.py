#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from articulation.data import SegmentationResult, load_rgbd_sequence_npz, load_tracks_npz
from articulation.joint.relative_motion import choose_reference_part, compute_relative_motion


def _load_segmentation(path: str | Path) -> SegmentationResult:
    payload = torch.load(path, map_location="cpu")
    return SegmentationResult(
        point_logits=payload["point_logits"],
        point_probs=payload["point_probs"],
        point_labels=payload["point_labels"],
        masks_per_frame=payload["masks_per_frame"],
        transforms_part0=payload["transforms_part0"],
        transforms_part1=payload["transforms_part1"],
        diagnostics=payload.get("diagnostics", {}),
    )


def _rgb_to_uint8(rgb_tchw: torch.Tensor) -> np.ndarray:
    rgb = rgb_tchw.detach().cpu().numpy().transpose(0, 2, 3, 1)  # [T,H,W,3]
    if rgb.max() <= 1.0:
        rgb = rgb * 255.0
    return np.clip(rgb, 0.0, 255.0).astype(np.uint8)


def _ensure_mask_shape(masks_tchw: torch.Tensor, h: int, w: int) -> torch.Tensor:
    if masks_tchw.shape[-2:] == (h, w):
        return masks_tchw
    return F.interpolate(masks_tchw, size=(h, w), mode="bilinear", align_corners=False)


def _save_mask_overlays(rgb_u8: np.ndarray, masks: torch.Tensor, out_dir: Path, alpha: float) -> np.ndarray:
    out_overlay = out_dir / "mask_overlay"
    out_part0 = out_dir / "mask_part0"
    out_part1 = out_dir / "mask_part1"
    out_overlay.mkdir(parents=True, exist_ok=True)
    out_part0.mkdir(parents=True, exist_ok=True)
    out_part1.mkdir(parents=True, exist_ok=True)

    labels = masks.argmax(dim=1).cpu().numpy()  # [T,H,W]
    colors = np.array([[255.0, 80.0, 80.0], [80.0, 140.0, 255.0]], dtype=np.float32)

    for t in range(rgb_u8.shape[0]):
        base = rgb_u8[t].astype(np.float32)
        overlay = base.copy()
        for part in (0, 1):
            m = labels[t] == part
            if np.any(m):
                overlay[m] = (1.0 - alpha) * overlay[m] + alpha * colors[part]

        plt.imsave(out_overlay / f"{t:04d}.png", overlay.astype(np.uint8))
        plt.imsave(out_part0 / f"{t:04d}.png", (labels[t] == 0).astype(np.float32), cmap="gray", vmin=0.0, vmax=1.0)
        plt.imsave(out_part1 / f"{t:04d}.png", (labels[t] == 1).astype(np.float32), cmap="gray", vmin=0.0, vmax=1.0)

    return labels


def _backproject_depth(depth_hw: np.ndarray, k_33: np.ndarray) -> np.ndarray:
    h, w = depth_hw.shape
    v, u = np.indices((h, w), dtype=np.float32)
    fx = float(k_33[0, 0])
    fy = float(k_33[1, 1])
    cx = float(k_33[0, 2])
    cy = float(k_33[1, 2])

    z = depth_hw
    x = (u - cx) * z / max(fx, 1e-8)
    y = (v - cy) * z / max(fy, 1e-8)
    return np.stack([x, y, z], axis=-1)


def _build_dense_segmented_cloud(
    seq_depth: torch.Tensor,
    seq_k: torch.Tensor,
    labels: np.ndarray,
    masks: torch.Tensor,
    frame_stride: int,
    sample_ratio: float,
    conf_thresh: float,
    min_depth: float,
    max_depth: float,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    conf = masks.max(dim=1).values.cpu().numpy()

    points_all: list[np.ndarray] = []
    colors_all: list[np.ndarray] = []
    for t in range(0, seq_depth.shape[0], max(1, frame_stride)):
        depth = seq_depth[t, 0].detach().cpu().numpy()
        k = seq_k[t].detach().cpu().numpy() if seq_k.ndim == 3 else seq_k.detach().cpu().numpy()
        xyz = _backproject_depth(depth, k)

        valid = np.isfinite(depth) & (depth > float(min_depth))
        if max_depth > 0.0:
            valid &= depth < float(max_depth)
        if conf_thresh > 0.0:
            valid &= conf[t] >= float(conf_thresh)
        if sample_ratio < 1.0:
            valid &= rng.random(depth.shape) < float(sample_ratio)

        if not np.any(valid):
            continue

        pts = xyz[valid]
        part = labels[t][valid]

        cols = np.zeros((pts.shape[0], 3), dtype=np.float32)
        cols[part == 0] = np.array([1.0, 0.35, 0.35], dtype=np.float32)
        cols[part == 1] = np.array([0.35, 0.55, 1.0], dtype=np.float32)

        points_all.append(pts.astype(np.float32))
        colors_all.append(cols)

    if not points_all:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    points = np.concatenate(points_all, axis=0)
    colors = np.concatenate(colors_all, axis=0)

    if max_points > 0 and points.shape[0] > max_points:
        idx = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[idx]
        colors = colors[idx]

    return points, colors


def _make_axis_lineset(
    axis_dir: np.ndarray,
    axis_point: np.ndarray,
    length: float,
):
    import open3d as o3d

    axis = axis_dir.astype(np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    p0 = axis_point.astype(np.float64) - length * axis
    p1 = axis_point.astype(np.float64) + length * axis

    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector(np.stack([p0, p1], axis=0))
    line.lines = o3d.utility.Vector2iVector(np.array([[0, 1]], dtype=np.int32))
    line.colors = o3d.utility.Vector3dVector(np.array([[0.2, 0.9, 0.2]], dtype=np.float64))
    return line


def _save_sparse_joint_view(
    tracks_npz: str | Path,
    seg: SegmentationResult,
    joint_payload: dict,
    out_dir: Path,
) -> None:
    import open3d as o3d

    tracks = load_tracks_npz(tracks_npz)
    ref_part, mov_part = choose_reference_part(tracks, seg)
    labels = seg.point_labels.long()
    rel = compute_relative_motion(
        xyz_part_ref=tracks.xyz[labels == ref_part],
        xyz_part_mov=tracks.xyz[labels == mov_part],
        valid_ref=tracks.valid[labels == ref_part],
        valid_mov=tracks.valid[labels == mov_part],
        reference_part=ref_part,
        moving_part=mov_part,
    )

    pts = rel.canonical_points.detach().cpu().numpy().astype(np.float64)
    if pts.shape[0] == 0:
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.2, 0.7, 1.0]], dtype=np.float64), (pts.shape[0], 1)))
    o3d.io.write_point_cloud(str(out_dir / "joint_canonical_points.ply"), pcd)

    axis_dir = np.asarray(joint_payload["axis_dir"], dtype=np.float64)
    axis_point = (
        np.asarray(joint_payload["axis_point"], dtype=np.float64)
        if joint_payload.get("axis_point", None) is not None
        else pts.mean(axis=0)
    )
    length = float(np.linalg.norm(np.max(pts, axis=0) - np.min(pts, axis=0)) * 0.25 + 1e-6)
    axis_line = _make_axis_lineset(axis_dir=axis_dir, axis_point=axis_point, length=length)
    o3d.io.write_line_set(str(out_dir / "joint_canonical_axis.ply"), axis_line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-npz", required=True)
    parser.add_argument("--segmentation", required=True)
    parser.add_argument("--joint", default=None, help="joint.pt produced by run_pipeline")
    parser.add_argument("--tracks-npz", default=None, help="optional tracks.npz for canonical joint-axis view")
    parser.add_argument("--output-dir", default="outputs/viz_3d")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--sample-ratio", type=float, default=0.05)
    parser.add_argument("--conf-thresh", type=float, default=0.5)
    parser.add_argument("--min-depth", type=float, default=1e-4)
    parser.add_argument("--max-depth", type=float, default=0.0, help="<=0 disables max-depth clipping")
    parser.add_argument("--max-points", type=int, default=350000)
    parser.add_argument("--mask-alpha", type=float, default=0.45)
    parser.add_argument("--show", action="store_true", help="open Open3D viewer")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seq = load_rgbd_sequence_npz(args.sequence_npz)
    seg = _load_segmentation(args.segmentation)

    masks = seg.masks_per_frame.detach().cpu().float()
    masks = _ensure_mask_shape(masks, seq.depth.shape[-2], seq.depth.shape[-1])

    rgb_u8 = _rgb_to_uint8(seq.rgb)
    labels = _save_mask_overlays(rgb_u8, masks, out_dir, alpha=float(args.mask_alpha))

    points, colors = _build_dense_segmented_cloud(
        seq_depth=seq.depth,
        seq_k=seq.K,
        labels=labels,
        masks=masks,
        frame_stride=int(args.frame_stride),
        sample_ratio=float(args.sample_ratio),
        conf_thresh=float(args.conf_thresh),
        min_depth=float(args.min_depth),
        max_depth=float(args.max_depth),
        max_points=int(args.max_points),
    )

    if points.shape[0] == 0:
        raise RuntimeError("No valid 3D points for visualization. Try lower --conf-thresh or higher --sample-ratio.")

    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    pcd_path = out_dir / "segmented_pointcloud.ply"
    o3d.io.write_point_cloud(str(pcd_path), pcd)

    geoms = [pcd]
    if args.joint is not None:
        joint_payload = torch.load(args.joint, map_location="cpu")
        axis_dir = np.asarray(joint_payload["axis_dir"], dtype=np.float64)
        axis_point = (
            np.asarray(joint_payload["axis_point"], dtype=np.float64)
            if joint_payload.get("axis_point", None) is not None
            else points.mean(axis=0).astype(np.float64)
        )
        axis_len = float(np.linalg.norm(np.max(points, axis=0) - np.min(points, axis=0)) * 0.25 + 1e-6)
        axis_line = _make_axis_lineset(axis_dir=axis_dir, axis_point=axis_point, length=axis_len)
        axis_path = out_dir / "joint_axis.ply"
        o3d.io.write_line_set(str(axis_path), axis_line)
        geoms.append(axis_line)

        if args.tracks_npz is not None:
            _save_sparse_joint_view(args.tracks_npz, seg, joint_payload, out_dir)

    print(f"Saved overlays to: {out_dir / 'mask_overlay'}")
    print(f"Saved segmented point cloud: {pcd_path}")
    if args.joint is not None:
        print(f"Saved axis line set: {out_dir / 'joint_axis.ply'}")
    if args.tracks_npz is not None and args.joint is not None:
        print(f"Saved canonical sparse joint view to: {out_dir}")

    if args.show:
        o3d.visualization.draw_geometries(geoms, window_name="ArtRig Segmentation + Joint")


if __name__ == "__main__":
    main()
