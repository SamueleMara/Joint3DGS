#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

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


def _save_mask_overlays(
    rgb_u8: np.ndarray,
    masks: torch.Tensor,
    out_dir: Path,
    alpha: float,
    show_progress: bool = False,
) -> np.ndarray:
    out_overlay = out_dir / "mask_overlay"
    out_part0 = out_dir / "mask_part0"
    out_part1 = out_dir / "mask_part1"
    out_overlay.mkdir(parents=True, exist_ok=True)
    out_part0.mkdir(parents=True, exist_ok=True)
    out_part1.mkdir(parents=True, exist_ok=True)

    import cv2

    labels = masks.argmax(dim=1).cpu().numpy()  # [T,H,W]
    colors = np.array([[255.0, 80.0, 80.0], [80.0, 140.0, 255.0]], dtype=np.float32)
    frame_iter = range(rgb_u8.shape[0])
    if show_progress:
        frame_iter = tqdm(frame_iter, desc="Viz3D/Mask Overlays", leave=False)
    for t in frame_iter:
        base = rgb_u8[t].astype(np.float32)
        overlay = base.copy()
        for part in (0, 1):
            m = labels[t] == part
            if np.any(m):
                overlay[m] = (1.0 - alpha) * overlay[m] + alpha * colors[part]
        overlay_u8 = np.clip(overlay, 0.0, 255.0).astype(np.uint8)
        mask0 = ((labels[t] == 0).astype(np.uint8) * 255)
        mask1 = ((labels[t] == 1).astype(np.uint8) * 255)

        cv2.imwrite(str(out_overlay / f"{t:04d}.png"), cv2.cvtColor(overlay_u8, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_part0 / f"{t:04d}.png"), mask0)
        cv2.imwrite(str(out_part1 / f"{t:04d}.png"), mask1)

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


def _subsample_points(
    points: np.ndarray,
    colors: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points, colors
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx], colors[idx]


def _write_point_cloud(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    o3d.io.write_point_cloud(str(path), pcd)


def _build_dense_segmented_clouds_over_frames(
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
    static_label: int = 0,
    export_frame_clouds: bool = True,
    frame_clouds_dir: Path | None = None,
    frame_max_points: int = 25000,
    show_progress: bool = False,
) -> dict[str, object]:
    rng = np.random.default_rng(0)
    conf = masks.max(dim=1).values.cpu().numpy()

    points_all: list[np.ndarray] = []
    colors_all: list[np.ndarray] = []
    points_static_all: list[np.ndarray] = []
    points_dynamic_all: list[np.ndarray] = []

    color_part0 = np.array([1.0, 0.35, 0.35], dtype=np.float32)
    color_part1 = np.array([0.35, 0.55, 1.0], dtype=np.float32)
    color_static = np.array([0.2, 0.9, 0.25], dtype=np.float32)
    color_dynamic = np.array([1.0, 0.6, 0.2], dtype=np.float32)

    frame_stats: list[dict[str, int]] = []

    if export_frame_clouds and frame_clouds_dir is not None:
        (frame_clouds_dir / "combined").mkdir(parents=True, exist_ok=True)
        (frame_clouds_dir / "static").mkdir(parents=True, exist_ok=True)
        (frame_clouds_dir / "dynamic").mkdir(parents=True, exist_ok=True)

    frame_iter = range(0, seq_depth.shape[0], max(1, frame_stride))
    if show_progress:
        frame_iter = tqdm(frame_iter, desc="Viz3D/Point Clouds", leave=False)
    for t in frame_iter:
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
            frame_stats.append({"frame": int(t), "points_total": 0, "points_static": 0, "points_dynamic": 0})
            continue

        pts = xyz[valid]
        part = labels[t][valid]
        is_static = part == int(static_label)
        is_dynamic = ~is_static

        cols = np.zeros((pts.shape[0], 3), dtype=np.float32)
        cols[part == 0] = color_part0
        cols[part == 1] = color_part1

        pts_static = pts[is_static].astype(np.float32)
        pts_dynamic = pts[is_dynamic].astype(np.float32)
        cols_static = np.tile(color_static, (pts_static.shape[0], 1)).astype(np.float32)
        cols_dynamic = np.tile(color_dynamic, (pts_dynamic.shape[0], 1)).astype(np.float32)

        points_all.append(pts.astype(np.float32))
        colors_all.append(cols)
        if pts_static.shape[0] > 0:
            points_static_all.append(pts_static)
        if pts_dynamic.shape[0] > 0:
            points_dynamic_all.append(pts_dynamic)

        frame_stats.append(
            {
                "frame": int(t),
                "points_total": int(pts.shape[0]),
                "points_static": int(pts_static.shape[0]),
                "points_dynamic": int(pts_dynamic.shape[0]),
            }
        )

        if export_frame_clouds and frame_clouds_dir is not None:
            pts_f, cols_f = _subsample_points(pts.astype(np.float32), cols, int(frame_max_points), rng)
            pts_s, cols_s = _subsample_points(pts_static, cols_static, int(frame_max_points), rng)
            pts_d, cols_d = _subsample_points(pts_dynamic, cols_dynamic, int(frame_max_points), rng)
            if pts_f.shape[0] > 0:
                _write_point_cloud(frame_clouds_dir / "combined" / f"{t:04d}.ply", pts_f, cols_f)
            if pts_s.shape[0] > 0:
                _write_point_cloud(frame_clouds_dir / "static" / f"{t:04d}.ply", pts_s, cols_s)
            if pts_d.shape[0] > 0:
                _write_point_cloud(frame_clouds_dir / "dynamic" / f"{t:04d}.ply", pts_d, cols_d)

    if not points_all:
        zeros = np.zeros((0, 3), dtype=np.float32)
        return {
            "points_all": zeros,
            "colors_all": zeros,
            "points_static": zeros,
            "colors_static": zeros,
            "points_dynamic": zeros,
            "colors_dynamic": zeros,
            "frame_stats": frame_stats,
        }

    points = np.concatenate(points_all, axis=0)
    colors = np.concatenate(colors_all, axis=0)
    points_static = (
        np.concatenate(points_static_all, axis=0)
        if points_static_all
        else np.zeros((0, 3), dtype=np.float32)
    )
    points_dynamic = (
        np.concatenate(points_dynamic_all, axis=0)
        if points_dynamic_all
        else np.zeros((0, 3), dtype=np.float32)
    )
    colors_static = np.tile(color_static, (points_static.shape[0], 1)).astype(np.float32)
    colors_dynamic = np.tile(color_dynamic, (points_dynamic.shape[0], 1)).astype(np.float32)

    points, colors = _subsample_points(points, colors, int(max_points), rng)
    points_static, colors_static = _subsample_points(points_static, colors_static, int(max_points), rng)
    points_dynamic, colors_dynamic = _subsample_points(points_dynamic, colors_dynamic, int(max_points), rng)

    return {
        "points_all": points,
        "colors_all": colors,
        "points_static": points_static,
        "colors_static": colors_static,
        "points_dynamic": points_dynamic,
        "colors_dynamic": colors_dynamic,
        "frame_stats": frame_stats,
    }


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
    out = _build_dense_segmented_clouds_over_frames(
        seq_depth=seq_depth,
        seq_k=seq_k,
        labels=labels,
        masks=masks,
        frame_stride=frame_stride,
        sample_ratio=sample_ratio,
        conf_thresh=conf_thresh,
        min_depth=min_depth,
        max_depth=max_depth,
        max_points=max_points,
        static_label=0,
        export_frame_clouds=False,
        frame_clouds_dir=None,
        frame_max_points=max_points,
        show_progress=False,
    )
    return out["points_all"], out["colors_all"]


def _infer_static_dynamic_parts(
    tracks_npz: str | Path,
    seg: SegmentationResult,
) -> tuple[int, int]:
    tracks = load_tracks_npz(tracks_npz)
    static_label, dynamic_label = choose_reference_part(tracks, seg)
    return int(static_label), int(dynamic_label)


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
    parser.add_argument("--frame-max-points", type=int, default=25000)
    parser.add_argument("--mask-alpha", type=float, default=0.45)
    parser.add_argument("--static-label", choices=["auto", "0", "1"], default="auto")
    parser.add_argument("--export-frame-clouds", dest="export_frame_clouds", action="store_true", default=True)
    parser.add_argument("--no-export-frame-clouds", dest="export_frame_clouds", action="store_false")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--show", action="store_true", help="open Open3D viewer")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seq = load_rgbd_sequence_npz(args.sequence_npz)
    seg = _load_segmentation(args.segmentation)

    masks = seg.masks_per_frame.detach().cpu().float()
    masks = _ensure_mask_shape(masks, seq.depth.shape[-2], seq.depth.shape[-1])

    rgb_u8 = _rgb_to_uint8(seq.rgb)
    labels = _save_mask_overlays(
        rgb_u8,
        masks,
        out_dir,
        alpha=float(args.mask_alpha),
        show_progress=args.progress,
    )

    static_label = 0
    dynamic_label = 1
    if args.static_label == "auto":
        if args.tracks_npz is not None:
            static_label, dynamic_label = _infer_static_dynamic_parts(args.tracks_npz, seg)
    else:
        static_label = int(args.static_label)
        dynamic_label = 1 - static_label

    clouds = _build_dense_segmented_clouds_over_frames(
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
        static_label=static_label,
        export_frame_clouds=bool(args.export_frame_clouds),
        frame_clouds_dir=out_dir / "frame_clouds",
        frame_max_points=int(args.frame_max_points),
        show_progress=args.progress,
    )
    points = clouds["points_all"]
    colors = clouds["colors_all"]
    if points.shape[0] == 0:
        raise RuntimeError("No valid 3D points for visualization. Try lower --conf-thresh or higher --sample-ratio.")

    import open3d as o3d

    pcd_path = out_dir / "segmented_pointcloud.ply"
    pcd_static_path = out_dir / "static_pointcloud.ply"
    pcd_dynamic_path = out_dir / "dynamic_pointcloud.ply"
    _write_point_cloud(pcd_path, points, colors)
    _write_point_cloud(pcd_static_path, clouds["points_static"], clouds["colors_static"])
    _write_point_cloud(pcd_dynamic_path, clouds["points_dynamic"], clouds["colors_dynamic"])

    pcd = o3d.io.read_point_cloud(str(pcd_path))

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
    print(f"Saved static point cloud (part {static_label}): {pcd_static_path}")
    print(f"Saved dynamic point cloud (part {dynamic_label}): {pcd_dynamic_path}")
    if args.export_frame_clouds:
        print(f"Saved per-frame clouds to: {out_dir / 'frame_clouds'}")
    if args.joint is not None:
        print(f"Saved axis line set: {out_dir / 'joint_axis.ply'}")
    if args.tracks_npz is not None and args.joint is not None:
        print(f"Saved canonical sparse joint view to: {out_dir}")

    if args.show:
        o3d.visualization.draw_geometries(geoms, window_name="ArtRig Segmentation + Joint")


if __name__ == "__main__":
    main()
