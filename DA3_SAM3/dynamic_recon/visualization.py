from __future__ import annotations

import logging
from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from dynamic_recon.fusion import LABEL_DYNAMIC, LABEL_STATIC, LABEL_UNCERTAIN
from dynamic_recon.geometry.projection import backproject

LOGGER = logging.getLogger(__name__)


def save_overlay_frames(
    frame_paths: list[Path],
    dynamic_score: np.ndarray,
    label_map: np.ndarray,
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_paths: list[Path] = []
    for idx, frame_path in enumerate(frame_paths):
        frame = np.asarray(Image.open(frame_path).convert("RGB"), dtype=np.uint8)
        overlay = frame.copy()
        overlay[label_map[idx] == LABEL_STATIC] = _blend_color(overlay[label_map[idx] == LABEL_STATIC], np.array([0, 255, 0], dtype=np.uint8))
        overlay[label_map[idx] == LABEL_DYNAMIC] = _blend_color(overlay[label_map[idx] == LABEL_DYNAMIC], np.array([255, 0, 0], dtype=np.uint8))
        overlay[label_map[idx] == LABEL_UNCERTAIN] = _blend_color(overlay[label_map[idx] == LABEL_UNCERTAIN], np.array([255, 255, 0], dtype=np.uint8))

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(frame)
        axes[0].set_title("RGB")
        axes[0].axis("off")
        heat = axes[1].imshow(dynamic_score[idx], cmap="magma")
        axes[1].set_title("Dynamic Score")
        axes[1].axis("off")
        fig.colorbar(heat, ax=axes[1], fraction=0.046, pad=0.04)
        fig.tight_layout()
        score_path = output_dir / f"{idx:06d}_score.png"
        fig.savefig(score_path)
        plt.close(fig)

        overlay_path = output_dir / f"{idx:06d}_overlay.png"
        Image.fromarray(overlay).save(overlay_path)
        overlay_paths.append(overlay_path)
    return overlay_paths


def write_video(image_paths: list[Path], output_path: Path, fps: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with imageio.get_writer(output_path, fps=max(fps, 1.0)) as writer:
            for path in image_paths:
                writer.append_data(imageio.imread(path))
        LOGGER.info("Saved visualization video to %s", output_path)
    except ValueError:
        fallback_path = output_path.with_suffix(".gif")
        frames = [imageio.imread(path) for path in image_paths]
        imageio.mimsave(fallback_path, frames, duration=max(1.0 / max(fps, 1.0), 0.05))
        LOGGER.warning("MP4 backend unavailable; wrote GIF fallback to %s", fallback_path)


def write_segmentation_videos(
    video_path: str | Path,
    sam_frames: list[object],
    output_dir: Path,
    fps: float,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = output_dir / "sam_overlay.mp4"
    panel_path = output_dir / "sam_panel.mp4"
    overlay_writer = imageio.get_writer(overlay_path, fps=max(fps, 1.0))
    panel_writer = imageio.get_writer(panel_path, fps=max(fps, 1.0))
    frames = _iter_frames(video_path)
    sam_by_frame = {int(item.frame_index): item for item in sam_frames}
    try:
        for frame_index, frame_rgb in frames:
            current = sam_by_frame.get(frame_index)
            if current is None:
                continue
            overlay = _render_sam_overlay(frame_rgb, current)
            panel = _render_sam_panel(frame_rgb, overlay, current)
            overlay_writer.append_data(overlay)
            panel_writer.append_data(panel)
    finally:
        overlay_writer.close()
        panel_writer.close()
    return {"sam_overlay": overlay_path, "sam_panel": panel_path}


def write_dynamic_masked_video(
    video_path: str | Path,
    sam_frames: list[object],
    output_path: Path,
    fps: float,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(output_path, fps=max(fps, 1.0))
    frames = _iter_frames(video_path)
    sam_by_frame = {int(item.frame_index): item for item in sam_frames}
    try:
        for frame_index, frame_rgb in frames:
            current = sam_by_frame.get(frame_index)
            if current is None:
                continue
            writer.append_data(_render_dynamic_only(frame_rgb, current))
    finally:
        writer.close()
    return output_path


def write_dynamic_probability_masked_video(
    video_path: str | Path,
    dynamic_prob_seq: np.ndarray,
    output_path: Path,
    fps: float,
    threshold: float = 0.45,
    neighbor_threshold: float = 0.35,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(output_path, fps=max(fps, 1.0))
    frames = _iter_frames(video_path)
    try:
        for frame_index, frame_rgb in frames:
            if frame_index >= len(dynamic_prob_seq):
                break
            prob_prev = dynamic_prob_seq[frame_index - 1] if frame_index > 0 else None
            prob_next = dynamic_prob_seq[frame_index + 1] if frame_index + 1 < len(dynamic_prob_seq) else None
            writer.append_data(
                _render_dynamic_probability_only(
                    frame_rgb,
                    dynamic_prob_seq[frame_index],
                    threshold=threshold,
                    prob_prev=prob_prev,
                    prob_next=prob_next,
                    neighbor_threshold=neighbor_threshold,
                )
            )
    finally:
        writer.close()
    return output_path


def write_pointcloud_camera_visualizations(
    da3_seq: object,
    poses: list[torch.Tensor],
    dynamic_prob_seq: np.ndarray | torch.Tensor | None,
    output_dir: Path,
    *,
    dynamic_threshold: float = 0.45,
    point_stride: int = 6,
    max_points: int = 120_000,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    static_xyz, static_rgb, dynamic_xyz, dynamic_rgb = _collect_labeled_world_points(
        da3_seq,
        poses,
        dynamic_prob_seq,
        dynamic_threshold=dynamic_threshold,
        point_stride=point_stride,
    )
    static_xyz, static_rgb = _downsample_cloud(static_xyz, static_rgb, max_points=max_points)
    dynamic_xyz, dynamic_rgb = _downsample_cloud(dynamic_xyz, dynamic_rgb, max_points=max_points)
    camera_xyz = _camera_centers_from_w2c(poses)

    outputs: dict[str, Path] = {}
    outputs["static_plot"] = output_dir / "scene_static_points_cameras.png"
    outputs["dynamic_plot"] = output_dir / "scene_dynamic_points_cameras.png"
    outputs["combined_plot"] = output_dir / "scene_static_dynamic_cameras.png"
    outputs["static_ply"] = output_dir / "scene_static_points.ply"
    outputs["dynamic_ply"] = output_dir / "scene_dynamic_points.ply"
    outputs["camera_centers_npy"] = output_dir / "camera_centers.npy"

    _plot_single_cloud_with_cameras(
        outputs["static_plot"],
        static_xyz,
        static_rgb,
        camera_xyz,
        title="Static Point Cloud + Estimated Cameras",
    )
    _plot_single_cloud_with_cameras(
        outputs["dynamic_plot"],
        dynamic_xyz,
        dynamic_rgb,
        camera_xyz,
        title="Dynamic Point Cloud + Estimated Cameras",
    )
    _plot_combined_cloud_with_cameras(
        outputs["combined_plot"],
        static_xyz,
        dynamic_xyz,
        camera_xyz,
    )
    _write_ply(outputs["static_ply"], static_xyz, static_rgb)
    _write_ply(outputs["dynamic_ply"], dynamic_xyz, dynamic_rgb)
    np.save(outputs["camera_centers_npy"], camera_xyz)
    return outputs


def write_pointcloud_over_time_video(
    da3_seq: object,
    poses: list[torch.Tensor],
    dynamic_prob_seq: np.ndarray | torch.Tensor | None,
    output_path: Path,
    *,
    fps: float,
    dynamic_threshold: float = 0.45,
    point_stride: int = 6,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(output_path, fps=max(float(fps), 1.0))
    camera_xyz = _camera_centers_from_w2c(poses)
    center, radius = _estimate_scene_bounds(da3_seq, poses, point_stride=point_stride)

    if dynamic_prob_seq is not None and torch.is_tensor(dynamic_prob_seq):
        dynamic_prob_seq = dynamic_prob_seq.detach().cpu().numpy()

    try:
        for frame_index, frame in enumerate(da3_seq.frames):
            world = backproject(frame.depth, frame.intrinsics, poses[frame_index]).detach().cpu().numpy()
            height, width = world.shape[:2]
            sample = np.zeros((height, width), dtype=bool)
            sample[:: max(int(point_stride), 1), :: max(int(point_stride), 1)] = True
            valid = np.isfinite(world[..., 2]) & (world[..., 2] > 1.0e-6)
            if dynamic_prob_seq is None or frame_index >= len(dynamic_prob_seq):
                dyn_mask = np.zeros((height, width), dtype=bool)
            else:
                prob = np.asarray(dynamic_prob_seq[frame_index], dtype=np.float32)
                if prob.shape != (height, width):
                    prob = cv2.resize(prob, (width, height), interpolation=cv2.INTER_LINEAR)
                dyn_mask = prob >= float(dynamic_threshold)
            static_xyz = world[sample & valid & (~dyn_mask)]
            dynamic_xyz = world[sample & valid & dyn_mask]
            writer.append_data(
                _render_pointcloud_time_frame(
                    static_xyz,
                    dynamic_xyz,
                    camera_xyz,
                    frame_index,
                    center,
                    radius,
                )
            )
    finally:
        writer.close()
    return output_path


def _iter_frames(video_path: str | Path):
    path = Path(video_path)
    if path.is_dir():
        frame_paths = sorted(
            [item for item in path.iterdir() if item.suffix.lower() in {".jpg", ".jpeg", ".png"}],
            key=lambda item: int(item.stem),
        )
        for frame_path in frame_paths:
            frame_rgb = np.asarray(Image.open(frame_path).convert("RGB"), dtype=np.uint8)
            yield int(frame_path.stem), frame_rgb
        return

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(video_path)
    try:
        frame_index = 0
        while True:
            success, frame_bgr = capture.read()
            if not success:
                break
            yield frame_index, cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_index += 1
    finally:
        capture.release()


def _collect_labeled_world_points(
    da3_seq: object,
    poses: list[torch.Tensor],
    dynamic_prob_seq: np.ndarray | torch.Tensor | None,
    *,
    dynamic_threshold: float,
    point_stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    static_xyz_chunks: list[np.ndarray] = []
    static_rgb_chunks: list[np.ndarray] = []
    dynamic_xyz_chunks: list[np.ndarray] = []
    dynamic_rgb_chunks: list[np.ndarray] = []
    stride = max(int(point_stride), 1)

    if dynamic_prob_seq is not None and torch.is_tensor(dynamic_prob_seq):
        dynamic_prob_seq = dynamic_prob_seq.detach().cpu().numpy()

    for frame_index, frame in enumerate(da3_seq.frames):
        world = backproject(frame.depth, frame.intrinsics, poses[frame_index]).detach().cpu().numpy()
        rgb = _frame_rgb_to_hwc(frame.rgb)
        height, width = world.shape[:2]
        sample_mask = np.zeros((height, width), dtype=bool)
        sample_mask[::stride, ::stride] = True

        if dynamic_prob_seq is None or frame_index >= len(dynamic_prob_seq):
            dyn_mask = np.zeros((height, width), dtype=bool)
        else:
            prob = np.asarray(dynamic_prob_seq[frame_index], dtype=np.float32)
            if prob.shape != (height, width):
                prob = cv2.resize(prob, (width, height), interpolation=cv2.INTER_LINEAR)
            dyn_mask = prob >= float(dynamic_threshold)

        valid_depth = np.isfinite(world[..., 2]) & (world[..., 2] > 1.0e-6)
        static_mask = sample_mask & valid_depth & (~dyn_mask)
        dynamic_mask = sample_mask & valid_depth & dyn_mask

        if static_mask.any():
            static_xyz_chunks.append(world[static_mask])
            static_rgb_chunks.append(rgb[static_mask])
        if dynamic_mask.any():
            dynamic_xyz_chunks.append(world[dynamic_mask])
            dynamic_rgb_chunks.append(rgb[dynamic_mask])

    static_xyz = np.concatenate(static_xyz_chunks, axis=0) if static_xyz_chunks else np.zeros((0, 3), dtype=np.float32)
    static_rgb = np.concatenate(static_rgb_chunks, axis=0) if static_rgb_chunks else np.zeros((0, 3), dtype=np.uint8)
    dynamic_xyz = np.concatenate(dynamic_xyz_chunks, axis=0) if dynamic_xyz_chunks else np.zeros((0, 3), dtype=np.float32)
    dynamic_rgb = np.concatenate(dynamic_rgb_chunks, axis=0) if dynamic_rgb_chunks else np.zeros((0, 3), dtype=np.uint8)
    return static_xyz, static_rgb, dynamic_xyz, dynamic_rgb


def _frame_rgb_to_hwc(rgb: torch.Tensor | np.ndarray) -> np.ndarray:
    if torch.is_tensor(rgb):
        rgb_np = rgb.detach().cpu().numpy()
    else:
        rgb_np = np.asarray(rgb)
    if rgb_np.ndim == 3 and rgb_np.shape[0] == 3:
        rgb_np = np.transpose(rgb_np, (1, 2, 0))
    if rgb_np.dtype != np.uint8:
        rgb_np = np.clip(rgb_np, 0.0, 1.0)
        rgb_np = (rgb_np * 255.0).astype(np.uint8)
    return rgb_np


def _downsample_cloud(xyz: np.ndarray, rgb: np.ndarray, *, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if xyz.shape[0] <= max_points:
        return xyz, rgb
    rng = np.random.default_rng(0)
    keep = rng.choice(xyz.shape[0], size=max_points, replace=False)
    return xyz[keep], rgb[keep]


def _camera_centers_from_w2c(poses: list[torch.Tensor]) -> np.ndarray:
    centers = []
    for pose in poses:
        pose_np = pose.detach().cpu().numpy()
        c2w = np.linalg.inv(pose_np)
        centers.append(c2w[:3, 3])
    return np.asarray(centers, dtype=np.float32)


def _plot_single_cloud_with_cameras(
    output_path: Path,
    points_xyz: np.ndarray,
    points_rgb: np.ndarray,
    camera_xyz: np.ndarray,
    *,
    title: str,
) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    if points_xyz.shape[0] > 0:
        colors = points_rgb.astype(np.float32) / 255.0 if points_rgb.shape[0] else None
        ax.scatter(points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2], c=colors, s=0.7, alpha=0.65, linewidths=0)
    if camera_xyz.shape[0] > 0:
        ax.plot(camera_xyz[:, 0], camera_xyz[:, 1], camera_xyz[:, 2], color="black", linewidth=1.4)
        ax.scatter(camera_xyz[:, 0], camera_xyz[:, 1], camera_xyz[:, 2], c="cyan", s=20, edgecolors="black", linewidths=0.4)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    _set_axes_equal_3d(ax, points_xyz, camera_xyz)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_combined_cloud_with_cameras(
    output_path: Path,
    static_xyz: np.ndarray,
    dynamic_xyz: np.ndarray,
    camera_xyz: np.ndarray,
) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    if static_xyz.shape[0] > 0:
        ax.scatter(static_xyz[:, 0], static_xyz[:, 1], static_xyz[:, 2], c="#48a868", s=0.7, alpha=0.5, linewidths=0, label="static")
    if dynamic_xyz.shape[0] > 0:
        ax.scatter(dynamic_xyz[:, 0], dynamic_xyz[:, 1], dynamic_xyz[:, 2], c="#d94f4f", s=0.7, alpha=0.7, linewidths=0, label="dynamic")
    if camera_xyz.shape[0] > 0:
        ax.plot(camera_xyz[:, 0], camera_xyz[:, 1], camera_xyz[:, 2], color="black", linewidth=1.4, label="camera path")
        ax.scatter(camera_xyz[:, 0], camera_xyz[:, 1], camera_xyz[:, 2], c="cyan", s=20, edgecolors="black", linewidths=0.4)
    ax.set_title("Static/Dynamic Point Clouds + Estimated Cameras")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    _set_axes_equal_3d(ax, static_xyz, dynamic_xyz, camera_xyz)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _set_axes_equal_3d(ax, *clouds: np.ndarray) -> None:
    valid = [cloud for cloud in clouds if cloud is not None and cloud.size > 0]
    if not valid:
        return
    stacked = np.concatenate(valid, axis=0)
    mins = np.nanmin(stacked, axis=0)
    maxs = np.nanmax(stacked, axis=0)
    center = 0.5 * (mins + maxs)
    radius = max(1.0e-3, 0.5 * float(np.max(maxs - mins)))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _estimate_scene_bounds(da3_seq: object, poses: list[torch.Tensor], *, point_stride: int) -> tuple[np.ndarray, float]:
    clouds: list[np.ndarray] = []
    step = max(int(point_stride), 1)
    for frame_index, frame in enumerate(da3_seq.frames):
        world = backproject(frame.depth, frame.intrinsics, poses[frame_index]).detach().cpu().numpy()
        sampled = world[::step, ::step, :].reshape(-1, 3)
        valid = np.isfinite(sampled[:, 2]) & (sampled[:, 2] > 1.0e-6)
        if valid.any():
            clouds.append(sampled[valid])
    camera_xyz = _camera_centers_from_w2c(poses)
    if camera_xyz.size > 0:
        clouds.append(camera_xyz)
    if not clouds:
        return np.zeros(3, dtype=np.float32), 1.0
    stacked = np.concatenate(clouds, axis=0)
    mins = np.nanmin(stacked, axis=0)
    maxs = np.nanmax(stacked, axis=0)
    center = 0.5 * (mins + maxs)
    radius = max(1.0e-3, 0.55 * float(np.max(maxs - mins)))
    return center.astype(np.float32), float(radius)


def _render_pointcloud_time_frame(
    static_xyz: np.ndarray,
    dynamic_xyz: np.ndarray,
    camera_xyz: np.ndarray,
    frame_index: int,
    center: np.ndarray,
    radius: float,
) -> np.ndarray:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    if static_xyz.size > 0:
        ax.scatter(static_xyz[:, 0], static_xyz[:, 1], static_xyz[:, 2], c="#48a868", s=1.2, alpha=0.5, linewidths=0)
    if dynamic_xyz.size > 0:
        ax.scatter(dynamic_xyz[:, 0], dynamic_xyz[:, 1], dynamic_xyz[:, 2], c="#d94f4f", s=1.4, alpha=0.8, linewidths=0)
    if camera_xyz.shape[0] > 0:
        ax.plot(camera_xyz[:, 0], camera_xyz[:, 1], camera_xyz[:, 2], color="black", linewidth=1.2, alpha=0.5)
        ax.scatter(camera_xyz[:, 0], camera_xyz[:, 1], camera_xyz[:, 2], c="lightgray", s=12, linewidths=0)
        current = camera_xyz[min(frame_index, camera_xyz.shape[0] - 1)]
        ax.scatter([current[0]], [current[1]], [current[2]], c="cyan", s=42, edgecolors="black", linewidths=0.6)
    ax.set_title(f"3D Points Over Time (frame {frame_index})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(float(center[0] - radius), float(center[0] + radius))
    ax.set_ylim(float(center[1] - radius), float(center[1] + radius))
    ax.set_zlim(float(center[2] - radius), float(center[2] + radius))
    fig.tight_layout()
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    plt.close(fig)
    return frame


def _write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)
    count = int(xyz.shape[0])
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {count}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for idx in range(count):
            x, y, z = xyz[idx]
            r, g, b = rgb[idx] if idx < rgb.shape[0] else (255, 255, 255)
            handle.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def _blend_color(values: np.ndarray, color: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    return (0.6 * values + 0.4 * color).astype(np.uint8)


def _render_sam_overlay(frame_rgb: np.ndarray, sam_frame: object) -> np.ndarray:
    overlay = frame_rgb.copy()
    masks = sam_frame.masks.detach().cpu().numpy() if hasattr(sam_frame.masks, "detach") else np.asarray(sam_frame.masks)
    masks = masks.astype(bool)
    if masks.ndim == 2:
        masks = masks[None]
    palette = np.array(
        [
            [255, 80, 80],
            [80, 200, 255],
            [255, 210, 80],
            [120, 255, 120],
            [255, 120, 220],
        ],
        dtype=np.uint8,
    )
    for idx, mask in enumerate(masks):
        color = palette[idx % len(palette)]
        overlay[mask] = _blend_color(overlay[mask], color)
        contour_mask = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color.tolist(), 2)
    label = f"frame {sam_frame.frame_index:04d}  objects {len(getattr(sam_frame, 'instance_ids', []))}"
    cv2.putText(overlay, label, (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return overlay


def _render_sam_panel(frame_rgb: np.ndarray, overlay: np.ndarray, sam_frame: object) -> np.ndarray:
    masks = sam_frame.masks.detach().cpu().numpy() if hasattr(sam_frame.masks, "detach") else np.asarray(sam_frame.masks)
    if masks.ndim == 3:
        mask_union = np.any(masks, axis=0)
    else:
        mask_union = masks.astype(bool)
    mask_img = (mask_union.astype(np.uint8) * 255)
    heat = cv2.applyColorMap(mask_img, cv2.COLORMAP_TURBO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    spacer = np.full((frame_rgb.shape[0], 12, 3), 18, dtype=np.uint8)
    return np.concatenate([frame_rgb, spacer, overlay, spacer, heat], axis=1)


def _render_dynamic_only(frame_rgb: np.ndarray, sam_frame: object) -> np.ndarray:
    masks = sam_frame.masks.detach().cpu().numpy() if hasattr(sam_frame.masks, "detach") else np.asarray(sam_frame.masks)
    if masks.ndim == 2:
        mask_union = masks.astype(bool)
    elif masks.ndim == 3:
        mask_union = np.any(masks.astype(bool), axis=0)
    else:
        raise ValueError(f"Unsupported SAM mask shape: {tuple(masks.shape)}")
    output = np.zeros_like(frame_rgb)
    output[mask_union] = frame_rgb[mask_union]
    return output


def _render_dynamic_probability_only(
    frame_rgb: np.ndarray,
    dynamic_prob: np.ndarray,
    threshold: float,
    prob_prev: np.ndarray | None = None,
    prob_next: np.ndarray | None = None,
    neighbor_threshold: float = 0.35,
) -> np.ndarray:
    prob = np.asarray(dynamic_prob, dtype=np.float32)
    if prob.shape != frame_rgb.shape[:2]:
        prob = cv2.resize(prob, (frame_rgb.shape[1], frame_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    prob = np.clip(prob, 0.0, 1.0)
    neighbor_support = np.zeros_like(prob, dtype=np.float32)
    if prob_prev is not None:
        prev = np.asarray(prob_prev, dtype=np.float32)
        if prev.shape != frame_rgb.shape[:2]:
            prev = cv2.resize(prev, (frame_rgb.shape[1], frame_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        neighbor_support = np.maximum(neighbor_support, np.clip(prev, 0.0, 1.0))
    if prob_next is not None:
        nxt = np.asarray(prob_next, dtype=np.float32)
        if nxt.shape != frame_rgb.shape[:2]:
            nxt = cv2.resize(nxt, (frame_rgb.shape[1], frame_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        neighbor_support = np.maximum(neighbor_support, np.clip(nxt, 0.0, 1.0))
    blended_prob = 0.75 * prob + 0.25 * neighbor_support
    mask = blended_prob >= threshold
    mask = mask.astype(np.uint8)
    # Keep small dynamic regions: avoid opening, use a light close only.
    kernel = np.ones((2, 2), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    alpha = np.clip(prob, 0.0, 1.0)[..., None] * mask[..., None].astype(np.float32)
    output = np.zeros_like(frame_rgb, dtype=np.float32)
    output = frame_rgb.astype(np.float32) * alpha
    return np.clip(output, 0, 255).astype(np.uint8)
