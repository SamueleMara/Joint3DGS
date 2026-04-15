from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import torch


def initialize_pose_sequence(da3_seq: object, cfg: object) -> object:
    if not getattr(cfg, "enabled", True) or len(da3_seq.frames) <= 1:
        return da3_seq
    base_poses = [frame.extrinsics.detach().clone() for frame in da3_seq.frames]
    mode = str(getattr(cfg, "camera_mode", "moving")).lower()
    if mode == "fixed":
        return _constant_pose_sequence(da3_seq, base_poses[0], source="pose_init_fixed")
    if mode == "auto" and _looks_like_static_camera(da3_seq, cfg):
        return _constant_pose_sequence(da3_seq, base_poses[0], source="pose_init_auto_static")

    poses = [base_poses[0].clone()]
    for index in range(len(da3_seq.frames) - 1):
        prev_frame = da3_seq.frames[index]
        next_frame = da3_seq.frames[index + 1]
        relative = _estimate_relative_pose(prev_frame, next_frame, cfg)
        if relative is None:
            fallback = next_frame.extrinsics.detach().clone()
            poses.append(fallback)
            continue
        poses.append(relative @ poses[-1])
    frames = []
    for frame, pose in zip(da3_seq.frames, poses):
        aux = dict(frame.aux)
        aux["pose_source"] = "cv2_pose_init"
        frames.append(replace(frame, extrinsics=pose, aux=aux))
    return replace(da3_seq, frames=frames)


def _constant_pose_sequence(da3_seq: object, pose: torch.Tensor, *, source: str) -> object:
    frames = []
    for frame in da3_seq.frames:
        aux = dict(frame.aux)
        aux["pose_source"] = source
        frames.append(replace(frame, extrinsics=pose.clone(), aux=aux))
    return replace(da3_seq, frames=frames)


def _looks_like_static_camera(da3_seq: object, cfg: object) -> bool:
    frames = da3_seq.frames
    if len(frames) <= 1:
        return True
    check_pairs = max(1, min(int(getattr(cfg, "static_check_pairs", 8)), len(frames) - 1))
    medians: list[float] = []
    for idx in range(check_pairs):
        flow_med = _estimate_median_pixel_flow(frames[idx], frames[idx + 1], cfg)
        if flow_med is not None and np.isfinite(flow_med):
            medians.append(float(flow_med))
    if not medians:
        return False
    return float(np.median(np.asarray(medians, dtype=np.float32))) <= float(getattr(cfg, "static_motion_median_px", 0.4))


def _estimate_median_pixel_flow(prev_frame: object, next_frame: object, cfg: object) -> float | None:
    prev_rgb = _to_gray(prev_frame.rgb)
    next_rgb = _to_gray(next_frame.rgb)
    prev_points = cv2.goodFeaturesToTrack(
        prev_rgb,
        maxCorners=int(getattr(cfg, "max_corners", 2000)),
        qualityLevel=float(getattr(cfg, "quality_level", 0.01)),
        minDistance=float(getattr(cfg, "min_distance", 7.0)),
    )
    if prev_points is None or len(prev_points) < 32:
        return None
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_rgb, next_rgb, prev_points, None)
    if next_points is None or status is None:
        return None
    prev_points = prev_points.reshape(-1, 2)
    next_points = next_points.reshape(-1, 2)
    status = status.reshape(-1).astype(bool)
    prev_points = prev_points[status]
    next_points = next_points[status]
    if prev_points.shape[0] < 32:
        return None
    disp = np.linalg.norm(next_points - prev_points, axis=1)
    if disp.size == 0:
        return None
    return float(np.median(disp))


def _estimate_relative_pose(prev_frame: object, next_frame: object, cfg: object) -> torch.Tensor | None:
    prev_rgb = _to_gray(prev_frame.rgb)
    next_rgb = _to_gray(next_frame.rgb)
    prev_points = cv2.goodFeaturesToTrack(
        prev_rgb,
        maxCorners=int(getattr(cfg, "max_corners", 2000)),
        qualityLevel=float(getattr(cfg, "quality_level", 0.01)),
        minDistance=float(getattr(cfg, "min_distance", 7.0)),
    )
    if prev_points is None or len(prev_points) < 16:
        return None
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_rgb, next_rgb, prev_points, None)
    if next_points is None or status is None:
        return None
    prev_points = prev_points.reshape(-1, 2)
    next_points = next_points.reshape(-1, 2)
    status = status.reshape(-1).astype(bool)
    prev_points = prev_points[status]
    next_points = next_points[status]
    if prev_points.shape[0] < 16:
        return None

    object_points, image_points = _build_pnp_correspondences(prev_points, next_points, prev_frame.depth, prev_frame.intrinsics)
    if object_points.shape[0] >= int(getattr(cfg, "min_inliers", 32)):
        pose = _solve_pnp(object_points, image_points, next_frame.intrinsics, cfg)
        if pose is not None:
            return pose.to(prev_frame.extrinsics)

    pose = _solve_essential(prev_points, next_points, prev_frame.intrinsics, prev_frame.depth)
    if pose is not None:
        return pose.to(prev_frame.extrinsics)
    return None


def _build_pnp_correspondences(
    prev_points: np.ndarray,
    next_points: np.ndarray,
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth.shape
    xs = np.clip(np.round(prev_points[:, 0]).astype(np.int32), 0, width - 1)
    ys = np.clip(np.round(prev_points[:, 1]).astype(np.int32), 0, height - 1)
    depth_np = depth.detach().cpu().numpy()
    depths = depth_np[ys, xs]
    valid = np.isfinite(depths) & (depths > 1.0e-6)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
    prev_points = prev_points[valid]
    next_points = next_points[valid]
    depths = depths[valid]
    k_inv = np.linalg.inv(intrinsics.detach().cpu().numpy())
    pixels = np.concatenate([prev_points, np.ones((prev_points.shape[0], 1), dtype=np.float32)], axis=1)
    cam_points = (k_inv @ pixels.T).T * depths[:, None]
    return cam_points.astype(np.float32), next_points.astype(np.float32)


def _solve_pnp(object_points: np.ndarray, image_points: np.ndarray, intrinsics: torch.Tensor, cfg: object) -> torch.Tensor | None:
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points,
        image_points,
        intrinsics.detach().cpu().numpy().astype(np.float32),
        None,
        reprojectionError=float(getattr(cfg, "pnp_reprojection_error", 3.0)),
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success or inliers is None or len(inliers) < int(getattr(cfg, "min_inliers", 32)):
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = rotation.astype(np.float32)
    pose[:3, 3] = tvec.reshape(3).astype(np.float32)
    return torch.from_numpy(pose)


def _solve_essential(prev_points: np.ndarray, next_points: np.ndarray, intrinsics: torch.Tensor, depth: torch.Tensor) -> torch.Tensor | None:
    k = intrinsics.detach().cpu().numpy().astype(np.float64)
    essential, mask = cv2.findEssentialMat(prev_points, next_points, k, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if essential is None:
        return None
    _, rotation, translation, mask_pose = cv2.recoverPose(essential, prev_points, next_points, k)
    if mask_pose is None or int(mask_pose.sum()) < 16:
        return None
    scale = _estimate_translation_scale(prev_points, depth)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = rotation.astype(np.float32)
    pose[:3, 3] = (translation.reshape(3) * scale).astype(np.float32)
    return torch.from_numpy(pose)


def _estimate_translation_scale(points: np.ndarray, depth: torch.Tensor) -> float:
    height, width = depth.shape
    xs = np.clip(np.round(points[:, 0]).astype(np.int32), 0, width - 1)
    ys = np.clip(np.round(points[:, 1]).astype(np.int32), 0, height - 1)
    depth_np = depth.detach().cpu().numpy()
    values = depth_np[ys, xs]
    values = values[np.isfinite(values) & (values > 1.0e-6)]
    if values.size == 0:
        return 0.05
    return float(np.median(values) * 0.05)


def _to_gray(rgb: torch.Tensor | None) -> np.ndarray:
    if rgb is None:
        raise ValueError("Pose initialization requires RGB frames")
    image = rgb.detach().cpu()
    if image.ndim == 3 and image.shape[0] == 3:
        image = image.permute(1, 2, 0)
    array = image.numpy()
    array = np.clip(array * 255.0 if array.max() <= 1.5 else array, 0, 255).astype(np.uint8)
    return cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
