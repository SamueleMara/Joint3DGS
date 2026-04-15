from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from dynamic_recon.progress import progress_iter


@dataclass(slots=True)
class _PairObservation:
    target_index: int
    source_index: int
    target_cam_points: torch.Tensor
    target_rgb: torch.Tensor
    target_static_weight: torch.Tensor
    source_static_weight: torch.Tensor
    intrinsics_target: torch.Tensor
    intrinsics_source: torch.Tensor
    source_rgb: torch.Tensor
    source_depth: torch.Tensor


def refine_poses_static_only(
    da3_seq: object,
    dynamic_probabilities: list[torch.Tensor],
    rgb_frames: list[torch.Tensor],
    cfg: object,
) -> list[torch.Tensor]:
    base_poses = [frame.extrinsics.detach().clone() for frame in da3_seq.frames]
    if not getattr(cfg, "enabled", True) or len(base_poses) <= 1:
        return base_poses
    mode = str(getattr(cfg, "camera_mode", "moving")).lower()
    if mode == "fixed":
        return [base_poses[0].clone() for _ in base_poses]
    if mode == "auto" and _looks_like_static_camera_from_poses(base_poses, cfg):
        return [base_poses[0].clone() for _ in base_poses]

    observations = _build_observations(da3_seq, dynamic_probabilities, rgb_frames, cfg)
    if not observations:
        return base_poses

    try:
        return _optimize_poses(observations, base_poses, cfg, device=observations[0].target_cam_points.device)
    except Exception as exc:
        if _should_fallback_to_cpu(exc, cfg, observations[0].target_cam_points.device):
            warnings.warn(
                "Pose refinement failed on CUDA; retrying on CPU. "
                f"Original error: {exc}"
            )
            try:
                return _optimize_poses(observations, base_poses, cfg, device=torch.device("cpu"))
            except Exception as exc_cpu:
                warnings.warn(
                    "Pose refinement failed on CPU fallback; returning unrefined poses. "
                    f"CPU error: {exc_cpu}"
                )
                return base_poses
        raise


def _optimize_poses(
    observations: list[_PairObservation],
    base_poses: list[torch.Tensor],
    cfg: object,
    *,
    device: torch.device,
) -> list[torch.Tensor]:
    base_poses_device = [pose.detach().clone().to(device) for pose in base_poses]
    observations_device = [_observation_to_device(obs, device) for obs in observations]
    pose_deltas = torch.zeros((len(base_poses_device), 6), dtype=base_poses_device[0].dtype, device=device, requires_grad=True)
    lr = float(cfg.lr)
    momentum = torch.zeros_like(pose_deltas)
    beta = 0.9
    clip = 0.05

    if device.type == "cuda":
        torch.cuda.synchronize(device=device)

    for _ in progress_iter(range(int(cfg.steps)), desc="Pose refine", total=int(cfg.steps)):
        if pose_deltas.grad is not None:
            pose_deltas.grad = None
        loss = _compute_loss(observations_device, base_poses_device, pose_deltas, cfg)
        if not torch.isfinite(loss) or not loss.requires_grad:
            break
        loss.backward()
        grad = pose_deltas.grad
        if grad is None or not torch.isfinite(grad).all():
            break
        with torch.no_grad():
            grad = torch.clamp(grad, min=-clip, max=clip)
            momentum.mul_(beta).add_(grad, alpha=1.0 - beta)
            pose_deltas.add_(momentum, alpha=-lr)

    refined_device = [_apply_pose_delta(pose, pose_deltas[index].detach()) for index, pose in enumerate(base_poses_device)]
    refined = [pose.to(base_poses[index].device) for index, pose in enumerate(refined_device)]
    return refined


def _observation_to_device(obs: _PairObservation, device: torch.device) -> _PairObservation:
    return _PairObservation(
        target_index=obs.target_index,
        source_index=obs.source_index,
        target_cam_points=obs.target_cam_points.to(device),
        target_rgb=obs.target_rgb.to(device),
        target_static_weight=obs.target_static_weight.to(device),
        source_static_weight=obs.source_static_weight.to(device),
        intrinsics_target=obs.intrinsics_target.to(device),
        intrinsics_source=obs.intrinsics_source.to(device),
        source_rgb=obs.source_rgb.to(device),
        source_depth=obs.source_depth.to(device),
    )


def _should_fallback_to_cpu(exc: Exception, cfg: object, device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    if not bool(getattr(cfg, "fallback_to_cpu_on_cuda_error", True)):
        return False
    message = str(exc).lower()
    cuda_signatures = (
        "cuda error",
        "unspecified launch failure",
        "illegal memory access",
        "device-side assert",
        "cublas",
        "cudnn",
    )
    return any(token in message for token in cuda_signatures)


def _build_observations(
    da3_seq: object,
    dynamic_probabilities: list[torch.Tensor],
    rgb_frames: list[torch.Tensor],
    cfg: object,
) -> list[_PairObservation]:
    observations: list[_PairObservation] = []
    frame_indices = list(range(len(da3_seq.frames)))
    if getattr(cfg, "keyframe_only", False):
        stride = max(int(cfg.keyframe_stride), 1)
        frame_indices = list(range(0, len(da3_seq.frames), stride))
        if frame_indices[-1] != len(da3_seq.frames) - 1:
            frame_indices.append(len(da3_seq.frames) - 1)

    for target_index in frame_indices:
        target_frame = da3_seq.frames[target_index]
        target_rgb = rgb_frames[target_index]
        target_dynamic = dynamic_probabilities[target_index].detach()
        static_mask = target_dynamic <= 0.5
        points_yx = _sample_static_points(static_mask)
        if points_yx is None:
            continue
        target_cam_points, target_rgb_samples, target_weights = _lift_target_camera_points(
            points_yx,
            target_frame.depth.detach(),
            target_frame.intrinsics.detach(),
            target_rgb.detach(),
            1.0 - target_dynamic,
        )
        if target_cam_points.numel() == 0:
            continue
        for source_index in _select_neighbors(target_index, len(da3_seq.frames), cfg):
            source_dynamic = dynamic_probabilities[source_index].detach()
            source_static = _sample_map_at_integer_points(1.0 - source_dynamic, points_yx)
            source_static = torch.clamp(source_static, min=0.0, max=1.0)
            observations.append(
                _PairObservation(
                    target_index=target_index,
                    source_index=source_index,
                    target_cam_points=target_cam_points,
                    target_rgb=target_rgb_samples,
                    target_static_weight=target_weights,
                    source_static_weight=source_static,
                    intrinsics_target=target_frame.intrinsics.detach(),
                    intrinsics_source=da3_seq.frames[source_index].intrinsics.detach(),
                    source_rgb=rgb_frames[source_index].detach(),
                    source_depth=da3_seq.frames[source_index].depth.detach(),
                )
            )
    return observations


def _compute_loss(
    observations: list[_PairObservation],
    base_poses: list[torch.Tensor],
    pose_deltas: torch.Tensor,
    cfg: object,
) -> torch.Tensor:
    total = pose_deltas.new_tensor(0.0)
    valid_terms = 0
    for obs in observations:
        target_pose = _apply_pose_delta(base_poses[obs.target_index], pose_deltas[obs.target_index])
        source_pose = _apply_pose_delta(base_poses[obs.source_index], pose_deltas[obs.source_index])
        world_points = _target_cam_to_world(obs.target_cam_points, target_pose)
        projected_uv, projected_depth = _project_world_points(world_points, obs.intrinsics_source, source_pose)
        sampled_depth, valid_depth = _sample_points(obs.source_depth, projected_uv)
        sampled_rgb, valid_rgb = _sample_points(obs.source_rgb, projected_uv)
        valid = valid_depth & valid_rgb & (projected_depth > 1.0e-6)
        if not torch.any(valid):
            continue
        weights = obs.target_static_weight * obs.source_static_weight * valid.float()
        denom = torch.clamp(weights.sum(), min=1.0e-6)
        depth_loss = ((sampled_depth - projected_depth).abs() * weights).sum() / denom
        rgb_loss = ((sampled_rgb - obs.target_rgb).abs().mean(dim=-1) * weights).sum() / denom
        total = total + float(getattr(cfg, "lambda_depth", 0.5)) * depth_loss + float(getattr(cfg, "lambda_rgb", 0.1)) * rgb_loss
        valid_terms += 1
    if valid_terms == 0:
        return pose_deltas.new_tensor(0.0)
    loss = total / valid_terms
    lambda_anchor = float(getattr(cfg, "lambda_pose_anchor", 1.0e-3))
    lambda_temporal_t = float(getattr(cfg, "lambda_temporal_translation", 5.0e-3))
    lambda_temporal_r = float(getattr(cfg, "lambda_temporal_rotation", 2.0e-3))
    if lambda_anchor > 0.0:
        loss = loss + lambda_anchor * pose_deltas.square().mean()
    if pose_deltas.shape[0] > 1:
        if lambda_temporal_t > 0.0:
            delta_t = pose_deltas[1:, :3] - pose_deltas[:-1, :3]
            loss = loss + lambda_temporal_t * delta_t.square().mean()
        if lambda_temporal_r > 0.0:
            delta_r = pose_deltas[1:, 3:] - pose_deltas[:-1, 3:]
            loss = loss + lambda_temporal_r * delta_r.square().mean()
    return loss


def _sample_static_points(static_mask: torch.Tensor, max_points: int = 1024) -> torch.Tensor | None:
    coords = torch.nonzero(static_mask, as_tuple=False)
    if coords.numel() == 0:
        return None
    if coords.shape[0] > max_points:
        step = max(coords.shape[0] // max_points, 1)
        coords = coords[::step][:max_points]
    return coords


def _lift_target_camera_points(
    points_yx: torch.Tensor,
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    rgb: torch.Tensor,
    static_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ys = points_yx[:, 0].long()
    xs = points_yx[:, 1].long()
    depth_samples = depth[ys, xs]
    valid = depth_samples > 1.0e-6
    if not torch.any(valid):
        empty = depth_samples[:0]
        return empty[:, None], empty[:, None], empty
    ys = ys[valid]
    xs = xs[valid]
    depth_samples = depth_samples[valid]
    pixels = torch.stack([xs.float(), ys.float(), torch.ones_like(xs, dtype=torch.float32)], dim=-1).to(depth_samples)
    cam_points = (torch.linalg.inv(intrinsics) @ pixels.T).T * depth_samples.unsqueeze(-1)
    rgb_hwc = rgb.permute(1, 2, 0) if rgb.ndim == 3 and rgb.shape[0] == 3 else rgb
    rgb_samples = rgb_hwc[ys, xs]
    weight_samples = torch.clamp(static_weight[ys, xs], min=0.0, max=1.0)
    return cam_points, rgb_samples, weight_samples


def _target_cam_to_world(cam_points: torch.Tensor, target_pose_w2c: torch.Tensor) -> torch.Tensor:
    cam_h = torch.cat([cam_points, torch.ones_like(cam_points[:, :1])], dim=-1)
    c2w = torch.linalg.inv(target_pose_w2c)
    world_h = (c2w @ cam_h.T).T
    return world_h[:, :3] / torch.clamp(world_h[:, 3:], min=1.0e-8)


def _project_world_points(world_points: torch.Tensor, intrinsics: torch.Tensor, extrinsics: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    world_h = torch.cat([world_points, torch.ones_like(world_points[:, :1])], dim=-1)
    cam_h = (extrinsics @ world_h.T).T
    cam = cam_h[:, :3] / torch.clamp(cam_h[:, 3:], min=1.0e-8)
    depth = cam[:, 2]
    uvw = (intrinsics @ cam.T).T
    uv = uvw[:, :2] / torch.clamp(depth.unsqueeze(-1), min=1.0e-6)
    return uv, depth


def _apply_pose_delta(extrinsics: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    translation = delta[:3]
    rotvec = delta[3:]
    rotation = _rodrigues(rotvec)
    upper = torch.cat([rotation, translation[:, None]], dim=1)
    bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=extrinsics.dtype, device=extrinsics.device)
    transform = torch.cat([upper, bottom], dim=0)
    return transform @ extrinsics


def _rodrigues(rotvec: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(rotvec)
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    if float(theta.detach().cpu()) < 1.0e-8:
        return eye + _skew(rotvec)
    axis = rotvec / theta
    K = _skew(axis)
    return eye + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)


def _skew(vector: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros((), dtype=vector.dtype, device=vector.device)
    row0 = torch.stack([zero, -vector[2], vector[1]])
    row1 = torch.stack([vector[2], zero, -vector[0]])
    row2 = torch.stack([-vector[1], vector[0], zero])
    return torch.stack([row0, row1, row2], dim=0)


def _select_neighbors(frame_index: int, num_frames: int, cfg: object) -> list[int]:
    stride = max(int(getattr(cfg, "keyframe_stride", 1)), 1)
    candidates = {frame_index - 1, frame_index + 1, frame_index - stride, frame_index + stride}
    return sorted(index for index in candidates if 0 <= index < num_frames and index != frame_index)


def _sample_map_at_integer_points(values: torch.Tensor, points_yx: torch.Tensor) -> torch.Tensor:
    ys = points_yx[:, 0].long()
    xs = points_yx[:, 1].long()
    return values[ys, xs]


def _sample_points(values: torch.Tensor, uv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if values.ndim == 2:
        source = values[None, None]
    elif values.ndim == 3:
        source = values[None]
    else:
        raise ValueError(f"Unsupported sample source shape: {tuple(values.shape)}")
    height, width = values.shape[-2:]
    norm_x = (uv[:, 0] / max(width - 1, 1)) * 2 - 1
    norm_y = (uv[:, 1] / max(height - 1, 1)) * 2 - 1
    grid = torch.stack([norm_x, norm_y], dim=-1).view(1, 1, -1, 2)
    sampled = F.grid_sample(source.float(), grid.float(), align_corners=True, mode="bilinear", padding_mode="zeros")
    valid = (norm_x >= -1) & (norm_x <= 1) & (norm_y >= -1) & (norm_y <= 1)
    if values.ndim == 2:
        return sampled[0, 0, 0], valid
    return sampled[0, :, 0].permute(1, 0), valid


def _looks_like_static_camera_from_poses(poses: list[torch.Tensor], cfg: object) -> bool:
    if len(poses) <= 1:
        return True
    trans: list[float] = []
    rot_deg: list[float] = []
    for idx in range(1, len(poses)):
        relative = poses[idx] @ torch.linalg.inv(poses[idx - 1])
        trans.append(float(torch.linalg.norm(relative[:3, 3]).detach().cpu()))
        rot_deg.append(_rotation_angle_deg(relative[:3, :3]))
    if not trans or not rot_deg:
        return False
    trans_med = float(np.median(np.asarray(trans, dtype=np.float32)))
    rot_med = float(np.median(np.asarray(rot_deg, dtype=np.float32)))
    trans_thr = float(getattr(cfg, "auto_static_translation_thresh", 1.0e-3))
    rot_thr = float(getattr(cfg, "auto_static_rotation_thresh_deg", 0.1))
    return trans_med <= trans_thr and rot_med <= rot_thr


def _rotation_angle_deg(rotation: torch.Tensor) -> float:
    trace = float(torch.trace(rotation).detach().cpu())
    cos_theta = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
    return float(np.degrees(np.arccos(cos_theta)))
