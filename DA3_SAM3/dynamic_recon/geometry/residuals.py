from __future__ import annotations

import torch

from dynamic_recon.flow import compute_dense_flow_pair, flow_forward_backward_consistency
from dynamic_recon.geometry.dynamic_prior import PairResiduals
from dynamic_recon.geometry.projection import backproject, project, sample_at_coords
from dynamic_recon.geometry.visibility import forward_backward_consistency, inside_image_mask, occlusion_mask, positive_depth_mask


def compute_pair_residuals(frame_t: torch.Tensor, frame_s: torch.Tensor, da3_t: object, da3_s: object, cfg: object) -> PairResiduals:
    rgb_t = frame_t.permute(1, 2, 0) if frame_t.ndim == 3 and frame_t.shape[0] == 3 else frame_t
    rgb_s = frame_s.permute(1, 2, 0) if frame_s.ndim == 3 and frame_s.shape[0] == 3 else frame_s
    world_t = backproject(da3_t.depth, da3_t.intrinsics, da3_t.extrinsics)
    uv_ts, z_ts = project(world_t, da3_s.intrinsics, da3_s.extrinsics)
    sampled_depth_s, valid_s = sample_at_coords(da3_s.depth, uv_ts)
    sampled_rgb_s, _ = sample_at_coords(rgb_s, uv_ts)
    world_from_s = backproject(sampled_depth_s, da3_s.intrinsics, da3_s.extrinsics)
    r_3d = torch.linalg.norm(world_from_s - world_t, dim=-1)
    r_rel = _relative_distance_change(world_t, world_from_s)
    r_depth = torch.abs(sampled_depth_s - z_ts)
    r_rgb = torch.abs(sampled_rgb_s - rgb_t).mean(dim=-1)
    uv_st, _ = project(world_from_s, da3_t.intrinsics, da3_t.extrinsics)
    height, width = da3_t.depth.shape
    yy, xx = torch.meshgrid(torch.arange(height, device=da3_t.depth.device), torch.arange(width, device=da3_t.depth.device), indexing="ij")
    original_uv = torch.stack([xx.float(), yy.float()], dim=-1)
    rigid_flow = uv_ts - original_uv
    observed_flow_fw = compute_dense_flow_pair(rgb_t, rgb_s).to(da3_t.depth.device)
    observed_flow_bw = compute_dense_flow_pair(rgb_s, rgb_t).to(da3_t.depth.device)
    flow_consistency = flow_forward_backward_consistency(observed_flow_fw, observed_flow_bw, cfg.flow_consistency_thresh)
    r_flow = torch.linalg.norm(observed_flow_fw - rigid_flow, dim=-1)
    r_cycle = torch.linalg.norm(uv_st - original_uv, dim=-1)
    visible = inside_image_mask(uv_ts, height, width).float()
    visible *= positive_depth_mask(z_ts).float()
    visible *= (~occlusion_mask(z_ts, sampled_depth_s, cfg.occlusion_abs_tol, cfg.occlusion_rel_tol)).float()
    visible *= valid_s.float()
    visible *= forward_backward_consistency(uv_ts, uv_st)
    visible *= flow_consistency
    return PairResiduals(uv_ts=uv_ts, z_ts=z_ts, r_3d=r_3d, r_rel=r_rel, r_flow=r_flow, r_depth=r_depth, r_rgb=r_rgb, r_cycle=r_cycle, visibility=visible)


def aggregate_pair_residuals(pairs: list[PairResiduals]) -> PairResiduals:
    if not pairs:
        raise ValueError("No pair residuals to aggregate")
    weights = torch.stack([torch.clamp(pair.visibility, min=1.0e-6) for pair in pairs], dim=0)
    norm = weights.sum(dim=0)

    def _reduce(name: str) -> torch.Tensor:
        stacked = torch.stack([getattr(pair, name) for pair in pairs], dim=0)
        return (stacked * weights).sum(dim=0) / norm

    return PairResiduals(
        uv_ts=torch.stack([pair.uv_ts for pair in pairs], dim=0).mean(dim=0),
        z_ts=torch.stack([pair.z_ts for pair in pairs], dim=0).mean(dim=0),
        r_3d=_reduce("r_3d"),
        r_rel=_reduce("r_rel"),
        r_flow=_reduce("r_flow"),
        r_depth=_reduce("r_depth"),
        r_rgb=_reduce("r_rgb"),
        r_cycle=_reduce("r_cycle"),
        visibility=torch.stack([pair.visibility for pair in pairs], dim=0).amax(dim=0),
    )


def compute_sequence_residuals(sequence_output: object, cfg: object) -> list[PairResiduals]:
    frames = sequence_output.frames
    outputs: list[PairResiduals] = []
    for frame_index, frame in enumerate(frames):
        pair_residuals: list[PairResiduals] = []
        for offset in getattr(cfg, "pair_offsets", [-1, 1]):
            neighbor_index = frame_index + int(offset)
            if neighbor_index < 0 or neighbor_index >= len(frames) or neighbor_index == frame_index:
                continue
            neighbor = frames[neighbor_index]
            pair_residuals.append(compute_pair_residuals(frame.rgb, neighbor.rgb, frame, neighbor, cfg))
        if pair_residuals:
            outputs.append(aggregate_pair_residuals(pair_residuals))
            continue

        height, width = frame.depth.shape
        zeros = torch.zeros((height, width), device=frame.depth.device, dtype=frame.depth.dtype)
        uv = torch.zeros((height, width, 2), device=frame.depth.device, dtype=frame.depth.dtype)
        outputs.append(
            PairResiduals(
                uv_ts=uv,
                z_ts=zeros,
                r_3d=zeros,
                r_rel=zeros,
                r_flow=zeros,
                r_depth=zeros,
                r_rgb=zeros,
                r_cycle=zeros,
                visibility=torch.ones_like(zeros),
            )
        )
    return outputs


def _relative_distance_change(world_t: torch.Tensor, world_from_s: torch.Tensor) -> torch.Tensor:
    height, width, _ = world_t.shape
    device = world_t.device
    dtype = world_t.dtype
    if height == 0 or width == 0:
        return torch.zeros((height, width), device=device, dtype=dtype)

    if width > 1:
        dist_t_x = torch.linalg.norm(world_t[:, 1:, :] - world_t[:, :-1, :], dim=-1)
        dist_s_x = torch.linalg.norm(world_from_s[:, 1:, :] - world_from_s[:, :-1, :], dim=-1)
        err_x = torch.abs(dist_t_x - dist_s_x)
        # Replicate the last valid column to recover [H, W].
        err_x = torch.cat([err_x, err_x[:, -1:].clone()], dim=1)
    else:
        err_x = torch.zeros((height, width), device=device, dtype=dtype)

    if height > 1:
        dist_t_y = torch.linalg.norm(world_t[1:, :, :] - world_t[:-1, :, :], dim=-1)
        dist_s_y = torch.linalg.norm(world_from_s[1:, :, :] - world_from_s[:-1, :, :], dim=-1)
        err_y = torch.abs(dist_t_y - dist_s_y)
        # Replicate the last valid row to recover [H, W].
        err_y = torch.cat([err_y, err_y[-1:, :].clone()], dim=0)
    else:
        err_y = torch.zeros((height, width), device=device, dtype=dtype)

    return 0.5 * (err_x + err_y)
