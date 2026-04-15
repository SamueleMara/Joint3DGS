from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from dataclasses import replace

from dynamic_recon.config.schema import PipelineConfig
from dynamic_recon.fusion.features import build_fusion_features
from dynamic_recon.fusion.model import DynamicFusionNet
from dynamic_recon.fusion.trainer import train_one_epoch
from dynamic_recon.geometry.dynamic_prior import build_dynamic_prior, temporal_smooth_priors
from dynamic_recon.geometry.projection import backproject, project, sample_at_coords
from dynamic_recon.geometry.pose_refine import refine_poses_static_only
from dynamic_recon.geometry.residuals import compute_sequence_residuals
from dynamic_recon.pipelines.infer_initial import run_initial_pass, run_segmentation_pass
from dynamic_recon.progress import progress_iter, stage_bar


def run_preliminary_pipeline(video_path: str | Path, outdir: str | Path, cfg: PipelineConfig) -> dict[str, object]:
    state = run_initial_pass(video_path, outdir, cfg)
    poses = [frame.extrinsics.clone() for frame in state["da3_seq"].frames]
    model: DynamicFusionNet | None = None
    dyn_prob_seq: torch.Tensor | None = None
    working_seq = _clone_sequence_with_poses(state["da3_seq"], poses)
    stage = stage_bar(desc="Outer optimization", total=cfg.pipeline.num_outer_iters)
    for outer_index in progress_iter(range(cfg.pipeline.num_outer_iters), desc="Outer iters", total=cfg.pipeline.num_outer_iters):
        sam_by_frame = {int(frame.frame_index): frame for frame in state["sam3_out"].frames}
        batches: list[dict[str, torch.Tensor]] = []
        ordered_indices: list[int] = []
        ordered_features: list[torch.Tensor] = []
        for frame_index, residual_frame in progress_iter(
            list(enumerate(working_seq.frames)),
            desc=f"Build fusion batches iter {outer_index + 1}",
            total=len(working_seq.frames),
        ):
            sam_frame = sam_by_frame.get(frame_index)
            if sam_frame is None:
                continue
            residual_bundle = state["residual_seq"][frame_index]
            features = build_fusion_features(residual_frame.rgb, residual_frame, residual_bundle, sam_frame)
            if model is None:
                model = DynamicFusionNet(in_channels=features.shape[0], base_channels=cfg.fusion.base_channels)
            target, weight = _build_fusion_supervision(
                state["dynamic_priors"][frame_index],
                residual_bundle,
                sam_frame,
                residual_frame.depth.shape,
                cfg.fusion,
            )
            batch = {
                "x": features.unsqueeze(0),
                "target": target.unsqueeze(0).unsqueeze(0),
                "weight": weight.unsqueeze(0).unsqueeze(0),
                "image": residual_frame.rgb.unsqueeze(0),
                "motion_abs": _normalize_map(residual_bundle.r_3d).unsqueeze(0).unsqueeze(0),
                "motion_rel": _normalize_map(residual_bundle.r_rel).unsqueeze(0).unsqueeze(0),
            }
            batch.update(
                _build_pairwise_3d_supervision(
                    frame_index,
                    working_seq.frames,
                    residual_bundle,
                    cfg.geometry,
                    cfg.fusion,
                )
            )
            batches.append(batch)
            ordered_indices.append(frame_index)
            ordered_features.append(features)
        if model is None or not batches:
            raise RuntimeError("No aligned SAM outputs were available for sequence fusion.")
        for _ in progress_iter(range(cfg.fusion.epochs_per_outer_iter), desc=f"Fusion epochs iter {outer_index + 1}", total=cfg.fusion.epochs_per_outer_iter):
            model = train_one_epoch(
                model,
                batches,
                cfg.fusion.lr,
                w_tv=float(getattr(cfg.fusion, "w_tv", 0.01)),
                w_abs_motion=float(getattr(cfg.fusion, "w_abs_motion", 0.4)),
                w_motion_rank=float(getattr(cfg.fusion, "w_motion_rank", 0.2)),
                w_contrastive=float(getattr(cfg.fusion, "w_contrastive", 0.25)),
                w_pair_3d_contrastive=float(getattr(cfg.fusion, "w_pair_3d_contrastive", 0.35)),
                contrastive_pairs=int(getattr(cfg.fusion, "contrastive_pairs", 1024)),
                contrastive_neighbor_radius=int(getattr(cfg.fusion, "contrastive_neighbor_radius", 6)),
                contrastive_beta=float(getattr(cfg.fusion, "contrastive_beta", 8.0)),
            )
        dyn_prob_frames: list[torch.Tensor] = []
        feature_map = {frame_index: feature for frame_index, feature in zip(ordered_indices, ordered_features)}
        for frame_index, residual_frame in progress_iter(
            list(enumerate(state["da3_seq"].frames)),
            desc=f"Predict fused masks iter {outer_index + 1}",
            total=len(state["da3_seq"].frames),
        ):
            feature = feature_map.get(frame_index)
            if feature is None:
                dyn_prob_frames.append(state["dynamic_priors"][frame_index].detach().clone())
                continue
            dyn_prob_frames.append(torch.sigmoid(model(feature.unsqueeze(0))["dynamic_logit"])[0, 0])
        dyn_prob_seq = _temporal_smooth_probabilities(torch.stack(dyn_prob_frames, dim=0), alpha=0.25)
        dyn_prob_seq = _backward_propagate_dynamic(
            dyn_prob_seq,
            decay=float(getattr(cfg.fusion, "backward_propagation_decay", 0.92)),
            iters=int(getattr(cfg.fusion, "backward_propagation_iters", 2)),
        )
        poses = refine_poses_static_only(
            working_seq,
            list(dyn_prob_seq),
            [frame.rgb for frame in working_seq.frames],
            cfg.pose_refine,
        )
        working_seq = _clone_sequence_with_poses(working_seq, poses)
        residual_seq = compute_sequence_residuals(working_seq, cfg.geometry)
        dynamic_priors = [build_dynamic_prior(pair, cfg.geometry) for pair in residual_seq]
        dynamic_priors = temporal_smooth_priors(dynamic_priors, cfg.geometry.temporal_smoothing_alpha)
        state["residual_seq"] = residual_seq
        state["dynamic_priors"] = dynamic_priors
        state["dynamic_prior"] = dynamic_priors[0]
        if cfg.pipeline.rerun_segmentation_each_iter and outer_index + 1 < cfg.pipeline.num_outer_iters:
            state = run_segmentation_pass(state, outdir, cfg, tag=f"iter_{outer_index + 1:02d}")
        stage.update(1)
    stage.close()
    return {
        **state,
        "config": cfg,
        "da3_seq": working_seq,
        "poses": poses,
        "dynamic_prob": dyn_prob_seq,
        "dynamic_prob_seq": dyn_prob_seq,
    }


def _resize_spatial_tensor(tensor: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tensor.ndim == 2:
        return F.interpolate(tensor.unsqueeze(0).unsqueeze(0), size=size, mode="bilinear", align_corners=False)[0, 0]
    if tensor.ndim == 3:
        return F.interpolate(tensor.unsqueeze(0), size=size, mode="bilinear", align_corners=False)[0]
    raise ValueError(f"Unsupported tensor shape for resize: {tuple(tensor.shape)}")


def _union_mask(masks: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    if masks.ndim == 2:
        union = masks.float()
    elif masks.ndim == 3:
        if masks.shape[0] == 0:
            union = torch.zeros(target_size, dtype=torch.float32, device=masks.device)
        else:
            union = masks.float().amax(dim=0)
    else:
        raise ValueError(f"Unsupported SAM mask shape: {tuple(masks.shape)}")
    if union.shape != target_size:
        union = _resize_spatial_tensor(union, target_size)
    return union


def _clone_sequence_with_poses(sequence: object, poses: list[torch.Tensor]) -> object:
    frames = [replace(frame, extrinsics=pose.clone()) for frame, pose in zip(sequence.frames, poses)]
    return replace(sequence, frames=frames)


def _build_fusion_supervision(
    dynamic_prior: torch.Tensor,
    residual_bundle: object,
    sam_frame: object,
    target_size: tuple[int, int],
    fusion_cfg: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    sam_union = _union_mask(sam_frame.masks, target_size)
    geo = _resize_spatial_tensor(dynamic_prior.float(), target_size)
    vis = _resize_spatial_tensor(residual_bundle.visibility.float(), target_size)
    dyn_geo_high = float(getattr(fusion_cfg, "dynamic_geo_high", 0.75))
    dyn_sam_high = float(getattr(fusion_cfg, "dynamic_sam_high", 0.7))
    static_geo_low = float(getattr(fusion_cfg, "static_geo_low", 0.15))
    static_sam_low = float(getattr(fusion_cfg, "static_sam_low", 0.1))
    min_vis_dynamic = float(getattr(fusion_cfg, "min_vis_dynamic", 0.35))
    min_vis_static = float(getattr(fusion_cfg, "min_vis_static", 0.45))

    confident_dynamic = (
        (((sam_union > dyn_sam_high) | (geo > dyn_geo_high)) & (vis > min_vis_dynamic))
        | ((sam_union > dyn_sam_high * 0.85) & (geo > dyn_geo_high * 0.85) & (vis > max(0.0, min_vis_dynamic - 0.05)))
    )
    confident_static = (sam_union < static_sam_low) & (geo < static_geo_low) & (vis > min_vis_static + 0.15)

    target = torch.full_like(geo, 0.5)
    target = torch.where(confident_dynamic, torch.ones_like(target), target)
    target = torch.where(confident_static, torch.zeros_like(target), target)

    weight = torch.zeros_like(geo)
    weight = torch.where(
        confident_dynamic,
        torch.full_like(weight, float(getattr(fusion_cfg, "w_dynamic_seed", 0.75))),
        weight,
    )
    weight = torch.where(
        confident_static,
        torch.full_like(weight, float(getattr(fusion_cfg, "w_static_seed", 0.75))),
        weight,
    )

    disagreement = torch.abs(sam_union - geo)
    agreement = 1.0 - disagreement
    sam_term = torch.clamp(sam_union, min=0.0, max=1.0) * float(getattr(fusion_cfg, "w_sam", 1.0))
    geo_term = torch.clamp(geo, min=0.0, max=1.0) * float(getattr(fusion_cfg, "w_geo", 1.0))
    weight = torch.maximum(weight, vis * agreement * 0.24)
    weight = torch.where(disagreement > 0.6, weight * 0.8, weight)
    weight = weight + 0.1 * sam_term + 0.1 * geo_term
    return target, torch.clamp(weight, min=0.0, max=4.0)


def _temporal_smooth_probabilities(prob_seq: torch.Tensor, alpha: float) -> torch.Tensor:
    if prob_seq.ndim != 3 or prob_seq.shape[0] <= 1 or alpha <= 0.0:
        return prob_seq
    forward: list[torch.Tensor] = []
    running = prob_seq[0]
    for current in prob_seq:
        running = (1.0 - alpha) * current + alpha * running
        forward.append(running)
    backward: list[torch.Tensor] = []
    running = prob_seq[-1]
    for current in reversed(prob_seq):
        running = (1.0 - alpha) * current + alpha * running
        backward.append(running)
    backward.reverse()
    return torch.stack([(left + right) * 0.5 for left, right in zip(forward, backward)], dim=0)


def _normalize_map(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    max_val = torch.clamp(values.amax(), min=1.0e-6)
    return torch.clamp(values / max_val, min=0.0, max=1.0)


def _backward_propagate_dynamic(prob_seq: torch.Tensor, decay: float, iters: int) -> torch.Tensor:
    if prob_seq.ndim != 3 or prob_seq.shape[0] <= 1:
        return prob_seq
    decay = float(decay)
    iters = int(iters)
    if iters <= 0 or decay <= 0.0:
        return prob_seq
    out = prob_seq.clone()
    for _ in range(iters):
        for frame_index in range(out.shape[0] - 2, -1, -1):
            propagated = torch.clamp(out[frame_index + 1] * decay, min=0.0, max=1.0)
            out[frame_index] = torch.maximum(out[frame_index], propagated)
    return torch.clamp(out, min=0.0, max=1.0)


def _build_pairwise_3d_supervision(
    frame_index: int,
    frames: list[object],
    residual_bundle: object,
    geometry_cfg: object,
    fusion_cfg: object,
) -> dict[str, torch.Tensor]:
    num_pairs = int(getattr(fusion_cfg, "pairwise_num_pairs", 2048))
    if num_pairs <= 0:
        return {}
    neighbor_index = _select_pair_neighbor(frame_index, len(frames), getattr(geometry_cfg, "pair_offsets", [-1, 1]))
    if neighbor_index is None:
        return {}
    frame_t = frames[frame_index]
    frame_s = frames[neighbor_index]

    world_t = backproject(frame_t.depth, frame_t.intrinsics, frame_t.extrinsics)
    uv_ts, z_ts = project(world_t, frame_s.intrinsics, frame_s.extrinsics)
    depth_s, valid_uv = sample_at_coords(frame_s.depth, uv_ts)
    world_from_s = backproject(depth_s, frame_s.intrinsics, frame_s.extrinsics)

    visibility = residual_bundle.visibility
    min_vis = float(getattr(fusion_cfg, "pairwise_min_visibility", 0.2))
    valid = valid_uv & (z_ts > 1.0e-6) & (depth_s > 1.0e-6) & (visibility > min_vis)
    valid_idx = torch.nonzero(valid.reshape(-1), as_tuple=False).squeeze(1)
    if valid_idx.numel() < 2:
        return {}

    k = min(num_pairs, int(valid_idx.numel()))
    if k <= 0:
        return {}
    pick_a = valid_idx[torch.randint(0, valid_idx.numel(), (k,), device=valid_idx.device)]
    pick_b = valid_idx[torch.randint(0, valid_idx.numel(), (k,), device=valid_idx.device)]
    non_degenerate = pick_a != pick_b
    if int(non_degenerate.sum().item()) == 0:
        return {}
    pick_a = pick_a[non_degenerate]
    pick_b = pick_b[non_degenerate]

    world_t_flat = world_t.reshape(-1, 3)
    world_s_flat = world_from_s.reshape(-1, 3)
    dist_t = torch.linalg.norm(world_t_flat[pick_a] - world_t_flat[pick_b], dim=-1)
    dist_s = torch.linalg.norm(world_s_flat[pick_a] - world_s_flat[pick_b], dim=-1)
    rel_change = torch.abs(dist_s - dist_t) / torch.clamp(dist_t, min=1.0e-3)

    abs_motion = _normalize_map(residual_bundle.r_3d).reshape(-1)
    abs_pair = 0.5 * (abs_motion[pick_a] + abs_motion[pick_b])
    rel_scale = float(getattr(fusion_cfg, "pairwise_rel_scale", 10.0))
    abs_scale = float(getattr(fusion_cfg, "pairwise_abs_scale", 4.0))
    target_static = torch.exp(-(rel_scale * rel_change + abs_scale * abs_pair))

    vis_flat = visibility.reshape(-1)
    vis_pair = 0.5 * (vis_flat[pick_a] + vis_flat[pick_b])
    baseline = torch.clamp(dist_t / (dist_t + 0.05), min=0.0, max=1.0)
    pair_weight = vis_pair * baseline

    finite = torch.isfinite(target_static) & torch.isfinite(pair_weight)
    if int(finite.sum().item()) == 0:
        return {}
    pick_a = pick_a[finite]
    pick_b = pick_b[finite]
    target_static = target_static[finite]
    pair_weight = pair_weight[finite]
    if pick_a.numel() == 0:
        return {}

    return {
        "pair_idx_a": pick_a.long(),
        "pair_idx_b": pick_b.long(),
        "pair_target_static": torch.clamp(target_static, min=0.0, max=1.0),
        "pair_weight": torch.clamp(pair_weight, min=0.0),
    }


def _select_pair_neighbor(frame_index: int, num_frames: int, offsets: list[int]) -> int | None:
    if num_frames <= 1:
        return None
    candidates: list[tuple[int, int]] = []
    for offset in offsets:
        if int(offset) == 0:
            continue
        neighbor = frame_index + int(offset)
        if 0 <= neighbor < num_frames:
            candidates.append((abs(int(offset)), neighbor))
    if not candidates:
        fallback = frame_index + 1 if frame_index + 1 < num_frames else frame_index - 1
        return fallback if 0 <= fallback < num_frames else None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]
