#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from tqdm.auto import tqdm

from articulation.data import TrackBatch, load_rgbd_sequence_npz, load_tracks_npz, save_tracks_npz
from articulation.external.depth_anything3_adapter import DepthAnything3Adapter
from articulation.external.joint_clue_adapter import JointClueEstimator
from articulation.features import build_feature_graph, initialize_two_part_logits
from articulation.joint.outputs import joint_result_to_dict
from articulation.joint.relative_motion import choose_reference_part, compute_relative_motion
from articulation.pipeline.stage1_segmentation import run_stage1_segmentation
from articulation.pipeline.stage2_joint import run_stage2_joint
from articulation.pipeline.tracking_utils import run_tracking_and_lift
from articulation.preprocess.depth_anything3 import estimate_depth_da3
from articulation.preprocess.masks import load_foreground_masks_from_dir
from articulation.utils import configure_logging, load_yaml_config
from articulation.utils.viz import (
    save_axis_3d,
    save_cog_trajectories,
    save_model_fit_comparison,
    save_moving_points_ref,
    save_point_label_overlay,
    save_segmentation_mask_preview,
    save_state_vs_time,
)
from articulation.external.dino_backend_adapter import DinoBackendAdapter


def _ensure_matplotlib_cache(out_dir: Path) -> None:
    cache = out_dir / ".cache" / "matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks-npz", default=None)
    parser.add_argument("--sequence-npz", default=None, help="RGBD sequence for DINO/tracking")
    parser.add_argument("--tracker-config", default="configs/tracker.yaml")
    parser.add_argument("--depth-config", default="configs/depth_anything3.yaml")
    parser.add_argument("--fg-mask-dir", default=None, help="Directory of binary foreground masks")
    parser.add_argument("--fg-mask-threshold", type=float, default=0.5)
    parser.add_argument("--use-dino-features", action="store_true")
    parser.add_argument("--seg-config", default="configs/segmentation.yaml")
    parser.add_argument("--joint-config", default="configs/joint.yaml")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--backend-repo", default=None, help="Path to external joint clue repo")
    parser.add_argument("--backend-callable", default=None, help="module:function callable")
    parser.add_argument("--progress", dest="progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--loss-debug", action="store_true", help="Print per-iteration loss debug info")
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--viz-frame", type=int, default=0)
    args = parser.parse_args()

    configure_logging()

    seg_cfg = load_yaml_config(args.seg_config)
    joint_cfg = load_yaml_config(args.joint_config)

    num_steps = 6 + (1 if args.viz else 0)
    pipeline_pbar = tqdm(total=num_steps, desc="Pipeline", disable=not args.progress, leave=False)

    if args.tracks_npz:
        tracks = load_tracks_npz(args.tracks_npz)
        seq = load_rgbd_sequence_npz(args.sequence_npz) if args.sequence_npz else None
    else:
        if args.sequence_npz is None:
            raise ValueError("--sequence-npz is required when tracks are not provided")
        seq = load_rgbd_sequence_npz(args.sequence_npz)
        if args.fg_mask_dir is not None:
            seq.fg_mask = load_foreground_masks_from_dir(
                args.fg_mask_dir,
                expected_frames=seq.T,
                image_size=(seq.H, seq.W),
                threshold=float(args.fg_mask_threshold),
            ).to(dtype=seq.rgb.dtype)
        depth_cfg = load_yaml_config(args.depth_config)
        if depth_cfg.get("enabled", False):
            adapter = DepthAnything3Adapter(
                repo_path=depth_cfg.get("repo_path"),
                model_dir=depth_cfg.get("model_dir") or None,
                model_name=depth_cfg.get("model_name", "da3-large"),
                device=depth_cfg.get("device", "cuda"),
            )
            seq.depth = estimate_depth_da3(
                seq.rgb,
                adapter,
                process_res=int(depth_cfg.get("process_res", 504)),
                process_res_method=str(depth_cfg.get("process_res_method", "upper_bound_resize")),
            )
        tracker_cfg = load_yaml_config(args.tracker_config)
        tracks = run_tracking_and_lift(
            sequence=seq,
            tracker_cfg=tracker_cfg,
            min_valid_ratio=float(seg_cfg.get("tracks", {}).get("min_valid_ratio", 0.7)),
            max_depth=seg_cfg.get("tracks", {}).get("max_depth", None),
        )
    pipeline_pbar.update(1)

    if args.use_dino_features or tracks.feature.shape[1] <= 1:
        if seq is None:
            raise ValueError("--sequence-npz is required when computing DINO features")
        model_name = seg_cfg.get("features", {}).get("model_name", "vit_small_patch14_dinov2")
        extractor = DinoBackendAdapter(model_name=model_name, device="cpu").build()
        feat_map = extractor.extract_dense(seq.rgb[tracks.anchor_frame])
        feat = extractor.sample_points(feat_map, tracks.xy[:, tracks.anchor_frame, :])
        tracks = TrackBatch(
            xy=tracks.xy,
            xyz=tracks.xyz,
            valid=tracks.valid,
            anchor_frame=tracks.anchor_frame,
            point_ids=tracks.point_ids,
            feature=feat,
            confidence=tracks.confidence,
        )
    pipeline_pbar.update(1)

    anchor_xy = tracks.xy[:, tracks.anchor_frame, :]
    graph = build_feature_graph(
        tracks.feature,
        xy=anchor_xy,
        num_neighbors=int(seg_cfg.get("features", {}).get("num_neighbors", 16)),
        spatial_gate_px=seg_cfg.get("features", {}).get("spatial_gate_px", None),
    )
    init_logits = initialize_two_part_logits(tracks.feature)
    pipeline_pbar.update(1)

    image_size = None
    if args.image_height is not None and args.image_width is not None:
        image_size = (args.image_height, args.image_width)

    seg = run_stage1_segmentation(
        tracks=tracks,
        graph=graph,
        cfg=seg_cfg,
        init_logits=init_logits,
        image_size=image_size,
        show_progress=args.progress,
        debug_losses=args.loss_debug,
    )
    pipeline_pbar.update(1)

    ext_cfg = joint_cfg.get("external", {}).get("joint_clue", {})
    if args.backend_repo is not None:
        ext_cfg["repo_path"] = args.backend_repo
    if args.backend_callable is not None:
        ext_cfg["backend_callable"] = args.backend_callable

    estimator = JointClueEstimator.from_config(ext_cfg)
    joint = run_stage2_joint(
        tracks=tracks,
        seg=seg,
        cfg=joint_cfg,
        clue_estimator=estimator,
        show_progress=args.progress,
        debug_losses=args.loss_debug,
    )
    pipeline_pbar.update(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_tracks_npz(tracks, out_dir / "tracks.npz")

    torch.save(
        {
            "point_logits": seg.point_logits.detach().cpu(),
            "point_probs": seg.point_probs.detach().cpu(),
            "point_labels": seg.point_labels.detach().cpu(),
            "masks_per_frame": seg.masks_per_frame.detach().cpu(),
            "transforms_part0": seg.transforms_part0.detach().cpu(),
            "transforms_part1": seg.transforms_part1.detach().cpu(),
            "diagnostics": seg.diagnostics,
        },
        out_dir / "segmentation.pt",
    )
    torch.save(joint_result_to_dict(joint), out_dir / "joint.pt")
    pipeline_pbar.update(1)

    if args.viz:
        _ensure_matplotlib_cache(out_dir)
        save_point_label_overlay(tracks, seg, out_dir / "viz_points_overlay.png", frame_idx=args.viz_frame)
        save_segmentation_mask_preview(seg.masks_per_frame, out_dir / "viz_masks_preview.png", frame_idx=args.viz_frame)
        save_cog_trajectories(tracks, seg, out_dir / "viz_cog_trajectories.png")
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
        save_moving_points_ref(rel, out_dir / "viz_moving_ref.png")
        save_axis_3d(rel.canonical_points, joint.axis_dir, joint.axis_point, out_dir / "viz_axis.png")
        save_model_fit_comparison(joint, out_dir / "viz_model_fit.png")
        save_state_vs_time(joint.state, out_dir / "viz_state.png")
        pipeline_pbar.update(1)

    pipeline_pbar.close()


if __name__ == "__main__":
    main()
