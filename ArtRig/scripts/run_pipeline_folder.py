#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from articulation.data import TrackBatch, load_rgbd_sequence_from_folders, save_tracks_npz
from articulation.external.dino_backend_adapter import DinoBackendAdapter
from articulation.external.joint_clue_adapter import JointClueEstimator
from articulation.features import build_feature_graph, initialize_two_part_logits
from articulation.joint.outputs import joint_result_to_dict
from articulation.joint.relative_motion import choose_reference_part, compute_relative_motion
from articulation.pipeline.stage1_segmentation import run_stage1_segmentation
from articulation.pipeline.stage2_joint import run_stage2_joint
from articulation.pipeline.tracking_utils import run_tracking_and_lift
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


def _ensure_matplotlib_cache(out_dir: Path) -> None:
    cache = out_dir / ".cache" / "matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))


def _infer_depth_scale(input_dir: Path, user_depth_scale: float | None) -> float:
    if user_depth_scale is not None:
        return float(user_depth_scale)
    meta_path = input_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            if "depth_scale" in meta:
                return float(meta["depth_scale"])
        except Exception:
            pass
    return 1.0


def _validate_extrinsics(path: Path, expected_t: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Expected extrinsics file not found: {path}")
    ext = np.load(path)
    if ext.ndim != 3:
        raise ValueError(f"extrinsics must be 3D, got shape {ext.shape}")
    if ext.shape[0] != expected_t:
        raise ValueError(f"extrinsics T={ext.shape[0]} does not match sequence T={expected_t}")
    if ext.shape[1:] == (4, 4):
        ext = ext[:, :3, :]
    if ext.shape[1:] != (3, 4):
        raise ValueError(f"extrinsics must be [T,3,4] or [T,4,4], got {ext.shape}")
    return ext


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Dataset folder with images/depth/fg_mask/intrinsics/extrinsics")
    parser.add_argument("--rgb-dir-name", default="images")
    parser.add_argument("--depth-dir-name", default="depth")
    parser.add_argument("--mask-dir-name", default="fg_mask")
    parser.add_argument("--intrinsics-file", default="intrinsics.npy")
    parser.add_argument("--extrinsics-file", default="extrinsics.npy")
    parser.add_argument("--depth-scale", type=float, default=None, help="Depth meters = depth_image * depth_scale")
    parser.add_argument("--tracker-config", default="configs/tracker.yaml")
    parser.add_argument("--seg-config", default="configs/segmentation.yaml")
    parser.add_argument("--joint-config", default="configs/joint.yaml")
    parser.add_argument("--use-dino-features", action="store_true")
    parser.add_argument("--backend-repo", default=None, help="Path to external joint clue repo")
    parser.add_argument("--backend-callable", default=None, help="module:function callable")
    parser.add_argument("--out-dir", default="outputs/folder_pipeline")
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--progress", dest="progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--loss-debug", action="store_true")
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--viz-frame", type=int, default=0)
    args = parser.parse_args()

    configure_logging()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input-dir not found: {input_dir}")

    rgb_dir = input_dir / args.rgb_dir_name
    depth_dir = input_dir / args.depth_dir_name
    mask_dir = input_dir / args.mask_dir_name
    intrinsics_path = input_dir / args.intrinsics_file
    extrinsics_path = input_dir / args.extrinsics_file

    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
    if not depth_dir.is_dir():
        raise FileNotFoundError(f"Depth directory not found: {depth_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Foreground mask directory not found: {mask_dir}")
    if not intrinsics_path.exists():
        raise FileNotFoundError(f"Intrinsics file not found: {intrinsics_path}")

    depth_scale = _infer_depth_scale(input_dir, args.depth_scale)
    seq = load_rgbd_sequence_from_folders(
        rgb_dir=rgb_dir,
        depth_dir=depth_dir,
        intrinsics=intrinsics_path,
        mask_dir=mask_dir,
        depth_scale=depth_scale,
        meta={"input_dir": str(input_dir), "depth_scale": depth_scale},
    )
    _validate_extrinsics(extrinsics_path, expected_t=seq.T)
    seq.meta["extrinsics_path"] = str(extrinsics_path)

    seg_cfg = load_yaml_config(args.seg_config)
    joint_cfg = load_yaml_config(args.joint_config)
    tracker_cfg = load_yaml_config(args.tracker_config)
    if tracker_cfg.get("backend", "precomputed") == "precomputed" and not tracker_cfg.get("tracks_npz"):
        tracker_cfg = dict(tracker_cfg)
        tracker_cfg["backend"] = "cotracker"
        tracker_cfg.setdefault("device", "cpu")

    num_steps = 6 + (1 if args.viz else 0)
    pipeline_pbar = tqdm(total=num_steps, desc="Pipeline", disable=not args.progress, leave=False)

    tracks = run_tracking_and_lift(
        sequence=seq,
        tracker_cfg=tracker_cfg,
        min_valid_ratio=float(seg_cfg.get("tracks", {}).get("min_valid_ratio", 0.7)),
        max_depth=seg_cfg.get("tracks", {}).get("max_depth", None),
    )
    pipeline_pbar.update(1)

    if args.use_dino_features or tracks.feature.shape[1] <= 1:
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

    meta = {
        "input_dir": str(input_dir),
        "rgb_dir": str(rgb_dir),
        "depth_dir": str(depth_dir),
        "mask_dir": str(mask_dir),
        "intrinsics_file": str(intrinsics_path),
        "extrinsics_file": str(extrinsics_path),
        "depth_scale": float(depth_scale),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
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
