#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from articulation.data import SegmentationResult, load_rgbd_sequence_npz, load_tracks_npz
from articulation.features import build_feature_graph, initialize_two_part_logits
from articulation.pipeline.stage0_matching import run_stage0_matching
from articulation.pipeline.stage1_segmentation import run_stage1_segmentation
from articulation.preprocess.multiview import load_multiview_sequence_from_folder, single_to_multiview_sequence
from articulation.utils import configure_logging, load_yaml_config
from articulation.utils.viz import save_cog_trajectories, save_point_label_overlay, save_segmentation_mask_preview


def _save_segmentation(path: Path, seg: SegmentationResult) -> None:
    payload = {
        "point_logits": seg.point_logits.detach().cpu(),
        "point_probs": seg.point_probs.detach().cpu(),
        "point_labels": seg.point_labels.detach().cpu(),
        "masks_per_frame": seg.masks_per_frame.detach().cpu(),
        "transforms_part0": seg.transforms_part0.detach().cpu(),
        "transforms_part1": seg.transforms_part1.detach().cpu(),
        "diagnostics": seg.diagnostics,
    }
    torch.save(payload, path)


def _ensure_matplotlib_cache(out_dir: Path) -> None:
    cache = out_dir / ".cache" / "matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks-npz", default=None)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--sequence-npz", default=None)

    parser.add_argument("--matching-config", default="configs/matching.yaml")
    parser.add_argument("--config", default="configs/segmentation.yaml")
    parser.add_argument("--output", default="outputs/segmentation.pt")

    parser.add_argument("--rgb-dir", default=None)
    parser.add_argument("--depth-dir", default=None)
    parser.add_argument("--mask-dir", default=None)
    parser.add_argument("--intrinsics-file", default=None)
    parser.add_argument("--extrinsics-file", default=None)
    parser.add_argument("--cameras-json", default=None)
    parser.add_argument("--extrinsics-convention", choices=["world_from_camera", "camera_from_world"], default="world_from_camera")
    parser.add_argument("--depth-scale", type=float, default=None)
    parser.add_argument("--depth-npy-scale", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=None)

    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--progress", dest="progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--loss-debug", action="store_true", help="Print per-iteration loss debug info")
    parser.add_argument("--viz-dir", default=None)
    parser.add_argument("--viz-frame", type=int, default=0)
    args = parser.parse_args()

    configure_logging()

    cfg = load_yaml_config(args.config)
    matching_cfg = load_yaml_config(args.matching_config)

    mv_seq = None
    if args.tracks_npz:
        tracks = load_tracks_npz(args.tracks_npz)
    else:
        if (args.input_dir is None) == (args.sequence_npz is None):
            raise ValueError("Provide exactly one of --input-dir or --sequence-npz when tracks are not provided")

        if args.sequence_npz is not None:
            seq = load_rgbd_sequence_npz(args.sequence_npz)
            mv_seq = single_to_multiview_sequence(seq)
        else:
            io_cfg = dict(matching_cfg.get("io", {}))
            mv_seq = load_multiview_sequence_from_folder(
                root=args.input_dir,
                rgb_dir=args.rgb_dir or io_cfg.get("rgb_dir", "images"),
                depth_dir=args.depth_dir or io_cfg.get("depth_dir", "depth"),
                mask_dir=args.mask_dir or io_cfg.get("mask_dir", "fg_mask"),
                intrinsics_file=args.intrinsics_file or io_cfg.get("intrinsics_file", "intrinsics.npy"),
                extrinsics_file=args.extrinsics_file or io_cfg.get("extrinsics_file", "extrinsics.npy"),
                depth_scale=float(args.depth_scale if args.depth_scale is not None else io_cfg.get("depth_scale", 1.0)),
                max_frames=args.max_frames,
                cameras_json=args.cameras_json or io_cfg.get("cameras_json", "metadata/cameras.json"),
                extrinsics_convention=str(args.extrinsics_convention),
                depth_npy_scale=float(args.depth_npy_scale if args.depth_npy_scale is not None else io_cfg.get("depth_npy_scale", 1.0)),
            )

        stage0 = run_stage0_matching(
            sequence=mv_seq,
            cfg=matching_cfg,
            show_progress=args.progress,
            debug=args.loss_debug,
        )
        tracks = stage0.tracks

    anchor_xy = tracks.xy[:, tracks.anchor_frame, :]
    graph = build_feature_graph(
        tracks.feature,
        xy=anchor_xy,
        num_neighbors=int(cfg.get("features", {}).get("num_neighbors", 16)),
        spatial_gate_px=cfg.get("features", {}).get("spatial_gate_px", None),
    )
    init_logits = initialize_two_part_logits(tracks.feature)

    image_size = None
    if args.image_height is not None and args.image_width is not None:
        image_size = (args.image_height, args.image_width)
    elif mv_seq is not None:
        image_size = (mv_seq.H, mv_seq.W)

    seg = run_stage1_segmentation(
        tracks=tracks,
        graph=graph,
        cfg=cfg,
        init_logits=init_logits,
        image_size=image_size,
        show_progress=args.progress,
        debug_losses=args.loss_debug,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    _save_segmentation(out, seg)

    if args.viz_dir is not None:
        viz_dir = Path(args.viz_dir)
        viz_dir.mkdir(parents=True, exist_ok=True)
        _ensure_matplotlib_cache(viz_dir)
        save_point_label_overlay(tracks, seg, viz_dir / "points_overlay.png", frame_idx=args.viz_frame)
        save_segmentation_mask_preview(seg.masks_per_frame, viz_dir / "masks_preview.png", frame_idx=args.viz_frame)
        save_cog_trajectories(tracks, seg, viz_dir / "cog_trajectories.png")


if __name__ == "__main__":
    main()
