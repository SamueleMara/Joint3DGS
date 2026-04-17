#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from articulation.data import load_rgbd_sequence_npz, save_tracks_npz
from articulation.pipeline.stage0_matching import run_stage0_matching
from articulation.preprocess.multiview import load_multiview_sequence_from_folder, single_to_multiview_sequence
from articulation.utils import configure_logging, load_yaml_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=None, help="Dataset root with rgb/depth/fg_mask + intrinsics/extrinsics")
    parser.add_argument("--sequence-npz", default=None, help="Single-view RGBD npz (legacy)")
    parser.add_argument("--config", default="configs/matching.yaml")
    parser.add_argument("--out-dir", default="outputs/matching")
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
    parser.add_argument("--progress", dest="progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    configure_logging()
    cfg = load_yaml_config(args.config)

    if args.input_dir is None and args.sequence_npz is None:
        raise ValueError("Provide either --input-dir or --sequence-npz")

    if args.input_dir is not None and args.sequence_npz is not None:
        raise ValueError("Use only one of --input-dir or --sequence-npz")

    if args.sequence_npz is not None:
        seq = load_rgbd_sequence_npz(args.sequence_npz)
        mv_seq = single_to_multiview_sequence(seq)
    else:
        io_cfg = dict(cfg.get("io", {}))
        rgb_dir = args.rgb_dir or io_cfg.get("rgb_dir", "images")
        depth_dir = args.depth_dir or io_cfg.get("depth_dir", "depth")
        mask_dir = args.mask_dir or io_cfg.get("mask_dir", "fg_mask")
        intrinsics_file = args.intrinsics_file or io_cfg.get("intrinsics_file", "intrinsics.npy")
        extrinsics_file = args.extrinsics_file or io_cfg.get("extrinsics_file", "extrinsics.npy")
        cameras_json = args.cameras_json or io_cfg.get("cameras_json", "metadata/cameras.json")
        depth_scale = float(args.depth_scale if args.depth_scale is not None else io_cfg.get("depth_scale", 1.0))
        depth_npy_scale = float(args.depth_npy_scale if args.depth_npy_scale is not None else io_cfg.get("depth_npy_scale", 1.0))

        mv_seq = load_multiview_sequence_from_folder(
            root=args.input_dir,
            rgb_dir=rgb_dir,
            depth_dir=depth_dir,
            mask_dir=mask_dir,
            intrinsics_file=intrinsics_file,
            extrinsics_file=extrinsics_file,
            depth_scale=depth_scale,
            max_frames=args.max_frames,
            cameras_json=cameras_json,
            extrinsics_convention=str(args.extrinsics_convention),
            depth_npy_scale=depth_npy_scale,
        )

    result = run_stage0_matching(
        sequence=mv_seq,
        cfg=cfg,
        show_progress=args.progress,
        debug=args.debug,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_tracks_npz(result.tracks, out_dir / "tracks.npz")
    (out_dir / "matching_diagnostics.json").write_text(json.dumps(result.diagnostics, indent=2))


if __name__ == "__main__":
    main()
