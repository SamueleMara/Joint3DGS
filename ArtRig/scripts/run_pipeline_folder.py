#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

from scripts.run_pipeline import main as pipeline_main


def main() -> None:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--rgb-dir-name", default="images")
    parser.add_argument("--depth-dir-name", default="depth")
    parser.add_argument("--mask-dir-name", default="fg_mask")
    parser.add_argument("--intrinsics-file", default="intrinsics.npy")
    parser.add_argument("--extrinsics-file", default="extrinsics.npy")
    parser.add_argument("--cameras-json", default="metadata/cameras.json")
    parser.add_argument("--extrinsics-convention", choices=["world_from_camera", "camera_from_world"], default="world_from_camera")
    parser.add_argument("--depth-scale", type=float, default=None)
    parser.add_argument("--depth-npy-scale", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=None)

    parser.add_argument("--matching-config", default="configs/matching.yaml")
    parser.add_argument("--seg-config", default="configs/segmentation.yaml")
    parser.add_argument("--joint-config", default="configs/joint.yaml")
    parser.add_argument("--out-dir", default="outputs/folder_pipeline")

    parser.add_argument("--backend-repo", default=None)
    parser.add_argument("--backend-callable", default=None)
    parser.add_argument("--progress", dest="progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--loss-debug", action="store_true")
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--viz-frame", type=int, default=0)

    args, unknown = parser.parse_known_args()

    forward = [
        "run_pipeline.py",
        "--input-dir",
        args.input_dir,
        "--rgb-dir",
        args.rgb_dir_name,
        "--depth-dir",
        args.depth_dir_name,
        "--mask-dir",
        args.mask_dir_name,
        "--intrinsics-file",
        args.intrinsics_file,
        "--extrinsics-file",
        args.extrinsics_file,
        "--cameras-json",
        args.cameras_json,
        "--extrinsics-convention",
        args.extrinsics_convention,
        "--matching-config",
        args.matching_config,
        "--seg-config",
        args.seg_config,
        "--joint-config",
        args.joint_config,
        "--out-dir",
        args.out_dir,
    ]

    if args.depth_scale is not None:
        forward += ["--depth-scale", str(args.depth_scale)]
    if args.depth_npy_scale is not None:
        forward += ["--depth-npy-scale", str(args.depth_npy_scale)]
    if args.max_frames is not None:
        forward += ["--max-frames", str(args.max_frames)]
    if args.backend_repo is not None:
        forward += ["--backend-repo", args.backend_repo]
    if args.backend_callable is not None:
        forward += ["--backend-callable", args.backend_callable]
    if args.progress:
        forward += ["--progress"]
    else:
        forward += ["--no-progress"]
    if args.loss_debug:
        forward += ["--loss-debug"]
    if args.viz:
        forward += ["--viz", "--viz-frame", str(args.viz_frame)]

    sys.argv = forward + unknown
    pipeline_main()


if __name__ == "__main__":
    main()
