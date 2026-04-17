#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from articulation.data.io_rgbd import load_rgb_sequence_from_video, save_rgbd_sequence_npz
from articulation.external.depth_anything3_adapter import DepthAnything3Adapter
from articulation.preprocess.depth_anything3 import estimate_depth_da3


def main() -> None:
    default_da3_repo = str(Path(__file__).resolve().parents[1] / "submodules" / "depth_anything_3")
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--model-name", default="da3-large")
    parser.add_argument(
        "--repo-path",
        default=default_da3_repo,
    )
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--process-res-method", default="upper_bound_resize")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    seq = load_rgb_sequence_from_video(args.video, args.intrinsics)
    adapter = DepthAnything3Adapter(
        repo_path=args.repo_path,
        model_dir=args.model_dir if args.model_dir else None,
        model_name=args.model_name,
        device=args.device,
    )
    depth = estimate_depth_da3(seq.rgb, adapter, process_res=args.process_res, process_res_method=args.process_res_method)
    seq.depth = depth
    save_rgbd_sequence_npz(seq, args.output)


if __name__ == "__main__":
    main()
