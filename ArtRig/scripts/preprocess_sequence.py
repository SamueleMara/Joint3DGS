#!/usr/bin/env python3
from __future__ import annotations

import argparse

from articulation.data.io_rgbd import load_rgbd_sequence_from_folders, save_rgbd_sequence_npz



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-dir", required=True)
    parser.add_argument("--depth-dir", required=True)
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--mask-dir", default=None)
    parser.add_argument("--depth-scale", type=float, default=1.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    seq = load_rgbd_sequence_from_folders(
        rgb_dir=args.rgb_dir,
        depth_dir=args.depth_dir,
        intrinsics=args.intrinsics,
        mask_dir=args.mask_dir,
        depth_scale=args.depth_scale,
    )
    save_rgbd_sequence_npz(seq, args.output)


if __name__ == "__main__":
    main()
