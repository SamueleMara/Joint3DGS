from __future__ import annotations

import argparse

from dynamic_recon.config.default import load_config
from dynamic_recon.pipelines.preprocess import run_preprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    run_preprocess(args.video, args.outdir, load_config(args.config))


if __name__ == "__main__":
    main()
