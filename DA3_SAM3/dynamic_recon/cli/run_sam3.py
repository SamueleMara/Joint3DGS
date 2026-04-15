from __future__ import annotations

import argparse

from dynamic_recon.config.default import load_config
from dynamic_recon.pipelines.infer_initial import run_initial_pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--seeds-dir", default=None)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--sam-backend", choices=("sam2", "sam3"), default=None)
    args = parser.parse_args()
    del args.seeds_dir
    cfg = load_config(args.config)
    if args.sam_backend == "sam2":
        cfg.sam3.version = "sam2.1"
        if not cfg.sam3.model_id or cfg.sam3.model_id.startswith("facebook/sam3"):
            cfg.sam3.model_id = "facebook/sam2.1-hiera-small"
    elif args.sam_backend == "sam3":
        cfg.sam3.version = "sam3.1"
        if cfg.sam3.model_id and cfg.sam3.model_id.startswith("facebook/sam2"):
            cfg.sam3.model_id = None
    run_initial_pass(args.video, args.outdir, cfg)


if __name__ == "__main__":
    main()
