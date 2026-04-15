from __future__ import annotations

import argparse
import torch

from dynamic_recon.config.default import load_config
from dynamic_recon.pipelines.export_results import export_results
from dynamic_recon.pipelines.iterate import run_preliminary_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--config", default="configs/quickstart.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-outer-iters", type=int, default=None)
    parser.add_argument("--sam-backend", choices=("sam2", "sam3"), default=None)
    parser.add_argument("--save-intermediates", action="store_true")
    parser.add_argument("--skip-da3", action="store_true")
    parser.add_argument("--skip-sam3", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    del args.skip_da3, args.skip_sam3, args.resume
    cfg = load_config(args.config)
    if args.device:
        cfg.device = args.device
    if args.num_outer_iters is not None:
        cfg.pipeline.num_outer_iters = args.num_outer_iters
    cfg.pipeline.save_intermediates = args.save_intermediates
    if args.sam_backend == "sam2":
        cfg.sam3.version = "sam2.1"
        if not cfg.sam3.model_id or cfg.sam3.model_id.startswith("facebook/sam3"):
            cfg.sam3.model_id = "facebook/sam2.1-hiera-small"
    elif args.sam_backend == "sam3":
        cfg.sam3.version = "sam3.1"
        if cfg.sam3.model_id and cfg.sam3.model_id.startswith("facebook/sam2"):
            cfg.sam3.model_id = None
    if str(cfg.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "Configured device is CUDA, but PyTorch cannot access a CUDA device. "
            "Check GPU driver/runtime (e.g., `nvidia-smi`), and ensure `CUDA_VISIBLE_DEVICES` is valid. "
            "If you intentionally want CPU, run with `--device cpu`."
        )
    state = run_preliminary_pipeline(args.video, args.outdir, cfg)
    export_results(args.outdir, state)


if __name__ == "__main__":
    main()
