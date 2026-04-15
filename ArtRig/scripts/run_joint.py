#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from articulation.data import SegmentationResult, load_tracks_npz
from articulation.external.joint_clue_adapter import JointClueEstimator
from articulation.joint.outputs import joint_result_to_dict
from articulation.pipeline.stage2_joint import run_stage2_joint
from articulation.utils import configure_logging, load_yaml_config



def _load_segmentation(path: str | Path) -> SegmentationResult:
    payload = torch.load(path, map_location="cpu")
    return SegmentationResult(
        point_logits=payload["point_logits"],
        point_probs=payload["point_probs"],
        point_labels=payload["point_labels"],
        masks_per_frame=payload["masks_per_frame"],
        transforms_part0=payload["transforms_part0"],
        transforms_part1=payload["transforms_part1"],
        diagnostics=payload.get("diagnostics", {}),
    )



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks-npz", required=True)
    parser.add_argument("--segmentation", required=True)
    parser.add_argument("--config", default="configs/joint.yaml")
    parser.add_argument("--output", default="outputs/joint.pt")
    parser.add_argument("--backend-repo", default=None, help="Path to external joint clue repo")
    parser.add_argument("--backend-callable", default=None, help="module:function callable")
    parser.add_argument("--progress", dest="progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--loss-debug", action="store_true", help="Print per-iteration loss debug info")
    args = parser.parse_args()

    configure_logging()

    tracks = load_tracks_npz(args.tracks_npz)
    seg = _load_segmentation(args.segmentation)
    cfg = load_yaml_config(args.config)

    ext_cfg = cfg.get("external", {}).get("joint_clue", {})
    if args.backend_repo is not None:
        ext_cfg["repo_path"] = args.backend_repo
    if args.backend_callable is not None:
        ext_cfg["backend_callable"] = args.backend_callable

    estimator = JointClueEstimator.from_config(ext_cfg)
    joint = run_stage2_joint(
        tracks=tracks,
        seg=seg,
        cfg=cfg,
        clue_estimator=estimator,
        show_progress=args.progress,
        debug_losses=args.loss_debug,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(joint_result_to_dict(joint), out)


if __name__ == "__main__":
    main()
