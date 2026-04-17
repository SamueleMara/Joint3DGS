#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from articulation.data import SegmentationResult, load_tracks_npz
from articulation.joint.relative_motion import choose_reference_part, compute_relative_motion
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
    parser.add_argument("--segmentation", required=True)
    parser.add_argument("--tracks-npz", required=True)
    parser.add_argument("--joint", default=None)
    parser.add_argument("--frame-idx", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/viz")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _ensure_matplotlib_cache(out_dir)

    tracks = load_tracks_npz(args.tracks_npz)
    seg = _load_segmentation(args.segmentation)

    save_point_label_overlay(tracks, seg, out_dir / "points_overlay.png", frame_idx=args.frame_idx)
    save_segmentation_mask_preview(seg.masks_per_frame, out_dir / "masks_preview.png", frame_idx=args.frame_idx)
    save_cog_trajectories(tracks, seg, out_dir / "cog_trajectories.png")

    if args.joint is not None:
        joint_payload = torch.load(args.joint, map_location="cpu")
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
        save_moving_points_ref(rel, out_dir / "moving_ref.png")
        save_axis_3d(rel.canonical_points, torch.tensor(joint_payload["axis_dir"]),
                     None if joint_payload["axis_point"] is None else torch.tensor(joint_payload["axis_point"]),
                     out_dir / "axis.png")
        save_state_vs_time(torch.tensor(joint_payload["state"]), out_dir / "state.png")

        # If candidates are present, generate comparison plot
        if "candidates" in joint_payload:
            from articulation.data.dataclasses import JointCandidateResult, JointResult

            candidates = [
                JointCandidateResult(
                    model_name=c["model_name"],
                    loss=c["loss"],
                    axis_dir=torch.tensor(joint_payload["axis_dir"]),
                    axis_point=None,
                    pitch=None,
                    state=torch.tensor(joint_payload["state"]),
                    pred_points=torch.zeros((1, 1, 3)),
                    diagnostics={},
                )
                for c in joint_payload["candidates"]
            ]
            joint = JointResult(
                best_model=joint_payload.get("best_model", ""),
                axis_dir=torch.tensor(joint_payload["axis_dir"]),
                axis_point=None,
                pitch=None,
                state=torch.tensor(joint_payload["state"]),
                candidates=candidates,
                diagnostics={},
            )
            save_model_fit_comparison(joint, out_dir / "model_fit.png")


if __name__ == "__main__":
    main()
