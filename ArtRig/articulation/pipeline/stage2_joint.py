from __future__ import annotations

from articulation.data.dataclasses import JointResult, SegmentationResult, TrackBatch
from articulation.external.joint_clue_adapter import JointClueEstimator
from articulation.joint.consensus import build_consensus
from articulation.joint.optimizer import optimize_joint_candidates
from articulation.joint.pointwise_init import build_pointwise_initialization
from articulation.joint.relative_motion import choose_reference_part, compute_relative_motion



def run_stage2_joint(
    tracks: TrackBatch,
    seg: SegmentationResult,
    cfg: dict,
    clue_estimator: JointClueEstimator | None = None,
    show_progress: bool = False,
    debug_losses: bool = False,
    wandb_run: object | None = None,
    wandb_prefix: str = "joint",
) -> JointResult:
    ref_part, mov_part = choose_reference_part(tracks, seg)
    labels = seg.point_labels.long()

    ref_mask = labels == ref_part
    mov_mask = labels == mov_part

    rel = compute_relative_motion(
        xyz_part_ref=tracks.xyz[ref_mask],
        xyz_part_mov=tracks.xyz[mov_mask],
        valid_ref=tracks.valid[ref_mask],
        valid_mov=tracks.valid[mov_mask],
        reference_part=ref_part,
        moving_part=mov_part,
    )

    estimator = clue_estimator or JointClueEstimator()
    init = build_pointwise_initialization(
        rel,
        estimator=estimator,
        num_point_samples=int(cfg.get("sampling", {}).get("num_point_samples", 256)),
        strategy=str(cfg.get("sampling", {}).get("strategy", "fps")),
    )

    consensus = build_consensus(init.clues)
    return optimize_joint_candidates(
        rel=rel,
        consensus=consensus,
        cfg=cfg,
        show_progress=show_progress,
        debug_losses=debug_losses,
        wandb_run=wandb_run,
        wandb_prefix=wandb_prefix,
    )
