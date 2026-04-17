from __future__ import annotations

from articulation.data.dataclasses import FeatureGraph, JointResult, MultiViewRGBDSequence, SegmentationResult, TrackBatch
from articulation.features.graph import build_feature_graph
from articulation.features.initialization import initialize_two_part_logits
from articulation.pipeline.stage0_matching import Stage0MatchingResult, run_stage0_matching
from articulation.pipeline.stage1_segmentation import run_stage1_segmentation
from articulation.pipeline.stage2_joint import run_stage2_joint



def run_full_pipeline(
    tracks: TrackBatch,
    graph: FeatureGraph,
    seg_cfg: dict,
    joint_cfg: dict,
    init_logits=None,
    image_size=None,
    show_progress: bool = False,
    debug_losses: bool = False,
) -> tuple[SegmentationResult, JointResult]:
    seg = run_stage1_segmentation(
        tracks=tracks,
        graph=graph,
        cfg=seg_cfg,
        init_logits=init_logits,
        image_size=image_size,
        show_progress=show_progress,
        debug_losses=debug_losses,
    )
    joint = run_stage2_joint(
        tracks=tracks,
        seg=seg,
        cfg=joint_cfg,
        show_progress=show_progress,
        debug_losses=debug_losses,
    )
    return seg, joint


def run_full_pipeline_from_multiview(
    sequence: MultiViewRGBDSequence,
    matching_cfg: dict,
    seg_cfg: dict,
    joint_cfg: dict,
    image_size: tuple[int, int] | None = None,
    show_progress: bool = False,
    debug_losses: bool = False,
) -> tuple[Stage0MatchingResult, SegmentationResult, JointResult]:
    stage0 = run_stage0_matching(sequence, matching_cfg, show_progress=show_progress, debug=debug_losses)
    tracks = stage0.tracks
    graph = build_feature_graph(
        tracks.feature,
        xy=tracks.xy[:, tracks.anchor_frame, :],
        num_neighbors=int(seg_cfg.get("features", {}).get("num_neighbors", 16)),
        spatial_gate_px=seg_cfg.get("features", {}).get("spatial_gate_px", None),
    )
    init_logits = initialize_two_part_logits(tracks.feature)
    if image_size is None:
        image_size = (sequence.H, sequence.W)
    seg, joint = run_full_pipeline(
        tracks=tracks,
        graph=graph,
        seg_cfg=seg_cfg,
        joint_cfg=joint_cfg,
        init_logits=init_logits,
        image_size=image_size,
        show_progress=show_progress,
        debug_losses=debug_losses,
    )
    return stage0, seg, joint
