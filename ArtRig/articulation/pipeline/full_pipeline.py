from __future__ import annotations

from articulation.data.dataclasses import FeatureGraph, JointResult, SegmentationResult, TrackBatch
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
