from __future__ import annotations

from articulation.data.dataclasses import JointResult, SegmentationResult



def summarize_segmentation(seg: SegmentationResult) -> dict:
    return {
        "num_points": int(seg.point_labels.numel()),
        "part1_ratio": float(seg.point_probs.mean().detach().cpu()),
        "loss_total_final": float(seg.diagnostics.get("loss_total", [None])[-1]) if seg.diagnostics.get("loss_total") else None,
    }



def summarize_joint(joint: JointResult) -> dict:
    return {
        "best_model": joint.best_model,
        "num_candidates": len(joint.candidates),
        "candidate_losses": {c.model_name: float(c.loss) for c in joint.candidates},
    }
