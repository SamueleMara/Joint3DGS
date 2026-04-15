"""Fusion model components plus compatibility mask fusion helpers."""

import numpy as np

from .features import build_fusion_features
from .model import DynamicFusionNet

LABEL_STATIC = np.uint8(0)
LABEL_DYNAMIC = np.uint8(1)
LABEL_UNCERTAIN = np.uint8(2)


def compute_dynamic_consistency(
    da3: object,
    sam3: object,
    dynamic_thresh: float,
    reproj_thresh_px: float,
    uncertain_margin: float,
    min_depth: float,
    max_depth: float,
) -> object:
    del reproj_thresh_px, min_depth, max_depth
    masks = sam3.masks.max(axis=1).astype(bool) if getattr(sam3, "masks", None) is not None else np.zeros_like(da3.depth, dtype=bool)
    dynamic_score = np.abs(da3.depth - np.roll(da3.depth, -1, axis=0)).astype(np.float32)
    label_map = np.full(dynamic_score.shape, LABEL_UNCERTAIN, dtype=np.uint8)
    label_map[np.logical_and(masks, dynamic_score >= dynamic_thresh + uncertain_margin)] = LABEL_DYNAMIC
    label_map[np.logical_and(~masks, dynamic_score <= max(dynamic_thresh - uncertain_margin, 0.0))] = LABEL_STATIC
    binary_dynamic = (label_map == LABEL_DYNAMIC).astype(np.uint8)
    return type(
        "FusionResult",
        (),
        {
            "dynamic_score": dynamic_score,
            "label_map": label_map,
            "binary_dynamic": binary_dynamic,
            "track_stats": {},
        },
    )()


__all__ = ["DynamicFusionNet", "build_fusion_features", "compute_dynamic_consistency", "LABEL_STATIC", "LABEL_DYNAMIC", "LABEL_UNCERTAIN"]
