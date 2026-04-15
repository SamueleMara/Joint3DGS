from __future__ import annotations

from collections import defaultdict

import torch

from articulation.segmentation.masks_from_points import masks_from_point_assignments



def merge_window_logits(
    point_ids: torch.Tensor,
    window_logits: list[tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Merge per-window logits for overlapping windows.

    Args:
        point_ids: [P_global]
        window_logits: list of (ids_w [Pw], logits_w [Pw])
    """
    acc = defaultdict(list)
    for ids_w, logits_w in window_logits:
        ids = ids_w.detach().cpu().tolist()
        vals = logits_w.detach().cpu().tolist()
        for pid, v in zip(ids, vals):
            acc[int(pid)].append(float(v))

    merged = torch.zeros_like(point_ids, dtype=torch.float32)
    for i, pid in enumerate(point_ids.detach().cpu().tolist()):
        vals = acc.get(int(pid), None)
        if vals:
            merged[i] = float(sum(vals) / len(vals))
    return merged



def build_dense_masks_from_tracks(
    xy: torch.Tensor,
    labels: torch.Tensor,
    image_size: tuple[int, int],
) -> torch.Tensor:
    return masks_from_point_assignments(xy=xy, labels=labels, image_size=image_size)
