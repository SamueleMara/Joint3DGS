from __future__ import annotations

import torch


def merge_overlapping_instances(masks: torch.Tensor) -> torch.Tensor:
    return masks.any(dim=0, keepdim=True)


def compute_instance_centroids(mask: torch.Tensor) -> tuple[float, float]:
    ys, xs = torch.where(mask > 0)
    if ys.numel() == 0:
        return 0.0, 0.0
    return float(xs.float().mean()), float(ys.float().mean())


def compute_instance_boxes(mask: torch.Tensor) -> tuple[int, int, int, int] | None:
    ys, xs = torch.where(mask > 0)
    if ys.numel() == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def instance_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    inter = torch.logical_and(a, b).sum().float()
    union = torch.logical_or(a, b).sum().float()
    if union == 0:
        return 0.0
    return float(inter / union)


def track_instance_lifetimes(sequence_masks: list[torch.Tensor]) -> list[int]:
    return [int(mask.any().item()) for mask in sequence_masks]


def export_instance_table(sequence_masks: list[torch.Tensor]) -> list[dict[str, int]]:
    return [{"frame_index": idx, "active_instances": int(mask.shape[0])} for idx, mask in enumerate(sequence_masks)]
