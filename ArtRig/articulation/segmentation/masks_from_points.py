from __future__ import annotations

import torch

from articulation.preprocess.masks import rasterize_point_labels



def masks_from_point_assignments(
    xy: torch.Tensor,
    labels: torch.Tensor,
    image_size: tuple[int, int],
) -> torch.Tensor:
    return rasterize_point_labels(xy=xy, labels=labels, image_size=image_size)
