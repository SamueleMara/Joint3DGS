from __future__ import annotations

import torch
import torch.nn.functional as F

from articulation.external.depth_anything3_adapter import DepthAnything3Adapter


def estimate_depth_da3(
    rgb: torch.Tensor,
    adapter: DepthAnything3Adapter,
    process_res: int = 504,
    process_res_method: str = "upper_bound_resize",
) -> torch.Tensor:
    depth = adapter.infer_depth(rgb, process_res=process_res, process_res_method=process_res_method)
    # Resize to match RGB if needed
    if depth.shape[-2:] != rgb.shape[-2:]:
        depth = F.interpolate(depth, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
    return depth
