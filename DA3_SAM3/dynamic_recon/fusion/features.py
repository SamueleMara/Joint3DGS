from __future__ import annotations

import torch
import torch.nn.functional as F


def build_fusion_features(frame_t: torch.Tensor, da3_t: object, residual_bundle_t: object, sam3_t: object, cfg: object | None = None) -> torch.Tensor:
    del cfg
    target_hw = da3_t.depth.shape
    sam_logit_union, sam_mask_union = _summarize_sam_outputs(sam3_t, target_hw)
    channels = [
        frame_t,
        da3_t.depth.unsqueeze(0),
        (da3_t.confidence if da3_t.confidence is not None else torch.ones_like(da3_t.depth)).unsqueeze(0),
        residual_bundle_t.r_3d.unsqueeze(0),
        residual_bundle_t.r_rel.unsqueeze(0),
        residual_bundle_t.r_flow.unsqueeze(0),
        residual_bundle_t.r_depth.unsqueeze(0),
        residual_bundle_t.r_rgb.unsqueeze(0),
        residual_bundle_t.r_cycle.unsqueeze(0),
        residual_bundle_t.visibility.unsqueeze(0),
        sam_logit_union,
        sam_mask_union,
    ]
    return torch.cat(channels, dim=0)


def _summarize_sam_outputs(sam3_t: object, target_hw: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    logits = getattr(sam3_t, "logits", None)
    masks = getattr(sam3_t, "masks", None)
    if logits is None or masks is None:
        zeros = torch.zeros((1, *target_hw), dtype=torch.float32)
        return zeros, zeros.clone()

    logits = logits.float()
    masks = masks.float()
    if logits.ndim == 2:
        logits = logits.unsqueeze(0)
    if masks.ndim == 2:
        masks = masks.unsqueeze(0)
    if logits.shape[0] == 0:
        logit_union = torch.zeros((1, *logits.shape[-2:]), dtype=logits.dtype, device=logits.device)
    else:
        logit_union = logits.amax(dim=0, keepdim=True)
    if masks.shape[0] == 0:
        mask_union = torch.zeros((1, *masks.shape[-2:]), dtype=masks.dtype, device=masks.device)
    else:
        mask_union = masks.amax(dim=0, keepdim=True)
    if logit_union.shape[-2:] != target_hw:
        logit_union = F.interpolate(logit_union.unsqueeze(0), size=target_hw, mode="bilinear", align_corners=False)[0]
    if mask_union.shape[-2:] != target_hw:
        mask_union = F.interpolate(mask_union.unsqueeze(0), size=target_hw, mode="nearest")[0]
    return logit_union, mask_union
