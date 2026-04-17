from __future__ import annotations

import torch



def point_accuracy(pred: torch.Tensor, gt: torch.Tensor) -> float:
    pred = pred.long().view(-1)
    gt = gt.long().view(-1)
    return float((pred == gt).float().mean())



def binary_iou(pred: torch.Tensor, gt: torch.Tensor) -> float:
    pred = pred.bool().view(-1)
    gt = gt.bool().view(-1)
    inter = (pred & gt).sum().float()
    union = (pred | gt).sum().float().clamp_min(1.0)
    return float(inter / union)



def binary_dice(pred: torch.Tensor, gt: torch.Tensor) -> float:
    pred = pred.bool().view(-1)
    gt = gt.bool().view(-1)
    inter = (pred & gt).sum().float()
    denom = pred.sum().float() + gt.sum().float()
    return float((2.0 * inter) / denom.clamp_min(1.0))
