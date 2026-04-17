from __future__ import annotations

import torch



def axis_direction_error_deg(pred: torch.Tensor, gt: torch.Tensor) -> float:
    p = pred / torch.linalg.norm(pred).clamp_min(1e-8)
    g = gt / torch.linalg.norm(gt).clamp_min(1e-8)
    cos = torch.abs(torch.dot(p, g)).clamp(0.0, 1.0)
    return float(torch.rad2deg(torch.acos(cos)))



def rmse_state(pred: torch.Tensor, gt: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((pred - gt) ** 2)))
