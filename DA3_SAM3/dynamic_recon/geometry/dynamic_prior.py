from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class PairResiduals:
    uv_ts: torch.Tensor
    z_ts: torch.Tensor
    r_3d: torch.Tensor
    r_rel: torch.Tensor
    r_flow: torch.Tensor
    r_depth: torch.Tensor
    r_rgb: torch.Tensor
    r_cycle: torch.Tensor
    visibility: torch.Tensor


def _normalize(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values
    scale = torch.mean(torch.abs(values))
    return values / torch.clamp(scale + 1.0, min=1.0e-6)


def build_dynamic_prior(pair: PairResiduals, cfg: object) -> torch.Tensor:
    logit = (
        cfg.alpha_3d * _normalize(pair.r_3d)
        + cfg.alpha_rel * _normalize(pair.r_rel)
        + cfg.alpha_flow * _normalize(pair.r_flow)
        + cfg.alpha_depth * _normalize(pair.r_depth)
        + cfg.alpha_rgb * _normalize(pair.r_rgb)
        + cfg.alpha_cycle * _normalize(pair.r_cycle)
        - cfg.alpha_vis * _normalize(pair.visibility)
    )
    return torch.sigmoid(logit)


def build_sequence_dynamic_prior(pairs: list[PairResiduals], cfg: object) -> torch.Tensor:
    priors = [build_dynamic_prior(pair, cfg) * torch.clamp(pair.visibility, min=0.0) for pair in pairs]
    weights = [torch.clamp(pair.visibility, min=1.0e-6) for pair in pairs]
    if not priors:
        raise ValueError("No pair residuals provided")
    return torch.stack(priors).sum(dim=0) / torch.stack(weights).sum(dim=0)


def temporal_smooth_priors(priors: list[torch.Tensor], alpha: float) -> list[torch.Tensor]:
    if not priors:
        return []
    alpha = float(alpha)
    if alpha <= 0.0:
        return [prior.clone() for prior in priors]
    smoothed_forward: list[torch.Tensor] = []
    running = priors[0].clone()
    for prior in priors:
        running = (1.0 - alpha) * prior + alpha * running
        smoothed_forward.append(running.clone())
    smoothed_backward: list[torch.Tensor] = []
    running = priors[-1].clone()
    for prior in reversed(priors):
        running = (1.0 - alpha) * prior + alpha * running
        smoothed_backward.append(running.clone())
    smoothed_backward.reverse()
    return [0.5 * (left + right) for left, right in zip(smoothed_forward, smoothed_backward)]
