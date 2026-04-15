from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_bce(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * weight).mean()


def edge_aware_tv(prob: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(prob[..., :, 1:] - prob[..., :, :-1])
    dy = torch.abs(prob[..., 1:, :] - prob[..., :-1, :])
    image_grad_x = torch.mean(torch.abs(image[..., :, 1:] - image[..., :, :-1]), dim=1, keepdim=True)
    image_grad_y = torch.mean(torch.abs(image[..., 1:, :] - image[..., :-1, :]), dim=1, keepdim=True)
    return (dx * torch.exp(-image_grad_x)).mean() + (dy * torch.exp(-image_grad_y)).mean()


def absolute_motion_loss(prob: torch.Tensor, motion_abs: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    target = torch.clamp(motion_abs, min=0.0, max=1.0)
    return ((prob - target).abs() * weight).mean()


def absolute_motion_ranking_loss(
    prob: torch.Tensor,
    motion_abs: torch.Tensor,
    weight: torch.Tensor,
    margin: float = 0.2,
    num_pairs: int = 1024,
) -> torch.Tensor:
    prob_flat = prob.reshape(-1)
    motion_flat = torch.clamp(motion_abs.reshape(-1), min=0.0, max=1.0)
    weight_flat = torch.clamp(weight.reshape(-1), min=0.0)
    if prob_flat.numel() < 2:
        return prob_flat.new_tensor(0.0)
    hi_thr = torch.quantile(motion_flat.detach(), 0.75)
    lo_thr = torch.quantile(motion_flat.detach(), 0.25)
    high_idx = torch.nonzero((motion_flat >= hi_thr) & (weight_flat > 0.0), as_tuple=False).squeeze(1)
    low_idx = torch.nonzero((motion_flat <= lo_thr) & (weight_flat > 0.0), as_tuple=False).squeeze(1)
    if high_idx.numel() == 0 or low_idx.numel() == 0:
        return prob_flat.new_tensor(0.0)
    k = int(min(num_pairs, high_idx.numel(), low_idx.numel()))
    if k <= 0:
        return prob_flat.new_tensor(0.0)
    hi_pick = high_idx[torch.randint(0, high_idx.numel(), (k,), device=prob.device)]
    lo_pick = low_idx[torch.randint(0, low_idx.numel(), (k,), device=prob.device)]
    # High-motion points should get larger dynamic probabilities than low-motion points.
    return F.relu(margin - (prob_flat[hi_pick] - prob_flat[lo_pick])).mean()


def contrastive_group_loss(
    prob: torch.Tensor,
    motion_rel: torch.Tensor,
    num_pairs: int = 1024,
    neighbor_radius: int = 6,
    beta: float = 8.0,
) -> torch.Tensor:
    if prob.ndim != 4 or motion_rel.ndim != 4:
        raise ValueError("contrastive_group_loss expects [B,1,H,W] tensors for prob and motion_rel.")
    b, _, h, w = prob.shape
    if h * w < 2:
        return prob.new_tensor(0.0)
    total = b * h * w
    k = int(min(num_pairs, total))
    if k <= 0:
        return prob.new_tensor(0.0)
    flat_prob = prob.reshape(-1)
    flat_rel = torch.clamp(motion_rel.reshape(-1), min=0.0, max=1.0)
    anchor = torch.randint(0, total, (k,), device=prob.device)
    a_b = anchor // (h * w)
    a_hw = anchor % (h * w)
    a_y = a_hw // w
    a_x = a_hw % w

    dy = torch.randint(-neighbor_radius, neighbor_radius + 1, (k,), device=prob.device)
    dx = torch.randint(-neighbor_radius, neighbor_radius + 1, (k,), device=prob.device)
    b_y = torch.clamp(a_y + dy, min=0, max=h - 1)
    b_x = torch.clamp(a_x + dx, min=0, max=w - 1)
    pair = a_b * (h * w) + b_y * w + b_x

    p_a = flat_prob[anchor]
    p_b = flat_prob[pair]
    rel_a = flat_rel[anchor]
    rel_b = flat_rel[pair]
    rel_delta = torch.abs(rel_a - rel_b)
    rel_mag = 0.5 * (rel_a + rel_b)
    # Large rigidity disagreement (or large absolute relative change) means they should not be in same group.
    target_same = torch.exp(-(beta * rel_delta + 0.5 * beta * rel_mag))
    same_group_prob = p_a * p_b + (1.0 - p_a) * (1.0 - p_b)
    same_group_prob = torch.clamp(same_group_prob, min=1.0e-5, max=1.0 - 1.0e-5)
    return F.binary_cross_entropy(same_group_prob, target_same)


def pairwise_static_contrastive_loss(
    prob: torch.Tensor,
    pair_idx_a: torch.Tensor,
    pair_idx_b: torch.Tensor,
    target_both_static: torch.Tensor,
    pair_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pairwise static-compatibility loss.

    For a pair (i, j), target_both_static is low when their 3D distance changes in time
    (or when absolute motion is high), so predicting both as static is penalized.
    """
    if prob.ndim != 4:
        raise ValueError("pairwise_static_contrastive_loss expects prob with shape [B,1,H,W].")
    if pair_idx_a.numel() == 0 or pair_idx_b.numel() == 0:
        return prob.new_tensor(0.0)

    static_prob = (1.0 - prob).reshape(-1)
    p_static_pair = static_prob[pair_idx_a] * static_prob[pair_idx_b]
    p_static_pair = torch.clamp(p_static_pair, min=1.0e-5, max=1.0 - 1.0e-5)
    target = torch.clamp(target_both_static.reshape(-1), min=0.0, max=1.0)
    per_pair = F.binary_cross_entropy(p_static_pair, target, reduction="none")
    if pair_weight is None:
        return per_pair.mean()
    weight = torch.clamp(pair_weight.reshape(-1), min=0.0)
    norm = torch.clamp(weight.sum(), min=1.0e-6)
    return (per_pair * weight).sum() / norm
