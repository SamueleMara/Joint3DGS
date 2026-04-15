from __future__ import annotations

import torch

from dynamic_recon.fusion.losses import (
    absolute_motion_loss,
    absolute_motion_ranking_loss,
    contrastive_group_loss,
    edge_aware_tv,
    pairwise_static_contrastive_loss,
    weighted_bce,
)
from dynamic_recon.fusion.model import DynamicFusionNet
from dynamic_recon.progress import progress_iter


def train_one_epoch(
    model: DynamicFusionNet,
    batches: list[dict[str, torch.Tensor]],
    lr: float,
    *,
    w_tv: float = 0.01,
    w_abs_motion: float = 0.4,
    w_motion_rank: float = 0.2,
    w_contrastive: float = 0.25,
    w_pair_3d_contrastive: float = 0.35,
    contrastive_pairs: int = 1024,
    contrastive_neighbor_radius: int = 6,
    contrastive_beta: float = 8.0,
) -> DynamicFusionNet:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for batch in progress_iter(batches, desc="Fusion epoch", total=len(batches)):
        pred = model(batch["x"])
        dyn_prob = torch.sigmoid(pred["dynamic_logit"])
        loss = weighted_bce(pred["dynamic_logit"], batch["target"], batch["weight"])
        loss = loss + float(w_tv) * edge_aware_tv(dyn_prob, batch["image"])
        if "motion_abs" in batch:
            loss = loss + float(w_abs_motion) * absolute_motion_loss(dyn_prob, batch["motion_abs"], batch["weight"])
            loss = loss + float(w_motion_rank) * absolute_motion_ranking_loss(
                dyn_prob,
                batch["motion_abs"],
                batch["weight"],
            )
        if "motion_rel" in batch:
            loss = loss + float(w_contrastive) * contrastive_group_loss(
                dyn_prob,
                batch["motion_rel"],
                num_pairs=int(contrastive_pairs),
                neighbor_radius=int(contrastive_neighbor_radius),
                beta=float(contrastive_beta),
            )
        if "pair_idx_a" in batch and "pair_idx_b" in batch and "pair_target_static" in batch:
            loss = loss + float(w_pair_3d_contrastive) * pairwise_static_contrastive_loss(
                dyn_prob,
                batch["pair_idx_a"],
                batch["pair_idx_b"],
                batch["pair_target_static"],
                batch.get("pair_weight"),
            )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model
