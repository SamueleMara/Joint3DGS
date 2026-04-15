from __future__ import annotations

from dataclasses import dataclass

import torch
from tqdm.auto import tqdm

from articulation.data.dataclasses import FeatureGraph, SegmentationResult, TrackBatch
from articulation.geometry.se3 import se3_exp
from articulation.segmentation.losses import (
    SegmentationLossWeights,
    balance_loss,
    cog_rigidity_loss,
    feature_smoothness_loss,
    motion_fit_loss,
    pairwise_rigidity_loss,
    rigidity_consistency_loss,
    total_segmentation_loss,
)
from articulation.segmentation.optimizer import build_segmentation_optimizer
from articulation.segmentation.propagation import build_dense_masks_from_tracks
from articulation.segmentation.variables import SegmentationVariables


@dataclass
class SegmentationSchedule:
    warmup_iters: int = 80
    pair_start_iter: int = 60
    cog_start_iter: int = 120


class SegmentationTrainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        loss_cfg = cfg.get("loss", {})
        self.loss_weights = SegmentationLossWeights(
            lambda_motion=float(loss_cfg.get("lambda_motion", 200.0)),
            lambda_smooth=float(loss_cfg.get("lambda_smooth", 10.0)),
            lambda_rigid=float(loss_cfg.get("lambda_rigid", loss_cfg.get("lambda_entropy", 0.01))),
            lambda_pair=float(loss_cfg.get("lambda_pair", 2.0)),
            lambda_cog=float(loss_cfg.get("lambda_cog", 1.0)),
            lambda_balance=float(loss_cfg.get("lambda_balance", 0.0)),
        )
        schedule_cfg = cfg.get("optimizer", {}).get("schedule", {})
        self.schedule = SegmentationSchedule(
            warmup_iters=int(schedule_cfg.get("warmup_iters", 80)),
            pair_start_iter=int(schedule_cfg.get("pair_start_iter", 60)),
            cog_start_iter=int(schedule_cfg.get("cog_start_iter", 120)),
        )

    def _twists_to_transforms(self, xi: torch.Tensor) -> torch.Tensor:
        return se3_exp(xi)

    def fit(
        self,
        tracks: TrackBatch,
        graph: FeatureGraph,
        init_logits: torch.Tensor | None = None,
        image_size: tuple[int, int] | None = None,
        show_progress: bool = False,
        debug_losses: bool = False,
        progress_desc: str | None = None,
    ) -> SegmentationResult:
        p, tw, _ = tracks.xyz.shape
        vars = SegmentationVariables(num_points=p, num_steps=tw - 1)
        vars = vars.to(tracks.xyz.device)

        if init_logits is not None:
            if init_logits.shape != (p,):
                raise ValueError("init_logits must be [P]")
            with torch.no_grad():
                vars.logits.copy_(init_logits.to(vars.logits.device, dtype=vars.logits.dtype))

        opt = build_segmentation_optimizer(vars, self.cfg.get("optimizer", {}))
        iters = int(self.cfg.get("optimizer", {}).get("iterations", 300))
        grad_clip = float(self.cfg.get("optimizer", {}).get("grad_clip", 1.0))
        debug_every = int(self.cfg.get("optimizer", {}).get("debug_every", 10))

        pair_cfg = self.cfg.get("sampling", {})
        pair_n = int(pair_cfg.get("num_pairs", 4096))
        loss_cfg = self.cfg.get("loss", {})
        pair_margin = float(loss_cfg.get("pair_margin", 0.01))
        pair_lambda_sep = float(loss_cfg.get("pair_lambda_sep", 1.0))

        diagnostics: dict[str, list[float]] = {
            "loss_total": [],
            "loss_motion": [],
            "loss_smooth": [],
            "loss_rigid": [],
            "loss_entropy": [],  # legacy name kept for downstream compatibility
            "loss_pair": [],
            "loss_cog": [],
            "loss_balance": [],
        }
        nonfinite_steps = 0

        pbar = tqdm(
            range(iters),
            desc=progress_desc or "Stage1/Segmentation",
            disable=not show_progress,
            leave=False,
        )
        for it in pbar:
            opt.zero_grad(set_to_none=True)

            w = vars.probs()
            T0 = self._twists_to_transforms(vars.xi_part0)
            T1 = self._twists_to_transforms(vars.xi_part1)

            l_motion = motion_fit_loss(tracks.xyz, tracks.valid, w, T0, T1)
            l_smooth = feature_smoothness_loss(w, graph)
            l_rigid = rigidity_consistency_loss(tracks.xyz, tracks.valid, w, T0, T1)

            use_pair = it >= self.schedule.pair_start_iter
            use_cog = it >= self.schedule.cog_start_iter
            use_balance = self.loss_weights.lambda_balance > 0.0

            l_pair = (
                pairwise_rigidity_loss(
                    tracks.xyz,
                    tracks.valid,
                    w,
                    num_pairs=pair_n,
                    margin=pair_margin,
                    lambda_sep=pair_lambda_sep,
                )
                if use_pair
                else w.new_tensor(0.0)
            )
            l_cog = cog_rigidity_loss(tracks.xyz, tracks.valid, w) if use_cog else w.new_tensor(0.0)
            l_bal = balance_loss(w) if use_balance else w.new_tensor(0.0)

            total = total_segmentation_loss(
                l_motion,
                l_smooth,
                l_rigid,
                l_pair,
                l_cog,
                l_bal,
                self.loss_weights,
                use_pair=use_pair,
                use_cog=use_cog,
                use_balance=use_balance,
            )
            all_finite = bool(
                torch.isfinite(l_motion)
                and torch.isfinite(l_smooth)
                and torch.isfinite(l_rigid)
                and torch.isfinite(l_pair)
                and torch.isfinite(l_cog)
                and torch.isfinite(l_bal)
                and torch.isfinite(total)
            )
            if all_finite:
                with torch.no_grad():
                    prev_logits = vars.logits.detach().clone()
                    prev_xi0 = vars.xi_part0.detach().clone()
                    prev_xi1 = vars.xi_part1.detach().clone()
                total.backward()
                torch.nn.utils.clip_grad_norm_(vars.parameters(), grad_clip)
                opt.step()
                with torch.no_grad():
                    vars.xi_part0.clamp_(-3.0, 3.0)
                    vars.xi_part1.clamp_(-3.0, 3.0)
                    params_finite = bool(
                        torch.isfinite(vars.logits).all()
                        and torch.isfinite(vars.xi_part0).all()
                        and torch.isfinite(vars.xi_part1).all()
                    )
                    if not params_finite:
                        nonfinite_steps += 1
                        vars.logits.copy_(prev_logits)
                        vars.xi_part0.copy_(prev_xi0)
                        vars.xi_part1.copy_(prev_xi1)
                        if debug_losses:
                            msg = f"[Seg][{it+1}/{iters}] non-finite params after step, state restored"
                            if show_progress:
                                tqdm.write(msg)
                            else:
                                print(msg)
            else:
                nonfinite_steps += 1
                with torch.no_grad():
                    vars.logits.nan_to_num_(0.0, 0.0, 0.0)
                    vars.xi_part0.nan_to_num_(0.0, 0.0, 0.0)
                    vars.xi_part1.nan_to_num_(0.0, 0.0, 0.0)
                    vars.xi_part0.clamp_(-3.0, 3.0)
                    vars.xi_part1.clamp_(-3.0, 3.0)
                l_motion = torch.nan_to_num(l_motion, nan=0.0, posinf=0.0, neginf=0.0)
                l_smooth = torch.nan_to_num(l_smooth, nan=0.0, posinf=0.0, neginf=0.0)
                l_rigid = torch.nan_to_num(l_rigid, nan=0.0, posinf=0.0, neginf=0.0)
                l_pair = torch.nan_to_num(l_pair, nan=0.0, posinf=0.0, neginf=0.0)
                l_cog = torch.nan_to_num(l_cog, nan=0.0, posinf=0.0, neginf=0.0)
                l_bal = torch.nan_to_num(l_bal, nan=0.0, posinf=0.0, neginf=0.0)
                total = torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)
                if debug_losses:
                    msg = f"[Seg][{it+1}/{iters}] non-finite loss detected, optimizer step skipped"
                    if show_progress:
                        tqdm.write(msg)
                    else:
                        print(msg)

            diagnostics["loss_total"].append(float(total.detach().cpu()))
            diagnostics["loss_motion"].append(float(l_motion.detach().cpu()))
            diagnostics["loss_smooth"].append(float(l_smooth.detach().cpu()))
            rigid_value = float(l_rigid.detach().cpu())
            diagnostics["loss_rigid"].append(rigid_value)
            diagnostics["loss_entropy"].append(rigid_value)
            diagnostics["loss_pair"].append(float(l_pair.detach().cpu()))
            diagnostics["loss_cog"].append(float(l_cog.detach().cpu()))
            diagnostics["loss_balance"].append(float(l_bal.detach().cpu()))

            if show_progress:
                pbar.set_postfix(
                    total=f"{diagnostics['loss_total'][-1]:.4g}",
                    motion=f"{diagnostics['loss_motion'][-1]:.4g}",
                    rigid=f"{rigid_value:.4g}",
                    smooth=f"{diagnostics['loss_smooth'][-1]:.4g}",
                )
            if debug_losses and (
                it == 0 or it == iters - 1 or ((it + 1) % max(1, debug_every) == 0)
            ):
                msg = (
                    f"[Seg][{it+1}/{iters}] total={diagnostics['loss_total'][-1]:.6f} "
                    f"motion={diagnostics['loss_motion'][-1]:.6f} "
                    f"rigid={rigid_value:.6f} "
                    f"smooth={diagnostics['loss_smooth'][-1]:.6f} "
                    f"pair={diagnostics['loss_pair'][-1]:.6f} "
                    f"cog={diagnostics['loss_cog'][-1]:.6f} "
                    f"balance={diagnostics['loss_balance'][-1]:.6f}"
                )
                if show_progress:
                    tqdm.write(msg)
                else:
                    print(msg)
        if show_progress:
            pbar.close()

        point_logits = torch.nan_to_num(vars.logits.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        point_probs = torch.sigmoid(point_logits)
        point_labels = (point_probs >= 0.5).long()

        if image_size is None:
            max_u = int(torch.ceil(tracks.xy[..., 0].max()).item()) + 1
            max_v = int(torch.ceil(tracks.xy[..., 1].max()).item()) + 1
            image_size = (max(1, max_v), max(1, max_u))

        masks_per_frame = build_dense_masks_from_tracks(tracks.xy, point_labels, image_size)

        transforms_part0 = self._twists_to_transforms(vars.xi_part0).detach()
        transforms_part1 = self._twists_to_transforms(vars.xi_part1).detach()
        diagnostics["nonfinite_steps"] = [float(nonfinite_steps)]

        return SegmentationResult(
            point_logits=point_logits,
            point_probs=point_probs,
            point_labels=point_labels,
            masks_per_frame=masks_per_frame,
            transforms_part0=transforms_part0,
            transforms_part1=transforms_part1,
            diagnostics=diagnostics,
        )
