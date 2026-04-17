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
    entropy_loss,
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
    canrigid_start_iter: int = 40
    pair_start_iter: int = 60
    cog_start_iter: int = 120
    entropy_decay_start: int = 40
    entropy_decay_end: int = 160


@dataclass
class SegmentationLossNormalization:
    enabled: bool = True
    beta: float = 0.9
    eps: float = 1e-6
    max_scale: float = 100.0


class SegmentationTrainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        loss_cfg = cfg.get("loss", {})
        self.loss_weights = SegmentationLossWeights(
            lambda_motion=float(loss_cfg.get("lambda_motion", 200.0)),
            lambda_smooth=float(loss_cfg.get("lambda_smooth", 10.0)),
            lambda_canrigid=float(loss_cfg.get("lambda_canrigid", loss_cfg.get("lambda_rigid", 5.0))),
            lambda_pair=float(loss_cfg.get("lambda_pair", 2.0)),
            lambda_cog=float(loss_cfg.get("lambda_cog", 1.0)),
            lambda_ent=float(loss_cfg.get("lambda_ent", loss_cfg.get("lambda_entropy", 0.001))),
            lambda_balance=float(loss_cfg.get("lambda_balance", 0.0)),
        )
        self.loss_norm = SegmentationLossNormalization(
            enabled=bool(loss_cfg.get("normalize_terms", True)),
            beta=float(loss_cfg.get("normalize_beta", 0.9)),
            eps=float(loss_cfg.get("normalize_eps", 1e-6)),
            max_scale=float(loss_cfg.get("normalize_max_scale", 100.0)),
        )
        schedule_cfg = cfg.get("optimizer", {}).get("schedule", {})
        self.schedule = SegmentationSchedule(
            warmup_iters=int(schedule_cfg.get("warmup_iters", 80)),
            canrigid_start_iter=int(schedule_cfg.get("canrigid_start_iter", schedule_cfg.get("canrigid_start", 40))),
            pair_start_iter=int(schedule_cfg.get("pair_start_iter", 60)),
            cog_start_iter=int(schedule_cfg.get("cog_start_iter", 120)),
            entropy_decay_start=int(schedule_cfg.get("entropy_decay_start", 40)),
            entropy_decay_end=int(schedule_cfg.get("entropy_decay_end", 160)),
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
        wandb_run: object | None = None,
        wandb_prefix: str = "seg",
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
            "loss_entropy_raw": [],
            "loss_entropy": [],  # legacy name kept for downstream compatibility
            "loss_pair": [],
            "loss_cog": [],
            "loss_balance": [],
            "loss_motion_eff": [],
            "loss_smooth_eff": [],
            "loss_rigid_eff": [],
            "loss_entropy_eff": [],
            "loss_pair_eff": [],
            "loss_cog_eff": [],
            "loss_balance_eff": [],
            "loss_motion_weighted": [],
            "loss_smooth_weighted": [],
            "loss_rigid_weighted": [],
            "loss_entropy_weighted": [],
            "loss_pair_weighted": [],
            "loss_cog_weighted": [],
            "loss_balance_weighted": [],
            "scale_motion": [],
            "scale_smooth": [],
            "scale_rigid": [],
            "scale_pair": [],
            "scale_cog": [],
            "scale_balance": [],
        }
        nonfinite_steps = 0
        loss_ema: dict[str, float] = {}

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
            use_canrigid = it >= self.schedule.canrigid_start_iter
            l_canrigid = (
                rigidity_consistency_loss(tracks.xyz, tracks.valid, w, T0, T1)
                if use_canrigid
                else w.new_tensor(0.0)
            )
            l_ent = entropy_loss(w)

            use_pair = it >= self.schedule.pair_start_iter
            use_cog = it >= self.schedule.cog_start_iter
            use_balance = self.loss_weights.lambda_balance > 0.0

            ent_factor = 1.0
            if it >= self.schedule.entropy_decay_start:
                if it >= self.schedule.entropy_decay_end:
                    ent_factor = 0.0
                else:
                    den = max(self.schedule.entropy_decay_end - self.schedule.entropy_decay_start, 1)
                    ent_factor = 1.0 - float(it - self.schedule.entropy_decay_start) / float(den)
            l_ent_eff = l_ent * float(max(ent_factor, 0.0))
            use_entropy = bool(self.loss_weights.lambda_ent > 0.0 and ent_factor > 0.0)

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

            def _effective(name: str, value: torch.Tensor, active: bool) -> tuple[torch.Tensor, float]:
                if not active:
                    return value.new_tensor(0.0), 0.0
                if not self.loss_norm.enabled:
                    return value, 1.0

                raw_abs = float(torch.abs(value.detach()).cpu())
                prev = loss_ema.get(name, None)
                if prev is None:
                    ema = max(raw_abs, self.loss_norm.eps)
                else:
                    beta = min(max(self.loss_norm.beta, 0.0), 0.9999)
                    ema = beta * prev + (1.0 - beta) * raw_abs
                    ema = max(ema, self.loss_norm.eps)
                loss_ema[name] = ema
                scale = 1.0 / max(ema, self.loss_norm.eps)
                scale = min(scale, self.loss_norm.max_scale)
                return value * float(scale), float(scale)

            l_motion_eff, s_motion = _effective("motion", l_motion, True)
            l_smooth_eff, s_smooth = _effective("smooth", l_smooth, True)
            l_canrigid_eff, s_rigid = _effective("rigid", l_canrigid, use_canrigid)
            l_pair_eff, s_pair = _effective("pair", l_pair, use_pair)
            l_cog_eff, s_cog = _effective("cog", l_cog, use_cog)
            l_bal_eff, s_bal = _effective("balance", l_bal, use_balance)

            total = total_segmentation_loss(
                l_motion_eff,
                l_smooth_eff,
                l_canrigid_eff,
                l_ent_eff,
                l_pair_eff,
                l_cog_eff,
                l_bal_eff,
                self.loss_weights,
                use_canrigid=use_canrigid,
                use_pair=use_pair,
                use_cog=use_cog,
                use_entropy=use_entropy,
                use_balance=use_balance,
            )
            all_finite = bool(
                torch.isfinite(l_motion)
                and torch.isfinite(l_smooth)
                and torch.isfinite(l_canrigid)
                and torch.isfinite(l_ent_eff)
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
                l_canrigid = torch.nan_to_num(l_canrigid, nan=0.0, posinf=0.0, neginf=0.0)
                l_ent_eff = torch.nan_to_num(l_ent_eff, nan=0.0, posinf=0.0, neginf=0.0)
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
            rigid_value = float(l_canrigid.detach().cpu())
            diagnostics["loss_rigid"].append(rigid_value)
            entropy_raw_value = float(l_ent.detach().cpu())
            entropy_eff_value = float(l_ent_eff.detach().cpu())
            diagnostics["loss_entropy_raw"].append(entropy_raw_value)
            diagnostics["loss_entropy"].append(float(l_ent_eff.detach().cpu()))
            diagnostics["loss_entropy_eff"].append(entropy_eff_value)
            diagnostics["loss_pair"].append(float(l_pair.detach().cpu()))
            diagnostics["loss_cog"].append(float(l_cog.detach().cpu()))
            diagnostics["loss_balance"].append(float(l_bal.detach().cpu()))

            motion_eff_f = float(l_motion_eff.detach().cpu())
            smooth_eff_f = float(l_smooth_eff.detach().cpu())
            rigid_eff_f = float(l_canrigid_eff.detach().cpu())
            pair_eff_f = float(l_pair_eff.detach().cpu())
            cog_eff_f = float(l_cog_eff.detach().cpu())
            bal_eff_f = float(l_bal_eff.detach().cpu())
            diagnostics["loss_motion_eff"].append(motion_eff_f)
            diagnostics["loss_smooth_eff"].append(smooth_eff_f)
            diagnostics["loss_rigid_eff"].append(rigid_eff_f)
            diagnostics["loss_pair_eff"].append(pair_eff_f)
            diagnostics["loss_cog_eff"].append(cog_eff_f)
            diagnostics["loss_balance_eff"].append(bal_eff_f)

            w_motion = self.loss_weights.lambda_motion * motion_eff_f
            w_smooth = self.loss_weights.lambda_smooth * smooth_eff_f
            w_rigid = self.loss_weights.lambda_canrigid * rigid_eff_f if use_canrigid else 0.0
            w_ent = self.loss_weights.lambda_ent * entropy_eff_value if use_entropy else 0.0
            w_pair = self.loss_weights.lambda_pair * pair_eff_f if use_pair else 0.0
            w_cog = self.loss_weights.lambda_cog * cog_eff_f if use_cog else 0.0
            w_bal = self.loss_weights.lambda_balance * bal_eff_f if use_balance else 0.0
            diagnostics["loss_motion_weighted"].append(w_motion)
            diagnostics["loss_smooth_weighted"].append(w_smooth)
            diagnostics["loss_rigid_weighted"].append(w_rigid)
            diagnostics["loss_entropy_weighted"].append(w_ent)
            diagnostics["loss_pair_weighted"].append(w_pair)
            diagnostics["loss_cog_weighted"].append(w_cog)
            diagnostics["loss_balance_weighted"].append(w_bal)
            diagnostics["scale_motion"].append(float(s_motion))
            diagnostics["scale_smooth"].append(float(s_smooth))
            diagnostics["scale_rigid"].append(float(s_rigid))
            diagnostics["scale_pair"].append(float(s_pair))
            diagnostics["scale_cog"].append(float(s_cog))
            diagnostics["scale_balance"].append(float(s_bal))

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
                    f"motion={diagnostics['loss_motion'][-1]:.6f} (w={w_motion:.6f}) "
                    f"rigid={rigid_value:.6f} (w={w_rigid:.6f}) "
                    f"smooth={diagnostics['loss_smooth'][-1]:.6f} (w={w_smooth:.6f}) "
                    f"pair={diagnostics['loss_pair'][-1]:.6f} (w={w_pair:.6f}) "
                    f"cog={diagnostics['loss_cog'][-1]:.6f} (w={w_cog:.6f}) "
                    f"balance={diagnostics['loss_balance'][-1]:.6f} (w={w_bal:.6f})"
                )
                if show_progress:
                    tqdm.write(msg)
                else:
                    print(msg)
            if wandb_run is not None:
                wb = {
                    f"{wandb_prefix}/iter": float(it + 1),
                    f"{wandb_prefix}/total": float(diagnostics["loss_total"][-1]),
                    f"{wandb_prefix}/raw/motion": float(diagnostics["loss_motion"][-1]),
                    f"{wandb_prefix}/raw/smooth": float(diagnostics["loss_smooth"][-1]),
                    f"{wandb_prefix}/raw/rigid": float(diagnostics["loss_rigid"][-1]),
                    f"{wandb_prefix}/raw/entropy": float(entropy_raw_value),
                    f"{wandb_prefix}/raw/pair": float(diagnostics["loss_pair"][-1]),
                    f"{wandb_prefix}/raw/cog": float(diagnostics["loss_cog"][-1]),
                    f"{wandb_prefix}/raw/balance": float(diagnostics["loss_balance"][-1]),
                    f"{wandb_prefix}/effective/entropy": float(entropy_eff_value),
                    f"{wandb_prefix}/weighted/motion": float(w_motion),
                    f"{wandb_prefix}/weighted/smooth": float(w_smooth),
                    f"{wandb_prefix}/weighted/rigid": float(w_rigid),
                    f"{wandb_prefix}/weighted/entropy": float(w_ent),
                    f"{wandb_prefix}/weighted/pair": float(w_pair),
                    f"{wandb_prefix}/weighted/cog": float(w_cog),
                    f"{wandb_prefix}/weighted/balance": float(w_bal),
                    f"{wandb_prefix}/scale/motion": float(s_motion),
                    f"{wandb_prefix}/scale/smooth": float(s_smooth),
                    f"{wandb_prefix}/scale/rigid": float(s_rigid),
                    f"{wandb_prefix}/scale/pair": float(s_pair),
                    f"{wandb_prefix}/scale/cog": float(s_cog),
                    f"{wandb_prefix}/scale/balance": float(s_bal),
                    f"{wandb_prefix}/schedule/entropy_factor": float(max(ent_factor, 0.0)),
                    f"{wandb_prefix}/schedule/use_canrigid": float(1.0 if use_canrigid else 0.0),
                    f"{wandb_prefix}/schedule/use_pair": float(1.0 if use_pair else 0.0),
                    f"{wandb_prefix}/schedule/use_cog": float(1.0 if use_cog else 0.0),
                    f"{wandb_prefix}/schedule/use_entropy": float(1.0 if use_entropy else 0.0),
                }
                wandb_run.log(wb)
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
