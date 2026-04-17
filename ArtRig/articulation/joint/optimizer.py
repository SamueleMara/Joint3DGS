from __future__ import annotations

import torch
from tqdm.auto import tqdm

from articulation.data.dataclasses import JointCandidateResult, JointResult, RelativeMotionResult
from articulation.joint.consensus import ConsensusResult
from articulation.joint.losses import (
    JointLossWeights,
    axis_point_reg,
    axis_prior_loss,
    pitch_reg,
    temporal_smoothness_loss,
    total_joint_loss,
    trajectory_reconstruction_loss,
)
from articulation.joint.models import PrismaticModel, RevoluteModel, ScrewModel
from articulation.joint.selection import select_best_candidate



def _build_model(model_name: str, t: int, consensus: ConsensusResult) -> torch.nn.Module:
    if model_name == "revolute":
        return RevoluteModel(num_frames=t, axis_dir=consensus.axis_dir, axis_point=consensus.axis_point)
    if model_name == "prismatic":
        return PrismaticModel(num_frames=t, axis_dir=consensus.axis_dir)
    if model_name == "screw":
        return ScrewModel(
            num_frames=t,
            axis_dir=consensus.axis_dir,
            axis_point=consensus.axis_point,
            pitch=consensus.pitch,
        )
    raise ValueError(f"Unknown model name: {model_name}")



def fit_candidate_model(
    model_name: str,
    rel: RelativeMotionResult,
    consensus: ConsensusResult,
    cfg: dict,
    show_progress: bool = False,
    debug_losses: bool = False,
    wandb_run: object | None = None,
    wandb_prefix: str = "joint",
) -> JointCandidateResult:
    t = rel.moving_points_rel.shape[1]
    model = _build_model(model_name, t=t, consensus=consensus).to(rel.moving_points_rel.device)

    loss_cfg = cfg.get("loss", {})
    weights = JointLossWeights(
        lambda_fit=float(loss_cfg.get("lambda_fit", 1.0)),
        lambda_temporal=float(loss_cfg.get("lambda_temporal", 0.1)),
        lambda_axis=float(loss_cfg.get("lambda_axis", 0.05)),
        lambda_axis_point=float(loss_cfg.get("lambda_axis_point", 0.0)),
        lambda_pitch=float(loss_cfg.get("lambda_pitch", 0.01)),
    )

    opt_cfg = cfg.get("optimizer", {})
    lr_axis = float(opt_cfg.get("lr_axis", 1e-2))
    lr_axis_point = float(opt_cfg.get("lr_axis_point", 5e-3))
    lr_state = float(opt_cfg.get("lr_state", 1e-2))
    lr_pitch = float(opt_cfg.get("lr_pitch", 1e-3))
    iters = int(opt_cfg.get("iterations", 500))
    debug_every = int(opt_cfg.get("debug_every", 25))

    params = []
    if hasattr(model, "axis_dir_raw"):
        params.append({"params": [model.axis_dir_raw], "lr": lr_axis})
    if hasattr(model, "axis_point"):
        params.append({"params": [model.axis_point], "lr": lr_axis_point})
    if hasattr(model, "state"):
        params.append({"params": [model.state], "lr": lr_state})
    if hasattr(model, "pitch"):
        params.append({"params": [model.pitch], "lr": lr_pitch})

    optim = torch.optim.Adam(params)

    axis_prior = consensus.axis_dir.to(rel.moving_points_rel.device)
    axis_point_prior = (
        consensus.axis_point.to(rel.moving_points_rel.device)
        if consensus.axis_point is not None
        else torch.zeros(3, device=rel.moving_points_rel.device)
    )

    loss_history: dict[str, list[float]] = {
        "loss_total": [],
        "loss_fit": [],
        "loss_temporal": [],
        "loss_axis": [],
        "loss_axis_point": [],
        "loss_pitch": [],
    }

    pbar = tqdm(
        range(iters),
        desc=f"Stage2/{model_name}",
        disable=not show_progress,
        leave=False,
    )
    for it in pbar:
        optim.zero_grad(set_to_none=True)
        pred = model(rel.canonical_points)

        l_fit = trajectory_reconstruction_loss(pred, rel.moving_points_rel, rel.valid, rel.weights)
        l_temp = temporal_smoothness_loss(model.state)
        l_axis = axis_prior_loss(model.axis_dir(), axis_prior)

        if hasattr(model, "axis_point"):
            l_axis_p = axis_point_reg(model.axis_point, axis_point_prior)
        else:
            l_axis_p = pred.new_tensor(0.0)

        if hasattr(model, "pitch"):
            l_pitch = pitch_reg(model.pitch)
        else:
            l_pitch = pred.new_tensor(0.0)

        loss = total_joint_loss(l_fit, l_temp, l_axis, l_axis_p, l_pitch, weights)
        loss.backward()
        optim.step()

        l_total = float(loss.detach().cpu())
        l_fit_f = float(l_fit.detach().cpu())
        l_temp_f = float(l_temp.detach().cpu())
        l_axis_f = float(l_axis.detach().cpu())
        l_axis_p_f = float(l_axis_p.detach().cpu())
        l_pitch_f = float(l_pitch.detach().cpu())
        w_fit = float(weights.lambda_fit * l_fit_f)
        w_temp = float(weights.lambda_temporal * l_temp_f)
        w_axis = float(weights.lambda_axis * l_axis_f)
        w_axis_p = float(weights.lambda_axis_point * l_axis_p_f)
        w_pitch = float(weights.lambda_pitch * l_pitch_f)

        loss_history["loss_total"].append(l_total)
        loss_history["loss_fit"].append(l_fit_f)
        loss_history["loss_temporal"].append(l_temp_f)
        loss_history["loss_axis"].append(l_axis_f)
        loss_history["loss_axis_point"].append(l_axis_p_f)
        loss_history["loss_pitch"].append(l_pitch_f)

        if show_progress:
            pbar.set_postfix(
                total=f"{l_total:.4g}",
                fit=f"{l_fit_f:.4g}",
                temp=f"{l_temp_f:.4g}",
                axis=f"{l_axis_f:.4g}",
            )
        if debug_losses and (
            it == 0 or it == iters - 1 or ((it + 1) % max(1, debug_every) == 0)
        ):
            msg = (
                f"[Joint:{model_name}][{it+1}/{iters}] total={l_total:.6f} "
                f"fit={l_fit_f:.6f} temp={l_temp_f:.6f} axis={l_axis_f:.6f} "
                f"axis_p={l_axis_p_f:.6f} pitch={l_pitch_f:.6f}"
            )
            if show_progress:
                tqdm.write(msg)
            else:
                print(msg)
        if wandb_run is not None:
            wandb_run.log(
                {
                    f"{wandb_prefix}/{model_name}/iter": float(it + 1),
                    f"{wandb_prefix}/{model_name}/total": float(l_total),
                    f"{wandb_prefix}/{model_name}/fit": float(l_fit_f),
                    f"{wandb_prefix}/{model_name}/temp": float(l_temp_f),
                    f"{wandb_prefix}/{model_name}/axis": float(l_axis_f),
                    f"{wandb_prefix}/{model_name}/axis_point": float(l_axis_p_f),
                    f"{wandb_prefix}/{model_name}/pitch": float(l_pitch_f),
                    f"{wandb_prefix}/{model_name}/weighted/fit": float(w_fit),
                    f"{wandb_prefix}/{model_name}/weighted/temp": float(w_temp),
                    f"{wandb_prefix}/{model_name}/weighted/axis": float(w_axis),
                    f"{wandb_prefix}/{model_name}/weighted/axis_point": float(w_axis_p),
                    f"{wandb_prefix}/{model_name}/weighted/pitch": float(w_pitch),
                }
            )
    if show_progress:
        pbar.close()

    with torch.no_grad():
        pred = model(rel.canonical_points)
        final_loss = float(trajectory_reconstruction_loss(pred, rel.moving_points_rel, rel.valid, rel.weights).cpu())

    axis_point = model.axis_point.detach().clone() if hasattr(model, "axis_point") else None
    pitch = model.pitch.detach().clone() if hasattr(model, "pitch") else None

    return JointCandidateResult(
        model_name=model_name,
        loss=final_loss,
        axis_dir=model.axis_dir().detach().clone(),
        axis_point=axis_point,
        pitch=pitch,
        state=model.state.detach().clone(),
        pred_points=pred.detach().clone(),
        diagnostics={
            "optimized": True,
            **loss_history,
        },
    )



def optimize_joint_candidates(
    rel: RelativeMotionResult,
    consensus: ConsensusResult,
    cfg: dict,
    show_progress: bool = False,
    debug_losses: bool = False,
    wandb_run: object | None = None,
    wandb_prefix: str = "joint",
) -> JointResult:
    candidates = cfg.get("models", {}).get("candidates", ["revolute", "prismatic", "screw"])

    cand_iter = tqdm(
        candidates,
        desc="Stage2/Candidate Models",
        disable=not show_progress,
        leave=False,
    )
    results: list[JointCandidateResult] = []
    for name in cand_iter:
        result = fit_candidate_model(
            name,
            rel,
            consensus,
            cfg,
            show_progress=show_progress,
            debug_losses=debug_losses,
            wandb_run=wandb_run,
            wandb_prefix=wandb_prefix,
        )
        results.append(result)
        if show_progress:
            best_so_far = min(results, key=lambda r: r.loss)
            cand_iter.set_postfix(best=best_so_far.model_name, loss=f"{best_so_far.loss:.4g}")
    if show_progress:
        cand_iter.close()
    best = select_best_candidate(results, screw_complexity_penalty=float(cfg.get("models", {}).get("screw_complexity_penalty", 0.05)))
    if wandb_run is not None:
        cand_losses = {r.model_name: float(r.loss) for r in results}
        wandb_run.log(
            {
                f"{wandb_prefix}/best_model": best.model_name,
                f"{wandb_prefix}/best_loss": float(best.loss),
                **{f"{wandb_prefix}/candidate_loss/{k}": v for k, v in cand_losses.items()},
            }
        )

    return JointResult(
        best_model=best.model_name,
        axis_dir=best.axis_dir,
        axis_point=best.axis_point,
        pitch=best.pitch,
        state=best.state,
        candidates=results,
        diagnostics={"num_candidates": len(results)},
    )
