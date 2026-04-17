import torch

from articulation.data.dataclasses import RelativeMotionResult
from articulation.joint.consensus import ConsensusResult
from articulation.joint.optimizer import optimize_joint_candidates


def _make_prismatic_motion(p=30, t=8):
    axis = torch.tensor([1.0, 0.0, 0.0])
    x0 = torch.randn(p, 3) * 0.2
    state = torch.linspace(0, 0.3, t)
    traj = x0[:, None, :] + state[None, :, None] * axis[None, None, :]
    valid = torch.ones(p, t, dtype=torch.bool)
    return x0, traj, valid, axis


def _make_revolute_motion(p=30, t=8):
    axis = torch.tensor([0.0, 0.0, 1.0])
    x0 = torch.randn(p, 3) * 0.2
    theta = torch.linspace(0, 0.5, t)
    cos = torch.cos(theta)[None, :, None]
    sin = torch.sin(theta)[None, :, None]
    x = x0[:, 0:1].unsqueeze(1)
    y = x0[:, 1:2].unsqueeze(1)
    z = x0[:, 2:3].unsqueeze(1)
    xr = x * cos - y * sin
    yr = x * sin + y * cos
    traj = torch.cat([xr, yr, z.repeat(1, t, 1)], dim=2)
    valid = torch.ones(p, t, dtype=torch.bool)
    return x0, traj, valid, axis


def _optimize(rel, axis):
    consensus = ConsensusResult(
        type_priors={"revolute": 0.34, "prismatic": 0.33, "screw": 0.33},
        axis_dir=axis,
        axis_point=None,
        pitch=None,
    )
    cfg = {
        "models": {"candidates": ["revolute", "prismatic"]},
        "loss": {
            "lambda_fit": 1.0,
            "lambda_temporal": 0.1,
            "lambda_axis": 0.05,
            "lambda_axis_point": 0.0,
            "lambda_pitch": 0.01,
        },
        "optimizer": {"lr_axis": 1e-2, "lr_axis_point": 1e-2, "lr_state": 1e-2, "lr_pitch": 1e-3, "iterations": 60},
    }
    return optimize_joint_candidates(rel=rel, consensus=consensus, cfg=cfg)


def test_prismatic_model_selection():
    x0, traj, valid, axis = _make_prismatic_motion()
    rel = RelativeMotionResult(
        reference_part=0,
        moving_part=1,
        canonical_points=x0,
        moving_points_rel=traj,
        valid=valid,
        weights=torch.ones(x0.shape[0]),
        ref_transform_inv=torch.eye(4).unsqueeze(0).repeat(traj.shape[1], 1, 1),
        diagnostics={},
    )
    result = _optimize(rel, axis)
    assert result.best_model == "prismatic"


def test_revolute_model_selection():
    x0, traj, valid, axis = _make_revolute_motion()
    rel = RelativeMotionResult(
        reference_part=0,
        moving_part=1,
        canonical_points=x0,
        moving_points_rel=traj,
        valid=valid,
        weights=torch.ones(x0.shape[0]),
        ref_transform_inv=torch.eye(4).unsqueeze(0).repeat(traj.shape[1], 1, 1),
        diagnostics={},
    )
    result = _optimize(rel, axis)
    assert result.best_model == "revolute"
