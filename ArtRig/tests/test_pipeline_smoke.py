import torch

from articulation.data import FeatureGraph, TrackBatch
from articulation.pipeline.full_pipeline import run_full_pipeline



def test_pipeline_smoke_tiny():
    torch.manual_seed(0)
    p, t = 24, 5
    xy = torch.rand(p, t, 2) * 32

    # Two-part synthetic motion
    x0 = torch.randn(p, 3)
    state = torch.linspace(0, 0.2, t)
    xyz = torch.zeros(p, t, 3)
    for ti in range(t):
        xyz[: p // 2, ti] = x0[: p // 2]
        xyz[p // 2 :, ti] = x0[p // 2 :] + torch.tensor([state[ti], 0.0, 0.0])

    valid = torch.ones(p, t, dtype=torch.bool)
    feat = torch.cat([torch.randn(p // 2, 8) - 1.0, torch.randn(p - p // 2, 8) + 1.0], dim=0)

    tracks = TrackBatch(
        xy=xy,
        xyz=xyz,
        valid=valid,
        anchor_frame=0,
        point_ids=torch.arange(p),
        feature=feat,
        confidence=None,
    )

    idx = torch.topk(torch.cdist(feat, feat), k=3, largest=False).indices[:, 1:]
    w = torch.ones_like(idx, dtype=torch.float32) / idx.shape[1]
    graph = FeatureGraph(nn_idx=idx, nn_weight=w)

    seg_cfg = {
        "features": {"num_neighbors": 2},
        "loss": {
            "lambda_motion": 10.0,
            "lambda_smooth": 1.0,
            "lambda_rigid": 0.01,
            "lambda_pair": 0.5,
            "lambda_cog": 0.5,
            "lambda_balance": 0.1,
            "pair_margin": 0.01,
            "pair_lambda_sep": 1.0,
        },
        "optimizer": {
            "lr_logits": 1e-2,
            "lr_twists": 1e-3,
            "iterations": 10,
            "grad_clip": 1.0,
            "schedule": {"pair_start_iter": 2, "cog_start_iter": 4},
        },
        "sampling": {"num_pairs": 64},
    }

    joint_cfg = {
        "sampling": {"num_point_samples": 16, "strategy": "random"},
        "models": {"candidates": ["revolute", "prismatic"], "screw_complexity_penalty": 0.05},
        "loss": {
            "lambda_fit": 1.0,
            "lambda_temporal": 0.1,
            "lambda_axis": 0.01,
            "lambda_axis_point": 0.0,
            "lambda_pitch": 0.01,
        },
        "optimizer": {
            "lr_axis": 1e-2,
            "lr_axis_point": 1e-2,
            "lr_state": 1e-2,
            "lr_pitch": 1e-3,
            "iterations": 20,
        },
    }

    seg, joint = run_full_pipeline(tracks, graph, seg_cfg=seg_cfg, joint_cfg=joint_cfg)
    assert seg.point_labels.shape[0] == p
    assert joint.best_model in {"revolute", "prismatic"}
