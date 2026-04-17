import torch

from articulation.matching.multiview_consistency import (
    MultiViewConsistencyWeights,
    multiview_consistency_score,
)


def _K():
    return torch.tensor([[120.0, 0.0, 32.0], [0.0, 120.0, 24.0], [0.0, 0.0, 1.0]], dtype=torch.float32)


def _T_identity():
    return torch.eye(4, dtype=torch.float32)


def _project(X, K):
    u = K[0, 0] * (X[0] / X[2]) + K[0, 2]
    v = K[1, 1] * (X[1] / X[2]) + K[1, 2]
    return torch.tensor([u, v], dtype=torch.float32)


def test_multiview_consistency_low_for_coherent_observations():
    base = torch.tensor([0.1, -0.05, 1.2], dtype=torch.float32)
    world = torch.stack([
        base,
        base + torch.tensor([0.001, -0.001, 0.0]),
        base + torch.tensor([-0.001, 0.001, 0.0]),
    ], dim=0)

    K_all = torch.stack([_K(), _K(), _K()], dim=0)
    T_all = torch.stack([_T_identity(), _T_identity(), _T_identity()], dim=0)
    xy = torch.stack([_project(w, _K()) for w in world], dim=0)
    desc = torch.randn(3, 16)

    res = multiview_consistency_score(
        world_points=world,
        observed_xy=xy,
        desc=desc,
        K_all=K_all,
        T_cw_all=T_all,
        weights=MultiViewConsistencyWeights(alpha_world=1.0, alpha_reproj=0.5, alpha_feat=0.0),
    )

    assert res.world_error < 0.01
    assert res.reproj_error < 1.0
    assert res.score < 0.5


def test_multiview_consistency_high_for_inconsistent_observations():
    world = torch.stack([
        torch.tensor([0.1, -0.05, 1.2], dtype=torch.float32),
        torch.tensor([0.8, 0.5, 1.2], dtype=torch.float32),
        torch.tensor([-0.7, 0.2, 1.2], dtype=torch.float32),
    ], dim=0)

    K_all = torch.stack([_K(), _K(), _K()], dim=0)
    T_all = torch.stack([_T_identity(), _T_identity(), _T_identity()], dim=0)
    # Fake all observations to the same pixel despite inconsistent world points.
    xy = torch.tensor([[40.0, 30.0], [40.0, 30.0], [40.0, 30.0]], dtype=torch.float32)
    desc = torch.randn(3, 16)

    res = multiview_consistency_score(
        world_points=world,
        observed_xy=xy,
        desc=desc,
        K_all=K_all,
        T_cw_all=T_all,
        weights=MultiViewConsistencyWeights(alpha_world=1.0, alpha_reproj=1.0, alpha_feat=0.0),
    )

    assert res.world_error > 0.2
    assert res.score > 0.2
