import torch

from articulation.joint.multiview import (
    aggregate_model_priors,
    robust_axis_point_consensus,
    robust_direction_consensus,
)


def test_aggregate_model_priors_prefers_lower_loss():
    per_view_losses = [
        {"revolute": 0.10, "prismatic": 0.30, "screw": 0.25},
        {"revolute": 0.12, "prismatic": 0.40, "screw": 0.22},
        {"revolute": 0.09, "prismatic": 0.35, "screw": 0.20},
    ]
    priors = aggregate_model_priors(per_view_losses, temperature=0.02)
    assert priors["revolute"] > priors["screw"] > priors["prismatic"]
    assert abs(sum(priors.values()) - 1.0) < 1e-6


def test_robust_direction_consensus_handles_sign_ambiguity():
    dirs = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.98, 0.05, 0.0],
            [-0.97, -0.04, 0.0],
        ],
        dtype=torch.float32,
    )
    fused, inlier, signs = robust_direction_consensus(dirs, max_angle_deg=10.0)
    # Sign-invariant check.
    assert torch.abs(torch.dot(fused, torch.tensor([1.0, 0.0, 0.0]))) > 0.98
    assert bool(inlier.all())
    assert signs.shape[0] == dirs.shape[0]


def test_robust_axis_point_consensus_recovers_intersection():
    torch.manual_seed(0)
    gt = torch.tensor([0.4, -0.2, 0.9], dtype=torch.float32)
    dirs = torch.tensor(
        [
            [0.2, 0.1, 1.0],
            [-0.3, 0.2, 0.9],
            [0.1, -0.4, 0.95],
            [-0.2, -0.3, 1.1],
        ],
        dtype=torch.float32,
    )
    dirs = dirs / torch.linalg.norm(dirs, dim=1, keepdim=True)

    pts = []
    for i in range(dirs.shape[0]):
        d = dirs[i]
        n = torch.tensor([d[1], -d[0], 0.0], dtype=torch.float32)
        if torch.linalg.norm(n) < 1e-6:
            n = torch.tensor([0.0, d[2], -d[1]], dtype=torch.float32)
        n = n / torch.linalg.norm(n)
        pts.append(gt + 0.01 * (i + 1) * n)
    pts = torch.stack(pts, dim=0)

    q, inliers = robust_axis_point_consensus(
        line_points=pts,
        line_dirs=dirs,
        max_dist=0.08,
    )
    assert q is not None
    assert bool(inliers.all())
    assert torch.linalg.norm(q - gt) < 0.05
