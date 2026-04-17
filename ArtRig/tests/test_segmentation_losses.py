import torch

from articulation.data import FeatureGraph
from articulation.geometry.se3 import se3_exp
from articulation.segmentation.losses import (
    cog_rigidity_loss,
    feature_smoothness_loss,
    motion_fit_loss,
    pairwise_rigidity_loss,
    rigidity_consistency_loss,
)


def _make_two_part_motion(p=20, t=5):
    x0 = torch.randn(p, 3)
    xyz = x0[:, None, :].repeat(1, t, 1)
    # part 1 moves along x
    move = torch.linspace(0, 0.2, t)
    xyz[p // 2 :, :, 0] += move[None, :]
    valid = torch.ones(p, t, dtype=torch.bool)
    w = torch.zeros(p)
    w[p // 2 :] = 1.0
    return xyz, valid, w


def test_rigidity_consistency_prefers_correct_partition():
    xyz, valid, w = _make_two_part_motion(p=30, t=6)
    t = xyz.shape[1]
    T0 = se3_exp(torch.zeros(t - 1, 6))
    xi1 = torch.zeros(t - 1, 6)
    xi1[:, 3] = torch.linspace(0, 0.2, t)[1:]
    T1 = se3_exp(xi1)

    l = rigidity_consistency_loss(xyz, valid, w, T0, T1)
    l_flip = rigidity_consistency_loss(xyz, valid, 1.0 - w, T0, T1)
    assert torch.isfinite(l)
    assert torch.isfinite(l_flip)
    assert l <= l_flip + 1e-6


def test_feature_smoothness_shape():
    w = torch.tensor([0.1, 0.2, 0.8, 0.9])
    idx = torch.tensor([[1, 2], [0, 2], [0, 3], [1, 2]])
    wei = torch.ones_like(idx, dtype=torch.float32) / 2.0
    g = FeatureGraph(nn_idx=idx, nn_weight=wei)
    l = feature_smoothness_loss(w, g)
    assert torch.isfinite(l)


def test_motion_fit_identity_small():
    p, t = 6, 4
    x0 = torch.randn(p, 1, 3)
    xyz = x0.repeat(1, t, 1)
    valid = torch.ones(p, t, dtype=torch.bool)
    w = torch.rand(p)
    T = se3_exp(torch.zeros(t - 1, 6))
    l = motion_fit_loss(xyz, valid, w, T, T)
    assert l < 1e-6


def test_pair_and_cog_rigidity_correct_partition():
    xyz, valid, w = _make_two_part_motion(p=30, t=6)
    l_pair = pairwise_rigidity_loss(xyz, valid, w, num_pairs=128)
    l_cog = cog_rigidity_loss(xyz, valid, w)
    assert torch.isfinite(l_pair)
    assert torch.isfinite(l_cog)

    l_cog_flip = cog_rigidity_loss(xyz, valid, 1.0 - w)
    assert l_cog <= l_cog_flip + 1e-4
