import torch

from articulation.data.dataclasses import KeypointBatch, MatchBatch
from articulation.matching.filters import filter_cross_time_matches, filter_same_time_multiview_matches


def _kp(xy, depth, valid=None):
    xy_t = torch.tensor(xy, dtype=torch.float32)
    d = torch.tensor(depth, dtype=torch.float32)
    n = xy_t.shape[0]
    if valid is None:
        valid_t = torch.ones(n, dtype=torch.bool)
    else:
        valid_t = torch.tensor(valid, dtype=torch.bool)
    return KeypointBatch(
        xy=xy_t,
        desc=torch.randn(n, 8),
        score=torch.ones(n),
        depth=d,
        valid=valid_t,
        t=0,
        v=0,
    )


def _K():
    return torch.tensor([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)


def _T():
    return torch.eye(4, dtype=torch.float32)


def test_same_time_multiview_3d_filtering_keeps_consistent_match():
    a = _kp([[10.0, 10.0], [20.0, 20.0]], [1.0, 1.0])
    b = _kp([[10.1, 10.0], [80.0, 80.0]], [1.0, 1.0])

    m = MatchBatch(
        idx_a=torch.tensor([0, 1]),
        idx_b=torch.tensor([0, 1]),
        confidence=torch.tensor([0.9, 0.9]),
        pair_type="same_time_multiview",
    )

    out = filter_same_time_multiview_matches(
        match=m,
        frame_a=a,
        frame_b=b,
        K_a=_K(),
        K_b=_K(),
        T_cw_a=_T(),
        T_cw_b=_T(),
        threshold_same_time=0.02,
    )

    assert out.idx_a.numel() == 1
    assert int(out.idx_a[0].item()) == 0
    assert int(out.idx_b[0].item()) == 0


def test_cross_time_filter_keeps_reasonable_motion_and_confidence():
    a = _kp([[10.0, 10.0], [20.0, 20.0]], [1.0, 1.0])
    b = _kp([[12.0, 10.0], [120.0, 120.0]], [1.05, 1.0])

    m = MatchBatch(
        idx_a=torch.tensor([0, 1]),
        idx_b=torch.tensor([0, 1]),
        confidence=torch.tensor([0.8, 0.05]),
        pair_type="cross_time_same_view",
    )

    out = filter_cross_time_matches(
        match=m,
        frame_a=a,
        frame_b=b,
        min_confidence=0.2,
        max_pixel_jump=20.0,
        max_depth_jump=0.2,
    )

    assert out.idx_a.numel() == 1
    assert int(out.idx_a[0].item()) == 0
    assert int(out.idx_b[0].item()) == 0
