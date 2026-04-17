import torch

from articulation.data.dataclasses import KeypointBatch, MatchBatch
from articulation.matching.build_tracks import PairMatchRecord, TrackBuildConfig, build_tracks_from_matches
from articulation.matching.multiview_consistency import MultiViewConsistencyWeights


def _kp(xy, depth, t, v):
    xy_t = torch.tensor(xy, dtype=torch.float32)
    n = xy_t.shape[0]
    return KeypointBatch(
        xy=xy_t,
        desc=torch.randn(n, 8),
        score=torch.ones(n),
        depth=torch.tensor(depth, dtype=torch.float32),
        valid=torch.ones(n, dtype=torch.bool),
        t=t,
        v=v,
    )


def test_track_graph_builds_two_tracks_on_toy_chain():
    keypoints = {
        (0, 0): _kp([[10.0, 10.0], [20.0, 20.0]], [1.0, 1.0], t=0, v=0),
        (1, 0): _kp([[11.0, 10.0], [21.0, 20.0]], [1.0, 1.0], t=1, v=0),
        (2, 0): _kp([[12.0, 10.0], [22.0, 20.0]], [1.0, 1.0], t=2, v=0),
    }

    m01 = MatchBatch(
        idx_a=torch.tensor([0, 1]),
        idx_b=torch.tensor([0, 1]),
        confidence=torch.tensor([0.9, 0.8]),
        pair_type="cross_time_same_view",
    )
    m12 = MatchBatch(
        idx_a=torch.tensor([0, 1]),
        idx_b=torch.tensor([0, 1]),
        confidence=torch.tensor([0.85, 0.75]),
        pair_type="cross_time_same_view",
    )

    pairs = [
        PairMatchRecord(frame_a=(0, 0), frame_b=(1, 0), pair_type="cross_time_same_view", match=m01),
        PairMatchRecord(frame_a=(1, 0), frame_b=(2, 0), pair_type="cross_time_same_view", match=m12),
    ]

    K = torch.tensor([[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]]], dtype=torch.float32)
    T_cw = torch.eye(4, dtype=torch.float32).view(1, 1, 4, 4).repeat(3, 1, 1, 1)

    out = build_tracks_from_matches(
        keypoints=keypoints,
        pair_matches=pairs,
        num_frames=3,
        num_views=1,
        K=K,
        T_cw=T_cw,
        anchor_frame=0,
        mv_weights=MultiViewConsistencyWeights(alpha_world=1.0, alpha_reproj=0.5, alpha_feat=0.0),
        cfg=TrackBuildConfig(
            min_component_obs=2,
            min_track_length=3,
            min_valid_ratio=1.0,
            max_mean_multiview_error=10.0,
            min_multiview_support_ratio=1.0,
        ),
    )

    tracks = out.tracks
    assert tracks.P == 2
    assert tracks.T == 3
    assert bool(torch.all(tracks.valid))
