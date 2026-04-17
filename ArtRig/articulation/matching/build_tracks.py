from __future__ import annotations

from dataclasses import dataclass

import torch

from articulation.data.dataclasses import KeypointBatch, MatchBatch, TrackBatch
from articulation.matching.multiview_consistency import (
    MultiViewConsistencyWeights,
    multiview_consistency_score,
)
from articulation.matching.track_graph import (
    ObservationNode,
    build_node_index,
    connected_components,
    match_to_edges,
)
from articulation.preprocess.lifting import lift_pixel_to_world


@dataclass
class PairMatchRecord:
    frame_a: tuple[int, int]
    frame_b: tuple[int, int]
    pair_type: str
    match: MatchBatch


@dataclass
class TrackBuildConfig:
    min_component_obs: int = 2
    min_track_length: int = 4
    min_valid_ratio: float = 0.5
    max_mean_multiview_error: float = 0.02
    min_multiview_support_ratio: float = 0.4
    min_confidence: float | None = None


@dataclass
class BuiltTracks:
    tracks: TrackBatch
    diagnostics: dict[str, float | int]


def _get_k_tv(K: torch.Tensor, t: int, v: int) -> torch.Tensor:
    if K.ndim == 3:
        return K[v]
    if K.ndim == 4:
        return K[t, v]
    raise ValueError(f"K must be [V,3,3] or [T,V,3,3], got {K.shape}")


def _observation_world(
    kp: KeypointBatch,
    k_idx: int,
    K_tv: torch.Tensor,
    T_cw_tv: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    if k_idx < 0 or k_idx >= kp.xy.shape[0]:
        raise IndexError("k_idx out of range")

    if kp.world is not None:
        xw = kp.world[k_idx]
        ok = bool(kp.valid[k_idx].item()) and bool(torch.isfinite(xw).all().item())
        return xw, ok

    xy = kp.xy[k_idx : k_idx + 1]
    depth = kp.depth[k_idx : k_idx + 1]
    world, valid = lift_pixel_to_world(xy, depth, K_tv, T_cw_tv)
    return world[0], bool(valid[0].item())


def _fuse_world(points: torch.Tensor) -> torch.Tensor:
    if points.shape[0] == 1:
        return points[0]
    return torch.median(points, dim=0).values


def _collect_comp_edge_conf(
    comp_nodes: list[ObservationNode],
    edges: list[tuple[ObservationNode, ObservationNode, float]],
) -> float:
    node_set = set(comp_nodes)
    vals: list[float] = []
    for a, b, c in edges:
        if a in node_set and b in node_set:
            vals.append(float(c))
    if not vals:
        return 1.0
    return float(sum(vals) / len(vals))


def build_tracks_from_matches(
    keypoints: dict[tuple[int, int], KeypointBatch],
    pair_matches: list[PairMatchRecord],
    num_frames: int,
    num_views: int,
    K: torch.Tensor,
    T_cw: torch.Tensor,
    anchor_frame: int,
    mv_weights: MultiViewConsistencyWeights,
    cfg: TrackBuildConfig,
) -> BuiltTracks:
    if num_frames <= 0:
        raise ValueError("num_frames must be > 0")
    if num_views <= 0:
        raise ValueError("num_views must be > 0")

    counts: dict[tuple[int, int], int] = {}
    for (t, v), kp in keypoints.items():
        counts[(int(t), int(v))] = int(kp.xy.shape[0])

    nodes, index = build_node_index(counts)

    edges: list[tuple[ObservationNode, ObservationNode, float]] = []
    for rec in pair_matches:
        edges.extend(match_to_edges(rec.match, rec.frame_a, rec.frame_b))

    comps = connected_components(nodes, index, edges)

    track_xy: list[torch.Tensor] = []
    track_xyz: list[torch.Tensor] = []
    track_valid: list[torch.Tensor] = []
    track_feat: list[torch.Tensor] = []
    track_conf: list[float] = []
    track_obs_count: list[torch.Tensor] = []
    track_mv_err: list[torch.Tensor] = []

    feature_dim = None
    for kp in keypoints.values():
        if kp.desc.ndim == 2 and kp.desc.shape[0] > 0:
            feature_dim = int(kp.desc.shape[1])
            break
    if feature_dim is None:
        feature_dim = 1

    for comp in comps:
        if len(comp) < int(cfg.min_component_obs):
            continue

        xy_t = torch.zeros((num_frames, 2), dtype=torch.float32)
        xyz_t = torch.zeros((num_frames, 3), dtype=torch.float32)
        valid_t = torch.zeros((num_frames,), dtype=torch.bool)
        obs_count_t = torch.zeros((num_frames,), dtype=torch.float32)
        mv_err_t = torch.full((num_frames,), float("inf"), dtype=torch.float32)

        desc_by_t: dict[int, list[torch.Tensor]] = {}

        for t in range(num_frames):
            obs_xy: list[torch.Tensor] = []
            obs_world: list[torch.Tensor] = []
            obs_desc: list[torch.Tensor] = []
            obs_K: list[torch.Tensor] = []
            obs_Tcw: list[torch.Tensor] = []

            for node in comp:
                if node.t != t:
                    continue
                kp = keypoints.get((node.t, node.v))
                if kp is None:
                    continue
                if node.k < 0 or node.k >= kp.xy.shape[0]:
                    continue

                K_tv = _get_k_tv(K, t=node.t, v=node.v).float()
                T_cw_tv = T_cw[node.t, node.v].float()
                xw, ok = _observation_world(kp, node.k, K_tv=K_tv, T_cw_tv=T_cw_tv)
                if not ok:
                    continue

                obs_xy.append(kp.xy[node.k].float())
                obs_world.append(xw.float())
                obs_desc.append(kp.desc[node.k].float())
                obs_K.append(K_tv)
                obs_Tcw.append(T_cw_tv)
                desc_by_t.setdefault(t, []).append(kp.desc[node.k].float())

            if not obs_world:
                continue

            obs_xy_t = torch.stack(obs_xy, dim=0)
            obs_world_t = torch.stack(obs_world, dim=0)
            obs_desc_t = torch.stack(obs_desc, dim=0)
            obs_K_t = torch.stack(obs_K, dim=0)
            obs_Tcw_t = torch.stack(obs_Tcw, dim=0)

            xy_t[t] = torch.median(obs_xy_t, dim=0).values
            xyz_t[t] = _fuse_world(obs_world_t)
            valid_t[t] = True
            obs_count_t[t] = float(obs_world_t.shape[0])

            mv_res = multiview_consistency_score(
                world_points=obs_world_t,
                observed_xy=obs_xy_t,
                desc=obs_desc_t,
                K_all=obs_K_t,
                T_cw_all=obs_Tcw_t,
                weights=mv_weights,
            )
            mv_err_t[t] = float(mv_res.score)

        if valid_t.sum().item() == 0:
            continue

        # Anchor feature: prefer anchor-frame observations, fallback to first valid time.
        anchor_desc = desc_by_t.get(anchor_frame, None)
        if not anchor_desc:
            t_valid = int(torch.where(valid_t)[0][0].item())
            anchor_desc = desc_by_t.get(t_valid, [])
        if anchor_desc:
            feat = torch.stack(anchor_desc, dim=0).mean(dim=0)
        else:
            feat = torch.zeros((feature_dim,), dtype=torch.float32)

        conf = _collect_comp_edge_conf(comp, edges)

        track_xy.append(xy_t)
        track_xyz.append(xyz_t)
        track_valid.append(valid_t)
        track_feat.append(feat)
        track_conf.append(conf)
        track_obs_count.append(obs_count_t)
        track_mv_err.append(mv_err_t)

    if not track_xyz:
        empty = TrackBatch(
            xy=torch.zeros((0, num_frames, 2), dtype=torch.float32),
            xyz=torch.zeros((0, num_frames, 3), dtype=torch.float32),
            valid=torch.zeros((0, num_frames), dtype=torch.bool),
            anchor_frame=int(anchor_frame),
            point_ids=torch.zeros((0,), dtype=torch.long),
            feature=torch.zeros((0, feature_dim), dtype=torch.float32),
            confidence=torch.zeros((0,), dtype=torch.float32),
            obs_count=torch.zeros((0, num_frames), dtype=torch.float32),
            multiview_error=torch.zeros((0, num_frames), dtype=torch.float32),
            meta={"num_components": int(len(comps)), "num_tracks_raw": 0},
        )
        return BuiltTracks(tracks=empty, diagnostics={"num_components": int(len(comps)), "num_tracks_raw": 0})

    xy = torch.stack(track_xy, dim=0)
    xyz = torch.stack(track_xyz, dim=0)
    valid = torch.stack(track_valid, dim=0)
    feat = torch.stack(track_feat, dim=0)
    conf = torch.tensor(track_conf, dtype=torch.float32)
    obs_count = torch.stack(track_obs_count, dim=0)
    mv_err = torch.stack(track_mv_err, dim=0)

    tracks = TrackBatch(
        xy=xy,
        xyz=xyz,
        valid=valid,
        anchor_frame=int(anchor_frame),
        point_ids=torch.arange(xy.shape[0], dtype=torch.long),
        feature=feat,
        confidence=conf,
        obs_count=obs_count,
        multiview_error=mv_err,
        meta={"num_components": int(len(comps)), "num_tracks_raw": int(xy.shape[0])},
    )

    filtered = filter_built_tracks(tracks, cfg)
    diagnostics = {
        "num_components": int(len(comps)),
        "num_tracks_raw": int(tracks.P),
        "num_tracks_filtered": int(filtered.P),
    }
    return BuiltTracks(tracks=filtered, diagnostics=diagnostics)


def filter_built_tracks(tracks: TrackBatch, cfg: TrackBuildConfig) -> TrackBatch:
    if tracks.P == 0:
        return tracks

    keep = torch.ones((tracks.P,), dtype=torch.bool)

    valid_ratio = tracks.valid.float().mean(dim=1)
    keep = keep & (valid_ratio >= float(cfg.min_valid_ratio))

    valid_len = tracks.valid.sum(dim=1)
    keep = keep & (valid_len >= int(cfg.min_track_length))

    if tracks.multiview_error is not None:
        finite_mv = torch.isfinite(tracks.multiview_error)
        mv_mask = tracks.valid & finite_mv
        mv_sum = torch.where(mv_mask, tracks.multiview_error, torch.zeros_like(tracks.multiview_error)).sum(dim=1)
        mv_den = mv_mask.float().sum(dim=1).clamp_min(1.0)
        mean_mv = mv_sum / mv_den
        support_ratio = mv_mask.float().mean(dim=1)

        keep = keep & (mean_mv <= float(cfg.max_mean_multiview_error))
        keep = keep & (support_ratio >= float(cfg.min_multiview_support_ratio))

    if cfg.min_confidence is not None and tracks.confidence is not None:
        keep = keep & (tracks.confidence >= float(cfg.min_confidence))

    if not bool(keep.any()):
        # Keep at least the best-confidence track to avoid empty downstream failures.
        if tracks.confidence is not None and tracks.confidence.numel() > 0:
            best = int(torch.argmax(tracks.confidence).item())
        else:
            best = 0
        keep = torch.zeros_like(keep)
        keep[best] = True

    conf = tracks.confidence[keep] if tracks.confidence is not None else None
    obs = tracks.obs_count[keep] if tracks.obs_count is not None else None
    mve = tracks.multiview_error[keep] if tracks.multiview_error is not None else None

    return TrackBatch(
        xy=tracks.xy[keep],
        xyz=tracks.xyz[keep],
        valid=tracks.valid[keep],
        anchor_frame=tracks.anchor_frame,
        point_ids=torch.arange(int(keep.sum().item()), dtype=torch.long),
        feature=tracks.feature[keep],
        confidence=conf,
        obs_count=obs,
        multiview_error=mve,
        meta=dict(tracks.meta),
    )
