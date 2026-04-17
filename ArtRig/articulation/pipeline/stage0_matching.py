from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

from tqdm.auto import tqdm

from articulation.data.dataclasses import KeypointBatch, MatchBatch, MultiViewRGBDSequence, TrackBatch
from articulation.external.dino_backend_adapter import DinoBackendAdapter
from articulation.features.keypoints import build_keypoint_detector, extract_keypoint_batch
from articulation.matching.build_tracks import (
    PairMatchRecord,
    TrackBuildConfig,
    build_tracks_from_matches,
)
from articulation.matching.filters import (
    filter_cross_time_matches,
    filter_same_time_multiview_matches,
)
from articulation.matching.matcher_wrapper import MatcherWrapper
from articulation.matching.multiview_consistency import MultiViewConsistencyWeights


@dataclass
class Stage0MatchingResult:
    tracks: TrackBatch
    keypoints: dict[tuple[int, int], KeypointBatch]
    matches: list[PairMatchRecord]
    diagnostics: dict[str, Any]


def _get_k_tv(K, t: int, v: int):
    if K.ndim == 3:
        return K[v]
    if K.ndim == 4:
        return K[t, v]
    raise ValueError(f"K must be [V,3,3] or [T,V,3,3], got {K.shape}")


def _build_pair_jobs(T: int, V: int, enable_cross_time_multiview: bool) -> list[tuple[tuple[int, int], tuple[int, int], str]]:
    jobs: list[tuple[tuple[int, int], tuple[int, int], str]] = []

    # Same-time multiview pairs.
    for t in range(T):
        for v1, v2 in combinations(range(V), 2):
            jobs.append(((t, v1), (t, v2), "same_time_multiview"))

    # Cross-time same-view pairs.
    for t in range(T - 1):
        for v in range(V):
            jobs.append(((t, v), (t + 1, v), "cross_time_same_view"))

    # Optional cross-time multiview pairs.
    if enable_cross_time_multiview:
        for t in range(T - 1):
            for v1 in range(V):
                for v2 in range(V):
                    if v1 == v2:
                        continue
                    jobs.append(((t, v1), (t + 1, v2), "cross_time_multiview"))

    return jobs


def run_stage0_matching(
    sequence: MultiViewRGBDSequence,
    cfg: dict,
    show_progress: bool = False,
    debug: bool = False,
) -> Stage0MatchingResult:
    feat_cfg = dict(cfg.get("features", {}))
    matcher_cfg = dict(cfg.get("matcher", {}))
    filt_cfg = dict(cfg.get("filtering", {}))
    mv_cfg = dict(cfg.get("multiview", {}))

    extractor = DinoBackendAdapter(
        model_name=str(feat_cfg.get("model_name", "vit_small_patch14_dinov2")),
        device=str(feat_cfg.get("device", "cpu")),
        repo_path=feat_cfg.get("repo_path"),
    ).build()

    detector = build_keypoint_detector(
        str(feat_cfg.get("detector", "shi_tomasi")),
        **dict(feat_cfg.get("detector_kwargs", {})),
    )

    num_kp = int(feat_cfg.get("num_keypoints", 1024))
    fg_erode_px = int(feat_cfg.get("fg_erode_px", 3))

    T = sequence.T
    V = sequence.V

    keypoints: dict[tuple[int, int], KeypointBatch] = {}
    extract_bar = tqdm(total=T * V, desc="Stage0/Features+KP", disable=not show_progress, leave=False)
    for t in range(T):
        for v in range(V):
            kp = extract_keypoint_batch(
                image=sequence.rgb[t, v],
                depth=sequence.depth[t, v],
                fg_mask=sequence.fg_mask[t, v],
                extractor=extractor,
                detector=detector,
                num_keypoints=num_kp,
                t=t,
                v=v,
                fg_erode_px=fg_erode_px,
            )
            keypoints[(t, v)] = kp
            extract_bar.update(1)
    extract_bar.close()

    matcher = MatcherWrapper.from_config(matcher_cfg)

    jobs = _build_pair_jobs(
        T=T,
        V=V,
        enable_cross_time_multiview=bool(mv_cfg.get("enable_cross_time_multiview", False)),
    )

    pair_matches: list[PairMatchRecord] = []
    raw_match_count = 0
    filt_match_count = 0

    match_bar = tqdm(jobs, desc="Stage0/Matching", disable=not show_progress, leave=False)
    for (ta, va), (tb, vb), pair_type in match_bar:
        fa = keypoints[(ta, va)]
        fb = keypoints[(tb, vb)]

        raw = matcher.match(fa, fb, pair_type=pair_type)
        raw_match_count += int(raw.idx_a.shape[0])

        if pair_type == "same_time_multiview":
            filtered = filter_same_time_multiview_matches(
                match=raw,
                frame_a=fa,
                frame_b=fb,
                K_a=_get_k_tv(sequence.K, ta, va),
                K_b=_get_k_tv(sequence.K, tb, vb),
                T_cw_a=sequence.T_cw[ta, va],
                T_cw_b=sequence.T_cw[tb, vb],
                threshold_same_time=float(filt_cfg.get("same_time_max_3d_error", 0.01)),
                fg_mask_a=sequence.fg_mask[ta, va],
                fg_mask_b=sequence.fg_mask[tb, vb],
            )
        else:
            filtered = filter_cross_time_matches(
                match=raw,
                frame_a=fa,
                frame_b=fb,
                min_confidence=float(filt_cfg.get("min_match_confidence", 0.2)),
                max_pixel_jump=float(filt_cfg.get("max_pixel_jump", 80.0)),
                max_depth_jump=filt_cfg.get("max_depth_jump", None),
                require_cycle_consistency=bool(filt_cfg.get("require_cycle_consistency", False)),
            )

        filt_match_count += int(filtered.idx_a.shape[0])
        pair_matches.append(
            PairMatchRecord(
                frame_a=(ta, va),
                frame_b=(tb, vb),
                pair_type=pair_type,
                match=filtered,
            )
        )

        if show_progress:
            match_bar.set_postfix(raw=raw_match_count, kept=filt_match_count)
    match_bar.close()

    built = build_tracks_from_matches(
        keypoints=keypoints,
        pair_matches=pair_matches,
        num_frames=T,
        num_views=V,
        K=sequence.K,
        T_cw=sequence.T_cw,
        anchor_frame=int(cfg.get("anchor_frame", 0)),
        mv_weights=MultiViewConsistencyWeights(
            alpha_world=float(mv_cfg.get("alpha_world", 1.0)),
            alpha_reproj=float(mv_cfg.get("alpha_reproj", 0.5)),
            alpha_feat=float(mv_cfg.get("alpha_feat", 0.2)),
        ),
        cfg=TrackBuildConfig(
            min_component_obs=int(filt_cfg.get("min_component_obs", 2)),
            min_track_length=int(filt_cfg.get("min_track_length", 4)),
            min_valid_ratio=float(filt_cfg.get("min_valid_ratio", 0.5)),
            max_mean_multiview_error=float(mv_cfg.get("max_mean_score", 0.02)),
            min_multiview_support_ratio=float(mv_cfg.get("min_multiview_support_ratio", 0.4)),
            min_confidence=(None if filt_cfg.get("min_track_confidence", None) is None else float(filt_cfg.get("min_track_confidence"))),
        ),
    )

    diagnostics = {
        "num_frames": T,
        "num_views": V,
        "num_keypoints_total": int(sum(kp.xy.shape[0] for kp in keypoints.values())),
        "num_pair_jobs": int(len(jobs)),
        "num_matches_raw": int(raw_match_count),
        "num_matches_kept": int(filt_match_count),
        **built.diagnostics,
    }

    if debug:
        print(
            "[Stage0] "
            f"kp_total={diagnostics['num_keypoints_total']} "
            f"matches={diagnostics['num_matches_raw']}->{diagnostics['num_matches_kept']} "
            f"tracks={diagnostics['num_tracks_filtered']}"
        )

    return Stage0MatchingResult(
        tracks=built.tracks,
        keypoints=keypoints,
        matches=pair_matches,
        diagnostics=diagnostics,
    )
