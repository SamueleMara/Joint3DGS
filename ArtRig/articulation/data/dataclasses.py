from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch


Tensor = torch.Tensor



def _assert_shape(name: str, tensor: Tensor, dims: int) -> None:
    if tensor.ndim != dims:
        raise ValueError(f"{name} must have {dims} dims, got {tensor.shape}")


@dataclass
class RGBDSequence:
    rgb: Tensor
    depth: Tensor
    fg_mask: Tensor
    K: Tensor
    frame_ids: list[int]
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _assert_shape("rgb", self.rgb, 4)
        _assert_shape("depth", self.depth, 4)
        _assert_shape("fg_mask", self.fg_mask, 4)
        if self.K.ndim not in (2, 3):
            raise ValueError(f"K must have shape [3,3] or [T,3,3], got {self.K.shape}")

        t = self.rgb.shape[0]
        if self.depth.shape[0] != t or self.fg_mask.shape[0] != t:
            raise ValueError("rgb/depth/fg_mask must share same T")
        if len(self.frame_ids) != t:
            raise ValueError(f"frame_ids length {len(self.frame_ids)} does not match T={t}")
        if self.rgb.shape[1] != 3:
            raise ValueError("rgb must be [T,3,H,W]")
        if self.depth.shape[1] != 1 or self.fg_mask.shape[1] != 1:
            raise ValueError("depth and fg_mask must be [T,1,H,W]")
        if self.K.shape[-2:] != (3, 3):
            raise ValueError("K must end with [3,3]")
        if self.K.ndim == 3 and self.K.shape[0] != t:
            raise ValueError("Per-frame K must be [T,3,3]")

    @property
    def T(self) -> int:
        return self.rgb.shape[0]

    @property
    def H(self) -> int:
        return self.rgb.shape[-2]

    @property
    def W(self) -> int:
        return self.rgb.shape[-1]

    def to(self, device: torch.device | str) -> "RGBDSequence":
        return RGBDSequence(
            rgb=self.rgb.to(device),
            depth=self.depth.to(device),
            fg_mask=self.fg_mask.to(device),
            K=self.K.to(device),
            frame_ids=list(self.frame_ids),
            meta=dict(self.meta),
        )


@dataclass
class MultiViewRGBDSequence:
    rgb: Tensor
    depth: Tensor
    fg_mask: Tensor
    K: Tensor
    T_cw: Tensor
    frame_ids: list[int]
    view_ids: list[int]
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _assert_shape("rgb", self.rgb, 5)
        _assert_shape("depth", self.depth, 5)
        _assert_shape("fg_mask", self.fg_mask, 5)
        _assert_shape("T_cw", self.T_cw, 4)
        if self.K.ndim not in (3, 4):
            raise ValueError(f"K must be [V,3,3] or [T,V,3,3], got {self.K.shape}")

        t, v = self.rgb.shape[0], self.rgb.shape[1]
        if self.depth.shape[:2] != (t, v) or self.fg_mask.shape[:2] != (t, v):
            raise ValueError("rgb/depth/fg_mask must share [T,V]")
        if self.rgb.shape[2] != 3:
            raise ValueError("rgb must be [T,V,3,H,W]")
        if self.depth.shape[2] != 1 or self.fg_mask.shape[2] != 1:
            raise ValueError("depth and fg_mask must be [T,V,1,H,W]")
        if self.T_cw.shape != (t, v, 4, 4):
            raise ValueError(f"T_cw must be [T,V,4,4], got {self.T_cw.shape}")
        if len(self.frame_ids) != t:
            raise ValueError(f"frame_ids length {len(self.frame_ids)} != T={t}")
        if len(self.view_ids) != v:
            raise ValueError(f"view_ids length {len(self.view_ids)} != V={v}")
        if self.K.shape[-2:] != (3, 3):
            raise ValueError("K must end with [3,3]")
        if self.K.ndim == 3 and self.K.shape[0] != v:
            raise ValueError("Per-view K must be [V,3,3]")
        if self.K.ndim == 4 and self.K.shape[:2] != (t, v):
            raise ValueError("Per-frame-view K must be [T,V,3,3]")

    @property
    def T(self) -> int:
        return self.rgb.shape[0]

    @property
    def V(self) -> int:
        return self.rgb.shape[1]

    @property
    def H(self) -> int:
        return self.rgb.shape[-2]

    @property
    def W(self) -> int:
        return self.rgb.shape[-1]

    def to(self, device: torch.device | str) -> "MultiViewRGBDSequence":
        return MultiViewRGBDSequence(
            rgb=self.rgb.to(device),
            depth=self.depth.to(device),
            fg_mask=self.fg_mask.to(device),
            K=self.K.to(device),
            T_cw=self.T_cw.to(device),
            frame_ids=list(self.frame_ids),
            view_ids=list(self.view_ids),
            meta=dict(self.meta),
        )


@dataclass
class KeypointBatch:
    xy: Tensor
    desc: Tensor
    score: Tensor
    depth: Tensor
    valid: Tensor
    t: int
    v: int
    world: Optional[Tensor] = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _assert_shape("xy", self.xy, 2)
        _assert_shape("desc", self.desc, 2)
        _assert_shape("score", self.score, 1)
        _assert_shape("depth", self.depth, 1)
        _assert_shape("valid", self.valid, 1)
        n = self.xy.shape[0]
        if self.xy.shape[1] != 2:
            raise ValueError("xy must be [N,2]")
        if self.desc.shape[0] != n:
            raise ValueError("desc must be [N,C]")
        if self.score.shape[0] != n or self.depth.shape[0] != n or self.valid.shape[0] != n:
            raise ValueError("score/depth/valid must be [N]")
        if self.world is not None:
            _assert_shape("world", self.world, 2)
            if self.world.shape != (n, 3):
                raise ValueError("world must be [N,3]")


@dataclass
class MatchBatch:
    idx_a: Tensor
    idx_b: Tensor
    confidence: Tensor
    pair_type: str
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _assert_shape("idx_a", self.idx_a, 1)
        _assert_shape("idx_b", self.idx_b, 1)
        _assert_shape("confidence", self.confidence, 1)
        if self.idx_a.shape[0] != self.idx_b.shape[0] or self.idx_a.shape[0] != self.confidence.shape[0]:
            raise ValueError("idx_a/idx_b/confidence must share same M")
        if self.pair_type not in {"same_time_multiview", "cross_time_same_view", "cross_time_multiview"}:
            raise ValueError(f"Unsupported pair_type: {self.pair_type}")


@dataclass
class TrackBatch:
    xy: Tensor
    xyz: Tensor
    valid: Tensor
    anchor_frame: int
    point_ids: Tensor
    feature: Tensor
    confidence: Optional[Tensor] = None
    obs_count: Optional[Tensor] = None
    multiview_error: Optional[Tensor] = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _assert_shape("xy", self.xy, 3)
        _assert_shape("xyz", self.xyz, 3)
        _assert_shape("valid", self.valid, 2)
        _assert_shape("point_ids", self.point_ids, 1)
        _assert_shape("feature", self.feature, 2)

        p, t, c = self.xy.shape
        if c != 2:
            raise ValueError("xy must be [P,T,2]")
        if self.xyz.shape != (p, t, 3):
            raise ValueError("xyz must be [P,T,3] and aligned with xy")
        if self.valid.shape != (p, t):
            raise ValueError("valid must be [P,T] and aligned with xy")
        if self.point_ids.shape[0] != p:
            raise ValueError("point_ids must be [P]")
        if self.feature.shape[0] != p:
            raise ValueError("feature must be [P,C]")
        if self.confidence is not None and self.confidence.shape != (p,):
            raise ValueError("confidence must be [P]")
        if self.obs_count is not None and self.obs_count.shape != (p, t):
            raise ValueError("obs_count must be [P,T]")
        if self.multiview_error is not None and self.multiview_error.shape != (p, t):
            raise ValueError("multiview_error must be [P,T]")
        if not (0 <= self.anchor_frame < t):
            raise ValueError(f"anchor_frame must be in [0,{t - 1}], got {self.anchor_frame}")

    @property
    def P(self) -> int:
        return self.xy.shape[0]

    @property
    def T(self) -> int:
        return self.xy.shape[1]

    def to(self, device: torch.device | str) -> "TrackBatch":
        return TrackBatch(
            xy=self.xy.to(device),
            xyz=self.xyz.to(device),
            valid=self.valid.to(device),
            anchor_frame=self.anchor_frame,
            point_ids=self.point_ids.to(device),
            feature=self.feature.to(device),
            confidence=None if self.confidence is None else self.confidence.to(device),
            obs_count=None if self.obs_count is None else self.obs_count.to(device),
            multiview_error=None if self.multiview_error is None else self.multiview_error.to(device),
            meta=dict(self.meta),
        )


@dataclass
class FeatureGraph:
    nn_idx: Tensor
    nn_weight: Tensor

    def __post_init__(self) -> None:
        _assert_shape("nn_idx", self.nn_idx, 2)
        _assert_shape("nn_weight", self.nn_weight, 2)
        if self.nn_idx.shape != self.nn_weight.shape:
            raise ValueError("nn_idx and nn_weight must have same shape [P,K]")


@dataclass
class SegmentationResult:
    point_logits: Tensor
    point_probs: Tensor
    point_labels: Tensor
    masks_per_frame: Tensor
    transforms_part0: Tensor
    transforms_part1: Tensor
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class RelativeMotionResult:
    reference_part: int
    moving_part: int
    canonical_points: Tensor
    moving_points_rel: Tensor
    valid: Tensor
    weights: Tensor
    ref_transform_inv: Tensor
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class JointCandidateResult:
    model_name: str
    loss: float
    axis_dir: Tensor
    axis_point: Optional[Tensor]
    pitch: Optional[Tensor]
    state: Tensor
    pred_points: Tensor
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class JointResult:
    best_model: str
    axis_dir: Tensor
    axis_point: Optional[Tensor]
    pitch: Optional[Tensor]
    state: Tensor
    candidates: list[JointCandidateResult]
    diagnostics: dict[str, Any] = field(default_factory=dict)
