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
class TrackBatch:
    xy: Tensor
    xyz: Tensor
    valid: Tensor
    anchor_frame: int
    point_ids: Tensor
    feature: Tensor
    confidence: Optional[Tensor] = None

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
