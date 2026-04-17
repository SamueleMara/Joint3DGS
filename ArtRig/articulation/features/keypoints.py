from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np
import torch

from articulation.data.dataclasses import KeypointBatch
from articulation.features.dino_wrapper import DinoFeatureExtractor


class BaseKeypointDetector(Protocol):
    def detect(self, image: torch.Tensor, fg_mask: torch.Tensor, num_keypoints: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Detect keypoints on one frame.

        Args:
            image: [3,H,W] float in [0,1]
            fg_mask: [1,H,W] or [H,W], binary-ish
            num_keypoints: max number of returned points

        Returns:
            xy: [N,2] pixel coordinates (u,v)
            score: [N] detector confidence
        """


@dataclass
class ShiTomasiKeypointDetector:
    quality_level: float = 0.01
    min_distance: float = 4.0
    block_size: int = 5
    use_harris: bool = False
    k: float = 0.04

    def detect(self, image: torch.Tensor, fg_mask: torch.Tensor, num_keypoints: int) -> tuple[torch.Tensor, torch.Tensor]:
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("image must be [3,H,W]")
        if fg_mask.ndim == 3:
            if fg_mask.shape[0] != 1:
                raise ValueError("fg_mask must be [1,H,W] or [H,W]")
            m = fg_mask[0]
        elif fg_mask.ndim == 2:
            m = fg_mask
        else:
            raise ValueError("fg_mask must be [1,H,W] or [H,W]")

        h, w = int(image.shape[-2]), int(image.shape[-1])
        if m.shape != (h, w):
            raise ValueError("fg_mask shape must match image H,W")

        img_u8 = (image.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
        mask_u8 = (m.detach().cpu().numpy() > 0.5).astype(np.uint8) * 255

        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=int(max(num_keypoints, 1)),
            qualityLevel=float(max(self.quality_level, 1e-6)),
            minDistance=float(max(self.min_distance, 1.0)),
            mask=mask_u8,
            blockSize=int(max(self.block_size, 2)),
            useHarrisDetector=bool(self.use_harris),
            k=float(self.k),
        )

        if corners is None or len(corners) == 0:
            return (
                torch.zeros((0, 2), dtype=torch.float32, device=image.device),
                torch.zeros((0,), dtype=torch.float32, device=image.device),
            )

        pts = torch.from_numpy(corners[:, 0, :]).float().to(image.device)

        # Lightweight score proxy: local gradient magnitude.
        gray_t = torch.from_numpy(gray).float().to(image.device) / 255.0
        gx = torch.zeros_like(gray_t)
        gy = torch.zeros_like(gray_t)
        gx[:, 1:-1] = 0.5 * (gray_t[:, 2:] - gray_t[:, :-2])
        gy[1:-1, :] = 0.5 * (gray_t[2:, :] - gray_t[:-2, :])
        grad = torch.sqrt(gx * gx + gy * gy)

        u = pts[:, 0].round().long().clamp(0, w - 1)
        v = pts[:, 1].round().long().clamp(0, h - 1)
        score = grad[v, u]

        return pts, score


@dataclass
class SIFTKeypointDetector:
    contrast_threshold: float = 0.04
    edge_threshold: float = 10.0

    def __post_init__(self) -> None:
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("OpenCV SIFT is unavailable in this build")
        self._sift = cv2.SIFT_create(
            contrastThreshold=float(self.contrast_threshold),
            edgeThreshold=float(self.edge_threshold),
        )

    def detect(self, image: torch.Tensor, fg_mask: torch.Tensor, num_keypoints: int) -> tuple[torch.Tensor, torch.Tensor]:
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("image must be [3,H,W]")
        if fg_mask.ndim == 3:
            m = fg_mask[0]
        elif fg_mask.ndim == 2:
            m = fg_mask
        else:
            raise ValueError("fg_mask must be [1,H,W] or [H,W]")

        img_u8 = (image.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
        mask_u8 = (m.detach().cpu().numpy() > 0.5).astype(np.uint8) * 255

        kps = self._sift.detect(gray, mask=mask_u8)
        if not kps:
            return (
                torch.zeros((0, 2), dtype=torch.float32, device=image.device),
                torch.zeros((0,), dtype=torch.float32, device=image.device),
            )

        kps = sorted(kps, key=lambda kp: float(kp.response), reverse=True)[: int(max(num_keypoints, 1))]
        xy = torch.tensor([[kp.pt[0], kp.pt[1]] for kp in kps], dtype=torch.float32, device=image.device)
        score = torch.tensor([float(max(kp.response, 0.0)) for kp in kps], dtype=torch.float32, device=image.device)
        return xy, score


def build_keypoint_detector(name: str, **kwargs) -> BaseKeypointDetector:
    key = str(name).strip().lower()
    if key in {"shi_tomasi", "good_features", "gftt"}:
        return ShiTomasiKeypointDetector(
            quality_level=float(kwargs.get("quality_level", 0.01)),
            min_distance=float(kwargs.get("min_distance", 4.0)),
            block_size=int(kwargs.get("block_size", 5)),
            use_harris=bool(kwargs.get("use_harris", False)),
            k=float(kwargs.get("k", 0.04)),
        )
    if key == "sift":
        return SIFTKeypointDetector(
            contrast_threshold=float(kwargs.get("contrast_threshold", 0.04)),
            edge_threshold=float(kwargs.get("edge_threshold", 10.0)),
        )
    raise ValueError(f"Unsupported keypoint detector: {name}")


def erode_foreground_mask(mask: torch.Tensor, erode_px: int) -> torch.Tensor:
    if mask.ndim == 3:
        if mask.shape[0] != 1:
            raise ValueError("mask must be [1,H,W] or [H,W]")
        m = mask[0]
    elif mask.ndim == 2:
        m = mask
    else:
        raise ValueError("mask must be [1,H,W] or [H,W]")

    if erode_px <= 0:
        return (m > 0.5).float().unsqueeze(0)

    m_u8 = ((m > 0.5).detach().cpu().numpy().astype(np.uint8)) * 255
    k = int(2 * erode_px + 1)
    kernel = np.ones((k, k), dtype=np.uint8)
    eroded = cv2.erode(m_u8, kernel, iterations=1)
    out = torch.from_numpy((eroded > 0).astype(np.float32)).to(mask.device)
    return out.unsqueeze(0)


def _sample_depth_nearest(depth: torch.Tensor, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if depth.ndim == 3:
        if depth.shape[0] != 1:
            raise ValueError("depth must be [1,H,W] or [H,W]")
        d = depth[0]
    elif depth.ndim == 2:
        d = depth
    else:
        raise ValueError("depth must be [1,H,W] or [H,W]")

    h, w = int(d.shape[0]), int(d.shape[1])
    u = xy[:, 0]
    v = xy[:, 1]
    inb = (u >= 0) & (u <= (w - 1)) & (v >= 0) & (v <= (h - 1))
    uu = u.round().long().clamp(0, w - 1)
    vv = v.round().long().clamp(0, h - 1)
    z = d[vv, uu]
    valid = inb & torch.isfinite(z) & (z > 0)
    z = torch.where(valid, z, torch.zeros_like(z))
    return z, valid


def extract_keypoint_batch(
    image: torch.Tensor,
    depth: torch.Tensor,
    fg_mask: torch.Tensor,
    extractor: DinoFeatureExtractor,
    detector: BaseKeypointDetector,
    num_keypoints: int,
    t: int,
    v: int,
    fg_erode_px: int = 0,
) -> KeypointBatch:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("image must be [3,H,W]")
    if depth.ndim not in (2, 3):
        raise ValueError("depth must be [H,W] or [1,H,W]")
    if fg_mask.ndim not in (2, 3):
        raise ValueError("fg_mask must be [H,W] or [1,H,W]")

    fg = erode_foreground_mask(fg_mask, erode_px=int(max(fg_erode_px, 0)))
    xy, score = detector.detect(image, fg, int(max(num_keypoints, 1)))

    if xy.shape[0] == 0:
        desc = torch.zeros((0, 1), dtype=torch.float32, device=image.device)
        z = torch.zeros((0,), dtype=torch.float32, device=image.device)
        valid = torch.zeros((0,), dtype=torch.bool, device=image.device)
        return KeypointBatch(
            xy=xy,
            desc=desc,
            score=score,
            depth=z,
            valid=valid,
            t=int(t),
            v=int(v),
            world=None,
            meta={"num_keypoints": 0},
        )

    feat_map = extractor.extract_dense(image)
    desc = extractor.sample_points(feat_map.to(image.device), xy)
    z, valid = _sample_depth_nearest(depth, xy)

    return KeypointBatch(
        xy=xy,
        desc=desc,
        score=score,
        depth=z,
        valid=valid,
        t=int(t),
        v=int(v),
        world=None,
        meta={
            "num_keypoints": int(xy.shape[0]),
            "fg_erode_px": int(max(fg_erode_px, 0)),
        },
    )
