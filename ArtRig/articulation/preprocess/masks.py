from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_foreground_masks_from_dir(
    mask_dir: str | Path,
    expected_frames: int | None = None,
    image_size: tuple[int, int] | None = None,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Load binary foreground masks from image directory as [T,1,H,W]."""
    mask_dir = Path(mask_dir)
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    mask_paths = sorted([p for p in mask_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS])
    if not mask_paths:
        raise ValueError(f"No mask images found in: {mask_dir}")
    if expected_frames is not None and len(mask_paths) != int(expected_frames):
        raise ValueError(f"Mask count {len(mask_paths)} != expected_frames {expected_frames}")

    h_target: int | None = None
    w_target: int | None = None
    if image_size is not None:
        h_target, w_target = int(image_size[0]), int(image_size[1])

    out: list[np.ndarray] = []
    for p in mask_paths:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f"Could not read mask image: {p}")
        if h_target is not None and w_target is not None and (m.shape[0] != h_target or m.shape[1] != w_target):
            m = cv2.resize(m, (w_target, h_target), interpolation=cv2.INTER_NEAREST)

        m = m.astype(np.float32)
        if m.max() > 1.0:
            m = m / 255.0
        m = (m >= float(threshold)).astype(np.float32)
        out.append(m)

    stacked = np.stack(out, axis=0)[:, None, :, :]
    return torch.from_numpy(stacked)



def rasterize_point_labels(
    xy: torch.Tensor,
    labels: torch.Tensor,
    image_size: tuple[int, int],
) -> torch.Tensor:
    """Rasterize tracked point labels into sparse masks.

    Args:
        xy: [P,T,2]
        labels: [P] in {0,1}
        image_size: (H, W)

    Returns:
        masks: [T,2,H,W] sparse point masks
    """
    p, t, _ = xy.shape
    h, w = image_size
    masks = torch.zeros((t, 2, h, w), dtype=torch.float32, device=xy.device)

    u = xy[..., 0].round().long()
    v = xy[..., 1].round().long()

    for ti in range(t):
        in_bounds = (u[:, ti] >= 0) & (u[:, ti] < w) & (v[:, ti] >= 0) & (v[:, ti] < h)
        idx = torch.where(in_bounds)[0]
        if idx.numel() == 0:
            continue
        uu = u[idx, ti]
        vv = v[idx, ti]
        ll = labels[idx].long().clamp(0, 1)
        masks[ti, ll, vv, uu] = 1.0
    return masks
