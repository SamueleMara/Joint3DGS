from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch

from articulation.data.dataclasses import RGBDSequence


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _sorted_images(dir_path: Path) -> list[Path]:
    paths = [p for p in dir_path.iterdir() if p.suffix.lower() in _IMAGE_EXTS]
    return sorted(paths)


def _load_intrinsics(path: Path) -> torch.Tensor:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        k = np.load(path)
    elif suffix == ".npz":
        data = np.load(path)
        if "K" in data:
            k = data["K"]
        else:
            first = list(data.keys())[0]
            k = data[first]
    elif suffix == ".json":
        payload = json.loads(path.read_text())
        k = np.array(payload["K"], dtype=np.float32)
    else:
        k = np.loadtxt(path, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    if k.shape[-2:] != (3, 3):
        raise ValueError(f"Expected K with trailing [3,3], got {k.shape}")
    return torch.from_numpy(k)


def _read_rgb(path: Path) -> torch.Tensor:
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read RGB image: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    return tensor


def _read_depth(path: Path, depth_scale: float) -> torch.Tensor:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Could not read depth image: {path}")
    depth = depth.astype(np.float32) * depth_scale
    return torch.from_numpy(depth).unsqueeze(0)


def _read_mask(path: Path) -> torch.Tensor:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask image: {path}")
    mask = (mask > 0).astype(np.float32)
    return torch.from_numpy(mask).unsqueeze(0)


def load_rgbd_sequence_from_folders(
    rgb_dir: str | Path,
    depth_dir: str | Path,
    intrinsics: str | Path,
    mask_dir: Optional[str | Path] = None,
    frame_ids: Optional[list[int]] = None,
    depth_scale: float = 1.0,
    meta: Optional[dict[str, Any]] = None,
) -> RGBDSequence:
    rgb_dir = Path(rgb_dir)
    depth_dir = Path(depth_dir)
    k_path = Path(intrinsics)

    rgb_files = _sorted_images(rgb_dir)
    depth_files = _sorted_images(depth_dir)

    if len(rgb_files) == 0 or len(depth_files) == 0:
        raise ValueError("RGB/depth folders must contain images")
    if len(rgb_files) != len(depth_files):
        raise ValueError("RGB and depth frame counts differ")

    if mask_dir is not None:
        mask_files = _sorted_images(Path(mask_dir))
        if len(mask_files) != len(rgb_files):
            raise ValueError("Mask frame count must match RGB frame count")
    else:
        mask_files = None

    rgb = torch.stack([_read_rgb(p) for p in rgb_files], dim=0)
    depth = torch.stack([_read_depth(p, depth_scale) for p in depth_files], dim=0)

    if mask_files is not None:
        fg_mask = torch.stack([_read_mask(p) for p in mask_files], dim=0)
    else:
        fg_mask = (depth > 0).float()

    k = _load_intrinsics(k_path)
    if frame_ids is None:
        frame_ids = list(range(rgb.shape[0]))

    sequence_meta: dict[str, Any] = {
        "rgb_dir": str(rgb_dir),
        "depth_dir": str(depth_dir),
        "intrinsics": str(k_path),
        "mask_dir": None if mask_dir is None else str(mask_dir),
    }
    if meta:
        sequence_meta.update(meta)

    return RGBDSequence(
        rgb=rgb,
        depth=depth,
        fg_mask=fg_mask,
        K=k,
        frame_ids=frame_ids,
        meta=sequence_meta,
    )


def load_rgb_sequence_from_video(
    video_path: str | Path,
    intrinsics: str | Path,
    mask_dir: Optional[str | Path] = None,
    frame_ids: Optional[list[int]] = None,
    meta: Optional[dict[str, Any]] = None,
) -> RGBDSequence:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError("No frames read from video")

    rgb = torch.from_numpy(np.stack(frames, axis=0)).permute(0, 3, 1, 2).float() / 255.0

    # placeholder depth (to be filled later)
    depth = torch.zeros((rgb.shape[0], 1, rgb.shape[2], rgb.shape[3]), dtype=rgb.dtype)

    if mask_dir is not None:
        mask_files = _sorted_images(Path(mask_dir))
        if len(mask_files) != len(frames):
            raise ValueError("Mask frame count must match RGB frame count")
        fg_mask = torch.stack([_read_mask(p) for p in mask_files], dim=0)
    else:
        fg_mask = torch.ones_like(depth)

    k = _load_intrinsics(Path(intrinsics))
    if frame_ids is None:
        frame_ids = list(range(rgb.shape[0]))

    sequence_meta: dict[str, Any] = {
        "video_path": str(video_path),
        "intrinsics": str(intrinsics),
        "mask_dir": None if mask_dir is None else str(mask_dir),
    }
    if meta:
        sequence_meta.update(meta)

    return RGBDSequence(
        rgb=rgb,
        depth=depth,
        fg_mask=fg_mask,
        K=k,
        frame_ids=frame_ids,
        meta=sequence_meta,
    )


def load_rgbd_sequence_npz(path: str | Path) -> RGBDSequence:
    payload = np.load(path, allow_pickle=True)

    if "rgb" in payload:
        rgb_np = payload["rgb"]
    elif "image" in payload:
        # DA3 export format uses key "image" and HWC layout.
        rgb_np = payload["image"]
    else:
        raise ValueError("RGB missing from npz (expected key 'rgb' or 'image').")

    if rgb_np.ndim != 4:
        raise ValueError(f"RGB tensor must be 4D, got shape {rgb_np.shape}")
    if rgb_np.shape[1] == 3:
        rgb = torch.from_numpy(rgb_np).float()
    elif rgb_np.shape[-1] == 3:
        rgb = torch.from_numpy(np.transpose(rgb_np, (0, 3, 1, 2))).float()
    else:
        raise ValueError(f"RGB tensor must be [T,3,H,W] or [T,H,W,3], got {rgb_np.shape}")
    if rgb.max().item() > 1.0:
        rgb = rgb / 255.0

    if "depth" in payload:
        depth_np = payload["depth"]
        if depth_np.ndim == 3:
            depth_np = depth_np[:, None, :, :]
        elif depth_np.ndim == 4 and depth_np.shape[-1] == 1:
            depth_np = np.transpose(depth_np, (0, 3, 1, 2))
        elif depth_np.ndim != 4:
            raise ValueError(f"Depth tensor must be [T,H,W] or [T,1,H,W], got {depth_np.shape}")
        depth = torch.from_numpy(depth_np).float()
    else:
        depth = torch.zeros((rgb.shape[0], 1, rgb.shape[2], rgb.shape[3]), dtype=rgb.dtype)

    if "fg_mask" in payload:
        fg_np = payload["fg_mask"]
        if fg_np.ndim == 3:
            fg_np = fg_np[:, None, :, :]
        elif fg_np.ndim == 4 and fg_np.shape[-1] == 1:
            fg_np = np.transpose(fg_np, (0, 3, 1, 2))
        elif fg_np.ndim != 4:
            raise ValueError(f"fg_mask tensor must be [T,H,W] or [T,1,H,W], got {fg_np.shape}")
        fg_mask = torch.from_numpy(fg_np).float()
    else:
        # Default to valid-depth mask for compatibility with DA3 exports.
        fg_mask = (depth > 0).float()

    if "K" in payload:
        k = torch.from_numpy(payload["K"]).float()
    elif "intrinsics" in payload:
        k = torch.from_numpy(payload["intrinsics"]).float()
    else:
        raise ValueError("K intrinsics missing from npz (expected key 'K' or 'intrinsics').")

    frame_ids = payload["frame_ids"].tolist() if "frame_ids" in payload else list(range(rgb.shape[0]))
    meta = payload["meta"].item() if "meta" in payload else {}

    return RGBDSequence(rgb=rgb, depth=depth, fg_mask=fg_mask, K=k, frame_ids=frame_ids, meta=meta)


def save_rgbd_sequence_npz(sequence: RGBDSequence, path: str | Path) -> None:
    np.savez_compressed(
        path,
        rgb=sequence.rgb.detach().cpu().numpy(),
        depth=sequence.depth.detach().cpu().numpy(),
        fg_mask=sequence.fg_mask.detach().cpu().numpy(),
        K=sequence.K.detach().cpu().numpy(),
        frame_ids=np.array(sequence.frame_ids, dtype=np.int64),
        meta=np.array(sequence.meta, dtype=object),
    )
