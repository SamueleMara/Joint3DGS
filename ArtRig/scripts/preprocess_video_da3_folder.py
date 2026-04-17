#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm


def _maybe_add_repo_path(path: str | None) -> None:
    if not path:
        return
    p = str(Path(path).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def _sample_video_frames(video_path: str | Path, target_fps: float, max_frames: int | None = None) -> tuple[list[np.ndarray], list[int], float, int]:
    if target_fps <= 0.0:
        raise ValueError("target fps must be > 0")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(src_fps) or src_fps <= 1e-6:
        src_fps = 30.0

    interval = 1.0 / float(target_fps)
    next_ts = 0.0

    frames: list[np.ndarray] = []
    sampled_indices: list[int] = []
    frame_idx = 0
    while cap.isOpened():
        ok, frame_bgr = cap.read()
        if not ok:
            break
        ts = frame_idx / src_fps
        if ts + 1e-9 >= next_ts:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            sampled_indices.append(frame_idx)
            next_ts += interval
            if max_frames is not None and len(frames) >= int(max_frames):
                frame_idx += 1
                break
        frame_idx += 1

    cap.release()
    if not frames:
        raise ValueError("No frames sampled from input video")
    return frames, sampled_indices, src_fps, frame_idx


def _load_masks_from_dir(mask_dir: str | Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = [p for p in Path(mask_dir).iterdir() if p.suffix.lower() in exts]
    return sorted(paths)


def _select_mask_paths(mask_paths: list[Path], sampled_indices: list[int], total_read_frames: int, sampled_count: int) -> list[Path]:
    if len(mask_paths) == sampled_count:
        return mask_paths
    if len(mask_paths) >= total_read_frames:
        out: list[Path] = []
        for idx in sampled_indices:
            if idx >= len(mask_paths):
                raise ValueError(f"Mask index {idx} out of bounds for {len(mask_paths)} mask files")
            out.append(mask_paths[idx])
        return out
    raise ValueError(
        "Mask count mismatch. Expected either same as sampled frames "
        f"({sampled_count}) or at least total source frames ({total_read_frames}), got {len(mask_paths)}"
    )


def _build_da3_model(repo_path: str | None, model_dir: str | None, model_name: str, device: str):
    _maybe_add_repo_path(repo_path)
    from depth_anything_3.api import DepthAnything3

    if model_dir:
        model = DepthAnything3.from_pretrained(model_dir)
    else:
        model = DepthAnything3(model_name=model_name)

    run_device = device
    if device.startswith("cuda") and not torch.cuda.is_available():
        run_device = "cpu"
    model = model.to(device=torch.device(run_device))
    return model, run_device


def _resize_and_adjust_intrinsics(
    depth: np.ndarray,
    conf: np.ndarray | None,
    intrinsics: np.ndarray | None,
    target_h: int,
    target_w: int,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    n, h, w = depth.shape
    if h == target_h and w == target_w:
        return depth, conf, intrinsics

    depth_out = np.zeros((n, target_h, target_w), dtype=np.float32)
    conf_out = None if conf is None else np.zeros((n, target_h, target_w), dtype=np.float32)
    intr_out = None if intrinsics is None else intrinsics.copy().astype(np.float32)

    sx = target_w / float(w)
    sy = target_h / float(h)
    for i in range(n):
        depth_out[i] = cv2.resize(depth[i], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        if conf_out is not None and conf is not None:
            conf_out[i] = cv2.resize(conf[i], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        if intr_out is not None:
            intr_out[i, 0, 0] *= sx
            intr_out[i, 0, 2] *= sx
            intr_out[i, 1, 1] *= sy
            intr_out[i, 1, 2] *= sy

    return depth_out, conf_out, intr_out


def main() -> None:
    default_da3_repo = str(Path(__file__).resolve().parents[1] / "submodules" / "depth_anything_3")
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--repo-path", default=default_da3_repo)
    parser.add_argument("--model-dir", default="", help="HF model id or local model directory")
    parser.add_argument("--model-name", default="da3-small")
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--process-res-method", default="upper_bound_resize")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--depth-scale", type=float, default=0.001, help="Depth meters = depth_png * depth_scale")
    parser.add_argument("--max-depth-m", type=float, default=65.0)
    parser.add_argument("--mask-dir", default=None, help="Optional precomputed foreground-mask folder")
    parser.add_argument("--conf-mask-threshold", type=float, default=0.0, help="Use DA3 conf threshold when mask-dir is not provided")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    rgb_dir = out_dir / "images"
    depth_dir = out_dir / "depth"
    mask_out_dir = out_dir / "fg_mask"
    conf_dir = out_dir / "confidence"
    for d in (rgb_dir, depth_dir, mask_out_dir, conf_dir):
        d.mkdir(parents=True, exist_ok=True)

    frames_rgb, sampled_indices, src_fps, total_read_frames = _sample_video_frames(
        args.video,
        target_fps=float(args.fps),
        max_frames=args.max_frames,
    )

    model, run_device = _build_da3_model(
        repo_path=args.repo_path,
        model_dir=args.model_dir if args.model_dir else None,
        model_name=args.model_name,
        device=args.device,
    )

    pred = model.inference(
        frames_rgb,
        process_res=int(args.process_res),
        process_res_method=str(args.process_res_method),
    )

    depth = np.asarray(pred.depth, dtype=np.float32)
    conf = np.asarray(pred.conf, dtype=np.float32) if getattr(pred, "conf", None) is not None else None
    intrinsics = np.asarray(pred.intrinsics, dtype=np.float32) if getattr(pred, "intrinsics", None) is not None else None
    extrinsics = np.asarray(pred.extrinsics, dtype=np.float32) if getattr(pred, "extrinsics", None) is not None else None

    n = len(frames_rgb)
    if depth.ndim != 3 or depth.shape[0] != n:
        raise ValueError(f"Unexpected DA3 depth shape: {depth.shape}")
    if intrinsics is None:
        raise RuntimeError("DA3 did not return intrinsics")
    if extrinsics is None:
        raise RuntimeError("DA3 did not return extrinsics")

    if intrinsics.ndim == 2:
        intrinsics = np.repeat(intrinsics[None, ...], n, axis=0)
    if intrinsics.shape != (n, 3, 3):
        raise ValueError(f"Unexpected intrinsics shape: {intrinsics.shape}")

    if extrinsics.ndim == 3 and extrinsics.shape[1:] == (4, 4):
        extrinsics = extrinsics[:, :3, :]
    if extrinsics.shape != (n, 3, 4):
        raise ValueError(f"Unexpected extrinsics shape: {extrinsics.shape}")

    target_h, target_w = int(frames_rgb[0].shape[0]), int(frames_rgb[0].shape[1])
    depth, conf, intrinsics = _resize_and_adjust_intrinsics(
        depth=depth,
        conf=conf,
        intrinsics=intrinsics,
        target_h=target_h,
        target_w=target_w,
    )
    assert intrinsics is not None

    selected_mask_paths: list[Path] | None = None
    if args.mask_dir is not None:
        mask_paths = _load_masks_from_dir(args.mask_dir)
        if not mask_paths:
            raise ValueError(f"No masks found in mask-dir: {args.mask_dir}")
        selected_mask_paths = _select_mask_paths(
            mask_paths=mask_paths,
            sampled_indices=sampled_indices,
            total_read_frames=total_read_frames,
            sampled_count=n,
        )

    for i in tqdm(range(n), desc="Writing dataset", leave=False):
        name = f"{i:06d}.png"

        rgb = frames_rgb[i]
        cv2.imwrite(str(rgb_dir / name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        depth_m = np.clip(depth[i], 0.0, float(args.max_depth_m))
        depth_raw = np.clip(np.round(depth_m / float(args.depth_scale)), 0, 65535).astype(np.uint16)
        cv2.imwrite(str(depth_dir / name), depth_raw)

        if conf is not None:
            conf_u8 = np.clip(conf[i], 0.0, 1.0) * 255.0
            cv2.imwrite(str(conf_dir / name), conf_u8.astype(np.uint8))

        if selected_mask_paths is not None:
            m = cv2.imread(str(selected_mask_paths[i]), cv2.IMREAD_GRAYSCALE)
            if m is None:
                raise FileNotFoundError(f"Could not read mask: {selected_mask_paths[i]}")
            if m.shape[:2] != (target_h, target_w):
                m = cv2.resize(m, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            m = (m > 127).astype(np.uint8) * 255
        else:
            m = (depth_m > 0.0)
            if conf is not None and float(args.conf_mask_threshold) > 0.0:
                m = m & (conf[i] >= float(args.conf_mask_threshold))
            m = m.astype(np.uint8) * 255
        cv2.imwrite(str(mask_out_dir / name), m)

    np.save(out_dir / "intrinsics.npy", intrinsics.astype(np.float32))
    np.save(out_dir / "extrinsics.npy", extrinsics.astype(np.float32))

    meta = {
        "video_path": str(Path(args.video).resolve()),
        "source_fps": float(src_fps),
        "output_fps": float(args.fps),
        "sampled_frame_indices": sampled_indices,
        "num_frames": int(n),
        "depth_scale": float(args.depth_scale),
        "max_depth_m": float(args.max_depth_m),
        "repo_path": str(Path(args.repo_path).resolve()),
        "model_dir": str(args.model_dir),
        "model_name": str(args.model_name),
        "process_res": int(args.process_res),
        "process_res_method": str(args.process_res_method),
        "device_requested": str(args.device),
        "device_used": run_device,
        "mask_source": "external" if selected_mask_paths is not None else "da3_depth_conf",
        "layout": {
            "images": "images/*.png",
            "depth": "depth/*.png",
            "fg_mask": "fg_mask/*.png",
            "confidence": "confidence/*.png",
            "intrinsics": "intrinsics.npy",
            "extrinsics": "extrinsics.npy",
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
