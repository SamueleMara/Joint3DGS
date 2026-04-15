from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from dynamic_recon.io_utils import save_json, save_npy
from dynamic_recon.types import DA3Result, VideoMetadata

LOGGER = logging.getLogger(__name__)


def run_da3(
    video_meta: VideoMetadata,
    output_dir: Path,
    model_name: str,
    use_ray_pose: bool,
    ref_view_strategy: str,
    chunk_size: int,
    allow_mock_models: bool,
) -> DA3Result:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        _import_da3()
        LOGGER.info("DA3 package import succeeded; using placeholder adapter path.")
        result = _mock_da3_result(video_meta, model_name, use_ray_pose, ref_view_strategy, chunk_size)
        result.metadata["backend"] = "official-import-placeholder"
    except ImportError:
        if not allow_mock_models:
            raise RuntimeError(
                "Depth Anything 3 is not installed. Run scripts/setup_submodules.sh and scripts/setup_env.sh "
                "or pass --allow-mock-models for a smoke run."
            )
        LOGGER.warning("Falling back to mock DA3 inference because the upstream package is unavailable.")
        result = _mock_da3_result(video_meta, model_name, use_ray_pose, ref_view_strategy, chunk_size)
        result.metadata["backend"] = "mock"

    save_npy(output_dir / "depth.npy", result.depth)
    save_npy(output_dir / "conf.npy", result.confidence)
    save_npy(output_dir / "intrinsics.npy", result.intrinsics)
    save_npy(output_dir / "extrinsics.npy", result.extrinsics_w2c)
    save_json(output_dir / "metadata.json", result.metadata)
    return result


def _import_da3() -> Any:
    for module_name in (
        "depth_anything_3",
        "depth_anything3",
        "depth_anything",
    ):
        try:
            return importlib.import_module(module_name)
        except ImportError:
            continue
    raise ImportError("Unable to import a Depth Anything 3 module")


def _mock_da3_result(
    video_meta: VideoMetadata,
    model_name: str,
    use_ray_pose: bool,
    ref_view_strategy: str,
    chunk_size: int,
) -> DA3Result:
    frames = len(video_meta.frame_paths)
    height, width = video_meta.frame_size
    depth = np.zeros((frames, height, width), dtype=np.float32)
    conf = np.zeros_like(depth)
    intrinsics = np.zeros((frames, 3, 3), dtype=np.float32)
    extrinsics = np.zeros((frames, 4, 4), dtype=np.float32)

    fx = fy = float(max(height, width))
    cx = width / 2.0
    cy = height / 2.0
    base_intrinsics = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    for idx, frame_path in enumerate(video_meta.frame_paths):
        image = np.asarray(Image.open(frame_path).convert("RGB"), dtype=np.float32) / 255.0
        gradient = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
        gray = image.mean(axis=2)
        depth[idx] = 1.0 + 3.0 * gray + gradient
        conf[idx] = np.clip(0.5 + 0.5 * gray, 0.0, 1.0)
        intrinsics[idx] = base_intrinsics
        extrinsics[idx] = np.eye(4, dtype=np.float32)
        extrinsics[idx, 0, 3] = -0.05 * idx

    metadata = {
        "model_name": model_name,
        "use_ray_pose": use_ray_pose,
        "ref_view_strategy": ref_view_strategy,
        "chunk_size": chunk_size,
        "depth_shape": list(depth.shape),
        "intrinsics_shape": list(intrinsics.shape),
        "extrinsics_convention": "OpenCV/COLMAP w2c 4x4",
    }
    return DA3Result(depth=depth, confidence=conf, intrinsics=intrinsics, extrinsics_w2c=extrinsics, metadata=metadata)
