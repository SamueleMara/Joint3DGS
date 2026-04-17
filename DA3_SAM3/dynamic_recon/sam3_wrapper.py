from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from dynamic_recon.io_utils import save_json, save_npz
from dynamic_recon.types import SAM3Result, SAM3Track, VideoMetadata

LOGGER = logging.getLogger(__name__)


def run_sam3(
    video_meta: VideoMetadata,
    output_dir: Path,
    prompts: list[str],
    checkpoint: str,
    prompt_json: Path | None,
    allow_mock_models: bool,
) -> SAM3Result:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        _import_sam3()
        LOGGER.info("SAM 3 package import succeeded; using placeholder adapter path.")
        result = _mock_sam3_result(video_meta, prompts)
        backend = "official-import-placeholder"
    except ImportError:
        if not allow_mock_models:
            raise RuntimeError(
                "SAM 3 is not installed. Run scripts/setup_submodules.sh and scripts/setup_env.sh "
                "or pass --allow-mock-models for a smoke run."
            )
        LOGGER.warning("Falling back to mock SAM 3 inference because the upstream package is unavailable.")
        result = _mock_sam3_result(video_meta, prompts)
        backend = "mock"

    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx in range(result.masks.shape[0]):
        for obj_idx, object_id in enumerate(result.object_ids):
            Image.fromarray((result.masks[frame_idx, obj_idx] * 255).astype(np.uint8)).save(
                masks_dir / f"{frame_idx:06d}_obj{object_id:03d}.png"
            )

    prompts_payload: dict[str, Any] = {"text_prompts": prompts, "checkpoint": checkpoint, "backend": backend}
    if prompt_json:
        prompts_payload["prompt_json"] = json.loads(prompt_json.read_text(encoding="utf-8"))
    save_json(output_dir / "prompts.json", prompts_payload)
    save_json(
        output_dir / "tracks.json",
        {
            "labels": result.labels,
            "tracks": [
                {
                    "object_id": track.object_id,
                    "label": track.label,
                    "frame_indices": track.frame_indices,
                    "scores": track.scores,
                }
                for track in result.tracks
            ],
        },
    )
    save_npz(output_dir / "masks.npz", masks=result.masks)
    return result


def _import_sam3() -> Any:
    for module_name in ("sam3", "segment_anything_3"):
        try:
            return importlib.import_module(module_name)
        except ImportError:
            continue
    raise ImportError("Unable to import a SAM 3 module")


def _mock_sam3_result(video_meta: VideoMetadata, prompts: list[str]) -> SAM3Result:
    frames = len(video_meta.frame_paths)
    height, width = video_meta.frame_size
    labels = {index + 1: prompt for index, prompt in enumerate(prompts)}
    object_ids = sorted(labels)
    masks = np.zeros((frames, len(object_ids), height, width), dtype=np.uint8)
    tracks: list[SAM3Track] = []

    for obj_idx, object_id in enumerate(object_ids):
        label = labels[object_id]
        size = max(min(height, width) // 6, 8)
        frame_indices: list[int] = []
        scores: list[float] = []
        for frame_idx in range(frames):
            image = Image.new("L", (width, height), 0)
            draw = ImageDraw.Draw(image)
            left = int((frame_idx * (size + 4) + obj_idx * size * 2) % max(width - size, 1))
            top = int((height * 0.2) + obj_idx * size)
            draw.rectangle([left, top, min(left + size, width - 1), min(top + size, height - 1)], fill=255)
            masks[frame_idx, obj_idx] = (np.asarray(image) > 0).astype(np.uint8)
            frame_indices.append(frame_idx)
            scores.append(0.9)
        tracks.append(SAM3Track(object_id=object_id, label=label, frame_indices=frame_indices, scores=scores))

    return SAM3Result(masks=masks, object_ids=object_ids, labels=labels, tracks=tracks)
