from __future__ import annotations

import logging
from pathlib import Path

import cv2
from PIL import Image

from dynamic_recon.io_utils import ensure_dir
from dynamic_recon.types import VideoMetadata

LOGGER = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ingest_video_or_frames(
    input_path: Path,
    output_dir: Path,
    fps: float | None,
    max_frames: int | None,
    resize_long_edge: int | None,
) -> VideoMetadata:
    frames_dir = ensure_dir(output_dir / "frames")
    if input_path.is_dir():
        frame_paths = _copy_image_folder(input_path, frames_dir, max_frames, resize_long_edge)
        timestamps = _folder_timestamps(len(frame_paths), fps)
        size = _read_image_size(frame_paths[0]) if frame_paths else (0, 0)
        return VideoMetadata(frame_paths=frame_paths, timestamps=timestamps, fps=fps or 0.0, frame_size=size)
    return _extract_video_frames(input_path, frames_dir, fps, max_frames, resize_long_edge)


def _copy_image_folder(
    input_dir: Path,
    frames_dir: Path,
    max_frames: int | None,
    resize_long_edge: int | None,
) -> list[Path]:
    frame_paths: list[Path] = []
    for index, image_path in enumerate(sorted(path for path in input_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)):
        if max_frames is not None and index >= max_frames:
            break
        output_path = frames_dir / f"{index:06d}.png"
        image = Image.open(image_path).convert("RGB")
        image = _resize_if_needed(image, resize_long_edge)
        image.save(output_path)
        frame_paths.append(output_path)
    LOGGER.info("Ingested %d frames from image folder", len(frame_paths))
    return frame_paths


def _extract_video_frames(
    video_path: Path,
    frames_dir: Path,
    fps: float | None,
    max_frames: int | None,
    resize_long_edge: int | None,
) -> VideoMetadata:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    sample_every = max(int(round(source_fps / fps)), 1) if fps and source_fps else 1
    frame_paths: list[Path] = []
    timestamps: list[float] = []
    frame_idx = 0
    saved_idx = 0

    while True:
        success, frame_bgr = capture.read()
        if not success:
            break
        if frame_idx % sample_every != 0:
            frame_idx += 1
            continue
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        image = _resize_if_needed(image, resize_long_edge)
        output_path = frames_dir / f"{saved_idx:06d}.png"
        image.save(output_path)
        frame_paths.append(output_path)
        timestamps.append(frame_idx / source_fps if source_fps else float(saved_idx))
        saved_idx += 1
        frame_idx += 1
        if max_frames is not None and saved_idx >= max_frames:
            break

    capture.release()
    size = _read_image_size(frame_paths[0]) if frame_paths else (0, 0)
    effective_fps = fps or source_fps
    LOGGER.info("Extracted %d frames from %s", len(frame_paths), video_path)
    return VideoMetadata(frame_paths=frame_paths, timestamps=timestamps, fps=effective_fps, frame_size=size)


def _resize_if_needed(image: Image.Image, resize_long_edge: int | None) -> Image.Image:
    if not resize_long_edge:
        return image
    width, height = image.size
    long_edge = max(width, height)
    if long_edge <= resize_long_edge:
        return image
    scale = resize_long_edge / long_edge
    new_size = (int(round(width * scale)), int(round(height * scale)))
    return image.resize(new_size, Image.Resampling.BILINEAR)


def _folder_timestamps(count: int, fps: float | None) -> list[float]:
    if not fps:
        return [float(index) for index in range(count)]
    return [index / fps for index in range(count)]


def _read_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
        return (height, width)
