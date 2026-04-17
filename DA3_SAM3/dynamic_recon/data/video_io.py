from __future__ import annotations

from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


def get_video_metadata(video_path: str | Path) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(video_path)
    meta = {
        "fps": float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
        "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    }
    capture.release()
    return meta


def sample_keyframes(num_frames: int, stride: int | None = None, max_frames: int | None = None) -> list[int]:
    step = max(stride or 1, 1)
    indices = list(range(0, num_frames, step))
    if max_frames is not None:
        indices = indices[:max_frames]
    return indices


def read_video_frames(
    video_path: str | Path,
    resize_long_edge: int | None = None,
    frame_range: tuple[int, int] | None = None,
    stride: int = 1,
    max_frames: int | None = None,
) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(video_path)
    start, end = frame_range or (0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
    frames: list[np.ndarray] = []
    frame_index = 0
    while True:
        success, frame_bgr = capture.read()
        if not success:
            break
        if frame_index < start:
            frame_index += 1
            continue
        if frame_index >= end:
            break
        if (frame_index - start) % max(stride, 1) != 0:
            frame_index += 1
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if resize_long_edge:
            frame_rgb = _resize(frame_rgb, resize_long_edge)
        frames.append(frame_rgb)
        if max_frames is not None and len(frames) >= max_frames:
            break
        frame_index += 1
    capture.release()
    return frames


def write_video(frames: list[np.ndarray], output_path: str | Path, fps: float) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output, fps=max(fps, 1.0)) as writer:
        for frame in frames:
            writer.append_data(frame)


def stable_frame_name(frame_index: int) -> str:
    return f"{frame_index:06d}.jpg"


def _resize(frame_rgb: np.ndarray, resize_long_edge: int) -> np.ndarray:
    height, width = frame_rgb.shape[:2]
    long_edge = max(height, width)
    if long_edge <= resize_long_edge:
        return frame_rgb
    scale = resize_long_edge / long_edge
    size = (int(round(width * scale)), int(round(height * scale)))
    return cv2.resize(frame_rgb, size, interpolation=cv2.INTER_LINEAR)
