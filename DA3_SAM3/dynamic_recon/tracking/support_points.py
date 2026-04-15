from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class SupportTrack:
    instance_id: int
    point_id: int
    frames: list[int]
    xy: list[tuple[float, float]]
    valid: list[bool]


def sample_support_points(mask: np.ndarray, num_points: int, method: str = "grid_or_farthest") -> np.ndarray:
    del method
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or num_points <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    coords = np.stack([xs, ys], axis=1).astype(np.float32)
    if coords.shape[0] <= num_points:
        return coords
    step = max(coords.shape[0] // num_points, 1)
    sampled = coords[::step][:num_points]
    if sampled.shape[0] < num_points:
        extra_idx = np.linspace(0, coords.shape[0] - 1, num_points - sampled.shape[0], dtype=np.int32)
        sampled = np.concatenate([sampled, coords[extra_idx]], axis=0)[:num_points]
    return sampled


def filter_support_points_by_texture(points: np.ndarray, image: np.ndarray, min_std: float = 0.0) -> np.ndarray:
    if points.size == 0 or min_std <= 0:
        return points
    image = image.astype(np.float32)
    height, width = image.shape[:2]
    keep: list[np.ndarray] = []
    for point in points:
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        x0 = max(x - 2, 0)
        x1 = min(x + 3, width)
        y0 = max(y - 2, 0)
        y1 = min(y + 3, height)
        patch = image[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        if float(np.std(patch)) >= min_std:
            keep.append(point)
    if not keep:
        return points[: min(len(points), 4)]
    return np.asarray(keep, dtype=np.float32)


def propagate_support_points(
    points: np.ndarray,
    masks_by_frame: list[np.ndarray],
    rgb_frames: list[np.ndarray],
    instance_id: int,
    frame_indices: list[int] | None = None,
    fb_consistency_tol: float = 1.5,
) -> list[SupportTrack]:
    if points.size == 0 or not masks_by_frame or not rgb_frames:
        return []
    frame_indices = frame_indices or list(range(len(masks_by_frame)))
    gray_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame.astype(np.uint8) for frame in rgb_frames]
    tracks: list[SupportTrack] = []
    for point_id, point in enumerate(points):
        track_xy: list[tuple[float, float]] = []
        track_valid: list[bool] = []
        current = np.asarray(point, dtype=np.float32).reshape(1, 1, 2)
        first_mask = masks_by_frame[0]
        current_xy, is_valid = _snap_to_mask(current[0, 0], first_mask)
        current[0, 0] = current_xy
        track_xy.append((float(current_xy[0]), float(current_xy[1])))
        track_valid.append(bool(is_valid))
        for local_index in range(1, len(masks_by_frame)):
            prev_gray = gray_frames[local_index - 1]
            next_gray = gray_frames[local_index]
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                next_gray,
                current,
                None,
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
            )
            if next_points is None or status is None or int(status[0, 0]) == 0:
                propagated_xy, is_valid = current[0, 0], False
            else:
                back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
                    next_gray,
                    prev_gray,
                    next_points,
                    None,
                    winSize=(15, 15),
                    maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
                )
                fb_error = np.inf
                if back_points is not None and back_status is not None and int(back_status[0, 0]) == 1:
                    fb_error = float(np.linalg.norm(back_points[0, 0] - current[0, 0]))
                if not np.isfinite(fb_error) or fb_error > fb_consistency_tol:
                    propagated_xy, is_valid = current[0, 0], False
                else:
                    propagated_xy, is_valid = _snap_to_mask(next_points[0, 0], masks_by_frame[local_index])
                    current = next_points.astype(np.float32)
                    current[0, 0] = propagated_xy
            track_xy.append((float(propagated_xy[0]), float(propagated_xy[1])))
            track_valid.append(bool(is_valid))
        tracks.append(
            SupportTrack(
                instance_id=int(instance_id),
                point_id=int(point_id),
                frames=[int(frame_idx) for frame_idx in frame_indices],
                xy=track_xy,
                valid=track_valid,
            )
        )
    return tracks


def _snap_to_mask(point_xy: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, bool]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return np.asarray(point_xy, dtype=np.float32), False
    coords = np.stack([xs, ys], axis=1).astype(np.float32)
    distances = np.linalg.norm(coords - point_xy[None], axis=1)
    nearest = coords[int(np.argmin(distances))]
    return nearest.astype(np.float32), True
