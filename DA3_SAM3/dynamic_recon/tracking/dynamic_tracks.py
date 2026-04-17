from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from dynamic_recon.tracking.support_points import (
    SupportTrack,
    filter_support_points_by_texture,
    propagate_support_points,
    sample_support_points,
)


@dataclass(slots=True)
class DynamicInstanceTrack:
    instance_id: int
    frames: list[int]
    valid: list[bool]
    area: list[int]
    boxes_xyxy: list[list[int] | None]
    centroids_xy: list[list[float] | None]


def export_dense_mask_tracks(sequence_frames: list[object]) -> list[dict[str, object]]:
    tracks = build_dynamic_instance_tracks(sequence_frames)
    return [asdict(track) for track in tracks]


def export_sparse_support_tracks(sequence_frames: list[object], rgb_frames: list[np.ndarray], num_points: int) -> list[SupportTrack]:
    tracks = build_dynamic_instance_tracks(sequence_frames)
    support_tracks: list[SupportTrack] = []
    frame_to_local = {int(frame.frame_index): local_index for local_index, frame in enumerate(sequence_frames)}
    for track in tracks:
        anchor_idx = _first_valid_index(track.valid)
        if anchor_idx is None:
            continue
        frame_index = track.frames[anchor_idx]
        local_anchor_index = frame_to_local[frame_index]
        masks_by_frame = [_mask_for_instance(item, track.instance_id) for item in sequence_frames]
        anchor_mask = masks_by_frame[local_anchor_index]
        points = sample_support_points(anchor_mask, num_points)
        points = filter_support_points_by_texture(points, rgb_frames[local_anchor_index], min_std=4.0)
        support_tracks.extend(
            propagate_support_points(
                points,
                masks_by_frame,
                rgb_frames,
                instance_id=track.instance_id,
                frame_indices=[int(item.frame_index) for item in sequence_frames],
            )
        )
    return support_tracks


def build_dynamic_instance_tracks(sequence_frames: list[object]) -> list[DynamicInstanceTrack]:
    instance_ids = sorted({int(instance_id) for frame in sequence_frames for instance_id in getattr(frame, "instance_ids", [])})
    tracks: list[DynamicInstanceTrack] = []
    for instance_id in instance_ids:
        frames: list[int] = []
        valid: list[bool] = []
        area: list[int] = []
        boxes_xyxy: list[list[int] | None] = []
        centroids_xy: list[list[float] | None] = []
        for frame in sequence_frames:
            frames.append(int(frame.frame_index))
            mask = _mask_for_instance(frame, instance_id)
            present = bool(mask.any())
            valid.append(present)
            if not present:
                area.append(0)
                boxes_xyxy.append(None)
                centroids_xy.append(None)
                continue
            ys, xs = np.where(mask)
            area.append(int(mask.sum()))
            boxes_xyxy.append([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])
            centroids_xy.append([float(xs.mean()), float(ys.mean())])
        tracks.append(
            DynamicInstanceTrack(
                instance_id=instance_id,
                frames=frames,
                valid=valid,
                area=area,
                boxes_xyxy=boxes_xyxy,
                centroids_xy=centroids_xy,
            )
        )
    return tracks


def _mask_for_instance(frame: object, instance_id: int) -> np.ndarray:
    masks = frame.masks.detach().cpu().numpy() if hasattr(frame.masks, "detach") else np.asarray(frame.masks)
    if masks.ndim == 2:
        masks = masks[None]
    instance_ids = [int(item) for item in getattr(frame, "instance_ids", [])]
    if instance_id not in instance_ids:
        return np.zeros(masks.shape[-2:], dtype=bool)
    return masks[instance_ids.index(instance_id)].astype(bool)


def _first_valid_index(values: list[bool]) -> int | None:
    for index, value in enumerate(values):
        if value:
            return index
    return None
