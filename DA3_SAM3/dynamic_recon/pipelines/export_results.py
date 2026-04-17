from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from dynamic_recon.progress import stage_bar
from dynamic_recon.visualization import (
    write_dynamic_masked_video,
    write_pointcloud_over_time_video,
    write_dynamic_probability_masked_video,
    write_pointcloud_camera_visualizations,
)


def export_results(run_dir: str | Path, state: dict[str, object]) -> dict[str, object]:
    run_dir = Path(run_dir)
    exports = run_dir / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    stage = stage_bar(desc="Export", total=4)

    fps = float(getattr(state["da3_seq"], "fps", 0.0) or 5.0)
    dynamic_prob_seq = state.get("dynamic_prob_seq")
    if dynamic_prob_seq is not None:
        dynamic_prob_np = dynamic_prob_seq.detach().cpu().numpy()
        fusion_cfg = state["config"].fusion
        masked_video_path = write_dynamic_probability_masked_video(
            state.get("segmentation_resource", state["source_video"]),
            dynamic_prob_np,
            exports / "dynamic_masked.mp4",
            fps,
            threshold=float(getattr(fusion_cfg, "export_dynamic_threshold", 0.65)),
            neighbor_threshold=float(getattr(fusion_cfg, "export_neighbor_threshold", 0.5)),
        )
    else:
        masked_video_path = write_dynamic_masked_video(
            state.get("segmentation_resource", state["source_video"]),
            state["sam3_out"].frames,
            exports / "dynamic_masked.mp4",
            fps,
        )
    stage.update(1)

    poses = np.stack([pose.detach().cpu().numpy() for pose in state["poses"]], axis=0)
    np.save(exports / "camera_poses.npy", poses)
    with (exports / "camera_poses.json").open("w", encoding="utf-8") as handle:
        json.dump(
            [
                {"frame_index": int(frame_index), "extrinsics": pose.tolist()}
                for frame_index, pose in enumerate(poses)
            ],
            handle,
            indent=2,
        )
    stage.update(1)

    fusion_cfg = state["config"].fusion
    pointcloud_outputs = write_pointcloud_camera_visualizations(
        state["da3_seq"],
        state["poses"],
        dynamic_prob_seq,
        exports,
        dynamic_threshold=float(getattr(fusion_cfg, "export_dynamic_threshold", 0.65)),
        point_stride=int(getattr(fusion_cfg, "export_point_stride", 6)),
        max_points=int(getattr(fusion_cfg, "export_max_points", 120000)),
    )
    stage.update(1)

    pointcloud_video_path = write_pointcloud_over_time_video(
        state["da3_seq"],
        state["poses"],
        dynamic_prob_seq,
        exports / "scene_pointcloud_over_time.mp4",
        fps=fps,
        dynamic_threshold=float(getattr(fusion_cfg, "export_dynamic_threshold", 0.65)),
        point_stride=int(getattr(fusion_cfg, "export_point_stride", 6)),
    )
    stage.update(1)
    stage.close()

    if not state["config"].pipeline.save_intermediates:
        _cleanup_intermediates(run_dir, state)

    return {
        "dynamic_masked_video": masked_video_path,
        "camera_poses": exports / "camera_poses.npy",
        "pointcloud_visualizations": pointcloud_outputs,
        "pointcloud_video": pointcloud_video_path,
    }


def _cleanup_intermediates(run_dir: Path, state: dict[str, object]) -> None:
    temp_root = state.get("temp_root")
    if temp_root:
        shutil.rmtree(Path(temp_root), ignore_errors=True)
    for name in ("frames", "frames_sam", "da3", "geometry", "sam3"):
        shutil.rmtree(run_dir / name, ignore_errors=True)
