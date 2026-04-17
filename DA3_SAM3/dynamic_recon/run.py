from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np

from dynamic_recon.da3_wrapper import run_da3
from dynamic_recon.fusion import LABEL_DYNAMIC, LABEL_STATIC, LABEL_UNCERTAIN, compute_dynamic_consistency
from dynamic_recon.geometry import fuse_point_cloud, save_point_cloud
from dynamic_recon.io_utils import ensure_dir, load_yaml, merge_dicts, save_json
from dynamic_recon.sam3_wrapper import run_sam3
from dynamic_recon.video import ingest_video_or_frames
from dynamic_recon.visualization import save_overlay_frames, write_video

LOGGER = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DA3 + SAM3 dynamic/static 3D reconstruction")
    parser.add_argument("--video", required=True, help="Path to an MP4 video or an image folder")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--config", default="configs/pipeline.default.yaml", help="Path to pipeline config YAML")
    parser.add_argument("--fps", type=float, default=None, help="Optional output FPS for frame extraction")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument("--prompts", default=None, help="Comma-separated SAM 3 text prompts")
    parser.add_argument("--sam3-checkpoint", default=None, help="SAM 3 checkpoint family")
    parser.add_argument("--da3-model", default=None, help="DA3 model name")
    parser.add_argument("--chunk-size", type=int, default=None, help="Chunk size for long-video inference")
    parser.add_argument("--use-ray-pose", action="store_true", help="Use ray pose estimation in DA3")
    parser.add_argument("--ref-view-strategy", default=None, help="DA3 reference view strategy")
    parser.add_argument("--dynamic-thresh", type=float, default=None, help="Dynamic fusion threshold")
    parser.add_argument("--confidence-thresh", type=float, default=None, help="Minimum DA3 confidence for fusion")
    parser.add_argument("--save-intermediate", action="store_true", help="Save intermediate artifacts")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and dependencies without inference")
    parser.add_argument("--allow-mock-models", action="store_true", help="Allow smoke-mode fallback when DA3 or SAM3 is unavailable")
    return parser


def load_config(config_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    config = load_yaml(config_path)
    overrides: dict[str, Any] = {
        "video": {
            "fps": args.fps if args.fps is not None else config.get("video", {}).get("fps"),
            "max_frames": args.max_frames if args.max_frames is not None else config.get("video", {}).get("max_frames"),
        },
        "da3": {
            "model_name": args.da3_model or config.get("da3", {}).get("model_name"),
            "use_ray_pose": args.use_ray_pose or config.get("da3", {}).get("use_ray_pose"),
            "ref_view_strategy": args.ref_view_strategy or config.get("da3", {}).get("ref_view_strategy"),
            "chunk_size": args.chunk_size or config.get("da3", {}).get("chunk_size"),
            "confidence_thresh": args.confidence_thresh if args.confidence_thresh is not None else config.get("da3", {}).get("confidence_thresh"),
        },
        "sam3": {
            "checkpoint": args.sam3_checkpoint or config.get("sam3", {}).get("checkpoint"),
        },
        "fusion": {
            "dynamic_thresh": args.dynamic_thresh if args.dynamic_thresh is not None else config.get("fusion", {}).get("dynamic_thresh"),
        },
        "output": {
            "save_intermediate": args.save_intermediate or config.get("output", {}).get("save_intermediate"),
            "allow_mock_models": args.allow_mock_models or config.get("output", {}).get("allow_mock_models"),
        },
    }
    if args.prompts:
        overrides.setdefault("sam3", {})["prompts"] = [item.strip() for item in args.prompts.split(",") if item.strip()]
    return merge_dicts(config, overrides)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args()
    config = load_config(Path(args.config), args)
    output_dir = Path(args.output)

    LOGGER.info("Stage 1/6: validate input and config")
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Missing input video or frame directory: {video_path}")

    if args.dry_run:
        _run_dry_check(video_path, output_dir, config)
        return

    LOGGER.info("Stage 2/6: ingest video frames")
    video_meta = ingest_video_or_frames(
        input_path=video_path,
        output_dir=output_dir,
        fps=config["video"]["fps"],
        max_frames=config["video"]["max_frames"],
        resize_long_edge=config["video"]["resize_long_edge"],
    )

    LOGGER.info("Stage 3/6: run DA3")
    da3_result = run_da3(
        video_meta=video_meta,
        output_dir=ensure_dir(output_dir / "depth"),
        model_name=config["da3"]["model_name"],
        use_ray_pose=config["da3"]["use_ray_pose"],
        ref_view_strategy=config["da3"]["ref_view_strategy"],
        chunk_size=int(config["da3"]["chunk_size"]),
        allow_mock_models=bool(config["output"]["allow_mock_models"]),
    )

    LOGGER.info("Stage 4/6: run SAM3")
    sam3_result = run_sam3(
        video_meta=video_meta,
        output_dir=ensure_dir(output_dir / "sam3"),
        prompts=list(config["sam3"]["prompts"]),
        checkpoint=str(config["sam3"]["checkpoint"]),
        prompt_json=Path(config["sam3"]["prompt_json"]) if config["sam3"].get("prompt_json") else None,
        allow_mock_models=bool(config["output"]["allow_mock_models"]),
    )

    LOGGER.info("Stage 5/6: fuse geometry and masks")
    fusion = compute_dynamic_consistency(
        da3=da3_result,
        sam3=sam3_result,
        dynamic_thresh=float(config["fusion"]["dynamic_thresh"]),
        reproj_thresh_px=float(config["fusion"]["reproj_thresh_px"]),
        uncertain_margin=float(config["fusion"]["uncertain_margin"]),
        min_depth=float(config["fusion"]["min_depth"]),
        max_depth=float(config["fusion"]["max_depth"]),
    )

    points_all, _ = fuse_point_cloud(
        da3_result.depth,
        da3_result.confidence,
        da3_result.intrinsics,
        da3_result.extrinsics_w2c,
        min_confidence=float(config["da3"]["confidence_thresh"]),
        min_depth=float(config["fusion"]["min_depth"]),
        max_depth=float(config["fusion"]["max_depth"]),
    )
    save_point_cloud(output_dir / "pointcloud_all_raw.ply", points_all)
    save_point_cloud(output_dir / "pointcloud_static_raw.ply", points_all)
    _export_labeled_masks(output_dir, fusion.label_map)
    _export_labeled_pointclouds(output_dir, fusion.label_map, da3_result, config)
    _export_pose_files(output_dir, da3_result)

    LOGGER.info("Stage 6/6: export visualizations and summary")
    overlay_paths = save_overlay_frames(video_meta.frame_paths, fusion.dynamic_score, fusion.label_map, ensure_dir(output_dir / "overlays"))
    if config["output"]["save_visualizations"]:
        write_video(overlay_paths, output_dir / "visualization.mp4", video_meta.fps or 5.0)

    summary = {
        "frames": len(video_meta.frame_paths),
        "fps": video_meta.fps,
        "frame_size": video_meta.frame_size,
        "track_stats": fusion.track_stats,
        "dynamic_pixels": int(np.sum(fusion.binary_dynamic)),
    }
    save_json(output_dir / "summary.json", summary)


def _run_dry_check(video_path: Path, output_dir: Path, config: dict[str, Any]) -> None:
    ensure_dir(output_dir)
    summary = {
        "video_exists": video_path.exists(),
        "output_dir": str(output_dir),
        "config": config,
        "da3_importable": _is_importable("depth_anything_3") or _is_importable("depth_anything3") or _is_importable("depth_anything"),
        "sam3_importable": _is_importable("sam3") or _is_importable("segment_anything_3"),
    }
    save_json(output_dir / "dry_run_summary.json", summary)
    LOGGER.info("Dry run completed. Summary written to %s", output_dir / "dry_run_summary.json")


def _is_importable(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def _export_labeled_masks(output_dir: Path, label_map: np.ndarray) -> None:
    mapping = {
        "masks_static": LABEL_STATIC,
        "masks_dynamic": LABEL_DYNAMIC,
        "masks_uncertain": LABEL_UNCERTAIN,
    }
    for folder_name, label in mapping.items():
        folder = ensure_dir(output_dir / folder_name)
        for idx in range(label_map.shape[0]):
            image = (label_map[idx] == label).astype(np.uint8) * 255
            from PIL import Image
            Image.fromarray(image).save(folder / f"{idx:06d}.png")


def _export_labeled_pointclouds(output_dir: Path, label_map: np.ndarray, da3_result: Any, config: dict[str, Any]) -> None:
    from dynamic_recon.geometry import backproject_depth, transform_points, w2c_to_c2w

    dynamic_points: list[np.ndarray] = []
    static_points: list[np.ndarray] = []
    for idx in range(da3_result.depth.shape[0]):
        world_points = transform_points(
            backproject_depth(da3_result.depth[idx], da3_result.intrinsics[idx]),
            w2c_to_c2w(da3_result.extrinsics_w2c[idx]),
        )
        valid = (
            (da3_result.confidence[idx] >= float(config["da3"]["confidence_thresh"]))
            & (da3_result.depth[idx] >= float(config["fusion"]["min_depth"]))
            & (da3_result.depth[idx] <= float(config["fusion"]["max_depth"]))
        )
        dynamic_points.append(world_points[np.logical_and(valid, label_map[idx] == LABEL_DYNAMIC)])
        static_points.append(world_points[np.logical_and(valid, label_map[idx] == LABEL_STATIC)])

    save_point_cloud(output_dir / "pointcloud_dynamic_fused.ply", _concat_points(dynamic_points))
    save_point_cloud(output_dir / "pointcloud_static_fused.ply", _concat_points(static_points))


def _concat_points(items: list[np.ndarray]) -> np.ndarray:
    non_empty = [item for item in items if item.size]
    if not non_empty:
        return np.zeros((0, 3), dtype=float)
    return np.concatenate(non_empty, axis=0)


def _export_pose_files(output_dir: Path, da3_result: Any) -> None:
    poses_dir = ensure_dir(output_dir / "poses")
    cameras_lines = []
    images_lines = []
    for idx, (intrinsic, extrinsic) in enumerate(zip(da3_result.intrinsics, da3_result.extrinsics_w2c, strict=True), start=1):
        width = int(round(intrinsic[0, 2] * 2))
        height = int(round(intrinsic[1, 2] * 2))
        cameras_lines.append(
            f"{idx} PINHOLE {width} {height} {intrinsic[0,0]:.6f} {intrinsic[1,1]:.6f} {intrinsic[0,2]:.6f} {intrinsic[1,2]:.6f}"
        )
        rotation = extrinsic[:3, :3]
        translation = extrinsic[:3, 3]
        qw, qx, qy, qz = _rotation_matrix_to_quaternion(rotation)
        images_lines.append(
            f"{idx} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} {translation[0]:.8f} {translation[1]:.8f} {translation[2]:.8f} {idx} {idx:06d}.png"
        )
        images_lines.append("")
    (poses_dir / "cameras.txt").write_text("\n".join(cameras_lines) + "\n", encoding="utf-8")
    (poses_dir / "images.txt").write_text("\n".join(images_lines), encoding="utf-8")


def _rotation_matrix_to_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    trace = np.trace(rotation)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (rotation[2, 1] - rotation[1, 2]) * s
        qy = (rotation[0, 2] - rotation[2, 0]) * s
        qz = (rotation[1, 0] - rotation[0, 1]) * s
    else:
        idx = int(np.argmax(np.diag(rotation)))
        if idx == 0:
            s = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
            qw = (rotation[2, 1] - rotation[1, 2]) / s
            qx = 0.25 * s
            qy = (rotation[0, 1] + rotation[1, 0]) / s
            qz = (rotation[0, 2] + rotation[2, 0]) / s
        elif idx == 1:
            s = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
            qw = (rotation[0, 2] - rotation[2, 0]) / s
            qx = (rotation[0, 1] + rotation[1, 0]) / s
            qy = 0.25 * s
            qz = (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
            qw = (rotation[1, 0] - rotation[0, 1]) / s
            qx = (rotation[0, 2] + rotation[2, 0]) / s
            qy = (rotation[1, 2] + rotation[2, 1]) / s
            qz = 0.25 * s
    return float(qw), float(qx), float(qy), float(qz)


if __name__ == "__main__":
    main()
