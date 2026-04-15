#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

from articulation.data import RGBDSequence, RelativeMotionResult, TrackBatch, save_tracks_npz
from articulation.external.dino_backend_adapter import DinoBackendAdapter
from articulation.external.joint_clue_adapter import JointClueEstimator
from articulation.features import build_feature_graph, initialize_two_part_logits
from articulation.joint.consensus import ConsensusResult
from articulation.joint.multiview import (
    aggregate_model_priors,
    fuse_signed_pitches,
    fuse_signed_states,
    robust_axis_point_consensus,
    robust_direction_consensus,
)
from articulation.joint.optimizer import fit_candidate_model
from articulation.joint.outputs import joint_result_to_dict
from articulation.joint.relative_motion import choose_reference_part, compute_relative_motion
from articulation.pipeline.stage1_segmentation import run_stage1_segmentation
from articulation.pipeline.stage2_joint import run_stage2_joint
from articulation.pipeline.tracking_utils import run_tracking_and_lift
from articulation.utils import configure_logging, load_yaml_config


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class ViewRun:
    camera: str
    sequence: RGBDSequence
    tracks: TrackBatch
    seg: Any
    joint: Any
    world_from_camera: torch.Tensor
    view_weight: float


def _ensure_matplotlib_cache(out_dir: Path) -> None:
    cache = out_dir / ".cache" / "matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))


def _maybe_add_repo_path(path: str | None) -> None:
    if not path:
        return
    p = str(Path(path).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def _sorted_files(dir_path: Path) -> list[Path]:
    files = [p for p in dir_path.iterdir() if p.is_file() and (p.suffix.lower() in _IMAGE_EXTS or p.suffix.lower() == ".npy")]
    return sorted(files)


def _match_frame_triplets(rgb_files: list[Path], depth_files: list[Path], mask_files: list[Path]) -> tuple[list[Path], list[Path], list[Path]]:
    if not rgb_files or not depth_files or not mask_files:
        raise ValueError("rgb/depth/mask folders must all be non-empty")

    rgb_map = {p.stem: p for p in rgb_files}
    depth_map = {p.stem: p for p in depth_files}
    mask_map = {p.stem: p for p in mask_files}
    common = sorted(set(rgb_map) & set(depth_map) & set(mask_map))
    if not common:
        raise ValueError("No common frame stems across rgb/depth/mask")

    return [rgb_map[k] for k in common], [depth_map[k] for k in common], [mask_map[k] for k in common]


def _read_rgb(path: Path) -> torch.Tensor:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read RGB image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0


def _read_depth(path: Path, depth_scale: float, depth_npy_scale: float) -> torch.Tensor:
    ext = path.suffix.lower()
    if ext == ".npy":
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2:
            raise ValueError(f"Depth npy must be [H,W], got {arr.shape} for {path}")
        depth = arr * float(depth_npy_scale)
    else:
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"Could not read depth image: {path}")
        depth = raw.astype(np.float32) * float(depth_scale)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth = np.maximum(depth, 0.0)
    return torch.from_numpy(depth).unsqueeze(0)


def _read_mask(path: Path) -> torch.Tensor:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Could not read mask image: {path}")
    return torch.from_numpy((m > 127).astype(np.float32)).unsqueeze(0)


def _normalize_np(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Cannot normalize near-zero vector")
    return v / n


def _camera_intrinsics_from_entry(entry: dict[str, Any]) -> np.ndarray:
    w = int(entry["width"])
    h = int(entry["height"])
    fx = entry.get("fx", None)
    fy = entry.get("fy", None)
    cx = entry.get("cx", None)
    cy = entry.get("cy", None)
    if fx is None or fy is None:
        fov_y_deg = float(entry.get("fov_y_deg", 45.0))
        fy_f = 0.5 * h / math.tan(math.radians(fov_y_deg) * 0.5)
        fx_f = fy_f
        cx_f = (w - 1) * 0.5
        cy_f = (h - 1) * 0.5
    else:
        fx_f = float(fx)
        fy_f = float(fy)
        cx_f = float((w - 1) * 0.5 if cx is None else cx)
        cy_f = float((h - 1) * 0.5 if cy is None else cy)
    return np.array([[fx_f, 0.0, cx_f], [0.0, fy_f, cy_f], [0.0, 0.0, 1.0]], dtype=np.float32)


def _lookat_world_from_camera(position: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    z = _normalize_np(target - position)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-8:
        alt_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        x = np.cross(z, alt_up)
    x = _normalize_np(x)
    y = _normalize_np(np.cross(z, x))

    t = np.eye(4, dtype=np.float32)
    t[:3, :3] = np.stack([x, y, z], axis=1)
    t[:3, 3] = position.astype(np.float32)
    return t


def _invert_4x4(t: np.ndarray) -> np.ndarray:
    r = t[:3, :3]
    tr = t[:3, 3]
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = r.T
    out[:3, 3] = -(r.T @ tr)
    return out


def _world_from_camera_from_entry(entry: dict[str, Any], extrinsics_convention: str) -> np.ndarray:
    pose = entry.get("pose_matrix_4x4", None)
    if pose is not None:
        mat = np.asarray(pose, dtype=np.float32)
        if mat.shape != (4, 4):
            raise ValueError(f"pose_matrix_4x4 must be [4,4], got {mat.shape}")
        if extrinsics_convention == "world_from_camera":
            return mat
        if extrinsics_convention == "camera_from_world":
            return _invert_4x4(mat)
        raise ValueError(f"Unknown extrinsics convention: {extrinsics_convention}")

    position = np.asarray(entry.get("position", [0.0, 0.0, 0.0]), dtype=np.float32)
    target = np.asarray(entry.get("target", [0.0, 0.0, 1.0]), dtype=np.float32)
    up = np.asarray(entry.get("up", [0.0, 0.0, 1.0]), dtype=np.float32)
    return _lookat_world_from_camera(position, target, up)


def _load_camera_entries(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text())
    cams = payload.get("cameras", [])
    out: dict[str, dict[str, Any]] = {}
    for c in cams:
        name = str(c.get("name", ""))
        if name:
            out[name] = c
    if not out:
        raise ValueError(f"No cameras found in metadata file: {path}")
    return out


def _load_sequence_for_camera(
    input_dir: Path,
    camera: str,
    camera_entry: dict[str, Any],
    rgb_root: str,
    depth_root: str,
    mask_root: str,
    depth_scale: float,
    depth_npy_scale: float,
    max_frames: int | None,
) -> RGBDSequence:
    rgb_dir = input_dir / rgb_root / camera
    depth_dir = input_dir / depth_root / camera
    mask_dir = input_dir / mask_root / camera
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"RGB dir missing for {camera}: {rgb_dir}")
    if not depth_dir.is_dir():
        raise FileNotFoundError(f"Depth dir missing for {camera}: {depth_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask dir missing for {camera}: {mask_dir}")

    rgb_f, depth_f, mask_f = _match_frame_triplets(_sorted_files(rgb_dir), _sorted_files(depth_dir), _sorted_files(mask_dir))
    if max_frames is not None:
        rgb_f = rgb_f[: max_frames]
        depth_f = depth_f[: max_frames]
        mask_f = mask_f[: max_frames]

    rgb = torch.stack([_read_rgb(p) for p in rgb_f], dim=0)
    depth = torch.stack([_read_depth(p, depth_scale=depth_scale, depth_npy_scale=depth_npy_scale) for p in depth_f], dim=0)
    fg_mask = torch.stack([_read_mask(p) for p in mask_f], dim=0)

    if rgb.shape[0] != depth.shape[0] or rgb.shape[0] != fg_mask.shape[0]:
        raise RuntimeError("RGB/depth/mask frame count mismatch after loading")
    if rgb.shape[-2:] != depth.shape[-2:] or rgb.shape[-2:] != fg_mask.shape[-2:]:
        raise ValueError(f"Resolution mismatch for {camera}: rgb={rgb.shape[-2:]} depth={depth.shape[-2:]} mask={fg_mask.shape[-2:]}")

    k = torch.from_numpy(_camera_intrinsics_from_entry(camera_entry))
    return RGBDSequence(
        rgb=rgb,
        depth=depth,
        fg_mask=fg_mask,
        K=k,
        frame_ids=list(range(rgb.shape[0])),
        meta={
            "input_dir": str(input_dir),
            "camera": camera,
            "rgb_dir": str(rgb_dir),
            "depth_dir": str(depth_dir),
            "mask_dir": str(mask_dir),
        },
    )


def _view_weight(tracks: TrackBatch, seg_diag: dict[str, Any]) -> float:
    p = float(max(int(tracks.P), 1))
    nonfinite = 0.0
    x = seg_diag.get("nonfinite_steps", None)
    if isinstance(x, list) and x:
        nonfinite = float(x[-1])
    elif isinstance(x, (int, float)):
        nonfinite = float(x)
    return float(math.log1p(p) / (1.0 + 0.05 * max(nonfinite, 0.0)))


def _save_segmentation(path: Path, seg: Any) -> None:
    torch.save(
        {
            "point_logits": seg.point_logits.detach().cpu(),
            "point_probs": seg.point_probs.detach().cpu(),
            "point_labels": seg.point_labels.detach().cpu(),
            "masks_per_frame": seg.masks_per_frame.detach().cpu(),
            "transforms_part0": seg.transforms_part0.detach().cpu(),
            "transforms_part1": seg.transforms_part1.detach().cpu(),
            "diagnostics": seg.diagnostics,
        },
        path,
    )


def _joint_candidate_map(joint: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for c in joint.candidates:
        out[str(c.model_name)] = c
    return out


def _to_world_points(x_cam: torch.Tensor, world_from_camera: torch.Tensor) -> torch.Tensor:
    r = world_from_camera[:3, :3]
    t = world_from_camera[:3, 3]
    return x_cam @ r.transpose(0, 1) + t


def _to_world_dirs(d_cam: torch.Tensor, world_from_camera: torch.Tensor) -> torch.Tensor:
    r = world_from_camera[:3, :3]
    return d_cam @ r.transpose(0, 1)


def _build_world_relative_motion(view: ViewRun) -> RelativeMotionResult:
    labels = view.seg.point_labels.long()
    ref_part, mov_part = choose_reference_part(view.tracks, view.seg)
    ref_mask = labels == ref_part
    mov_mask = labels == mov_part

    rel = compute_relative_motion(
        xyz_part_ref=view.tracks.xyz[ref_mask],
        xyz_part_mov=view.tracks.xyz[mov_mask],
        valid_ref=view.tracks.valid[ref_mask],
        valid_mov=view.tracks.valid[mov_mask],
        reference_part=ref_part,
        moving_part=mov_part,
    )

    can_world = _to_world_points(rel.canonical_points, view.world_from_camera)
    mov_world = _to_world_points(rel.moving_points_rel, view.world_from_camera)
    return RelativeMotionResult(
        reference_part=rel.reference_part,
        moving_part=rel.moving_part,
        canonical_points=can_world,
        moving_points_rel=mov_world,
        valid=rel.valid,
        weights=rel.weights,
        ref_transform_inv=rel.ref_transform_inv,
        diagnostics=dict(rel.diagnostics),
    )


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, tuple):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return float(obj.detach().cpu())
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Multiview dataset root with rgb/<cam>, depth_*/<cam>, mask/<cam>")
    parser.add_argument("--rgb-root", default="rgb")
    parser.add_argument("--depth-root", default="depth_npy")
    parser.add_argument("--mask-root", default="mask")
    parser.add_argument("--camera-pattern", default="cam_*")
    parser.add_argument("--cameras-json", default="metadata/cameras.json")
    parser.add_argument("--extrinsics-convention", choices=["world_from_camera", "camera_from_world"], default="world_from_camera")
    parser.add_argument("--depth-scale", type=float, default=0.001, help="Scale for image depth files (meters = raw * depth_scale)")
    parser.add_argument("--depth-npy-scale", type=float, default=1.0, help="Scale for .npy depth files (meters = raw * depth_npy_scale)")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--tracker-config", default="configs/tracker.yaml")
    parser.add_argument("--seg-config", default="configs/segmentation.yaml")
    parser.add_argument("--joint-config", default="configs/joint.yaml")
    parser.add_argument("--use-dino-features", action="store_true")
    parser.add_argument("--backend-repo", default=None, help="Path to external joint clue repo")
    parser.add_argument("--backend-callable", default=None, help="module:function callable")
    parser.add_argument("--cotracker-repo", default=".third_party/repos/co-tracker", help="Added to PYTHONPATH for backend=cotracker")
    parser.add_argument("--out-dir", default="outputs/multiview_pipeline")
    parser.add_argument("--fusion-temperature", type=float, default=0.02)
    parser.add_argument("--fusion-angle-deg", type=float, default=12.0)
    parser.add_argument("--fusion-point-thresh", type=float, default=0.08)
    parser.add_argument("--refine-global", dest="refine_global", action="store_true", default=True)
    parser.add_argument("--no-refine-global", dest="refine_global", action="store_false")
    parser.add_argument("--progress", dest="progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--loss-debug", action="store_true")
    args = parser.parse_args()

    configure_logging()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _ensure_matplotlib_cache(out_dir)

    seg_cfg = load_yaml_config(args.seg_config)
    joint_cfg = load_yaml_config(args.joint_config)
    tracker_cfg = load_yaml_config(args.tracker_config)
    if tracker_cfg.get("backend", "precomputed") == "precomputed" and not tracker_cfg.get("tracks_npz"):
        tracker_cfg = dict(tracker_cfg)
        tracker_cfg["backend"] = "cotracker"
        tracker_cfg.setdefault("device", "cpu")
    if tracker_cfg.get("backend") == "cotracker":
        _maybe_add_repo_path(args.cotracker_repo)

    rgb_root = input_dir / args.rgb_root
    if not rgb_root.is_dir():
        raise FileNotFoundError(f"RGB root not found: {rgb_root}")

    cam_entries = _load_camera_entries(input_dir / args.cameras_json)
    cameras = sorted([p.name for p in rgb_root.glob(args.camera_pattern) if p.is_dir()])
    if not cameras:
        raise ValueError(f"No camera folders matching pattern '{args.camera_pattern}' under {rgb_root}")

    ext_cfg = dict(joint_cfg.get("external", {}).get("joint_clue", {}))
    if args.backend_repo is not None:
        ext_cfg["repo_path"] = args.backend_repo
    if args.backend_callable is not None:
        ext_cfg["backend_callable"] = args.backend_callable
    estimator = JointClueEstimator.from_config(ext_cfg)

    view_runs: list[ViewRun] = []
    cams_iter = tqdm(cameras, desc="Multiview/Cameras", disable=not args.progress, leave=False)
    for camera in cams_iter:
        if camera not in cam_entries:
            raise KeyError(f"Camera {camera} missing from metadata file {args.cameras_json}")
        c_entry = cam_entries[camera]
        world_from_camera_np = _world_from_camera_from_entry(c_entry, args.extrinsics_convention)
        world_from_camera = torch.from_numpy(world_from_camera_np).float()

        seq = _load_sequence_for_camera(
            input_dir=input_dir,
            camera=camera,
            camera_entry=c_entry,
            rgb_root=args.rgb_root,
            depth_root=args.depth_root,
            mask_root=args.mask_root,
            depth_scale=float(args.depth_scale),
            depth_npy_scale=float(args.depth_npy_scale),
            max_frames=args.max_frames,
        )

        view_dir = out_dir / "views" / camera
        view_dir.mkdir(parents=True, exist_ok=True)
        stage_pbar = tqdm(total=5, desc=f"{camera}/Pipeline", disable=not args.progress, leave=False)

        tracks = run_tracking_and_lift(
            sequence=seq,
            tracker_cfg=tracker_cfg,
            min_valid_ratio=float(seg_cfg.get("tracks", {}).get("min_valid_ratio", 0.7)),
            max_depth=seg_cfg.get("tracks", {}).get("max_depth", None),
        )
        stage_pbar.update(1)

        if args.use_dino_features or tracks.feature.shape[1] <= 1:
            model_name = seg_cfg.get("features", {}).get("model_name", "vit_small_patch14_dinov2")
            extractor = DinoBackendAdapter(model_name=model_name, device="cpu").build()
            feat_map = extractor.extract_dense(seq.rgb[tracks.anchor_frame])
            feat = extractor.sample_points(feat_map, tracks.xy[:, tracks.anchor_frame, :])
            tracks = TrackBatch(
                xy=tracks.xy,
                xyz=tracks.xyz,
                valid=tracks.valid,
                anchor_frame=tracks.anchor_frame,
                point_ids=tracks.point_ids,
                feature=feat,
                confidence=tracks.confidence,
            )
        stage_pbar.update(1)

        graph = build_feature_graph(
            tracks.feature,
            xy=tracks.xy[:, tracks.anchor_frame, :],
            num_neighbors=int(seg_cfg.get("features", {}).get("num_neighbors", 16)),
            spatial_gate_px=seg_cfg.get("features", {}).get("spatial_gate_px", None),
        )
        init_logits = initialize_two_part_logits(tracks.feature)
        seg = run_stage1_segmentation(
            tracks=tracks,
            graph=graph,
            cfg=seg_cfg,
            init_logits=init_logits,
            image_size=(seq.H, seq.W),
            show_progress=args.progress,
            debug_losses=args.loss_debug,
        )
        stage_pbar.update(1)

        joint = run_stage2_joint(
            tracks=tracks,
            seg=seg,
            cfg=joint_cfg,
            clue_estimator=estimator,
            show_progress=args.progress,
            debug_losses=args.loss_debug,
        )
        stage_pbar.update(1)

        save_tracks_npz(tracks, view_dir / "tracks.npz")
        _save_segmentation(view_dir / "segmentation.pt", seg)
        torch.save(joint_result_to_dict(joint), view_dir / "joint.pt")
        stage_pbar.update(1)
        if args.progress:
            stage_pbar.close()

        vw = _view_weight(tracks, seg.diagnostics)
        view_runs.append(
            ViewRun(
                camera=camera,
                sequence=seq,
                tracks=tracks,
                seg=seg,
                joint=joint,
                world_from_camera=world_from_camera,
                view_weight=vw,
            )
        )
    if args.progress:
        cams_iter.close()

    per_view_losses: list[dict[str, float]] = []
    per_view_weights: list[float] = []
    per_view_summary: dict[str, Any] = {}
    losses_by_camera: dict[str, dict[str, float]] = {}
    for v in view_runs:
        c_map = _joint_candidate_map(v.joint)
        losses = {k: float(c_map[k].loss) for k in ("revolute", "prismatic", "screw") if k in c_map}
        per_view_losses.append(losses)
        per_view_weights.append(float(v.view_weight))
        losses_by_camera[v.camera] = losses
        per_view_summary[v.camera] = {
            "num_points": int(v.tracks.P),
            "view_weight": float(v.view_weight),
            "best_model": str(v.joint.best_model),
            "candidate_losses": losses,
            "nonfinite_steps": v.seg.diagnostics.get("nonfinite_steps", []),
        }

    priors = aggregate_model_priors(
        per_view_losses=per_view_losses,
        view_weights=per_view_weights,
        temperature=float(args.fusion_temperature),
    )
    fused_model = max(priors.items(), key=lambda kv: kv[1])[0]

    dirs_world: list[torch.Tensor] = []
    point_dirs_world: list[torch.Tensor] = []
    points_world: list[torch.Tensor] = []
    point_weights: list[float] = []
    states: list[torch.Tensor] = []
    pitches: list[float | None] = []
    cand_weights: list[float] = []
    cand_losses: list[float] = []
    for v in view_runs:
        c_map = _joint_candidate_map(v.joint)
        cand = c_map.get(fused_model, None)
        if cand is None:
            continue
        dir_world = _to_world_dirs(cand.axis_dir.view(1, 3), v.world_from_camera)[0]
        dirs_world.append(dir_world)
        if cand.axis_point is not None:
            pt_world = _to_world_points(cand.axis_point.view(1, 3), v.world_from_camera)[0]
            points_world.append(pt_world)
            point_dirs_world.append(dir_world)
        states.append(cand.state.detach().float())
        pitches.append(None if cand.pitch is None else float(cand.pitch.detach().cpu()))
        view_losses = losses_by_camera.get(v.camera, {})
        if view_losses:
            l_min = min(view_losses.values())
            conf = math.exp(-(float(cand.loss) - l_min) / max(float(args.fusion_temperature), 1e-6))
        else:
            conf = 1.0
        fused_w = float(v.view_weight) * float(conf)
        cand_weights.append(fused_w)
        if cand.axis_point is not None:
            point_weights.append(fused_w)
        cand_losses.append(float(cand.loss))

    if not dirs_world:
        raise RuntimeError("No candidate directions available for fusion")

    dirs_t = torch.stack(dirs_world, dim=0).float()
    w_t = torch.tensor(cand_weights, dtype=torch.float32)
    fused_dir, inlier_mask, _ = robust_direction_consensus(
        dirs_t,
        weights=w_t,
        max_angle_deg=float(args.fusion_angle_deg),
    )

    fused_point = None
    axis_point_inlier_count = 0
    if fused_model in {"revolute", "screw"} and points_world:
        pts_t = torch.stack(points_world, dim=0).float()
        dirs_for_points = torch.stack(point_dirs_world, dim=0).float()
        weights_for_points = torch.tensor(point_weights, dtype=torch.float32)
        fused_point, point_inliers = robust_axis_point_consensus(
            pts_t,
            dirs_for_points,
            weights=weights_for_points,
            max_dist=float(args.fusion_point_thresh),
        )
        axis_point_inlier_count = int(point_inliers.sum().item())

    fused_state = fuse_signed_states(states=states, dirs=dirs_t, fused_dir=fused_dir, weights=w_t)
    fused_pitch = fuse_signed_pitches(pitches=pitches, dirs=dirs_t, fused_dir=fused_dir, weights=w_t)
    fused_loss = float(np.average(np.array(cand_losses, dtype=np.float64), weights=np.array(cand_weights, dtype=np.float64)))

    refined = None
    if args.refine_global:
        rel_world_list = [_build_world_relative_motion(v) for v in view_runs]
        t_min = min(int(r.moving_points_rel.shape[1]) for r in rel_world_list)
        cat_can = torch.cat([r.canonical_points for r in rel_world_list], dim=0)
        cat_mov = torch.cat([r.moving_points_rel[:, :t_min, :] for r in rel_world_list], dim=0)
        cat_valid = torch.cat([r.valid[:, :t_min] for r in rel_world_list], dim=0)
        cat_w = torch.cat([r.weights for r in rel_world_list], dim=0)
        rel_world = RelativeMotionResult(
            reference_part=-1,
            moving_part=-1,
            canonical_points=cat_can,
            moving_points_rel=cat_mov,
            valid=cat_valid,
            weights=cat_w,
            ref_transform_inv=torch.eye(4).unsqueeze(0).repeat(t_min, 1, 1),
            diagnostics={"num_views": len(rel_world_list), "num_points": int(cat_can.shape[0])},
        )
        seed = ConsensusResult(
            type_priors={k: float(v) for k, v in priors.items()},
            axis_dir=fused_dir.detach().clone(),
            axis_point=None if fused_point is None else fused_point.detach().clone(),
            pitch=None if fused_pitch is None else float(fused_pitch),
        )
        ref_cfg = dict(joint_cfg)
        m_cfg = dict(ref_cfg.get("models", {}))
        m_cfg["candidates"] = [fused_model]
        ref_cfg["models"] = m_cfg
        cand = fit_candidate_model(
            model_name=fused_model,
            rel=rel_world,
            consensus=seed,
            cfg=ref_cfg,
            show_progress=args.progress,
            debug_losses=args.loss_debug,
        )
        refined = cand
        fused_model = cand.model_name
        fused_dir = cand.axis_dir.detach().clone()
        fused_point = None if cand.axis_point is None else cand.axis_point.detach().clone()
        fused_pitch = None if cand.pitch is None else float(cand.pitch.detach().cpu())
        fused_state = cand.state.detach().clone()
        fused_loss = float(cand.loss)

    fused_payload = {
        "best_model": fused_model,
        "type_priors": {k: float(v) for k, v in priors.items()},
        "axis_dir_world": fused_dir.detach().cpu().tolist(),
        "axis_point_world": None if fused_point is None else fused_point.detach().cpu().tolist(),
        "pitch": None if fused_pitch is None else float(fused_pitch),
        "state": None if fused_state is None else fused_state.detach().cpu().tolist(),
        "loss": float(fused_loss),
        "diagnostics": {
            "num_views": len(view_runs),
            "fusion_model": fused_model,
            "direction_inliers": int(inlier_mask.sum().item()),
            "direction_total": int(inlier_mask.numel()),
            "axis_point_inliers": int(axis_point_inlier_count),
            "global_refined": bool(refined is not None),
        },
        "per_view": per_view_summary,
    }
    if refined is not None:
        fused_payload["refined_candidate"] = {
            "model_name": refined.model_name,
            "loss": float(refined.loss),
            "diagnostics": _jsonify(refined.diagnostics),
        }

    torch.save(fused_payload, out_dir / "joint_fused_world.pt")
    (out_dir / "joint_fused_world.json").write_text(json.dumps(_jsonify(fused_payload), indent=2))
    (out_dir / "multiview_run_meta.json").write_text(
        json.dumps(
            {
                "input_dir": str(input_dir.resolve()),
                "rgb_root": args.rgb_root,
                "depth_root": args.depth_root,
                "mask_root": args.mask_root,
                "cameras_json": args.cameras_json,
                "cameras": [v.camera for v in view_runs],
                "num_views": len(view_runs),
                "frames_per_view": {v.camera: int(v.sequence.T) for v in view_runs},
                "fused_output": str((out_dir / "joint_fused_world.pt").resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
