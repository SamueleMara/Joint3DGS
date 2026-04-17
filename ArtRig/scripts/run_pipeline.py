#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from articulation.data import (
    MultiViewRGBDSequence,
    RGBDSequence,
    TrackBatch,
    load_rgbd_sequence_npz,
    load_tracks_npz,
    save_rgbd_sequence_npz,
    save_tracks_npz,
)
from articulation.external.dino_backend_adapter import DinoBackendAdapter
from articulation.external.joint_clue_adapter import JointClueEstimator
from articulation.features import build_feature_graph, initialize_two_part_logits
from articulation.joint.outputs import joint_result_to_dict
from articulation.joint.relative_motion import choose_reference_part, compute_relative_motion
from articulation.pipeline.stage0_matching import run_stage0_matching
from articulation.pipeline.stage1_segmentation import run_stage1_segmentation
from articulation.pipeline.stage2_joint import run_stage2_joint
from articulation.preprocess.multiview import load_multiview_sequence_from_folder, single_to_multiview_sequence
from articulation.utils import configure_logging, load_yaml_config
from articulation.utils.wandb_monitor import WandbSystemMonitor
from articulation.utils.viz import (
    save_axis_3d,
    save_cog_trajectories,
    save_model_fit_comparison,
    save_moving_points_ref,
    save_point_label_overlay,
    save_segmentation_mask_preview,
    save_state_vs_time,
)


def _ensure_matplotlib_cache(out_dir: Path) -> None:
    cache = out_dir / ".cache" / "matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))


def _init_wandb(
    args: argparse.Namespace,
    out_dir: Path,
    matching_cfg: dict,
    seg_cfg: dict,
    joint_cfg: dict,
):
    if not args.wandb:
        return None
    try:
        import wandb
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "W&B logging requested but wandb is not installed. Install with `pip install wandb`."
        ) from exc

    tags = [t.strip() for t in str(args.wandb_tags).split(",") if t.strip()]
    run_name = args.wandb_name or _build_default_wandb_name(args=args, out_dir=out_dir)
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=run_name,
        group=args.wandb_group or None,
        mode=args.wandb_mode,
        tags=tags,
        dir=str(out_dir / "wandb"),
        config={
            "matching": matching_cfg,
            "segmentation": seg_cfg,
            "joint": joint_cfg,
            "out_dir": str(out_dir),
        },
    )
    return run


def _infer_dataset_name(args: argparse.Namespace, out_dir: Path) -> str:
    if args.input_dir:
        p = Path(args.input_dir).expanduser()
        name = p.name
        # If user points to an internal subfolder (rgb/images/depth/mask), use parent folder as dataset id.
        if name.lower() in {"rgb", "images", "depth", "depth_npy", "mask", "fg_mask"} and p.parent.name:
            name = p.parent.name
        return name or "dataset"
    if args.sequence_npz:
        return Path(args.sequence_npz).stem or "sequence"
    if args.tracks_npz:
        return Path(args.tracks_npz).stem or "tracks"
    return out_dir.name or "run"


def _build_default_wandb_name(args: argparse.Namespace, out_dir: Path) -> str:
    dataset_name = _infer_dataset_name(args=args, out_dir=out_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return f"{dataset_name}_{timestamp}"


def _log_stage0_to_wandb(wandb_run: object | None, stage0_diag: dict[str, object], prefix: str = "stage0") -> None:
    if wandb_run is None:
        return
    payload: dict[str, float] = {}
    for key, value in stage0_diag.items():
        if isinstance(value, bool):
            payload[f"{prefix}/{key}"] = float(1.0 if value else 0.0)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            payload[f"{prefix}/{key}"] = float(value)
    if payload:
        wandb_run.log(payload)


def _get_anchor_image(mv_seq: MultiViewRGBDSequence | None, anchor_frame: int) -> torch.Tensor | None:
    if mv_seq is None:
        return None
    if mv_seq.T == 0:
        return None
    t = int(max(0, min(anchor_frame, mv_seq.T - 1)))
    return mv_seq.rgb[t, 0]


def _compute_features_if_missing(
    tracks: TrackBatch,
    mv_seq: MultiViewRGBDSequence | None,
    seg_cfg: dict,
) -> TrackBatch:
    if tracks.feature.shape[1] > 1:
        return tracks
    if mv_seq is None:
        return tracks

    img = _get_anchor_image(mv_seq, tracks.anchor_frame)
    if img is None:
        return tracks

    model_name = seg_cfg.get("features", {}).get("model_name", "vit_small_patch14_dinov2")
    extractor = DinoBackendAdapter(model_name=model_name, device="cpu").build()
    feat_map = extractor.extract_dense(img)
    feat = extractor.sample_points(feat_map, tracks.xy[:, tracks.anchor_frame, :])

    return TrackBatch(
        xy=tracks.xy,
        xyz=tracks.xyz,
        valid=tracks.valid,
        anchor_frame=tracks.anchor_frame,
        point_ids=tracks.point_ids,
        feature=feat,
        confidence=tracks.confidence,
        obs_count=tracks.obs_count,
        multiview_error=tracks.multiview_error,
        meta=dict(tracks.meta),
    )


def _clamp_idx(idx: int, size: int) -> int:
    if size <= 0:
        return 0
    return int(max(0, min(int(idx), size - 1)))


def _select_view_sequence(mv_seq: MultiViewRGBDSequence, view_idx: int) -> tuple[RGBDSequence, int]:
    if mv_seq.V <= 0 or mv_seq.T <= 0:
        raise ValueError("Cannot extract a view from empty multi-view sequence")

    v = _clamp_idx(view_idx, mv_seq.V)
    if mv_seq.K.ndim == 3:
        k = mv_seq.K[v]
    else:
        k = mv_seq.K[:, v]

    seq = RGBDSequence(
        rgb=mv_seq.rgb[:, v],
        depth=mv_seq.depth[:, v],
        fg_mask=mv_seq.fg_mask[:, v],
        K=k,
        frame_ids=list(mv_seq.frame_ids),
        meta={**mv_seq.meta, "selected_view_idx": int(v)},
    )
    return seq, v


def _pca_rgb(feat_chw: torch.Tensor) -> np.ndarray:
    c, h, w = feat_chw.shape
    x = feat_chw.permute(1, 2, 0).reshape(-1, c).float()
    x = x - x.mean(dim=0, keepdim=True)
    _, _, v = torch.pca_lowrank(x, q=min(3, c))
    x3 = x @ v[:, : min(3, c)]
    if x3.shape[1] < 3:
        pad = torch.zeros((x3.shape[0], 3 - x3.shape[1]), dtype=x3.dtype, device=x3.device)
        x3 = torch.cat([x3, pad], dim=1)

    lo = torch.quantile(x3, 0.01, dim=0, keepdim=True)
    hi = torch.quantile(x3, 0.99, dim=0, keepdim=True)
    x3 = (x3 - lo) / (hi - lo + 1e-8)
    x3 = x3.clamp(0.0, 1.0)
    return x3.reshape(h, w, 3).cpu().numpy()


def _norm_map(feat_chw: torch.Tensor) -> np.ndarray:
    n = torch.linalg.norm(feat_chw.float(), dim=0)
    n = (n - n.min()) / (n.max() - n.min() + 1e-8)
    return n.cpu().numpy()


def _save_dino_feature_maps(
    image_chw: torch.Tensor,
    model_name: str,
    device: str,
    out_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    extractor = DinoBackendAdapter(model_name=model_name, device=device).build()
    h_in, w_in = int(image_chw.shape[-2]), int(image_chw.shape[-1])
    h_model = max(14, (h_in // 14) * 14)
    w_model = max(14, (w_in // 14) * 14)
    if (h_model, w_model) != (h_in, w_in):
        image_for_model = F.interpolate(
            image_chw.unsqueeze(0),
            size=(h_model, w_model),
            mode="bilinear",
            align_corners=False,
        )[0]
    else:
        image_for_model = image_chw

    with torch.inference_mode():
        fmap = extractor.extract_dense(image_for_model)  # [C,h,w]
    pca = _pca_rgb(fmap)
    norm = _norm_map(fmap)

    pca_up = F.interpolate(
        torch.from_numpy(pca).permute(2, 0, 1).unsqueeze(0),
        size=(h_in, w_in),
        mode="bilinear",
        align_corners=False,
    )[0].permute(1, 2, 0).numpy()
    norm_up = F.interpolate(
        torch.from_numpy(norm).unsqueeze(0).unsqueeze(0),
        size=(h_in, w_in),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()

    img_np = image_chw.permute(1, 2, 0).detach().cpu().numpy()
    plt.imsave(out_dir / "input_rgb.png", np.clip(img_np, 0.0, 1.0))
    plt.imsave(out_dir / "dino_pca.png", np.clip(pca_up, 0.0, 1.0))
    plt.imsave(out_dir / "dino_norm.png", np.clip(norm_up, 0.0, 1.0), cmap="viridis")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(np.clip(img_np, 0.0, 1.0))
    axes[0].set_title("Input RGB")
    axes[0].axis("off")
    axes[1].imshow(np.clip(pca_up, 0.0, 1.0))
    axes[1].set_title("DINO features (PCA)")
    axes[1].axis("off")
    im = axes[2].imshow(np.clip(norm_up, 0.0, 1.0), cmap="viridis")
    axes[2].set_title("DINO feature norm")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "dino_feature_map_overview.png", dpi=160)
    plt.close(fig)
    del extractor


def _run_basic_viz(
    tracks: TrackBatch,
    seg,
    joint,
    out_dir: Path,
    frame_idx: int,
    prefix: str = "",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _ensure_matplotlib_cache(out_dir)

    save_point_label_overlay(tracks, seg, out_dir / f"{prefix}points_overlay.png", frame_idx=frame_idx)
    save_segmentation_mask_preview(seg.masks_per_frame, out_dir / f"{prefix}masks_preview.png", frame_idx=frame_idx)
    save_cog_trajectories(tracks, seg, out_dir / f"{prefix}cog_trajectories.png")
    ref_part, mov_part = choose_reference_part(tracks, seg)
    labels = seg.point_labels.long()
    rel = compute_relative_motion(
        xyz_part_ref=tracks.xyz[labels == ref_part],
        xyz_part_mov=tracks.xyz[labels == mov_part],
        valid_ref=tracks.valid[labels == ref_part],
        valid_mov=tracks.valid[labels == mov_part],
        reference_part=ref_part,
        moving_part=mov_part,
    )
    save_moving_points_ref(rel, out_dir / f"{prefix}moving_ref.png")
    save_axis_3d(rel.canonical_points, joint.axis_dir, joint.axis_point, out_dir / f"{prefix}axis.png")
    save_model_fit_comparison(joint, out_dir / f"{prefix}model_fit.png")
    save_state_vs_time(joint.state, out_dir / f"{prefix}state.png")


def _save_dense_3d_viz(
    sequence: RGBDSequence,
    seg,
    joint_payload: dict,
    tracks_path: Path,
    out_dir: Path,
    frame_stride: int,
    sample_ratio: float,
    conf_thresh: float,
    min_depth: float,
    max_depth: float,
    max_points: int,
    frame_max_points: int,
    static_label: str,
    export_frame_clouds: bool,
    mask_alpha: float,
    show: bool,
    show_progress: bool = False,
) -> dict[str, object]:
    from scripts import visualize_segmentation_3d as viz3d

    out_dir.mkdir(parents=True, exist_ok=True)
    masks = seg.masks_per_frame.detach().cpu().float()
    masks = viz3d._ensure_mask_shape(masks, sequence.depth.shape[-2], sequence.depth.shape[-1])

    rgb_u8 = viz3d._rgb_to_uint8(sequence.rgb)
    labels = viz3d._save_mask_overlays(
        rgb_u8,
        masks,
        out_dir,
        alpha=float(mask_alpha),
        show_progress=show_progress,
    )

    if static_label == "auto":
        static_part, dynamic_part = viz3d._infer_static_dynamic_parts(tracks_path, seg)
    else:
        static_part = int(static_label)
        dynamic_part = 1 - static_part

    clouds = viz3d._build_dense_segmented_clouds_over_frames(
        seq_depth=sequence.depth,
        seq_k=sequence.K,
        labels=labels,
        masks=masks,
        frame_stride=int(max(1, frame_stride)),
        sample_ratio=float(sample_ratio),
        conf_thresh=float(conf_thresh),
        min_depth=float(min_depth),
        max_depth=float(max_depth),
        max_points=int(max_points),
        static_label=int(static_part),
        export_frame_clouds=bool(export_frame_clouds),
        frame_clouds_dir=out_dir / "frame_clouds",
        frame_max_points=int(frame_max_points),
        show_progress=show_progress,
    )
    points = clouds["points_all"]
    colors = clouds["colors_all"]
    if points.shape[0] == 0:
        return {
            "num_points": 0,
            "overlay_dir": str(out_dir / "mask_overlay"),
            "warning": "No valid 3D points. Lower --viz-3d-conf-thresh or raise --viz-3d-sample-ratio.",
            "static_label": int(static_part),
            "dynamic_label": int(dynamic_part),
        }

    import open3d as o3d

    pcd_path = out_dir / "segmented_pointcloud.ply"
    pcd_static_path = out_dir / "static_pointcloud.ply"
    pcd_dynamic_path = out_dir / "dynamic_pointcloud.ply"
    viz3d._write_point_cloud(pcd_path, points, colors)
    viz3d._write_point_cloud(pcd_static_path, clouds["points_static"], clouds["colors_static"])
    viz3d._write_point_cloud(pcd_dynamic_path, clouds["points_dynamic"], clouds["colors_dynamic"])
    pcd = o3d.io.read_point_cloud(str(pcd_path))

    axis_dir = np.asarray(joint_payload["axis_dir"], dtype=np.float64)
    axis_point = (
        np.asarray(joint_payload["axis_point"], dtype=np.float64)
        if joint_payload.get("axis_point", None) is not None
        else points.mean(axis=0).astype(np.float64)
    )
    axis_len = float(np.linalg.norm(np.max(points, axis=0) - np.min(points, axis=0)) * 0.25 + 1e-6)
    axis_line = viz3d._make_axis_lineset(axis_dir=axis_dir, axis_point=axis_point, length=axis_len)
    axis_path = out_dir / "joint_axis.ply"
    o3d.io.write_line_set(str(axis_path), axis_line)
    frame_stats_path = out_dir / "frame_cloud_stats.json"
    frame_stats_path.write_text(json.dumps(clouds.get("frame_stats", []), indent=2))

    viz3d._save_sparse_joint_view(
        tracks_npz=tracks_path,
        seg=seg,
        joint_payload=joint_payload,
        out_dir=out_dir,
    )

    if show:
        o3d.visualization.draw_geometries([pcd, axis_line], window_name="ArtRig Segmented Cloud + Joint Axis")

    return {
        "pointcloud_path": str(pcd_path),
        "static_pointcloud_path": str(pcd_static_path),
        "dynamic_pointcloud_path": str(pcd_dynamic_path),
        "axis_path": str(axis_path),
        "num_points": int(points.shape[0]),
        "num_points_static": int(clouds["points_static"].shape[0]),
        "num_points_dynamic": int(clouds["points_dynamic"].shape[0]),
        "static_label": int(static_part),
        "dynamic_label": int(dynamic_part),
        "overlay_dir": str(out_dir / "mask_overlay"),
        "frame_clouds_dir": str(out_dir / "frame_clouds") if export_frame_clouds else None,
        "frame_cloud_stats": str(frame_stats_path),
    }


def _run_viz_all_bundle(
    out_dir: Path,
    tracks: TrackBatch,
    seg,
    joint,
    joint_payload: dict,
    sequence: RGBDSequence | None,
    selected_view_idx: int | None,
    seg_cfg: dict,
    args: argparse.Namespace,
) -> dict[str, object]:
    bundle_dir = Path(args.viz_all_dir) if args.viz_all_dir is not None else out_dir / "viz_all"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    _ensure_matplotlib_cache(bundle_dir)

    num_steps = 4
    viz_pbar = tqdm(
        total=num_steps,
        desc="Pipeline/VizAll",
        disable=not args.progress,
        leave=False,
    )

    outputs: dict[str, object] = {
        "bundle_dir": str(bundle_dir),
    }

    frame_idx = _clamp_idx(args.viz_frame, tracks.T)
    summary_dir = bundle_dir / "summary"
    _run_basic_viz(tracks, seg, joint, summary_dir, frame_idx=frame_idx)
    outputs["summary_dir"] = str(summary_dir)
    viz_pbar.update(1)

    if sequence is None:
        outputs["warning"] = "No sequence was loaded in this run, skipped DINO and dense 3D visualizations."
        viz_pbar.close()
        return outputs

    selected_view = 0 if selected_view_idx is None else int(selected_view_idx)
    outputs["selected_view_idx"] = selected_view

    if args.viz_export_sequence_npz:
        seq_npz_path = bundle_dir / f"sequence_view_{selected_view:03d}.npz"
        save_rgbd_sequence_npz(sequence, seq_npz_path)
        outputs["sequence_npz"] = str(seq_npz_path)
    viz_pbar.update(1)

    if args.viz_all_no_dino:
        outputs["dino_skipped"] = True
        viz_pbar.update(1)
    else:
        dino_dir = bundle_dir / "dino"
        dino_model_name = (
            args.viz_dino_model
            if args.viz_dino_model is not None
            else str(seg_cfg.get("features", {}).get("model_name", "vit_small_patch14_dinov2"))
        )
        dino_frame = _clamp_idx(args.viz_dino_frame if args.viz_dino_frame is not None else frame_idx, sequence.T)
        _save_dino_feature_maps(
            image_chw=sequence.rgb[dino_frame],
            model_name=dino_model_name,
            device=args.viz_dino_device,
            out_dir=dino_dir,
        )
        outputs["dino_dir"] = str(dino_dir)
        outputs["dino_frame"] = int(dino_frame)
        viz_pbar.update(1)

    dense3d_dir = bundle_dir / "seg3d"
    dense3d = _save_dense_3d_viz(
        sequence=sequence,
        seg=seg,
        joint_payload=joint_payload,
        tracks_path=out_dir / "tracks.npz",
        out_dir=dense3d_dir,
        frame_stride=args.viz_3d_frame_stride,
        sample_ratio=args.viz_3d_sample_ratio,
        conf_thresh=args.viz_3d_conf_thresh,
        min_depth=args.viz_3d_min_depth,
        max_depth=args.viz_3d_max_depth,
        max_points=args.viz_3d_max_points,
        frame_max_points=args.viz_3d_frame_max_points,
        static_label=args.viz_3d_static_label,
        export_frame_clouds=args.viz_3d_export_frame_clouds,
        mask_alpha=args.viz_mask_alpha,
        show=args.viz_show_3d,
        show_progress=args.progress,
    )
    outputs["seg3d"] = dense3d
    viz_pbar.update(1)
    viz_pbar.close()
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks-npz", default=None, help="Skip Stage0 and load tracks directly")
    parser.add_argument("--input-dir", default=None, help="Dataset root for Stage0 matching")
    parser.add_argument("--sequence-npz", default=None, help="Single-view RGBD npz for Stage0 matching")

    parser.add_argument("--matching-config", default="configs/matching.yaml")
    parser.add_argument("--seg-config", default="configs/segmentation.yaml")
    parser.add_argument("--joint-config", default="configs/joint.yaml")

    parser.add_argument("--rgb-dir", default=None)
    parser.add_argument("--depth-dir", default=None)
    parser.add_argument("--mask-dir", default=None)
    parser.add_argument("--intrinsics-file", default=None)
    parser.add_argument("--extrinsics-file", default=None)
    parser.add_argument("--cameras-json", default=None)
    parser.add_argument("--extrinsics-convention", choices=["world_from_camera", "camera_from_world"], default="world_from_camera")
    parser.add_argument("--depth-scale", type=float, default=None)
    parser.add_argument("--depth-npy-scale", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=None)

    # Legacy args kept for CLI compatibility.
    parser.add_argument("--tracker-config", default=None)
    parser.add_argument("--depth-config", default=None)
    parser.add_argument("--fg-mask-dir", default=None)
    parser.add_argument("--fg-mask-threshold", type=float, default=0.5)

    parser.add_argument("--backend-repo", default=None, help="Path to external joint clue repo")
    parser.add_argument("--backend-callable", default=None, help="module:function callable")

    parser.add_argument("--out-dir", default="outputs/pipeline")
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--image-width", type=int, default=None)

    parser.add_argument("--progress", dest="progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--loss-debug", action="store_true")
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--viz-frame", type=int, default=0)
    parser.add_argument("--viz-all", action="store_true", help="Run full visualization bundle (summary + DINO maps + dense 3D point cloud)")
    parser.add_argument("--viz-all-dir", default=None, help="Output directory for --viz-all (default: <out-dir>/viz_all)")
    parser.add_argument("--viz-view-idx", type=int, default=0, help="Camera/view index used by --viz-all for image/depth-based visualizations")
    parser.add_argument("--viz-dino-frame", type=int, default=None, help="Frame index for DINO visualization (default: --viz-frame)")
    parser.add_argument("--viz-dino-device", default="cpu", help="Device for DINO visualization model inference")
    parser.add_argument("--viz-dino-model", default=None, help="Override DINO model name for visualization")
    parser.add_argument("--viz-all-no-dino", action="store_true", help="Skip DINO feature-map export inside --viz-all")
    parser.add_argument("--viz-export-sequence-npz", action="store_true", help="Export selected RGB-D view as NPZ under --viz-all output")
    parser.add_argument("--viz-3d-frame-stride", type=int, default=2)
    parser.add_argument("--viz-3d-sample-ratio", type=float, default=0.1)
    parser.add_argument("--viz-3d-conf-thresh", type=float, default=0.5)
    parser.add_argument("--viz-3d-min-depth", type=float, default=1e-4)
    parser.add_argument("--viz-3d-max-depth", type=float, default=0.0, help="<=0 disables max-depth clipping")
    parser.add_argument("--viz-3d-max-points", type=int, default=350000)
    parser.add_argument("--viz-3d-frame-max-points", type=int, default=25000)
    parser.add_argument("--viz-3d-static-label", choices=["auto", "0", "1"], default="auto")
    parser.add_argument("--viz-3d-export-frame-clouds", dest="viz_3d_export_frame_clouds", action="store_true", default=True)
    parser.add_argument("--viz-3d-no-frame-clouds", dest="viz_3d_export_frame_clouds", action="store_false")
    parser.add_argument("--viz-mask-alpha", type=float, default=0.45)
    parser.add_argument("--viz-show-3d", action="store_true", help="Open Open3D viewer after saving dense 3D visualization")
    parser.add_argument("--wandb", dest="wandb", action="store_true", default=True, help="Enable Weights & Biases logging (default: enabled)")
    parser.add_argument("--no-wandb", dest="wandb", action="store_false", help="Disable Weights & Biases logging")
    parser.add_argument("--wandb-project", default="ArtRig")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-tags", default="")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument(
        "--wandb-system-monitor",
        dest="wandb_system_monitor",
        action="store_true",
        default=True,
        help="Enable periodic CPU/RAM/GPU/power telemetry logging to W&B (default: enabled)",
    )
    parser.add_argument(
        "--no-wandb-system-monitor",
        dest="wandb_system_monitor",
        action="store_false",
        help="Disable periodic system telemetry logging to W&B",
    )
    parser.add_argument(
        "--wandb-system-interval",
        type=float,
        default=2.0,
        help="Seconds between periodic W&B system telemetry logs",
    )
    args = parser.parse_args()

    configure_logging()

    seg_cfg = load_yaml_config(args.seg_config)
    joint_cfg = load_yaml_config(args.joint_config)
    matching_cfg = load_yaml_config(args.matching_config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = _init_wandb(
        args=args,
        out_dir=out_dir,
        matching_cfg=matching_cfg,
        seg_cfg=seg_cfg,
        joint_cfg=joint_cfg,
    )
    wandb_monitor: WandbSystemMonitor | None = None
    if (
        wandb_run is not None
        and args.wandb_system_monitor
        and str(args.wandb_mode).lower() != "disabled"
    ):
        wandb_monitor = WandbSystemMonitor(
            wandb_run=wandb_run,
            interval_sec=float(max(0.5, args.wandb_system_interval)),
            prefix="system",
        )
        wandb_monitor.start()
    if wandb_run is not None and getattr(wandb_run, "url", None):
        msg = f"[W&B] Run URL: {wandb_run.url}"
        if args.progress:
            tqdm.write(msg)
        else:
            print(msg)

    stage0_diag: dict[str, object] = {}
    mv_seq: MultiViewRGBDSequence | None = None

    if args.tracks_npz:
        tracks = load_tracks_npz(args.tracks_npz)
        if args.sequence_npz is not None:
            seq = load_rgbd_sequence_npz(args.sequence_npz)
            mv_seq = single_to_multiview_sequence(seq)
        elif args.input_dir is not None and args.viz_all:
            io_cfg = dict(matching_cfg.get("io", {}))
            rgb_dir = args.rgb_dir or io_cfg.get("rgb_dir", "images")
            depth_dir = args.depth_dir or io_cfg.get("depth_dir", "depth")
            mask_dir = args.mask_dir or io_cfg.get("mask_dir", "fg_mask")
            intrinsics_file = args.intrinsics_file or io_cfg.get("intrinsics_file", "intrinsics.npy")
            extrinsics_file = args.extrinsics_file or io_cfg.get("extrinsics_file", "extrinsics.npy")
            cameras_json = args.cameras_json or io_cfg.get("cameras_json", "metadata/cameras.json")
            depth_scale = float(args.depth_scale if args.depth_scale is not None else io_cfg.get("depth_scale", 1.0))
            depth_npy_scale = float(args.depth_npy_scale if args.depth_npy_scale is not None else io_cfg.get("depth_npy_scale", 1.0))
            mv_seq = load_multiview_sequence_from_folder(
                root=args.input_dir,
                rgb_dir=rgb_dir,
                depth_dir=depth_dir,
                mask_dir=mask_dir,
                intrinsics_file=intrinsics_file,
                extrinsics_file=extrinsics_file,
                depth_scale=depth_scale,
                max_frames=args.max_frames,
                cameras_json=cameras_json,
                extrinsics_convention=str(args.extrinsics_convention),
                depth_npy_scale=depth_npy_scale,
            )
    else:
        if (args.input_dir is None) == (args.sequence_npz is None):
            raise ValueError("Provide exactly one of --input-dir or --sequence-npz when --tracks-npz is not set")

        if args.sequence_npz is not None:
            seq = load_rgbd_sequence_npz(args.sequence_npz)
            mv_seq = single_to_multiview_sequence(seq)
        else:
            io_cfg = dict(matching_cfg.get("io", {}))
            rgb_dir = args.rgb_dir or io_cfg.get("rgb_dir", "images")
            depth_dir = args.depth_dir or io_cfg.get("depth_dir", "depth")
            mask_dir = args.mask_dir or io_cfg.get("mask_dir", "fg_mask")
            intrinsics_file = args.intrinsics_file or io_cfg.get("intrinsics_file", "intrinsics.npy")
            extrinsics_file = args.extrinsics_file or io_cfg.get("extrinsics_file", "extrinsics.npy")
            cameras_json = args.cameras_json or io_cfg.get("cameras_json", "metadata/cameras.json")
            depth_scale = float(args.depth_scale if args.depth_scale is not None else io_cfg.get("depth_scale", 1.0))
            depth_npy_scale = float(args.depth_npy_scale if args.depth_npy_scale is not None else io_cfg.get("depth_npy_scale", 1.0))

            mv_seq = load_multiview_sequence_from_folder(
                root=args.input_dir,
                rgb_dir=rgb_dir,
                depth_dir=depth_dir,
                mask_dir=mask_dir,
                intrinsics_file=intrinsics_file,
                extrinsics_file=extrinsics_file,
                depth_scale=depth_scale,
                max_frames=args.max_frames,
                cameras_json=cameras_json,
                extrinsics_convention=str(args.extrinsics_convention),
                depth_npy_scale=depth_npy_scale,
            )

        stage0 = run_stage0_matching(
            sequence=mv_seq,
            cfg=matching_cfg,
            show_progress=args.progress,
            debug=args.loss_debug,
        )
        tracks = stage0.tracks
        stage0_diag = dict(stage0.diagnostics)
        _log_stage0_to_wandb(wandb_run=wandb_run, stage0_diag=stage0_diag, prefix="stage0")

    tracks = _compute_features_if_missing(tracks, mv_seq=mv_seq, seg_cfg=seg_cfg)

    anchor_xy = tracks.xy[:, tracks.anchor_frame, :]
    graph = build_feature_graph(
        tracks.feature,
        xy=anchor_xy,
        num_neighbors=int(seg_cfg.get("features", {}).get("num_neighbors", 16)),
        spatial_gate_px=seg_cfg.get("features", {}).get("spatial_gate_px", None),
    )
    init_logits = initialize_two_part_logits(tracks.feature)

    image_size = None
    if args.image_height is not None and args.image_width is not None:
        image_size = (args.image_height, args.image_width)
    elif mv_seq is not None:
        image_size = (mv_seq.H, mv_seq.W)

    num_steps = 3 + (1 if args.viz else 0) + (1 if args.viz_all else 0)
    pbar = tqdm(total=num_steps, desc="Pipeline", disable=not args.progress, leave=False)

    seg = run_stage1_segmentation(
        tracks=tracks,
        graph=graph,
        cfg=seg_cfg,
        init_logits=init_logits,
        image_size=image_size,
        show_progress=args.progress,
        debug_losses=args.loss_debug,
        wandb_run=wandb_run,
        wandb_prefix="seg",
    )
    pbar.update(1)

    ext_cfg = dict(joint_cfg.get("external", {}).get("joint_clue", {}))
    if args.backend_repo is not None:
        ext_cfg["repo_path"] = args.backend_repo
    if args.backend_callable is not None:
        ext_cfg["backend_callable"] = args.backend_callable

    estimator = JointClueEstimator.from_config(ext_cfg)
    joint = run_stage2_joint(
        tracks=tracks,
        seg=seg,
        cfg=joint_cfg,
        clue_estimator=estimator,
        show_progress=args.progress,
        debug_losses=args.loss_debug,
        wandb_run=wandb_run,
        wandb_prefix="joint",
    )
    pbar.update(1)
    save_tracks_npz(tracks, out_dir / "tracks.npz")

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
        out_dir / "segmentation.pt",
    )
    joint_payload = joint_result_to_dict(joint)
    torch.save(joint_payload, out_dir / "joint.pt")

    run_meta = {
        "stage0": stage0_diag,
        "tracks": {
            "num_points": int(tracks.P),
            "num_frames": int(tracks.T),
            "has_multiview_error": bool(tracks.multiview_error is not None),
        },
    }
    if wandb_run is not None:
        run_url = getattr(wandb_run, "url", None)
        run_meta["wandb"] = {
            "enabled": True,
            "project": str(args.wandb_project),
            "entity": None if args.wandb_entity is None else str(args.wandb_entity),
            "run_name": str(getattr(wandb_run, "name", "")),
            "run_id": str(getattr(wandb_run, "id", "")),
            "run_url": None if run_url in (None, "", "None") else str(run_url),
            "mode": str(args.wandb_mode),
            "system_monitor_enabled": bool(args.wandb_system_monitor),
            "system_monitor_interval_sec": float(max(0.5, args.wandb_system_interval)),
        }
    pbar.update(1)

    if args.viz:
        frame_idx = _clamp_idx(args.viz_frame, tracks.T)
        _run_basic_viz(
            tracks=tracks,
            seg=seg,
            joint=joint,
            out_dir=out_dir,
            frame_idx=frame_idx,
            prefix="viz_",
        )
        run_meta["viz"] = {
            "enabled": True,
            "dir": str(out_dir),
            "frame_idx": int(frame_idx),
        }
        pbar.update(1)

    if args.viz_all:
        viz_sequence: RGBDSequence | None = None
        viz_view_idx: int | None = None
        if mv_seq is not None:
            viz_sequence, viz_view_idx = _select_view_sequence(mv_seq, args.viz_view_idx)
            # Free full multi-view tensors before heavy visualization to lower peak RAM.
            mv_seq = None
            gc.collect()

        viz_all_outputs = _run_viz_all_bundle(
            out_dir=out_dir,
            tracks=tracks,
            seg=seg,
            joint=joint,
            joint_payload=joint_payload,
            sequence=viz_sequence,
            selected_view_idx=viz_view_idx,
            seg_cfg=seg_cfg,
            args=args,
        )
        run_meta["viz_all"] = viz_all_outputs
        pbar.update(1)

    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2))
    if wandb_run is not None:
        if wandb_monitor is not None:
            wandb_monitor.stop()
        wandb_run.log(
            {
                "run/num_tracks": float(tracks.P),
                "run/num_frames": float(tracks.T),
                "run/seg_nonfinite_steps": float(seg.diagnostics.get("nonfinite_steps", [0.0])[0]),
                "run/joint_best_model": joint.best_model,
                "run/joint_best_loss": float(min(c.loss for c in joint.candidates)),
            }
        )
        wandb_run.summary["run_meta_path"] = str(out_dir / "run_meta.json")
        wandb_run.summary["out_dir"] = str(out_dir)
        wandb_run.finish()
    pbar.close()


if __name__ == "__main__":
    main()
