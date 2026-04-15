from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import torch
from PIL import Image

from dynamic_recon.config.schema import PipelineConfig
from dynamic_recon.da3.wrapper import run_da3_on_frames
from dynamic_recon.data.cache_io import save_array, save_json
from dynamic_recon.data.video_io import get_video_metadata, read_video_frames, stable_frame_name
from dynamic_recon.geometry.dynamic_prior import build_dynamic_prior, temporal_smooth_priors
from dynamic_recon.geometry.pose_init import initialize_pose_sequence
from dynamic_recon.geometry.residuals import compute_sequence_residuals
from dynamic_recon.progress import progress_iter, stage_bar


def run_preprocess(video_path: str | Path, outdir: str | Path, cfg: PipelineConfig) -> dict[str, object]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stage = stage_bar(desc="Preprocess", total=4)
    frames = read_video_frames(
        video_path,
        resize_long_edge=cfg.video.resize_long_edge,
        stride=cfg.video.stride,
        max_frames=cfg.video.max_frames,
    )
    metadata = get_video_metadata(video_path)
    stage.update(1)
    if cfg.pipeline.save_intermediates:
        frames_dir = outdir / "frames"
        seg_frames_dir = outdir / "frames_sam"
    else:
        temp_root = Path(tempfile.mkdtemp(prefix="dynamic_recon_", dir=outdir))
        frames_dir = temp_root / "frames"
        seg_frames_dir = temp_root / "frames_sam"
    frames_dir.mkdir(parents=True, exist_ok=True)
    seg_frames_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(progress_iter(frames, desc="Writing frames", total=len(frames))):
        Image.fromarray(frame).save(frames_dir / stable_frame_name(idx))
        seg_frame = frame
        if cfg.sam3.resize_long_edge is not None:
            seg_frame = _resize_for_segmentation(frame, int(cfg.sam3.resize_long_edge))
        Image.fromarray(seg_frame).save(seg_frames_dir / stable_frame_name(idx))
    stage.update(1)
    da3_seq = run_da3_on_frames(frames, cfg.da3, source_video=str(video_path))
    da3_seq.fps = float(metadata["fps"])
    da3_seq = initialize_pose_sequence(da3_seq, cfg.pose_init)
    if cfg.pipeline.save_intermediates:
        da3_dir = outdir / "da3"
        for idx, frame_out in enumerate(progress_iter(da3_seq.frames, desc="Caching DA3 outputs", total=len(da3_seq.frames))):
            save_array(da3_dir / "depth" / f"{idx:06d}.npy", frame_out.depth.detach().cpu().numpy())
            save_array(da3_dir / "intrinsics" / f"{idx:06d}.npy", frame_out.intrinsics.detach().cpu().numpy())
            save_array(da3_dir / "extrinsics" / f"{idx:06d}.npy", frame_out.extrinsics.detach().cpu().numpy())
            if frame_out.confidence is not None:
                save_array(da3_dir / "confidence" / f"{idx:06d}.npy", frame_out.confidence.detach().cpu().numpy())
        save_json(da3_dir / "metadata.json", {"fps": da3_seq.fps, "height": da3_seq.height, "width": da3_seq.width, "source_video": str(video_path)})
    stage.update(1)
    residual_seq = compute_sequence_residuals(da3_seq, cfg.geometry)
    dynamic_priors = [build_dynamic_prior(pair, cfg.geometry) for pair in residual_seq]
    dynamic_priors = temporal_smooth_priors(dynamic_priors, cfg.geometry.temporal_smoothing_alpha)
    if cfg.pipeline.save_intermediates:
        geometry_dir = outdir / "geometry"
        for name in ("residual_3d", "residual_rel", "residual_flow", "residual_depth", "residual_feat", "residual_cycle", "visibility", "dynamic_prior"):
            (geometry_dir / name).mkdir(parents=True, exist_ok=True)
        for idx, (pair, dynamic_prior) in enumerate(zip(residual_seq, dynamic_priors)):
            save_array(geometry_dir / "residual_3d" / f"{idx:06d}.npy", pair.r_3d.detach().cpu().numpy())
            save_array(geometry_dir / "residual_rel" / f"{idx:06d}.npy", pair.r_rel.detach().cpu().numpy())
            save_array(geometry_dir / "residual_flow" / f"{idx:06d}.npy", pair.r_flow.detach().cpu().numpy())
            save_array(geometry_dir / "residual_depth" / f"{idx:06d}.npy", pair.r_depth.detach().cpu().numpy())
            save_array(geometry_dir / "residual_feat" / f"{idx:06d}.npy", pair.r_rgb.detach().cpu().numpy())
            save_array(geometry_dir / "residual_cycle" / f"{idx:06d}.npy", pair.r_cycle.detach().cpu().numpy())
            save_array(geometry_dir / "visibility" / f"{idx:06d}.npy", pair.visibility.detach().cpu().numpy())
            save_array(geometry_dir / "dynamic_prior" / f"{idx:06d}.npy", dynamic_prior.detach().cpu().numpy())
    stage.update(1)
    stage.close()
    return {
        "frames": frames,
        "frames_dir": str(frames_dir),
        "segmentation_frames_dir": str(seg_frames_dir),
        "da3_seq": da3_seq,
        "residual_seq": residual_seq,
        "dynamic_priors": dynamic_priors,
        "dynamic_prior": dynamic_priors[0],
        "source_video": str(video_path),
        "temp_root": None if cfg.pipeline.save_intermediates else str(temp_root),
    }


def _resize_for_segmentation(frame: np.ndarray, resize_long_edge: int) -> np.ndarray:
    height, width = frame.shape[:2]
    long_edge = max(height, width)
    if long_edge <= resize_long_edge:
        return frame
    scale = resize_long_edge / float(long_edge)
    out_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return np.asarray(Image.fromarray(frame).resize(out_size, resample=Image.BILINEAR))
