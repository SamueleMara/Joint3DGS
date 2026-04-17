from __future__ import annotations

from dataclasses import dataclass

import torch
from tqdm.auto import tqdm

from articulation.data.dataclasses import FeatureGraph, SegmentationResult, TrackBatch
from articulation.preprocess.windows import WindowSpec, build_sliding_windows
from articulation.segmentation.propagation import merge_window_logits
from articulation.segmentation.trainer import SegmentationTrainer


@dataclass
class WindowingConfig:
    size: int = 8
    stride: int = 4
    final_iters: int = 50



def _slice_tracks_window(tracks: TrackBatch, window: WindowSpec) -> TrackBatch:
    xy = tracks.xy[:, window.start:window.end, :]
    xyz = tracks.xyz[:, window.start:window.end, :]
    valid = tracks.valid[:, window.start:window.end]
    return TrackBatch(
        xy=xy,
        xyz=xyz,
        valid=valid,
        anchor_frame=0,
        point_ids=tracks.point_ids,
        feature=tracks.feature,
        confidence=tracks.confidence,
    )



def _run_windowed_segmentation(
    tracks: TrackBatch,
    graph: FeatureGraph,
    cfg: dict,
    init_logits: torch.Tensor | None,
    image_size: tuple[int, int] | None,
    show_progress: bool = False,
    debug_losses: bool = False,
    wandb_run: object | None = None,
    wandb_prefix: str = "seg",
) -> SegmentationResult:
    w_cfg = cfg.get("window", {})
    windowing = WindowingConfig(
        size=int(w_cfg.get("size", 8)),
        stride=int(w_cfg.get("stride", 4)),
        final_iters=int(w_cfg.get("final_iters", 50)),
    )

    windows = build_sliding_windows(tracks.T, windowing.size, windowing.stride)
    trainer = SegmentationTrainer(cfg)

    window_logits: list[tuple[torch.Tensor, torch.Tensor]] = []
    win_iter = tqdm(
        windows,
        desc="Stage1/Window Passes",
        disable=not show_progress,
        leave=False,
    )
    for wi, w in enumerate(win_iter):
        w_tracks = _slice_tracks_window(tracks, w)
        seg_w = trainer.fit(
            tracks=w_tracks,
            graph=graph,
            init_logits=init_logits,
            image_size=image_size,
            show_progress=show_progress,
            debug_losses=debug_losses,
            progress_desc=f"Stage1/Window {wi+1}/{len(windows)}",
            wandb_run=wandb_run,
            wandb_prefix=f"{wandb_prefix}/window_{wi+1:02d}",
        )
        window_logits.append((tracks.point_ids, seg_w.point_logits.detach().cpu()))
    if show_progress:
        win_iter.close()

    merged = merge_window_logits(tracks.point_ids, window_logits)

    # Optional final refinement on the full sequence using merged logits
    cfg_final = dict(cfg)
    opt_cfg = dict(cfg.get("optimizer", {}))
    opt_cfg["iterations"] = windowing.final_iters
    cfg_final["optimizer"] = opt_cfg

    trainer_final = SegmentationTrainer(cfg_final)
    return trainer_final.fit(
        tracks=tracks,
        graph=graph,
        init_logits=merged,
        image_size=image_size,
        show_progress=show_progress,
        debug_losses=debug_losses,
        progress_desc="Stage1/Final Refinement",
        wandb_run=wandb_run,
        wandb_prefix=f"{wandb_prefix}/final",
    )



def run_stage1_segmentation(
    tracks: TrackBatch,
    graph: FeatureGraph,
    cfg: dict,
    init_logits: torch.Tensor | None = None,
    image_size: tuple[int, int] | None = None,
    show_progress: bool = False,
    debug_losses: bool = False,
    wandb_run: object | None = None,
    wandb_prefix: str = "seg",
) -> SegmentationResult:
    if tracks.T < 2:
        raise ValueError("Segmentation requires at least 2 frames")

    if "window" in cfg and cfg["window"] is not None:
        w_cfg = cfg["window"]
        if isinstance(w_cfg, dict):
            size = int(w_cfg.get("size", tracks.T))
            if size < tracks.T:
                return _run_windowed_segmentation(
                    tracks,
                    graph,
                    cfg,
                    init_logits,
                    image_size,
                    show_progress=show_progress,
                    debug_losses=debug_losses,
                    wandb_run=wandb_run,
                    wandb_prefix=wandb_prefix,
                )

    trainer = SegmentationTrainer(cfg)
    return trainer.fit(
        tracks=tracks,
        graph=graph,
        init_logits=init_logits,
        image_size=image_size,
        show_progress=show_progress,
        debug_losses=debug_losses,
        progress_desc="Stage1/Segmentation",
        wandb_run=wandb_run,
        wandb_prefix=wandb_prefix,
    )
