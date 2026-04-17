from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from dynamic_recon.config.schema import PipelineConfig
from dynamic_recon.data.cache_io import save_array, save_json
from dynamic_recon.pipelines.preprocess import run_preprocess
from dynamic_recon.progress import progress_iter, stage_bar
from dynamic_recon.sam3.prompts import build_prompt_schedule
from dynamic_recon.sam3.wrapper import add_prompts, propagate_masks, start_video_session


def run_initial_pass(video_path: str | Path, outdir: str | Path, cfg: PipelineConfig) -> dict[str, object]:
    stage = stage_bar(desc="Initial segmentation pass", total=4)
    state = run_preprocess(video_path, outdir, cfg)
    stage.update(1)
    state = run_segmentation_pass(state, outdir, cfg, tag="initial")
    stage.update(1)
    stage.update(1)
    stage.close()
    return state


def run_segmentation_pass(
    state: dict[str, object],
    outdir: str | Path,
    cfg: PipelineConfig,
    *,
    tag: str,
) -> dict[str, object]:
    prompts = build_prompt_schedule(
        [prior.detach().cpu().numpy() for prior in state["dynamic_priors"]],
        max_regions=cfg.sam3.max_regions_per_frame,
        min_region_area=cfg.sam3.min_region_area,
        points_per_region=cfg.sam3.points_per_region,
        seed_threshold_high=cfg.geometry.seed_threshold_high,
        max_prompt_frames=cfg.sam3.max_prompt_frames,
    )
    sam_resource: str | Path = state.get("segmentation_frames_dir", state["frames_dir"])
    session = start_video_session(sam_resource, cfg.sam3)
    session = add_prompts(session, prompts)
    sam3_out = propagate_masks(session)

    if cfg.pipeline.save_intermediates:
        sam3_dir = Path(outdir) / "sam3"
        pass_dir = sam3_dir / tag
        (pass_dir / "prompts").mkdir(parents=True, exist_ok=True)
        (pass_dir / "logits").mkdir(parents=True, exist_ok=True)
        (pass_dir / "masks").mkdir(parents=True, exist_ok=True)
        save_json(pass_dir / "prompts" / "000000.json", [asdict(prompt) for prompt in prompts])
        for frame in progress_iter(sam3_out.frames, desc=f"Caching SAM outputs {tag}", total=len(sam3_out.frames)):
            save_array(pass_dir / "logits" / f"{frame.frame_index:06d}.npy", frame.logits.detach().cpu().numpy())
            save_array(pass_dir / "masks" / f"{frame.frame_index:06d}.npy", frame.masks.detach().cpu().numpy())

        if tag == "initial":
            legacy_dir = sam3_dir
            (legacy_dir / "prompts").mkdir(parents=True, exist_ok=True)
            (legacy_dir / "logits").mkdir(parents=True, exist_ok=True)
            (legacy_dir / "masks").mkdir(parents=True, exist_ok=True)
            save_json(legacy_dir / "prompts" / "000000.json", [asdict(prompt) for prompt in prompts])
            for frame in sam3_out.frames:
                save_array(legacy_dir / "logits" / f"{frame.frame_index:06d}.npy", frame.logits.detach().cpu().numpy())
                save_array(legacy_dir / "masks" / f"{frame.frame_index:06d}.npy", frame.masks.detach().cpu().numpy())

    return {**state, "sam3_out": sam3_out, "segmentation_resource": str(sam_resource), "sam_prompts": prompts}
