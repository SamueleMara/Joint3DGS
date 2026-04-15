from __future__ import annotations

from dataclasses import asdict
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from dynamic_recon.config.schema import DA3Config
from dynamic_recon.da3.types import DA3FrameOutput, DA3SequenceOutput
from dynamic_recon.data.video_io import get_video_metadata, read_video_frames
from dynamic_recon.progress import progress_iter


class DA3Adapter:
    def infer_sequence(self, frames: list[np.ndarray], cfg: DA3Config, source_video: str = "") -> DA3SequenceOutput:
        raise NotImplementedError


class MockDA3Adapter(DA3Adapter):
    def infer_sequence(self, frames: list[np.ndarray], cfg: DA3Config, source_video: str = "") -> DA3SequenceOutput:
        if not frames:
            raise ValueError("No frames provided to DA3 adapter")
        height, width = frames[0].shape[:2]
        base_k = torch.tensor(
            [[float(max(height, width)), 0.0, width / 2.0], [0.0, float(max(height, width)), height / 2.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        outputs: list[DA3FrameOutput] = []
        for index, frame in enumerate(progress_iter(frames, desc="DA3 mock frames", total=len(frames))):
            rgb = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0
            gray = rgb.mean(dim=0)
            gradient = torch.linspace(0.0, 1.0, width, dtype=torch.float32).view(1, width)
            depth = 1.0 + 3.0 * gray + gradient
            confidence = torch.clamp(0.5 + 0.5 * gray, 0.0, 1.0)
            extrinsics = torch.eye(4, dtype=torch.float32)
            extrinsics[0, 3] = -0.05 * index
            outputs.append(
                DA3FrameOutput(
                    frame_index=index,
                    depth=depth,
                    intrinsics=base_k.clone(),
                    extrinsics=extrinsics,
                    confidence=confidence,
                    rgb=rgb,
                    aux={"adapter": "mock", "cfg": asdict(cfg)},
                )
            )
        return DA3SequenceOutput(frames=outputs, fps=0.0, height=height, width=width, source_video=source_video)


class OfficialDA3Adapter(DA3Adapter):
    def __init__(self, api_cls: Any) -> None:
        self.api_cls = api_cls
        self.model_cache: dict[str, Any] = {}

    def infer_sequence(self, frames: list[np.ndarray], cfg: DA3Config, source_video: str = "") -> DA3SequenceOutput:
        if not frames:
            raise ValueError("No frames provided to DA3 adapter")
        model_id = cfg.checkpoint or cfg.model_name
        if not model_id:
            raise RuntimeError(
                "DA3 requires `da3.model_name` or `da3.checkpoint`. "
                "Per the upstream model cards, prefer a `-1.1` model such as `depth-anything/DA3-LARGE-1.1`."
            )
        model = self._get_model(model_id)
        chunk_size = cfg.sequence_chunk_size
        if chunk_size is None or chunk_size <= 0 or chunk_size >= len(frames):
            return self._infer_single_chunk(model, frames, cfg, source_video=source_video, start_index=0)

        overlap = max(cfg.sequence_chunk_overlap, 0)
        step = max(chunk_size - overlap, 1)
        chunks: list[tuple[int, list[np.ndarray]]] = []
        for start in range(0, len(frames), step):
            end = min(start + chunk_size, len(frames))
            chunk_frames = frames[start:end]
            if not chunk_frames:
                continue
            chunks.append((start, chunk_frames))
            if end >= len(frames):
                break

        outputs: list[DA3FrameOutput] = []
        final_height = 0
        final_width = 0
        for start_index, chunk_frames in progress_iter(chunks, desc="DA3 sequence chunks", total=len(chunks)):
            chunk_output = self._infer_single_chunk(model, chunk_frames, cfg, source_video=source_video, start_index=start_index)
            final_height = chunk_output.height
            final_width = chunk_output.width
            outputs.extend(chunk_output.frames)
        return DA3SequenceOutput(frames=outputs, fps=0.0, height=final_height, width=final_width, source_video=source_video)

    def _get_model(self, model_id: str) -> Any:
        if model_id in self.model_cache:
            return self.model_cache[model_id]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = self.api_cls.from_pretrained(model_id)
        model = model.to(device)
        self.model_cache[model_id] = model
        return model

    def _infer_single_chunk(
        self,
        model: Any,
        frames: list[np.ndarray],
        cfg: DA3Config,
        *,
        source_video: str,
        start_index: int,
    ) -> DA3SequenceOutput:
        prediction = self._run_model_with_retries(model, frames, cfg)
        if prediction.intrinsics is None or prediction.extrinsics is None:
            raise RuntimeError(
                "DA3 did not return intrinsics/extrinsics. "
                "Use an any-view model from the model cards, preferably `depth-anything/DA3-LARGE-1.1`, "
                "or `depth-anything/DA3NESTED-GIANT-LARGE-1.1` if you need metric scale."
            )
        extrinsics = prediction.extrinsics
        if extrinsics.ndim == 3 and extrinsics.shape[-2:] == (3, 4):
            padded = np.zeros((extrinsics.shape[0], 4, 4), dtype=extrinsics.dtype)
            padded[:, :3, :4] = extrinsics
            padded[:, 3, 3] = 1.0
            extrinsics = padded
        outputs: list[DA3FrameOutput] = []
        out_height, out_width = prediction.depth.shape[-2:]
        for local_index, frame in enumerate(progress_iter(frames, desc="DA3 output packaging", total=len(frames))):
            rgb = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0
            if rgb.shape[-2:] != (out_height, out_width):
                rgb = F.interpolate(rgb.unsqueeze(0), size=(out_height, out_width), mode="bilinear", align_corners=False)[0]
            outputs.append(
                DA3FrameOutput(
                    frame_index=start_index + local_index,
                    depth=torch.from_numpy(prediction.depth[local_index]).float(),
                    intrinsics=torch.from_numpy(prediction.intrinsics[local_index]).float(),
                    extrinsics=torch.from_numpy(extrinsics[local_index]).float(),
                    confidence=None if prediction.conf is None else torch.from_numpy(prediction.conf[local_index]).float(),
                    rgb=rgb,
                    aux={
                        "adapter": "official",
                        "model_name": cfg.checkpoint or cfg.model_name,
                        "process_res": int(out_height),
                        "use_ray_pose": cfg.use_ray_pose,
                        "ref_view_strategy": cfg.ref_view_strategy,
                        "is_metric": getattr(prediction, "is_metric", None),
                        "chunk_start": start_index,
                        "chunk_size": len(frames),
                    },
                )
            )
        return DA3SequenceOutput(frames=outputs, fps=0.0, height=int(out_height), width=int(out_width), source_video=source_video)

    def _run_model_with_retries(self, model: Any, frames: list[np.ndarray], cfg: DA3Config) -> Any:
        retry_resolutions = [cfg.process_res, *[res for res in cfg.oom_retry_process_res if res != cfg.process_res]]
        prediction = None
        last_exc: Exception | None = None
        for process_res in progress_iter(retry_resolutions, desc="DA3 inference retries", total=len(retry_resolutions)):
            try:
                prediction = model.inference(
                    image=frames,
                    process_res=process_res,
                    process_res_method=cfg.process_res_method,
                    use_ray_pose=cfg.use_ray_pose,
                    ref_view_strategy=cfg.ref_view_strategy,
                )
                break
            except torch.OutOfMemoryError as exc:
                last_exc = exc
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
        if prediction is None:
            attempted = ", ".join(str(res) for res in retry_resolutions)
            chunk_msg = ""
            if cfg.sequence_chunk_size:
                chunk_msg = f" Current chunk size: {cfg.sequence_chunk_size} frames."
            raise RuntimeError(
                "DA3 ran out of CUDA memory for all configured processing resolutions. "
                f"Attempted process_res values: {attempted}.{chunk_msg} "
                "Lower `video.max_frames`, increase `video.stride`, reduce `video.resize_long_edge`, "
                "lower `da3.process_res`/`da3.oom_retry_process_res`, or enable smaller `da3.sequence_chunk_size`."
            ) from last_exc
        return prediction


def build_adapter(cfg: DA3Config | None = None) -> DA3Adapter:
    cfg = cfg or DA3Config()
    try:
        api_module = import_module("depth_anything_3.api")
        return OfficialDA3Adapter(api_module.DepthAnything3)
    except Exception as exc:
        if cfg.allow_mock:
            return MockDA3Adapter()
        raise RuntimeError(
            "Failed to import the official DA3 API from `depth_anything_3.api`. "
            "Install the DA3 submodule package and provide an accessible checkpoint/model id."
        ) from exc


def load_video_for_da3(video_path: str | Path, cfg: DA3Config) -> tuple[list[np.ndarray], dict[str, float | int]]:
    metadata = get_video_metadata(video_path)
    frames = read_video_frames(video_path, stride=cfg.stride)
    return frames, metadata
