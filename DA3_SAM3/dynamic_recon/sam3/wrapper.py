from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from contextlib import nullcontext
import warnings

import numpy as np
import torch
from PIL import Image

from dynamic_recon.data.video_io import read_video_frames
from dynamic_recon.progress import progress_iter


@dataclass(slots=True)
class SAM3FrameOutput:
    frame_index: int
    logits: torch.Tensor
    masks: torch.Tensor
    instance_ids: list[int]
    scores: torch.Tensor | None = None


@dataclass(slots=True)
class SAM3SequenceOutput:
    frames: list[SAM3FrameOutput]


def load_sam3_predictor(cfg: object) -> object:
    version = getattr(cfg, "version", "sam3.1")
    if str(version).startswith("sam2"):
        try:
            sam2_module = import_module("sam2.sam2_video_predictor")
            model_id = getattr(cfg, "model_id", None) or _default_sam2_model_id(str(version))
            predictor = sam2_module.SAM2VideoPredictor.from_pretrained(model_id)
            return {"backend": "sam2", "predictor": predictor, "model_id": model_id}
        except Exception as exc:
            if getattr(cfg, "allow_mock", False):
                warnings.warn(
                    "Falling back to SAM mock backend because SAM2 predictor initialization failed. "
                    f"Reason: {exc}"
                )
                return {"cfg": cfg, "mock": True, "reason": str(exc)}
            raise RuntimeError(
                "Failed to build the official SAM2 video predictor. "
                "Ensure the `sam2` package is installed and that the configured SAM2 model id is accessible."
            ) from exc
    try:
        autocast_context = _enable_sam3_autocast()
        builder = import_module("sam3.model_builder")
        predictor = builder.build_sam3_predictor(
            checkpoint_path=getattr(cfg, "checkpoint", None),
            version=version,
        )
        return {"backend": "sam3", "predictor": predictor, "autocast_context": autocast_context}
    except Exception as exc:
        if getattr(cfg, "allow_mock", False):
            warnings.warn(
                "Falling back to SAM mock backend because SAM3 predictor initialization failed. "
                f"Reason: {exc}"
            )
            return {"cfg": cfg, "mock": True, "reason": str(exc)}
        raise RuntimeError(
            "Failed to build the official SAM3 video predictor. "
            "Ensure the `sam3` package is installed and that checkpoint access on Hugging Face is available, "
            "or set `sam3.checkpoint` to a local checkpoint path. "
            "The upstream code also requires `psutil` and Hugging Face authentication for auto-download."
        ) from exc


def start_video_session(video_path_or_frames: object, cfg: object) -> dict[str, object]:
    predictor_bundle = load_sam3_predictor(cfg)
    if isinstance(predictor_bundle, dict) and predictor_bundle.get("mock"):
        frames = video_path_or_frames
        if isinstance(video_path_or_frames, (str, Path)):
            frames = _read_frames_for_mock(video_path_or_frames)
        return {"video": frames, "cfg": cfg, "predictor": predictor_bundle, "mock": True}
    resource_path = video_path_or_frames
    if not isinstance(video_path_or_frames, (str, Path)):
        raise RuntimeError(
            "The official SAM video predictor expects a video path or frame folder path. "
            "Pass an MP4 path or export JPEG frames to disk before starting a session."
        )
    backend = predictor_bundle["backend"]
    predictor = predictor_bundle["predictor"]
    if backend == "sam2":
        inference_state = predictor.init_state(
            str(resource_path),
            offload_video_to_cpu=getattr(cfg, "offload_video_to_cpu", False),
            offload_state_to_cpu=getattr(cfg, "offload_state_to_cpu", False),
            async_loading_frames=getattr(cfg, "async_loading_frames", False),
        )
        return {
            "video": str(resource_path),
            "cfg": cfg,
            "backend": backend,
            "predictor": predictor,
            "inference_state": inference_state,
        }
    response = predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=str(resource_path),
            offload_video_to_cpu=getattr(cfg, "offload_video_to_cpu", False),
        )
    )
    return {"video": str(resource_path), "cfg": cfg, "backend": backend, "predictor": predictor, "session_id": response["session_id"]}


def add_prompts(session: dict[str, object], prompts: list[object]) -> dict[str, object]:
    if session.get("mock"):
        session["prompts"] = prompts
        return session
    if session.get("backend") == "sam2":
        predictor = session["predictor"]
        inference_state = session["inference_state"]
        with _sam2_autocast():
            for obj_id, prompt in enumerate(progress_iter(prompts, desc="SAM2 prompts", total=len(prompts)), start=1):
                points: list[list[float]] = []
                labels: list[int] = []
                for x, y in prompt.positive_points:
                    points.append([float(x), float(y)])
                    labels.append(1)
                for x, y in getattr(prompt, "negative_points", []):
                    points.append([float(x), float(y)])
                    labels.append(0)
                box = None
                if prompt.box_xyxy is not None:
                    x0, y0, x1, y1 = prompt.box_xyxy
                    box = [float(x0), float(y0), float(x1), float(y1)]
                predictor.add_new_points_or_box(
                    inference_state,
                    frame_idx=prompt.frame_index,
                    obj_id=obj_id,
                    points=points or None,
                    labels=labels or None,
                    box=box,
                )
        session["prompts"] = prompts
        return session
    predictor = session["predictor"]
    session_id = session["session_id"]
    for obj_id, prompt in enumerate(progress_iter(prompts, desc="SAM3 prompts", total=len(prompts)), start=1):
        request = {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": prompt.frame_index,
            "obj_id": obj_id,
        }
        has_points = bool(prompt.positive_points or getattr(prompt, "negative_points", None))
        if prompt.positive_points:
            request["points"] = [[x, y] for x, y in prompt.positive_points]
            request["point_labels"] = [1] * len(prompt.positive_points)
        if getattr(prompt, "negative_points", None):
            all_points = request.get("points", [])
            all_labels = request.get("point_labels", [])
            all_points.extend([[x, y] for x, y in prompt.negative_points])
            all_labels.extend([0] * len(prompt.negative_points))
            request["points"] = all_points
            request["point_labels"] = all_labels
        if not has_points and prompt.box_xyxy is not None:
            x0, y0, x1, y1 = prompt.box_xyxy
            request["bounding_boxes"] = [[x0, y0, x1 - x0, y1 - y0]]
        predictor.handle_request(request=request)
    session["prompts"] = prompts
    return session


def propagate_masks(session: dict[str, object]) -> SAM3SequenceOutput:
    if session.get("mock"):
        frames = session["video"]
        prompts = session.get("prompts", [])
        outputs: list[SAM3FrameOutput] = []
        for index, frame in enumerate(frames):
            height, width = frame.shape[:2]
            logits = torch.zeros(1, height, width, dtype=torch.float32)
            if prompts:
                logits[:, height // 4 : height // 2, width // 4 : width // 2] = 4.0
            masks = logits > 0
            outputs.append(SAM3FrameOutput(frame_index=index, logits=logits, masks=masks, instance_ids=[1], scores=torch.tensor([1.0])))
        return SAM3SequenceOutput(frames=outputs)

    predictor = session["predictor"]
    if session.get("backend") == "sam2":
        inference_state = session["inference_state"]
        outputs: list[SAM3FrameOutput] = []
        with _sam2_autocast():
            for frame_index, obj_ids, masks in predictor.propagate_in_video(inference_state):
                mask_tensor = _normalize_sam2_masks(masks).detach().cpu()
                outputs.append(
                    SAM3FrameOutput(
                        frame_index=int(frame_index),
                        logits=mask_tensor,
                        masks=mask_tensor > 0,
                        instance_ids=[int(item) for item in obj_ids],
                        scores=torch.ones(mask_tensor.shape[0], dtype=torch.float32),
                    )
                )
        outputs.sort(key=lambda item: item.frame_index)
        return SAM3SequenceOutput(frames=outputs)
    session_id = session["session_id"]
    outputs: list[SAM3FrameOutput] = []
    stream = predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    )
    for response in progress_iter(stream, desc="SAM3 propagate"):
        frame_index = int(response["frame_index"])
        raw = response["outputs"]
        obj_ids = _extract_obj_ids(raw)
        masks = _extract_masks(raw)
        logits = _extract_logits(raw, masks)
        scores = _extract_scores(raw, len(obj_ids))
        outputs.append(
            SAM3FrameOutput(
                frame_index=frame_index,
                logits=logits,
                masks=masks > 0,
                instance_ids=obj_ids,
                scores=scores,
            )
        )
    outputs.sort(key=lambda item: item.frame_index)
    return SAM3SequenceOutput(frames=outputs)


def _extract_obj_ids(raw: object) -> list[int]:
    if isinstance(raw, tuple) and len(raw) >= 2:
        ids = raw[1]
        return [int(item) for item in ids]
    if isinstance(raw, dict) and "out_obj_ids" in raw:
        ids = raw["out_obj_ids"]
        if isinstance(ids, np.ndarray):
            return [int(item) for item in ids.tolist()]
        if torch.is_tensor(ids):
            return [int(item) for item in ids.detach().cpu().tolist()]
    return []


def _extract_masks(raw: object) -> torch.Tensor:
    if isinstance(raw, tuple):
        if len(raw) >= 4:
            tensor = raw[3]
        elif len(raw) >= 3:
            tensor = raw[2]
        else:
            tensor = None
    elif isinstance(raw, dict):
        tensor = raw.get("out_binary_masks")
    else:
        tensor = None
    if tensor is None:
        return torch.zeros((0, 1, 1), dtype=torch.float32)
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if not torch.is_tensor(tensor):
        raise RuntimeError(f"Unsupported SAM3 mask output type: {type(tensor)}")
    return tensor.float()


def _extract_logits(raw: object, masks: torch.Tensor) -> torch.Tensor:
    if isinstance(raw, tuple) and len(raw) >= 3:
        candidate = raw[2]
        if isinstance(candidate, np.ndarray):
            candidate = torch.from_numpy(candidate)
        if torch.is_tensor(candidate):
            return candidate.float()
    if isinstance(raw, dict) and "out_binary_masks" in raw:
        return masks.float()
    return masks.float()


def _extract_scores(raw: object, count: int) -> torch.Tensor | None:
    if isinstance(raw, tuple) and len(raw) >= 5:
        scores = raw[4]
        if isinstance(scores, np.ndarray):
            return torch.from_numpy(scores).float().reshape(-1)
        if torch.is_tensor(scores):
            return scores.float().reshape(-1)
    if isinstance(raw, dict) and "scores" in raw:
        scores = raw["scores"]
        if isinstance(scores, np.ndarray):
            return torch.from_numpy(scores).float().reshape(-1)
        if torch.is_tensor(scores):
            return scores.float().reshape(-1)
    if count == 0:
        return None
    return torch.ones(count, dtype=torch.float32)


def _enable_sam3_autocast() -> object | None:
    if not torch.cuda.is_available():
        return None
    context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    context.__enter__()
    return context


def _sam2_autocast():
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else nullcontext()


def _default_sam2_model_id(version: str) -> str:
    if version.startswith("sam2.1"):
        return "facebook/sam2.1-hiera-small"
    return "facebook/sam2-hiera-small"


def _normalize_sam2_masks(masks: object) -> torch.Tensor:
    if isinstance(masks, np.ndarray):
        masks = torch.from_numpy(masks)
    if not torch.is_tensor(masks):
        raise RuntimeError(f"Unsupported SAM2 mask output type: {type(masks)}")
    masks = masks.float()
    if masks.ndim == 4 and masks.shape[1] == 1:
        return masks[:, 0]
    if masks.ndim == 3:
        return masks
    raise RuntimeError(f"Unsupported SAM2 mask tensor shape: {tuple(masks.shape)}")


def _read_frames_for_mock(resource_path: str | Path) -> list[np.ndarray]:
    path = Path(resource_path)
    if path.is_dir():
        frame_paths = sorted(
            [item for item in path.iterdir() if item.suffix.lower() in {".jpg", ".jpeg", ".png"}],
            key=lambda item: int(item.stem),
        )
        if not frame_paths:
            raise FileNotFoundError(path)
        return [np.asarray(Image.open(frame_path).convert("RGB"), dtype=np.uint8) for frame_path in frame_paths]
    return read_video_frames(path)
