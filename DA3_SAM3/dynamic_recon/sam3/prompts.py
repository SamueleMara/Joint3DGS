from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class PromptRegion:
    frame_index: int
    box_xyxy: tuple[int, int, int, int] | None
    positive_points: list[tuple[int, int]]
    negative_points: list[tuple[int, int]]
    score: float


def extract_connected_components(mask: np.ndarray) -> list[np.ndarray]:
    binary = (mask > 0).astype(np.uint8)
    if binary.max() == 0:
        return []
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components: list[np.ndarray] = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 4:
            continue
        components.append(labels == label)
    return components


def mask_to_boxes(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def select_top_regions(mask: np.ndarray, score_map: np.ndarray, max_regions: int) -> list[np.ndarray]:
    components = extract_connected_components(mask)
    scored = sorted(components, key=lambda component: float(score_map[component].mean()) if np.any(component) else 0.0, reverse=True)
    return scored[:max_regions]


def sample_click_points(mask: np.ndarray, num_points_per_region: int) -> list[tuple[int, int]]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return []
    stride = max(1, ys.size // max(num_points_per_region, 1))
    return [(int(xs[idx]), int(ys[idx])) for idx in range(0, ys.size, stride)][:num_points_per_region]


def build_prompts_from_dynamic_prior(
    dynamic_prior: np.ndarray,
    frame_index: int,
    max_regions: int,
    min_region_area: int,
    points_per_region: int,
    seed_threshold_high: float,
) -> list[PromptRegion]:
    threshold = max(float(seed_threshold_high), float(np.quantile(dynamic_prior, 0.9)))
    seed_mask = _build_seed_mask(dynamic_prior, threshold)
    regions = [region for region in select_top_regions(seed_mask, dynamic_prior, max_regions) if int(region.sum()) >= int(min_region_area)]
    prompts: list[PromptRegion] = []
    for region in regions:
        box = mask_to_boxes(region)
        prompts.append(
            PromptRegion(
                frame_index=frame_index,
                box_xyxy=box,
                positive_points=sample_click_points(region, points_per_region),
                negative_points=_sample_negative_ring(box, dynamic_prior.shape, points_per_region),
                score=float(dynamic_prior[region].mean()) if np.any(region) else 0.0,
            )
        )
    if prompts:
        return prompts
    return [_fallback_prompt(dynamic_prior, frame_index=frame_index, points_per_region=points_per_region)]


def build_prompt_schedule(
    dynamic_priors: list[np.ndarray],
    max_regions: int,
    min_region_area: int,
    points_per_region: int,
    seed_threshold_high: float,
    max_prompt_frames: int,
) -> list[PromptRegion]:
    if not dynamic_priors:
        return []
    scored_frames = sorted(
        [(frame_index, float(np.max(prior))) for frame_index, prior in enumerate(dynamic_priors)],
        key=lambda item: item[1],
        reverse=True,
    )
    selected_frames = [frame_index for frame_index, score in scored_frames if score >= seed_threshold_high][:max_prompt_frames]
    if not selected_frames:
        selected_frames = [scored_frames[0][0]]
    prompts: list[PromptRegion] = []
    for frame_index in sorted(selected_frames):
        prompts.extend(
            build_prompts_from_dynamic_prior(
                dynamic_priors[frame_index],
                frame_index=frame_index,
                max_regions=max_regions,
                min_region_area=min_region_area,
                points_per_region=points_per_region,
                seed_threshold_high=seed_threshold_high,
            )
        )
    return prompts


def _fallback_prompt(dynamic_prior: np.ndarray, frame_index: int, points_per_region: int) -> PromptRegion:
    height, width = dynamic_prior.shape[:2]
    flat_index = int(np.argmax(dynamic_prior))
    y, x = np.unravel_index(flat_index, dynamic_prior.shape)
    radius = max(min(height, width) // 16, 8)
    x0 = max(0, x - radius)
    y0 = max(0, y - radius)
    x1 = min(width - 1, x + radius)
    y1 = min(height - 1, y + radius)
    positive_points = [(int(x), int(y))]
    if points_per_region > 1:
        positive_points.extend(
            [
                (int(max(x0, x - radius // 2)), int(y)),
                (int(min(x1, x + radius // 2)), int(y)),
            ][: points_per_region - 1]
        )
    return PromptRegion(
        frame_index=frame_index,
        box_xyxy=(int(x0), int(y0), int(x1), int(y1)),
        positive_points=positive_points,
        negative_points=_sample_negative_ring((int(x0), int(y0), int(x1), int(y1)), dynamic_prior.shape, points_per_region),
        score=float(dynamic_prior[y, x]),
    )


def _sample_negative_ring(
    box_xyxy: tuple[int, int, int, int] | None,
    image_shape: tuple[int, ...],
    num_points: int,
) -> list[tuple[int, int]]:
    if box_xyxy is None or num_points <= 0:
        return []
    height, width = image_shape[:2]
    x0, y0, x1, y1 = box_xyxy
    pad = max((x1 - x0 + y1 - y0) // 8, 4)
    ring = [
        (max(0, x0 - pad), max(0, y0 - pad)),
        (min(width - 1, x1 + pad), max(0, y0 - pad)),
        (max(0, x0 - pad), min(height - 1, y1 + pad)),
        (min(width - 1, x1 + pad), min(height - 1, y1 + pad)),
        ((x0 + x1) // 2, max(0, y0 - pad)),
        ((x0 + x1) // 2, min(height - 1, y1 + pad)),
        (max(0, x0 - pad), (y0 + y1) // 2),
        (min(width - 1, x1 + pad), (y0 + y1) // 2),
    ]
    return [(int(x), int(y)) for x, y in ring[:num_points]]


def _build_seed_mask(dynamic_prior: np.ndarray, threshold: float) -> np.ndarray:
    binary = (dynamic_prior > threshold).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.dilate(binary, kernel, iterations=1)
    return binary.astype(bool)
