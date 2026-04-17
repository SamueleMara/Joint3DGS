from __future__ import annotations

import cv2
import numpy as np
import torch


def compute_dense_flow_pair(frame_t: torch.Tensor | np.ndarray, frame_s: torch.Tensor | np.ndarray) -> torch.Tensor:
    image_t = _to_gray_u8(frame_t)
    image_s = _to_gray_u8(frame_s)
    flow = cv2.calcOpticalFlowFarneback(
        image_t,
        image_s,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=21,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    return torch.from_numpy(flow.astype(np.float32))


def flow_forward_backward_consistency(flow_fw: torch.Tensor, flow_bw: torch.Tensor, threshold: float) -> torch.Tensor:
    height, width = flow_fw.shape[:2]
    yy, xx = torch.meshgrid(
        torch.arange(height, device=flow_fw.device, dtype=torch.float32),
        torch.arange(width, device=flow_fw.device, dtype=torch.float32),
        indexing="ij",
    )
    target_x = torch.clamp(xx + flow_fw[..., 0], 0.0, float(width - 1))
    target_y = torch.clamp(yy + flow_fw[..., 1], 0.0, float(height - 1))
    x0 = torch.floor(target_x).long()
    y0 = torch.floor(target_y).long()
    sampled_bw = flow_bw[y0, x0]
    cycle_error = torch.linalg.norm(flow_fw + sampled_bw, dim=-1)
    return torch.exp(-cycle_error / max(float(threshold), 1.0e-6))


def _to_gray_u8(frame: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(frame, torch.Tensor):
        tensor = frame.detach().cpu()
        if tensor.ndim == 3 and tensor.shape[0] == 3:
            image = tensor.permute(1, 2, 0).numpy()
        else:
            image = tensor.numpy()
    else:
        image = np.asarray(frame)
    if image.ndim == 3 and image.shape[-1] == 3:
        if image.dtype != np.uint8:
            image = np.clip(image * 255.0 if image.max() <= 1.5 else image, 0, 255).astype(np.uint8)
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0 if image.max() <= 1.5 else image, 0, 255).astype(np.uint8)
    return image
