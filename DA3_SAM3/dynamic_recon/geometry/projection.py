from __future__ import annotations

import torch
import torch.nn.functional as F

from dynamic_recon.geometry.camera import from_homogeneous, to_homogeneous, w2c_to_c2w


def pixel_grid(height: int, width: int, device: torch.device | str, homogeneous: bool = True) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    grid = torch.stack([xx.float(), yy.float()], dim=-1)
    if homogeneous:
        ones = torch.ones(height, width, 1, device=device)
        return torch.cat([grid, ones], dim=-1)
    return grid


def backproject(depth: torch.Tensor, k: torch.Tensor, e_w2c: torch.Tensor) -> torch.Tensor:
    height, width = depth.shape
    pixels = pixel_grid(height, width, depth.device, homogeneous=True)
    k_inv = torch.linalg.inv(k)
    cam = torch.einsum("ij,hwj->hwi", k_inv, pixels) * depth.unsqueeze(-1)
    world_h = torch.einsum("ij,hwj->hwi", w2c_to_c2w(e_w2c), to_homogeneous(cam))
    return from_homogeneous(world_h)


def project(world_points: torch.Tensor, k: torch.Tensor, e_w2c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cam_h = torch.einsum("ij,hwj->hwi", e_w2c, to_homogeneous(world_points))
    cam = from_homogeneous(cam_h)
    z_cam = cam[..., 2]
    uvw = torch.einsum("ij,hwj->hwi", k, cam)
    uv = uvw[..., :2] / torch.clamp(z_cam.unsqueeze(-1), min=1.0e-6)
    return uv, z_cam


def sample_at_coords(image_or_map: torch.Tensor, uv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if image_or_map.ndim == 2:
        source = image_or_map[None, None]
    elif image_or_map.ndim == 3:
        source = image_or_map.permute(2, 0, 1).unsqueeze(0)
    else:
        raise ValueError(f"Unsupported source shape: {image_or_map.shape}")
    height, width = uv.shape[:2]
    norm_x = (uv[..., 0] / max(width - 1, 1)) * 2 - 1
    norm_y = (uv[..., 1] / max(height - 1, 1)) * 2 - 1
    grid = torch.stack([norm_x, norm_y], dim=-1)[None]
    sampled = F.grid_sample(source.float(), grid.float(), align_corners=True, mode="bilinear", padding_mode="zeros")
    valid = (norm_x >= -1) & (norm_x <= 1) & (norm_y >= -1) & (norm_y <= 1)
    if image_or_map.ndim == 2:
        return sampled[0, 0], valid
    return sampled[0].permute(1, 2, 0), valid


def reproject_static(depth_t: torch.Tensor, k_t: torch.Tensor, e_t: torch.Tensor, k_s: torch.Tensor, e_s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    world_points = backproject(depth_t, k_t, e_t)
    return project(world_points, k_s, e_s)
