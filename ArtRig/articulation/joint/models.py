from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from articulation.geometry.lines import normalize_direction



def _rodrigues_rotate(points: torch.Tensor, axis: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Rotate points around axis through origin.

    points: [P,3], axis: [3], theta: [T]
    returns: [P,T,3]
    """
    u = normalize_direction(axis)
    p = points[:, None, :]
    th = theta[None, :, None]

    cos = torch.cos(th)
    sin = torch.sin(th)

    dot = (p * u[None, None, :]).sum(dim=-1, keepdim=True)
    cross = torch.cross(u[None, None, :].expand_as(p), p, dim=-1)

    return p * cos + cross * sin + u[None, None, :] * dot * (1.0 - cos)


class RevoluteModel(nn.Module):
    def __init__(self, num_frames: int, axis_dir: torch.Tensor, axis_point: torch.Tensor | None):
        super().__init__()
        self.axis_dir_raw = nn.Parameter(axis_dir.clone().float())
        self.axis_point = nn.Parameter((torch.zeros(3) if axis_point is None else axis_point.clone().float()))
        self.state = nn.Parameter(torch.zeros(num_frames))

    def axis_dir(self) -> torch.Tensor:
        return normalize_direction(self.axis_dir_raw)

    def forward(self, canonical_points: torch.Tensor) -> torch.Tensor:
        q = canonical_points - self.axis_point[None, :]
        r = _rodrigues_rotate(q, self.axis_dir(), self.state)
        return r + self.axis_point[None, None, :]


class PrismaticModel(nn.Module):
    def __init__(self, num_frames: int, axis_dir: torch.Tensor):
        super().__init__()
        self.axis_dir_raw = nn.Parameter(axis_dir.clone().float())
        self.state = nn.Parameter(torch.zeros(num_frames))

    def axis_dir(self) -> torch.Tensor:
        return normalize_direction(self.axis_dir_raw)

    def forward(self, canonical_points: torch.Tensor) -> torch.Tensor:
        u = self.axis_dir()[None, None, :]
        return canonical_points[:, None, :] + self.state[None, :, None] * u


class ScrewModel(nn.Module):
    def __init__(self, num_frames: int, axis_dir: torch.Tensor, axis_point: torch.Tensor | None, pitch: float | None):
        super().__init__()
        self.axis_dir_raw = nn.Parameter(axis_dir.clone().float())
        self.axis_point = nn.Parameter((torch.zeros(3) if axis_point is None else axis_point.clone().float()))
        self.state = nn.Parameter(torch.zeros(num_frames))
        self.pitch = nn.Parameter(torch.tensor(0.0 if pitch is None else float(pitch), dtype=torch.float32))

    def axis_dir(self) -> torch.Tensor:
        return normalize_direction(self.axis_dir_raw)

    def forward(self, canonical_points: torch.Tensor) -> torch.Tensor:
        u = self.axis_dir()
        q = canonical_points - self.axis_point[None, :]
        r = _rodrigues_rotate(q, u, self.state)
        trans = (self.pitch * self.state)[None, :, None] * u[None, None, :]
        return r + self.axis_point[None, None, :] + trans


@dataclass
class JointModelBundle:
    name: str
    model: nn.Module
