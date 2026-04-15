"""Synthetic trajectory generators for axis-constrained motion."""

from typing import Optional

import numpy as np

from trajectory_constraint_fit import build_axis_frame, normalize


def random_axis(rng: np.random.Generator):
    """Generate a random axis direction and canonical axis point."""
    u = normalize(rng.normal(size=3))
    c = rng.normal(size=3)
    c = c - np.dot(c, u) * u
    return u, c


def _noise(points: np.ndarray, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    if noise_std <= 0.0:
        return points
    return points + rng.normal(scale=noise_std, size=points.shape)


def generate_circle_trajectory(
    num_samples: int = 120,
    radius: float = 1.0,
    h0: float = 0.3,
    turns: float = 1.5,
    noise_std: float = 0.01,
    u: Optional[np.ndarray] = None,
    c: Optional[np.ndarray] = None,
    seed: int = 0,
    return_params: bool = False,
) -> np.ndarray:
    """Generate circular motion around a fixed axis."""
    rng = np.random.default_rng(seed)
    if u is None or c is None:
        u, c = random_axis(rng)
    frame = build_axis_frame(u)
    phi = np.linspace(0.0, 2.0 * np.pi * turns, num_samples)
    local = np.column_stack((radius * np.cos(phi), radius * np.sin(phi), np.full_like(phi, h0)))
    points = c + local @ frame.T
    points = _noise(points, noise_std, rng)
    if return_params:
        return points, {
            "motion_type": "circle",
            "axis_direction": u,
            "axis_point": c,
            "radius": radius,
            "h0": h0,
            "phi": phi,
        }
    return points


def generate_translation_trajectory(
    num_samples: int = 120,
    radius: float = 0.9,
    phi0: float = 0.8,
    h_span: float = 2.0,
    noise_std: float = 0.01,
    u: Optional[np.ndarray] = None,
    c: Optional[np.ndarray] = None,
    seed: int = 1,
    return_params: bool = False,
) -> np.ndarray:
    """Generate translation parallel to the axis at constant offset."""
    rng = np.random.default_rng(seed)
    if u is None or c is None:
        u, c = random_axis(rng)
    frame = build_axis_frame(u)
    h = np.linspace(-0.5 * h_span, 0.5 * h_span, num_samples)
    local = np.column_stack(
        (
            np.full_like(h, radius * np.cos(phi0)),
            np.full_like(h, radius * np.sin(phi0)),
            h,
        )
    )
    points = c + local @ frame.T
    points = _noise(points, noise_std, rng)
    if return_params:
        return points, {
            "motion_type": "translation",
            "axis_direction": u,
            "axis_point": c,
            "radius": radius,
            "phi0": phi0,
            "h": h,
        }
    return points


def generate_screw_trajectory(
    num_samples: int = 140,
    radius: float = 1.1,
    h0: float = -0.4,
    pitch: float = 0.25,
    turns: float = 2.0,
    noise_std: float = 0.01,
    u: Optional[np.ndarray] = None,
    c: Optional[np.ndarray] = None,
    seed: int = 2,
    return_params: bool = False,
) -> np.ndarray:
    """Generate screw motion around a fixed axis."""
    rng = np.random.default_rng(seed)
    if u is None or c is None:
        u, c = random_axis(rng)
    frame = build_axis_frame(u)
    phi = np.linspace(0.0, 2.0 * np.pi * turns, num_samples)
    local = np.column_stack((radius * np.cos(phi), radius * np.sin(phi), h0 + pitch * phi))
    points = c + local @ frame.T
    points = _noise(points, noise_std, rng)
    if return_params:
        return points, {
            "motion_type": "screw",
            "axis_direction": u,
            "axis_point": c,
            "radius": radius,
            "h0": h0,
            "pitch": pitch,
            "phi": phi,
        }
    return points
