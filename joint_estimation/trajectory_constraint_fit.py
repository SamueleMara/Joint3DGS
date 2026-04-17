"""Fit a fixed-axis motion constraint to a sampled 3D trajectory."""

from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np
from scipy.optimize import least_squares


EPS = 1e-9


@dataclass
class ClassificationResult:
    label: str
    axis_direction: np.ndarray
    axis_point: np.ndarray
    radius: float
    pitch: Optional[float]
    residuals: Dict[str, float]
    confidence: float
    rho: np.ndarray
    phi: np.ndarray
    h: np.ndarray
    warnings: tuple

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def normalize(v: np.ndarray) -> np.ndarray:
    """Return a unit vector."""
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    if norm < EPS:
        raise ValueError("Cannot normalize a near-zero vector.")
    return v / norm


def build_axis_frame(u: np.ndarray) -> np.ndarray:
    """Construct an orthonormal basis [e1 e2 u]."""
    u = normalize(u)
    basis_candidates = np.eye(3)
    reference = basis_candidates[np.argmin(np.abs(basis_candidates @ u))]
    e1 = reference - np.dot(reference, u) * u
    e1 = normalize(e1)
    e2 = normalize(np.cross(u, e1))
    return np.column_stack((e1, e2, u))


def project_to_axis_frame(points: np.ndarray, u: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Transform points into the axis-aligned frame."""
    frame = build_axis_frame(u)
    return (points - c) @ frame


def unwrap_angles(phi: np.ndarray) -> np.ndarray:
    """Unwrap an angle sequence in radians."""
    return np.unwrap(np.asarray(phi, dtype=float))


def wrapped_angle_distance(a: np.ndarray, b: float) -> np.ndarray:
    """Shortest signed wrapped angular distance."""
    return np.angle(np.exp(1j * (np.asarray(a) - b)))


def cylindrical_coordinates(points: np.ndarray, u: np.ndarray, c: np.ndarray):
    """Return cylindrical coordinates around the fitted axis."""
    axis_frame = project_to_axis_frame(points, u, c)
    rho = np.linalg.norm(axis_frame[:, :2], axis=1)
    phi_wrapped = np.arctan2(axis_frame[:, 1], axis_frame[:, 0])
    phi = unwrap_angles(phi_wrapped)
    h = axis_frame[:, 2]
    return rho, phi, h, phi_wrapped


def _validate_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if points.shape[0] < 3:
        raise ValueError("At least 3 samples are required.")
    return points


def _canonical_axis_point(c: np.ndarray, u: np.ndarray) -> np.ndarray:
    return np.asarray(c, dtype=float) - np.dot(c, u) * u


def _cylinder_residuals(params: np.ndarray, points: np.ndarray) -> np.ndarray:
    u = normalize(params[:3])
    c = _canonical_axis_point(params[3:6], u)
    radius = max(abs(params[6]), EPS)
    radial = np.linalg.norm((points - c) - np.outer((points - c) @ u, u), axis=1)
    return radial - radius


def _fixed_direction_cylinder_residuals(params: np.ndarray, points: np.ndarray, u: np.ndarray) -> np.ndarray:
    c = _canonical_axis_point(params[:3], u)
    radius = max(abs(params[3]), EPS)
    radial = np.linalg.norm((points - c) - np.outer((points - c) @ u, u), axis=1)
    return radial - radius


def _principal_directions(points: np.ndarray) -> np.ndarray:
    centered = points - np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return vh


def fit_axis_with_direction(points: np.ndarray, direction: np.ndarray) -> Dict[str, np.ndarray]:
    """Estimate axis point and radius for a fixed direction."""
    points = _validate_points(points)
    u = normalize(direction)
    centroid = np.mean(points, axis=0)
    c0 = _canonical_axis_point(centroid, u)
    radial = np.linalg.norm((points - c0) - np.outer((points - c0) @ u, u), axis=1)
    x0 = np.hstack((c0, max(np.mean(radial), EPS)))
    result = least_squares(_fixed_direction_cylinder_residuals, x0=x0, args=(points, u), method="trf")
    c = _canonical_axis_point(result.x[:3], u)
    radius = float(max(abs(result.x[3]), EPS))
    return {"axis_direction": u, "axis_point": c, "radius": radius}


def fit_cylinder_axis(points: np.ndarray) -> Dict[str, np.ndarray]:
    """Estimate a common axis and radius for the trajectory."""
    points = _validate_points(points)
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    vh = _principal_directions(points)

    candidates = []
    for idx in (0, 1, 2):
        direction = normalize(vh[idx])
        c0 = _canonical_axis_point(centroid, direction)
        radial = np.linalg.norm(centered - np.outer(centered @ direction, direction), axis=1)
        r0 = max(np.mean(radial), EPS)
        candidates.append(np.hstack((direction, c0, r0)))
        candidates.append(np.hstack((-direction, c0, r0)))

    best = None
    for x0 in candidates:
        result = least_squares(_cylinder_residuals, x0=x0, args=(points,), method="trf")
        residual_norm = float(np.mean(result.fun ** 2))
        if best is None or residual_norm < best[0]:
            best = (residual_norm, result.x)

    assert best is not None
    params = best[1]
    u = normalize(params[:3])
    c = _canonical_axis_point(params[3:6], u)
    radius = float(max(abs(params[6]), EPS))
    return {"axis_direction": u, "axis_point": c, "radius": radius}


def fit_circle_model(rho: np.ndarray, h: np.ndarray) -> Dict[str, float]:
    """Fit a constant-radius, constant-height model."""
    radius = float(np.mean(rho))
    h0 = float(np.mean(h))
    residual = float(np.mean((rho - radius) ** 2 + (h - h0) ** 2))
    return {"radius": radius, "h0": h0, "residual": residual}


def fit_translation_model(rho: np.ndarray, phi_wrapped: np.ndarray) -> Dict[str, float]:
    """Fit a constant-radius, constant-angle model."""
    radius = float(np.mean(rho))
    phi0 = float(np.angle(np.mean(np.exp(1j * phi_wrapped))))
    angle_error = wrapped_angle_distance(phi_wrapped, phi0)
    residual = float(np.mean((rho - radius) ** 2 + angle_error ** 2))
    return {"radius": radius, "phi0": phi0, "residual": residual}


def fit_screw_model(rho: np.ndarray, phi: np.ndarray, h: np.ndarray) -> Dict[str, float]:
    """Fit a constant-radius helix model."""
    radius = float(np.mean(rho))
    design = np.column_stack((np.ones_like(phi), phi))
    h0, pitch = np.linalg.lstsq(design, h, rcond=None)[0]
    residual = float(np.mean((rho - radius) ** 2 + (h - (h0 + pitch * phi)) ** 2))
    return {"radius": radius, "h0": float(h0), "pitch": float(pitch), "residual": residual}


def classify_trajectory(points: np.ndarray, times: Optional[np.ndarray] = None) -> Dict[str, object]:
    """Classify a trajectory as circle, translation, or screw."""
    del times  # Reserved for future irregular-sampling extensions.
    points = _validate_points(points)
    centroid = np.mean(points, axis=0)
    principal_dirs = _principal_directions(points)
    scale = float(np.var(points) + EPS)

    circle_axis = fit_axis_with_direction(points, principal_dirs[2])
    circle_rho, circle_phi, circle_h, circle_phi_wrapped = cylindrical_coordinates(
        points, circle_axis["axis_direction"], circle_axis["axis_point"]
    )
    circle = fit_circle_model(circle_rho, circle_h)

    translation_u = normalize(principal_dirs[0])
    translation_c = _canonical_axis_point(centroid, translation_u)
    trans_rho, trans_phi, trans_h, trans_phi_wrapped = cylindrical_coordinates(points, translation_u, translation_c)
    translation = fit_translation_model(trans_rho, trans_phi_wrapped)
    if float(np.mean(trans_rho)) < 0.1 * max(np.std(points), EPS):
        translation["residual"] = float(np.mean((trans_rho - np.mean(trans_rho)) ** 2))

    screw_axis = fit_cylinder_axis(points)
    screw_rho, screw_phi, screw_h, screw_phi_wrapped = cylindrical_coordinates(
        points, screw_axis["axis_direction"], screw_axis["axis_point"]
    )
    screw = fit_screw_model(screw_rho, screw_phi, screw_h)

    candidates = {
        "circle": {
            "axis": circle_axis,
            "rho": circle_rho,
            "phi": circle_phi,
            "h": circle_h,
            "phi_wrapped": circle_phi_wrapped,
            "model": circle,
        },
        "translation": {
            "axis": {
                "axis_direction": translation_u,
                "axis_point": translation_c,
                "radius": float(np.mean(trans_rho)),
            },
            "rho": trans_rho,
            "phi": trans_phi,
            "h": trans_h,
            "phi_wrapped": trans_phi_wrapped,
            "model": translation,
        },
        "screw": {
            "axis": screw_axis,
            "rho": screw_rho,
            "phi": screw_phi,
            "h": screw_h,
            "phi_wrapped": screw_phi_wrapped,
            "model": screw,
        },
    }
    residuals = {label: data["model"]["residual"] for label, data in candidates.items()}
    sorted_models = sorted(residuals.items(), key=lambda item: item[1])
    best_label, best_residual = sorted_models[0]
    second_best = sorted_models[1][1]
    normalized_residuals = {key: value / scale for key, value in residuals.items()}
    confidence = float((second_best - best_residual) / (second_best + EPS))
    best = candidates[best_label]

    warnings = []
    if best["axis"]["radius"] < 1e-3:
        warnings.append("Estimated radius is very small; axis fit may be ill-conditioned.")
    if np.linalg.norm(points[-1] - points[0]) < 1e-3 and np.std(points, axis=0).max() < 1e-3:
        warnings.append("Trajectory motion is very small; classification may be unreliable.")
    if np.ptp(best["phi"]) < 0.35 and np.ptp(best["h"]) < 0.35:
        warnings.append("Trajectory covers only a short arc/extent; confidence is reduced.")

    pitch = best["model"]["pitch"] if best_label == "screw" else None
    circle_center = None
    estimated_motion = {}
    if best_label == "circle":
        circle_center = best["axis"]["axis_point"] + best["model"]["h0"] * best["axis"]["axis_direction"]
        estimated_motion = {
            "angle_start_rad": float(best["phi"][0]),
            "angle_end_rad": float(best["phi"][-1]),
            "delta_angle_rad": float(best["phi"][-1] - best["phi"][0]),
        }
    elif best_label == "translation":
        estimated_motion = {
            "delta_displacement": float(best["h"][-1] - best["h"][0]),
        }
    elif best_label == "screw":
        estimated_motion = {
            "delta_angle_rad": float(best["phi"][-1] - best["phi"][0]),
            "delta_displacement": float(best["h"][-1] - best["h"][0]),
        }
    result = ClassificationResult(
        label=best_label,
        axis_direction=best["axis"]["axis_direction"],
        axis_point=best["axis"]["axis_point"],
        radius=best["axis"]["radius"],
        pitch=pitch,
        residuals=normalized_residuals,
        confidence=max(0.0, min(1.0, confidence)),
        rho=best["rho"],
        phi=best["phi"],
        h=best["h"],
        warnings=tuple(warnings),
    )
    result_dict = result.to_dict()
    result_dict["circle_center"] = circle_center
    result_dict["estimated_motion"] = estimated_motion
    return result_dict
