"""Demo for axis-constrained trajectory classification."""

import argparse
import json
import os
import random

import numpy as np

from simulate_trajectories import (
    generate_circle_trajectory,
    generate_screw_trajectory,
    generate_translation_trajectory,
)
from trajectory_constraint_fit import classify_trajectory


def _format_vector(v: np.ndarray) -> str:
    return np.array2string(v, precision=4, suppress_small=True)


def _print_result(name: str, result):
    print(f"\n=== {name} ===")
    print(f"predicted label: {result['label']}")
    print(f"axis direction: {_format_vector(result['axis_direction'])}")
    print(f"axis point: {_format_vector(result['axis_point'])}")
    if result["label"] == "circle":
        print(f"fitted radius: {result['radius']:.4f}")
        print(f"fitted circle center: {_format_vector(result['circle_center'])}")
        print(f"estimated angle start: {result['estimated_motion']['angle_start_rad']:.4f} rad")
        print(f"estimated angle end: {result['estimated_motion']['angle_end_rad']:.4f} rad")
        print(f"estimated delta angle: {result['estimated_motion']['delta_angle_rad']:.4f} rad")
    elif result["label"] == "translation":
        print(f"estimated delta displacement: {result['estimated_motion']['delta_displacement']:.4f}")
    elif result["label"] == "screw":
        print(f"fitted radius: {result['radius']:.4f}")
        print(f"fitted pitch: {result['pitch']:.6f}")
        print(f"estimated delta angle: {result['estimated_motion']['delta_angle_rad']:.4f} rad")
        print(f"estimated delta displacement: {result['estimated_motion']['delta_displacement']:.4f}")
    print("normalized residuals:")
    for label, residual in result["residuals"].items():
        print(f"  {label}: {residual:.6f}")
    print(f"confidence: {result['confidence']:.4f}")
    if result["warnings"]:
        print("warnings:")
        for warning in result["warnings"]:
            print(f"  - {warning}")


def _axis_direction_alignment(a: np.ndarray, b: np.ndarray) -> float:
    return float(abs(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b))))


def _percent_error(value: float, reference: float) -> float:
    reference = max(abs(reference), 1e-9)
    return 100.0 * abs(value) / reference


def _print_ground_truth_comparison(result, metadata):
    if metadata is None:
        return
    alignment = _axis_direction_alignment(result["axis_direction"], metadata["axis_direction"])
    print("ground-truth comparison:")
    print(f"  axis direction alignment: {100.0 * alignment:.4f}%")
    if metadata.get("motion_type") == "translation":
        print("  axis point/radius: not uniquely identifiable from a single translated point")
    else:
        axis_point_error = float(np.linalg.norm(result["axis_point"] - metadata["axis_point"]))
        radius_error = float(abs(result["radius"] - metadata["radius"]))
        axis_point_error_pct = _percent_error(axis_point_error, metadata["radius"])
        radius_error_pct = _percent_error(radius_error, metadata["radius"])
        print(f"  axis point error: {axis_point_error_pct:.4f}% of true radius")
        print(f"  radius error: {radius_error_pct:.4f}%")
        if metadata.get("motion_type") == "screw":
            pitch_error = float(abs(result["pitch"] - metadata["pitch"]))
            pitch_error_pct = _percent_error(pitch_error, metadata["pitch"])
            print(f"  pitch error: {pitch_error_pct:.4f}%")


def _set_equal_3d_axes(ax, points: np.ndarray):
    center = points.mean(axis=0)
    extents = np.ptp(points, axis=0)
    radius = 0.5 * np.max(extents)
    radius = max(radius, 1.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    try:
        ax.set_proj_type("ortho")
    except TypeError:
        pass


def _reconstruct_predicted_motion(result):
    axis_point = result["axis_point"]
    axis_direction = result["axis_direction"]
    frame = np.column_stack(
        (
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )
    )
    from trajectory_constraint_fit import build_axis_frame

    frame = build_axis_frame(axis_direction)
    num_samples = len(result["rho"])
    radius = float(result["radius"])

    if result["label"] == "circle":
        phi = result["phi"]
        h = np.full(num_samples, np.mean(result["h"]))
    elif result["label"] == "translation":
        phi = np.full(num_samples, np.mean(result["phi"]))
        h = np.linspace(np.min(result["h"]), np.max(result["h"]), num_samples)
    else:
        phi = result["phi"]
        pitch = 0.0 if result["pitch"] is None else float(result["pitch"])
        h0 = float(np.mean(result["h"] - pitch * phi))
        h = h0 + pitch * phi

    local = np.column_stack((radius * np.cos(phi), radius * np.sin(phi), h))
    return axis_point + local @ frame.T


def _fit_error_stats(points: np.ndarray, fitted_curve: np.ndarray):
    distances = np.linalg.norm(points - fitted_curve, axis=1)
    return float(np.mean(distances)), float(np.max(distances))


def _load_points(path: str) -> np.ndarray:
    extension = os.path.splitext(path)[1].lower()
    if extension == ".csv":
        points = np.loadtxt(path, delimiter=",")
    elif extension == ".npy":
        points = np.load(path)
    elif extension == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            if "points" not in payload:
                raise ValueError("JSON trajectory file must contain a 'points' field.")
            payload = payload["points"]
        points = np.asarray(payload, dtype=float)
    else:
        raise ValueError("Unsupported trajectory format. Use .csv, .npy, or .json.")

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Trajectory file must contain an Nx3 array of xyz samples.")
    return points


def _plot_3d_results(dataset):
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(16, 5))
    for column, (name, points, result, metadata) in enumerate(dataset, start=1):
        ax3d = figure.add_subplot(1, len(dataset), column, projection="3d")
        ax3d.plot(points[:, 0], points[:, 1], points[:, 2], color="tab:blue", lw=1.8, label="samples")
        ax3d.scatter(points[:, 0], points[:, 1], points[:, 2], s=12, color="tab:blue", alpha=0.75)

        axis_point = result["axis_point"]
        axis_direction = result["axis_direction"]
        span = max(np.ptp(points, axis=0).max(), 1.0)
        t = np.linspace(-0.8 * span, 0.8 * span, 2)
        axis_line = axis_point + np.outer(t, axis_direction)
        fitted_curve = _reconstruct_predicted_motion(result)
        mean_fit_error, max_fit_error = _fit_error_stats(points, fitted_curve)

        ax3d.plot(
            axis_line[:, 0],
            axis_line[:, 1],
            axis_line[:, 2],
            color="tab:red",
            lw=2.2,
            label="fitted axis",
        )
        ax3d.plot(
            fitted_curve[:, 0],
            fitted_curve[:, 1],
            fitted_curve[:, 2],
            color="tab:orange",
            lw=1.6,
            linestyle="--",
            label="predicted motion",
        )
        show_true_axis = metadata is not None and metadata.get("motion_type") != "translation"
        if show_true_axis:
            true_axis_line = metadata["axis_point"] + np.outer(t, metadata["axis_direction"])
            ax3d.plot(
                true_axis_line[:, 0],
                true_axis_line[:, 1],
                true_axis_line[:, 2],
                color="tab:green",
                lw=1.8,
                linestyle=":",
                label="true axis",
            )
        ax3d.scatter(
            axis_point[0],
            axis_point[1],
            axis_point[2],
            color="black",
            s=30,
            label="axis point",
        )
        ax3d.set_title(f"{name} -> {result['label']}")
        ax3d.set_xlabel("x")
        ax3d.set_ylabel("y")
        ax3d.set_zlabel("z")
        ax3d.view_init(elev=18, azim=-62)
        plotted_points = [points, axis_line, fitted_curve]
        if show_true_axis:
            plotted_points.append(true_axis_line)
        _set_equal_3d_axes(ax3d, np.vstack(plotted_points))
        note = "mean fit err: %.4f\nmax fit err: %.4f" % (mean_fit_error, max_fit_error)
        if metadata is not None and metadata.get("motion_type") == "translation":
            note += "\ntranslation axis point is non-unique"
        ax3d.text2D(
            0.03,
            0.03,
            note,
            transform=ax3d.transAxes,
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
        ax3d.legend(loc="upper right", fontsize=8)

    figure.suptitle("Trajectory classification with fitted 3D axis models", fontsize=14)
    figure.tight_layout()
    plt.show()


def _random_dataset(seed: int = None):
    if seed is None:
        seed = random.randint(0, 10**6)

    rng = random.Random(seed)
    generators = {
        "circle": generate_circle_trajectory,
        "translation": generate_translation_trajectory,
        "screw": generate_screw_trajectory,
    }
    chosen_label = rng.choice(list(generators))

    generator_kwargs = {"seed": seed, "return_params": True}
    if chosen_label == "circle":
        generator_kwargs["turns"] = rng.uniform(0.35, 1.85)
    elif chosen_label == "screw":
        generator_kwargs["turns"] = rng.uniform(0.5, 2.25)

    points, metadata = generators[chosen_label](**generator_kwargs)
    return chosen_label, points, metadata, seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plot", action="store_true", help="Skip the 3D matplotlib visualization.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for the demo trajectory.")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to a real trajectory file (.csv, .npy, or .json) containing an Nx3 point array.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run the original three-trajectory showcase instead of a single random sample.",
    )
    args = parser.parse_args()

    if args.input is not None:
        points = _load_points(args.input)
        datasets = [(os.path.basename(args.input), points, None)]
    elif args.all:
        datasets = [
            ("circle",) + generate_circle_trajectory(return_params=True),
            ("translation",) + generate_translation_trajectory(return_params=True),
            ("screw",) + generate_screw_trajectory(return_params=True),
        ]
    else:
        chosen_label, points, metadata, used_seed = _random_dataset(seed=args.seed)
        print(f"randomly selected trajectory: {chosen_label}")
        print(f"seed: {used_seed}")
        datasets = [(chosen_label, points, metadata)]

    results = []
    for name, points, metadata in datasets:
        result = classify_trajectory(points)
        _print_result(name, result)
        _print_ground_truth_comparison(result, metadata)
        results.append((name, points, result, metadata))

    if not args.no_plot:
        _plot_3d_results(results)


if __name__ == "__main__":
    main()
