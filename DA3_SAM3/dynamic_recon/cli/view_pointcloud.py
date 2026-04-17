from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize static/dynamic point clouds with estimated camera poses.")
    parser.add_argument("--run-dir", default="outputs/run_sam3")
    parser.add_argument("--exports-dir", default=None)
    parser.add_argument("--backend", choices=("auto", "open3d", "matplotlib"), default="auto")
    parser.add_argument("--max-points", type=int, default=120000)
    parser.add_argument("--point-size", type=float, default=0.6)
    parser.add_argument("--save", default=None, help="Optional image path for matplotlib backend.")
    parser.add_argument("--no-show", action="store_true", help="Do not open an interactive window (matplotlib only).")
    args = parser.parse_args()

    exports_dir = Path(args.exports_dir) if args.exports_dir else Path(args.run_dir) / "exports"
    static_ply = exports_dir / "scene_static_points.ply"
    dynamic_ply = exports_dir / "scene_dynamic_points.ply"
    if not static_ply.exists() or not dynamic_ply.exists():
        raise FileNotFoundError(
            f"Missing pointcloud files in {exports_dir}. Expected "
            "`scene_static_points.ply` and `scene_dynamic_points.ply`. "
            "Run pipeline export first."
        )
    camera_centers = _load_camera_centers(exports_dir)
    static_xyz, static_rgb = _load_ascii_ply(static_ply)
    dynamic_xyz, dynamic_rgb = _load_ascii_ply(dynamic_ply)
    static_xyz, static_rgb = _downsample(static_xyz, static_rgb, int(args.max_points), seed=0)
    dynamic_xyz, dynamic_rgb = _downsample(dynamic_xyz, dynamic_rgb, int(args.max_points), seed=1)

    backend = args.backend
    if backend == "auto":
        backend = "open3d" if _has_open3d() else "matplotlib"
    if backend == "open3d":
        _view_open3d(static_xyz, static_rgb, dynamic_xyz, dynamic_rgb, camera_centers)
        return
    _view_matplotlib(
        static_xyz,
        static_rgb,
        dynamic_xyz,
        dynamic_rgb,
        camera_centers,
        point_size=float(args.point_size),
        save_path=None if args.save is None else Path(args.save),
        show=not bool(args.no_show),
        default_save=exports_dir / "scene_view_cli.png",
    )


def _has_open3d() -> bool:
    try:
        import open3d  # noqa: F401

        return True
    except Exception:
        return False


def _load_camera_centers(exports_dir: Path) -> np.ndarray:
    camera_centers_path = exports_dir / "camera_centers.npy"
    if camera_centers_path.exists():
        return np.load(camera_centers_path)
    poses_path = exports_dir / "camera_poses.npy"
    if not poses_path.exists():
        raise FileNotFoundError(
            f"Missing `{camera_centers_path}` and `{poses_path}` in {exports_dir}. "
            "Run pipeline export first."
        )
    poses = np.load(poses_path)
    centers = []
    for pose in poses:
        c2w = np.linalg.inv(pose)
        centers.append(c2w[:3, 3])
    return np.asarray(centers, dtype=np.float32)


def _load_ascii_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as handle:
        header_lines = 0
        vertex_count = 0
        for line in handle:
            header_lines += 1
            stripped = line.strip()
            if stripped.startswith("element vertex"):
                parts = stripped.split()
                if len(parts) >= 3:
                    vertex_count = int(parts[-1])
            if stripped == "end_header":
                break
    if vertex_count <= 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    data = np.loadtxt(path, skiprows=header_lines, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    xyz = data[:, :3].astype(np.float32)
    if data.shape[1] >= 6:
        rgb = np.clip(data[:, 3:6], 0, 255).astype(np.uint8)
    else:
        rgb = np.full((xyz.shape[0], 3), 255, dtype=np.uint8)
    return xyz, rgb


def _downsample(xyz: np.ndarray, rgb: np.ndarray, max_points: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or xyz.shape[0] <= max_points:
        return xyz, rgb
    rng = np.random.default_rng(seed)
    keep = rng.choice(xyz.shape[0], size=max_points, replace=False)
    return xyz[keep], rgb[keep]


def _view_open3d(
    static_xyz: np.ndarray,
    static_rgb: np.ndarray,
    dynamic_xyz: np.ndarray,
    dynamic_rgb: np.ndarray,
    camera_centers: np.ndarray,
) -> None:
    try:
        import open3d as o3d
    except Exception as exc:
        raise RuntimeError("Open3D backend requested, but `open3d` is not installed.") from exc

    geometries = []
    if static_xyz.shape[0] > 0:
        static_pcd = o3d.geometry.PointCloud()
        static_pcd.points = o3d.utility.Vector3dVector(static_xyz.astype(np.float64))
        static_pcd.colors = o3d.utility.Vector3dVector((static_rgb.astype(np.float32) / 255.0).astype(np.float64))
        geometries.append(static_pcd)
    if dynamic_xyz.shape[0] > 0:
        dynamic_pcd = o3d.geometry.PointCloud()
        dynamic_pcd.points = o3d.utility.Vector3dVector(dynamic_xyz.astype(np.float64))
        dynamic_pcd.colors = o3d.utility.Vector3dVector((dynamic_rgb.astype(np.float32) / 255.0).astype(np.float64))
        geometries.append(dynamic_pcd)

    if camera_centers.shape[0] > 0:
        cam_pcd = o3d.geometry.PointCloud()
        cam_pcd.points = o3d.utility.Vector3dVector(camera_centers.astype(np.float64))
        cam_pcd.paint_uniform_color([0.0, 1.0, 1.0])
        geometries.append(cam_pcd)
        if camera_centers.shape[0] > 1:
            lines = [[idx, idx + 1] for idx in range(camera_centers.shape[0] - 1)]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(camera_centers.astype(np.float64))
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.0, 0.0, 0.0]], dtype=np.float64), (len(lines), 1)))
            geometries.append(line_set)

    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2))
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Static/Dynamic Point Clouds + Estimated Cameras",
        width=1600,
        height=1000,
    )


def _view_matplotlib(
    static_xyz: np.ndarray,
    static_rgb: np.ndarray,
    dynamic_xyz: np.ndarray,
    dynamic_rgb: np.ndarray,
    camera_centers: np.ndarray,
    *,
    point_size: float,
    save_path: Path | None,
    show: bool,
    default_save: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    if static_xyz.shape[0] > 0:
        ax.scatter(
            static_xyz[:, 0],
            static_xyz[:, 1],
            static_xyz[:, 2],
            c=static_rgb.astype(np.float32) / 255.0,
            s=point_size,
            alpha=0.45,
            linewidths=0,
            label="static",
        )
    if dynamic_xyz.shape[0] > 0:
        ax.scatter(
            dynamic_xyz[:, 0],
            dynamic_xyz[:, 1],
            dynamic_xyz[:, 2],
            c=dynamic_rgb.astype(np.float32) / 255.0,
            s=point_size,
            alpha=0.8,
            linewidths=0,
            label="dynamic",
        )
    if camera_centers.shape[0] > 0:
        ax.plot(camera_centers[:, 0], camera_centers[:, 1], camera_centers[:, 2], color="black", linewidth=1.6, label="camera path")
        ax.scatter(camera_centers[:, 0], camera_centers[:, 1], camera_centers[:, 2], c="cyan", s=20, edgecolors="black", linewidths=0.4)

    ax.set_title("Static/Dynamic Point Clouds + Estimated Cameras")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    _set_axes_equal(ax, static_xyz, dynamic_xyz, camera_centers)
    ax.legend(loc="upper right")
    fig.tight_layout()

    target = save_path if save_path is not None else default_save
    fig.savefig(target, dpi=180)
    print(f"Saved matplotlib view to: {target}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _set_axes_equal(ax, *clouds: np.ndarray) -> None:
    valid = [cloud for cloud in clouds if cloud is not None and cloud.size > 0]
    if not valid:
        return
    stacked = np.concatenate(valid, axis=0)
    mins = np.nanmin(stacked, axis=0)
    maxs = np.nanmax(stacked, axis=0)
    center = 0.5 * (mins + maxs)
    radius = max(1.0e-3, 0.5 * float(np.max(maxs - mins)))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


if __name__ == "__main__":
    main()
