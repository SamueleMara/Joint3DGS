from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from articulation.data.dataclasses import MultiViewRGBDSequence, RGBDSequence
from articulation.data.io_cameras import expand_intrinsics_for_multiview, load_extrinsics_as_T_cw, load_intrinsics


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"}


def _sorted_paths(dir_path: Path) -> list[Path]:
    return sorted([p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS])


def _read_rgb(path: Path) -> torch.Tensor:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


def _read_depth(path: Path, depth_scale: float, depth_npy_scale: float) -> torch.Tensor:
    if path.suffix.lower() == ".npy":
        d = np.load(path).astype(np.float32)
        if d.ndim == 3 and d.shape[0] == 1:
            d = d[0]
        if d.ndim != 2:
            raise ValueError(f"Depth npy must be [H,W], got {d.shape} at {path}")
        depth = d * float(depth_npy_scale)
    else:
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"Could not read depth image: {path}")
        depth = raw.astype(np.float32) * float(depth_scale)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth = np.maximum(depth, 0.0)
    return torch.from_numpy(depth).unsqueeze(0)


def _read_mask(path: Path) -> torch.Tensor:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Could not read foreground mask: {path}")
    return torch.from_numpy((m > 127).astype(np.float32)).unsqueeze(0)


def _is_single_view_flat(dir_path: Path) -> bool:
    if not dir_path.is_dir():
        return False
    files = [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS]
    subdirs = [p for p in dir_path.iterdir() if p.is_dir()]
    return len(files) > 0 and len(subdirs) == 0


def _looks_like_multiview_root(dir_path: Path) -> bool:
    if not dir_path.is_dir():
        return False
    subs = [p.name for p in dir_path.iterdir() if p.is_dir()]
    return len(subs) > 0 and all(s.startswith("cam_") for s in subs)


def _resolve_dataset_root_and_rgb_dir(root: Path, rgb_dir: str) -> tuple[Path, str]:
    if (root / rgb_dir).is_dir():
        return root, rgb_dir

    # Allow passing the rgb folder directly (e.g. .../fr3_joint1/rgb).
    if _looks_like_multiview_root(root):
        parent = root.parent
        return parent, root.name

    for cand in ["rgb", "images"]:
        if (root / cand).is_dir():
            return root, cand

    return root, rgb_dir


def _resolve_existing_subdir(root: Path, requested: str, candidates: list[str], label: str) -> str:
    if (root / requested).is_dir():
        return requested
    for cand in candidates:
        if (root / cand).is_dir():
            return cand
    raise FileNotFoundError(f"Could not find {label} directory under {root}. Tried {[requested] + candidates}")


def _collect_views(root: Path, subdir_name: str) -> list[tuple[int, str, Path]]:
    p = root / subdir_name
    if not p.is_dir():
        raise FileNotFoundError(f"Missing directory: {p}")

    if _is_single_view_flat(p):
        return [(0, "cam_000", p)]

    views = sorted([d for d in p.iterdir() if d.is_dir()])
    if not views:
        raise ValueError(f"No view folders found in {p}")
    return [(i, v.name, v) for i, v in enumerate(views)]


def _align_triplets(rgb_files: list[Path], depth_files: list[Path], mask_files: list[Path]) -> tuple[list[Path], list[Path], list[Path]]:
    rgb_map = {p.stem: p for p in rgb_files}
    depth_map = {p.stem: p for p in depth_files}
    mask_map = {p.stem: p for p in mask_files}

    keys = sorted(set(rgb_map) & set(depth_map) & set(mask_map))
    if not keys:
        raise ValueError("No common frame stems across rgb/depth/fg_mask")

    return [rgb_map[k] for k in keys], [depth_map[k] for k in keys], [mask_map[k] for k in keys]


def _normalize_np(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Cannot normalize near-zero vector")
    return v / n


def _camera_intrinsics_from_entry(entry: dict) -> np.ndarray:
    w = int(entry["width"])
    h = int(entry["height"])
    fx = entry.get("fx", None)
    fy = entry.get("fy", None)
    cx = entry.get("cx", None)
    cy = entry.get("cy", None)

    if fx is None or fy is None:
        fov_y_deg = float(entry.get("fov_y_deg", 45.0))
        fy_f = 0.5 * h / math.tan(math.radians(fov_y_deg) * 0.5)
        fx_f = fy_f
        cx_f = (w - 1) * 0.5
        cy_f = (h - 1) * 0.5
    else:
        fx_f = float(fx)
        fy_f = float(fy)
        cx_f = float((w - 1) * 0.5 if cx is None else cx)
        cy_f = float((h - 1) * 0.5 if cy is None else cy)

    return np.array([[fx_f, 0.0, cx_f], [0.0, fy_f, cy_f], [0.0, 0.0, 1.0]], dtype=np.float32)


def _lookat_world_from_camera(position: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    z = _normalize_np(target - position)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-8:
        alt_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        x = np.cross(z, alt_up)
    x = _normalize_np(x)
    y = _normalize_np(np.cross(z, x))

    t = np.eye(4, dtype=np.float32)
    t[:3, :3] = np.stack([x, y, z], axis=1)
    t[:3, 3] = position.astype(np.float32)
    return t


def _invert_4x4(t: np.ndarray) -> np.ndarray:
    r = t[:3, :3]
    tr = t[:3, 3]
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = r.T
    out[:3, 3] = -(r.T @ tr)
    return out


def _world_from_camera_from_entry(entry: dict, extrinsics_convention: str) -> np.ndarray:
    pose = entry.get("pose_matrix_4x4", None)
    if pose is not None:
        mat = np.asarray(pose, dtype=np.float32)
        if mat.shape != (4, 4):
            raise ValueError(f"pose_matrix_4x4 must be [4,4], got {mat.shape}")
        if extrinsics_convention == "world_from_camera":
            return mat
        if extrinsics_convention == "camera_from_world":
            return _invert_4x4(mat)
        raise ValueError(f"Unknown extrinsics convention: {extrinsics_convention}")

    position = np.asarray(entry.get("position", [0.0, 0.0, 0.0]), dtype=np.float32)
    target = np.asarray(entry.get("target", [0.0, 0.0, 1.0]), dtype=np.float32)
    up = np.asarray(entry.get("up", [0.0, 0.0, 1.0]), dtype=np.float32)
    return _lookat_world_from_camera(position, target, up)


def _load_camera_entries(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text())
    cams = payload.get("cameras", [])
    out: dict[str, dict] = {}
    for c in cams:
        name = str(c.get("name", ""))
        if name:
            out[name] = c
    if not out:
        raise ValueError(f"No cameras found in metadata file: {path}")
    return out


def load_multiview_sequence_from_folder(
    root: str | Path,
    rgb_dir: str = "images",
    depth_dir: str = "depth",
    mask_dir: str = "fg_mask",
    intrinsics_file: str = "intrinsics.npy",
    extrinsics_file: str = "extrinsics.npy",
    depth_scale: float = 1.0,
    max_frames: int | None = None,
    cameras_json: str = "metadata/cameras.json",
    extrinsics_convention: str = "world_from_camera",
    depth_npy_scale: float = 1.0,
) -> MultiViewRGBDSequence:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    root, rgb_dir = _resolve_dataset_root_and_rgb_dir(root, rgb_dir)
    depth_dir = _resolve_existing_subdir(root, depth_dir, ["depth_npy", "depth_png", "depth"], "depth")
    mask_dir = _resolve_existing_subdir(root, mask_dir, ["mask", "fg_mask"], "mask")

    rgb_views = _collect_views(root, rgb_dir)
    depth_views = _collect_views(root, depth_dir)
    mask_views = _collect_views(root, mask_dir)

    rgb_by_name = {name: p for _, name, p in rgb_views}
    depth_by_name = {name: p for _, name, p in depth_views}
    mask_by_name = {name: p for _, name, p in mask_views}

    common_names = sorted(set(rgb_by_name) & set(depth_by_name) & set(mask_by_name))
    if not common_names:
        raise ValueError("No common camera names across rgb/depth/fg_mask")

    view_ids: list[int] = []
    view_names: list[str] = []
    rgb_all: list[torch.Tensor] = []
    depth_all: list[torch.Tensor] = []
    mask_all: list[torch.Tensor] = []
    frame_ids: list[int] | None = None

    for vidx, name in enumerate(common_names):
        rgb_files = _sorted_paths(rgb_by_name[name])
        depth_files = _sorted_paths(depth_by_name[name])
        mask_files = _sorted_paths(mask_by_name[name])

        rgb_files, depth_files, mask_files = _align_triplets(rgb_files, depth_files, mask_files)

        if max_frames is not None:
            rgb_files = rgb_files[: int(max_frames)]
            depth_files = depth_files[: int(max_frames)]
            mask_files = mask_files[: int(max_frames)]

        rgb_v = torch.stack([_read_rgb(p) for p in rgb_files], dim=0)
        depth_v = torch.stack(
            [_read_depth(p, depth_scale=float(depth_scale), depth_npy_scale=float(depth_npy_scale)) for p in depth_files],
            dim=0,
        )
        mask_v = torch.stack([_read_mask(p) for p in mask_files], dim=0)

        if rgb_v.shape[0] == 0:
            raise ValueError(f"No frames found for view {name}")
        if rgb_v.shape[-2:] != depth_v.shape[-2:] or rgb_v.shape[-2:] != mask_v.shape[-2:]:
            raise ValueError(f"Resolution mismatch in view {name}")

        if frame_ids is None:
            frame_ids = list(range(int(rgb_v.shape[0])))
        elif len(frame_ids) != int(rgb_v.shape[0]):
            raise ValueError("All views must have the same frame count")

        view_ids.append(vidx)
        view_names.append(name)
        rgb_all.append(rgb_v)
        depth_all.append(depth_v)
        mask_all.append(mask_v)

    assert frame_ids is not None
    T = len(frame_ids)
    V = len(view_ids)

    rgb = torch.stack(rgb_all, dim=1)  # [T,V,3,H,W]
    depth = torch.stack(depth_all, dim=1)
    fg_mask = torch.stack(mask_all, dim=1)

    intr_path = root / intrinsics_file
    extr_path = root / extrinsics_file

    if intr_path.exists() and extr_path.exists():
        K_raw = load_intrinsics(intr_path).float()
        K = expand_intrinsics_for_multiview(K_raw, T=T, V=V)
        T_cw = load_extrinsics_as_T_cw(extr_path, T=T, V=V).float()
    else:
        cam_path = root / cameras_json
        if not cam_path.exists():
            raise FileNotFoundError(
                f"Missing camera calibration files. Expected either ({intr_path}, {extr_path}) "
                f"or metadata file {cam_path}"
            )

        entries = _load_camera_entries(cam_path)
        K_list: list[np.ndarray] = []
        T_cw_static_list: list[np.ndarray] = []
        for name in common_names:
            if name not in entries:
                raise KeyError(f"Camera {name} missing from metadata file: {cam_path}")
            e = entries[name]
            K_list.append(_camera_intrinsics_from_entry(e))
            T_wc = _world_from_camera_from_entry(e, extrinsics_convention=extrinsics_convention)
            T_cw_static_list.append(_invert_4x4(T_wc))

        K = torch.from_numpy(np.stack(K_list, axis=0)).float()  # [V,3,3]
        T_cw_static = torch.from_numpy(np.stack(T_cw_static_list, axis=0)).float()  # [V,4,4]
        T_cw = T_cw_static.unsqueeze(0).expand(T, -1, -1, -1).contiguous()  # [T,V,4,4]

    return MultiViewRGBDSequence(
        rgb=rgb,
        depth=depth,
        fg_mask=fg_mask,
        K=K,
        T_cw=T_cw,
        frame_ids=frame_ids,
        view_ids=view_ids,
        meta={
            "root": str(root),
            "rgb_dir": str(root / rgb_dir),
            "depth_dir": str(root / depth_dir),
            "mask_dir": str(root / mask_dir),
            "view_names": view_names,
            "intrinsics_file": str(intr_path),
            "extrinsics_file": str(extr_path),
            "cameras_json": str(root / cameras_json),
            "depth_scale": float(depth_scale),
            "depth_npy_scale": float(depth_npy_scale),
        },
    )


def single_to_multiview_sequence(sequence: RGBDSequence) -> MultiViewRGBDSequence:
    """Wrap a single-view RGBDSequence into V=1 multiview format."""
    T = sequence.T
    rgb = sequence.rgb.unsqueeze(1)       # [T,1,3,H,W]
    depth = sequence.depth.unsqueeze(1)   # [T,1,1,H,W]
    fg_mask = sequence.fg_mask.unsqueeze(1)

    if sequence.K.ndim == 2:
        K = sequence.K.unsqueeze(0)  # [1,3,3]
    elif sequence.K.ndim == 3:
        if sequence.K.shape[0] == T:
            K = sequence.K.unsqueeze(1)  # [T,1,3,3]
        else:
            raise ValueError(f"Single-view sequence K must be [3,3] or [T,3,3], got {sequence.K.shape}")
    else:
        raise ValueError(f"Unsupported single-view K shape: {sequence.K.shape}")

    T_cw = torch.eye(4, dtype=sequence.rgb.dtype).unsqueeze(0).unsqueeze(0).repeat(T, 1, 1, 1)
    return MultiViewRGBDSequence(
        rgb=rgb,
        depth=depth,
        fg_mask=fg_mask,
        K=K,
        T_cw=T_cw,
        frame_ids=list(sequence.frame_ids),
        view_ids=[0],
        meta={**dict(sequence.meta), "wrapped_single_view": True},
    )
