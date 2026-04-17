from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(arr, dtype=np.float32))


def _load_array(path: str | Path) -> np.ndarray:
    p = Path(path)
    s = p.suffix.lower()
    if s == ".npy":
        return np.load(p)
    if s == ".npz":
        data = np.load(p)
        if "K" in data:
            return data["K"]
        if "intrinsics" in data:
            return data["intrinsics"]
        if "T_cw" in data:
            return data["T_cw"]
        if "extrinsics" in data:
            return data["extrinsics"]
        first = list(data.keys())[0]
        return data[first]
    if s == ".json":
        payload = json.loads(p.read_text())
        if "K" in payload:
            return np.asarray(payload["K"], dtype=np.float32)
        if "intrinsics" in payload:
            return np.asarray(payload["intrinsics"], dtype=np.float32)
        if "T_cw" in payload:
            return np.asarray(payload["T_cw"], dtype=np.float32)
        if "extrinsics" in payload:
            return np.asarray(payload["extrinsics"], dtype=np.float32)
        raise ValueError(f"Unsupported JSON camera schema in {p}")
    return np.loadtxt(p, dtype=np.float32)


def load_intrinsics(path: str | Path) -> torch.Tensor:
    arr = np.asarray(_load_array(path), dtype=np.float32)
    if arr.shape[-2:] != (3, 3):
        raise ValueError(f"Intrinsics must end with [3,3], got {arr.shape}")
    return _to_tensor(arr)


def _to_4x4(extr: np.ndarray) -> np.ndarray:
    if extr.shape[-2:] == (4, 4):
        return extr.astype(np.float32)
    if extr.shape[-2:] == (3, 4):
        out = np.zeros((*extr.shape[:-2], 4, 4), dtype=np.float32)
        out[..., :3, :4] = extr.astype(np.float32)
        out[..., 3, 3] = 1.0
        return out
    raise ValueError(f"Extrinsics must end with [3,4] or [4,4], got {extr.shape}")


def expand_intrinsics_for_multiview(K: torch.Tensor, T: int, V: int) -> torch.Tensor:
    """Normalize K to [T,V,3,3] or [V,3,3] if static per-view."""
    if K.ndim == 2:
        if K.shape != (3, 3):
            raise ValueError(f"K must be [3,3], got {K.shape}")
        return K.unsqueeze(0).expand(V, -1, -1)

    if K.ndim == 3:
        if K.shape[-2:] != (3, 3):
            raise ValueError(f"K must end with [3,3], got {K.shape}")
        if K.shape[0] == V:
            return K
        if K.shape[0] == T:
            return K.unsqueeze(1).expand(-1, V, -1, -1)
        if K.shape[0] == 1:
            return K.expand(V, -1, -1)
        raise ValueError(f"Could not broadcast K with shape {K.shape} to T={T}, V={V}")

    if K.ndim == 4:
        if K.shape[:2] != (T, V) or K.shape[-2:] != (3, 3):
            raise ValueError(f"K must be [T,V,3,3], got {K.shape} for T={T}, V={V}")
        return K

    raise ValueError(f"Unsupported K shape: {K.shape}")


def load_extrinsics_as_T_cw(path: str | Path, T: int, V: int) -> torch.Tensor:
    arr = np.asarray(_load_array(path), dtype=np.float32)
    arr = _to_4x4(arr)

    if arr.ndim == 2:
        arr = arr[None, None, ...]
    elif arr.ndim == 3:
        if arr.shape[0] == T:
            arr = arr[:, None, ...]
        elif arr.shape[0] == V:
            arr = arr[None, ...]
        elif arr.shape[0] == 1:
            arr = arr[None, ...]
        else:
            raise ValueError(f"Ambiguous extrinsics shape {arr.shape} for T={T}, V={V}")
    elif arr.ndim == 4:
        pass
    else:
        raise ValueError(f"Unsupported extrinsics shape: {arr.shape}")

    if arr.ndim == 4:
        if arr.shape[0] == 1:
            arr = np.repeat(arr, T, axis=0)
        if arr.shape[1] == 1:
            arr = np.repeat(arr, V, axis=1)

    if arr.shape != (T, V, 4, 4):
        raise ValueError(f"Extrinsics could not be broadcast to [T,V,4,4], got {arr.shape}")

    return _to_tensor(arr)
