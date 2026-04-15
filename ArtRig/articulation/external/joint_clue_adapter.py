from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Optional
import sys

import numpy as np



def _resolve_callable(dotted: str) -> Callable[[np.ndarray], dict[str, Any]]:
    if ":" not in dotted:
        raise ValueError("backend_callable must be in format 'module.submodule:function'")
    mod_name, fn_name = dotted.split(":", 1)
    mod = import_module(mod_name)
    fn = getattr(mod, fn_name)
    if not callable(fn):
        raise TypeError(f"Loaded backend target is not callable: {dotted}")
    return fn



def _maybe_add_repo_path(path: str | None) -> None:
    if not path:
        return
    p = str(Path(path).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


@dataclass
class JointClueEstimator:
    """Adapter around external point-trajectory joint clue module.

    If no backend is provided, falls back to a lightweight heuristic for development.
    """

    backend: Optional[Callable[[np.ndarray], dict[str, Any]]] = None

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> "JointClueEstimator":
        cfg = cfg or {}
        repo_path = cfg.get("repo_path")
        callable_name = cfg.get("backend_callable")
        _maybe_add_repo_path(repo_path)

        if callable_name:
            backend = _resolve_callable(str(callable_name))
            return cls(backend=backend)
        return cls(backend=None)

    def infer(self, xyz_traj: np.ndarray) -> dict[str, Any]:
        if xyz_traj.ndim != 2 or xyz_traj.shape[1] != 3:
            raise ValueError("xyz_traj must be [T,3]")

        if self.backend is not None:
            raw = self.backend(xyz_traj)
            return self._normalize_output(raw)

        return self._heuristic_output(xyz_traj)

    @staticmethod
    def _normalize_output(raw: dict[str, Any]) -> dict[str, Any]:
        type_scores = raw.get("type_scores") or {}
        out = {
            "type_scores": {
                "revolute": float(type_scores.get("revolute", 0.33)),
                "prismatic": float(type_scores.get("prismatic", 0.33)),
                "screw": float(type_scores.get("screw", 0.34)),
            },
            "axis_dir": raw.get("axis_dir", None),
            "axis_point": raw.get("axis_point", None),
            "pitch": raw.get("pitch", None),
            "confidence": float(raw.get("confidence", 1.0)),
        }
        axis_dir = out["axis_dir"]
        if axis_dir is None:
            out["axis_dir"] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            axis_dir = np.asarray(axis_dir, dtype=np.float32)
            n = np.linalg.norm(axis_dir) + 1e-8
            out["axis_dir"] = axis_dir / n
        if out["axis_point"] is not None:
            out["axis_point"] = np.asarray(out["axis_point"], dtype=np.float32)
        if out["pitch"] is not None:
            out["pitch"] = float(out["pitch"])
        return out

    @staticmethod
    def _heuristic_output(xyz_traj: np.ndarray) -> dict[str, Any]:
        d = np.diff(xyz_traj, axis=0)
        speed = np.linalg.norm(d, axis=1)
        disp = xyz_traj[-1] - xyz_traj[0]
        disp_n = np.linalg.norm(disp) + 1e-8
        axis = disp / disp_n

        straightness = disp_n / (np.sum(speed) + 1e-8)
        prismatic = float(np.clip(straightness, 0.0, 1.0))
        revolute = float(np.clip(1.0 - straightness, 0.0, 1.0))
        screw = 0.2 * min(prismatic, revolute)
        s = revolute + prismatic + screw + 1e-8

        return {
            "type_scores": {
                "revolute": revolute / s,
                "prismatic": prismatic / s,
                "screw": screw / s,
            },
            "axis_dir": axis.astype(np.float32),
            "axis_point": None,
            "pitch": None,
            "confidence": float(np.clip(np.mean(speed), 0.0, 1.0)),
        }
