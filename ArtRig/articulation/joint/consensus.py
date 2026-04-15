from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ConsensusResult:
    type_priors: dict[str, float]
    axis_dir: torch.Tensor
    axis_point: torch.Tensor | None
    pitch: float | None



def aggregate_type_scores(clues: list[dict]) -> dict[str, float]:
    scores = {"revolute": 0.0, "prismatic": 0.0, "screw": 0.0}
    if len(clues) == 0:
        return {k: 1.0 / 3.0 for k in scores}

    total_w = 0.0
    for c in clues:
        w = float(c.get("confidence", 1.0))
        ts = c.get("type_scores", {})
        for k in scores:
            scores[k] += w * float(ts.get(k, 0.0))
        total_w += w

    if total_w <= 0:
        return {k: 1.0 / 3.0 for k in scores}

    s = sum(scores.values()) + 1e-8
    return {k: float(v / s) for k, v in scores.items()}



def aggregate_axis_dirs(clues: list[dict]) -> torch.Tensor:
    if len(clues) == 0:
        return torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

    dirs = []
    ws = []
    for c in clues:
        a = c.get("axis_dir", None)
        if a is None:
            continue
        d = torch.as_tensor(a, dtype=torch.float32)
        if d.numel() != 3:
            continue
        n = torch.linalg.norm(d)
        if n < 1e-8:
            continue
        dirs.append(d / n)
        ws.append(float(c.get("confidence", 1.0)))

    if len(dirs) == 0:
        return torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

    ref = dirs[0]
    acc = torch.zeros(3, dtype=torch.float32)
    for d, w in zip(dirs, ws):
        sign = 1.0 if torch.dot(d, ref) >= 0 else -1.0
        acc = acc + sign * w * d

    return acc / torch.linalg.norm(acc).clamp_min(1e-8)



def aggregate_axis_point_and_pitch(clues: list[dict]) -> tuple[torch.Tensor | None, float | None]:
    points = []
    pws = []
    pitches = []
    for c in clues:
        w = float(c.get("confidence", 1.0))
        ap = c.get("axis_point", None)
        if ap is not None:
            points.append(torch.as_tensor(ap, dtype=torch.float32))
            pws.append(w)
        pitch = c.get("pitch", None)
        if pitch is not None:
            pitches.append((w, float(pitch)))

    axis_point = None
    if points:
        w = torch.tensor(pws, dtype=torch.float32)
        p = torch.stack(points, dim=0)
        axis_point = (w[:, None] * p).sum(dim=0) / w.sum().clamp_min(1e-8)

    pitch = None
    if pitches:
        w = sum(pi[0] for pi in pitches)
        if w > 0:
            pitch = sum(pi[0] * pi[1] for pi in pitches) / w

    return axis_point, pitch



def build_consensus(clues: list[dict]) -> ConsensusResult:
    priors = aggregate_type_scores(clues)
    axis = aggregate_axis_dirs(clues)
    axis_point, pitch = aggregate_axis_point_and_pitch(clues)
    return ConsensusResult(type_priors=priors, axis_dir=axis, axis_point=axis_point, pitch=pitch)
