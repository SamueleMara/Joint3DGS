from __future__ import annotations

import math

import torch


def _normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if v.shape[-1] != 3:
        raise ValueError("Expected vectors with trailing dim 3")
    return v / torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(eps)


def aggregate_model_priors(
    per_view_losses: list[dict[str, float]],
    view_weights: list[float] | None = None,
    temperature: float = 0.02,
) -> dict[str, float]:
    models = ("revolute", "prismatic", "screw")
    if not per_view_losses:
        return {m: 1.0 / 3.0 for m in models}

    if view_weights is None:
        view_weights = [1.0] * len(per_view_losses)
    if len(view_weights) != len(per_view_losses):
        raise ValueError("view_weights length must match per_view_losses")

    accum = {m: 0.0 for m in models}
    temp = max(float(temperature), 1e-6)

    for losses, vw in zip(per_view_losses, view_weights):
        if vw <= 0:
            continue
        vals = [float(losses.get(m, 1e9)) for m in models]
        min_l = min(vals)
        logits = [math.exp(-(v - min_l) / temp) for v in vals]
        z = sum(logits) + 1e-12
        probs = [v / z for v in logits]
        for m, p in zip(models, probs):
            accum[m] += float(vw) * float(p)

    z = sum(accum.values())
    if z <= 1e-12:
        return {m: 1.0 / 3.0 for m in models}
    return {m: float(accum[m] / z) for m in models}


def robust_direction_consensus(
    dirs: torch.Tensor,
    weights: torch.Tensor | None = None,
    max_angle_deg: float = 12.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Consensus of unoriented directions.

    Returns:
        fused_dir: [3]
        inlier_mask: [N] bool
        signs: [N] in {-1,+1} aligning each input dir to fused_dir
    """
    if dirs.ndim != 2 or dirs.shape[1] != 3:
        raise ValueError("dirs must be [N,3]")
    n = dirs.shape[0]
    if n == 0:
        raise ValueError("Need at least one direction")
    d = _normalize(dirs.float())
    if weights is None:
        w = torch.ones(n, dtype=torch.float32, device=d.device)
    else:
        if weights.shape != (n,):
            raise ValueError("weights must be [N]")
        w = weights.float().clamp_min(0.0)

    cos_thr = math.cos(math.radians(max(float(max_angle_deg), 0.0)))
    best_score = -1.0
    best_mean_abs_cos = -1.0
    best_seed_idx = 0
    best_inlier = torch.ones(n, dtype=torch.bool, device=d.device)

    for i in range(n):
        seed = d[i]
        abs_cos = torch.abs(torch.sum(d * seed[None, :], dim=1))
        inlier = abs_cos >= cos_thr
        score = float(torch.sum(w * inlier.float()).item())
        mean_abs = float(torch.mean(abs_cos[inlier]).item()) if bool(inlier.any()) else 0.0
        if score > best_score or (score == best_score and mean_abs > best_mean_abs_cos):
            best_score = score
            best_mean_abs_cos = mean_abs
            best_seed_idx = i
            best_inlier = inlier

    seed = d[best_seed_idx]
    signs = torch.where(torch.sum(d * seed[None, :], dim=1) >= 0.0, 1.0, -1.0)
    aligned = d * signs[:, None]
    ww = (w * best_inlier.float())[:, None]
    fused = torch.sum(ww * aligned, dim=0)
    if float(torch.linalg.norm(fused).item()) < 1e-8:
        fused = aligned[best_seed_idx]
    fused = _normalize(fused[None, :])[0]

    signs_final = torch.where(torch.sum(d * fused[None, :], dim=1) >= 0.0, 1.0, -1.0)
    return fused, best_inlier, signs_final


def weighted_line_intersection(
    line_points: torch.Tensor,
    line_dirs: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if line_points.ndim != 2 or line_points.shape[1] != 3:
        raise ValueError("line_points must be [N,3]")
    if line_dirs.shape != line_points.shape:
        raise ValueError("line_dirs must be [N,3]")
    n = line_points.shape[0]
    if n == 0:
        raise ValueError("Need at least one line")

    d = _normalize(line_dirs.float())
    p = line_points.float()
    if weights is None:
        w = torch.ones(n, dtype=torch.float32, device=p.device)
    else:
        if weights.shape != (n,):
            raise ValueError("weights must be [N]")
        w = weights.float().clamp_min(0.0)

    eye = torch.eye(3, dtype=torch.float32, device=p.device)
    a = torch.zeros((3, 3), dtype=torch.float32, device=p.device)
    b = torch.zeros((3,), dtype=torch.float32, device=p.device)
    for i in range(n):
        pi = p[i]
        di = d[i]
        proj = eye - torch.outer(di, di)
        wi = w[i]
        a = a + wi * proj
        b = b + wi * (proj @ pi)

    try:
        q = torch.linalg.solve(a, b)
    except RuntimeError:
        q = torch.linalg.pinv(a) @ b
    return q


def robust_axis_point_consensus(
    line_points: torch.Tensor,
    line_dirs: torch.Tensor,
    weights: torch.Tensor | None = None,
    max_dist: float = 0.08,
    refinement_steps: int = 3,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Consensus of axis point for a set of 3D lines."""
    if line_points.ndim != 2 or line_points.shape[1] != 3:
        raise ValueError("line_points must be [N,3]")
    if line_dirs.shape != line_points.shape:
        raise ValueError("line_dirs must be [N,3]")
    n = line_points.shape[0]
    if n == 0:
        return None, torch.zeros((0,), dtype=torch.bool)

    if weights is None:
        w = torch.ones(n, dtype=torch.float32, device=line_points.device)
    else:
        if weights.shape != (n,):
            raise ValueError("weights must be [N]")
        w = weights.float().clamp_min(0.0)

    p = line_points.float()
    d = _normalize(line_dirs.float())

    q = weighted_line_intersection(p, d, w)
    inlier = torch.ones(n, dtype=torch.bool, device=p.device)
    thr = max(float(max_dist), 0.0)

    for _ in range(max(int(refinement_steps), 0)):
        rel = q[None, :] - p
        proj = torch.sum(rel * d, dim=1, keepdim=True) * d
        dist = torch.linalg.norm(rel - proj, dim=1)
        new_inlier = dist <= thr
        if not bool(new_inlier.any()):
            break
        if torch.equal(new_inlier, inlier):
            break
        inlier = new_inlier
        q = weighted_line_intersection(p[inlier], d[inlier], w[inlier])

    return q, inlier


def fuse_signed_states(
    states: list[torch.Tensor],
    dirs: torch.Tensor,
    fused_dir: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if not states:
        return None
    if dirs.ndim != 2 or dirs.shape[1] != 3:
        raise ValueError("dirs must be [N,3]")
    n = len(states)
    if dirs.shape[0] != n:
        raise ValueError("states and dirs size mismatch")

    t = min(int(s.numel()) for s in states)
    if t <= 0:
        return None

    dd = _normalize(dirs.float())
    fd = _normalize(fused_dir.float()[None, :])[0]
    sign = torch.where(torch.sum(dd * fd[None, :], dim=1) >= 0.0, 1.0, -1.0)

    if weights is None:
        w = torch.ones(n, dtype=torch.float32, device=dirs.device)
    else:
        if weights.shape != (n,):
            raise ValueError("weights must be [N]")
        w = weights.float().clamp_min(0.0)
    ws = w / w.sum().clamp_min(1e-8)

    acc = torch.zeros(t, dtype=torch.float32, device=dirs.device)
    for i, s in enumerate(states):
        acc = acc + ws[i] * (sign[i] * s.detach().float().to(dirs.device).view(-1)[:t])
    return acc


def fuse_signed_pitches(
    pitches: list[float | None],
    dirs: torch.Tensor,
    fused_dir: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> float | None:
    if not pitches:
        return None
    vals: list[float] = []
    idxs: list[int] = []
    for i, p in enumerate(pitches):
        if p is None:
            continue
        vals.append(float(p))
        idxs.append(i)
    if not vals:
        return None

    dd = _normalize(dirs.float())
    fd = _normalize(fused_dir.float()[None, :])[0]
    if weights is None:
        w = torch.ones(dd.shape[0], dtype=torch.float32, device=dd.device)
    else:
        w = weights.float().to(dd.device).clamp_min(0.0)

    num = 0.0
    den = 0.0
    for v, i in zip(vals, idxs):
        sgn = 1.0 if float(torch.dot(dd[i], fd).item()) >= 0.0 else -1.0
        wi = float(w[i].item())
        num += wi * sgn * float(v)
        den += wi
    if den <= 1e-8:
        return None
    return float(num / den)
