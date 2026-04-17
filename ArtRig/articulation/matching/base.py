from __future__ import annotations

from typing import Protocol

import torch
import torch.nn.functional as F

from articulation.data.dataclasses import KeypointBatch, MatchBatch


class BaseMatcher(Protocol):
    def match(self, frame_a: KeypointBatch, frame_b: KeypointBatch, pair_type: str) -> MatchBatch:
        ...


def descriptor_mutual_nn_match(
    frame_a: KeypointBatch,
    frame_b: KeypointBatch,
    pair_type: str,
    min_confidence: float = 0.0,
    max_matches: int | None = None,
) -> MatchBatch:
    if frame_a.desc.ndim != 2 or frame_b.desc.ndim != 2:
        raise ValueError("Descriptors must be [N,C]")
    na = frame_a.desc.shape[0]
    nb = frame_b.desc.shape[0]

    if na == 0 or nb == 0:
        return MatchBatch(
            idx_a=torch.zeros((0,), dtype=torch.long),
            idx_b=torch.zeros((0,), dtype=torch.long),
            confidence=torch.zeros((0,), dtype=torch.float32),
            pair_type=pair_type,
            meta={"backend": "descriptor_mutual_nn", "num_matches": 0},
        )

    da = F.normalize(frame_a.desc.float(), dim=1)
    db = F.normalize(frame_b.desc.float(), dim=1)

    sim = da @ db.transpose(0, 1)  # [Na,Nb]

    best_b = torch.argmax(sim, dim=1)
    best_a = torch.argmax(sim, dim=0)

    arange_a = torch.arange(na, device=sim.device)
    b_of_a = best_b
    a_of_b = best_a[b_of_a]
    mutual = a_of_b == arange_a

    if not bool(mutual.any()):
        return MatchBatch(
            idx_a=torch.zeros((0,), dtype=torch.long),
            idx_b=torch.zeros((0,), dtype=torch.long),
            confidence=torch.zeros((0,), dtype=torch.float32),
            pair_type=pair_type,
            meta={"backend": "descriptor_mutual_nn", "num_matches": 0},
        )

    idx_a = arange_a[mutual]
    idx_b = b_of_a[mutual]
    conf = sim[idx_a, idx_b]

    conf_mask = conf >= float(min_confidence)
    idx_a = idx_a[conf_mask]
    idx_b = idx_b[conf_mask]
    conf = conf[conf_mask]

    if max_matches is not None and idx_a.numel() > int(max_matches):
        topk = torch.topk(conf, k=int(max_matches), largest=True)
        keep = topk.indices
        idx_a = idx_a[keep]
        idx_b = idx_b[keep]
        conf = conf[keep]

    return MatchBatch(
        idx_a=idx_a.detach().cpu().long(),
        idx_b=idx_b.detach().cpu().long(),
        confidence=conf.detach().cpu().float(),
        pair_type=pair_type,
        meta={
            "backend": "descriptor_mutual_nn",
            "num_matches": int(idx_a.numel()),
            "min_confidence": float(min_confidence),
        },
    )
