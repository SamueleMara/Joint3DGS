from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from articulation.data.dataclasses import KeypointBatch, MatchBatch
from articulation.external.matcher_backend_adapter import MatcherBackendAdapter
from articulation.matching.base import descriptor_mutual_nn_match


def _coerce_backend_output(raw: MatchBatch | dict[str, Any], pair_type: str) -> MatchBatch:
    if isinstance(raw, MatchBatch):
        return raw

    if not isinstance(raw, dict):
        raise TypeError("Matcher backend must return MatchBatch or dict")

    idx_a = torch.as_tensor(raw.get("idx_a", []), dtype=torch.long)
    idx_b = torch.as_tensor(raw.get("idx_b", []), dtype=torch.long)
    conf = torch.as_tensor(raw.get("confidence", raw.get("conf", [])), dtype=torch.float32)

    if idx_a.ndim != 1 or idx_b.ndim != 1 or conf.ndim != 1:
        raise ValueError("Backend output idx_a/idx_b/confidence must be 1D")
    if idx_a.numel() != idx_b.numel() or idx_a.numel() != conf.numel():
        raise ValueError("Backend output idx_a/idx_b/confidence must align")

    return MatchBatch(
        idx_a=idx_a,
        idx_b=idx_b,
        confidence=conf,
        pair_type=pair_type,
        meta={k: v for k, v in raw.items() if k not in {"idx_a", "idx_b", "confidence", "conf"}},
    )


@dataclass
class OmniGlueMatcher:
    backend_callable: str | None = None
    repo_path: str | None = None
    min_confidence: float = 0.2
    max_matches: int | None = None

    def __post_init__(self) -> None:
        self._backend: Callable[[KeypointBatch, KeypointBatch, str], MatchBatch | dict[str, Any]] | None = None
        if self.backend_callable:
            self._backend = MatcherBackendAdapter(
                backend_callable=self.backend_callable,
                repo_path=self.repo_path,
            ).build()

    def match(self, frame_a: KeypointBatch, frame_b: KeypointBatch, pair_type: str) -> MatchBatch:
        if self._backend is None:
            return descriptor_mutual_nn_match(
                frame_a,
                frame_b,
                pair_type=pair_type,
                min_confidence=float(self.min_confidence),
                max_matches=self.max_matches,
            )

        raw = self._backend(frame_a, frame_b, pair_type)
        out = _coerce_backend_output(raw, pair_type=pair_type)

        keep = out.confidence >= float(self.min_confidence)
        idx_a = out.idx_a[keep]
        idx_b = out.idx_b[keep]
        conf = out.confidence[keep]

        if self.max_matches is not None and idx_a.numel() > int(self.max_matches):
            topk = torch.topk(conf, k=int(self.max_matches), largest=True)
            sel = topk.indices
            idx_a = idx_a[sel]
            idx_b = idx_b[sel]
            conf = conf[sel]

        meta = dict(out.meta)
        meta["backend"] = "omniglue"
        meta["num_matches"] = int(idx_a.numel())
        return MatchBatch(idx_a=idx_a, idx_b=idx_b, confidence=conf, pair_type=pair_type, meta=meta)
