from __future__ import annotations

from dataclasses import dataclass

from articulation.data.dataclasses import KeypointBatch, MatchBatch
from articulation.matching.base import BaseMatcher, descriptor_mutual_nn_match
from articulation.matching.lightglue_adapter import LightGlueMatcher
from articulation.matching.omniglue_adapter import OmniGlueMatcher


@dataclass
class MatcherWrapper:
    backend: BaseMatcher

    @classmethod
    def from_config(cls, cfg: dict) -> "MatcherWrapper":
        backend_name = str(cfg.get("backend", "omniglue")).lower()
        min_conf = float(cfg.get("min_confidence", 0.2))
        max_matches = cfg.get("max_matches", None)
        max_matches_int = None if max_matches is None else int(max_matches)

        if backend_name == "omniglue":
            impl = OmniGlueMatcher(
                backend_callable=cfg.get("backend_callable"),
                repo_path=cfg.get("repo_path"),
                min_confidence=min_conf,
                max_matches=max_matches_int,
            )
            return cls(backend=impl)

        if backend_name == "lightglue":
            impl = LightGlueMatcher(
                backend_callable=cfg.get("backend_callable"),
                repo_path=cfg.get("repo_path"),
                min_confidence=min_conf,
                max_matches=max_matches_int,
            )
            return cls(backend=impl)

        if backend_name in {"descriptor", "mutual_nn"}:
            class _InternalMatcher:
                def match(self, frame_a: KeypointBatch, frame_b: KeypointBatch, pair_type: str) -> MatchBatch:
                    return descriptor_mutual_nn_match(
                        frame_a,
                        frame_b,
                        pair_type=pair_type,
                        min_confidence=min_conf,
                        max_matches=max_matches_int,
                    )

            return cls(backend=_InternalMatcher())

        raise ValueError(f"Unsupported matcher backend: {backend_name}")

    def match(self, frame_a: KeypointBatch, frame_b: KeypointBatch, pair_type: str) -> MatchBatch:
        return self.backend.match(frame_a, frame_b, pair_type)
