from __future__ import annotations

from articulation.data.dataclasses import JointCandidateResult



def select_best_candidate(
    candidates: list[JointCandidateResult],
    screw_complexity_penalty: float = 0.05,
) -> JointCandidateResult:
    if not candidates:
        raise ValueError("No joint candidates provided")

    best = None
    best_score = float("inf")
    for c in candidates:
        score = c.loss
        if c.model_name == "screw":
            score += screw_complexity_penalty
        if score < best_score:
            best_score = score
            best = c

    assert best is not None
    return best
