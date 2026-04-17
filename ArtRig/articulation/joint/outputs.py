from __future__ import annotations

from articulation.data.dataclasses import JointResult



def joint_result_to_dict(result: JointResult) -> dict:
    return {
        "best_model": result.best_model,
        "axis_dir": result.axis_dir.detach().cpu().tolist(),
        "axis_point": None if result.axis_point is None else result.axis_point.detach().cpu().tolist(),
        "pitch": None if result.pitch is None else float(result.pitch.detach().cpu()),
        "state": result.state.detach().cpu().tolist(),
        "candidates": [
            {
                "model_name": c.model_name,
                "loss": float(c.loss),
                "diagnostics": c.diagnostics,
            }
            for c in result.candidates
        ],
        "diagnostics": result.diagnostics,
    }
