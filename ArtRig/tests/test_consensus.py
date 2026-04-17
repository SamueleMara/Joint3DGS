import torch

from articulation.joint.consensus import aggregate_axis_dirs, aggregate_type_scores



def test_axis_consensus_sign_ambiguity():
    clues = [
        {"axis_dir": [1.0, 0.0, 0.0], "confidence": 1.0},
        {"axis_dir": [-1.0, 0.0, 0.0], "confidence": 1.0},
        {"axis_dir": [0.9, 0.1, 0.0], "confidence": 0.5},
    ]
    a = aggregate_axis_dirs(clues)
    assert torch.isclose(torch.abs(a[0]), torch.tensor(1.0), atol=1e-3)



def test_type_consensus_normalized():
    clues = [
        {"type_scores": {"revolute": 1.0, "prismatic": 0.0, "screw": 0.0}, "confidence": 1.0},
        {"type_scores": {"revolute": 0.0, "prismatic": 1.0, "screw": 0.0}, "confidence": 1.0},
    ]
    s = aggregate_type_scores(clues)
    total = s["revolute"] + s["prismatic"] + s["screw"]
    assert abs(total - 1.0) < 1e-6
