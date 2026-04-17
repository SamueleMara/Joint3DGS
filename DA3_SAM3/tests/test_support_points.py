from __future__ import annotations

import numpy as np

from dynamic_recon.tracking.support_points import sample_support_points


def test_support_points_inside_mask() -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 3:7] = 1
    points = sample_support_points(mask, 4)
    assert len(points) == 4
    for x, y in points:
        assert mask[int(y), int(x)] == 1
