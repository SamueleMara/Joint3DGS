from __future__ import annotations

import torch

from articulation.segmentation.variables import SegmentationVariables



def build_segmentation_optimizer(vars: SegmentationVariables, cfg: dict) -> torch.optim.Optimizer:
    lr_logits = float(cfg.get("lr_logits", 1e-2))
    lr_twists = float(cfg.get("lr_twists", 1e-4))

    return torch.optim.Adam(
        [
            {"params": [vars.logits], "lr": lr_logits},
            {"params": [vars.xi_part0, vars.xi_part1], "lr": lr_twists},
        ]
    )
