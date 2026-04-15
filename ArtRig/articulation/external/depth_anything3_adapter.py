from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import sys

import numpy as np
import torch


@dataclass
class DepthAnything3Adapter:
    repo_path: str | None = None
    model_dir: str | None = None
    model_name: str = "da3-large"
    device: str = "cuda"

    def _ensure_repo_path(self) -> None:
        if not self.repo_path:
            return
        p = str(Path(self.repo_path).resolve())
        if p not in sys.path:
            sys.path.insert(0, p)

    def build(self):
        self._ensure_repo_path()
        from depth_anything_3.api import DepthAnything3

        if self.model_dir:
            model = DepthAnything3.from_pretrained(self.model_dir)
        else:
            model = DepthAnything3(model_name=self.model_name)
        model = model.to(device=torch.device(self.device))
        return model

    @torch.inference_mode()
    def infer_depth(
        self,
        rgb: torch.Tensor,
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
    ) -> torch.Tensor:
        """Infer depth for RGB sequence.

        Args:
            rgb: [T,3,H,W] in [0,1]
        Returns:
            depth: [T,1,H,W] float32
        """
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("rgb must be [T,3,H,W]")

        model = self.build()
        images = []
        for frame in rgb:
            img = (frame.clamp(0, 1) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
            images.append(img)

        pred = model.inference(
            images,
            process_res=process_res,
            process_res_method=process_res_method,
            export_format="mini_npz",
        )
        depth = torch.from_numpy(pred.depth).float()  # [T,H,W]
        if depth.ndim != 3:
            raise ValueError("DA3 depth output must be [T,H,W]")
        depth = depth.unsqueeze(1)
        return depth
