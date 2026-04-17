from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from articulation.data.io_features import sample_features_at_xy


class DinoFeatureExtractor:
    def __init__(self, model_name: str = "vit_small_patch14_dinov2", device: str = "cpu"):
        self.device = torch.device(device)
        self.model_name = model_name
        self._last_input_hw: tuple[int, int] | None = None

        try:
            import timm
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "timm is required for DinoFeatureExtractor. Install timm or provide external adapter."
            ) from e

        create_kwargs = dict(pretrained=True, features_only=True)
        try:
            try:
                # Support variable input resolution (e.g. DA3 504px outputs) when available.
                self.model = timm.create_model(model_name, dynamic_img_size=True, **create_kwargs)
            except TypeError:
                self.model = timm.create_model(model_name, **create_kwargs)
        except Exception:
            # Offline/uncached fallback: keep architecture and continue with random weights.
            create_kwargs["pretrained"] = False
            try:
                self.model = timm.create_model(model_name, dynamic_img_size=True, **create_kwargs)
            except TypeError:
                self.model = timm.create_model(model_name, **create_kwargs)
        self.model.eval().to(self.device)

    @torch.inference_mode()
    def extract_dense(self, image: torch.Tensor) -> torch.Tensor:
        """Returns dense feature map [C,h,w] for a single image [3,H,W]."""
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("image must be [3,H,W]")

        h_in, w_in = int(image.shape[-2]), int(image.shape[-1])

        patch_h, patch_w = 14, 14
        try:
            pe = self.model.model.patch_embed
            p = pe.patch_size
            if isinstance(p, tuple):
                patch_h, patch_w = int(p[0]), int(p[1])
            else:
                patch_h = patch_w = int(p)
        except Exception:
            pass

        h_model = max(patch_h, (h_in // patch_h) * patch_h)
        w_model = max(patch_w, (w_in // patch_w) * patch_w)
        if (h_model, w_model) != (h_in, w_in):
            image = F.interpolate(
                image.unsqueeze(0),
                size=(h_model, w_model),
                mode="bilinear",
                align_corners=False,
            )[0]

        x = image.unsqueeze(0).to(self.device)
        feats = self.model(x)
        if isinstance(feats, (list, tuple)):
            fmap = feats[-1]
        else:
            fmap = feats

        if fmap.ndim != 4:
            raise ValueError(f"Unexpected feature map shape: {fmap.shape}")

        self._last_input_hw = (h_in, w_in)
        return fmap.squeeze(0).detach().cpu()

    @torch.inference_mode()
    def sample_points(self, feat_map: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        if feat_map.ndim != 3:
            raise ValueError("feat_map must be [C,h,w]")
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("xy must be [P,2]")
        xy_feat = xy
        if self._last_input_hw is not None:
            h_in, w_in = self._last_input_hw
            _, h_feat, w_feat = feat_map.shape
            if h_in > 1 and w_in > 1 and (h_in != h_feat or w_in != w_feat):
                sy = (h_feat - 1) / (h_in - 1)
                sx = (w_feat - 1) / (w_in - 1)
                xy_feat = xy.clone()
                xy_feat[:, 0] = xy_feat[:, 0] * sx
                xy_feat[:, 1] = xy_feat[:, 1] * sy
        return sample_features_at_xy(feat_map.to(xy.device), xy_feat)
