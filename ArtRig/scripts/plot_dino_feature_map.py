#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from articulation.data import load_rgbd_sequence_npz
from articulation.features.dino_wrapper import DinoFeatureExtractor


def _load_image_from_video(video_path: str | Path, frame_idx: int) -> torch.Tensor:
    cap = cv2.VideoCapture(str(video_path))
    i = 0
    frame = None
    while cap.isOpened():
        ok, fr = cap.read()
        if not ok:
            break
        if i == frame_idx:
            frame = fr
            break
        i += 1
    cap.release()
    if frame is None:
        raise ValueError(f"Could not read frame {frame_idx} from video: {video_path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0


def _load_image_from_path(image_path: str | Path) -> torch.Tensor:
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


def _pca_rgb(feat_chw: torch.Tensor) -> np.ndarray:
    c, h, w = feat_chw.shape
    x = feat_chw.permute(1, 2, 0).reshape(-1, c).float()
    x = x - x.mean(dim=0, keepdim=True)
    _, _, v = torch.pca_lowrank(x, q=min(3, c))
    x3 = x @ v[:, : min(3, c)]
    if x3.shape[1] < 3:
        pad = torch.zeros((x3.shape[0], 3 - x3.shape[1]), dtype=x3.dtype, device=x3.device)
        x3 = torch.cat([x3, pad], dim=1)

    lo = torch.quantile(x3, 0.01, dim=0, keepdim=True)
    hi = torch.quantile(x3, 0.99, dim=0, keepdim=True)
    x3 = (x3 - lo) / (hi - lo + 1e-8)
    x3 = x3.clamp(0.0, 1.0)
    return x3.reshape(h, w, 3).cpu().numpy()


def _norm_map(feat_chw: torch.Tensor) -> np.ndarray:
    n = torch.linalg.norm(feat_chw.float(), dim=0)
    n = (n - n.min()) / (n.max() - n.min() + 1e-8)
    return n.cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--sequence-npz", default=None)
    src.add_argument("--video", default=None)
    src.add_argument("--image", default=None)
    parser.add_argument("--frame-idx", type=int, default=0)
    parser.add_argument("--model-name", default="vit_small_patch14_dinov2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="outputs/dino_feature_map")
    args = parser.parse_args()

    if args.sequence_npz is not None:
        seq = load_rgbd_sequence_npz(args.sequence_npz)
        if not (0 <= args.frame_idx < seq.T):
            raise ValueError(f"frame-idx {args.frame_idx} out of range [0,{seq.T - 1}]")
        image = seq.rgb[args.frame_idx]
    elif args.video is not None:
        image = _load_image_from_video(args.video, frame_idx=args.frame_idx)
    else:
        image = _load_image_from_path(args.image)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    extractor = DinoFeatureExtractor(model_name=args.model_name, device=args.device)
    h_in, w_in = int(image.shape[-2]), int(image.shape[-1])
    h_model = max(14, (h_in // 14) * 14)
    w_model = max(14, (w_in // 14) * 14)
    if (h_model, w_model) != (h_in, w_in):
        image_for_model = F.interpolate(
            image.unsqueeze(0),
            size=(h_model, w_model),
            mode="bilinear",
            align_corners=False,
        )[0]
    else:
        image_for_model = image

    fmap = extractor.extract_dense(image_for_model)  # [C,h,w]

    pca = _pca_rgb(fmap)
    norm = _norm_map(fmap)

    h, w = image.shape[-2], image.shape[-1]
    pca_up = F.interpolate(
        torch.from_numpy(pca).permute(2, 0, 1).unsqueeze(0),
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    )[0].permute(1, 2, 0).numpy()
    norm_up = F.interpolate(
        torch.from_numpy(norm).unsqueeze(0).unsqueeze(0),
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()

    img_np = image.permute(1, 2, 0).cpu().numpy()
    plt.imsave(out_dir / "input_rgb.png", np.clip(img_np, 0.0, 1.0))
    plt.imsave(out_dir / "dino_pca.png", np.clip(pca_up, 0.0, 1.0))
    plt.imsave(out_dir / "dino_norm.png", np.clip(norm_up, 0.0, 1.0), cmap="viridis")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(np.clip(img_np, 0.0, 1.0))
    axes[0].set_title("Input RGB")
    axes[0].axis("off")
    axes[1].imshow(np.clip(pca_up, 0.0, 1.0))
    axes[1].set_title("DINO features (PCA)")
    axes[1].axis("off")
    im = axes[2].imshow(np.clip(norm_up, 0.0, 1.0), cmap="viridis")
    axes[2].set_title("DINO feature norm")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "dino_feature_map_overview.png", dpi=160)
    plt.close(fig)

    print(f"Saved DINO visualizations to: {out_dir}")
    print(f"Feature map shape: {tuple(fmap.shape)}")


if __name__ == "__main__":
    main()
