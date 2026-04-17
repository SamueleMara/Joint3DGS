from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch

from articulation.data.dataclasses import RGBDSequence, TrackBatch


@dataclass
class AllTrackerBackend:
    repo_path: str
    checkpoint: str
    seqlen: int | None = None
    inference_iters: int = 4
    query_frame: int = 0
    rate: int = 4
    device: str = "cuda"

    def _load_model(self, seqlen: int):
        repo = str(Path(self.repo_path).resolve())
        if repo not in sys.path:
            sys.path.insert(0, repo)

        from nets.alltracker import Net  # type: ignore

        model = Net(seqlen=seqlen)
        ckpt = torch.load(self.checkpoint, map_location="cpu")
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)
        return model

    def track(self, sequence: RGBDSequence) -> TrackBatch:
        if not Path(self.checkpoint).exists():
            raise FileNotFoundError(f"AllTracker checkpoint not found: {self.checkpoint}")

        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        seqlen = int(self.seqlen or sequence.T)
        model = self._load_model(seqlen=seqlen).to(device)
        model.eval()

        # AllTracker expects 0-255 RGB
        video = (sequence.rgb * 255.0).to(device)
        video = video.unsqueeze(0)  # [1,T,3,H,W]
        b, t, c, h, w = video.shape

        # build grid
        repo = str(Path(self.repo_path).resolve())
        if repo not in sys.path:
            sys.path.insert(0, repo)
        import utils.basic  # type: ignore

        grid_xy = utils.basic.gridcloud2d(1, h, w, norm=False, device=device).float()
        grid_xy = grid_xy.permute(0, 2, 1).reshape(1, 1, 2, h, w)

        # forward
        flows_e, visconf_maps_e, _, _ = model.forward_sliding(
            video[:, self.query_frame :],
            iters=self.inference_iters,
            sw=None,
            is_training=False,
        )
        traj_maps_e = flows_e + grid_xy  # [1,Tf,2,H,W]
        visconf_maps_e = visconf_maps_e

        if self.query_frame > 0:
            backward_flows_e, backward_visconf_maps_e, _, _ = model.forward_sliding(
                video[:, : self.query_frame + 1].flip([1]),
                iters=self.inference_iters,
                sw=None,
                is_training=False,
            )
            backward_traj_maps_e = backward_flows_e + grid_xy
            backward_traj_maps_e = backward_traj_maps_e.flip([1])[:, :-1]
            backward_visconf_maps_e = backward_visconf_maps_e.flip([1])[:, :-1]
            traj_maps_e = torch.cat([backward_traj_maps_e, traj_maps_e], dim=1)
            visconf_maps_e = torch.cat([backward_visconf_maps_e, visconf_maps_e], dim=1)

        # subsample grid
        rate = max(1, int(self.rate))
        trajs = traj_maps_e[:, :, :, ::rate, ::rate].reshape(1, -1, 2, (h // rate) * (w // rate)).permute(0, 1, 3, 2)
        visconfs = visconf_maps_e[:, :, :, ::rate, ::rate].reshape(1, -1, 2, (h // rate) * (w // rate)).permute(0, 1, 3, 2)

        # shape to [N,T,2]
        tracks = trajs[0].permute(1, 0, 2).contiguous()
        vis = (visconfs[0, :, :, 0] > 0.5).permute(1, 0).contiguous()

        p, t2, _ = tracks.shape
        return TrackBatch(
            xy=tracks.float(),
            xyz=torch.zeros((p, t2, 3), dtype=tracks.dtype, device=tracks.device),
            valid=vis.bool(),
            anchor_frame=self.query_frame,
            point_ids=torch.arange(p, device=tracks.device),
            feature=torch.zeros((p, 1), dtype=tracks.dtype, device=tracks.device),
            confidence=None,
        )
