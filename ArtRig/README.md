# ArtRig

Two-part articulated object reconstruction from RGB-D:

1. Stage 0: DINO features + keypoints + matching + 3D lifting + multiview-consistent sparse tracks
2. Stage 1: rigidity-driven two-part segmentation
3. Stage 2: joint estimation (revolute / prismatic / screw)

## Repo Layout

- `articulation/pipeline/stage0_matching.py`
- `articulation/pipeline/stage1_segmentation.py`
- `articulation/pipeline/stage2_joint.py`
- `scripts/run_matching.py`
- `scripts/run_segmentation.py`
- `scripts/run_joint.py`
- `scripts/run_pipeline.py`
- `scripts/preprocess_video_da3_folder.py`

Top-level wrappers:
- `run_matching.py`
- `run_pipeline.py`
- `run_pipeline_folder.py`
- `preprocess_video_da3_folder.py`

## Environment

```bash
conda create -y -n art-rig python=3.10 pip
conda activate art-rig
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

## Submodules

All third-party repos should stay under `submodules/`.

Initialize existing submodules:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

Optional additional submodules (for external matcher/feature backends):

```bash
git submodule add https://github.com/facebookresearch/dinov2 submodules/dinov2
git submodule add https://github.com/google-research/omniglue submodules/omniglue
git submodule add https://github.com/cvg/LightGlue submodules/lightglue
# optional external joint clue backend
# git submodule add <JOINT_CLUE_REPO_URL> submodules/joint_clue_repo
```

Install everything (project + installable submodules):

```bash
bash scripts/install_submodules.sh
```

## Input Format (Folder)

```text
dataset/
  images/000000.png ...
  depth/000000.png ...         # or .npy files
  fg_mask/000000.png ...
  intrinsics.npy               # [3,3], [V,3,3], [T,3,3], or [T,V,3,3]
  extrinsics.npy               # [T,3,4]/[T,4,4] or [T,V,3,4]/[T,V,4,4]
```

## Preprocess Video With Depth-Anything-3

```bash
python preprocess_video_da3_folder.py \
  --video /path/to/video.mp4 \
  --output-dir /path/to/dataset_folder \
  --fps 1.0 \
  --repo-path submodules/depth_anything_3 \
  --model-name da3-small \
  --device cpu
```

## Run Stages

Stage 0 only:

```bash
python run_matching.py \
  --input-dir /path/to/dataset_folder \
  --config configs/matching.yaml \
  --out-dir outputs/matching_run
```

Stage 1 only:

```bash
python scripts/run_segmentation.py \
  --tracks-npz outputs/matching_run/tracks.npz \
  --config configs/segmentation.yaml \
  --output outputs/segmentation.pt
```

Stage 2 only:

```bash
python scripts/run_joint.py \
  --tracks-npz outputs/matching_run/tracks.npz \
  --segmentation outputs/segmentation.pt \
  --config configs/joint.yaml \
  --output outputs/joint.pt
```

End-to-end:

```bash
python run_pipeline.py \
  --input-dir /path/to/dataset_folder \
  --matching-config configs/matching.yaml \
  --seg-config configs/segmentation.yaml \
  --joint-config configs/joint.yaml \
  --out-dir outputs/full_run \
  --progress --loss-debug
```

Stage-1 loss balancing is enabled by default (`configs/segmentation.yaml`):

- online term normalization (`loss.normalize_terms: true`)
- weighted debug print now shows both raw and weighted contributions

For multiview datasets like `fr3_joint1` (`rgb/cam_*`, `depth_npy/cam_*`, `mask/cam_*`, `metadata/cameras.json`), this works directly:

```bash
python run_pipeline.py \
  --input-dir /home/samuelemara/Joint3DGS/ArtRig/inputs/fr3_joint1/rgb \
  --depth-dir depth_npy \
  --mask-dir mask \
  --cameras-json metadata/cameras.json \
  --depth-npy-scale 1.0 \
  --matching-config configs/matching.yaml \
  --seg-config configs/segmentation.yaml \
  --joint-config configs/joint.yaml \
  --out-dir outputs/fr3_joint1_full \
  --progress --loss-debug
```

Single-view legacy NPZ input is also supported:

```bash
python run_pipeline.py \
  --sequence-npz /path/to/results.npz \
  --matching-config configs/matching.yaml \
  --out-dir outputs/full_run_npz
```

## Visualization

Use existing utilities:

- `scripts/visualize_outputs.py`
- `scripts/visualize_segmentation_3d.py`
- `scripts/plot_dino_feature_map.py`

Quick previews from the same pipeline command:

```bash
python run_pipeline.py ... --viz
```

All-in-one visualization bundle (summary + DINO map + dense segmented 3D cloud + joint axis) from the same command:

```bash
python run_pipeline.py ... \
  --viz-all \
  --viz-export-sequence-npz \
  --viz-view-idx 0 \
  --viz-frame 0
```

This writes everything under `<out-dir>/viz_all/`, including:

- `viz_all/seg3d/segmented_pointcloud.ply`
- `viz_all/seg3d/static_pointcloud.ply`
- `viz_all/seg3d/dynamic_pointcloud.ply`
- `viz_all/seg3d/frame_clouds/static/*.ply`
- `viz_all/seg3d/frame_clouds/dynamic/*.ply`
- `viz_all/seg3d/frame_cloud_stats.json`

If RAM is tight, skip DINO map export:

```bash
python run_pipeline.py ... --viz-all --viz-all-no-dino
```

Tune 3D export density:

```bash
python run_pipeline.py ... --viz-all \
  --viz-3d-frame-stride 2 \
  --viz-3d-sample-ratio 0.1 \
  --viz-3d-conf-thresh 0.5 \
  --viz-3d-frame-max-points 25000
```

## W&B Logging

W&B logging is enabled by default in `run_pipeline.py`.
If `--wandb-name` is not provided, the default name is:
`<dataset_name>_YYYYMMDD_HHMM` (for example, `fr3_joint1_20260417_0934`).

```bash
python run_pipeline.py ... \
  --wandb-project ArtRig \
  --wandb-name fr3_joint1_run1
```

Disable it explicitly with `--no-wandb`.

By default, the wrapper also streams system telemetry to W&B every 2 seconds:

- CPU usage
- RAM usage
- per-GPU utilization/memory
- per-GPU power draw (W) when NVML is available

Disable telemetry with `--no-wandb-system-monitor`, or adjust rate with `--wandb-system-interval`.

`run_meta.json` stores W&B metadata (`run_id`, `run_url`, mode).  
For no-cloud debugging, use `--wandb-mode offline`.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```
