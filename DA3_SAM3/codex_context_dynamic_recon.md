# Codex context: DA3 + SAM3 dynamic/static 3D video pipeline

## Goal

Create a reproducible research-style Python project that:

1. installs **Depth Anything 3 (DA3)** and **SAM 3** as git submodules,
2. downloads their checkpoints without committing them,
3. runs DA3 on a video to estimate:
   - per-frame depth,
   - camera intrinsics,
   - camera extrinsics / poses,
   - a fused point cloud,
4. runs SAM 3 on the same video to segment and track dynamic objects,
5. combines the 2D SAM 3 tracks with a **3D static-vs-dynamic consistency test** based on depth and estimated camera motion,
6. exports:
   - camera poses in COLMAP/OpenCV-friendly format,
   - fused point cloud `.ply`,
   - per-frame dynamic/static masks,
   - a visualization video.

The final system must work from a single command such as:

```bash
python -m dynamic_recon.run \
  --video data/input.mp4 \
  --prompts "person,car,dog" \
  --output outputs/run01
```

---

## Canonical upstream dependencies

Use these official upstream repos as git submodules:

- `third_party/depth_anything_3` → `https://github.com/ByteDance-Seed/Depth-Anything-3.git`
- `third_party/sam3` → `https://github.com/facebookresearch/sam3.git`

Do **not** reimplement their core models. Wrap their official APIs.

---

## Important compatibility rule

Try a **single shared environment first**:

- Python `3.12`
- PyTorch `2.10`
- CUDA wheel line compatible with SAM 3 official install

Reason: SAM 3 is the stricter dependency. If a shared env becomes unstable because of version conflicts, do **not** block the project. Keep the same repo layout, but isolate DA3 and SAM3 behind thin wrappers and allow a fallback mode with two environments / subprocess calls.

Implementation priority:

1. shared env if it works,
2. otherwise dual-env wrapper mode,
3. never hardcode local absolute paths.

---

## Repository changes to make

Create or update the project with this structure:

```text
.
├── .gitmodules
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── scripts/
│   ├── setup_submodules.sh
│   ├── setup_env.sh
│   ├── download_checkpoints.sh
│   ├── extract_frames.sh
│   └── run_demo.sh
├── configs/
│   └── pipeline.default.yaml
├── third_party/
│   ├── depth_anything_3/
│   └── sam3/
├── dynamic_recon/
│   ├── __init__.py
│   ├── run.py
│   ├── io_utils.py
│   ├── video.py
│   ├── da3_wrapper.py
│   ├── sam3_wrapper.py
│   ├── geometry.py
│   ├── fusion.py
│   ├── visualization.py
│   └── types.py
└── outputs/
```

---

## Git submodule tasks

Add the submodules with exact commands:

```bash
git submodule add https://github.com/ByteDance-Seed/Depth-Anything-3.git third_party/depth_anything_3
git submodule add https://github.com/facebookresearch/sam3.git third_party/sam3
git submodule update --init --recursive
```

Commit the `.gitmodules` file and submodule pointers.

---

## Environment setup tasks

Create `scripts/setup_env.sh` that:

1. creates a Python 3.12 virtualenv or conda env,
2. installs a CUDA-enabled PyTorch build compatible with the current machine,
3. installs this repo in editable mode,
4. installs DA3 and SAM3 in editable mode from the submodules,
5. installs any extra runtime deps needed by the orchestration code.

### Shared-env target

Use this order:

```bash
# example skeleton, adapt CUDA index URL if needed
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install xformers
pip install -e third_party/depth_anything_3
pip install -e third_party/sam3
pip install -e .
```

Also install practical runtime deps if missing:

- opencv-python
- numpy
- scipy
- pillow
- pyyaml
- tqdm
- imageio
- imageio-ffmpeg
- trimesh
- open3d
- matplotlib
- huggingface_hub

If shared install fails, document the failure and implement:

- `scripts/setup_env_da3.sh`
- `scripts/setup_env_sam3.sh`

and make the wrappers support subprocess execution.

---

## Checkpoint download tasks

Create `scripts/download_checkpoints.sh`.

### DA3
Use the official `DepthAnything3.from_pretrained(...)` path or cache the requested model through Hugging Face.

Default model choice:

- quality-first default: `depth-anything/DA3-LARGE-1.1`
- optional stronger non-commercial preset: `depth-anything/DA3NESTED-GIANT-LARGE-1.1`

Expose this as a config value.

### SAM 3
Use the official SAM 3 package/checkpoint flow.

Requirements:

- read Hugging Face token from env,
- fail with a clear message if checkpoint access has not been granted,
- never commit downloaded checkpoints.

Support a config entry for selecting the checkpoint family, preferring the latest compatible default from upstream.

---

## Functional pipeline to implement

### Step 1: video ingestion

In `dynamic_recon/video.py`:

- accept MP4 or image folder input,
- extract RGB frames to `outputs/.../frames/`,
- optionally downsample FPS,
- keep a mapping from frame index → timestamp.

### Step 2: DA3 inference for depth + poses

In `dynamic_recon/da3_wrapper.py`:

- load DA3 from the submodule install,
- run inference over frame lists or chunks,
- save:
  - `depth.npy` or chunked `.npz`,
  - `conf.npy`,
  - `extrinsics.npy`,
  - `intrinsics.npy`,
- export metadata JSON with shapes and conventions.

Use DA3’s official API outputs directly where possible.

Implementation notes:

- prefer `DA3-LARGE-1.1` as the default pose model,
- expose `use_ray_pose` as a config option,
- expose `ref_view_strategy`, with `middle` preferred for temporally ordered videos unless testing shows another option is better,
- support chunked inference for long videos.

### Step 3: point cloud construction

In `dynamic_recon/geometry.py`:

Implement utilities for:

- converting DA3 extrinsics into a documented convention,
- backprojecting depth pixels into 3D camera points,
- transforming them into world coordinates,
- fusing them into a global point cloud,
- optionally filtering by confidence and depth range.

Save:

- `pointcloud_static_raw.ply`
- `pointcloud_all_raw.ply`

Prefer a simple confidence-aware fusion first. No TSDF is required unless needed later.

### Step 4: SAM 3 video prompting and tracking

In `dynamic_recon/sam3_wrapper.py`:

Implement a wrapper around the official video predictor.

Support:

- text prompts from CLI/config, e.g. `"person,car,bicycle"`,
- optional box/point prompts loaded from a JSON file,
- tracked masks for each object id across frames,
- saving masks as PNGs and compressed arrays.

Save outputs like:

```text
outputs/run01/sam3/
  masks/
  tracks.json
  prompts.json
```

### Step 5: 3D dynamic/static consistency scoring

In `dynamic_recon/fusion.py` implement the core logic.

For each frame pair `(t, t+1)` and pixel `p_t = (u, v)` with depth `d_t`:

1. backproject `p_t` using `K_t` and `d_t` to camera coordinates,
2. transform to world coordinates using the DA3 pose,
3. project that world point into frame `t+1` using `K_{t+1}` and pose `T_{t+1}`,
4. sample depth and confidence near the projected location in frame `t+1`,
5. reconstruct the corresponding 3D point from frame `t+1`,
6. compute a **static-world consistency error**.

Use at least these cues:

- 3D position inconsistency:
  - `||X_world_from_t - X_world_from_tplus1||`
- reprojection consistency
- depth disagreement along the viewing ray
- DA3 confidence gating
- occlusion / out-of-frame handling

Produce a scalar `dynamic_score[u, v]`.

### Step 6: fuse SAM 3 with the 3D score

The final label should combine:

- `sam_dynamic_prior` from SAM 3 tracked masks,
- `geom_dynamic_score` from 3D inconsistency.

Recommended first-pass fusion logic:

- if inside a SAM 3 tracked object and 3D inconsistency is high → `dynamic`
- if outside SAM 3 masks and 3D inconsistency is low → `static`
- if they disagree → `uncertain`
- provide a config switch for either:
  - hard thresholds, or
  - weighted probabilistic fusion.

Deliver both:

- ternary label map: `static / dynamic / uncertain`
- binary dynamic mask

Also compute per-track 3D motion statistics:

- mean 3D displacement,
- robust median displacement,
- fraction of dynamic pixels per object.

### Step 7: static point cloud export

Use the fused labels to export:

- static-only point cloud
- dynamic-only point cloud
- optional colored point cloud by label

Save:

```text
outputs/run01/
  poses/
  depth/
  pointcloud_all_raw.ply
  pointcloud_static_fused.ply
  pointcloud_dynamic_fused.ply
  masks_dynamic/
  masks_static/
  overlays/
  summary.json
```

### Step 8: visualization

In `dynamic_recon/visualization.py`:

Create videos showing:

- RGB frames,
- SAM 3 tracked masks,
- geometric dynamic score heatmap,
- final dynamic/static/uncertain overlay.

Use consistent colors:

- green = static
- red = dynamic
- yellow = uncertain

---

## Pose convention requirements

Be explicit everywhere about pose conventions.

- DA3 returns extrinsics in OpenCV world-to-camera / COLMAP-style format.
- Implement utility functions to convert between:
  - `w2c` 3x4,
  - `w2c` 4x4,
  - `c2w` 4x4.

Add unit tests for these conversions.

Never silently mix conventions.

---

## Long-video requirements

Support chunked / streaming execution for long videos.

Rules:

- process frames in chunks,
- preserve stable file naming,
- merge chunk outputs into a single global result,
- never hold the entire video in GPU memory,
- make chunk size configurable.

---

## CLI requirements

Implement:

```bash
python -m dynamic_recon.run --help
```

Required arguments:

- `--video`
- `--output`

Important optional arguments:

- `--fps`
- `--max-frames`
- `--prompts`
- `--sam3-checkpoint`
- `--da3-model`
- `--chunk-size`
- `--use-ray-pose`
- `--ref-view-strategy`
- `--dynamic-thresh`
- `--confidence-thresh`
- `--save-intermediate`

---

## Config file requirements

Create `configs/pipeline.default.yaml` with sensible defaults, including:

```yaml
video:
  fps: 5
  max_frames: null
  resize_long_edge: 768

da3:
  model_name: depth-anything/DA3-LARGE-1.1
  use_ray_pose: true
  ref_view_strategy: middle
  chunk_size: 32
  confidence_thresh: 0.2

sam3:
  prompts: [person, car]
  checkpoint: default
  text_prompt_frame: 0

fusion:
  dynamic_thresh: 0.08
  reproj_thresh_px: 3.0
  uncertain_margin: 0.02
  min_depth: 0.1
  max_depth: 100.0
  neighborhood_radius: 2

output:
  save_intermediate: true
  save_visualizations: true
```

Threshold values are only initial guesses. Keep them configurable.

---

## Engineering requirements

- use type hints,
- use dataclasses or small typed containers for outputs,
- keep all paths relative/project-root friendly,
- add logging with clear stage boundaries,
- fail fast on missing checkpoints,
- do not swallow exceptions,
- add a dry-run mode that checks configs and dependencies without full inference.

---

## Testing requirements

Add at least lightweight tests or smoke checks for:

1. pose convention conversions,
2. depth backprojection,
3. projection from world points into image coordinates,
4. fusion logic on a tiny synthetic example,
5. CLI config parsing.

If full model tests are too heavy, include mock-backed smoke tests.

---

## README requirements

Update `README.md` so a new user can:

1. clone the repo,
2. init submodules,
3. create the environment,
4. authenticate to Hugging Face if needed,
5. download checkpoints,
6. run the demo command,
7. understand the output folder structure.

Include a short section called **Known limitations**:

- DA3 poses may drift on highly dynamic or texture-poor videos,
- SAM 3 tracks depend on prompt quality and checkpoint availability,
- 3D dynamic classification is sensitive to pose/depth noise and occlusions,
- the first version is research-grade, not production-grade.

---

## Acceptance criteria

The task is complete only when all of the following are true:

1. the repo contains both upstream projects as submodules,
2. the environment setup scripts exist and are documented,
3. the pipeline runs end-to-end on a short demo video,
4. DA3 outputs poses, intrinsics, depth, and a point cloud,
5. SAM 3 outputs tracked object masks,
6. the fusion stage exports dynamic/static/uncertain masks,
7. the final output contains separate static and dynamic point clouds,
8. the README explains exactly how to reproduce the run.

---

## Implementation guidance

Prefer simple, inspectable code over premature optimization.

For the first usable version:

- use DA3 for geometry and pose,
- use SAM 3 for object priors,
- use a nearest-neighbor or bilinear sampling-based static-consistency test,
- use confidence filtering and thresholded fusion,
- save lots of intermediate artifacts for debugging.

Do not attempt to build a full differentiable bundle adjustment system in v1.
This repo is an orchestration and fusion project, not a reimplementation of MegaSaM.

---

## Nice-to-have follow-ups

If time allows after the main pipeline works:

- optional Open3D viewer for the labeled point cloud,
- support for manual prompt seeding from a JSON annotation file,
- temporal smoothing of dynamic masks,
- per-object 3D trajectories,
- export COLMAP text files for downstream reconstruction tools,
- optional feature matching / BA refinement on static-only regions.

