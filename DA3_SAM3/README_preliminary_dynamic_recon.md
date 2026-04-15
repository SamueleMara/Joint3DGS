# Preliminary Dynamic Recon

## Purpose

This repository contains a preliminary monocular video pipeline for combining:

- Depth Anything 3 geometry and camera estimates
- static-world reprojection residuals
- a soft geometry-derived dynamic prior
- SAM3 promptable video masks
- lightweight fusion and export logic

This is a preliminary stage for dynamic/static video understanding. It is not yet a full 4D scene optimization or 3DGS pipeline.

## What is implemented

### Package structure

The current implementation is organized as:

```text
dynamic_recon/
├─ cli/
├─ config/
├─ da3/
├─ data/
├─ fusion/
├─ geometry/
├─ pipelines/
├─ sam3/
└─ tracking/
```

### Implemented modules

`dynamic_recon/config`
- dataclass config schema
- YAML config loading

`dynamic_recon/data`
- OpenCV video reading
- frame naming
- cache read/write helpers

`dynamic_recon/da3`
- DA3 output containers
- thin adapter/wrapper layer
- official `DepthAnything3.from_pretrained(...)` adapter path
- configurable mock fallback via `da3.allow_mock`

`dynamic_recon/geometry`
- torch camera transforms
- projection and backprojection
- image/map sampling
- visibility logic
- pair residual computation
- dynamic prior computation
- placeholder static-only pose refinement hook

`dynamic_recon/sam3`
- prompt generation from geometry seeds
- thin video wrapper normalization
- official predictor path via `sam3.model_builder.build_sam3_predictor(...)`
- temporary SAM2 backend path via `sam2.sam2_video_predictor.SAM2VideoPredictor`
- instance helper utilities

`dynamic_recon/fusion`
- stacked per-pixel fusion features
- lightweight CNN fusion head
- basic losses and one-epoch trainer

`dynamic_recon/tracking`
- static reprojection tracks
- dynamic region track export helpers
- support-point sampling and propagation placeholders

`dynamic_recon/pipelines`
- `preprocess.py`
- `infer_initial.py`
- `iterate.py`
- `export_results.py`

`dynamic_recon/cli`
- runnable command-line interfaces for the preliminary pipeline

### Repo-level changes

- setup/install shell scripts were moved from `scripts/` to `setup/`
- Python launcher scripts were added to `scripts/`
- new configs were added:
  - `configs/base.yaml`
  - `configs/quickstart.yaml`
  - `configs/highres.yaml`
  - `configs/convergence.yaml`
  - `configs/convergence_7gb.yaml`
  - `configs/convergence_7gb_safe.yaml`
- new tests were added:
  - `tests/test_projection.py`
  - `tests/test_visibility.py`
  - `tests/test_dynamic_prior.py`
  - `tests/test_support_points.py`
  - `tests/test_fusion_shapes.py`

## Installation

### 1. Conda environment

Current environment name:

```bash
conda activate da3-sam3-dynamic
```

If you need to rebuild it:

```bash
./setup/setup_env.sh
```

What `setup/setup_env.sh` does:

- creates or reuses the conda env `da3-sam3-dynamic`
- installs `torch==2.10.0` and `torchvision` from the CUDA 12.8 wheel index
- installs common runtime dependencies
- installs `third_party/depth_anything_3`, `third_party/sam2`, `third_party/sam3`, and this repo in editable mode
- installs the SAM3 fast-inference dependencies used by the official upstream code:
  - `einops`
  - `ninja`
  - `flash-attn-3`
- compiles the SAM2 CUDA extension in-place if `nvcc` is available

Environment assumptions used in this repo:

- Python `3.12`
- PyTorch CUDA build `cu128`
- CUDA toolkit available at `/usr/local/cuda-12.8`
- `TORCH_CUDA_ARCH_LIST=8.9` for RTX 4070 Laptop / Ada-class builds

Optional split setup paths:

```bash
./setup/setup_env_da3.sh
./setup/setup_env_sam3.sh
```

### 2. Submodules

The upstream repos should live here:

- `third_party/depth_anything_3`
- `third_party/sam2`
- `third_party/sam3`

To initialize or refresh them:

```bash
./setup/setup_submodules.sh
```

Expected local layout after setup:

```text
third_party/
├─ depth_anything_3/
├─ sam2/
└─ sam3/
```

All three repos are installed into the conda env in editable mode, so Python imports resolve directly into those folders.

### 3. Checkpoints

Checkpoints are not stored in git.

If you have Hugging Face access configured:

```bash
export HF_TOKEN=...
./setup/download_checkpoints.sh
```

For SAM3, the official upstream instructions require checkpoint access approval plus local Hugging Face authentication before auto-downloads will work:

```bash
hf auth login
```

You can either let the upstream predictor auto-download checkpoints at runtime or set `sam3.checkpoint` to a local checkpoint path.

If the `hf` CLI is not available in your env, the equivalent Python login is:

```bash
conda run -n da3-sam3-dynamic python -c "from huggingface_hub import login; login(token='YOUR_TOKEN', add_to_git_credential=False)"
```

### 4. DA3 model selection

The DA3 wrapper now follows the upstream API and model-card guidance from the official repository.

Preferred DA3 models:

- `depth-anything/DA3-LARGE-1.1`
  - recommended default any-view model
  - supports relative depth, pose estimation, and pose-conditioned depth
- `depth-anything/DA3NESTED-GIANT-LARGE-1.1`
  - stronger nested metric model
  - non-commercial license

Relevant DA3 config fields:

- `da3.model_name`
- `da3.checkpoint`
- `da3.use_ray_pose`
- `da3.ref_view_strategy`
- `da3.process_res`
- `da3.process_res_method`
- `da3.sequence_chunk_size`
- `da3.sequence_chunk_overlap`

For temporally ordered videos, `ref_view_strategy: middle` is the current default in this repo.
For low-VRAM GPUs, `da3.sequence_chunk_size` can be used to run DA3 on smaller temporal chunks instead of the full sampled video at once. This reduces memory pressure but weakens global multi-view consistency across chunk boundaries.

### 5. SAM2 installation and compilation

This repo uses the official SAM2 code from:

- `third_party/sam2`

The conda env install is editable:

```bash
conda activate da3-sam3-dynamic
python -m pip install -e third_party/sam2
```

To build the optional SAM2 CUDA extension manually:

```bash
conda activate da3-sam3-dynamic
cd third_party/sam2
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export TORCH_CUDA_ARCH_LIST=8.9
export SAM2_BUILD_ALLOW_ERRORS=0
python setup.py build_ext --inplace
```

Why this is needed:

- without the compiled extension, SAM2 still runs, but you see the `_C` warning and post-processing is skipped
- with the compiled extension, `sam2._C` should resolve from:
  - `third_party/sam2/sam2/_C.so`

Verification:

```bash
conda run -n da3-sam3-dynamic python -c "import sam2, sam2._C; print(sam2.__file__); print(sam2._C.__file__)"
```

Repo-specific note:

- this repo preloads Torch shared libraries from the conda env before importing `sam2._C`
- this was necessary so the compiled extension can find `libc10.so`, `libtorch_cpu.so`, and related Torch libs at runtime

### 6. SAM3 installation and runtime requirements

The SAM3 wrapper now follows the upstream request/stream API from the official repository.

Current behavior:

- it uses `sam3.model_builder.build_sam3_predictor(...)`
- `sam3.version` selects the upstream family
- this repo now defaults to `sam3.1`
- if `sam3.checkpoint` is unset, the upstream code attempts a Hugging Face download
- if `sam3.allow_mock: false`, startup fails hard when the official predictor cannot be built
- if `sam3.version` starts with `sam2`, the wrapper uses the open SAM2 video predictor instead

Relevant SAM3 config fields:

- `sam3.version`
- `sam3.model_id`
- `sam3.checkpoint`
- `sam3.gpus_to_use`
- `sam3.offload_video_to_cpu`
- `sam3.offload_state_to_cpu`
- `sam3.allow_mock`

Install path used by this repo:

```bash
conda activate da3-sam3-dynamic
python -m pip install -e third_party/sam3
python -m pip install einops ninja
python -m pip install flash-attn-3 --no-deps --index-url https://download.pytorch.org/whl/cu128
```

SAM3 access requirements:

- approved Hugging Face access to `facebook/sam3.1`
- a valid Hugging Face login in the current machine/user environment

Verification:

```bash
conda run -n da3-sam3-dynamic python -c "from sam3.model_builder import build_sam3_predictor; p = build_sam3_predictor(version='sam3.1'); print(type(p).__name__)"
```

Repo-specific runtime notes:

- `sam3` is imported from `third_party/sam3`
- this repo currently uses the official predictor on the MP4 path for `sam3`
- `sam2` uses the sampled JPEG frame directory exported by preprocess
- the vendored `third_party/sam3/sam3/perflib/fa3.py` was patched in this repo to use `bfloat16` instead of `float8`, because the local Ada GPU + installed FlashAttention path requires `fp16`/`bf16`
- the vendored `third_party/sam3/sam3/model/sam3_multiplex_tracking.py` was patched so point-prompt frames are marked correctly before propagation

Important upstream constraints:

- the official video predictor expects a video path or a JPEG frame directory on disk
- it does not consume the in-memory `numpy` frame list used by the older mock path
- for `sam2`, the JPEG folder must use numeric filenames such as `000000.jpg`

### 7. Making SAM2 and SAM3 accessible in the conda env

The expected import targets are:

```bash
conda run -n da3-sam3-dynamic python -c "import sam2, sam3; print(sam2.__file__); print(sam3.__file__)"
```

They should resolve into:

- `third_party/sam2/...`
- `third_party/sam3/...`

If they do not, reinstall editable packages:

```bash
conda activate da3-sam3-dynamic
python -m pip uninstall -y SAM-2 sam3
python -m pip install -e third_party/sam2 -e third_party/sam3
```

## CLI overview

### `python -m dynamic_recon.cli.run_pipeline`

This is the main entrypoint.

What it does:

- loads the video
- runs the preliminary DA3 path
- computes first-pass geometry residuals and dynamic prior
- derives prompts and runs the segmentation backend selected with `--sam-backend`
- runs the lightweight fusion step
- exports dynamic probability and track artifacts

Arguments:

- `--video`: required input video path
- `--outdir`: required output directory
- `--config`: config path, default `configs/quickstart.yaml`
- `--device`: optional config override
- `--num-outer-iters`: optional outer-loop override
- `--sam-backend {sam2,sam3}`: choose open SAM2 or gated SAM3
- `--skip-da3`: accepted for forward compatibility, not yet wired
- `--skip-sam3`: accepted for forward compatibility, not yet wired
- `--resume`: accepted for forward compatibility, not yet wired

Example:

```bash
python -m dynamic_recon.cli.run_pipeline \
  --video /path/to/input.mp4 \
  --outdir outputs/demo_run \
  --config configs/quickstart.yaml \
  --sam-backend sam2
```

### `python -m dynamic_recon.cli.run_da3`

Purpose:

- preprocess-only path
- extracts frames
- runs the DA3 wrapper through the upstream `DepthAnything3` API when a real model id or checkpoint is configured
- caches DA3 outputs
- computes initial residuals and dynamic prior

Arguments:

- `--video`
- `--outdir`
- `--config`

Example:

```bash
python -m dynamic_recon.cli.run_da3 \
  --video /path/to/input.mp4 \
  --outdir outputs/demo_run \
  --config configs/base.yaml
```

### `python -m dynamic_recon.cli.run_sam3`

Purpose:

- runs the preliminary DA3/geometry step
- converts the dynamic prior into prompts
- runs the selected segmentation backend
- stores logits and masks

Arguments:

- `--video`
- `--seeds-dir`
  - accepted right now but not used yet
- `--outdir`
- `--config`
- `--sam-backend {sam2,sam3}`

Example:

```bash
python -m dynamic_recon.cli.run_sam3 \
  --video /path/to/input.mp4 \
  --outdir outputs/demo_run \
  --config configs/base.yaml \
  --sam-backend sam3
```

### `python -m dynamic_recon.cli.export_tracks`

Purpose:

- records a track export request into the run directory
- current implementation writes the export request JSON and serves as the CLI skeleton for fuller export behavior

Arguments:

- `--run-dir`
- `--query-frame`
- `--query-grid-step`
- `--mode` with `static|dynamic|both`

Example:

```bash
python -m dynamic_recon.cli.export_tracks \
  --run-dir outputs/demo_run \
  --query-frame 0 \
  --mode both
```

## Python launcher scripts

These are thin wrappers around the module CLIs:

- `python scripts/run_preliminary_pipeline.py ...`
- `python scripts/run_preliminary_da3.py ...`
- `python scripts/run_preliminary_sam3.py ...`
- `python scripts/export_preliminary_tracks.py ...`

## Output layout

A typical run writes:

```text
outputs/<run_name>/
├─ frames/
├─ da3/
│  ├─ depth/
│  ├─ intrinsics/
│  ├─ extrinsics/
│  ├─ confidence/
│  └─ metadata.json
├─ geometry/
│  ├─ residual_3d/
│  ├─ residual_depth/
│  ├─ residual_feat/
│  ├─ residual_cycle/
│  ├─ visibility/
│  └─ dynamic_prior/
├─ sam3/
│  ├─ prompts/
│  ├─ logits/
│  └─ masks/
└─ exports/
   ├─ dynamic_prob.npy
   ├─ dynamic_prob.png
   ├─ static_tracks.json
   ├─ dynamic_tracks.json
   └─ support_tracks.json
```

## Recommended usage

### Fast run

```bash
conda activate da3-sam3-dynamic

python -m dynamic_recon.cli.run_pipeline \
  --video /path/to/input.mp4 \
  --outdir outputs/demo_run \
  --config configs/quickstart.yaml
```

### Step-by-step

```bash
python -m dynamic_recon.cli.run_da3 \
  --video /path/to/input.mp4 \
  --outdir outputs/demo_run \
  --config configs/base.yaml

python -m dynamic_recon.cli.run_sam3 \
  --video /path/to/input.mp4 \
  --outdir outputs/demo_run \
  --config configs/base.yaml

python -m dynamic_recon.cli.export_tracks \
  --run-dir outputs/demo_run \
  --query-frame 0 \
  --mode both
```

## Config files

- `configs/base.yaml`
  - baseline settings for the preliminary pipeline
- `configs/quickstart.yaml`
  - reduced scale and iteration count for smoke runs
- `configs/highres.yaml`
  - larger scale and more outer iterations
- `configs/convergence.yaml`
  - higher-budget preset for longer real runs on larger GPUs
  - disables mock DA3/SAM3 fallback
  - defaults to `sam3.version: sam3.1`
  - increases outer-loop count, fusion epochs, pair baselines, and pose refinement steps
- `configs/convergence_7gb.yaml`
  - lower-VRAM convergence-oriented preset for GPUs around 8 GB
  - keeps the same outer-loop budget but reduces DA3 frame count and inference resolution
  - enables chunked DA3 inference
- `configs/convergence_7gb_safe.yaml`
  - stricter low-VRAM preset for GPUs that still OOM on `configs/convergence_7gb.yaml`
  - uses fewer frames, lower DA3 resolution, and smaller DA3 chunks
  - currently routes the segmentation stage through open `SAM2.1` with CPU offload enabled

## Verification

Verified in the dedicated conda environment:

- `python -m dynamic_recon.cli.run_pipeline --help`
- synthetic smoke run of the preliminary pipeline
- tests:
  - projection
  - visibility
  - dynamic prior
  - support-point sampling
  - fusion output shapes

## Known limitations

- The current `run_pipeline` implementation is still a preliminary scaffold. A config with more iterations can increase runtime and optimization budget, but it does not guarantee true convergence in the optimization sense.
- The DA3 wrapper is now wired to the official `DepthAnything3.from_pretrained(...)` API, but the rest of the pipeline around it is still preliminary.
- The SAM3 wrapper now uses the official predictor API, but successful runtime still depends on `psutil`, Hugging Face authentication, and valid checkpoint access.
- Some CLI flags are placeholders for future resume/skip logic:
  - `--skip-da3`
  - `--skip-sam3`
  - `--resume`
  - `--seeds-dir` in `run_sam3`
- `dynamic_recon.geometry.pose_refine` is currently a placeholder hook rather than a full optimizer.
- The fusion trainer is intentionally minimal.
- High-fidelity runs still depend on correct DA3 and SAM3 checkpoints plus upstream-compatible model configuration.
- Real DA3 inference requires a valid model id or checkpoint path. The upstream repo recommends the refreshed `-1.1` family. This repo defaults to `depth-anything/DA3-LARGE-1.1`.
- Real SAM3 inference requires either a local checkpoint or authenticated Hugging Face access to the official checkpoint repo. This repo defaults to `sam3.version: sam3.1`.
