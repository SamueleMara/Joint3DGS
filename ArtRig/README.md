# ArtRig (articulation-rigidity)

Two-stage articulated object pipeline:
1. segmentation by rigidity/motion consistency
2. joint estimation from relative motion (revolute/prismatic/screw)

Includes:
- single-view wrappers (`run_pipeline.py`, `run_pipeline_folder.py`)
- multiview wrapper (`run_pipeline_multiview.py`)
- DA3 preprocessing (`preprocess_video_da3_folder.py`)

## 1. Clone + Submodules

Clone with submodules:

```bash
git clone --recurse-submodules <YOUR_GITHUB_URL> ArtRig
cd ArtRig
```

If you already cloned without submodules:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

Current submodules:
- `submodules/depth_anything_3` (Depth Anything v3)
- `submodules/alltracker`
- `submodules/tapnet`

## 2. Environment Setup

```bash
conda create -y -n art-rig python=3.10 pip
conda activate art-rig
python -m pip install --upgrade pip
python -m pip install -e ".[dev,extras]"
```

Install editable submodule packages that are Python-installable:

```bash
python -m pip install -e submodules/depth_anything_3 --no-deps --no-build-isolation
python -m pip install -e submodules/tapnet --no-deps --no-build-isolation
```

`alltracker` is consumed via `repo_path` and is not installed as a package.

## 3. Optional: CoTracker Local Repo

If you use `backend: cotracker`, keep CoTracker available at:
- `.third_party/repos/co-tracker`

The wrappers accept `--cotracker-repo` when needed.

## 4. Input Formats

### 4.1 Single-view folder format (`run_pipeline_folder.py`)

Expected structure:

```text
dataset/
  images/*.png
  depth/*.png
  fg_mask/*.png
  intrinsics.npy
  extrinsics.npy
  meta.json   # optional, can include depth_scale
```

### 4.2 Multiview format (`run_pipeline_multiview.py`)

Expected structure:

```text
dataset/
  rgb/cam_000/*.png
  depth_npy/cam_000/*.npy   # or another depth root
  mask/cam_000/*.png
  ...
  metadata/cameras.json
```

## 5. Main Commands

### 5.1 Preprocess video with DA3 into folder dataset

```bash
python preprocess_video_da3_folder.py \
  --video /path/to/video.mp4 \
  --output-dir /path/to/dataset_folder \
  --fps 1.0 \
  --model-dir depth-anything/DA3-SMALL \
  --device cpu
```

### 5.2 Run full single-view folder pipeline

```bash
python run_pipeline_folder.py \
  --input-dir /path/to/dataset_folder \
  --tracker-config configs/tracker.yaml \
  --seg-config configs/segmentation.yaml \
  --joint-config configs/joint.yaml \
  --use-dino-features \
  --out-dir outputs/folder_pipeline \
  --progress --loss-debug
```

### 5.3 Run full multiview training on `fr3_joint1`

```bash
python run_pipeline_multiview.py \
  --input-dir /home/samuelemara/Joint3DGS/ArtRig/inputs/fr3_joint1 \
  --rgb-root rgb \
  --depth-root depth_npy \
  --mask-root mask \
  --camera-pattern 'cam_*' \
  --cameras-json metadata/cameras.json \
  --depth-npy-scale 1.0 \
  --tracker-config outputs/tracker_cotracker_cpu_ckpt.yaml \
  --seg-config configs/segmentation.yaml \
  --joint-config configs/joint.yaml \
  --out-dir outputs/fr3_joint1_multiview_full \
  --progress --refine-global
```

Fused multiview result:
- `outputs/fr3_joint1_multiview_full/joint_fused_world.json`
- `outputs/fr3_joint1_multiview_full/joint_fused_world.pt`

## 6. Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

## 7. Notes for GitHub

- `outputs/`, caches, and local artifacts are ignored by `.gitignore`.
- Submodules are managed by `.gitmodules`; run `git submodule update --init --recursive` after clone.
