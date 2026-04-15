# Codex Context: DA3 + SAM3 Preliminary Static/Dynamic Understanding and Pixel Tracking

## Objective

Implement a **preliminary monocular video pipeline** in Python that:

1. runs **Depth Anything 3 (DA3)** over a video to estimate per-frame depth and camera parameters;
2. uses those outputs to compute **static-world reprojection residuals** and derive a **soft dynamic prior**;
3. uses **SAM3** to turn high-confidence moving regions into **temporally consistent dynamic instance masks**;
4. fuses geometry and SAM3 signals into a refined **static/dynamic pixel classification**;
5. refines poses using only confident **static** pixels;
6. tracks:
   - **static pixels** by 3D reprojection under refined poses/depth;
   - **dynamic regions** by SAM3 masks plus sparse support-point tracks.

This is **not** the full Shape-of-Motion / 3DGS system yet. It is a buildable preliminary stage focused on:
- static vs dynamic understanding,
- pixel/region tracking,
- iterative consistency between geometry and video segmentation.

## Important design principle

Do **not** trust a single source of truth.

Use:
- DA3 geometry and poses,
- geometric residuals under a static-world assumption,
- SAM3 promptable video masks,
- temporal consistency,
- optional sparse point tracks inside dynamic instances,

and optimize them to converge to a stable soft static/dynamic labeling and stable tracks.

This is inspired by the **optimization philosophy** of Shape of Motion:
fuse several noisy priors into one coherent dynamic interpretation of the video,
rather than trusting any single predictor.

---

## Repository / environment assumptions

These are already present and should be treated as the current ground truth:

- **Repo root:** `DA3_SAM3`
- **DA3 submodule:** `depth_anything_3`
- **DA3 nested submodule:** `salad`
- **SAM3 submodule:** `sam3`

Conda environment:
- **env name:** `da3-sam3-dynamic-recon`
- **env path:** `/home/samuelemara/miniconda3/envs/da3-sam3-dynamic-recon`

Installed in that conda env:
- Python `3.12.13`
- `torch 2.10.0+cu128`
- `torchvision 0.25.0+cu128`
- repo package `dynamic-recon` in editable mode
- runtime deps from `requirements.txt`
- dev deps from `requirements-dev.txt`, including `pytest`

Editable install target:
- Project source: `DA3_SAM3`

Not confirmed yet:
- `pip install -e third_party/depth_anything_3`
- `pip install -e third_party/sam3`
- model checkpoints are not downloaded

Activate with:
```bash
conda activate da3-sam3-dynamic-recon
```

---

## What Codex should build

Codex should implement a **clean, testable Python package** inside the repo that adds:

1. **video ingestion**
2. **DA3 inference wrapper**
3. **camera/depth cache writer**
4. **geometric residual computation**
5. **initial dynamic prior generation**
6. **SAM3 seeding + propagation wrapper**
7. **fusion model / refinement logic**
8. **static-only pose refinement utilities**
9. **static and dynamic tracking exporters**
10. **CLI entrypoints**
11. **basic tests**

The code should be modular and runnable from the command line on a single input video.

---

## Non-goals for this iteration

Do **not** implement yet:
- 3D Gaussian Splatting
- canonical-space dynamic Gaussians
- SE(3) basis optimization
- full end-to-end training through DA3 and SAM3
- full differentiable bundle adjustment over every module
- distributed training or cluster orchestration

Those can come later.

This iteration is about making a **reliable preliminary system**.

---

## High-level algorithm

For video frames `I_t`, DA3 estimates:
- intrinsics `K_t`
- extrinsics `E_t` (world-to-camera)
- depth `D_t`
- possibly confidence or auxiliary outputs

From those, define a static-world reprojection test.

For a pixel `p = (u, v)` in frame `t`, backproject into 3D:
```text
X_t(p) = E_t^{-1} ( D_t(p) * K_t^{-1} * p_h )
```
where `p_h = [u, v, 1]^T`.

Assuming the point is static, project it into frame `s`:
```text
p_hat_{t->s}(p) = project( K_s, E_s, X_t(p) )
```

Then compute residuals:
- 3D consistency residual
- depth consistency residual
- feature / photometric consistency residual
- forward-backward cycle residual
- visibility confidence

Combine them into an initial soft dynamic prior `g_t(p)`.

Then:
- convert the strongest moving regions into SAM3 prompts,
- use SAM3 to propagate dynamic instance masks,
- fuse the geometry prior and SAM3 masks,
- update static/dynamic probabilities,
- refine poses using only confident static pixels,
- recompute residuals,
- iterate.

---

## Recommended package layout

Codex should create something like:

```text
DA3_SAM3/
├─ dynamic_recon/
│  ├─ __init__.py
│  ├─ config/
│  │  ├─ default.py
│  │  └─ schema.py
│  ├─ data/
│  │  ├─ video_io.py
│  │  ├─ frame_dataset.py
│  │  └─ cache_io.py
│  ├─ da3/
│  │  ├─ __init__.py
│  │  ├─ wrapper.py
│  │  ├─ adapters.py
│  │  └─ types.py
│  ├─ geometry/
│  │  ├─ __init__.py
│  │  ├─ camera.py
│  │  ├─ projection.py
│  │  ├─ visibility.py
│  │  ├─ residuals.py
│  │  ├─ dynamic_prior.py
│  │  └─ pose_refine.py
│  ├─ sam3/
│  │  ├─ __init__.py
│  │  ├─ wrapper.py
│  │  ├─ prompts.py
│  │  ├─ propagation.py
│  │  └─ instances.py
│  ├─ fusion/
│  │  ├─ __init__.py
│  │  ├─ features.py
│  │  ├─ model.py
│  │  ├─ losses.py
│  │  └─ trainer.py
│  ├─ tracking/
│  │  ├─ __init__.py
│  │  ├─ static_tracks.py
│  │  ├─ dynamic_tracks.py
│  │  └─ support_points.py
│  ├─ pipelines/
│  │  ├─ __init__.py
│  │  ├─ preprocess.py
│  │  ├─ infer_initial.py
│  │  ├─ iterate.py
│  │  └─ export_results.py
│  ├─ viz/
│  │  ├─ __init__.py
│  │  ├─ overlay.py
│  │  ├─ video_debug.py
│  │  └─ pointcloud.py
│  └─ cli/
│     ├─ __init__.py
│     ├─ run_pipeline.py
│     ├─ run_da3.py
│     ├─ run_sam3.py
│     └─ export_tracks.py
├─ tests/
│  ├─ test_projection.py
│  ├─ test_visibility.py
│  ├─ test_dynamic_prior.py
│  ├─ test_support_points.py
│  └─ test_fusion_shapes.py
└─ configs/
   ├─ base.yaml
   ├─ quickstart.yaml
   └─ highres.yaml
```

If `dynamic_recon/` already exists, extend it instead of duplicating.

---

## File-by-file implementation guidance

## 1) `dynamic_recon/data/video_io.py`

Implement:
- `read_video_frames(video_path) -> list[np.ndarray]`
- `write_video(frames, output_path, fps)`
- `sample_keyframes(num_frames, stride=None, max_frames=None)`
- `get_video_metadata(video_path)`

Requirements:
- use OpenCV for I/O
- preserve frame indices
- return RGB arrays internally, not BGR
- support optional resize at read time
- support frame-range selection

Also provide a stable frame naming convention:
```text
frame_000000.png
frame_000001.png
...
```

## 2) `dynamic_recon/data/cache_io.py`

Implement structured cache read/write helpers.

Suggested cache tree:
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
│  ├─ masks/
│  ├─ instances/
│  └─ tracks/
├─ fusion/
│  ├─ masks/
│  ├─ logits/
│  └─ checkpoints/
└─ exports/
```

Use:
- `.npy` for arrays
- `.npz` for grouped numeric results
- `.json` for metadata
- `.png` for human-readable masks only when needed

Implement helpers:
- `save_array(path, array)`
- `load_array(path)`
- `save_json(path, obj)`
- `load_json(path)`

---

## 3) `dynamic_recon/da3/types.py`

Define typed containers, either `dataclass` or `TypedDict`.

Recommended:
```python
@dataclass
class DA3FrameOutput:
    frame_index: int
    depth: torch.Tensor          # [H, W]
    intrinsics: torch.Tensor     # [3, 3]
    extrinsics: torch.Tensor     # [4, 4] world-to-camera
    confidence: torch.Tensor | None = None
    rgb: torch.Tensor | None = None
    aux: dict[str, Any] | None = None
```

And:
```python
@dataclass
class DA3SequenceOutput:
    frames: list[DA3FrameOutput]
    fps: float
    height: int
    width: int
    source_video: str
```

---

## 4) `dynamic_recon/da3/wrapper.py`

Goal:
hide DA3-specific API details behind a stable project interface.

Implement:
- `load_da3_model(cfg) -> Any`
- `run_da3_on_frames(frames, cfg) -> DA3SequenceOutput`
- `run_da3_on_video(video_path, cfg) -> DA3SequenceOutput`

Important:
- keep this wrapper **thin**
- adapt to the actual installed DA3 API instead of hard-coding assumptions everywhere else
- if DA3 supports chunked or windowed processing, expose `window_size`, `stride`, `batch_size`
- if DA3 can return intrinsics/extrinsics directly, use them
- if it needs different inference modes for pose/depth, wrap that here

Add a single internal adapter layer:
```python
class DA3Adapter:
    def infer_sequence(self, frames: list[np.ndarray], cfg) -> DA3SequenceOutput: ...
```

This makes later DA3 API changes localized.

### Practical wrapper behavior
The wrapper should:
- convert numpy RGB frames to the DA3 input format
- move tensors to the configured device
- normalize outputs to:
  - depth `[H, W]`
  - intrinsics `[3,3]`
  - extrinsics `[4,4]`
- keep frame indices aligned

### Fallback behavior
If DA3 does not expose confidence, return `None`.
If DA3 does not expose intrinsics or extrinsics in a mode, raise a clear error with suggestions.

---

## 5) `dynamic_recon/geometry/camera.py`

Implement:
- `invert_extrinsics(E_wc) / invert_w2c(E_w2c)`
- `compose_extrinsics(A, B)`
- `normalize_intrinsics(K, width, height)` if needed
- SE(3)-safe utilities:
  - `to_homogeneous(points)`
  - `from_homogeneous(points_h)`

Use `torch` implementations, not NumPy-only, so the same functions can support refinement later.

---

## 6) `dynamic_recon/geometry/projection.py`

Implement the core math.

Functions:
- `pixel_grid(height, width, device, homogeneous=True) -> [H, W, 3]`
- `backproject(depth, K, E_w2c) -> world_points`
- `project(world_points, K, E_w2c) -> uv, z_cam`
- `sample_at_coords(image_or_map, uv) -> sampled_values, valid_mask`
- `reproject_static(depth_t, K_t, E_t, K_s, E_s)`

### Conventions
Pick one convention and document it clearly:
- `E_w2c`: world-to-camera 4x4 transform
- `X_world`
- `X_cam = E_w2c @ X_world_h`
- `uv = K @ X_cam[:3]`, divide by z

Return values:
- `uv`: float coordinates in pixel space
- `z_cam`: camera-space depth
- `valid_mask`: inside image and positive depth

Use `torch.nn.functional.grid_sample` for differentiable sampling if possible.

---

## 7) `dynamic_recon/geometry/visibility.py`

Implement robust visibility logic to avoid false dynamic labels from occlusions.

Functions:
- `inside_image_mask(uv, height, width)`
- `positive_depth_mask(z_cam, min_depth=1e-6)`
- `occlusion_mask(pred_depth, observed_depth, tol_abs, tol_rel)`
- `forward_backward_consistency(...)`

Define visibility confidence:
```python
vis_conf = inside * positive_depth * not_occluded * cycle_consistent
```
or a soft equivalent.

Use soft scores when possible, not only booleans.

---

## 8) `dynamic_recon/geometry/residuals.py`

This is the core of the preliminary system.

Implement per-pixel residuals between frame `t` and frame `s` under the **static-world assumption**.

### Required residual maps

1. **3D consistency residual**
```text
r_3d(p,t,s) = || X_s( p_hat_{t->s}(p) ) - X_t(p) ||_2
```
Implementation detail:
- backproject frame `t`
- reproject into frame `s`
- sample depth from frame `s`
- backproject sampled points from frame `s`
- compare in world coordinates

2. **Depth residual**
```text
r_depth(p,t,s) = | D_s( p_hat_{t->s}(p) ) - z_hat_{t->s}(p) |
```

3. **Photometric / feature residual**
Initially implement simple RGB residual:
```text
r_rgb(p,t,s) = || I_s( p_hat_{t->s}(p) ) - I_t(p) ||_1
```
Later optionally replace or augment with deep features.

4. **Forward-backward cycle residual**
Project `t -> s -> t`, compare return location to original pixel.

5. **Visibility confidence**
A confidence map that downweights:
- projected points outside the image,
- invalid or negative depths,
- strong occlusions,
- depth discontinuity neighborhoods.

### API
Implement:
```python
@dataclass
class PairResiduals:
    uv_ts: torch.Tensor
    z_ts: torch.Tensor
    r_3d: torch.Tensor
    r_depth: torch.Tensor
    r_rgb: torch.Tensor
    r_cycle: torch.Tensor
    visibility: torch.Tensor
```

and:
```python
def compute_pair_residuals(
    frame_t, frame_s, da3_t, da3_s, cfg
) -> PairResiduals:
    ...
```

---

## 9) `dynamic_recon/geometry/dynamic_prior.py`

Turn pairwise residuals into an initial soft dynamic prior.

Start with a simple logistic formulation:
```python
logit = (
    alpha_3d   * normalize(r_3d)
    + alpha_z  * normalize(r_depth)
    + alpha_rgb * normalize(r_rgb)
    + alpha_cycle * normalize(r_cycle)
    - alpha_vis * normalize(visibility)
)
g = torch.sigmoid(logit)
```

Then aggregate across several source/target pairs:
- adjacent pairs: `(t, t+1)`, `(t, t-1)`
- medium-baseline pairs
- optionally keyframe pairs

Aggregation suggestion:
```python
g_t = weighted_mean([g_t_from_s1, g_t_from_s2, ...], weights=visibility)
```

Also compute:
- `static_conf_t = 1 - g_t`
- `moving_seed_mask_t = g_t > seed_threshold_high`
- `static_seed_mask_t = g_t < static_threshold_low`

### Important
This stage should be conservative:
- use **high precision / low recall** thresholds for moving seeds
- it is fine to miss some dynamic pixels initially
- avoid flooding SAM3 with noisy prompts

---

## 10) `dynamic_recon/sam3/prompts.py`

Convert geometry-derived seeds into SAM3 prompts.

Implement:
- `extract_connected_components(mask)`
- `mask_to_boxes(mask)`
- `select_top_regions(mask, score_map, max_regions)`
- `sample_click_points(mask, num_points_per_region)`
- `build_prompts_from_dynamic_prior(...)`

Recommended prompt strategy:
- for each connected component in high-confidence moving seeds:
  - compute bounding box
  - sample 1-3 positive points inside
  - optionally sample negative points around the boundary / static ring
- keep prompts stable across nearby frames

Prompt data structure:
```python
@dataclass
class PromptRegion:
    frame_index: int
    box_xyxy: tuple[int, int, int, int] | None
    positive_points: list[tuple[int, int]]
    negative_points: list[tuple[int, int]]
    score: float
```

---

## 11) `dynamic_recon/sam3/wrapper.py`

Wrap the actual SAM3 video predictor.

Implement:
- `load_sam3_predictor(cfg)`
- `start_video_session(video_path_or_frames, cfg)`
- `add_prompts(session, prompts)`
- `propagate_masks(session) -> sequence outputs`

Normalize outputs into a stable project format:
```python
@dataclass
class SAM3FrameOutput:
    frame_index: int
    logits: torch.Tensor        # [N_inst, H, W] or [H, W] for merged
    masks: torch.Tensor         # bool / float
    instance_ids: list[int]
    scores: torch.Tensor | None = None
```

And sequence container:
```python
@dataclass
class SAM3SequenceOutput:
    frames: list[SAM3FrameOutput]
```

### Practical notes
- Do not assume one API shape forever.
- Keep the wrapper thin and defensive.
- Save per-frame raw SAM3 logits if available, not only binary masks.

This is important because the fusion model should consume **soft logits**, not only thresholded masks.

---

## 12) `dynamic_recon/sam3/instances.py`

Implement utilities to turn SAM3 raw outputs into temporally consistent instance structures.

Functions:
- `merge_overlapping_instances`
- `compute_instance_centroids`
- `compute_instance_boxes`
- `instance_iou`
- `track_instance_lifetimes`
- `export_instance_table`

Track IDs should remain stable across the sequence as provided by SAM3 when possible.
If SAM3 returns unstable IDs, add a post-hoc IoU + centroid association pass.

---

## 13) `dynamic_recon/tracking/support_points.py`

For dynamic regions, add sparse support points inside each SAM3 instance.

Goal:
dynamic pixel tracking is difficult to define densely at first pass.
Use:
- region identity from SAM3
- sparse support tracks inside each region

Implement:
- `sample_support_points(mask, num_points, method="grid_or_farthest")`
- `filter_support_points_by_texture(...)`
- `propagate_support_points(...)`

For propagation, simplest options:
- use local geometry and mask overlap
- or integrate an optical-flow / point tracker later

For this first iteration, allow a simpler approach:
- keep support points tied to instance masks frame-to-frame by nearest valid projection / optical flow placeholder

Return:
```python
@dataclass
class SupportTrack:
    instance_id: int
    point_id: int
    frames: list[int]
    xy: list[tuple[float, float]]
    valid: list[bool]
```

---

## 14) `dynamic_recon/fusion/features.py`

Build fused per-pixel features.

Inputs should include:
- RGB image `[3,H,W]`
- depth `[1,H,W]`
- optional DA3 confidence `[1,H,W]`
- residual maps:
  - `r_3d`
  - `r_depth`
  - `r_rgb`
  - `r_cycle`
  - `visibility`
- SAM3 merged logits or per-instance max-logit
- optional boundary map / image gradient
- optional temporal summary from neighboring frames

Create one stacked tensor:
```python
fusion_input_t: [C, H, W]
```

Implement:
- `build_fusion_features(frame_t, da3_t, residual_bundle_t, sam3_t, cfg)`

---

## 15) `dynamic_recon/fusion/model.py`

Implement a lightweight fusion head.

Do **not** build a giant model first.

A small UNet or shallow CNN is enough:
- input channels `C`
- output:
  - `dynamic_logit [1,H,W]`
  - optionally `boundary_logit [1,H,W]`

Recommended model:
- 3-4 levels max
- GroupNorm or BatchNorm
- ReLU / SiLU
- no fancy dependencies

Suggested API:
```python
class DynamicFusionNet(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32):
        ...
    def forward(self, x) -> dict[str, torch.Tensor]:
        return {
            "dynamic_logit": dynamic_logit,
            "boundary_logit": boundary_logit,   # optional
        }
```

---

## 16) `dynamic_recon/fusion/losses.py`

The fusion head should not pretend to have hard ground truth.
Use weak/self-supervised consistency losses.

Recommended losses:

### A. SAM agreement loss
Only where SAM3 is confident.
```python
L_sam = weighted_bce(pred_dyn, sam3_dyn_soft, weight=sam3_conf)
```

### B. Geometry consistency loss
Predicted static pixels should have low residuals:
```python
L_geo = mean( (1 - pred_dyn_prob) * robust(r_3d + r_depth + r_cycle) )
```

### C. Static prior anchoring
Where geometry is very confidently static:
```python
L_static_seed = BCE(pred_dyn_prob[static_seed], 0)
```

### D. Dynamic prior anchoring
Where geometry is very confidently dynamic:
```python
L_dynamic_seed = BCE(pred_dyn_prob[moving_seed], 1)
```

### E. Spatial smoothness / edge-aware regularization
Encourage piecewise smooth masks, but preserve edges:
```python
L_tv = edge_aware_tv(pred_dyn_prob, image)
```

### F. Temporal consistency
Along correspondences or tracks:
```python
L_temp = | pred_dyn_t(p) - pred_dyn_s(p_hat) |
```

Total loss:
```python
L = (
    w_sam * L_sam
    + w_geo * L_geo
    + w_static_seed * L_static_seed
    + w_dynamic_seed * L_dynamic_seed
    + w_tv * L_tv
    + w_temp * L_temp
)
```

Use robust penalties like Charbonnier or Huber where appropriate.

---

## 17) `dynamic_recon/fusion/trainer.py`

Implement:
- dataset assembly from cached outputs
- minibatch sampling by frame or frame-pair
- one-epoch train loop
- validation/debug metrics

Need not be heavy.
This can be:
- frame-based training
- or short temporal windows

### Suggested training phases

#### Phase 1
Train only fusion head on fixed DA3 + fixed SAM3 outputs.

#### Phase 2
After fusion stabilizes, rerun pose refinement using confident static pixels, then regenerate residuals.

#### Phase 3
Retrain / finetune fusion head with updated geometry residuals.

This alternating scheme is easier and more stable than full end-to-end training.

---

## 18) `dynamic_recon/geometry/pose_refine.py`

This is a simplified static-only pose refinement stage.

Do not try to fully re-train DA3.

Instead:
- keep DA3 depth fixed initially,
- refine per-frame extrinsics (and optionally a small depth scale/shift),
- optimize only on confident static pixels.

Possible implementation:
- parameterize each pose update as a small SE(3) delta
- use `torch.optim.Adam`
- loss based on static reprojection residuals across neighboring frames

Suggested API:
```python
def refine_poses_static_only(
    sequence_output,
    dynamic_probs,
    rgb_frames,
    cfg,
) -> list[torch.Tensor]:
    ...
```

### Pose refinement loss
Use only pixels with high static probability:
```python
L_pose = mean(
    static_weight * robust(r_3d + lambda_z * r_depth + lambda_rgb * r_rgb)
)
```

Where:
```python
static_weight = clamp(1 - dynamic_prob, min=0, max=1)
```

### Practical simplification
At first:
- refine only a subset of keyframe poses
- interpolate or keep others fixed
- or refine a short sliding window

This reduces instability.

---

## 19) `dynamic_recon/tracking/static_tracks.py`

Static tracking is the easiest part.

Given a set of query pixels in frame `t`:
- backproject them using `D_t, K_t, E_t`
- project them into any target frame `s`

Implement:
- `track_static_pixels(query_frame_idx, query_points_xy, da3_seq, poses_refined)`

Return:
```python
@dataclass
class StaticTrackResult:
    query_frame_idx: int
    points_xy: np.ndarray          # [N,2]
    tracks_xy: np.ndarray          # [T,N,2]
    valid: np.ndarray              # [T,N]
```

Also support dense-grid queries and random sparse queries.

---

## 20) `dynamic_recon/tracking/dynamic_tracks.py`

Dynamic tracking in this preliminary version should be **region-centric**, not pretending to be perfect dense identity tracking.

Implement:
- `track_dynamic_instances_from_sam3(...)`
- `attach_support_points_to_instances(...)`
- `export_dense_mask_tracks(...)`
- `export_sparse_support_tracks(...)`

Output two things:
1. dynamic mask tracks per instance across frames
2. sparse support-point tracks inside each instance

This is the recommended compromise for the first build.

---

## 21) `dynamic_recon/pipelines/preprocess.py`

Implement:
- extract frames
- run DA3
- cache outputs
- compute residual maps
- build initial dynamic prior

CLI should make this step runnable independently.

---

## 22) `dynamic_recon/pipelines/infer_initial.py`

Implement the first full pass:
1. load video
2. run DA3
3. compute residuals
4. build moving seeds
5. run SAM3
6. save overlays and masks

This is the fastest route to a useful prototype.

---

## 23) `dynamic_recon/pipelines/iterate.py`

Implement the alternating optimization loop.

Pseudo-code:
```python
for iter_idx in range(cfg.num_outer_iters):
    # 1. Geometry residuals from current DA3/poses
    residuals = compute_all_pair_residuals(...)

    # 2. Geometry-derived dynamic prior
    dyn_prior = build_dynamic_prior(residuals, ...)

    # 3. SAM3 seeding and propagation
    prompts = build_prompts_from_dynamic_prior(dyn_prior, ...)
    sam3_out = run_sam3(video, prompts, ...)

    # 4. Fusion training / inference
    fusion_features = build_all_fusion_features(...)
    fusion_model = train_or_update_fusion_head(...)
    dyn_prob = infer_dynamic_probs(...)

    # 5. Static-only pose refinement
    poses_refined = refine_poses_static_only(
        da3_seq, dyn_prob, rgb_frames, cfg.pose_refine
    )

    # 6. Update DA3 sequence state / cached extrinsics
    update_pose_cache(poses_refined)

    # 7. Export debugging artifacts
    export_debug_videos(...)
```

Outer iterations can start at `2` or `3`.

---

## 24) `dynamic_recon/pipelines/export_results.py`

Export:
- dynamic probability video
- binary dynamic mask video
- static probability video
- SAM3 instance overlay video
- static track visualizations
- dynamic support-point track visualizations
- point cloud colored by static/dynamic probability if a simple point cloud export is needed

---

## 25) CLI entrypoints

Implement these CLIs:

### `python -m dynamic_recon.cli.run_da3`
Arguments:
- `--video`
- `--outdir`
- `--config`
- `--device`
- `--max-frames`
- `--stride`

### `python -m dynamic_recon.cli.run_sam3`
Arguments:
- `--video`
- `--seeds-dir`
- `--outdir`
- `--config`

### `python -m dynamic_recon.cli.run_pipeline`
Arguments:
- `--video`
- `--outdir`
- `--config`
- `--device`
- `--num-outer-iters`
- `--skip-da3`
- `--skip-sam3`
- `--resume`

### `python -m dynamic_recon.cli.export_tracks`
Arguments:
- `--run-dir`
- `--query-frame`
- `--query-grid-step`
- `--mode static|dynamic|both`

---

## Config design

Use YAML config files plus Python dataclass loading if desired.

`configs/base.yaml` should define at least:

```yaml
device: cuda
dtype: float32
seed: 0

video:
  resize_long_edge: null
  max_frames: null
  stride: 1

da3:
  enabled: true
  checkpoint: null
  batch_size: 1
  window_size: 16
  stride: 8
  mixed_precision: true

geometry:
  pair_offsets: [-2, -1, 1, 2]
  alpha_3d: 1.0
  alpha_depth: 0.5
  alpha_rgb: 0.2
  alpha_cycle: 0.5
  alpha_vis: 1.0
  seed_threshold_high: 0.8
  static_threshold_low: 0.2
  occlusion_abs_tol: 0.05
  occlusion_rel_tol: 0.1

sam3:
  enabled: true
  checkpoint: null
  max_regions_per_frame: 8
  points_per_region: 3
  negative_ring: 5

fusion:
  enabled: true
  base_channels: 32
  lr: 1.0e-3
  batch_size: 2
  epochs_per_outer_iter: 3
  w_sam: 1.0
  w_geo: 1.0
  w_static_seed: 0.5
  w_dynamic_seed: 0.5
  w_tv: 0.05
  w_temp: 0.1

pose_refine:
  enabled: true
  lr: 1.0e-4
  steps: 100
  lambda_depth: 0.5
  lambda_rgb: 0.1
  keyframe_only: true
  keyframe_stride: 4

pipeline:
  num_outer_iters: 2
  save_debug_every_iter: true
```

---

## Expected tensor conventions

Codex should keep tensor conventions consistent:

- RGB images:
  - torch: `[3, H, W]`
  - numpy: `[H, W, 3]`
- depth: `[H, W]`
- masks/logits: `[1, H, W]` or `[H, W]`
- multi-instance SAM3 logits: `[N_inst, H, W]`
- intrinsics: `[3, 3]`
- extrinsics: `[4, 4]`
- world points dense: `[H, W, 3]`

Document these conventions in module docstrings.

---

## Practical heuristics and safeguards

### 1. Use conservative moving seeds
Only seed SAM3 from the strongest geometric movers at first.
False positives early can poison everything downstream.

### 2. Prefer soft masks
Store probabilities/logits whenever possible.
Do not throw away uncertainty too early.

### 3. Handle occlusions explicitly
Many "dynamic" detections are just occlusion/disocclusion.
Visibility gating is mandatory.

### 4. Static-only pose refinement
Never let high-confidence dynamic regions dominate pose updates.

### 5. Start with low frame count
For the first implementation, test on:
- 30 to 100 frames
- modest resolution
- one clear moving object

### 6. Cache everything
DA3 and SAM3 are expensive.
Every stage should be restartable from cache.

### 7. Avoid full end-to-end optimization initially
Train only the small fusion head first.
Alternate with pose refinement.

---

## Metrics to compute

Implement lightweight debugging metrics even without full labels:

### Geometry diagnostics
- mean residual on confident static pixels
- mean residual on confident dynamic pixels
- fraction of pixels marked valid by visibility

### Consistency diagnostics
- mean agreement between fusion mask and SAM3
- temporal label consistency along static reprojections
- temporal label consistency within SAM3 instance tracks

### Track diagnostics
- static track forward-backward reprojection error
- dynamic support-point survival length
- average instance IoU over time

These are enough to detect whether the pipeline is stabilizing.

---

## Visualizations Codex should export

For every run, export videos/images for:
- DA3 depth
- residual_3d
- residual_depth
- dynamic prior
- SAM3 instance masks
- fused dynamic probability
- fused binary dynamic mask
- static-only overlay
- static track overlay
- dynamic support-point track overlay

These overlays are essential for debugging.

---

## Tests Codex should add

At minimum:

### `tests/test_projection.py`
- backproject then project returns original pixels for simple synthetic setup

### `tests/test_visibility.py`
- points outside image are marked invalid
- negative camera depth invalid
- occlusion logic behaves on synthetic depth maps

### `tests/test_dynamic_prior.py`
- larger residuals produce larger dynamic scores
- high visibility reduces false dynamics when residuals are small

### `tests/test_support_points.py`
- support points stay inside mask on sampling
- correct number of points returned

### `tests/test_fusion_shapes.py`
- fusion model accepts expected input shapes
- output shape matches `[B,1,H,W]`

Use tiny synthetic tensors so tests run fast.

---

## Suggested implementation order

Codex should implement in this exact order:

### Phase 0: repo plumbing
- config loader
- cache utils
- video I/O
- CLI skeletons

### Phase 1: geometry-only baseline
- DA3 wrapper
- projection/backprojection
- residual maps
- dynamic prior
- debug export

### Phase 2: SAM3 integration
- prompt generation
- SAM3 wrapper
- propagation
- instance export

### Phase 3: fusion
- fused features
- small fusion network
- weak/self-supervised losses
- trainer
- debug metrics

### Phase 4: pose refinement
- static-only pose updates
- rerender residuals
- alternating outer loop

### Phase 5: tracking exports
- static reprojection tracks
- dynamic region tracks
- support-point tracks

Do not jump straight to the full loop.

---

## Example outer-loop behavior

### Iteration 0
- DA3 fixed
- residuals from initial poses/depth
- seed SAM3
- train fusion head

### Iteration 1
- refine poses with static pixels from fusion
- recompute residuals
- rerun SAM3 using improved seeds / optionally reuse previous masks
- retrain fusion head

### Iteration 2
- export best masks and tracks
- stop unless metrics clearly keep improving

---

## Pseudocode for the full prototype

```python
def run_preliminary_pipeline(video_path: str, cfg):
    frames, meta = load_video(video_path, cfg.video)

    da3_seq = run_or_load_da3(frames, cfg.da3)

    poses = [f.extrinsics.clone() for f in da3_seq.frames]

    fusion_model = None
    dyn_prob = None

    for outer_iter in range(cfg.pipeline.num_outer_iters):
        residual_bundle = compute_sequence_residuals(
            frames=frames,
            da3_seq=da3_seq,
            poses=poses,
            cfg=cfg.geometry,
        )

        dyn_prior = build_sequence_dynamic_prior(
            residual_bundle,
            cfg.geometry,
        )

        prompts = build_prompts_from_sequence_dynamic_prior(
            dyn_prior,
            frames,
            cfg.sam3,
        )

        sam3_out = run_or_load_sam3(
            frames=frames,
            prompts=prompts,
            cfg=cfg.sam3,
            run_tag=f"iter_{outer_iter}",
        )

        fusion_dataset = build_fusion_dataset(
            frames=frames,
            da3_seq=da3_seq,
            residual_bundle=residual_bundle,
            sam3_out=sam3_out,
            dyn_prior=dyn_prior,
            cfg=cfg.fusion,
        )

        fusion_model = train_or_update_fusion_model(
            model=fusion_model,
            dataset=fusion_dataset,
            cfg=cfg.fusion,
        )

        dyn_prob = infer_sequence_dynamic_probs(
            model=fusion_model,
            dataset=fusion_dataset,
            cfg=cfg.fusion,
        )

        if cfg.pose_refine.enabled:
            poses = refine_poses_static_only(
                da3_seq=da3_seq,
                poses=poses,
                frames=frames,
                dyn_prob=dyn_prob,
                cfg=cfg.pose_refine,
            )

        export_iteration_debug(
            frames=frames,
            da3_seq=da3_seq,
            residual_bundle=residual_bundle,
            dyn_prior=dyn_prior,
            sam3_out=sam3_out,
            dyn_prob=dyn_prob,
            poses=poses,
            iter_idx=outer_iter,
            cfg=cfg,
        )

    static_tracks = export_static_tracks(frames, da3_seq, poses, dyn_prob, cfg)
    dynamic_tracks = export_dynamic_tracks(frames, sam3_out, dyn_prob, cfg)

    return {
        "da3_seq": da3_seq,
        "poses": poses,
        "dynamic_prob": dyn_prob,
        "sam3_out": sam3_out,
        "static_tracks": static_tracks,
        "dynamic_tracks": dynamic_tracks,
    }
```

---

## Minimal first-run command sequence

Codex should make the following workflow possible:

```bash
conda activate da3-sam3-dynamic-recon

python -m dynamic_recon.cli.run_pipeline \
  --video /path/to/input.mp4 \
  --outdir outputs/demo_run \
  --config configs/quickstart.yaml
```

And optionally step-by-step:

```bash
python -m dynamic_recon.cli.run_da3 \
  --video /path/to/input.mp4 \
  --outdir outputs/demo_run

python -m dynamic_recon.cli.run_sam3 \
  --video /path/to/input.mp4 \
  --seeds-dir outputs/demo_run/geometry/dynamic_prior \
  --outdir outputs/demo_run

python -m dynamic_recon.cli.export_tracks \
  --run-dir outputs/demo_run \
  --query-frame 0 \
  --mode both
```

---

## Notes about checkpoints and local editable installs

Codex should **not** assume model checkpoints already exist.

Add clear error messages:
- if DA3 checkpoint path is missing
- if SAM3 checkpoint path is missing
- if the submodule package import fails

If local editable installs are needed, surface commands like:
```bash
pip install -e ./depth_anything_3
pip install -e ./sam3
```

But do not hard-code those installs into Python code.

The implementation should:
- fail clearly,
- explain what path / checkpoint is missing,
- allow resume once the assets are present.

---

## Coding standards

- Python 3.12
- type hints throughout
- docstrings on public functions
- avoid giant scripts; keep modules small
- prefer `torch` for geometry math
- use `dataclasses` for structured outputs
- use `pathlib.Path`
- log with Python `logging`, not print everywhere
- make every pipeline stage resumable from cache

---

## Deliverables Codex should produce

1. New/updated Python modules implementing the preliminary pipeline
2. CLI entrypoints
3. YAML configs
4. Tests
5. A short `README_preliminary_dynamic_recon.md` describing:
   - how to activate the environment,
   - how to place checkpoints,
   - how to run the quickstart,
   - what outputs to expect

---

## Final implementation target

After implementation, the system should be able to take one monocular video and produce:

- cached DA3 depth + camera outputs
- geometry-derived soft dynamic prior
- SAM3 dynamic instance masks through time
- refined soft static/dynamic mask per frame
- static reprojection tracks
- dynamic region tracks plus sparse support-point tracks
- debug videos showing the convergence of geometry and segmentation cues

That is the correct preliminary foundation before introducing a richer 4D scene representation later.
