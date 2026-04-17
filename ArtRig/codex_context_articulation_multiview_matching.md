# Codex Implementation Context: Two-Part Articulated Segmentation and Joint Estimation
# from Multiview RGB-D + Feature Matching + 3D Tracks

## Purpose

Build a Python project that reconstructs a **two-part articulated foreground object**
from **multiview RGB-D sequences over time**.

We assume:
- multiple synchronized views per time step
- RGB, depth, foreground masks, and calibrated cameras are available
- the foreground contains exactly two rigid parts:
  1) a reference / still / base part
  2) a moving part
- the moving part is connected to the reference part by one joint

The pipeline must:
1. extract robust image features from masked foreground regions
2. match points across views and across time
3. lift matches to 3D using depth and camera calibration
4. assemble sparse 3D trajectories over time
5. enforce multiview consistency to improve track quality
6. segment tracked points into two rigid parts using rigidity-driven optimization
7. propagate sparse part labels to dense foreground masks
8. estimate the joint type and joint parameters from the relative motion of the moving part
9. use the already-existing external point-trajectory joint clue module as an initializer

This implementation intentionally keeps the **two-stage structure**:
- Stage 0: feature extraction + matching + 3D track construction
- Stage 1: segmentation by rigidity
- Stage 2: joint estimation from relative motion

This document replaces the old tracker-first stage with a **new feature extraction + matching stage** and adds **multiview consistency constraints** as a first-class component.

---

# 1. High-Level Method Summary

## 1.1 Main idea

We do NOT start from dense per-pixel segmentation.
We start from **reliable sparse foreground trajectories** built from multiview feature matches.

Pipeline:
1. Extract frozen visual features from each masked foreground image
2. Detect sparse foreground keypoints
3. Match them:
   - across views at the same time
   - across adjacent times in the same view
   - optionally across time and view for extra robustness
4. Lift matched pixels into 3D using depth
5. Build sparse 3D point trajectories over time
6. Enforce **multiview consistency** to keep only geometrically valid tracks
7. Fit a 2-body rigid explanation with soft assignments
8. Use those soft assignments to derive:
   - point-level static/moving labels
   - dense per-frame masks
9. Estimate the joint from the relative motion between the two parts

This means the segmentation is **explained by rigidity in 3D**, not only by appearance.

---

# 2. Non-Negotiable Constraints

1. Python only
2. Use PyTorch for optimization and geometry where practical
3. Use frozen pretrained visual features
4. Use matching backends behind wrappers/adapters
5. Keep the two-stage structure:
   - Stage 1 segmentation
   - Stage 2 joint estimation
6. Keep the external point-trajectory joint clue backend and wrap it cleanly
7. All core module interfaces must use dataclasses
8. Support noisy tracks, missing depth, and occlusions
9. Handle axis sign ambiguity in joint estimation
10. Keep the code modular, testable, and configurable
11. Put external research repositories under a top-level `submodules/` folder
12. Do not hard-code direct imports from external repos across the core codebase; always use adapters

---

# 3. Repository Layout

Use this structure:

```text
articulation_rigidity/
├── pyproject.toml
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── submodules/
│   ├── dinov2/
│   ├── omniglue/
│   ├── lightglue/
│   └── <existing_joint_clue_repo>/
├── configs/
│   ├── default.yaml
│   ├── matching.yaml
│   ├── segmentation.yaml
│   ├── joint.yaml
│   ├── pipeline.yaml
│   └── dataset/
│       ├── multiview_rgbd.yaml
│       └── synthetic.yaml
├── scripts/
│   ├── install_submodules.sh
│   ├── run_matching.py
│   ├── run_segmentation.py
│   ├── run_joint.py
│   ├── run_pipeline.py
│   ├── preprocess_sequence.py
│   └── visualize_outputs.py
├── articulation/
│   ├── __init__.py
│   ├── data/
│   │   ├── dataclasses.py
│   │   ├── dataset.py
│   │   ├── io_rgbd.py
│   │   ├── io_tracks.py
│   │   ├── io_features.py
│   │   └── io_cameras.py
│   ├── preprocess/
│   │   ├── masks.py
│   │   ├── windows.py
│   │   ├── depth.py
│   │   ├── lifting.py
│   │   ├── filtering.py
│   │   └── multiview.py
│   ├── features/
│   │   ├── dino_wrapper.py
│   │   ├── keypoints.py
│   │   ├── graph.py
│   │   ├── neighbors.py
│   │   └── initialization.py
│   ├── matching/
│   │   ├── base.py
│   │   ├── matcher_wrapper.py
│   │   ├── omniglue_adapter.py
│   │   ├── lightglue_adapter.py
│   │   ├── filters.py
│   │   ├── multiview_consistency.py
│   │   ├── track_graph.py
│   │   └── build_tracks.py
│   ├── geometry/
│   │   ├── so3.py
│   │   ├── se3.py
│   │   ├── transforms.py
│   │   ├── pca.py
│   │   ├── lines.py
│   │   ├── invariants.py
│   │   ├── robust.py
│   │   ├── kabsch.py
│   │   └── trajectories.py
│   ├── segmentation/
│   │   ├── variables.py
│   │   ├── losses.py
│   │   ├── optimizer.py
│   │   ├── propagation.py
│   │   ├── masks_from_points.py
│   │   └── trainer.py
│   ├── joint/
│   │   ├── relative_motion.py
│   │   ├── pointwise_init.py
│   │   ├── consensus.py
│   │   ├── models.py
│   │   ├── losses.py
│   │   ├── optimizer.py
│   │   ├── selection.py
│   │   └── outputs.py
│   ├── external/
│   │   ├── joint_clue_adapter.py
│   │   ├── dino_backend_adapter.py
│   │   └── matcher_backend_adapter.py
│   ├── pipeline/
│   │   ├── stage0_matching.py
│   │   ├── stage1_segmentation.py
│   │   ├── stage2_joint.py
│   │   └── full_pipeline.py
│   ├── evaluation/
│   │   ├── segmentation_metrics.py
│   │   ├── joint_metrics.py
│   │   └── diagnostics.py
│   └── utils/
│       ├── config.py
│       ├── logging.py
│       ├── random.py
│       ├── tensors.py
│       ├── viz.py
│       └── timing.py
└── tests/
    ├── test_so3_se3.py
    ├── test_matching_filters.py
    ├── test_multiview_consistency.py
    ├── test_track_building.py
    ├── test_segmentation_losses.py
    ├── test_joint_models.py
    ├── test_consensus.py
    └── test_pipeline_smoke.py
```

---

# 4. Dependency Installation and Submodule Instructions

## 4.1 Python dependencies

Use these runtime dependencies:
- python >= 3.10
- torch
- torchvision
- numpy
- scipy
- opencv-python
- omegaconf
- hydra-core
- tqdm
- matplotlib
- scikit-learn
- einops
- open3d
- trimesh
- timm
- networkx
- pytest (dev)
- ruff (dev)
- mypy (optional)

## 4.2 External research repos under `submodules/`

Put external repos under:
- `submodules/dinov2`
- `submodules/omniglue`
- `submodules/lightglue`
- `submodules/<existing_joint_clue_repo>`

## 4.3 Example git submodule commands

Codex should set up the project assuming this pattern:

```bash
git submodule add <DINO_REPO_URL> submodules/dinov2
git submodule add <OMNIGLUE_REPO_URL> submodules/omniglue
git submodule add <LIGHTGLUE_REPO_URL> submodules/lightglue
git submodule add <JOINT_CLUE_REPO_URL> submodules/joint_clue_repo
git submodule update --init --recursive
```

Do NOT scatter external repos elsewhere.
All adapters must point into `submodules/`.

## 4.4 Installation script requirements

Implement `scripts/install_submodules.sh` that does the following:

```bash
#!/usr/bin/env bash
set -euo pipefail

python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt

# Local project
python -m pip install -e .

# Editable submodules if they expose installable packages
if [ -f submodules/dinov2/setup.py ] || [ -f submodules/dinov2/pyproject.toml ]; then
  python -m pip install -e submodules/dinov2
fi
if [ -f submodules/omniglue/setup.py ] || [ -f submodules/omniglue/pyproject.toml ]; then
  python -m pip install -e submodules/omniglue
fi
if [ -f submodules/lightglue/setup.py ] || [ -f submodules/lightglue/pyproject.toml ]; then
  python -m pip install -e submodules/lightglue
fi
if [ -f submodules/joint_clue_repo/setup.py ] || [ -f submodules/joint_clue_repo/pyproject.toml ]; then
  python -m pip install -e submodules/joint_clue_repo
fi
```

## 4.5 Adapter rule

The core code must not import external submodules directly.
Instead:
- `external/dino_backend_adapter.py`
- `external/matcher_backend_adapter.py`
- `external/joint_clue_adapter.py`

must wrap all external interactions.

---

# 5. Core Data Contracts

## 5.1 Multiview RGB-D sequence

```python
@dataclass
class MultiViewRGBDSequence:
    rgb: torch.Tensor              # [T, V, 3, H, W], float32 in [0,1]
    depth: torch.Tensor            # [T, V, 1, H, W], float32 meters
    fg_mask: torch.Tensor          # [T, V, 1, H, W], bool or float {0,1}
    K: torch.Tensor                # [V,3,3] or [T,V,3,3]
    T_cw: torch.Tensor             # [T,V,4,4], world->camera
    frame_ids: list[int]
    view_ids: list[int]
    meta: dict[str, Any]
```

## 5.2 Sparse keypoint observations

```python
@dataclass
class KeypointBatch:
    xy: torch.Tensor               # [N,2]
    desc: torch.Tensor             # [N,C]
    score: torch.Tensor            # [N]
    depth: torch.Tensor            # [N]
    valid: torch.Tensor            # [N]
    t: int
    v: int
```

## 5.3 Pairwise matches

```python
@dataclass
class MatchBatch:
    idx_a: torch.Tensor            # [M]
    idx_b: torch.Tensor            # [M]
    confidence: torch.Tensor       # [M]
    pair_type: str                 # "same_time_multiview" | "cross_time_same_view" | "cross_time_multiview"
    meta: dict[str, Any]
```

## 5.4 Track batch after multiview fusion

```python
@dataclass
class TrackBatch:
    xy: torch.Tensor               # sparse mapping or padded tensor
    xyz: torch.Tensor              # [P, T, 3], fused world points
    valid: torch.Tensor            # [P, T], bool
    anchor_frame: int
    point_ids: torch.Tensor        # [P]
    feature: torch.Tensor          # [P, C], anchor-frame feature vectors
    confidence: torch.Tensor       # [P]
    obs_count: torch.Tensor        # [P, T]
    multiview_error: torch.Tensor  # [P, T]
    meta: dict[str, Any]
```

## 5.5 Feature graph

```python
@dataclass
class FeatureGraph:
    nn_idx: torch.Tensor           # [P, K]
    nn_weight: torch.Tensor        # [P, K]
```

## 5.6 Segmentation result

```python
@dataclass
class SegmentationResult:
    point_logits: torch.Tensor         # [P]
    point_probs: torch.Tensor          # [P]
    point_labels: torch.Tensor         # [P], {0,1}
    masks_per_frame: torch.Tensor      # [T, V, 2, H, W]
    transforms_part0: torch.Tensor     # [Tw-1, 4, 4] or [T, 4, 4]
    transforms_part1: torch.Tensor     # [Tw-1, 4, 4] or [T, 4, 4]
    diagnostics: dict[str, Any]
```

## 5.7 Relative motion result

```python
@dataclass
class RelativeMotionResult:
    reference_part: int
    moving_part: int
    canonical_points: torch.Tensor     # [Pm, 3]
    moving_points_rel: torch.Tensor    # [Pm, T, 3]
    valid: torch.Tensor                # [Pm, T]
    weights: torch.Tensor              # [Pm]
    ref_transform_inv: torch.Tensor    # [T, 4, 4]
    diagnostics: dict[str, Any]
```

## 5.8 Joint results

```python
@dataclass
class JointCandidateResult:
    model_name: str
    loss: float
    axis_dir: torch.Tensor
    axis_point: Optional[torch.Tensor]
    pitch: Optional[torch.Tensor]
    state: torch.Tensor
    pred_points: torch.Tensor
    diagnostics: dict[str, Any]

@dataclass
class JointResult:
    best_model: str
    axis_dir: torch.Tensor
    axis_point: Optional[torch.Tensor]
    pitch: Optional[torch.Tensor]
    state: torch.Tensor
    candidates: list[JointCandidateResult]
    diagnostics: dict[str, Any]
```

---

# 6. Stage 0 — New Feature Extraction + Matching + Multiview Consistency

## 6.1 Goal of Stage 0

Construct a reliable sparse `TrackBatch` from multiview RGB-D data.

This stage must produce:
- sparse foreground point identities
- per-point anchor features
- 3D world trajectories over time
- validity masks
- multiview consistency scores

The rest of the pipeline depends on this stage being robust.

## 6.2 New feature extraction module

This replaces the old tracker-first entry point.

Implement `features/dino_wrapper.py` and `features/keypoints.py`.

For each `(t, v)`:
1. read RGB image and foreground mask
2. erode the mask slightly to avoid depth edges
3. extract dense frozen DINO-family features over the image
4. detect sparse keypoints inside the foreground mask
5. sample dense descriptors at keypoint locations
6. attach depth, score, and metadata

Required interface:

```python
class DinoFeatureExtractor:
    def __init__(self, model_name: str, device: str):
        ...

    def extract_dense(self, image: torch.Tensor) -> torch.Tensor:
        # returns [C,h,w]
        ...

    def sample_points(self, feat_map: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        # returns [P,C]
        ...
```

Keypoint detector must be behind an adapter.
Could be:
- SuperPoint
- SIFT
- external detector backend

Core code must not depend on one specific detector.

## 6.3 Matching stage

Implement `matching/base.py`, `matching/matcher_wrapper.py`, `matching/omniglue_adapter.py`, and `matching/lightglue_adapter.py`.

Required matcher interface:

```python
class BaseMatcher(Protocol):
    def match(
        self,
        frame_a: KeypointBatch,
        frame_b: KeypointBatch
    ) -> MatchBatch:
        ...
```

Support three pair types:
1. `same_time_multiview`: `(t, v1) <-> (t, v2)`
2. `cross_time_same_view`: `(t, v) <-> (t+1, v)`
3. `cross_time_multiview`: optional `(t, v1) <-> (t+1, v2)` for stronger linking

Default matching strategy:
- use `same_time_multiview` for strong 3D identity building
- use `cross_time_same_view` for temporal track extension
- enable `cross_time_multiview` only as an optional robustness pass

## 6.4 Geometric filtering of matches

Implement `matching/filters.py`.

### Same-time multiview filtering
For a match between views at the same time:
- both points must be inside the foreground mask
- both must have valid depth
- lift both to world coordinates
- world 3D distance must be below threshold

If `X_a` and `X_b` are lifted world points at same time, keep the match only if:

```text
||X_a - X_b|| < threshold_same_time
```

### Cross-time filtering
For adjacent-time same-view or cross-view temporal matches:
- require valid depth where available
- require confidence above threshold
- optionally require cycle consistency
- reject absurdly large jumps
- reject motion that is too inconsistent with neighboring matched points

Do NOT require same 3D position across time, because articulation is allowed.

## 6.5 3D lifting

Implement `preprocess/lifting.py`.

Use camera intrinsics and extrinsics to lift pixels into world coordinates.

For pixel `(u, v)` with depth `z`:

```text
x_cam = (u - cx) / fx * z
y_cam = (v - cy) / fy * z
z_cam = z
X_world = T_wc @ [x_cam, y_cam, z_cam, 1]^T
```

Required function:

```python
def lift_pixel_to_world(
    xy: torch.Tensor,
    depth: torch.Tensor,
    K: torch.Tensor,
    T_cw: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```

## 6.6 Multiview consistency (NEW REQUIRED COMPONENT)

Implement `matching/multiview_consistency.py`.

This is required to improve track quality and reduce false correspondences.

### Principle
A good track must be consistent across all available views at the same time.

For a track `p` at time `t`, if it is observed in multiple views:
1. lift all observations to world coordinates
2. these lifted points should agree closely
3. their reprojections into the other views should agree with observed pixels
4. the descriptor support across views should be mutually coherent

### Required multiview consistency checks

#### A. World-point agreement
For all valid same-time observations of the same track:

```text
L_mv_world = mean_{a,b} ||X_{t,a} - X_{t,b}||
```

This should be small.

#### B. Cross-view reprojection consistency
For each same-time multiview observation, fuse a world point `X̄_t` and reproject into all supporting views:

```text
u_hat_{t,v} = project(K_v, T_cw[t,v], X̄_t)
```

Penalize reprojection error to the observed keypoint location:

```text
L_mv_reproj = mean_v ||u_hat_{t,v} - u_obs_{t,v}||
```

#### C. Descriptor coherence across views
For same-time multiview grouped observations, feature similarity should remain high:

```text
L_mv_feat = mean_{a,b} (1 - cosine(desc_a, desc_b))
```

This is a soft check, not a hard constraint.

### Required implementation behavior

- keep a per-track per-time multiview support set
- compute `multiview_error[p,t]`
- reject or downweight track states with large multiview inconsistency
- use multiview consistency both:
  - during track building
  - during later track filtering

### Suggested aggregate multiview score

```text
mv_score[p,t] =
    alpha_world  * L_mv_world
  + alpha_reproj * L_mv_reproj
  + alpha_feat   * L_mv_feat
```

Lower is better.

Track filtering can use:
- mean multiview score across valid times
- fraction of times with acceptable multiview support

## 6.7 Track graph construction

Implement `matching/track_graph.py` and `matching/build_tracks.py`.

Graph definition:
- node = one keypoint observation `(t, v, k)`
- edge = one accepted match

Track building strategy:
1. use same-time multiview edges to group observations of the same physical point
2. extend groups through adjacent-time matches
3. optionally use cross-time multiview edges to bridge missing links
4. prune inconsistent components
5. for each track and time, fuse multiple same-time observations into one world point
6. compute and store multiview consistency diagnostics

Fusion rule:
- robust average or median of all valid same-time same-track world points

Output:
- `xyz[P,T,3]`
- `valid[P,T]`
- `multiview_error[P,T]`

## 6.8 Track filtering

Implement:
- minimum valid ratio
- minimum temporal length
- outlier rejection on velocity/acceleration
- maximum mean multiview inconsistency
- minimum multiview support fraction
- optional confidence threshold

These filters are critical because all rigidity losses are sensitive to bad trajectories.

---

# 7. Stage 1 — Segmentation by Rigidity

## 7.1 Goal

Given sparse 3D trajectories `xyz[P,T,3]`, estimate a soft assignment `w_p` of each point to:
- part 0 = reference/still part
- part 1 = moving part

This stage must remain a **two-part soft assignment optimization**.

## 7.2 Variables to optimize

Implement `segmentation/variables.py`.

```python
class SegmentationVariables(nn.Module):
    def __init__(self, num_points: int, num_steps: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_points))
        self.xi_part0 = nn.Parameter(torch.zeros(num_steps, 6))
        self.xi_part1 = nn.Parameter(torch.zeros(num_steps, 6))

    def probs(self) -> torch.Tensor:
        return torch.sigmoid(self.logits)
```

Where:
- `logits[p]` => soft membership score of point `p`
- `w[p] = sigmoid(logits[p])` => probability point belongs to part 1
- `xi_part*` => SE(3) twists from anchor to each frame

## 7.3 SE(3) parameterization

Implement in `geometry/se3.py`.

Required:
- `so3_exp`
- `se3_exp`
- `transform_points`
- batching support

Use exponential maps for stability.

## 7.4 Motion fit loss

Implement `motion_fit_loss()` in `segmentation/losses.py`.

For each point `p`, anchor frame `0`, and time `t > 0`:

```text
res0 = || T0_t(X0_p) - Xt_p ||
res1 = || T1_t(X0_p) - Xt_p ||
norm = ||X0_p - Xt_p|| + eps
```

```text
L_motion =
    sum_(p,t valid)
      (1 - w_p) * huber(res0 / norm)
    + w_p       * huber(res1 / norm)
```

## 7.5 Feature smoothness loss

Implement `feature_smoothness_loss()` on the anchor-frame feature graph.

```text
L_smooth = sum_p sum_q alpha_pq * abs(w_p - w_q)
```

## 7.6 Canonical-frame rigidity consistency loss

This is the key geometric assignment loss.

If a point belongs to part 0, then after mapping its observed trajectory into the canonical frame of part 0, it should stay almost fixed over time.

If it belongs to part 1, the same must hold in part 1 canonical coordinates.

For point `p`:

```text
Xcan0_t,p = inverse(T0_t) * X_t,p
Xcan1_t,p = inverse(T1_t) * X_t,p
Var0_p = Var_t(Xcan0_t,p)
Var1_p = Var_t(Xcan1_t,p)
```

```text
L_canrigid =
    sum_p
      (1 - w_p) * Var0_p
    + w_p       * Var1_p
```

Interpretation:
The point should be assigned to the part whose canonical frame makes it most stationary over time.

## 7.7 Entropy loss

Implement binary entropy:

```text
L_ent = -sum_p [w_p log w_p + (1-w_p) log(1-w_p)]
```

Policy:
- keep only a very small entropy weight for warmup
- default behavior should rely primarily on geometric losses
- optionally anneal entropy to zero

## 7.8 Pairwise relative-distance rigidity loss

Implement `pairwise_rigidity_loss()`.

For pair `(i,j)` and valid time `t`:

```text
d_t = ||X_t_i - X_t_j||
d_0 = ||X_0_i - X_0_j||
delta_t = |d_t - d_0|
s_ij = w_i*w_j + (1-w_i)*(1-w_j)
d_ij = 1 - s_ij
```

```text
L_pair =
    sum_(i,j)
      s_ij * mean_t huber(delta_t)
    + lambda_sep * d_ij * mean_t relu(margin - delta_t)
```

## 7.9 Center-of-geometry rigidity loss

Implement `cog_rigidity_loss()`.

```text
c0_t = sum_p (1-w_p) * X_t_p / sum_p (1-w_p)
c1_t = sum_p w_p * X_t_p / sum_p w_p
r0_t,p = ||X_t,p - c0_t||
r1_t,p = ||X_t,p - c1_t||
```

```text
L_cog =
    sum_(p,t valid)
      (1 - w_p) * huber( |r0_t,p - r0_0,p| / (r0_0,p + eps) )
    + w_p       * huber( |r1_t,p - r1_0,p| / (r1_0,p + eps) )
```

## 7.10 Optional anti-collapse balance loss

```text
L_balance = (mean(w) - 0.5)^2
```

Use only if collapse happens.

## 7.11 Final segmentation loss

```text
L_seg =
    lambda_motion   * L_motion
  + lambda_smooth   * L_smooth
  + lambda_canrigid * L_canrigid
  + lambda_pair     * L_pair
  + lambda_cog      * L_cog
  + lambda_ent      * L_ent
  + lambda_balance  * L_balance
```

## 7.12 Initialization

Use DINO-like anchor-frame features to initialize the two soft clusters.

Recommended:
- KMeans with 2 clusters
- convert cluster assignment to initial logits `{-2, +2}`

DINO features are used for:
- initialization
- neighbor graph
- pair sampling prior
- sparse-to-dense propagation

## 7.13 Optimization schedule

Recommended:
1. warmup:
   - `L_motion + L_smooth + tiny L_ent`
2. add `L_canrigid`
3. add `L_pair`
4. add `L_cog`
5. if collapse detected, optionally activate `L_balance`
6. anneal entropy down

## 7.14 Sliding windows

Implement per-window segmentation on temporal windows.

## 7.15 Propagation to dense masks

Process:
1. rasterize sparse point labels back into each frame/view
2. restrict to the foreground mask
3. propagate sparse labels inside the foreground using:
   - image-space nearest neighbors
   - optionally DINO feature similarity
4. produce dense two-channel masks:
   - part 0 mask
   - part 1 mask

---

# 8. Stage 2 — Joint Estimation

## 8.1 Goal

Given the segmented tracks:
1. choose reference part and moving part
2. express moving-part trajectories in the reference-part frame
3. use the external pointwise joint clue module for initialization
4. build consensus on joint type and axis direction
5. fit candidate kinematic models globally on all moving-part points
6. choose the best model

Candidate joint types:
- revolute
- prismatic
- screw

## 8.2 Reference part selection

Choose the part with lower average motion magnitude as reference.

## 8.3 Relative motion computation

Use the segmented reference-part points to estimate per-frame rigid transforms of the reference part.
Then transform moving-part points into the reference-part canonical frame.
Do NOT just subtract centroids.

## 8.4 External pointwise joint clue adapter

Keep the existing module.
Wrap it cleanly in `external/joint_clue_adapter.py`.

Required interface:

```python
class JointClueEstimator:
    def infer(self, xyz_traj: np.ndarray) -> dict:
        ...
```

## 8.5 Pointwise initialization and consensus

Process:
1. sample moving-part points
2. call the external clue backend on each trajectory
3. aggregate type scores
4. aggregate axis directions with sign alignment
5. estimate initial axis point / pitch when available

## 8.6 Joint models

Implement in `joint/models.py`:
- RevoluteModel
- PrismaticModel
- ScrewModel

All differentiable in PyTorch.

## 8.7 Joint losses

Implement in `joint/losses.py`:
- trajectory reconstruction
- temporal smoothness
- axis prior regularization
- optional axis point regularization
- screw pitch regularization

## 8.8 Model selection

Fit all candidate models:
- revolute
- prismatic
- screw

Choose best by:

```text
score = optimized_loss + complexity_penalty[model]
```

---

# 9. Matching + Multiview Consistency + Segmentation + Joint Estimation Coupling

## 9.1 Core principle

Matching builds the trajectories.
Multiview consistency validates the trajectories.
Segmentation explains the trajectories by two rigid motions.
Joint estimation explains the moving part relative to the reference part.

So the order is:

multiview RGB-D
-> sparse features
-> sparse matches
-> multiview-consistent 3D lifted trajectories
-> rigidity-based two-part segmentation
-> reference-frame relative motion
-> joint estimation

This ordering must remain explicit in the codebase.

---

# 10. CLI Requirements

## run_matching.py
Inputs:
- sequence path
- matcher backend config
- feature config

Outputs:
- sparse tracks
- multiview consistency diagnostics
- visualizations

## run_segmentation.py
Inputs:
- track source
- segmentation config

Outputs:
- point labels
- dense masks
- diagnostics

## run_joint.py
Inputs:
- segmentation result
- joint config
- external clue backend config

Outputs:
- joint type
- axis direction
- axis point
- pitch if screw
- state over time
- visualizations

## run_pipeline.py
Runs all stages end-to-end.

---

# 11. Visualization Requirements

Must implement:
1. sparse matches across views and time
2. multiview consistency heatmaps per track and time
3. 3D trajectories colored by soft label
4. dense masks per frame/view
5. canonical-frame trajectory variance plots
6. part centroids over time
7. moving-part trajectories in reference frame
8. 3D joint axis visualization
9. per-model fit comparison
10. state-vs-time plots

Use matplotlib and open3d.

---

# 12. Unit Tests Required

## test_so3_se3.py
- zero twist -> identity
- transform correctness
- batch behavior

## test_matching_filters.py
- same-time multiview matches fail when 3D disagreement too large
- valid matches survive

## test_multiview_consistency.py
- world agreement score low for correct multiview observations
- reprojection consistency low for coherent fused world points
- inconsistent cross-view observations get high error

## test_track_building.py
- graph assembly creates correct sparse trajectories on toy data

## test_segmentation_losses.py
Synthetic two-part articulated trajectories:
- correct grouping gives low motion fit
- correct grouping gives low canonical rigidity
- correct grouping gives low pairwise rigidity
- correct grouping gives low CoG rigidity
- entropy remains numerically stable

## test_joint_models.py
Synthetic revolute/prismatic/screw trajectories:
- correct type is selected
- axis direction recovered approximately
- state recovered approximately

## test_consensus.py
- sign ambiguity handled correctly
- weighted axis aggregation stable

## test_pipeline_smoke.py
Small synthetic end-to-end run.

---

# 13. Config Recommendations

## matching.yaml
```yaml
matcher:
  backend: "omniglue"   # or "lightglue"
features:
  model_name: "vit_small_patch14_dinov2"
  num_keypoints: 1024
  fg_erode_px: 3

filtering:
  same_time_max_3d_error: 0.01
  min_match_confidence: 0.2
  min_track_length: 4
  min_valid_ratio: 0.5

multiview:
  alpha_world: 1.0
  alpha_reproj: 0.5
  alpha_feat: 0.2
  max_mean_score: 0.02
  min_multiview_support_ratio: 0.4
  enable_cross_time_multiview: false
```

## segmentation.yaml
```yaml
window:
  size: 8
  stride: 4

features:
  num_neighbors: 16
  spatial_gate_px: 48

loss:
  lambda_motion: 200.0
  lambda_smooth: 10.0
  lambda_canrigid: 5.0
  lambda_pair: 2.0
  lambda_cog: 1.0
  lambda_ent: 0.001
  lambda_balance: 0.0
  pair_margin: 0.01
  pair_lambda_sep: 1.0

optimizer:
  lr_logits: 1e-2
  lr_twists: 1e-4
  iterations: 300
  grad_clip: 1.0
  schedule:
    warmup_iters: 40
    canrigid_start_iter: 40
    pair_start_iter: 80
    cog_start_iter: 120
    entropy_decay_start: 40
    entropy_decay_end: 160

sampling:
  num_pairs: 4096
```

## joint.yaml
```yaml
sampling:
  num_point_samples: 256
  strategy: "fps"

models:
  candidates: ["revolute", "prismatic", "screw"]
  screw_complexity_penalty: 0.05

loss:
  lambda_fit: 1.0
  lambda_temporal: 0.1
  lambda_axis: 0.05
  lambda_axis_point: 0.0
  lambda_pitch: 0.01

optimizer:
  lr_axis: 1e-2
  lr_axis_point: 5e-3
  lr_state: 1e-2
  lr_pitch: 1e-3
  iterations: 500
```

---

# 14. Failure Modes and Mitigations

## Segmentation collapse
Mitigation:
- feature-based initialization
- motion + smoothness warmup
- tiny entropy only at start
- optional weak balance term

## Bad matches corrupt tracks
Mitigation:
- strong same-time multiview filtering
- confidence thresholds
- cycle checks when available
- descriptor coherence checks

## Bad tracks corrupt rigidity losses
Mitigation:
- min valid ratio
- multiview support thresholds
- robust penalties
- pair subsampling
- overlap threshold

## Canonical rigidity unstable with too few valid frames
Mitigation:
- masked variance
- minimum valid frames per point
- ignore low-support points in `L_canrigid`

## CoG instability
Mitigation:
- clamp centroid denominators
- activate CoG after warmup
- optional balance prior

## Wrong joint due to screw overfitting
Mitigation:
- complexity penalty
- pitch regularization
- axis prior from pointwise clues

---

# 15. Final Instruction to Codex

Implement this as a research-grade modular project.

Most important requirements:
- Stage 0 must build sparse 3D trajectories from multiview RGB-D using frozen features + matcher + depth lifting
- the old tracker-first section is replaced by this new feature extraction + matching stage
- external research repos must live under `submodules/`
- include installation instructions and an install script for those submodules
- add multiview consistency checks and use them both during track building and later filtering
- Stage 1 must segment points into two parts using rigidity-driven losses
- the key assignment logic must come from geometric rigidity, especially the canonical-frame rigidity loss
- Stage 2 must estimate the joint from moving-part trajectories in the reference-part frame
- do not replace the external pointwise joint clue backend; wrap it and use it as initialization
- DINO-like features are used for initialization, smoothing, propagation, and pair sampling
- the final dense segmentation must be derived from sparse rigid tracks plus feature-based propagation inside the foreground mask
