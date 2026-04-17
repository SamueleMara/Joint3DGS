# Codex Implementation Context: Two-Stage Articulated Object Segmentation and Joint Estimation from RGB-D + Tracks

## Purpose

This document is a **deep implementation context file** for Codex to build a Python project that reconstructs a **two-part articulated object** from a monocular RGB-D video under **free-moving capture**, using a **two-stage pipeline**:

1. **Part segmentation by motion rigidity**
2. **Joint estimation by relative motion of the moving part w.r.t. the reference part**

This design intentionally preserves:
- the **two-step procedure**
- **feature-based initialization** from a frozen visual model (DINO-family features or equivalent)
- a **reference-vs-moving part** formulation in joint estimation
- usage of an **existing external point-trajectory joint clue module** that, given a point trajectory over time, returns a **joint clue** and an **axis versor**

This design intentionally replaces the original FreeArtGS `L_init` term with **physics-based rigidity losses** based on:
- **pairwise relative distance constancy**
- **distance-to-center-of-geometry constancy**

and upgrades joint estimation from a lightweight closed-form stage into a more robust **global optimization over all moving-part points**, while still using the external point-wise joint-clue module for initialization.

---

# 1. Research/Method Summary

## 1.1 High-level goal

Given:
- RGB frames `I_t`
- depth maps `D_t`
- foreground masks `M_t`
- camera intrinsics `K`
- dense or semi-dense tracked points across time

we want to estimate:

1. a decomposition of tracked object points into **two rigid parts**
2. the **reference part** and **moving part**
3. the relative motion of the moving part in the reference-part frame
4. the **joint type** among:
   - revolute
   - prismatic
   - screw
5. the **joint axis direction**
6. the **joint axis location / anchor**
7. the per-frame **joint state**
   - angle for revolute
   - translation for prismatic
   - angle (+ pitch) for screw

---

## 1.2 Key design decisions

### Keep
- feature initialization from DINO-like embeddings
- motion-fitting over short windows
- soft two-part assignments
- a second stage using relative motion between two candidate parts
- one part chosen as reference / approximately fixed
- the already-working external point-trajectory joint-clue module

### Replace
Original `L_init` BCE-to-feature-cluster prior is replaced or strongly augmented by:
- **pairwise rigidity loss**
- **center-of-geometry rigidity loss**

### Add
Global joint optimization over the moving-part point cloud trajectories using candidate kinematic models:
- revolute
- prismatic
- screw

---

# 2. Non-Negotiable Constraints for Codex

Codex must implement under these constraints:

1. **Python only**
2. Prefer **PyTorch** for differentiable optimization
3. Use **frozen pretrained visual features**
4. The segmentation stage must remain a **two-part soft assignment optimization**
5. Joint estimation must use the **existing external module** as an initializer / clue provider, not be rewritten
6. The code must be:
   - modular
   - testable
   - reproducible
   - configurable
7. Provide a clean command-line interface for:
   - segmentation only
   - joint estimation only
   - full pipeline
8. Code should avoid hard dependencies on a specific tracker implementation; tracking must be behind a wrapper
9. Axis direction must be treated with **sign ambiguity awareness** (`u` and `-u` equivalent where appropriate)
10. Optimization must be robust to:
   - noisy tracks
   - missing depth
   - outlier frames
   - short motion intervals with limited articulation

---

# 3. Repository Architecture to Implement

Use this exact or very close structure:

```text
articulation_rigidity/
├── pyproject.toml
├── README.md
├── configs/
│   ├── default.yaml
│   ├── segmentation.yaml
│   ├── joint.yaml
│   ├── pipeline.yaml
│   └── dataset/
│       ├── real_rgbd.yaml
│       └── synthetic.yaml
├── scripts/
│   ├── run_pipeline.py
│   ├── run_segmentation.py
│   ├── run_joint.py
│   ├── preprocess_sequence.py
│   └── visualize_outputs.py
├── articulation/
│   ├── __init__.py
│   ├── data/
│   │   ├── dataclasses.py
│   │   ├── dataset.py
│   │   ├── io_rgbd.py
│   │   ├── io_tracks.py
│   │   └── io_features.py
│   ├── preprocess/
│   │   ├── masks.py
│   │   ├── windows.py
│   │   ├── depth.py
│   │   ├── lifting.py
│   │   └── filtering.py
│   ├── tracking/
│   │   ├── base.py
│   │   ├── tracker_wrapper.py
│   │   └── track_filters.py
│   ├── features/
│   │   ├── dino_wrapper.py
│   │   ├── graph.py
│   │   ├── neighbors.py
│   │   └── initialization.py
│   ├── geometry/
│   │   ├── so3.py
│   │   ├── se3.py
│   │   ├── transforms.py
│   │   ├── pca.py
│   │   ├── lines.py
│   │   ├── invariants.py
│   │   └── robust.py
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
│   │   └── tracker_backend_adapter.py
│   ├── pipeline/
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
    ├── test_segmentation_losses.py
    ├── test_joint_models.py
    ├── test_consensus.py
    └── test_pipeline_smoke.py
```

---

# 4. Dependencies

## 4.1 Required

Use these runtime dependencies:

- `python >= 3.10`
- `torch`
- `torchvision`
- `numpy`
- `scipy`
- `opencv-python`
- `omegaconf`
- `hydra-core`
- `tqdm`
- `matplotlib`
- `scikit-learn`
- `einops`
- `open3d`
- `trimesh`
- `timm`

## 4.2 Optional but recommended

- `faiss-cpu` or `faiss-gpu` for feature nearest neighbors
- `pytransform3d` for debugging transforms and screws
- `pytest`
- `ruff`
- `mypy`

## 4.3 External backends not hard-coded into core

Do not hard-code direct imports from research repos across the codebase. Wrap them via adapters.

Backends:
- tracker backend (AllTracker or equivalent)
- DINO backend (via `timm` or external repo)
- external pointwise joint clue backend

---

# 5. Data Contracts

Codex must implement strict dataclasses for exchange between modules.

## 5.1 RGB-D sequence

```python
@dataclass
class RGBDSequence:
    rgb: torch.Tensor          # [T, 3, H, W], float32 in [0,1]
    depth: torch.Tensor        # [T, 1, H, W], float32 meters
    fg_mask: torch.Tensor      # [T, 1, H, W], bool or float {0,1}
    K: torch.Tensor            # [3,3] or [T,3,3]
    frame_ids: list[int]
    meta: dict[str, Any]
```

## 5.2 Track batch

```python
@dataclass
class TrackBatch:
    xy: torch.Tensor           # [P, T, 2], pixel coordinates
    xyz: torch.Tensor          # [P, T, 3], lifted points
    valid: torch.Tensor        # [P, T], bool
    anchor_frame: int
    point_ids: torch.Tensor    # [P]
    feature: torch.Tensor      # [P, C], anchor-frame feature vectors
    confidence: torch.Tensor   # [P], optional tracker confidence
```

## 5.3 Feature graph

```python
@dataclass
class FeatureGraph:
    nn_idx: torch.Tensor       # [P, K]
    nn_weight: torch.Tensor    # [P, K]
```

## 5.4 Segmentation output

```python
@dataclass
class SegmentationResult:
    point_logits: torch.Tensor         # [P]
    point_probs: torch.Tensor          # [P]
    point_labels: torch.Tensor         # [P], {0,1}
    masks_per_frame: torch.Tensor      # [T, 2, H, W]
    transforms_part0: torch.Tensor     # [Tw-1, 4, 4] or [T, 4, 4]
    transforms_part1: torch.Tensor     # [Tw-1, 4, 4] or [T, 4, 4]
    diagnostics: dict[str, Any]
```

## 5.5 Relative motion output

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

## 5.6 Joint outputs

```python
@dataclass
class JointCandidateResult:
    model_name: str                    # "revolute" | "prismatic" | "screw"
    loss: float
    axis_dir: torch.Tensor             # [3]
    axis_point: Optional[torch.Tensor] # [3]
    pitch: Optional[torch.Tensor]      # scalar for screw
    state: torch.Tensor                # [T]
    pred_points: torch.Tensor          # [Pm, T, 3]
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

# 6. Stage 1 — Segmentation by Rigidity

## 6.1 Required conceptual behavior

We must estimate a soft assignment `w_p` of tracked points into two rigid parts. We keep:
- motion fit
- feature smoothness
- entropy term

We replace/augment the original DINO-init BCE regularization with:
- pairwise relative-distance rigidity
- center-of-geometry rigidity

DINO remains:
- an initialization source
- a neighbor-graph source

---

## 6.2 Variables to optimize

Use a `SegmentationVariables` module:

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

Notes:
- `logits` represent point membership scores
- `w = sigmoid(logits)` are probabilities of belonging to part 1
- `xi_part*` are SE(3) twists for transforms between anchor and each later frame in the window

---

## 6.3 Transform parameterization

Implement in `geometry/se3.py`:

- `so3_exp`
- `se3_exp`
- `transform_points`

Represent an SE(3) transform by a 6D twist:
- first 3 = axis-angle rotation vector
- last 3 = translation

Use exponential maps for numerical stability.

### Required functions

```python
def so3_exp(omega: torch.Tensor) -> torch.Tensor:
    ...

def se3_exp(xi: torch.Tensor) -> torch.Tensor:
    ...

def transform_points(T: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    ...
```

All must support batching.

---

## 6.4 Motion fit loss

For each point and frame in a window, points should be explained by one of two rigid transforms.

Implement `motion_fit_loss()`:

Inputs:
- `xyz`: `[P, Tw, 3]`
- `valid`: `[P, Tw]`
- `w`: `[P]`
- transforms for both parts

Behavior:
- anchor at frame 0 within each window
- for each `t > 0`, transform `X_{0,p}` by both candidate transforms
- compute residual to observed `X_{t,p}`
- use weighted soft assignment

Formula to approximate:

```python
res0 = || T0_t(X0_p) - Xt_p ||
res1 = || T1_t(X0_p) - Xt_p ||

loss = (1 - w_p) * huber(res0 / (||X0_p - Xt_p|| + eps))
     + w_p       * huber(res1 / (||X0_p - Xt_p|| + eps))
```

Use only valid point/frame pairs.

---

## 6.5 Feature smoothness loss

Implement `feature_smoothness_loss()` on the DINO neighbor graph.

For each point `p` and feature neighbors `q`:
- penalize `|w_p - w_q|` weighted by feature similarity

Formula:

```python
L_smooth = sum_p sum_q alpha_pq * abs(w_p - w_q)
```

Use precomputed neighbors from anchor-frame features.

---

## 6.6 Entropy loss

Implement binary entropy on soft assignments:

```python
L_ent = -sum_p [w_p log w_p + (1-w_p) log(1-w_p)]
```

This encourages confident assignments.

Numerical stability:
- clamp probabilities to `[1e-6, 1-1e-6]`

---

## 6.7 New pairwise relative-distance rigidity loss

This is one of the core changes.

### Principle
If points belong to the same rigid part, pairwise distance should stay constant over time.

### Inputs
- `xyz`: `[P, Tw, 3]`
- `valid`: `[P, Tw]`
- `w`: `[P]`
- sampled point pairs `(i,j)`

### Pairwise invariance
For each pair `(i,j)` and each valid frame `t`, define:

```python
d_t = ||X_t_i - X_t_j||
d_0 = ||X_0_i - X_0_j||
delta = |d_t - d_0|
```

### Soft same-part probability

```python
s_ij = w_i*w_j + (1-w_i)*(1-w_j)
d_ij = 1 - s_ij
```

### Loss
Use a contrastive margin-style form:

```python
L_pair = sum_(i,j)
    s_ij * mean_t huber(delta_t)
    + lambda_sep * d_ij * mean_t relu(margin - delta_t)
```

Interpretation:
- same-part pairs should have low distance variation
- different-part pairs are encouraged not to look rigid

### Implementation notes
- do **not** use all pairs
- sample `M` pairs per iteration
- prefer neighbors in feature or spatial space
- ignore pairs with insufficient frame overlap
- use robust penalties

---

## 6.8 New center-of-geometry rigidity loss

This is the second core change.

### Principle
For a rigid part, distance from each point to the part centroid should remain approximately constant over time.

### Soft centroids

For each frame `t`:

```python
c0_t = sum_p (1-w_p) * X_t_p / sum_p (1-w_p)
c1_t = sum_p w_p * X_t_p / sum_p w_p
```

### Radial invariance
For each point `p`:
- if it belongs to part 0, radial distance to `c0_t` should remain stable
- if it belongs to part 1, radial distance to `c1_t` should remain stable

Use anchor-frame radial distance as reference.

Loss form:

```python
r0_t_p = ||X_t_p - c0_t||
r1_t_p = ||X_t_p - c1_t||
r0_0_p = ||X_0_p - c0_0||
r1_0_p = ||X_0_p - c1_0||

L_cog =
    sum_t sum_p
      (1-w_p) * huber( |r0_t_p - r0_0_p| / (r0_0_p + eps) )
    + w_p * huber( |r1_t_p - r1_0_p| / (r1_0_p + eps) )
```

### Why normalized
Without normalization, large-radius points dominate.

### Safety
Centroids can become unstable when one cluster collapses.
Therefore optionally add a weak cluster-balance term.

---

## 6.9 Optional anti-collapse regularization

This is optional, but Codex should implement it as a switch.

```python
L_balance = (mean(w) - 0.5)^2
```

Use a **small weight only**. The purpose is only to prevent the trivial all-points-one-part collapse early in training.

---

## 6.10 Final segmentation loss

Implement a total segmentation loss:

```python
L_seg =
    lambda_motion * L_motion
  + lambda_smooth * L_smooth
  + lambda_entropy * L_ent
  + lambda_pair * L_pair
  + lambda_cog * L_cog
  + lambda_balance * L_balance   # optional
```

DINO initialization is used as:
- initial value of logits
- source of graph neighbors
- optionally source of initial pair sampling neighborhoods

Do **not** keep the old BCE-to-init loss as default.

---

## 6.11 Optimization schedule

Codex should implement a staged schedule because full loss from iteration 0 may be unstable.

Recommended schedule:
1. warmup:
   - motion + smooth + entropy
2. add pairwise rigidity
3. add CoG rigidity
4. optionally activate anti-collapse only if collapse detected

This should be configurable.

---

## 6.12 Sliding windows

Implement in `preprocess/windows.py`:

- fixed-length windows
- stride configurable
- each window anchored at first frame

`WindowSpec`:
```python
@dataclass
class WindowSpec:
    start: int
    end: int
    anchor: int
```

Run segmentation per window, then merge.

---

## 6.13 Merge/propagate window results

Implement in `segmentation/propagation.py`:

Responsibilities:
- merge logits / labels across overlapping windows
- propagate soft labels to untracked pixels using feature neighbors
- produce dense per-frame masks

Approach:
- accumulate point logits over windows
- average or confidence-weight them
- threshold for hard labels
- rasterize tracked labels back to image plane
- fill holes by nearest-neighbor in feature space or image-space postprocessing

---

# 7. Feature Extraction and Initialization

## 7.1 Feature extractor wrapper

Implement `features/dino_wrapper.py` with a frozen model interface.

Required interface:

```python
class DinoFeatureExtractor:
    def __init__(self, model_name: str, device: str):
        ...

    def extract_dense(self, image: torch.Tensor) -> torch.Tensor:
        """Returns dense feature map [C, h, w]."""

    def sample_points(self, feat_map: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        """Returns point features [P, C]."""
```

Use `timm` by default if possible.

---

## 7.2 Initialization of part probabilities

Implement `features/initialization.py`.

Goal:
- use features to obtain a reasonable initial two-cluster partition

Methods allowed:
- KMeans on point features
- spectral clustering on feature graph
- simple 2-means in feature space

Prefer:
- KMeans with 2 clusters
- convert clusters to logits, e.g. `{-2, +2}`

Optional:
- use motion magnitude as a secondary cue for cluster polarity

---

## 7.3 Neighbor graph construction

Implement `features/graph.py`.

Graph:
- `K` nearest neighbors in feature space
- optional spatial gate in anchor image coordinates
- weights by cosine similarity or Gaussian over feature distance

Output:
- `nn_idx`
- `nn_weight`

This graph is used for:
- smoothness loss
- pair sampling
- propagation/fill

---

# 8. Tracking and 3D Lifting

## 8.1 Tracker wrapper

Implement `tracking/base.py` and `tracking/tracker_wrapper.py`.

The core code must interact only through:

```python
class BaseTracker(Protocol):
    def track(self, sequence: RGBDSequence) -> TrackBatch:
        ...
```

This allows swapping:
- AllTracker
- TAPIR-like tracker
- custom tracked points

---

## 8.2 Lift 2D tracks to 3D

Implement `preprocess/lifting.py`.

Function:
```python
def lift_tracks_to_3d(
    xy: torch.Tensor, 
    depth: torch.Tensor, 
    K: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        xyz: [P,T,3]
        valid: [P,T]
    """
```

Use:
- nearest or bilinear depth sampling
- invalid if:
  - out of bounds
  - depth missing / non-positive
  - depth too large if max-depth set

Equation:
For pixel `(u,v)` and depth `z`:

```python
x = (u - cx) / fx * z
y = (v - cy) / fy * z
z = z
```

Output in camera coordinates.

---

## 8.3 Track filtering

Implement:
- min valid ratio filter
- trajectory smoothness/outlier filter
- optional tracker confidence filter

These filters are critical because rigidity losses are sensitive to noisy tracks.

---

# 9. Stage 2 — Joint Estimation and Optimization

## 9.1 Required behavior

After segmentation:
1. choose reference part and moving part
2. express moving-part points in the reference-part coordinate frame
3. use the external point-trajectory module on sampled point trajectories
4. build consensus for joint type and axis direction
5. initialize candidate joint models
6. fit all candidate models globally over all moving-part points
7. choose the best one

Candidate models:
- revolute
- prismatic
- screw

---

## 9.2 Reference part selection

Implement in `joint/relative_motion.py`.

Criterion:
- choose the part with lower average motion magnitude as reference

Possible metrics:
- average centroid displacement over time
- average rigid transform translation norm
- average point displacement magnitude

Keep this simple and configurable.

---

## 9.3 Reference-frame conversion

This step is essential.

You must compute moving-part trajectories in the **reference-part canonical frame**.

Implement:
```python
def compute_relative_motion(
    xyz_part_ref: torch.Tensor, 
    xyz_part_mov: torch.Tensor,
    valid_ref: torch.Tensor,
    valid_mov: torch.Tensor
) -> RelativeMotionResult:
    ...
```

Recommended approach:
- estimate per-frame rigid transform of reference part back to canonical
- apply inverse reference motion to moving-part points

For each frame `t`:
- estimate `T_ref_t` aligning reference-part points at frame `t` to canonical/reference frame
- compute `T_ref_t^{-1}` or consistent convention
- transform moving-part points into reference frame

This is more correct than just subtracting centroids.

Use robust rigid alignment (Procrustes / Kabsch) over reference-part points if needed.

---

## 9.4 External pointwise joint clue adapter

Implement in `external/joint_clue_adapter.py`.

This wraps the already-working module.

Required interface:

```python
class JointClueEstimator:
    def infer(self, xyz_traj: np.ndarray) -> dict:
        """
        Input:
            xyz_traj: [T,3]
        Output:
            {
                "type_scores": {"revolute": float, "prismatic": float, "screw": float},
                "axis_dir": np.ndarray,      # [3]
                "axis_point": Optional[np.ndarray],
                "pitch": Optional[float],
                "confidence": float
            }
        """
```

If the underlying module only returns:
- type clue
- axis versor

then adapter must still normalize output to the above dict and fill unavailable fields with `None`.

---

## 9.5 Pointwise initialization stage

Implement in `joint/pointwise_init.py`.

Steps:
1. subsample moving-part canonical points
2. retrieve their relative trajectories in the reference frame
3. call the external joint clue module per trajectory
4. collect clues with confidences

Sampling strategies:
- random
- farthest point sampling
- confidence-weighted sampling

Recommended default:
- farthest point sampling in canonical moving-part cloud

---

## 9.6 Consensus stage

Implement in `joint/consensus.py`.

Responsibilities:
- aggregate pointwise type clues into global type priors
- aggregate axis directions robustly
- optionally estimate initial pitch for screw

### Type consensus
Simple options:
- weighted vote by confidence
- weighted mean of type scores

### Axis consensus
Important: axis direction has sign ambiguity.
If a point predicts `u` and another predicts `-u`, they may represent the same axis direction.

Algorithm:
1. pick the highest-confidence axis as reference
2. flip any axis whose dot product with the reference is negative
3. weighted-average
4. renormalize

### Axis point / pitch consensus
If available from pointwise outputs:
- use weighted median or RANSAC-like aggregation
If unavailable:
- leave to later initialization heuristics

---

## 9.7 Kinematic models to optimize

Implement in `joint/models.py`.

### RevoluteModel

Parameters:
- `axis_raw`: unconstrained `[3]`, normalized internally
- `axis_point`: `[3]`
- `theta`: `[T]`

Forward equation:
For canonical point `Y`:
```python
u = normalize(axis_raw)
pred_t = R(u, theta_t) @ (Y - o) + o
```

### PrismaticModel

Parameters:
- `axis_raw`: `[3]`
- `disp`: `[T]`

Equation:
```python
pred_t = Y + disp_t * u
```

### ScrewModel

Parameters:
- `axis_raw`
- `axis_point`
- `theta`
- `pitch`

Equation:
```python
pred_t = R(u, theta_t) @ (Y - o) + o + pitch * theta_t * u
```

Implement all in PyTorch, fully differentiable.

---

## 9.8 Initializing model parameters

Implement helpers in `joint/models.py` or `joint/optimizer.py`.

### For revolute
- `axis_raw` from consensus axis
- `axis_point` initialized from moving-part centroid, or centroid projected orthogonally to axis
- `theta` initialized from angle progression if possible, else zeros

### For prismatic
- `axis_raw` from consensus axis
- `disp_t` initialized by projection of centroid displacement onto axis

### For screw
- `axis_raw` from consensus axis
- `axis_point` like revolute
- `theta_t` from projected rotation-like motion if possible
- `pitch`:
  - from external clue if available
  - else initialize near zero

---

## 9.9 Global joint losses

Implement in `joint/losses.py`.

### 1. Trajectory reconstruction loss
For observed relative moving-part trajectories `X_obs[p,t]` and predicted trajectories `X_pred[p,t]`:

```python
L_fit = sum_p,t valid[p,t] * weight[p] * huber(||X_pred[p,t] - X_obs[p,t]||)
```

### 2. Temporal smoothness
Use on state variables:
- `theta_t` or `disp_t`

```python
L_temporal = sum_t huber(state[t+1] - state[t])
```

### 3. Axis prior regularization
Weakly keep optimized axis near consensus axis:

```python
L_axis_reg = 1 - dot(u, u_init)^2
```

Use squared cosine because sign ambiguity may not matter.

### 4. Optional axis location regularization
Weak prior on axis point near initialized location if optimization drifts wildly.

### 5. Screw pitch regularization
Keep pitch small unless data needs it:

```python
L_pitch = huber(pitch)
```

with low weight.

### Total per candidate model

```python
L_joint =
    lambda_fit * L_fit
  + lambda_temporal * L_temporal
  + lambda_axis * L_axis_reg
  + lambda_axis_point * L_axis_point_reg
  + lambda_pitch * L_pitch   # screw only
```

---

## 9.10 Model selection

Implement in `joint/selection.py`.

Fit all candidate models:
- revolute
- prismatic
- screw

Select best by:
```python
score = optimized_loss + complexity_penalty[model]
```

Recommended default penalties:
- revolute: `0.0`
- prismatic: `0.0`
- screw: small positive penalty

The screw penalty prevents trivial overfitting.

---

# 10. Mathematical Details Codex Must Preserve

## 10.1 Same-part probability
Given soft membership probability `w_i` that point `i` belongs to part 1:

```python
s_ij = w_i*w_j + (1-w_i)*(1-w_j)
```

This is the probability two points are on the same part.

## 10.2 Different-part probability
```python
d_ij = 1 - s_ij
```

## 10.3 Pairwise rigidity
Same-part pairs should preserve relative distance over time.

## 10.4 CoG rigidity
Within a rigid body, point-to-centroid radius stays approximately constant.

## 10.5 Relative motion
Joint estimation must use moving-part trajectories in the reference-part frame.

## 10.6 Global multi-point fitting
Do not select a joint only from one point’s clue. Use pointwise clues only as priors/init, then fit globally over all moving-part points.

---

# 11. Command Line Interfaces to Implement

## 11.1 `scripts/run_segmentation.py`
Inputs:
- sequence path
- track source or tracker config
- feature config
- segmentation config

Outputs:
- point labels
- dense masks
- segmentation diagnostics
- optional visualizations

## 11.2 `scripts/run_joint.py`
Inputs:
- segmentation result
- relative motion inputs
- external joint clue backend config
- joint optimization config

Outputs:
- joint type
- axis direction
- axis point
- pitch if screw
- per-frame state
- visualization

## 11.3 `scripts/run_pipeline.py`
Runs both stages end-to-end.

---

# 12. Visualization Requirements

Codex must implement debug visualization because this project will otherwise be hard to validate.

Provide:
1. tracked points colored by soft label
2. per-frame dense masks of the two parts
3. CoG trajectories of the two candidate parts
4. moving-part trajectories in reference frame
5. axis visualization in 3D
6. per-model fit comparison
7. state-vs-time plots

Use:
- `matplotlib`
- `open3d` for 3D point cloud + axis line visualization

---

# 13. Unit Tests Required

## 13.1 `test_so3_se3.py`
Must verify:
- identity under zero twist
- composition sanity
- transformed points correctness

## 13.2 `test_segmentation_losses.py`
Create synthetic rigid-two-part motion and test:
- same-part pair rigidity loss is low
- cross-part articulated pairs tend to higher residual
- CoG rigidity loss is low for correct grouping
- entropy behaves numerically stably

## 13.3 `test_joint_models.py`
Synthetic data for:
- revolute motion
- prismatic motion
- screw motion

Verify optimization recovers:
- correct type
- approximate axis direction
- approximate state

## 13.4 `test_consensus.py`
Verify:
- axis sign alignment works
- weighted axis mean behaves correctly
- type consensus behaves correctly

## 13.5 `test_pipeline_smoke.py`
Small synthetic smoke test for full pipeline.

---

# 14. Config Details

Codex must provide Hydra/OmegaConf configs.

## 14.1 `configs/segmentation.yaml`

Suggested fields:
```yaml
window:
  size: 8
  stride: 4

tracks:
  min_valid_ratio: 0.7
  max_depth: 3.0

features:
  model_name: "vit_small_patch14_dinov2"
  num_neighbors: 16
  spatial_gate_px: 48

loss:
  lambda_motion: 200.0
  lambda_smooth: 10.0
  lambda_entropy: 0.01
  lambda_pair: 2.0
  lambda_cog: 1.0
  lambda_balance: 0.0
  pair_margin: 0.01
  pair_lambda_sep: 1.0

optimizer:
  lr_logits: 1e-2
  lr_twists: 1e-4
  iterations: 300
  grad_clip: 1.0
  schedule:
    warmup_iters: 80
    pair_start_iter: 60
    cog_start_iter: 120

sampling:
  num_pairs: 4096
```

## 14.2 `configs/joint.yaml`

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

# 15. Important Failure Modes and Mitigations

## 15.1 Segmentation collapse
All points assigned to one part.

Mitigation:
- strong feature smoothness + entropy warm start
- optional weak balance loss
- staged activation of pair/cog losses

## 15.2 Noisy pairwise rigidity
Bad tracks or limited frame overlap can corrupt pair loss.

Mitigation:
- valid masks
- robust penalties
- pair subsampling
- minimum overlap threshold

## 15.3 CoG instability
Soft centroid becomes unstable when one cluster is tiny.

Mitigation:
- clamp denominator
- optional balance prior
- activate CoG loss after warmup

## 15.4 Wrong joint type due to overfitting
Screw model overfits.

Mitigation:
- complexity penalty
- pitch regularization
- use pointwise clue priors

## 15.5 Axis location drift
Axis direction is good but axis point is unstable.

Mitigation:
- use multi-point global trajectory loss
- initialize from part centroid / boundary region
- optionally regularize axis point weakly

---

# 16. Coding Style Requirements

Codex must:
- use type hints everywhere
- add docstrings to public functions/classes
- separate tensor code from NumPy code
- keep adapters isolated from core logic
- use dataclasses for module interfaces
- write readable logging
- avoid giant files
- avoid hard-coded constants in source; place them in configs

---

# 17. Implementation Order for Codex

Codex should implement in this order:

## Phase 1
- data dataclasses
- RGB-D I/O
- track lifting
- DINO wrapper
- feature graph
- SO(3)/SE(3) utilities

## Phase 2
- segmentation variables
- motion fit loss
- smoothness loss
- entropy loss
- segmentation optimizer
- simple mask propagation

## Phase 3
- pairwise rigidity loss
- CoG rigidity loss
- improved propagation and diagnostics

## Phase 4
- reference motion estimation
- external joint clue adapter
- pointwise initialization
- consensus fusion

## Phase 5
- revolute/prismatic/screw models
- joint losses
- candidate fitting
- model selection

## Phase 6
- CLI scripts
- visualization
- tests
- docs

---

# 18. Pseudocode Summary

## 18.1 Segmentation

```python
sequence = load_rgbd_sequence(...)
tracks = tracker.track(sequence)
tracks.xyz, tracks.valid = lift_tracks_to_3d(...)
features = dino.extract_dense(sequence.rgb[anchor])
track_features = dino.sample_points(features, tracks.xy[:, anchor])
graph = build_feature_graph(track_features)

init_logits = initialize_two_clusters(track_features)

vars = SegmentationVariables(...)
vars.logits.data = init_logits

for iter in range(num_iters):
    w = sigmoid(vars.logits)
    L_motion = motion_fit_loss(...)
    L_smooth = feature_smoothness_loss(...)
    L_ent = entropy_loss(...)

    L = lambda_motion * L_motion + lambda_smooth * L_smooth + lambda_entropy * L_ent

    if iter >= pair_start:
        L += lambda_pair * pairwise_rigidity_loss(...)

    if iter >= cog_start:
        L += lambda_cog * cog_rigidity_loss(...)

    if balance_enabled:
        L += lambda_balance * balance_loss(...)

    optimize(L)

seg_result = build_segmentation_result(...)
```

## 18.2 Joint estimation

```python
rel = compute_relative_motion(seg_result, ...)
sampled_trajs = sample_point_trajectories(rel.canonical_points, rel.moving_points_rel)

clues = [joint_clue_backend.infer(traj) for traj in sampled_trajs]
consensus = build_consensus(clues)

candidate_results = []
for model_name in ["revolute", "prismatic", "screw"]:
    model = build_model(model_name, consensus, rel)
    result = optimize_joint_model(model, rel, config)
    candidate_results.append(result)

best = select_best_model(candidate_results)
```

---

# 19. Explicit Note About External Point-Trajectory Module

Codex must **not** rewrite or replace the user’s already-working module that, from a point trajectory over time, returns:
- the clue of the joint
- the axis versor

Instead:
- wrap it cleanly
- normalize its outputs
- use it as a pointwise prior/initializer
- fuse multiple pointwise outputs into a robust global initialization
- then refine with global optimization over all moving-part points

This is a key requirement.

---

# 20. Deliverables Expected from Codex

Codex should produce:
1. a complete Python package implementing the above
2. configs
3. CLI scripts
4. tests
5. docstrings
6. minimal README instructions
7. modular adapters for:
   - tracker
   - DINO backend
   - external joint clue backend

---

# 21. Final Instruction to Codex

Implement the system as a **research-grade, modular, debuggable Python project**, prioritizing:
- clarity
- correctness
- physical consistency
- extensibility

Do not oversimplify the losses.
Do not remove the two-stage decomposition.
Do not replace the external pointwise joint clue module.
Use it to initialize and stabilize a stronger global joint fit.

The segmentation stage must be driven by **rigidity physics**, not only feature priors.
The joint stage must be driven by **relative motion in the reference-part frame** and refined by **global multi-point optimization**.
