# Method Status

Target pipeline:

1. Run DA3 on the whole video.
2. Compute static-world reprojection residual maps.
3. Threshold only the highest-confidence moving regions.
4. Use those regions to seed SAM3.
5. Get dynamic instance masks over time.
6. Train a lightweight fusion network to predict final soft dynamic masks.
7. Refine poses using only confident static pixels.
8. Track static pixels by 3D reprojection.
9. Track dynamic regions by SAM3 masks plus sparse support points.

Current implementation mapping:

- `DA3 on the whole video`
  - [dynamic_recon/pipelines/preprocess.py](/home/samuelemara/Joint3DGS/DA3_SAM3/dynamic_recon/pipelines/preprocess.py)
  - Whole-video mode is config-driven and chunked for VRAM safety.

- `Static-world reprojection residual maps`
  - [dynamic_recon/geometry/residuals.py](/home/samuelemara/Joint3DGS/DA3_SAM3/dynamic_recon/geometry/residuals.py)

- `Highest-confidence moving regions`
  - [dynamic_recon/sam3/prompts.py](/home/samuelemara/Joint3DGS/DA3_SAM3/dynamic_recon/sam3/prompts.py)

- `Seed SAM3`
  - [dynamic_recon/pipelines/infer_initial.py](/home/samuelemara/Joint3DGS/DA3_SAM3/dynamic_recon/pipelines/infer_initial.py)
  - [dynamic_recon/sam3/wrapper.py](/home/samuelemara/Joint3DGS/DA3_SAM3/dynamic_recon/sam3/wrapper.py)

- `Dynamic instance masks over time`
  - [dynamic_recon/sam3/wrapper.py](/home/samuelemara/Joint3DGS/DA3_SAM3/dynamic_recon/sam3/wrapper.py)

- `Lightweight fusion network`
  - [dynamic_recon/fusion/model.py](/home/samuelemara/Joint3DGS/DA3_SAM3/dynamic_recon/fusion/model.py)
  - [dynamic_recon/fusion/features.py](/home/samuelemara/Joint3DGS/DA3_SAM3/dynamic_recon/fusion/features.py)
  - [dynamic_recon/fusion/trainer.py](/home/samuelemara/Joint3DGS/DA3_SAM3/dynamic_recon/fusion/trainer.py)

- `Pose refinement on confident static pixels`
  - [dynamic_recon/geometry/pose_refine.py](/home/samuelemara/Joint3DGS/DA3_SAM3/dynamic_recon/geometry/pose_refine.py)
  - Implemented as lightweight local 6D refinement, not full bundle adjustment.

- `Static pixel tracking by 3D reprojection`
  - [dynamic_recon/tracking/static_tracks.py](/home/samuelemara/Joint3DGS/DA3_SAM3/dynamic_recon/tracking/static_tracks.py)

- `Dynamic tracking by SAM masks plus sparse support points`
  - [dynamic_recon/tracking/dynamic_tracks.py](/home/samuelemara/Joint3DGS/DA3_SAM3/dynamic_recon/tracking/dynamic_tracks.py)
  - [dynamic_recon/tracking/support_points.py](/home/samuelemara/Joint3DGS/DA3_SAM3/dynamic_recon/tracking/support_points.py)
  - Support-point propagation uses optical flow plus mask gating.

Current outer loop:

- DA3
- residuals
- confident dynamic seeding
- SAM propagation
- fusion training
- static-only pose refinement
- residual recomputation
- SAM rerun on updated priors each outer iteration

Current minimal outputs:

- [exports/dynamic_masked.mp4](/home/samuelemara/Joint3DGS/DA3_SAM3/outputs)
- [exports/camera_poses.npy](/home/samuelemara/Joint3DGS/DA3_SAM3/outputs)
- [exports/camera_poses.json](/home/samuelemara/Joint3DGS/DA3_SAM3/outputs)

Known limits:

- Whole-video runs are still limited by GPU memory and disk throughput.
- Pose refinement is still lightweight.
- Dynamic support tracking is heuristic rather than learned.
