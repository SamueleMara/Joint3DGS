from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class VideoConfig:
    resize_long_edge: int | None = None
    max_frames: int | None = None
    stride: int = 1


@dataclass(slots=True)
class DA3Config:
    enabled: bool = True
    checkpoint: str | None = None
    model_name: str = "depth-anything/DA3-LARGE-1.1"
    batch_size: int = 1
    window_size: int = 16
    stride: int = 8
    mixed_precision: bool = True
    use_ray_pose: bool = False
    ref_view_strategy: str = "middle"
    process_res: int = 504
    process_res_method: str = "upper_bound_resize"
    oom_retry_process_res: list[int] = field(default_factory=lambda: [560, 448, 392])
    sequence_chunk_size: int | None = None
    sequence_chunk_overlap: int = 0
    allow_mock: bool = True


@dataclass(slots=True)
class GeometryConfig:
    pair_offsets: list[int] = field(default_factory=lambda: [-2, -1, 1, 2])
    alpha_3d: float = 1.0
    alpha_rel: float = 0.8
    alpha_flow: float = 1.0
    alpha_depth: float = 0.5
    alpha_rgb: float = 0.2
    alpha_cycle: float = 0.5
    alpha_vis: float = 1.0
    seed_threshold_high: float = 0.8
    temporal_smoothing_alpha: float = 0.3
    static_threshold_low: float = 0.2
    flow_consistency_thresh: float = 1.5
    occlusion_abs_tol: float = 0.05
    occlusion_rel_tol: float = 0.1


@dataclass(slots=True)
class SAM3Config:
    enabled: bool = True
    checkpoint: str | None = None
    model_id: str | None = None
    version: str = "sam3.1"
    resize_long_edge: int | None = None
    gpus_to_use: list[int] | None = None
    offload_video_to_cpu: bool = False
    offload_state_to_cpu: bool = False
    async_loading_frames: bool = False
    max_regions_per_frame: int = 8
    max_prompt_frames: int = 8
    min_region_area: int = 32
    points_per_region: int = 3
    negative_ring: int = 5
    allow_mock: bool = True


@dataclass(slots=True)
class FusionConfig:
    enabled: bool = True
    base_channels: int = 32
    lr: float = 1.0e-3
    batch_size: int = 2
    epochs_per_outer_iter: int = 3
    w_sam: float = 1.0
    w_geo: float = 1.0
    w_static_seed: float = 0.5
    w_dynamic_seed: float = 0.5
    w_tv: float = 0.05
    w_temp: float = 0.1
    w_abs_motion: float = 0.4
    w_motion_rank: float = 0.2
    w_contrastive: float = 0.25
    w_pair_3d_contrastive: float = 0.35
    contrastive_pairs: int = 1024
    contrastive_neighbor_radius: int = 6
    contrastive_beta: float = 8.0
    pairwise_num_pairs: int = 2048
    pairwise_rel_scale: float = 10.0
    pairwise_abs_scale: float = 4.0
    pairwise_min_visibility: float = 0.2
    dynamic_geo_high: float = 0.7
    dynamic_sam_high: float = 0.6
    static_geo_low: float = 0.18
    static_sam_low: float = 0.12
    min_vis_dynamic: float = 0.3
    min_vis_static: float = 0.35
    export_dynamic_threshold: float = 0.55
    export_neighbor_threshold: float = 0.45
    export_point_stride: int = 6
    export_max_points: int = 120000
    backward_propagation_decay: float = 0.92
    backward_propagation_iters: int = 2


@dataclass(slots=True)
class PoseRefineConfig:
    enabled: bool = True
    lr: float = 1.0e-4
    steps: int = 100
    lambda_depth: float = 0.5
    lambda_rgb: float = 0.1
    lambda_temporal_translation: float = 5.0e-3
    lambda_temporal_rotation: float = 2.0e-3
    lambda_pose_anchor: float = 1.0e-3
    keyframe_only: bool = True
    keyframe_stride: int = 4
    camera_mode: str = "moving"  # moving | fixed | auto
    auto_static_translation_thresh: float = 1.0e-3
    auto_static_rotation_thresh_deg: float = 0.1
    fallback_to_cpu_on_cuda_error: bool = True


@dataclass(slots=True)
class PoseInitConfig:
    enabled: bool = True
    max_corners: int = 2000
    quality_level: float = 0.01
    min_distance: float = 7.0
    pnp_reprojection_error: float = 3.0
    min_inliers: int = 32
    camera_mode: str = "moving"  # moving | fixed | auto
    static_check_pairs: int = 8
    static_motion_median_px: float = 0.4


@dataclass(slots=True)
class PipelineRuntimeConfig:
    num_outer_iters: int = 2
    save_debug_every_iter: bool = False
    save_intermediates: bool = False
    rerun_segmentation_each_iter: bool = True


@dataclass(slots=True)
class PipelineConfig:
    device: str = "cuda"
    dtype: str = "float32"
    seed: int = 0
    video: VideoConfig = field(default_factory=VideoConfig)
    da3: DA3Config = field(default_factory=DA3Config)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    sam3: SAM3Config = field(default_factory=SAM3Config)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    pose_init: PoseInitConfig = field(default_factory=PoseInitConfig)
    pose_refine: PoseRefineConfig = field(default_factory=PoseRefineConfig)
    pipeline: PipelineRuntimeConfig = field(default_factory=PipelineRuntimeConfig)
