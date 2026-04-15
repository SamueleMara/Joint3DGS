# Dynamic Recon

Research-style repository for DA3 + SAM3 video analysis. The current repo state is centered on the preliminary static/dynamic understanding pipeline implemented under `dynamic_recon/`.

Main docs:

- [README_preliminary_dynamic_recon.md](/home/samuelemara/Joint3DGS/DA3_SAM3/README_preliminary_dynamic_recon.md)

Quick start:

```bash
conda activate da3-sam3-dynamic
python -m dynamic_recon.cli.run_pipeline \
  --video /path/to/input.mp4 \
  --outdir outputs/demo_run \
  --config configs/quickstart.yaml
```

Backend selection:

- `--sam-backend sam2`
  - uses the locally vendored and compiled `third_party/sam2`
  - best choice when you want the open model path and lower setup friction
- `--sam-backend sam3`
  - uses the locally vendored `third_party/sam3`
  - requires Hugging Face authentication and approved access to `facebook/sam3.1`

For GPUs around 8 GB VRAM, prefer `configs/convergence_7gb.yaml`, and if that still OOMs use `configs/convergence_7gb_safe.yaml`.
`configs/convergence_7gb_safe.yaml` is currently set to use open SAM2.1 instead of gated SAM3.

Setup scripts now live under `setup/`:

- `setup/setup_env.sh`
- `setup/setup_submodules.sh`
- `setup/download_checkpoints.sh`

Python launcher scripts now live under `scripts/`:

- `scripts/run_preliminary_pipeline.py`
- `scripts/run_preliminary_da3.py`
- `scripts/run_preliminary_sam3.py`
- `scripts/export_preliminary_tracks.py`

Installation details for DA3, SAM2, and SAM3:

- [README_preliminary_dynamic_recon.md](/home/samuelemara/Joint3DGS/DA3_SAM3/README_preliminary_dynamic_recon.md)
