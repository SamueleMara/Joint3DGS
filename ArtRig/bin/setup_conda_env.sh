#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
ENV_PREFIX="${ENV_PREFIX:-$ROOT_DIR/.third_party/conda_env}"
PYTHON_BIN="$ENV_PREFIX/bin/python"

if [ ! -x "$CONDA_BIN" ]; then
  echo "Conda binary not found at $CONDA_BIN"
  exit 1
fi

mkdir -p "$ROOT_DIR/.third_party"

CONDA_NO_PLUGINS=true "$CONDA_BIN" create -y -p "$ENV_PREFIX" python=3.10 pip
CONDA_NO_PLUGINS=true "$CONDA_BIN" run -p "$ENV_PREFIX" python -m pip install --upgrade pip
CONDA_NO_PLUGINS=true "$CONDA_BIN" run -p "$ENV_PREFIX" python -m pip install \
  torch torchvision numpy scipy opencv-python omegaconf hydra-core tqdm matplotlib \
  scikit-learn einops open3d trimesh timm faiss-cpu pytransform3d pytest ruff mypy
CONDA_NO_PLUGINS=true "$CONDA_BIN" run -p "$ENV_PREFIX" python -m pip install -e "$ROOT_DIR"

echo "Environment ready: $ENV_PREFIX"
echo "Use interpreter: $PYTHON_BIN"
