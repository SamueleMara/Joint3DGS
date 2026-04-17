#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
ENV_NAME="${ENV_NAME:-da3-only}"
eval "$(conda shell.bash hook)"
conda create -y -n "${ENV_NAME}" python=3.12 || true
conda activate "${ENV_NAME}"
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install xformers -r requirements.txt -e third_party/depth_anything_3 -e .
