#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ENV_NAME="${ENV_NAME:-da3-sam3-dynamic}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  CONDA_EXTRACT_THREADS=1 CONDA_VERIFY_THREADS=1 conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi
conda activate "${ENV_NAME}"
python -m pip install --upgrade pip wheel
python -m pip install "setuptools<81"
python -m pip install torch==2.10.0 torchvision --index-url "${TORCH_INDEX_URL}"
python -m pip install xformers
python -m pip install einops ninja
python -m pip install flash-attn-3 --no-deps --index-url "${TORCH_INDEX_URL}"
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip install -e third_party/sam2 -e third_party/depth_anything_3 -e third_party/sam3 -e .

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
if [[ -x "${CUDA_HOME}/bin/nvcc" ]]; then
  pushd third_party/sam2 >/dev/null
  PATH="${CUDA_HOME}/bin:${PATH}" \
  CUDA_HOME="${CUDA_HOME}" \
  TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
  SAM2_BUILD_ALLOW_ERRORS=0 \
  conda run -n "${ENV_NAME}" python setup.py build_ext --inplace
  popd >/dev/null
fi
