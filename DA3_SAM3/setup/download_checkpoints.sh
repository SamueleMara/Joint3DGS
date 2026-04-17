#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODEL_NAME="${DA3_MODEL_NAME:-depth-anything/DA3-LARGE-1.1}"
SAM3_MODEL_NAME="${SAM3_MODEL_NAME:-facebook/sam3.1-hiera-large}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints}"
mkdir -p "${CHECKPOINT_DIR}"

DA3_MODEL_NAME="${MODEL_NAME}" CHECKPOINT_DIR="${CHECKPOINT_DIR}" python - <<'PY'
import os
from huggingface_hub import snapshot_download
repo = os.environ.get("DA3_MODEL_NAME", "depth-anything/DA3-LARGE-1.1")
target = os.path.join(os.environ.get("CHECKPOINT_DIR", "checkpoints"), "da3")
snapshot_download(repo_id=repo, local_dir=target, local_dir_use_symlinks=False)
print(target)
PY

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is required for SAM3 checkpoints" >&2
  exit 1
fi

HF_TOKEN="${HF_TOKEN}" SAM3_MODEL_NAME="${SAM3_MODEL_NAME}" CHECKPOINT_DIR="${CHECKPOINT_DIR}" python - <<'PY'
import os
from huggingface_hub import snapshot_download

repo = os.environ["SAM3_MODEL_NAME"]
target = os.path.join(os.environ.get("CHECKPOINT_DIR", "checkpoints"), "sam3")
snapshot_download(
    repo_id=repo,
    local_dir=target,
    local_dir_use_symlinks=False,
    token=os.environ["HF_TOKEN"],
)
print(target)
PY
