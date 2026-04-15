#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

VIDEO_PATH="${1:?usage: extract_frames.sh <video> <output_dir> [fps]}"
OUTPUT_DIR="${2:?usage: extract_frames.sh <video> <output_dir> [fps]}"
FPS="${3:-}"

CMD=(python -m dynamic_recon.run --video "${VIDEO_PATH}" --output "${OUTPUT_DIR}" --dry-run)
if [ -n "${FPS}" ]; then
  CMD+=(--fps "${FPS}")
fi

"${CMD[@]}"
