#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

VIDEO_PATH="${1:-data/input.mp4}"
OUTPUT_DIR="${2:-outputs/demo}"
PROMPTS="${3:-person,car}"

python -m dynamic_recon.run \
  --video "${VIDEO_PATH}" \
  --output "${OUTPUT_DIR}" \
  --prompts "${PROMPTS}" \
  --allow-mock-models
