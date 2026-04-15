#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "error: run inside a git repository" >&2
  exit 1
fi

git submodule add https://github.com/ByteDance-Seed/Depth-Anything-3.git third_party/depth_anything_3 2>/dev/null || true
git submodule add https://github.com/facebookresearch/sam3.git third_party/sam3 2>/dev/null || true
git submodule update --init --recursive
