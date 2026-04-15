#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
THIRD_DIR="$ROOT_DIR/.third_party"
ENV_PREFIX="${ENV_PREFIX:-$THIRD_DIR/conda_env}"
REPOS_DIR="$THIRD_DIR/repos"
CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
REQ_FILE="$ROOT_DIR/env/requirements-pip.txt"

mkdir -p "$THIRD_DIR" "$REPOS_DIR" "$THIRD_DIR/conda_pkgs" "$THIRD_DIR/cache"

if [ ! -x "$CONDA_BIN" ]; then
  echo "Conda binary not found: $CONDA_BIN"
  exit 1
fi

export CONDA_NO_PLUGINS=true
export CONDA_PKGS_DIRS="$THIRD_DIR/conda_pkgs"
export XDG_CACHE_HOME="$THIRD_DIR/cache"
export CONDA_SOLVER=classic
export PIP_CACHE_DIR="$THIRD_DIR/cache/pip"

echo "[1/6] Creating conda env at: $ENV_PREFIX"
"$CONDA_BIN" create --solver classic -y -p "$ENV_PREFIX" python=3.10 pip

echo "[2/6] Upgrading pip"
"$CONDA_BIN" run -p "$ENV_PREFIX" python -m pip install --upgrade pip

echo "[3/6] Installing core Python dependencies"
"$CONDA_BIN" run -p "$ENV_PREFIX" python -m pip install -r "$REQ_FILE"

echo "[4/6] Installing project in editable mode"
"$CONDA_BIN" run -p "$ENV_PREFIX" python -m pip install -e "$ROOT_DIR"

echo "[5/6] Cloning/updating third-party repos"
if [ ! -d "$REPOS_DIR/dinov2/.git" ]; then
  git clone https://github.com/facebookresearch/dinov2.git "$REPOS_DIR/dinov2"
else
  git -C "$REPOS_DIR/dinov2" pull --ff-only
fi

if [ ! -d "$REPOS_DIR/co-tracker/.git" ]; then
  git clone https://github.com/facebookresearch/co-tracker.git "$REPOS_DIR/co-tracker"
else
  git -C "$REPOS_DIR/co-tracker" pull --ff-only
fi

# TAPIR repo (override with TAPIR_REPO_URL if needed)
TAPIR_REPO_URL="${TAPIR_REPO_URL:-https://github.com/google-research/tapir.git}"
if [ ! -d "$REPOS_DIR/tapir/.git" ]; then
  git clone "$TAPIR_REPO_URL" "$REPOS_DIR/tapir" || true
else
  git -C "$REPOS_DIR/tapir" pull --ff-only || true
fi

echo "[6/6] Installing third-party repos into conda env"
"$CONDA_BIN" run -p "$ENV_PREFIX" python -m pip install -e "$REPOS_DIR/dinov2" --no-deps --no-build-isolation
"$CONDA_BIN" run -p "$ENV_PREFIX" python -m pip install -e "$REPOS_DIR/co-tracker" --no-deps --no-build-isolation

if [ -d "$REPOS_DIR/tapir" ]; then
  if [ -f "$REPOS_DIR/tapir/pyproject.toml" ] || [ -f "$REPOS_DIR/tapir/setup.py" ]; then
    "$CONDA_BIN" run -p "$ENV_PREFIX" python -m pip install -e "$REPOS_DIR/tapir" --no-deps --no-build-isolation || true
  else
    echo "TAPIR repo has no installable package metadata; using repo_path import only."
  fi
fi

echo

echo "Provisioning complete"
echo "Env python: $ENV_PREFIX/bin/python"
echo "Third-party repos: $REPOS_DIR"
