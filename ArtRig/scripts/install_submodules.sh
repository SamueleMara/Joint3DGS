#!/usr/bin/env bash
set -euo pipefail

python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt

# Local project
python -m pip install -e .

# Editable submodules if they expose installable packages.
for module_dir in \
  submodules/dinov2 \
  submodules/omniglue \
  submodules/lightglue \
  submodules/joint_clue_repo \
  submodules/depth_anything_3 \
  submodules/tapnet
  do
  if [ -f "$module_dir/setup.py" ] || [ -f "$module_dir/pyproject.toml" ]; then
    python -m pip install -e "$module_dir"
  fi
done
