# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import ctypes
from pathlib import Path

from hydra import initialize_config_module
from hydra.core.global_hydra import GlobalHydra


def _preload_torch_libs() -> None:
    try:
        import torch
    except Exception:
        return
    torch_lib_dir = Path(torch.__file__).resolve().parent / "lib"
    for name in ("libc10.so", "libtorch_cpu.so", "libtorch_python.so", "libc10_cuda.so"):
        candidate = torch_lib_dir / name
        if candidate.is_file():
            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)


_preload_torch_libs()

if not GlobalHydra.instance().is_initialized():
    initialize_config_module("sam2", version_base="1.2")
