from __future__ import annotations

from typing import Any

import torch



def to_device(obj: Any, device: str | torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, device) for v in obj)
    if hasattr(obj, "to"):
        return obj.to(device)
    return obj
