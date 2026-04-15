from __future__ import annotations

import torch



def hat(omega: torch.Tensor) -> torch.Tensor:
    if omega.shape[-1] != 3:
        raise ValueError("omega must end with dim 3")
    wx, wy, wz = omega.unbind(dim=-1)
    z = torch.zeros_like(wx)
    row0 = torch.stack([z, -wz, wy], dim=-1)
    row1 = torch.stack([wz, z, -wx], dim=-1)
    row2 = torch.stack([-wy, wx, z], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)



def so3_exp(omega: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Exponential map from axis-angle to rotation matrix."""
    if omega.shape[-1] != 3:
        raise ValueError("omega must end with dim 3")

    theta = torch.linalg.norm(omega, dim=-1, keepdim=True)
    omega_hat = hat(omega)
    omega_hat2 = omega_hat @ omega_hat

    I = torch.eye(3, device=omega.device, dtype=omega.dtype)
    I = I.expand(*omega.shape[:-1], 3, 3)

    theta2 = theta * theta
    a = torch.where(theta > eps, torch.sin(theta) / theta, 1.0 - theta2 / 6.0)
    b = torch.where(theta > eps, (1.0 - torch.cos(theta)) / theta2, 0.5 - theta2 / 24.0)

    return I + a.unsqueeze(-1) * omega_hat + b.unsqueeze(-1) * omega_hat2
