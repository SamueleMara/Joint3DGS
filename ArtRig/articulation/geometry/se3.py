from __future__ import annotations

import torch

from articulation.geometry.so3 import hat, so3_exp



def se3_exp(xi: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Exponential map from twist [w,v] to SE(3) matrix."""
    if xi.shape[-1] != 6:
        raise ValueError("xi must end with dim 6")

    omega = xi[..., :3]
    v = xi[..., 3:]

    theta = torch.linalg.norm(omega, dim=-1, keepdim=True)
    omega_hat = hat(omega)
    omega_hat2 = omega_hat @ omega_hat

    R = so3_exp(omega, eps=eps)

    I = torch.eye(3, device=xi.device, dtype=xi.dtype)
    I = I.expand(*xi.shape[:-1], 3, 3)

    theta2 = theta * theta
    theta_safe = torch.clamp(theta, min=eps)
    theta2_safe = theta_safe * theta_safe
    theta3_safe = theta2_safe * theta_safe

    # Safe exact forms.
    b_exact = (1.0 - torch.cos(theta)) / theta2_safe
    c_exact = (theta - torch.sin(theta)) / theta3_safe

    # Small-angle Taylor expansions.
    b_taylor = 0.5 - theta2 / 24.0 + (theta2 * theta2) / 720.0
    c_taylor = 1.0 / 6.0 - theta2 / 120.0 + (theta2 * theta2) / 5040.0

    b = torch.where(theta > eps, b_exact, b_taylor)
    c = torch.where(theta > eps, c_exact, c_taylor)

    V = I + b.unsqueeze(-1) * omega_hat + c.unsqueeze(-1) * omega_hat2
    t = (V @ v.unsqueeze(-1)).squeeze(-1)

    T = torch.zeros(*xi.shape[:-1], 4, 4, device=xi.device, dtype=xi.dtype)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T



def transform_points(T: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Apply SE(3) transform(s) to points.

    Supported:
    - T: [4,4], X: [N,3]
    - T: [B,4,4], X: [B,N,3] or [N,3]
    """
    if T.shape[-2:] != (4, 4):
        raise ValueError("T must end with [4,4]")
    if X.shape[-1] != 3:
        raise ValueError("X must end with dim 3")

    R = T[..., :3, :3]
    t = T[..., :3, 3]

    if T.ndim == 2:
        return X @ R.transpose(-1, -2) + t

    if X.ndim == 2:
        x = X.unsqueeze(0).expand(T.shape[0], -1, -1)
    else:
        x = X
        if x.shape[0] != T.shape[0]:
            raise ValueError("Batch mismatch between T and X")

    return x @ R.transpose(-1, -2) + t.unsqueeze(1)
