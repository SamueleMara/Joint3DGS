import torch

from articulation.geometry import invert_transform, se3_exp, so3_exp, transform_points


def test_so3_identity_zero_vector():
    w = torch.zeros(3)
    R = so3_exp(w)
    assert torch.allclose(R, torch.eye(3), atol=1e-6)


def test_se3_identity_zero_twist():
    xi = torch.zeros(6)
    T = se3_exp(xi)
    assert torch.allclose(T, torch.eye(4), atol=1e-6)


def test_transform_invert_roundtrip():
    xi = torch.tensor([0.1, -0.2, 0.05, 0.2, -0.1, 0.3])
    T = se3_exp(xi)
    Ti = invert_transform(T)
    x = torch.randn(8, 3)
    y = transform_points(T, x)
    xr = transform_points(Ti, y)
    assert torch.allclose(x, xr, atol=1e-5)


def test_composition_sanity():
    xi1 = torch.tensor([0.05, 0.0, 0.0, 0.1, 0.0, 0.0])
    xi2 = torch.tensor([0.0, 0.05, 0.0, 0.0, 0.1, 0.0])
    T1 = se3_exp(xi1)
    T2 = se3_exp(xi2)
    x = torch.randn(6, 3)
    y = transform_points(T2, transform_points(T1, x))
    T12 = T2 @ T1
    y2 = transform_points(T12, x)
    assert torch.allclose(y, y2, atol=1e-5)
