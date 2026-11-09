"""Tests of the masks with a smooth rim (brief TESTS.md, part D).

The oracles are the logistic formula built in the test from
``torch.fft.fftfreq`` indices, the symmetry ``i -> -i``, the Sommerfeld
expansion of the disc area, and the angle of each pixel.
"""

from __future__ import annotations

import math

import pytest
import torch

from scatterem.physics import optics as O


def signed_index(n: int) -> torch.Tensor:
    return torch.fft.fftfreq(n, dtype=torch.float64) * n


def logistic(shape, radius, rim, px_ratio, index=signed_index) -> torch.Tensor:
    iy, ix = index(shape[0]), index(shape[1])
    d = torch.sqrt(iy[:, None] ** 2 + (px_ratio * ix[None, :]) ** 2)
    return 1.0 / (1.0 + torch.exp(4.0 * (d - radius) / rim))


def arange_index(n: int) -> torch.Tensor:
    return torch.arange(n, dtype=torch.float64)


SHAPES = [(8, 8), (9, 9), (8, 9), (9, 8)]


# --- D.1 values -------------------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("radius", [0.0, 1.5, 3.0])
@pytest.mark.parametrize("rim", [0.5, 2.0])
@pytest.mark.parametrize("px_ratio", [1.0, 0.7, 1.6])
def test_disc_equals_logistic_formula(shape, radius, rim, px_ratio) -> None:
    got = O.smooth_disc(
        shape, radius, rim_px=rim, px_ratio=px_ratio, dtype=torch.float64
    )
    ref = logistic(shape, radius, rim, px_ratio)
    torch.testing.assert_close(got, ref, rtol=4e-16, atol=1e-300)
    got32 = O.smooth_disc(
        shape, radius, rim_px=rim, px_ratio=px_ratio, dtype=torch.float32
    )
    torch.testing.assert_close(got32, ref.float(), rtol=1.2e-7, atol=1e-38)


def test_value_check_detects_non_dft_order() -> None:
    got = O.smooth_disc((9, 9), 3.0, dtype=torch.float64)
    wrong = logistic((9, 9), 3.0, O.RIM_WIDTH_PX, 1.0, index=arange_index)
    assert not torch.allclose(got, wrong, rtol=1e-6)


def test_default_rim_width() -> None:
    got = O.smooth_disc((16, 16), 4.0, dtype=torch.float64)
    torch.testing.assert_close(got, logistic((16, 16), 4.0, O.RIM_WIDTH_PX, 1.0))


# --- D.2 symmetry -----------------------------------------------------------


def mirror(mask: torch.Tensor, axis: int) -> torch.Tensor:
    n = mask.shape[axis]
    idx = (-torch.arange(n)) % n
    return mask.index_select(axis, idx)


@pytest.mark.parametrize("shape", SHAPES)
def test_masks_are_symmetric(shape) -> None:
    masks = [
        O.smooth_disc(shape, 2.3, px_ratio=1.3),
        O.smooth_annulus(shape, 1.0, 2.7, rim_px=0.8, px_ratio=0.9),
    ]
    for mask in masks:
        for axis in (0, 1):
            assert torch.equal(mask, mirror(mask, axis))


def test_symmetry_check_detects_shifted_mask() -> None:
    mask = torch.roll(O.smooth_disc((9, 8), 2.3), 1, dims=0)
    assert not torch.equal(mask, mirror(mask, 0))


def test_px_ratio_none_means_square_pixels() -> None:
    for shape in ((9, 8), (8, 11)):
        square = O.smooth_disc(shape, 2.3, px_ratio=1.0, dtype=torch.float64)
        assert torch.equal(O.smooth_disc(shape, 2.3, dtype=torch.float64), square)
        ring = O.smooth_annulus(shape, 1.0, 2.7, px_ratio=1.0, dtype=torch.float64)
        assert torch.equal(O.smooth_annulus(shape, 1.0, 2.7, dtype=torch.float64), ring)
    stretched = O.smooth_disc((9, 8), 2.3, px_ratio=1.3, dtype=torch.float64)
    assert not torch.equal(O.smooth_disc((9, 8), 2.3, dtype=torch.float64), stretched)


# --- D.3 area ---------------------------------------------------------------


def area_reference(radius: float, rim: float, px_ratio: float) -> float:
    return (math.pi * radius**2 + math.pi**3 * rim**2 / 48) / px_ratio


@pytest.mark.parametrize("px_ratio", [1.0, 0.8])
@pytest.mark.parametrize("rim", [2.0, 1.0])
def test_disc_area(px_ratio: float, rim: float) -> None:
    mask = O.smooth_disc(
        (256, 256), 40.0, rim_px=rim, px_ratio=px_ratio, dtype=torch.float64
    )
    total = mask.sum().item()
    assert abs(total - area_reference(40.0, rim, px_ratio)) <= 1.0
    assert abs(total - area_reference(40.05, rim, px_ratio)) > 1.0  # control


# --- D.4 illumination disc --------------------------------------------------


def test_illumination_disc_follows_the_angle() -> None:
    shape, g_y, g_x, lam, angle = (64, 48), 0.05, 0.07, 0.025, 0.02
    mask = O.illumination_disc(
        shape, (g_y, g_x), angle, lambda_a=lam, dtype=torch.float64
    )
    ky = signed_index(64)[:, None] * g_y
    kx = signed_index(48)[None, :] * g_x
    theta = lam * torch.sqrt(ky**2 + kx**2)
    keep = (theta - angle).abs() > lam * g_y * O.RIM_WIDTH_PX
    assert keep.sum() > 0.9 * keep.numel()
    assert torch.equal((mask > 0.5)[keep], (theta < angle)[keep])
    isotropic = O.smooth_disc(shape, angle / (lam * g_y), dtype=torch.float64)
    assert not torch.equal((isotropic > 0.5)[keep], (theta < angle)[keep])  # control


# --- D.5 annulus ------------------------------------------------------------


def test_annulus_regions() -> None:
    p, q, w = 8.0, 20.0, 1.0
    mask = O.smooth_annulus((64, 64), p, q, rim_px=w, dtype=torch.float64)
    iy, ix = signed_index(64)[:, None], signed_index(64)[None, :]
    d = torch.sqrt(iy**2 + ix**2)
    inside = (d > p + 3 * w) & (d < q - 3 * w)
    outside = (d < p - 3 * w) | (d > q + 3 * w)
    assert ((mask[inside] - 1).abs() < 1e-3).all()
    assert (mask[outside].abs() < 1e-3).all()
    assert not (mask[outside] > 0.5).any()  # control target: a disc fails here
    disc = O.smooth_disc((64, 64), q, rim_px=w, dtype=torch.float64)
    assert (disc[outside] > 0.5).any()


def test_annulus_with_equal_radii_is_small() -> None:
    for r in (0.0, 3.0, 7.5):
        assert O.smooth_annulus((32, 32), r, r, rim_px=1.5).max() <= 0.25


# --- D.6 range, dtype, device -----------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_range_dtype_device(dtype) -> None:
    for mask in (
        O.smooth_disc((12, 10), 3.0, dtype=dtype, device="cpu"),
        O.smooth_annulus((12, 10), 1.0, 3.0, dtype=dtype),
        O.illumination_disc((12, 10), (0.1, 0.1), 0.01, lambda_a=0.02, dtype=dtype),
    ):
        assert mask.dtype == dtype and mask.device.type == "cpu"
        assert not mask.requires_grad
        assert (mask >= 0).all() and (mask <= 1).all()
    assert O.smooth_disc((4, 4), 1.0).dtype == torch.get_default_dtype()


def test_cuda_masks_agree_with_cpu(require_cuda) -> None:
    for dtype in (torch.float32, torch.float64):
        eps = torch.finfo(dtype).eps
        cpu = O.smooth_annulus((33, 40), 2.0, 9.0, px_ratio=1.2, dtype=dtype)
        gpu = O.smooth_annulus(
            (33, 40), 2.0, 9.0, px_ratio=1.2, dtype=dtype, device=require_cuda
        )
        assert gpu.device.type == "cuda" and gpu.dtype == dtype
        assert ((gpu.cpu() - cpu).abs() <= 2 * eps).all()


# --- D.7 refusals -----------------------------------------------------------


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (lambda: O.smooth_disc((8, 8), -1.0), ValueError),
        (lambda: O.smooth_disc((8, 8), math.inf), ValueError),
        (lambda: O.smooth_disc((8, 8), math.nan), ValueError),
        (lambda: O.smooth_disc((8, 8), 1.0, rim_px=0.0), ValueError),
        (lambda: O.smooth_disc((8, 8), 1.0, rim_px=-1.0), ValueError),
        (lambda: O.smooth_disc((8, 8), 1.0, rim_px=math.inf), ValueError),
        (lambda: O.smooth_disc((8, 8), 1.0, px_ratio=0.0), ValueError),
        (lambda: O.smooth_disc((8, 8), 1.0, px_ratio=math.nan), ValueError),
        (lambda: O.smooth_disc((8, 0), 1.0), ValueError),
        (lambda: O.smooth_disc((8, 8, 8), 1.0), ValueError),
        (lambda: O.smooth_disc((8, 8.0), 1.0), TypeError),
        (lambda: O.smooth_disc((8, True), 1.0), TypeError),
        (lambda: O.smooth_disc((8, 8), "1"), TypeError),
        (lambda: O.smooth_disc((8, 8), 1.0, dtype=torch.int32), TypeError),
        (lambda: O.smooth_disc((8, 8), 1.0, dtype=torch.complex64), TypeError),
        (lambda: O.smooth_annulus((8, 8), 3.0, 2.0), ValueError),
        (lambda: O.smooth_annulus((8, 8), -1.0, 2.0), ValueError),
        (
            lambda: O.illumination_disc((8, 8), (0.0, 0.1), 0.01, lambda_a=0.02),
            ValueError,
        ),
        (
            lambda: O.illumination_disc((8, 8), (0.1, -0.1), 0.01, lambda_a=0.02),
            ValueError,
        ),
        (
            lambda: O.illumination_disc((8, 8), (0.1, math.inf), 0.01, lambda_a=0.02),
            ValueError,
        ),
        (
            lambda: O.illumination_disc((8, 8), (0.1, 0.1), -0.01, lambda_a=0.02),
            ValueError,
        ),
        (
            lambda: O.illumination_disc((8, 8), (0.1, 0.1), 0.01, lambda_a=0.0),
            ValueError,
        ),
    ],
)
def test_bad_mask_inputs_are_refused(call, error) -> None:
    with pytest.raises(error):
        call()


def test_refusal_check_detects_accepted_input() -> None:
    with pytest.raises(pytest.fail.Exception):
        with pytest.raises(ValueError):
            O.smooth_annulus((8, 8), 2.0, 2.0)
