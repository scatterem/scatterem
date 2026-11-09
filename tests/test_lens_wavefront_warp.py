"""Tests of the Warp device functions of the lens wavefront (TESTS.md, part C).

The oracle is the float64 torch evaluation of the same float32 inputs. The
bit-identity test compares the integer views of two float32 results.
"""

from __future__ import annotations


import numpy as np
import pytest
import torch
import warp as wp

from scatterem.physics import optics as O
from scatterem.physics.kernels import lens_wavefront_wp as LW

wp.init()
DEVICES = ["cpu"] + (["cuda"] if wp.is_cuda_available() else [])
EPS32 = 2.0**-23


@wp.kernel
def _general_kernel(
    lens: wp.array(dtype=wp.float32),
    fy: wp.array(dtype=wp.float32),
    fx: wp.array(dtype=wp.float32),
    lam: wp.array(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    out[i] = LW.lens_wavefront_f32(lens, fy[i], fx[i], lam[i])


def run(kernel, lens: np.ndarray, fy, fx, lam, device: str) -> np.ndarray:
    arrays = [wp.array(np.asarray(a, np.float32), device=device) for a in (fy, fx, lam)]
    out = wp.zeros(len(fy), dtype=wp.float32, device=device)
    w_lens = LW.lens_to_warp(torch.tensor(lens, dtype=torch.float32), device=device)
    wp.launch(kernel, dim=len(fy), inputs=[w_lens, *arrays, out], device=device)
    return out.numpy()


def reference(lens32: np.ndarray, fy32, fx32, lam32) -> tuple[np.ndarray, np.ndarray]:
    """Return float64 ``W`` and the bound of the docstring, per point."""
    c = torch.tensor(lens32, dtype=torch.float64)
    w, bound = [], []
    for y, x, lam in zip(fy32, fx32, lam32, strict=True):
        fy = torch.tensor([float(y)], dtype=torch.float64)
        fx = torch.tensor([float(x)], dtype=torch.float64)
        w.append(O.lens_wavefront(c, fy, fx, lambda_a=float(lam)).item())
        basis = O.lens_wavefront(torch.diag(c.abs()), fy, fx, lambda_a=float(lam))
        bound.append(32 * EPS32 * basis.abs().sum().item() + 4 * EPS32 * abs(w[-1]))
    return np.array(w), np.array(bound)


def inputs(rng: np.random.Generator, n: int):
    fy = rng.uniform(-1.5, 1.5, n).astype(np.float32)
    fx = rng.uniform(-1.5, 1.5, n).astype(np.float32)
    lam = rng.uniform(0.0197, 0.037, n).astype(np.float32)
    scale = np.array([0.02 ** (1 - n) * 100.0 for n, _, _ in O.LENS_TERMS])
    lens = (rng.standard_normal(12) * scale).astype(np.float32)
    return lens, fy, fx, lam


@pytest.mark.parametrize("device", DEVICES)
def test_general_function_matches_float64(device: str) -> None:
    lens, fy, fx, lam = inputs(np.random.default_rng(1), 300)
    fy[:3], fx[:3] = 0.0, 0.0
    got = run(_general_kernel, lens, fy, fx, lam, device).astype(np.float64)
    ref, bound = reference(lens, fy, fx, lam)
    assert np.all(np.abs(got - ref) <= bound)
    assert np.all(got[:3] == 0.0)


@pytest.mark.parametrize("device", DEVICES)
def test_general_check_detects_permuted_lens(device: str) -> None:
    lens, fy, fx, lam = inputs(np.random.default_rng(2), 100)
    swapped = lens.copy()
    swapped[[1, 2]] = lens[[2, 1]]
    got = run(_general_kernel, swapped, fy, fx, lam, device).astype(np.float64)
    ref, bound = reference(lens, fy, fx, lam)
    assert np.any(np.abs(got - ref) > bound)


@wp.kernel
def _rows_kernel(
    lenses: wp.array2d(dtype=wp.float32),
    fy: wp.array(dtype=wp.float32),
    fx: wp.array(dtype=wp.float32),
    lam: wp.array(dtype=wp.float32),
    general: wp.array(dtype=wp.float32),
    focus: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    general[i] = LW.lens_wavefront_f32(lenses[i], fy[i], fx[i], lam[i])
    focus[i] = LW.lens_wavefront_focus_f32(lenses[i], fy[i], fx[i], lam[i])


def both(lenses: np.ndarray, fy, fx, lam, device: str):
    """Return the general and the single-entry results, one lens per point."""
    n = len(fy)
    args = [wp.array(np.asarray(a, np.float32), device=device) for a in (fy, fx, lam)]
    outs = [wp.zeros(n, dtype=wp.float32, device=device) for _ in range(2)]
    w_lens = wp.array(lenses.astype(np.float32), device=device)
    wp.launch(_rows_kernel, dim=n, inputs=[w_lens, *args, *outs], device=device)
    return outs[0].numpy(), outs[1].numpy()


def focus_cases(rng: np.random.Generator, zeros: str, n: int = 10_000):
    fy = rng.uniform(-3, 3, n).astype(np.float32)
    fx = rng.uniform(-3, 3, n).astype(np.float32)
    fy[:50], fx[:50] = 0.0, 0.0
    lam = rng.uniform(0.005, 0.1, n).astype(np.float32)
    lenses = np.zeros((n, 12), np.float32)
    if zeros == "minus":
        lenses[:, 1:] = -0.0
    elif zeros == "mixed":
        lenses[:, 1:] = np.where(rng.random((n, 11)) < 0.5, 0.0, -0.0)
    lenses[:, 0] = rng.standard_normal(n) * 1e3
    lenses[::97, 0] = 0.0
    lenses[1::97, 0] = -0.0
    return lenses, fy, fx, lam


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("zeros", ["plus", "minus", "mixed"])
def test_focus_function_is_bit_identical(device: str, zeros: str) -> None:
    lenses, fy, fx, lam = focus_cases(np.random.default_rng(3), zeros)
    if zeros == "minus":
        assert np.all(np.signbit(lenses[:, 1:]))
    general, focus = both(lenses, fy, fx, lam, device)
    assert np.array_equal(general.view(np.int32), focus.view(np.int32))


@pytest.mark.parametrize("device", DEVICES)
def test_focus_check_detects_other_entry(device: str) -> None:
    lenses, fy, fx, lam = focus_cases(np.random.default_rng(4), "plus", 500)
    lenses[:, 5] = 1e-3
    general, focus = both(lenses, fy, fx, lam, device)
    assert not np.array_equal(general.view(np.int32), focus.view(np.int32))


def test_lens_to_warp_keeps_order_and_refuses_bad_shape() -> None:
    t = torch.arange(12, dtype=torch.float64)
    assert np.array_equal(LW.lens_to_warp(t).numpy(), np.arange(12, dtype=np.float32))
    assert np.array_equal(LW.lens_to_warp(list(range(12))).numpy(), np.arange(12))
    with pytest.raises(ValueError, match="shape"):
        LW.lens_to_warp(torch.zeros(11))


