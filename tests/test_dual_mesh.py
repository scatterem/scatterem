"""Tests of the DFT sample frequencies of a 2-D grid (``dual_mesh``).

Oracles: exact rational frequencies from the contract rule, the DFT of a pure
plane wave, the scaling law and the layout rule.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
import torch

from scatterem.utils.spectral_mesh import dual_mesh


def _contract_index(i: int, n: int) -> int:
    """Signed index m: 0..ceil(N/2)-1, then -floor(N/2)..-1."""
    positive = math.ceil(n / 2)
    return i if i < positive else i - n


def _exact_axis(n: int, d: float, half: bool, nyquist_positive: bool = False):
    out = []
    for i in range(n):
        m = _contract_index(i, n)
        if nyquist_positive and n % 2 == 0 and m == -n // 2:
            m = n // 2
        value = Fraction(m, n) / Fraction(d)
        if half:
            value += Fraction(1, 2 * n) / Fraction(d)
        out.append(float(value))
    return np.array(out)


def _max_ulp_error(got: np.ndarray, ref: np.ndarray) -> float:
    spacing = np.spacing(np.maximum(np.abs(ref), np.finfo(np.float64).tiny))
    return float(np.max(np.abs(got - ref) / spacing))


@pytest.mark.parametrize("n_yx", [(7, 10), (8, 9), (1, 5), (2, 3)])
@pytest.mark.parametrize("half", [False, True])
def test_values_equal_exact_rational_frequencies(n_yx, half):
    pitch = (0.37, 1.9)
    grid = dual_mesh(n_yx, pitch, staggered=half, dtype=torch.float64).numpy()
    ref_y = _exact_axis(n_yx[0], pitch[0], half)
    ref_x = _exact_axis(n_yx[1], pitch[1], half)
    assert grid.shape == (2, *n_yx)
    assert _max_ulp_error(grid[0], np.broadcast_to(ref_y[:, None], n_yx)) <= 1
    assert _max_ulp_error(grid[1], np.broadcast_to(ref_x[None, :], n_yx)) <= 1


def test_exact_value_check_detects_positive_nyquist_term():
    """Control: a reference with a positive Nyquist term fails the check."""
    grid = dual_mesh((8, 9), (0.37, 1.9), dtype=torch.float64, staggered=False).numpy()
    wrong = _exact_axis(8, 0.37, False, nyquist_positive=True)
    assert _max_ulp_error(grid[0][:, 0], wrong) > 1e6


def _layout_ok(grid: torch.Tensor) -> bool:
    comp_y, comp_x = grid
    if not (torch.equal(comp_y, comp_y[:, :1].expand_as(comp_y))):
        return False
    if not (torch.equal(comp_x, comp_x[:1, :].expand_as(comp_x))):
        return False
    shifted = torch.fft.fftshift(grid, dim=(-2, -1))
    return bool(
        (torch.diff(shifted[0][:, 0]) > 0).all()
        and (torch.diff(shifted[1][0, :]) > 0).all()
    )


@pytest.mark.parametrize("half", [False, True])
def test_layout_rows_and_columns(half):
    grid = dual_mesh((6, 9), (1.3, 0.4), staggered=half, dtype=torch.float64)
    assert _layout_ok(grid)


def test_layout_check_detects_exchanged_components():
    """Control: exchanged components fail on a non-square grid."""
    grid = dual_mesh((6, 9), 1.0, dtype=torch.float64, staggered=False)
    swapped = grid.flip(0)
    assert torch.equal(swapped[0], grid[1])
    assert not _layout_ok(swapped)


def _plane_wave_peak(n_yx, pitch, entry, grid):
    qy = float(grid[0][entry])
    qx = float(grid[1][entry])
    y = torch.arange(n_yx[0], dtype=torch.float64)[:, None] * pitch[0]
    x = torch.arange(n_yx[1], dtype=torch.float64)[None, :] * pitch[1]
    wave = torch.exp(2j * math.pi * (qy * y + qx * x))
    spectrum = torch.fft.fft2(wave).abs()
    flat = int(spectrum.argmax())
    peak = divmod(flat, n_yx[1])
    rest = spectrum.clone()
    rest[peak] = 0
    return peak, float(rest.max() / spectrum[peak])


def test_plane_wave_peaks_at_its_grid_entry():
    n_yx, pitch = (12, 15), (0.7, 1.6)
    grid = dual_mesh(n_yx, pitch, dtype=torch.float64, staggered=False)
    rng = np.random.default_rng(11)
    for _ in range(5):
        entry = (int(rng.integers(n_yx[0])), int(rng.integers(n_yx[1])))
        peak, leak = _plane_wave_peak(n_yx, pitch, entry, grid)
        assert peak == entry
        assert leak < 1e-9


def test_plane_wave_check_detects_neighbouring_frequency():
    """Control: the frequency of a neighbouring entry does not peak at (3, 5)."""
    n_yx, pitch = (12, 15), (0.7, 1.6)
    grid = dual_mesh(n_yx, pitch, dtype=torch.float64, staggered=False)
    peak, _ = _plane_wave_peak(n_yx, pitch, (3, 4), grid)
    assert peak != (3, 5)


def test_doubling_the_pitch_halves_the_frequencies():
    base = dual_mesh((9, 12), (0.3, 0.8), staggered=True, dtype=torch.float64)
    double = dual_mesh((9, 12), (0.6, 1.6), staggered=True, dtype=torch.float64)
    assert torch.equal(double, base / 2)
    triple = dual_mesh((9, 12), (0.9, 2.4), staggered=True, dtype=torch.float64)
    assert not torch.equal(triple, base / 2)  # control


def test_scalar_pitch_applies_to_both_axes():
    one = dual_mesh((5, 6), 0.25, dtype=torch.float64, staggered=False)
    two = dual_mesh((5, 6), [0.25, 0.25], dtype=torch.float64, staggered=False)
    three = dual_mesh(
        (5, 6), torch.tensor([0.25, 0.25]), dtype=torch.float64, staggered=False
    )
    four = dual_mesh(
        (5, 6), np.array([0.25, 0.25]), dtype=torch.float64, staggered=False
    )
    assert torch.equal(one, two) and torch.equal(one, three)
    assert torch.equal(one, four)


def test_dtype_and_default_dtype():
    grid32 = dual_mesh((7, 8), (0.37, 1.9), dtype=torch.float32, staggered=False)
    grid64 = dual_mesh((7, 8), (0.37, 1.9), dtype=torch.float64, staggered=False)
    assert grid32.dtype == torch.float32 and grid64.dtype == torch.float64
    ref = grid64.to(torch.float32).numpy()
    spacing = np.spacing(np.maximum(np.abs(ref), np.float32(1e-30)))
    assert np.all(np.abs(grid32.numpy() - ref) <= spacing)
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        assert dual_mesh((3, 4), 1.0, staggered=False).dtype == torch.float64
        torch.set_default_dtype(torch.float32)
        assert dual_mesh((3, 4), 1.0, staggered=False).dtype == torch.float32
    finally:
        torch.set_default_dtype(old)
    assert dual_mesh((3, 4), 1.0, staggered=False).device.type == "cpu"


def test_cuda_equals_cpu(require_cuda):
    cpu = dual_mesh((7, 8), (0.37, 1.9), staggered=True, dtype=torch.float64)
    gpu = dual_mesh(
        (7, 8), (0.37, 1.9), staggered=True, dtype=torch.float64, device=require_cuda
    )
    assert gpu.device.type == "cuda"
    assert torch.equal(gpu.cpu(), cpu)


@pytest.mark.parametrize("pitch", [0, -1.0, float("nan"), float("inf"), (1, 2, 3)])
def test_bad_pitch_raises_value_error(pitch):
    with pytest.raises(ValueError, match="pitch"):
        dual_mesh((4, 4), pitch, staggered=False)


@pytest.mark.parametrize("pitch", ["1", True, 1j, None, (1.0, "a")])
def test_bad_pitch_type_raises_type_error(pitch):
    with pytest.raises(TypeError, match="pitch"):
        dual_mesh((4, 4), pitch, staggered=False)


@pytest.mark.parametrize("n_yx", [(0, 4), (-2, 4), (4, 4, 4), (4,)])
def test_bad_counts_raise_value_error(n_yx):
    with pytest.raises(ValueError, match="n_yx"):
        dual_mesh(n_yx, 1.0, staggered=False)


@pytest.mark.parametrize("n_yx", [(4.0, 4), (4.5, 4), (True, 4), "44", 4])
def test_bad_count_types_raise_type_error(n_yx):
    with pytest.raises(TypeError, match="n_yx"):
        dual_mesh(n_yx, 1.0, staggered=False)


@pytest.mark.parametrize("dtype", [torch.int32, torch.complex64, torch.float16])
def test_bad_dtype_raises_type_error(dtype):
    with pytest.raises(TypeError, match="dtype"):
        dual_mesh((4, 4), 1.0, dtype=dtype, staggered=False)


def test_valid_call_does_not_raise():
    """Control for the refusal tests."""
    out = dual_mesh(
        np.array([4, 5]), np.float32(0.5), dtype=torch.float64, staggered=False
    )
    assert out.shape == (2, 4, 5)


def test_pitch_and_staggered_have_no_default():
    """The caller must give the pitch and the half-step switch."""
    with pytest.raises(TypeError, match="pitch"):
        dual_mesh((4, 4), staggered=False)
    with pytest.raises(TypeError, match="staggered"):
        dual_mesh((4, 4), 1.0)
