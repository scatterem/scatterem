"""Tests of the shell means of a spectrum (``shell_means``).

Oracles: exact properties of a weighted mean (constants, linearity), analytic
isotropic functions (a Gaussian and ``|q|^2``), a second grid with equal
pitches, and the outer-limit rule computed from the contract frequency list.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
import torch

from scatterem.utils.spectral_mesh import shell_means


def _axis_freq(n: int, d: float) -> np.ndarray:
    m = np.arange(n)
    m = np.where(m < math.ceil(n / 2), m, m - n)
    return m / (n * d)


def _abs_q(shape, pitch) -> np.ndarray:
    q2 = np.zeros(shape)
    for axis, (n, d) in enumerate(zip(shape, pitch, strict=True)):
        view = [1] * len(shape)
        view[axis] = n
        q2 = q2 + (_axis_freq(n, d) ** 2).reshape(view)
    return np.sqrt(q2)


def _q_lim(shape, pitch) -> float:
    """Smallest of the largest positive axis frequencies."""
    return min(_axis_freq(n, d).max() for n, d in zip(shape, pitch, strict=True))


def _spacing(shape, pitch) -> float:
    return max(1.0 / (n * d) for n, d in zip(shape, pitch, strict=True))


@pytest.mark.parametrize(
    ("shape", "pitch"), [((33, 40), (1.0, 0.6)), ((12, 15, 18), (0.5, 0.7, 0.9))]
)
def test_constant_array_gives_constant_profile(shape, pitch):
    values = np.full(shape, 3.7)
    result = shell_means(values, pitch)
    assert np.allclose(result.shell_average, 3.7, rtol=1e-13, atol=0)
    changed = values.copy()
    changed.flat[5] = 10.0  # control: one changed sample shows
    assert np.ptp(shell_means(changed, pitch).shell_average) > 1e-3


def test_profile_is_linear():
    rng = np.random.default_rng(5)
    f = rng.standard_normal((40, 36))
    g = rng.standard_normal((40, 36))
    pitch = (0.8, 1.1)
    left = shell_means(2.5 * f - 0.75 * g, pitch).shell_average
    right = (
        2.5 * shell_means(f, pitch).shell_average
        - 0.75 * shell_means(g, pitch).shell_average
    )
    scale = np.abs(left).max()
    assert np.max(np.abs(left - right)) <= 1e-12 * scale
    wrong = (
        2.5 * shell_means(f**2, pitch).shell_average
        - 0.75 * shell_means(g, pitch).shell_average
    )
    assert np.max(np.abs(left - wrong)) > 1e-3 * scale  # control


def _gauss_error(n: int, width_factor: float = 1.0) -> float:
    shape, pitch = (n, n), (1.0, 1.0)
    q0 = 0.3 * _q_lim(shape, pitch)
    values = np.exp(-((_abs_q(shape, pitch) / (width_factor * q0)) ** 2))
    q, mean = shell_means(values, pitch)
    return float(np.max(np.abs(mean - np.exp(-((q / q0) ** 2)))[3:]))


@pytest.mark.parametrize(("n", "bound"), [(64, 5e-3), (256, 1e-3)])
def test_isotropic_gaussian_meets_accuracy_goal(n, bound):
    assert _gauss_error(n) <= bound
    assert _gauss_error(n, 1.05) > bound  # control: 5 % wider Gaussian


@pytest.mark.parametrize(
    ("shape", "pitch"), [((64, 96), (1.0, 0.7)), ((256, 256), (1.0, 1.0))]
)
def test_q_squared_matches_annulus_mean(shape, pitch):
    """``f = |q|^2``: bracket and closeness to the exact annulus mean.

    Every sample that adds to shell ``k`` lies in ``((k-1) D, (k+1) D)``, so
    the mean lies in ``[((k-1) D)^2, ((k+1) D)^2]``. The continuum value of
    a symmetric tent is ``(k^2 + 1/6) D^2``; the annulus
    ``[(k - 1/2) D, (k + 1/2) D]`` has the mean ``(k^2 + 1/4) D^2``. The two
    differ by ``D^2 / 12``. The lattice adds at most ``2 k D |m1|`` with a
    first moment ``|m1| <= D / (4 k)``, that is ``D^2 / 2``. The bound
    ``D^2`` covers the sum with margin.
    """
    q_abs = _abs_q(shape, pitch)
    q, mean = shell_means(q_abs**2, pitch)
    d = _spacing(shape, pitch)
    k = np.arange(len(q))
    assert np.all(mean[1:] >= ((k[1:] - 1) * d) ** 2 * (1 - 1e-12))
    assert np.all(mean[1:] <= ((k[1:] + 1) * d) ** 2 * (1 + 1e-12))
    annulus = (k**2 + 0.25) * d**2
    assert np.all(np.abs(mean - annulus)[3:] <= d**2)
    wrong = ((k + 1) * d) ** 2  # control
    assert np.any(np.abs(mean - wrong)[3:] > d**2)


def test_rings_are_in_physical_frequency():
    shape, pitch = (64, 96), (1.0, 0.7)
    q0 = 0.3 * _q_lim((64, 64), (1.0, 1.0))
    values = np.exp(-((_abs_q(shape, pitch) / q0) ** 2))
    q, mean = shell_means(values, pitch)
    assert np.max(np.abs(mean - np.exp(-((q / q0) ** 2)))[3:]) <= 5e-3
    q_ref, mean_ref = shell_means(
        np.exp(-((_abs_q((64, 64), (1.0, 1.0)) / q0) ** 2)), 1.0
    )
    n = min(len(q), len(q_ref))
    assert np.allclose(q[:n], q_ref[:n], rtol=1e-12)
    assert np.max(np.abs(mean[:n] - mean_ref[:n])[3:]) <= 5e-3
    # control: isotropic in index distance, not in physical frequency
    iy = np.fft.fftfreq(64) * 64
    ix = np.fft.fftfreq(96) * 96
    index_r = np.hypot(iy[:, None], ix[None, :])
    wrong = np.exp(-((index_r / (q0 * 64)) ** 2))
    q_w, mean_w = shell_means(wrong, pitch)
    assert np.max(np.abs(mean_w - np.exp(-((q_w / q0) ** 2)))[3:]) > 5e-3


@pytest.mark.parametrize(
    ("shape", "pitch"),
    [((64, 96), (1.0, 0.7)), ((65, 40), (0.5, 1.3)), ((31, 32, 33), (1, 1, 1))],
)
def test_outer_limit_rule(shape, pitch):
    q, _ = shell_means(np.ones(shape), pitch)
    d = _spacing(shape, pitch)
    q_lim = _q_lim(shape, pitch)
    assert np.allclose(np.diff(q), d, rtol=1e-12)
    assert q[0] == 0.0
    assert q[-1] + d <= q_lim * (1 + 1e-12)
    assert q_lim < q[-1] + 2 * d
    extra = q[-1] + d  # control: one more shell breaks the rule
    assert not extra + d <= q_lim * (1 + 1e-12)


def test_numpy_and_tensor_agree_and_input_is_unchanged():
    rng = np.random.default_rng(8)
    values = rng.random((30, 34))
    keep = values.copy()
    a = shell_means(values, (1.0, 0.9))
    tensor = torch.from_numpy(values.copy())
    b = shell_means(tensor, (1.0, 0.9))
    assert np.array_equal(a.shell_wavenumber, b.shell_wavenumber) and np.array_equal(
        a.shell_average, b.shell_average
    )
    c = shell_means(tensor.to(torch.float32), (1.0, 0.9))
    assert np.allclose(c.shell_average, a.shell_average, rtol=1e-6)
    assert np.array_equal(values, keep)
    assert (
        a.shell_average.dtype == np.float64 and a.shell_wavenumber.dtype == np.float64
    )
    integers = shell_means(np.arange(30 * 34).reshape(30, 34), 1.0)
    floats = shell_means(np.arange(30 * 34).reshape(30, 34).astype(float), 1.0)
    assert np.array_equal(integers.shell_average, floats.shell_average)


def test_cuda_equals_cpu(require_cuda):
    rng = np.random.default_rng(9)
    values = rng.random((40, 50))
    cpu = shell_means(values, (0.5, 0.6))
    gpu = shell_means(torch.from_numpy(values).to(require_cuda), (0.5, 0.6))
    assert np.allclose(gpu.shell_average, cpu.shell_average, rtol=1e-12, atol=0)


@pytest.mark.parametrize(
    ("values", "pitch"),
    [
        (np.ones(10), 1.0),
        (np.ones((8, 8)), (1.0, 1.0, 1.0)),
        (np.ones((8, 8)), 0),
        (np.ones((8, 8)), -1.0),
        (np.ones((8, 8)), float("nan")),
        (np.ones((8, 2)), 1.0),
        (np.array([[1.0, np.nan, 1.0]] * 4), 1.0),
    ],
)
def test_bad_values_raise_value_error(values, pitch):
    with pytest.raises(ValueError):
        shell_means(values, pitch)


@pytest.mark.parametrize(
    "values",
    [
        np.ones((8, 8), complex),
        torch.ones(8, 8, dtype=torch.complex64),
        np.ones((8, 8), bool),
        [[1.0] * 8] * 8,
    ],
)
def test_bad_types_raise_type_error(values):
    with pytest.raises(TypeError, match="spectrum_samples"):
        shell_means(values, 1.0)


def test_valid_call_does_not_raise():
    """Control for the refusal tests."""
    assert len(shell_means(np.ones((8, 8)), 1.0).shell_wavenumber) == 3


@pytest.mark.slow
def test_cost_of_a_2048_array():
    values = np.random.default_rng(1).random((2048, 2048))
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        best = math.inf
        for _ in range(3):
            start = time.perf_counter()
            shell_means(values, 1.0)
            best = min(best, time.perf_counter() - start)
    finally:
        torch.set_num_threads(threads)
    assert best < 1.0


def test_pitch_has_no_default():
    """The caller must give the pitch."""
    with pytest.raises(TypeError, match="pitch"):
        shell_means(np.ones((8, 8)))
