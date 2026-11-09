"""Fourier sample frequencies of grids and shell means of spectra.

The module gives the DFT sample frequencies of a 2-D grid and the mean of a
spectrum over shells of constant spatial frequency ``|q|`` (rings in 2-D,
spherical shells in 3-D).

Literature: Briggs and Henson (1995), *The DFT: An Owner's Manual for the
Discrete Fourier Transform* (the DFT frequencies and their order); Bracewell
(2000), *The Fourier Transform and Its Applications* (2-D transforms and
circular symmetry); Frank (2006), *Three-Dimensional Electron Microscopy of
Macromolecular Assemblies* (the rotational average of a power spectrum).
This code was written for scatterem from this literature.

Conventions: the forward kernel is ``exp(-2 pi i k n / N)``, as in
``torch.fft.fftn``. On an axis with ``N`` points and pitch ``d`` the sample
frequencies are ``m / (N d)`` [cycles per length unit], with ``m`` in DFT
order ``0, 1, ..., ceil(N/2) - 1, -floor(N/2), ..., -1``.
"""

from __future__ import annotations

import logging
import math
from fractions import Fraction
from typing import Any, NamedTuple

import numpy as np
import torch

from scatterem.utils._checks import (
    float64_tensor,
    pitches,
    point_counts,
    real_dtype,
)

__all__ = ["ShellMeans", "dual_mesh", "shell_means"]

logger = logging.getLogger(__name__)


def _dft_index(i: int, n: int) -> int:
    """Return the signed DFT frequency index of array index ``i``."""
    return i if i < (n + 1) // 2 else i - n


def _axis_frequencies(n: int, d: float, staggered: bool) -> list[float]:
    """Return the frequencies of one axis, each rounded once to float64."""
    shift = 1 if staggered else 0
    step = Fraction(d)
    return [
        float(Fraction(2 * _dft_index(i, n) + shift, 2 * n) / step) for i in range(n)
    ]


def dual_mesh(
    n_yx: Any,
    pitch: Any,
    *,
    staggered: bool,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the DFT sample frequencies of a 2-D grid, both components.

    The frequencies of a sampled field form the dual (reciprocal) lattice of
    its sampling lattice. Each value is computed in exact rational
    arithmetic and then rounded once to the output dtype.

    Parameters
    ----------
    n_yx : sequence of int
        The number of grid points ``(Ny, Nx)``. Each count is a positive
        integer.
    pitch : float or sequence of float
        The real-space pitch ``(d_y, d_x)`` [length]. One number applies to
        both axes. There is no default: give 1.0 for frequencies in cycles
        per sample.
    staggered : bool
        If True, add half of the frequency step to each axis:
        ``1 / (2 Ny d_y)`` and ``1 / (2 Nx d_x)``. Use it for samples that
        sit midway between the grid points. Give False for the standard DFT
        grid. There is no default.
    dtype : torch.dtype, optional
        ``torch.float32`` or ``torch.float64``. The default is
        ``torch.get_default_dtype()``.
    device : torch.device or str, optional
        The device of the result. The default is the CPU.

    Returns
    -------
    torch.Tensor
        Shape ``(2, Ny, Nx)`` [1/length], in FFT order. Entry ``[0, i, j]``
        is the ``y`` frequency of row ``i`` and entry ``[1, i, j]`` is the
        ``x`` frequency of column ``j``. Entry 0 holds the y frequencies and
        entry 1 the x frequencies. Use
        ``torch.fft.fftshift(out, dim=(-2, -1))`` to centre it.

    Raises
    ------
    TypeError
        If a count is not an integer, a pitch is not a real number, or the
        dtype is not float32 or float64.
    ValueError
        If a count is smaller than 1, the number of pitches is not 2, or a
        pitch is not finite and positive.
    """
    ny, nx = point_counts(n_yx)
    d_y, d_x = pitches(pitch, 2)
    if not isinstance(staggered, bool | np.bool_):
        raise TypeError(f"staggered: expected a bool, got {staggered!r}")
    out_dtype = real_dtype(dtype)
    target = torch.device("cpu") if device is None else torch.device(device)
    q_y = torch.tensor(_axis_frequencies(ny, d_y, bool(staggered)), dtype=torch.float64)
    q_x = torch.tensor(_axis_frequencies(nx, d_x, bool(staggered)), dtype=torch.float64)
    q_y = q_y.to(dtype=out_dtype, device=target)
    q_x = q_x.to(dtype=out_dtype, device=target)
    return torch.stack(
        (q_y[:, None].expand(ny, nx), q_x[None, :].expand(ny, nx)), dim=0
    )


class ShellMeans(NamedTuple):
    """Shell frequencies and shell means of a spectrum.

    Parameters
    ----------
    shell_wavenumber : numpy.ndarray
        Shape ``(K,)``, float64. The frequency of each shell [1/length],
        equally spaced and increasing, starting at 0.
    shell_average : numpy.ndarray
        Shape ``(K,)``, float64. The weighted mean of the samples of each
        shell, in the unit of the input values.
    """

    shell_wavenumber: np.ndarray
    shell_average: np.ndarray


def _shell_layout(shape: tuple[int, ...], steps: tuple[float, ...]) -> tuple:
    """Return the shell spacing [1/length] and the number of shells.

    The spacing is the coarsest frequency step of all axes. Shell ``k`` is
    in the output only if ``(k + 1)`` spacings do not exceed the largest
    positive frequency of every axis. The comparison is exact.
    """
    deltas = [Fraction(1, n) / Fraction(d) for n, d in zip(shape, steps, strict=True)]
    spacing = max(deltas)
    q_lim = min(((n - 1) // 2) * dq for n, dq in zip(shape, deltas, strict=True))
    n_shells = math.floor(q_lim / spacing)
    return spacing, n_shells, q_lim


def shell_means(spectrum_samples: Any, pitch: Any) -> ShellMeans:
    """Return the mean of a spectrum over shells of constant ``|q|``.

    Each sample has the frequency ``|q| = sqrt(q_0^2 + q_1^2 + ...)`` from
    the DFT frequencies of its axes (see ``dual_mesh``). The shells are
    equally spaced in ``|q|`` with the spacing ``D = max_a 1 / (N_a d_a)``,
    the coarsest frequency step. Shell ``k`` is at ``q_k = k D``.

    A sample at ``|q| = x D`` with ``k0 = floor(x)`` and ``t = x - k0`` gives
    the weight ``(1 - t) J`` to shell ``k0`` and ``t J`` to shell ``k0 + 1``
    (linear interpolation). The factor ``J = (D / |q|)^(n - 1)`` for ``n``
    axes cancels the growth of the number of samples with ``|q|``, so the
    effective weight in ``|q|`` is a symmetric tent about ``q_k``. The bias
    for a smooth ``f(|q|)`` is then about ``f''(q_k) D^2 / 12``. The sample
    at ``|q| = 0`` gives the weight 1 to shell 0. The shell mean is
    ``sum w v / sum w``, so it is linear in the values and a constant array
    gives that constant.

    Outer limit: ``q_lim`` is the smallest of the largest positive axis
    frequencies, ``min_a floor((N_a - 1) / 2) / (N_a d_a)``. The output holds
    the shells ``k`` with ``(k + 1) D <= q_lim``: every shell in the output
    has all of its support inside the grid. The largest shell frequency is
    thus in ``[q_lim - 2 D, q_lim - D]``.

    Parameters
    ----------
    spectrum_samples : numpy.ndarray or torch.Tensor
        Real or integer array with two or more axes, in FFT order on every
        axis (zero frequency at the first element), for example
        ``abs(torch.fft.fftn(x)) ** 2``. A tensor can be on any device.
    pitch : float or sequence of float
        The real-space pitch of each axis [length], or one value for all
        axes. There is no default: give 1.0 for frequencies in cycles per
        sample.

    Returns
    -------
    ShellMeans
        ``(shell_wavenumber, shell_average)``: two new NumPy float64 arrays of
        equal length. ``shell_wavenumber`` is
        in [1/length].

    Raises
    ------
    TypeError
        If ``spectrum_samples`` is complex, bool or not an array, or a pitch is not a
        real number.
    ValueError
        If ``spectrum_samples`` has fewer than two axes, an axis has fewer than 3
        points, ``spectrum_samples`` holds NaN or infinite values, the number of
        pitches is wrong, or a pitch is not finite and positive.
    """
    data = float64_tensor(spectrum_samples, "spectrum_samples")
    if data.ndim < 2:
        raise ValueError(
            f"spectrum_samples: expected an array with 2 or more axes, got shape "
            f"{tuple(data.shape)}"
        )
    steps = pitches(pitch, data.ndim)
    shape = tuple(data.shape)
    n_bad = int((~torch.isfinite(data)).sum().item())
    if n_bad:
        raise ValueError(f"spectrum_samples: {n_bad} samples are NaN or infinite")
    spacing, n_shells, q_lim = _shell_layout(shape, steps)
    if n_shells < 1:
        raise ValueError(
            f"spectrum_samples: every axis needs at least 3 points for one shell, got "
            f"shape {shape}"
        )
    dev = data.device
    x_sq = torch.zeros((), dtype=torch.float64, device=dev)
    for axis, (n, d) in enumerate(zip(shape, steps, strict=True)):
        index = torch.arange(n, device=dev, dtype=torch.float64)
        index = torch.where(index < (n + 1) // 2, index, index - n)
        ratio = float(Fraction(1, n) / Fraction(d) / spacing)
        view = [1] * data.ndim
        view[axis] = n
        x_sq = x_sq + ((index * ratio) ** 2).reshape(view)
    x = torch.sqrt(x_sq).reshape(-1)
    flat = data.reshape(-1)
    k0 = torch.floor(x)
    t = x - k0
    k0 = k0.to(torch.int64)
    jac = torch.where(x > 0, x, 1.0) ** (1 - data.ndim)
    keep = k0 < n_shells
    k0, t, jac, flat = k0[keep], t[keep], jac[keep], flat[keep]
    w_lo = (1.0 - t) * jac
    w_hi = t * jac
    size = n_shells + 1
    num = torch.bincount(k0, w_lo * flat, minlength=size)
    num = num + torch.bincount(k0 + 1, w_hi * flat, minlength=size)
    den = torch.bincount(k0, w_lo, minlength=size)
    den = den + torch.bincount(k0 + 1, w_hi, minlength=size)
    mean = (num[:n_shells] / den[:n_shells]).cpu().numpy()
    q = float(spacing) * np.arange(n_shells, dtype=np.float64)
    logger.debug(
        "shell_means: %d shells, spacing %g, q_lim %g",
        n_shells,
        float(spacing),
        float(q_lim),
    )
    return ShellMeans(shell_wavenumber=q, shell_average=mean)
