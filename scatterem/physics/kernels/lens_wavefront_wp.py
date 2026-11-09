"""NVIDIA Warp device functions for the lens wavefront of orders 1 to 3.

This module was written for scatterem from Krivanek, Dellby and Lupini
(1999), Ultramicroscopy 78, 1-11; Haider, Uhlemann and Zach (2000),
Ultramicroscopy 81, 163-175; Kirkland (2020), Advanced Computing in Electron
Microscopy, 3rd ed., ch. 3; and the NVIDIA Warp documentation (device
functions, kernels, arrays).

The device functions compute the same wavefront ``W`` [rad] as
:func:`scatterem.physics.optics.lens_wavefront`, for one frequency point,
in float32. They read the 12 coefficients [Angstrom] from a Warp array in the
order of ``scatterem.physics.optics.LENS_LAYOUT``. A kernel of another module
calls them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import warp as wp

from scatterem.physics._lens_basis import TERMS

__all__ = ["lens_to_warp", "lens_wavefront_f32", "lens_wavefront_focus_f32"]

_TWO_PI = wp.constant(wp.float32(2.0 * math.pi))
_HALF = wp.constant(wp.float32(0.5))
_THREE = wp.constant(wp.float32(3.0))
_FOUR = wp.constant(wp.float32(4.0))
_TWO = wp.constant(wp.float32(2.0))
_ZERO = wp.constant(wp.float32(0.0))


@wp.func
def _focus_term(c: wp.float32, u: wp.float32, v: wp.float32) -> wp.float32:
    """Return the slot-0 term ``c (u^2 + v^2) / 2`` [Angstrom rad^2]."""
    return c * ((u * u + v * v) * _HALF)  # pyright: ignore[reportOperatorIssue]


@wp.func
def _term(c: wp.float32, p: wp.float32) -> wp.float32:
    """Return ``c p``, and exact ``+0`` for ``c = +-0`` even if ``p`` is inf."""
    return wp.where(c == _ZERO, _ZERO, c * p)


@wp.func
def lens_wavefront_f32(
    lens: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    fy: wp.float32,
    fx: wp.float32,
    lambda_a: wp.float32,
) -> wp.float32:
    """Return the lens wavefront ``W`` [rad] at one frequency point, in float32.

    Accuracy. For inputs that are exact in float32, the result ``W32`` and the
    float64 value ``W64`` of the same inputs satisfy

        |W32 - W64| <= 32 eps (2 pi / lambda) sum_j |c_j P_j| / (n_j + 1)
                       + 4 eps |W64|,   eps = 2^-23,

    with ``P_j`` the basis polynomial of slot ``j``. Each term takes at most
    about 10 roundings, the sum of 12 terms adds 11 and the scaling adds 2.

    Parameters
    ----------
    lens : wp.array of wp.float32
        The 12 coefficients [Angstrom] in the order of ``LENS_LAYOUT``.
    fy, fx : wp.float32
        Frequency coordinates ``k_y`` and ``k_x`` [1/Angstrom].
    lambda_a : wp.float32
        Wavelength [Angstrom], larger than zero.

    Returns
    -------
    wp.float32
        Wavefront [rad].
    """
    u = lambda_a * fx
    v = lambda_a * fy
    t0 = _focus_term(lens[0], u, v)
    r2 = u * u + v * v  # pyright: ignore[reportOperatorIssue]
    zr2 = u * u - v * v  # pyright: ignore[reportOperatorIssue]
    zi2 = _TWO * (u * v)  # pyright: ignore[reportOperatorIssue]
    zr3 = zr2 * u - zi2 * v
    zi3 = zr2 * v + zi2 * u
    zr4 = zr2 * zr2 - zi2 * zi2
    zi4 = _TWO * (zr2 * zi2)
    rest = _term(lens[1], zr2 * _HALF)
    rest = rest + _term(lens[2], zi2 * _HALF)
    rest = rest + _term(lens[3], (r2 * u) / _THREE)
    rest = rest + _term(lens[4], (r2 * v) / _THREE)
    rest = rest + _term(lens[5], zr3 / _THREE)
    rest = rest + _term(lens[6], zi3 / _THREE)
    rest = rest + _term(lens[7], (r2 * r2) / _FOUR)
    rest = rest + _term(lens[8], (r2 * zr2) / _FOUR)
    rest = rest + _term(lens[9], (r2 * zi2) / _FOUR)
    rest = rest + _term(lens[10], zr4 / _FOUR)
    rest = rest + _term(lens[11], zi4 / _FOUR)
    s = (t0 + rest) + _ZERO
    return s * (_TWO_PI / lambda_a)


@wp.func
def lens_wavefront_focus_f32(
    lens: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    fy: wp.float32,
    fx: wp.float32,
    lambda_a: wp.float32,
) -> wp.float32:
    """Return ``W`` [rad] for a lens vector whose sole non-zero slot is 0.

    The function reads only ``lens[0]`` (the defocus slot ``C10``). If the
    other 11 slots are ``+0.0`` or ``-0.0`` and all inputs are finite, it
    returns the same float32 bits as :func:`lens_wavefront_f32`. This needs
    the default Warp setting without fast-math; fast-math would void it.

    Rule: the host picks ``lens_wavefront_focus_f32`` or
    ``lens_wavefront_f32`` before the launch (for example as a kernel template
    argument or by building one of two kernels). Do not choose between them
    inside a kernel with a run-time condition.

    Parameters
    ----------
    lens : wp.array of wp.float32
        The 12 coefficients [Angstrom] in the order of ``LENS_LAYOUT``.
    fy, fx : wp.float32
        Frequency coordinates ``k_y`` and ``k_x`` [1/Angstrom].
    lambda_a : wp.float32
        Wavelength [Angstrom], larger than zero.

    Returns
    -------
    wp.float32
        Wavefront [rad].
    """
    u = lambda_a * fx
    v = lambda_a * fy
    t0 = _focus_term(lens[0], u, v)
    s = (t0 + _ZERO) + _ZERO
    return s * (_TWO_PI / lambda_a)


def lens_to_warp(
    lens: torch.Tensor | Sequence[float],
    *,
    device: str | torch.device | None = None,
) -> wp.array:
    """Return a lens vector as a 1-D float32 Warp array in the same order.

    A float32 contiguous tensor on the target device is shared without a
    copy. Other inputs are cast to float32 and copied first.

    Parameters
    ----------
    lens : torch.Tensor or sequence of float
        Lens vector of shape ``(12,)`` [Angstrom].
    device : str or torch.device, optional
        Target device. ``None`` means the device of the tensor, or CPU for a
        sequence.

    Returns
    -------
    wp.array
        Shape ``(12,)``, dtype ``wp.float32``.

    Raises
    ------
    ValueError
        If the shape is not ``(12,)``.
    """
    wp.init()
    if isinstance(lens, torch.Tensor):
        t = lens.detach()
    else:
        t = torch.as_tensor(lens, dtype=torch.float32)
    if tuple(t.shape) != (len(TERMS),):
        raise ValueError(f"lens must have shape (12,), got {tuple(t.shape)}")
    if device is not None:
        t = t.to(torch.device(device))
    t = t.to(torch.float32).contiguous()
    return wp.from_torch(t, dtype=wp.float32)
