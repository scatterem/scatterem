"""Torch special functions of the EELS transition potentials (float64).

Private helper of :mod:`.transition_potentials`, written by the scatterem
authors from the standard series, closed forms and recurrences of the
spherical Bessel functions, the associated Legendre functions and the
spherical harmonics (Condon-Shortley phase).
"""

from __future__ import annotations

from math import factorial, pi, sqrt

import torch
from torch import Tensor

__all__ = [
    "assoc_legendre_torch",
    "interp_torch",
    "spherical_bessel_torch",
    "spherical_harmonic_torch",
    "ylm_norm",
]


def spherical_bessel_torch(
    lmax: int, x: Tensor, x_thr: float = 2.0, n_series: int = 12
) -> Tensor:
    """Spherical Bessel functions ``j_0 .. j_lmax`` of a float64 tensor.

    Level ``l`` uses its power series with ``n_series + 2 l`` terms where
    ``x < max(x_thr, l)``, and the closed forms of ``j_0``, ``j_1`` with the
    upward recurrence ``j_(l+1) = (2l+1)/x j_l - j_(l-1)`` elsewhere (stable
    for ``x >= l``). Level ``l`` does not depend on ``lmax``.

    Parameters
    ----------
    lmax : int
        Highest order.
    x : Tensor
        Arguments, ``x >= 0``.
    x_thr : float, optional
        Lowest argument at which the recurrence is used.
    n_series : int, optional
        Number of series terms of ``j_0``.

    Returns
    -------
    Tensor
        Shape ``(lmax + 1, *x.shape)``, float64.
    """
    x = x.to(torch.float64)
    out = torch.empty((lmax + 1, *x.shape), dtype=torch.float64, device=x.device)
    lowest = max(float(x_thr), 1e-300)
    xr = torch.where(x >= lowest, x, torch.full_like(x, max(float(x_thr), 1.0)))
    sin, cos = torch.sin(xr), torch.cos(xr)
    rec_prev = sin / xr
    rec = sin / (xr * xr) - cos / xr
    for ell in range(lmax + 1):
        if ell == 0:
            level = rec_prev
        elif ell == 1:
            level = rec
        else:
            rec_prev, rec = rec, (2 * ell - 1) / xr * rec - rec_prev
            level = rec
        use_series = x < max(float(x_thr), float(ell))
        if bool(use_series.any()):
            xs = torch.where(use_series, x, torch.zeros_like(x))
            half = -0.5 * xs * xs
            term = torch.ones_like(xs)
            total = torch.ones_like(xs)
            for k in range(1, n_series + 2 * ell):
                term = term * half / (k * (2 * ell + 2 * k + 1))
                total = total + term
            dfact = 1.0
            for odd in range(3, 2 * ell + 2, 2):
                dfact *= odd
            level = torch.where(use_series, xs**ell / dfact * total, level)
        out[ell] = level
    return out


def assoc_legendre_torch(ell: int, am: int, ct: Tensor, st: Tensor) -> Tensor:
    """Associated Legendre ``P_ell^am(cos theta)`` with the Condon-Shortley phase.

    Starts from ``P_m^m = (-1)^m (2m-1)!! sin^m theta`` and steps in ``ell``
    with ``(l-m) P_l^m = (2l-1) cos theta P_(l-1)^m - (l+m-1) P_(l-2)^m``.

    Parameters
    ----------
    ell : int
        Degree.
    am : int
        Order, ``0 <= am``.
    ct, st : Tensor
        ``cos theta`` and ``sin theta``.

    Returns
    -------
    Tensor
        ``P_ell^am``, zero when ``am > ell``.
    """
    if am > ell:
        return torch.zeros_like(ct)
    dfact = 1.0
    for odd in range(1, 2 * am, 2):
        dfact *= odd
    p_prev = (-1.0) ** am * dfact * st**am
    if ell == am:
        return p_prev
    p = (2 * am + 1) * ct * p_prev
    for deg in range(am + 2, ell + 1):
        p_prev, p = p, ((2 * deg - 1) * ct * p - (deg + am - 1) * p_prev) / (deg - am)
    return p


def ylm_norm(ell: int, am: int) -> float:
    """Normalisation ``sqrt((2l+1)/(4 pi) (l-m)!/(l+m)!)`` of ``Y_l^m``.

    Parameters
    ----------
    ell, am : int
        Degree and order, ``0 <= am <= ell``.

    Returns
    -------
    float
        The factor.
    """
    return sqrt((2 * ell + 1) / (4.0 * pi) * factorial(ell - am) / factorial(ell + am))


def spherical_harmonic_torch(
    m: int, ell: int, phi: Tensor, theta: Tensor, cache: dict | None = None
) -> Tensor:
    """Complex spherical harmonic ``Y_ell^m(theta, phi)`` (Condon-Shortley).

    Equal to ``scipy.special.sph_harm_y(ell, m, theta, phi)``.

    Parameters
    ----------
    m, ell : int
        Order and degree.
    phi : Tensor
        Azimuth.
    theta : Tensor
        Polar angle.
    cache : dict, optional
        Value store shared by calls with the same ``phi`` and ``theta``. It
        holds ``cos theta``, ``sin theta``, ``i phi`` and ``exp(i |m| phi)``;
        the result is bit-identical to ``cache=None``.

    Returns
    -------
    Tensor
        complex128, with the shape of ``phi``.
    """
    if cache is None:
        cache = {}
    if "ct" not in cache:
        theta = theta.to(torch.float64)
        cache["ct"], cache["st"] = torch.cos(theta), torch.sin(theta)
        cache["phi_c"] = 1j * phi.to(torch.float64).to(torch.complex128)
    am = abs(int(m))
    key = ("e", am)
    if key not in cache:
        cache[key] = torch.exp(am * cache["phi_c"])
    y = ylm_norm(ell, am) * assoc_legendre_torch(ell, am, cache["ct"], cache["st"])
    y = y * cache[key]
    if m < 0:
        y = (-1.0) ** am * torch.conj(y)
    return y


def interp_torch(xq: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    """Linear interpolation with ``numpy.interp`` semantics on a uniform grid.

    Parameters
    ----------
    xq : Tensor
        Query points, any shape.
    xp : Tensor
        Uniform, ascending sample points, shape ``(N,)``, ``N >= 2``.
    fp : Tensor
        Samples, shape ``(N,)``, real or complex.

    Returns
    -------
    Tensor
        Interpolated values with the shape of ``xq`` and the dtype of ``fp``;
        clamped to ``fp[0]`` and ``fp[-1]`` outside ``xp``.
    """
    n = xp.shape[0]
    x0 = xp[0]
    dx = (xp[-1] - x0) / (n - 1)
    idx = torch.floor((xq - x0) / dx).clamp(0, n - 2).to(torch.long)
    t = ((xq - xp[idx]) / dx).clamp(0.0, 1.0).to(xp.dtype)
    return (1.0 - t) * fp[idx] + t * fp[idx + 1]
