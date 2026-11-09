"""Radial integration nodes and K tables of the EELS transition potentials.

Private helper of :mod:`.transition_potentials`, written by the scatterem
authors from the literature (Manson 1972; Brown, Ciston and Ophus 2019) and
standard numerical quadrature: composite Simpson nodes on ``[0, R_cut]``, the radial
integral ``I_lambda(K) = int u_c j_lambda(K r) u_b dr`` (``j_0 - 1`` for
``lambda = 0``) and its slope in ``K``, and cubic Hermite lookup from a uniform
``K`` table.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, sqrt

import numpy as np
import torch
from scipy.interpolate import CubicSpline
from scipy.special import spherical_jn
from torch import Tensor

from .radial import _HARTREE_EV, _RYDBERG_EV, AtomicRadialWavefunction

#: Bound-state tail weight ``int_(R_cut)^inf u_b^2 dr`` beyond the radial cut.
#: 1e-14 leaves a relative error of 1.5e-5 in the small O K dipole integral;
#: 1e-20 brings it below 1e-8.
_TAIL = 1e-20
#: Below this argument ``j_0(x) - 1`` uses its power series.
_J0M1_SERIES = 0.1
#: Rows of momentum transfers per numpy block of the radial integral.
_CHUNK = 256


@dataclass(frozen=True)
class _Nodes:
    """Simpson nodes of the radial integral and the weighted product."""

    r: np.ndarray
    p: np.ndarray
    r_cut: float


def _radial_nodes(
    bound: AtomicRadialWavefunction, excited: AtomicRadialWavefunction, kmax: float
) -> _Nodes:
    """Simpson nodes on ``[0, R_cut]`` and ``w u_b u_c`` on them.

    ``kmax`` is the largest momentum transfer in 1/Bohr. For ``l' = l`` the
    continuum is first orthogonalised to the bound orbital.
    """
    rb, ub = bound.r, bound.u
    seg = 0.5 * (ub[1:] ** 2 + ub[:-1] ** 2) * np.diff(rb)
    tail = np.concatenate((np.cumsum(seg[::-1])[::-1], [0.0]))
    r_cut = float(rb[np.flatnonzero(tail <= _TAIL)[0]])
    r_cut = min(r_cut, bound.rmax, excited.rmax)
    k = sqrt(max(excited.energy, 0.0) / _RYDBERG_EV)
    kappa = sqrt(2.0 * abs(bound.energy) / _HARTREE_EV)
    omega = k + float(kmax) + kappa
    h = min(0.1 / omega, r_cut / 2000.0)
    m = ceil(r_cut / (2.0 * h))
    r = np.linspace(0.0, r_cut, 2 * m + 1)
    w = np.full(r.size, 2.0)
    w[1::2] = 4.0
    w[0] = w[-1] = 1.0
    w *= (r[1] - r[0]) / 3.0
    # Not-a-knot ends: a natural spline forces u'' = 0 at r = 0, but an s
    # orbital has u''(0) = -2 Z u'(0); that costs 7e-5 at large K (1s, Z = 8).
    u_b = CubicSpline(bound.r, bound.u, bc_type="not-a-knot")(r)
    u_c = CubicSpline(excited.r, excited.u, bc_type="not-a-knot")(r)
    if excited.l == bound.l:
        # Pauli: the ejected electron cannot occupy the initial orbital. Remove
        # the overlap (Gram-Schmidt, on these nodes), so that the matrix
        # element vanishes at K = 0 and at large K.
        u_c = u_c - (np.dot(w * u_b, u_c) / np.dot(w * u_b, u_b)) * u_b
    return _Nodes(r=r, p=w * u_b * u_c, r_cut=r_cut)


def _k_table(kmax: float, r_cut: float) -> np.ndarray:
    """Uniform momentum table ``0 .. >= kmax`` (1/Bohr), step ``0.05/R_cut``.

    The grid values come from this table by cubic Hermite interpolation of
    ``I_lambda`` and its slope (see :func:`_hermite_np`).
    """
    dk = 0.05 / r_cut
    return dk * np.arange(max(1, ceil(kmax / dk)) + 1, dtype=np.float64)


def _j0m1_series(x2):
    """``j_0(x) - 1`` from ``x^2`` by its power series (``x < 0.1``)."""
    return x2 * (-1.0 / 6.0 + x2 * (1.0 / 120.0 + x2 * (-1.0 / 5040.0 + x2 / 362880.0)))


def _overlaps_np(lams: Sequence[int], nodes: _Nodes, kb: np.ndarray) -> np.ndarray:
    """``I_lambda(K)`` for ``K = kb`` (1/Bohr) with numpy and scipy."""
    kb = np.asarray(kb, dtype=np.float64).ravel()
    out = np.empty((len(lams), kb.size), dtype=np.float64)
    for start in range(0, kb.size, _CHUNK):
        x = kb[start : start + _CHUNK].reshape(-1, 1) * nodes.r.reshape(1, -1)
        for i, lam in enumerate(lams):
            j = spherical_jn(lam, x)
            if lam == 0:
                j = np.where(x < _J0M1_SERIES, _j0m1_series(x * x), j - 1.0)
            out[i, start : start + _CHUNK] = (j * nodes.p).sum(axis=-1)
    return out


def _slopes_np(lams: Sequence[int], nodes: _Nodes, kb: np.ndarray) -> np.ndarray:
    """``d I_lambda / dK`` for ``K = kb`` (1/Bohr): ``int u_c r j_lambda'(K r) u_b dr``.

    The ``- 1`` of the ``lambda = 0`` integrand is constant in ``K``, so it does
    not enter the slope.
    """
    kb = np.asarray(kb, dtype=np.float64).ravel()
    pr = nodes.p * nodes.r
    out = np.empty((len(lams), kb.size), dtype=np.float64)
    for start in range(0, kb.size, _CHUNK):
        x = kb[start : start + _CHUNK].reshape(-1, 1) * nodes.r.reshape(1, -1)
        for i, lam in enumerate(lams):
            dj = spherical_jn(lam, x, derivative=True)
            out[i, start : start + _CHUNK] = (dj * pr).sum(axis=-1)
    return out


def _hermite_np(x: np.ndarray, dk: float, f: np.ndarray, df: np.ndarray) -> np.ndarray:
    """Cubic Hermite interpolation from values and slopes on ``K_j = j dk``.

    The error is at most ``dk^4 max|f^(4)| / 384``. A linear table cannot
    follow the ``K^2`` start of an orthogonalised ``lambda = 0`` integral.
    """
    s = np.asarray(x, dtype=np.float64) / dk
    j = np.clip(np.floor(s), 0, f.size - 2).astype(np.int64)
    t = s - j
    u = 1.0 - t
    return u * u * ((1.0 + 2.0 * t) * f[j] + t * dk * df[j]) + t * t * (
        (3.0 - 2.0 * t) * f[j + 1] - u * dk * df[j + 1]
    )


def _hermite_torch(x: Tensor, dk: float, f: Tensor, df: Tensor) -> Tensor:
    """Torch twin of :func:`_hermite_np` (same formula and operation order)."""
    s = x / dk
    j = torch.floor(s).clamp(0, f.shape[0] - 2).to(torch.long)
    t = s - j.to(s.dtype)
    u = 1.0 - t
    return u * u * ((1.0 + 2.0 * t) * f[j] + t * dk * df[j]) + t * t * (
        (3.0 - 2.0 * t) * f[j + 1] - u * dk * df[j + 1]
    )
