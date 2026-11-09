"""Numerov integration of a continuum radial state in a central potential.

Private helper of :mod:`.radial`, written by the scatterem authors from the
radial Schrödinger equation, the Numerov method and the per-Rydberg energy
normalisation of Manson, Phys. Rev. A 6, 1013 (1972). Hartree atomic units:
``r`` in Bohr, energies in Hartree, the potential given as ``r v(r)`` in
Ha Bohr.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import spherical_jn, spherical_yn

__all__ = [
    "grid_integral",
    "neutral_radius",
    "numerov",
    "positive_start",
    "solve_continuum",
]

#: |y| above which the Numerov solution is rescaled (closed channels).
_RESCALE = 1e150
#: Matching: ``|s r v|`` must stay below this (times max(1, k)) beyond R_match.
_MATCH_TOL = 1e-8
#: An orbital whose weight beyond ``r`` is below this counts as enclosed. With
#: 1e-6 the cut lies before the low-density region where a semi-local
#: exchange-correlation potential is numerical noise, and the phase shifts of
#: GPAW atoms agree to 3e-3 rad between radial grids of 150 and 600 points
#: per node (C, O, Ti, Sr; 5-100 eV).
_ENCLOSED_TAIL = 1e-6
#: No matching radius is smaller than this (Bohr).
_MIN_MATCH = 10.0


def numerov(f: np.ndarray, u0: float, u1: float, dx: float) -> np.ndarray:
    """Solve ``y'' = f y`` on a uniform grid with the Numerov method.

    The recurrence, from the Taylor series of ``y`` to sixth order, is
    ``(1 - h^2 f_(i+1)/12) y_(i+1) = 2 (1 + 5 h^2 f_i/12) y_i
    - (1 - h^2 f_(i-1)/12) y_(i-1)``. The global error is ``O(h^4)``. If
    ``|y|`` grows beyond 1e150, the whole solution so far is scaled down; the
    result is then correct up to one overall factor.

    Parameters
    ----------
    f : ndarray
        Shape ``(N,)``. Coefficient ``f(x_i)`` on the grid.
    u0, u1 : float
        The first two values ``y_0``, ``y_1``.
    dx : float
        Grid step.

    Returns
    -------
    ndarray
        Shape ``(N,)``. The solution ``y_i``, float64.
    """
    f = np.asarray(f, dtype=np.float64)
    # d = h^2 f / 12 is small; adding it separately avoids the cancellation of
    # the form (12 - 10 (1 - d)).
    d = (dx * dx / 12.0) * f
    y = np.empty_like(f)
    y[0] = u0
    if f.size > 1:
        y[1] = u1
    for i in range(1, f.size - 1):
        step = (2.0 * y[i] - y[i - 1]) + (10.0 * d[i] * y[i] + d[i - 1] * y[i - 1])
        y[i + 1] = step / (1.0 - d[i + 1])
        if abs(y[i + 1]) > _RESCALE:
            y[: i + 2] /= _RESCALE
    return y


def grid_integral(r: np.ndarray, f: np.ndarray) -> float:
    """Integrate ``f(r)`` over a smooth, possibly non-uniform radial grid.

    The grid is taken as a smooth map ``r(g)`` of the sample index ``g``. The
    integral ``int f dr = int f (dr/dg) dg`` is then a trapezoid sum in ``g``,
    with ``dr/dg`` from a cubic spline of ``r(g)``. For an integrand that
    vanishes with its derivatives at both ends (a bound orbital), this sum
    converges much faster than the trapezoid rule in ``r``.

    Parameters
    ----------
    r : ndarray
        Shape ``(N,)``. Ascending radial grid.
    f : ndarray
        Shape ``(N,)``. Integrand on ``r``.

    Returns
    -------
    float
        The integral.
    """
    g = np.arange(r.size, dtype=np.float64)
    drdg = CubicSpline(g, r).derivative()(g)
    return float(np.trapezoid(f * drdg, g))


def neutral_radius(r: np.ndarray, states) -> float | None:
    """Smallest radius beyond which every bound orbital has negligible weight.

    Beyond this radius the atom's charge is enclosed. By Gauss's law the
    electrostatic potential of the neutral atom is then zero. What is left is
    the exchange-correlation potential of a vanishing density, which a
    semi-local functional gives only to within numerical noise (about 1e-3 Ha,
    grid dependent) and which ends in a constant floor.

    Parameters
    ----------
    r : ndarray
        Shape ``(N,)``. Radial grid of the orbitals, Bohr.
    states : sequence
        ``(n, l, eigenvalue, u)`` tuples, ``u`` on ``r``.

    Returns
    -------
    float or None
        The smallest grid radius ``>= 10`` Bohr at which every
        ``int_r^inf u^2 dr`` is at most ``_ENCLOSED_TAIL`` (1e-6), or
        ``None`` if there are no states or no such radius.
    """
    if not states:
        return None
    dr = np.diff(r)
    worst = np.zeros(r.size)
    for state in states:
        u2 = np.asarray(state[3], dtype=np.float64) ** 2
        piece = 0.5 * (u2[1:] + u2[:-1]) * dr
        tail = np.concatenate((np.cumsum(piece[::-1])[::-1], [0.0]))
        worst = np.maximum(worst, tail)
    idx = np.flatnonzero((worst <= _ENCLOSED_TAIL) & (r >= _MIN_MATCH))
    return float(r[idx[0]]) if idx.size else None


def positive_start(u: np.ndarray) -> np.ndarray:
    """Flip the sign of ``u`` so that it is positive near the origin.

    Parameters
    ----------
    u : ndarray
        Radial function on an ascending grid.

    Returns
    -------
    ndarray
        ``u`` or ``-u``: the first sample above 1e-3 of the peak is positive.
    """
    big = np.flatnonzero(np.abs(u) > 1e-3 * float(np.abs(u).max()))
    if big.size and u[big[0]] < 0.0:
        return -u
    return u


def _matching_radius(
    r: np.ndarray, srv: np.ndarray, k: float, r_neutral: float | None
) -> float:
    """Radius beyond which the continuum equation is taken as free.

    It is the smaller of two radii: the first grid radius ``>= 10`` Bohr
    beyond which ``|s r v|`` stays negligible, and ``r_neutral`` (the atom's
    charge is enclosed). A warning names the residual if neither exists.
    """
    tol = _MATCH_TOL * max(1.0, k)
    small = np.abs(srv) <= tol
    tail_ok = np.logical_and.accumulate(small[::-1])[::-1]
    idx = np.flatnonzero(tail_ok & (r >= _MIN_MATCH))
    candidates = [float(r[idx[0]])] if idx.size else []
    if r_neutral is not None:
        candidates.append(max(_MIN_MATCH, float(r_neutral)))
    if candidates:
        return min(candidates)
    beyond = r >= _MIN_MATCH
    residual = float(np.abs(srv[beyond]).max()) if beyond.any() else 0.0
    if residual > tol:
        warnings.warn(
            f"the atomic potential has not decayed at the end of the GPAW grid "
            f"(|r v| = {residual:.3e} Ha Bohr); matching there",
            RuntimeWarning,
            stacklevel=4,
        )
    return max(_MIN_MATCH, float(r[-1]))


def _log_grid(srv, energy: float, r0: float, r_out: float) -> tuple:
    """Log grid ``r0 .. r_out`` with local phase step ``q dx <= 0.05``, ``dx <= 0.01``."""
    probe = np.geomspace(r0, r_out, 4000)
    q = probe * np.sqrt(np.maximum(0.0, 2.0 * (energy - srv(probe) / probe)))
    dx = min(0.01, 0.05 / max(float(q.max()), 1e-30))
    span = math.log(r_out / r0)
    n_steps = math.ceil(span / dx)
    dx = span / n_steps
    r = np.exp(math.log(r0) + dx * np.arange(n_steps + 1))
    r[-1] = r_out
    return r, dx


def solve_continuum(
    r_pot: np.ndarray,
    srv_pot: np.ndarray,
    r0: float,
    lprime: int,
    energy: float,
    r_neutral: float | None = None,
) -> tuple:
    """Per-Ry normalised continuum state in the central potential ``v_s``.

    Solves ``u'' = [l'(l'+1)/r^2 + 2 (v_s - E)] u`` on the log grid
    ``x = ln r`` with ``u = r^(1/2) w``, where
    ``w'' = [(l' + 1/2)^2 + 2 r^2 (v_s - E)] w`` (exact) is integrated
    outward by Numerov from the regular start ``u ~ r^(l'+1) (1 + c r)``.
    Beyond the matching radius (``|r v_s|`` negligible, or the atom's charge
    enclosed) the potential is zero and ``u = a jhat(k r) + b nhat(k r)`` is fitted at two radii about
    ``pi/(2k)`` apart; ``u`` is then scaled to the asymptotic amplitude
    ``(pi k)^(-1/2)``, ``k = sqrt(2 E)`` in 1/Bohr.

    Parameters
    ----------
    r_pot : ndarray
        Radial grid of the potential, Bohr, ascending.
    srv_pot : ndarray
        ``r v_s(r)`` on ``r_pot``, Ha Bohr (already scaled).
    r0 : float
        First point of the log grid, Bohr.
    lprime : int
        Orbital angular momentum.
    energy : float
        Kinetic energy ``E > 0`` in Hartree.
    r_neutral : float, optional
        Radius (Bohr) beyond which the atom's charge is enclosed, from
        :func:`neutral_radius`. The potential is set to zero beyond the
        matching radius, which is at most this value.

    Returns
    -------
    tuple
        ``(r, u)``: the log grid with ``r = 0`` prepended, and ``u`` on it.
    """
    lp = int(lprime)
    k = math.sqrt(2.0 * energy)
    r_pot = np.asarray(r_pot, dtype=np.float64)
    srv_pot = np.asarray(srv_pot, dtype=np.float64)
    r_match = _matching_radius(r_pot, srv_pot, k, r_neutral)
    quarter = math.pi / (2.0 * k)
    r_out = max(20.0, r_match + quarter)
    spline = CubicSpline(r_pot, srv_pot, extrapolate=True)

    def srv(rr: np.ndarray) -> np.ndarray:
        # Zero beyond R_match, which is where |r v_s| became negligible.
        return np.where(rr < r_match, spline(np.minimum(rr, r_match)), 0.0)

    r, dx = _log_grid(srv, energy, r0, r_out)
    g = (lp + 0.5) ** 2 + 2.0 * r * srv(r) - 2.0 * r * r * energy
    c = float(srv(np.array([0.0]))[0]) / (lp + 1.0)
    w_start = r[:2] ** (lp + 0.5) * (1.0 + c * r[:2])
    u = np.sqrt(r) * numerov(g, w_start[0], w_start[1], dx)

    i2 = r.size - 1
    i1 = int(np.searchsorted(r, r_out - quarter))
    i1 = min(max(i1, int(np.searchsorted(r, r_match))), i2 - 1)
    z = k * r[[i1, i2]]
    basis = np.stack([z * spherical_jn(lp, z), z * spherical_yn(lp, z)], axis=1)
    a, b = np.linalg.solve(basis, u[[i1, i2]])
    u = u / (math.hypot(a, b) * math.sqrt(math.pi * k))
    return np.concatenate(([0.0], r)), np.concatenate(([0.0], u))
