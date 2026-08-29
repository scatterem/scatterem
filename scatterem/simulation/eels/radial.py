"""Radial wavefunctions for core-loss EELS transition potentials.

A :class:`RadialWavefunction` stores ``u(r) = r * R(r)`` on a radial grid in
Bohr together with the orbital energy in eV.  Two families of providers build
them:

* **GPAW-backed** (physically accurate, requires the optional ``gpaw``
  dependency).  Bound states come from an all-electron atomic DFT calculation;
  continuum states are obtained by integrating the radial Schroedinger equation
  (Numerov) in the all-electron effective potential.  This mirrors the approach
  used by abTEM but is re-implemented from the underlying physics.

* **Analytic** (hydrogenic bound state + free spherical-Bessel continuum).  These
  are *not* quantitatively accurate but let the whole transition-potential and
  multislice pipeline run and be tested without GPAW installed.

Computed wavefunctions are cached on disk (keyed by their physical parameters)
so that repeated simulations do not redo the atomic calculations.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from importlib.util import find_spec
from math import factorial
from typing import Optional

import numpy as np
from scipy.special import genlaguerre, spherical_jn

__all__ = [
    "AtomicRadialWavefunction",
    "RadialWavefunction",
    "gpaw_available",
    "bound_wavefunction",
    "continuum_wavefunction",
    "hydrogenic_bound_wavefunction",
    "free_continuum_wavefunction",
    "cache_dir",
]

# Atomic-unit conversions.
_HARTREE_EV = 27.211386245988
_RYDBERG_EV = _HARTREE_EV / 2.0
_BOHR_ANG = 0.52917721090380


def gpaw_available() -> bool:
    """True if the optional :mod:`gpaw` dependency can be imported."""
    return find_spec("gpaw") is not None


def cache_dir() -> str:
    """Directory used to cache atomic wavefunctions / transition potentials.

    Override with the ``SCATTEREM_EELS_CACHE`` environment variable.
    """
    path = os.environ.get(
        "SCATTEREM_EELS_CACHE",
        os.path.join(
            os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
            "scatterem",
            "eels",
        ),
    )
    os.makedirs(path, exist_ok=True)
    return path


@dataclass
class AtomicRadialWavefunction:
    """A radial wavefunction ``u(r) = r * R(r)``.

    Parameters
    ----------
    n, l : int | None
        Principal and orbital angular-momentum quantum numbers.  ``n`` is
        ``None`` for continuum states.
    energy : float
        Orbital energy in eV.  Negative for bound states, positive (energy above
        the ionization threshold) for continuum states.
    r : ndarray
        Radial grid in Bohr.
    u : ndarray
        ``u(r) = r * R(r)`` sampled on ``r``.
    """

    n: Optional[int]
    l: int
    energy: float
    r: np.ndarray
    u: np.ndarray

    def __post_init__(self) -> None:
        self.r = np.asarray(self.r, dtype=np.float64)
        self.u = np.asarray(self.u, dtype=np.float64)
        self.energy = float(self.energy)

    def __call__(self, r) -> np.ndarray:
        """Evaluate ``u(r)`` (linear interpolation, zero outside the grid)."""
        return np.interp(r, self.r, self.u, left=0.0, right=0.0)

    @property
    def rmax(self) -> float:
        return float(self.r[-1])


# Descriptive name; the implementation is independent of abTEM's similarly named
# class. ``RadialWavefunction`` is kept as a backward-compatible alias.
RadialWavefunction = AtomicRadialWavefunction


# --------------------------------------------------------------------------- #
# Analytic providers (no GPAW needed)
# --------------------------------------------------------------------------- #
def hydrogenic_bound_wavefunction(
    Z: int, n: int, l: int, n_points: int = 4000, rmax: float = 40.0
) -> RadialWavefunction:
    """Analytic hydrogenic bound state (effective nuclear charge ``Z``).

    Useful for testing the transition-potential machinery without GPAW.  The
    energy is the hydrogenic eigenvalue ``-Z^2 / (2 n^2)`` Hartree.
    """
    if not (0 <= l < n):
        raise ValueError(f"require 0 <= l < n, got n={n}, l={l}")
    r = np.linspace(1e-6, rmax, n_points)  # Bohr
    rho = 2.0 * Z * r / n
    # Normalised hydrogenic radial function R_nl(r) (atomic units).
    norm = np.sqrt(
        (2.0 * Z / n) ** 3 * factorial(n - l - 1) / (2.0 * n * factorial(n + l))
    )
    laguerre = genlaguerre(n - l - 1, 2 * l + 1)(rho)
    R = norm * np.exp(-rho / 2.0) * rho**l * laguerre
    energy = -(Z**2) / (2.0 * n**2) * _HARTREE_EV
    return RadialWavefunction(n=n, l=l, energy=energy, r=r, u=r * R)


def free_continuum_wavefunction(
    l: int, epsilon: float, n_points: int = 8000, rmax: float = 60.0
) -> RadialWavefunction:
    """Free-particle continuum state: ``u_l(r) = r * k * j_l(k r)``.

    ``epsilon`` is the energy above threshold in eV.  This neglects the nuclear
    Coulomb attraction and is *not* energy-normalised the way the GPAW backend
    is, so it must not be used for absolute cross-sections -- it exists only to
    let the transition-potential / multislice pipeline run and be tested without
    GPAW.  Use ``backend="gpaw"`` for physically meaningful results.
    """
    r = np.linspace(1e-6, rmax, n_points)  # Bohr
    k = np.sqrt(2.0 * epsilon / _HARTREE_EV)  # 1/Bohr
    u = r * k * spherical_jn(l, k * r)
    return RadialWavefunction(n=None, l=l, energy=float(epsilon), r=r, u=u)


# --------------------------------------------------------------------------- #
# GPAW-backed providers
# --------------------------------------------------------------------------- #
def _numerov(f: np.ndarray, u0: float, u1: float, dx: float) -> np.ndarray:
    """Integrate ``u''(x) = f(x) u(x)`` outward via the Numerov method."""
    u = np.zeros_like(f)
    u[0] = u0
    u[1] = u1
    h2 = dx * dx
    h12 = h2 / 12.0
    w0 = u[0] * (1.0 - h12 * f[0])
    w1 = u[1] * (1.0 - h12 * f[1])
    for i in range(2, f.size):
        w2 = 2.0 * w1 - w0 + h2 * f[i - 1] * u[i - 1]
        u[i] = w2 / (1.0 - h12 * f[i])
        w0, w1 = w1, w2
    return u


def _gpaw_bound(Z: int, n: int, l: int, xc: str) -> RadialWavefunction:
    import contextlib
    import io

    from ase.data import chemical_symbols
    from gpaw.atom.all_electron import AllElectron

    with contextlib.redirect_stdout(io.StringIO()):
        ae = AllElectron(chemical_symbols[Z], xcname=xc)
        ae.run()

    # Locate the (n, l) orbital among the computed eigenstates.
    idx = None
    for j, (nj, lj) in enumerate(zip(ae.n_j, ae.l_j)):
        if nj == n and lj == l:
            idx = j
            break
    if idx is None:
        raise ValueError(f"orbital n={n}, l={l} not found for Z={Z}")

    energy = ae.e_j[idx] * _HARTREE_EV  # eV (negative)
    return RadialWavefunction(
        n=n, l=l, energy=energy, r=np.asarray(ae.r), u=np.asarray(ae.u_j[idx])
    )


def _gpaw_continuum(
    Z: int, lprime: int, epsilon: float, xc: str, potential_scale: float = 1.0
) -> RadialWavefunction:
    import contextlib
    import io

    from ase.data import chemical_symbols
    from gpaw.atom.aeatom import AllElectronAtom
    from scipy.interpolate import interp1d

    with contextlib.redirect_stdout(io.StringIO()):
        ae = AllElectronAtom(chemical_symbols[Z], xc=xc)
        ae.run()
        ae.scalar_relativistic = True
        ae.refine()

    # Effective radial potential V(r) (in Rydberg-consistent units: -2 * vr / r).
    vr = interp1d(
        ae.rgd.r_g,
        -2.0 * ae.vr_sg[0],
        fill_value="extrapolate",
        bounds_error=False,
    )

    r = np.linspace(1e-12, 40.0, 1_000_000)  # Bohr
    ef = epsilon / _RYDBERG_EV
    with np.errstate(divide="ignore", invalid="ignore"):
        veff = vr(r) / r
        centrifugal = lprime * (lprime + 1) / r**2
    # ``potential_scale`` is an optional empirical multiplier on the effective
    # potential term ``(centrifugal - veff)``.  The default is
    # 1.0: Brown 2019 Eqs. (8)-(9), and full Dirac/FAC continuum solvers,
    # carry no such factor.  abTEM applies ``1.02`` as an ad-hoc scalar-
    # relativistic deepening; setting ``potential_scale=1.02`` reproduces abTEM's
    # continuum to ~1e-7.  It matters almost only for the barrier-free l'=0
    # (monopole) s-wave: its overlap with the bound state is near-orthogonal, so
    # a ~2% potential change (a small near-origin phase shift) is amplified into
    # a large change in the monopole transition potential, while the dipole
    # (held off the nucleus by its centrifugal barrier) is unaffected.
    f = (centrifugal - veff) * potential_scale - ef  # u'' = f u  (Rydberg units)

    dx = r[1] - r[0]
    u = _numerov(f, 0.0, 1e-12, dx)
    # Energy-normalise the continuum state.  In the asymptotic region the
    # solution is a free sinusoid ``u ~ A sin(k_loc r + delta)`` whose envelope
    # is ``A = sqrt(u^2 + u'^2 / k_loc^2)`` with the *local* wavenumber
    # ``k_loc^2 = -f`` (since ``u'' = f u`` and ``f -> -ef < 0`` far out).  Using
    # the true asymptotic amplitude — rather than the global peak ``max|u|``,
    # which overshoots the sinusoid envelope by an l-dependent 6-15% — gives the
    # correct Manson-1972 energy normalisation ``u / (sqrt(pi) eps^(1/4))``.
    up = np.gradient(u, dx)
    kloc2 = -f
    mask = (r > 0.5 * r[-1]) & (kloc2 > 0)
    # Evaluate the envelope only on the asymptotic, classically allowed tail
    # (kloc2 > 0).  Computing it over all r would take sqrt of a negative number
    # in the inner classically-forbidden region (a spurious RuntimeWarning); the
    # masked result is numerically identical.
    if mask.any():
        A = np.median(np.sqrt(u[mask] ** 2 + up[mask] ** 2 / kloc2[mask]))
    else:
        A = 0.0
    if np.isfinite(A) and A > 0:
        u = u / A / (np.sqrt(np.pi) * ef**0.25)
    return RadialWavefunction(n=None, l=lprime, energy=float(epsilon), r=r, u=u)


# --------------------------------------------------------------------------- #
# Public dispatch with on-disk caching
# --------------------------------------------------------------------------- #
def _cache_path(tag: str) -> str:
    digest = hashlib.sha1(tag.encode()).hexdigest()[:16]
    return os.path.join(cache_dir(), f"radial_{digest}.npz")


def _load_or_build(tag: str, builder) -> RadialWavefunction:
    path = _cache_path(tag)
    if os.path.exists(path):
        data = np.load(path, allow_pickle=False)
        n = int(data["n"]) if not np.isnan(data["n"]) else None
        return RadialWavefunction(
            n=n,
            l=int(data["l"]),
            energy=float(data["energy"]),
            r=data["r"],
            u=data["u"],
        )
    wf = builder()
    np.savez(
        path,
        n=np.nan if wf.n is None else wf.n,
        l=wf.l,
        energy=wf.energy,
        r=wf.r,
        u=wf.u,
    )
    return wf


def bound_wavefunction(
    Z: int,
    n: int,
    l: int,
    xc: str = "PBE",
    backend: str = "auto",
    use_cache: bool = True,
) -> RadialWavefunction:
    """Bound radial wavefunction for orbital ``(n, l)`` of element ``Z``.

    ``backend`` is ``"gpaw"``, ``"hydrogenic"`` or ``"auto"`` (GPAW if available,
    otherwise hydrogenic).
    """
    if backend == "auto":
        backend = "gpaw" if gpaw_available() else "hydrogenic"
    if backend == "hydrogenic":
        builder = lambda: hydrogenic_bound_wavefunction(Z, n, l)
        tag = f"bound|hydrogenic|{Z}|{n}|{l}"
    elif backend == "gpaw":
        builder = lambda: _gpaw_bound(Z, n, l, xc)
        tag = f"bound|gpaw|{xc}|{Z}|{n}|{l}"
    else:
        raise ValueError(f"unknown backend {backend!r}")
    return _load_or_build(tag, builder) if use_cache else builder()


def continuum_wavefunction(
    Z: int,
    lprime: int,
    epsilon: float,
    xc: str = "PBE",
    backend: str = "auto",
    use_cache: bool = True,
    potential_scale: float = 1.0,
) -> RadialWavefunction:
    """Continuum radial wavefunction of angular momentum ``lprime`` at energy
    ``epsilon`` (eV above threshold) for element ``Z``.

    ``potential_scale`` multiplies the effective-potential term of the GPAW
    continuum radial equation (default 1.0 = the published Brown 2019 physics;
    abTEM uses 1.02).  It is ignored by the analytic ``hydrogenic`` backend, whose
    free-particle continuum has no atomic potential.
    """
    if epsilon <= 0:
        # The energy-normalised continuum diverges (1/epsilon**0.25) at the
        # threshold and the free state collapses to zero; require eps > 0.
        raise ValueError(f"continuum energy epsilon must be > 0, got {epsilon}")
    if backend == "auto":
        backend = "gpaw" if gpaw_available() else "hydrogenic"
    if backend == "hydrogenic":
        builder = lambda: free_continuum_wavefunction(lprime, epsilon)
        tag = f"cont|free|{lprime}|{epsilon:.6g}"
    elif backend == "gpaw":
        builder = lambda: _gpaw_continuum(Z, lprime, epsilon, xc, potential_scale)
        # ``v3`` marks the asymptotic-amplitude normalisation + the configurable
        # ``potential_scale`` (encoded below); bumping the version invalidates
        # caches written by earlier normalisers.
        tag = f"cont|gpaw|v3|{xc}|{Z}|{lprime}|{epsilon:.6g}|ps{potential_scale:.6g}"
    else:
        raise ValueError(f"unknown backend {backend!r}")
    return _load_or_build(tag, builder) if use_cache else builder()
