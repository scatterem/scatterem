"""Radial one-electron states for the EELS transition potentials.

Written by the scatterem authors from the Schrödinger equation, the
energy-normalisation convention of Manson, Phys. Rev. A 6, 1013 (1972), and
the multislice EELS formalism of Brown, Ciston and Ophus, Phys. Rev. Research 1,
033186 (2019).

Every state is ``u(r) = r R(r)`` on a radial grid in Bohr, with ``u > 0`` near
the origin. Bound states have ``int u^2 dr = 1``. Continuum states are
normalised per Rydberg: ``u -> (pi k)^(-1/2) sin(k r - l pi/2 + delta_l)`` with
``k`` in 1/Bohr. Families: ``"hydrogenic"`` (analytic bound state, free
spherical-Bessel continuum) and ``"gpaw"`` (GPAW all-electron bound state,
Numerov continuum in its Kohn-Sham potential; GPAW is the optional extra
``scatterem[eels]`` and is imported only inside the GPAW helpers).
"""

from __future__ import annotations

import contextlib
import functools
import io
import math
import os
from dataclasses import dataclass
from importlib.util import find_spec
from math import factorial

import numpy as np
from scipy.special import genlaguerre, spherical_jn

from ._radial_numerics import numerov
from ._radial_store import AtomRecord as _AtomRecord
from ._radial_store import atom_record as _atom_record
from ._radial_store import cache_file, load_or_build, record_bound, record_continuum
from ._wavefunction_choice import HydrogenicFallbackWarning, resolve_wavefunctions

__all__ = [
    "AtomicRadialWavefunction",
    "HydrogenicFallbackWarning",
    "gpaw_available",
    "resolve_wavefunctions",
    "bound_wavefunction",
    "continuum_wavefunction",
    "hydrogenic_bound_wavefunction",
    "free_continuum_wavefunction",
    "cache_dir",
]

_HARTREE_EV = 27.211386245988  # Hartree energy E_h in eV (CODATA 2018)
#: Rydberg energy in eV, E_h / 2.
_RYDBERG_EV = _HARTREE_EV / 2.0
#: Bohr radius a_0 in Angstrom (CODATA 2018).
_BOHR_ANG = 0.529177210903

_numerov = numerov  # the private name that tests import

#: Version tag of the radial cache. A new tag never reads files of older code.
_CACHE_TAG = "cr3"
#: Radial grid density of the GPAW all-electron atom (grid points per node).
_GPAW_GPERNODE = 600


def gpaw_available() -> bool:
    """Tell if ``gpaw`` can be imported, without importing it.

    Returns
    -------
    bool
        ``True`` if ``gpaw`` is importable, ``False`` otherwise.
    """
    try:
        return find_spec("gpaw") is not None
    except (ValueError, ImportError):
        return False


def cache_dir() -> str:
    """Return the EELS on-disk cache directory and create it if it is missing.

    ``SCATTEREM_EELS_CACHE`` overrides the location. Otherwise the directory is
    ``$XDG_CACHE_HOME/scatterem/eels``, or ``~/.cache/scatterem/eels`` when
    ``XDG_CACHE_HOME`` is not set. The environment is read at call time.

    Returns
    -------
    str
        Path to the (created if missing) cache directory.
    """
    path = os.environ.get("SCATTEREM_EELS_CACHE")
    if not path:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
            os.path.expanduser("~"), ".cache"
        )
        path = os.path.join(base, "scatterem", "eels")
    os.makedirs(path, exist_ok=True)
    return path


@dataclass
class AtomicRadialWavefunction:
    """Radial one-electron state ``u(r) = r R(r)`` on a grid in Bohr.

    Parameters
    ----------
    n, l : int | None
        Principal and orbital angular-momentum quantum numbers. ``n`` is
        ``None`` for continuum states.
    energy : float
        Orbital energy in eV. Negative for bound states, positive (energy
        above the ionization threshold) for continuum states.
    r : ndarray
        Radial grid in Bohr, strictly ascending.
    u : ndarray
        ``u(r) = r * R(r)`` sampled on ``r``.
    """

    n: int | None
    l: int
    energy: float
    r: np.ndarray
    u: np.ndarray

    def __post_init__(self) -> None:
        """Cast the fields to their canonical types and validate the grid.

        Raises
        ------
        ValueError
            If ``r`` and ``u`` are not 1-D arrays of the same length, or if
            ``r`` is not strictly ascending.
        """
        self.r = np.ascontiguousarray(self.r, dtype=np.float64)
        self.u = np.ascontiguousarray(self.u, dtype=np.float64)
        self.energy = float(self.energy)
        self.l = int(self.l)
        if self.n is not None:
            self.n = int(self.n)
        if self.r.ndim != 1 or self.u.ndim != 1 or self.r.shape != self.u.shape:
            raise ValueError(
                f"r and u must be 1-D arrays of equal length, got shapes "
                f"{self.r.shape} and {self.u.shape}"
            )
        if self.r.size > 1 and not np.all(np.diff(self.r) > 0.0):
            raise ValueError("the radial grid r must be strictly ascending")

    def __call__(self, r) -> np.ndarray:
        """Interpolate ``u`` linearly at ``r``; exactly 0 outside the grid.

        Parameters
        ----------
        r : array_like
            Radii in Bohr.

        Returns
        -------
        ndarray
            ``u(r)``, float64, with the shape of ``r``.
        """
        return np.interp(np.asarray(r, dtype=np.float64), self.r, self.u, 0.0, 0.0)

    @property
    def rmax(self) -> float:
        """Outer end of the radial grid.

        Returns
        -------
        float
            The last (largest) value of ``r``, in Bohr.
        """
        return float(self.r[-1])


def hydrogenic_bound_wavefunction(
    Z: int, n: int, l: int, n_points: int | None = None, rmax: float = 40.0
) -> AtomicRadialWavefunction:
    """Normalised hydrogenic bound state on a uniform grid.

    With ``rho = 2 Z r / n``,
    ``R_nl = N rho^l exp(-rho/2) L^(2l+1)_(n-l-1)(rho)`` and
    ``N = sqrt((2Z/n)^3 (n-l-1)! / (2n (n+l)!))``, where ``L`` is the
    generalised Laguerre polynomial with ``L^(a)_k(0) = binom(k+a, k)``.
    The energy is ``-Z^2/(2 n^2)`` Hartree.

    The default grid has at least 4000 points and at least 10 points per
    orbital length ``n / Z`` Bohr, so a heavy K shell is sampled as finely
    as a light one.

    Parameters
    ----------
    Z : int
        Effective nuclear charge.
    n : int
        Principal quantum number.
    l : int
        Orbital angular-momentum quantum number, must satisfy ``0 <= l < n``.
    n_points : int, optional
        Number of samples on the radial grid ``linspace(0, rmax, n_points)``.
        ``None`` (default) chooses it from ``Z / n`` as described above.
    rmax : float, optional
        Outer edge of the radial grid, in Bohr.

    Returns
    -------
    AtomicRadialWavefunction
        The normalised hydrogenic bound state ``u(r) = r * R_nl(r)``.

    Raises
    ------
    ValueError
        If ``l`` is not in ``0 <= l < n``.
    """
    n, l = int(n), int(l)
    if not 0 <= l < n:
        raise ValueError(f"hydrogenic states require 0 <= l < n, got n={n}, l={l}")
    if n_points is None:  # >= 4000 points and >= 10 per orbital length n/Z Bohr
        n_points = max(4000, math.ceil(10.0 * float(Z) / n * float(rmax)) + 1)
    r = np.linspace(0.0, float(rmax), int(n_points))
    rho = 2.0 * Z * r / n
    norm = math.sqrt(
        (2.0 * Z / n) ** 3 * factorial(n - l - 1) / (2.0 * n * factorial(n + l))
    )
    radial = norm * rho**l * np.exp(-rho / 2.0) * genlaguerre(n - l - 1, 2 * l + 1)(rho)
    energy = -(Z**2) / (2.0 * n**2) * _HARTREE_EV
    return AtomicRadialWavefunction(n=n, l=l, energy=energy, r=r, u=r * radial)


def free_continuum_wavefunction(
    l: int, epsilon: float, n_points: int = 8000, rmax: float = 60.0
) -> AtomicRadialWavefunction:
    """Free continuum state, normalised per Rydberg.

    ``u(r) = (pi k)^(-1/2) k r j_l(k r)``, with ``k = sqrt(epsilon / Ry)`` in
    1/Bohr. Far out ``u -> (pi k)^(-1/2) sin(k r - l pi/2)``, so
    ``int u_E u_E' dr = delta(E - E')`` with ``E`` in Ry (Manson 1972).

    Parameters
    ----------
    l : int
        Orbital angular-momentum quantum number of the continuum state.
    epsilon : float
        Energy above the ionization threshold, in eV. Must be > 0.
    n_points : int, optional
        Number of samples on the radial grid ``linspace(0, rmax, n_points)``.
    rmax : float, optional
        Outer edge of the radial grid, in Bohr.

    Returns
    -------
    AtomicRadialWavefunction
        The free spherical-Bessel continuum state ``u_l(r)``, units
        Bohr^(-1/2) Ry^(-1/2).

    Raises
    ------
    ValueError
        If ``epsilon <= 0``.
    """
    if not epsilon > 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    k = math.sqrt(float(epsilon) / _RYDBERG_EV)  # 1/Bohr
    r = np.linspace(0.0, float(rmax), int(n_points))
    u = k * r * spherical_jn(int(l), k * r) / math.sqrt(math.pi * k)
    return AtomicRadialWavefunction(n=None, l=int(l), energy=float(epsilon), r=r, u=u)


@functools.cache
def _gpaw_atom(Z: int, xc: str) -> _AtomRecord:
    """Run a spin-paired, scalar-relativistic GPAW all-electron atom.

    The calculation is deterministic, so the in-process memo keeps the
    callers pure. The grid has ``_GPAW_GPERNODE`` = 600 points per node, 4
    times the GPAW default, which leaves core levels unconverged by up to
    9 eV (Sr 1s); with 600, doubling moves Au 2p and 3d by < 0.15 eV.

    Parameters
    ----------
    Z : int
        Atomic number.
    xc : str
        Exchange-correlation functional name.

    Returns
    -------
    _AtomRecord
        Grid, ``r v(r)`` and the occupied states.
    """
    from ase.data import chemical_symbols
    from gpaw.atom.all_electron import AllElectron

    with contextlib.redirect_stdout(io.StringIO()):
        atom = AllElectron(
            chemical_symbols[int(Z)],
            xcname=xc,
            scalarrel=True,
            gpernode=_GPAW_GPERNODE,
        )
        atom.run()
    return _atom_record(atom)


def _gpaw_bound(Z: int, n: int, l: int, xc: str) -> AtomicRadialWavefunction:
    """Bound state ``(n, l)`` of the GPAW all-electron atom.

    The energy is the Kohn-Sham eigenvalue in eV. It under-binds core levels
    by up to tens of eV; that is the physics of the Kohn-Sham model, not an
    error.

    Parameters
    ----------
    Z : int
        Atomic number.
    n, l : int
        Quantum numbers of the occupied orbital.
    xc : str
        Exchange-correlation functional name.

    Returns
    -------
    AtomicRadialWavefunction
        ``u = r R`` with ``int u^2 dr = 1`` and ``u > 0`` near the origin.

    Raises
    ------
    ValueError
        If the ground-state configuration does not contain ``(n, l)``.
    RuntimeError
        If the GPAW state is not normalised to 1e-4 (grid-index quadrature).
    """
    r, u, e_ha = record_bound(_gpaw_atom(int(Z), xc), int(Z), int(n), int(l))
    return AtomicRadialWavefunction(
        n=int(n), l=int(l), energy=e_ha * _HARTREE_EV, r=r, u=u
    )


def _gpaw_continuum(
    Z: int, lprime: int, epsilon: float, xc: str, potential_scale: float = 1.0
) -> AtomicRadialWavefunction:
    """Continuum state in the self-consistent potential of the GPAW atom.

    The potential is the ground-state Kohn-Sham potential of the neutral atom
    (no core hole), scaled by ``potential_scale``. It is set to zero beyond the
    radius where every bound orbital is enclosed (tail weight below 1e-6):
    there the neutral atom has no electrostatic field, and the semi-local
    exchange-correlation potential is numerical noise that ends in a constant
    floor (about -8.5e-4 Ha for PBE). This cut is a model choice; moving it
    changes the phase shifts by up to about 0.02 rad. The continuum is
    non-relativistic and normalised per Rydberg:
    ``u -> (pi k)^(-1/2) sin(k r - l' pi/2 + delta)``.

    Parameters
    ----------
    Z : int
        Atomic number.
    lprime : int
        Orbital angular momentum of the continuum state.
    epsilon : float
        Energy above the threshold, in eV. Must be > 0.
    xc : str
        Exchange-correlation functional name.
    potential_scale : float, optional
        Multiplier ``s`` on the whole potential. ``0`` gives the free state.

    Returns
    -------
    AtomicRadialWavefunction
        The continuum state on a log grid with ``r = 0`` prepended.
    """
    if not epsilon > 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    r, u = record_continuum(
        _gpaw_atom(int(Z), xc),
        float(potential_scale),
        1e-6 / max(int(Z), 1),
        int(lprime),
        float(epsilon) / _HARTREE_EV,
    )
    return AtomicRadialWavefunction(
        n=None, l=int(lprime), energy=float(epsilon), r=r, u=u
    )


def _cache_path(tag: str) -> str:
    """File in :func:`cache_dir` for the canonical key ``tag``."""
    return cache_file(cache_dir(), tag)


def _load_or_build(tag: str, builder) -> AtomicRadialWavefunction:
    """Read the cached state of ``tag``, or build it and write it atomically."""
    return load_or_build(_cache_path(tag), builder, AtomicRadialWavefunction)


def bound_wavefunction(
    Z: int,
    n: int,
    l: int,
    xc: str = "PBE",
    wavefunctions: str = "auto",
    use_cache: bool = True,
) -> AtomicRadialWavefunction:
    """Bound radial state ``(n, l)`` of element ``Z``, cached on disk.

    Parameters
    ----------
    Z : int
        Atomic number of the element.
    n : int
        Principal quantum number.
    l : int
        Orbital angular-momentum quantum number.
    xc : str, optional
        Exchange-correlation functional passed to the GPAW wavefunctions.
    wavefunctions : str, optional
        ``"auto"`` (the default: GPAW if importable, otherwise hydrogenic with
        a :class:`~scatterem.simulation.eels.physics.radial.HydrogenicFallbackWarning`), ``"gpaw"`` or ``"hydrogenic"``.
    use_cache : bool, optional
        Whether to read/write the on-disk cache in :func:`cache_dir`.

    Returns
    -------
    AtomicRadialWavefunction
        The bound-state radial wavefunction ``u(r) = r * R_nl(r)``.
    """
    family = resolve_wavefunctions(wavefunctions)
    if family == "gpaw":
        tag = f"{_CACHE_TAG}|bound|gpaw|Z={Z}|n={n}|l={l}|xc={xc}"

        def builder():
            return _gpaw_bound(Z, n, l, xc)

    else:
        tag = f"{_CACHE_TAG}|bound|hydrogenic|Z={Z}|n={n}|l={l}"

        def builder():
            return hydrogenic_bound_wavefunction(Z, n, l)

    return _load_or_build(tag, builder) if use_cache else builder()


def continuum_wavefunction(
    Z: int,
    lprime: int,
    epsilon: float,
    xc: str = "PBE",
    wavefunctions: str = "auto",
    use_cache: bool = True,
    potential_scale: float = 1.0,
) -> AtomicRadialWavefunction:
    """Continuum radial state ``l'`` at ``epsilon`` eV above threshold, cached.

    Parameters
    ----------
    Z : int
        Atomic number of the element.
    lprime : int
        Orbital angular-momentum quantum number of the continuum state.
    epsilon : float
        Energy above the ionization threshold, in eV. Must be > 0.
    xc : str, optional
        Exchange-correlation functional passed to the GPAW wavefunctions.
    wavefunctions : str, optional
        ``"auto"`` (the default: GPAW if importable, otherwise hydrogenic with
        a :class:`~scatterem.simulation.eels.physics.radial.HydrogenicFallbackWarning`), ``"gpaw"`` or ``"hydrogenic"``.
    use_cache : bool, optional
        Whether to read/write the on-disk cache in :func:`cache_dir`.
    potential_scale : float, optional
        Multiplier on the atomic potential of the GPAW continuum equation;
        ``1.0`` leaves it unchanged. It is ignored by the analytic
        ``hydrogenic`` wavefunctions, whose free-particle continuum has no
        atomic potential.

    Returns
    -------
    AtomicRadialWavefunction
        The continuum radial wavefunction ``u_lprime(r)``, normalised per
        Rydberg (Manson 1972).

    Raises
    ------
    ValueError
        If ``epsilon <= 0``.
    """
    if not epsilon > 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    family = resolve_wavefunctions(wavefunctions)
    eps = float(epsilon)
    if family == "gpaw":
        scale = float(potential_scale)
        tag = f"{_CACHE_TAG}|cont|gpaw|Z={Z}|l={lprime}|eps={eps!r}|xc={xc}|s={scale!r}"

        def builder():
            return _gpaw_continuum(Z, lprime, epsilon, xc, potential_scale)

    else:
        tag = f"{_CACHE_TAG}|cont|hydrogenic|Z={Z}|l={lprime}|eps={eps!r}"

        def builder():
            return free_continuum_wavefunction(lprime, epsilon)

    return _load_or_build(tag, builder) if use_cache else builder()
