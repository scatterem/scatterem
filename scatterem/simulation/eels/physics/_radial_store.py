"""Records and on-disk cache files of the EELS radial states.

Private helper of :mod:`.radial`, written by the scatterem authors.
"""

from __future__ import annotations

import hashlib
import math
import os
import zipfile
from dataclasses import dataclass

import numpy as np

from ._radial_numerics import (
    grid_integral,
    neutral_radius,
    positive_start,
    solve_continuum,
)

__all__ = [
    "AtomRecord",
    "atom_record",
    "cache_file",
    "load_or_build",
    "record_bound",
    "record_continuum",
]

#: Largest accepted deviation of a GPAW orbital's norm from 1. A larger one
#: means that the record holds a different quantity than ``u = r R``.
_NORM_GUARD = 1e-4


@dataclass(frozen=True)
class AtomRecord:
    """Plain record of a spherical all-electron atom (Hartree atomic units).

    Attributes
    ----------
    r : ndarray
        Radial grid in Bohr, ascending.
    rv : ndarray
        ``r v(r)`` of the total Kohn-Sham potential on ``r``, in Ha Bohr.
    states : tuple
        One ``(n, l, eigenvalue_Ha, u)`` tuple per occupied state.
    """

    r: np.ndarray
    rv: np.ndarray
    states: tuple


def atom_record(ae) -> AtomRecord:
    """Read the grid, potential and eigenstates from a finished GPAW atom.

    The attribute names are those of the GPAW ``AllElectron`` object after
    ``run()``: ``r`` (grid, Bohr), ``vr`` (``r v(r)`` of the total Kohn-Sham
    potential, Ha Bohr), and per state ``n_j``, ``l_j``, ``e_j`` (Hartree)
    and ``u_j`` (``u = r R``). This is the one place that depends on them.

    Parameters
    ----------
    ae : object
        A GPAW ``AllElectron`` instance after ``run()``.

    Returns
    -------
    AtomRecord
        Copies of the grid, ``r v(r)`` and the states, as float64 arrays.
    """
    states = tuple(
        (int(n), int(ell), float(e), np.array(u, dtype=np.float64))
        for n, ell, e, u in zip(ae.n_j, ae.l_j, ae.e_j, ae.u_j, strict=True)
    )
    return AtomRecord(
        r=np.array(ae.r, dtype=np.float64),
        rv=np.array(ae.vr, dtype=np.float64),
        states=states,
    )


def record_bound(record: AtomRecord, Z: int, n: int, ell: int) -> tuple:
    """Normalised bound orbital ``(n, ell)`` of an atom record.

    The norm is taken in grid-index space (:func:`grid_integral`): the GPAW
    grid is far from uniform, and the trapezoid rule in ``r`` is off by up to
    4e-4 there.

    Parameters
    ----------
    record : AtomRecord
        The atom.
    Z, n, ell : int
        Atomic number (for messages) and quantum numbers ``n``, ``l``.

    Returns
    -------
    tuple
        ``(r, u, eigenvalue_Ha)`` with ``r[0] = 0``, ``int u^2 dr = 1`` and
        ``u > 0`` near the origin.

    Raises
    ------
    ValueError
        If the record has no state ``(n, ell)``.
    RuntimeError
        If the stored norm differs from 1 by more than ``_NORM_GUARD``.
    """
    found = [st for st in record.states if (int(st[0]), int(st[1])) == (n, ell)]
    if not found:
        raise ValueError(f"orbital n={n}, l={ell} not found for Z={Z} in GPAW")
    r = np.asarray(record.r, dtype=np.float64)
    u = np.asarray(found[0][3], dtype=np.float64)
    norm = grid_integral(r, u * u)
    if abs(norm - 1.0) > _NORM_GUARD:
        raise RuntimeError(
            f"GPAW state n={n}, l={ell} of Z={Z} has norm {norm:.8f}, not 1"
        )
    u = positive_start(u / math.sqrt(norm))
    if r[0] > 0.0:
        r, u = np.concatenate(([0.0], r)), np.concatenate(([0.0], u))
    return r, u, float(found[0][2])


def record_continuum(
    record: AtomRecord, scale: float, r0: float, lprime: int, energy: float
) -> tuple:
    """Continuum state in ``scale`` times the potential of an atom record.

    Parameters
    ----------
    record : AtomRecord
        The atom.
    scale : float
        Multiplier on the potential.
    r0 : float
        First point of the log grid, Bohr.
    lprime : int
        Orbital angular momentum.
    energy : float
        Kinetic energy in Hartree, > 0.

    Returns
    -------
    tuple
        ``(r, u)`` from :func:`.solve_continuum`, with the potential cut at the
        radius where the record's orbitals are enclosed.
    """
    r = np.asarray(record.r, dtype=np.float64)
    return solve_continuum(
        r,
        scale * np.asarray(record.rv, dtype=np.float64),
        r0,
        lprime,
        energy,
        r_neutral=neutral_radius(r, record.states),
    )


def cache_file(directory: str, tag: str) -> str:
    """Cache file of the canonical key ``tag``.

    Parameters
    ----------
    directory : str
        Cache directory.
    tag : str
        Canonical key string.

    Returns
    -------
    str
        ``<directory>/radial_<sha256(tag)[:24]>.npz``.
    """
    digest = hashlib.sha256(tag.encode("utf-8")).hexdigest()[:24]
    return os.path.join(directory, f"radial_{digest}.npz")


def load_or_build(path: str, builder, factory):
    """Read a radial state from ``path``, or build it and write it atomically.

    A file that cannot be read is rebuilt and overwritten. The new file is
    written as ``*.tmp.npz`` in the same directory and moved into place with
    ``os.replace``, so a parallel reader never sees a partial file. ``n`` is
    stored as NaN for a continuum state.

    Parameters
    ----------
    path : str
        Cache file.
    builder : callable
        Returns the state when the file is missing or unreadable.
    factory : callable
        ``factory(n=, l=, energy=, r=, u=)`` makes the state from the file.

    Returns
    -------
    object
        The state.
    """
    if os.path.exists(path):
        try:
            with np.load(path) as data:
                n_raw = float(data["n"])
                return factory(
                    n=None if math.isnan(n_raw) else int(n_raw),
                    l=int(data["l"]),
                    energy=float(data["energy"]),
                    r=data["r"],
                    u=data["u"],
                )
        except (OSError, KeyError, ValueError, EOFError, zipfile.BadZipFile):
            pass
    wf = builder()
    tmp = f"{path[:-4]}.{os.getpid()}.tmp.npz"
    np.savez(
        tmp,
        n=np.nan if wf.n is None else float(wf.n),
        l=wf.l,
        energy=wf.energy,
        r=wf.r,
        u=wf.u,
    )
    os.replace(tmp, path)
    return wf
