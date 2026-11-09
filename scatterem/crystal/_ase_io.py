"""Read atoms through ASE into plain numpy arrays.

The physics layer may not import ASE, but
``scatterem.crystal`` may, inside a function. This module is that one place: it
reads the extended-XYZ and ASE ``.traj`` formats with ``ase.io.read`` and
returns numpy arrays, so ``read_atom_set``
uses ASE's readers without depending on ASE types. CIF goes through
:meth:`scatterem.crystal.phase.Phase.from_cif`, which also reads ASE.

Velocities are converted from ASE's unit system, which ASE defines for the
momenta it stores: length in Angstrom, time in Angstrom * sqrt(u / eV), so
1 fs is ``ase.units.fs`` time units and one ASE velocity unit is
``1000 * ase.units.fs`` (about 98.2) Angstrom/ps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

#: The ASE format names this module reads.
ASE_FORMATS = ("extxyz", "traj")


@dataclass(frozen=True)
class AseFrame:
    """One frame as ASE read it.

    Attributes
    ----------
    cell : numpy.ndarray
        Shape ``(3, 3)`` lattice-vector rows, Angstrom.
    positions : numpy.ndarray
        Shape ``(N, 3)`` Cartesian positions, Angstrom.
    numbers : numpy.ndarray
        Shape ``(N,)`` atomic numbers, int64.
    velocities : numpy.ndarray or None
        Shape ``(N, 3)``, Angstrom/ps; None when the frame stores no momenta.
    """

    cell: NDArray[np.float64]
    positions: NDArray[np.float64]
    numbers: NDArray[np.int64]
    velocities: NDArray[np.float64] | None


def read_frames(path: str | Path, format: str) -> list[AseFrame]:
    """Read every frame of an extended-XYZ or ASE ``.traj`` file.

    Parameters
    ----------
    path : str or Path
        File to read.
    format : str
        ``"extxyz"`` or ``"traj"``.

    Returns
    -------
    list of AseFrame
        One entry per frame, in file order.

    Raises
    ------
    ValueError
        For another format.
    """
    if format not in ASE_FORMATS:
        raise ValueError(f"format must be one of {ASE_FORMATS}, got {format!r}")
    from ase import units
    from ase.io import read

    # ase.io.read is typed Atoms | list[Atoms]; index=":" gives a list.
    images: Any = read(str(path), index=":", format=format)
    per_ps = 1000.0 * units.fs
    frames = []
    for atoms in images:
        velocities = None
        if atoms.has("momenta"):
            velocities = np.asarray(atoms.get_velocities(), dtype=float) * per_ps
        frames.append(
            AseFrame(
                cell=np.asarray(atoms.get_cell()[:], dtype=float),
                positions=np.asarray(atoms.get_positions(), dtype=float),
                numbers=np.asarray(atoms.get_atomic_numbers(), dtype=np.int64),
                velocities=velocities,
            )
        )
    return frames
