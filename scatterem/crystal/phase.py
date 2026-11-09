"""One crystalline phase: a general cell and its P1 sites, read from a CIF.

:class:`Phase` keeps the crystal in its own cell, which can be oblique. Its
sites use the names, units and rules of
:class:`scatterem.physics.atom_set.AtomSet`: ``fractional`` (along the cell
edges), ``element_z``, ``site_fraction``, ``msd`` (<u^2>, Angstrom^2) and
``adp`` (anisotropic U tensors). ``cell`` holds the edge vectors as rows. So
the same crystal in a rectangular box for a multislice run is::

    AtomSet(phase.cell, phase.fractional, phase.element_z,
            site_fraction=phase.site_fraction, msd=phase.msd, adp=phase.adp,
            remark=phase.name)

This module is in the ``crystal`` layer, below ``physics``, so it does not
import ``AtomSet``; it checks the sites with its own code by the same rules.

:meth:`Phase.from_cif` reads a CIF through ASE (imported inside the method,
so importing this module does not import ASE). It reads the isotropic thermal
value of each site line; ``DEFAULT_U2`` applies where the file has none.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scatterem.crystal._cif_sites import cif_site_arrays

__all__ = ["Phase"]

#: Mean-square displacement <u^2> [Angstrom^2] for an atom whose CIF line
#: gives no ``U_iso`` or ``B_iso`` value, and for a :class:`Phase` or an
#: ``AtomSet`` built without ``msd``. One value for both CIF routes
#: (``Phase.from_cif`` and ``physics.atom_set.read_atom_set``), set by the
#: owner on 2026-10-22.
DEFAULT_U2 = 0.01

#: Site fractions in ``(1, 1 + _FULL_SITE_ROUNDING]`` are file rounding and
#: become 1. It is the ``FULL_OCCUPANCY_TOLERANCE`` of
#: ``physics.atom_set`` (the layering contract keeps ``crystal`` from
#: importing it); a test checks that the two are equal.
_FULL_SITE_ROUNDING = 1e-4

#: Largest atomic number of the element table.
_MAX_Z = 118

#: Largest asymmetry of a tensor, relative to its largest entry.
_TENSOR_SYMMETRY = 1e-10


class Phase:
    """One crystalline phase: a general cell and its P1 sites (immutable).

    Parameters
    ----------
    cell : array_like
        Shape ``(3, 3)``, any real dtype, Angstrom. The cell edge vectors as
        rows; any orientation, orthogonal or oblique, right- or left-handed.
    fractional : array_like
        Shape ``(N, 3)``, any real dtype. Site positions as fractions of the
        cell edges (component ``i`` along row ``i`` of `cell`). ``N`` may be
        0. Kept as given (no wrap).
    element_z : array_like
        Shape ``(N,)``, integers 1 to 118. Atomic number of each site row.
    site_fraction : None, float or array_like, optional
        ``(N,)`` or scalar, in [0, 1]. Occupancy of each row; None means
        full. A mixed site has one row per species, all at the same
        position. Checked as in ``AtomSet``.
    msd : None, float or array_like, optional
        ``(N,)`` or scalar, Angstrom^2, >= 0. Isotropic mean-square
        displacement <u^2> along one direction. None gives
        :data:`DEFAULT_U2`.
    adp : None or array_like, optional
        Shape ``(N, 3, 3)``, Angstrom^2. Symmetric anisotropic displacement
        tensors ``U``, Cartesian in the frame of the `cell` rows. None
        (default) means isotropic sites (`msd` only). A structure factor
        uses a tensor as ``exp(-2 pi^2 g^T U g)``.
    name : str, optional
        Free-form label.
    point_group : str or None, optional
        Hermann-Mauguin point-group symbol, if known. No code reads it yet.

    Raises
    ------
    ValueError
        For a bad argument. The message starts with the argument name.

    Notes
    -----
    The arrays are read-only and every property returns the stored array.
    """

    __slots__ = (
        "_cell",
        "_fractional",
        "_z",
        "_site_fraction",
        "_msd",
        "_adp",
        "_name",
        "_point_group",
    )

    def __init__(
        self,
        cell,
        fractional,
        element_z,
        *,
        site_fraction=None,
        msd=None,
        adp=None,
        name: str = "",
        point_group: str | None = None,
    ):
        rows = _real_array(cell, "cell")
        if rows.shape != (3, 3):
            raise ValueError(f"cell: must have shape (3, 3); got {rows.shape}")
        f = _real_array(fractional, "fractional")
        if f.ndim != 2 or f.shape[1] != 3:
            raise ValueError(f"fractional: must have shape (N, 3); got {f.shape}")
        n = f.shape[0]
        z = _site_numbers(element_z, n)
        fill = _per_site(site_fraction, "site_fraction", n, 1.0)
        if n and (fill.min() < 0.0 or fill.max() > 1.0 + _FULL_SITE_ROUNDING):
            raise ValueError(
                f"site_fraction: values must be in [0, 1]; got min {fill.min()}, "
                f"max {fill.max()}"
            )
        fill = np.minimum(fill, 1.0)
        u2 = _per_site(msd, "msd", n, DEFAULT_U2)
        if n and u2.min() < 0.0:
            raise ValueError(f"msd: values must be >= 0; got min {u2.min()}")
        tensors = None if adp is None else _site_tensors(adp, n)
        if not isinstance(name, str):
            raise ValueError(f"name: must be a str; got {type(name).__name__}")
        if point_group is not None and not isinstance(point_group, str):
            raise ValueError(
                f"point_group: must be a str or None; got {type(point_group).__name__}"
            )
        for slot, value in (
            ("_cell", rows),
            ("_fractional", f),
            ("_z", z),
            ("_site_fraction", fill),
            ("_msd", u2),
            ("_adp", tensors),
        ):
            if value is not None:
                value.setflags(write=False)
            object.__setattr__(self, slot, value)
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_point_group", point_group)

    def __setattr__(self, name, value):
        raise AttributeError("Phase is immutable; build a new Phase")

    def __delattr__(self, name):
        raise AttributeError("Phase is immutable")

    def __reduce__(self):
        """Pickle and copy through the constructor (it checks the arrays again)."""
        return _rebuild_phase, (
            self._cell, self._fractional, self._z, self._site_fraction,
            self._msd, self._adp, self._name, self._point_group,
        )  # fmt: skip

    # ----------------------------------------------------------- read access
    @property
    def cell(self) -> np.ndarray:
        """(3, 3) float64, Angstrom. Cell edge vectors as rows (read-only)."""
        return self._cell

    @property
    def fractional(self) -> np.ndarray:
        """(N, 3) float64. Site positions along the cell edges (read-only)."""
        return self._fractional

    @property
    def element_z(self) -> np.ndarray:
        """(N,) int64. Atomic number of each site row (read-only)."""
        return self._z

    @property
    def site_fraction(self) -> np.ndarray:
        """(N,) float64. Occupancy of each site row in [0, 1] (read-only)."""
        return self._site_fraction

    @property
    def msd(self) -> np.ndarray:
        """(N,) float64, Angstrom^2. Isotropic <u^2> (read-only)."""
        return self._msd

    @property
    def adp(self) -> np.ndarray | None:
        """(N, 3, 3) float64, Angstrom^2, frame of the cell rows, or None."""
        return self._adp

    @property
    def name(self) -> str:
        """Free-form label; empty when not given."""
        return self._name

    @property
    def point_group(self) -> str | None:
        """Hermann-Mauguin point-group symbol, or None."""
        return self._point_group

    @property
    def row_total(self) -> int:
        """Number of site rows N (a mixed site has one row per species)."""
        return int(self._z.shape[0])

    def __len__(self) -> int:
        return self.row_total

    @property
    def volume(self) -> float:
        """Cell volume ``|det(cell)|`` [Angstrom^3]."""
        return float(abs(np.linalg.det(self._cell)))

    def __repr__(self) -> str:
        return f"Phase(name={self._name!r}, row_total={self.row_total})"

    def number_density(self) -> dict[int, float]:
        """Sum the site fractions per element and divide by the cell volume.

        A phase without sites is not an error: it has no elements, so the
        result is the empty dict.

        Returns
        -------
        dict[int, float]
            Atomic number to number density [atoms / Angstrom^3].

        Raises
        ------
        ValueError
            If the cell volume is not positive.
        """
        volume = self.volume
        if not volume > 0.0:
            raise ValueError(
                f"phase {self._name!r} has non-positive cell volume {volume}"
            )
        z, fill = self._z, self._site_fraction
        return {int(e): float(fill[z == e].sum() / volume) for e in np.unique(z)}

    # ------------------------------------------------------------ conversion
    @classmethod
    def from_atom_set(cls, structure, *, name: str | None = None) -> Phase:
        """Return the phase whose cell is the box of an ``AtomSet``.

        Parameters
        ----------
        structure : scatterem.physics.atom_set.AtomSet
            The cell is ``diag(structure.sides)``; the site arrays are copied.
        name : str, optional
            Label. None (default) takes ``structure.remark``.

        Returns
        -------
        Phase
        """
        return cls(
            np.diag(structure.sides),
            structure.fractional,
            structure.element_z,
            site_fraction=structure.site_fraction,
            msd=structure.msd,
            adp=structure.adp,
            name=structure.remark if name is None else name,
        )

    @classmethod
    def from_cif(
        cls, path, *, msd_default: float = DEFAULT_U2, name: str = ""
    ) -> Phase:
        """Read a CIF into a :class:`Phase` (symmetry-expanded to P1).

        Parameters
        ----------
        path : str or Path
            CIF file, whatever its extension. ASE applies the symmetry
            operations, so the returned sites are the full cell content.
            ASE reads the last data block that holds a structure, as
            ``ase.io.read`` does.
        msd_default : float
            Mean-square thermal displacement <u^2> [Angstrom^2], finite and
            >= 0, for a site whose line gives no thermal value (``?``, ``.``
            or no column). Default :data:`DEFAULT_U2` (0.01).
        name : str
            Label; defaults to the file stem.

        Returns
        -------
        Phase
            Symmetry-expanded (P1) phase read from `path`.

        Raises
        ------
        RuntimeError
            If ASE cannot read `path` as a CIF file.
        ValueError
            If a thermal value is text other than ``?`` / ``.``. A negative
            value does not raise: a warning is logged and the default applies.

        Notes
        -----
        ``msd`` comes from ``_atom_site_U_iso_or_equiv`` [Angstrom^2] or,
        where a line has no U value, from ``_atom_site_B_iso_or_equiv`` with
        B = 8 pi^2 <u^2> (De Graef, *Introduction to Conventional
        Transmission Electron Microscopy*, 2003, p. 149). An uncertainty in
        brackets, such as ``0.0123(4)``, is dropped. Each expanded site takes
        the value of its own site line; on a mixed site each species row
        takes the value of its own line. A ``Uani`` line gives its
        ``U_iso_or_equiv`` value in ``msd``.

        The ``_atom_site_aniso_U_*`` (or ``_B_*``, ``B = 8 pi^2 U``) loop, if
        present, gives ``adp``: ``U_cart = M N U N M^T`` (``M`` the lattice
        vectors as columns, ``N = diag(|a*_i|)``), rotated to every
        symmetry-expanded site by the operation that generated it. A line
        without an aniso entry gets ``<u^2> I``. Without the loop ``adp`` is
        None.
        """
        path = Path(path)
        cell, frac, z, occ, u2, tensors = cif_site_arrays(path, float(msd_default))
        return cls(
            cell,
            frac,
            z,
            site_fraction=occ,
            msd=u2,
            adp=tensors,
            name=name or path.stem,
        )


def _real_array(x, label: str) -> np.ndarray:
    """``x`` as a new finite float64 array; refuse a non-real dtype."""
    raw = np.asarray(x)
    if raw.size and raw.dtype.kind not in "iuf":
        raise ValueError(f"{label}: must be real numbers; got dtype {raw.dtype}")
    out = np.array(raw, dtype=np.float64)
    if not np.isfinite(out).all():
        raise ValueError(f"{label}: values must be finite")
    return out


def _site_numbers(element_z, n: int) -> np.ndarray:
    """Atomic numbers as int64; each an integer in 1.._MAX_Z."""
    z = _real_array(element_z, "element_z")
    if z.shape != (n,):
        raise ValueError(f"element_z: must have shape ({n},); got {z.shape}")
    if not (z == np.rint(z)).all():
        raise ValueError("element_z: values must be integers")
    if n and (z.min() < 1 or z.max() > _MAX_Z):
        raise ValueError(f"element_z: values must be in 1..{_MAX_Z}")
    return z.astype(np.int64)


def _per_site(x, label: str, n: int, default: float) -> np.ndarray:
    """A scalar or an (n,) array as (n,) float64; None gives `default`."""
    if x is None:
        return np.full(n, default, dtype=np.float64)
    values = _real_array(x, label)
    if values.ndim == 0:
        return np.full(n, float(values), dtype=np.float64)
    if values.shape != (n,):
        raise ValueError(
            f"{label}: must be a scalar or have shape ({n},); got {values.shape}"
        )
    return values


def _site_tensors(adp, n: int) -> np.ndarray:
    """(n, 3, 3) float64 symmetric tensors."""
    tensors = _real_array(adp, "adp")
    if tensors.shape != (n, 3, 3):
        raise ValueError(f"adp: must have shape ({n}, 3, 3); got {tensors.shape}")
    if n:
        asym = np.abs(tensors - tensors.transpose(0, 2, 1)).max()
        if asym > _TENSOR_SYMMETRY * max(np.abs(tensors).max(), 1e-30):
            raise ValueError(f"adp: tensors must be symmetric; asymmetry {asym:g}")
    return tensors


def _rebuild_phase(cell, fractional, z, fill, msd, adp, name, point_group) -> Phase:
    return Phase(
        cell,
        fractional,
        z,
        site_fraction=fill,
        msd=msd,
        adp=adp,
        name=name,
        point_group=point_group,
    )


def cif_to_arrays(
    path,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None
]:
    """Read a CIF into plain arrays (symmetry-expanded to P1).

    This is the reader under :meth:`Phase.from_cif`, for readers that build
    their own container (``physics.atom_set.read_atom_set``). It marks a
    missing thermal value with NaN, so the caller applies its own default.

    Parameters
    ----------
    path : str or Path
        CIF file, whatever its extension.

    Returns
    -------
    rows : numpy.ndarray
        Shape ``(3, 3)``, float64. Lattice vectors as rows [Angstrom].
    fractional : numpy.ndarray
        Shape ``(N, 3)``, float64. Fractional coordinates in ``[0, 1)``.
    z : numpy.ndarray
        Shape ``(N,)``, int64. Atomic numbers.
    occupancy : numpy.ndarray
        Shape ``(N,)``, float64. A mixed site gives one entry per species.
    u2 : numpy.ndarray
        Shape ``(N,)``, float64. Mean-square displacement <u^2>
        [Angstrom^2]. NaN where the file gives no thermal value.
    tensors : numpy.ndarray or None
        Shape ``(N, 3, 3)``, float64. Anisotropic displacement tensors U
        [Angstrom^2], Cartesian in the frame of `rows`. None when the file
        has no anisotropic loop. A site without an anisotropic entry gets
        ``u2 * I`` (NaN where `u2` is NaN).

    Raises
    ------
    RuntimeError
        If the file cannot be parsed as a CIF.
    """
    return cif_site_arrays(Path(path), float("nan"))
