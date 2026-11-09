"""A frozen set of atoms in an orthogonal box, ready for a multislice run.

An :class:`AtomSet` holds one structure in a rectangular box whose edges lie
along the Cartesian axes. Box axis 0 maps to grid rows ("y"), axis 1 to grid
columns ("x"), and axis 2 is the beam direction (depth increases toward
+axis 2). Lengths are in Å, angles in radians, mean-square displacements
<u²> in Å².

The container is immutable. Its arrays are read-only, so a property returns
the stored array without a copy, and several structures can share arrays.
Every operation returns a new structure.

Public names: :class:`AtomSet`, :func:`has_partial_occupancy`,
:func:`read_atom_set`, :data:`FULL_OCCUPANCY_TOLERANCE`,
:data:`DEFAULT_TILT_SINE`, :data:`MAX_SUPERCELL_FACTOR`.
"""

from __future__ import annotations

import numbers

import numpy as np

from scatterem.physics._atom_set_checks import (
    check_atom_data,
    check_fractional,
    real,
    rot_from_axis_angle,
)
from scatterem.physics._atom_set_io import read_structure_arrays
from scatterem.physics._orthobox import (
    DEFAULT_TILT_SINE,
    MAX_SUPERCELL_FACTOR,
    check_lattice,
    check_tilt_sine,
    plan_axis_aligned,
    plan_orthobox,
)

__all__ = [
    "DEFAULT_TILT_SINE",
    "FULL_OCCUPANCY_TOLERANCE",
    "MAX_SUPERCELL_FACTOR",
    "AtomSet",
    "has_partial_occupancy",
    "read_atom_set",
]

#: Site fractions at or above ``1 - FULL_OCCUPANCY_TOLERANCE`` count as full.
#: Reason: the supported formats write occupancies with four decimals or
#: more (CIF practice ``0.3333``; muSTEM and Kirkland files ``1.0000``). A
#: full site written with four decimals is off by at most 5e-5 when rounded
#: and 1e-4 when truncated (``0.9999``). A real vacancy fraction below 1e-4
#: changes the scattered intensity by less than 1e-4, which no multislice
#: comparison resolves. This is the only copy of the value.
FULL_OCCUPANCY_TOLERANCE = 1e-4


def has_partial_occupancy(site_fraction) -> bool:
    """Return True if any site fraction is below ``1 - FULL_OCCUPANCY_TOLERANCE``.

    Parameters
    ----------
    site_fraction : array_like
        Any shape, any real dtype. Callers must not pass NaN.

    Returns
    -------
    bool
        False for empty input. A value exactly at ``1 - tolerance`` is full.
    """
    a = np.asarray(site_fraction)
    return a.size > 0 and bool(a.min() < 1.0 - FULL_OCCUPANCY_TOLERANCE)


class AtomSet:
    """One atomic structure in a rectangular, axis-aligned box (immutable).

    Parameters
    ----------
    box_spec : array_like
        Any real dtype, Å. Either (3,) side lengths of a rectangular box, or
        (3, 3) lattice vectors as rows (any orientation, any handedness,
        orthogonal or oblique). Rows are reduced to a rectangular box with
        :func:`scatterem.physics._orthobox.plan_orthobox`; the beam (axis 2)
        is then along ``row0 x row1``, and every periodic image carries the
        data of its source atom.
    fractional : array_like
        (N, 3), any real dtype. Fractional coordinates of `box_spec`. N may
        be 0. For a (3,) box they are kept as given (no wrap: a finite
        specimen may have atoms outside [0, 1)). For (3, 3) rows the result
        is in [0, 1).
    element_z : array_like
        (N,), integers or integer-valued floats, 1 to 118. Atomic number Z.
    site_fraction : None, float or array_like, optional
        (N,) or scalar, in [0, 1]. Site occupancy. None means full
        occupancy. Values in (1, 1 + FULL_OCCUPANCY_TOLERANCE] are file
        rounding and become 1.
    msd : None, float or array_like, optional
        (N,) or scalar, Å², >= 0. Isotropic mean-square displacement <u²>
        along one direction. None gives
        :data:`scatterem.crystal.phase.DEFAULT_U2`. Never derived from `adp`.
    adp : None or array_like, optional
        (N, 3, 3), Å². Symmetric anisotropic U tensors, Cartesian in the
        frame of `box_spec` (the box for (3,), the rows for (3, 3)). For
        (3, 3) rows they become ``Q U Q^T`` with ``Q`` the frame of the
        reduction (its small strain is neglected; Grosse-Kunstleve and Adams
        2002).
    remark : str, optional
        Free text.
    tilt_sine : float, optional
        Largest sine of the tilt of a new box edge from its lattice vector in
        the reduction of (3, 3) rows, in (0, 0.1]. Always validated.

    Raises
    ------
    ValueError
        For a bad argument. The message starts with the argument name.
    """

    __slots__ = (
        "_sides",
        "_fractional",
        "_z",
        "_site_fraction",
        "_msd",
        "_adp",
        "_remark",
    )

    def __init__(
        self,
        box_spec,
        fractional,
        element_z,
        *,
        site_fraction=None,
        msd=None,
        adp=None,
        remark: str = "",
        tilt_sine: float = DEFAULT_TILT_SINE,
    ):
        raw_box = real(box_spec, "box_spec")
        if raw_box.shape == (3, 3):
            edge_rows = check_lattice(raw_box, "box_spec")
        elif raw_box.shape == (3,):
            edge_rows = None
            sides = np.array(raw_box, dtype=np.float64)
            if not np.isfinite(sides).all() or (sides <= 0).any():
                raise ValueError(
                    f"box_spec: side lengths must be finite and > 0; got {sides}"
                )
        else:
            raise ValueError(
                f"box_spec: must have shape (3,) or (3, 3); got {raw_box.shape}"
            )
        sine = check_tilt_sine(tilt_sine)
        f = check_fractional(fractional)
        n = f.shape[0]
        z, fill, u2, adp = check_atom_data(
            n, element_z, site_fraction, msd, adp, FULL_OCCUPANCY_TOLERANCE
        )
        if not isinstance(remark, str):
            raise ValueError(f"remark: must be a str; got {type(remark).__name__}")
        if edge_rows is not None:
            plan = plan_orthobox(edge_rows, sine)
            f, source = plan.apply(f)
            sides = np.array(plan.sides)
            z, fill, u2 = z[source], fill[source], u2[source]
            if adp is not None:
                Q = plan.frame
                adp = (Q @ adp @ Q.T)[source]
        self._store(sides, f, z, fill, u2, adp, remark)  # pyright: ignore[reportPossiblyUnboundVariable]

    def _store(self, sides, f, z, fill, u2, adp, remark):
        for name, value in (
            ("_sides", sides),
            ("_fractional", f),
            ("_z", z),
            ("_site_fraction", fill),
            ("_msd", u2),
            ("_adp", adp),
        ):
            if value is not None:
                value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_remark", remark)

    @classmethod
    def _from_validated(cls, sides, fractional, z, fill, msd, adp, remark) -> AtomSet:
        """Store arrays that the class made itself, without validation."""
        obj = cls.__new__(cls)
        obj._store(sides, fractional, z, fill, msd, adp, remark)
        return obj

    def __setattr__(self, name, value):
        raise AttributeError(
            "AtomSet is immutable; use with_fractional or another operation"
        )

    def __delattr__(self, name):
        raise AttributeError("AtomSet is immutable")

    def __reduce__(self):
        """Pickle and copy through ``_from_validated``; it freezes the arrays."""
        return type(self)._from_validated, (
            self._sides, self._fractional, self._z, self._site_fraction,
            self._msd, self._adp, self._remark,
        )  # fmt: skip

    # ----------------------------------------------------------- read access
    @property
    def sides(self) -> np.ndarray:
        """(3,) float64, Å. Side lengths of the rectangular box (read-only)."""
        return self._sides

    @property
    def fractional(self) -> np.ndarray:
        """(N, 3) float64. Fractional coordinates along the box axes (read-only)."""
        return self._fractional

    @property
    def element_z(self) -> np.ndarray:
        """(N,) int64. Atomic number Z of each atom (read-only)."""
        return self._z

    @property
    def site_fraction(self) -> np.ndarray:
        """(N,) float64. Site occupancy in [0, 1] (read-only)."""
        return self._site_fraction

    @property
    def msd(self) -> np.ndarray:
        """(N,) float64, Å². Isotropic <u²> along one direction (read-only)."""
        return self._msd

    @property
    def adp(self) -> np.ndarray | None:
        """(N, 3, 3) float64, Å², box frame, or None (read-only)."""
        return self._adp

    @property
    def remark(self) -> str:
        """Free text; empty when not given."""
        return self._remark

    @property
    def row_total(self) -> int:
        """Number of atom rows N (a mixed site has one row per species)."""
        return int(self._z.shape[0])

    def __len__(self) -> int:
        return self.row_total

    @property
    def box_volume(self) -> float:
        """Box volume, Å³."""
        return float(np.prod(self._sides))

    @property
    def xyz_angstrom(self) -> np.ndarray:
        """(N, 3) float64, Å. ``fractional * sides`` (a new array per call)."""
        return self._fractional * self._sides

    def __repr__(self) -> str:
        return f"AtomSet(row_total={self.row_total}, sides={self._sides.tolist()}, remark={self._remark!r})"

    # ------------------------------------------------------------ operations
    def with_fractional(self, fractional) -> AtomSet:
        """Return a new structure with new positions and all other data shared.

        Parameters
        ----------
        fractional : array_like
            (N, 3), any real dtype, the same N. Fractional coordinates of
            this box. One float64 copy is made; no wrap, no reduction.

        Returns
        -------
        AtomSet
            Shares sides, Z, site fractions, msd, adp and remark with this
            structure. This structure does not change.
        """
        f = check_fractional(fractional)
        if f.shape[0] != self.row_total:
            raise ValueError(
                f"fractional: must have shape ({self.row_total}, 3); got {f.shape}"
            )
        return AtomSet._from_validated(
            self._sides,
            f,
            self._z,
            self._site_fraction,
            self._msd,
            self._adp,
            self._remark,
        )

    def _centroid(self) -> np.ndarray:
        if self.row_total == 0:
            return np.asarray(self._sides) / 2.0
        return self.xyz_angstrom.mean(axis=0)

    def with_rotation(
        self,
        *,
        axis_vec,
        turn_radians,
        fixed_point=None,
        as_crystal: bool = False,
        padding: float = 0.0,
        tilt_sine: float = DEFAULT_TILT_SINE,
    ) -> AtomSet:
        """Return the structure after a rigid rotation.

        A point at Cartesian ``r`` (box frame, Å) moves to
        ``fixed_point + R @ (r - fixed_point)``, with ``R`` the active,
        right-handed turn of :func:`scatterem.rotation.axis_angle.to_matrix`.
        Z, site fractions, <u²> and the atom order are kept; tensors become
        ``R U R^T``. This structure does not change. All parameters are
        keyword-only, so an axis and an angle cannot be swapped by mistake.

        Parameters
        ----------
        axis_vec : array_like
            (3,), any real dtype, any non-zero length, box frame. Turn axis.
        turn_radians : float
            Turn angle, radians.
        fixed_point : array_like, optional
            (3,), Å, box frame. The point that does not move. Default: the
            mean Cartesian position of the atom rows (each row counts once);
            for N = 0 the box centre ``sides / 2``. Reason: the mean position
            belongs to the material, not to the box, so the default does not
            change when a caller adds or removes vacuum, and a specimen that
            is not centred in its box turns about its own middle. For the
            finite specimen the result does not depend on this point, because
            the new box is fitted around the turned content (a different
            point gives only a common shift, which the fit removes). For a
            periodic crystal it sets the origin of the turned lattice.
        as_crystal : bool, optional
            False (default): finite specimen. True: periodic crystal. The
            default never fails; a periodic result exists only for special
            rotations, so the caller must ask for it.
        padding : float, optional
            Å, >= 0, only for ``as_crystal=False``. Default 0: the new box
            always holds the turned input box, so vacuum in the input stays.
        tilt_sine : float, optional
            Only for ``as_crystal=True``: largest sine of the tilt of a box
            edge from its lattice vector, in (0, 0.1].

        Returns
        -------
        AtomSet
            The turned structure, a new object.

        Notes
        -----
        Finite specimen: the input box turns about its centre into a tilted
        block. Its extent along box axis ``i`` is the turned centre plus or
        minus ``sum_j |R_ij| sides_j / 2``. The new box is the smallest box
        with faces normal to the box axes that holds this extent and every
        turned atom with `padding` added on each side. So each atom keeps a
        gap of at least `padding` to every face. The result is not periodic:
        give enough `padding` for a calculation with periodic boundaries.

        Periodic crystal: the result is periodic, and is the same crystal in
        the new frame, when each box axis is parallel (within `tilt_sine`)
        to a lattice vector of the turned crystal with indices of at most
        ``MAX_AXIS_INDEX`` (24). Examples: any multiple of a quarter turn
        about a box axis; a turn that brings a low-index zone axis onto a
        box axis. Each box edge is the shortest such vector. For any other
        rotation it raises ``ValueError``; use ``as_crystal=False`` then.
        """
        a = real(turn_radians, "turn_radians")
        if a.ndim != 0 or not np.isfinite(a):
            raise ValueError(
                f"turn_radians: must be a finite real scalar; got {turn_radians!r}"
            )
        ax = np.array(real(axis_vec, "axis_vec"), dtype=np.float64)
        if ax.shape != (3,) or not np.isfinite(ax).all() or not np.any(ax):
            raise ValueError(
                f"axis_vec: must be a finite, non-zero (3,) vector; got {axis_vec!r}"
            )
        if fixed_point is None:
            p = self._centroid()
        else:
            p = np.array(real(fixed_point, "fixed_point"), dtype=np.float64)
            if p.shape != (3,) or not np.isfinite(p).all():
                raise ValueError(
                    f"fixed_point: must be a finite (3,) vector, Å; got {fixed_point!r}"
                )
        if not isinstance(as_crystal, (bool, np.bool_)):
            raise ValueError(f"as_crystal: must be a bool; got {as_crystal!r}")
        pad = real(padding, "padding")
        if pad.ndim != 0 or not np.isfinite(pad) or pad < 0:
            raise ValueError(
                f"padding: must be a finite number >= 0 (Å); got {padding!r}"
            )
        pad = float(pad)
        R = rot_from_axis_angle(ax, float(a))
        adp = None if self._adp is None else R @ self._adp @ R.T
        if as_crystal:
            if pad != 0.0:
                raise ValueError("padding: applies only when as_crystal is False")
            turned = np.diag(self._sides) @ R.T
            try:
                plan = plan_axis_aligned(turned, tilt_sine)
            except ValueError as e:
                raise ValueError(
                    f"{e}; this rotation gives no periodic crystal, use as_crystal=False"
                ) from e
            B = plan.matrix.astype(np.float64) @ turned
            shift = (p - R @ p) @ np.linalg.inv(B)
            f, source = plan.apply(self._fractional, shift)
            return AtomSet._from_validated(
                np.array(plan.sides),
                f,
                self._z[source],
                self._site_fraction[source],
                self._msd[source],
                None if adp is None else adp[source],
                self._remark,
            )
        r = p + (self.xyz_angstrom - p) @ R.T
        half = self._sides / 2.0
        centre = p + R @ (half - p)
        reach = np.abs(R) @ half
        low_edge = centre - reach
        high_edge = centre + reach
        if self.row_total:
            low_edge = np.minimum(low_edge, r.min(axis=0) - pad)
            high_edge = np.maximum(high_edge, r.max(axis=0) + pad)
        sides = high_edge - low_edge
        return AtomSet._from_validated(
            sides,
            (r - low_edge) / sides,
            self._z,
            self._site_fraction,
            self._msd,
            adp,
            self._remark,
        )

    def replicated(self, replicas) -> AtomSet:
        """Return the n0 x n1 x n2 periodic supercell.

        Parameters
        ----------
        replicas : sequence of int
            Three integers >= 1, the copies along box axes 0, 1 and 2
            (Python or NumPy integers; bool and float are refused).

        Returns
        -------
        AtomSet
            New sides ``sides * replicas``. Rows are atom-major: all
            ``M = n0 * n1 * n2`` copies of atom ``j`` come first as one
            block, then those of atom ``j + 1``. Inside a block the copy
            index is ``i0 + n0 * (i1 + n1 * i2)`` (``i0`` runs fastest). So
            row ``j * M + i0 + n0 * (i1 + n1 * i2)`` is atom ``j`` moved by
            ``(i0, i1, i2)`` box lengths, and its source is ``row // M``.
            Positions are ``(f + i) / replicas``. A value that rounds to
            exactly 1.0 from an input below 1 becomes the float just below
            1, so input in [0, 1) gives output in [0, 1). Replicas
            (1, 1, 1) give a structure that shares the arrays. This structure
            does not change.
        """
        try:
            items = list(replicas)
        except TypeError:
            raise ValueError(
                f"replicas: must be a sequence of three integers; got {replicas!r}"
            ) from None
        if len(items) != 3 or any(
            isinstance(c, (bool, np.bool_)) or not isinstance(c, numbers.Integral)
            for c in items
        ):
            raise ValueError(f"replicas: must be three integers; got {replicas!r}")
        c = np.array([int(x) for x in items], dtype=np.int64)
        if (c < 1).any():
            raise ValueError(f"replicas: each count must be >= 1; got {c.tolist()}")
        if (c == 1).all():
            return AtomSet._from_validated(*self.__reduce__()[1])
        # np.indices over (n2, n1, n0) runs its last index (i0) fastest.
        cell_steps = np.indices(c[::-1]).reshape(3, -1).T[:, ::-1].astype(np.float64)  # pyright: ignore[reportCallIssue, reportArgumentType]
        m = cell_steps.shape[0]
        f = self._fractional
        out = ((f[:, None, :] + cell_steps[None, :, :]) / c).reshape(-1, 3)
        below = np.repeat(f < 1.0, m, axis=0)
        out[(out >= 1.0) & below] = np.nextafter(1.0, 0.0)
        adp = None if self._adp is None else np.repeat(self._adp, m, axis=0)
        return AtomSet._from_validated(
            self._sides * c,
            out,
            np.repeat(self._z, m),
            np.repeat(self._site_fraction, m),
            np.repeat(self._msd, m),
            adp,
            self._remark,
        )


def read_atom_set(
    source_file,
    *,
    file_style: str | None = None,
    displacement_kind: str | None = None,
    position_kind: str | None = None,
    tilt_sine: float = DEFAULT_TILT_SINE,
) -> AtomSet:
    """Build an :class:`AtomSet` from one structure file on disk.

    Parameters
    ----------
    source_file : str or os.PathLike
        The file. A missing file raises ``FileNotFoundError``.
    file_style : str, optional
        ``"kirkland"``, ``"poscar"``, ``"mustem"``, ``"cif"`` or
        ``"extxyz"``. None selects by the lower-case extension: ``.xyz``
        kirkland, ``.p1`` poscar, ``.xtl`` mustem, ``.cif`` cif, ``.extxyz``
        extxyz.
    displacement_kind : str, optional
        Quantity of the thermal numbers in the file: ``"mean_square"``
        (<u²>, Å²), ``"rms_length"`` (Å, squared on reading) or
        ``"b_factor"`` (Debye-Waller B, Å², divided by 8 pi²; Willis and
        Pryor 1975, Trueblood et al. 1996). None: the quantity the format
        specification defines (kirkland ``"rms_length"``, mustem
        ``"mean_square"``, poscar ``"mean_square"``), so a file that follows
        its specification reads correctly without the option. CIF values
        are already <u²> and are used as they are; extxyz has no thermal
        column. The value is validated for every file_style. If the file gives
        no number for an atom, that atom keeps ``DEFAULT_U2`` with no
        conversion.
    position_kind : str, optional
        ``"relative"`` (fractional) or ``"absolute"`` (Cartesian Å):
        overrides the coordinate type of the format, for kirkland (default
        absolute) and mustem (default relative) only. Other readers refuse a
        value other than None: POSCAR states the type in the file, CIF and
        extxyz have one type.
    tilt_sine : float, optional
        Largest tilt sine of the reduction of (3, 3) lattice rows.

    Returns
    -------
    AtomSet
        The remark is the comment or title line (kirkland, poscar, mustem) or
        the file name (cif, extxyz).

    Raises
    ------
    ValueError
        For a bad option or a line that the file_style cannot parse (the message
        names the file and the 1-based line number).
    """
    sine = check_tilt_sine(tilt_sine)
    data = read_structure_arrays(
        source_file, file_style, displacement_kind, position_kind
    )
    return AtomSet(
        data["box_spec"],
        data["fractional"],
        data["element_z"],
        site_fraction=data["site_fraction"],
        msd=data["msd"],
        adp=data["adp"],
        remark=data["remark"],
        tilt_sine=sine,
    )
