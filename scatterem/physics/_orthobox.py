"""Reduction of a general lattice to a rectangular, axis-aligned supercell.

Multislice needs a rectangular cell with the beam along one cell axis. An
oblique lattice ``A`` (lattice vectors as rows, Å) is represented by a
supercell ``B = P @ A`` with an integer matrix ``P`` whose rows ``b_i`` are
orthogonal within a strain bound (International Tables Vol. A, ch. 5.1).
Old fractional coordinates ``f`` become ``f @ inv(P)`` in the new basis.

The work has two public steps, so that a caller can decide the reduction
once and apply it to many sets of positions:

1. :func:`plan_orthobox` (or :func:`plan_axis_aligned`) decides ``P``, the
   result frame and the side lengths from the lattice rows alone. It returns
   an :class:`OrthoPlan`.
2. :meth:`OrthoPlan.apply` maps fractional positions of the old cell to all
   their images in the new cell.

Result frame of :func:`plan_orthobox`: axis 2 (the beam) is exactly
``+n = a0 x a1 / |a0 x a1|``; axis 0 is along the in-plane edge ``b0``
(exactly along ``a0`` when the plane lattice of rows 0 and 1 is already
rectangular); axis 1 is ``n x e0``. The frame is right-handed. A left-handed
input gives ``det P < 0``; the map from old to new Cartesian coordinates is
then still a proper rotation followed by a shear of determinant +1, so the
crystal is never mirrored.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

import numpy as np

from scatterem.physics._lattice_search import (
    adjugate,
    beam_search,
    hnf_diagonal,
    order_plane_pair,
    plane_search,
)

__all__ = [
    "DEFAULT_TILT_SINE",
    "MAX_AXIS_INDEX",
    "MAX_SUPERCELL_FACTOR",
    "OrthoPlan",
    "check_lattice",
    "check_tilt_sine",
    "plan_axis_aligned",
    "plan_orthobox",
]

#: Default tilt sine (dimensionless) of the reduction. Each new edge
#: may tilt from its box axis by an angle whose sine is at most this value.
#: Reasons, measured with :func:`plan_orthobox` on 2026-11-08: (1) bond
#: lengths change by at most 0.5 %, 0.015 Å for a 3 Å bond, well below the
#: thermal rms displacement of about 0.1 Å; (2) every crystal cell of the
#: tests keeps the smallest supercell that any bound up to 1e-2 gives: the
#: cubic cell with beta = 90.1 deg (sin 0.1 deg = 1.75e-3) and the TE6 and
#: hub CIFs stay at 1 to 3 cells, the monoclinic cell (5.1, 6.2, 7.3 Å;
#: beta 103.7 deg) needs 3 cells from 4.08e-3 up, Sb2S3 22 cells from
#: 2.59e-3 up. The cost: the triclinic test cells (4.1, 5.3, 6.2 Å; 81, 97,
#: 104 deg and 4.1, 5.2, 6.3 Å; 71.3, 82.4, 97.5 deg) need 93 and 174 cells,
#: against 72 and 95 at 1e-2. A caller can pass a larger bound.
DEFAULT_TILT_SINE = 5e-3

#: Bound on |det P|, the number of input cells in the supercell. 1e4 cells
#: of 10 to 100 atoms give 1e5 to 1e6 atoms, the upper scale of callers.
#: It also bounds the search time.
MAX_SUPERCELL_FACTOR = 10**4

#: Bound on the lattice-vector indices in :func:`plan_axis_aligned`. It
#: covers zone axes up to index 24 (periods up to about 170 Å for a 4 Å
#: cell), more than any practical periodic tilt; the search then costs
#: about 1.2e5 vectors per call.
MAX_AXIS_INDEX = 24

#: Largest tilt sine accepted. Above 10 % strain the supercell is no longer
#: a useful model of the input crystal.
_MAX_TILT_SINE = 0.1

#: A lattice is singular when |det A| <= this times the product of the row
#: norms (a volume sine below 1e-10: the inverse loses more than 10 digits).
_SINGULAR = 1e-10

#: New fractional values within this of an integer are set to the integer.
#: This moves an atom by less than 1e-12 of an edge (below float64 noise
#: for edges up to 1e3 Å) and puts face atoms exactly on 0.
_SNAP = 1e-12

#: Relative width of a tie in the length comparisons of the searches.
_TIE = 1e-12


def check_tilt_sine(tilt_sine) -> float:
    """Return `tilt_sine` as a float after the checks of the reduction.

    Raises
    ------
    ValueError
        If it is not a finite real number in (0, 0.1].
    """
    if isinstance(tilt_sine, bool) or not isinstance(tilt_sine, numbers.Real):
        raise ValueError(f"tilt_sine: must be a real number; got {tilt_sine!r}")
    value = float(tilt_sine)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"tilt_sine: must be finite and > 0; got {value!r}")
    if value > _MAX_TILT_SINE:
        raise ValueError(f"tilt_sine: must be <= {_MAX_TILT_SINE}; got {value!r}")
    return value


def check_lattice(lattice, name: str = "lattice") -> np.ndarray:
    """Return lattice rows as a new (3, 3) float64 array after checks.

    Raises
    ------
    ValueError
        For a wrong shape, a non-finite value, a zero row or singular rows.
        The message starts with `name`.
    """
    raw = np.asarray(lattice)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name}: must be a real array; got dtype {raw.dtype}")
    edge_rows = np.array(raw, dtype=np.float64)
    if edge_rows.shape != (3, 3):
        raise ValueError(
            f"{name}: lattice rows must have shape (3, 3); got {edge_rows.shape}"
        )
    if not np.isfinite(edge_rows).all():
        raise ValueError(f"{name}: values must be finite")
    norms = np.linalg.norm(edge_rows, axis=1)
    for i in range(3):
        if norms[i] == 0.0:
            raise ValueError(f"{name}: row {i} has zero length")
    if abs(np.linalg.det(edge_rows)) <= _SINGULAR * float(np.prod(norms)):
        raise ValueError(f"{name}: the lattice rows are singular (no volume)")
    return edge_rows


def _read_only(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class OrthoPlan:
    """One reduction of lattice rows to a rectangular cell.

    Attributes
    ----------
    lattice : numpy.ndarray
        (3, 3) float64, Å. The input rows ``A``.
    matrix : numpy.ndarray
        (3, 3) int64. ``P``: the new edges as rows in the old basis.
    frame : numpy.ndarray
        (3, 3) float64. ``Q``, a proper rotation. Its rows are the new box
        axes in the old Cartesian frame; old vector ``v`` (column) has the
        unstrained new coordinates ``Q @ v``.
    sides : numpy.ndarray
        (3,) float64, Å. Side lengths ``L`` of the new cell.
    strain : numpy.ndarray
        (3,) float64. Edge strain ``|L_i e_i - b_i| / |b_i|``, the sine of
        the angle between new edge ``b_i`` and its box axis.
    linear_map : numpy.ndarray
        (3, 3) float64. ``M = inv(B) @ diag(L)``; ``v_new = v_old @ M`` for
        Cartesian row vectors. It maps each edge ``b_i`` exactly onto
        ``L_i e_i`` and agrees with the new positions.
    translations : numpy.ndarray
        (abs(det P), 3) int64. Coset representatives of ``Z^3 / Z^3 P``, in
        C order with the last index fastest.
    tilt_sine : float
        The largest tilt sine used.
    """

    lattice: np.ndarray
    matrix: np.ndarray
    frame: np.ndarray
    sides: np.ndarray
    strain: np.ndarray
    linear_map: np.ndarray
    translations: np.ndarray
    tilt_sine: float

    @property
    def n_cells(self) -> int:
        """Number of input cells in the new cell, ``abs(det P)``."""
        return int(self.translations.shape[0])

    def apply(self, fractional, shift=None) -> tuple[np.ndarray, np.ndarray]:
        """Map old fractional positions to all their images in the new cell.

        Parameters
        ----------
        fractional : array_like
            (N, 3), any real dtype. Fractional coordinates of the old cell.
        shift : array_like, optional
            (3,). Extra fractional shift in the new basis, added before the
            wrap.

        Returns
        -------
        fractional_new : numpy.ndarray
            (abs(det P) * N, 3) float64, each value in [0, 1). The images are
            translation-major: image ``t`` of atom ``j`` is row ``t * N + j``.
            Each image appears exactly once, also on faces, edges and corners.
        source : numpy.ndarray
            (abs(det P) * N,) int64. Index of the source atom of each row.
        """
        raw = np.asarray(fractional)
        if raw.dtype.kind not in "iuf":
            raise ValueError(f"fractional: must be a real array; got dtype {raw.dtype}")
        f = np.asarray(raw, dtype=np.float64)
        if f.ndim != 2 or f.shape[1] != 3:
            raise ValueError(f"fractional: must have shape (N, 3); got {f.shape}")
        if not np.isfinite(f).all():
            raise ValueError("fractional: values must be finite")
        adj, det = adjugate(self.matrix)
        g = (f[None, :, :] + self.translations[:, None, :]) @ adj.astype(np.float64)
        g = (g / det).reshape(-1, 3)
        if shift is not None:
            g = g + np.asarray(shift, dtype=np.float64).reshape(3)
        nearest = np.rint(g)
        snap = np.abs(g - nearest) <= _SNAP
        g[snap] = nearest[snap]
        g -= np.floor(g)
        g[g >= 1.0] = 0.0
        source = np.tile(np.arange(f.shape[0], dtype=np.int64), self.n_cells)
        return g, source

    def vectors(self, v, strained: bool = True) -> np.ndarray:
        """Express Cartesian vectors of the old frame in the new frame.

        Parameters
        ----------
        v : array_like
            (..., 3), Å or any unit (displacements, velocities).
        strained : bool
            True: ``v @ M``, the map that agrees with the new positions.
            False: ``v @ Q.T``, the pure rotation (strain neglected).

        Returns
        -------
        numpy.ndarray
            (..., 3) float64.
        """
        v = np.asarray(v, dtype=np.float64)
        return v @ (self.linear_map if strained else self.frame.T)


def _translations(p: np.ndarray) -> np.ndarray:
    diag = hnf_diagonal(p)
    grid = np.indices(diag, dtype=np.int64).reshape(3, -1).T
    return _read_only(np.ascontiguousarray(grid))


def _bound_error(sine_cap: float) -> ValueError:
    return ValueError(
        f"tilt_sine: no rectangular supercell with at most {MAX_SUPERCELL_FACTOR} "
        f"cells meets tilt_sine {sine_cap:g}; increase tilt_sine"
    )


def _finish(A, P, Q, sine_cap) -> OrthoPlan:
    B = P.astype(np.float64) @ A
    sides = np.einsum("ij,ij->i", B, Q)
    strain = np.linalg.norm(sides[:, None] * Q - B, axis=1) / np.linalg.norm(B, axis=1)
    linear_map = np.linalg.inv(B) @ np.diag(sides)
    return OrthoPlan(
        lattice=_read_only(A),
        matrix=_read_only(P.astype(np.int64)),
        frame=_read_only(Q),
        sides=_read_only(sides),
        strain=_read_only(strain),
        linear_map=_read_only(linear_map),
        translations=_translations(P),
        tilt_sine=sine_cap,
    )


def plan_orthobox(lattice, tilt_sine: float = DEFAULT_TILT_SINE) -> OrthoPlan:
    """Decide the reduction of lattice rows to a rectangular cell.

    Parameters
    ----------
    lattice : array_like
        (3, 3), any real dtype, Å. Lattice vectors as rows, any orientation,
        any handedness. Rows 0 and 1 span the plane normal to the beam.
    tilt_sine : float
        Largest edge strain (sine of the tilt of a new edge from its box
        axis), in (0, 0.1]. Default :data:`DEFAULT_TILT_SINE`.

    Returns
    -------
    OrthoPlan

    Notes
    -----
    ``P`` has in-plane rows 0 and 1 (third entry 0) and a beam row
    ``(m0, m1, s k)`` with ``s = sign(a2 . n)``, so ``|det P| = d k``.

    - ``d`` is the smallest index of a sublattice of ``Z a0 + Z a1`` whose
      Gauss-reduced basis has ``|cos| <= tilt_sine``; the sublattices are the
      2-D Hermite normal forms (Hart and Forcade 2008). The reduced basis has
      the smallest ``|cos|`` of all bases of its sublattice, so the test is
      exact.
    - ``k`` is the smallest positive integer for which the closest vector of
      the plane lattice brings ``s k a2`` within ``tilt_sine`` of ``n``.

    The minimum of ``|det P|`` is proven within this class of ``P``. A
    matrix outside the class tilts ``b0`` or ``b1`` out of the plane by at
    least ``|a2 . n|``; it meets the bound only with edges longer than
    ``|a2 . n| / tilt_sine``.

    Rows that are orthogonal within `tilt_sine` give ``P = diag(1, 1, s)``
    and ``L_i = |a_i|`` (to second order in the strain).

    Raises
    ------
    ValueError
        For bad `lattice` or `tilt_sine`, or when ``|det P|`` would exceed
        :data:`MAX_SUPERCELL_FACTOR` (the message names `tilt_sine`).
    """
    A = check_lattice(lattice, "lattice")
    sine_cap = check_tilt_sine(tilt_sine)
    normal = np.cross(A[0], A[1])
    n = normal / np.linalg.norm(normal)
    gram = A[:2] @ A[:2].T
    beam = beam_search(A, n, sine_cap, MAX_SUPERCELL_FACTOR)
    if beam is None:
        raise _bound_error(sine_cap)
    k, m0, m1, s = beam
    if abs(gram[0, 1]) <= sine_cap * math.sqrt(gram[0, 0] * gram[1, 1]):
        b0, b1 = np.array([1, 0]), np.array([0, 1])
    else:
        found = plane_search(gram, sine_cap, MAX_SUPERCELL_FACTOR // k)
        if found is None:
            raise _bound_error(sine_cap)
        _, r0, r1 = found
        b0, b1 = order_plane_pair(r0, r1, gram)
    P = np.array(
        [[b0[0], b0[1], 0], [b1[0], b1[1], 0], [m0, m1, s * k]], dtype=np.int64
    )
    e0 = b0.astype(np.float64) @ A[:2]
    e0 = e0 - (e0 @ n) * n
    e0 /= np.linalg.norm(e0)
    Q = np.stack([e0, np.cross(n, e0), n])
    return _finish(A, P, Q, sine_cap)


def plan_axis_aligned(lattice, tilt_sine: float = DEFAULT_TILT_SINE) -> OrthoPlan:
    """Decide a reduction whose box axes are the current Cartesian axes.

    For each axis ``j`` the edge ``b_j`` is the shortest lattice vector
    ``m @ A`` with ``max|m_i| <= MAX_AXIS_INDEX``, ``(m @ A)_j > 0`` and a
    tilt from axis ``j`` (sine) of at most `tilt_sine` (ties: smallest
    ``|m|_1``, then lexicographic). ``frame`` is the identity and
    ``L_j = (b_j)_j``.

    Parameters
    ----------
    lattice : array_like
        (3, 3), any real dtype, Å. Lattice rows.
    tilt_sine : float
        Largest edge tilt (sine), in (0, 0.1].

    Returns
    -------
    OrthoPlan

    Raises
    ------
    ValueError
        When an axis has no such lattice vector, or ``|det P|`` is 0 or above
        :data:`MAX_SUPERCELL_FACTOR`. The message names `tilt_sine` and the
        axis.
    """
    A = check_lattice(lattice, "lattice")
    sine_cap = check_tilt_sine(tilt_sine)
    r = np.arange(-MAX_AXIS_INDEX, MAX_AXIS_INDEX + 1, dtype=np.int64)
    m = np.stack(np.meshgrid(r, r, r, indexing="ij"), axis=-1).reshape(-1, 3)
    m = m[np.abs(m).sum(axis=1) > 0]
    v = m.astype(np.float64) @ A
    norm2 = np.einsum("ij,ij->i", v, v)
    edge_rows = []
    for j in range(3):
        others = [i for i in range(3) if i != j]
        perp = np.sqrt(v[:, others[0]] ** 2 + v[:, others[1]] ** 2)
        ok = np.flatnonzero((v[:, j] > 0) & (perp <= sine_cap * np.sqrt(norm2)))
        if ok.size == 0:
            raise ValueError(
                f"tilt_sine: no lattice vector with indices up to {MAX_AXIS_INDEX} "
                f"lies along box axis {j} within tilt_sine {sine_cap:g}"
            )
        ok = ok[norm2[ok] <= norm2[ok].min() * (1.0 + _TIE)]
        mm = m[ok]
        order = np.lexsort((mm[:, 2], mm[:, 1], mm[:, 0], np.abs(mm).sum(axis=1)))
        edge_rows.append(mm[order[0]])
    P = np.array(edge_rows, dtype=np.int64)
    det = abs(adjugate(P)[1])
    if det == 0 or det > MAX_SUPERCELL_FACTOR:
        raise ValueError(
            f"tilt_sine: the box edges found within tilt_sine {sine_cap:g} span "
            f"{det} cells (allowed 1..{MAX_SUPERCELL_FACTOR}); axis edges {P.tolist()}"
        )
    return _finish(A, P, np.eye(3), sine_cap)
