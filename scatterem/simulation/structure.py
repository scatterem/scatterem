r"""Crystal-structure container, structure-file readers and supercell geometry.

Written from the published formulation of the underlying crystallography and
multislice specimen setup (references below): the equations, the published
file-format definitions and the requirements of the consumers inside this
package.

What it provides
----------------
:class:`Structure`
    A plain host-side (NumPy, CPU) container for a periodic -- or
    vacuum-padded finite -- atomic model, plus readers for the three structure
    file formats used in the electron-microscopy simulation community, an
    orthorhombic-supercell construction, supercell tiling and rigid rotation.

The atom table
--------------
``Structure.atoms`` is an ``(n_atoms, 6)`` float64 array whose column layout is
a hard contract shared with ~10 consumer modules::

    column   0    1    2    3    4      5
             x    y    z    Z    occ    <u^2>

* ``x, y, z``  -- FRACTIONAL coordinates, nominally in ``[0, 1)``, i.e.
  ``r_cartesian = (x, y, z) * unitcell`` in Angstrom.
* ``Z``        -- atomic number, stored as an exact integer-valued float.
* ``occ``      -- site occupancy in ``[0, 1]``.
* ``<u^2>``    -- the ONE-dimensional mean-square thermal displacement
  ``<u_x^2> = <u_y^2> = <u_z^2>`` in Angstrom^2 (see :ref:`debye-waller`).

``Structure.unitcell`` is the ``(3,)`` vector of orthorhombic cell edge lengths
``(a, b, c)`` in Angstrom.  It is the only cell representation the rest of the
package understands; a non-orthorhombic input matrix is converted to an
orthorhombic supercell at construction time.

.. _debye-waller:

Debye-Waller conventions
------------------------
The isotropic Debye-Waller factor multiplies the elastic atomic scattering
amplitude by

.. math::   \exp(-B s^2), \qquad s = \sin\theta/\lambda = q/2,

and is related to the one-dimensional mean-square displacement by the textbook
identity

.. math::   B = 8\pi^2 \langle u^2\rangle,
            \qquad \exp(-Bs^2) = \exp(-2\pi^2 \langle u^2\rangle q^2).

Column 5 stores :math:`\langle u^2\rangle` in Angstrom^2, which is exactly what
a frozen-phonon consumer needs: it draws an independent zero-mean Gaussian
displacement of standard deviation :math:`\sqrt{\langle u^2\rangle}` for each
Cartesian axis.  Adopting the three-dimensional convention
:math:`\langle u_{tot}^2\rangle = 3\langle u^2\rangle` instead would make the
thermal smearing wrong by :math:`\sqrt{3}`.

Vector convention
-----------------
Coordinates are ROW vectors throughout.  A 3x3 cell matrix ``M`` carries the
three edge vectors **a**, **b**, **c** as its ROWS, so Cartesian positions are
``r = f @ M`` for fractional row vectors ``f``.  Rotation matrices are built in
the active column-vector convention (:func:`_rot_matrix`) and applied on the
RIGHT of row vectors, ``r' = r @ R``; since ``r @ R = (R^T r^T)^T`` this rotates
the atoms by **minus** ``theta`` about the given axis.  That sign is externally
observable (tilt-series handedness) and is part of the contract -- see
:meth:`Structure.rotate`.

References
----------
* E. J. Kirkland, *Advanced Computing in Electron Microscopy*, 2nd ed.,
  Springer 2010 -- Ch. 5 (frozen phonons / thermal diffuse scattering) and
  Ch. 6 (specimen setup, orthogonal periodically-continuable supercells).
* International Tables for Crystallography Vol. C -- the ``B`` <-> ``<u^2>``
  definition; Peng, Ren, Dudarev & Whelan, *Acta Cryst.* **A52** (1996) 456 for
  tabulated values.
* C. Ophus, *Adv. Struct. Chem. Imaging* **3**:13 (2017) and L. Rangel DaCosta
  et al., *Micron* **151** (2021) 103141 -- the Prismatic ``.xyz`` format.
* K. Momma & F. Izumi, *J. Appl. Cryst.* **44** (2011) 1272 -- VESTA, which
  writes the VASP-5/POSCAR-shaped ``.p1`` export read here.
* L. J. Allen, A. J. D'Alfonso & S. D. Findlay, *Ultramicroscopy* **151**
  (2015) 11 -- muSTEM, whose ``.xtl`` format is read here.
* O. Rodrigues (1840); see any rigid-body-kinematics text for the rotation
  formula used by :func:`_rot_matrix`.
"""

from __future__ import annotations

import os
import re
import warnings

import numpy as np

# The element-symbol table is a published data table that already lives in
# ``scattering_factors``; it is 1-based on the atomic number with a sentinel
# empty string at index 0, so ``atomic_symbol.index("Ca") == 20``.  It is
# re-exported here (unchanged) because the structure-file readers are its main
# consumer, and mirrored into a dict so that symbol -> Z costs O(1) instead of a
# linear scan of 119 strings per atom.
from scatterem.simulation.scattering_factors import atomic_symbol as _atomic_symbol

atomic_symbol = _atomic_symbol

_SYMBOL_TO_Z = {sym: z for z, sym in enumerate(_atomic_symbol) if sym}
_SYMBOL_TO_Z_LOWER = {sym.lower(): z for z, sym in enumerate(_atomic_symbol) if sym}

#: Index dtype for the maps returned by :func:`find_equivalent_sites` and
#: :func:`_remove_common_factors`.
_int = np.int32
#: Storage dtype for coordinates, cell edges and every other float here.
_float = np.float64

_ELEMENT_RE = re.compile(r"[A-Za-z]+")

#: Safety cap on the atom count an orthorhombic-supercell construction may
#: produce.  The pseudo-rational tiling can legitimately multiply the atom count
#: by up to ``(1/EPS)**3``; without a cap a badly conditioned monoclinic cell
#: silently asks for ~10^6 times more atoms than the input.
_MAX_SUPERCELL_ATOMS = 50_000_000

__all__ = [
    "Structure",
    "atomic_symbol",
    "find_equivalent_sites",
]


# ---------------------------------------------------------------------------
# Rotations
# ---------------------------------------------------------------------------


def _rot_matrix(theta, u):
    r"""Rodrigues rotation matrix for angle ``theta`` about the axis ``u``.

    For a unit axis :math:`\hat n` and angle :math:`t` the active rotation of a
    COLUMN vector :math:`v` is

    .. math::

        R(t,\hat n)\,v = v\cos t + (\hat n \times v)\sin t
                         + \hat n\,(\hat n\cdot v)(1-\cos t),

    which in matrix form is

    .. math::

        R(t,\hat n) = I\cos t + \sin t\,[\hat n]_\times
                      + (1-\cos t)\,\hat n\,\hat n^{\mathsf T},
        \qquad
        [\hat n]_\times = \begin{pmatrix}0&-n_3&n_2\\ n_3&0&-n_1\\
                                          -n_2&n_1&0\end{pmatrix}.

    Written out with :math:`c=\cos t`, :math:`s=\sin t`::

        [[ c + n1^2(1-c),      n1 n2 (1-c) - n3 s,  n1 n3 (1-c) + n2 s ],
         [ n2 n1 (1-c) + n3 s,  c + n2^2(1-c),      n2 n3 (1-c) - n1 s ],
         [ n3 n1 (1-c) - n2 s,  n3 n2 (1-c) + n1 s,  c + n3^2(1-c)     ]]

    The result is a proper rotation: :math:`R R^{\mathsf T} = I`,
    :math:`\det R = +1`, and
    :math:`R(t,\hat n)^{\mathsf T} = R(-t,\hat n) = R(t,-\hat n)`.  It is
    numerically identical to
    ``scipy.spatial.transform.Rotation.from_rotvec(t * n_hat).as_matrix()``.

    Parameters
    ----------
    theta : float
        Rotation angle in RADIANS.  Scalar only -- this is not vectorised over
        angles.
    u : array_like, shape (3,)
        Rotation axis.  Normalised internally, so its magnitude is irrelevant.

    Returns
    -------
    ndarray, shape (3, 3), dtype float64

    Notes
    -----
    Argument order is ``(theta, axis)``.  Applying the result to ROW vectors on
    the right (``r @ R``, the convention used by :meth:`Structure.rotate` and by
    ``builders.geometry.rotate_about_axis``) rotates by ``-theta``.
    """
    u = np.asarray(u, dtype=_float).reshape(3)
    norm = np.sqrt(u @ u)
    if norm == 0.0:
        raise ValueError("_rot_matrix: rotation axis must be a nonzero vector")
    n1, n2, n3 = u / norm

    c = np.cos(theta)
    s = np.sin(theta)
    C = 1.0 - c

    return np.array(
        [
            [c + n1 * n1 * C, n1 * n2 * C - n3 * s, n1 * n3 * C + n2 * s],
            [n2 * n1 * C + n3 * s, c + n2 * n2 * C, n2 * n3 * C - n1 * s],
            [n3 * n1 * C - n2 * s, n3 * n2 * C + n1 * s, c + n3 * n3 * C],
        ],
        dtype=_float,
    )


def _remove_common_factors(nums):
    """Divide a set of integers by their greatest common divisor.

    Returns ``nums / gcd(nums)``, i.e. the same ratio expressed in lowest
    terms.  Used to reduce the ``(n1, n2)`` tiling pair of the pseudo-rational
    orthogonalisation (:meth:`Structure._pseudo_rational_tiling`).

    Parameters
    ----------
    nums : array_like of int

    Returns
    -------
    ndarray of int32
        A freshly allocated array; the input is never modified.  A single
        division by ``gcd`` always suffices, because ``gcd(n/g) == 1`` by
        definition of the greatest common divisor.
    """
    arr = np.asarray(nums, dtype=np.int64)
    g = int(np.gcd.reduce(arr.reshape(-1))) if arr.size else 1
    if g in (0, 1):
        return arr.astype(_int)
    return (arr // g).astype(_int)


# ---------------------------------------------------------------------------
# Equivalent (co-sited) atoms
# ---------------------------------------------------------------------------


def find_equivalent_sites(positions, EPS=1e-3, wrap=False):
    """Group atoms that occupy the same crystallographic site.

    Physics
    -------
    A partially-occupied site shared by two or more chemical species is
    represented in a structure file by two or more atom records at (nearly)
    identical coordinates.  Under the frozen-phonon / frozen-lattice
    approximation those records are ONE physical atom, so in a given phonon
    configuration they must all receive the SAME random thermal displacement --
    otherwise the thermal-diffuse scattering from that site is decorrelated and
    spuriously reduced.  ``make_potential`` draws one independent Gaussian
    displacement per atom row and then gathers that array through the index map
    returned here, which makes the displacements of co-sited atoms identical.

    Algorithm
    ---------
    Atoms closer than ``EPS`` are linked, the connected components of the
    resulting graph are extracted, and every atom is mapped to the SMALLEST
    atom index in its component (a star, not a chain).  A single gather through
    the returned map therefore gives one common displacement per group for any
    site multiplicity, not just for the two-fold case.

    The close pairs are found with a k-d tree fixed-radius query, so the cost is
    ``O(n log n)`` in time and ``O(n + n_pairs)`` in memory.  An all-pairs
    distance matrix would need ``n(n-1)/2`` doubles -- about 40 GB at 10^5
    atoms -- and is simply not usable on a realistic supercell.

    Parameters
    ----------
    positions : ndarray, shape (n_atoms, 3)
        FRACTIONAL coordinates (a slice of ``Structure.atoms[:, :3]``).
    EPS : float, default 1e-3
        Coincidence threshold, in the same (fractional, dimensionless) units as
        ``positions``.  The implied physical tolerance is therefore roughly
        ``EPS`` times the cell edge: ~0.008 A in an 8 A cell but ~0.05 A in a
        50 A supercell.
    wrap : bool, default False
        If True, measure distances with the minimum-image convention on the
        unit fractional cube, so that sites related by a lattice translation
        (fractional ``z = 0`` and ``z = 1``) are recognised as one site.  The
        default is False because coordinates are not required to be reduced to
        ``[0, 1)`` everywhere in this package.

    Returns
    -------
    ndarray, shape (n_atoms,), dtype int32
        ``out[i]`` is the representative index of atom ``i``; the identity map
        when no pair is closer than ``EPS``.  Every entry is a valid index in
        ``[0, n_atoms)``.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    pos = np.ascontiguousarray(positions, dtype=_float)
    natoms = pos.shape[0]
    if natoms < 2:
        return np.arange(natoms, dtype=_int)

    if wrap:
        tree = cKDTree(np.mod(pos, 1.0), boxsize=1.0)
    else:
        tree = cKDTree(pos)
    pairs = tree.query_pairs(EPS, output_type="ndarray")

    if pairs.shape[0] == 0:
        return np.arange(natoms, dtype=_int)

    # Connected components of the coincidence graph, then the minimum member of
    # each component as its representative.
    ones = np.ones(pairs.shape[0], dtype=np.int8)
    graph = coo_matrix(
        (ones, (pairs[:, 0], pairs[:, 1])), shape=(natoms, natoms)
    ).tocsr()
    ncomp, labels = connected_components(graph, directed=False)

    rep = np.full(ncomp, natoms, dtype=np.int64)
    np.minimum.at(rep, labels, np.arange(natoms, dtype=np.int64))
    return rep[labels].astype(_int)


#: Backwards-compatible private alias (the historical spelling used by
#: ``potentials.make_potential``).
_find_equivalent_sites = find_equivalent_sites


# ---------------------------------------------------------------------------
# Cell geometry helpers
# ---------------------------------------------------------------------------


def _is_orthorhombic(M, EPS):
    """True if the 3x3 cell matrix ``M`` is axis-aligned orthorhombic.

    The test is on the LARGEST off-diagonal element,
    ``max_{i != j} |M_ij| < EPS`` (an absolute threshold, in Angstrom).  Testing
    the largest element rather than the signed sum matters: equal-and-opposite
    shears -- e.g. rows ``(4, 0.5, 0)``, ``(-0.5, 5, 0)``, ``(0, 0, 6)`` -- sum
    to zero and would otherwise be accepted as orthorhombic, silently
    discarding a 0.5 A shear.
    """
    off = M - np.diag(np.diag(M))
    return bool(np.max(np.abs(off)) < EPS)


def _align_cell_to_axes(M, EPS):
    r"""Rigidly rotate a cell matrix so **a** lies along +x and **b** in the xy plane.

    This is a rigid rotation of the whole crystal, so the FRACTIONAL
    coordinates of the atoms are unchanged; only the cell matrix is rewritten.

    Stage 1.  The angle between **a** and :math:`\hat x` is
    :math:`t_1 = \arccos(a_x/|a|)`, and the rotation that carries **a** onto
    :math:`\hat x` is :math:`R(t_1, \hat n_1)` with
    :math:`\hat n_1 = \widehat{a \times \hat x}`.  (Check: with
    :math:`\hat n_1 \perp a` Rodrigues gives
    :math:`R a = |a|\hat a\cos t_1 + |a|(\hat x - \hat a\cos t_1) = |a|\hat x`.)
    With rows as edge vectors this is applied as ``M <- M @ R.T``.

    Stage 2.  With **a** along x, the residual out-of-plane tilt of **b** is
    :math:`t_2 = \operatorname{atan2}(b_z, b_y)` and
    :math:`R(-t_2, \hat x)` brings **b** into the xy plane while leaving **a**
    untouched.

    Both stages are skipped when the corresponding angle is below ``EPS``.
    """
    M = np.array(M, dtype=_float, copy=True)

    # --- stage 1: a -> +x ------------------------------------------------
    a = M[0]
    a_norm = np.sqrt(a @ a)
    if a_norm > 0.0:
        t1 = np.arccos(np.clip(a[0] / a_norm, -1.0, 1.0))
        if t1 > EPS:
            axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
            if np.sqrt(axis @ axis) <= 1e-12 * a_norm:
                # a is antiparallel to x: the cross product degenerates, and a
                # half turn about any perpendicular axis does the job.
                axis = np.array([0.0, 1.0, 0.0])
            M = M @ _rot_matrix(t1, axis).T

    # --- stage 2: b -> xy plane -----------------------------------------
    b = M[1]
    t2 = np.arctan2(b[2], b[1])
    if abs(t2) > EPS:
        M = M @ _rot_matrix(-t2, np.array([1.0, 0.0, 0.0])).T

    return M


def _orthogonalise_pair(M, dim1, dim2, EPS):
    r"""Plan one pseudo-rational orthogonalisation pass on a cell matrix.

    Makes edge vector ``dim2`` perpendicular to edge vector ``dim1`` by tiling
    the cell by a rational approximation of their overlap and then subtracting
    the exact projection.

    Let :math:`a = M[\mathrm{dim}_1]`, :math:`b = M[\mathrm{dim}_2]` and define
    the reduced overlap

    .. math::  r = \frac{a\cdot b}{|a|^2}

    (:math:`r|a|` is the length of the projection of **b** on
    :math:`\hat a`).  If :math:`|r| < \varepsilon` the residual shear that the
    final diagonal extraction discards is below :math:`\varepsilon |a|`
    Angstrom and the pass does nothing.

    Otherwise approximate :math:`r` by a rational number with denominator at
    most :math:`1/\varepsilon`,

    .. math::  n_1 = \operatorname{round}(|r|/\varepsilon), \quad
               n_2 = \operatorname{round}(1/\varepsilon), \quad
               (n_1, n_2) \leftarrow (n_1, n_2)/\gcd(n_1, n_2),

    and tile by :math:`n_1` along ``dim1`` and :math:`n_2` along ``dim2``.  In
    the tiled cell :math:`a' = n_1 a`, :math:`b' = n_2 b` and

    .. math::  \frac{a'\cdot b'}{|a'|^2} = \frac{n_2}{n_1} r \approx 1,

    i.e. **b'** overhangs **a'** by almost exactly ONE tiled a-period, so
    subtracting that period is (to the rational-approximation error) a
    lattice-preserving operation.  The Gram-Schmidt step

    .. math::  b'' = b' - \frac{a'\cdot b'}{a'\cdot a'}\,a'

    then leaves :math:`b'' \perp a'`.

    Returns
    -------
    (M_new, n1, n2)
        ``M_new`` is the tiled-and-orthogonalised matrix; ``n1``/``n2`` are the
        tiling factors that must be applied to the atoms along ``dim1``/``dim2``
        for ``M_new`` to describe them.  ``(M.copy(), 1, 1)`` if the pair was
        already orthogonal.

    Notes
    -----
    The early-out threshold is on ``r`` and not on the true direction cosine
    ``|a.b| / (|a||b|)``.  ``r`` is the quantity the rational approximation is
    built from -- an early-out on the cosine could leave ``r`` below ``EPS``,
    where ``round(|r|/EPS)`` degenerates to zero -- and it is the quantity that
    bounds the discarded shear in Angstrom.  On a borderline cell with
    ``|b| >> |a|`` the two criteria differ.
    """
    M = np.array(M, dtype=_float, copy=True)
    a = M[dim1]
    b = M[dim2]
    aa = a @ a
    if aa == 0.0:
        raise ValueError("cell edge vector %d has zero length" % dim1)
    r = (a @ b) / aa
    if abs(r) < EPS:
        return M, 1, 1

    n1, n2 = _remove_common_factors([int(round(abs(r) / EPS)), int(round(1.0 / EPS))])
    n1 = max(int(n1), 1)
    n2 = max(int(n2), 1)

    M[dim1] = a * n1
    M[dim2] = b * n2
    a = M[dim1]
    M[dim2] = M[dim2] - ((a @ M[dim2]) / (a @ a)) * a
    return M, n1, n2


# ---------------------------------------------------------------------------
# File readers
# ---------------------------------------------------------------------------


def _bulk_columns(lines, float_cols, str_col=None):
    """Parse selected columns of a whitespace-separated text block.

    The whole atom block goes through NumPy's C text reader in one call per
    dtype, instead of one ``str.split`` plus half a dozen Python ``float()``
    calls per atom.  On a 200k-atom file that is the difference between ~0.2 s
    and ~0.8 s.  A ragged block (lines with differing field counts) falls back
    to a per-line split, which is slower but tolerant.

    Parameters
    ----------
    lines : list of str
    float_cols : sequence of int
        Zero-based indices of the numeric fields to return, in order.
    str_col : int, optional
        Index of one text field (the element label) to return as well.

    Returns
    -------
    (ndarray (n, len(float_cols)) float64, ndarray (n,) str or None)
    """
    if not lines:
        raise ValueError("structure file contains no atom records")
    float_cols = tuple(int(c) for c in float_cols)
    ncol_min = max(float_cols + ((str_col,) if str_col is not None else ())) + 1
    try:
        table = np.loadtxt(
            lines, dtype=_float, usecols=float_cols, comments=None, ndmin=2
        )
        labels = (
            None
            if str_col is None
            else np.loadtxt(
                lines, dtype=str, usecols=(int(str_col),), comments=None, ndmin=1
            )
        )
    except ValueError:
        # Ragged block: split line by line and keep the fields we need.
        rows = [ln.split() for ln in lines]
        if min(len(r) for r in rows) < ncol_min:
            raise ValueError(
                "atom records need at least %d whitespace-separated fields; "
                "the shortest has %d" % (ncol_min, min(len(r) for r in rows))
            ) from None
        table = np.array([[r[c] for c in float_cols] for r in rows], dtype=_float)
        labels = None if str_col is None else np.array([r[str_col] for r in rows])
    if table.shape[0] != len(lines):
        raise ValueError(
            "expected %d atom records, parsed %d" % (len(lines), table.shape[0])
        )
    return table, labels


def _symbols_to_Z(labels):
    """Resolve site labels such as ``'Ca1'`` or ``'O12'`` to atomic numbers.

    The species is the leading maximal run of ASCII letters of the label, looked
    up in :data:`atomic_symbol` (whose list index IS the atomic number).  Only
    the handful of DISTINCT labels is resolved; the answer is broadcast back to
    the atoms with an inverse-index gather, so the cost is independent of the
    atom count.

    An exact (case-sensitive) match is tried first, then a case-insensitive one,
    so that ``'CA1'`` still resolves to calcium rather than raising.
    """
    uniq, inverse = np.unique(np.asarray(labels), return_inverse=True)
    Z = np.empty(uniq.size, dtype=_float)
    for i, label in enumerate(uniq):
        m = _ELEMENT_RE.match(str(label))
        if m is None:
            raise ValueError(
                "cannot read an element symbol from the site label %r" % (label,)
            )
        sym = m.group(0)
        z = _SYMBOL_TO_Z.get(sym)
        if z is None:
            z = _SYMBOL_TO_Z_LOWER.get(sym.lower())
        if z is None:
            raise ValueError(
                "unknown element symbol %r (from site label %r)" % (sym, label)
            )
        Z[i] = z
    return Z[inverse.reshape(-1)]


def _read_xyz(lines):
    """Read a Prismatic XYZ file (already split into lines, title consumed).

    Layout::

        line 1   title
        line 2   a b c                       cell edge lengths [Angstrom]
        line 3+  Z  x  y  z  occ  <u^2>      one atom per line
        ...      -1                          terminator (EOF also accepted)

    The coordinates are CARTESIAN Angstrom in the Prismatic specification; see
    :meth:`Structure.fromfile` for how ``atomic_coordinates`` handles that.

    Reference: Ophus, Adv. Struct. Chem. Imaging 3:13 (2017); Rangel DaCosta
    et al., Micron 151 (2021) 103141.
    """
    cell = np.array(lines[1].split()[:3], dtype=_float)
    if cell.size != 3:
        raise ValueError(".xyz line 2 must hold three cell edge lengths")

    body = []
    for ln in lines[2:]:
        s = ln.strip()
        if s == "-1":
            break
        if s:
            body.append(s)

    # Prismatic column order is Z x y z occ <u^2>; ask for [x, y, z, Z, occ,
    # <u^2>] directly so no permutation copy is needed afterwards.
    table, _ = _bulk_columns(body, (1, 2, 3, 0, 4, 5))
    return cell, table[:, :3], table[:, 3], table[:, 4], table[:, 5]


def _read_p1(lines):
    """Read a VESTA P1 export (VASP-5/POSCAR-shaped; title consumed).

    Layout::

        line 1   title
        line 2   global scale factor
        line 3-5 3x3 cell matrix, rows = a, b, c        [Angstrom]
        line 6   element symbols           (not used: the per-atom labels are
                                            authoritative, since they also
                                            carry the site distinction)
        line 7   per-species atom counts    (only the sum is used)
        line 8   'Direct' | 'Cartesian'     (optionally preceded by
                                             'Selective dynamics')
        then     x  y  z  label  occ  <u^2>  ...        one atom per line

    Species come from the per-atom labels, which is what carries the site
    distinction (``Ca1``, ``O3``, ...) in a P1 export.

    The POSCAR global scale factor is honoured: a positive value multiplies the
    cell, a negative value gives the target cell VOLUME (VASP convention).  The
    ``Direct``/``Cartesian`` keyword is honoured too -- a Cartesian file is
    converted to fractional coordinates through the inverse cell matrix.

    Reference: Momma & Izumi, J. Appl. Cryst. 44 (2011) 1272.
    """
    scale = float(lines[1].split()[0])
    M = np.array([ln.split()[:3] for ln in lines[2:5]], dtype=_float)
    if scale < 0.0:
        # Negative scale means "this is the target volume".
        volume = abs(float(np.linalg.det(M)))
        scale = (abs(scale) / volume) ** (1.0 / 3.0) if volume > 0.0 else 1.0
    if scale != 1.0:
        M = M * scale

    counts = np.array(lines[6].split(), dtype=np.int64)
    natoms = int(counts.sum())

    first = 8
    mode = lines[7].strip()
    if mode[:1] in ("S", "s"):  # optional 'Selective dynamics' line
        mode = lines[8].strip()
        first = 9
    cartesian = mode[:1] in ("C", "c", "K", "k")

    body = [ln for ln in lines[first:] if ln.strip()][:natoms]
    if len(body) < natoms:
        raise ValueError(
            "the per-species counts declare %d atoms but only %d atom records "
            "follow" % (natoms, len(body))
        )

    table, labels = _bulk_columns(body, (0, 1, 2, 4, 5), str_col=3)
    coords = table[:, :3]
    Z = _symbols_to_Z(labels)
    occ = table[:, 3]
    dwf = table[:, 4]

    if cartesian:
        coords = np.mod(coords @ np.linalg.inv(M), 1.0)

    return M, coords, Z, occ, dwf


def _read_xtl(lines):
    """Read a muSTEM XTL file (title consumed).

    Layout::

        line 1   title
        line 2   a b c                       cell edge lengths [Angstrom]
        line 3   accelerating voltage [kV]   (not used by this container)
        line 4   number of species blocks
        then per block:
                 element symbol             (informational)
                 count  Z  occ  <u^2>
                 count lines of  x y z      fractional coordinates

    The per-block ``Z``, ``occ`` and ``<u^2>`` are broadcast to every atom of
    the block.  Blocks are accumulated in a list and concatenated once, so the
    cost is linear in the atom count.

    Reference: Allen, D'Alfonso & Findlay, Ultramicroscopy 151 (2015) 11 (muSTEM).
    """

    def nonblank(i):
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines):
            raise ValueError(".xtl file ended in the middle of a species block")
        return i

    i = nonblank(1)
    cell = np.array(lines[i].split()[:3], dtype=_float)
    # The next line is the accelerating voltage: a beam parameter, not a
    # property of the specimen, so it is not stored on the container.
    i = nonblank(nonblank(i + 1) + 1)
    nspecies = int(float(lines[i].split()[0]))

    coord_blocks = []
    Z_blocks = []
    occ_blocks = []
    dwf_blocks = []

    for _ in range(nspecies):
        i = nonblank(i + 1)  # element symbol line (informational)
        i = nonblank(i + 1)
        head = lines[i].split()
        count = int(float(head[0]))
        Z_b, occ_b, dwf_b = (float(head[1]), float(head[2]), float(head[3]))

        block = lines[i + 1 : i + 1 + count]
        if len(block) == count and all(ln.strip() for ln in block):
            i += count  # common case: no blank lines inside the block
        else:  # tolerate blank lines, one line at a time
            block = []
            j = i + 1
            while len(block) < count:
                j = nonblank(j)
                block.append(lines[j])
                j += 1
            i = j - 1

        coord_blocks.append(_bulk_columns(block, (0, 1, 2))[0])
        Z_blocks.append(np.full(count, Z_b, dtype=_float))
        occ_blocks.append(np.full(count, occ_b, dtype=_float))
        dwf_blocks.append(np.full(count, dwf_b, dtype=_float))

    if not coord_blocks:
        raise ValueError(".xtl file declares no species blocks")
    coords = np.concatenate(coord_blocks, axis=0)
    return (
        cell,
        coords,
        np.concatenate(Z_blocks),
        np.concatenate(occ_blocks),
        np.concatenate(dwf_blocks),
    )


# ---------------------------------------------------------------------------
# The container
# ---------------------------------------------------------------------------


class Structure:
    """A periodic (or vacuum-padded finite) atomic model.

    Parameters
    ----------
    unitcell : array_like
        Either the three orthorhombic cell edge lengths ``(a, b, c)`` in
        Angstrom, or a 3x3 matrix whose ROWS are the cell edge vectors **a**,
        **b**, **c** in Cartesian Angstrom.  A matrix that is axis-aligned
        orthorhombic (to ``EPS``) is reduced to its diagonal; any other matrix
        triggers :meth:`_orthorhombic_supercell`, which tiles the crystal into
        an orthorhombic supercell.
    atoms : array_like, shape (n_atoms, 4)
        ``[x, y, z, Z]`` -- fractional coordinates and atomic number.  Exactly
        four columns; ``occ`` and ``dwf`` are appended to give the six-column
        table documented in the module docstring.
    dwf : array_like, shape (n_atoms,) or None
        One-dimensional mean-square thermal displacement ``<u^2>`` [A^2].
        ``None`` gives every atom 0.01 A^2.  NOTE the argument ORDER: ``dwf`` is
        the third positional argument and ``occ`` the fourth, which is the
        reverse of the stored column order (column 4 is ``occ``, column 5 is
        ``<u^2>``).  Pass them by keyword.
    occ : array_like, shape (n_atoms,) or None
        Site occupancies.  ``None`` gives every atom 1.0.
    Title : str
        Free-form title; the first line of the source file for
        :meth:`fromfile`.
    EPS : float, default 1e-2
        Tolerance for the orthorhombicity test and, if it fails, for the
        pseudo-rational supercell construction.

    Attributes
    ----------
    unitcell : ndarray, shape (3,), float64
        Orthorhombic cell edge lengths in Angstrom.  Freshly allocated and
        writable -- it never aliases the caller's array.
    atoms : ndarray, shape (n_atoms, 6), float64
        ``[x, y, z, Z, occ, <u^2>]``; see the module docstring.
    Title : str
    fractional_occupancy : numpy.bool_
        True iff any atom has ``|occ - 1| > 1e-3``.  Consumers use it to gate
        the correlated-displacement path (see :func:`find_equivalent_sites`).
        It is computed once, at construction; :meth:`tile` and :meth:`rotate`
        cannot change occupancies, so it stays valid.

    Notes
    -----
    Host NumPy only: this module never imports torch, never sees a device and
    performs no Fourier transform.  Conversion to tensors happens in the
    consumers.
    """

    def __init__(self, unitcell, atoms, dwf, occ=None, Title="", EPS=1e-2):
        atoms = np.asarray(atoms, dtype=_float)
        if atoms.ndim != 2 or atoms.shape[1] != 4:
            raise ValueError(
                "atoms must have shape (n_atoms, 4) = [x, y, z, Z]; got %r. "
                "The stored table gains the occupancy and <u^2> columns here, "
                "so passing a wider array would silently break the six-column "
                "layout that the rest of the package relies on." % (atoms.shape,)
            )
        natoms = atoms.shape[0]

        occ_col = (
            np.ones(natoms, dtype=_float)
            if occ is None
            else np.asarray(occ, dtype=_float).reshape(natoms)
        )
        dwf_col = (
            np.full(natoms, 0.01, dtype=_float)
            if dwf is None
            else np.asarray(dwf, dtype=_float).reshape(natoms)
        )

        table = np.empty((natoms, 6), dtype=_float)
        table[:, :4] = atoms
        table[:, 4] = occ_col
        table[:, 5] = dwf_col
        self.atoms = table

        self.Title = str(Title)
        # numpy.bool_, as the consumers expect; `is True` comparisons on it fail.
        self.fractional_occupancy = np.any(np.abs(table[:, 4] - 1.0) > 1e-3)

        cell = np.array(unitcell, dtype=_float)
        if cell.ndim == 1 and cell.size == 3:
            self.unitcell = cell
        elif cell.shape == (3, 3):
            if _is_orthorhombic(cell, EPS):
                self.unitcell = np.diag(cell).copy()
            else:
                self.unitcell = cell
                self._orthorhombic_supercell(EPS)
        else:
            raise ValueError(
                "unitcell must be three edge lengths or a 3x3 matrix of edge "
                "vectors (rows); got shape %r" % (cell.shape,)
            )

    # -- construction helpers ------------------------------------------------

    @classmethod
    def _assemble(cls, unitcell, atoms, Title, fractional_occupancy):
        """Build an instance from already-validated parts, with no copying."""
        obj = cls.__new__(cls)
        obj.unitcell = unitcell
        obj.atoms = atoms
        obj.Title = Title
        obj.fractional_occupancy = fractional_occupancy
        return obj

    def __repr__(self):  # pragma: no cover - diagnostics only
        uc = np.asarray(self.unitcell)
        return "Structure(%d atoms, cell=[%.4g, %.4g, %.4g] A, Title=%r)" % (
            self.atoms.shape[0],
            uc.flat[0],
            uc.flat[1],
            uc.flat[2],
            self.Title,
        )

    # -- readers -------------------------------------------------------------

    @classmethod
    def fromfile(
        cls,
        fnam,
        temperature_factor_units="ums",
        atomic_coordinates="fractional",
        EPS=1e-2,
        T=None,
    ):
        r"""Read a structure file and return a :class:`Structure`.

        The format is chosen from the lowercased filename extension:
        ``.xyz`` (Prismatic), ``.p1`` (VESTA P1 / POSCAR-shaped) or ``.xtl``
        (muSTEM).  The first line of every format is the title.

        Parameters
        ----------
        fnam : str or os.PathLike
        temperature_factor_units : {'ums', 'urms', 'B'}, default 'ums'
            Units of the thermal parameter in the file.  With
            :math:`B = 8\pi^2\langle u^2\rangle`:

            ============  ===========================================
            ``'ums'``     already :math:`\langle u^2\rangle` [A^2]
            ``'urms'``    :math:`u_{rms}` [A]; squared on read
            ``'B'``       crystallographic B [A^2]; divided by 8 pi^2
            ============  ===========================================

        atomic_coordinates : {'fractional', 'cartesian'}, default 'fractional'
            Interpretation of the coordinates in the FILE.

            .. warning::

               The Prismatic ``.xyz`` specification stores CARTESIAN Angstrom,
               so physically correct ``.xyz`` loading needs
               ``atomic_coordinates='cartesian'``.  The default is
               ``'fractional'`` for backwards compatibility, and with that
               default the Angstrom values are stored unchanged in the
               fractional slot -- every consumer then treats e.g. 4.0725 A as
               the fractional coordinate 4.0725, which the downstream
               ceiling-then-modulo pixel mapping and the ``z mod 1`` slice
               assignment quietly alias.  Existing parity tests lock this in.

            Unlike the historical behaviour, an unrecognised value now raises
            instead of silently doing nothing.
        EPS : float, default 1e-2
            Passed to the constructor (orthorhombicity / supercell tolerance).
        T : array_like, shape (3, 3), optional
            Optional cell transformation (zone-axis or supercell redefinition),
            applied consistently in the row-vector convention:

            .. math::

               r_{cart} = f M, \qquad M' = M T^{\mathsf T},
               \qquad f' = r_{cart}\,(M')^{-1} \bmod 1.

            A length-3 cell is promoted to ``diag(a, b, c)`` first, so ``T``
            works with every reader.

        Returns
        -------
        Structure
        """
        fnam = os.fspath(fnam)
        ext = os.path.splitext(fnam)[1].lower()

        # A context manager, so a parse error cannot leak the descriptor.
        with open(fnam, "r") as f:
            lines = f.read().splitlines()
        if not lines:
            raise ValueError("%s is empty" % fnam)
        Title = lines[0].strip()

        if ext == ".xyz":
            cell, coords, Z, occ, dwf = _read_xyz(lines)
        elif ext == ".p1":
            cell, coords, Z, occ, dwf = _read_p1(lines)
        elif ext == ".xtl":
            cell, coords, Z, occ, dwf = _read_xtl(lines)
        else:
            raise ValueError(
                "unrecognised structure file extension '%s' "
                "(expected '.xyz', '.p1' or '.xtl')" % ext
            )

        # 1. thermal parameter -> one-dimensional <u^2> in A^2
        if temperature_factor_units == "ums":
            pass
        elif temperature_factor_units == "urms":
            dwf = dwf * dwf
        elif temperature_factor_units == "B":
            dwf = dwf / (8.0 * np.pi**2)
        else:
            raise ValueError(
                "temperature_factor_units must be 'ums' (<u^2> [A^2]), "
                "'urms' (u_rms [A]) or 'B' (Debye-Waller B [A^2]); got %r"
                % (temperature_factor_units,)
            )

        # 2. coordinate convention
        if atomic_coordinates == "cartesian":
            if cell.ndim == 2:
                # Correct for a general cell, not only a diagonal one.
                coords = np.mod(coords @ np.linalg.inv(cell), 1.0)
            else:
                coords = np.mod(coords / cell, 1.0)
        elif atomic_coordinates != "fractional":
            raise ValueError(
                "atomic_coordinates must be 'fractional' or 'cartesian'; got %r"
                % (atomic_coordinates,)
            )

        # 3. optional cell transformation
        if T is not None:
            T = np.asarray(T, dtype=_float).reshape(3, 3)
            M = cell if cell.ndim == 2 else np.diag(cell)
            cart = coords @ M
            M = M @ T.T
            coords = np.mod(cart @ np.linalg.inv(M), 1.0)
            cell = M

        atoms = np.empty((coords.shape[0], 4), dtype=_float)
        atoms[:, :3] = coords
        atoms[:, 3] = Z
        return cls(cell, atoms, dwf=dwf, occ=occ, Title=Title, EPS=EPS)

    # -- supercell tiling ----------------------------------------------------

    def tile(self, x=1, y=1, z=1):
        r"""Build the ``x`` by ``y`` by ``z`` supercell IN PLACE.

        The cell edge along dimension :math:`d` becomes :math:`n_d` times
        longer and every atom is replicated at all integer lattice
        translations, with the coordinates re-expressed as fractions of the NEW
        cell:

        .. math::

           f^{new}_d(\text{atom}, m) = \frac{f_d(\text{atom}) + m_d}{n_d},
           \qquad 0 \le m_d < n_d .

        Output ORDERING is part of the contract: copies are enumerated in C
        order over :math:`m = (m_x, m_y, m_z)` -- flat index
        :math:`m_x n_y n_z + m_y n_z + m_z` -- and the original atom order is
        preserved within each copy.  The table is therefore copy-major and
        atom-minor: rows ``[c*n_atoms, (c+1)*n_atoms)`` belong to copy ``c``.
        Consumers that pair a per-atom weight vector with ``structure.atoms``
        rely on this.

        Columns 3-5 (``Z``, ``occ``, ``<u^2>``) are copied verbatim into every
        replica; neither ``Title`` nor ``fractional_occupancy`` changes.

        Returns
        -------
        Structure
            ``self`` -- this method MUTATES the structure.  Copy first if you
            need to keep the untiled model.

        Notes
        -----
        Works whether ``unitcell`` is currently a length-3 vector (each edge
        length scaled) or a 3x3 matrix (each ROW scaled by its own factor); the
        latter is needed while the orthorhombic-supercell machinery is running.
        """
        n = np.array([int(x), int(y), int(z)], dtype=np.int64)
        if np.any(n < 1):
            raise ValueError("tiling factors must be >= 1; got %r" % (tuple(n),))

        uc = np.asarray(self.unitcell, dtype=_float)
        if uc.ndim == 2:
            self.unitcell = uc * n[:, np.newaxis]
        else:
            self.unitcell = uc * n

        ncopies = int(n.prod())
        if ncopies == 1:
            return self

        atoms = self.atoms
        natoms = atoms.shape[0]

        # Loop-invariant work done once: the integer offset table, in C order
        # over (m_x, m_y, m_z), and the single division by the tiling vector.
        offsets = np.stack(
            np.meshgrid(
                np.arange(n[0], dtype=_float),
                np.arange(n[1], dtype=_float),
                np.arange(n[2], dtype=_float),
                indexing="ij",
            ),
            axis=-1,
        ).reshape(ncopies, 3)

        out = np.empty((ncopies * natoms, 6), dtype=_float)
        nf = n.astype(_float)

        # The division by the tiling vector is hoisted out of the copy loop:
        # ``(f + m)/n`` is evaluated as ``f/n + m/n``, one divide over the
        # SOURCE table instead of one over every copy.  Floating-point division
        # is the expensive operation here (it is the bulk of the runtime for a
        # large supercell) and the two groupings agree to within one unit in
        # the last place, i.e. ~1e-16 of a cell edge.
        src = atoms[:, :3] / nf
        offsets /= nf

        if natoms >= 256:
            # Enough work per copy to amortise the loop: do the arithmetic in a
            # small CONTIGUOUS scratch buffer (SIMD-friendly, cache-resident)
            # and pay the strided write only once per column block.
            tail = atoms[:, 3:]
            scratch = np.empty((natoms, 3), dtype=_float)
            for c in range(ncopies):
                np.add(src, offsets[c], out=scratch)
                block = out[c * natoms : (c + 1) * natoms]
                block[:, :3] = scratch
                block[:, 3:] = tail
        else:
            # Many tiny copies: one broadcast beats ``ncopies`` NumPy calls.
            # ``view`` reshapes the OUTPUT buffer itself -- reshaping a column
            # slice would be non-contiguous and would silently produce a copy.
            view = out.reshape(ncopies, natoms, 6)
            np.add(
                src[np.newaxis, :, :],
                offsets[:, np.newaxis, :],
                out=view[:, :, :3],
            )
            view[:, :, 3:] = atoms[np.newaxis, :, 3:]

        self.atoms = out
        return self

    # -- orthorhombic supercell ---------------------------------------------

    def _orthorhombic_supercell(self, EPS=1e-2):
        r"""Turn a non-orthorhombic 3x3 cell into an orthorhombic supercell, in place.

        A multislice or PRISM calculation needs a rectangular, periodically
        continuable box (Kirkland, *Advanced Computing in Electron Microscopy*,
        2nd ed., Ch. 6).  Four stages:

        1. rigidly rotate the cell so **a** lies along +x;
        2. rotate about x so **b** lies in the xy plane;
        3. three pseudo-rational orthogonalisation passes on the dimension
           pairs (0,1), (0,2), (1,2) -- see :func:`_orthogonalise_pair`.  The
           order matters and is safe: once **b** and **c** are both
           perpendicular to **a**, subtracting a multiple of **b** from **c**
           preserves ``c . a = 0``;
        4. discard the now EPS-small off-diagonal elements by taking the
           matrix diagonal.

        Stages 1 and 2 are rigid rotations of the whole crystal, so they leave
        the fractional coordinates untouched and act on the cell matrix alone.

        Performance
        -----------
        Stages 1-3 depend only on the 3x3 cell, so the whole plan is computed
        on the matrix first; the atoms are then tiled ONCE by the product of
        the three passes' factors and transformed ONCE by the composed 3x3 map.
        The straightforward interleaving would materialise the (up to
        ``1/EPS**2``-fold) intermediate atom table three times over.  The total
        multiplier is checked against :data:`_MAX_SUPERCELL_ATOMS` before any
        memory is committed, because a badly conditioned cell can legitimately
        ask for ``(1/EPS)**3 = 10**6`` times more atoms.
        """
        M0 = np.array(self.unitcell, dtype=_float, copy=True).reshape(3, 3)

        # Stages 1 and 2: rigid alignment (cell only).
        M_rot = _align_cell_to_axes(M0, EPS)

        # Stage 3, planned on the matrix alone.
        M = M_rot
        tiling = np.ones(3, dtype=np.int64)
        for dim1, dim2 in ((0, 1), (0, 2), (1, 2)):
            M, n1, n2 = _orthogonalise_pair(M, dim1, dim2, EPS)
            tiling[dim1] *= n1
            tiling[dim2] *= n2

        ncopies = int(tiling.prod())
        if ncopies * self.atoms.shape[0] > _MAX_SUPERCELL_ATOMS:
            raise ValueError(
                "orthorhombic supercell construction would need %d x %d = %d "
                "atoms (tiling %r), above the %d cap. The cell is poorly "
                "approximated by a rational supercell at EPS=%g; try a larger "
                "EPS or supply an already-orthorhombic cell."
                % (
                    ncopies,
                    self.atoms.shape[0],
                    ncopies * self.atoms.shape[0],
                    tuple(int(t) for t in tiling),
                    _MAX_SUPERCELL_ATOMS,
                    EPS,
                )
            )

        # Apply to the atoms: one tiling, one composed coordinate map.
        self.unitcell = M_rot
        M_tiled = M_rot * tiling[:, np.newaxis].astype(_float)
        if ncopies > 1:
            self.tile(*(int(t) for t in tiling))
        if ncopies > 1 or not np.array_equal(M, M_tiled):
            # Row-vector convention: f_new = (f_old M_old M_new^-1) mod 1.
            transform = M_tiled @ np.linalg.inv(M)
            self.atoms[:, :3] = np.mod(self.atoms[:, :3] @ transform, 1.0)

        # Stage 4: drop the residual shear.
        off = np.max(np.abs(M - np.diag(np.diag(M))))
        if off > EPS * float(np.max(np.abs(np.diag(M)))):
            warnings.warn(
                "orthorhombic supercell: discarding a residual shear of %.3g A "
                "(the rational approximation at EPS=%g did not fully "
                "orthogonalise the cell)" % (off, EPS),
                stacklevel=2,
            )
        self.unitcell = np.diag(M).copy()

    def _pseudo_rational_tiling(self, dim1, dim2, EPS=1e-2):
        """Make edge vector ``dim2`` orthogonal to edge vector ``dim1``, in place.

        Tiles the crystal by a rational approximation of the two vectors'
        overlap and then removes the projection exactly; see
        :func:`_orthogonalise_pair` for the derivation.  Requires
        ``self.unitcell`` to be a 3x3 matrix (rows = edge vectors), and mutates
        both the cell and the atom table.  Does nothing when the two vectors
        are already orthogonal to within ``EPS``.
        """
        M = np.asarray(self.unitcell, dtype=_float)
        if M.shape != (3, 3):
            raise ValueError(
                "_pseudo_rational_tiling needs a 3x3 cell matrix; got shape %r"
                % (M.shape,)
            )
        M_new, n1, n2 = _orthogonalise_pair(M, dim1, dim2, EPS)
        if n1 == 1 and n2 == 1 and np.array_equal(M_new, M):
            return

        tiling = [1, 1, 1]
        tiling[dim1] = n1
        tiling[dim2] = n2
        M_tiled = M * np.array(tiling, dtype=_float)[:, np.newaxis]

        self.tile(*tiling)
        self.unitcell = M_tiled
        transform = M_tiled @ np.linalg.inv(M_new)
        self.atoms[:, :3] = np.mod(self.atoms[:, :3] @ transform, 1.0)
        self.unitcell = M_new

    # -- rigid rotation ------------------------------------------------------

    def rotate(
        self,
        theta,
        axis,
        origin=(0.5, 0.5, 0.5),
        wrap=True,
        margin=0.0,
        rebox=True,
    ):
        r"""Rigidly rotate the atoms and re-box them; returns a NEW structure.

        The atom set is rotated by ``theta`` about the direction ``axis``
        through the fractional point ``origin`` and then re-expressed in a
        (possibly new) axis-aligned orthorhombic box.  ``self`` is not modified.

        Pipeline
        --------
        1. Cartesian positions and the pivot:
           :math:`r = f \odot L`, :math:`o = \mathrm{origin} \odot L`, with
           :math:`L` the cell edge lengths.
        2. :math:`r' = (r - o) R + o` with :math:`R` from :func:`_rot_matrix`.

           .. warning::

              **Sign convention.**  ``R`` is the ACTIVE rotation of a COLUMN
              vector, but it is applied to ROW vectors on the right.  Since
              ``r @ R = (R^T r^T)^T`` and :math:`R^{\mathsf T} = R(-\theta)`,
              the atoms are rotated by **minus** ``theta`` about ``axis``.
              Example: an atom at fractional ``(0.75, 0.5, 0.5)`` of a
              4x4x4 A cell sits 1 A from the centre along +x, and
              ``rotate(pi/2, [0, 0, 1])`` puts it at Cartesian ``(2, 1, 2)``,
              i.e. 1 A along **-y**.  Using the textbook active convention here
              would mirror every tilt series.

        3. Re-boxing, selected by ``(rebox, wrap)``:

           * ``rebox=False`` -- keep the cell verbatim and just divide by it.
             Atoms leaving the box are neither wrapped nor clipped; a warning
             naming the observed fractional range is issued instead.  Use this
             for a tilt SERIES of a vacuum-padded finite object, so that every
             tilt shares one cell, field of view and object placement.
           * ``rebox=True, wrap=True`` (default) -- fit a new axis-aligned box
             to the rotated CELL CORNERS, using the five fractional corners
             000, 100, 010, 001, 111: the new origin is their componentwise
             minimum and the new edge lengths their componentwise peak-to-peak
             range.  Atoms are then mapped into that box modulo 1.  Five
             corners are exact for a rotation about a cardinal axis (the
             remaining corners 110, 101, 011 add nothing to the bounding box
             there); for a general off-axis direction the box can be
             undersized, and the modulo-1 reduction then maps atoms into wrong
             periodic images.
           * ``rebox=True, wrap=False`` -- fit the box to the true rotated ATOM
             extent: origin = componentwise minimum minus ``margin``, edges =
             peak-to-peak plus ``2*margin``, with NO modulo-1 reduction.  Exact
             for a finite, non-periodic specimen at an arbitrary tilt.

        Parameters
        ----------
        theta : float
            Rotation angle in RADIANS.
        axis : array_like, shape (3,)
            Any nonzero direction; normalised internally.
        origin : array_like, shape (3,), default (0.5, 0.5, 0.5)
            FRACTIONAL coordinates of the pivot.
        wrap : bool, default True
            See above.  Only consulted when ``rebox`` is True.
        margin : float, default 0.0
            Vacuum padding in Angstrom added on EACH side in the
            ``wrap=False`` branch (total extent grows by ``2*margin``).
        rebox : bool, default True
            See above.

        Returns
        -------
        Structure
            A new object.  Columns 3-5, ``Title`` and ``fractional_occupancy``
            are carried over unchanged.

        Notes
        -----
        Coordinate columns are read as FRACTIONAL.  Applied to a ``.xyz``
        structure loaded with the default ``atomic_coordinates='fractional'``
        (where the columns hold Cartesian Angstrom) this multiplies Angstrom by
        Angstrom -- deterministic, but not physically meaningful.
        """
        uc = np.asarray(self.unitcell, dtype=_float)
        if uc.ndim != 1 or uc.size != 3:
            raise ValueError(
                "rotate() needs the orthorhombic edge-length form of the cell; "
                "got shape %r" % (uc.shape,)
            )

        R = _rot_matrix(theta, axis)
        o = np.asarray(origin, dtype=_float).reshape(3) * uc

        natoms = self.atoms.shape[0]
        out = np.empty((natoms, 6), dtype=_float)
        out[:, 3:] = self.atoms[:, 3:]

        # The coordinate arithmetic runs in a CONTIGUOUS (n, 3) scratch buffer and
        # is copied into the six-column table once, at the end.  Writing straight
        # into ``out[:, :3]`` would save that copy but makes every intermediate a
        # strided view (stride 6 doubles): BLAS cannot write into it, so the matmul
        # falls off its fast path and each of the following in-place passes touches
        # one cache line per two coordinates.  Measured on this host, the strided
        # form is 1.1-1.7x SLOWER over 5e2 - 1e6 atoms.  The results are
        # bit-identical either way -- same operations, same order.

        # The whole affine map from OLD fractional to NEW fractional
        # coordinates is fixed by the 3x3 cell and the pivot, so it is folded
        # into one 3x3 matrix and one offset before the atoms are touched:
        #
        #     r' = (f L - o) R + o = f A0 + c0,    A0 = diag(L) R,  c0 = o - oR
        #     f' = (r' - o_new) / L_new = f (A0/L_new) + (c0 - o_new)/L_new
        #
        # so one matmul writing straight into the output buffer plus one
        # broadcast add replaces the five full-size passes of the literal
        # scale / translate / rotate / translate / rescale chain.
        A0 = uc[:, np.newaxis] * R
        c0 = o - o @ R

        if not rebox:
            new_uc = uc.copy()
            cart = self.atoms[:, :3] @ (A0 / new_uc)
            cart += c0 / new_uc
            lo = float(cart.min())
            hi = float(cart.max())
            if lo < -1e-9 or hi > 1.0 + 1e-9:
                warnings.warn(
                    "rotate(rebox=False): rotated atoms lie outside the "
                    "retained cell (fractional range [%.6g, %.6g], expected "
                    "[0, 1)); they are neither wrapped nor clipped -- pad the "
                    "cell with more vacuum before rotating." % (lo, hi),
                    stacklevel=2,
                )
        elif wrap:
            # Bounding box of the rotated cell corners 000, 100, 010, 001, 111.
            corners = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                ],
                dtype=_float,
            )
            verts = corners @ A0 + c0
            new_o = verts.min(axis=0)
            new_uc = np.ptp(verts, axis=0)
            cart = self.atoms[:, :3] @ (A0 / new_uc)
            cart += (c0 - new_o) / new_uc
            cart %= 1.0
        else:
            # The box follows the rotated ATOMS, so their Cartesian positions
            # are needed first; they are built in the scratch buffer.
            cart = self.atoms[:, :3] @ A0
            cart += c0
            new_o = cart.min(axis=0) - margin
            new_uc = np.ptp(cart, axis=0) + 2.0 * margin
            if margin == 0.0:
                # Without this the extremal atoms would sit at fraction exactly
                # 1.0, violating the [0, 1) convention every consumer assumes.
                new_uc = new_uc * (1.0 + 1e-10)
            # A specimen that is flat along one axis has zero extent there;
            # unit length keeps the division finite (and the atoms at 0).
            new_uc = np.where(new_uc > 0.0, new_uc, 1.0)
            cart -= new_o
            cart /= new_uc

        out[:, :3] = cart
        return self._assemble(new_uc, out, self.Title, self.fractional_occupancy)
