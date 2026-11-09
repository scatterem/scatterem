"""Anisotropic displacement tensors from a CIF, expanded to P1, Cartesian.

A CIF gives ``U^ij`` (``_atom_site_aniso_U_11`` ... ``_U_23``) or ``B^ij``
(``B = 8 pi^2 U``) per site line, in the convention
``cov(dx_i, dx_j) = a*_i a*_j U^ij`` for the fractional displacement ``dx``
(Trueblood et al., Acta Cryst. A52 (1996) 770-781). The Cartesian tensor is
``U_cart = M N U N M^T`` with ``M`` the lattice vectors as columns and
``N = diag(|a*_i|)`` (Grosse-Kunstleve and Adams, J. Appl. Cryst. 35 (2002)
477-480). An atom that the space-group operation ``x -> R x + t`` generated
from its site line has the fractional covariance ``R cov R^T``.

ASE reads the block; this module only reads the aniso loop and applies the
transformations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["expanded_tensors"]

logger = logging.getLogger(__name__)
_NPD_TOL = 1e-12  # Angstrom^2; the eigenvalue bar of the positive-definite check

_U_TAGS = tuple(
    f"_atom_site_aniso_u_{ij}" for ij in ("11", "22", "33", "12", "13", "23")
)
_B_TAGS = tuple(
    f"_atom_site_aniso_b_{ij}" for ij in ("11", "22", "33", "12", "13", "23")
)
_LABEL_TAG = "_atom_site_aniso_label"
_U2_PER_B = 1.0 / (8.0 * np.pi**2)
_POSITION_TOL = 1e-4  # fractional; ASE's symprec is 1e-3


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else [value]


def _aniso_lines(block: Any, path: Path) -> dict[str, np.ndarray]:
    """``{label: U^ij (3, 3)}`` of the aniso loop; empty without one."""
    tags, scale = _U_TAGS, 1.0
    if _U_TAGS[0] not in block:
        if _B_TAGS[0] not in block:
            return {}
        tags, scale = _B_TAGS, _U2_PER_B
    labels = [str(v) for v in _as_list(block[_LABEL_TAG])]
    cols = []
    for tag in tags:
        values = _as_list(block.get(tag, [0.0] * len(labels)))
        for label, v in zip(labels, values, strict=True):
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise ValueError(
                    f"{path}: site {label!r} has {tag} {v!r}, not a number"
                )
        cols.append(np.asarray(values, dtype=np.float64) * scale)
    u11, u22, u33, u12, u13, u23 = cols
    out = {}
    for i, label in enumerate(labels):
        out[label] = np.array(
            [
                [u11[i], u12[i], u13[i]],
                [u12[i], u22[i], u23[i]],
                [u13[i], u23[i], u33[i]],
            ]
        )
    return out


def _generating_rotation(
    x_site: np.ndarray, x_atom: np.ndarray, rotations: np.ndarray, shifts: np.ndarray
) -> np.ndarray:
    """A rotation ``R`` of an operation with ``R x_site + t = x_atom`` (mod 1)."""
    images = np.einsum("sij,j->si", rotations, x_site) + shifts
    d = images - x_atom
    d -= np.rint(d)
    hit = np.flatnonzero(np.abs(d).max(axis=1) <= _POSITION_TOL)
    if hit.size == 0:
        raise RuntimeError("no space-group operation maps the site line onto the atom")
    return rotations[hit[0]]


def expanded_tensors(
    block: Any, atoms: Any, cell: np.ndarray, line_u2: np.ndarray, path: Path
) -> list[dict] | None:
    """Per expanded atom, ``{symbol: U_cart (3, 3)}`` with the key ``None`` too.

    Parameters
    ----------
    block : ase.io.cif.CIFBlock
        The CIF block.
    atoms : ase.Atoms
        ``block.get_atoms()`` (symmetry-expanded, with ``spacegroup_kinds``).
    cell : numpy.ndarray
        ``(3, 3)`` lattice rows [Angstrom] of ``atoms``.
    line_u2 : numpy.ndarray
        ``(n_lines,)`` isotropic ``<u^2>`` per site line (already defaulted);
        a line without an aniso entry gets ``<u^2> I``.
    path : Path
        File name for messages.

    Returns
    -------
    list of dict or None
        None when the block has no aniso loop (the caller keeps no tensor).
        Otherwise, for each atom of ``atoms``, the Cartesian tensor
        [Angstrom^2] in the frame of ``cell`` for each species on its site
        (the merge rule of ``crystal._cif_sites._site_u2``).
    """
    lines = _aniso_lines(block, path)
    if not lines:
        return None
    for label in [
        k for k, u in lines.items() if np.linalg.eigvalsh(u).min() < -_NPD_TOL
    ]:
        # a non-positive-definite ADP (checkCIF flags these): the site keeps
        # its U_iso, as before aniso support, instead of failing downstream
        logger.warning(
            "%s: the aniso tensor of site %s is not positive definite; "
            "the site uses its isotropic value",
            path,
            label,
        )
        del lines[label]
    labels = [str(v) for v in _as_list(block.get("_atom_site_label", []))]
    symbols = block.get_symbols()
    coords = block.get_unsymmetrized_structure().get_scaled_positions()
    rotations, shifts = block.get_spacegroup(True).get_op()
    kinds = atoms.arrays.get("spacegroup_kinds")
    kinds = np.arange(len(atoms)) if kinds is None else kinds
    frac = atoms.get_scaled_positions(wrap=True)
    m = cell.T  # columns are the lattice vectors
    n = np.diag(np.linalg.norm(np.linalg.inv(cell).T, axis=1))  # |a*_i|
    out = []
    for a, k in enumerate(int(k) for k in kinds):
        rot = _generating_rotation(coords[k], frac[a], rotations, shifts)
        near = np.flatnonzero(np.linalg.norm(coords - coords[k], axis=1) < 1e-3)
        site: dict = {}
        for j in [k, *(int(j) for j in near if j != k)]:
            label = labels[j] if j < len(labels) else ""
            if label in lines:
                cov = rot @ (n @ lines[label] @ n) @ rot.T
                tensor = m @ cov @ m.T
            else:
                tensor = float(line_u2[j]) * np.eye(3)
            site.setdefault(None, tensor)
            site[symbols[j]] = tensor
        out.append(site)
    return out
