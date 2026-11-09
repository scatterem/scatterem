"""The CIF reader under ``Phase.from_cif`` and ``crystal.phase.cif_to_arrays``.

ASE parses the file and applies the symmetry operations; this module adds the
isotropic thermal values of the site lines, the anisotropic tensors
(:mod:`scatterem.crystal._cif_aniso`) and one row per species of a mixed site.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from scatterem.crystal._cif_aniso import expanded_tensors
from scatterem.crystal.elements import SYMBOLS

logger = logging.getLogger(__name__)

#: <u^2> per unit of the crystallographic B factor, from B = 8 pi^2 <u^2>.
U2_PER_B = 1.0 / (8.0 * np.pi**2)

#: CIF columns of the isotropic thermal value, as ASE spells them (lower case).
_U_ISO_TAG = "_atom_site_u_iso_or_equiv"
_B_ISO_TAG = "_atom_site_b_iso_or_equiv"

#: Distance in fractional coordinates below which ASE puts two CIF lines on
#: one site (the ``symprec`` default of ``ase.spacegroup.crystal``).
_ASE_SYMPREC = 0.001


def cif_site_arrays(
    path: Path, u2_default: float
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None
]:
    """The arrays of ``crystal.phase.cif_to_arrays``, with `u2_default` for a gap.

    Returns ``(rows, fractional, z, occupancy, u2, tensors)``; see
    :func:`scatterem.crystal.phase.cif_to_arrays`.
    """
    block, atoms = _read_cif_block(path)
    cell = np.asarray(atoms.get_cell()[:], dtype=np.float64)
    # wrap=True: CIF fractional coordinates are only defined modulo 1, and
    # ase can return values that drift a hair outside [0, 1) (e.g. -1e-16 or
    # symmetry-generated positions landing exactly on 1.0) after applying the
    # space-group operators. Wrapping keeps the result in [0, 1) rather than
    # relying on whatever the CIF's listed symmetry operators produce.
    frac = np.asarray(atoms.get_scaled_positions(wrap=True), dtype=np.float64)
    Z = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
    site_occ = _site_occupancies(atoms)
    line_u2 = _line_u2(block, path)
    line_u2 = np.where(np.isnan(line_u2), u2_default, line_u2)
    site_u2 = _site_u2(block, atoms, line_u2)
    _check_merge_rule(site_occ, site_u2, path)

    site_u = expanded_tensors(block, atoms, cell, line_u2, path)
    pos: list[np.ndarray] = []
    zs: list[int] = []
    occs: list[float] = []
    u2s: list[float] = []
    tensors: list[np.ndarray] = []
    for a in range(len(atoms)):
        occ_map = site_occ[a]
        if not occ_map:
            pos.append(frac[a])
            zs.append(int(Z[a]))
            occs.append(1.0)
            u2s.append(site_u2[a][None])
            if site_u is not None:
                tensors.append(site_u[a][None])
            continue
        for sym, occ in occ_map.items():
            pos.append(frac[a])
            zs.append(_symbol_to_Z(sym, path))
            occs.append(float(occ))
            u2s.append(site_u2[a][sym])
            if site_u is not None:
                tensors.append(site_u[a][sym])
    return (
        cell.copy(),
        np.asarray(pos, dtype=np.float64).reshape(-1, 3),
        np.asarray(zs, dtype=np.int64),
        np.asarray(occs, dtype=np.float64),
        np.asarray(u2s, dtype=np.float64),
        None if site_u is None else np.stack(tensors),
    )


def _read_cif_block(path: Path) -> tuple[Any, Any]:
    """Return the last CIF block with a structure and its expanded ``Atoms``.

    These are the calls of ``ase.io.read(path, format="cif")`` with its
    defaults, made one by one so the raw block stays at hand for the thermal
    columns that ASE does not put into ``Atoms``.
    """
    from ase.io.cif import parse_cif

    try:
        blocks = [b for b in parse_cif(str(path)) if b.has_structure()]
        block = blocks[-1]
        atoms = block.get_atoms()
    except Exception as e:
        # ase raises a bare, message-less AssertionError from deep inside its
        # CIF parser for a malformed/non-CIF file, which otherwise surfaces
        # with no indication of *which* file was being read. Re-raise a
        # fresh exception naming the path; `from e` keeps the original type
        # and message (however empty) attached as __cause__, so nothing is
        # swallowed. RuntimeError (not `type(e)(...)`) because some
        # exceptions ase could plausibly raise (e.g. UnicodeDecodeError)
        # need constructor args beyond a single message string.
        raise RuntimeError(
            f"Phase.from_cif: failed to read {path} as a CIF file "
            f"({type(e).__name__}: {e})"
        ) from e
    return block, atoms


def _thermal_column(block: Any, tag: str, path: Path) -> np.ndarray:
    """One thermal column per site line as floats; NaN for ``?``, ``.`` or none."""
    n = len(block.get_symbols())
    if tag not in block:
        return np.full(n, np.nan)
    raw = block[tag]
    values = raw if isinstance(raw, list) else [raw]
    labels = _site_labels(block, n)
    out = np.full(n, np.nan)
    for i, value in enumerate(values):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[i] = float(value)
        elif value not in ("?", "."):
            raise ValueError(
                f"{path}: site {labels[i]!r} has {tag} {value!r}, not a number"
            )
    return out


def _site_labels(block: Any, n: int) -> list[str]:
    """The ``_atom_site_label`` of each site line, or ``""`` without that column."""
    raw = block.get("_atom_site_label")
    if raw is None:
        return [""] * n
    return [str(v) for v in (raw if isinstance(raw, list) else [raw])]


def _line_u2(block: Any, path: Path) -> np.ndarray:
    """<u^2> [Angstrom^2] per CIF site line: U first, else B / (8 pi^2), else NaN.

    A negative value is not physical. The function logs a warning with the
    file, the site label and the value, and returns NaN for that line, so the
    default applies.
    """
    u = _thermal_column(block, _U_ISO_TAG, path)
    b = _thermal_column(block, _B_ISO_TAG, path)
    from_u = ~np.isnan(u)
    u2 = np.where(from_u, u, b * U2_PER_B)
    negative = np.flatnonzero(u2 < 0.0)
    if negative.size:
        labels = _site_labels(block, len(u2))
        for i in negative:
            tag, value = (_U_ISO_TAG, u[i]) if from_u[i] else (_B_ISO_TAG, b[i])
            logger.warning(
                "%s: site %r has a negative %s %r; the default <u^2> is used",
                path,
                labels[i],
                tag,
                float(value),
            )
        u2[negative] = np.nan
    return u2


def _check_merge_rule(
    site_occ: list[dict | None], site_u2: list[dict], path: Path
) -> None:
    """Raise if a species of ASE's occupancy dict has no thermal value.

    ``_site_u2`` copies the merge rule of ASE 3.29. If a later ASE changes
    that rule, this check names the file and the ASE version, instead of a
    bare ``KeyError`` in the row assembly.
    """
    for occ_map, u2_map in zip(site_occ, site_u2, strict=True):
        missing = set(occ_map or ()) - set(u2_map)
        if missing:
            import ase

            raise RuntimeError(
                f"Phase.from_cif: {path}: species {sorted(missing)} of the ASE "
                f"occupancy merge have no thermal value. The merge rule of "
                f"ASE {ase.__version__} differs from the ASE 3.29 rule that "
                "_site_u2 copies."
            )


def _site_u2(block: Any, atoms: Any, line_u2: np.ndarray) -> list[dict]:
    """Per-atom ``{symbol: <u^2>}``; the key ``None`` holds the atom's own line.

    ``spacegroup_kinds`` maps each atom to the CIF line that ASE expanded. On a
    mixed site ASE also merges the other lines within ``_ASE_SYMPREC`` into
    the occupancy dict (``ase.spacegroup.crystal``: the line itself first,
    then the others in file order, a later line of the same symbol winning).
    This function builds the thermal dict by the same rule, so each species
    row takes the value of the line that gave its occupancy.
    """
    symbols = block.get_symbols()
    kinds = atoms.arrays.get("spacegroup_kinds")
    if kinds is None:
        kinds = np.arange(len(atoms))
    coords = block.get_unsymmetrized_structure().get_scaled_positions()
    by_kind: dict[int, dict] = {}
    for k in {int(k) for k in kinds}:
        near = np.flatnonzero(np.linalg.norm(coords - coords[k], axis=1) < _ASE_SYMPREC)
        lines = [k, *(int(j) for j in near if j != k)]
        site: dict = {None: float(line_u2[k])}
        for j in lines:
            site[symbols[j]] = float(line_u2[j])
        by_kind[k] = site
    return [by_kind[int(k)] for k in kinds]


def _site_occupancies(atoms: Any) -> list[dict | None]:
    """Per-atom ``{symbol: occ}`` from ``atoms.info['occupancy']``, or ``None``.

    The occupancy dict is keyed by **asymmetric-unit site index**, not by atom
    index; ase records the atom -> site map in the ``spacegroup_kinds`` array.
    Indexing by atom would misalign whenever a CIF lists several species on one
    position: ase collapses a shared site to a single atom but still allocates
    one occupancy-dict key per input line that named that site, so the key
    count can exceed the atom count (verified against ase 3.28 -- a CIF with
    ``Fe 0.5`` / ``Ni 0.5`` on one position yields 2 atoms but 3 occupancy
    keys; ``spacegroup_kinds`` correctly picks the key that already carries
    the merged ``{"Fe": 0.5, "Ni": 0.5}`` dict).
    """
    occ_info = atoms.info.get("occupancy")
    n = len(atoms)
    if not occ_info:
        return [None] * n
    kinds = atoms.arrays.get("spacegroup_kinds")
    if kinds is None:
        kinds = np.arange(n)
    return [occ_info.get(str(int(kinds[a]))) for a in range(n)]


def _symbol_to_Z(symbol: str, path: Path) -> int:
    """Atomic number of an ASE symbol (case-sensitive; ``""`` gives 0)."""
    try:
        return SYMBOLS.index(str(symbol))
    except ValueError as exc:  # pragma: no cover - malformed CIF
        raise ValueError(f"unknown element symbol {symbol!r} in {path}") from exc
