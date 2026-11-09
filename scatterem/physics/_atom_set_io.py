"""File readers of the structure container (private).

:func:`read_structure_arrays` selects one of five readers and returns plain
arrays. It does not build the container, so this module does not import
:mod:`scatterem.physics.atom_set`.

Readers (name, extension, layout source):

- ``"kirkland"``, ``.xyz``: Kirkland, *Advanced Computing in Electron
  Microscopy* (2010, 2020); Ophus 2017; Pryor et al. 2017.
- ``"poscar"``, ``.p1``: VASP POSCAR, version 5 layout (VASP manual; Momma
  and Izumi 2011).
- ``"mustem"``, ``.xtl``: muSTEM crystal file (muSTEM manual; Allen et al.
  2015).
- ``"cif"``, ``.cif``: through :func:`scatterem.crystal.phase.cif_to_arrays`.
- ``"extxyz"``, ``.extxyz``: through
  :func:`scatterem.crystal._ase_io.read_frames`.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from scatterem.crystal import _ase_io, phase
from scatterem.crystal.elements import atomic_number

__all__ = ["DISPLACEMENT_KINDS", "SUFFIX_STYLE", "FILE_STYLES", "read_structure_arrays"]

#: File style names.
FILE_STYLES = ("kirkland", "poscar", "mustem", "cif", "extxyz")

#: Lower-case file extension to file style.
SUFFIX_STYLE = {
    ".xyz": "kirkland",
    ".p1": "poscar",
    ".xtl": "mustem",
    ".cif": "cif",
    ".extxyz": "extxyz",
}

#: Spellings of the displacement option: ``"mean_square"`` is <u²> (Å²),
#: ``"rms_length"`` the root-mean-square displacement (Å), ``"b_factor"`` the
#: Debye-Waller B (Å²).
DISPLACEMENT_KINDS = ("mean_square", "rms_length", "b_factor")

#: The quantity that each format specification defines for its thermal
#: column. A file that follows its specification reads correctly with the
#: default. Kirkland writes the rms displacement ("wobble"); muSTEM writes
#: <u²>; the extra POSCAR column is this module's own choice, <u²>.
_DEFAULT_DISPLACEMENT_KIND = {
    "kirkland": "rms_length",
    "poscar": "mean_square",
    "mustem": "mean_square",
}

#: The position kinds: ``"relative"`` is fractional, ``"absolute"`` Cartesian
#: Å. The override applies to kirkland and mustem only.
_POSITION_KINDS = ("relative", "absolute")
_DEFAULT_POSITION_KIND = {"kirkland": "absolute", "mustem": "relative"}

#: <u²> per unit of B, from B = 8 pi² <u²> (Willis and Pryor 1975, ch. 4).
_MSD_PER_B = 1.0 / (8.0 * math.pi**2)


def as_mean_square(values: np.ndarray, quantity: str) -> np.ndarray:
    """Convert thermal values read from a file to <u²> in Å².

    Parameters
    ----------
    values : numpy.ndarray
        (N,) float64, in the unit of `quantity`.
    quantity : str
        One of :data:`DISPLACEMENT_KINDS`.

    Returns
    -------
    numpy.ndarray
        (N,) float64, Å².
    """
    if quantity == "rms_length":
        return values * values
    if quantity == "b_factor":
        return values * _MSD_PER_B
    return values


def _error(source_file, line: int, text: str) -> ValueError:
    return ValueError(f"source_file: {source_file} line {line}: {text}")


def _floats(tokens, source_file, line: int) -> list[float]:
    try:
        return [float(t) for t in tokens]
    except ValueError:
        raise _error(
            source_file, line, f"cannot read numbers from {' '.join(tokens)!r}"
        ) from None


def _parses_as_float(word: str) -> bool:
    try:
        float(word)
    except ValueError:
        return False
    return True


def _kirkland(source_file, lines, displacement_kind, position_kind):
    """Kirkland XYZ.

    Line 1 is a comment (the remark). Line 2 holds the three rectangular cell
    lengths ``a b c`` (Å). Each later line is one atom, ``Z x y z fill u``:
    ``Z`` an integer token, ``x y z`` Cartesian Å, the occupancy and the
    thermal value (rms displacement, Å, by the published description). The
    atom list stops at the end of the file or at a line that starts with the
    token ``-1``; the reader skips all lines after that marker. Blank lines
    are skipped. Any other line raises ``ValueError`` with its line number.
    Fractional positions are ``x / a`` per axis; with
    ``position_kind="relative"`` the numbers are kept as written.
    """
    if len(lines) < 2:
        raise _error(source_file, len(lines) + 1, "file ends before the cell line")
    cell_tokens = lines[1].split()
    if len(cell_tokens) != 3:
        raise _error(
            source_file, 2, f"expected 3 cell lengths, got {len(cell_tokens)} tokens"
        )
    cell = np.array(_floats(cell_tokens, source_file, 2), dtype=np.float64)
    # (1-based line number, words) of every non-blank line after the cell
    # line, cut at the first line whose first word is the -1 marker.
    atom_lines = [
        (i + 3, words) for i, words in enumerate(t.split() for t in lines[2:]) if words
    ]
    stop = next(
        (k for k, (_, words) in enumerate(atom_lines) if words[0] == "-1"),
        len(atom_lines),
    )
    z, xyz, fill, u_col = [], [], [], []
    for number, words in atom_lines[:stop]:
        if len(words) != 6:
            raise _error(
                source_file,
                number,
                f"expected 6 numbers: Z, x, y, z, site fraction, displacement; got {len(words)}",
            )
        try:
            z.append(int(words[0]))
        except ValueError:
            raise _error(
                source_file, number, f"Z must be an integer; got {words[0]!r}"
            ) from None
        values = _floats(words[1:], source_file, number)
        xyz.append(values[:3])
        fill.append(values[3])
        u_col.append(values[4])
    xyz = np.array(xyz, dtype=np.float64).reshape(-1, 3)
    fractional = xyz if position_kind == "relative" else xyz / cell
    return {
        "box_spec": cell,
        "fractional": fractional,
        "element_z": np.array(z, dtype=np.int64),
        "site_fraction": np.array(fill, dtype=np.float64),
        "msd": as_mean_square(np.array(u_col, dtype=np.float64), displacement_kind),
        "adp": None,
        "remark": lines[0].strip(),
    }


def _poscar_extras(rest, selective, source_file, number) -> list[float]:
    """[occupancy, thermal value] from the words after the 3 coordinates.

    Selective flags ``T``/``F`` come first (checked, discarded). One next word
    that is not a number is a site label (v0.3 and VESTA ``x y z label occ u``)
    and is skipped. Then up to two numbers, in the `displacement_kind` quantity;
    the first other word ends them. Missing values are 1 and NaN. The label
    does not set the species; the element line and the counts do.
    """
    if selective:
        flags = [t.upper() for t in rest[:3]]
        if len(flags) != 3 or any(f not in ("T", "F") for f in flags):
            raise _error(source_file, number, "expected three selective flags T or F")
        rest = rest[3:]
    if rest and not _parses_as_float(rest[0]):
        rest = rest[1:]
    extras = []
    for t in rest:
        if not _parses_as_float(t):
            break
        extras.append(float(t))
    if len(extras) > 2:
        raise _error(source_file, number, f"{len(extras)} numeric extras are ambiguous")
    return extras + [1.0, math.nan][len(extras) :]


def _poscar(source_file, lines, displacement_kind, position_kind):
    """VASP POSCAR, version 5 layout (VESTA "P1" export).

    Lines: comment (remark); scale factor (a single number; a positive value
    scales all lengths, a negative value gives the cell volume in Å³); three
    lattice rows; element symbols (a token such as ``Fe_pv`` or ``Fe/x`` is
    cut at ``_`` or ``/``); counts; an optional line starting with
    ``S``/``s`` (selective dynamics); the coordinate line (``D``/``d``
    direct, ``C``/``c``/``K``/``k`` Cartesian Å times the scale); after it,
    each atom has a line of its own.

    Atom line: three coordinates, then the words that :func:`_poscar_extras`
    reads. An atom without extras gets occupancy 1 and the scatterem default
    <u²>. Lines after the last atom are ignored.
    """
    if len(lines) < 8:
        raise _error(
            source_file, len(lines) + 1, "file ends before the coordinate line"
        )
    scale_tokens = lines[1].split()
    if len(scale_tokens) != 1:
        raise _error(
            source_file, 2, f"expected one scale factor, got {len(scale_tokens)} tokens"
        )
    scale = _floats(scale_tokens, source_file, 2)[0]
    if scale == 0.0 or not math.isfinite(scale):
        raise _error(
            source_file, 2, f"scale factor must be finite and non-zero; got {scale}"
        )
    edge_rows = []
    for number in (3, 4, 5):
        tokens = lines[number - 1].split()
        if len(tokens) < 3:
            raise _error(source_file, number, "expected 3 numbers of a lattice row")
        edge_rows.append(_floats(tokens[:3], source_file, number))
    lattice = np.array(edge_rows, dtype=np.float64)
    symbols = lines[5].split()
    if not symbols or any(_parses_as_float(s) for s in symbols):
        raise _error(
            source_file, 6, "element line missing (the VASP 4 layout is not read)"
        )
    z_species = []
    for s in symbols:
        try:
            z_species.append(atomic_number(s.split("_")[0].split("/")[0]))
        except ValueError:
            raise _error(source_file, 6, f"unknown element symbol {s!r}") from None
    count_tokens = lines[6].split()
    try:
        counts = [int(t) for t in count_tokens]
    except ValueError:
        raise _error(source_file, 7, "counts must be integers") from None
    if len(counts) != len(symbols) or min(counts) < 1:
        raise _error(source_file, 7, "expected one positive count per element symbol")
    k = 7
    selective = lines[k].strip()[:1] in ("S", "s")
    if selective:
        k += 1
    if k >= len(lines):
        raise _error(source_file, k + 1, "file ends before the coordinate line")
    kind = lines[k].strip()[:1]
    if kind not in ("D", "d", "C", "c", "K", "k"):
        raise _error(source_file, k + 1, "expected 'Direct' or 'Cartesian'")
    cartesian = kind not in ("D", "d")
    atom_total = sum(counts)
    xyz = np.zeros((atom_total, 3))
    fill = np.ones(atom_total)
    u_col = np.full(atom_total, np.nan)
    for i in range(atom_total):
        number = k + 2 + i
        if number > len(lines):
            raise _error(
                source_file, number, f"file ends after {i} of {atom_total} atoms"
            )
        tokens = lines[number - 1].split()
        if len(tokens) < 3:
            raise _error(source_file, number, "expected 3 coordinates")
        xyz[i] = _floats(tokens[:3], source_file, number)
        fill[i], u_col[i] = _poscar_extras(tokens[3:], selective, source_file, number)
    if scale < 0:
        factor = (-scale / abs(np.linalg.det(lattice))) ** (1.0 / 3.0)
    else:
        factor = scale
    lattice = lattice * factor
    if cartesian:
        xyz = np.linalg.solve(lattice.T, (xyz * factor).T).T
    read = ~np.isnan(u_col)
    msd = np.full(atom_total, phase.DEFAULT_U2)
    msd[read] = as_mean_square(u_col[read], displacement_kind)
    return {
        "box_spec": lattice,
        "fractional": xyz,
        "element_z": np.repeat(np.array(z_species, dtype=np.int64), counts),
        "site_fraction": fill,
        "msd": msd,
        "adp": None,
        "remark": lines[0].strip(),
    }


def mustem_lattice(a, b, c, alpha, beta, gamma) -> np.ndarray:
    """Lattice rows (Å) from lengths (Å) and angles (degrees).

    Right-handed, with ``a0 = (a, 0, 0)``, ``a1`` in the x-y plane and the
    metric of International Tables Vol. B, ch. 1.1. The beam of the
    reduction is then along ``a x b`` (``c*``).

    Raises
    ------
    ValueError
        If the angles give no cell (the message names the angles).
    """
    ca, cb, cg = (math.cos(math.radians(x)) for x in (alpha, beta, gamma))
    sg = math.sin(math.radians(gamma))
    t_y = (ca - cb * cg) / sg
    w2 = 1.0 - cb * cb - t_y * t_y
    if not w2 > 0.0:
        raise ValueError(f"angles: {alpha}, {beta}, {gamma} degrees give no cell")
    return np.array(
        [[a, 0.0, 0.0], [b * cg, b * sg, 0.0], [c * cb, c * t_y, c * math.sqrt(w2)]],
        dtype=np.float64,
    )


def _mustem(source_file, lines, displacement_kind, position_kind):
    """muSTEM crystal file.

    Lines: title (remark); ``a b c`` or ``a b c alpha beta gamma`` (Å,
    degrees); beam energy (keV, read and ignored); number of species; per
    species a symbol line, a line ``count Z fill u`` (``u`` is <u²> in Å² by
    the manual; ``Z`` may be written as ``38.0``) and ``count`` lines of
    three fractional coordinates. Blank lines after the title are skipped.
    Missing angles, or three angles of exactly 90, give a rectangular cell
    with the positions as written. Other angles give the lattice rows of
    :func:`mustem_lattice`. With ``position_kind="absolute"`` the positions
    are Å in that frame.
    """
    body = [
        (i + 1, line.split()) for i, line in enumerate(lines) if i > 0 and line.split()
    ]
    it = iter(body)

    def take(what):
        try:
            return next(it)
        except StopIteration:
            raise _error(
                source_file, len(lines) + 1, f"file ends before the {what}"
            ) from None

    number, tokens = take("cell line")
    if len(tokens) not in (3, 6):
        raise _error(
            source_file,
            number,
            f"expected 3 lengths and optional 3 angles, got {len(tokens)} tokens",
        )
    params = _floats(tokens, source_file, number)
    oblique = len(params) == 6 and any(x != 90.0 for x in params[3:])
    if oblique:
        try:
            cell = mustem_lattice(*params)
        except ValueError as e:
            raise _error(source_file, number, str(e)) from None
    else:
        cell = np.array(params[:3], dtype=np.float64)
    number, tokens = take("beam energy")
    _floats(tokens[:1], source_file, number)
    number, tokens = take("number of species")
    try:
        species_total = int(tokens[0])
    except ValueError:
        raise _error(
            source_file, number, "number of species must be an integer"
        ) from None
    z, xyz, fill, u_col = [], [], [], []
    for _ in range(species_total):
        take("species symbol")
        number, tokens = take("species line")
        if len(tokens) < 4:
            raise _error(source_file, number, "expected 'count Z occupancy thermal'")
        try:
            count = int(tokens[0])
        except ValueError:
            raise _error(source_file, number, "count must be an integer") from None
        zf, o, t = _floats(tokens[1:4], source_file, number)
        if zf != round(zf):
            raise _error(
                source_file, number, f"Z must be an integer; got {tokens[1]!r}"
            )
        for _ in range(count):
            number, tokens = take("coordinates")
            if len(tokens) < 3:
                raise _error(source_file, number, "expected 3 coordinates")
            xyz.append(_floats(tokens[:3], source_file, number))
            z.append(int(round(zf)))
            fill.append(o)
            u_col.append(t)
    xyz = np.array(xyz, dtype=np.float64).reshape(-1, 3)
    if position_kind == "absolute":
        xyz = np.linalg.solve(cell.T, xyz.T).T if oblique else xyz / cell
    return {
        "box_spec": cell,
        "fractional": xyz,
        "element_z": np.array(z, dtype=np.int64),
        "site_fraction": np.array(fill, dtype=np.float64),
        "msd": as_mean_square(np.array(u_col, dtype=np.float64), displacement_kind),
        "adp": None,
        "remark": lines[0].strip(),
    }


def _cif(source_file, displacement_kind, position_kind):
    try:
        edge_rows, f_rel, z, fill, u2, adp_rows = phase.cif_to_arrays(source_file)
    except RuntimeError as e:
        raise ValueError(f"source_file: {source_file}: {e}") from e
    u2 = np.where(np.isnan(u2), phase.DEFAULT_U2, u2)
    if adp_rows is not None:
        adp_rows = np.array(adp_rows, dtype=np.float64)
        # A site without a full tensor in the file gets the isotropic default.
        tensor_gap = np.isnan(adp_rows).reshape(len(adp_rows), -1).any(axis=1)
        adp_rows[tensor_gap] = np.eye(3) * phase.DEFAULT_U2
    return {
        "box_spec": edge_rows,
        "fractional": f_rel,
        "element_z": z,
        "site_fraction": fill,
        "msd": u2,
        "adp": adp_rows,
        "remark": Path(source_file).name,
    }


def _extxyz(source_file, displacement_kind, position_kind):
    try:
        frames = _ase_io.read_frames(source_file, "extxyz")
    except ValueError as e:
        raise ValueError(f"source_file: {source_file}: {e}") from e
    if len(frames) != 1:
        raise ValueError(
            f"source_file: file holds {len(frames)} frames; read_atom_set reads one frame"
        )
    frame = frames[0]
    cell = np.asarray(frame.cell, dtype=np.float64)
    if not cell.any():
        raise ValueError(f"source_file: {source_file}: frame has no cell")
    positions = np.asarray(frame.positions, dtype=np.float64).reshape(-1, 3)
    return {
        "box_spec": cell,
        "fractional": np.linalg.solve(cell.T, positions.T).T,
        "element_z": np.asarray(frame.numbers, dtype=np.int64),
        "site_fraction": None,
        "msd": None,
        "adp": None,
        "remark": Path(source_file).name,
    }


def read_structure_arrays(
    source_file, file_style=None, displacement_kind=None, position_kind=None
) -> dict:
    """Read a structure file into plain arrays.

    Parameters
    ----------
    source_file : str or os.PathLike
        The file.
    file_style : str, optional
        One of :data:`FILE_STYLES`. None selects by the lower-case extension
        (:data:`SUFFIX_STYLE`).
    displacement_kind : str, optional
        Quantity of the thermal numbers in the file, one of
        :data:`DISPLACEMENT_KINDS`. None: the quantity of the format
        specification. It is validated for every reader; CIF and extxyz
        ignore it.
    position_kind : str, optional
        ``"relative"`` or ``"absolute"``, for kirkland and mustem only.

    Returns
    -------
    dict
        ``box_spec`` ((3,) lengths or (3, 3) rows, Å), ``fractional``
        (N, 3), ``element_z`` (N,), ``site_fraction`` (N,) or None, ``msd``
        (N,) Å² or None, ``adp`` (N, 3, 3) Å² or None, ``remark`` (str).
    """
    source_file = Path(source_file)
    if file_style is None:
        suffix = source_file.suffix.lower()
        if suffix not in SUFFIX_STYLE:
            raise ValueError(
                f"source_file: unknown extension {suffix!r}; known: {sorted(SUFFIX_STYLE)} "
                f"(or give file_style, one of {FILE_STYLES})"
            )
        file_style = SUFFIX_STYLE[suffix]
    elif file_style not in FILE_STYLES:
        raise ValueError(
            f"file_style: unknown file style {file_style!r}; known: {FILE_STYLES}"
        )
    if displacement_kind is not None and displacement_kind not in DISPLACEMENT_KINDS:
        raise ValueError(
            f"displacement_kind: unknown quantity {displacement_kind!r}; known: {DISPLACEMENT_KINDS}"
        )
    if position_kind is not None:
        if position_kind not in _POSITION_KINDS:
            raise ValueError(
                f"position_kind: unknown kind {position_kind!r}; known: {_POSITION_KINDS}"
            )
        if file_style not in _DEFAULT_POSITION_KIND:
            raise ValueError(
                f"position_kind: applies only to the readers {tuple(_DEFAULT_POSITION_KIND)}, "
                f"not to {file_style!r}"
            )
    if not source_file.is_file():
        raise FileNotFoundError(f"no such file: {source_file}")
    if file_style == "cif":
        return _cif(source_file, displacement_kind, position_kind)
    if file_style == "extxyz":
        return _extxyz(source_file, displacement_kind, position_kind)
    displacement_kind = displacement_kind or _DEFAULT_DISPLACEMENT_KIND[file_style]
    position_kind = position_kind or _DEFAULT_POSITION_KIND.get(file_style)
    lines = source_file.read_text().splitlines()
    parse = {"kirkland": _kirkland, "poscar": _poscar, "mustem": _mustem}[file_style]
    return parse(source_file, lines, displacement_kind, position_kind)
