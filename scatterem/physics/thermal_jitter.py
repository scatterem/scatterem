"""Frozen-phonon displacements of atoms and groups of rows at one site.

Literature: in the frozen-phonon method each atom moves by an independent
normal displacement with the variance ``<u^2>`` in each configuration
(Loane, Xu and Silcox 1991, Acta Cryst. A47, 267-278; Kirkland 2020,
*Advanced Computing in Electron Microscopy*, 3rd ed., ch. 7). The gradient
with respect to ``<u^2>`` is the pathwise derivative with common random
numbers (Glasserman 2004, *Monte Carlo Methods in Financial Engineering*,
ch. 7). Written for scatterem from this literature.

Units: displacements [Angstrom], ``<u^2>`` [Angstrom^2], fractional
coordinates [dimensionless].
"""

from __future__ import annotations

import logging
import math
import numbers
import secrets
from dataclasses import dataclass

import numpy as np
import torch

from ._stack_checks import check_stack_request
from .atom_set import AtomSet
from .row_replacements import RowReplacements, atom_data, common_device

__all__ = [
    "COLOCATION_TOLERANCE",
    "ThermalJitter",
    "colocated_rows",
    "jitter_offsets",
]

_log = logging.getLogger(__name__)

#: Largest distance [Angstrom] between rows that describe one site.
#: Reason: rows of a mixed site are copies of one written position; four
#: decimals in a 100 Angstrom cell give at most 0.005 Angstrom. The shortest
#: real distance between atoms (H-H, 0.74 Angstrom) is 74 times larger.
COLOCATION_TOLERANCE = 0.01

_MAX_BINS = 2**20


@dataclass(frozen=True)
class ThermalJitter:
    """Request for one frozen-phonon configuration of a build.

    Parameters
    ----------
    draw_seed : int, optional
        Integer in ``0 .. 2**63 - 1`` that fixes the draws. It seeds a local
        ``torch.Generator`` on the build device; the global generators do not
        change. Without ``draw_seed``, the object takes a fresh 63-bit value
        from :func:`secrets.randbits`, stores it and logs it at INFO, so a
        run can be repeated afterwards. Every build with one object uses
        its ``draw_seed``.
    transverse_only : bool, optional
        If True (default), the atoms move only along ``y`` and ``x``. If
        False, they also move along the beam.

    Raises
    ------
    TypeError
        If ``draw_seed`` is not an integer or ``transverse_only`` is not a
        bool.
    ValueError
        If ``draw_seed`` is outside ``0 .. 2**63 - 1``.

    Notes
    -----
    Stream order (float64 standard normal values on the build device):
    block 1 has the shape ``(c_y, c_x, G, 2)`` (copy ``i_y`` slowest, then
    ``i_x``, then the row group, then the axis ``(y, x)``). With
    ``transverse_only=False``, block 2 ``(c_y, c_x, G)`` follows. So block 1 is the same with and
    without block 2. CPU and CUDA generators give different streams.
    """

    draw_seed: int | None = None
    transverse_only: bool = True

    def __post_init__(self) -> None:
        seed = self.draw_seed
        if seed is None:
            seed = secrets.randbits(63)
            _log.info("ThermalJitter: no draw_seed given; using %d", seed)
        elif isinstance(seed, (bool, np.bool_)) or not isinstance(
            seed, numbers.Integral
        ):
            raise TypeError(f"draw_seed: must be an integer; got {seed!r}")
        seed = int(seed)
        if not 0 <= seed < 2**63:
            raise ValueError(f"draw_seed: must be in 0 .. 2**63 - 1; got {seed}")
        flat = self.transverse_only
        if not isinstance(flat, (bool, np.bool_)):
            raise TypeError(f"transverse_only: must be a bool; got {flat!r}")
        object.__setattr__(self, "draw_seed", seed)
        object.__setattr__(self, "transverse_only", bool(flat))

    @property
    def moves_along_beam(self) -> bool:
        """True if the atoms also move along the beam."""
        return not self.transverse_only


def _plain_fractional(source, sides):
    """Return float64 numpy coordinates (N, 3) and sides (3,)."""
    if isinstance(source, AtomSet):
        if sides is not None:
            raise ValueError("sides: must be None for an AtomSet source")
        return np.asarray(source.fractional, np.float64), np.asarray(source.sides)
    if sides is None:
        raise ValueError("sides: needed for an array source")
    if isinstance(source, torch.Tensor):
        source = source.detach().cpu().numpy()
    f = np.asarray(source, dtype=np.float64)
    s = np.asarray(sides, dtype=np.float64)
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"source: must have shape (N, 3); got {f.shape}")
    if s.shape != (3,) or not np.isfinite(s).all() or (s <= 0).any():
        raise ValueError(f"sides: must be 3 finite values > 0; got {sides!r}")
    if not np.isfinite(f).all():
        raise ValueError("source: all coordinates must be finite")
    return f, s


def _close_pairs(f, sides, tol, bins):
    """Return row pairs ``(i, j)``, ``i < j``, at distance <= ``tol``."""
    n = f.shape[0]
    w = f - np.floor(f)
    cell = np.minimum((w * bins).astype(np.int64), bins - 1)
    key = (cell[:, 0] * bins[1] + cell[:, 1]) * bins[2] + cell[:, 2]
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    rows = np.arange(n)
    found_i, found_j = [], []
    offsets = {
        tuple(int(v) % int(b) for v, b in zip((a, b_, c), bins, strict=True))
        for a in (-1, 0, 1)
        for b_ in (-1, 0, 1)
        for c in (-1, 0, 1)
    }
    for off in sorted(offsets):
        nb = (cell + np.array(off)) % bins
        nkey = (nb[:, 0] * bins[1] + nb[:, 1]) * bins[2] + nb[:, 2]
        start = np.searchsorted(sorted_key, nkey, side="left")
        stop = np.searchsorted(sorted_key, nkey, side="right")
        hits = stop - start
        if not hits.any():
            continue
        i = np.repeat(rows, hits)
        first = np.repeat(start - np.cumsum(hits) + hits, hits)
        j = order[first + np.arange(i.size)]
        keep = i < j
        found_i.append(i[keep])
        found_j.append(j[keep])
    if not found_i:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    i = np.concatenate(found_i)
    j = np.concatenate(found_j)
    d = f[i] - f[j]
    d -= np.round(d)
    near = np.max(np.abs(d * sides), axis=1) <= tol
    return i[near], j[near]


def colocated_rows(source, sides=None, *, tolerance=COLOCATION_TOLERANCE):
    """Return group labels of rows that describe one site.

    Two rows are in one group if a chain of rows links them, with each step
    at a minimum-image Chebyshev distance ``max_i |sides_i (d_i - round(d_i))|``
    of at most ``tolerance`` (``d`` the difference of fractional coordinates,
    all three box axes). The cell is periodic, so rows at ``0.0`` and
    ``0.99999`` can be one site. The chain rule makes the result independent
    of the row order.

    Parameters
    ----------
    source : AtomSet or array_like or torch.Tensor
        An ``AtomSet`` (then ``sides`` is None), or ``(N, 3)`` fractional
        coordinates together with ``sides``.
    sides : array_like, optional
        ``(3,)`` box side lengths [Angstrom] for an array source.
    tolerance : float, optional
        Largest step [Angstrom], finite, > 0 and below half of the smallest
        side. Default :data:`COLOCATION_TOLERANCE` (0.01 Angstrom).

    Returns
    -------
    numpy.ndarray
        The int64 labels ``(N,)`` in ``0 .. G - 1``. Labels follow the first row
        of each group in row order, so row 0 has label 0.

    Raises
    ------
    ValueError
        For a bad shape, a value that is not finite or a bad tolerance.

    Notes
    -----
    Rows go into periodic cube bins of side at least ``tolerance``; each row
    looks at the 27 neighbour bins, and a label propagation finds the
    connected components. Time and memory are O(N) for the bounded number of
    rows per bin of a real structure.
    """
    f, s = _plain_fractional(source, sides)
    if isinstance(tolerance, (bool, np.bool_)) or not isinstance(
        tolerance, numbers.Real
    ):
        raise TypeError(f"tolerance: must be a real number; got {tolerance!r}")
    tol = float(tolerance)
    if not (math.isfinite(tol) and 0.0 < tol < 0.5 * float(s.min())):
        raise ValueError(
            f"tolerance: must be finite, > 0 and below half the smallest side "
            f"{float(s.min())} Angstrom; got {tolerance!r}"
        )
    n = f.shape[0]
    if n == 0:
        return np.empty(0, np.int64)
    bins = np.maximum(np.minimum(np.floor(s / tol), _MAX_BINS), 1).astype(np.int64)
    i, j = _close_pairs(f, s, tol, bins)
    labels = np.arange(n, dtype=np.int64)
    while i.size:
        new = labels.copy()
        np.minimum.at(new, i, labels[j])
        np.minimum.at(new, j, labels[i])
        new = new[new]
        if np.array_equal(new, labels):
            break
        labels = new
    _, out = np.unique(labels, return_inverse=True)
    return out.astype(np.int64).reshape(n)


def group_msd(msd: torch.Tensor, labels: torch.Tensor, groups: int) -> torch.Tensor:
    """Return the mean ``<u^2>`` of each group [Angstrom^2], with the graph."""
    total = torch.zeros(groups, dtype=msd.dtype, device=msd.device)
    total = total.index_add(0, labels, msd)
    count = torch.bincount(labels, minlength=groups).to(msd.dtype)
    return total / count


def check_msd_derivative(msd: torch.Tensor, group_values: torch.Tensor) -> None:
    """Refuse a derivative request where a group has ``<u^2> = 0``."""
    if msd.requires_grad and bool((group_values.detach() == 0).any()):
        raise ValueError(
            "overrides.msd: requires grad, but a row group has <u^2> = 0; the "
            "derivative of sqrt(<u^2>) does not exist there. Start from a "
            "small positive value."
        )


def standard_draws(jitter: ThermalJitter, replicas, groups: int, device):
    """Return the standard normal blocks ``xi`` and ``zeta`` (or None)."""
    gen = torch.Generator(device=device)
    seed = jitter.draw_seed
    if seed is None:  # __post_init__ always sets it
        raise ValueError("ThermalJitter.draw_seed is not set")
    gen.manual_seed(seed)
    c_y, c_x = replicas
    opts = {"generator": gen, "dtype": torch.float64, "device": device}
    xi = torch.randn((c_y, c_x, groups, 2), **opts)
    zeta = torch.randn((c_y, c_x, groups), **opts) if jitter.moves_along_beam else None
    return xi, zeta


def offsets_from_data(data, sides, replicas, jitter, device):
    """Return offsets ``(c_y, c_x, N, 3)`` [Angstrom], labels and group msd."""
    frac = data["fractional"].detach().cpu().numpy()
    labels_np = colocated_rows(frac, sides)
    groups = int(labels_np.max()) + 1 if labels_np.size else 0
    labels = torch.from_numpy(labels_np).to(device)
    u_g = group_msd(data["msd"], labels, groups)
    check_msd_derivative(data["msd"], u_g)
    xi, zeta = standard_draws(jitter, replicas, groups, device)
    root = torch.sqrt(u_g)
    plane = root[None, None, :, None] * xi
    if zeta is None:
        beam = torch.zeros_like(plane[..., 0])
    else:
        beam = root[None, None, :] * zeta
    out = torch.cat([plane, beam[..., None]], dim=-1)
    return out[:, :, labels, :], labels, u_g


def jitter_offsets(
    atom_set: AtomSet,
    replicas,
    jitter: ThermalJitter,
    *,
    overrides: RowReplacements | None = None,
    device=None,
) -> torch.Tensor:
    """Return the thermal offsets of one build, per cell copy [Angstrom].

    Each row group (see :func:`colocated_rows`) of each cell copy moves by
    ``sqrt(u_g) * xi`` along ``y`` and ``x``, and with
    ``jitter.transverse_only=False`` by ``sqrt(u_g) * zeta`` along the beam. ``u_g`` is the arithmetic mean of
    the ``<u^2>`` of the rows of the group (replacement values when given):
    it is symmetric in the rows and smooth, so the pathwise derivative
    ``xi / (2 G_g sqrt(u_g))`` reaches each of the ``G_g`` rows. The values
    ``xi`` and ``zeta`` are the stream of :class:`ThermalJitter`, the same
    that :func:`~scatterem.physics.potential_stack.potential_stack` uses.

    Parameters
    ----------
    atom_set : AtomSet
        The structure.
    replicas : tuple of int
        Copy counts ``(c_y, c_x)``.
    jitter : ThermalJitter
        Draw seed and beam-direction switch.
    overrides : RowReplacements, optional
        Replacement values; the groups use the replaced coordinates.
    device : torch.device or str, optional
        Device. Default: the device of the replacement tensors, else the CPU.

    Returns
    -------
    torch.Tensor
        The float64 offsets ``(c_y, c_x, N, 3)`` [Angstrom]; axis 2 is 0 with
        ``transverse_only``. Copy ``(i_y, i_x)`` is the cell moved by
        ``(i_y sides[0], i_x sides[1])``. The fractional beam offset is
        ``offset[..., 2] / sides[2]``. The result carries the graph of
        ``overrides.msd``.

    Raises
    ------
    ValueError
        If ``overrides.msd`` requires grad and a group has ``u_g = 0``: the
        offset there is 0, but its derivative does not exist.
    """
    if not isinstance(jitter, ThermalJitter):
        raise TypeError(f"jitter: must be a ThermalJitter; got {jitter!r}")
    req = check_stack_request((1, 1), atom_set, replicas=replicas)
    if overrides is not None:
        if not isinstance(overrides, RowReplacements):
            raise TypeError(f"overrides: must be a RowReplacements; got {overrides!r}")
        overrides.checked_for(atom_set.row_total)
    given = () if overrides is None else overrides.tensors
    dev = torch.device(device) if device is not None else None
    dev = dev or common_device("overrides", given) or torch.device("cpu")
    data = atom_data(atom_set, dev, overrides)
    if atom_set.row_total == 0:
        return torch.zeros((*req.replicas, 0, 3), dtype=torch.float64, device=dev)
    out, _, _ = offsets_from_data(data, atom_set.sides, req.replicas, jitter, dev)
    return out
