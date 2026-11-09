"""Per-site replacement tensors for the atom data of a potential build.

Literature: the replacement values feed the projected potential of
Kirkland (2020, *Advanced Computing in Electron Microscopy*, 3rd ed.,
ch. 5 to 7) and the pathwise derivatives of Glasserman (2004, *Monte Carlo
Methods in Financial Engineering*, ch. 7). Written for scatterem from this
literature.

Units: fractional coordinates [dimensionless], ``<u^2>`` [Angstrom^2], site
fractions [dimensionless].
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields

import numpy as np
import torch

from .atom_set import FULL_OCCUPANCY_TOLERANCE, AtomSet

__all__ = ["RowReplacements"]

_log = logging.getLogger(__name__)

_SHAPES = {
    "beam_fraction": (),
    "plane_fraction": (2,),
    "msd": (),
    "site_fraction": (),
}


def _check_field(name: str, value) -> int:
    """Check one replacement tensor and return its row count ``N``."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name}: must be a torch.Tensor; got {type(value).__name__}")
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name}: must be float32 or float64; got {value.dtype}")
    tail = _SHAPES[name]
    if value.dim() != 1 + len(tail) or tuple(value.shape[1:]) != tail:
        want = "(N,)" if not tail else "(N, 2)"
        raise ValueError(f"{name}: must have shape {want}; got {tuple(value.shape)}")
    plain = value.detach()
    if not bool(torch.isfinite(plain).all()):
        raise ValueError(f"{name}: all values must be finite")
    if name == "msd" and bool((plain < 0).any()):
        raise ValueError(f"{name}: all values must be >= 0 [Angstrom^2]")
    if name == "site_fraction":
        top = 1.0 + FULL_OCCUPANCY_TOLERANCE
        if bool((plain < 0).any()) or bool((plain > top).any()):
            raise ValueError(f"{name}: all values must be in [0, {top}]")
    return int(value.shape[0])


@dataclass(frozen=True, eq=False)
class RowReplacements:
    """Tensors that replace per-atom data of an ``AtomSet`` in a build.

    Each field is optional. A given field has one entry per atom row and
    replaces that data of the ``AtomSet``. Each tensor may require grad; the
    build keeps its graph.

    Parameters
    ----------
    beam_fraction : torch.Tensor, optional
        ``(N,)`` fractional beam coordinates [dimensionless].
    plane_fraction : torch.Tensor, optional
        ``(N, 2)`` fractional coordinates along ``y`` and ``x``.
    msd : torch.Tensor, optional
        ``(N,)`` mean-square displacement ``<u^2>`` [Angstrom^2], >= 0.
    site_fraction : torch.Tensor, optional
        ``(N,)`` site fractions in ``[0, 1 + FULL_OCCUPANCY_TOLERANCE]``.
        Values above 1 are used as given (a clip has no derivative). A build
        with ``ignore_site_fraction=True`` refuses this field, because it
        would ignore the values without notice.

    Raises
    ------
    TypeError
        If a field is not a tensor, or has an integer, bool or complex dtype.
    ValueError
        If a shape is wrong, the fields disagree on ``N``, a value is not
        finite, or an ``msd`` or site fraction is out of range.

    Notes
    -----
    The discrete decisions of a build (slice of an atom, groups of rows at
    one site, whether site fractions are in use) use the replacement values
    as an ``AtomSet`` with these values would. The object stores the
    tensors as given, without a copy.
    """

    beam_fraction: torch.Tensor | None = None
    plane_fraction: torch.Tensor | None = None
    msd: torch.Tensor | None = None
    site_fraction: torch.Tensor | None = None

    def __post_init__(self) -> None:
        counts = {
            f.name: _check_field(f.name, getattr(self, f.name))
            for f in fields(self)
            if getattr(self, f.name) is not None
        }
        if len(set(counts.values())) > 1:
            raise ValueError(f"overrides: fields differ in row count N; got {counts}")

    @property
    def tensors(self) -> tuple[torch.Tensor, ...]:
        """The given replacement tensors, in field order."""
        return tuple(
            getattr(self, f.name)
            for f in fields(self)
            if getattr(self, f.name) is not None
        )

    @property
    def row_count(self) -> int | None:
        """Common ``N`` of the given fields, or None without fields."""
        given = self.tensors
        return int(given[0].shape[0]) if given else None

    def checked_for(self, row_total: int) -> RowReplacements:
        """Return this object after a check against an atom count.

        Parameters
        ----------
        row_total : int
            Number of rows of the ``AtomSet``.

        Returns
        -------
        RowReplacements
            This object.

        Raises
        ------
        ValueError
            If the fields have another ``N``.
        """
        n = self.row_count
        if n is not None and n != int(row_total):
            raise ValueError(
                f"overrides: fields have N = {n}; the atom set has {row_total} rows"
            )
        return self

    def apply_to(self, atom_set: AtomSet, device) -> dict[str, torch.Tensor]:
        """Return the atom data with the replacement values in place.

        Parameters
        ----------
        atom_set : AtomSet
            The structure.
        device : torch.device or str
            Target device.

        Returns
        -------
        dict
            The float64 tensors on ``device``: ``"fractional"`` ``(N, 3)``,
            ``"msd"`` ``(N,)`` [Angstrom^2] and ``"site_fraction"`` ``(N,)``.
            A float32 replacement converts exactly to float64; the graph
            stays.
        """
        self.checked_for(atom_set.row_total)
        return atom_data(atom_set, device, self)


def atom_data(atom_set: AtomSet, device, overrides: RowReplacements | None):
    """Return float64 atom data on ``device`` with optional replacements."""
    f = torch.from_numpy(np.array(atom_set.fractional, dtype=np.float64)).to(device)

    def pick(name, base):
        value = None if overrides is None else getattr(overrides, name)
        if value is None:
            return base
        return value.to(device=device, dtype=torch.float64)

    plane = pick("plane_fraction", f[:, 0:2])
    beam = pick("beam_fraction", f[:, 2])
    msd = pick(
        "msd", torch.from_numpy(np.array(atom_set.msd, dtype=np.float64)).to(device)
    )
    fill = pick(
        "site_fraction",
        torch.from_numpy(np.array(atom_set.site_fraction, dtype=np.float64)).to(device),
    )
    return {
        "fractional": torch.cat([plane, beam[:, None]], dim=1),
        "msd": msd,
        "site_fraction": fill,
    }


def common_device(name: str, tensors) -> torch.device | None:
    """Return the one device of some tensors, or None without tensors."""
    devices = []
    for t in tensors:
        if isinstance(t, torch.Tensor) and t.device not in devices:
            devices.append(t.device)
    if len(devices) > 1:
        raise ValueError(
            f"{name}: tensor inputs are on more than one device: "
            f"{[str(d) for d in devices]}; give device= or move them"
        )
    return devices[0] if devices else None
