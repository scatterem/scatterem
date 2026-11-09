"""Input checks shared by the routes that build projected potentials.

Literature: the inputs describe the periodic field of view and the grid of
the projected potential of Kirkland (2020, *Advanced Computing in Electron
Microscopy*, 3rd ed., ch. 5 and 6). Written for scatterem from this
literature.

Units: lengths [Angstrom]; grid point counts and copy counts are integers.
"""

from __future__ import annotations

import logging
import numbers
from typing import NamedTuple

import numpy as np
import torch

from .atom_set import AtomSet
from .row_replacements import common_device

__all__ = [
    "StackRequest",
    "check_stack_request",
    "resolve_stack_device",
]

_log = logging.getLogger(__name__)

_COMPLEX_PARTNER = {torch.complex64: torch.float32, torch.complex128: torch.float64}


class StackRequest(NamedTuple):
    """Validated values of a potential request.

    Parameters
    ----------
    replicas : tuple of int
        Copy counts ``(c_y, c_x)``.
    lengths : tuple of float
        Field-of-view lengths ``(c_y sides[0], c_x sides[1])`` [Angstrom].
    points : tuple of int
        Grid point counts ``(Ny, Nx)``.
    real_dtype : torch.dtype
        The real dtype, float32 or float64.
    """

    replicas: tuple[int, int]
    lengths: tuple[float, float]
    points: tuple[int, int]
    real_dtype: torch.dtype


def _pair(value, name: str) -> tuple[int, int]:
    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__"):
        raise TypeError(f"{name}: must be a pair of integers; got {value!r}")
    items = list(value)
    if len(items) != 2:
        raise ValueError(f"{name}: must hold 2 integers; got {len(items)} values")
    for v in items:
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Integral):
            raise TypeError(f"{name}: must hold integers; got {value!r}")
    out = (int(items[0]), int(items[1]))
    if min(out) < 1:
        raise ValueError(f"{name}: each value must be at least 1; got {out}")
    return out


def real_dtype_for(dtype) -> torch.dtype:
    """Return the real build dtype for a requested dtype.

    ``None`` gives ``torch.get_default_dtype()``. A complex dtype gives its
    real partner (``complex64 -> float32``, ``complex128 -> float64``).
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    if dtype in (torch.float32, torch.float64):
        return dtype
    if dtype in _COMPLEX_PARTNER:
        return _COMPLEX_PARTNER[dtype]
    raise TypeError(
        f"dtype: must be float32, float64, complex64 or complex128; got {dtype!r}"
    )


def scale_values(atom_scale, rows: int):
    """Check per-atom coefficients; return a tensor or float64 numpy array."""
    if isinstance(atom_scale, torch.Tensor):
        if atom_scale.is_complex() or atom_scale.dtype == torch.bool:
            raise TypeError(f"atom_scale: must be real; got dtype {atom_scale.dtype}")
        values = atom_scale
        plain = atom_scale.detach()
    else:
        try:
            arr = np.asarray(atom_scale)
        except (TypeError, ValueError):
            raise TypeError(
                f"atom_scale: must be numeric; got {atom_scale!r}"
            ) from None
        if arr.dtype.kind not in "iuf":
            raise TypeError(f"atom_scale: must be real numbers; got dtype {arr.dtype}")
        values = arr.astype(np.float64)
        plain = torch.from_numpy(values)
    if tuple(values.shape) != (rows,):
        raise ValueError(
            f"atom_scale: must have shape ({rows},); got {tuple(values.shape)}"
        )
    if not bool(torch.isfinite(plain.to(torch.float64)).all()):
        raise ValueError("atom_scale: all values must be finite")
    return values


def check_stack_request(
    points,
    atom_set,
    *,
    replicas=None,
    atom_scale=None,
    scale_squared: bool = False,
    dtype=None,
) -> StackRequest:
    """Check the shared inputs of a potential build.

    Parameters
    ----------
    points : tuple of int
        Grid point counts ``(Ny, Nx)``, each at least 1.
    atom_set : AtomSet
        The structure.
    replicas : tuple of int, optional
        Copy counts ``(c_y, c_x)`` of the cell, each at least 1. Default
        None: one copy per axis, ``(1, 1)``.
    atom_scale : array_like or torch.Tensor, optional
        Per-atom real, finite coefficients ``(N,)``.
    scale_squared : bool, optional
        Use ``c_j**2`` instead of ``c_j``.
    dtype : torch.dtype, optional
        One of float32, float64, complex64 or complex128. Default the torch default
        dtype. A complex dtype selects its real partner.

    Returns
    -------
    StackRequest
        ``(replicas, lengths, points, real_dtype)`` with Python ints and
        floats; ``lengths`` [Angstrom].

    Raises
    ------
    TypeError
        For a value of a wrong type (also ``bool`` as an integer, a complex
        or non-numeric ``atom_scale``, a bad ``dtype``).
    ValueError
        For a wrong pair length, a count below 1, or an ``atom_scale`` with a
        wrong shape or a value that is not finite.
    """
    if not isinstance(atom_set, AtomSet):
        raise TypeError(f"atom_set: must be an AtomSet; got {type(atom_set).__name__}")
    pts = _pair(points, "points")
    reps = (1, 1) if replicas is None else _pair(replicas, "replicas")
    if atom_scale is not None:
        scale_values(atom_scale, atom_set.row_total)
    if not isinstance(scale_squared, (bool, np.bool_)):
        raise TypeError(f"scale_squared: must be a bool; got {scale_squared!r}")
    real = real_dtype_for(dtype)
    sides = atom_set.sides
    lengths = (float(reps[0] * sides[0]), float(reps[1] * sides[1]))
    return StackRequest(reps, lengths, pts, real)


def resolve_stack_device(device, *tensors) -> torch.device:
    """Return the device of a build.

    Rules:

    1. A given ``device`` wins. Tensor inputs move to it.
    2. Else the device of the tensor inputs (form spectra, coefficients,
       replacement tensors). Inputs on more than one device raise.
    3. Else the CPU. The rule never selects a GPU by itself.

    Parameters
    ----------
    device : torch.device or str or None
        Requested device.
    *tensors : object
        Inputs; only ``torch.Tensor`` objects count.

    Returns
    -------
    torch.device
        The device of the build.

    Raises
    ------
    ValueError
        If ``device`` is None and the tensors are on more than one device.
    """
    if device is not None:
        return torch.device(device)
    found = common_device("tensor inputs", tensors)
    return found if found is not None else torch.device("cpu")
