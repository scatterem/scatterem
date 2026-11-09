"""The per-slice plan that ``prism_eels_image`` tabulates before its slice loop."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from torch import Tensor

from scatterem.simulation._wave.partitioned_smatrix import PartitionedScatteringMatrix
from scatterem.simulation._wave.smatrix import ScatteringMatrix

from ._detector_exit import DetectorExitSMatrix
from ._scan_accumulator import _OrderedScanAccumulator


class _SlicePlan(NamedTuple):
    """Everything the per-slice contraction reads, tabulated before the loop.

    A NamedTuple rather than a dataclass so the body below can unpack it in
    one statement: these names were all locals of one very long function, and
    the point of the split is to move the contraction out without renaming
    anything inside it.
    """

    S1: PartitionedScatteringMatrix | ScatteringMatrix
    S2: DetectorExitSMatrix
    #: the image accumulator (``img`` of shape ``(P,)`` or ``(P, n_q)``)
    acc: _OrderedScanAccumulator
    scan_pixels: Tensor
    # qEELS binning: `None` keeps the scalar map, otherwise (ndet,) -> bin index.
    q_bin: Tensor | None
    n_q: int
    partitioned: bool
    n_trans: int
    wy: int
    wx: int
    H: Tensor
    Hq: Tensor
    ky: Tensor
    kx: Tensor
    cy: Tensor
    cx: Tensor
    all_shifts: np.ndarray
    #: per slice: the row indices of its sites, in slice-grouped order
    slice_groups: list[np.ndarray]
    iy_by_slice: list[Tensor]
    ix_by_slice: list[Tensor]
    #: per slice: the windows as flat ``iy * nx + ix`` indices, ``(na, wy*wx)``
    lin_by_slice: list[Tensor]
    scan_by_slice: list
    scan_idx_by_slice: list
    atom_counts_by_slice: list
    #: slice-loop invariants carried from one slice to the next (see
    #: `_accumulate_slice`): ``coeffs``, and the last ``shifts`` / ``h_crops`` pair
    reuse: dict
    #: contraction order: ``"auto"``, ``"window"`` (the ``"columns"`` contract)
    #: or ``"probe"``
    order: str = "auto"
    #: most scan positions one atom contracts at once (``None``: no bound)
    scan_chunk: int | None = None
    #: full-grid reconstruction workspace in bytes (``None``: module default)
    workspace: int | None = None
