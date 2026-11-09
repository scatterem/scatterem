"""Share of each atom in each slice along the beam.

Literature: the multislice slices follow Kirkland (2020, *Advanced Computing
in Electron Microscopy*, 3rd ed., ch. 6). The normal profile along the beam
is the Debye-Waller smearing of a vibrating atom (Kirkland 2020, ch. 7).
Written for scatterem from this literature.

Units: beam coordinates are fractional [dimensionless]; ``<u^2>``
[Angstrom^2]; the box depth [Angstrom]; shares [dimensionless].

Models (``BEAM_SPREADS``):

- ``"binned"``: share 1 in the slice that holds the reduced coordinate
  (:meth:`~scatterem.physics.beam_partition.BeamPartition.bin_index`).
- ``"tent"``: a tent profile between the two nearest slice midpoints,
  periodic along the beam.
- ``"smeared"``: the integral of a periodic normal density with standard
  deviation ``sqrt(<u^2>)`` over each slice.
"""

from __future__ import annotations

import logging
import math
from typing import NamedTuple

import torch

from .beam_partition import BeamPartition

__all__ = [
    "BEAM_SPREADS",
    "GAUSS_IMAGE_REACH",
    "AxialShares",
    "axial_shares",
]

_log = logging.getLogger(__name__)

#: Spellings of the models of the distribution along the beam.
BEAM_SPREADS = ("binned", "tent", "smeared")

#: Periodic images of the normal profile reach this many standard
#: deviations. Reason: the sum over all slices and images telescopes to
#: ``Phi((1 - r + M) / s) - Phi((-r - M) / s)`` with both arguments at least
#: 8 in magnitude, and ``Phi(-8) = 6.2e-16``; so the shares add up to 1
#: within ``1.3e-15`` plus rounding, 100 times below the bound ``1e-12``.
GAUSS_IMAGE_REACH = 8


class AxialShares(NamedTuple):
    """Sparse shares of items in slices.

    Entries are sorted by ``slab``, then by ``item``.

    Parameters
    ----------
    item : torch.Tensor
        The int64 ``(P,)`` index into the inputs.
    slab : torch.Tensor
        The int64 ``(P,)`` slice index.
    share : torch.Tensor
        The float64 ``(P,)`` share [dimensionless], with the autograd graph of
        the beam coordinates and ``<u^2>``.
    """

    item: torch.Tensor
    slab: torch.Tensor
    share: torch.Tensor


def check_spread(beam_spread) -> str:
    """Return a valid model spelling or raise."""
    if not isinstance(beam_spread, str):
        raise TypeError(f"beam_spread: must be a str; got {beam_spread!r}")
    if beam_spread not in BEAM_SPREADS:
        raise ValueError(
            f"beam_spread: must be one of {BEAM_SPREADS}; got {beam_spread!r}"
        )
    return beam_spread


def _reduce(z: torch.Tensor) -> torch.Tensor:
    """Reduce into ``[0, 1)`` and keep the derivative 1."""
    r = z - torch.floor(z.detach())
    return torch.where(r.detach() >= 1.0, r - 1.0, r)


def _sorted(item, slab, share) -> AxialShares:
    key = slab * (int(item.max()) + 1 if item.numel() else 1) + item
    order = torch.argsort(key, stable=True)
    return AxialShares(item[order], slab[order], share[order])


def _binned(partition, z, edge_snap, snap_dtype) -> AxialShares:
    k = partition.bin_index(z.detach(), edge_snap=edge_snap, margin_dtype=snap_dtype)
    keep = k >= 0
    item = torch.nonzero(keep).reshape(-1)
    share = torch.ones(item.shape, dtype=torch.float64, device=z.device)
    return _sorted(item, k[keep], share)


def _tent(partition: BeamPartition, z: torch.Tensor) -> AxialShares:
    s = partition.count
    dev = z.device
    if s == 1:
        item = torch.arange(z.shape[0], device=dev)
        share = torch.ones(z.shape, dtype=torch.float64, device=dev)
        return AxialShares(item, torch.zeros_like(item), share)
    e = partition.edges
    mids = torch.tensor(
        [float((e[k] + e[k + 1]) / 2) for k in range(s)],
        dtype=torch.float64,
        device=dev,
    )
    r = _reduce(z)
    k = torch.searchsorted(mids, r.detach().contiguous(), right=True) - 1
    before, after = k < 0, k >= s - 1
    low_slab = torch.where(before | after, torch.full_like(k, s - 1), k)
    high_slab = torch.where(before | after, torch.zeros_like(k), k + 1)
    lo = mids[low_slab] - before.to(torch.float64)
    hi = mids[high_slab] + after.to(torch.float64)
    width = hi - lo
    share_low = (hi - r) / width
    share_high = (r - lo) / width
    item = torch.arange(z.shape[0], device=dev)
    return _sorted(
        torch.cat([item, item]),
        torch.cat([low_slab, high_slab]),
        torch.cat([share_low, share_high]),
    )


def _smeared(partition, z, msd, depth, edge_snap, snap_dtype) -> AxialShares:
    s = partition.count
    dev = z.device
    sigma = torch.sqrt(msd) / float(depth)
    zero = msd.detach() == 0
    safe = torch.where(zero, torch.ones_like(sigma), sigma)
    reach = float(sigma.detach().max()) if sigma.numel() else 0.0
    m = math.ceil(GAUSS_IMAGE_REACH * reach)
    images = torch.arange(-m, m + 1, dtype=torch.float64, device=dev)
    e = partition.edge_values(torch.float64, dev)
    r = _reduce(z)
    # Arguments (K, S + 1, images) of Phi at each edge, per image.
    arg = (e[None, :, None] - r[:, None, None] - images[None, None, :]) / safe[
        :, None, None
    ]
    lo, hi = arg[:, :-1, :], arg[:, 1:, :]
    tail = torch.special.ndtr(-lo) - torch.special.ndtr(-hi)
    body = torch.special.ndtr(hi) - torch.special.ndtr(lo)
    shares = torch.where(lo > 0, tail, body).sum(dim=-1)
    whole = _binned(partition, z, edge_snap, snap_dtype)
    onehot = torch.zeros((z.shape[0], s), dtype=torch.float64, device=dev)
    onehot[whole.item, whole.slab] = 1.0
    shares = torch.where(zero[:, None], onehot, shares)
    slab = torch.arange(s, device=dev).repeat_interleave(z.shape[0])
    item = torch.arange(z.shape[0], device=dev).repeat(s)
    flat = shares.transpose(0, 1).reshape(-1)
    if not flat.requires_grad:
        keep = flat != 0
        return AxialShares(item[keep], slab[keep], flat[keep])
    return AxialShares(item, slab, flat)


def axial_shares(
    partition: BeamPartition,
    beam_fraction: torch.Tensor,
    msd: torch.Tensor,
    depth: float,
    *,
    beam_spread: str = "binned",
    edge_snap: bool = True,
    snap_dtype: torch.dtype | None = None,
) -> AxialShares:
    """Return the share of each item in each slice.

    Rules:

    - ``"binned"``: share 1 in ``partition.bin_index(r, edge_snap=...)``.
      Items in no slice get no entry.
    - ``"tent"``: with the midpoints ``m_k = (e_k + e_{k+1}) / 2`` and
      ``m_k <= r < m_{k+1}``, slice ``k`` gets ``(m_{k+1} - r) / (m_{k+1} -
      m_k)`` and slice ``k + 1`` gets ``(r - m_k) / (m_{k+1} - m_k)``. Below
      ``m_0`` the pair is ``(S - 1, 0)`` with midpoints ``m_{S-1} - 1`` and
      ``m_0``; at or after ``m_{S-1}`` it is ``(S - 1, 0)`` with ``m_{S-1}``
      and ``m_0 + 1``. On a midpoint the slice gets 1. At a midpoint,
      autograd gives the slope of the segment that starts there (the
      right-hand derivative).
    - ``"smeared"``: with ``s = sqrt(<u^2>) / depth``, slice ``k`` gets
      ``sum_{m=-M..M} [Phi((e_{k+1} - r - m) / s) - Phi((e_k - r - m) / s)]``
      with ``M = ceil(GAUSS_IMAGE_REACH * max s)``. Each difference is taken
      on the tail side. An item with ``<u^2> = 0`` gets the binned share,
      exactly.

    Parameters
    ----------
    partition : BeamPartition
        The slicing. ``"tent"`` and ``"smeared"`` need a partition that
        covers the cell.
    beam_fraction : torch.Tensor
        ``(K,)`` real beam coordinates [fractional], after any draw along the
        beam, not reduced.
    msd : torch.Tensor
        ``(K,)`` ``<u^2>`` of the row group of each item [Angstrom^2].
    depth : float
        Box length along the beam ``sides[2]`` [Angstrom].
    beam_spread : str, optional
        One of :data:`BEAM_SPREADS`. Default ``"binned"``.
    edge_snap : bool, optional
        Edge snap of the binned share (see :meth:`~scatterem.physics.beam_partition.BeamPartition.bin_index`).
    snap_dtype : torch.dtype, optional
        The dtype of the coordinate source, which sets the snap margin. Default
        the dtype of ``beam_fraction``.

    Returns
    -------
    AxialShares
        Sparse ``(item, slab, share)``. Memory is O(K) for ``"binned"`` and
        ``"tent"`` and O(K S) for ``"smeared"``.

    Raises
    ------
    ValueError
        For an unknown model, or ``"tent"``/``"smeared"`` with a partition
        that does not cover the cell.
    """
    if not isinstance(partition, BeamPartition):
        raise TypeError(f"partition: must be a BeamPartition; got {partition!r}")
    spread = check_spread(beam_spread)
    if spread != "binned" and not partition.covers_cell:
        raise ValueError(
            f"partition: beam_spread={spread!r} needs slices that cover the cell "
            f"from 0 to 1; got edges {partition.edges[0]} .. {partition.edges[-1]}"
        )
    if snap_dtype is None and isinstance(beam_fraction, torch.Tensor):
        snap_dtype = (
            torch.float32 if beam_fraction.dtype == torch.float32 else torch.float64
        )
    z = beam_fraction.to(torch.float64)
    if spread == "binned":
        return _binned(partition, z, edge_snap, snap_dtype)
    if spread == "tent":
        return _tent(partition, z)
    u = msd.to(device=z.device, dtype=torch.float64)
    return _smeared(partition, z, u, float(depth), edge_snap, snap_dtype)
