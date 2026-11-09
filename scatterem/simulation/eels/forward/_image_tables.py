"""Host-side tables of the double-channeling PRISM STEM-EELS image.

Split out of ``prism_eels_image.py`` unchanged, plus the ``"both"`` qEELS
binning: where the atoms are (slice grouping, crop windows, sub-pixel shifts,
the scan positions each atom reaches), which detector beams are collected, and
how their intensities are binned. All bookkeeping -- no contraction, so none
of the bit-identity contracts of the accumulation body are involved.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from ._detector_exit import _periodic_window_stack


def _crop_window(inelastic_crop: int | None, ny: int, nx: int) -> tuple[int, int]:
    """The ``(wy, wx)`` transition-potential window, full grid when not cropped."""
    if inelastic_crop is None:
        return ny, nx
    crop = int(inelastic_crop)
    return max(1, ny // crop), max(1, nx // crop)


def _site_shifts(sites: np.ndarray, ny: int, nx: int):
    """Every atom's sub-pixel shift, tabulated once.

    Same float64 arithmetic as the per-atom ``ay - round(ay) + ny // 2``; numpy
    and python both round half to even, so the table is bit-identical to the
    per-atom spelling it replaces.
    """
    site_y = sites[:, 0] % 1.0 * ny
    site_x = sites[:, 1] % 1.0 * nx
    shifts = np.stack(
        [site_y - np.round(site_y) + ny // 2, site_x - np.round(site_x) + nx // 2],
        axis=1,
    )
    return site_y, site_x, shifts


def _slice_grouping(site_slice: np.ndarray, nz: int):
    """Atom indices per slice, and the site order that concatenates them.

    The window tables are built in this SLICE-GROUPED order so each slice's
    block of window rows is a ``torch.split`` view of the same single host
    transfer. Slicing a site-ordered device stack with the numpy ``in_slice``
    instead would be a *synchronous pageable H2D of the index* twice per slice:
    measured at ~227 us of host stall each, which is more than the batched
    gather saves on a call whose host is the critical path.
    """
    slice_groups = [np.nonzero(site_slice == i)[0] for i in range(nz)]
    group_counts = [int(group.size) for group in slice_groups]
    order = (
        np.concatenate(slice_groups)
        if any(group_counts)
        else np.zeros(0, dtype=np.int64)
    )
    return slice_groups, group_counts, order


def _detector_beams(
    gpts: tuple[int, int],
    gridsize: Sequence[float],
    wavelength: float,
    detector_mrad: float,
    factors: tuple[int, int],
    device,
) -> Tensor:
    """Grid pixels inside the collection semi-angle, PRISM-subsampled.

    S2 is a PRISM scattering matrix too, so its output beams are decimated by
    the same interpolation factor as S1 -- without that S2 would carry ~f**2 too
    many columns and be intractable. ``f = 1`` keeps every beam.
    """
    ny, nx = gpts
    fy, fx = factors
    qy = torch.fft.fftfreq(ny, d=gridsize[0] / ny, device=device)
    qx = torch.fft.fftfreq(nx, d=gridsize[1] / nx, device=device)
    q2 = qy[:, None] ** 2 + qx[None, :] ** 2
    beta_q = detector_mrad / 1000.0 / wavelength
    iy_grid = torch.arange(ny, device=device)[:, None].expand(ny, nx)
    ix_grid = torch.arange(nx, device=device)[None, :].expand(ny, nx)
    det_mask = (q2 <= beta_q**2) & (iy_grid % fy == 0) & (ix_grid % fx == 0)
    return torch.nonzero(det_mask, as_tuple=False)  # (ndet, 2)


def _masked_scan_tables(
    sites: np.ndarray,
    order: np.ndarray,
    offs: np.ndarray,
    group_counts: list[int],
    scan: tuple[Tensor, Tensor],
    gpts: tuple[int, int],
    factors: tuple[int, int],
    device,
) -> tuple[list, list, list]:
    """Which scan positions each atom contributes to, per slice.

    The PRISM crop mask is a function of the SITE and the scan grid only -- both
    known before the slice loop -- so the whole ``(Nsite, P)`` mask is built in
    one broadcast pass and converted to per-atom scan-index lists in ONE host
    transfer. The per-atom form paid ~14 elementwise launches, a
    ``bool(mask.any())`` DEVICE SYNC and two boolean-mask ``nonzero``s per
    ionized atom, on a call that is ~2/3 host-issue.

    The arithmetic is unchanged: broadcasting a ``(Nsite, 1)`` float64 site
    coordinate against the ``(P,)`` scan coordinate forms the same float64
    products as the scalar-per-atom spelling, and ``nonzero`` returns indices in
    ascending order so an ``index_select`` gathers exactly what the boolean mask
    did.

    Rows are in the same SLICE-GROUPED order as the window tables, so each
    slice's atoms are one contiguous block of the single ``nonzero`` -- a
    ``torch.split`` view, costing nothing. Reordering the rows does not change
    any atom's indices: ``nonzero`` is row-major and ascending within a row
    either way.
    """
    ny, nx = gpts
    fy, fx = factors
    ry, rx = scan
    nz = len(group_counts)
    n_site = int(sites.shape[0])
    scan_by_slice: list = [None] * nz
    scan_idx_by_slice: list = [None] * nz
    atom_counts_by_slice: list = [None] * nz
    sy_t = torch.as_tensor(sites[order, 0], dtype=torch.float64, device=device)
    sx_t = torch.as_tensor(sites[order, 1], dtype=torch.float64, device=device)
    mask_all = (
        torch.abs((ry / ny - sy_t.view(-1, 1) + 0.5) % 1.0 - 0.5) <= 0.5 / fy
    ) & (torch.abs((rx / nx - sx_t.view(-1, 1) + 0.5) % 1.0 - 0.5) <= 0.5 / fx)
    nzi = torch.nonzero(mask_all, as_tuple=False)  # (M, 2), row-major ascending
    counts = torch.bincount(nzi[:, 0], minlength=n_site).tolist()  # one transfer
    flat_scan = nzi[:, 1].contiguous()
    per_slice = [int(sum(counts[offs[i] : offs[i + 1]])) for i in range(nz)]
    blocks = list(torch.split(flat_scan, per_slice))
    for i in range(nz):
        if group_counts[i] == 0:
            continue
        atom_counts_by_slice[i] = counts[offs[i] : offs[i + 1]]
        scan_by_slice[i] = blocks[i]
        scan_idx_by_slice[i] = list(torch.split(blocks[i], atom_counts_by_slice[i]))
    return scan_by_slice, scan_idx_by_slice, atom_counts_by_slice


def _fold_detectors(power: Tensor, q_bin: Tensor | None, n_q: int) -> Tensor:
    """Reduce ``(..., ndet)`` intensity over the detector beams.

    With ``q_bin`` ``None`` this is the scalar map's plain sum. Otherwise each
    beam is added into its momentum bin (qEELS), so the trailing axis becomes
    ``n_q`` and summing THAT axis reproduces the plain sum exactly -- both are
    the same float64 additions, only grouped differently.

    Callers keep the non-qEELS spelling inline rather than routing it through
    here, because the scalar path's reduction order is pinned by the bit-identity
    contracts in the contraction body.
    """
    if q_bin is None:
        return power.sum(dim=-1)
    out = torch.zeros(power.shape[:-1] + (n_q,), dtype=power.dtype, device=power.device)
    return out.index_add_(-1, q_bin, power)


def _qeels_bins(
    det_idx: Tensor,
    qeels_axis: int | Literal["both"] | None,
    grid: tuple[int, int],
    P: int,
    device,
) -> tuple[Tensor, Tensor | None, int, Tensor | None]:
    """The accumulator and, for qEELS, the detector-beam -> momentum-bin map.

    Each beam is binned by its SIGNED frequency along the resolved axis; the
    perpendicular axis is projected away by the binning itself (every beam
    lands in a bin regardless of its perpendicular momentum). ``None`` keeps the
    scalar map and a flat ``(P,)`` accumulator. ``"both"`` keeps every beam in
    its own bin, with the ``(n_q, 2)`` signed ``(qy, qx)`` pixel frequencies.
    """
    ny, nx = grid
    if qeels_axis is None:
        return torch.zeros(P, dtype=torch.float64, device=device), None, 0, None
    if isinstance(qeels_axis, str):
        if qeels_axis != "both":
            raise ValueError("qeels_axis must be None, 0, 1, or 'both'")
        sizes = torch.tensor([ny, nx], device=det_idx.device)
        q_pix = (det_idx + sizes // 2) % sizes - sizes // 2  # (ndet, 2) signed
        n_q = int(det_idx.shape[0])
        q_bin = torch.arange(n_q, device=det_idx.device)
        img = torch.zeros((P, n_q), dtype=torch.float64, device=device)
        return img, q_bin, n_q, q_pix
    if qeels_axis not in (0, 1):
        raise ValueError("qeels_axis must be None, 0, 1, or 'both'")
    a = int(qeels_axis)
    nres = ny if a == 0 else nx
    g_res = (det_idx[:, a] + nres // 2) % nres - nres // 2  # signed (ndet,)
    q_pix, q_bin = torch.unique(g_res, sorted=True, return_inverse=True)
    n_q = int(q_pix.shape[0])
    img = torch.zeros((P, n_q), dtype=torch.float64, device=device)
    return img, q_bin, n_q, q_pix


def _slice_tables(
    sites: np.ndarray,
    site_slice: np.ndarray,
    nz: int,
    grid: tuple[int, int],
    window: tuple[int, int],
    scan_yx: tuple[Tensor, Tensor],
    factor: tuple[int, int],
    P: int,
    prism_mask: bool,
    device,
):
    """Everything the slice loop reads that depends only on WHERE the atoms are.

    Split out of :func:`prism_eels_image` unchanged: the atom grouping, each
    atom's crop-window indices, and the scan positions it contributes to. All
    host-side bookkeeping -- no contraction, so none of the bit-identity
    contracts in the accumulation body are involved.
    """
    ny, nx = grid
    wy, wx = window
    ry, rx = scan_yx
    fy, fx = factor
    site_y, site_x, all_shifts = _site_shifts(sites, ny, nx)
    slice_groups, group_counts, order = _slice_grouping(site_slice, nz)
    iy_all = _periodic_window_stack(np.round(site_y)[order], wy, ny, device)
    ix_all = _periodic_window_stack(np.round(site_x)[order], wx, nx, device)
    # The same windows as row-major flat indices into the (ny, nx) grid, for the
    # whole call in one launch, so each slice's crop stack is one `index_select`.
    lin_all = (iy_all[:, :, None] * nx + ix_all[:, None, :]).reshape(-1, wy * wx)

    # Which scan positions each atom contributes to -- see _masked_scan_tables for
    # why the whole (Nsite, P) mask is built in one broadcast pass, and why its
    # rows are in the same SLICE-GROUPED order as the window tables above.
    offs = np.concatenate([[0], np.cumsum(group_counts)]).astype(np.int64)
    scan_by_slice: list = [None] * nz
    scan_idx_by_slice: list = [None] * nz
    atom_counts_by_slice: list = [None] * nz
    if prism_mask:
        scan_by_slice, scan_idx_by_slice, atom_counts_by_slice = _masked_scan_tables(
            sites, order, offs, group_counts, (ry, rx), (ny, nx), (fy, fx), device
        )
    else:
        every = torch.arange(P, device=device)
        for i in range(nz):
            if group_counts[i] == 0:
                continue
            atom_counts_by_slice[i] = [P] * group_counts[i]
            scan_idx_by_slice[i] = [every] * group_counts[i]
    return (
        all_shifts,
        slice_groups,
        *(list(torch.split(t, group_counts)) for t in (iy_all, ix_all, lin_all)),
        scan_by_slice,
        scan_idx_by_slice,
        atom_counts_by_slice,
    )
