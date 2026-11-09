"""Reuse of one full-grid NNW reconstruction per slice in ``prism_eels_image``.

A partitioned ``S1`` or ``S2`` rebuilds its columns from the ``Bp`` parents on
every atom's crop window. When the windows of one slice together cover at
least the whole grid, one pointwise reconstruction on the full grid does the
same work once, and each atom then crops it. The reconstruction is pointwise,
so a crop of the full-grid result equals the windowed result to rounding.

The reuse is bounded by an additional-workspace budget that both legs share
(``prism_eels_image(reconstruction_workspace=...)``; the default is
:data:`_EELS_RECONSTRUCTION_BYTES`, and 0 turns the reuse off). It is not used
when the windows are sparse, when the focal referencing is on (the windowed
propagation is not pointwise), or when ``S`` needs gradients.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

# Default additional reconstruction workspace (bytes), shared by both legs. The
# estimate in _reconstruct_slice_or_none includes double-precision phase
# temporaries even for complex64 wave fields. It does not bound the total
# simulation memory. ``prism_eels_image(reconstruction_workspace=...)``
# overrides it for one call; 0 gives the windowed path of the BiP-PRISM paper.
_EELS_RECONSTRUCTION_BYTES = 1 << 31


def _workspace_bytes(value: int | None) -> int | None:
    """Check ``prism_eels_image(reconstruction_workspace=...)``: bytes >= 0 or None."""
    if value is None:
        return None
    if int(value) < 0:
        raise ValueError(
            f"reconstruction_workspace must be >= 0 bytes or None: {value!r}"
        )
    return int(value)


def _reconstruct_slice_or_none(
    matrix, n_targets: int, n_active: int, wy: int, wx: int, budget: int
) -> Tensor | None:
    """Full-grid NNW columns of ``matrix`` when the slice's windows cover the grid.

    The window method keeps the magnitude replacement and the ``S2`` vacuum
    phase. ``S1.reconstruct_columns()`` has different magnitude semantics on
    purpose, so it is not used here.

    Parameters
    ----------
    matrix : PartitionedScatteringMatrix or DetectorExitSMatrix
        The partitioned matrix at the current plane.
    n_targets : int
        Number of reconstructed columns (aperture or detector beams).
    n_active : int
        Number of atoms of the slice with at least one scan position.
    wy, wx : int
        Crop window shape.
    budget : int
        Additional workspace (bytes) still available to this reconstruction.

    Returns
    -------
    Tensor or None
        The reconstructed columns ``(n_targets, Ny, Nx)``, or ``None`` when
        the reuse does not apply. ``None`` costs no device work.
    """
    ny, nx = matrix.S.shape[-2:]
    if n_active < 2 or n_active * wy * wx < ny * nx:
        return None
    if matrix._backprop_slices() or matrix.S.requires_grad:
        return None
    # Eight complex128-sized live fields per target and parent conservatively
    # cover the phase construction, the interpolation, the magnitude
    # replacement and their copies.
    workspace = 8 * 16 * (n_targets + matrix.S.shape[0]) * ny * nx
    if workspace > budget:
        return None
    iy = torch.arange(ny, device=matrix.S.device)
    ix = torch.arange(nx, device=matrix.S.device)
    return matrix._nnw_window(matrix.S, iy, ix)


def _slice_reconstructions(
    S1,
    S2,
    partitioned: bool,
    n_beams: int,
    counts: Sequence[int],
    wy: int,
    wx: int,
    budget: int | None = None,
) -> tuple[Tensor | None, Tensor | None]:
    """One slice's reusable ``S1`` and ``S2`` reconstructions, or ``None`` each.

    The ``S1`` result, when there is one, is subtracted from the budget that
    the ``S2`` reconstruction may use, because it stays alive with it.

    Parameters
    ----------
    S1 : PartitionedScatteringMatrix or ScatteringMatrix
        Probe matrix at the slice.
    S2 : DetectorExitSMatrix
        Detector matrix at the slice.
    partitioned : bool
        Whether ``S1`` is partitioned.
    n_beams : int
        Number of aperture beams of ``S1``.
    counts : sequence of int
        Scan positions per atom of the slice.
    wy, wx : int
        Crop window shape.
    budget : int or None, optional
        Additional workspace (bytes) for both reconstructions. ``None`` reads
        :data:`_EELS_RECONSTRUCTION_BYTES` at call time; 0 turns the reuse off.

    Returns
    -------
    (Tensor or None, Tensor or None)
        Full-grid ``S1`` columns ``(n_beams, Ny, Nx)`` and ``S2`` columns
        ``(ndet, Ny, Nx)``.
    """
    active = sum(int(count) > 0 for count in counts)
    if budget is None:
        budget = _EELS_RECONSTRUCTION_BYTES
    s1_recon = s2_recon = None
    if partitioned:
        s1_recon = _reconstruct_slice_or_none(S1, n_beams, active, wy, wx, budget)
        if s1_recon is not None:
            budget -= s1_recon.numel() * s1_recon.element_size()
    if S2.partitioned:
        s2_recon = _reconstruct_slice_or_none(S2, S2.ndet, active, wy, wx, budget)
    return s1_recon, s2_recon


def _s1_window(
    S1, recon: Tensor | None, partitioned: bool, iy: Tensor, ix: Tensor
) -> Tensor:
    """``S1`` columns on the crop window ``(Nbeams, wy, wx)``.

    A crop of the slice's full-grid reconstruction when there is one, else the
    windowed NNW reconstruction (partitioned) or a direct crop.
    """
    if recon is not None:
        return recon[:, iy[:, None], ix[None, :]]
    if partitioned:
        return S1.reconstruct_columns_window(iy, ix)
    return S1.S[:, iy][:, :, ix]


def _s2_window(S2, recon: Tensor | None, iy: Tensor, ix: Tensor) -> Tensor:
    """``S2`` columns on the crop window ``(ndet, wy, wx)``.

    A crop of the slice's full-grid reconstruction when there is one, else
    :meth:`DetectorExitSMatrix.columns_window`.
    """
    if recon is not None:
        return recon[:, iy[:, None], ix[None, :]]
    return S2.columns_window(iy, ix)
