"""The PRISM parent waves and the build's propagator check (``PrismScanOp``).

Moved verbatim out of :mod:`.prism` (a frozen size offender) so that the L2
block-graph routes of :mod:`._prism_s_build_graph` could be wired into the
S-matrix build without growing it.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["check_propagator_grid", "initial_parent_waves"]


def check_propagator_grid(propagator, MY: int, MX: int) -> None:
    """Refuse a propagator not built for the full ``(MY, MX)`` simulation grid.

    Parameters
    ----------
    propagator : Propagator
        The inter-slice propagator handed to the build.
    MY, MX : int
        The simulation grid ``s_meta.M``.

    Raises
    ------
    ValueError
        If the propagator's shape is not ``(MY, MX)``.
    """
    prop_shape = tuple(propagator._shape)
    if prop_shape != (MY, MX):
        raise ValueError(
            f"Propagator shape {prop_shape} does not match s_meta.M "
            f"({MY}, {MX}). PRISM requires a propagator built for the "
            f"full simulation grid; rebuild via Propagator(..., "
            f"shape=s_meta.M, ...) before calling forward."
        )


def initial_parent_waves(op, lo: int = 0, hi: int | None = None) -> Tensor:
    """``PrismScanOp._initial_parent_waves``: IFFT of a delta at each parent beam.

    Parameters
    ----------
    op : PrismScanOp
        The operator (its ``s_meta``, ``fy`` / ``fx`` and build dtype).
    lo, hi : int
        The beam range ``[lo, hi)``; ``hi=None`` is ``Bp``.

    Returns
    -------
    torch.Tensor
        ``(hi - lo, MY, MX)`` plane waves in the build dtype.
    """
    MY, MX = int(op.s_meta.M[0]), int(op.s_meta.M[1])
    Bp = op.s_meta.Bp
    hi = Bp if hi is None else int(hi)
    lo = int(lo)
    nb = hi - lo
    device = op.s_meta.device
    coords = op.s_meta.parent_beams_coords  # (Bp, 2), int on (MY, MX) grid
    idx_y = coords[lo:hi, 0].long() % MY
    idx_x = coords[lo:hi, 1].long() % MX
    # Detector-grid (P = (MY/fy, MX/fx)) amplitude correction sqrt(fy*fx);
    # f==1 (recon) is a no-op (byte-identical).
    fscale = float(op.fy * op.fx) ** 0.5 if (op.fy != 1 or op.fx != 1) else None
    # fp16 build narrows to complex32; fp32/mixed stays complex64.
    out_dtype = op._build_dtype()
    # Assemble the parent waves ONE BEAM-CHUNK at a time: the per-beam IFFT is
    # independent (ifft2(psi)[k] == ifft2(psi[k])), so this is byte-identical
    # to a single (nb, MY, MX) IFFT but never holds psi_fft AND its output at
    # once (~2x27 GB at 20 nm, for callers at the FULL aperture). Sub-range callers (the build's
    # per-chunk loop) pass nb <= chunk -> one iteration -> same as before.
    chunk = max(1, int(6e8 // max(1, int(MY) * int(MX))))
    if nb <= chunk:
        # Single shot (the build's per-chunk calls, recon): no separate output
        # buffer -- ifft IS the result. (The old code path; keeps the elastic
        # build's peak unchanged.)
        psi = torch.zeros((nb, MY, MX), dtype=torch.complex64, device=device)
        psi[torch.arange(nb, device=device), idx_y, idx_x] = 1.0
        waves = torch.fft.ifft2(psi, norm="ortho")
        del psi
        if fscale is not None:
            waves *= fscale
        return waves if out_dtype == torch.complex64 else waves.to(out_dtype)
    # Large aperture (full-Bp call): assemble per beam-chunk so
    # psi_fft and its IFFT never both span the full (Bp, MY, MX) (~2x27 GB).
    waves = torch.empty((nb, MY, MX), dtype=out_dtype, device=device)
    for k in range(0, nb, chunk):
        kj = min(k + chunk, nb)
        psi = torch.zeros((kj - k, MY, MX), dtype=torch.complex64, device=device)
        psi[torch.arange(kj - k, device=device), idx_y[k:kj], idx_x[k:kj]] = 1.0
        w = torch.fft.ifft2(psi, norm="ortho")
        if fscale is not None:
            w = w * fscale
        waves[k:kj] = w.to(out_dtype)
    return waves
