"""One-kernel refinement tail of :func:`scatterem.utils.subpixel._peak_offsets`.

Private, lazily imported by ``_peak_offsets`` for CUDA inputs.

Why this exists
---------------
After the separable DFT evaluation, ``_peak_offsets`` finds the best sample of the
``(nbatch, n_side, n_side)`` patch and fits a parabola through it along rows and
along columns.  The torch spelling of that tail is ~65 small launches per call
(``argmax``, ``div``/``%``, ``arange``, two gathers of the offsets and of the
patch, two 25-op ``_parabolic_vertices`` calls, the step multiply and add, two
copies), on arrays of a few hundred elements.  On ``d31_bf_selfcal/medium/selfcal``
the row is host-launch bound and calls ``_peak_offsets`` 55 times, so the tail is
~20 ms of a ~168 ms row (replaying its answer instead reads
1.1377x).  This kernel does the tail in one launch, one thread per image.

Numerics
--------
BIT-EXACT to the torch spelling, by construction and checked with ``torch.equal``
on the real call and on planted ties, NaNs, flat and boundary peaks:

- the argmax scans the patch in flat (row-major) order and keeps the FIRST index
  of the maximum, and the first NaN over any number, which is ATen's ``argmax``
  rule (``GreaterOrNan`` with the lower index on a tie);
- the parabola is the same IEEE operations in the same order as
  ``_parabolic_vertices`` (``(left - 2*middle) + right``, ``(0.5*(left - right)) /
  curvature``, clamp that lets NaN through, zero off the interior or on a flat
  curvature), and ``offset + step*vertex`` rounds the product before the sum.
  Products, sums, differences and the quotient use the round-to-nearest
  intrinsics, so nvrtc cannot contract a multiply into a following add as an FMA
  (one rounding where ATen has two).
- the offsets are read from the caller's ``row_offsets`` / ``col_offsets``, the
  same tensors the torch spelling gathers from.

Gating
------
The ``subpixel_tail`` row of :mod:`scatterem.utils._backends`, whose predicate is
:func:`refine_tail_supported`: the patch is a non-empty, square-sided
``(nbatch, n, n)`` CUDA float64 tensor with no autograd, and both offset tables
are ``(nbatch, n)`` float64 tensors on the same device with no autograd.
Otherwise :func:`refine_tail` returns ``None`` and the caller keeps the torch
ops.  The row also needs :func:`~scatterem.physics._warp_interop.warp_sees_cuda`, and ``SCATTEREM_USE_WARP=0``
selects the torch ops.
"""

from __future__ import annotations

import torch
import warp as wp

from scatterem.physics import _warp_interop as wi
from scatterem.physics._warp_rn import dadd_rn as _dadd
from scatterem.physics._warp_rn import ddiv_rn as _ddiv
from scatterem.physics._warp_rn import dmul_rn as _dmul
from scatterem.physics._warp_rn import dsub_rn as _dsub
from scatterem.utils._backends import SUBPIXEL_TAIL
from scatterem.utils.env import resolve_backend
from scatterem.utils.warp.launch import ensure_warp_initialized

__all__ = ["CALLS", "refine_tail", "refine_tail_supported"]

#: ``[ran, declined]``.  Host-side counters only, so a probe can prove the fast path
#: was TAKEN rather than infer it from a timing.
CALLS = [0, 0]


@wp.func
def _vertex(left: wp.float64, middle: wp.float64, right: wp.float64) -> wp.float64:
    """``_parabolic_vertices`` for one interior sample, in steps."""
    curvature = _dadd(_dsub(left, _dmul(wp.float64(2.0), middle)), right)
    if curvature == wp.float64(0.0):
        return wp.float64(0.0)
    v = _ddiv(_dmul(_dsub(left, right), wp.float64(0.5)), curvature)
    # torch's clamp returns NaN unchanged, and keeps the sign of a zero.
    if v < wp.float64(-0.5):
        v = wp.float64(-0.5)
    if v > wp.float64(0.5):
        v = wp.float64(0.5)
    return v


@wp.kernel
def _refine_tail_kernel(
    patch: wp.array3d(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
    row_off: wp.array2d(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
    col_off: wp.array2d(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
    step: wp.float64,
    out: wp.array2d(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
):
    b = wp.tid()
    n = patch.shape[1]
    # argmax over the flattened patch: first NaN, else first maximum.
    best_r = int(0)  # noqa: UP018 -- declares a mutable Warp local; a bare literal is a constant
    best_c = int(0)  # noqa: UP018 -- declares a mutable Warp local; a bare literal is a constant
    best = patch[b, 0, 0]
    for r in range(n):
        for c in range(n):
            v = patch[b, r, c]
            if not wp.isnan(best):
                if wp.isnan(v) or v > best:
                    best = v
                    best_r = r
                    best_c = c
    v_row = wp.float64(0.0)
    if best_r > 0 and best_r < n - 1:
        v_row = _vertex(
            patch[b, best_r - 1, best_c],
            patch[b, best_r, best_c],
            patch[b, best_r + 1, best_c],
        )
    v_col = wp.float64(0.0)
    if best_c > 0 and best_c < n - 1:
        v_col = _vertex(
            patch[b, best_r, best_c - 1],
            patch[b, best_r, best_c],
            patch[b, best_r, best_c + 1],
        )
    out[b, 0] = _dadd(row_off[b, best_r], _dmul(step, v_row))
    out[b, 1] = _dadd(col_off[b, best_c], _dmul(step, v_col))


def _wrap(t: torch.Tensor):
    """``wi.from_torch`` on a contiguous float64 view, as a raw ctype."""
    return wi.from_torch(t.contiguous(), dtype=wp.float64, return_ctype=True)


def _no_grad(t: torch.Tensor) -> bool:
    return not (torch.is_grad_enabled() and t.requires_grad)


def refine_tail_supported(
    patch: torch.Tensor, row_offsets: torch.Tensor, col_offsets: torch.Tensor
) -> bool:
    """True if :func:`refine_tail` may replace the torch refinement tail."""
    if not (
        patch.is_cuda
        and patch.dtype == torch.float64
        and patch.dim() == 3
        and patch.shape[0] > 0
        and patch.shape[1] > 0
        and patch.shape[1] == patch.shape[2]
        and _no_grad(patch)
    ):
        return False
    return all(
        t.device == patch.device
        and t.dtype == torch.float64
        and t.shape == patch.shape[:2]
        and _no_grad(t)
        for t in (row_offsets, col_offsets)
    )


def refine_tail(
    patch: torch.Tensor,
    row_offsets: torch.Tensor,
    col_offsets: torch.Tensor,
    step: float,
) -> torch.Tensor | None:
    """``refined`` of ``_peak_offsets`` from its ``patch`` and offset tables.

    Returns a ``(nbatch, 2)`` float64 tensor, or ``None`` without launching
    anything when the backend row declines; the caller must then run the torch
    ops itself.
    """
    if resolve_backend(SUBPIXEL_TAIL, None, patch, row_offsets, col_offsets) != "warp":
        CALLS[1] += 1
        return None
    ensure_warp_initialized()
    out = torch.empty((patch.shape[0], 2), dtype=torch.float64, device=patch.device)
    wp.launch(
        _refine_tail_kernel,
        dim=patch.shape[0],
        inputs=[_wrap(patch), _wrap(row_offsets), _wrap(col_offsets), float(step)],
        outputs=[_wrap(out)],
        device=wi.device_from_torch(patch.device),
        # Warp's default stream is not PyTorch's current stream; launching on
        # Warp's own would leave the kernel unordered against the ATen ops that
        # produce ``patch`` and consume ``out`` (as in _overlap_counts_warp.py).
        stream=wi.stream_from_torch(torch.cuda.current_stream(patch.device)),
    )
    CALLS[0] += 1
    return out
