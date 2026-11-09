"""Clamp and per-row float64 copy of ``Dataset4DStem.__init__`` in one pass.

Private, lazily imported by :func:`._init_passes.clamp_sqrt_total` for CUDA
inputs.

Why this exists
---------------
The constructor clamps the cube in place (``clamp_min_(0)``, one read and one
write of the whole cube) and then, for the dose sum, copies each scan row to
float64 (``row.to(torch.float64).sum()``), which reads every value again.  On
``d29_pad_taper/large/realspace`` the cube is 3.3 GB and the clamp pass is
~10 ms of a ~67 ms row.  This kernel reads each element once,
writes it back only where the clamp changes it, and writes the float64 copy of
the clamped value.

Numerics
--------
BIT-EXACT to the torch spelling:

* The clamp matches CUDA ``clamp_min_(0)`` element by element: NaN is kept with
  its payload, negatives, ``-inf`` and ``-0.0`` become ``+0.0``.  The test is
  ``v <= 0``, which is False for NaN; a ``+0.0`` is rewritten with ``+0.0``.
* The float64 copy is the exact widening ``row.to(torch.float64)`` makes.  The
  NaN branch converts the LOADED value: a reassigned local was compiled to a
  select that returned the canonical NaN ``0x7fffffff``, not the payload.
* On the slot path the sums are NOT reimplemented: torch's own ``.sum()`` runs
  on each row.  Each row sits in its own slot of a scratch buffer, and every
  slot starts on a 512-byte boundary, the alignment of a fresh caching-allocator
  block.  With unaligned slots torch's reduction took another vectorised path and
  two of 51 probe cases differed in the last bits.
* For a row of at least 2048 elements the default path writes no float64 copy:
  :mod:`._sum_order_warp` adds the widened values in the order of torch's own
  ``.sum()``, so the bits are the same (row ``clamp_sum_order``).

Amplitudes
----------
With ``sqrt=True`` (``transform_to_amplitudes``) the same pass also takes the
square root: ``x = sqrt(clamp(x))`` is written back and its float64 copy feeds
the sums.  The torch spelling is ``clamp_min_(0)``, then ``sqrt_()``, two more
full passes.  The root is ``__fsqrt_rn``, IEEE round-to-nearest like the ATen
``sqrt`` kernel, so the value is the same; ``v <= 0`` writes ``+0.0``, which is
``sqrt(+0.0)``.  On ``d29_pad_taper/large/fourier_shift`` this
removes the clamp and the ``sqrt_`` passes.  Without the clamp torch's
``sqrt`` of a negative is NaN, so the caller asks for the root only together
with the clamp.

Gating
------
The ``clamp_cast`` row of :mod:`scatterem.data._backends`, whose predicate is
:func:`clamp_cast_supported`: ``array`` is a non-empty contiguous CUDA float32
tensor with at least one dimension, no autograd, and fewer than ``2**31``
elements (the kernel indexes with 32-bit integers).  Otherwise
:func:`clamp_and_row_sums_f64` returns ``None`` and the caller keeps the torch
ops.  ``SCATTEREM_USE_WARP=0`` selects the torch ops.  When the row
``clamp_sum_order`` chooses ``"torch"``, this pass still runs but sums through
the float64 slots.
"""

from __future__ import annotations

import torch
import warp as wp

from scatterem.data._backends import CLAMP_CAST, CLAMP_SUM_ORDER
from scatterem.physics import _warp_interop as wi
from scatterem.physics._warp_rn import fsqrt_rn as fsqrt
from scatterem.utils.env import resolve_backend

from ._sum_order_warp import row_sums_torch_order

__all__ = [
    "CALLS",
    "clamp_and_row_sums_f64",
    "clamp_cast_supported",
]

#: ``[ran, declined]``.  Host-side counters only, so a probe can prove the fast path
#: was TAKEN rather than infer it from a timing.
CALLS = [0, 0]

#: Float64 row slots per launch.  The torch spelling holds one float64 row at a
#: time; two slots hold one more, whatever the cube size.  A byte budget is the
#: wrong shape here: on a small cube it covers every row, and the scratch becomes
#: twice the float32 cube.  On a large benchmark cube we measured 1.339x on
#: the clamp + dose stage with two rows per launch, 1.364x with eleven, and 1.293x
#: with one.
_ROWS_PER_LAUNCH = 2


@wp.kernel
def _clamp_cast_kernel(
    x: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    base: int,
    row: int,
    out: wp.array2d(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
):
    j, e = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    i = base + j * row + e
    v = x[i]
    if v <= wp.float32(0.0):
        x[i] = wp.float32(0.0)
        out[j, e] = wp.float64(0.0)
    else:
        out[j, e] = wp.float64(v)


@wp.kernel
def _clamp_sqrt_cast_kernel(
    x: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    base: int,
    row: int,
    out: wp.array2d(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
):
    j, e = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    i = base + j * row + e
    v = x[i]
    if v <= wp.float32(0.0):
        x[i] = wp.float32(0.0)
        out[j, e] = wp.float64(0.0)
    else:
        r = fsqrt(v)
        x[i] = r
        out[j, e] = wp.float64(r)


def clamp_cast_supported(array: torch.Tensor) -> bool:
    """The ``clamp_cast`` row's predicate: may the Warp pass take ``array``?

    Parameters
    ----------
    array : torch.Tensor
        The cube; its first axis is the outer slice.

    Returns
    -------
    bool
        Whether the Warp pass may run.
    """
    return (
        not torch.compiler.is_compiling()
        and array.is_cuda
        and (array.device.index or 0) < wp.get_cuda_device_count()
        and array.dtype is torch.float32
        and array.dim() >= 1
        and array.is_contiguous()
        and 0 < array.numel() < 2**31
        and not (torch.is_grad_enabled() and array.requires_grad)
        and not torch.cuda.is_current_stream_capturing()
    )


def clamp_and_row_sums_f64(
    array: torch.Tensor, sqrt: bool = False
) -> torch.Tensor | None:
    """``array.clamp_min_(0)``, then ``stack([array[i].to(float64).sum() ...])``.

    With ``sqrt=True``, ``array.sqrt_()`` runs between the two (see
    "Amplitudes" in the module docstring).

    Parameters
    ----------
    array : torch.Tensor
        The cube, clamped IN PLACE when the Warp pass runs.
    sqrt : bool, optional
        Also take the square root in place.

    Returns
    -------
    torch.Tensor or None
        The ``(array.shape[0],)`` float64 per-row sums, still on the device; or
        ``None`` without touching ``array`` when the row ``clamp_cast`` does
        not choose ``"warp"``.  The caller must then run the torch ops itself.
    """
    if resolve_backend(CLAMP_CAST, None, array) != "warp":
        CALLS[1] += 1
        return None
    n_rows = int(array.shape[0])
    row = array.numel() // n_rows
    if resolve_backend(CLAMP_SUM_ORDER, None, row) == "warp":
        CALLS[0] += 1
        return row_sums_torch_order(array, sqrt)
    slot = -(-row // 64) * 64  # 64 float64 = 512 bytes
    k = min(n_rows, _ROWS_PER_LAUNCH)
    buf = torch.empty((k, slot), dtype=torch.float64, device=array.device)
    rshape = tuple(array.shape[1:])
    # Named locals: a raw ctype does not keep its tensor alive.
    flat = array.reshape(-1)
    x_ct = wi.from_torch(flat, dtype=wp.float32, return_ctype=True)
    buf_ct = wi.from_torch(buf, dtype=wp.float64, return_ctype=True)
    device = wi.device_from_torch(array.device)
    # Warp's default stream is not PyTorch's current stream; launch on the torch
    # stream so the kernel is ordered against the ops around it.
    stream = wi.stream_from_torch(torch.cuda.current_stream(array.device))
    sums = []
    for r0 in range(0, n_rows, k):
        m = min(k, n_rows - r0)
        wp.launch(
            _clamp_sqrt_cast_kernel if sqrt else _clamp_cast_kernel,
            dim=(m, row),
            inputs=[x_ct, r0 * row, row],
            outputs=[buf_ct],
            device=device,
            stream=stream,
        )
        for j in range(m):
            sums.append(buf[j, :row].view(rshape).sum())
    CALLS[0] += 1
    return torch.stack(sums)
