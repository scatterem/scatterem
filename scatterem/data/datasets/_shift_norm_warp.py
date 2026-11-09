"""The constructor's normalization written by the fftshift pass.

Private, lazily imported by :func:`._init_passes.shift_and_clamp` for CUDA inputs.

Why this exists
---------------
With ``fourier_shift_dim`` on the last two axes and the default
``normalize=True``, the fused shift pass of :mod:`._shift_clamp_warp` writes the
shifted, clamped (and rooted) cube.  The constructor then reads that cube for
``c = temp_3d.mean(0).max()``, and reads and writes it again for ``/= c``.
:func:`fftshift_last2_clamp_normalize_and_row_sums_f64` finds ``c`` first from
the SOURCE, through the shift map with the clamp and the root in registers
(:mod:`._mean0_warp`, including torch's row halves above 2 GiB).  The shift pass
then stores ``r / c`` and still sums the undivided ``r``.  On
``d29_pad_taper/large/fourier_shift`` this removes the ``/=`` pass and moves
the mean's read from the shifted cube to the source.

Numerics
--------
BIT-EXACT to the shift pass followed by ``mean(0).max()`` and ``/=``.  Each
stored value is one IEEE float32 division of the value of
``_shift_clamp_load`` by the same ``c``, as ATen's ``div_`` by a CUDA scalar
tensor makes it.  A NaN row sum or a NaN ``c`` has a payload that depends on
the add order, so then the pass hands the call back (``None``).

Gating
------
The ``fftshift_normalize`` row of :mod:`scatterem.data._backends`, whose
predicate is :func:`shift_normalize_supported`: a 4-D CUDA cube whose
``mean(0)`` configuration :mod:`._mean0_warp` copies, for every row half.  The
rows of :mod:`._shift_clamp_warp` must also choose ``"warp"``.  Otherwise the
function returns ``None``, and the caller takes the shift pass alone and lets
the constructor normalize.
"""

from __future__ import annotations

import torch
import warp as wp

from scatterem.data._backends import FFTSHIFT_NORMALIZE
from scatterem.physics import _warp_interop as wi
from scatterem.utils.env import resolve_backend

from . import _clamp_cast_warp, _shift_clamp_warp
from ._mean0_warp import mean0_fits, mean0_max
from ._shift_clamp_warp import _shift_src
from ._sum_order_warp import block_tree_sum, fsqrt, raw_order_sums

__all__ = [
    "CALLS",
    "fftshift_last2_clamp_normalize_and_row_sums_f64",
    "shift_normalize_supported",
]

#: ``[ran, declined]``.  Host-side counters only, so a probe can prove the fast
#: path was TAKEN rather than infer it from a timing.
CALLS = [0, 0]


@wp.func
def _shift_mean_elem(
    src: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    i: int,
    ny: int,
    nx: int,
    sy: int,
    sx: int,
    sqrt: int,
) -> wp.float32:
    # Element `i` of the shifted, clamped (and rooted) cube: the value that
    # `_shift_clamp_load` stores.  Nothing is written.
    v = src[_shift_src(i, ny, nx, sy, sx)]
    if v <= wp.float32(0.0):
        return wp.float32(0.0)
    if sqrt != 0:
        return fsqrt(v)
    return v


@wp.func
def _shift_mean_load(
    src: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    e: int,
    ny: int,
    nx: int,
    sy: int,
    sx: int,
    sqrt: int,
) -> wp.vec4:
    return wp.vec4(
        _shift_mean_elem(src, e, ny, nx, sy, sx, sqrt),
        _shift_mean_elem(src, e + 1, ny, nx, sy, sx, sqrt),
        _shift_mean_elem(src, e + 2, ny, nx, sy, sx, sqrt),
        _shift_mean_elem(src, e + 3, ny, nx, sy, sx, sqrt),
    )


@wp.kernel
def _shift_mean0_partial_kernel(
    src: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    ny: int,
    nx: int,
    sy: int,
    sx: int,
    sqrt: int,
    r0: int,
    n: int,
    k: int,
    grid_x: int,
    step: int,
    scratch: wp.array(dtype=wp.vec4f),  # pyright: ignore[reportInvalidTypeForm]
):
    # The thread map of `_mean0_warp` over the rows r0 .. r0 + n - 1 (one row
    # half of torch's split iterator, or all rows).
    tid = wp.tid()
    x = tid % 32  # pyright: ignore[reportOperatorIssue]
    y = (tid // 32) % 4  # pyright: ignore[reportOperatorIssue]
    blk = tid // 128  # pyright: ignore[reportOperatorIssue]
    bx = blk % grid_x
    by = blk // grid_x
    col = (x + bx * 32) * 4
    if col >= k:
        return
    idx = y + by * 4
    a0 = wp.vec4(0.0)
    a1 = wp.vec4(0.0)
    a2 = wp.vec4(0.0)
    a3 = wp.vec4(0.0)
    while idx + 3 * step < n:
        v0 = _shift_mean_load(src, (r0 + idx) * k + col, ny, nx, sy, sx, sqrt)
        v1 = _shift_mean_load(src, (r0 + idx + step) * k + col, ny, nx, sy, sx, sqrt)
        v2 = _shift_mean_load(
            src, (r0 + idx + 2 * step) * k + col, ny, nx, sy, sx, sqrt
        )
        v3 = _shift_mean_load(
            src, (r0 + idx + 3 * step) * k + col, ny, nx, sy, sx, sqrt
        )
        a0 = a0 + v0
        a1 = a1 + v1
        a2 = a2 + v2
        a3 = a3 + v3
        idx += 4 * step
    if idx < n:
        a0 = a0 + _shift_mean_load(src, (r0 + idx) * k + col, ny, nx, sy, sx, sqrt)
        idx += step
    if idx < n:
        a1 = a1 + _shift_mean_load(src, (r0 + idx) * k + col, ny, nx, sy, sx, sqrt)
        idx += step
    if idx < n:
        a2 = a2 + _shift_mean_load(src, (r0 + idx) * k + col, ny, nx, sy, sx, sqrt)
    scratch[(by * 4 + y) * (k / 4) + col / 4] = ((a0 + a1) + a2) + a3


@wp.func
def _shift_norm_load(
    src: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    x: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    c: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    i: int,
    ny: int,
    nx: int,
    sy: int,
    sx: int,
    sqrt: int,
) -> wp.float64:
    # `_shift_clamp_load`, storing `r / c` (the element of `div_`) and returning
    # the undivided `r` for the dose sum.
    v = src[_shift_src(i, ny, nx, sy, sx)]
    if v <= wp.float32(0.0):
        x[i] = wp.float32(0.0) / c[0]
        return wp.float64(0.0)
    if sqrt != 0:
        r = fsqrt(v)
        x[i] = r / c[0]
        return wp.float64(r)
    x[i] = v / c[0]
    return wp.float64(v)


@wp.kernel
def _shift_norm_sum_order_kernel(
    src: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    x: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    c: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    ny: int,
    nx: int,
    sy: int,
    sx: int,
    sqrt: int,
    row: int,
    ctas: int,
    part: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
):
    # `_shift_clamp_warp._shift_clamp_sum_order_kernel` with `_shift_norm_load`;
    # the thread map and the add order are the same.
    tid = wp.tid()
    t = tid % 512  # pyright: ignore[reportOperatorIssue]
    rb = tid // 512  # pyright: ignore[reportOperatorIssue]
    b = rb % ctas
    base = (rb // ctas) * row
    nvec = row // 4
    idx = t + b * 512
    stride = 512 * ctas
    a0 = wp.float64(0.0)
    a1 = wp.float64(0.0)
    a2 = wp.float64(0.0)
    a3 = wp.float64(0.0)
    while idx < nvec:
        e = base + idx * 4
        a0 = a0 + _shift_norm_load(src, x, c, e, ny, nx, sy, sx, sqrt)
        a1 = a1 + _shift_norm_load(src, x, c, e + 1, ny, nx, sy, sx, sqrt)
        a2 = a2 + _shift_norm_load(src, x, c, e + 2, ny, nx, sy, sx, sqrt)
        a3 = a3 + _shift_norm_load(src, x, c, e + 3, ny, nx, sy, sx, sqrt)
        idx += stride
    if b == 0:
        k = nvec * 4 + t
        if k < row:
            a0 = a0 + _shift_norm_load(src, x, c, base + k, ny, nx, sy, sx, sqrt)
    s = block_tree_sum(((a0 + a1) + a2) + a3)
    if t == 0:
        part[rb] = s


def shift_normalize_supported(array: torch.Tensor) -> bool:
    """The ``fftshift_normalize`` row's predicate: may the shift pass also divide?

    Parameters
    ----------
    array : torch.Tensor
        The unshifted cube ``(scan y, scan x, detector y, detector x)``.

    Returns
    -------
    bool
        Whether ``array`` is a 4-D CUDA cube and :mod:`._mean0_warp` copies
        the ``mean(0)`` configuration of every row half.
    """
    if not (array.is_cuda and array.dim() == 4 and array.numel() > 0):
        return False
    k = int(array.shape[-2]) * int(array.shape[-1])
    return mean0_fits(array.numel() // k, k, array.device)


def fftshift_last2_clamp_normalize_and_row_sums_f64(
    array: torch.Tensor, sqrt: bool = False
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """The fused fftshift pass plus the constructor's normalization.

    Same bits as ``_shift_clamp_warp.fftshift_last2_clamp_and_row_sums_f64``
    followed by ``mean(0).max()`` and ``/=``.

    Parameters
    ----------
    array : torch.Tensor
        The 4-D cube.  It is only read.
    sqrt : bool, optional
        Also take the square root.

    Returns
    -------
    tuple of torch.Tensor or None
        ``(x, sums)``: ``x`` is the shifted, clamped (and rooted) copy divided
        by ``c = x.reshape(N, K).mean(0).max()`` (``N`` scan positions, ``K``
        detector pixels), the value ``Dataset4DStem.__init__`` divides by with
        ``normalize=True``; ``sums`` are the float64 row sums of the UNDIVIDED
        values, which the constructor sums before it normalizes.  ``None`` when
        a row in "Gating" does not choose ``"warp"``, or when a row sum or
        ``c`` is NaN; the caller then takes the shift pass alone.
    """
    if resolve_backend(
        FFTSHIFT_NORMALIZE, None, array
    ) != "warp" or not _shift_clamp_warp._routed(array):
        CALLS[1] += 1
        return None
    ny, nx = int(array.shape[-2]), int(array.shape[-1])
    k = ny * nx
    f32 = {"dtype": wp.float32, "return_ctype": True}
    # Named locals: a raw ctype does not keep its tensor alive.
    src_flat = array.reshape(-1)
    src_ct = wi.from_torch(src_flat, **f32)
    shift = [ny, nx, ny // 2, nx // 2, int(sqrt)]
    c = mean0_max(
        _shift_mean0_partial_kernel,
        [src_ct, *shift],
        array.numel() // k,
        k,
        array.device,
    )
    out = torch.empty_like(array)
    out_flat = out.reshape(-1)
    inputs = [src_ct, wi.from_torch(out_flat, **f32), wi.from_torch(c, **f32), *shift]
    n_rows = int(array.shape[0])
    row = array.numel() // n_rows
    sums = raw_order_sums(
        _shift_norm_sum_order_kernel, inputs, n_rows, row, array.device
    )
    # A NaN payload depends on the add order, and the undivided values are gone:
    # hand the whole call back.  One host read; the caller reads the sums next.
    if bool(torch.isnan(sums).any() | torch.isnan(c).any()):
        CALLS[1] += 1
        return None
    CALLS[0] += 1
    _shift_clamp_warp.CALLS[0] += 1  # the shift pass ran, fused
    _clamp_cast_warp.CALLS[0] += 1  # the clamp + dose pass ran, fused
    return out, sums
