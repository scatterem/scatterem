"""Detector fftshift written by the constructor's clamp + dose pass.

Private, lazily imported by :func:`._init_passes.fourier_shift` for CUDA inputs.

Why this exists
---------------
With ``fourier_shift_dim`` on the last two axes, ``Dataset4DStem.__init__``
first makes a shifted copy (``physics.fourier.fftshift_last2``: one read and one
write of the cube).  Then the clamp + dose pass of :mod:`._clamp_cast_warp` runs
over the copy (one more read and write).  On ``d29_pad_taper/large/fourier_shift``
the copy is 7 ms of a 25 ms row.  The kernel here does both at once: each thread
reads the source element at its UNSHIFTED index, writes the clamped value (or
its root) into the new buffer, and adds it to the row sum.

Numerics
--------
BIT-EXACT to the two-pass spelling.  Output ``(y, x)`` reads
``((y + ny - sy) % ny, (x + nx - sx) % nx)``, the map of ``fftshift_last2``.
Each stored float32 is the value the copy would have carried through the clamp
and ``__fsqrt_rn``; the stored value is the LOADED one, so a NaN keeps its
payload.  The sums add the same float64 values over the SHIFTED row, in the
order of :mod:`._sum_order_warp`, as the two-pass spelling does.

Gating
------
The ``fftshift_clamp`` row of :mod:`scatterem.data._backends`, whose predicate
is :func:`shift_clamp_supported`; the rows ``clamp_cast`` and
``clamp_sum_order`` must also choose ``"warp"``.  Otherwise
:func:`fftshift_last2_clamp_and_row_sums_f64` returns ``None``, and the caller
shifts first and clamps after.
"""

from __future__ import annotations

import torch
import warp as wp

from scatterem.data._backends import CLAMP_CAST, CLAMP_SUM_ORDER, FFTSHIFT_CLAMP
from scatterem.physics import _warp_interop as wi
from scatterem.physics._warp_rn import fsqrt_rn as fsqrt
from scatterem.utils.env import resolve_backend

from . import _clamp_cast_warp
from ._clamp_cast_warp import clamp_cast_supported
from ._sum_order_warp import block_tree_sum, order_sums

__all__ = ["CALLS", "fftshift_last2_clamp_and_row_sums_f64", "shift_clamp_supported"]

#: ``[ran, declined]``.  Host-side counters only, so a probe can prove the fast
#: path was TAKEN rather than infer it from a timing.
CALLS = [0, 0]


@wp.func
def _shift_src(i: int, ny: int, nx: int, sy: int, sx: int) -> int:
    # Source index of element `i` of `fftshift(src, dim=(-2, -1))`.
    plane = ny * nx
    p = i // plane
    r = i - p * plane
    y = r // nx
    xx = r - y * nx
    ys = y + ny - sy
    if ys >= ny:
        ys = ys - ny
    xs = xx + nx - sx
    if xs >= nx:
        xs = xs - nx
    return p * plane + ys * nx + xs


@wp.func
def _shift_clamp_load(
    src: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    x: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    i: int,
    ny: int,
    nx: int,
    sy: int,
    sx: int,
    sqrt: int,
) -> wp.float64:
    # `_clamp_load` on `x = fftshift(src)`, except that `x[i]` is always
    # written: it is a new buffer.
    v = src[_shift_src(i, ny, nx, sy, sx)]
    if v <= wp.float32(0.0):
        x[i] = wp.float32(0.0)
        return wp.float64(0.0)
    if sqrt != 0:
        r = fsqrt(v)
        x[i] = r
        return wp.float64(r)
    x[i] = v
    return wp.float64(v)


@wp.kernel
def _shift_clamp_sum_order_kernel(
    src: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    x: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    ny: int,
    nx: int,
    sy: int,
    sx: int,
    sqrt: int,
    row: int,
    ctas: int,
    part: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
):
    # `_sum_order_warp._clamp_sum_order_kernel` with `_shift_clamp_load`; the
    # thread map and the add order are the same.
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
        a0 = a0 + _shift_clamp_load(src, x, e, ny, nx, sy, sx, sqrt)
        a1 = a1 + _shift_clamp_load(src, x, e + 1, ny, nx, sy, sx, sqrt)
        a2 = a2 + _shift_clamp_load(src, x, e + 2, ny, nx, sy, sx, sqrt)
        a3 = a3 + _shift_clamp_load(src, x, e + 3, ny, nx, sy, sx, sqrt)
        idx += stride
    if b == 0:
        k = nvec * 4 + t
        if k < row:
            a0 = a0 + _shift_clamp_load(src, x, base + k, ny, nx, sy, sx, sqrt)
    s = block_tree_sum(((a0 + a1) + a2) + a3)
    if t == 0:
        part[rb] = s


def shift_clamp_supported(array: torch.Tensor) -> bool:
    """The ``fftshift_clamp`` row's predicate: may the fused pass take ``array``?

    Parameters
    ----------
    array : torch.Tensor
        The unshifted cube; its first axis is the outer slice.

    Returns
    -------
    bool
        Whether the cube has at least two dimensions and the ``clamp_cast``
        pass may take it.
    """
    return array.dim() >= 2 and clamp_cast_supported(array)


def _routed(array: torch.Tensor) -> bool:
    """Whether every row in "Gating" chooses ``"warp"`` for ``array``.

    ``_shift_norm_warp`` asks this too, so its pass runs only where this one may.
    """
    if resolve_backend(FFTSHIFT_CLAMP, None, array) != "warp" or (
        resolve_backend(CLAMP_CAST, None, array) != "warp"
    ):
        return False
    row = array.numel() // int(array.shape[0])
    return resolve_backend(CLAMP_SUM_ORDER, None, row) == "warp"


def fftshift_last2_clamp_and_row_sums_f64(
    array: torch.Tensor, sqrt: bool = False
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """``x = fftshift(array, dim=(-2, -1))``, clamped, and its float64 row sums.

    With ``sqrt=True`` the pass also takes the square root of each clamped
    value.

    Parameters
    ----------
    array : torch.Tensor
        The cube.  It is only read.
    sqrt : bool, optional
        Also take the square root.

    Returns
    -------
    tuple of torch.Tensor or None
        ``(x, sums)``: ``x`` is a new contiguous buffer and ``sums`` the
        ``(array.shape[0],)`` float64 row sums on the device.  ``None``
        without an allocation when a row in "Gating" does not choose
        ``"warp"``; the caller must then shift and clamp itself.
    """
    if not _routed(array):
        CALLS[1] += 1
        return None
    row = array.numel() // int(array.shape[0])
    ny, nx = int(array.shape[-2]), int(array.shape[-1])
    out = torch.empty_like(array)
    # Named locals: a raw ctype does not keep its tensor alive.
    src_flat = array.reshape(-1)
    out_flat = out.reshape(-1)
    f32 = {"dtype": wp.float32, "return_ctype": True}
    inputs = [
        wi.from_torch(src_flat, **f32),
        wi.from_torch(out_flat, **f32),
        ny,
        nx,
        ny // 2,
        nx // 2,
        int(sqrt),
    ]
    sums = order_sums(_shift_clamp_sum_order_kernel, inputs, out, row)
    CALLS[0] += 1
    _clamp_cast_warp.CALLS[0] += 1  # the clamp + dose pass ran, fused
    return out, sums
