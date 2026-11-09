"""Crop copy written by the constructor's clamp + dose pass.

Private, lazily imported by :func:`._init_passes.crop_array` for CUDA inputs.

Why this exists
---------------
``Dataset4DStem.crop(clone=True)`` first clones the strided slice of the parent
(one read of the slice, one write of the crop).  The constructor then runs the
clamp + dose pass of :mod:`._clamp_cast_warp` over the clone (one more read).
:func:`crop_clamp_and_row_sums_f64` does both at once: each thread reads the
parent at the strided index of its element, clamps it, writes it into the new
contiguous buffer and adds it to the row sum.  On
``d29_pad_taper/large/crop`` this removes the clone's write and read of the
0.6 GB crop.

With the default ``normalize=True`` the constructor then reads the crop once
more for ``c = temp_3d.mean(0).max()``, and reads and writes it again for
``/= c``.  :func:`crop_clamp_normalize_and_row_sums_f64` finds ``c`` first from
the parent (:mod:`._mean0_warp`, with the clamp in registers), and the crop pass
then stores ``x / c`` and still sums the undivided ``x``.  The ``/=``
pass and the mean's read of the crop go.

Numerics
--------
BIT-EXACT to the torch spelling.  The crop holds the same values in the same
order as ``parent[index].clone()``.  The clamp is that of
``_clamp_cast_kernel``, and the stored value is the LOADED one, so a NaN keeps
its payload.  The sums add the same float64 values in the order of
:mod:`._sum_order_warp`.  The divided value is one IEEE float32 division by the
same ``c``, as ATen's ``div_`` by a CUDA scalar tensor makes it.  A NaN row sum
or a NaN ``c`` has a payload that depends on the add order, so then the
normalizing pass hands the call back (``None``); the undivided values are gone,
so it cannot re-sum them.

Gating
------
The ``crop_clamp`` row of :mod:`scatterem.data._backends`, whose predicate is
:func:`crop_clamp_supported`: a non-empty 4-D view of the parent's CUDA storage,
with non-negative strides, inside the parent.  The rows ``clamp_cast`` (on the
parent) and ``clamp_sum_order`` must also choose ``"warp"``.  The
normalizing pass also needs the ``crop_normalize`` row
(:func:`crop_normalize_supported`: a ``mean(0)`` configuration that
:mod:`._mean0_warp` copies, under 2 GiB).  Otherwise the functions return
``None`` and the caller clones (and the constructor normalizes).
"""

from __future__ import annotations

import torch
import warp as wp

from scatterem.data._backends import (
    CLAMP_CAST,
    CLAMP_SUM_ORDER,
    CROP_CLAMP,
    CROP_NORMALIZE,
)
from scatterem.physics import _warp_interop as wi
from scatterem.utils.env import resolve_backend

from . import _clamp_cast_warp
from ._mean0_warp import mean0_config, mean0_max
from ._sum_order_warp import block_tree_sum, order_sums, raw_order_sums

__all__ = [
    "CALLS",
    "crop_clamp_and_row_sums_f64",
    "crop_clamp_normalize_and_row_sums_f64",
    "crop_clamp_supported",
    "crop_normalize_supported",
]

#: ``[ran, declined, ran with the normalization, normalization declined]``.
#: Host-side counters only, so a probe can prove the fast path was TAKEN rather
#: than infer it from a timing.
CALLS = [0, 0, 0, 0]


@wp.func
def _crop_clamp_load(
    src: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    x: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    i: int,
    off: int,
    s1: int,
    s2: int,
    s3: int,
    st0: int,
    st1: int,
    st2: int,
    st3: int,
) -> wp.float64:
    # Element `i` of the contiguous crop reads the parent at the view's strided
    # index, then `_clamp_load`.  `x[i]` is always written: it is a new buffer.
    c3 = i % s3
    t = i / s3
    c2 = t % s2
    t = t / s2
    c1 = t % s1
    c0 = t / s1
    v = src[off + c0 * st0 + c1 * st1 + c2 * st2 + c3 * st3]
    if v <= wp.float32(0.0):
        x[i] = wp.float32(0.0)
        return wp.float64(0.0)
    x[i] = v
    return wp.float64(v)


@wp.kernel
def _crop_clamp_sum_order_kernel(
    src: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    x: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    off: int,
    s1: int,
    s2: int,
    s3: int,
    st0: int,
    st1: int,
    st2: int,
    st3: int,
    row: int,
    ctas: int,
    part: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
):
    # `_sum_order_warp._clamp_sum_order_kernel` with `_crop_clamp_load`; the
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
        a0 = a0 + _crop_clamp_load(src, x, e, off, s1, s2, s3, st0, st1, st2, st3)
        a1 = a1 + _crop_clamp_load(src, x, e + 1, off, s1, s2, s3, st0, st1, st2, st3)
        a2 = a2 + _crop_clamp_load(src, x, e + 2, off, s1, s2, s3, st0, st1, st2, st3)
        a3 = a3 + _crop_clamp_load(src, x, e + 3, off, s1, s2, s3, st0, st1, st2, st3)
        idx += stride
    if b == 0:
        k = nvec * 4 + t
        if k < row:
            a0 = a0 + _crop_clamp_load(
                src, x, base + k, off, s1, s2, s3, st0, st1, st2, st3
            )
    s = block_tree_sum(((a0 + a1) + a2) + a3)
    if t == 0:
        part[rb] = s


@wp.func
def _crop_mean_load(
    src: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    r: int,
    j: int,
    off: int,
    s1: int,
    s3: int,
    st0: int,
    st1: int,
    st2: int,
    st3: int,
) -> wp.vec4:
    # Clamped crop pixels j .. j+3 of scan position r (`r = c0 * s1 + c1`); the
    # clamp of `_crop_clamp_load`, nothing written.
    p = off + (r / s1) * st0 + (r % s1) * st1
    v = wp.vec4(0.0)
    for q in range(4):
        e = src[p + ((j + q) / s3) * st2 + ((j + q) % s3) * st3]
        if e > wp.float32(0.0) or e != e:
            v[q] = e
    return v


@wp.kernel
def _crop_mean0_partial_kernel(
    src: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    off: int,
    s1: int,
    s3: int,
    st0: int,
    st1: int,
    st2: int,
    st3: int,
    r0: int,
    n: int,
    k: int,
    grid_x: int,
    step: int,
    scratch: wp.array(dtype=wp.vec4f),  # pyright: ignore[reportInvalidTypeForm]
):
    # One thread per torch thread (x, y) of block (bx, by), block (32, 4), over
    # the rows r0 .. r0 + n - 1 (see `_mean0_warp`).  Stores the thread's value
    # after `((a0 + a1) + a2) + a3` at scratch[by, y, col / 4].
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
        r = r0 + idx
        v0 = _crop_mean_load(src, r, col, off, s1, s3, st0, st1, st2, st3)
        v1 = _crop_mean_load(src, r + step, col, off, s1, s3, st0, st1, st2, st3)
        v2 = _crop_mean_load(src, r + 2 * step, col, off, s1, s3, st0, st1, st2, st3)
        v3 = _crop_mean_load(src, r + 3 * step, col, off, s1, s3, st0, st1, st2, st3)
        a0 = a0 + v0
        a1 = a1 + v1
        a2 = a2 + v2
        a3 = a3 + v3
        idx += 4 * step
    if idx < n:
        a0 = a0 + _crop_mean_load(src, r0 + idx, col, off, s1, s3, st0, st1, st2, st3)
        idx += step
    if idx < n:
        a1 = a1 + _crop_mean_load(src, r0 + idx, col, off, s1, s3, st0, st1, st2, st3)
        idx += step
    if idx < n:
        a2 = a2 + _crop_mean_load(src, r0 + idx, col, off, s1, s3, st0, st1, st2, st3)
    scratch[(by * 4 + y) * (k / 4) + col / 4] = ((a0 + a1) + a2) + a3


@wp.func
def _crop_norm_load(
    src: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    x: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    c: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    i: int,
    off: int,
    s1: int,
    s2: int,
    s3: int,
    st0: int,
    st1: int,
    st2: int,
    st3: int,
) -> wp.float64:
    # `_crop_clamp_load`, storing `v / c` (the element of `div_`) and returning
    # the undivided `v` for the dose sum.
    c3 = i % s3
    t = i / s3
    c2 = t % s2
    t = t / s2
    c1 = t % s1
    c0 = t / s1
    v = src[off + c0 * st0 + c1 * st1 + c2 * st2 + c3 * st3]
    if v <= wp.float32(0.0):
        x[i] = wp.float32(0.0) / c[0]
        return wp.float64(0.0)
    x[i] = v / c[0]
    return wp.float64(v)


@wp.kernel
def _crop_norm_sum_order_kernel(
    src: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    x: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    c: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    off: int,
    s1: int,
    s2: int,
    s3: int,
    st0: int,
    st1: int,
    st2: int,
    st3: int,
    row: int,
    ctas: int,
    part: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
):
    # `_crop_clamp_sum_order_kernel` with `_crop_norm_load`; same thread map.
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
        a0 = a0 + _crop_norm_load(src, x, c, e, off, s1, s2, s3, st0, st1, st2, st3)
        a1 = a1 + _crop_norm_load(src, x, c, e + 1, off, s1, s2, s3, st0, st1, st2, st3)
        a2 = a2 + _crop_norm_load(src, x, c, e + 2, off, s1, s2, s3, st0, st1, st2, st3)
        a3 = a3 + _crop_norm_load(src, x, c, e + 3, off, s1, s2, s3, st0, st1, st2, st3)
        idx += stride
    if b == 0:
        k = nvec * 4 + t
        if k < row:
            a0 = a0 + _crop_norm_load(
                src, x, c, base + k, off, s1, s2, s3, st0, st1, st2, st3
            )
    s = block_tree_sum(((a0 + a1) + a2) + a3)
    if t == 0:
        part[rb] = s


def _offset(parent: torch.Tensor, view: torch.Tensor) -> int:
    """The view's first element, counted from the parent's first element."""
    return int(view.storage_offset()) - int(parent.storage_offset())


def crop_clamp_supported(parent: torch.Tensor, view: torch.Tensor) -> bool:
    """The ``crop_clamp`` row's predicate: may the fused crop pass take ``view``?

    Parameters
    ----------
    parent : torch.Tensor
        The dataset's cube.  The ``clamp_cast`` row checks it.
    view : torch.Tensor
        ``parent[index]`` with basic slices.

    Returns
    -------
    bool
        Whether ``view`` is a non-empty 4-D view of ``parent``'s CUDA storage,
        with non-negative strides, whose last element lies inside ``parent``.
    """
    ok = (
        parent.is_cuda
        and view.is_cuda
        and view.dim() == 4
        and view.numel() > 0
        and view.dtype == parent.dtype
        and view.device == parent.device
        and view.untyped_storage().data_ptr() == parent.untyped_storage().data_ptr()
        and _offset(parent, view) >= 0
        and all(s >= 0 for s in view.stride())
    )
    if not ok:
        return False
    last = _offset(parent, view) + sum(
        (n - 1) * s for n, s in zip(view.shape, view.stride(), strict=True)
    )
    return last < parent.numel()


def crop_normalize_supported(parent: torch.Tensor, view: torch.Tensor) -> bool:
    """The ``crop_normalize`` row's predicate: may the crop pass also divide?

    Parameters
    ----------
    parent : torch.Tensor
        The dataset's cube.
    view : torch.Tensor
        ``parent[index]`` with basic slices.

    Returns
    -------
    bool
        Whether the ``crop_clamp`` predicate holds and :mod:`._mean0_warp`
        copies the ``mean(0)`` configuration of the crop, as one part (the lab
        did not copy torch's row halves for the crop).
    """
    if not crop_clamp_supported(parent, view):
        return False
    n_pos, k = view.shape[0] * view.shape[1], view.shape[2] * view.shape[3]
    return mean0_config(n_pos, k, view.device) is not None


def _routed(norm: bool, parent: torch.Tensor, view: torch.Tensor) -> bool:
    """Whether every row in "Gating" chooses ``"warp"`` for these operands."""
    row = view.numel() // int(view.shape[0]) if view.dim() == 4 else 0
    return (
        (not norm or resolve_backend(CROP_NORMALIZE, None, parent, view) == "warp")
        and resolve_backend(CROP_CLAMP, None, parent, view) == "warp"
        and resolve_backend(CLAMP_CAST, None, parent) == "warp"
        and resolve_backend(CLAMP_SUM_ORDER, None, row) == "warp"
    )


def _crop_inputs(parent: torch.Tensor, view: torch.Tensor) -> list:
    """The offset, the crop's shape (dims 1-3) and the view's strides."""
    s1, s2, s3 = (int(s) for s in view.shape[1:])
    return [_offset(parent, view), s1, s2, s3, *(int(s) for s in view.stride())]


def crop_clamp_and_row_sums_f64(
    parent: torch.Tensor, view: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """``x = view.clone()``, clamped, and its float64 row sums, in one pass.

    Parameters
    ----------
    parent : torch.Tensor
        The contiguous cube.  It is only read.
    view : torch.Tensor
        ``parent[index]`` with basic slices.

    Returns
    -------
    tuple of torch.Tensor or None
        ``(x, sums)``: ``x`` is a new contiguous, clamped buffer and ``sums``
        the ``(x.shape[0],)`` float64 row sums on the device.  ``None`` without
        an allocation when a row in "Gating" does not choose ``"warp"``; the
        caller must then clone and clamp itself.
    """
    if not _routed(False, parent, view):
        CALLS[1] += 1
        return None
    out = torch.empty(tuple(view.shape), dtype=torch.float32, device=view.device)
    # Named locals: a raw ctype does not keep its tensor alive.
    src_flat = parent.reshape(-1)
    out_flat = out.reshape(-1)
    f32 = {"dtype": wp.float32, "return_ctype": True}
    inputs = [
        wi.from_torch(src_flat, **f32),
        wi.from_torch(out_flat, **f32),
        *_crop_inputs(parent, view),
    ]
    row = view.numel() // int(view.shape[0])
    sums = order_sums(_crop_clamp_sum_order_kernel, inputs, out, row)
    CALLS[0] += 1
    _clamp_cast_warp.CALLS[0] += 1  # the clamp + dose pass ran, fused
    return out, sums


def crop_clamp_normalize_and_row_sums_f64(
    parent: torch.Tensor, view: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """:func:`crop_clamp_and_row_sums_f64` plus the constructor's normalization.

    Same bits as clone + clamp + row sums + ``mean(0).max()`` + ``/=``.

    Parameters
    ----------
    parent : torch.Tensor
        The contiguous cube.  It is only read.
    view : torch.Tensor
        ``parent[index]`` with basic slices.

    Returns
    -------
    tuple of torch.Tensor or None
        ``(x, sums)``: ``x`` is the clamped crop divided by
        ``c = x_clamped.reshape(N, K).mean(0).max()`` (``N`` scan positions,
        ``K`` detector pixels), the value ``Dataset4DStem.__init__`` divides by
        with ``normalize=True``; ``sums`` are the float64 row sums of the
        UNDIVIDED clamped crop, which the constructor sums before it
        normalizes.  ``None`` when a row in "Gating" does not choose
        ``"warp"``, or when a row sum or ``c`` is NaN; the caller then takes
        :func:`crop_clamp_and_row_sums_f64` and lets the constructor normalize.
    """
    if not _routed(True, parent, view):
        CALLS[3] += 1
        return None
    f32 = {"dtype": wp.float32, "return_ctype": True}
    # Named locals: a raw ctype does not keep its tensor alive.
    src_flat = parent.reshape(-1)
    src_ct = wi.from_torch(src_flat, **f32)
    geo = _crop_inputs(parent, view)
    s1 = geo[1]
    n_pos, k = int(view.shape[0]) * s1, geo[2] * geo[3]
    # The partial kernel takes the offset, s1, s3 and the four strides.
    mean_geo = [geo[0], s1, geo[3], *geo[4:]]
    c = mean0_max(
        _crop_mean0_partial_kernel, [src_ct, *mean_geo], n_pos, k, view.device
    )
    out = torch.empty(tuple(view.shape), dtype=torch.float32, device=view.device)
    out_flat = out.reshape(-1)
    inputs = [
        src_ct,
        wi.from_torch(out_flat, **f32),
        wi.from_torch(c, **f32),
        *geo,
    ]
    n_rows = int(view.shape[0])
    row = view.numel() // n_rows
    sums = raw_order_sums(_crop_norm_sum_order_kernel, inputs, n_rows, row, view.device)
    # A NaN payload depends on the add order, and the undivided rows are gone:
    # hand the whole call back.  One host read; the caller reads the sums next.
    if bool(torch.isnan(sums).any() | torch.isnan(c).any()):
        CALLS[3] += 1
        return None
    CALLS[2] += 1
    _clamp_cast_warp.CALLS[0] += 1  # the clamp + dose pass ran, fused
    return out, sums
