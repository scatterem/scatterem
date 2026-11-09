"""The constructor's ``mean(0).max()`` in torch 2.11's order.

Private.  :mod:`._crop_clamp_warp` and :mod:`._shift_norm_warp`
use it to find the normalization constant of
``Dataset4DStem.__init__`` BEFORE the cube exists, so their fused pass can store
``x / c`` and the constructor's ``/=`` pass goes.

Why this exists
---------------
With the default ``normalize=True`` the constructor reads the cube for
``c = temp_3d.mean(0).max()`` and then reads and writes it again for
``self._array /= c``.  On the benchmark rows these two legs are about 43 % of the
row.  The fused producer passes (crop, fftshift, pad) can divide while they
write, if ``c`` is known first.  This module computes ``c`` from the producer's
SOURCE: each caller gives a partial kernel that reads the source through its
element map (with the clamp, and the root, in registers).

Numerics
--------
BIT-EXACT to ``x.reshape(N, K).mean(0)`` of a fresh contiguous float32 ``x``
(``N`` scan positions, ``K`` detector pixels), for the configuration copied
here.  The order is that of torch 2.11's ``reduce_kernel`` for an outer
reduction (``ATen/native/cuda/Reduce.cuh``):

* Output vector 4, block (32, 4).  Thread ``(x, y)`` of block ``(bx, by)`` owns
  columns ``4 (x + 32 bx) ..`` and rows ``y + 4 by + s i`` (``s = 4 ctas``) in
  four accumulators (row ``idx + j s`` into accumulator ``j``, tail rows into
  accumulators 0, 1, 2), then ``((a0 + a1) + a2) + a3``.
* The block tree over the four warps: ``(v0 + v2) + (v1 + v3)``.
* With ``ctas > 1`` torch's last block: thread ``y`` adds the partials
  ``y, y + 4, ..`` to 0, then the same tree.
* ``MeanOps.project``: ``* factor`` with ``factor = float(K) / float(N K)`` in
  float32.
* Above 2 GiB ``TensorIterator::with_32bit_indexing`` halves the row dim (first
  half first) until each part fits.  Each half is reduced with its own
  ``ctas``, added into the output (``out + value``), and only the last half
  projects (:func:`mean0_leaves`).

Every step is one float32 add or multiply, so the bits are the same.  The lab
proved the order with a torch emulation on 87 shapes, including the row-half
split (a benchmark script).  ``c`` itself is torch's own
``mean.max()``.

Gating
------
:func:`mean0_config` returns ``None`` outside the copied configuration:
``K % 4 != 0`` or ``K < 128`` (another output vector or block shape), or
``N < 64`` (another split over the warps).  The callers' rows decline then.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import warp as wp

from scatterem.physics import _warp_interop as wi

__all__ = ["mean0_config", "mean0_fits", "mean0_leaves", "mean0_max"]

#: ``device index -> SMs * (max threads per SM // 128)``, torch's target grid.
_TARGET: dict[int, int] = {}


@wp.func
def _tree4(
    s: wp.array(dtype=wp.vec4f),  # pyright: ignore[reportInvalidTypeForm]
    base: int,
    kq: int,
) -> wp.vec4:
    # torch's block_y_reduce over 4 rows: (v0 + v2) + (v1 + v3).
    return (s[base] + s[base + 2 * kq]) + (s[base + kq] + s[base + 3 * kq])


@wp.kernel
def _mean0_final_acc_kernel(
    scratch: wp.array(dtype=wp.vec4f),  # pyright: ignore[reportInvalidTypeForm]
    k: int,
    ctas: int,
    acc: int,
    proj: int,
    factor: wp.float32,
    out: wp.array(dtype=wp.vec4f),  # pyright: ignore[reportInvalidTypeForm]
):
    # One thread per 4 outputs: each block's tree, then torch's last block, then
    # torch's sub-iterator steps: with `acc` the value is added to the output
    # (`out + value`), and only with `proj` is it multiplied by the factor
    # (`MeanOps.project`).
    q = wp.tid()
    kq = k // 4
    v = wp.vec4(0.0)
    if ctas == 1:
        v = _tree4(scratch, q, kq)  # pyright: ignore[reportArgumentType]
    else:
        t0 = wp.vec4(0.0)
        t1 = wp.vec4(0.0)
        t2 = wp.vec4(0.0)
        t3 = wp.vec4(0.0)
        c = int(0)  # noqa: UP018 -- declares a mutable Warp local; a bare literal is a constant
        while c < ctas:
            t0 = t0 + _tree4(scratch, c * 4 * kq + q, kq)  # pyright: ignore[reportOperatorIssue]
            if c + 1 < ctas:
                t1 = t1 + _tree4(scratch, (c + 1) * 4 * kq + q, kq)  # pyright: ignore[reportOperatorIssue]
            if c + 2 < ctas:
                t2 = t2 + _tree4(scratch, (c + 2) * 4 * kq + q, kq)  # pyright: ignore[reportOperatorIssue]
            if c + 3 < ctas:
                t3 = t3 + _tree4(scratch, (c + 3) * 4 * kq + q, kq)  # pyright: ignore[reportOperatorIssue]
            c += 4
        v = (t0 + t2) + (t1 + t3)
    if acc != 0:
        v = out[q] + v
    if proj != 0:
        v = v * factor
    out[q] = v


def mean0_config(n: int, k: int, device: torch.device) -> tuple[int, int] | None:
    """``(grid_x, ctas)`` of torch's ``mean(0)`` over a fresh float32 ``[n, k]``.

    ``setReduceConfig`` (outer reduction): output vector 4 needs ``k % 4 == 0``
    (a fresh buffer is aligned); block (32, 4) needs ``k // 4 >= 32`` and
    ``n >= 4``; the split over the 4 warps needs ``n >= 64``.  Above 2 GiB torch
    splits the iterator; :func:`mean0_leaves` gives the parts.

    Parameters
    ----------
    n : int
        The number of rows (scan positions) of one part.
    k : int
        The number of columns (detector pixels).
    device : torch.device
        The CUDA device.

    Returns
    -------
    tuple of int or None
        ``(grid_x, ctas)``, or ``None`` outside the configuration copied here.
    """
    if k % 4 or k // 4 < 32 or n < 64 or n * k * 4 >= 2**31 - 1:
        return None
    key = device.index if device.index is not None else torch.cuda.current_device()
    target = _TARGET.get(key)
    if target is None:
        p = torch.cuda.get_device_properties(key)
        target = p.multi_processor_count * (p.max_threads_per_multi_processor // 128)
        _TARGET[key] = target
    grid_x = -(-(k // 4) // 32)
    vpt = -(-n // 4)
    ctas = 1
    if vpt >= 256 and grid_x <= target:
        ctas = max(min(-(-target // grid_x), -(-vpt // 16)), -(-vpt // 256))
    return grid_x, ctas


def mean0_leaves(start: int, n: int, k: int) -> list[tuple[int, int]]:
    """Row parts of torch's ``with_32bit_indexing`` for a float32 ``[n, k]``.

    Parameters
    ----------
    start : int
        The first row.
    n : int
        The number of rows.
    k : int
        The number of columns.

    Returns
    -------
    list of tuple of int
        ``(first row, rows)`` per part: the row dim halved (first half first)
        until each part fits 32-bit indexing.
    """
    imax = 2**31 - 1
    if n * k <= imax and 1 + ((n - 1) * k + (k - 1)) * 4 <= imax:
        return [(start, n)]
    h = n // 2
    return mean0_leaves(start, h, k) + mean0_leaves(start + h, n - h, k)


def mean0_fits(n: int, k: int, on: torch.device) -> bool:
    """Whether every row part of ``[n, k]`` has a copied configuration.

    Parameters
    ----------
    n : int
        The number of rows (scan positions).
    k : int
        The number of columns (detector pixels).
    on : torch.device
        The CUDA device.

    Returns
    -------
    bool
        Whether :func:`mean0_max` may run on this shape.
    """
    return all(mean0_config(m, k, on) is not None for _, m in mean0_leaves(0, n, k))


def mean0_max(
    partial: Callable[..., None], src_inputs: list, n: int, k: int, on: torch.device
) -> torch.Tensor:
    """``c = x.reshape(n, k).mean(0).max()`` of the cube that ``partial`` reads.

    Parameters
    ----------
    partial : warp.Kernel
        A partial kernel with the thread map of "Numerics": its arguments are
        ``*src_inputs, r0, n, k, grid_x, step, scratch``.
    src_inputs : list
        Its leading arguments (the source and its element map).
    n : int
        The number of rows (scan positions) of ``x``.
    k : int
        The number of columns (detector pixels) of ``x``.
    on : torch.device
        The CUDA device.  Every part must have a configuration
        (:func:`mean0_fits`).

    Returns
    -------
    torch.Tensor
        ``c`` as a ``(1,)`` float32 tensor on the device.  No host sync.
    """
    leaves = mean0_leaves(0, n, k)
    device = wi.device_from_torch(on)
    # Warp's default stream is not PyTorch's current stream; launch on the torch
    # stream so the kernels are ordered against the ops around them.
    stream = wi.stream_from_torch(torch.cuda.current_stream(on))
    mean = torch.empty((k // 4, 4), dtype=torch.float32, device=on)
    mean_ct = wi.from_torch(mean, dtype=wp.vec4f, return_ctype=True)
    # `MeanOps`' factor of the WHOLE iterator: float(outputs) / numel, in float32.
    f32 = torch.float32
    factor = float(
        torch.tensor(float(k), dtype=f32) / torch.tensor(float(n * k), dtype=f32)
    )
    # Named locals: a raw ctype does not keep its tensor alive.
    scratches = []
    for li, (r0, m) in enumerate(leaves):
        cfg = mean0_config(m, k, on)
        assert cfg is not None  # the caller's row checked every part
        grid_x, ctas = cfg
        scratch = torch.empty((ctas * k, 4), dtype=torch.float32, device=on)
        scratches.append(scratch)
        scratch_ct = wi.from_torch(scratch, dtype=wp.vec4f, return_ctype=True)
        wp.launch(
            partial,
            dim=grid_x * ctas * 128,
            inputs=[*src_inputs, r0, m, k, grid_x, 4 * ctas],
            outputs=[scratch_ct],
            device=device,
            stream=stream,
            block_dim=128,
        )
        last = li == len(leaves) - 1
        wp.launch(
            _mean0_final_acc_kernel,
            dim=k // 4,
            inputs=[scratch_ct, k, ctas, int(li > 0), int(last), factor],
            outputs=[mean_ct],
            device=device,
            stream=stream,
        )
    del scratches
    return mean.max().reshape(1)
