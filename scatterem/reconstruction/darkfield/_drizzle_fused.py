"""The drizzle splat of :func:`~scatterem.reconstruction.darkfield.drizzle.drizzle_resample`
as ONE Warp gather kernel.

Why this exists
---------------
The eager splat builds, for every live ``(oy, ox)`` window offset, a full-size
weight, index and mask over all ``N*H*W`` input pixels -- ``(P, N, H, W)``
tensors -- and then scatters them with two ``index_add_`` per offset.  On the
flagship FF-STEM figure (``r14_ffstem/gd2o3/tcdf``: ``N = 8`` frames of
``512 x 512`` per call, ``U = 2``, ``P = 9``) that is ~40 ATen launches per call
and **~12.8 ms of device time per row** (``index_add_`` 4.7, the window
``copy_``/``where``/``mul``/``add``/``bitwise_and`` 8.1), on a row that is 93 %
device-bound -- to produce two ``(H*U, W*U)`` grids from 8 MiB of input.

The splat is SEPARABLE and every output pixel receives at most
``ceil((2*radius+1)/U)**2`` terms per frame, so it can be GATHERED instead: one
thread per output pixel walks the few input pixels whose drop overlaps it.  No
window is materialised and no atomic is issued.

What is and is not the same
---------------------------
* **Every term is the eager term, word for word.**  The drop centre is
  ``i*U + shift`` (the product is exact), the overlap is
  ``(min(t + 0.5, c + hw) - max(t - 0.5, c - hw)).clamp_min(0) * inv2hw`` per
  axis with ``hw``/``inv2hw`` rounded to float32 exactly as the eager op rounds a
  Python scalar, the weight is ``wy * wx`` and the flux ``v * (wy * wx)``.  The
  multiplies are ``__fmul_rn`` so nvrtc cannot contract a product into the
  following add; ``torch.round`` is round-half-to-EVEN and so is ``rint``
  (``wp.round`` is half-AWAY and would move a drop sitting exactly on a
  half-pixel).
* **The SUM ORDER is not.**  The eager ``index_add_`` is an ``atomicAdd`` on
  CUDA, so its order is already unspecified launch to launch; this kernel sums
  each pixel's terms in a fixed ``(n, i, j)`` order.  The difference is float32
  reassociation, graded against the eager path in the report.
* **Zero-weight terms.**  The eager path visits only offsets that carry weight
  SOMEWHERE and parks out-of-bounds entries on pixel 0 with weight ``+0.0``;
  this kernel visits every term within the structurally live radius
  (:func:`_live_radius`, a superset of those offsets).  Both add exact zeros for finite
  input, so they agree bit for bit on which terms are nonzero; they can differ
  only in where a NON-FINITE input value's ``x * 0.0`` lands.

Scope of the gate
-----------------
CUDA only (``wp.launch`` on a CPU device is a single-threaded scalar loop),
float32 accumulation only, forward-only (``drizzle_resample`` is
``@torch.no_grad`` and the kernel has no adjoint, so no gradcheck obligation
arises), and ``SCATTEREM_DRIZZLE_FUSED=0`` restores the eager splat per call
(the kill switch of the ``drizzle`` row in ``scatterem/reconstruction/_backends.py``,
which calls :func:`drizzle_fused_supported`).  The predicate is entirely
host-known, so it costs no device sync.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import warp as wp
from torch import Tensor

from scatterem.physics import _warp_interop as wi
from scatterem.physics._warp_rn import fadd_rn as _fadd
from scatterem.physics._warp_rn import fmul_rn as _fmul
from scatterem.physics._warp_rn import fsub_rn as _fsub


@wp.func
def _axis_weight(t: wp.float32, c: wp.float32, hw: wp.float32, inv2hw: wp.float32):
    """The eager per-axis weight at output coordinate ``t`` for drop centre ``c``:
    ``(min(t + 0.5, c + hw) - max(t - 0.5, c - hw)).clamp_min(0) * inv2hw``."""
    seg = _fsub(
        wp.min(_fadd(t, 0.5), _fadd(c, hw)), wp.max(_fsub(t, 0.5), _fsub(c, hw))
    )
    return _fmul(wp.max(seg, 0.0), inv2hw)


@wp.kernel
def _drizzle_gather_kernel(
    vals: wp.array3d(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    shifts: wp.array2d(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    upsample: wp.int32,
    hw: wp.float32,
    inv2hw: wp.float32,
    radius: wp.int32,
    accum: wp.array2d(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    hits: wp.array2d(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
):
    """``accum[Y, X] = sum v * (wy * wx)`` and ``hits[Y, X] = sum wy * wx`` over
    every ``(n, i, j)`` whose rounded drop centre lies within ``radius`` of
    ``(Y, X)`` -- exactly the eager window's in-bounds entries."""
    y, x = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    n_frames = vals.shape[0]
    h_in = vals.shape[1]
    w_in = vals.shape[2]
    yf = float(y)
    xf = float(x)
    uf = float(upsample)
    rf = float(radius)
    a = float(0.0)  # noqa: UP018 -- declares a mutable Warp local; a bare literal is a constant
    h = float(0.0)  # noqa: UP018 -- declares a mutable Warp local; a bare literal is a constant
    for n in range(n_frames):
        dy = shifts[n, 0]
        dx = shifts[n, 1]
        # rint(i*U + d) within radius of the pixel => i*U + d within
        # radius + 0.5 of it; one extra row/col each side absorbs the rounding
        # of the division, and the exact test below decides membership.
        i0 = wp.max(int(wp.floor((yf - rf - 0.5 - dy) / uf)) - 1, 0)
        i1 = wp.min(int(wp.ceil((yf + rf + 0.5 - dy) / uf)) + 1, h_in - 1)
        j0 = wp.max(int(wp.floor((xf - rf - 0.5 - dx) / uf)) - 1, 0)
        j1 = wp.min(int(wp.ceil((xf + rf + 0.5 - dx) / uf)) + 1, w_in - 1)
        for i in range(i0, i1 + 1):
            cy = _fadd(_fmul(float(i), uf), dy)
            if wp.abs(yf - wp.rint(cy)) <= rf:  # pyright: ignore[reportOperatorIssue]
                wy = _axis_weight(yf, cy, hw, inv2hw)  # pyright: ignore[reportArgumentType]
                for j in range(j0, j1 + 1):
                    cx = _fadd(_fmul(float(j), uf), dx)
                    if wp.abs(xf - wp.rint(cx)) <= rf:  # pyright: ignore[reportOperatorIssue]
                        w = _fmul(wy, _axis_weight(xf, cx, hw, inv2hw))  # pyright: ignore[reportArgumentType]
                        a = _fadd(a, _fmul(vals[n, i, j], w))
                        h = _fadd(h, w)
    accum[y, x] = a
    hits[y, x] = h


def drizzle_fused_supported(images: Tensor, shifts: Tensor) -> bool:
    """Whether :func:`drizzle_splat_fused` can replace the eager splat here.

    ``images``/``shifts`` are the operands AFTER ``drizzle_resample``'s own dtype
    and device coercion.  Host-known only, so no device sync."""
    return (
        images.device.type == "cuda"
        and images.dtype == torch.float32
        and shifts.dtype == torch.float32
        and shifts.device == images.device
        and images.dim() == 3
    )


def _live_radius(hw: float, radius: int) -> int:
    """The widest integer offset ``k = t - rint(c)`` that can carry weight.

    For ``|k| >= hw + 1`` the drop and the output pixel are disjoint in float32
    too: ``t -/+ 0.5`` is representable and on the far side of ``c +/- hw``, so
    the rounded overlap is ``<= 0`` and the term adds ``+-0.0``, a no-op on a
    finite sum.  Dropping those terms keeps every nonzero one in its ``(n, i,
    j)`` order, so the result is bit-identical.  ``hw`` is
    rounded to float32 first because that is the value the kernel compares."""
    hw32 = float(np.float32(hw))
    return min(int(radius), int(math.ceil(hw32 + 1.0)) - 1)


def drizzle_splat_fused(
    images: Tensor,
    shifts: Tensor,
    upsample: int,
    hw: float,
    inv2hw: float,
    radius: int,
) -> tuple[Tensor, Tensor]:
    """``(accum, hits)``, each ``(H*U, W*U)`` float32, in one launch.  Caller
    checks :func:`drizzle_fused_supported`.  ``radius`` is the eager window's;
    the kernel walks only :func:`_live_radius` of it."""
    n, h_in, w_in = (int(s) for s in images.shape)
    hh, wh = h_in * int(upsample), w_in * int(upsample)
    accum = torch.empty((hh, wh), device=images.device, dtype=torch.float32)
    hits = torch.empty((hh, wh), device=images.device, dtype=torch.float32)
    if hh == 0 or wh == 0:
        return accum, hits
    if n == 0:
        return accum.zero_(), hits.zero_()
    wp.launch(
        _drizzle_gather_kernel,
        dim=(hh, wh),
        inputs=[
            wi.from_torch(images.contiguous(), dtype=wp.float32),
            wi.from_torch(shifts.contiguous(), dtype=wp.float32),
            int(upsample),
            float(hw),
            float(inv2hw),
            _live_radius(hw, radius),
        ],
        outputs=[
            wi.from_torch(accum, dtype=wp.float32),
            wi.from_torch(hits, dtype=wp.float32),
        ],
        device=wi.device_from_torch(images.device),
        # Warp's default stream is not PyTorch's current stream; a launch left
        # there is unordered against the ATen ops around it (physics/_ramp_fused.py).
        stream=wi.stream_from_torch(torch.cuda.current_stream(images.device)),
    )
    return accum, hits
