"""Fused GPU evaluation of the rotation objective (optional fast path).

This module computes the same two objective curves as the reference torch
implementation in :mod:`._rotation_solver`, with two Triton launches:

1. A grid over tiles of ``32 × 32`` cells. Each program loads the four
   corner values of every cell of its tile (a ``33 × 33`` block per
   component), converts them to float64, applies the cell-centred stencil and
   writes six tile sums ``K, D, M, K_P, D_P, M_P`` to a float64 buffer
   ``(n_tiles, 6)``.
2. One program. It reduces the buffer in tile order (fixed order, no atomics)
   and evaluates both curves for all angles in blocks of 1024.

Reason for two launches: launch 1 needs the full GPU; launch 2 is tiny and
must see all partial sums in a fixed order, so the result is deterministic.
One launch with atomics would make the result depend on the schedule.
Extra memory: ``48 · n_tiles + 16 · A`` bytes (the buffer and the ``cos 2θ``,
``sin 2θ`` tables). It does not grow with ``A × n0·n1``.

Agreement bound: kernel and reference compute the same float64 terms and
differ only in the summation order. ``max |values_triton − values_torch| <=
1e-9`` in the normalised units of the objective (curves lie in ``[0, 2]``).
That is far below float32 rounding, so the chosen angle is the same unless
two trial angles differ by less than the bound.

``import scatterem`` never imports this module. :mod:`._rotation_solver` imports it
lazily, and a failed ``import triton`` only sets ``_TRITON = None``.
"""

from __future__ import annotations

import logging
import math
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

try:  # Triton is optional.
    import triton
    import triton.language as tl
except ImportError:  # also when sys.modules["triton"] is None
    # Any, not None: the kernels below are defined only when Triton imported.
    triton = tl = cast(Any, None)

_TRITON = triton

__all__ = ["rotation_curves_triton", "triton_rotation_supported"]

logger = logging.getLogger(__name__)

_TILE = 32
_LANE_WIDTH = 1024

if _TRITON is not None:

    @triton.jit
    def _tile_moments_kernel(
        fx_ptr,
        fy_ptr,
        out_ptr,
        n0,
        n1,
        s0,
        s1,
        n_t1,
        TILE: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
    ):
        p0 = tl.program_id(0)
        p1 = tl.program_id(1)
        i = p0 * TILE + tl.arange(0, TILE)[:, None]
        j = p1 * TILE + tl.arange(0, TILE)[None, :]
        mask = (i < n0 - 1) & (j < n1 - 1)
        o = i * n1 + j
        x00 = tl.load(fx_ptr + o, mask=mask, other=0.0).to(tl.float64)
        x10 = tl.load(fx_ptr + o + n1, mask=mask, other=0.0).to(tl.float64)
        x01 = tl.load(fx_ptr + o + 1, mask=mask, other=0.0).to(tl.float64)
        x11 = tl.load(fx_ptr + o + n1 + 1, mask=mask, other=0.0).to(tl.float64)
        y00 = tl.load(fy_ptr + o, mask=mask, other=0.0).to(tl.float64)
        y10 = tl.load(fy_ptr + o + n1, mask=mask, other=0.0).to(tl.float64)
        y01 = tl.load(fy_ptr + o + 1, mask=mask, other=0.0).to(tl.float64)
        y11 = tl.load(fy_ptr + o + n1 + 1, mask=mask, other=0.0).to(tl.float64)
        a = 0.5 * ((x11 - x01) + (x10 - x00)) / s0
        b = 0.5 * ((x11 - x10) + (x01 - x00)) / s1
        c = 0.5 * ((y11 - y01) + (y10 - y00)) / s0
        d = 0.5 * ((y11 - y10) + (y01 - y00)) / s1
        kap = c - b
        dlt = a + d
        kap_p = a - d
        dlt_p = b + c
        base = out_ptr + (p0 * n_t1 + p1) * 6
        tl.store(base + 0, tl.sum(tl.sum(kap * kap, axis=1), axis=0))
        tl.store(base + 1, tl.sum(tl.sum(dlt * dlt, axis=1), axis=0))
        tl.store(base + 2, tl.sum(tl.sum(kap * dlt, axis=1), axis=0))
        tl.store(base + 3, tl.sum(tl.sum(kap_p * kap_p, axis=1), axis=0))
        tl.store(base + 4, tl.sum(tl.sum(dlt_p * dlt_p, axis=1), axis=0))
        tl.store(base + 5, tl.sum(tl.sum(kap_p * dlt_p, axis=1), axis=0))

    @triton.jit
    def _curves_kernel(
        buf_ptr,
        n_tiles,
        c2_ptr,
        s2_ptr,
        out_ptr,
        angle_count,
        DIVERGENCE: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
        LANE_WIDTH: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
    ):
        acc_k = tl.zeros([LANE_WIDTH], dtype=tl.float64)
        acc_d = tl.zeros([LANE_WIDTH], dtype=tl.float64)
        acc_m = tl.zeros([LANE_WIDTH], dtype=tl.float64)
        acc_kp = tl.zeros([LANE_WIDTH], dtype=tl.float64)
        acc_dp = tl.zeros([LANE_WIDTH], dtype=tl.float64)
        acc_mp = tl.zeros([LANE_WIDTH], dtype=tl.float64)
        for start in range(0, n_tiles, LANE_WIDTH):
            t = start + tl.arange(0, LANE_WIDTH)
            m = t < n_tiles
            acc_k += tl.load(buf_ptr + t * 6 + 0, mask=m, other=0.0)
            acc_d += tl.load(buf_ptr + t * 6 + 1, mask=m, other=0.0)
            acc_m += tl.load(buf_ptr + t * 6 + 2, mask=m, other=0.0)
            acc_kp += tl.load(buf_ptr + t * 6 + 3, mask=m, other=0.0)
            acc_dp += tl.load(buf_ptr + t * 6 + 4, mask=m, other=0.0)
            acc_mp += tl.load(buf_ptr + t * 6 + 5, mask=m, other=0.0)
        k = tl.sum(acc_k, axis=0)
        d = tl.sum(acc_d, axis=0)
        mm = tl.sum(acc_m, axis=0)
        kp = tl.sum(acc_kp, axis=0)
        dp = tl.sum(acc_dp, axis=0)
        mp = tl.sum(acc_mp, axis=0)
        f = 0.5 * ((k + d) + (kp + dp))
        pos = f > 0.0
        f_safe = tl.where(pos, f, 1.0)
        for start in range(0, angle_count, LANE_WIDTH):
            a = start + tl.arange(0, LANE_WIDTH)
            m = a < angle_count
            c2 = tl.load(c2_ptr + a, mask=m, other=0.0)
            s2 = tl.load(s2_ptr + a, mask=m, other=0.0)
            if DIVERGENCE:
                n_0 = 0.5 * (k + d) - 0.5 * (k - d) * c2 - mm * s2
                n_1 = 0.5 * (kp + dp) - 0.5 * (kp - dp) * c2 - mp * s2
            else:
                n_0 = 0.5 * (k + d) + 0.5 * (k - d) * c2 + mm * s2
                n_1 = 0.5 * (kp + dp) + 0.5 * (kp - dp) * c2 + mp * s2
            tl.store(out_ptr + a, tl.where(pos, n_0 / f_safe, 0.0), mask=m)
            tl.store(
                out_ptr + angle_count + a, tl.where(pos, n_1 / f_safe, 0.0), mask=m
            )


def triton_rotation_supported(comp_a0, comp_a1, angle_grid_deg=None) -> bool:
    """Say whether the fused kernel handles these inputs exactly.

    Callable on its own; never raises. Declined cases:

    1. Triton cannot be imported, or CUDA is not available.
    2. A field is not a ``torch.Tensor``, not on a CUDA device, or the two
       fields are on different devices.
    3. A dtype is not ``torch.float32`` (float64 goes to the reference, which
       is float64 inside already).
    4. A field is not 2-D, the shapes differ, or ``n0 < 2`` or ``n1 < 2``.
    5. ``n0 · n1 >= 2**31`` (the kernel uses 32-bit offsets).
    6. A field requires grad (the kernel has no backward).
    7. ``angle_grid_deg`` is given and is empty, not 1-D or not finite.

    Non-contiguous fields are accepted; the launch makes a contiguous copy.
    """
    try:
        if _TRITON is None or not torch.cuda.is_available():
            return False
        if not (isinstance(comp_a0, Tensor) and isinstance(comp_a1, Tensor)):
            return False
        if (
            not (comp_a0.is_cuda and comp_a1.is_cuda)
            or comp_a0.device != comp_a1.device
        ):
            return False
        if comp_a0.dtype != torch.float32 or comp_a1.dtype != torch.float32:
            return False
        if comp_a0.ndim != 2 or comp_a0.shape != comp_a1.shape:
            return False
        n0, n1 = comp_a0.shape
        if n0 < 2 or n1 < 2 or n0 * n1 >= 2**31:
            return False
        if comp_a0.requires_grad or comp_a1.requires_grad:
            return False
        if angle_grid_deg is not None:
            if isinstance(angle_grid_deg, Tensor):
                angle_grid_deg = angle_grid_deg.detach().cpu().numpy()
            arr = np.asarray(angle_grid_deg, dtype=np.float64)
            if arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr)):
                return False
        return True
    except Exception:  # noqa: BLE001 - the predicate never raises
        return False


def rotation_curves_triton(
    comp_a0,
    comp_a1,
    angle_grid_deg,
    *,
    curve_kind: str = "curl",
    scan_spacing=(1.0, 1.0),
    strict: bool = False,
) -> Tensor | None:
    """Run the fused kernel and return both curves, or None.

    Parameters
    ----------
    comp_a0, comp_a1 : Tensor
        float32 CUDA ``(n0, n1)`` fields.
    angle_grid_deg : array_like
        1-D finite trial angles in degrees.
    curve_kind : {"curl", "divergence"}, optional
    scan_spacing : (float, float), optional
    strict : bool, optional
        Default False: a Triton error is logged at DEBUG and gives None.
        True: the error propagates. Tests use it, so that a broken kernel
        fails instead of falling back without notice.

    Returns
    -------
    Tensor or None
        float64 ``(2, A)`` on the device of the field; row 0 as measured,
        row 1 swapped. None when the predicate declines or (not ``strict``)
        Triton raises; the caller then runs the reference implementation.
    """
    if not triton_rotation_supported(comp_a0, comp_a1, angle_grid_deg):
        return None
    try:
        from scatterem.data.calibration._rotation_ops import _angle_trig

        s0, s1 = float(scan_spacing[0]), float(scan_spacing[1])
        fx = comp_a0.contiguous()
        fy = comp_a1.contiguous()
        n0, n1 = fx.shape
        n_t0 = math.ceil((n0 - 1) / _TILE)
        n_t1 = math.ceil((n1 - 1) / _TILE)
        buf = torch.empty((n_t0 * n_t1, 6), dtype=torch.float64, device=fx.device)
        angles = np.asarray(
            angle_grid_deg.detach().cpu().numpy()
            if isinstance(angle_grid_deg, Tensor)
            else angle_grid_deg,
            dtype=np.float64,
        )
        c2, s2 = _angle_trig(angles, fx.device)
        out = torch.empty((2, angles.size), dtype=torch.float64, device=fx.device)
        with torch.cuda.device(fx.device):
            _tile_moments_kernel[(n_t0, n_t1)](  # pyright: ignore[reportIndexIssue]
                fx, fy, buf, n0, n1, s0, s1, n_t1, TILE=_TILE
            )
            _curves_kernel[(1,)](  # pyright: ignore[reportIndexIssue]
                buf,
                n_t0 * n_t1,
                c2,
                s2,
                out,
                angles.size,
                DIVERGENCE=curve_kind == "divergence",
                LANE_WIDTH=_LANE_WIDTH,
            )
        return out
    except Exception as err:  # noqa: BLE001 - any Triton failure falls back to torch
        if strict:
            raise
        logger.debug("fused rotation kernel did not run: %r", err)
        return None
