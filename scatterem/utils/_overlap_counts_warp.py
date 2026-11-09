"""One-kernel ``double_and_triple_pixel_counts_torch`` for CUDA inputs.

Private, lazily imported by
:func:`scatterem.utils.transfer.double_and_triple_pixel_counts_torch`.

Why this exists
---------------
The torch spelling evaluates the triple- and three pair-overlap areas as ~70 separate
float64 elementwise launches (``.double()``, ``zeros_like``, compares, clamps,
``arccos``/``arcsin``, ``sqrt``, ``where`` ...).  On the analytical-SSNR path of
``direct_ptychography(return_snr=True)`` the grid is small (96^2 at
``r14_ffstem/carbon``), so the block is pure host issue: ~650 us per call, on a row
that is host-bound end to end.

Warp, not Triton, on purpose: the FF-STEM path already runs Warp kernels, so this
module's first launch in a process is a cached-module load, whereas a Triton kernel
there was the process's FIRST Triton launch and paid Triton's runtime setup -- a
measured +45 ms on the first ``direct_ptychography`` call per process.

Numerics
--------
BIT-EXACT to the torch spelling, by construction and checked bitwise (int64 views,
so the sign of zero counts) on the real call and adversarial grids.  Every step is
the same IEEE operation in the same order: ATen divides a tensor by a CPU scalar as a
multiply by the host-computed reciprocal, so ``d / (2R)`` is ``d * (1/(2R))`` here
too; ``x**2`` is ``x * x``; ``2 * q`` is ``q + q`` (exact); ``A12`` is ``A01`` (the
same call on the same arguments).  Products, sums and the square root use the
round-to-nearest intrinsics so nvrtc cannot contract a multiply into a following add
as an FMA (one rounding where ATen has two); ``acos``/``asin`` are the CUDA math
library routines ATen's float64 kernels call.  The host constants are formed from
the caller's ``R`` and ``delta_k`` objects with the SAME expressions the torch
spelling uses.

Gating
------
The ``overlap_counts`` row of :mod:`scatterem.utils._backends`, whose predicate is
:func:`overlap_counts_supported`: ``q`` is a CUDA float32/float64 tensor with no
autograd; ``R`` and ``delta_k`` are real python/numpy scalars (not tensors, not
bool), finite and positive, and so are the derived constants.  Otherwise
:func:`overlap_counts` returns ``None`` and the caller keeps the torch ops.
"""

from __future__ import annotations

import math
import numbers

import torch
import warp as wp

from scatterem.physics import _warp_interop as wi
from scatterem.physics._warp_rn import dadd_rn as _dadd
from scatterem.physics._warp_rn import dmul_rn as _dmul
from scatterem.physics._warp_rn import dsqrt_rn as _dsqrt
from scatterem.physics._warp_rn import dsub_rn as _dsub
from scatterem.utils._backends import OVERLAP_COUNTS
from scatterem.utils.env import resolve_backend
from scatterem.utils.warp.launch import ensure_warp_initialized

__all__ = ["CALLS", "overlap_consts", "overlap_counts", "overlap_counts_supported"]

#: ``[ran, declined]``.  Host-side counters only, so a probe can prove the fast path
#: was TAKEN rather than infer it from a timing.
CALLS = [0, 0]


@wp.func_native("return acos(a);")
def _dacos(a: wp.float64) -> wp.float64: ...


@wp.func_native("return asin(a);")
def _dasin(a: wp.float64) -> wp.float64: ...


@wp.func
def _clamp_unit(x: wp.float64) -> wp.float64:
    return wp.min(wp.max(x, wp.float64(-1.0)), wp.float64(1.0))


@wp.func
def _pair_area(
    d: wp.float64,
    two_r: wp.float64,
    inv_two_r: wp.float64,
    four_r2: wp.float64,
    two_r2: wp.float64,
) -> wp.float64:
    # transfer.pair_overlap_area_torch, op for op
    arg = _clamp_unit(_dmul(d, inv_two_r))
    s = wp.max(_dsub(four_r2, _dmul(d, d)), wp.float64(0.0))
    a = _dsub(_dmul(two_r2, _dacos(arg)), _dmul(_dmul(wp.float64(0.5), d), _dsqrt(s)))
    if d < two_r:
        return a
    return wp.float64(0.0)


@wp.func
def _counts(
    q: wp.float64,
    two_r: wp.float64,
    inv_two_r: wp.float64,
    four_r2: wp.float64,
    two_r2: wp.float64,
    r: wp.float64,
    inv_r: wp.float64,
    r2: wp.float64,
    pi_r2: wp.float64,
    inv_pix: wp.float64,
) -> wp.vec2d:
    # transfer.triple_overlap_area_torch, op for op
    arg3 = _clamp_unit(_dmul(q, inv_r))
    s3 = wp.max(_dsub(r2, _dmul(q, q)), wp.float64(0.0))
    a3 = _dsub(
        _dsub(pi_r2, _dmul(two_r2, _dasin(arg3))), _dmul(_dadd(q, q), _dsqrt(s3))
    )
    if not (q <= r):
        a3 = wp.float64(0.0)
    # transfer.double_and_triple_overlap_areas_torch: A12 == A01
    a01 = _pair_area(q, two_r, inv_two_r, four_r2, two_r2)
    a02 = _pair_area(_dadd(q, q), two_r, inv_two_r, four_r2, two_r2)
    a2 = _dsub(_dadd(_dadd(a01, a01), a02), _dmul(wp.float64(3.0), a3))
    if q >= two_r:
        a2 = wp.float64(0.0)
        a3 = wp.float64(0.0)
    return wp.vec2d(_dmul(a2, inv_pix), _dmul(a3, inv_pix))


@wp.kernel
def _overlap_counts_f32_kernel(
    q: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    n2: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
    n3: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
    two_r: wp.float64,
    inv_two_r: wp.float64,
    four_r2: wp.float64,
    two_r2: wp.float64,
    r: wp.float64,
    inv_r: wp.float64,
    r2: wp.float64,
    pi_r2: wp.float64,
    inv_pix: wp.float64,
):
    i = wp.tid()
    v = _counts(
        wp.float64(q[i]),
        two_r,
        inv_two_r,
        four_r2,
        two_r2,
        r,
        inv_r,
        r2,
        pi_r2,
        inv_pix,
    )
    n2[i] = v[0]  # pyright: ignore[reportIndexIssue]
    n3[i] = v[1]  # pyright: ignore[reportIndexIssue]


@wp.kernel
def _overlap_counts_f64_kernel(
    q: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
    n2: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
    n3: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
    two_r: wp.float64,
    inv_two_r: wp.float64,
    four_r2: wp.float64,
    two_r2: wp.float64,
    r: wp.float64,
    inv_r: wp.float64,
    r2: wp.float64,
    pi_r2: wp.float64,
    inv_pix: wp.float64,
):
    i = wp.tid()
    v = _counts(q[i], two_r, inv_two_r, four_r2, two_r2, r, inv_r, r2, pi_r2, inv_pix)
    n2[i] = v[0]  # pyright: ignore[reportIndexIssue]
    n3[i] = v[1]  # pyright: ignore[reportIndexIssue]


def _is_real_scalar(x) -> bool:
    return (
        isinstance(x, numbers.Real)
        and not isinstance(x, bool)
        and not torch.is_tensor(x)
        and math.isfinite(float(x))
        and float(x) > 0.0
    )


def overlap_consts(R, delta_k) -> tuple[float, ...] | None:
    """The host constants ``_counts`` takes, or ``None`` if ``R``/``delta_k`` fail
    the gate.  The SAME expressions, on the SAME scalar objects, as the torch
    spelling.  Shared with ``reconstruction._ssnr_analytical_warp``."""
    if not (_is_real_scalar(R) and _is_real_scalar(delta_k)):
        return None
    pix_area = delta_k**2
    consts = (
        2 * R,
        1.0 / (2 * R),
        4 * R**2,
        2 * R**2,
        R,
        1.0 / R,
        R**2,
        torch.pi * R**2,
        1.0 / pix_area,
    )
    if not all(math.isfinite(float(c)) and float(c) > 0.0 for c in consts):
        return None
    return tuple(float(c) for c in consts)


def overlap_counts_supported(q, R, delta_k) -> bool:
    """True if :func:`overlap_counts` may replace the torch overlap counts."""
    return (
        torch.is_tensor(q)
        and q.is_cuda
        and q.dtype in (torch.float32, torch.float64)
        and not (torch.is_grad_enabled() and q.requires_grad)
        and overlap_consts(R, delta_k) is not None
    )


def overlap_counts(
    q: torch.Tensor, R, delta_k
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """``(N2, N3)`` of :func:`~scatterem.utils.transfer.double_and_triple_pixel_counts_torch`.

    Returns ``None`` without launching anything when the backend row declines;
    the caller must then run the torch ops itself.
    """
    if resolve_backend(OVERLAP_COUNTS, None, q, R, delta_k) != "warp":
        CALLS[1] += 1
        return None
    consts = overlap_consts(R, delta_k)
    assert consts is not None  # the predicate checked it
    ensure_warp_initialized()
    qf = q.contiguous().view(-1)
    n2 = torch.empty(q.shape, dtype=torch.float64, device=q.device)
    n3 = torch.empty_like(n2)
    if qf.numel():
        kernel = (
            _overlap_counts_f32_kernel
            if q.dtype == torch.float32
            else _overlap_counts_f64_kernel
        )
        wp.launch(
            kernel,
            dim=qf.numel(),
            inputs=[
                wi.from_torch(qf),
                wi.from_torch(n2.view(-1)),
                wi.from_torch(n3.view(-1)),
                *consts,
            ],
            device=wi.device_from_torch(q.device),
            # Warp's default stream is not PyTorch's current stream; launching on
            # Warp's own would leave the kernel unordered against the ATen ops
            # that produce ``q`` and consume ``n2``/``n3``.
            stream=wi.stream_from_torch(torch.cuda.current_stream(q.device)),
        )
    CALLS[0] += 1
    return n2, n3
