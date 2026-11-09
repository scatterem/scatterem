"""Half-spectrum Fourier-weighted fusion for ``fuse_images_fourier_weighted``.

That is :func:`scatterem.reconstruction.fusion._band.fuse_images_fourier_weighted`.

Private, lazily imported by that function so that ``import scatterem`` never requires
Triton -- same arrangement as :mod:`scatterem.nn.functional._bcast_cmul_triton`.

Why this exists
---------------
The shipped fusion is ``Re(ifft2(fft2(a) * w1 + fft2(b) * w2))`` for two REAL images
``a``, ``b`` and two REAL weights: two real->complex casts, two c2c forward transforms
and one c2c inverse whose imaginary half is thrown away.  At ``r14_ffstem/gd2o3``'s
1024^2 grid that block is device-bound (0.38 ms of a 1.6 ms ``fuse_warm`` row).

The same image comes from the half spectrum.  For any complex ``X``,
``Re(ifft2(X)) = ifft2(herm(X))`` with ``herm(X)(q) = (X(q) + conj X(-q)) / 2``; for a
real image ``conj F(-q) = F(q)``, so with real weights
``herm(F1 w1 + F2 w2) = F1 s1 + F2 s2`` where ``s = (w(q) + w(-q)) / 2``.  That is
Hermitian, so it is ``irfft2`` of its half.  **The symmetrisation is what makes this
exact for EVERY real weight**, not just the radial ones ``fused_full_field`` builds:
dropping it (``irfft2(rfft2(a) * w1[:, :W//2+1] + ...)``) is wrong by ~5e-2 of peak on
an asymmetric weight, and ~ulp on a radial one only because ``w(q) == w(-q)`` there.
(a benchmark script, arm B0 vs B5.)

``hermitian_combine`` does the symmetrise-multiply-add in ONE kernel, in place into
the first half spectrum.  A torch spelling of the symmetrisation (a mirror gather plus
two elementwise ops per weight) gives back more than half the saving at 1024^2.

Numerics
--------
NOT bit-exact to the shipped spelling: different transforms (r2c/c2r instead of c2c)
and ``fma(F1, s1, F2 * s2)`` instead of two rounded products and an add.  Against a
complex128 evaluation of the SHIPPED formula at 1024^2 it is 5.0e-07 of peak where the
shipped float32 path itself is 5.2e-07 (asymmetric weights; radial 5.6e-07 vs 5.8e-07).

Gating
------
:func:`fuse_half` returns ``None`` -- and the caller keeps the shipped ops -- unless:
CUDA; ``a``, ``b``, ``w1``, ``w2`` all float32, 2-D, the SAME shape, contiguous; no
autograd on any of them (forward-only kernel); ``H * W >= MIN_NUMEL`` (below it the
block is host-bound and the extra launch loses: 0.797x at 512^2, 1.218x at 768^2,
2.149x at 1024^2 on an RTX A6000); fewer than ``2**29`` elements (int32 offsets).
``SCATTEREM_FUSE_HALF_C2R=0`` forces the shipped path (read per call, used for A/B).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from scatterem.utils._backends import FUSE_HALF
from scatterem.utils.env import resolve_backend

__all__ = ["fuse_half", "CALLS", "MIN_NUMEL"]

#: ``[ran, declined]``.  Host-side counters only, so a probe can prove the fast path
#: was TAKEN rather than infer it from a timing.
CALLS = [0, 0]

#: Smallest ``H * W`` that takes the half-spectrum path (768^2, the smallest measured win).
MIN_NUMEL = 768 * 768

BLOCK_DEFAULT = 1024


@triton.jit
def _hermitian_combine_kernel(
    F1,  # (H, WH, 2) f32, real view of rfft2(a)  [in/out]
    F2,  # (H, WH, 2) f32, real view of rfft2(b)  [in]
    W1,  # (H, W) f32 contiguous                  [in]
    W2,  # (H, W) f32 contiguous                  [in]
    H,
    W,
    WH,
    S1Y,
    S1X,
    S2Y,
    S2X,
    N,  # H * WH
    BLOCK: tl.constexpr,
):
    e = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = e < N
    y = e // WH
    x = e - y * WH
    # index mirror -q of q = (y, x) on the FULL grid
    my = (H - y) % H
    mx = (W - x) % W
    a1 = tl.load(W1 + y * W + x, mask=m, other=0.0)
    b1 = tl.load(W1 + my * W + mx, mask=m, other=0.0)
    a2 = tl.load(W2 + y * W + x, mask=m, other=0.0)
    b2 = tl.load(W2 + my * W + mx, mask=m, other=0.0)
    s1 = (a1 + b1) * 0.5
    s2 = (a2 + b2) * 0.5
    o1 = 2 * (y * S1Y + x * S1X)
    o2 = 2 * (y * S2Y + x * S2X)
    f1r = tl.load(F1 + o1, mask=m, other=0.0)
    f1i = tl.load(F1 + o1 + 1, mask=m, other=0.0)
    f2r = tl.load(F2 + o2, mask=m, other=0.0)
    f2i = tl.load(F2 + o2 + 1, mask=m, other=0.0)
    tl.store(F1 + o1, tl.math.fma(f1r, s1, f2r * s2), mask=m)
    tl.store(F1 + o1 + 1, tl.math.fma(f1i, s1, f2i * s2), mask=m)


def fuse_half(
    a: torch.Tensor,
    b: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    block: int = BLOCK_DEFAULT,
) -> torch.Tensor | None:
    """``Re(ifft2(fft2(a, ortho) * w1 + fft2(b, ortho) * w2, ortho))`` via the half spectrum.

    Returns ``None`` without launching anything when a gate in the module docstring
    fails; the caller must then run the shipped ops itself.
    """
    if resolve_backend(FUSE_HALF) != "triton":
        CALLS[1] += 1
        return None
    ts = (a, b, w1, w2)
    if not all(t.is_cuda and t.dtype is torch.float32 and t.dim() == 2 for t in ts):
        CALLS[1] += 1
        return None
    if not all(t.shape == a.shape and t.is_contiguous() for t in ts):
        CALLS[1] += 1
        return None
    if torch.is_grad_enabled() and any(t.requires_grad for t in ts):
        CALLS[1] += 1
        return None
    h, w = a.shape
    if h * w < MIN_NUMEL or 2 * h * w >= 2**30:
        CALLS[1] += 1
        return None
    wh = w // 2 + 1
    # Transform a (1, H, W) view so the half spectrum comes back ROW-major; the plain
    # 2-D call returns it column-major, which the kernel's row-major weight reads
    # would then walk against.
    f1 = torch.fft.rfft2(a[None], dim=(1, 2), norm="ortho")[0]
    f2 = torch.fft.rfft2(b[None], dim=(1, 2), norm="ortho")[0]
    n = h * wh
    _hermitian_combine_kernel[(triton.cdiv(n, block),)](  # pyright: ignore[reportIndexIssue]
        torch.view_as_real(f1),
        torch.view_as_real(f2),
        w1,
        w2,
        h,
        w,
        wh,
        f1.stride(0),
        f1.stride(1),
        f2.stride(0),
        f2.stride(1),
        n,
        BLOCK=block,
        num_warps=4,
    )
    CALLS[0] += 1
    return torch.fft.irfft2(f1, s=(h, w), dim=(0, 1), norm="ortho")
