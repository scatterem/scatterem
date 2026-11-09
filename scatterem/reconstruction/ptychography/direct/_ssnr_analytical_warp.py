"""Two-kernel elementwise tail of :func:`direct_ptychography_ssnr` for CUDA inputs.

Private, called by :func:`.ctf.direct_ptychography_ssnr` (1.0626x
on ``r14_ffstem/carbon/ptycho``).

Why this exists
---------------
After the PCTF kernel, the analytical SSNR is a chain of small elementwise ops on
the ``[Nqy, Nqx]`` grid: the PCTF's bright-field normalisation (``sqrt(Ky**2 +
Kx**2) < alpha/lambda``, ``2 * A.sum()``, divide -- 7 launches), an ``fftshift``
(gone on main), the double/triple overlap counts, and the
noise/SNR epilogue (add, divide, sqrt, cast, divide, square, scale, masked fill).
On ``r14_ffstem/carbon/ptycho`` the grid is 96^2, so every one of them is pure host
issue on a host-bound row (PCTF tail + epilogue + overlap wrapper ~0.4 ms of a
3.4 ms row).  Here they are two launches: a bright-field pixel count
and one per-pixel kernel that evaluates the overlap counts (the same ``wp.func``
the one-kernel ``double_and_triple_pixel_counts_torch`` uses) and the epilogue.

Numerics
--------
BIT-EXACT to the torch spelling, by construction and checked bitwise.  Every step
is the same IEEE operation in the same order: ``x**2`` is ``x * x``; ATen divides a
tensor by a CPU scalar as a multiply by the host-computed reciprocal, so ``/
Nalpha`` is ``* (1.0 / Nalpha)``; ATen compares and multiplies a float32 tensor
with a python scalar after casting the scalar to float32, so ``alpha/lambda`` and
``fluence`` are cast on the host with ``numpy.float32`` (round to nearest, as
C++'s cast); ``pctf / (2 * A.sum())`` divides by the int64 count converted to
float32 (exact below 2**24, gated).  Products, sums, quotients and square roots
use the round-to-nearest intrinsics so nvrtc cannot contract a multiply into a
following add.  The count is an integer atomic, so it is deterministic.

Gating
------
The ``ssnr_epilogue`` row of :mod:`scatterem.reconstruction._backends`, whose
predicate is :func:`ssnr_epilogue_supported`: every tensor is CUDA float32
without autograd on one device, the overlap-count constants are valid (the gate
of :mod:`scatterem.utils._overlap_counts_warp`), ``Nalpha``, ``fluence`` and
``alpha / lambda`` are finite with ``Nalpha > 0``, and ``0 < n_k < 2**23`` (the
float32 count is exact).  Otherwise :func:`ssnr_epilogue` returns ``None`` and
the caller keeps the torch ops.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import warp as wp

from scatterem.physics import _warp_interop as wi
from scatterem.physics._warp_rn import fadd_rn as _fadd
from scatterem.physics._warp_rn import fdiv_rn as _fdiv
from scatterem.physics._warp_rn import fmul_rn as _fmul
from scatterem.physics._warp_rn import fsqrt_rn as _fsqrt
from scatterem.reconstruction._backends import SSNR_EPILOGUE
from scatterem.utils._overlap_counts_warp import (
    _counts,
    _dadd,
    _dmul,
    _dsqrt,
    overlap_consts,
)
from scatterem.utils.env import resolve_backend

__all__ = ["CALLS", "ssnr_epilogue", "ssnr_epilogue_supported"]

#: ``[ran, declined]``.  Host-side counters only, so a probe can prove the fast path
#: was TAKEN rather than infer it from a timing.
CALLS = [0, 0]


@wp.kernel
def _bright_field_count_kernel(
    kx: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    ky: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    thresh: wp.float32,
    count: wp.array(dtype=wp.int32),  # pyright: ignore[reportInvalidTypeForm]
):
    # phase_contrast_transfer_function: K = sqrt(Ky**2 + Kx**2); A = K < a/l
    i = wp.tid()
    k = _fsqrt(_fadd(_fmul(ky[i], ky[i]), _fmul(kx[i], kx[i])))
    if k < thresh:
        wp.atomic_add(count, 0, 1)


@wp.kernel
def _ssnr_epilogue_kernel(
    pctf: wp.array2d(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    q: wp.array2d(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    count: wp.array(dtype=wp.int32),  # pyright: ignore[reportInvalidTypeForm]
    out: wp.array2d(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    two_r: wp.float64,
    inv_two_r: wp.float64,
    four_r2: wp.float64,
    two_r2: wp.float64,
    r: wp.float64,
    inv_r: wp.float64,
    r2: wp.float64,
    pi_r2: wp.float64,
    inv_pix: wp.float64,
    inv_nalpha: wp.float64,
    fluence: wp.float32,
):
    iy, ix = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    # double_and_triple_pixel_counts_torch
    v = _counts(
        wp.float64(q[iy, ix]),
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
    # sqrt((N2 + N3) / Nalpha).to(float32)
    noise = wp.float32(_dsqrt(_dmul(_dadd(v[0], v[1]), inv_nalpha)))  # pyright: ignore[reportIndexIssue]
    # pctf / (2 * A.sum())
    p = _fdiv(pctf[iy, ix], wp.float32(count[0] * 2))
    # (pctf / noise) ** 2 * fluence; [noise == 0] = 0
    s = _fdiv(p, noise)
    s = _fmul(_fmul(s, s), fluence)
    if noise == wp.float32(0.0):
        s = wp.float32(0.0)
    out[iy, ix] = s


def _f32_cuda(t) -> bool:
    return (
        torch.is_tensor(t)
        and t.is_cuda
        and t.dtype == torch.float32
        and not (torch.is_grad_enabled() and t.requires_grad)
    )


def _inv_nalpha(Nalpha) -> float:
    return 1.0 / float(Nalpha) if float(Nalpha) > 0.0 else math.inf


def ssnr_epilogue_supported(
    pctf_unnormalized, Kx, Ky, semiconvergence_angle, wavelength, q, R, delta_k,
    Nalpha, fluence,
) -> bool:  # fmt: skip
    """True if :func:`ssnr_epilogue` may replace the torch tail."""
    if not (
        _f32_cuda(pctf_unnormalized)
        and pctf_unnormalized.dim() == 2
        and _f32_cuda(Kx)
        and _f32_cuda(Ky)
        and Kx.dim() == 1
        and Ky.shape == Kx.shape
        and 0 < Kx.shape[0] < 2**23
        and _f32_cuda(q)
        and q.shape == pctf_unnormalized.shape
        and q.device == pctf_unnormalized.device == Kx.device == Ky.device
    ):
        return False
    thresh = semiconvergence_angle / wavelength
    return (
        overlap_consts(R, delta_k) is not None
        and math.isfinite(_inv_nalpha(Nalpha))
        and math.isfinite(float(fluence))
        and math.isfinite(thresh)
    )


def ssnr_epilogue(
    pctf_unnormalized: torch.Tensor,
    Kx: torch.Tensor,
    Ky: torch.Tensor,
    semiconvergence_angle,
    wavelength,
    q: torch.Tensor,
    R,
    delta_k,
    Nalpha,
    fluence: float,
) -> torch.Tensor | None:
    """The analytical SNR map ``direct_ptychography_ssnr`` returns, from the
    UNNORMALISED PCTF and the ``[Nqy, Nqx]`` radial frequency grid ``q``.

    Returns ``None`` without launching anything when the backend row declines;
    the caller must then run the torch ops itself.
    """
    args = (pctf_unnormalized, Kx, Ky, semiconvergence_angle, wavelength, q)
    if resolve_backend(SSNR_EPILOGUE, None, *args, R, delta_k, Nalpha, fluence) != (
        "warp"
    ):
        CALLS[1] += 1
        return None
    consts = overlap_consts(R, delta_k)
    assert consts is not None  # the predicate checked it
    thresh = semiconvergence_angle / wavelength
    device = pctf_unnormalized.device
    wp_device = wi.device_from_torch(device)
    # Warp's default stream is not PyTorch's current stream (see
    # _overlap_counts_warp): launch on torch's, so the kernels are ordered after
    # the PCTF kernel and before the caller's consumers.
    stream = wi.stream_from_torch(torch.cuda.current_stream(device))
    count = torch.zeros(1, dtype=torch.int32, device=device)
    wp.launch(
        _bright_field_count_kernel,
        dim=Kx.shape[0],
        inputs=[
            wi.from_torch(Kx),
            wi.from_torch(Ky),
            float(np.float32(thresh)),
            wi.from_torch(count),
        ],
        device=wp_device,
        stream=stream,
    )
    out = torch.empty_like(pctf_unnormalized)
    wp.launch(
        _ssnr_epilogue_kernel,
        dim=tuple(pctf_unnormalized.shape),
        inputs=[
            wi.from_torch(pctf_unnormalized.contiguous()),
            wi.from_torch(q.contiguous()),
            wi.from_torch(count),
            wi.from_torch(out),
            *consts,
            _inv_nalpha(Nalpha),
            float(np.float32(float(fluence))),
        ],
        device=wp_device,
        stream=stream,
    )
    CALLS[0] += 1
    return out
