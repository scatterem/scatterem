"""The ramp/IFFT half of the projection, in torch, around the Warp kernels."""

from __future__ import annotations

import torch
from torch import Tensor

from scatterem.physics._frequency_grid import fft_frequency_grid

from ._fwd_eager import fwd_eager
from ._ramp_kernels import _prism_ramp_mul, _ramp_mul_tile
from .torch_oracle import _ramp_is_identity


class _RampIFFT2(torch.autograd.Function):
    """``ifft2(basis_fft[:, None] * ramp[None, :, None], norm="forward")``.

    FORWARD is the same two operations in the same order as the eager spelling,
    so it is bit-identical -- this exists only to replace the BACKWARD.

    Autograd's chain costs three full passes over the (Nmodes, K, Bp, M1, M2)
    cotangent (2.0 GiB at r2/medium): ``fft2``, then ``mul``'s adjoint writing a
    full-size ``g * conj(ramp)`` temporary, then ``sum_to_size`` reading it back
    to reduce over K.  Measured (medium shape): 24.181 ms total = fft2
    12.547 + temp 6.406 + sum 3.463.  Here the last two collapse into one fused
    Warp kernel and the ``fft2`` stays -- it is a hard floor, because the
    K-reduction carries per-pixel Fourier weights and so cannot be moved to the
    other side of the transform (in real space ``conj(ramp)`` is a dense
    fractional-shift operator, not a cheap one).  ``torch.compile`` cannot fuse
    this: inductor does not code-generate complex operators, falls back to eager
    and is exactly as slow -- do not retry it.

    ``fft2`` is the exact adjoint of ``ifft2(norm="forward")`` (the unnormalised
    inverse DFT), verified elementwise against autograd at max rel 0.0 by
    a benchmark script.

    The reduction sums the K cotangents in a different association than
    ``mul``-then-``sum`` does, and Warp's ``acc += a * b`` contracts to FMA (which
    does not round the product), so ``grad_basis_fft`` is NOT bit-identical --
    ~1.3e-7 relative, about 1 fp32 ulp, and closer to a complex128 reduction
    than the eager chain is at 3 of the 8 stored reduction digests.  The R2
    oracle gates reductions at a derived ``8*sqrt(N_terms)*eps`` and separately
    requires that tolerance to accept fp64 truth; before that fix this
    change could not be landed at all.
    """

    @staticmethod
    def forward(ctx, basis_fft: Tensor, ramp: Tensor) -> Tensor:
        ctx.save_for_backward(ramp)
        # ``forward`` runs with grad DISABLED, so ``_ramp_mul``'s gate takes the
        # bit-exact Warp route here even though both inputs require grad; the
        # hand-written ``backward`` below is untouched by it.
        shifted = _ramp_mul(basis_fft, ramp)
        return torch.fft.ifft2(shifted, norm="forward")

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride]  # torch stub
        ctx, grad_out: Tensor
    ):
        (ramp,) = ctx.saved_tensors
        # fft2 = adjoint of ifft2(norm="forward"); then multiply-by-conj(ramp)
        # and reduce over K in one pass.
        g = torch.fft.fft2(grad_out)
        g_real = torch.view_as_real(g)
        if not g_real.is_contiguous():
            g_real = g_real.contiguous()
        out_real = torch.ops.scatterem.prism_ramp_adjoint(
            g_real, torch.view_as_real(ramp).contiguous()
        )
        return torch.view_as_complex(out_real), None


def _subpixel_ramp(M1: int, M2: int, dr: Tensor, device) -> Tensor:
    """``exp(-2 pi i (qy dry_k + qx drx_k))`` as ``(K, M1, M2)`` complex64.

    The one spelling of the sub-pixel ramp on the Warp route, shared by
    :func:`_build_basis_rs` and the L2-blocked forward (``_l2_block``) so
    both multiply by the same bits.
    """
    q = fft_frequency_grid((M1, M2), [1, 1], device=str(device))  # (2, M1, M2)
    qy = q[0]
    qx = q[1]

    dry = dr[:, 0]
    drx = dr[:, 1]

    return torch.exp(
        (-2j * torch.pi)
        * (qy[None, :, :] * dry[:, None, None] + qx[None, :, :] * drx[:, None, None])
    ).to(torch.complex64)


# Not physics.fourier.phase_ramp: Warp ramp + adjoint
# (r2_prism_project: forward bit-exact).
def _build_basis_rs(
    basis_fft: Tensor,
    dr: Tensor,
    precision: str = "fp32",
    fold_k: bool = False,
    ramp_identity: bool | None = None,
) -> Tensor:
    """Per-position real-space basis (Nmodes, K, Bp, M1, M2) via ramp + ifft2.

    This is the FFT half of the projection — kept in torch.  ``dr`` is the
    sub-pixel offset in PIXELS; the ramp and IFFT normalisation match
    ``_prism_project_torch`` exactly (see its docstring).

    Args:
        basis_fft: (Nmodes, Bp, M1, M2) complex64 — position-independent basis.
        dr:        (K, 2) float32 — sub-pixel offsets in PIXELS.
        precision: "fp32" (default, unchanged: FFT + result in complex64) or a
                   half-precision route ("fp16" / "mixed"): FFT in complex32 / fp16
                   AND the returned real-space basis kept in complex32 — a float16
                   real view that feeds the dtype-generic Warp kernel. For "fp16"
                   the kernel multiplies in fp16; for "mixed" it casts each operand
                   to fp32 before the multiply. Either way the parent-beam sum
                   re-promotes to fp32 inside the kernel.

    NOTE: this materialises all K at once.  Memory-bounded chunking is Task P5.

    The inverse transform uses ``norm="forward"``, which IS the unnormalised
    inverse DFT -- i.e. exactly ``ifft2(..., norm="ortho") * (M1*M2)**0.5``, the
    form this function used to write and the form ``_prism_project_torch`` still
    writes.  Spelling it as one ``norm`` keeps the identical arithmetic while
    dropping two full-tensor passes over ``(Nmodes, K, Bp, M1, M2)`` -- the
    largest tensor in the projection: torch's own ortho rescale, and the
    out-of-place ``* sqrt(M1*M2)``, whose separate output buffer made the peak
    3x the result instead of 2x.  Measured 2.0-2.4x on the step and exactly
    equal (``torch.equal``) wherever ``sqrt(M1*M2)`` is a power of two.
    """
    Nmodes, Bp, M1, M2 = basis_fft.shape
    K = dr.shape[0]
    device = basis_fft.device
    # Build the ramp in complex64 (the FFT precision is chosen below). This keeps
    # the sub-pixel phase accurate; the narrowing to complex32 (if any) happens on
    # the ``shifted`` product just before the ifft2.
    cdtype = torch.complex64

    # Identity ramp: build the real-space basis once and broadcast over K.  See
    # _ramp_is_identity for why the forward is bit-exact.  Note the caller's
    # ``view_as_real(...).contiguous()`` still materialises the K axis, so the win
    # on this route is the K-fold redundant FFT and the ``shifted`` transient --
    # not the final kernel-input buffer.
    if _ramp_is_identity(basis_fft, dr) if ramp_identity is None else ramp_identity:
        base = torch.fft.ifft2(basis_fft.to(cdtype), norm="forward")
        if precision in ("fp16", "mixed"):
            base = base.to(torch.complex32)
        folded = base[:, None, :, :, :]  # (Nmodes, 1, Bp, M1, M2)
        # ``fold_k``: hand the caller the SINGLE k slice and let the kernel do the
        # broadcast with an index (``k_bcast``).  Returning the length-1 axis rather
        # than a slice of the expand matters under autograd: ``expand(...)[:, :1]``
        # has a narrow adjoint that allocates the full (Nmodes, K, Bp, M1, M2) zero
        # buffer this fold exists to avoid, whereas ``unsqueeze``'s adjoint is a
        # reshape.  The caller must pass the matching ``k_bcast`` to the kernel.
        if fold_k:
            return folded
        return folded.expand(Nmodes, K, Bp, M1, M2)

    ramp = _subpixel_ramp(M1, M2, dr, device)  # (K, M1, M2) complex64

    # Build the shifted basis directly in kernel layout (Nmodes, K, Bp, M1, M2):
    # ifft2 only touches the last two dims, so the K/Bp order is free to choose —
    # choosing kernel order here removes the post-ifft permute().contiguous() copy.
    if precision in ("fp16", "mixed"):
        # cuFFT has no half-precision (complex32) plan for many size/batch combos
        # -- CUFFT_INVALID_SIZE at large or non-power-of-2 N (e.g. anisotropic-f
        # tips with N=(2944,1536)). FFT in complex64 and NARROW the result to
        # complex32 afterwards: the transient fp32 FFT is per-K-chunk (bounded),
        # while the returned basis -- and the stored S-matrix -- stay complex32 so
        # the Warp kernel still receives a half-precision basis. For "fp16" the
        # kernel multiplies in fp16; for "mixed" it casts each operand to fp32
        # before the multiply (re-promoting the per-beam product to fp32 either
        # way). The basis FFT being fp32 only makes it MORE accurate.
        shifted = (
            basis_fft.to(cdtype)[:, None, :, :, :] * ramp[None, :, None, :, :]
        )  # (Nmodes, K, Bp, M1, M2) complex64
        out = torch.fft.ifft2(shifted, norm="forward")
        return out.to(torch.complex32)
    return _shift_and_ifft2(basis_fft.to(cdtype), ramp)  # complex64


#: K-tile of :func:`_prism_ramp_mul_kernel`.  The tile is the UNROLL factor, so it
#: is a compile-time constant and the kernel is only eligible when it divides K.
#: The reason is speed: the same body with a *dynamic* inner ``for k in range(K)``
#: loop is slower (1.350x vs 1.675x at r2/medium).  The bits no longer depend on
#: the loop form, because ``cmul_aten`` pins every FMA (it once did: NVRTC
#: contracted the plain product differently in the rolled and unrolled loops).


def _ramp_mul(basis_fft: Tensor, ramp: Tensor) -> Tensor:
    """``basis_fft[:, None] * ramp[None, :, None]``, bit-exact, on the fast route.

    Refused (falling back to the eager broadcast multiply, which is what every
    number above is measured against) when no instantiated tile divides K, when
    the tensors are not the CUDA complex64 the kernel is written for, or when a
    gradient has to flow THROUGH the multiply -- the kernel carries no adjoint,
    by design, so the differentiable spellings stay on the autograd chain.
    """
    if (
        basis_fft.is_cuda
        and basis_fft.dtype == torch.complex64
        and ramp.dtype == torch.complex64
        and _ramp_mul_tile(ramp.shape[0]) is not None
        and not (
            torch.is_grad_enabled() and (basis_fft.requires_grad or ramp.requires_grad)
        )
    ):
        basis_real = torch.view_as_real(basis_fft).contiguous()
        ramp_real = torch.view_as_real(ramp).contiguous()
        if fwd_eager(basis_real, ramp_real):  # the op body, no dispatch
            out_real = _prism_ramp_mul._init_fn(basis_real, ramp_real)
        else:
            out_real = torch.ops.scatterem.prism_ramp_mul(basis_real, ramp_real)
        return torch.view_as_complex(out_real)
    return basis_fft[:, None, :, :, :] * ramp[None, :, None, :, :]


def _shift_and_ifft2(basis_fft: Tensor, ramp: Tensor) -> Tensor:
    """``ifft2(basis_fft[:, None] * ramp[None, :, None], norm="forward")``.

    Takes ``_RampIFFT2``'s fused backward when a gradient is actually going to be
    taken *through basis_fft only*.  A differentiable ``dr`` needs a gradient
    through the ramp as well, which the fused kernel does not produce, so that
    case stays on the eager chain.  Forward arithmetic is identical either way.
    """
    if (
        torch.is_grad_enabled()
        and basis_fft.requires_grad
        and not ramp.requires_grad
        and basis_fft.is_cuda
    ):
        return _RampIFFT2.apply(basis_fft, ramp)
    return torch.fft.ifft2(_ramp_mul(basis_fft, ramp), norm="forward")
