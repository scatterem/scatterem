"""Warp kernels for the Fourier-shift ramp: its adjoint, and its product."""

from __future__ import annotations

import torch
import warp as wp
from torch import Tensor

from scatterem.physics import _warp_interop as wi
from scatterem.physics._aten_cmul_order import aten_cmul_imag_swapped
from scatterem.physics._cmul_warp import cmul_aten


def _make_prism_ramp_adjoint_kernel(T: int):
    # ``T`` and ``ACC`` are baked in at Warp codegen time from this closure, the
    # same construction ``_make_prism_project_bwd_sdt_gather_tiled_kernel`` uses
    # and for the same reason: the per-thread accumulator is a fixed-length Warp
    # vector, so its length is part of the kernel's type and cannot be a kernel
    # argument.
    ACC = wp.types.vector(length=2 * T, dtype=wp.float32)

    @wp.kernel
    def _kernel(
        g: wp.array3d(dtype=wp.vec2),  # (Nmodes*K*Bp, M1, M2) float32 real view
        ramp: wp.array3d(dtype=wp.vec2),  # (K, M1, M2)  # pyright: ignore
        K: int,
        Bp_over_T: int,
        out: wp.array3d(dtype=wp.vec2),  # (Nmodes*Bp, M1, M2)  # pyright: ignore
    ):
        # The adjoint of ``basis_fft[:, None] * ramp[None, :, None]``: multiply by
        # conj(ramp) and reduce over the broadcast K axis, in ONE pass.  Torch
        # spells this as two: a FULL-SIZE ``g * conj(ramp)`` temporary and then a
        # ``sum_to_size`` that reads it back, so ~6.3 GiB of traffic at r2/medium
        # where one fused pass is ~2.27 GiB.
        #
        # One thread owns T consecutive parent beams at the same (y, x), and the
        # K loop is OUTSIDE the tile loop.  That ordering is the whole point: the
        # ramp value for a given (k, y, x) is shared by every (n, b), so with k
        # outer it is loaded K times per thread instead of K*T, and the T loads of
        # ``g`` that follow it are mutually independent, so the tile buys T-way
        # memory-level parallelism on the streaming operand as well.  (An earlier version wrote
        # this loop the other way round -- ramp INSIDE the tile loop -- so the tile
        # bought neither, which is why its sweep read flat at T=2/4/8.  Measured:
        # k-outer is 1.0568x / 1.2521x / 1.3662x / 1.4347x at T=2/4/8/16 over the
        # shipped T=4 t-outer kernel, bit-exact, at r2/medium's real shape.)
        #
        # The SUMMATION ORDER is unchanged -- each output still accumulates over k
        # ascending in its own fp32 register pair with the same two FMA chains --
        # so this is bit-identical to the t-outer spelling, verified elementwise at
        # both registered sizes by a one-off probe.
        nb, y, x = wp.tid()  # pyright: ignore
        n = nb / Bp_over_T
        b0 = (nb - n * Bp_over_T) * T
        Bp = Bp_over_T * T
        acc = ACC()
        for k in range(K):
            rv = ramp[k, y, x]
            base = (n * K + k) * Bp + b0
            for t in range(T):
                gv = g[base + t, y, x]
                # g * conj(ramp)
                acc[2 * t] += gv[0] * rv[0] + gv[1] * rv[1]
                acc[2 * t + 1] += gv[1] * rv[0] - gv[0] * rv[1]
        for t in range(T):
            out[n * Bp + b0 + t, y, x] = wp.vec2(acc[2 * t], acc[2 * t + 1])

    return _kernel


# Largest tile that divides Bp.  A sweep of the k-outer kernel at BOTH registered
# shapes: at r2/medium (Bp=512, K=8, M=256) it rises 1.0568x -> 1.4924x over
# T=2..64 and is 0.975x of a same-process reduction roof at T=64, but at r2/small
# (Bp=128, K=16, M=128) it PEAKS at T=16 (1.2081x) and collapses to 0.41x at
# T=128, where ``Bp // T == 1`` leaves only M1*M2 threads to fill the card.  16 is
# the largest tile that is at or near the optimum on both, and capping the list
# there bounds that failure mode: ``Bp % T == 0`` implies ``Bp // T >= 1``, so the
# thinnest grid this can produce is Nmodes*M1*M2 threads at Bp == 16.
_RAMP_ADJOINT_TILES = (16, 8, 4, 2, 1)

#: One kernel instance per tile, built at import time so a launch never pays Warp
#: codegen.  Distinct closures, distinct Warp kernel keys.
_RAMP_ADJOINT_KERNELS = {
    _T: _make_prism_ramp_adjoint_kernel(_T) for _T in _RAMP_ADJOINT_TILES
}


@torch.library.custom_op("scatterem::prism_ramp_adjoint", mutates_args=())
def _prism_ramp_adjoint(
    g_real: Tensor,  # (Nmodes, K, Bp, M1, M2, 2) float32
    ramp_real: Tensor,  # (K, M1, M2, 2) float32
) -> Tensor:  # (Nmodes, Bp, M1, M2, 2) float32
    Nmodes, K, Bp, M1, M2, _ = g_real.shape
    T = next(t for t in _RAMP_ADJOINT_TILES if Bp % t == 0)
    out = torch.empty(
        (Nmodes * Bp, M1, M2, 2), dtype=torch.float32, device=g_real.device
    )
    wp.launch(
        kernel=_RAMP_ADJOINT_KERNELS[T],
        dim=(Nmodes * (Bp // T), M1, M2),
        inputs=[
            wi.from_torch(
                g_real.reshape(Nmodes * K * Bp, M1, M2, 2),
                dtype=wp.vec2,
                requires_grad=False,
            ),
            wi.from_torch(ramp_real, dtype=wp.vec2, requires_grad=False),
            int(K),
            int(Bp // T),
        ],
        outputs=[wi.from_torch(out, dtype=wp.vec2, requires_grad=False)],
        device=wi.device_from_torch(g_real.device),
    )
    return out.reshape(Nmodes, Bp, M1, M2, 2)


@_prism_ramp_adjoint.register_fake
def _(g_real, ramp_real):
    Nmodes, _K, Bp, M1, M2, _ = g_real.shape
    return g_real.new_empty((Nmodes, Bp, M1, M2, 2), dtype=torch.float32)


#: unrolled one (two one-off probes).
_RAMP_MUL_TILE = 8

#: Tiles the fused ramp multiply is instantiated at, largest first; the launch
#: takes the largest one that divides K (:data:`_RAMP_MUL_KERNELS`).  The first version
#: shipped ONLY the 8, so a caller with ``K % 8 != 0`` silently fell back to the eager
#: broadcast -- which is every chunk of ``prism_project(chunk=...)``, the
#: memory-bounded route this argument exists for.  At ``r2_prism_project/medium``
#: the registered ``project_chunked`` row runs ``chunk = K // 4 = 2`` and paid the
#: eager multiply four times, which is the whole 2.5 ms by which that row exceeds
#: its unchunked twin.
#:
#: 1 is deliberately ABSENT.  The tile's prize is deleting the ``basis`` re-read
#: that torch's output-ordered TensorIterator forces (see the kernel docstring),
#: and at T=1 there is no re-read to delete: measured **0.9365x** at r2/medium's
#: shape, i.e. a LOSS, so an odd K correctly keeps the eager broadcast.
#: Measured at r2/medium (Bp=512, M=256), each BIT-EXACT (``torch.equal``,
#: ndiff = 0 of 268 435 456) against ``basis_fft[:, None] * ramp[None, :, None]``:
#: T=8 1.5993x, T=4 1.4939x, T=2 1.2431x at K=8; T=2 **1.157-1.227x** at K=2.
_RAMP_MUL_TILES = (8, 4, 2)


# ``basis * ramp`` with the bits of ATen's complex64 product, pinned with
# ``__fmaf_rn`` (``cmul_aten``).  The product was written as plain ``a*c - b*d``
# and left to NVRTC's contraction, which matched ATen's sm_86 code but not its
# sm_120 code, where ATen fuses the OTHER imaginary product; ``im_swap`` selects
# the device's order (``scatterem.physics._aten_cmul_order``).
@wp.kernel
def _prism_ramp_mul_kernel(
    basis: wp.array3d(dtype=wp.vec2),  # (Bp, M1, M2)      one mode  # pyright: ignore
    ramp: wp.array3d(dtype=wp.vec2),  # (K, M1, M2)  # pyright: ignore
    out: wp.array4d(dtype=wp.vec2),  # (K, Bp, M1, M2)   one mode  # pyright: ignore
    Bp: int,
    k0: int,
    im_swap: int,  # ATen's imaginary FMA on this device (_aten_cmul_order)
):
    """``out[k0+i, p] = basis[p] * ramp[k0+i]`` for ``i`` in ``0..TILE-1``.

    One thread owns a PIXEL ``(y, x)`` and loops over parent beams, which is the
    whole point.  ``torch``'s TensorIterator can only order its loop by the
    OUTPUT's strides, so on a (Nmodes, K, Bp, M1, M2) output K is the outer axis
    and ``basis_fft`` is streamed from DRAM once per k -- 8 times, 2.147 GB at
    r2/medium where 268 MiB is unique.  Allocating the output as
    (Nmodes, Bp, K, M1, M2) inverts that and is WORSE (6.78 ms vs 6.40): the
    2 GiB store stream evicts L2 in either order, so whichever operand you try to
    keep resident is flushed, and no blocking helps (nb=1..512 swept, best
    1.033x).  Here the TILE ramp values for this pixel go into REGISTERS once and
    ``basis`` is read exactly once, so neither operand depends on the cache at
    all: measured 6.402 -> 3.822 ms = 1.675x, against a 3.017 ms store-only floor
    (``fill_``), i.e. within 0.8 ms of the compulsory traffic.

    BIT-EXACT against the torch spelling (``torch.equal``, max|d| = 0.000e+00 at
    both registered sizes): ``cmul_aten`` pins ATen's FMAs in the device's order.
    The unrolled form is for speed (:data:`_RAMP_MUL_TILE`).
    """
    y, x = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    r0 = ramp[k0 + 0, y, x]
    r1 = ramp[k0 + 1, y, x]
    r2 = ramp[k0 + 2, y, x]
    r3 = ramp[k0 + 3, y, x]
    r4 = ramp[k0 + 4, y, x]
    r5 = ramp[k0 + 5, y, x]
    r6 = ramp[k0 + 6, y, x]
    r7 = ramp[k0 + 7, y, x]
    for p in range(Bp):
        b = basis[p, y, x]
        br = b[0]
        bi = b[1]
        out[k0 + 0, p, y, x] = cmul_aten(br, bi, r0[0], r0[1], im_swap)
        out[k0 + 1, p, y, x] = cmul_aten(br, bi, r1[0], r1[1], im_swap)
        out[k0 + 2, p, y, x] = cmul_aten(br, bi, r2[0], r2[1], im_swap)
        out[k0 + 3, p, y, x] = cmul_aten(br, bi, r3[0], r3[1], im_swap)
        out[k0 + 4, p, y, x] = cmul_aten(br, bi, r4[0], r4[1], im_swap)
        out[k0 + 5, p, y, x] = cmul_aten(br, bi, r5[0], r5[1], im_swap)
        out[k0 + 6, p, y, x] = cmul_aten(br, bi, r6[0], r6[1], im_swap)
        out[k0 + 7, p, y, x] = cmul_aten(br, bi, r7[0], r7[1], im_swap)


def _make_prism_ramp_mul_kernel(T: int):
    """:func:`_prism_ramp_mul_kernel`'s body at an arbitrary tile ``T``.

    Same construction as :func:`_make_prism_ramp_adjoint_kernel`: ``T`` is baked
    in from this closure at Warp codegen time, so ``range(T)`` has a compile-time
    bound and unrolls, like the hand-written 8-fold body (the dynamic ``range(K)``
    spelling is slower, see :data:`_RAMP_MUL_TILE`; the bits are pinned by
    ``cmul_aten`` either way).  The register-held ramp
    values need a fixed-length Warp vector, whose length is part of the kernel's
    type and so cannot be a kernel argument.
    """
    RAMP_T = wp.types.vector(length=2 * T, dtype=wp.float32)

    @wp.kernel
    def _kernel(
        basis: wp.array3d(dtype=wp.vec2),  # (Bp, M1, M2)      one mode
        ramp: wp.array3d(dtype=wp.vec2),  # (K, M1, M2)  # pyright: ignore
        out: wp.array4d(dtype=wp.vec2),  # (K, Bp, M1, M2)   one mode  # pyright: ignore
        Bp: int,
        k0: int,
        im_swap: int,  # ATen's imaginary FMA on this device (_aten_cmul_order)
    ):
        y, x = wp.tid()  # pyright: ignore
        rv = RAMP_T()
        for i in range(T):
            r = ramp[k0 + i, y, x]
            rv[2 * i] = r[0]
            rv[2 * i + 1] = r[1]
        for p in range(Bp):
            b = basis[p, y, x]
            br = b[0]
            bi = b[1]
            for i in range(T):
                out[k0 + i, p, y, x] = cmul_aten(
                    br, bi, rv[2 * i], rv[2 * i + 1], im_swap
                )

    return _kernel


#: One kernel instance per tile, built at import time so a launch never pays Warp
#: codegen.  ``_RAMP_MUL_TILE`` keeps the hand-written body that was measured and
#: shipped, so the ``K % 8 == 0`` route -- every unchunked caller -- is not merely
#: bit-exact but the identical kernel it was before the smaller tiles existed.
_RAMP_MUL_KERNELS = {
    _T: (
        _prism_ramp_mul_kernel
        if _T == _RAMP_MUL_TILE
        else _make_prism_ramp_mul_kernel(_T)
    )
    for _T in _RAMP_MUL_TILES
}


@torch.library.custom_op("scatterem::prism_ramp_mul", mutates_args=())
def _prism_ramp_mul(
    basis_real: Tensor,  # (Nmodes, Bp, M1, M2, 2) float32
    ramp_real: Tensor,  # (K, M1, M2, 2) float32
) -> Tensor:  # (Nmodes, K, Bp, M1, M2, 2) float32
    Nmodes, Bp, M1, M2, _ = basis_real.shape
    K = ramp_real.shape[0]
    im_swap = aten_cmul_imag_swapped(basis_real.device)
    if im_swap is None:
        # The custom op cannot decline, so with ATen's order unknown (a first
        # call under CUDA graph capture) it runs the eager product -- the same
        # bits by definition.
        prod = (
            torch.view_as_complex(basis_real)[:, None]
            * torch.view_as_complex(ramp_real)[None, :, None]
        )
        return torch.view_as_real(prod).contiguous()
    out = torch.empty(
        (Nmodes, K, Bp, M1, M2, 2), dtype=torch.float32, device=basis_real.device
    )
    wdev = wi.device_from_torch(basis_real.device)
    ramp_wp = wi.from_torch(ramp_real, dtype=wp.vec2, requires_grad=False)
    tile = _ramp_mul_tile(K)
    if tile is None:  # callers refuse odd K (see ``_ramp_mul``) before this point
        raise ValueError(f"no ramp-multiply tile for K={K}")
    kernel = _RAMP_MUL_KERNELS[tile]
    for m in range(Nmodes):
        basis_wp = wi.from_torch(basis_real[m], dtype=wp.vec2, requires_grad=False)
        out_wp = wi.from_torch(out[m], dtype=wp.vec2, requires_grad=False)
        for k0 in range(0, K, tile):
            wp.launch(
                kernel=kernel,
                dim=(M1, M2),
                inputs=[basis_wp, ramp_wp, out_wp, int(Bp), int(k0), int(im_swap)],
                device=wdev,
            )
    return out


@_prism_ramp_mul.register_fake
def _(basis_real, ramp_real):
    Nmodes, Bp, M1, M2, _ = basis_real.shape
    K = ramp_real.shape[0]
    return basis_real.new_empty((Nmodes, K, Bp, M1, M2, 2), dtype=torch.float32)


def _ramp_mul_tile(K: int) -> int | None:
    """Largest instantiated tile that divides ``K``, or ``None`` if there is none.

    ``None`` means an odd ``K``, where the only surviving tile would be 1 and 1 is
    a measured loss -- see :data:`_RAMP_MUL_TILES`.
    """
    return next((t for t in _RAMP_MUL_TILES if K % t == 0), None)
