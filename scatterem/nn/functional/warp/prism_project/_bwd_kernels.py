"""Warp backward kernels: the basis adjoint and the S-matrix scatter-add."""

from __future__ import annotations

import warp as wp

from ._funcs import (
    _cconj,
    _cmul,
)

# ===========================================================================
# Fused Warp backward (Task P6)
# ===========================================================================
#
# Adjoint of the forward sum (PyTorch complex convention, out = a*b ->
# grad_a = grad_out*conj(b), grad_b = grad_out*conj(a); see transmit.py):
#
#   exit[mk,y,x] = Σ_p basis_rs[mk,p,y,x] · Sdt[p, gy, gx]
#       with gy = (py_k+y)%SY, gx = (px_k+x)%SX
#
#   grad_basis_rs[mk,p,y,x] = grad_exit[mk,y,x] · conj(Sdt[p, gy, gx])
#   grad_Sdt[p, gy, gx]    += grad_exit[mk,y,x] · conj(basis_rs[mk,p,y,x])
#                                                     (scatter / atomic-add
#                                                      through the cyclic crop)
#
# Only this Warp op needs a hand adjoint: torch autograd then propagates
# grad_basis_rs back through the ramp + ifft2 to grad_basis_fft and grad_dr.


@wp.kernel
def _prism_project_bwd_basis_kernel(
    grad_out: wp.array(dtype=wp.vec2f, ndim=3),  # (Nmodes*K, M1, M2)  # pyright: ignore
    Sdt: wp.array(dtype=wp.vec2f, ndim=3),  # (Bp, SY, SX)  # pyright: ignore
    positions: wp.array(dtype=wp.int32, ndim=2),  # (K, 2)  # pyright: ignore
    K: int,
    SY: int,
    SX: int,
    grad_basis_rs: wp.array(dtype=wp.vec2f, ndim=4),  # (Nmodes*K, Bp, M1, M2)
) -> None:
    # One thread per (mk, p, y, x): pure gather, no atomics needed since each
    # grad_basis_rs element is written exactly once.
    mk, p, yx = wp.tid()  # pyright: ignore
    M2 = grad_out.shape[2]
    y = yx // M2
    x = yx % M2
    k = mk % K

    py = positions[k, 0]
    px = positions[k, 1]
    gy = (py + y) % SY
    gx = (px + x) % SX
    if gy < 0:
        gy += SY
    if gx < 0:
        gx += SX

    grad_basis_rs[mk, p, y, x] = _cmul(grad_out[mk, y, x], _cconj(Sdt[p, gy, gx]))


@wp.kernel
def _prism_project_bwd_basis_bcast_kernel(
    grad_out: wp.array(dtype=wp.vec2f, ndim=3),  # (Nmodes*K, M1, M2)  # pyright: ignore
    Sdt: wp.array(dtype=wp.vec2f, ndim=3),  # (Bp, SY, SX)  # pyright: ignore
    positions: wp.array(dtype=wp.int32, ndim=2),  # (K, 2)  # pyright: ignore
    K: int,
    SY: int,
    SX: int,
    grad_basis_rs: wp.array(dtype=wp.vec2f, ndim=4),  # (Nmodes*1, Bp, M1, M2)
) -> None:
    # Adjoint of the K-BROADCAST basis layout (``k_bcast``): the forward read one
    # stored row for every position, so this reduces all K cotangents back into
    # that single row.  One thread per OUTPUT element (m, p, y, x), looping k in a
    # register -- the same shape of trade as for grad_Sdt:
    #   * no atomics, even though K threads' worth of work lands on one element;
    #   * no K-sized (Nmodes*K, Bp, M1, M2) buffer and no separate torch sum over
    #     K afterwards -- the reduction happens where the values are produced;
    #   * deterministic: the sum runs in ascending k on one thread, so it does not
    #     depend on the order atomic units happen to retire.
    # ``grad_out`` still carries all K rows; only the basis is folded.
    m, p, yx = wp.tid()  # pyright: ignore
    M2 = grad_out.shape[2]
    y = yx // M2
    x = yx % M2

    acc = wp.vec2(0.0, 0.0)
    for k in range(K):
        py = positions[k, 0]
        px = positions[k, 1]
        gy = (py + y) % SY
        gx = (px + x) % SX
        if gy < 0:
            gy += SY
        if gx < 0:
            gx += SX
        acc = acc + _cmul(grad_out[m * K + k, y, x], _cconj(Sdt[p, gy, gx]))

    grad_basis_rs[m, p, y, x] = acc


#: Number of parent beams one thread of the tiled bcast kernel below owns.
#: A first sweep varied this alone (a one-off probe, tiled vs the untiled kernel above:
#: TP = 4 / 8 / 16 -> 1.187x / 1.208x / 1.203x) and read 8 as the plateau.  A second
#: sweep varied it JOINTLY with the two knobs the first held fixed -- the k-unroll width and
#: ``block_dim`` -- and the joint optimum is a different corner:
#:
#:     TP=8  k=1 bd=256 (first point)   1.0000x     72 regs, 24 warps/SM
#:     TP=8  k=1 bd=512                 0.9703x     72 regs, 16 warps/SM
#:     TP=4  k=1 bd=256                 0.9807x     56 regs, 32 warps/SM
#:     TP=4  k=2 bd=256                 1.0426x     56 regs, 32 warps/SM
#:     TP=4  k=2 bd=512                 1.0588x     56 regs, 32 warps/SM  <- shipped
#:     TP=2  k=4 bd=256 / TP=16 k=1     0.9287x / 0.9979x
#:
#: (r2/medium, Bp=512, Sdt 512^2, basis 256^2, K=8, Nmodes=1, A6000/sm_86;
#: a benchmark script, six alternating in-process cycles, every arm
#: ``torch.equal`` against the shipped kernel's own output.)  **No single knob is
#: positive on its own** -- narrowing the tile alone loses 2 %, widening the block
#: alone loses 3 % -- which is why three separate runs sweeping one axis at a time
#: found nothing.  The mechanism is that TP is simultaneously the cotangent-reuse
#: width (the first reading) AND the number of independent ``Sdt`` loads a thread has
#: in flight per k step, and the accumulator is ``2*TP`` floats so it is also the
#: register bill.  Halving TP buys the 64-register step (24 -> 32 resident warps)
#: and the k-unroll buys the loads in flight back; ``block_dim`` then pays because
#: 512 consecutive ``yx`` is two whole output rows per block.
#:
#: NOT a traffic fix, and the second sweep measured that too: ``Sdt`` loads are ``Bp*M1*M2*K``
#: independently of TP, so if this kernel were bound by the 2x request redundancy
#: on ``Sdt`` then SMALL TP would win, because the
#: L2 working set needed to capture that redundancy is ``~TP * 1.9 MiB`` against a
#: 6 MiB L2.  TP = 2 measures **0.9287x**.  The redundancy is not what this kernel
#: is waiting for.
#:
#: (This is a SEPARATE optimum from ``_PRISM_BWD_SDT_TILE_P``: different kernel,
#: different per-thread state.)
_PRISM_BWD_BASIS_TILE_P = 4
_PRISM_BWD_BASIS_TILE_ACC = wp.types.vector(
    length=2 * _PRISM_BWD_BASIS_TILE_P, dtype=wp.float32
)

#: How many k iterations of the tiled bcast kernel are unrolled into one
#: predicate-free body.  This is the knob that makes ``TP = 4`` pay: it restores
#: the ``2*TP = 8`` independent in-flight ``Sdt`` loads that ``TP = 8`` had, at
#: ``TP = 4``'s register cost.  ``K`` is a runtime argument and need not be a
#: multiple of this, so the kernel splits the loop and pays the remainder in an
#: epilogue -- an ``if k < K`` INSIDE the unrolled body instead measures 1.0010x
#: against 1.0588x at identical registers and occupancy, because the branch is
#: what stops the compiler hoisting both iterations' loads.
_PRISM_BWD_BASIS_K_UNROLL = 2

#: Threads per block for the tiled bcast kernel.  Not warp's default (256).
#: Worth 1.0588x vs 1.0426x at ``TP = 4``, and **0.9703x** on the shipped
#: ``TP = 8`` body -- a sweep of block_dim on that body alone and correctly found
#: every value at or below 1.00x.
_PRISM_BWD_BASIS_BLOCK_DIM = 512


@wp.kernel
def _prism_project_bwd_basis_bcast_tiled_kernel(
    grad_out: wp.array(dtype=wp.vec2f, ndim=3),  # (Nmodes*K, M1, M2)  # pyright: ignore
    Sdt: wp.array(dtype=wp.vec2f, ndim=3),  # (Bp, SY, SX)  # pyright: ignore
    positions: wp.array(dtype=wp.int32, ndim=2),  # (K, 2)  # pyright: ignore
    K: int,
    SY: int,
    SX: int,
    Bp: int,
    grad_basis_rs: wp.array(dtype=wp.vec2f, ndim=4),  # (Nmodes*1, Bp, M1, M2)
) -> None:
    # p-TILED variant of _prism_project_bwd_basis_bcast_kernel.  That kernel owns one
    # output (m, p, y, x) and loops k, issuing TWO global loads per accumulate step:
    #   * ``Sdt[p, gy, gx]``            -- depends on p;
    #   * ``grad_out[m*K + k, y, x]``   -- does NOT depend on p, so the untiled
    #     kernel re-requests the whole cotangent once per parent beam (512 times at
    #     r2/medium, thousands in production).
    # Here one thread owns _PRISM_BWD_BASIS_TILE_P consecutive beams and loads the
    # cotangent ONCE for all of them: loads per accumulate fall from 2 to (TP+1)/TP,
    # i.e. requested traffic 4.56 -> 2.69 GB at r2/medium, and the crop arithmetic is
    # paid once per tile instead of once per beam.
    #
    # MEASURED MECHANISM, and it is not the one the traffic count suggests: the
    # cotangent is only 4 MiB against this card's 6 MiB L2, so those redundant
    # requests were already L2 HITS -- what they cost was L2->SM bandwidth and load
    # slots, not DRAM bandwidth.  Removing 7/8 of them buys 1.21x (4.13 -> 3.41 ms),
    # where the DRAM-traffic view would have predicted ~1.7x.  The crop arithmetic
    # that looked expensive -- two runtime integer divisions per accumulate step, 268 M
    # of them, against the FORWARD kernel's 0.5 M for the same maths -- turns out to
    # cost nothing measurable: replacing both with a host pre-wrap and a single
    # compare-subtract (the trick _prism_project_bwd_sdt_gather_kernel uses) measured
    # 1.008x / 1.011x, i.e. inside noise, both alone and on top of this tiling
    # (a one-off probe v1/v2/v4).  So the index arithmetic is left exactly as
    # the untiled kernel has it.
    #
    # NOT bit-identical with the untiled kernel: ~1.7e-7 relative, 0.5-1.1 fp32 ulp.
    # The summation order is unchanged (each output still sums k ascending in its own
    # fp32 register pair with the same _cmul/_cconj), so it is not reassociation --
    # The same effect was documented on the tiled grad_Sdt gather and attributed it to
    # multiply-add contraction in the vector accumulator.  That attribution is NOT
    # confirmed: two rewrites that should defeat contraction -- the accumulate as two
    # scalar adds through a named product, and both operands pushed through explicit
    # wp.float32 casts the way _fwd_prod does -- leave the difference unchanged at
    # 1.7e-7 (a one-off probe v5/v6).  Whatever the cause, it is not
    # addressable at the expression level, so the gate for this hand-written
    # adjoint is complex128 truth, not torch.equal:
    # a benchmark script builds grad_basis_rs in fp64 from the
    # definition and requires the tiled kernel to be no further from it than the
    # untiled kernel is.
    m, pt, yx = wp.tid()  # pyright: ignore
    M2 = grad_out.shape[2]
    y = yx // M2
    x = yx % M2
    p0 = pt * _PRISM_BWD_BASIS_TILE_P

    acc = _PRISM_BWD_BASIS_TILE_ACC()
    # the k loop is SPLIT, not predicated.  The main body runs whole
    # ``_PRISM_BWD_BASIS_K_UNROLL``-wide groups with a compile-time inner trip
    # count and NO branch, so the compiler can hoist all ``K_UNROLL * TILE_P``
    # independent ``Sdt`` loads of a group above the arithmetic; the remainder
    # (0 .. K_UNROLL-1 iterations, and always 0 at every registered size) is paid
    # once afterwards.  Per-output summation order is unchanged -- still k
    # ascending through the same fp32 register pair -- so this is bit-identical
    # to the un-unrolled body, verified with ``torch.equal``, not asserted.
    nkb = K // _PRISM_BWD_BASIS_K_UNROLL
    for kb in range(nkb):
        for u in range(_PRISM_BWD_BASIS_K_UNROLL):
            k = kb * _PRISM_BWD_BASIS_K_UNROLL + u
            py = positions[k, 0]
            px = positions[k, 1]
            gy = (py + y) % SY
            gx = (px + x) % SX
            if gy < 0:
                gy += SY
            if gx < 0:
                gx += SX
            g = grad_out[m * K + k, y, x]  # ONE load for the whole tile
            for i in range(_PRISM_BWD_BASIS_TILE_P):
                # A tail lane CLAMPS to the last valid beam rather than branching:
                # it accumulates a valid value that is simply not written back, and
                # it reads an address a sibling lane reads anyway (so it stays in
                # cache).
                p = wp.min(p0 + i, Bp - 1)
                a = wp.vec2(acc[2 * i], acc[2 * i + 1])
                a = a + _cmul(g, _cconj(Sdt[p, gy, gx]))
                acc[2 * i] = a[0]
                acc[2 * i + 1] = a[1]

    for k in range(nkb * _PRISM_BWD_BASIS_K_UNROLL, K):  # odd-K epilogue
        py = positions[k, 0]
        px = positions[k, 1]
        gy = (py + y) % SY
        gx = (px + x) % SX
        if gy < 0:
            gy += SY
        if gx < 0:
            gx += SX
        g = grad_out[m * K + k, y, x]
        for i in range(_PRISM_BWD_BASIS_TILE_P):
            p = wp.min(p0 + i, Bp - 1)
            a = wp.vec2(acc[2 * i], acc[2 * i + 1])
            a = a + _cmul(g, _cconj(Sdt[p, gy, gx]))
            acc[2 * i] = a[0]
            acc[2 * i + 1] = a[1]

    for i in range(_PRISM_BWD_BASIS_TILE_P):
        p = p0 + i
        if p < Bp:
            grad_basis_rs[m, p, y, x] = wp.vec2(acc[2 * i], acc[2 * i + 1])


@wp.kernel
def _prism_project_bwd_sdt_kernel(
    grad_out: wp.array(dtype=wp.vec2f, ndim=3),  # (Nmodes*K, M1, M2)  # pyright: ignore
    basis_rs: wp.array(dtype=wp.vec2f, ndim=4),  # (Nmodes*Kb, Bp, M1, M2)
    positions: wp.array(dtype=wp.int32, ndim=2),  # (K, 2)  # pyright: ignore
    K: int,
    SY: int,
    SX: int,
    kb: int,
    grad_Sdt: wp.array(dtype=wp.vec2f, ndim=3),  # (Bp, SY, SX)  # pyright: ignore
) -> None:
    # One thread per (mk, p, y, x): scatter-add into grad_Sdt through the cyclic
    # crop — many (mk, y, x) map to the same (gy, gx), so atomic_add is required.
    #
    # ``kb`` == 1 means the basis is stored K-broadcast (one row per mode), so the
    # basis row is ``mk // K`` instead of ``mk``.  It is a launch-wide constant, so
    # the branch is uniform; with kb == 0 the index arithmetic is unchanged.
    mk, p, yx = wp.tid()  # pyright: ignore
    M2 = grad_out.shape[2]
    y = yx // M2
    x = yx % M2
    k = mk % K
    mb = mk
    if kb == 1:
        mb = mk // K

    py = positions[k, 0]
    px = positions[k, 1]
    gy = (py + y) % SY
    gx = (px + x) % SX
    if gy < 0:
        gy += SY
    if gx < 0:
        gx += SX

    g = _cmul(grad_out[mk, y, x], _cconj(basis_rs[mb, p, y, x]))
    wp.atomic_add(grad_Sdt, p, gy, gx, g)
