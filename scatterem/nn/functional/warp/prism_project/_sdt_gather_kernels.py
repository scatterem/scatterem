"""Warp gather kernels for the S-matrix half of the PRISM adjoint.

Split out of ``_bwd_kernels`` (which keeps the basis adjoint and the atomic
scatter) when both gather kernels got an ``accumulate`` flag, so that the
chunked grad path can add every chunk into ONE shared ``grad_Sdt`` buffer (see
``_chunked``).  With ``accumulate == 0`` each store is unconditional and the
kernels compute exactly what they did before the move.
"""

from __future__ import annotations

import warp as wp

from ._funcs import (
    _cconj,
    _cmul,
)

#: Use the gather route only while the window covers at least this fraction of the
#: S-matrix grid, i.e. while ``SY*SX <= _SDT_GATHER_MAX_WASTE * M1*M2``.  The gather
#: tests all K positions per output element and keeps the ``(M1*M2)/(SY*SX)`` share
#: that lands in the window, so its wasted-iteration factor is the grid
#: oversampling — INDEPENDENT of K (measured: the K=8 and K=32 ratios agree to 2 %).
#: Measured on an A6000 at Bp=512, M=128 (a one-off probe), gather vs the
#: atomic scatter it replaces:  f²=9 → 1.45x, f²=16 → 1.21x / 1.18x (K=8 / K=32),
#: f²=25 → 1.05x (inside this box's noise), f²=36 → 0.90x / 0.74x, f²=64 → 0.73x.
#: 16 is the last threshold where every measured point wins by more than noise.
_SDT_GATHER_MAX_WASTE = 16


@wp.kernel
def _prism_project_bwd_sdt_gather_kernel(
    grad_out: wp.array(dtype=wp.vec2f, ndim=3),  # (Nmodes*K, M1, M2)  # pyright: ignore
    # (Nmodes*K, Bp, M1, M2)
    basis_rs: wp.array(dtype=wp.vec2f, ndim=4),  # pyright: ignore[reportInvalidTypeForm]
    # (K, 2), pre-wrapped (see below)
    positions_n: wp.array(dtype=wp.int32, ndim=2),  # pyright: ignore[reportInvalidTypeForm]
    K: int,
    Nmodes: int,
    SY: int,
    SX: int,
    kb: int,
    accumulate: int,
    grad_Sdt: wp.array(dtype=wp.vec2f, ndim=3),  # (Bp, SY, SX)  # pyright: ignore
) -> None:
    # The transpose of _prism_project_bwd_sdt_kernel: one thread per OUTPUT element
    # (p, gy, gx), gathering the contributions instead of scattering into them.
    #
    # The crop gy = (py_k + y) % SY is a TRANSLATION, so for each k it inverts to a
    # single candidate y = gy - py_k (mod SY), which contributes iff y < M1.  It is
    # the only candidate because y ranges over [0, M1) and M1 <= SY, so two distinct
    # y in the window cannot be congruent mod SY — the caller enforces that bound.
    #
    # Consequences, all of which the scatter pays for and this does not:
    #   * no atomics — every element is accumulated in a register by one thread;
    #   * no zero-fill — every element of grad_Sdt is written exactly once, so the
    #     caller allocates with empty() instead of zeros();
    #   * deterministic — the sum runs in a fixed k-then-m order rather than in
    #     whatever order the atomic units happened to retire.
    #
    # ``positions_n`` must already be wrapped into [0,SY) x [0,SX) so that a single
    # ``+= SY`` restores a negative difference.  A full ``%`` here would put two
    # integer divisions in the inner loop, which is the cost this route is trying
    # to avoid; the caller does it once, on a (K, 2) tensor.
    p, gy, gx = wp.tid()  # pyright: ignore
    M1 = basis_rs.shape[2]
    M2 = basis_rs.shape[3]

    # ``accumulate`` == 1 adds into ``grad_Sdt`` instead of overwriting it, so the
    # chunked grad path can share ONE (Bp, SY, SX) accumulator across chunks rather
    # than materialising one per chunk and letting autograd sum them.
    # ``hit`` then lets a thread whose output element no position in THIS chunk
    # covers skip the read-modify-write entirely -- its contribution is exactly
    # zero.  With ``accumulate`` == 0 the store is unconditional, as it always was.
    acc = wp.vec2(0.0, 0.0)
    hit = int(0)  # noqa: UP018 -- a mutable Warp local; a bare literal is a constant
    for k in range(K):
        y = gy - positions_n[k, 0]
        if y < 0:
            y += SY
        if y < M1:
            x = gx - positions_n[k, 1]
            if x < 0:
                x += SX
            if x < M2:
                hit = 1
                # y, x depend only on k, so the mode loop is nested inside and the
                # index arithmetic is paid once per (k, output element).
                for m in range(Nmodes):
                    mk = m * K + k
                    # ``kb`` == 1: basis stored K-broadcast, so its row is ``m``.
                    mb = mk
                    if kb == 1:
                        mb = m
                    acc = acc + _cmul(grad_out[mk, y, x], _cconj(basis_rs[mb, p, y, x]))

    if accumulate == 0:
        grad_Sdt[p, gy, gx] = acc
    else:
        if hit == 1:
            grad_Sdt[p, gy, gx] = grad_Sdt[p, gy, gx] + acc


#: Number of parent beams one thread of the tiled gather kernel below owns.
#: MEASURED sweep on an A6000 at ``r2/medium`` (Bp=512, Sdt 512^2, basis 256^2,
#: K=8, Nmodes=1, kb=1), a one-off probe in a fresh process per
#: setting (the constant is baked into the Warp module), each launch also timing
#: the UNTILED kernel as an in-process anchor -- anchor 5.43-5.46 ms in all five,
#: so the tiled numbers are comparable across launches:
#:
#:     TP        2       4       8      16      32
#:     ms    3.883   3.357   4.041   4.681   4.782
#:
#: The optimum is an interior one at **4** -- both neighbours are worse, so it is
#: not noise -- and it beats the 8 this kernel originally shipped with (chosen by
#: analogy with the forward TK=16, never swept) by 1.20-1.22x.  Past 4 the
#: 2*TP-float accumulator vector costs more occupancy than the amortised cotangent
#: load buys back; TP=32's 64 accumulator registers are the same spill the forward saw.
#: Keep it a power of two so the tail arithmetic stays cheap.  The optimum is
#: register-file dependent: another card, or a much larger Bp, wants its own sweep.
#:
#: ...AND THE SWEEP ABOVE COVERS ONLY ``kb=1``.  ``kb`` selects between two basis
#: LAYOUTS with an 8x difference in working set -- K-broadcast ``(Nmodes, Bp, M1,
#: M2)`` = 256 MiB at ``r2/medium`` versus general ``(Nmodes*K, Bp, M1, M2)`` =
#: 2048 MiB -- and it is a *runtime* argument, so one constant cannot serve both.
#: Re-running the same sweep at ``kb=0`` (same probe, same anchor protocol) gives a
#: monotonically different curve with its optimum 8x higher:
#:
#:     TP        2       4       8      16      32      64
#:     kb=1  3.927   3.503*  4.293   4.993   5.144   9.480
#:     kb=0  6.691   5.931   5.521   5.305   5.227*  9.378
#:
#: Both curves are monotone up to their peak and worse at both neighbours of it, so
#: neither optimum is a noise pick; both collapse at 64 on the same register spill.
#: The bigger working set is why: at ``kb=0`` the basis rows a tile needs are 8x
#: further apart, so a wider tile is what amortises the cotangent load and keeps
#: enough loads in flight, and it can afford the registers because far less of the
#: basis stays resident anyway.
#:
#: So the kernel is instantiated TWICE, from ONE body, and the launch dispatches on
#: the same ``kb`` test it already computes.  The tile count has to be a
#: compile-time constant (it sets the unrolled tile loop AND the accumulator
#: vector's length, which is part of the kernel's type), hence a factory rather
#: than a kernel argument.  Both instantiations carry the same Warp kernel key and
#: coexist in this module; Warp 1.13 keeps them distinct (verified by launching both
#: and checking each against its own tile size, a one-off probe).
_PRISM_BWD_SDT_TILE_P = 4  # kb == 1, K-broadcast basis
_PRISM_BWD_SDT_TILE_P_KB0 = 32  # kb == 0, general basis


def _make_prism_project_bwd_sdt_gather_tiled_kernel(TILE_P: int):
    # ``TILE_P`` and ``ACC`` are baked in at Warp codegen time from this closure.
    ACC = wp.types.vector(length=2 * TILE_P, dtype=wp.float32)

    @wp.kernel
    def _kernel(
        # (Nmodes*K, M1, M2)
        grad_out: wp.array(dtype=wp.vec2f, ndim=3),  # pyright: ignore[reportInvalidTypeForm]
        # (Nmodes*Kb, Bp, M1, M2)
        basis_rs: wp.array(dtype=wp.vec2f, ndim=4),  # pyright: ignore[reportInvalidTypeForm]
        # (K, 2), pre-wrapped
        positions_n: wp.array(dtype=wp.int32, ndim=2),  # pyright: ignore[reportInvalidTypeForm]
        K: int,
        Nmodes: int,
        SY: int,
        SX: int,
        kb: int,
        Bp: int,
        accumulate: int,
        grad_Sdt: wp.array(dtype=wp.vec2f, ndim=3),  # (Bp, SY, SX)  # pyright: ignore
    ) -> None:
        # p-TILED variant of _prism_project_bwd_sdt_gather_kernel.  Everything the
        # untiled kernel does inside its (k, m) loop EXCEPT the basis load and the
        # accumulate is independent of ``p``:
        #   * the window inversion y = gy - py_k (mod SY) and its two bounds tests --
        #     3/4 of the iterations do nothing else, since only M1/SY * M2/SX of the
        #     candidates land in the window;
        #   * the ``grad_out[mk, y, x]`` load itself.
        # The untiled kernel repeats all of it once per parent beam, i.e. Bp times
        # (512 at r2/medium, thousands in production).  Here one thread owns
        # TILE_P consecutive beams: the index arithmetic and the cotangent load are
        # paid ONCE for the whole tile, and loads per accumulate fall from 2 to
        # (TP+1)/TP.
        #
        # The SUMMATION ORDER is unchanged: each output element still accumulates over
        # (k, then m) ascending, in its own fp32 register pair, with the same
        # _cmul/_cconj.  Tiling changes which thread does the work, not the order of
        # the additions.  It is NOT bit-identical all the same -- measured ~1e-7
        # relative, i.e. 1-2 fp32 ulp, because the accumulator here lives in a vector
        # register pair and the compiler contracts the multiply-add differently than in
        # the untiled kernel's straight-line `acc = acc + _cmul(...)`.  The tiled
        # FORWARD kernel is bit-identical only because _fwd_prod's explicit float32
        # casts block that contraction on both sides.  The disagreement is also
        # measured to be tile-size INDEPENDENT (1.46e-07 at every TP from 2 to 32),
        # so the second instantiation adds no error the first did not already have.
        #
        # Since this is a HAND-WRITTEN ADJOINT, the gate is therefore complex128 truth
        # rather than torch.equal: a benchmark script builds grad_Sdt
        # in fp64 from the definition and requires EACH instantiation to be no further
        # from it than the untiled kernel is, at both layouts and at Bp values that
        # are tail cases for TILE_P = 4 and for TILE_P = 32.
        pt, gy, gx = wp.tid()  # pyright: ignore
        p0 = pt * TILE_P
        M1 = basis_rs.shape[2]
        M2 = basis_rs.shape[3]

        # ``accumulate`` == 1: add into ``grad_Sdt``, and skip the store entirely
        # for an element no position in THIS chunk covers (see the untiled kernel).
        acc = ACC()
        hit = int(0)  # noqa: UP018 -- a mutable Warp local
        for k in range(K):
            y = gy - positions_n[k, 0]
            if y < 0:
                y += SY
            if y < M1:
                x = gx - positions_n[k, 1]
                if x < 0:
                    x += SX
                if x < M2:
                    hit = 1
                    for m in range(Nmodes):
                        mk = m * K + k
                        # ``kb`` == 1: basis stored K-broadcast, so its row is ``m``.
                        mb = mk
                        if kb == 1:
                            mb = m
                        g = grad_out[mk, y, x]  # ONE load for the whole tile
                        for i in range(TILE_P):
                            # A tail lane CLAMPS to the last valid beam rather than
                            # branching: it recomputes a valid accumulator that is
                            # simply not written back, and it reads an address another
                            # lane is reading anyway (so it stays in cache).
                            p = wp.min(p0 + i, Bp - 1)
                            a = wp.vec2(acc[2 * i], acc[2 * i + 1])
                            a = a + _cmul(g, _cconj(basis_rs[mb, p, y, x]))
                            acc[2 * i] = a[0]
                            acc[2 * i + 1] = a[1]

        store = int(1)  # noqa: UP018 -- a mutable Warp local
        if accumulate == 1:
            if hit == 0:
                store = 0
        if store == 1:
            for i in range(TILE_P):
                p = p0 + i
                if p < Bp:
                    v = wp.vec2(acc[2 * i], acc[2 * i + 1])
                    if accumulate == 1:
                        v = v + grad_Sdt[p, gy, gx]
                    grad_Sdt[p, gy, gx] = v

    return _kernel


#: ``kb == 1`` (K-broadcast basis).  Name and value are what the sweeps established, so
#: the probes that sweep this constant keep working unchanged.
_prism_project_bwd_sdt_gather_tiled_kernel = (
    _make_prism_project_bwd_sdt_gather_tiled_kernel(_PRISM_BWD_SDT_TILE_P)
)
#: ``kb == 0`` (general basis).  Same body, 8x the tile.
_prism_project_bwd_sdt_gather_tiled_kb0_kernel = (
    _make_prism_project_bwd_sdt_gather_tiled_kernel(_PRISM_BWD_SDT_TILE_P_KB0)
)
