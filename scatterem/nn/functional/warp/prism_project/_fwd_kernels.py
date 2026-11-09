"""Warp forward kernels for the PRISM projection, and their overloads."""

from __future__ import annotations

from typing import Any

import warp as wp

from ._funcs import _fwd_prod


@wp.kernel
def _prism_project_fwd_kernel(
    basis_rs: wp.array(dtype=Any, ndim=4),  # (Nmodes*K, Bp, M1, M2) input dtype
    Sdt: wp.array(dtype=Any, ndim=3),  # (Bp, SY, SX) input dtype  # pyright: ignore
    positions: wp.array(dtype=wp.int32, ndim=2),  # (K, 2)  # pyright: ignore
    K: int,
    SY: int,
    SX: int,
    periodic: int,  # 1 = periodic (cyclic) crop, 0 = non-periodic (zero OOB)
    mul_fp16: int,  # 1 = multiply in input precision; 0 = cast each operand to
    #                 fp32 before the multiply (compute-precision selector; only
    #                 meaningful for vec2h inputs — a no-op for vec2 fp32 inputs)
    bcast: int,  # 1 = ``basis_rs`` carries a SINGLE k slice per mode (leading
    #              axis is Nmodes, not Nmodes*K) and every position reads it;
    #              0 = the leading axis is the full Nmodes*K (historical layout)
    out: wp.array(dtype=wp.vec2, ndim=3),  # FIXED fp32 output (exit waves)
) -> None:
    # Leading axis fuses (Nmodes, K) -> mk; recover k = mk % K (row-major
    # flatten of a (Nmodes, K) shape, so the K index is the fast axis).
    #
    # Generic over the INPUT value type (wp.vec2 fp32 / wp.vec2h fp16): the
    # product runs in the precision selected by ``mul_fp16`` (input fp16 vs
    # per-element fp32 cast), but the parent-beam sum accumulates into an fp32
    # wp.vec2 register and the exit wave is written fp32. Many beams (Bp) reduce
    # onto each output element, so fp16 accumulation would lose precision — the
    # fp32 accumulator preserves it (mirrors transmit.py's fp32 scatter
    # accumulator).
    mk, y, x = wp.tid()  # pyright: ignore
    k = mk % K

    # Basis row for this thread.  With ``bcast == 0`` the basis has one row per
    # (mode, position) and the row IS mk.  With ``bcast == 1`` the K rows of a
    # mode are all equal (identity sub-pixel ramp), so only one is stored and the
    # row is the mode index mk // K — a stride-0 read that every position shares.
    mb = mk
    if bcast == 1:
        mb = mk // K

    py = positions[k, 0]
    px = positions[k, 1]

    acc = wp.vec2(0.0, 0.0)  # fp32 accumulator
    Bp = basis_rs.shape[1]

    if periodic == 1:
        # Cyclic crop: wrap indices modulo (SY, SX).
        gy = (py + y) % SY
        gx = (px + x) % SX
        if gy < 0:
            gy += SY
        if gx < 0:
            gx += SX
        for p in range(Bp):
            acc = acc + _fwd_prod(basis_rs[mb, p, y, x], Sdt[p, gy, gx], mul_fp16)
    else:
        # Non-periodic crop: skip out-of-bounds contributions (contribute 0).
        for p in range(Bp):
            gy = py + y
            gx = px + x
            if gy >= 0 and gy < SY and gx >= 0 and gx < SX:
                acc = acc + _fwd_prod(basis_rs[mb, p, y, x], Sdt[p, gy, gx], mul_fp16)

    out[mk, y, x] = acc


# Number of scan positions one thread of the tiled kernel below owns.  Measured
# sweep on an A6000 at the production shape (Bp=9361, Sdt 512^2, basis 256^2,
# a one-off probe): TK = 2/4/8/16/32 -> 1.84/3.29/5.81/7.46/3.77x over the
# untiled kernel.  16 is the peak; 32 falls off a cliff (64 accumulator registers
# spill, and only 256 blocks are left to fill 84 SMs).
_PRISM_FWD_TILE_K = 16
#: Narrow instantiation of the SAME body, used when there are fewer positions than
#: ``_PRISM_FWD_TILE_K``.  The tail lanes of a TK-wide tile clamp to ``k = K-1`` by
#: design, so at ``K < TK`` a fraction ``(TK-K)/TK`` of the accumulator registers,
#: the index registers and the whole unrolled inner loop is thrown away on lanes
#: whose output is never written back.  The docstring above measured "TK > K costs
#: nothing" against the UNTILED kernel, never against a narrower tile: at ``K = 8``
#: on ``r2/medium/project_fwd_backward`` the 8-wide tile is 1.099x the
#: 16-wide one back-to-back (a 45-arm sweep, every arm ``torch.equal``-gated).
#: The tile width has to be a compile-time constant -- it sets the unrolled loop AND
#: the accumulator vector's length, which is part of the kernel's type -- hence a
#: factory, mirroring ``_make_prism_project_bwd_sdt_gather_tiled_kernel`` below.
_PRISM_FWD_TILE_K_NARROW = 8
#: Kept as module-level names at the production width for the probes that import
#: them; the kernel bodies below take their own from the factory closure.
_PRISM_FWD_TILE_ACC = wp.types.vector(length=2 * _PRISM_FWD_TILE_K, dtype=wp.float32)
_PRISM_FWD_TILE_IDX = wp.types.vector(length=_PRISM_FWD_TILE_K, dtype=wp.int32)


def _make_prism_project_fwd_tiled_kernel(TK: int):
    # ``TK``, ``ACC`` and ``IDX`` are baked in at Warp codegen time from this closure.
    ACC = wp.types.vector(length=2 * TK, dtype=wp.float32)
    IDX = wp.types.vector(length=TK, dtype=wp.int32)

    @wp.kernel
    def _kernel(
        basis_rs: wp.array(dtype=Any, ndim=4),  # (Nmodes, Bp, M1, M2) -- bcast layout
        Sdt: wp.array(dtype=Any, ndim=3),  # (Bp, SY, SX) input dtype  # pyright: ignore
        positions: wp.array(dtype=wp.int32, ndim=2),  # (K, 2)  # pyright: ignore
        K: int,
        ntile: int,  # ceil(K / TK)
        SY: int,
        SX: int,
        periodic: int,  # 1 = periodic (cyclic) crop, 0 = non-periodic (zero OOB)
        mul_fp16: int,  # see _fwd_prod
        out: wp.array(dtype=wp.vec2, ndim=3),  # FIXED fp32 output (exit waves)
    ) -> None:
        # k-TILED variant of _prism_project_fwd_kernel, for the BROADCAST basis only
        # (identity sub-pixel ramp: ``basis_rs`` holds ONE k slice per mode, so
        # ``basis_rs[mode, p, y, x]`` does not depend on k at all).  The untiled
        # kernel still issues that load once per (k, y, x, p), i.e. it re-requests
        # the whole basis K times -- 314 GB per launch at the production shape, which
        # is the entire measured cost.  Here one thread owns TK positions and loads
        # the basis element ONCE for all of them: loads per output element fall from
        # 2 to (TK+1)/TK.
        #
        # BIT-EXACT with the untiled kernel by construction, and gated on that: each
        # output element still accumulates over p in the same order, in the same fp32
        # register, with the same _fwd_prod. Tiling changes which thread does the
        # work, not the arithmetic. (torch.equal, 0.000e+00, at every TK measured.)
        mt, y, x = wp.tid()  # pyright: ignore
        mode = mt // ntile
        k0 = (mt % ntile) * TK

        # Hoist the gather coordinates out of the p loop -- they are k-dependent but
        # p-independent, so recomputing them Bp times is pure waste.  A tail tile
        # CLAMPS to the last valid position rather than branching: the extra lanes
        # recompute a valid output that is simply not written back, and they read
        # addresses another lane is reading anyway (so they stay in cache).
        #
        # The whole periodic/non-periodic split is ONE branch, taken once, so the
        # in-bounds mask ``ok`` exists only on the non-periodic side.  It is dead on
        # the periodic side (every lane is in bounds after the wrap), and building it
        # there would keep TK integer registers live across the entire p loop for
        # nothing -- which is the register pressure the tile width is limited by.
        gy = IDX()
        gx = IDX()
        acc = ACC()  # fp32 accumulators, one complex pair per tile lane
        Bp = basis_rs.shape[1]

        if periodic == 1:
            for i in range(TK):
                k = wp.min(k0 + i, K - 1)
                ry = positions[k, 0] + y
                rx = positions[k, 1] + x
                ry = ry % SY
                rx = rx % SX
                if ry < 0:
                    ry += SY
                if rx < 0:
                    rx += SX
                gy[i] = ry
                gx[i] = rx
            for p in range(Bp):
                b = basis_rs[mode, p, y, x]
                for i in range(TK):
                    a = wp.vec2(acc[2 * i], acc[2 * i + 1])
                    a = a + _fwd_prod(b, Sdt[p, gy[i], gx[i]], mul_fp16)
                    acc[2 * i] = a[0]
                    acc[2 * i + 1] = a[1]
        else:
            ok = IDX()
            for i in range(TK):
                k = wp.min(k0 + i, K - 1)
                ry = positions[k, 0] + y
                rx = positions[k, 1] + x
                if ry >= 0 and ry < SY and rx >= 0 and rx < SX:
                    ok[i] = 1
                else:
                    # Out of bounds contributes zero; park the index at 0 so the
                    # (masked-off) load is always in range.
                    ok[i] = 0
                    ry = 0
                    rx = 0
                gy[i] = ry
                gx[i] = rx
            for p in range(Bp):
                b = basis_rs[mode, p, y, x]
                for i in range(TK):
                    if ok[i] == 1:
                        a = wp.vec2(acc[2 * i], acc[2 * i + 1])
                        a = a + _fwd_prod(b, Sdt[p, gy[i], gx[i]], mul_fp16)
                        acc[2 * i] = a[0]
                        acc[2 * i + 1] = a[1]

        for i in range(TK):
            k = k0 + i
            if k < K:
                out[mode * K + k, y, x] = wp.vec2(acc[2 * i], acc[2 * i + 1])

    return _kernel


#: Production tile (K >= _PRISM_FWD_TILE_K).  Name and value are what the
#: first sweeps established, so the probes that sweep this constant keep working unchanged.
_prism_project_fwd_tiled_kernel = _make_prism_project_fwd_tiled_kernel(
    _PRISM_FWD_TILE_K
)
#: Narrow tile (K < _PRISM_FWD_TILE_K).  Same body, half the lanes.
_prism_project_fwd_tiled_narrow_kernel = _make_prism_project_fwd_tiled_kernel(
    _PRISM_FWD_TILE_K_NARROW
)


# Pre-declare both concrete input instantiations at module load so the kernel is
# not JIT-recompiled per launch (which would break CUDA-graph capture). Overload
# ONLY the Any-typed inputs; the accumulator and output stay fixed at the fp32
# wp.vec2 declared in the signature. wp.vec2 = fp32 (complex64) value type,
# wp.vec2h = fp16 (complex32) value type. Mirrors transmit.py:101-127.
for _vt in (wp.vec2h, wp.vec2):
    for _kern in (
        _prism_project_fwd_kernel,
        _prism_project_fwd_tiled_kernel,
        _prism_project_fwd_tiled_narrow_kernel,
    ):
        wp.overload(
            _kern,
            {
                "basis_rs": wp.array(dtype=_vt, ndim=4),
                "Sdt": wp.array(dtype=_vt, ndim=3),
            },
        )
del _vt, _kern
