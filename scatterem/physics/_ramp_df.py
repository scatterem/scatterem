"""The Fourier-shift ramp as a certified float-float Warp kernel.

Two kernels sit beside ``_ramp_fused._ramp_phase_c64_kernel`` and give the same
complex64 bits with less float64 work:

* ``_ramp_phase_c64_df_kernel`` replaces the float64 ``sincos`` with a
  certified float-float ``sincos`` on the FP32 units.  GA102 runs
  float64 at 1/64 of the float32 rate, and ``sincos`` was 56.6 % of
  a probe-build benchmark.  We measured the kernel at
  0.247 -> 0.118 ms per call and found 0 of 1 548 161 928 words different
  from the float64 kernel.
* ``_ramp_mul_c64_df_kernel`` also applies ``fourier_shift_stack``'s broadcast
  ``wq * phase`` before the store.  This removes one full read and write
  pass over the ``(P, ny, nx)`` phase.  We measured the process peak at
  545.0 -> 529.0 MiB and found 0 differing words over 84 cases.

The Warp source below is carried verbatim from the lab.  Do not simplify an
``fma`` or a constant: the bits depend on each one.

Routing (``simulation/_backends.py``): ``physics.propagation`` asks
:func:`ramp_phase_df_supported` (row ``eels_ramp``, impl ``"warp_df"``) and
:func:`ramp_wq_fold_supported` (row ``eels_ramp_wq_fold``) directly.  Layer 0
may not import the registry, so the caller keeps the predicates.
"""

from __future__ import annotations

import torch
import warp as wp
from torch import Tensor

from scatterem.physics import _warp_interop as wi
from scatterem.physics._aten_cmul_order import (
    aten_cmul_imag_swapped,
    aten_cmul_swap_flag,
)
from scatterem.physics._cmul_warp import cmul_aten
from scatterem.physics._ramp_fused import (
    _dadd,
    _dmul,
    _launch_ramp_phase,
    _wp_freqs,
    ramp_phase_fused_supported,
)
from scatterem.physics._warp_init import (
    ensure_warp_initialized as _ensure_warp_initialized,
)

# The same complex64 bits as ``f32(sincos(u))`` with the float64 work cut to an
# exact/once-rounded prefix.  GA102 runs float64 at 1/64 of the
# float32 rate and ``sincos`` costs ~45 float64 instructions here, so:
#
# 1. ``r = fma(-k, P2, fma(-k, P1, u))`` is the two-constant Cody-Waite residual
#    of ``u`` mod pi/2 (``P1+P2`` = pi/2 to 2^-107; the inner ``fma`` is exact
#    because ``u`` and ``k*P1`` are multiples of 2^-53 that differ by < 1), so
#    ``r`` carries relative error <= 2^-53 plus <= 2^-88 absolute, and keeps
#    RELATIVE accuracy when ``u`` sits next to a multiple of pi/2 -- the
#    quarter-integer ``t`` of an integer scan raster that made an absolute
#    certificate fall back in every warp.
# 2. ``sin r`` / ``cos r`` are Taylor series to degree 17/16 (truncation
#    < 2^-46 for |r| <= 0.9) evaluated as a float-float on FP32 units: the
#    terms above 2^-24 in double-float, the rest in float.  The quadrant is an
#    exact swap/negate, so each output is +-sin r or +-cos r and its error is
#    RELATIVE (~2^-42).
# 3. The result ``V`` is CERTIFIED: CUDA's ``sincos`` is within 2 ulp(double)
#    of the true value, so with ``E = 2^-36|V| + 2^-80`` (>= 16x both errors)
#    every value in ``[V-E, V+E]`` rounds to the same float32 iff the strict
#    half-gap test below holds -- then that float32 IS the direct kernel's.
#    Anything else (test fails, |u| >= 2^19, NaN/inf, tiny |u|, |r| > 0.9)
#    takes the verbatim ``sincos`` path.  ``u == 0`` gives ``(1, u)`` exactly.
_DF_SINCOS_SRC = """
struct F2 { float h, l; };
auto fts = [](float a, float b) {
    float s = __fadd_rn(a, b);
    return F2{s, __fsub_rn(b, __fsub_rn(s, a))};
};
auto ts = [](float a, float b) {
    float s = __fadd_rn(a, b);
    float bb = __fsub_rn(s, a);
    return F2{s, __fadd_rn(__fsub_rn(a, __fsub_rn(s, bb)), __fsub_rn(b, bb))};
};
auto mul = [&](F2 a, F2 b) {
    float p = __fmul_rn(a.h, b.h);
    float e = __fmaf_rn(a.h, b.h, -p);
    e = __fmaf_rn(a.h, b.l, e);
    e = __fmaf_rn(a.l, b.h, e);
    return fts(p, e);
};
auto add = [&](F2 a, F2 b) {
    F2 s = ts(a.h, b.h);
    return fts(s.h, __fadd_rn(s.l, __fadd_rn(a.l, b.l)));
};
auto addf = [&](F2 a, float b) {
    F2 s = ts(a.h, b);
    return fts(s.h, __fadd_rn(s.l, a.l));
};
auto K = [](unsigned h, unsigned l) {
    return F2{__uint_as_float(h), __uint_as_float(l)};
};
auto cert = [&](F2 v) {
    float ay = fabsf(v.h);
    if (!(ay > 1.0e-30f)) return false;
    float d = v.h < 0.0f ? -v.l : v.l;
    float e = __fmaf_rn(ay, 1.4551915228366852e-11f, 8.271806125530277e-25f);
    unsigned b = __float_as_uint(ay);
    float up = __fsub_rn(__uint_as_float(b + 1u), ay);
    float dn = __fsub_rn(ay, __uint_as_float(b - 1u));
    return __fadd_rn(d, e) < 0.5f * up && __fsub_rn(d, e) > -0.5f * dn;
};
float oc = 0.0f, os = 0.0f;
bool ok = false;
float uf = __double2float_rn(u);
if (u == 0.0) {
    oc = 1.0f;
    os = uf;
    ok = true;
} else if (fabsf(uf) < 524288.0f && fabsf(uf) > 1.0e-18f) {
    float kf = rintf(__fmul_rn(uf, __uint_as_float(0x3f22f983u)));
    double kd = (double)kf;
    double r = __fma_rn(-kd, 6.123233995736766e-17, __fma_rn(-kd, 1.5707963267948966, u));
    float rh = __double2float_rn(r);
    float rl = __double2float_rn(__dsub_rn(r, (double)rh));
    if (fabsf(rh) <= 0.9f) {
        F2 x = F2{rh, rl};
        F2 z = mul(x, x);
        float zh = z.h;
        float q1 = __fmaf_rn(zh, __fmaf_rn(zh, __fmaf_rn(zh,
            __uint_as_float(0x274a963cu), __uint_as_float(0xab573f9fu)),
            __uint_as_float(0x2f309231u)), __uint_as_float(0xb2d7322bu));
        F2 w = addf(K(0x3638ef1du, 0x292ad8e6u), __fmul_rn(zh, q1));
        w = add(K(0xb9500d01u, 0x2c3fcbfdu), mul(z, w));
        w = add(K(0x3c088889u, 0xafeeeeefu), mul(z, w));
        w = add(K(0xbe2aaaabu, 0x31aaaaabu), mul(z, w));
        F2 s = add(x, mul(x, mul(z, w)));
        float q2 = __fmaf_rn(zh, __fmaf_rn(zh, __fmaf_rn(zh,
            __uint_as_float(0x29573f9fu), __uint_as_float(0xad49cba5u)),
            __uint_as_float(0x310f76c7u)), __uint_as_float(0xb493f27eu));
        F2 v = addf(K(0x37d00d01u, 0xaabfcbfdu), __fmul_rn(zh, q2));
        v = add(K(0xbab60b61u, 0x2e13e93fu), mul(z, v));
        v = add(K(0x3d2aaaabu, 0xb0aaaaabu), mul(z, v));
        v = add(F2{-0.5f, 0.0f}, mul(z, v));
        F2 c = add(F2{1.0f, 0.0f}, mul(z, v));
        int q = ((int)kf) & 3;
        F2 rc = (q == 0) ? c : (q == 1) ? F2{-s.h, -s.l} : (q == 2) ? F2{-c.h, -c.l} : s;
        F2 rs = (q == 0) ? s : (q == 1) ? c : (q == 2) ? F2{-s.h, -s.l} : F2{-c.h, -c.l};
        if (cert(rc) && cert(rs)) {
            oc = rc.h;
            os = rs.h;
            ok = true;
        }
    }
}
if (!ok) {
    double2 r2;
    sincos(u, &r2.y, &r2.x);
    oc = __double2float_rn(r2.x);
    os = __double2float_rn(r2.y);
}
return wp::vec_t<2, wp::float32>(oc, os);
"""

# ``wq * phase`` with the bits of torch's complex64 product:
# nvcc contracts ``c10::complex<float>``'s ``operator*`` into exactly this FMA
# form -- measured 0 of 4 194 304 words; no contraction differs on 995 520 and
# the mirrored FMA on 1 300 508.  ``a`` is ``wq`` (the left operand), ``b`` the
# phase.  The imaginary FMA's order is the device's (``cmul_aten``, ``im_swap``).


@wp.func_native(_DF_SINCOS_SRC)
def _df_sincos_f32(u: wp.float64) -> wp.vec2f: ...


@wp.kernel
def _ramp_phase_c64_df_kernel(
    ky: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
    kx: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
    sy: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
    sx: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
    minus_two_pi: wp.float64,
    out: wp.array4d(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
):
    p, i, j = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    t = _dadd(_dmul(ky[i], sy[p]), _dmul(kx[j], sx[p]))
    u = _dmul(minus_two_pi, t)
    cs = _df_sincos_f32(u)
    out[p, i, j, 0] = cs[0]
    out[p, i, j, 1] = cs[1]


@wp.kernel
def _ramp_mul_c64_df_kernel(
    ky: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
    kx: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
    sy: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
    sx: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
    minus_two_pi: wp.float64,
    wq: wp.array3d(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    im_swap: wp.int32,  # ATen's imaginary FMA on this device (_aten_cmul_order)
    out: wp.array4d(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
):
    p, i, j = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    t = _dadd(_dmul(ky[i], sy[p]), _dmul(kx[j], sx[p]))
    u = _dmul(minus_two_pi, t)
    cs = _df_sincos_f32(u)
    z = cmul_aten(wq[i, j, 0], wq[i, j, 1], cs[0], cs[1], im_swap)
    out[p, i, j, 0] = z[0]
    out[p, i, j, 1] = z[1]


def ramp_phase_df_supported(waves: Tensor, shifts: Tensor) -> bool:
    """Whether the certified float-float kernel builds the ramp for these operands.

    The kernel emits a float32 only where it can prove that it equals the float64
    kernel's word; every other element takes the verbatim float64 ``sincos``.  So
    it takes exactly the operands the float64 kernel takes
    (:func:`~scatterem.physics._ramp_fused.ramp_phase_fused_supported`).
    """
    return ramp_phase_fused_supported(waves, shifts)


def ramp_wq_fold_supported(waves: Tensor, shifts: Tensor, fold: bool) -> bool:
    """Whether ``wq * phase`` can ride on the float-float ramp kernel.

    ``fold`` is ``fourier_shift_stack``'s single-chunk power-of-two condition.
    The kernel writes one ``(P, ny, nx)`` product, so ``waves`` must be 2-D: with
    leading wave axes the broadcast product has a different shape.  ``wq`` (the
    FFT of ``waves``) is an operand of the kernel, and the kernel has no adjoint.
    So the route is also closed when a caller differentiates with respect to
    ``waves``.  Off this route, the caller keeps the separate torch multiply.
    """
    return (
        fold
        and waves.ndim == 2
        and not (torch.is_grad_enabled() and waves.requires_grad)
        and ramp_phase_df_supported(waves, shifts)
        and aten_cmul_imag_swapped(waves.device) is not None
    )


def ramp_phase_df(
    ky: Tensor,
    kx: Tensor,
    sy: Tensor,
    sx: Tensor,
    minus_two_pi: float,
    shape: tuple,
) -> Tensor:
    """:func:`~scatterem.physics._ramp_fused.ramp_phase_fused` on the DF kernel.

    Same arguments, same complex64 bits.  The launch uses PyTorch's current
    stream for the reason ``ramp_phase_fused`` gives.
    """
    return _launch_ramp_phase(
        _ramp_phase_c64_df_kernel, ky, kx, sy, sx, minus_two_pi, shape
    )


def ramp_mul_fused(
    ky: Tensor,
    kx: Tensor,
    sy: Tensor,
    sx: Tensor,
    minus_two_pi: float,
    wq: Tensor,
) -> Tensor:
    """``wq * ramp_phase_df(...)`` for a 2-D complex64 ``wq``, in one kernel.

    The ramp's only reader in ``fourier_shift_stack`` is that broadcast multiply,
    a full read and write pass over the ``(P, ny, nx)`` phase.  This kernel
    applies it before the store instead, with the same bits (see ``cmul_aten``).
    """
    _ensure_warp_initialized()
    ny = int(ky.shape[0])
    nx = int(kx.shape[0])
    P = int(sy.shape[0])
    out = torch.empty((P, ny, nx), dtype=torch.complex64, device=ky.device)
    wp.launch(
        _ramp_mul_c64_df_kernel,
        dim=(P, ny, nx),
        inputs=[
            _wp_freqs(ky),
            _wp_freqs(kx),
            wi.from_torch(sy.contiguous(), dtype=wp.float64),
            wi.from_torch(sx.contiguous(), dtype=wp.float64),
            wp.float64(minus_two_pi),
            wi.from_torch(torch.view_as_real(wq.contiguous()), dtype=wp.float32),
            aten_cmul_swap_flag(ky.device),
            wi.from_torch(torch.view_as_real(out), dtype=wp.float32),
        ],
        device=wi.device_from_torch(ky.device),
        # Same stream rule as ``ramp_phase_fused``.
        stream=wi.stream_from_torch(torch.cuda.current_stream(ky.device)),
    )
    return out
