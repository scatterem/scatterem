"""Round-to-nearest float32 / float64 arithmetic (and ``hypotf``) as Warp functions.

nvcc contracts a multiply that feeds an add or a subtract into one FMA, and
NVRTC makes that choice per device. A Warp kernel that must give torch's bits
for a separately rounded ``a * b + c`` spells each operation with its CUDA
``__<p><op>_rn`` intrinsic, which the compiler never contracts. Each kernel
module used to declare its own copies of these one-line ``wp.func_native``
functions; they all live here now, and a kernel module imports the ones it
uses (usually under its historical ``_fmul`` / ``_dadd`` names).

Each function has a host branch with the plain C operator, so that a Warp
module that holds one still builds for the CPU (Warp compiles the whole module
for the launch device). On CUDA the branch is the intrinsic, exactly as
before.
"""

from __future__ import annotations

import warp as wp

__all__ = [
    "dadd_rn",
    "ddiv_rn",
    "dmul_rn",
    "drcp_rn",
    "dsqrt_rn",
    "dsub_rn",
    "fadd_rn",
    "fdiv_rn",
    "fmul_rn",
    "frcp_rn",
    "fsqrt_rn",
    "fsub_rn",
    "hypotf",
]

_BINARY = """
#if defined(__CUDA_ARCH__)
return __{p}{op}_rn(a, b);
#else
return a {sym} b;
#endif
"""

_UNARY = """
#if defined(__CUDA_ARCH__)
return __{p}{op}_rn(a);
#else
return {host};
#endif
"""


@wp.func_native(_BINARY.format(p="f", op="mul", sym="*"))
def fmul_rn(a: wp.float32, b: wp.float32) -> wp.float32: ...


@wp.func_native(_BINARY.format(p="f", op="add", sym="+"))
def fadd_rn(a: wp.float32, b: wp.float32) -> wp.float32: ...


@wp.func_native(_BINARY.format(p="f", op="sub", sym="-"))
def fsub_rn(a: wp.float32, b: wp.float32) -> wp.float32: ...


@wp.func_native(_BINARY.format(p="f", op="div", sym="/"))
def fdiv_rn(a: wp.float32, b: wp.float32) -> wp.float32: ...


@wp.func_native(_UNARY.format(p="f", op="sqrt", host="sqrtf(a)"))
def fsqrt_rn(a: wp.float32) -> wp.float32: ...


@wp.func_native(_UNARY.format(p="f", op="rcp", host="1.0f / a"))
def frcp_rn(a: wp.float32) -> wp.float32: ...


@wp.func_native(_BINARY.format(p="d", op="mul", sym="*"))
def dmul_rn(a: wp.float64, b: wp.float64) -> wp.float64: ...


@wp.func_native(_BINARY.format(p="d", op="add", sym="+"))
def dadd_rn(a: wp.float64, b: wp.float64) -> wp.float64: ...


@wp.func_native(_BINARY.format(p="d", op="sub", sym="-"))
def dsub_rn(a: wp.float64, b: wp.float64) -> wp.float64: ...


@wp.func_native(_BINARY.format(p="d", op="div", sym="/"))
def ddiv_rn(a: wp.float64, b: wp.float64) -> wp.float64: ...


@wp.func_native(_UNARY.format(p="d", op="sqrt", host="sqrt(a)"))
def dsqrt_rn(a: wp.float64) -> wp.float64: ...


@wp.func_native(_UNARY.format(p="d", op="rcp", host="1.0 / a"))
def drcp_rn(a: wp.float64) -> wp.float64: ...


# Not an ``_rn`` intrinsic: CUDA's ``hypotf`` (the same name on the host), which
# the kernels use for ``|z|`` where torch's ``abs`` of a complex64 does.
@wp.func_native("return hypotf(a, b);")
def hypotf(a: wp.float32, b: wp.float32) -> wp.float32: ...
