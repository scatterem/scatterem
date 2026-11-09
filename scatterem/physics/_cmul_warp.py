"""ATen's complex64 product ``a * b`` as a Warp function, in the device's FMA order.

nvcc contracts ``c10::complex<float>::operator*`` into one FMA per half.  A Warp
kernel that must give ATen's bits spells those FMAs with ``__fmaf_rn``: left to
NVRTC, the contraction is the compiler's choice, and it does not follow ATen on
every device.  The real half is ``fma(ar, br, -(ai*bi))`` everywhere.  The
imaginary half is ``fma(ar, bi, ai*br)`` in ATen's sm_75 to sm_90 code and
``fma(ai, br, ar*bi)`` in its sm_120 code; ``swap`` (from
:func:`scatterem.physics._aten_cmul_order.aten_cmul_imag_swapped`) selects the
second.  ``swap`` is a kernel argument with one value for the whole launch, so
the branch does not diverge.
"""

from __future__ import annotations

import warp as wp

__all__ = ["cmul_aten"]


# No caller launches it on the CPU.  The CPU branch exists only so that a Warp
# module holding this function still builds there (Warp compiles the whole
# module for the launch device), e.g. for a CPU launch of a sibling kernel.
_SRC = """
#if defined(__CUDA_ARCH__)
const float re = __fmaf_rn(ar, br, -__fmul_rn(ai, bi));
const float im = swap ? __fmaf_rn(ai, br, __fmul_rn(ar, bi))
                      : __fmaf_rn(ar, bi, __fmul_rn(ai, br));
#else
const float pi = ai * bi;
const float ps = swap ? ar * bi : ai * br;
const float re = fmaf(ar, br, -pi);
const float im = swap ? fmaf(ai, br, ps) : fmaf(ar, bi, ps);
#endif
return wp::vec_t<2, wp::float32>(re, im);
"""


@wp.func_native(_SRC)
def cmul_aten(
    ar: wp.float32, ai: wp.float32, br: wp.float32, bi: wp.float32, swap: wp.int32
) -> wp.vec2f: ...
