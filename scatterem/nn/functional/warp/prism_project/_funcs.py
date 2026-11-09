"""The generic complex ``wp.func`` helpers the kernels here are built from.

Kept in their own module so the forward and backward kernel modules share ONE
definition of each.  They are ``Any``-typed on purpose -- Warp specialises them
per call site, which is what lets the same source serve the fp16 (``wp.vec2h``)
and fp32 (``wp.vec2``) instantiations.
"""

from __future__ import annotations

from typing import Any

import warp as wp

# ``_cmul`` / ``_cconj`` are shared with transmit.py; the kernel modules here
# import them from this module.
from scatterem.nn.functional.warp._complex_funcs import _cconj as _cconj
from scatterem.nn.functional.warp._complex_funcs import _cmul as _cmul

# ===========================================================================
# Fused Warp forward (Task P4)
# ===========================================================================
#
# The clean split (per the design spec):
#   - ramp + ifft2 (FFTs) stay in torch: build the per-position real-space
#     basis ``basis_rs`` of shape (Nmodes, K, Bp, M1, M2).  Chunking this to
#     bound its memory is the NEXT task (P5); here we materialise all K at once.
#   - a single Warp kernel does the cyclic-crop gather + parent-beam sum with
#     register accumulation per output element: for (m, k, y, x), loop
#     p in [0, Bp) and accumulate
#         basis_rs[m,k,p,y,x] * Sdt[p, (py_k+y)%SY, (px_k+x)%SX]
#     into a wp.vec2f register, written once -> exit[m,k,y,x].
#
# Complex tensors are passed as their view_as_real (vec2f) layout, mirroring
# the repo convention in transmit.py.


@wp.func
def _fwd_prod(a: Any, b: Any, mul_fp16: int):
    # Per-beam complex product for the forward sum, returned as an fp32 wp.vec2.
    #
    # ``a``/``b`` are the input value type (wp.vec2 fp32 / wp.vec2h fp16):
    #   * fp32 inputs (wp.vec2): the cast/multiply is fp32 either way; ``mul_fp16``
    #     is a no-op.
    #   * fp16 inputs (wp.vec2h): ``mul_fp16 == 1`` multiplies in fp16 (the current
    #     "fp16" preset); ``mul_fp16 == 0`` first casts each operand to wp.vec2
    #     (fp32) and multiplies in fp32 (the "mixed" preset — fp32 arithmetic on a
    #     complex32-stored S-matrix, NO full-tensor upcast).
    # The product is always returned fp32 for the fp32 accumulator.
    if mul_fp16 == 1:
        prod = _cmul(a, b)  # input precision
        return wp.vec2(wp.float32(prod[0]), wp.float32(prod[1]))
    af = wp.vec2(wp.float32(a[0]), wp.float32(a[1]))
    bf = wp.vec2(wp.float32(b[0]), wp.float32(b[1]))
    return _cmul(af, bf)  # fp32 multiply
