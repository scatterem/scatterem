"""Generic complex ``wp.func`` helpers shared by the transmit and PRISM kernels.

The helpers are ``Any``-typed on purpose. Warp specialises them per call site,
so one source serves the fp16 (``wp.vec2h``) and fp32 (``wp.vec2``) kernels.
:mod:`scatterem.utils.warp.complex` has fp32-only (``wp.vec2``) versions.
"""

from typing import Any

import warp as wp


@wp.func
def _cmul(a: Any, b: Any):
    # Complex multiply for a 2-component value vector (real, imag). Generic over
    # the value type: wp.vec2 (float32 components, complex64) or wp.vec2h
    # (float16 components, complex32). Construct the result via ``type(a)(...)``
    # so the component type matches the inputs (no float32 literal leaks into
    # the fp16 path).
    return type(a)(a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


@wp.func
def _cconj(a: Any):
    # Complex conjugate; generic over the value type (wp.vec2 / wp.vec2h).
    return type(a)(a[0], -a[1])
