"""``acc += |psi|**2`` for a float64 accumulator, as one Warp kernel.

Why this exists
---------------
``prism_eels.prism_transition_potential``'s inner chunk loop ends in

    out[c0:c1] += _abs_complex(ex).pow(2)

over a ``(scan_chunk, ny, nx)`` complex64 ``ex`` and a **float64** accumulator.
That is three device kernels, and each writes a full-size plane for the next one
to read back:

    h = hypot(ex.real, ex.imag)      read  8 + write 4
    p = h.pow(2)                     read  4 + write 4
    out += p                         read  4 + read 8 + write 8   (f32->f64 in
                                                                   TensorIterator)

= **40 B/element** to move 8 B/element of input into an accumulator.  One thread
per element keeps ``h`` and ``h*h`` in registers and read-modify-writes ``out``
once: **24 B/element**, the compulsory traffic and nothing else.

Measured at the real ``(64, 192, 192)`` shape (a benchmark script,
median of 9 after 3 warm-ups): the eager chain **148.9 us**, this kernel **92.7 us**
= 1.606x, against a 1.667x traffic bound, i.e. 96 % of its own roof.  The chunk
loop runs 256 times per ``r7_eels/wide/simulate`` (16 site x transition pairs x 16
chunks of P = 1024), so that is ~14.4 ms of a ~347 ms row.

Bit-exactness is a contract here, not an aspiration
---------------------------------------------------
``r7_eels`` is gated on ``sha256`` of its output cube, so this kernel has to
reproduce ``float64(f32_square(hypotf(re, im)))`` word for word.  Three spellings
are load-bearing:

* **``hypotf`` via :func:`warp.func_native`, not ``libdevice.hypot`` and not
  ``sqrtf(re*re + im*im)``.**  ATen's complex ``abs_kernel_cuda`` is
  ``thrust::abs``, i.e. ``hypotf``; the two alternatives are each **1 ulp** off on
  16.9 % / 20.6 % of 2 359 296 elements (both measured).
  ``hypotf`` emitted directly is 0 differing words -- measured, not
  reasoned about, by ``torch.equal`` at the real shape.
* **``__fmul_rn(h, h)``**, because ``Tensor.pow(2)`` is ``h*h`` bit for bit
  and the round-to-nearest intrinsic is that multiply with no licence for nvrtc to
  contract it into something else.
* **``__dadd_rn``** for the accumulate, for the same reason -- and note the
  float32->float64 conversion of ``h*h`` is exact, which is *why* the eager chain's
  promotion inside ``add_`` and this kernel's ``wp.float64(...)`` are the same bits.

``physics._ramp_fused``'s note applies verbatim: the intrinsics are a guarantee, not a
demonstrated necessity, and nobody should repeat a "measured to be load-bearing"
claim about them without re-measuring it.

Scope of the gate
-----------------
CUDA only (``wp.launch`` on a CPU device is a single-threaded scalar loop, so the
fusion *loses* there), complex64 source into a float64 accumulator
only, both operands contiguous, and forward-only: the kernel has **no adjoint**, so
a caller differentiating through the synthesis keeps the eager chain and no
gradcheck obligation arises.  The predicate is entirely host-known, so it costs no
device sync.
"""

from __future__ import annotations

import torch
import warp as wp
from torch import Tensor

from scatterem.physics import _warp_interop as wi
from scatterem.physics._warp_rn import dadd_rn as _dadd
from scatterem.physics._warp_rn import fmul_rn as _fmul
from scatterem.physics._warp_rn import hypotf as _hypotf


@wp.kernel
def _abs2_accumulate_kernel(
    psi: wp.array3d(dtype=wp.vec2f),  # pyright: ignore[reportInvalidTypeForm]
    out: wp.array3d(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
):
    """``out += float64(hypotf(psi.re, psi.im) ** 2)``, one pass."""
    c, i, j = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    v = psi[c, i, j]
    m = _hypotf(v[0], v[1])
    out[c, i, j] = _dadd(out[c, i, j], wp.float64(_fmul(m, m)))


def abs2_accumulate_fused_supported(psi: Tensor, acc: Tensor) -> bool:
    """Whether the fused kernel reproduces ``acc += psi.abs().pow(2)`` here.

    Everything is host-known, so no device sync.  ``acc`` must be contiguous
    because the kernel writes through it in place; ``acc`` is a leading-axis slice
    of the caller's accumulator, which is contiguous, but a caller that hands a
    strided view must keep the eager path rather than silently write a copy.
    """
    return (
        acc.device.type == "cuda"
        and psi.device == acc.device
        and psi.dtype == torch.complex64
        and acc.dtype == torch.float64
        and psi.shape == acc.shape
        and psi.dim() == 3
        and psi.is_contiguous()
        and acc.is_contiguous()
        and not (torch.is_grad_enabled() and (psi.requires_grad or acc.requires_grad))
    )


def abs2_accumulate_fused(acc: Tensor, psi: Tensor) -> None:
    """``acc += psi.abs().pow(2)`` in place, one kernel.  Caller checks support."""
    dim = tuple(int(s) for s in acc.shape)
    wp.launch(
        _abs2_accumulate_kernel,
        dim=dim,
        inputs=[wi.from_torch(torch.view_as_real(psi), dtype=wp.vec2f)],
        outputs=[wi.from_torch(acc, dtype=wp.float64)],
        device=wi.device_from_torch(acc.device),
        # NOT a detail: Warp's default stream for a device is not PyTorch's
        # current stream, so a launch left on Warp's own stream is unordered
        # against the ATen ops that produced ``psi`` and consume ``acc``.  Same
        # rule, and the same measured failure, as ``physics/_ramp_fused.py``.
        stream=wi.stream_from_torch(torch.cuda.current_stream(acc.device)),
    )
