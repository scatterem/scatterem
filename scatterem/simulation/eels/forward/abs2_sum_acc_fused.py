"""``acc += |psi|**2 .sum(dim=1)`` for a float64 accumulator, as one Warp kernel.

Why this exists
---------------
:func:`..multislice_eels.multislice_transition_potential` and
:func:`.prism_eels.prism_transition_potential_hybrid` both end their inner site
body in

    acc += psi_n.abs().pow_(2).sum(dim=1)

over a ``(Pb, n_trans, Ny, Nx)`` complex64 stack and a ``(Pb, Ny, Nx)`` **float64**
accumulator.  That is four device kernels, and each writes a full-size plane for
the next one to read back (per element of the ``n_trans`` stack, ``E``, plus per
element of the output, ``O``):

    h = abs(psi_n)                   E: read 8 + write 4
    h.pow_(2)                        E: read 4 + write 4
    s = h.sum(dim=1)                 E: read 4          O: write 4
    acc += s                                            O: read 4 + read 8 + write 8

= **24 B per stack element** to move 8 B of input into the accumulator.  One thread
per *output* element walks the transition axis in registers and read-modify-writes
``acc`` once: **8 B per stack element** plus 16 B per output.

This is the reduction-bearing sibling of :mod:`.abs2_acc_fused`,
which fuses the same tail where there is no ``sum``.

Reduction order
---------------
The fused tail of :mod:`.abs2_acc_fused` leaves this sibling out because a float32 reduction over the transition
axis is *allowed* to reassociate.  It does, and an ascending serial accumulate is
**measured wrong**: at ``n_trans = 27`` it disagrees with ATen by ``max|d| 1.9e-05``
(it is bit-exact only at ``n_trans = 4``, which is why a sequential spelling passes
``r7_eels/wide`` and fails everywhere else).

ATen's ``reduce_kernel<nt, vt0=4, ReduceOp<float, ...>>`` keeps **four** partial
accumulators, feeds element ``t`` to accumulator ``t % 4`` in ascending ``t``, and
finishes ``((s0 + s1) + s2) + s3``.  That is what the kernel below spells, and it is
verified by ``torch.equal`` against the eager chain at every registered
``(P, n_trans, Ny, Nx)`` shape of both call sites -- ``n_trans`` 4 and 27, three
grids.  It is an empirical match to an ATen implementation detail, not a theorem:
if a future torch changes ``vt0`` or splits the reduce axis across threads, this
kernel becomes a *tolerance* change rather than a bit-exact one, and ``r7_eels``'s
``sha256`` gate is what will say so.

Bit-exactness is a contract here, not an aspiration
---------------------------------------------------
``r7_eels`` is gated on ``sha256`` of its output cube.  The spellings that carry
that, all of them inherited from :mod:`.abs2_acc_fused` and its
measurements:

* **``hypotf`` via :func:`warp.func_native`**, because ATen's complex ``abs`` is
  ``thrust::abs`` = ``hypotf``; ``libdevice.hypot`` and ``sqrtf(re*re+im*im)`` are
  each 1 ulp off on ~17-21 % of elements.
* **``__fmul_rn(h, h)``** for ``pow_(2)``, and **``__fadd_rn``** for the transition
  sum, so nvrtc has no licence to contract the pair into an FMA -- which would be a
  *different* result from the separate square-then-add ATen performs.
* **``__dadd_rn``** for the accumulate; the float32->float64 conversion of the sum
  is exact, which is why ATen's promotion inside ``add_`` and ``wp.float64(...)``
  here are the same bits.

Scope of the gate
-----------------
CUDA only (a ``wp.launch`` on a CPU device is a single-threaded scalar loop, so the
fusion *loses* there), complex64 source into a float64 accumulator
only, both contiguous, and forward-only: the kernel has **no adjoint**, so a caller
differentiating through the site walk keeps the eager chain and no gradcheck
obligation arises.  The predicate is entirely host-known, so it costs no device
sync.
"""

from __future__ import annotations

import torch
import warp as wp
from torch import Tensor

from scatterem.physics import _warp_interop as wi
from scatterem.physics._warp_rn import dadd_rn as _dadd
from scatterem.physics._warp_rn import fadd_rn as _fadd
from scatterem.physics._warp_rn import fmul_rn as _fmul
from scatterem.physics._warp_rn import hypotf as _hypotf


@wp.func
def _abs2(v: wp.vec2f) -> wp.float32:
    m = _hypotf(v[0], v[1])  # pyright: ignore[reportIndexIssue]
    return _fmul(m, m)


@wp.kernel
def _abs2_sum_accumulate_kernel(
    psi: wp.array4d(dtype=wp.vec2f),  # pyright: ignore[reportInvalidTypeForm]
    n_trans: wp.int32,
    out: wp.array3d(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
):
    """``out += float64(sum_t hypotf(psi[t].re, psi[t].im) ** 2)``, one pass.

    The summation order is **ATen's**, not the obvious one -- see the module
    docstring's "reduction order" section.  Four partial accumulators fed
    round-robin in ascending ``t``, then combined ``((s0+s1)+s2)+s3``.
    """
    p, i, j = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    s0 = float(0.0)  # noqa: UP018 -- declares a mutable Warp local; a bare literal is a constant
    s1 = float(0.0)  # noqa: UP018 -- declares a mutable Warp local; a bare literal is a constant
    s2 = float(0.0)  # noqa: UP018 -- declares a mutable Warp local; a bare literal is a constant
    s3 = float(0.0)  # noqa: UP018 -- declares a mutable Warp local; a bare literal is a constant
    nb = n_trans / 4
    for b in range(nb):
        t = b * 4
        s0 = _fadd(s0, _abs2(psi[p, t, i, j]))
        s1 = _fadd(s1, _abs2(psi[p, t + 1, i, j]))
        s2 = _fadd(s2, _abs2(psi[p, t + 2, i, j]))
        s3 = _fadd(s3, _abs2(psi[p, t + 3, i, j]))
    t = nb * 4  # pyright: ignore[reportOperatorIssue]
    if t < n_trans:
        s0 = _fadd(s0, _abs2(psi[p, t, i, j]))
    if t + 1 < n_trans:
        s1 = _fadd(s1, _abs2(psi[p, t + 1, i, j]))
    if t + 2 < n_trans:
        s2 = _fadd(s2, _abs2(psi[p, t + 2, i, j]))
    s = _fadd(_fadd(_fadd(s0, s1), s2), s3)
    out[p, i, j] = _dadd(out[p, i, j], wp.float64(s))


def abs2_sum_accumulate_fused_supported(psi: Tensor, acc: Tensor) -> bool:
    """Whether the fused kernel reproduces ``acc += psi.abs().pow_(2).sum(dim=1)``.

    Everything is host-known, so no device sync.  ``acc`` must be contiguous because
    the kernel writes through it in place.
    """
    return (
        acc.device.type == "cuda"
        and psi.device == acc.device
        and psi.dtype == torch.complex64
        and acc.dtype == torch.float64
        and psi.dim() == 4
        and acc.dim() == 3
        and psi.shape[0] == acc.shape[0]
        and psi.shape[2] == acc.shape[1]
        and psi.shape[3] == acc.shape[2]
        and psi.is_contiguous()
        and acc.is_contiguous()
        and not (torch.is_grad_enabled() and (psi.requires_grad or acc.requires_grad))
    )


def abs2_sum_accumulate_fused(acc: Tensor, psi: Tensor) -> None:
    """``acc += psi.abs().pow_(2).sum(dim=1)`` in place.  Caller checks support."""
    wp.launch(
        _abs2_sum_accumulate_kernel,
        dim=tuple(int(s) for s in acc.shape),
        inputs=[
            wi.from_torch(torch.view_as_real(psi), dtype=wp.vec2f),
            int(psi.shape[1]),
        ],
        outputs=[wi.from_torch(acc, dtype=wp.float64)],
        device=wi.device_from_torch(acc.device),
        # NOT a detail: Warp's default stream for a device is not PyTorch's current
        # stream, so a launch left on Warp's own stream is unordered against the
        # ATen ops that produced ``psi`` and consume ``acc``.  Same rule, and the
        # same measured failure, as ``physics/_ramp_fused.py``.
        stream=wi.stream_from_torch(torch.cuda.current_stream(acc.device)),
    )
