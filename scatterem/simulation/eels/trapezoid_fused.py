"""The weighted-trapezoid integrand of ``_radial_overlaps_torch`` as one kernel.

Why this exists
---------------
``transition_potentials._radial_overlaps_torch`` reduces the radial overlap
integral ``∫ u_b(r) j_l''(qr) u_e(r) dr`` with

    integrand = sel * ube.view(1, 1, -1)          # (n_lpp, Nq, Nr)
    integral  = torch.trapezoid(integrand, r, dim=-1)

on a ``(n_lpp, Nq=512, Nr=20000)`` float64 argument matrix -- 82 MB *per level*.
``torch.trapezoid`` is itself three more passes (``x.diff()``, then
``(y[..., :-1] + y[..., 1:])``, then ``* dx``, then the reduce and a scalar
halving), so the chain materialises **three** full-size intermediates to produce
a ``(n_lpp, Nq)`` answer:

    mul  sel*ube          [3,512,20000]   0.724 ms   679 GB/s
    add  left+right       [3,512,19999]   0.719              (L2 reuse)
    mul  * dx             [3,512,19999]   0.724      679
    sum                   [3,512,19999]   0.363      677
                                          --------
                                          2.53 ms per 3-level group

Every one of those passes is *at this card's DRAM roof* (perf-lab QD91 measured
673-680 GB/s across the whole row), which is exactly why rate-based screening
said there was nothing here: the defect is not that the passes are slow, it is
that three of the four intermediates need not exist.  One thread per
``(level, q, i)`` keeps the chain in registers and writes the summand once.

What this does NOT do, and why
------------------------------
The **reduction stays in torch**.  ``r7_eels`` is gated on a golden ``sha256`` of
its output, and a hand-written reduction would reassociate ``.sum(-1)`` -- the
same reason QD91 declined collapsing the whole chain into a weighted matvec
(``sel @ (ube*w)``), which is the obvious 5x-traffic win and is *not* bit-exact.
The kernel produces the identical ``(n_lpp, Nq, Nr-1)`` contiguous summand tensor
that ``* dx`` produced before, so torch's reduce sees the same values in the same
layout and therefore uses the same tree.

Bit-exactness is a contract here, not an aspiration
---------------------------------------------------
Four roundings per element, in this order, are what the eager chain performs:

    lo = fl(sel[a,q,i]   * ube[i])
    hi = fl(sel[a,q,i+1] * ube[i+1])
    t  = fl(fl(lo + hi) * dx[i])

* **``__dmul_rn`` / ``__dadd_rn`` are load-bearing here**, unlike in
  ``ramp_fused.py`` where they are belt-and-braces: ``lo + hi`` is literally
  ``a*b + c*d``, the shape nvrtc contracts into an FMA, and
  ``diffraction_tomo/kinematic_fused.py`` records that contraction changing
  results in this very repository.  Do not respell them with plain ``*``/``+``
  without re-running the exactness probe.
* **The halving is exact and therefore free to move.**  ``torch.trapezoid``
  divides by 2 (an exact binary operation for every non-subnormal double), so
  ``((l+r)*dx).sum(-1)/2``, ``((l+r)*dx/2).sum(-1)`` and ``((l+r)/2*dx).sum(-1)``
  are bitwise identical -- verified over the real reduction length, not assumed.
  That is what makes it legitimate to apply the scale *after* the reduce here.
* ``dx`` is ``r.diff()``, which is bitwise the ``aten::sub`` ``torch.trapezoid``
  performs on the same ``r``.

Scope of the gate
-----------------
CUDA only (``wp.launch`` on a CPU device is a single-threaded scalar loop and the
fusion *loses* there by ~6x -- perf-lab D302), float64 only, contiguous operands
only, and **forward-only**: the kernel has no adjoint, so a caller
differentiating through the radial overlaps keeps the eager chain and no
gradcheck obligation arises.
"""

from __future__ import annotations

import torch
import warp as wp
from torch import Tensor

# ``a * b`` and ``a + b`` with the round-to-nearest intrinsics, so nvrtc cannot
# contract ``lo + hi`` -- which is ``a*b + c*d`` -- into an FMA.  See the module
# docstring: here this is load-bearing, not belt-and-braces.
_MUL_SRC = "return __dmul_rn(a, b);"
_ADD_SRC = "return __dadd_rn(a, b);"


@wp.func_native(_MUL_SRC)
def _dmul(a: wp.float64, b: wp.float64) -> wp.float64: ...


@wp.func_native(_ADD_SRC)
def _dadd(a: wp.float64, b: wp.float64) -> wp.float64: ...


@wp.kernel
def _trapz_weighted_kernel(
    sel: wp.array3d(dtype=wp.float64),  # (n, Nq, Nr)
    ube: wp.array(dtype=wp.float64),  # (Nr,)
    dx: wp.array(dtype=wp.float64),  # (Nr-1,)
    out: wp.array3d(dtype=wp.float64),  # (n, Nq, Nr-1)
):
    a, q, i = wp.tid()
    lo = _dmul(sel[a, q, i], ube[i])
    hi = _dmul(sel[a, q, i + 1], ube[i + 1])
    out[a, q, i] = _dmul(_dadd(lo, hi), dx[i])


def trapezoid_weighted_supported(sel: Tensor, ube: Tensor, r: Tensor) -> bool:
    """Whether the fused kernel reproduces the eager chain for these operands.

    Every clause is host-known, so the predicate costs no device sync.
    """
    return (
        sel.device.type == "cuda"
        and sel.dtype == torch.float64
        and ube.dtype == torch.float64
        and r.dtype == torch.float64
        and sel.dim() == 3
        and ube.dim() == 1
        and r.dim() == 1
        and sel.shape[-1] == ube.shape[0] == r.shape[0]
        and sel.shape[-1] >= 2
        and sel.is_contiguous()
        and ube.is_contiguous()
        and r.is_contiguous()
        and not (
            torch.is_grad_enabled()
            and (sel.requires_grad or ube.requires_grad or r.requires_grad)
        )
    )


def trapezoid_weighted_fused(sel: Tensor, ube: Tensor, r: Tensor) -> Tensor:
    """``torch.trapezoid(sel * ube.view(1, 1, -1), r, dim=-1)``, one kernel + one reduce.

    ``sel`` is ``(n, Nq, Nr)``, ``ube`` and ``r`` are ``(Nr,)``.  Returns
    ``(n, Nq)``.
    """
    n, nq, nr = (int(s) for s in sel.shape)
    dx = r.diff()
    summand = torch.empty((n, nq, nr - 1), dtype=torch.float64, device=sel.device)
    wp.launch(
        _trapz_weighted_kernel,
        dim=(n, nq, nr - 1),
        inputs=[
            wp.from_torch(sel, dtype=wp.float64),
            wp.from_torch(ube, dtype=wp.float64),
            wp.from_torch(dx, dtype=wp.float64),
            wp.from_torch(summand, dtype=wp.float64),
        ],
        device=wp.device_from_torch(sel.device),
        # NOT a detail: Warp's default stream for a device is not PyTorch's
        # current stream, so a launch left on Warp's own stream is unordered
        # against the ATen reduce that consumes ``summand`` -- and is invisible
        # to a ``torch.cuda.graph`` capture.  Same rule as ``ramp_fused.py``.
        stream=wp.stream_from_torch(torch.cuda.current_stream(sel.device)),
    )
    # The halving is exact, hence free to apply after the reduce -- see the
    # module docstring.
    return summand.sum(-1) / 2.0
