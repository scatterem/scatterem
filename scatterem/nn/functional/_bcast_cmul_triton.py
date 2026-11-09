"""Plane-resident broadcast complex multiply ``w[b] *= plane`` for large beam stacks.

Private, lazily imported by the two call sites in the PRISM S-matrix build loop
(the S-matrix build of :mod:`scatterem.nn.scan_exit_wave.prism` and
:meth:`scatterem.nn.modules.Propagator.forward`) so that ``import scatterem`` never
requires Triton -- same arrangement as :mod:`scatterem.nn.functional._conj_mul_reduce_triton`.
Triton ships with the CUDA builds of PyTorch; CPU/ROCm-less builds do not have it.

Why this exists
---------------
The multislice loop over a large parent-beam chunk runs two broadcast complex64
multiplies per slice -- the transmission ``T[z] * w`` and the Fresnel ramp
``w *= ramp`` -- against a ``(MY, MX)`` plane and a ``(B, MY, MX)`` wave stack.  At
the production shape ``(2288, 512, 512)`` (4.80 GB) torch runs both at **482 GiB/s**
while this card's in-place read+write roof is **636 GiB/s**: a 1.31x gap.  It is not a
spelling problem -- ``w.mul_(ramp)``, ``mul_(ramp.unsqueeze(0))``, the flattened
2-D form and ``torch.mul(..., out=w)`` were measured within 0.2 % of each other
(a benchmark script).

The cause is L2 residency, and it is visible as a step in the plane size at constant
total traffic (a one-off probe):

    plane 512^2 = 2.00 MiB   roof 14.055 ms   broadcast 18.442 ms   gap 1.312x
    plane 256^2 = 0.50 MiB   roof 15.029 ms   broadcast 14.273 ms   gap 0.950x
    plane 128^2 = 0.12 MiB   roof 15.047 ms   broadcast 14.204 ms   gap 0.944x

A flat elementwise kernel touches the plane once per element of ``w``, so the plane
is read 4.8 GB / 2 MB = 2288 times over.  A 0.5 MiB plane survives in the 6 MiB L2
against that stream and costs nothing; a 2 MiB plane does not, and the misses come
back as DRAM traffic.  Writing the same arithmetic as a flat Triton kernel reproduces
torch exactly (18.432 ms), which confirms the arithmetic was never the problem;
Triton's ``evict_first``/``evict_last`` hints recover only part of it (16.562 ms).

This kernel makes the plane loop-invariant instead: each program owns one ``BLOCK``
tile of the plane, loads it **once into registers**, and then strides over the beam
axis, so total plane traffic is 2 MB rather than 4.8 GB.  Measured at the production
shape: **14.190 ms, 629.8 GiB/s = 99.0 % of the 636 GiB/s roof, 1.305x**.

Deferred scaling (``scale=``)
----------------------------
On a grid that is NOT a power of two, :meth:`Propagator.forward` cannot ride the
round-trip's ``1/N`` in on the phase ramp (it is not bit-exact there), so
``ifft2(norm=None)`` pays torch's SEPARATE full-field scaling pass over the whole beam
stack -- a read+write of 1.228 GB per slice at the ``r3_simulator/small`` PRISM shape.
``scale`` lets the transmit that immediately follows absorb it: the propagator takes the
UNNORMALIZED inverse (``norm="forward"``) and this kernel computes ``plane * (w * scale)``
in the pass it was already making, deleting the other one.  That is bit-exact by
construction rather than by luck -- the same two roundings happen in the same order, only
the memory round trip between them is gone -- and it is checked as such
(a one-off probe: ``torch.equal`` at ``(4165,192,192)``, and
``ifft2(norm=None) == ifft2(norm="forward") * (1/N)`` bitwise at 96^2/128^2/192^2).

Numerics -- a bit-exactness contract, not a tolerance
----------------------------------------------------
The result is ``torch.equal`` to the torch op it replaces, and it has to be: this
runs inside a 32-slice multislice chain whose output is held at a sha256,
so a one-ulp move in slice 0 is a visible change in the exit wave.

Two things make that hold, and both are checked by a one-off probe over 26
shape/order/dtype combinations rather than assumed:

* **Operand order is preserved.**  Complex multiply is commutative in exact
  arithmetic but *not* bitwise once the compiler contracts a product into an FMA --
  which of the two products gets rounded first depends on which operand is first.
  ``torch.equal(w * plane, plane * w)`` is **False** at these shapes, so
  ``plane_first`` selects the matching spelling.
* **The FMA is written explicitly.**  ``a*c - b*d`` spelled as three ops is at the
  mercy of the backend's contraction decision: the same expression is bit-exact
  against torch in the plane-resident kernel and *not* bit-exact in a flat kernel
  (6.1e-08).  ``tl.math.fma`` pins it, so the kernel does not depend on Triton's
  contraction defaults.  A torch build that changed nvcc's contraction would break
  the contract loudly (the sha256), not silently.
* **The imaginary FMA follows the device.**  ATen's sm_120 code fuses the other
  product of the imaginary half (``fma(xi, yr, xr*yi)`` instead of
  ``fma(xr, yi, xi*yr)``).  :mod:`scatterem.physics._aten_cmul_order` measures the
  order once per device; ``IM_SWAP`` selects it, and the kernel declines when
  ATen uses neither.

Gating
------
The fast path declines -- and the caller keeps the torch op -- unless every condition
below holds.  The thresholds are the mechanism, not tuning: below them torch is
already at the roof (see the table above), so taking this kernel there would be a
regression risk for no gain.

* CUDA, complex64, contiguous, ``plane.shape == w.shape[-2:]``, ``w.ndim >= 3``;
* the plane is at least 2 MiB (``MY*MX >= 2**18``) and a whole number of ``BLOCK``
  tiles -- i.e. exactly the regime where the plane stops fitting alongside the
  stream;
* the stack is at least 256 MiB, so the stream really does evict the plane.  This
  deliberately keeps the ordinary ``(NZ, 512, 512)`` conventional-multislice
  propagate (67 MB, measured at 671 GB/s -- already at the roof) on the torch path;
  both size floors apply only without a ``scale`` (with one, the win is a deleted
  pass at any size);
* **no autograd.**  In-place writes from a Triton kernel are invisible to autograd,
  so a live graph would get silently wrong gradients -- this repository's recorded
  failure mode.  Same gate shape as ``physics.propagation.propagate``'s in-place branch.
  A forward-mode dual operand is declined too (``_forward_ad.carries_tangent``):
  the kernel multiplies the primal only, so the tangent would not be propagated.
* ``SCATTEREM_BCAST_CMUL=0`` forces the torch path (used for the A/B measurement).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from scatterem.nn._backends import BCAST_CMUL
from scatterem.nn.functional._forward_ad import carries_tangent
from scatterem.physics._aten_cmul_order import aten_cmul_imag_swapped
from scatterem.utils.env import resolve_backend

__all__ = ["bcast_cmul_", "BLOCK_DEFAULT", "GROUPS_DEFAULT", "CALLS"]

#: ``[ran, declined]``.  Host-side counters only (no sync, no device work); they exist
#: so a probe can prove the fast path was actually TAKEN rather than infer it from a
#: timing -- a silent decline reads exactly like "no speedup available".
CALLS = [0, 0]

# 1024 complex elements per program-tile; 512^2 is a whole 256 tiles.  GROUPS is the
# beam-axis split, i.e. how many programs share one plane tile: (256, 16) = 4096
# programs keeps the card busy while each program still amortises its plane load over
# 143 beams.  Measured at (2288,512,512): g=16 14.190 ms / g=64 14.322 / g=256 14.826.
BLOCK_DEFAULT = 1024
GROUPS_DEFAULT = 16

# Below these the plane already lives in L2 for free and torch is at the roof.
_MIN_PLANE_ELEMS = 1 << 18  # 512^2 complex64 = 2 MiB
_MIN_STACK_BYTES = 256 << 20  # 256 MiB


@triton.jit
def _bcast_cmul_kernel(
    W,  # (NB, PLANE, 2) f32 -- real view of the complex64 stack  [in/out]
    P,  # (PLANE, 2) f32     -- real view of the plane            [in]
    PLANE,
    NB,
    BPP,  # beams per program
    SCALE,  # real prefactor on W, applied BEFORE the multiply (see DO_SCALE)
    BLOCK: tl.constexpr,
    PLANE_FIRST: tl.constexpr,
    DO_SCALE: tl.constexpr,
    IM_SWAP: tl.constexpr,  # ATen's imaginary FMA on this device (_aten_cmul_order)
):
    t = tl.program_id(0)  # which plane tile
    g = tl.program_id(1)  # which beam group
    off = t * BLOCK + tl.arange(0, BLOCK)

    # The plane tile is loaded ONCE and stays in registers for the whole beam loop --
    # this is the entire point of the kernel.  ``evict_last`` asks L2 to keep it for
    # the other beam groups working on the same tile.
    pr = tl.load(P + 2 * off, eviction_policy="evict_last")
    pi = tl.load(P + 2 * off + 1, eviction_policy="evict_last")

    # ``w`` is read and written interleaved (2*BLOCK contiguous floats) and split into
    # re/im with tl.split, so both accesses are fully coalesced 16-byte transactions.
    o2 = 2 * (t * BLOCK) + tl.arange(0, 2 * BLOCK)
    b1 = tl.minimum(g * BPP + BPP, NB)
    for b in range(g * BPP, b1):
        o = o2 + 2 * b * PLANE
        wf = tl.load(W + o, eviction_policy="evict_first")
        wr, wi = tl.split(tl.reshape(wf, (BLOCK, 2)))
        if DO_SCALE:
            # Two SEPARATE f32 multiplies, deliberately not folded into the FMAs
            # below: this has to round exactly where torch's own scaling pass
            # rounds, i.e. `w * SCALE` materialised in f32 and only then
            # multiplied by the plane.  Fusing it into the products would drop
            # that rounding and break the bit-exactness contract.
            wr = wr * SCALE
            wi = wi * SCALE
        if PLANE_FIRST:
            # torch's ``plane * w``: the (im*im) product is rounded, then one FMA.
            cr = tl.math.fma(pr, wr, -(pi * wi))
            if IM_SWAP:
                ci = tl.math.fma(pi, wr, pr * wi)
            else:
                ci = tl.math.fma(pr, wi, pi * wr)
        else:
            # torch's ``w * plane`` / ``w.mul_(plane)``.
            cr = tl.math.fma(wr, pr, -(wi * pi))
            if IM_SWAP:
                ci = tl.math.fma(wi, pr, wr * pi)
            else:
                ci = tl.math.fma(wr, pi, wi * pr)
        tl.store(
            W + o,
            tl.reshape(tl.join(cr, ci), (2 * BLOCK,)),
            eviction_policy="evict_first",
        )


def bcast_cmul_(
    w: torch.Tensor,
    plane: torch.Tensor,
    plane_first: bool,
    block: int = BLOCK_DEFAULT,
    groups: int = GROUPS_DEFAULT,
    scale: float | None = None,
) -> bool:
    """Multiply ``w`` in place by the broadcast ``plane``; ``True`` if it ran.

    ``plane_first`` picks which torch expression is being reproduced bitwise:
    ``True`` for ``plane * w`` (the transmission ``T[z].unsqueeze(0) * w``), ``False``
    for ``w * plane`` / ``w.mul_(plane)`` (the Fresnel ramp).

    ``scale`` (real) reproduces ``plane * (w * scale)``, i.e. it absorbs a separate
    full-field scaling pass into this one -- see "Deferred scaling" above.

    Returns ``False`` without touching ``w`` when any gate in the module docstring
    fails; the caller must then run the torch op itself.  Never raises on a shape it
    does not like -- declining is the contract.
    """
    if resolve_backend(BCAST_CMUL) != "triton":
        CALLS[1] += 1
        return False
    if not (w.is_cuda and plane.is_cuda):
        CALLS[1] += 1
        return False
    if w.dtype is not torch.complex64 or plane.dtype is not torch.complex64:
        CALLS[1] += 1
        return False
    if w.ndim < 3 or tuple(plane.shape) != tuple(w.shape[-2:]):
        CALLS[1] += 1
        return False
    if not (w.is_contiguous() and plane.is_contiguous()):
        CALLS[1] += 1
        return False
    if torch.is_grad_enabled() and (w.requires_grad or plane.requires_grad):
        CALLS[1] += 1
        return False
    if carries_tangent(w, plane):  # the kernel would drop the tangent
        CALLS[1] += 1
        return False
    npx = plane.numel()
    # The plane-size floor is about L2 residency, so it applies only when the
    # multiply is the whole job.  With a ``scale`` to absorb, the win is a DELETED
    # PASS over ``w`` and is independent of the plane size -- measured at
    # (4165,192,192), a 288 KiB plane: 3.656 ms = 1.012x this card's read+write
    # roof, against 7.231 ms for the two torch passes it replaces.
    if npx % block != 0 or (scale is None and npx < _MIN_PLANE_ELEMS):
        CALLS[1] += 1
        return False
    # Same reasoning for the stack floor: it guards the scale-free multiply
    # only.  An L2-sized block of the PRISM build (a 16-beam block at 192^2 is
    # 4.7 MiB) was declined by it, so torch paid ``x * inv_n`` as a separate pass.
    if scale is None and w.numel() * 8 < _MIN_STACK_BYTES:
        CALLS[1] += 1
        return False
    im_swap = aten_cmul_imag_swapped(w.device)
    if im_swap is None:
        CALLS[1] += 1
        return False
    nb = w.numel() // npx
    bpp = (nb + groups - 1) // groups
    _bcast_cmul_kernel[(npx // block, groups)](
        torch.view_as_real(w.view(nb, npx)),
        torch.view_as_real(plane.view(npx)),
        npx,
        nb,
        bpp,
        1.0 if scale is None else float(scale),
        BLOCK=block,
        PLANE_FIRST=plane_first,
        DO_SCALE=scale is not None,
        IM_SWAP=im_swap,
        num_warps=4,
    )
    CALLS[0] += 1
    return True
