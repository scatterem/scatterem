"""K-summed conjugate ramp product ``out[m, b] = sum_k grad[m, b, k] * conj(ramp[k])``.

Private, lazily imported by its one call site
(``scatterem.nn.functional.warp.prism_project._torch_adjoint._RampMulKSum``) so that
``import scatterem`` never requires Triton -- same arrangement as
:mod:`scatterem.nn.functional._bcast_mul_conj_triton`.

Why this exists
---------------
On the torch route with a sub-pixel ramp, the per-position basis is
``ifft2(basis_fft[:, :, None] * ramp)``, and autograd's adjoint of that multiply
w.r.t. ``basis_fft`` is ``(grad * ramp.conj()).sum_to_size(...)``: a full
``(Nm, Bp, K, M1, M2)`` product (2 GiB at ``r2_prism_project/medium``) written only
for the reduction over ``K`` to read it back.  Measured on
``medium/project_backward_joint --no-warp``: the ``mul`` 9.44 ms plus the
``sum`` 3.45 ms = 34 % of the row's device time.  The Warp route already folds the
two (``_RampIFFT2`` + ``prism_ramp_adjoint``); this is the same fold for the
torch route: read ``grad`` once, write the K-summed result once.

Numerics -- a bit-exactness contract, not a tolerance
----------------------------------------------------
The result is ``torch.equal`` to the eager expression at the registered shapes
(two benchmark scripts, random cotangent):

* the product is spelled the way ATen's complex64 ``mul`` rounds ``g * w``
  (``g`` first): ``fma(gr, wr, -(gi*wi))``, ``fma(gr, wi, gi*wr)``, with the
  conj applied to ``ramp`` in registers as an exact sign flip (``* -1.0``, not
  unary minus -- Triton lowers that as ``0 - x``); on a device whose
  ATen build fuses the other imaginary product (sm_120) the imaginary half is
  ``fma(gi, wr, gr*wi)`` (:mod:`scatterem.physics._aten_cmul_order`);
* the sum over ``K`` follows TensorIterator's outer-dimension thread reduce
  (``vt0 = 4``): four accumulators seeded with ``0``, term ``k`` added to
  accumulator ``k % 4`` in ``k`` order, then combined ``((a0 + a1) + a2) + a3``.
  A plain ``k``-order sum differs in ~58 % of words; this order in 0.

If ATen ever splits this reduction across threads (tiny outputs), the order and
so the bits would differ; the result stays a correct, deterministic fp32 sum.

Gating
------
Declines -- and the caller keeps the eager expression -- unless both operands are
contiguous CUDA ``complex64`` with no lazy conj / neg bit, ``grad`` is
``(Nm, Bp, K, *P)`` against a ``(1, 1, K, *P)`` ramp, the plane is a whole number of
``BLOCK`` tiles, the flat float offsets fit in int32, and no operand needs a
gradient under grad mode.  The gate is the ``ramp_adjoint_ksum`` backend row's
predicate, :func:`ramp_adjoint_ksum_supported`.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from scatterem.nn._backends import RAMP_ADJOINT_KSUM
from scatterem.physics._aten_cmul_order import aten_cmul_imag_swapped
from scatterem.utils.env import resolve_backend

__all__ = ["CALLS", "ramp_adjoint_ksum", "ramp_adjoint_ksum_supported"]

#: ``[ran, declined]``, host-side only -- proves the fast path was TAKEN.
CALLS = [0, 0]

BLOCK_DEFAULT = 1024
GROUPS_DEFAULT = 16


@triton.jit
def _ramp_adjoint_ksum_kernel(
    Out,  # (NB, PLANE, 2) f32  -- real view of the K-summed output [out]
    G,  # (NB, K, PLANE, 2) f32 -- real view of the cotangent       [in]
    R,  # (K, PLANE, 2) f32     -- real view of the (unconjugated) ramp [in]
    PLANE,
    NB,
    BPP,  # beams per program
    K,
    BLOCK: tl.constexpr,
    IM_SWAP: tl.constexpr,  # ATen's imaginary FMA on this device (_aten_cmul_order)
):
    t = tl.program_id(0)  # which plane tile
    g = tl.program_id(1)  # which beam group
    o2 = 2 * (t * BLOCK) + tl.arange(0, 2 * BLOCK)
    b1 = tl.minimum(g * BPP + BPP, NB)
    for b in range(g * BPP, b1):
        a0r = tl.zeros((BLOCK,), dtype=tl.float32)
        a0i = tl.zeros((BLOCK,), dtype=tl.float32)
        a1r = tl.zeros((BLOCK,), dtype=tl.float32)
        a1i = tl.zeros((BLOCK,), dtype=tl.float32)
        a2r = tl.zeros((BLOCK,), dtype=tl.float32)
        a2i = tl.zeros((BLOCK,), dtype=tl.float32)
        a3r = tl.zeros((BLOCK,), dtype=tl.float32)
        a3i = tl.zeros((BLOCK,), dtype=tl.float32)
        for k in range(K):
            rf = tl.load(R + o2 + 2 * k * PLANE, eviction_policy="evict_last")
            wr, ri = tl.split(tl.reshape(rf, (BLOCK, 2)))
            wi = ri * -1.0  # conj(ramp), exactly
            gf = tl.load(
                G + o2 + 2 * (b * K + k) * PLANE, eviction_policy="evict_first"
            )
            gr, gi = tl.split(tl.reshape(gf, (BLOCK, 2)))
            # ATen's ``g * w``.
            cr = tl.math.fma(gr, wr, (gi * wi) * -1.0)
            if IM_SWAP:
                ci = tl.math.fma(gi, wr, gr * wi)
            else:
                ci = tl.math.fma(gr, wi, gi * wr)
            j = k % 4
            if j == 0:
                a0r = a0r + cr
                a0i = a0i + ci
            elif j == 1:
                a1r = a1r + cr
                a1i = a1i + ci
            elif j == 2:
                a2r = a2r + cr
                a2i = a2i + ci
            else:
                a3r = a3r + cr
                a3i = a3i + ci
        accr = ((a0r + a1r) + a2r) + a3r
        acci = ((a0i + a1i) + a2i) + a3i
        tl.store(
            Out + o2 + 2 * b * PLANE,
            tl.reshape(tl.join(accr, acci), (2 * BLOCK,)),
            eviction_policy="evict_first",
        )


def _plain(t: torch.Tensor) -> bool:
    return (
        t.is_cuda
        and t.dtype is torch.complex64
        and t.is_contiguous()
        and not (t.is_conj() or t.is_neg())
        and not (torch.is_grad_enabled() and t.requires_grad)
    )


def ramp_adjoint_ksum_supported(
    grad: torch.Tensor, ramp: torch.Tensor, block: int = BLOCK_DEFAULT
) -> bool:
    """True if :func:`ramp_adjoint_ksum` may replace the eager product + K-sum."""
    if not (_plain(grad) and _plain(ramp)):
        return False
    if aten_cmul_imag_swapped(grad.device) is None:
        return False
    if (
        grad.ndim < 4
        or ramp.ndim != grad.ndim
        or tuple(ramp.shape[:2]) != (1, 1)
        or tuple(ramp.shape[2:]) != tuple(grad.shape[2:])
    ):
        return False
    nb, K = grad.shape[1:3]
    npx = grad[0, 0, 0].numel()
    return not (
        npx == 0 or nb == 0 or K == 0 or npx % block != 0 or 2 * nb * K * npx >= 2**31
    )


def ramp_adjoint_ksum(
    grad: torch.Tensor,
    ramp: torch.Tensor,
    block: int = BLOCK_DEFAULT,
    groups: int = GROUPS_DEFAULT,
) -> torch.Tensor | None:
    """``(grad * ramp.conj()).sum(2)``, or ``None``.

    ``grad`` is ``(Nm, Bp, K, *P)`` and ``ramp`` is ``(1, 1, K, *P)``; the result
    is ``(Nm, Bp, *P)``, contiguous.  Returns ``None`` without launching anything
    when a gate in the module docstring fails -- declining is the contract.
    """
    if resolve_backend(RAMP_ADJOINT_KSUM, None, grad, ramp, block) != "triton":
        CALLS[1] += 1
        return None
    nm, nb, K = grad.shape[:3]
    npx = grad[0, 0, 0].numel()
    out = torch.empty(
        (nm, nb) + tuple(grad.shape[3:]), dtype=torch.complex64, device=grad.device
    )
    bpp = (nb + groups - 1) // groups
    r_real = torch.view_as_real(ramp.view(K, npx))
    for m in range(nm):
        _ramp_adjoint_ksum_kernel[(npx // block, groups)](  # pyright: ignore[reportIndexIssue]
            torch.view_as_real(out[m].view(nb, npx)),
            torch.view_as_real(grad[m].view(nb * K, npx)),
            r_real,
            npx,
            nb,
            bpp,
            K,
            BLOCK=block,
            IM_SWAP=aten_cmul_imag_swapped(grad.device),
            num_warps=4,
        )
    CALLS[0] += 1
    return out
