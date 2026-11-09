"""Fused mode-axis conjugate-product reduction ``out[n] = sum_m g[m,n] * conj(w[m,n])``.

Private, lazily imported by the one call site that needs it (the backward of
:func:`scatterem.nn.ptychography.multislice_transmit_slice`'s broadcast transmission
multiply) so that ``import scatterem`` never requires Triton -- same arrangement as
:mod:`scatterem.nn.functional._bcast_cmul_triton` and
:mod:`scatterem.nn.functional._conj_mul_reduce_triton`.

Why this exists
---------------
``waves = T.unsqueeze(0) * waves`` is one line, but autograd's ``MulBackward0`` spends
three full-tensor passes producing the ``T`` gradient out of it.  At a
benchmark shape -- wave field ``(4, 256, 128, 128)`` complex64 = 128 MiB, ``T`` slice
32 MiB -- the profiler attributes, per transmit step:

    mul   [4,..]x[4,..]   0.553 ms   grad * conj(waves)
    copy_ [4,..]          0.372 ms   resolve_conj(waves)      <- the conj tax
    sum   [4,..]          0.240 ms   reduce the mode axis
    -------------------   1.165 ms, and x9 slices = 10.5 ms of a 52.5 ms training step

The ``copy_`` is there because ``grad * other.conj()`` has no conj-view support in
TensorIterator: every ``.conj()`` on an autograd path is a ``resolve_conj()``, i.e. a
full read+write of the saved operand.  It is invisible in a forward profile and absent
from the real-dtype version of the same expression.  The ``sum`` then re-reads a 128 MiB
intermediate that was just written.

Compulsory traffic for the whole thing is read ``g`` + read ``w`` + write ``out`` =
288 MiB; the eager chain moves ~800 MiB.  Fusing it into one kernel measured
**1.244 -> 0.482 ms = 2.580x, 583 GiB/s = 94 % of this card's 618 GiB/s in-place
read+write roof** (a benchmark script).

Numerics -- a bit-exactness contract, not a tolerance
----------------------------------------------------
The result is ``torch.equal`` to ``torch.mul(g, w.conj()).sum(0, keepdim=True)``, and that
is checked over 60 shape/mode/magnitude combinations by
a benchmark script rather than assumed.  Two things make it hold:

* **The products are written as explicit FMAs in torch's own contraction pattern.**
  ``c10::complex`` multiply is ``(ar*br - ai*bi, ar*bi + ai*br)`` and nvcc contracts each
  into ``fma(first, second, -(third*fourth))``.  With ``b = conj(w) = (wr, -wi)`` -- an
  exact sign flip -- that is ``fma(gr, wr, gi*wi)`` and ``fma(gr, -wi, gi*wr)``.
  Spelling it with ``tl.math.fma`` pins it instead of leaving it to Triton's contraction
  default; the same lesson as ``_bcast_cmul_triton``, where an un-pinned ``a*c - b*d``
  was bit-exact in one kernel shape and 6.1e-08 off in another.  ATen's sm_120 code
  fuses the other product of the imaginary half (``fma(gi, wr, gr*(-wi))``);
  ``IM_SWAP`` (from :mod:`scatterem.physics._aten_cmul_order`) selects the
  device's order.
* **The accumulation is strictly sequential** ``((p0 + p1) + p2) + p3`` in float32, which
  is what ``at::acc_type<float, true> == float`` and a thread-per-output reduction give.

That second one is *conditional*, and the gates below exist to enforce the condition
rather than hope for it.  a one-off probe sweeps N at each M and finds torch
switches summation strategy -- hence order -- when the reduction stops being
thread-per-output:

    M= 1  exact for every N (2^0..2^22)
    M= 2  exact for every N
    M= 4  exact for N >= 2          (N=1 alone fails: no output parallelism to exploit)
    M= 8  exact for NO N            (torch reduces 8 pairwise/vectorised, we do not)
    M=16  exact for NO N

Bit-exactness matters even though this tensor feeds ``grad_object``, which a check
can hold at a derived ``atol`` rather than a digest: a fused transmit adjoint that
moves one ulp is hard to find later.  A contract that either holds exactly or
declines is worth more than a tolerance that has to be re-argued at every shape.

Gating
------
The fast path declines -- and the caller keeps the torch chain -- unless CUDA, complex64,
both operands contiguous with matching shape, ``M <= _MAX_MODES`` and
``N >= _MIN_ELEMS``.  Both thresholds are mechanism, not tuning, and each has two
independent reasons:

* ``_MAX_MODES = 4`` is where the summation order still matches torch (table above).  It
  is also where ``tl.static_range`` specialisation stays bounded -- M is a compile-time
  constant, so every distinct M is its own compile.  M = 8 simply
  keeps the torch path: no gain, no regression.
* ``_MIN_ELEMS = 2**18`` is where the fused path becomes *faster*.  Measured at M=4:
  N=2^16 0.707x (SLOWER -- Triton's launch overhead against a 2 MiB stack), N=2^18
  1.237x, N=2^20 2.087x, N=2^22 2.582x.  It is also 17 orders of two above the N>=2
  exactness edge, so the numeric contract is nowhere near its boundary.

``SCATTEREM_CONJ_MUL_REDUCE=0`` forces the torch path; that is how the A/B is taken.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from scatterem.nn._backends import (
    CONJ_MUL_BCAST_PRODUCTS,
    CONJ_MUL_REDUCE,
    CONJ_MUL_REDUCE_BCAST,
)
from scatterem.physics._aten_cmul_order import aten_cmul_imag_swapped
from scatterem.utils.env import resolve_backend

__all__ = [
    "conj_mul_bcast_products",
    "conj_mul_bcast_products_declines",
    "conj_mul_reduce",
    "conj_mul_reduce_bcast",
    "conj_mul_reduce_bcast_declines",
    "conj_mul_reduce_bcast_op",
    "BLOCK_DEFAULT",
    "CALLS",
    "CALLS_DUAL",
    "CALLS_PRODUCTS",
]

#: ``[ran, declined]``.  Host-side counters only (no sync, no device work); they exist so
#: a probe can prove the fast path was actually TAKEN rather than infer it from a timing
#: -- a silent decline reads exactly like "no speedup available".
CALLS = [0, 0]

#: Same, for the dual-output entry point :func:`conj_mul_reduce_bcast`.
CALLS_DUAL = [0, 0]

#: 1024 complex elements per program.  Measured at (4,256,128,128): 256 -> 0.493 ms,
#: 512 -> 0.484, 1024 -> 0.482, 2048 -> 0.487 (num_warps=4 throughout).  The curve is
#: flat to 1 % across 512-2048, so this is a plateau, not a tuned edge.
BLOCK_DEFAULT = 1024
_NUM_WARPS = 4

#: See "Gating" in the module docstring -- both are measured crossovers, not tuning.
_MIN_ELEMS = 1 << 18  # elements per mode; below this torch is FASTER (0.707x at 2^16)
_MAX_MODES = 4  # above this torch's summation order changes and exactness is lost


@triton.jit
def _conj_mul_reduce_kernel(
    G,  # (M, N, 2) f32 -- real view of the complex64 upstream gradient  [in]
    W,  # (M, N, 2) f32 -- real view of the complex64 saved operand      [in]
    OUT,  # (N, 2) f32  -- real view of the complex64 reduction output   [out]
    N,
    M: tl.constexpr,
    BLOCK: tl.constexpr,
    IM_SWAP: tl.constexpr,  # ATen's imaginary FMA on this device (_aten_cmul_order)
):
    t = tl.program_id(0)
    # Both operands are read interleaved (2*BLOCK contiguous floats) and split into
    # re/im with tl.split, so every access is a fully coalesced 16-byte transaction --
    # a strided `2*off` / `2*off+1` pair measured the same here but is not guaranteed to.
    o2 = 2 * t * BLOCK + tl.arange(0, 2 * BLOCK)
    m2 = o2 < 2 * N
    accr = tl.zeros((BLOCK,), dtype=tl.float32)
    acci = tl.zeros((BLOCK,), dtype=tl.float32)
    for m in tl.static_range(M):
        o = o2 + 2 * m * N
        gr, gi = tl.split(tl.reshape(tl.load(G + o, mask=m2, other=0.0), (BLOCK, 2)))
        wr, wi = tl.split(tl.reshape(tl.load(W + o, mask=m2, other=0.0), (BLOCK, 2)))
        # torch's ``g * resolve_conj(w)``: see the module docstring.  conj(w) = (wr, -wi)
        # is an exact sign flip, so -(gi * (-wi)) is exactly gi * wi.
        accr = accr + tl.math.fma(gr, wr, gi * wi)
        if IM_SWAP:
            acci = acci + tl.math.fma(gi, wr, gr * (wi * -1.0))
        else:
            acci = acci + tl.math.fma(gr, -wi, gi * wr)
    tl.store(OUT + o2, tl.reshape(tl.join(accr, acci), (2 * BLOCK,)), mask=m2)


@triton.jit
def _conj_mul_reduce_bcast_kernel(
    G,  # (M, N, 2) f32 -- real view of the complex64 upstream gradient       [in]
    W,  # (M, N, 2) f32 -- real view of the saved exit waves                  [in]
    T,  # (N, 2) f32    -- real view of the saved transmission slice          [in]
    OW,  # (M, N, 2) f32 -- real view of grad_waves = g * conj(T)             [out]
    OT,  # (N, 2) f32    -- real view of grad_T = sum_m g[m] * conj(w[m])     [out]
    N,
    M: tl.constexpr,
    BLOCK: tl.constexpr,
    IM_SWAP: tl.constexpr,  # ATen's imaginary FMA on this device (_aten_cmul_order)
):
    """Both halves of the transmit adjoint, reading ``g`` once and holding ``T`` in registers.

    Arithmetic is identical to :func:`_conj_mul_reduce_kernel` for ``OT``; ``OW`` is the
    same FMA pattern against ``T`` instead of ``w``.  See ``conj_mul_reduce_bcast``.
    """
    t = tl.program_id(0)
    o2 = 2 * t * BLOCK + tl.arange(0, 2 * BLOCK)
    m2 = o2 < 2 * N
    # T is loop-invariant: loaded ONCE per program and reused for every mode, so its
    # traffic is 32 MiB rather than the 4x32 a flat broadcast pays (a plane-residency
    # effect -- T is 32 MiB against a 6 MiB L2, so every mode-pass misses).
    tr, ti = tl.split(tl.reshape(tl.load(T + o2, mask=m2, other=0.0), (BLOCK, 2)))
    accr = tl.zeros((BLOCK,), dtype=tl.float32)
    acci = tl.zeros((BLOCK,), dtype=tl.float32)
    for m in tl.static_range(M):
        o = o2 + 2 * m * N
        gr, gi = tl.split(tl.reshape(tl.load(G + o, mask=m2, other=0.0), (BLOCK, 2)))
        wr, wi = tl.split(tl.reshape(tl.load(W + o, mask=m2, other=0.0), (BLOCK, 2)))
        accr = accr + tl.math.fma(gr, wr, gi * wi)
        if IM_SWAP:
            acci = acci + tl.math.fma(gi, wr, gr * (wi * -1.0))
        else:
            acci = acci + tl.math.fma(gr, -wi, gi * wr)
        # torch's ``mul(g, T.unsqueeze(0).conj())``: g is the FIRST operand there, and
        # operand order is bitwise-significant once a product contracts into an FMA
        # (``_bcast_cmul_triton``: torch.equal(w*plane, plane*w) is False).  Same g here.
        gwr = tl.math.fma(gr, tr, gi * ti)
        if IM_SWAP:
            gwi = tl.math.fma(gi, tr, gr * (ti * -1.0))
        else:
            gwi = tl.math.fma(gr, -ti, gi * tr)
        tl.store(OW + o, tl.reshape(tl.join(gwr, gwi), (2 * BLOCK,)), mask=m2)
    tl.store(OT + o2, tl.reshape(tl.join(accr, acci), (2 * BLOCK,)), mask=m2)


def conj_mul_reduce_bcast(
    g: torch.Tensor, w: torch.Tensor, t: torch.Tensor, block: int = BLOCK_DEFAULT
) -> tuple | None:
    """``(sum_m g[m]*conj(w[m]), g*conj(t))`` in one pass, or ``None`` if declined.

    ``g``/``w`` are ``(M, *rest)`` complex64 and ``t`` is ``(*rest,)``; the results are
    ``(1, *rest)`` and ``(M, *rest)``.  This is :func:`conj_mul_reduce` plus the OTHER
    half of the same adjoint, and the reason to compute them together is traffic, not
    tidiness -- ``g`` is the largest operand and both halves need it:

        separate  grad_T (g 128 + w 128 + out 32)  +  grad_waves (g 128 + T 32 + out 128)
                  = 576 MiB at the benchmark shape
        together  g 128 + w 128 + T 32 + grad_waves 128 + grad_T 32 = 448 MiB

    Measured at ``(4,256,128,128)`` complex64 (a benchmark script):
    the shipped pair costs **1.224 ms** (fused grad_T 0.616 + eager grad_waves 0.742,
    the latter only **379 GiB/s** against its own compulsory traffic) and this kernel
    **0.748 ms = 1.635x, 585 GiB/s = 94.6 % of this card's 618 GiB/s in-place roof**.

    Numerics: ``grad_waves`` is ``torch.equal`` to ``torch.mul(g, t.unsqueeze(0).conj())``
    and ``grad_T`` to ``torch.mul(g, w.conj()).sum(0, keepdim=True)``.  For ``grad_waves``
    that holds **unconditionally** rather than in a regime -- it is pure elementwise, and
    the module docstring's ``_MAX_MODES`` caveat is a statement about torch's *summation
    order*, of which there is none here.  The gate below is therefore inherited from the
    ``grad_T`` half, which does reduce.  This matters because ``grad_waves`` carries
    ``grad_probe``: a fused transmit adjoint must not move this tensor by one ulp.
    """
    if resolve_backend(CONJ_MUL_REDUCE_BCAST, None, g, w, t) != "triton":
        CALLS_DUAL[1] += 1
        return None
    CALLS_DUAL[0] += 1
    return _conj_mul_reduce_bcast_launch(g, w, t, block)


def _conj_mul_reduce_bcast_launch(
    g: torch.Tensor, w: torch.Tensor, t: torch.Tensor, block: int = BLOCK_DEFAULT
) -> tuple:
    """Allocate + launch, with NO gate, no env read and no counter.

    Split out of :func:`conj_mul_reduce_bcast` so the eager entry point and the
    ``torch.compile`` one (:func:`conj_mul_reduce_bcast_op`) share ONE launch and cannot
    drift in numerics.  Callers must have checked
    :func:`conj_mul_reduce_bcast_declines` first.

    The custom op cannot decline, so when ATen's imaginary FMA order is unknown
    (:func:`~scatterem.physics._aten_cmul_order.aten_cmul_imag_swapped` is
    ``None``, e.g. a first call under CUDA graph capture) this runs the eager
    expression itself -- the same bits by definition.
    """
    im_swap = aten_cmul_imag_swapped(g.device)
    if im_swap is None:
        return (
            torch.mul(g, w.conj()).sum(0, keepdim=True),
            torch.mul(g, t.unsqueeze(0).conj()),
        )
    nm = g.shape[0]
    n = t.numel()
    grad_t = torch.empty((1,) + tuple(g.shape[1:]), device=g.device, dtype=g.dtype)
    grad_w = torch.empty_like(g)
    _conj_mul_reduce_bcast_kernel[(triton.cdiv(n, block),)](  # pyright: ignore[reportIndexIssue]
        torch.view_as_real(g.view(nm, n)),
        torch.view_as_real(w.view(nm, n)),
        torch.view_as_real(t.view(n)),
        torch.view_as_real(grad_w.view(nm, n)),
        torch.view_as_real(grad_t.view(n)),
        n,
        M=nm,
        BLOCK=block,
        IM_SWAP=im_swap,
        num_warps=_NUM_WARPS,
    )
    return grad_t, grad_w


def conj_mul_reduce_bcast_declines(
    g: torch.Tensor, w: torch.Tensor, t: torch.Tensor
) -> bool:
    """Do ``g``/``w``/``t`` fail any gate of the DUAL kernel?  (Pure tensor metadata.)

    Everything here is shape/dtype/device/contiguity, i.e. exactly what Dynamo can settle
    at trace time as guards -- which is why the compiled route can check the same gate the
    eager route checks instead of a weaker one.  The env switch and the counters, which it
    deliberately does NOT include, are the untraceable half.
    """
    if conj_mul_reduce_declines(g, w):
        return True
    if t.dtype is not torch.complex64 or not t.is_contiguous():
        return True
    return t.shape != g.shape[1:]


def conj_mul_reduce_declines(g: torch.Tensor, w: torch.Tensor) -> bool:
    """Do ``g``/``w`` fail any gate in the module docstring?  (No counters, no allocation.)"""
    return _products_decline(g, w) or g.shape[0] > _MAX_MODES


def _products_decline(g: torch.Tensor, w: torch.Tensor) -> bool:
    """Every gate of :func:`conj_mul_reduce_declines` except ``_MAX_MODES``."""
    if not (g.is_cuda and w.is_cuda):
        return True
    if g.dtype is not torch.complex64 or w.dtype is not torch.complex64:
        return True
    if g.shape != w.shape or g.ndim < 2:
        return True
    if not (g.is_contiguous() and w.is_contiguous()):
        return True
    return g.numel() // g.shape[0] < _MIN_ELEMS


def conj_mul_reduce(
    g: torch.Tensor, w: torch.Tensor, block: int = BLOCK_DEFAULT
) -> torch.Tensor | None:
    """``sum_m g[m] * conj(w[m])`` keeping the reduced axis, or ``None`` if declined.

    ``g`` and ``w`` are ``(M, *rest)`` complex64; the result is ``(1, *rest)``.  Returns
    ``None`` without allocating when any gate in the module docstring fails, so the
    caller must keep its own torch chain as the fallback.  Never raises on a shape it
    does not like -- declining is the contract.
    """
    if resolve_backend(CONJ_MUL_REDUCE, None, g, w) != "triton":
        CALLS[1] += 1
        return None
    im_swap = aten_cmul_imag_swapped(g.device)
    if im_swap is None:
        CALLS[1] += 1
        return None
    nm = g.shape[0]
    n = g.numel() // nm
    out = torch.empty((1,) + tuple(g.shape[1:]), device=g.device, dtype=g.dtype)
    _conj_mul_reduce_kernel[(triton.cdiv(n, block),)](  # pyright: ignore[reportIndexIssue]
        torch.view_as_real(g.view(nm, n)),
        torch.view_as_real(w.view(nm, n)),
        torch.view_as_real(out.view(n)),
        n,
        M=nm,
        BLOCK=block,
        IM_SWAP=im_swap,
        num_warps=_NUM_WARPS,
    )
    CALLS[0] += 1
    return out


# ------------------------------------------ above ``_MAX_MODES``: fuse the products only
# ``_MAX_MODES`` refuses the dual kernel above 4 modes because torch's ``sum(0)`` stops
# reducing in the order the kernel's accumulator does -- but nothing in the refusal is
# about the PRODUCTS.  Above the edge the eager fallback paid, per transmit slice,
# ``resolve_conj(waves)`` (a full read+write), ``grad * conj_waves``, ``resolve_conj(T)``
# and a non-vectorised broadcast ``grad * conj_T``.  This kernel writes both product
# tensors in one pass reading ``g`` once and ``T`` once per program, and the mode
# reduction stays torch's own ``P.sum(0, keepdim=True)`` on the identical ``P`` -- so
# ``grad_T`` is bit-exact by construction at ANY mode count, not in a regime.  The
# per-element FMA spellings are the dual kernel's, which are ``torch.equal`` to torch's
# ``mul(g, w.conj())`` / ``mul(g, t.unsqueeze(0).conj())``.

#: ``[ran, declined]`` for :func:`conj_mul_bcast_products`.
CALLS_PRODUCTS = [0, 0]


@triton.jit
def _conj_mul_bcast_products_kernel(
    G,  # (M, N, 2) f32 -- real view of the complex64 upstream gradient       [in]
    W,  # (M, N, 2) f32 -- real view of the saved exit waves                  [in]
    T,  # (N, 2) f32    -- real view of the saved transmission slice          [in]
    OW,  # (M, N, 2) f32 -- real view of grad_waves = g * conj(T)             [out]
    OP,  # (M, N, 2) f32 -- real view of the products g[m] * conj(w[m])       [out]
    N,
    M: tl.constexpr,
    BLOCK: tl.constexpr,
    IM_SWAP: tl.constexpr,  # ATen's imaginary FMA on this device (_aten_cmul_order)
):
    t = tl.program_id(0)
    o2 = 2 * t * BLOCK + tl.arange(0, 2 * BLOCK)
    m2 = o2 < 2 * N
    tr, ti = tl.split(tl.reshape(tl.load(T + o2, mask=m2, other=0.0), (BLOCK, 2)))
    for m in tl.static_range(M):
        o = o2 + 2 * m * N
        gr, gi = tl.split(tl.reshape(tl.load(G + o, mask=m2, other=0.0), (BLOCK, 2)))
        wr, wi = tl.split(tl.reshape(tl.load(W + o, mask=m2, other=0.0), (BLOCK, 2)))
        pr = tl.math.fma(gr, wr, gi * wi)
        if IM_SWAP:
            pi = tl.math.fma(gi, wr, gr * (wi * -1.0))
        else:
            pi = tl.math.fma(gr, -wi, gi * wr)
        tl.store(OP + o, tl.reshape(tl.join(pr, pi), (2 * BLOCK,)), mask=m2)
        gwr = tl.math.fma(gr, tr, gi * ti)
        if IM_SWAP:
            gwi = tl.math.fma(gi, tr, gr * (ti * -1.0))
        else:
            gwi = tl.math.fma(gr, -ti, gi * tr)
        tl.store(OW + o, tl.reshape(tl.join(gwr, gwi), (2 * BLOCK,)), mask=m2)


def conj_mul_bcast_products_declines(
    g: torch.Tensor, w: torch.Tensor, t: torch.Tensor
) -> bool:
    """Do ``g``/``w``/``t`` fail a gate of :func:`conj_mul_bcast_products`?

    The dual kernel's gates (:func:`conj_mul_reduce_bcast_declines`) without
    ``_MAX_MODES``: there is no reduction here.  Pure tensor metadata.
    """
    if _products_decline(g, w):
        return True
    if t.dtype is not torch.complex64 or not t.is_contiguous():
        return True
    return t.shape != g.shape[1:]


def conj_mul_bcast_products(
    g: torch.Tensor, w: torch.Tensor, t: torch.Tensor, block: int = BLOCK_DEFAULT
) -> tuple | None:
    """``(g*conj(w), g*conj(t))`` in one pass, or ``None`` if declined.

    The caller reduces the first result with ``.sum(0, keepdim=True)`` itself, so
    that reduction is torch's own; both products are ``torch.equal`` to torch's
    ``mul``.  Asks the ``conj_mul_bcast_products`` backend row.
    """
    if resolve_backend(CONJ_MUL_BCAST_PRODUCTS, None, g, w, t) != "triton":
        CALLS_PRODUCTS[1] += 1
        return None
    im_swap = aten_cmul_imag_swapped(g.device)
    if im_swap is None:
        CALLS_PRODUCTS[1] += 1
        return None
    CALLS_PRODUCTS[0] += 1
    nm = g.shape[0]
    n = t.numel()
    prod = torch.empty_like(g)
    grad_w = torch.empty_like(g)
    _conj_mul_bcast_products_kernel[(triton.cdiv(n, block),)](  # pyright: ignore[reportIndexIssue]
        torch.view_as_real(g.view(nm, n)),
        torch.view_as_real(w.view(nm, n)),
        torch.view_as_real(t.view(n)),
        torch.view_as_real(grad_w.view(nm, n)),
        torch.view_as_real(prod.view(nm, n)),
        n,
        M=nm,
        BLOCK=block,
        IM_SWAP=im_swap,
        num_warps=_NUM_WARPS,
    )
    return prod, grad_w


# ----------------------------------------------------- the torch.compile entry point
# The first fused adjoint was EAGER-ONLY: ``_fused_transmit_adjoint_eligible`` declined
# under ``torch.compiler.is_compiling()``, so the compiled route paid the unfused
# ``MulBackward0`` chain.  What Dynamo cannot trace is not the kernel -- it is the host
# Python wrapped around it: an ``os.environ`` read, a mutated global counter list, and an
# ``Optional`` return whose ``None`` is a data-dependent branch.  An opaque
# ``custom_op`` has a fixed schema and is never traced INTO, so all three problems go
# away at once and the compiled backward captures the same kernel the eager one runs.
#
# This matters more than a normal "also works under compile" note, because the measured
# compile winner for this model is ``backend="cudagraphs"`` -- a CUDA-graph replay of
# EAGER kernels (Inductor arms are ~0.7x here: it has no complex codegen, so every
# complex op falls back with a full HBM round-trip).  A graph replay is only as fast as
# the kernels captured into it, so declining the fusion at trace time made the fusion
# unreachable for the route that is otherwise fastest.
#
# NOT differentiable, deliberately: this is only ever called from inside
# ``_BroadcastTransmit.backward``, so registering an autograd formula would be dead code
# claiming double-backward support that the eager path does not have either.


@torch.library.custom_op("scatterem::conj_mul_reduce_bcast", mutates_args=())
def conj_mul_reduce_bcast_op(
    g: torch.Tensor, w: torch.Tensor, t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(sum_m g[m]*conj(w[m]), g*conj(t))`` as an opaque op -- see the comment above.

    Bit-identical to :func:`conj_mul_reduce_bcast` (same
    :func:`_conj_mul_reduce_bcast_launch`), but it does NOT decline: the caller must have
    checked :func:`conj_mul_reduce_bcast_declines`, which is pure tensor metadata and so
    is settled at trace time.  ``BLOCK_DEFAULT`` is not a parameter here because a
    compiled graph has no way to vary it per call.
    """
    return _conj_mul_reduce_bcast_launch(g, w, t, BLOCK_DEFAULT)


@conj_mul_reduce_bcast_op.register_fake
def _(g: torch.Tensor, w: torch.Tensor, t: torch.Tensor):
    return (
        torch.empty((1,) + tuple(g.shape[1:]), device=g.device, dtype=g.dtype),
        torch.empty_like(g),
    )
