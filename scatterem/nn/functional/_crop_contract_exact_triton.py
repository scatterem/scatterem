"""Bit-exact fused crop contraction for the NO-GRAD PRISM forward block body.

Private, lazily imported by its one call site
(``scatterem.nn.functional.warp.prism_project._torch_blocks._contract_blocks``,
the block loop of ``_prism_project_torch``) so that ``import scatterem`` never
requires Triton -- same arrangement as
:mod:`scatterem.nn.functional._crop_contract_triton`.

Why this exists
---------------
``_prism_project_torch``'s no-grad loop runs, per block of ``nb`` parent beams and
``K`` positions::

    S_dt_crop = Sdt_flat[blk].index_select(1, idx).view(nb, K, M1, M2)
    out[:, ksl] += (basis_k * S_dt_crop[None]).sum(1)

The crop (128 MiB per block at ``r2_prism_project/medium``) is written only for the
product to read it, and the product (another 128 MiB) only for the sum to read it.
This kernel reads each crop value straight from ``Sdt_flat`` through ``idx``, forms
the product in registers, sums over the block's ``p`` and adds the running
``exit_wave`` -- one read of ``basis_k``, one gather of ``Sdt``, one write.

Numerics -- a bit-exactness contract, not a tolerance
----------------------------------------------------
``crop_contract`` (the grad path's kernel) sums ``p`` in two levels and is NOT
``torch.equal`` to the eager body, which is why the no-grad path never took it
(r3's PRISM cube references are sha256).  This one is:

* each term is spelled the way ATen's complex64 ``mul`` rounds ``basis * crop``
  (``basis`` first): ``fma(br, cr, -(bi*ci))``, ``fma(br, ci, bi*cr)`` -- the same
  spelling as ``crop_contract`` and ``ramp_adjoint_ksum``;
* the sum over ``p`` follows TensorIterator's outer-dimension thread reduce
  (``vt0 = 4``): four accumulators seeded with ``0``, term ``p`` added to
  accumulator ``p % 4`` in ``p`` order, then combined ``((a0 + a1) + a2) + a3``
  (measured).  Padding terms past ``nb`` are ``+0.0``; an accumulator
  seeded with ``+0`` can never hold ``-0``, so adding them is an identity;
* the running sum is ``out + part``, one complex add, as eager does it.

ATen keeps that order only while ONE thread owns all ``nb`` terms of an output.
``setReduceConfig`` splits the input across threads once the per-thread value
count reaches ``16 * block_height`` (64 at the vectorised outer config, block
32x4).  From ``nb = 64`` the kernel follows that split too: four
"threads" ``y`` each sum ``p = y, y + 4, ...`` in the order above, then
``block_y_reduce`` combines them ``(T0 + T2) + (T1 + T3)``.  At
``ceil(nb / 4) >= 256`` ATen also splits across CTAs, so the kernel declines
``nb >= 1021`` (and the split path needs ``K * npx >= 512`` so block.y is 4).
``r2_prism_project/medium`` runs ``PB = 32``; its ``project_chunked``, r2
``small`` and the r3 PRISM blocks run ``PB >= 64``.

Gating
------
Declines -- and the caller keeps the eager body -- unless everything is CUDA
``complex64`` and contiguous, ``Nm == 1``, ``basis`` is ``(1, nb, 1 or K, *P)``
against a ``(nb, SY*SX)`` S-matrix block and a flat int32 index of ``K*prod(P)``
entries, ``1 <= nb < 1021``, the plane is a whole number of tiles, offsets fit in
int32, and nothing needs a gradient or is being compiled.  The gate is the
``crop_contract_exact`` backend row's predicate,
:func:`crop_contract_exact_supported`.

``basis`` may be a ``(p, k)``-permuted view (the torch route's K-outer
``ifft2`` output); each plane must be contiguous, the ``p``/``k`` strides are
kernel arguments.  The summation order is the kernel's, not the layout's, so the
bits do not depend on it.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from scatterem.nn._backends import CROP_CONTRACT_EXACT
from scatterem.physics._aten_cmul_order import aten_cmul_imag_swapped
from scatterem.utils.env import resolve_backend

__all__ = ["CALLS", "crop_contract_exact", "crop_contract_exact_supported"]

#: ``[ran, declined]``, host-side only -- proves the fast path was TAKEN.
CALLS = [0, 0]

BLOCK_DEFAULT = 128
KB_DEFAULT = 2
#: ATen splits the reduction across threads from here on (see the docstring).
_MAX_NB = 64
#: ATen's block-y split stays 4 threads until ``ceil(nb / 4)`` reaches 256,
#: where it also splits the input across CTAs (verified: 1020 exact, 1021 not).
_MAX_NB_SPLIT = 1021


@triton.jit
def _psum4(
    S,
    B,
    off_g,
    off_b,
    m2,
    NBP,
    SYX,
    BP_STRIDE,
    START,
    STEP: tl.constexpr,
    KB: tl.constexpr,
    BLOCK: tl.constexpr,
    IM_SWAP: tl.constexpr,
):
    """Sum of the terms ``p = START, START + STEP, ...`` in ATen's thread order.

    The thread's ``j``-th term goes to accumulator ``j % 4``; the four are then
    combined ``((a0 + a1) + a2) + a3`` (TensorIterator's ``thread_reduce``).
    """
    a0r = tl.zeros((KB, BLOCK), dtype=tl.float32)
    a0i = tl.zeros((KB, BLOCK), dtype=tl.float32)
    a1r = tl.zeros((KB, BLOCK), dtype=tl.float32)
    a1i = tl.zeros((KB, BLOCK), dtype=tl.float32)
    a2r = tl.zeros((KB, BLOCK), dtype=tl.float32)
    a2i = tl.zeros((KB, BLOCK), dtype=tl.float32)
    a3r = tl.zeros((KB, BLOCK), dtype=tl.float32)
    a3i = tl.zeros((KB, BLOCK), dtype=tl.float32)
    for p0 in range(START, NBP, 4 * STEP):
        for j in tl.static_range(4):
            p = p0 + j * STEP
            pm = p < NBP
            sf = tl.load(S + off_g + 2 * p * SYX, mask=m2 & pm, other=0.0)
            cr, ci = tl.split(tl.reshape(sf, (KB, BLOCK, 2)))
            bf = tl.load(
                B + off_b + 2 * p * BP_STRIDE,
                mask=m2 & pm,
                other=0.0,
                eviction_policy="evict_first",
            )
            br, bi = tl.split(tl.reshape(bf, (KB, BLOCK, 2)))
            # ATen's ``basis * crop``; ``* -1.0`` is an exact sign flip (Triton
            # lowers unary minus to ``0 - x``).  ``IM_SWAP``: the
            # imaginary FMA of this device's ATen build (_aten_cmul_order).
            tr = tl.math.fma(br, cr, (bi * ci) * -1.0)
            if IM_SWAP:
                ti = tl.math.fma(bi, cr, br * ci)
            else:
                ti = tl.math.fma(br, ci, bi * cr)
            if j == 0:
                a0r = a0r + tr
                a0i = a0i + ti
            elif j == 1:
                a1r = a1r + tr
                a1i = a1i + ti
            elif j == 2:
                a2r = a2r + tr
                a2i = a2i + ti
            else:
                a3r = a3r + tr
                a3i = a3i + ti
    return ((a0r + a1r) + a2r) + a3r, ((a0i + a1i) + a2i) + a3i


@triton.jit
def _crop_contract_exact_kernel(
    Out,  # (K, PLANE, 2) f32         -- real view of the output        [in/out]
    B,  # (NBP, KB_ALL, PLANE, 2) f32 -- real view of the basis block   [in]
    S,  # (NBP, SYX, 2) f32           -- real view of the S-matrix rows [in]
    IDX,  # (K, PLANE) int32 flat crop index into SYX                   [in]
    PLANE,
    NBP,
    K,
    SYX,
    BK_STRIDE,  # 0 for the K-broadcast basis, else PLANE
    BP_STRIDE,  # basis stride over p, in complex elements
    ACC: tl.constexpr,
    SPLIT: tl.constexpr,
    KB: tl.constexpr,
    BLOCK: tl.constexpr,
    IM_SWAP: tl.constexpr,  # ATen's imaginary FMA on this device (_aten_cmul_order)
):
    t = tl.program_id(0)  # which plane tile
    kb = tl.program_id(1)  # which block of positions
    ks = kb * KB + tl.arange(0, KB)
    kmask = ks < K
    pix = t * BLOCK + tl.arange(0, BLOCK)
    pix2 = 2 * (t * BLOCK) + tl.arange(0, 2 * BLOCK)
    m1 = kmask[:, None] & (pix[None, :] >= 0)
    m2 = kmask[:, None] & (pix2[None, :] >= 0)
    g = tl.load(IDX + ks[:, None] * PLANE + pix[None, :], mask=m1, other=0)
    g2 = 2 * g
    off_g = tl.reshape(tl.join(g2, g2 + 1), (KB, 2 * BLOCK))
    off_b = 2 * ks[:, None] * BK_STRIDE + pix2[None, :]
    if SPLIT == 1:
        accr, acci = _psum4(
            S, B, off_g, off_b, m2, NBP, SYX, BP_STRIDE, 0, 1, KB, BLOCK, IM_SWAP
        )
    else:
        # ``nb >= 64``: ATen's block-y split over 4 threads -- thread ``y``
        # owns ``p = y, y + 4, ...``, then ``block_y_reduce``'s shared-memory tree
        # ``(T0 + T2) + (T1 + T3)``.
        t0r, t0i = _psum4(
            S, B, off_g, off_b, m2, NBP, SYX, BP_STRIDE, 0, 4, KB, BLOCK, IM_SWAP
        )
        t1r, t1i = _psum4(
            S, B, off_g, off_b, m2, NBP, SYX, BP_STRIDE, 1, 4, KB, BLOCK, IM_SWAP
        )
        t2r, t2i = _psum4(
            S, B, off_g, off_b, m2, NBP, SYX, BP_STRIDE, 2, 4, KB, BLOCK, IM_SWAP
        )
        t3r, t3i = _psum4(
            S, B, off_g, off_b, m2, NBP, SYX, BP_STRIDE, 3, 4, KB, BLOCK, IM_SWAP
        )
        accr = (t0r + t2r) + (t1r + t3r)
        acci = (t0i + t2i) + (t1i + t3i)
    off_o = 2 * ks[:, None] * PLANE + pix2[None, :]
    res = tl.reshape(tl.join(accr, acci), (KB, 2 * BLOCK))
    if ACC:
        res = tl.load(Out + off_o, mask=m2, other=0.0) + res
    tl.store(Out + off_o, res, mask=m2)


def _operands_ok(basis, sdt, idx, out) -> bool:
    tensors = (basis, sdt, idx) + (() if out is None else (out,))
    if not all(t.is_cuda for t in tensors):
        return False
    if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        return False
    if basis.dtype is not torch.complex64 or sdt.dtype is not torch.complex64:
        return False
    # ``view_as_real`` refuses a lazily-conjugated tensor and would drop a neg bit.
    if any(t.is_conj() or t.is_neg() for t in (basis, sdt) + tensors[3:]):
        return False
    return idx.dtype is torch.int32 and basis.ndim >= 4 and sdt.ndim == 2


def _shapes_ok(basis, sdt, idx, K: int, out, block: int) -> bool:
    nm, nbp, kb_all = basis.shape[:3]
    npx = basis[0, 0, 0].numel()
    split = 1 if nbp < _MAX_NB else 4
    if (
        nm != 1
        or not 1 <= nbp < _MAX_NB_SPLIT
        # ATen's 32x4 block needs ``K * npx / 4 >= 128`` outputs; fewer and
        # block.y grows past 4, so the split order changes.
        or (split == 4 and K * npx < 512)
        or kb_all not in (1, K)
        or sdt.shape[0] != nbp
        or idx.numel() != K * npx
    ):
        return False
    plane_ok = basis.stride(-1) == 1 and (
        basis.ndim < 5 or basis.stride(-2) == basis.shape[-1]
    )
    if not (plane_ok and sdt.is_contiguous() and idx.is_contiguous()):
        return False
    oshape = (1, K) + tuple(basis.shape[3:])
    if out is not None and (
        tuple(out.shape) != oshape
        or out.dtype is not torch.complex64
        or not out.is_contiguous()
    ):
        return False
    return not (
        npx == 0
        or K == 0
        or npx % block != 0
        or 2 * sdt.numel() >= 2**31
        or 2 * basis.numel() >= 2**31
        # the eager product ATen reduces must be 32-bit indexable, else it
        # splits the iterator (possibly along ``p``)
        or 8 * nbp * K * npx >= 2**31
    )


def crop_contract_exact_supported(
    basis: torch.Tensor,
    sdt: torch.Tensor,
    idx: torch.Tensor,
    K: int,
    out: torch.Tensor | None = None,
    block: int = BLOCK_DEFAULT,
) -> bool:
    """True if :func:`crop_contract_exact` may replace the eager block body."""
    if torch.compiler.is_compiling():
        return False
    return (
        _operands_ok(basis, sdt, idx, out)
        and _shapes_ok(basis, sdt, idx, K, out, block)
        and aten_cmul_imag_swapped(basis.device) is not None
    )


def crop_contract_exact(
    basis: torch.Tensor,
    sdt: torch.Tensor,
    idx: torch.Tensor,
    K: int,
    out: torch.Tensor | None = None,
    block: int = BLOCK_DEFAULT,
    kb: int = KB_DEFAULT,
) -> torch.Tensor | None:
    """``(basis * sdt.index_select(1, idx).view(nb, K, *P)[None]).sum(1)`` [+ ``out``].

    ``basis`` is ``(1, nb, Kb, *P)`` with ``Kb`` either 1 (broadcast over the
    positions) or ``K``; ``sdt`` is the block's ``(nb, SY*SX)`` S-matrix rows and
    ``idx`` the flat int32 crop index.  With ``out`` given, the block's sum is
    added to it IN PLACE (``out + part``) and ``out`` is returned; otherwise a new
    ``(1, K, *P)`` tensor holding ``part`` is.  Bit-identical to the eager body
    (module docstring).  Returns ``None`` without launching anything when the
    ``crop_contract_exact`` row declines -- declining is the contract.
    """
    args = (basis, sdt, idx, K, out, block)
    if resolve_backend(CROP_CONTRACT_EXACT, None, *args) != "triton":
        CALLS[1] += 1
        return None
    nbp, kb_all = basis.shape[1:3]
    npx = basis[0, 0, 0].numel()
    acc = out is not None
    if out is None:
        out = torch.empty(
            (1, K) + tuple(basis.shape[3:]), dtype=torch.complex64, device=basis.device
        )
    _crop_contract_exact_kernel[(npx // block, triton.cdiv(K, kb))](  # pyright: ignore[reportIndexIssue]
        torch.view_as_real(out[0].view(K, npx)),
        torch.view_as_real(basis[0]),
        torch.view_as_real(sdt),
        idx,
        npx,
        nbp,
        K,
        sdt.shape[1],
        0 if kb_all == 1 else basis.stride(2),
        basis.stride(1),
        ACC=acc,
        SPLIT=1 if nbp < _MAX_NB else 4,
        KB=kb,
        BLOCK=block,
        IM_SWAP=aten_cmul_imag_swapped(basis.device),
        num_warps=4,
    )
    CALLS[0] += 1
    return out
