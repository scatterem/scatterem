"""Fused crop contraction ``out[m, k] = sum_p basis[m, p, k] * crop[p, k]``.

Private, lazily imported by its one call site
(``scatterem.nn.functional.warp.prism_project._torch_adjoint``'s
``_CropContractFrozenBasis.forward``)
so that ``import scatterem`` never requires Triton -- same arrangement as
:mod:`scatterem.nn.functional._bcast_mul_conj_triton`.

Why this exists
---------------
On the graph-building PRISM projection (the reconstruction direction) the forward
contraction is spelled ``(basis_k * S_dt_crop[None]).sum(1)``: the full
``(Nmodes, Bp, K, M1, M2)`` product is written (2 GiB at ``r2_prism_project/medium``)
and then read back by the sum over parent beams ``p``.  The product exists only to
be reduced.  This kernel forms each term in registers and accumulates over ``p``,
so it reads the crop and the basis once and writes only the ``(Nmodes, K, M1, M2)``
exit wave.  On the integer-pixel scan the basis is a stride-0 K-broadcast
``(Nmodes, Bp, 1, M1, M2)``; one basis tile is then loaded per ``p`` and reused
across the program's ``K`` rows in registers.

Numerics -- tolerance, not bit-exact
------------------------------------
Each term is spelled as torch's ``basis * crop`` compiles (the broadcast operand
first: the ``im*im`` product rounded, one FMA per half, pinned with ``tl.math.fma``;
the imaginary FMA follows the device's ATen build, see
:mod:`scatterem.physics._aten_cmul_order`).
The terms are then added in ``p`` order in float32, in two levels (runs of
``PC`` beams summed into a partial, the partials into the total).  TensorIterator's
reduction over ``Bp`` terms is not sequential (``_conj_mul_reduce_triton`` measured
its order changing above 4 terms), so the result is NOT ``torch.equal`` to the
eager expression.  It is deterministic, and independent of ``K`` (each output
element sums over ``p`` alone), so scan chunking cannot change it.  Only the grad
path reaches it; the no-grad forward keeps its blocked eager loop.

Gating
------
Declines -- and the caller keeps the eager expression -- unless both operands are
CUDA ``complex64`` with no lazy conj / neg bit, the crop is contiguous and the
basis has contiguous planes (its ``p`` / ``k`` strides are free), the basis is
``(Nm, Bp, 1 or K, *P)`` against a ``(Bp, K, *P)`` crop, the flat float offsets fit
in int32, and nothing is being compiled.  The gate is the ``crop_contract`` backend
row's predicate, :func:`crop_contract_supported`.

Gather mode
------------------
With ``crop=None`` the kernel reads each crop value as ``sdt[p, idx[k, pix]]``
instead of from a materialised crop, so the ``(Bp, K, M1, M2)`` ``index_select``
(a 2 GiB write plus its read-back at ``r2_prism_project/medium``) disappears.  The
values are the same floats, so the output is ``torch.equal`` to the crop mode.  The
frozen-probe call site uses it because its backward never reads the crop; the joint
one takes it too, since ``grad_basis`` learnt to gather the crop the same
way (:mod:`scatterem.nn.functional._bcast_mul_conj_triton`).  It additionally needs
a contiguous ``(Bp, SY*SX)`` ``sdt`` and a contiguous int32 ``idx``; a declined
gather falls back to the crop mode at the call site.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from scatterem.nn._backends import CROP_CONTRACT
from scatterem.physics._aten_cmul_order import aten_cmul_imag_swapped
from scatterem.utils.env import resolve_backend

__all__ = ["CALLS", "GATHER_CALLS", "crop_contract", "crop_contract_supported"]

#: ``[ran, declined]``, host-side only -- proves the fast path was TAKEN.
CALLS = [0, 0]
#: runs that took the index-gather mode (a subset of ``CALLS[0]``).
GATHER_CALLS = [0]

BLOCK_DEFAULT = 256
KB_DEFAULT = 8
PC_DEFAULT = 16
# the gather mode's loads are scalar (the offsets are data), so it needs
# ~8x the programs to cover their latency: at ``r2_prism_project/medium`` the
# kernel reads 45.99 ms at the crop mode's 256x8 tile (256 programs), 12.71 at
# 128x8, and 6.25-6.36 ms for every tile from 64x2 to 256x1 (>= 2048 programs).
GATHER_BLOCK_DEFAULT = 128
GATHER_KB_DEFAULT = 2


@triton.jit
def _crop_contract_kernel(
    Out,  # (K, PLANE, 2) f32        -- real view of one mode's output  [out]
    B,  # (Bp, KB_ALL, PLANE, 2)   -- real view of one mode's basis   [in]
    S,  # (Bp, K, PLANE, 2) f32 crop, or (Bp, SYX, 2) f32 S-matrix if GATHER
    IDX,  # (K, PLANE) int32 flat crop index into SYX (read only if GATHER)
    PLANE,
    NBP,
    K,
    SYX,
    BK_STRIDE,  # 0 for the K-broadcast basis, else PLANE
    BP_STRIDE,  # basis stride over p, in complex elements
    KBCAST: tl.constexpr,
    GATHER: tl.constexpr,
    KB: tl.constexpr,
    PC: tl.constexpr,
    BLOCK: tl.constexpr,
    IM_SWAP: tl.constexpr,  # ATen's imaginary FMA on this device (_aten_cmul_order)
):
    t = tl.program_id(0)  # which plane tile
    kb = tl.program_id(1)  # which block of positions
    ks = kb * KB + tl.arange(0, KB)
    kmask = ks < K
    pix2 = 2 * (t * BLOCK) + tl.arange(0, 2 * BLOCK)
    off_s = 2 * ks[:, None] * PLANE + pix2[None, :]  # (KB, 2*BLOCK)
    m2 = kmask[:, None] & (pix2[None, :] >= 0)
    if GATHER:
        # the crop element S_dt_crop[p, k, pix] IS Sdt[p, IDX[k, pix]], so
        # the index is loaded once and each term's crop value read straight from
        # the S-matrix -- the (Bp, K, M1, M2) crop is never written or read back.
        pix = t * BLOCK + tl.arange(0, BLOCK)
        m1 = kmask[:, None] & (pix[None, :] >= 0)
        g = tl.load(IDX + ks[:, None] * PLANE + pix[None, :], mask=m1, other=0)
        g2 = 2 * g
        off_g = tl.reshape(tl.join(g2, g2 + 1), (KB, 2 * BLOCK))
    accr = tl.zeros((KB, BLOCK), dtype=tl.float32)
    acci = tl.zeros((KB, BLOCK), dtype=tl.float32)
    # Two-level sum: PC consecutive beams into a partial, partials into the total.
    # A flat sequential sum over all Bp terms measured ~7x eager's error at
    # ``r2_prism_project/medium`` (Bp = 512).  Beams past NBP load as zero.
    for p0 in range(0, NBP, PC):
        parr = tl.zeros((KB, BLOCK), dtype=tl.float32)
        pari = tl.zeros((KB, BLOCK), dtype=tl.float32)
        for j in tl.static_range(PC):
            p = p0 + j
            pm = p < NBP
            if GATHER:
                sf = tl.load(
                    S + off_g + 2 * p * SYX,  # pyright: ignore[reportPossiblyUnboundVariable]
                    mask=m2 & pm,
                    other=0.0,
                )
            else:
                sf = tl.load(S + off_s + 2 * p * K * PLANE, mask=m2 & pm, other=0.0)
            wr, wi = tl.split(tl.reshape(sf, (KB, BLOCK, 2)))
            if KBCAST:
                bf = tl.load(
                    B + pix2 + 2 * p * BP_STRIDE,
                    mask=(pix2 >= 0) & pm,
                    other=0.0,
                    eviction_policy="evict_first",
                )
                pr1, pi1 = tl.split(tl.reshape(bf, (BLOCK, 2)))
                pr = pr1[None, :]
                pi = pi1[None, :]
            else:
                off_b = 2 * ks[:, None] * BK_STRIDE + pix2[None, :]
                bf = tl.load(B + off_b + 2 * p * BP_STRIDE, mask=m2 & pm, other=0.0)
                pr, pi = tl.split(tl.reshape(bf, (KB, BLOCK, 2)))
            # ``basis * crop`` as torch compiles it: the basis (first operand) leads.
            # ``* -1.0`` is a sign flip; Triton lowers unary minus to ``0 - x``.
            parr += tl.math.fma(pr, wr, (pi * wi) * -1.0)
            if IM_SWAP:
                pari += tl.math.fma(pi, wr, pr * wi)
            else:
                pari += tl.math.fma(pr, wi, pi * wr)
        accr += parr
        acci += pari
    tl.store(Out + off_s, tl.reshape(tl.join(accr, acci), (KB, 2 * BLOCK)), mask=m2)


def _operands_ok(basis, src, shape):
    if not (basis.is_cuda and src.is_cuda):
        return False
    if basis.dtype is not torch.complex64 or src.dtype is not torch.complex64:
        return False
    # ``view_as_real`` refuses a lazily-conjugated tensor and would drop a neg bit.
    if basis.is_conj() or basis.is_neg() or src.is_conj() or src.is_neg():
        return False
    if basis.ndim < 4 or len(shape) != basis.ndim - 1:
        return False
    nbp, kb_all = basis.shape[1:3]
    K = shape[1]
    if shape[0] != nbp or shape[2:] != tuple(basis.shape[3:]) or kb_all not in (1, K):
        return False
    # the basis only needs contiguous planes; its p and k strides are
    # kernel arguments, so the K-outer view of the frozen-probe grad block (the
    # Warp ``_ramp_mul`` output, permuted) is read in place.  A contiguous basis
    # passes the same two strides as before.
    return (
        src.is_contiguous()
        and basis[0, 0, 0].is_contiguous()
        and min(basis.stride()[1:3]) >= 0
    )


def _span(basis) -> int:
    """Largest element offset + 1 of one mode (= its numel when contiguous)."""
    return 1 + sum(
        (n - 1) * s for n, s in zip(basis.shape[1:], basis.stride()[1:], strict=True)
    )


def _gather_ok(src, idx, nbp, n_idx):
    return (
        idx is not None
        and src.ndim == 2
        and src.shape[0] == nbp
        and idx.device == src.device
        and idx.dtype is torch.int32
        and idx.is_contiguous()
        and idx.numel() == n_idx
    )


def _block(gather: bool, block: int | None) -> int:
    if block is not None:
        return block
    return GATHER_BLOCK_DEFAULT if gather else BLOCK_DEFAULT


def crop_contract_supported(
    basis: torch.Tensor,
    crop: torch.Tensor | None,
    sdt: torch.Tensor | None = None,
    idx: torch.Tensor | None = None,
    crop_shape: tuple[int, ...] | None = None,
    block: int | None = None,
) -> bool:
    """True if :func:`crop_contract` may replace the eager product + ``p``-sum."""
    if torch.compiler.is_compiling():
        return False
    gather = crop is None
    src = sdt if gather else crop
    shape = crop_shape if gather else (None if crop is None else tuple(crop.shape))
    if src is None or shape is None:
        return False
    shape = tuple(shape)
    if not _operands_ok(basis, src, shape):
        return False
    nbp = basis.shape[1]
    K = shape[1]
    npx = basis[0, 0, 0].numel()
    if gather and not _gather_ok(src, idx, nbp, K * npx):
        return False
    if aten_cmul_imag_swapped(basis.device) is None:
        return False
    return not (
        npx == 0
        or nbp == 0
        or K == 0
        or npx % _block(gather, block) != 0
        or 2 * src.numel() >= 2**31
        or 2 * _span(basis) >= 2**31
    )


def crop_contract(
    basis: torch.Tensor,
    crop: torch.Tensor | None,
    block: int | None = None,
    kb: int | None = None,
    pc: int = PC_DEFAULT,
    *,
    sdt: torch.Tensor | None = None,
    idx: torch.Tensor | None = None,
    crop_shape: tuple[int, ...] | None = None,
) -> torch.Tensor | None:
    """``(basis * crop[None]).sum(1)`` as a new tensor, or ``None``.

    ``basis`` is ``(Nm, Bp, Kb, *P)`` with ``Kb`` either 1 (broadcast over the
    positions) or ``K``; ``crop`` is ``(Bp, K, *P)``.  The result is
    ``(Nm, K, *P)``, contiguous.  Returns ``None`` without launching anything when
    a gate in the module docstring fails -- declining is the contract.

    With ``crop=None`` the crop is read through its index instead:
    ``crop == sdt.index_select(1, idx).view(crop_shape)`` for ``sdt`` of shape
    ``(Bp, SY*SX)`` and a flat int32 ``idx`` of ``K*prod(P)`` entries.  The terms,
    and therefore the result, are bit-identical to passing the materialised crop,
    because a gather copies values exactly.
    """
    gather = crop is None
    block = _block(gather, block)
    if kb is None:
        kb = GATHER_KB_DEFAULT if gather else KB_DEFAULT
    args = (basis, crop, sdt, idx, crop_shape, block)
    if resolve_backend(CROP_CONTRACT, None, *args) != "triton":
        CALLS[1] += 1
        return None
    shape = tuple(crop_shape if gather else crop.shape)  # pyright: ignore[reportArgumentType]
    nm, nbp, kb_all = basis.shape[:3]
    K = shape[1]
    npx = basis[0, 0, 0].numel()
    out = torch.empty((nm, K) + shape[2:], dtype=torch.complex64, device=basis.device)
    kbcast = kb_all == 1 and K > 1
    if gather:
        s_real = torch.view_as_real(sdt)  # pyright: ignore[reportArgumentType]
        idx_arg = idx
        syx = sdt.shape[1]  # pyright: ignore[reportOptionalMemberAccess]
    else:
        s_real = torch.view_as_real(crop.view(nbp, K, npx))
        idx_arg = s_real  # unread: GATHER is a constexpr False
        syx = 0
    for m in range(nm):
        _crop_contract_kernel[(npx // block, triton.cdiv(K, kb))](  # pyright: ignore[reportIndexIssue]
            torch.view_as_real(out[m].view(K, npx)),
            torch.view_as_real(basis[m]),
            s_real,
            idx_arg,
            npx,
            nbp,
            K,
            syx,
            0 if kb_all == 1 else basis.stride(2),
            basis.stride(1) if nbp > 1 else 0,
            KBCAST=kbcast,
            GATHER=gather,
            KB=kb,
            PC=pc,
            BLOCK=block,
            IM_SWAP=aten_cmul_imag_swapped(basis.device),
            num_warps=4,
        )
    CALLS[0] += 1
    GATHER_CALLS[0] += gather
    return out
