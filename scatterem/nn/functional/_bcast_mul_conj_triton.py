"""Broadcast conjugate product ``out[m, b] = plane[m] * conj(stack[b])``.

Private, lazily imported by its one call site
(``scatterem.nn.functional.warp.prism_project._CropContractFrozenBasis.backward``)
so that ``import scatterem`` never requires Triton -- same arrangement as
:mod:`scatterem.nn.functional._bcast_cmul_triton`.

Why this exists
---------------
The joint PRISM backward computes ``grad_basis = grad.unsqueeze(1) *
S_dt_crop[None].conj()``: a small ``(Nmodes, K, M1, M2)`` cotangent broadcast over
the ``Bp`` parent beams of the ``(Bp, K, M1, M2)`` crop.  ``mul`` has no conj-view
support in TensorIterator, so the ``.conj()`` is a ``resolve_conj`` -- a full read
and write of the crop (2 GiB each way at ``r2_prism_project/medium``) -- before the
multiply reads it back.  Counted in crop-sized complex64 passes, the eager spelling
is 2 (resolve) + 2 (mul: read + write) = 4; this kernel reads the crop once and
writes the product once = 2.  The small cotangent is loaded once per program tile
and kept in registers across the beam loop, as in ``_bcast_cmul_triton``, so its
broadcast re-read never reaches DRAM.

Numerics -- a bit-exactness contract, not a tolerance
----------------------------------------------------
The result is ``torch.equal`` to the eager expression.  ``conj`` is an exact sign
flip of the imaginary part, so loading ``stack`` and negating ``si`` in registers
hands the product exactly the operand ``resolve_conj`` would have written.  The
product is then spelled the way ``_bcast_cmul_triton`` spells torch's ``plane * w``
(the broadcast operand FIRST): the ``im*im`` product rounded, then one FMA per
half, pinned with ``tl.math.fma`` rather than left to Triton's contraction default.
Checked on the production operands (a benchmark script).  The
imaginary FMA follows the device's ATen build (``IM_SWAP``, from
:mod:`scatterem.physics._aten_cmul_order`); both kernels decline where ATen
uses neither order.

Gating
------
Declines -- and the caller keeps the eager expression -- unless both operands are
contiguous CUDA ``complex64``, ``stack.shape[1:] == plane.shape[1:]``, the plane is
a whole number of ``BLOCK`` tiles, the flat float offsets fit in int32, and no
operand needs a gradient under grad mode (a Triton write is invisible to autograd;
the call site runs it only in a non-``create_graph`` backward).

K-summed sibling
-----------------------
On an integer-pixel scan the basis is one plane broadcast over the ``K`` positions
(a stride-0 expand), so its gradient is the product above summed over ``K``.
Eagerly that is the full ``(Nm, Bp, K, M1, M2)`` product (2 GiB at
``r2_prism_project/medium``) followed by a reduction over ``K``.
:func:`bcast_mul_conj_ksum` forms the same per-term products, spelled as above,
and adds them in ``k`` order in registers, so only the ``(Nm, Bp, 1, M1, M2)``
sum is written.  Same gates (its own ``bcast_mul_conj_ksum`` backend row); not
bit-exact against the eager reduction (its summation order is TensorIterator's),
deterministic.

Gather mode
------------------
With ``stack=None`` both functions read each stack value as ``sdt[b, idx[pix]]``
instead of from a materialised stack, so the joint PRISM forward no longer has to
write (and save) the ``(Bp, K, M1, M2)`` crop -- a 2 GiB ``index_select`` at
``r2_prism_project/medium``.  The values are the same floats, so the result is
``torch.equal`` to passing ``sdt.index_select(1, idx).view(...)`` as the stack.  It
additionally needs a contiguous ``(NB, SYX)`` ``sdt`` and a contiguous int32
``idx`` with one entry per stack element of one beam.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from scatterem.nn._backends import (
    BCAST_MUL_CONJ,
    BCAST_MUL_CONJ_KSUM,
    KSUM_GATHER_WIDE,
)
from scatterem.physics._aten_cmul_order import aten_cmul_imag_swapped
from scatterem.utils.env import resolve_backend

__all__ = [
    "CALLS",
    "GATHER_CALLS",
    "KSUM_CALLS",
    "WIDE_CALLS",
    "bcast_mul_conj",
    "bcast_mul_conj_ksum",
    "bcast_mul_conj_ksum_supported",
    "bcast_mul_conj_supported",
    "ksum_gather_wide_supported",
]

#: ``[ran, declined]``, host-side only -- proves the fast path was TAKEN.
CALLS = [0, 0]
#: the same, for :func:`bcast_mul_conj_ksum`.
KSUM_CALLS = [0, 0]
#: runs of either function that took the index-gather mode.
GATHER_CALLS = [0]

BLOCK_DEFAULT = 1024
GROUPS_DEFAULT = 16
# the K-summed gather reads 4.37 ms at 512 (6.88 at 1024, 5.05 at 256) on
# ``r2_prism_project/medium/project_fwd_backward``'s operands; the plain
# gather is flat (6.42-6.51 ms) from 128 to 1024, so it keeps the default.
KSUM_GATHER_BLOCK_DEFAULT = 512
# the K-summed gather's WIDE launch -- BLOCK 1024, 2 warps, 8 beam groups
# (64 beams per program at ``Bp=512``), taken when it still gives the card at least
# ``KSUM_GATHER_WIDE_MIN_PROGS_PER_SM`` programs per SM.  Measured in row on
# ``r2_prism_project/*/project_backward --no-warp`` against 512/4/16:
# 128 programs (small) 0.90x, 288 (Bp=256, M=192) 1.03x, 512 (Bp=256, M=256)
# 1.07x, 512 (medium, Bp=512) 1.15-1.18x.  BLOCK, warps and groups only remap
# pixels and beams to programs; each output element's K loop is unchanged, so
# every launch is bit-identical.  The ``ksum_gather_wide`` row decides.
KSUM_GATHER_WIDE_LAUNCH = (1024, 2, 8)
KSUM_GATHER_WIDE_MIN_PROGS_PER_SM = 3
#: gather-mode ksum runs that took the wide launch.
WIDE_CALLS = [0]
_SM_COUNT: dict[int, int] = {}


@triton.jit
def _bcast_mul_conj_kernel(
    Out,  # (NB, PLANE, 2) f32 -- real view of the output          [out]
    S,  # (NB, PLANE, 2) f32 stack, or (NB, SYX, 2) f32 if GATHER  [in]
    IDX,  # (PLANE,) int32 flat index into SYX (read only if GATHER)
    P,  # (PLANE, 2) f32     -- real view of the broadcast plane  [in]
    PLANE,
    NB,
    BPP,  # beams per program
    SYX,
    GATHER: tl.constexpr,
    BLOCK: tl.constexpr,
    IM_SWAP: tl.constexpr,  # ATen's imaginary FMA on this device (_aten_cmul_order)
):
    t = tl.program_id(0)  # which plane tile
    g = tl.program_id(1)  # which beam group
    off = t * BLOCK + tl.arange(0, BLOCK)
    pr = tl.load(P + 2 * off, eviction_policy="evict_last")
    pi = tl.load(P + 2 * off + 1, eviction_policy="evict_last")
    o2 = 2 * (t * BLOCK) + tl.arange(0, 2 * BLOCK)
    if GATHER:
        g2 = 2 * tl.load(IDX + off)
    b1 = tl.minimum(g * BPP + BPP, NB)
    for b in range(g * BPP, b1):
        o = o2 + 2 * b * PLANE
        if GATHER:
            # stack[b, pix] IS sdt[b, IDX[pix]] -- the same two floats.
            wr = tl.load(S + g2 + 2 * b * SYX)  # pyright: ignore[reportPossiblyUnboundVariable]
            si = tl.load(S + g2 + 1 + 2 * b * SYX)  # pyright: ignore[reportPossiblyUnboundVariable]
        else:
            sf = tl.load(S + o, eviction_policy="evict_first")
            wr, si = tl.split(tl.reshape(sf, (BLOCK, 2)))
        # resolve_conj, exactly.  NOT ``-si``: Triton lowers unary minus as
        # ``0 - x``, which maps +0 to +0 where a sign flip gives -0, and torch's
        # conj (and the negation inside its product) flip the sign bit.
        # Multiplying by -1.0 is exact and flips the sign of a zero too.
        wi = si * -1.0
        # torch's ``plane * w`` with ``w = conj(stack)``.
        cr = tl.math.fma(pr, wr, (pi * wi) * -1.0)
        if IM_SWAP:
            ci = tl.math.fma(pi, wr, pr * wi)
        else:
            ci = tl.math.fma(pr, wi, pi * wr)
        tl.store(
            Out + o,
            tl.reshape(tl.join(cr, ci), (2 * BLOCK,)),
            eviction_policy="evict_first",
        )


@triton.jit
def _bcast_mul_conj_ksum_kernel(
    Out,  # (NB, PLANE, 2) f32  -- real view of the K-summed output   [out]
    S,  # (NB, K, PLANE, 2) f32 stack, or (NB, SYX, 2) f32 if GATHER [in]
    IDX,  # (K, PLANE) int32 flat index into SYX (read only if GATHER)
    P,  # (K, PLANE, 2) f32     -- real view of the broadcast plane  [in]
    PLANE,
    NB,
    BPP,  # beams per program
    K,
    SYX,
    GATHER: tl.constexpr,
    BLOCK: tl.constexpr,
    IM_SWAP: tl.constexpr,  # ATen's imaginary FMA on this device (_aten_cmul_order)
):
    t = tl.program_id(0)  # which plane tile
    g = tl.program_id(1)  # which beam group
    o2 = 2 * (t * BLOCK) + tl.arange(0, 2 * BLOCK)
    pix = t * BLOCK + tl.arange(0, BLOCK)
    b1 = tl.minimum(g * BPP + BPP, NB)
    for b in range(g * BPP, b1):
        accr = tl.zeros((BLOCK,), dtype=tl.float32)
        acci = tl.zeros((BLOCK,), dtype=tl.float32)
        for k in range(K):
            pf = tl.load(P + o2 + 2 * k * PLANE, eviction_policy="evict_last")
            pr, pi = tl.split(tl.reshape(pf, (BLOCK, 2)))
            if GATHER:
                # stack[b, k, pix] IS sdt[b, IDX[k, pix]].
                g2 = 2 * tl.load(IDX + k * PLANE + pix, eviction_policy="evict_last")
                wr = tl.load(S + g2 + 2 * b * SYX)
                si = tl.load(S + g2 + 1 + 2 * b * SYX)
            else:
                sf = tl.load(
                    S + o2 + 2 * (b * K + k) * PLANE, eviction_policy="evict_first"
                )
                wr, si = tl.split(tl.reshape(sf, (BLOCK, 2)))
            # The product exactly as ``_bcast_mul_conj_kernel`` spells it.
            wi = si * -1.0
            accr += tl.math.fma(pr, wr, (pi * wi) * -1.0)
            if IM_SWAP:
                acci += tl.math.fma(pi, wr, pr * wi)
            else:
                acci += tl.math.fma(pr, wi, pi * wr)
        tl.store(
            Out + o2 + 2 * b * PLANE,
            tl.reshape(tl.join(accr, acci), (2 * BLOCK,)),
            eviction_policy="evict_first",
        )


def _plain(t: torch.Tensor) -> bool:
    """A contiguous CUDA ``complex64`` operand the kernels may read as floats.

    ``view_as_real`` refuses a lazily-conjugated tensor and would silently drop a
    negative bit; the eager multiply resolves such a view itself, so decline
    rather than resolve a copy here.  A Triton write is invisible to autograd,
    so an operand that needs a gradient under grad mode declines too.
    """
    return (
        t.is_cuda
        and t.dtype is torch.complex64
        and t.is_contiguous()
        and not (t.is_conj() or t.is_neg())
        and not (torch.is_grad_enabled() and t.requires_grad)
    )


def _stack_ok(
    plane: torch.Tensor,
    stack: torch.Tensor | None,
    sdt: torch.Tensor | None,
    idx: torch.Tensor | None,
    per_beam: int,
) -> bool:
    """The stack's gates, or the gather mode's: ``sdt`` ``(NB, SYX)``."""
    if stack is not None:
        return _plain(stack) and tuple(stack.shape[1:]) == tuple(plane.shape[1:])
    return (
        sdt is not None
        and _plain(sdt)
        and sdt.ndim == 2
        and idx is not None
        and idx.device == sdt.device
        and idx.dtype is torch.int32
        and idx.is_contiguous()
        and idx.numel() == per_beam
        and 2 * sdt.numel() < 2**31
    )


def bcast_mul_conj_supported(
    plane: torch.Tensor,
    stack: torch.Tensor | None,
    sdt: torch.Tensor | None = None,
    idx: torch.Tensor | None = None,
    block: int = BLOCK_DEFAULT,
) -> bool:
    """True if :func:`bcast_mul_conj` may replace the eager conj-then-mul."""
    if not (_plain(plane) and plane.ndim >= 2):
        return False
    if aten_cmul_imag_swapped(plane.device) is None:
        return False
    npx = plane[0].numel()
    if not _stack_ok(plane, stack, sdt, idx, npx):
        return False
    nb = (sdt if stack is None else stack).shape[0]  # pyright: ignore[reportOptionalMemberAccess]
    return not (npx == 0 or nb == 0 or npx % block != 0 or 2 * nb * npx >= 2**31)


def ksum_gather_wide_supported(device: torch.device, npx: int, nb: int) -> bool:
    """True if the gather-mode ksum launch still fills the card when wide."""
    block, _, groups = KSUM_GATHER_WIDE_LAUNCH
    if not (device.type == "cuda" and npx % block == 0 and nb >= groups):
        return False
    idx = device.index if device.index is not None else torch.cuda.current_device()
    n_sm = _SM_COUNT.get(idx)
    if n_sm is None:
        n_sm = torch.cuda.get_device_properties(idx).multi_processor_count
        _SM_COUNT[idx] = n_sm
    return (npx // block) * groups >= KSUM_GATHER_WIDE_MIN_PROGS_PER_SM * n_sm


def _ksum_gather_launch(device: torch.device, npx: int, nb: int):
    """``(BLOCK, num_warps, groups)`` for one unforced gather-mode ksum launch."""
    if resolve_backend(KSUM_GATHER_WIDE, None, device, npx, nb) == "wide":
        return KSUM_GATHER_WIDE_LAUNCH
    return KSUM_GATHER_BLOCK_DEFAULT, 4, GROUPS_DEFAULT


def _ksum_block(stack: torch.Tensor | None, block: int | None) -> int:
    if block is not None:
        return block
    return KSUM_GATHER_BLOCK_DEFAULT if stack is None else BLOCK_DEFAULT


def bcast_mul_conj_ksum_supported(
    plane: torch.Tensor,
    stack: torch.Tensor | None,
    sdt: torch.Tensor | None = None,
    idx: torch.Tensor | None = None,
    block: int | None = None,
) -> bool:
    """True if :func:`bcast_mul_conj_ksum` may replace the eager product + K-sum."""
    if not (_plain(plane) and plane.ndim >= 3):
        return False
    if aten_cmul_imag_swapped(plane.device) is None:
        return False
    K = plane.shape[1]
    npx = plane[0, 0].numel()
    if not _stack_ok(plane, stack, sdt, idx, K * npx):
        return False
    nb = (sdt if stack is None else stack).shape[0]  # pyright: ignore[reportOptionalMemberAccess]
    return not (
        npx == 0
        or nb == 0
        or K == 0
        or npx % _ksum_block(stack, block) != 0
        or 2 * nb * K * npx >= 2**31
    )


def _stack_args(
    stack: torch.Tensor | None,
    sdt: torch.Tensor | None,
    idx: torch.Tensor | None,
    rows: int,
    npx: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """``(S, IDX, SYX)`` kernel arguments for the stack or the gather mode."""
    if stack is None:
        return torch.view_as_real(sdt), idx, sdt.shape[1]  # pyright: ignore[reportArgumentType, reportReturnType, reportOptionalMemberAccess]
    s_real = torch.view_as_real(stack.view(rows, npx))
    return s_real, s_real, 0  # IDX unread: GATHER is a constexpr False


def bcast_mul_conj_ksum(
    plane: torch.Tensor,
    stack: torch.Tensor | None,
    block: int | None = None,
    groups: int | None = None,
    *,
    sdt: torch.Tensor | None = None,
    idx: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """``(plane.unsqueeze(1) * stack[None].conj()).sum(2, keepdim=True)``, or ``None``.

    ``plane`` is ``(Nm, K, *P)`` and ``stack`` is ``(NB, K, *P)``; the result is
    ``(Nm, NB, 1, *P)``, contiguous.  Declines under the same gates as
    :func:`bcast_mul_conj`.  With ``stack=None`` the stack is read as
    ``sdt.index_select(1, idx).view(NB, K, *P)`` without being written.
    An unforced gather (``block`` and ``groups`` both ``None``) picks its launch
    from the ``ksum_gather_wide`` row once the shapes are known.
    """
    auto = stack is None and block is None and groups is None
    block = _ksum_block(stack, block)
    if resolve_backend(BCAST_MUL_CONJ_KSUM, None, plane, stack, sdt, idx, block) != (
        "triton"
    ):
        KSUM_CALLS[1] += 1
        return None
    gather = stack is None
    K = plane.shape[1]
    npx = plane[0, 0].numel()
    nb = (sdt if gather else stack).shape[0]  # pyright: ignore[reportOptionalMemberAccess]
    num_warps = 4
    if groups is None:
        groups = GROUPS_DEFAULT
    if auto:
        block, num_warps, groups = _ksum_gather_launch(plane.device, npx, nb)
    out = torch.empty(
        (plane.shape[0], nb, 1) + tuple(plane.shape[2:]),
        dtype=torch.complex64,
        device=plane.device,
    )
    bpp = (nb + groups - 1) // groups
    s_real, idx_arg, syx = _stack_args(stack, sdt, idx, nb * K, npx)
    for m in range(plane.shape[0]):
        _bcast_mul_conj_ksum_kernel[(npx // block, groups)](  # pyright: ignore[reportIndexIssue]
            torch.view_as_real(out[m].view(nb, npx)),
            s_real,
            idx_arg,
            torch.view_as_real(plane[m].view(K, npx)),
            npx,
            nb,
            bpp,
            K,
            syx,
            GATHER=gather,
            BLOCK=block,
            IM_SWAP=aten_cmul_imag_swapped(plane.device),
            num_warps=num_warps,
        )
    KSUM_CALLS[0] += 1
    GATHER_CALLS[0] += gather
    WIDE_CALLS[0] += auto and block != KSUM_GATHER_BLOCK_DEFAULT
    return out


def bcast_mul_conj(
    plane: torch.Tensor,
    stack: torch.Tensor | None,
    block: int = BLOCK_DEFAULT,
    groups: int = GROUPS_DEFAULT,
    *,
    sdt: torch.Tensor | None = None,
    idx: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """``plane.unsqueeze(1) * stack[None].conj()`` as a new tensor, or ``None``.

    ``plane`` is ``(Nm, *P)`` and ``stack`` is ``(NB, *P)``; the result is
    ``(Nm, NB, *P)``, contiguous.  Returns ``None`` without launching anything when
    a gate in the module docstring fails -- declining is the contract.  With
    ``stack=None`` the stack is read as ``sdt.index_select(1, idx).view(NB, *P)``
    without being written.
    """
    if resolve_backend(BCAST_MUL_CONJ, None, plane, stack, sdt, idx, block) != (
        "triton"
    ):
        CALLS[1] += 1
        return None
    gather = stack is None
    npx = plane[0].numel()
    nb = (sdt if gather else stack).shape[0]  # pyright: ignore[reportOptionalMemberAccess]
    out = torch.empty(
        (plane.shape[0], nb) + tuple(plane.shape[1:]),
        dtype=torch.complex64,
        device=plane.device,
    )
    bpp = (nb + groups - 1) // groups
    s_real, idx_arg, syx = _stack_args(stack, sdt, idx, nb, npx)
    for m in range(plane.shape[0]):
        _bcast_mul_conj_kernel[(npx // block, groups)](  # pyright: ignore[reportIndexIssue]
            torch.view_as_real(out[m].view(nb, npx)),
            s_real,
            idx_arg,
            torch.view_as_real(plane[m].view(npx)),
            npx,
            nb,
            bpp,
            syx,
            GATHER=gather,
            BLOCK=block,
            IM_SWAP=aten_cmul_imag_swapped(plane.device),
            num_warps=4,
        )
    CALLS[0] += 1
    GATHER_CALLS[0] += gather
    return out
