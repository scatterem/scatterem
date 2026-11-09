"""Per-output-pixel gather for the PRISM crop-contract ``grad_Sdt``.

Private, lazily imported by its one call site
(``scatterem.nn.functional.warp.prism_project._torch_adjoint``'s
``_CropContractFrozenBasis.backward``) so that ``import scatterem`` never requires
Triton -- same arrangement as :mod:`scatterem.nn.functional._bcast_mul_conj_triton`.

Why this exists
---------------
The window backward builds ``grad_Sdt`` as::

    acc = zeros((Bp, SY, SX))                                   # fill
    for k, rect, m:  acc[:, rect] += conj(g)[m, k, rect] * basis[m, :, k, rect]
    grad_Sdt = conj(acc)                                        # in place

i.e. one full write for the fill, a read-modify-write of the ``(Bp, M1, M2)``
window per scan position, and a read-modify-write of the whole gradient for the
conj.  At ``r2_prism_project/medium`` (``Bp=512``, ``K=8``, ``M=256``, ``S=512``)
that is ~9 GiB of traffic for a 1 GiB result and a 2 GiB ``basis``.  Here every
output element is produced by ONE lane: it walks the ``K`` positions, adds the
product only where that position's periodic crop covers the pixel, and stores
``conj(acc)`` once -- ``basis`` read once, ``grad_Sdt`` written once (~3 GiB).

Numerics -- the same sum in the same order, NOT bit-exact
---------------------------------------------------------
Each lane starts from ``+0`` and adds, in ``k`` order with the mode index inner,
the same terms the shipped loop adds into that element.  Each product
``conj(g) * b`` is spelled with every multiply and add rounded once and no FMA
contraction (``libdevice`` ``*_rn``), which is word-equal to ATen's contiguous
complex64 ``addcmul_``.  The shipped loop's ``addcmul_`` always broadcasts the
``conj(g)`` plane over ``Bp`` and so takes ATen's strided kernel, whose rounding
differs from its contiguous one in ~6 % of words (``_d871_spell.py``): the
two bodies agree to one rounding per term (max rel ~1e-7 of peak), the error
class already accepted for the loop itself.  The final conj multiplies the
imaginary half by ``-1.0`` (a sign flip, not ``0 - x``).  Unlike
the atomics that the loop replaced, the result is run-to-run deterministic.

Gating
------
Declines -- the caller keeps the shipped window loop -- unless ``grad`` and
``basis`` are CUDA ``complex64`` with the crop-contract layout and no lazy
conj / neg bit, ``grad`` contiguous and ``basis`` with contiguous ``(M1, M2)``
planes (its m/b/k strides are passed in, so the ``_dr0`` stride-0 K-broadcast
view is served), ``K <= KMAX`` (the per-lane position walk is ``O(K)``),
every flat float offset fits in int32, and no operand needs a gradient under grad
mode (a Triton write is invisible to autograd; the call site runs it only in a
non-``create_graph`` backward).  The gate is the ``prism_window_gather`` backend
row's predicate, :func:`window_gather_supported`.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice as L

from scatterem.nn._backends import PRISM_WINDOW_GATHER
from scatterem.utils.env import resolve_backend

__all__ = ["CALLS", "KMAX", "window_gather_grad_sdt", "window_gather_supported"]

#: ``[ran, declined]``, host-side only -- proves the fast path was TAKEN.
CALLS = [0, 0]

#: Largest scan-position count the per-lane walk is used for.
KMAX = 64

BLOCK_X_DEFAULT = 256
BEAMS_PER_PROGRAM = 16
# 8 warps is 1.118x on the kernel and 1.096x on
# ``r2_prism_project/medium/project_backward --no-warp`` (1/2/16 are slower;
# flat at ``small``).  The per-lane K walk is unchanged, so the bits do not move.
NUM_WARPS = 8


@triton.jit
def _window_gather_kernel(
    Out,  # (Bp, SY, SX, 2) f32 -- real view of grad_Sdt          [out]
    B,  # (NM, Bp, K, M1, M2, 2) f32 -- real view of basis_k      [in]
    G,  # (NM, K, M1, M2, 2) f32 -- real view of the cotangent    [in]
    POS,  # (K, 2) int32 -- positions reduced mod (SY, SX)        [in]
    SY,
    SX,
    M1,
    M2,
    K,
    NB,
    SB_M,  # basis_k strides over (m, b, k) in complex elements; 0 on k
    SB_B,  # for the ``_dr0`` K-broadcast view
    SB_K,
    NM: tl.constexpr,
    BPP: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    y = tl.program_id(0)
    xt = tl.program_id(1)
    grp = tl.program_id(2)
    x = xt * BLOCK_X + tl.arange(0, BLOCK_X)
    xin = x < SX
    plane = M1 * M2
    for bb in range(BPP):
        b = grp * BPP + bb
        if b < NB:
            acc_r = tl.zeros((BLOCK_X,), dtype=tl.float32)
            acc_i = tl.zeros((BLOCK_X,), dtype=tl.float32)
            for k in range(K):
                py = tl.load(POS + 2 * k)
                dy = y - py
                dy = tl.where(dy < 0, dy + SY, dy)
                if dy < M1:
                    px = tl.load(POS + 2 * k + 1)
                    dx = x - px
                    dx = tl.where(dx < 0, dx + SX, dx)
                    cov = xin & (dx < M2)
                    for m in tl.static_range(NM):
                        go = 2 * ((m * K + k) * plane + dy * M2 + dx)
                        gr = tl.load(G + go, mask=cov, other=0.0)
                        gi = tl.load(G + go + 1, mask=cov, other=0.0)
                        bo = 2 * (m * SB_M + b * SB_B + k * SB_K + dy * M2 + dx)
                        br = tl.load(
                            B + bo, mask=cov, other=0.0, eviction_policy="evict_first"
                        )
                        bi = tl.load(
                            B + bo + 1,
                            mask=cov,
                            other=0.0,
                            eviction_policy="evict_first",
                        )
                        # conj(g), exactly: a sign flip, never ``0 - x``.
                        cgi = gi * -1.0
                        # conj(g) * b with every product and sum rounded once,
                        # no contraction: ATen's contiguous complex64 addcmul_.
                        pr = L.sub_rn(L.mul_rn(gr, br), L.mul_rn(cgi, bi))
                        pi = L.add_rn(L.mul_rn(gr, bi), L.mul_rn(cgi, br))
                        acc_r = tl.where(cov, acc_r + pr, acc_r)
                        acc_i = tl.where(cov, acc_i + pi, acc_i)
            oo = 2 * ((b * SY + y) * SX + x)
            tl.store(Out + oo, acc_r, mask=xin)
            # the shipped in-place ``conj_physical_``, exactly.
            tl.store(Out + oo + 1, acc_i * -1.0, mask=xin)


def _complex64_cuda(t: torch.Tensor) -> bool:
    return (
        t.is_cuda
        and t.dtype is torch.complex64
        and not (t.is_conj() or t.is_neg())
        and not (torch.is_grad_enabled() and t.requires_grad)
    )


def window_gather_supported(
    grad: torch.Tensor,
    basis_k: torch.Tensor,
    positions: Sequence[Sequence[int]],
    grid: tuple[int, int],
) -> bool:
    """True if :func:`window_gather_grad_sdt` may replace the window loop."""
    if not (_complex64_cuda(grad) and _complex64_cuda(basis_k)):
        return False
    if grad.ndim != 4 or basis_k.ndim != 5:
        return False
    nm, K, M1, M2 = grad.shape
    if basis_k.shape[0] != nm or tuple(basis_k.shape[2:]) != (K, M1, M2):
        return False
    # ``basis_k`` only needs contiguous ``(M1, M2)`` planes; its m/b/k
    # strides are kernel arguments, so the ``_dr0`` stride-0 K view is taken too.
    sbm, sbb, sbk, sby, sbx = basis_k.stride()
    if not (
        grad.is_contiguous() and sbx == 1 and sby == M2 and min(sbm, sbb, sbk) >= 0
    ):
        return False
    SY, SX = int(grid[0]), int(grid[1])
    # largest element offset + 1 (= numel when contiguous)
    span = 1 + sum(
        (n - 1) * s for n, s in zip(basis_k.shape, basis_k.stride(), strict=True)
    )
    return not (
        K == 0
        or K > KMAX
        or len(positions) != K
        or M1 > SY
        or M2 > SX
        or 2 * span >= 2**31
        or 2 * basis_k.shape[1] * SY * SX >= 2**31
    )


def window_gather_grad_sdt(
    grad: torch.Tensor,
    basis_k: torch.Tensor,
    positions: Sequence[Sequence[int]],
    grid: tuple[int, int],
    block_x: int = BLOCK_X_DEFAULT,
) -> torch.Tensor | None:
    """``grad_Sdt`` of shape ``(Bp, SY, SX)`` for the crop contract, or ``None``.

    ``grad`` is the ``(Nmodes, K, M1, M2)`` cotangent, ``basis_k`` the
    ``(Nmodes, Bp, K, M1, M2)`` saved basis and ``positions`` the host-side
    ``(py, px)`` list.  Returns ``None`` without launching anything when a gate in
    the module docstring fails -- declining is the contract.
    """
    if (
        resolve_backend(PRISM_WINDOW_GATHER, None, grad, basis_k, positions, grid)
        != "triton"
    ):
        CALLS[1] += 1
        return None
    nm, K, M1, M2 = grad.shape
    SY, SX = int(grid[0]), int(grid[1])
    nb = basis_k.shape[1]
    sbm, sbb, sbk = basis_k.stride()[:3]
    pos = torch.tensor(
        [(int(py) % SY, int(px) % SX) for py, px in positions],
        dtype=torch.int32,
        device=grad.device,
    )
    out = torch.empty((nb, SY, SX), dtype=torch.complex64, device=grad.device)
    grid3 = (SY, triton.cdiv(SX, block_x), triton.cdiv(nb, BEAMS_PER_PROGRAM))
    _window_gather_kernel[grid3](  # pyright: ignore[reportIndexIssue]
        torch.view_as_real(out),
        torch.view_as_real(basis_k),
        torch.view_as_real(grad),
        pos,
        SY,
        SX,
        M1,
        M2,
        K,
        nb,
        sbm,
        sbb,
        sbk,
        NM=nm,
        BPP=BEAMS_PER_PROGRAM,
        BLOCK_X=block_x,
        num_warps=NUM_WARPS,
    )
    CALLS[0] += 1
    return out
