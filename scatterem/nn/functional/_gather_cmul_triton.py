"""Fused gather + transmit multiply ``T[idx] * w`` for the plain multislice route.

Private, lazily imported by the plain multislice route
so that ``import scatterem`` never requires Triton -- same arrangement as
:mod:`scatterem.nn.functional._bcast_cmul_triton`.

Why this exists
---------------
On the library-default (non-Warp) forward, each multislice slice gathers its
per-position transmission patches from the per-unique-pixel stack and then multiplies
them into the waves::

    patches = T_stack[z].index_select(0, idx).reshape(K, ph, pw)   # write 128 MiB
    waves = patches.unsqueeze(0) * waves                             # read 256, write 128

At ``r3_simulator/medium`` (``K=256``, ``256x256`` patches, 32 slices) both passes
run at the DRAM roof, and together they are ~30 % of the row: the gathered patch
stack exists only to be read back once.  The Warp route fuses the two
(``fused_transmit``); this kernel does the same for the default route, so the patch
stack is never written: per slice it reads ``w`` + ``idx`` and writes the product,
320 MiB instead of 576 MiB.  The per-unique-pixel slice ``T_stack[z]`` is ~1 MiB and
stays L2-resident, so gathering it costs no DRAM.

Numerics -- a bit-exactness contract, not a tolerance
----------------------------------------------------
The result is ``torch.equal`` to ``T[idx].unsqueeze(0) * w``: a gather copies bits,
and the multiply is spelled with the explicit FMAs torch's ``patch * w`` compiles to
(the ``PLANE_FIRST`` spelling of ``_bcast_cmul_triton``), so the same two roundings
happen in the same order.  A test can therefore hold the simulated cube at a sha256.
The imaginary FMA follows the device's ATen build (``IM_SWAP``, from
:mod:`scatterem.physics._aten_cmul_order`); the kernel declines where ATen
uses neither order.

Gating
------
Declines -- and the caller keeps the torch ops -- unless: CUDA; complex64 ``T`` (1-D,
contiguous) and ``w`` (contiguous, ``w.numel()`` a whole multiple of ``idx.numel()``);
a 1-D contiguous int32/int64 ``idx``; no autograd on either operand (the kernel is
forward-only); fewer than ``2**31`` float words.  ``SCATTEREM_GATHER_CMUL=0`` forces
the torch path (read per call, used for the A/B measurement).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from scatterem.nn._backends import GATHER_CMUL
from scatterem.nn._backends_graphs import GATHER_CMUL_SEP
from scatterem.physics._aten_cmul_order import _cached_order, aten_cmul_imag_swapped
from scatterem.utils.env import resolve_backend

__all__ = [
    "CALLS",
    "CALLS_SEP",
    "gather_cmul",
    "gather_cmul_sep",
    "gather_cmul_sep_supported",
    "gather_defer_norm_supported",
    "separable_tables",
    "warp_transmit_imag_swapped",
]

#: ``[ran, declined]``.  Host-side counters only, so a probe can prove the fast path
#: was TAKEN rather than infer it from a timing.
CALLS = [0, 0]

BLOCK_DEFAULT = 1024


@triton.jit
def _gather_cmul_kernel(
    W,  # (NM, N, 2) f32 -- real view of the complex64 waves     [in]
    T,  # (U, 2) f32     -- real view of the per-unique-pixel T   [in]
    IDX,  # (N,) int      -- per-element index into T              [in]
    OUT,  # (NM, N, 2) f32 -- real view of the result              [out]
    N,
    NM,
    BLOCK: tl.constexpr,
    IM_SWAP: tl.constexpr,  # ATen's imaginary FMA on this device (_aten_cmul_order)
):
    pid = tl.program_id(0)
    e = pid * BLOCK + tl.arange(0, BLOCK)
    mask = e < N
    j = tl.load(IDX + e, mask=mask, other=0)
    pr = tl.load(T + 2 * j, mask=mask, other=0.0, eviction_policy="evict_last")
    pi = tl.load(T + 2 * j + 1, mask=mask, other=0.0, eviction_policy="evict_last")

    o2 = 2 * (pid * BLOCK) + tl.arange(0, 2 * BLOCK)
    m2 = o2 < 2 * N
    for m in range(NM):
        o = o2 + 2 * m * N
        wf = tl.load(W + o, mask=m2, other=0.0, eviction_policy="evict_first")
        wr, wi = tl.split(tl.reshape(wf, (BLOCK, 2)))
        # torch's ``patch * w``: the (im*im) product is rounded, then one FMA.
        cr = tl.math.fma(pr, wr, -(pi * wi))
        if IM_SWAP:
            ci = tl.math.fma(pi, wr, pr * wi)
        else:
            ci = tl.math.fma(pr, wi, pi * wr)
        tl.store(
            OUT + o,
            tl.reshape(tl.join(cr, ci), (2 * BLOCK,)),
            mask=m2,
            eviction_policy="evict_first",
        )


def gather_cmul(
    T: torch.Tensor, idx: torch.Tensor, w: torch.Tensor, block: int = BLOCK_DEFAULT
) -> torch.Tensor | None:
    """``T.index_select(0, idx).reshape(w.shape[1:]).unsqueeze(0) * w``, fused.

    Returns ``None`` without launching anything when a gate in the module docstring
    fails; the caller must then run the torch ops itself.
    """
    if resolve_backend(GATHER_CMUL) != "triton":
        CALLS[1] += 1
        return None
    n = idx.numel()
    if not (T.is_cuda and w.is_cuda and idx.is_cuda):
        CALLS[1] += 1
        return None
    if T.dtype is not torch.complex64 or w.dtype is not torch.complex64:
        CALLS[1] += 1
        return None
    if idx.dtype not in (torch.int32, torch.int64) or idx.dim() != 1 or T.dim() != 1:
        CALLS[1] += 1
        return None
    if not (T.is_contiguous() and w.is_contiguous() and idx.is_contiguous()):
        CALLS[1] += 1
        return None
    if n == 0 or w.numel() % n != 0 or 2 * w.numel() >= 2**31:
        CALLS[1] += 1
        return None
    if torch.is_grad_enabled() and (T.requires_grad or w.requires_grad):
        CALLS[1] += 1
        return None
    im_swap = aten_cmul_imag_swapped(w.device)
    if im_swap is None:
        CALLS[1] += 1
        return None
    nm = w.numel() // n
    out = torch.empty_like(w)
    _gather_cmul_kernel[(triton.cdiv(n, block),)](  # pyright: ignore[reportIndexIssue]
        torch.view_as_real(w),
        torch.view_as_real(T),
        idx,
        torch.view_as_real(out),
        n,
        nm,
        BLOCK=block,
        IM_SWAP=im_swap,
        num_warps=4,
    )
    CALLS[0] += 1
    return out


# ---------------------------------------------------------------------------
# Separable-index variant (with a 64-bit gather; the same kernel serves
# the Warp route, rounded as the Warp kernel rounds).
#
# The batched unique-with-inverse step builds ``inv[k, y, x]`` as a box-linear offset,
# so it is a sum of a row table and a column table:
#     inv[k, y, x] == R[k, y] + C[k, x],  R = inv[:, :, 0],
#                                         C = inv[:, 0, :] - inv[:, 0, :1].
# :func:`separable_tables` checks that identity EXACTLY on the whole index (a
# verified gate, not an argued one), so the kernel computes the same ``j`` from
# two small L1-resident tables instead of loading the full-size index plane.
# ``T[j]`` is then read as ONE 8-byte word and split by bitcast: a bit copy.
# Row ``gather_cmul_sep`` selects the route.
# ---------------------------------------------------------------------------

#: ``[ran, declined]`` for :func:`gather_cmul_sep`.
CALLS_SEP = [0, 0]

#: Device index -> the Warp ``fused_transmit`` imaginary FMA order (or ``None``).
_WARP_ORDER: dict[int, bool | None] = {}


def gather_cmul_sep_supported(inv: torch.Tensor) -> bool:
    """The ``gather_cmul_sep`` row's predicate: may :func:`separable_tables` try ``inv``?

    A ``(K, ph, pw)`` contiguous CUDA int32 index, outside ``torch.compile``
    tracing and outside a CUDA-graph capture (the proof reads the host).

    Parameters
    ----------
    inv : torch.Tensor
        The per-position index into the per-unique-pixel stack.

    Returns
    -------
    bool
        Whether the separable route may be tried.
    """
    return (
        inv.is_cuda
        and inv.dtype == torch.int32
        and inv.dim() == 3
        and inv.is_contiguous()
        and not torch.compiler.is_compiling()
        and not torch.cuda.is_current_stream_capturing()
    )


def gather_defer_norm_supported(waves: torch.Tensor) -> bool:
    """The ``gather_defer_norm`` row's predicate: is there a ``1/N`` to defer?

    CUDA complex64 waves on a grid that is not a power of two on both axes.  Only
    there does ``Propagator.forward`` run a separate scaling pass after the
    inverse FFT; on a power-of-two grid it folds the ``1/N`` onto its ramp.

    Parameters
    ----------
    waves : torch.Tensor
        ``(..., ph, pw)`` waves of the slice loop.

    Returns
    -------
    bool
        Whether the separable slice loop may defer the ``1/N``.
    """
    ph, pw = (int(n) for n in waves.shape[-2:])
    pow2 = ph & (ph - 1) == 0 and pw & (pw - 1) == 0
    return waves.is_cuda and waves.dtype == torch.complex64 and not pow2


def separable_tables(
    inv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """``(R, C)`` int32 tables with ``inv == R[:, :, None] + C[:, None, :]``, or ``None``.

    The identity is verified on every element with ``torch.equal`` (one host
    sync).  ``None`` when row ``gather_cmul_sep`` declines or when ``inv`` has any
    other form; the caller then keeps the full index.

    Parameters
    ----------
    inv : torch.Tensor
        ``(K, ph, pw)`` CUDA int32 index.

    Returns
    -------
    tuple of torch.Tensor or None
        The ``(K, ph)`` row table and the ``(K, pw)`` column table.
    """
    if resolve_backend(GATHER_CMUL_SEP, None, inv) != "separable":
        return None
    R = inv[:, :, 0].contiguous()
    C = (inv[:, 0, :] - inv[:, 0, :1]).contiguous()
    if not torch.equal(inv, R[:, :, None] + C[:, None, :]):
        return None
    return R, C


def _warp_probe(device: torch.device) -> bool | None:
    """Run ``fused_transmit`` on the ATen probe's operands and name its FMA order."""
    from scatterem.nn.functional.warp.transmit import fused_transmit
    from scatterem.physics._aten_cmul_order import _classify, _operands

    pr, pi, wr, wi = _operands()
    n = pr.numel()
    f32 = torch.float32
    T = torch.complex(pr.to(f32), pi.to(f32)).to(device)
    w = torch.complex(wr.to(f32), wi.to(f32)).to(device).reshape(1, 1, 1, n)
    inv = torch.arange(n, dtype=torch.int32, device=device).reshape(1, 1, n)
    got = torch.view_as_real(fused_transmit(T, inv, w)).reshape(n, 2).cpu()
    return _classify(got, pr, pi, wr, wi)


def warp_transmit_imag_swapped(device: torch.device | int | str) -> bool | None:
    """The imaginary FMA order of the Warp ``fused_transmit`` product on ``device``.

    In the kernel's names (patch ``p``, wave ``w``): ``True`` for
    ``fma(pi, wr, pr*wi)``, the order nvcc gave the Warp ``_cmul`` on an
    sm_86; ``False`` for ``fma(pr, wi, pi*wr)``; ``None`` when the Warp
    kernel uses neither, the device is not CUDA, or the first call cannot run the
    probe now (CUDA-graph capture, ``torch.compile`` tracing, a ``RuntimeError``).
    Asked once per device on the exact operands of
    :mod:`scatterem.physics._aten_cmul_order`, because nvcc may contract the
    product differently on another architecture, as it does for ATen on sm_120.

    Parameters
    ----------
    device : torch.device or int or str
        The CUDA device.

    Returns
    -------
    bool or None
        The order, or ``None`` (then :func:`gather_cmul_sep` declines).
    """
    return _cached_order(_WARP_ORDER, device, _warp_probe)


@triton.jit
def _gather_cmul_sep_kernel(
    W,  # (NM, N, 2) f32 -- real view of the complex64 waves     [in]
    T64,  # (U,) int64    -- the complex64 T viewed as 8-byte words [in]
    R,  # (K, PH) int32  -- row table                            [in]
    C,  # (K, PW) int32  -- column table                         [in]
    OUT,  # (NM, N, 2) f32 -- real view of the result              [out]
    N,
    NM,
    SCALE,  # real prefactor on W, applied before the product (see DO_SCALE)
    PH: tl.constexpr,
    PW: tl.constexpr,
    BLOCK: tl.constexpr,
    IM_SWAP: tl.constexpr,  # the imaginary FMA order of the product copied
    WARP_NEG: tl.constexpr,  # the Warp kernel's sign flip in the real half
    DO_SCALE: tl.constexpr,
):
    pid = tl.program_id(0)
    e = pid * BLOCK + tl.arange(0, BLOCK)
    mask = e < N
    k = e // (PH * PW)
    y = (e // PW) % PH
    x = e % PW
    j = tl.load(R + k * PH + y, mask=mask, other=0) + tl.load(
        C + k * PW + x, mask=mask, other=0
    )
    v = tl.load(T64 + j, mask=mask, other=0, eviction_policy="evict_last")
    pr = (v & 0xFFFFFFFF).to(tl.int32).to(tl.float32, bitcast=True)
    pi = (v >> 32).to(tl.int32).to(tl.float32, bitcast=True)

    o2 = 2 * (pid * BLOCK) + tl.arange(0, 2 * BLOCK)
    m2 = o2 < 2 * N
    for m in range(NM):
        o = o2 + 2 * m * N
        wf = tl.load(W + o, mask=m2, other=0.0, eviction_policy="evict_first")
        wr, wi = tl.split(tl.reshape(wf, (BLOCK, 2)))
        if DO_SCALE:
            # The propagator's deferred inverse-FFT 1/N: two
            # separate f32 multiplies, rounded where torch's scaling pass rounds.
            # Torch multiplies by the complex (s, 0), so only the sign of an
            # exact-zero component can differ; the values are the same.
            wr = wr * SCALE
            wi = wi * SCALE
        # The (im*im) product is rounded, then one FMA.  ``* -1.0`` is nvcc's sign
        # flip in the Warp kernel: Triton's unary minus gives ``-(+0) == +0``,
        # which differs from Warp when the wave value is an exact ``(-0, +0)``.
        if WARP_NEG:
            cr = tl.math.fma(pr, wr, (pi * wi) * -1.0)
        else:
            cr = tl.math.fma(pr, wr, -(pi * wi))
        if IM_SWAP:
            ci = tl.math.fma(pi, wr, pr * wi)
        else:
            ci = tl.math.fma(pr, wi, pi * wr)
        tl.store(
            OUT + o,
            tl.reshape(tl.join(cr, ci), (2 * BLOCK,)),
            mask=m2,
            eviction_policy="evict_first",
        )


def _sep_operands_ok(T, R, C, w) -> bool:
    """:func:`gather_cmul`'s tensor gates, for the table form of the index."""
    if not (T.is_cuda and w.is_cuda and R.is_cuda and C.is_cuda):
        return False
    if T.dtype is not torch.complex64 or w.dtype is not torch.complex64:
        return False
    if R.dtype != torch.int32 or C.dtype != torch.int32 or T.dim() != 1:
        return False
    if not (T.is_contiguous() and w.is_contiguous()):
        return False
    if not (R.is_contiguous() and C.is_contiguous() and R.dim() == C.dim() == 2):
        return False
    n = R.shape[0] * R.shape[1] * C.shape[1]
    if C.shape[0] != R.shape[0] or n == 0 or w.numel() % n != 0:
        return False
    if 2 * w.numel() >= 2**31:
        return False
    return not (torch.is_grad_enabled() and (T.requires_grad or w.requires_grad))


def gather_cmul_sep(
    T: torch.Tensor,
    R: torch.Tensor,
    C: torch.Tensor,
    w: torch.Tensor,
    block: int = BLOCK_DEFAULT,
    warp_cmul: bool = False,
    scale: float | None = None,
) -> torch.Tensor | None:
    """:func:`gather_cmul` with ``idx = (R[:, :, None] + C[:, None, :]).reshape(-1)``.

    ``R``/``C`` must come from :func:`separable_tables`.  Same gates as
    :func:`gather_cmul`; returns ``None`` without launching when one fails.
    ``warp_cmul=False`` rounds the product as ATen does on this device
    (bit-identical to :func:`gather_cmul`); ``warp_cmul=True`` rounds it as the
    Warp ``fused_transmit`` kernel does (bit-identical to it).  A real
    ``scale`` gives ``T[idx] * (w * scale)``: the kernel takes over the
    propagator's deferred ``1/N``.

    Parameters
    ----------
    T : torch.Tensor
        ``(U,)`` complex64 per-unique-pixel transmission.
    R, C : torch.Tensor
        ``(K, ph)`` and ``(K, pw)`` int32 tables of the index.
    w : torch.Tensor
        ``(..., K, ph, pw)`` contiguous complex64 waves.
    block : int, optional
        Elements per program.
    warp_cmul : bool, optional
        Round as the Warp kernel instead of ATen.
    scale : float or None, optional
        A real factor on ``w``, applied before the product; ``None`` for none.

    Returns
    -------
    torch.Tensor or None
        The transmitted waves, or ``None``.
    """
    if not _sep_operands_ok(T, R, C, w):
        CALLS_SEP[1] += 1
        return None
    order = warp_transmit_imag_swapped if warp_cmul else aten_cmul_imag_swapped
    im_swap = order(w.device)
    if im_swap is None:
        CALLS_SEP[1] += 1
        return None
    K, ph = R.shape
    pw = C.shape[1]
    n = K * ph * pw
    out = torch.empty_like(w)
    _gather_cmul_sep_kernel[(triton.cdiv(n, block),)](  # pyright: ignore[reportIndexIssue]
        torch.view_as_real(w),
        T.view(torch.int64),
        R,
        C,
        torch.view_as_real(out),
        n,
        w.numel() // n,
        1.0 if scale is None else float(scale),
        PH=ph,
        PW=pw,
        BLOCK=block,
        IM_SWAP=im_swap,
        WARP_NEG=bool(warp_cmul),
        DO_SCALE=scale is not None,
        num_warps=4,
    )
    CALLS_SEP[0] += 1
    return out
