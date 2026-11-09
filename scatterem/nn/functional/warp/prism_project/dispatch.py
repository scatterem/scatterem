"""The public entry point, and the Warp route it dispatches to."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

from scatterem.nn._backends import PRISM_PROJECT
from scatterem.utils.env import resolve_use_warp

# Imported for its REGISTRATION side effect as much as for reading: defining
# ``ops`` is what puts ``scatterem::prism_project_fwd`` -- and the autograd
# wired onto it -- into ``torch.ops``, which the Warp route below calls.
from . import ops as ops
from ._basis import BeamletBasis, as_route_basis
from ._chunked import _prism_project_warp_chunked
from ._fwd_eager import fwd_eager
from ._l2_block import prism_proj_l2_blocked
from ._ramp_kernels import _RAMP_MUL_TILES
from .ramp import _build_basis_rs
from .torch_oracle import _items_per_budget, _prism_project_torch, _ramp_is_identity


def _prism_project_warp(
    Sdt: Tensor,
    basis_fft: Tensor | BeamletBasis,
    positions: Tensor,
    dr: Tensor,
    precision: str = "fp32",
    periodic: bool = True,
    ramp_identity: bool | None = None,
) -> Tensor:
    """Fused-Warp PRISM exit-wave projection (forward).

    Same contract as :func:`_prism_project_torch`.  The ramp + ifft2 (FFTs)
    run in torch to produce the per-position real-space basis; the cyclic-crop
    gather + parent-beam sum run in a single Warp kernel.

    Args:
        precision: "fp32" (default), "mixed", or "fp16".
            * "fp32": all inputs run through the kernel in complex64. If ``Sdt``
              arrives in complex32 it is upcast to complex64 first.
            * "mixed": "store in fp16, project in fp32" WITHOUT a full-tensor
              upcast. ``basis_rs`` (complex32 from the fp16 FFT) and ``Sdt`` (kept
              complex32) feed the dtype-generic Warp kernel as float16 real views;
              the kernel casts each operand to fp32 per element (register cast)
              before the multiply, so the arithmetic is fp32 but neither tensor is
              ever materialised in complex64. Parent-beam sum accumulates in fp32;
              exit wave written fp32 (returned complex64).
            * "fp16": true fp16 compute. Same complex32 inputs as "mixed", but the
              per-beam product runs in fp16 (no per-element upcast); the
              parent-beam sum still accumulates in fp32 and the exit wave is
              written fp32 (returned complex64).
        periodic: If ``True`` (default), use cyclic-crop (periodic boundary).
                  If ``False``, out-of-bounds contributions are zeroed out
                  (finite-specimen / non-periodic boundary).

    Returns ``(Nmodes, K, M1, M2)`` complex64 (exit waves are always fp32).
    """
    if isinstance(basis_fft, BeamletBasis):  # the kernels take ONE tensor
        basis_fft = basis_fft.dense()
    # Backward stays fp32: the bwd kernels are fp32-only. If a half-precision
    # route is requested while grad is enabled, fall back to fp32 inputs so
    # autograd is exact. Forward simulation runs under no_grad, so this only
    # guards misuse.
    grad_active = torch.is_grad_enabled() and (
        Sdt.requires_grad or basis_fft.requires_grad or dr.requires_grad
    )
    if precision in ("fp16", "mixed") and grad_active:
        precision = "fp32"

    if precision == "fp32" and periodic and not grad_active:
        # the L2-blocked graph (``_l2_block``), bit-identical.  Resolve the
        # identity-ramp question once (it syncs) and hand the answer on.
        if ramp_identity is None:
            ramp_identity = _ramp_is_identity(basis_fft, dr)
        if not ramp_identity:
            blocked = prism_proj_l2_blocked(Sdt, basis_fft, positions, dr)
            if blocked is not None:
                return blocked

    if precision in ("fp16", "mixed"):
        # Half-precision input route: keep Sdt in complex32 (float16 real view);
        # the FFT in _build_basis_rs produces a complex32 basis_rs. The kernel
        # re-promotes the per-beam product to fp32 before accumulating. For
        # "mixed" the kernel additionally casts each operand to fp32 before the
        # multiply (fp32 arithmetic on the complex32-stored tensors — no
        # full-tensor upcast); for "fp16" the multiply itself runs in fp16.
        if Sdt.dtype != torch.complex32:
            Sdt = Sdt.to(torch.complex32)
        # "fp16" multiplies in fp16; "mixed" casts each operand to fp32 first.
        mul_fp16 = precision == "fp16"
    else:
        # fp32 path: if Sdt was stored complex32, upcast so the fp32 kernel
        # receives float32 inputs.
        if Sdt.dtype == torch.complex32:
            Sdt = Sdt.to(torch.complex64)
        mul_fp16 = False  # no-op for fp32 (vec2) inputs

    basis_rs = _build_basis_rs(
        basis_fft, dr, precision, fold_k=True, ramp_identity=ramp_identity
    )  # (Nmodes, K or 1, Bp, M1, M2) complex64 (fp32) / complex32 (mixed/fp16)
    # Identity sub-pixel ramp: _build_basis_rs returns the K axis as a stride-0
    # broadcast of ONE slice.  ``.contiguous()`` below would expand it back out --
    # a (Nmodes, K, Bp, M1, M2) buffer that is K copies of the same bytes, and the
    # single largest allocation on this path (9.8 GiB at r3/small, 157 GiB at
    # r3/medium).  Keep the one slice and let the kernel do the broadcast with an
    # index instead: same reads, K times less memory and no K-fold copy.
    #
    # Holds under grad too: the bwd kernels now carry the broadcast index, and the
    # basis gradient is reduced over K inside the kernel that produces it
    # (_prism_project_bwd_basis_bcast_kernel) instead of being materialised at full
    # K and summed by the expand adjoint afterwards.  ``dr`` gradients are still
    # excluded, by _ramp_is_identity itself -- the fold drops the ramp from the
    # graph, so a differentiable dr must not take this route.
    K_pos = int(positions.shape[0])
    k_bcast = K_pos if (basis_rs.shape[1] == 1 and K_pos > 1) else 0
    basis_rs_real = torch.view_as_real(basis_rs).contiguous()
    Sdt_real = torch.view_as_real(Sdt).contiguous()
    pos = positions.to(torch.int32).contiguous()
    if fwd_eager(basis_rs_real, Sdt_real, pos):  # the op body, no dispatch
        fwd = ops._prism_project_fwd._init_fn
    else:
        fwd = torch.ops.scatterem.prism_project_fwd
    out_real = fwd(
        basis_rs_real, Sdt_real, pos, periodic, mul_fp16, k_bcast
    )  # always float32 -> complex64 exit waves
    return torch.view_as_complex(out_real)


def _warp_budget_chunk(
    basis_fft: Tensor | BeamletBasis, dr: Tensor, K: int
) -> tuple[int | None, bool | None]:
    """Position block for a forward ``chunk=None`` call on the Warp route.

    With an ACTIVE sub-pixel ramp, ``_build_basis_rs`` materialises the shifted
    basis ``(Nmodes, K, Bp, M1, M2)`` complex64 for every position at once (and
    its ``ifft2`` output next to it) -- 11 GiB at ``K = 324, Bp = 1109, 64^2``,
    and a forward-simulation chunk is thousands of positions.  So the same
    byte budget as the torch oracle (:func:`~.torch_oracle._items_per_budget`)
    sizes a position block for it, rounded down to a multiple of the widest
    ``_RAMP_MUL_TILES`` and never below it, so every full block keeps the fused
    ramp multiply (which needs a tile dividing its K) and a small call is not
    split at all.  Chunking only re-batches positions -- the kernels compute
    each exit wave from its own position alone -- so the result is
    bit-identical.

    The identity ramp (every ``dr`` zero) folds K away and has nothing to
    bound, so it stays one pass; the answer is returned so that pass skips its
    own ``dr.any()`` sync.  Callers use this only when no gradient is live: the
    chunked backward reassociates the gradient sums.

    Parameters
    ----------
    basis_fft : Tensor or BeamletBasis
        ``(Nmodes, Bp, M1, M2)`` basis (only its shape is read).
    dr : Tensor
        ``(K, 2)`` sub-pixel offsets.
    K : int
        Number of positions.

    Returns
    -------
    chunk : int or None
        Positions per block, or ``None`` for a single pass.
    ramp_identity : bool or None
        The resolved identity-ramp answer when ``dr`` was inspected, else
        ``None`` (let the single pass decide).
    """
    Nmodes, Bp, M1, M2 = basis_fft.shape
    tile = _RAMP_MUL_TILES[0]
    per_k = Nmodes * Bp * M1 * M2 * torch.complex64.itemsize
    block = _items_per_budget(per_k, K)
    block = max(tile, block - block % tile)
    if block >= K:
        return None, None
    if _ramp_is_identity(basis_fft, dr):
        return None, True
    return block, False


def prism_project(
    Sdt: Tensor,
    basis_fft: Tensor | BeamletBasis,
    positions: Tensor,
    dr: Tensor,
    chunk: int | None = None,
    precision: str = "fp32",
    periodic: bool = True,
    use_warp: bool | None = None,
) -> Tensor:
    """PRISM exit-wave projection dispatcher.

    When ``use_warp`` is True (or ``None`` and ``SCATTEREM_USE_WARP`` is one of
    ``1`` / ``true`` / ``yes``; see
    :func:`scatterem.utils.env.resolve_use_warp`), runs the fused Warp forward
    (:func:`_prism_project_warp`); otherwise falls back to the readable torch
    oracle (:func:`_prism_project_torch`).  Both paths obey the same contract:

        ``exit[m,k,y,x] = Σ_p basis_k[m,p,y,x] · Sdt[p,(py_k+y)%SY,(px_k+x)%SX]``

    Args:
        Sdt, basis_fft, positions, dr: see :func:`_prism_project_torch`.
        chunk: Number of positions to process at once.  ``K`` processes all
            positions in a single pass.  ``None`` (default) does too, except on
            the Warp route with no gradient live and an active sub-pixel ramp,
            where a block sized by the projection's byte budget bounds the
            shifted basis (see :func:`_warp_budget_chunk`).  Smaller values
            bound peak memory proportionally: only ``chunk`` positions are
            materialised at once in ``_build_basis_rs``.  The result is
            bit-identical regardless of ``chunk`` (chunking only changes
            batching, not arithmetic).

            That saving applies only when the sub-pixel ramp is ACTIVE.  When
            every ``dr`` is zero, ``_build_basis_rs`` folds the K axis to a
            single broadcast slice that is already K-independent, so there is
            nothing left to bound and chunking costs a little memory rather
            than saving it (the preallocated full-K output plus a per-chunk
            slice).  Chunk for memory only when positions carry sub-pixel
            offsets.
        precision: "fp32" (default, unchanged fp32 FFT), "mixed" (complex32
            storage projected with fp32 arithmetic — a per-element register cast
            in the Warp kernel, no full-tensor upcast), or "fp16" (complex32
            storage projected with fp16 arithmetic). Warp path only — the torch
            oracle is always fp32.
        periodic: If ``True`` (default), use cyclic-crop (periodic boundary
            conditions — correct for crystalline specimens).  If ``False``,
            out-of-bounds contributions are zeroed out (non-periodic boundary
            — correct for finite specimens such as nanoparticle tips).

    Returns:
        exit_wave: (Nmodes, K, M1, M2) complex.

    Differentiable in ``Sdt``, ``basis_fft`` and ``dr``.  The Warp ``custom_op``
    carries a hand adjoint
    (cyclic-crop gather + atomic scatter-add); torch autograd then propagates
    ``grad_basis_rs`` back through the ramp + ifft2 to ``grad_basis_fft`` and
    ``grad_dr`` for free.  When chunked AND any input requires grad, the Warp
    route runs the whole chunk loop as ONE autograd node
    (``_chunked._PrismProjectChunked``), so its backward can accumulate
    ``grad_Sdt`` into a single buffer; the torch-oracle route keeps the per-chunk
    ``torch.cat``.

    Note on ``chunk`` UNDER GRAD: chunking bounds the ``grad_basis_rs`` peak, NOT
    the whole backward.  Every chunk's ``basis_rs`` stays alive for its own
    adjoint, and ``grad_Sdt`` is the size of the S-matrix regardless of ``chunk``
    -- measured at ``r2/medium``, ``chunk = K/4`` buys 1.18x less peak for 1.39x
    more time and ``chunk = 1`` buys 1.22x for 1.79x (with one shared accumulator; before
    it those costs were 1.87x and 3.02x).
    """
    K = positions.shape[0]
    Nmodes = basis_fft.shape[0]
    M1, M2 = basis_fft.shape[-2], basis_fft.shape[-1]

    # Warp vs torch: explicit ``use_warp``, else SCATTEREM_USE_WARP (``"0"`` is
    # torch), else torch -- the ``prism_project`` row of ``scatterem.nn._backends``.
    _uw = use_warp = resolve_use_warp(PRISM_PROJECT, use_warp)
    basis_fft = as_route_basis(basis_fft, _uw, Sdt, dr)

    requires_grad = torch.is_grad_enabled() and any(
        t.requires_grad for t in (Sdt, basis_fft, dr)
    )
    ident: bool | None = None
    if chunk is None and _uw and not requires_grad:  # the Warp byte budget
        chunk, ident = _warp_budget_chunk(basis_fft, dr, K)

    # Resolve chunk size: None or larger-than-K -> single pass (no loop overhead).
    if chunk is None or chunk >= K:
        if _uw:
            return _prism_project_warp(
                Sdt, basis_fft, positions, dr, precision, periodic, ident
            )
        # oracle is always fp32
        return _prism_project_torch(Sdt, basis_fft, positions, dr, periodic=periodic)

    # Position-chunked loop: positions[i : i+chunk] independently; peak ~ chunk.
    # ONE device->host transfer for the whole call instead of one PER CHUNK.
    # ``_ramp_is_identity`` ends in ``bool(dr.any())``, a sync, and it sits at the
    # top of every chunk's projection -- so a K-position call chunked n ways paid n
    # syncs, each stalling the loop mid-flight.  Measured cost saturates at
    # ~0.16 ms per sync on an A6000, i.e. 1.0020x at 1 chunk but **1.0496x at 8**
    # (a one-off probe), because the GPU work per call is fixed while the
    # sync count grows -- and fine chunking is exactly the large-volume regime
    # ``chunk`` exists for (peak 5380 -> 1797 MiB over that same sweep).
    #
    # Per-chunk semantics are UNCHANGED, which is why this resolves a per-POSITION
    # mask rather than one whole-tensor answer: chunk i folds iff its own ``dr``
    # rows are all zero, exactly as ``_ramp_is_identity(basis_fft, dr[i:i+chunk])``
    # would have decided.  Collapsing to ``not dr.any()`` for the whole call would
    # be cheaper still and is WRONG -- a mixed ``dr`` has all-zero chunks that must
    # keep folding.  ``dr != 0`` matches ``dr.any()`` element for element (both
    # treat -0.0 as zero and NaN as nonzero).
    if torch.is_grad_enabled() and dr.requires_grad:
        # The fold is refused for every chunk (it drops the ramp from the graph, so
        # a differentiable dr would silently get no gradient -- see
        # _ramp_is_identity).  No mask needed, so no sync at all.
        dr_nonzero = None
    else:
        dr_nonzero = (dr != 0).any(dim=1).tolist()

    def _project_chunk(pos_sl: Tensor, dr_sl: Tensor, ident: bool | None) -> Tensor:
        if use_warp:
            return _prism_project_warp(
                Sdt,
                basis_fft,
                pos_sl,
                dr_sl,
                precision=precision,
                periodic=periodic,
                ramp_identity=ident,
            )
        # oracle is always fp32
        return _prism_project_torch(
            Sdt, basis_fft, pos_sl, dr_sl, periodic=periodic, ramp_identity=ident
        )

    def _ident(i: int) -> bool:
        return False if dr_nonzero is None else not any(dr_nonzero[i : i + chunk])

    if requires_grad and use_warp:
        # ONE autograd node, so the backward shares ONE grad_Sdt buffer.
        args = (Sdt, cast(Tensor, basis_fft), positions, dr, chunk, periodic, _ident)
        return _prism_project_warp_chunked(*args)
    if requires_grad:
        # Torch-oracle route.  Out-of-place accumulation: in-place index assignment
        # into a pre-allocated buffer is not autograd-compatible.  torch.cat keeps
        # the graph intact while still bounding peak memory to ~chunk positions
        # (each chunk's basis_rs is freed after its output is produced).
        chunks = [
            _project_chunk(positions[i : i + chunk], dr[i : i + chunk], _ident(i))
            for i in range(0, K, chunk)
        ]
        return torch.cat(chunks, dim=1)

    # No-grad fast path: write each slice into a pre-allocated buffer.
    # Always complex64: _prism_project_warp/_prism_project_torch always return
    # complex64 (fp32 accumulator even when Sdt is complex32 for the mixed
    # preset), so allocating with Sdt.dtype would silently narrow the result
    # when Sdt is complex32.
    out = torch.empty((Nmodes, K, M1, M2), dtype=torch.complex64, device=Sdt.device)
    for i in range(0, K, chunk):
        sl = slice(i, i + chunk)
        out[:, sl] = _project_chunk(positions[sl], dr[sl], _ident(i))
    return out
