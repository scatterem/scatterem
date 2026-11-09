"""The readable torch oracle the fused Warp route is graded against.

Must stay PURE TORCH: the ``warp_matches_torch`` / ``grad_matches_torch_route``
invariants use it as the reference.  The fused CUDA shortcuts of its block loop
(``_torch_blocks``) and adjoint (``_torch_adjoint``) are backend rows that decline
to the torch spelling.
"""

from __future__ import annotations

import torch
from torch import Tensor

from scatterem.physics._frequency_grid import fft_frequency_grid

from ._basis import BeamletBasis
from ._torch_adjoint import _CropContractFrozenBasis
from ._torch_blocks import _contract_blocks, _Crop, _grad_block_basis


def _ramp_is_identity(basis_fft: Tensor | BeamletBasis, dr: Tensor) -> bool:
    """Whether the real-space basis may be built once and broadcast over K.

    ``dr == 0`` (an integer-pixel scan — the common forward-simulation case) makes
    the sub-pixel ramp ``exp(0)`` exactly ``1+0j``, and a complex64 multiply by
    ``(1, 0)`` is the bit-for-bit identity.  Every one of the K slices of the
    shifted basis is then equal to ``basis_fft``, so the per-position ``ifft2``
    repeats one transform K times: a single ifft2 over the ``(Nmodes, Bp)`` batch,
    broadcast back over K as a stride-0 view, replaces it.  cuFFT is
    slice-for-slice bit-identical between the two batch counts, so the FORWARD is
    unchanged element for element — see a benchmark script,
    which checks that against the pre-change code for both routes.

    The BACKWARD is *not* bit-identical: the adjoint of the broadcast sums the K
    cotangents and then transforms once, where the old path transformed each of
    the K slices and summed afterwards.  That is pure reassociation of a
    mathematically equal sum (measured rel 2e-7, ~2 ULP of fp32).  An earlier version declined the
    fold under grad *for that reason alone*, because the R2 gradient references were
    then gated below fp32's own reproducibility floor; a later measurement showed that the same
    gate rejected complex128 truth and re-derived it at ``sqrt(N_terms)*eps``, which
    removes the objection.  So the fold now applies to ``basis_fft.requires_grad``
    too — the PRISM reconstruction direction with a refined probe, where the K-fold
    redundant transform was being paid on the forward *and* again on the backward.

    ``dr.requires_grad`` still refuses, and that one is not a tolerance question:
    the fold drops the ramp from the graph entirely, so ``dr`` would silently get no
    gradient rather than a slightly reassociated one.  Do not widen it further.
    """
    if torch.is_grad_enabled() and dr.requires_grad:
        return False
    return not bool(dr.any())


#: Byte budget for ONE ``(Nmodes, PB, nk, M1, M2)`` slab of ``_prism_project_torch``.
#:
#: The body is blocked over scan positions (``_torch_project_block``) and, once a
#: single position exceeds the budget, over parent beams as well
#: (``_torch_parent_block``), so the ramp multiply, the ``ifft2``, the cyclic-crop
#: gather and the parent-beam contraction all run on one slab and only the
#: ``(Nmodes, K, M1, M2)`` result survives; the loop holds ~3 slabs at once.
#:
#: 128 MiB is the knee measured at ``r2_prism_project/medium``: the row is
#: FLAT in the block size (47.4 / 45.5 / 44.5 / 44.1 / 43.8 ms at 8 / 16 / 32 / 64 /
#: 128) while peak is strictly LINEAR in it, so the budget buys back memory at
#: almost no time.
_TORCH_PROJECT_BLOCK_BYTES = 128 * 1024 * 1024


def _items_per_budget(item_bytes: int, n: int) -> int:
    """How many items of ``item_bytes`` each fit :data:`_TORCH_PROJECT_BLOCK_BYTES`.

    The one budget rule of the projection, shared by both routes: the torch
    oracle's position and parent blocks below and the Warp route's position
    block (``dispatch._warp_budget_chunk``).  The budget is read at call time,
    so a test can monkeypatch it.

    Parameters
    ----------
    item_bytes : int
        Bytes one item (a scan position, or a parent beam) adds to the slab.
    n : int
        Number of items there are.

    Returns
    -------
    int
        Items per block, clamped to ``[1, n]``: an item larger than the whole
        budget still gets a block of one.
    """
    return max(1, min(n, _TORCH_PROJECT_BLOCK_BYTES // max(1, item_bytes)))


def _torch_project_block(basis_fft, K: int) -> int:
    """Scan-position block size for :func:`_prism_project_torch`'s byte budget.

    The budget bounds the ``(Nmodes, PB, nk, M1, M2)`` intermediate, ``PB`` being
    the parent-beam block of :func:`_torch_parent_block` (all of ``Bp`` unless ONE
    position already exceeds the budget).

    Blocking ``k`` cannot change a result: every block reduces the same parents
    in the same order, so each output is the same sum of the same terms at every
    block size -- byte-identical to ``KB = K``, i.e. to the single-pass form the
    grad path keeps below.

    ON THE CPU A BLOCK IS ONE POSITION.  The sum is the same at every block
    size; the PRODUCT under it is not.  ATen's threaded complex multiply rounds
    the last few elements of each thread's share differently from the rest (its
    scalar tail contracts ``ar*cr - ai*ci`` into a fused multiply-add, its
    vector body does not: 14 of 324,608 elements at 16 threads, none at one),
    so the bits of a position depended on where it sat in the block, hence on
    how the caller chunked its scan -- a chunk of exactly one position took the
    contiguous kernel where every other chunking took the broadcast one, and
    the simulated image moved by 1e-5.  With one position per block every
    position is the same kernel on the same shapes, whatever ``K`` is.  It is
    also the faster spelling there: the contiguous kernel beats the broadcast
    one (20.4 -> 13.0 ms at ``Bp = 317, M = 32``, 74.7 -> 22.8 ms at ``Bp =
    1000, M = 64``).  A CUDA kernel computes every element with the same code,
    so blocks there are sized by the byte budget (one position per block would
    cost a kernel launch each: 0.57 -> 9.3 ms at ``Bp = 61, M = 16, K = 512``).
    """
    if basis_fft.device.type == "cpu":
        return 1
    Nmodes, _, M1, M2 = basis_fft.shape
    per_k = Nmodes * _torch_parent_block(basis_fft) * M1 * M2 * basis_fft.element_size()
    return _items_per_budget(per_k, K)


def _torch_parent_block(basis_fft) -> int:
    """Parent-beam block size: all of ``Bp`` while ONE position fits the budget.

    The position block above bounds nothing once a single position's
    ``(Nmodes, Bp, M1, M2)`` exceeds the budget -- and at f = 1 that is the size
    of the S-matrix: the basis, its transform, the crop and their product were
    each one more copy of it (pure PRISM peaked at 5.2x its S-matrix, 25.7 GB
    for 4.9 GB at 256 px).  So the parent axis is blocked too.

    The size depends on ``(Nmodes, M1, M2)`` and the budget ONLY, never on
    ``K``: the partial sums are then the same partition of the parents however
    the caller chunked its scan, which keeps the chunk-size bit-identity.  (An
    earlier parent blocking was abandoned because ITS size was derived from
    ``K``: 457 beams in ten groups at ``K = 36`` and two at ``K = 7`` moved the
    cube by 5e-4.)  While one position fits there is one block, and the
    arithmetic is what it always was.
    """
    Nmodes, Bp, M1, M2 = basis_fft.shape
    per_p = Nmodes * M1 * M2 * basis_fft.element_size()
    return _items_per_budget(per_p, Bp)


# Kept copy: builds its own Fourier-shift ramp (not physics.fourier.phase_ramp), as
# the independent r2_prism_project oracle must share no code with what it checks.
def _prism_project_torch(
    Sdt: Tensor,
    basis_fft: Tensor | BeamletBasis,
    positions: Tensor,
    dr: Tensor,
    periodic: bool = True,
    ramp_identity: bool | None = None,
) -> Tensor:
    """Readable torch reference for PRISM exit-wave projection.

    This is the oracle implementation: it is pure torch, and it is the reference
    the ``warp_matches_torch`` / ``grad_matches_torch_route`` invariants grade the
    fused Warp route against, so it must stay pure torch.  It is no longer the
    memory hog its name suggests: the body is blocked over the parent-beam axis
    under ``_TORCH_PROJECT_BLOCK_BYTES``, so the per-position basis and the
    cropped S-matrix patches are materialised one block of ``p`` at a time.

    Args:
        Sdt:        (Bp, SY, SX) complex — de-tilted S-matrix columns.
                    ``Sdt[p, r] = S[p, r] * exp(-2πi h_p · r)``
                    (output of the detilt step in PrismScanOp.forward step 4).
        basis_fft:  (Nmodes, Bp, M1, M2) complex, as a tensor or FACTORED
                    (``basis.BeamletBasis``, scattered one block of parents at
                    a time) — position-independent beamlet
                    basis in Fourier space.  ``basis_fft[m, p, qy, qx]`` holds
                    the weighted probe amplitude ``Σ_b w_{p,b} Ψ_m(h_b)``
                    scattered onto the (M1, M2) beam grid (no K axis, no IFFT,
                    no normalisation prefactor — that lives in ifft2 below).
        positions:  (K, 2) int — integer scan positions (py_k, px_k) in object
                    pixels.  The cyclic crop ``Sdt[p, (py_k+y)%SY, (px_k+x)%SX]``
                    consumes this (when ``periodic=True``).
        dr:         (K, 2) float — sub-pixel offsets in PIXELS (dy_k, dx_k).
                    The Fourier-shift ramp
                    ``exp(-2πi (qy·dry_k + qx·drx_k))`` applies these.
                    Pass ``torch.zeros((K, 2), ...)`` for dr=0.
        periodic:   If ``True`` (default), the crop uses modulo arithmetic so
                    S-matrix columns wrap cyclically — correct for periodic
                    crystals.  If ``False``, out-of-bounds indices contribute
                    zero — correct for finite specimens.

    Returns:
        exit_wave:  (Nmodes, K, M1, M2) complex — projected exit waves.

    Notes:
        - ``dr`` uses PIXEL units and pixel-frequency
          ``q = fft_frequency_grid((M1, M2), [1, 1])`` (``scatterem.physics._frequency_grid``),
          exactly mirroring ``SubpixShift`` (modules.py:993).  No physical ``dx``
          is needed or used.
        - The IFFT normalisation matches the current ``PrismScanOp.forward``
          step 6.  The amplitude wanted there is the ORTHO inverse transform with
          its ``1/sqrt(M1*M2)`` cancelled again, i.e. the plain unnormalised
          inverse DFT -- so it is spelled ``norm="forward"`` (torch's name for
          "put the whole ``1/N`` on the forward transform, none on the inverse")
          rather than as ``ifft2(..., norm="ortho") * (M1*M2)**0.5``.  The two are
          the same transform; the second spelling buys the ``sqrt`` back with an
          extra read-modify-write pass over a tensor the size of the shifted
          basis -- 2 GiB read + 2 GiB written at ``r2_prism_project/medium``, to
          multiply by a constant.  The same defect was measured one target over
          (``_direct_ptychography``, 15.0 % of its device time) and this is the
          same fix inside a different library call.  The swap is BIT-EXACT
          (``torch.equal``) whenever ``sqrt(M1*M2)`` is an exact power of two --
          true of every power-of-two grid, incl. both registered ``r2`` sizes --
          because scaling by ``1/sqrt(N)`` and back then cancels with no
          rounding; on other shapes it deviates by 6-10e-08 of peak with
          strictly FEWER roundings, since this IS the unnormalised transform
          rather than a scaled-then-unscaled one (a one-off probe).
        - At ``dr=0`` the ramp is identically 1 and the result is bit-identical
          (within fp32 rounding) to applying it explicitly.  (This note used to
          say "to the current einsum path"; there is no longer an einsum path --
          see the contraction comment in section 3+4.)
    """
    Bp, SY, SX = Sdt.shape
    Nmodes, Bp2, M1, M2 = basis_fft.shape
    K = positions.shape[0]
    device = Sdt.device
    dtype = Sdt.dtype  # complex64

    assert Bp == Bp2, f"Sdt Bp={Bp} != basis_fft Bp={Bp2}"

    # ------------------------------------------------------------------
    # 1. Build the sub-pixel ramp for each position k.
    #    Shape: (K, M1, M2).
    #    q in pixel-frequency units (matching SubpixShift convention).
    # ------------------------------------------------------------------
    # Identity ramp: build the real-space basis once and broadcast over K.
    # See _ramp_is_identity for why this is bit-exact in the forward.
    ramp_id = (
        _ramp_is_identity(basis_fft, dr) if ramp_identity is None else ramp_identity
    )
    if ramp_id:
        ramp = None
    else:
        q = fft_frequency_grid((M1, M2), [1, 1], device=str(device))  # (2, M1, M2)
        qy = q[0]  # (M1, M2)
        qx = q[1]  # (M1, M2)

        dry = dr[:, 0]  # (K,)
        drx = dr[:, 1]  # (K,)

        # (K, M1, M2) — same shape logic as SubpixShift — carried with the two
        # leading singleton axes the blocked body broadcasts it against.
        ramp = torch.exp(
            (-2j * torch.pi)
            * (
                qy[None, :, :] * dry[:, None, None]
                + qx[None, :, :] * drx[:, None, None]
            )
        ).to(dtype)[None, None]  # (1, 1, K, M1, M2) complex64

    # ------------------------------------------------------------------
    # 2. Index grid for the crop of Sdt at each position.
    #    Periodic:     S_dt_crop[p, k, y, x] = Sdt[p, (py_k+y)%SY, (px_k+x)%SX]
    #    Non-periodic: same but out-of-bounds entries contribute 0.
    # ------------------------------------------------------------------
    py = positions[:, 0].long()  # (K,)
    px = positions[:, 1].long()  # (K,)
    my = torch.arange(M1, device=device)  # (M1,)
    mx = torch.arange(M2, device=device)  # (M2,)
    gy_raw = py[:, None, None] + my[None, :, None]  # (K, M1, 1)  raw (unmodded)
    gx_raw = px[:, None, None] + mx[None, None, :]  # (K, 1, M2)  raw (unmodded)
    if periodic:
        gy = gy_raw % SY  # (K, M1, 1)
        gx = gx_raw % SX  # (K, 1, M2)
        in_bounds = None
    else:
        # Out-of-bounds mask: True where (gy_raw, gx_raw) is within bounds.
        in_bounds = ((gy_raw >= 0) & (gy_raw < SY) & (gx_raw >= 0) & (gx_raw < SX))[
            None
        ]  # (1, K, M1, M2) via broadcasting, one leading axis for the p block
        # Clamp indices for the gather (safe even where masked).
        gy = gy_raw.expand(K, M1, M2).clamp(0, SY - 1)  # (K, M1, M2)
        gx = gx_raw.expand(K, M1, M2).clamp(0, SX - 1)  # (K, M1, M2)

    # The crop below is spelled as ONE ``index_select`` against a flattened
    # ``(Bp, SY*SX)`` view rather than as the two-index advanced-indexing form
    # ``Sdt[blk, gy, gx]``.  It is the SAME permutation -- ``torch.equal``, and a
    # gather cannot reassociate anything -- but it is 1.73x faster, because
    # ATen's multi-index ``index`` kernel does its own broadcast + 64-bit address
    # arithmetic per output element and never vectorises, while ``index_select``
    # over a single 32-bit index runs at the copy roof.  Measured in situ at
    # ``r2_prism_project/medium`` (one 128 MiB
    # block, all arms ``torch.equal`` to the shipped one, against a 632.5 GB/s
    # same-process copy roof):
    #
    #     Sdt[blk, gy, gx]                  0.746 ms   334.9 GB/s   (shipped)
    #     index_select, int64 index         0.575 ms   434.4 GB/s
    #     index_select, int32 index         0.431 ms   579.9 GB/s   <- this
    #     Sdt_flat[blk][:, flat_i64]        0.527 ms   474.1 GB/s
    #     22 contiguous rectangle copies    0.496 ms   504.4 GB/s
    #     rowsel + torch.gather             0.967 ms   258.6 GB/s
    #
    # The int32 index is worth 1.33x over the int64 one on its own: the index is
    # re-read once per row of the output, so halving it is what makes it stay
    # L2-resident.  ``SY*SX`` is the largest value it must hold, so the dtype is
    # chosen from that and NOT assumed -- an S-matrix with more than 2^31 pixels
    # per parent beam falls back to int64 and merely loses the 1.33x.
    #
    # The BACKWARD is where most of the prize is, and it was not predicted: the
    # adjoint of the two-index form is ``index_put_(..., accumulate=True)``,
    # which ATen serves with its sorted, deterministic scatter; the adjoint of
    # ``index_select`` is ``index_add_``, a plain atomic scatter over the same
    # elements.  Same arithmetic, same memory, **4.05x** on
    # ``medium/project_backward`` (131.049 -> 32.368 ms) and **4.87x** at
    # ``small``.  The price is that ``grad_Sdt`` now reassociates: it moved
    # 7.3e-08 of peak against the HEAD spelling, and the oracle's
    # warp-vs-torch residual 1.04e-07 -> 1.56e-07 against a 1e-4 tolerance.
    # ``grad_basis_fft`` is ``torch.equal`` and the forward is ``torch.equal``
    # on all 32 routes probed (periodic/finite x dr=0/dr!=0 x wrapping x
    # non-power-of-two x contiguous/not; a one-off probe).
    #
    # The flat view must be free: ``reshape`` on a non-contiguous ``Sdt`` would
    # copy the whole S-matrix (1 GiB at ``medium``), so a non-contiguous caller
    # keeps the original spelling rather than paying for the faster one.  The
    # index itself is bounded by the tensor it indexes into: it is
    # ``K*M1*M2*4`` bytes against the crop's ``PB*K*M1*M2*8``, i.e. never more
    # than half of ONE parent beam's slab (2.0 MiB of the 128 MiB block at
    # ``medium``, and that 2.0 MiB is the whole of this change's peak cost).
    if Sdt.is_contiguous():
        _crop_idx = gy * SX + gx  # (K, M1, 1) + (K, 1, M2) -> (K, M1, M2)
        _crop_idx = _crop_idx.to(
            torch.int32 if SY * SX < 2**31 else torch.int64
        ).reshape(-1)
        Sdt_flat = Sdt.view(Bp, SY * SX)
    else:
        _crop_idx = None
        Sdt_flat = None

    # ------------------------------------------------------------------
    # 3+4. Ramp, IFFT, crop and the contraction over parent beams p, BLOCKED
    #      over k -- and over p once one position exceeds the budget -- so no
    #      (Nmodes, Bp, K, M1, M2) and no (Nmodes, Bp, M1, M2) is ever resident.
    #
    #      Per block:  basis_fft[:, blk, None] * ramp -> ifft2 -> * Sdt[blk, gy, gx]
    #                  -> sum over the block's p, accumulated into exit_wave.
    #
    #      This is the same arithmetic reassociated: the parent-beam sum is
    #      performed block by block instead of in one einsum.  The contraction is
    #      spelled ``(a*b).sum(1)`` and NOT ``einsum("mkpyx,kpyx->mkyx", ...)``,
    #      on BOTH paths.  einsum is not merely equivalent but SLOWER, because it
    #      reduces this pattern through ``bmm``, which has to make both operands
    #      contiguous in a layout neither is in: 3.02x forward at ``medium``, at
    #      identical peak (the permutes themselves are worth 0.04 ms of 37.7;
    #      a one-off probe).  Its ADJOINT is worse still and in both
    #      currencies -- ``bmm``'s backward is two more ``bmm``s, each re-paying
    #      for its own contiguous copies, where ``mul``+``sum``'s is a stride-0
    #      expand plus two elementwise passes.  Measured in situ at ``medium``,
    #      ``PB = Bp`` held fixed in both arms so this is purely the spelling
    #      (2 cycles, all populations DISJOINT):
    #
    #          project_backward        191.333 -> 131.141 ms  1.4590x  5144 -> 4096 MiB
    #          project_fwd_backward    261.606 -> 175.016 ms  1.4948x  6153 -> 6153 MiB
    #          project_backward_joint  268.498 -> 173.236 ms  1.5499x  7192 -> 6144 MiB
    #
    # ------------------------------------------------------------------
    # The BLOCKING, by contrast, is a WIN with no grad and a measured TRADE under
    # it, so THAT is gated: when a graph is being built, every block's
    # intermediates stay live for the backward, so the peak win shrinks while the
    # adjoint pays one set of nodes per block.  A measurement of the ungated version
    # at ``medium``, A/B in alternating separately-launched arms, all three
    # populations DISJOINT:
    #
    #     project_backward        191.526 -> 231.745 ms  0.8265x   5144 -> 2112 MiB
    #     project_fwd_backward    260.713 -> 276.145 ms  0.9441x   8201 -> 3912 MiB
    #     project_backward_joint  267.679 -> 298.050 ms  0.8981x   7192 -> 2496 MiB
    #
    # Trading 6-17 % of the RECONSTRUCTION direction's time for 2.1-2.9x less peak
    # is a real trade and a human call, not one to make unattended -- so the
    # graph-building path keeps ``PB = Bp``, i.e. the single-pass memory profile it
    # has always had.  The forward-only path (simulation, and every
    # ``torch.no_grad()`` caller) takes the blocked loop.  Note that the gate is on
    # ``PB`` ALONE: the contraction spelling above is a win in both directions, so
    # unlike that change it is not gated and needs no separate grad arm.
    grad_path = torch.is_grad_enabled() and any(
        t.requires_grad for t in (Sdt, basis_fft, dr)
    )
    KB = K if grad_path else _torch_project_block(basis_fft, K)
    PB = Bp if grad_path else _torch_parent_block(basis_fft)
    if K == 0:
        # The einsum this replaces returned an empty ``(Nmodes, 0, M1, M2)`` for
        # that input, so keep doing so.
        return Sdt.new_empty((Nmodes, 0, M1, M2))

    # ``Sdt`` needs a gradient (frozen probe, or joint) and the grad path runs as
    # ONE parent block and ONE position block: same forward, hand adjoint --
    # conj on ``grad`` not ``basis_k``, per-position window scatter
    # (see ``_torch_adjoint``).  ``0 + x == x`` exactly, so the
    # forward is bit-identical to ``_contract_blocks``' single block.
    if (
        grad_path
        and PB == Bp
        and KB == K
        and K > 0
        and Sdt.requires_grad
        and Sdt_flat is not None
        and in_bounds is None
        and isinstance(basis_fft, Tensor)
    ):
        if ramp is None:
            # the UNEXPANDED (Nmodes, Bp, 1, M1, M2) basis -- the Function
            # broadcasts it over K, so its gradient comes back already K-summed.
            basis_k = torch.fft.ifft2(basis_fft, norm="forward")[:, :, None]
        else:  # the Warp ramp multiply where it pays (K-outer view)
            basis_k = _grad_block_basis(basis_fft, ramp, Sdt)
        args = (basis_k, Sdt_flat, _crop_idx, (Bp, K, M1, M2), positions, (SY, SX))
        return _CropContractFrozenBasis.apply(*args)

    crop = _Crop(Sdt, Sdt_flat, _crop_idx, gy, gx, in_bounds)
    return _contract_blocks(basis_fft, ramp, crop, (Nmodes, Bp, K, M1, M2), PB, KB)
