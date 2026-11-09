"""Linear-scaling double-channeling PRISM STEM-EELS *image* (Brown & Ophus).

This is the detector-integrated counterpart of
:func:`scatterem.simulation.eels.forward.prism_eels.prism_transition_potential`. It
produces a 2D STEM-EELS map ``I(x, y)`` for a chosen edge using **two** scattering
matrices and per-atom transition-potential cropping, the algorithm that makes the
FePt-nanoparticle-scale simulation of Brown, Ciston & Ophus, Phys. Rev. Research
**1**, 033186 (2019) tractable:

* ``S1`` (``ScatteringMatrix``) propagates the probe-forming beams to the
  ionization slice (reused across all scan positions).
* ``S2`` propagates the inelastically-scattered wave from the ionization slice to
  the **detector** -- built once per slice as a *transpose* multislice of the
  detector-accepted output beams and peeled forward slice by slice, so it is
  reused across **all atoms** in that slice.
* For each ionized atom the small matrix ``Sn[d, b] = sum_r S2_d(r) H_n0(r) S1_b(r)``
  is formed over a crop window around the atom (the transition potential is
  localized), then applied to every scan illumination vector:
  ``I(R) += sum_d |sum_b Sn[d, b] psi_illum[b, R]|^2`` over the PRISM crop region.
  ``Sn`` is the operator ``S_n`` of Eq. (11) of the paper, restricted to the
  detector beams and the crop window (Sec. IV B).

Because ``S2`` is built per slice (not per atom) and the per-atom work is a small
cropped matmul, the cost is independent of the number of probe positions and scales
benignly with the number of ionized atoms -- unlike the 4D dual-S, which re-propagates
per atom. The transpose-multislice ``S2`` reproduces ``fft(forward_exit(w))`` exactly,
so with no crop this equals the detector-integrated 4D dual-S.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import pi
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from scatterem.physics.beam_partition import BeamPartition
from scatterem.physics.propagation import propagator_kernel
from scatterem.simulation._wave.partitioned_smatrix import PartitionedScatteringMatrix
from scatterem.simulation._wave.prism_beams import _factor_pair
from scatterem.simulation._wave.smatrix import ScatteringMatrix

from ..physics.transition_potentials import TransitionPotentials
from ._detector_exit import (
    DetectorExitSMatrix,
    _periodic_window,
)
from ._image_tables import (
    _crop_window,
    _detector_beams,
    _fold_detectors,
    _qeels_bins,
    _slice_tables,
)
from ._reconstruction_reuse import (
    _s1_window,
    _s2_window,
    _slice_reconstructions,
    _workspace_bytes,
)
from ._scan_accumulator import _OrderedScanAccumulator
from ._slice_plan import _SlicePlan

__all__ = ["DetectorExitSMatrix", "prism_eels_image"]


# Upper bound (bytes) on the transition-batched per-atom intermediate in
# prism_eels_image; the n_trans loop is chunked to keep peak memory under this
# while still batching enough transitions to fill the GEMM. ~2 GiB comfortably
# batches every FePt / LAO-STO edge (incl. La-M, n_trans=75) in one shot.
_EELS_TRANS_BLOCK_BYTES = 1 << 31

# Upper bound (bytes) on the full-grid intermediate of the batched per-atom
# sub-pixel shift below; the atom axis is chunked to keep the peak at a few
# atoms' worth of what the per-atom loop already allocated one at a time.
_EELS_SHIFT_BLOCK_BYTES = 1 << 25

# Upper bound (bytes) on one slice's stacked crop windows in _crop_stack_or_none.
# Same role and same value as _EELS_SHIFT_BLOCK_BYTES: keep the batched form's peak
# at a few atoms' worth of what the per-atom loop already allocated one at a time.
# Above it the per-atom path is taken, which is what shipped before the batching.
_EELS_CROP_BLOCK_BYTES = 1 << 25

# Upper bounds (bytes) on one slice's stacked `h * S1` product and on one slice's
# stacked masked coefficients.  Same role and same value as the two caps above:
# keep the batched form's peak at a few atoms' worth of what the per-atom loop
# already allocated one at a time; above them the per-atom path is taken, which is
# what shipped before the batching.
_EELS_HS1_BLOCK_BYTES = 1 << 25
_EELS_COEFF_BLOCK_BYTES = 1 << 25

# Which of the two mathematically identical contraction orders to use for the
# per-atom coupling (see :func:`_contract_orders` and the per-atom body of
# :func:`prism_eels_image`).  "auto" picks the cheaper one from the actual shapes;
# "window" and "probe" force one, which is what the equivalence tests exercise.
_EELS_CONTRACT_ORDER = "auto"

# Cost of one ``"window"`` FLOP relative to one ``"probe"`` FLOP, MEASURED.  The two
# orders' FLOP counts are both right; what a FLOP count cannot see is that the
# ``"window"`` order has to materialise and stage the ``(n_trans, nbeams, wy, wx)``
# product before it can contract it, so each of its FLOPs arrives with an extra full
# pass of memory traffic that the ``"probe"`` order never pays.  Measured on an
# RTX A6000 the ``"window"`` order retires its FLOPs ~5x slower, which is why the
# unweighted comparison switches to it about 5x too early in ``Pm``.  See
# :func:`_contract_orders` for the calibration table.
_EELS_WINDOW_COST_SCALE = 5.0


def _contract_orders(
    counts: Sequence[int],
    n_trans: int,
    ndet: int,
    nbeams: int,
    W: int,
    forced: str | None = None,
) -> list:
    """One slice's per-atom contraction order, decided entirely on the HOST.

    The per-atom coupling is a three-way contraction over the crop window, and it
    can be associated two ways.  Both compute the same sum; they differ only in
    which pair is contracted first:

    * ``"window"`` -- contract the window for every ``(t, d, b)`` triple to get
      ``Sn[t,d,b] = sum_wv S2c[d,wv] h[t,wv] S1c[b,wv]``, then project onto the
      scan, ``amp[t,p,d] = sum_b Sn[t,d,b] c[p,b]``.  ``Sn`` is
      scan-independent, so it is reused across the atom's scan positions.
      cost ~ ``n_trans*ndet*nbeams*W + n_trans*Pm*ndet*nbeams``
    * ``"probe"`` -- synthesize the probe on the window first (the physical order:
      ``psi = c @ S1c`` is the PRISM probe restricted to the window), multiply by
      the transition potential and propagate to the detector,
      ``amp[t,p,d] = sum_wv S2c[d,wv] h[t,wv] psi[p,wv]``.
      cost ~ ``Pm*nbeams*W + n_trans*Pm*W*ndet``

    The ratio of the leading terms is ``nbeams/Pm``, so ``"probe"`` wins whenever
    fewer scan positions contribute to this atom than there are aperture beams --
    the common case, and strongly so under the PRISM aliasing mask where
    ``Pm ~ P/(fy*fx)``.  It also never materialises the
    ``(n_trans, nbeams, wy, wx)`` intermediate that dominates memory traffic in the
    ``"window"`` order, which is why the caller skips
    :func:`_hs1_stack_or_none` outright when no atom of the slice takes it.

    **The two costs are not comparable FLOP for FLOP, and that is what
    ``_EELS_WINDOW_COST_SCALE`` corrects.**  Comparing them unweighted puts the
    crossover at ``Pm = nbeams``; measured, it is at ``Pm ~ 5*nbeams``, because
    ``"window"`` time is nearly *independent* of ``Pm`` (it is dominated by staging
    the intermediate above, which is scan-independent) while ``"probe"`` starts far
    cheaper and grows linearly in ``Pm``.  Calibration, whole-call medians on an
    RTX A6000, ``auto`` against each order forced, ``n_trans = 27`` throughout:

    ===================================== ====== ======= ====== ====== ==========
    geometry                              nbeams Pm      probe  window truth
    ===================================== ====== ======= ====== ====== ==========
    128 px, f=2, no crop (W=16384, nd=37)     49      16   4.74  12.14 probe
    128 px, f=2, no crop                      49      64   6.34  12.03 probe
    128 px, f=2, no crop                      49     144   8.86  12.00 probe
    128 px, f=2, no crop                      49     256  12.38  12.03 window
    192 px, f=4, crop 22 (W=64, nd=21)        29      64 107.51 115.96 probe
    192 px, f=4, crop 22                      29     100 108.91 115.35 probe
    256 px, f=4, crop 22 (W=121, nd=37)       49     100  27.64  28.01 probe
    ===================================== ====== ======= ====== ====== ==========

    A scale of 5 reproduces the truth on all of those, on the 128 px sweep's
    remaining points, and on an independent ``ndet = 137`` sweep (probe at ``Pm = 144``, window at
    ``Pm = 272`` and ``1024``).
    It is a **calibration**, not a derivation: it was fitted to the 128 px sweep and
    then checked against three further ``(n_trans, ndet, nbeams, W)`` geometries it
    was not fitted to.  Getting it wrong costs time and nothing else -- both orders
    are mathematically identical, so a mis-dispatch is a missed win, never a wrong
    answer, which is why a measured constant is an acceptable form of fix here.

    ``counts`` is the slice's per-atom ``Pm``, which the caller already has on the
    host (it is the ``bincount`` of the PRISM mask, transferred once before the
    slice loop).  So the decision costs no device work and NO SYNC -- the whole
    point of taking it here rather than off ``c_masked.shape[0]`` inside the loop.

    ``forced`` (``"window"``, ``"probe"`` or ``"auto"``) overrides the module
    default ``_EELS_CONTRACT_ORDER``; ``None`` keeps that default.
    """
    if forced is None or forced == "auto":
        forced = _EELS_CONTRACT_ORDER
    if forced != "auto":
        return [forced] * len(counts)
    orders = []
    for pm in counts:
        pm = int(pm)
        cost_window = _EELS_WINDOW_COST_SCALE * (
            n_trans * ndet * nbeams * W + n_trans * pm * ndet * nbeams
        )
        cost_probe = pm * nbeams * W + n_trans * pm * W * ndet
        orders.append("probe" if cost_probe <= cost_window else "window")
    return orders


def _crop_stack_or_none(
    S: Tensor, iy: Tensor, ix: Tensor, wy: int, wx: int, lin: Tensor | None = None
) -> Tensor | None:
    """``(na, C, wy, wx)`` crop windows for one slice's ``na`` atoms at once.

    Row ``j`` is elementwise equal to ``S[:, iy[j]][:, :, ix[j]]`` -- both spellings
    are pure gathers of the same elements of ``S``, with no arithmetic -- so this is
    bit-identical to the per-atom form by construction.  What it buys is the launch
    count: one advanced index per slice instead of two per atom, and it never
    materialises the ``(C, wy, nx)`` intermediate that the chained form builds and
    then throws ``1 - wx/nx`` of away.

    Returns ``None`` when there is nothing to batch (a single atom) or when the
    stack would exceed :data:`_EELS_CROP_BLOCK_BYTES`, in which case the caller
    keeps the per-atom path.  Both are properties of the problem, not fitted
    crossovers: at one atom the batched form *is* the per-atom form plus a permute.

    The permuted view is returned WITHOUT ``.contiguous()``: the copy is legal (both
    spellings are bit-identical) but measured 0.967x in situ -- the consumers are a
    broadcast multiply and two einsums, which do not pay back a full extra pass over
    the stack.

    ``lin`` (optional, ``(na, wy*wx)`` int64) is the same window as a row-major flat
    index into ``S``'s last two axes, ``iy[:, :, None] * nx + ix[:, None, :]``.  On a
    contiguous ``S`` the gather is then ONE ``index_select`` into the same
    ``(C, na, wy, wx)`` contiguous buffer the advanced index writes -- same elements,
    same shape, same strides after the permute -- instead of the advanced index's
    three launches.
    """
    na = iy.shape[0]
    if na < 2:
        return None
    if na * S.shape[0] * wy * wx * S.element_size() > _EELS_CROP_BLOCK_BYTES:
        return None
    if lin is not None and S.dim() == 3 and S.is_contiguous():
        flat = S.view(S.shape[0], -1).index_select(1, lin.reshape(-1))
        return flat.view(S.shape[0], na, wy, wx).permute(1, 0, 2, 3)
    return S[:, iy[:, :, None], ix[:, None, :]].permute(1, 0, 2, 3)


def _hs1_stack_or_none(h_crops: Tensor, S1_win: Tensor | None) -> Tensor | None:
    """``(na, n_trans, Nbeams, wy, wx)`` -- one slice's ``h_crop[:, None] * S1c[None]``.

    **Bit-identical to the per-atom form, and -- unlike an atom-batched GEMM -- the
    two GEMMs downstream still see byte-for-byte the same problem.**  Two separate
    reasons, both structural:

    * the product itself is an elementwise complex multiply, so broadcasting
      ``(na, nt, 1, wy, wx)`` against ``(na, 1, B, wy, wx)`` forms exactly the same
      per-element products as the per-atom ``(nt, 1, wy, wx) * (1, B, wy, wx)``; and
    * the returned stack is contiguous, so ``out[j, t0:t0 + tblk]`` is a contiguous
      view with the *same shape and the same strides* as the per-atom product it
      replaces.  ``cuBLAS`` therefore selects the same kernel and reduces in the same
      order -- which is the thing a measurement found going wrong (up to 4.4e-04) when the
      atom axis was folded into the contraction instead.

    Returns ``None`` when ``S1_win`` is unavailable (partitioned ``S1``, a single
    atom, or a crop stack over :data:`_EELS_CROP_BLOCK_BYTES`) or when the product
    would exceed :data:`_EELS_HS1_BLOCK_BYTES`, in which case the caller keeps the
    per-atom multiply.  Both are properties of the problem, not fitted crossovers.

    The win is host-side: at production site counts this call is ~2/3 host-issue, and
    the per-atom spelling pays one ``aten::mul`` dispatch plus one ``cudaLaunchKernel``
    per ionized atom for a product whose total element count is unchanged.
    """
    if S1_win is None:
        return None
    na = int(h_crops.shape[0])
    if na < 2:
        return None
    nbytes = (
        na
        * int(h_crops.shape[1])
        * int(S1_win.shape[1])
        * int(h_crops.shape[-2])
        * int(h_crops.shape[-1])
        * h_crops.element_size()
    )
    if nbytes > _EELS_HS1_BLOCK_BYTES:
        return None
    return h_crops[:, :, None] * S1_win[:, None]


def _coeff_stack_or_none(
    coeffs: Tensor, sel_cat: Tensor | None, counts: Sequence[int]
) -> list | None:
    """One slice's ``coeffs.index_select(0, sel)`` per atom, in ONE gather.

    ``sel_cat`` is this slice's atoms' scan indices already concatenated in atom
    order (a ``torch.split`` view of the single ``nonzero`` the caller takes before
    the slice loop), so entry ``j`` is elementwise equal to
    ``coeffs.index_select(0, sel_j)``: ``index_select`` preserves row order, and
    splitting dim 0 of the contiguous result yields a contiguous ``(Pm, Nbeams)``
    tensor with the same shape and strides as the per-atom gather.  The GEMM that
    consumes it is therefore unchanged, exactly as for :func:`_hs1_stack_or_none`.

    Returns ``None`` when there is nothing to batch (fewer than two atoms, or no
    scan positions at all), when the caller has no concatenated index (``prism_mask``
    off, where every atom shares the *same* ``arange(P)`` and batching would
    replicate it ``na`` times), or when the stack would exceed
    :data:`_EELS_COEFF_BLOCK_BYTES`.
    """
    if sel_cat is None or len(counts) < 2:
        return None
    total = int(sel_cat.numel())
    if total == 0:
        return None
    if total * int(coeffs.shape[1]) * coeffs.element_size() > _EELS_COEFF_BLOCK_BYTES:
        return None
    return list(torch.split(coeffs.index_select(0, sel_cat), list(counts)))


def _probe_batch_size(
    orders: Sequence[str],
    counts: Sequence[int],
    sel_cat: Tensor | None,
    S1_win: Tensor | None,
    S2_win: Tensor | None,
    na: int,
    n_trans: int,
    ndet: int,
    W: int,
) -> int | None:
    """This slice's common ``Pm`` when its whole "probe"-order body can be batched.

    The per-atom body of the ``"probe"`` order is ``psi = c @ s1``,
    ``hpsi = h * psi``, ``amp = hpsi @ s2^T`` and a float64 reduction -- and every
    one of those runs at the SAME shape for every atom of the slice as soon as the
    atoms' scan counts agree.  The operands are already stacked contiguously along
    the atom axis (``s1_flat``, ``s2_flat``, ``h_flat``, and the single
    ``index_select`` behind :func:`_coeff_stack_or_none`), so the whole slice is one
    launch train instead of ``na`` of them.

    **Bit-identical to the per-atom form, because the GEMM shapes and
    strides do not change.**  Batching the atom axis leaves both contractions' lengths
    (``Nbeams`` and ``W``) and both operands' leading strides exactly as the
    per-atom GEMMs see them -- it does not fold the atom axis *into* a contraction,
    which is the operation measured to reassociate cuBLAS by 4.4e-04.  Measured
    rather than argued: ``torch.equal`` holds for both ``bmm``s, for the broadcast
    multiply and for the float64 ``sum`` at the production shapes, with the
    library's own ``float32_matmul_precision='medium'`` in force.

    Returns ``None`` -- keeping the per-atom path that shipped before -- whenever
    any of the following is a property of this slice rather than a tuning choice:

    * fewer than two atoms, or any atom taking the ``"window"`` order (which forms
      a different intermediate entirely);
    * the crop stacks or the concatenated scan index are unavailable (a partitioned
      ``S1``/``S2``, a crop stack over :data:`_EELS_CROP_BLOCK_BYTES`, or
      ``prism_mask`` off, where every atom shares the same ``arange(P)`` and the
      batched gather would replicate it ``na`` times);
    * the atoms' scan counts differ, or any of them is empty; or
    * the batched ``(na, n_trans, Pm, W)`` working set would exceed
      :data:`_EELS_TRANS_BLOCK_BYTES`, the same budget the per-atom transition
      blocking already respects.  Requiring the WHOLE transition axis to fit is
      what keeps the result bit-identical: the per-atom loop reduces and scatters
      one ``t``-block at a time, so a batched form with a different block size
      would partition the same float64 sum differently.
    """
    if na < 2 or sel_cat is None or S1_win is None or S2_win is None:
        return None
    if any(o != "probe" for o in orders):
        return None
    counts = [int(c) for c in counts]
    if len(counts) != na or len(set(counts)) != 1 or counts[0] <= 0:
        return None
    Pm = counts[0]
    per_t = Pm * W + Pm * ndet
    if na * n_trans * per_t * 16 > _EELS_TRANS_BLOCK_BYTES:
        return None
    return Pm


def _shift_crop_stack(
    Hq: Tensor,
    ky: Tensor,
    kx: Tensor,
    shifts_yx: np.ndarray,
    cy: Tensor,
    cx: Tensor,
    dtype,
) -> Tensor:
    """``physics.fourier.fourier_shift(H, s)[:, cy][:, :, cx]`` for many shifts.

    ``Hq`` is ``fft2(H)`` -- the same tensor for every atom, so it is transformed
    once by the caller instead of once per atom.

    **Bitwise-identical to the per-atom loop it replaces**, for the reasons
    ``physics.propagation.fourier_shift_stack`` documents for the same restructuring
    on the scan axis: the ramp is the same float64 elementwise arithmetic (a
    broadcast multiply against ``(na,1,1,1)`` shifts forms the same products as a
    scalar multiply per atom), the complex multiply against ``Hq`` is elementwise,
    and cuFFT's batched 2-D transform is bitwise independent per batch element.
    Verified with ``torch.equal`` on the returned image, not assumed.

    The win is not arithmetic: the per-atom form pays a float64 ramp build, an
    ``fft2(H)``, an ``ifft2`` and two gathers per ionized atom, and at production
    site counts this path is host-issue bound.
    """
    n_trans = Hq.shape[0]
    na = int(shifts_yx.shape[0])
    if na == 1:
        # Nothing to batch, so the batched form can only add overhead (a device
        # shift vector, a leading unit axis, a staging copy).  This is a structural
        # dispatch, not a fitted crossover: it fires exactly when the atom axis has
        # length one, and it keeps the fft2(H) hoist, which is where the whole
        # single-atom saving is.
        phase = torch.exp(
            -2j * pi * (ky * float(shifts_yx[0, 0]) + kx * float(shifts_yx[0, 1]))
        ).to(dtype)
        full = torch.fft.ifft2(Hq * phase, dim=(-2, -1))
        return full[:, cy][:, :, cx][None]
    sh = torch.as_tensor(shifts_yx, dtype=torch.float64, device=Hq.device)
    sy = sh[:, 0].view(na, 1, 1, 1)
    sx = sh[:, 1].view(na, 1, 1, 1)
    per_atom = n_trans * Hq.shape[-2] * Hq.shape[-1] * 8
    blk = max(1, int(_EELS_SHIFT_BLOCK_BYTES // max(per_atom, 1)))
    if blk >= na:  # one chunk: no staging buffer needed
        phase = torch.exp(-2j * pi * (ky * sy + kx * sx)).to(dtype)
        full = torch.fft.ifft2(Hq[None] * phase, dim=(-2, -1))
        return full[:, :, cy][:, :, :, cx]
    out = torch.empty(
        (na, n_trans, int(cy.numel()), int(cx.numel())), dtype=dtype, device=Hq.device
    )
    for a0 in range(0, na, blk):
        a1 = min(a0 + blk, na)
        phase = torch.exp(-2j * pi * (ky * sy[a0:a1] + kx * sx[a0:a1])).to(dtype)
        full = torch.fft.ifft2(Hq[None] * phase, dim=(-2, -1))
        out[a0:a1] = full[:, :, cy][:, :, :, cx]
    return out


def _accumulate_batched(
    acc: _OrderedScanAccumulator,
    coeffs: Tensor,
    scan_sel: Tensor,
    scan_idx: list,
    S1_win: Tensor,
    S2_win: Tensor,
    h_flat: Tensor,
    na: int,
    Pm_b: int,
    Nbeams: int,
    n_trans: int,
    ndet: int,
    W: int,
    q_bin: Tensor | None,
    n_q: int,
) -> None:
    """One slice's atoms in ONE launch train, when the batching was accepted.

    `S1_win.reshape(...).contiguous()` and `S2_win.reshape(...)` build exactly
    the `s1_flat` / `s2_flat` stacks the per-atom branch builds lazily, and
    `coeffs.index_select(0, scan_sel)` is the same single gather
    `_coeff_stack_or_none` takes -- viewed as `(na, Pm, Nbeams)` instead of
    split into `na` contiguous rows. Same operands, same shapes, same strides,
    so this stays bit-identical to the per-atom spelling.
    """
    psi = torch.bmm(
        coeffs.index_select(0, scan_sel).view(na, Pm_b, Nbeams),
        S1_win.reshape(na, Nbeams, W).contiguous(),
    )  # (na, Pm, W)
    hpsi = h_flat[:, :, None, :] * psi[:, None]  # (na, n_trans, Pm, W)
    amp = torch.bmm(
        hpsi.reshape(na, n_trans * Pm_b, W),
        S2_win.reshape(na, ndet, W).contiguous().transpose(1, 2),
    )  # (na, n_trans*Pm, ndet)
    power = amp.abs().pow(2).to(torch.float64).view(na, n_trans, Pm_b, ndet)
    contrib = (
        power.sum(dim=(1, 3))  # (na, Pm)
        if q_bin is None
        else _fold_detectors(power.sum(dim=1), q_bin, n_q)  # (na, Pm, n_q)
    )
    # The scatter stays PER ATOM IN ATOM ORDER: two atoms of one slice can hit the
    # same scan position, so one atomic scatter with duplicate keys would be a
    # different (and nondeterministic) float64 summation order.  The accumulator
    # keeps that order without a launch per atom.
    acc.add(scan_idx, contrib, scan_sel, contrib.flatten(0, 1))


def _scan_blocks(
    sel: Tensor, coeffs: Tensor, scan_chunk: int | None
) -> list[tuple[Tensor, Tensor]]:
    """Split one atom's scan keys and coefficients into ``scan_chunk`` blocks.

    The blocks bound the scan-dependent intermediates. One block (the whole
    atom) is the unbounded spelling, so a call without chunking runs exactly
    the operations it ran before ``scan_chunk`` existed.
    """
    n = int(coeffs.shape[0])
    if scan_chunk is None or n <= scan_chunk:
        return [(sel, coeffs)]
    return [
        (sel[c0 : c0 + scan_chunk], coeffs[c0 : c0 + scan_chunk])
        for c0 in range(0, n, scan_chunk)
    ]


def _accumulate_probe_order(
    acc: _OrderedScanAccumulator,
    blocks: list[tuple[Tensor, Tensor]],
    s1f: Tensor,
    s2f: Tensor,
    hj: Tensor,
    fold,
) -> None:
    """One atom's ``"probe"``-order contraction, block by block of scan positions.

    ``s1f`` ``(Nbeams, W)`` and ``s2f`` ``(ndet, W)`` are the window columns,
    ``hj`` ``(n_trans, W)`` the shifted transition potentials, and ``fold``
    reduces ``(nt, Pm, ndet)`` intensity to the image rows.
    """
    n_trans, W = hj.shape
    ndet = s2f.shape[0]
    # `.transpose(0, 1)` is a free view that cuBLAS consumes as op=T -- no
    # staging copy, which is why this order needs no staging copy.
    s2t = s2f.transpose(0, 1)  # (W, ndet)
    for sel_b, c_b in blocks:
        # (Pm, W): the PRISM probe of Eq. (7) restricted to the window, shared
        # by every transition of this atom.
        psi = c_b @ s1f
        Pm = c_b.shape[0]
        # Spelled with `@` rather than einsum: torch.einsum was measured at
        # ~13.6 us of HOST parse per call for zero extra device work on a body
        # that is dispatch-bound, and matmul lowers straight to the GEMM.
        per_t = Pm * W + Pm * ndet
        tblk = max(1, min(n_trans, _EELS_TRANS_BLOCK_BYTES // (per_t * 16)))
        for t0 in range(0, n_trans, tblk):
            hpsi = hj[t0 : t0 + tblk, None, :] * psi[None]  # (nt, Pm, W)
            amp = hpsi @ s2t  # (nt, Pm, ndet)
            # |amp|^2 summed over this block's transitions (they add
            # incoherently) and the detector beams, in fp64.
            acc.add((sel_b,), (fold(amp.abs().pow(2).to(torch.float64)),))


def _accumulate_slice(plan: _SlicePlan, i: int) -> None:
    """Add slice ``i``'s ionized atoms to the image, in place.

    One PRISM advance of ``S1``, one peel of ``S2``, then every atom of the
    slice contracted against the shifted transition potentials. Which
    contraction order each atom takes, and whether the slice-wide ``h * S1``
    product and coefficient gather are worth building, is decided on the host
    from counts transferred before the loop -- so the planning costs no device
    work and no sync.
    """
    (
        S1,
        S2,
        acc,
        scan_pixels,
        q_bin,
        n_q,
        partitioned,
        n_trans,
        wy,
        wx,
        H,
        Hq,
        ky,
        kx,
        cy,
        cx,
        all_shifts,
        slice_groups,
        iy_by_slice,
        ix_by_slice,
        lin_by_slice,
        scan_by_slice,
        scan_idx_by_slice,
        atom_counts_by_slice,
        reuse,
        order,
        scan_chunk,
        workspace,
    ) = plan
    in_slice = slice_groups[i]
    if in_slice.size == 0:
        return

    def fold(power: Tensor) -> Tensor:
        """(n_trans, Pm, ndet) -> (Pm,), or (Pm, n_q) when qEELS is on."""
        if q_bin is None:
            return power.sum(dim=(0, 2))
        return _fold_detectors(power.sum(dim=0), q_bin, n_q)

    S1.advance_to(i)
    S2.peel_to(i)
    # Two slice-loop invariants, computed once instead of once per slice.
    # `coeffs_at` reads only the beam frequencies, `coeffs0` and the fixed scan --
    # none of which `advance_to` touches -- so every slice's call returned the same
    # tensor.  `_shift_crop_stack` depends on the slice only through its atoms'
    # sub-pixel shifts, and the atoms of one lateral column repeat the same shifts
    # slice after slice, so a slice whose shift table equals the previous slice's
    # reuses its crops (the previous `h_crops` is dropped when replaced, so this
    # holds no extra memory).  Nothing downstream writes to either tensor; both
    # reuses are the same ops on the same inputs, hence bit-identical.
    if "coeffs" not in reuse:
        reuse["coeffs"] = S1.coeffs_at(scan_pixels)  # (P, Nbeams)
    coeffs = reuse["coeffs"]
    # All atoms of the slice shift the SAME transition potentials: one batched,
    # bitwise-identical transform (_shift_crop_stack), not one per atom.
    shifts = all_shifts[in_slice]
    prev = reuse.get("shifts")
    if prev is None or not np.array_equal(shifts, prev):
        reuse["h_crops"] = None
        reuse["h_crops"] = _shift_crop_stack(Hq, ky, kx, shifts, cy, cx, H.dtype)
        reuse["shifts"] = shifts
    h_crops = reuse["h_crops"]
    win = (iy_by_slice[i], ix_by_slice[i], wy, wx, lin_by_slice[i])
    iy_sl, ix_sl = win[0], win[1]  # (na, wy), (na, wx)
    # ONE flat `index_select` per matrix for all the slice's crop windows.
    S1_win = None if partitioned else _crop_stack_or_none(S1.S, *win)
    S2_win = None if S2.partitioned else _crop_stack_or_none(S2.S, *win)
    # Shapes that are fixed for the whole slice -- only `Pm` varies per atom --
    # so the contraction order of every atom is decided here, on the host, from
    # the scan counts that were already transferred before the slice loop.
    na = int(in_slice.size)
    ndet = S2.ndet
    Nbeams = int(coeffs.shape[1])
    W = wy * wx
    orders = _contract_orders(
        atom_counts_by_slice[i], n_trans, ndet, Nbeams, W, forced=order
    )
    any_window = any(o == "window" for o in orders)
    # One elementwise `h * S1` product and one coefficient gather for the WHOLE
    # slice, both leaving every GEMM's problem shape and strides untouched (see
    # the two helpers).  `None` means the per-atom spelling below is kept.
    # The `h * S1` product exists ONLY to feed the "window" order's first GEMM,
    # so a slice whose atoms all take the "probe" order never builds it -- which
    # is the memory-traffic half of the reassociation's win, not just a guard.
    hs1_all = _hs1_stack_or_none(h_crops, S1_win) if any_window else None
    # The whole slice's "probe"-order body in ONE launch train instead of one
    # per atom (bit-identical to the per-atom form).  Decided entirely on the host, from counts that were
    # already transferred before the slice loop, so it costs no device work and
    # no sync.  `None` means the per-atom loop below runs exactly as it shipped.
    Pm_b = _probe_batch_size(
        orders,
        atom_counts_by_slice[i],
        scan_by_slice[i],
        S1_win,
        S2_win,
        na,
        n_trans,
        ndet,
        W,
    )
    # A slice with an atom above the scan-chunk bound keeps the per-atom loop,
    # which splits that atom's scan positions into bounded blocks.
    if scan_chunk is not None and any(
        int(count) > scan_chunk for count in atom_counts_by_slice[i]
    ):
        Pm_b = None
    c_all = (
        None
        if Pm_b is not None
        else _coeff_stack_or_none(coeffs, scan_by_slice[i], atom_counts_by_slice[i])
    )
    # `h_crops` is contiguous in every branch of _shift_crop_stack, so merging its
    # window axes is a free view and row `j` has exactly the shape and strides of
    # the per-atom `h_crops[j].reshape(n_trans, W)` -- see the note on `s1_flat`.
    h_flat = h_crops.reshape(na, n_trans, W)
    # The two `permute(...).contiguous()` staging copies the first GEMM needs are
    # built ONCE PER SLICE instead of once per atom, lazily (many configurations
    # never reach the explicit-GEMM branch below at all).  Row `j` of each stack
    # is byte-for-byte the tensor the per-atom spelling built: permuting only the
    # NON-leading axes of a stack and making it contiguous leaves each leading-axis
    # slice contiguous with exactly the shape and strides of the per-atom copy, so
    # cuBLAS sees the same problem and the result is bit-identical.
    hs1_perm = None  # (na, wx, wy, n_trans, Nbeams), contiguous
    s2_perm = None  # (na, ndet, wx, wy), contiguous
    # The "probe" order's two GEMM operands, flattened over the window axis and
    # staged ONCE PER SLICE, lazily (a slice whose atoms all take the "window"
    # order never builds them).  They are made contiguous deliberately: the crop
    # stacks are permuted views whose row `j` has leading stride `na*W` instead of
    # `W`, so feeding them straight to the GEMM would hand cuBLAS a DIFFERENT `ld`
    # than the per-atom fallback below hands it for the same problem -- exactly the
    # "same shape, same strides, same kernel, same reduction order" invariant that
    # `_hs1_stack_or_none` documents and that was measured going wrong (4.4e-04)
    # when it was violated.  After the copy, `s1_flat[j]` / `s2_flat[j]` are
    # byte-for-byte the tensors the per-atom `reshape` produces, so the batched and
    # unbatched spellings of the "probe" order stay bit-identical to each other.
    # The copy is bounded by _EELS_CROP_BLOCK_BYTES (the stacks only exist below
    # it) and is the same one-per-slice staging the "window" order already pays for
    # `s2_perm`.
    s1_flat = None  # (na, Nbeams, W), contiguous
    s2_flat = None  # (na, ndet, W), contiguous
    if Pm_b is not None:
        # The whole slice in one launch train; the per-atom loop below is the
        # fallback for a slice the batching declined.
        _accumulate_batched(
            acc,
            coeffs,
            scan_by_slice[i],
            scan_idx_by_slice[i],
            S1_win,
            S2_win,
            h_flat,
            na,
            Pm_b,
            Nbeams,
            n_trans,
            ndet,
            W,
            q_bin,
            n_q,
        )
        return
    s1_recon, s2_recon = _slice_reconstructions(
        S1, S2, partitioned, Nbeams, atom_counts_by_slice[i], wy, wx, workspace
    )
    for j, _s in enumerate(in_slice):
        sel = scan_idx_by_slice[i][j]  # scan positions this atom contributes to
        if sel.numel() == 0:  # host-side: the mask was reduced once, above
            continue
        iy = iy_sl[j]
        ix = ix_sl[j]
        S2c = S2_win[j] if S2_win is not None else _s2_window(S2, s2_recon, iy, ix)

        c_masked = (
            c_all[j] if c_all is not None else coeffs.index_select(0, sel)
        )  # (Pm, Nbeams)
        blocks = _scan_blocks(sel, c_masked, scan_chunk)

        if orders[j] == "probe":
            # Reassociated coupling: synthesize the PRISM probe on the window,
            # apply the transition potential, propagate to the detector.  This is
            # an EXACT reassociation of the same sum the "window" branch below
            # takes (see :func:`_contract_orders`), so the two agree to fp64
            # round-off, not to a physical approximation.
            #
            # It reuses every per-slice hoist that is not specific to the other
            # order: the batched sub-pixel shift (`h_flat`), both crop stacks and
            # the single coefficient gather.  Only the `h * S1` stack is skipped,
            # because this order never forms that product at all.
            if S1_win is not None:
                if s1_flat is None:
                    s1_flat = S1_win.reshape(na, Nbeams, W).contiguous()
                s1f = s1_flat[j]  # (Nbeams, W)
            else:
                s1f = _s1_window(S1, s1_recon, partitioned, iy, ix).reshape(Nbeams, W)
            if S2_win is not None:
                if s2_flat is None:
                    s2_flat = S2_win.reshape(na, ndet, W).contiguous()
                s2f = s2_flat[j]  # (ndet, W)
            else:
                s2f = S2c.reshape(ndet, W)
            _accumulate_probe_order(acc, blocks, s1f, s2f, h_flat[j], fold)
            continue

        # Scalar-combinable S1 columns on the crop window. When partitioned, they
        # are NNW-reconstructed on the window only, or cropped from the slice's
        # one full-grid reconstruction when the windows cover the grid.
        if hs1_all is not None:
            S1c = None  # the product is already formed for the whole slice
        elif S1_win is not None:
            S1c = S1_win[j]  # (Nbeams, wy, wx)
        else:
            S1c = _s1_window(S1, s1_recon, partitioned, iy, ix)  # (Nbeams, wy, wx)

        # Batch the n_trans transitions into two large GEMMs instead of a
        # Python loop of 2*n_trans tiny ones. S2c, S1c and c_masked are all
        # transition-independent -- only h_crop varies with t -- so folding t
        # into the contraction removes ~2*(n_trans-1) kernel launches per atom
        # and enlarges the matmul enough to actually engage the TF32 tensor
        # cores (measured ~1.6-2x op-level / 2-5x end-to-end at FePt / LAO-STO
        # shapes; the batched GEMM is ~2.6x faster with TF32 on than off).
        # Kept in the working dtype -- a bf16 real-pair decomposition was
        # slower here (cast overhead at these window sizes) and less accurate.
        # The t axis is chunked so the (nt, Nbeams, wy, wx) intermediate stays
        # bounded (~2 GiB batches every FePt / LAO-STO edge in one shot).
        per_t = Nbeams * wy * wx + ndet * Nbeams + blocks[0][1].shape[0] * ndet
        tblk = max(1, min(n_trans, _EELS_TRANS_BLOCK_BYTES // (per_t * 16)))
        for t0 in range(0, n_trans, tblk):
            if hs1_all is not None:
                HS1 = hs1_all[j, t0 : t0 + tblk]  # (nt, Nbeams, wy, wx)
            else:
                h_blk = h_crops[j][t0 : t0 + tblk]  # (nt, wy, wx)
                HS1 = h_blk[:, None] * S1c[None]  # (nt, Nbeams, wy, wx)
            # `torch.einsum` costs ~13.6 us of HOST parse per call here for ZERO
            # extra device work, and this loop runs 2 x n_atoms times on a call
            # that is dispatch-bound.  Spell the two contractions as the exact
            # `reshape` + `bmm` einsum itself lowers to -- captured op-for-op,
            # shapes AND strides, so cuBLAS cannot dispatch differently and the
            # result is bit-identical.  Guarded, because einsum
            # DEGENERATES on a size-1 axis: at Nbeams == 1 the second contraction
            # becomes a plain `mul` (no GEMM at all, so no TF32) and at nt == 1 it
            # blocks its `bmm` differently.  Those shapes keep einsum.
            nt_b, nb_b = HS1.shape[0], HS1.shape[1]
            kwin = HS1.shape[2] * HS1.shape[3]
            if nt_b > 1 and nb_b > 1 and kwin > 1:
                # (1, ndet, v*w) @ (1, v*w, nt*b): einsum contracts the window in
                # (v, w) order, so both operands are permuted before contiguous().
                if S2_win is not None:
                    if s2_perm is None:
                        s2_perm = S2_win.permute(0, 1, 3, 2).contiguous()
                    lhs = s2_perm[j].view(1, ndet, kwin)
                else:
                    lhs = S2c.permute(0, 2, 1).contiguous().view(1, ndet, kwin)
                # A t-block that is not the whole transition axis is a stride-slice
                # of the permuted stack's second-innermost axis, i.e. NOT the
                # contiguous tensor the per-atom copy produces -- so the hoist only
                # fires when the block covers every transition.
                if hs1_all is not None and nt_b == n_trans:
                    if hs1_perm is None:
                        hs1_perm = hs1_all.permute(0, 4, 3, 1, 2).contiguous()
                    rhs = hs1_perm[j].view(1, kwin, nt_b * nb_b)
                else:
                    rhs = (
                        HS1.permute(3, 2, 0, 1).contiguous().view(1, kwin, nt_b * nb_b)
                    )
                Sn = (
                    torch.bmm(lhs, rhs).view(ndet, nt_b, nb_b).permute(1, 0, 2)
                )  # (nt, ndet, Nbeams)
                # (1, nt*ndet, b) @ (1, b, Pm), with einsum's OWN row order (nt, ndet)
                # -- which costs the same `reshape` copy einsum pays.  The cheaper
                # (ndet, nt) order makes the left operand a free view of Sn's
                # buffer, but `amp` then has different STRIDES, and the consumer
                # below reduces over two of its axes: measured, that reassociates
                # the float64 sum by 1 ULP and breaks the `dense` sha256 contract on
                # real data (random-valued probes cannot see it).  Same strides is
                # what makes this bit-exact, so keep them.
                lhs_sn = Sn.reshape(1, nt_b * ndet, nb_b)
                for sel_b, c_b in blocks:
                    amp = (
                        torch.bmm(lhs_sn, c_b.permute(1, 0).unsqueeze(0))
                        .view(nt_b, ndet, c_b.shape[0])
                        .permute(0, 2, 1)
                    )  # (nt, Pm, ndet)
                    # |amp|^2 summed over this block's transitions (they add
                    # incoherently) and the detector beams, in fp64.
                    acc.add((sel_b,), (fold(amp.abs().pow(2).to(torch.float64)),))
            else:
                Sn = torch.einsum("dwv,tbwv->tdb", S2c, HS1)  # (nt, ndet, Nbeams)
                for sel_b, c_b in blocks:
                    amp = torch.einsum("tdb,pb->tpd", Sn, c_b)  # (nt, Pm, ndet)
                    acc.add((sel_b,), (fold(amp.abs().pow(2).to(torch.float64)),))


#: The public ``contract`` names, mapped to the contraction orders of
#: :func:`_contract_orders` (``"columns"`` is the ``"window"`` order).
_ORDER_BY_CONTRACT = {"auto": "auto", "columns": "window", "probe": "probe"}


def _build_matrices(
    probe_q: Tensor,
    transmissions: Tensor,
    kernel: Tensor,
    det_idx: Tensor,
    factor: tuple[int, int],
    partition: dict | None,
    partition_s2: dict | None,
    *,
    mag_preserve: bool,
    focal_backprop: float | str | None,
    physics: tuple[float, float, Sequence[float]],
) -> tuple[PartitionedScatteringMatrix | ScatteringMatrix, DetectorExitSMatrix]:
    """Build the probe matrix ``S1`` and the detector matrix ``S2``.

    Fresnel referencing applies to ``S1`` through ``focal_backprop``; each
    partition dictionary may override its own matrix's setting with a
    ``"focal_backprop"`` key. The ``S2`` vacuum de-chirp is independent and on
    by default. ``physics`` is ``(wavelength, slice_distance, gridsize)`` for
    the windowed referencing.
    """
    wavelength, slice_distance, gridsize = physics
    if partition is not None:
        part = dict(partition)
        S1 = PartitionedScatteringMatrix(
            probe_q,
            transmissions,
            kernel,
            interpolation_factor=factor,
            mag_preserve=mag_preserve,
            focal_backprop=part.pop("focal_backprop", focal_backprop),
            wavelength=wavelength,
            slice_distance=slice_distance,
            gridsize=gridsize,
            **part,
        )
    else:
        S1 = ScatteringMatrix(
            probe_q, transmissions, kernel, interpolation_factor=factor
        )
    S2 = DetectorExitSMatrix(
        det_idx,
        transmissions,
        kernel,
        partition=partition_s2,
        mag_preserve=mag_preserve,
        focal_backprop=(partition_s2 or {}).get("focal_backprop"),
        wavelength=wavelength,
        slice_distance=slice_distance,
        gridsize=gridsize,
    )
    return S1, S2


def prism_eels_image(
    probe_q: Tensor,
    transmissions: Tensor,
    transition_potentials: TransitionPotentials,
    sites: np.ndarray,
    scan_pixels: Tensor,
    *,
    wavelength: float,
    gridsize: Sequence[float],
    slice_distance: float,
    detector_mrad: float,
    interpolation_factor=1,
    inelastic_crop: int | None = None,
    prism_mask: bool = True,
    scan_shape: Sequence[int] | None = None,
    partition: dict | None = None,
    partition_s2: dict | None = None,
    qeels_axis: int | Literal["both"] | None = None,
    contract: Literal["auto", "columns", "probe"] = "auto",
    mag_preserve: bool = True,
    focal_backprop: float | str | None = None,
    scan_chunk: int | None = 8192,
    reconstruction_workspace: int | None = None,
) -> Tensor | tuple[Tensor, Tensor]:
    """Double-channeling PRISM STEM-EELS image ``I(x, y)``.

    Parameters
    ----------
    probe_q : Tensor (Ny, Nx)
        Probe-forming aperture in reciprocal space (complex).
    transmissions : Tensor (NZ, Ny, Nx)
        Elastic transmission functions (``|T| = 1``; no absorption).
    transition_potentials : TransitionPotentials
        Transition potentials (``.array`` (n_trans, Ny, Nx), centred at origin).
    sites : ndarray (Nsite, 3)
        Fractional ``(y, x, z)`` of the ionized atoms.
    scan_pixels : Tensor (P, 2)
        Scan positions in pixels.
    wavelength, gridsize, slice_distance
        As elsewhere.
    detector_mrad : float
        EELS detector collection semi-angle [mrad]; output beams within it.
    interpolation_factor : int or (int, int)
        PRISM interpolation factor for ``S1`` (and the scan crop region).
    inelastic_crop : int, optional
        Crop the transition-potential / scattering-matrix window to
        ``(Ny, Nx) // inelastic_crop`` around each atom (the localized transition).
        ``None`` uses the full grid (exact; equals the detector-integrated 4D dual-S).
    prism_mask : bool
        Restrict each atom's contribution to scan positions within the PRISM
        ``1/f`` crop region around it (matches the interpolation-factor periodicity).
    scan_shape : (int, int), optional
        If given, the flat ``(P,)`` image is reshaped to ``scan_shape``.
    partition : dict, optional
        If given (e.g. ``{"n_radial": 4}``), build ``S1`` on ``Bp`` parent beams
        (:class:`PartitionedScatteringMatrix`) and reconstruct the exact aperture
        columns on each atom's crop window via natural-neighbor interpolation. This
        cheapens the ``S1`` build/advance + memory (``Bp`` parents) at the cost of
        the NNW reconstruction error. A ``"focal_backprop"`` key overrides the
        ``focal_backprop`` argument for ``S1``. ``None`` uses the exact per-pixel
        ``S1``. When one slice's windows cover the grid, both partitioned legs
        reuse one full-grid reconstruction per slice, within the additional
        workspace that ``reconstruction_workspace`` sets; sparse windows and
        focal referencing stay windowed.
    partition_s2 : dict, optional
        Same idea for ``S2`` (the detector matrix): build the transpose multislice
        on ``Bp`` parent detector beams and NNW-reconstruct the detector columns on
        the crop window, shrinking the resident ``S2`` from ``ndet`` to ``Bp``
        columns at a second NNW error. ``"dechirp"`` (default True) removes and
        restores the scalar vacuum beam phase around the interpolation (see
        :class:`DetectorExitSMatrix`); ``"focal_backprop"`` turns on the Fresnel
        referencing of ``S2`` (off unless given here). ``None`` keeps the full
        per-beam ``S2``.
    qeels_axis : {0, 1, "both"}, optional
        Momentum-resolved (qEELS) output. ``None`` (default) returns the scalar
        elemental map. ``0`` or ``1`` *resolves* that detector-momentum axis and
        projects the perpendicular one away: each beam's ``|amp_d|^2`` is binned
        by its signed frequency along ``qeels_axis`` rather than summed over all
        ``d``. ``"both"`` keeps every detector beam in its own bin (no
        projection). Summing the spectrum over its momentum axis returns the map
        to float64 round-off -- the same additions, grouped differently, so the
        agreement is ~1 ULP (measured 2e-16 relative) rather than bitwise. Works
        with ``partition`` / ``partition_s2`` (the binning is post-``|amp|^2``);
        for a faithful resolved axis use ``interpolation_factor == 1`` along it.
    contract : {"auto", "columns", "probe"}, default "auto"
        How the per-atom contraction is associated (same result to rounding,
        different cost). ``"columns"`` forms the scan-independent
        ``(n_det, B)`` coupling ``S2c (H S1c)`` once per atom and applies the
        scan coefficients last; it is best when ``P`` is very large or the crop
        window is the full grid. ``"probe"`` synthesises the probe on the crop
        window first (``probe = c S1c``) and contracts it against the detector
        columns; it avoids the ``O(n_det B |window|)`` intermediate and is
        faster for small crop windows and modest ``P``. ``"auto"`` picks the
        cheaper order per atom from the shapes (see ``_contract_orders``).
    mag_preserve : bool, default True
        Magnitude-preserving NNW reconstruction of the partitioned ``S1`` and
        ``S2`` columns: keep the complex sum's phase but restore the
        interpolated magnitude. This counters the phase-decoherence amplitude
        loss of the plain complex average and converges to the exact result at
        full parents. ``False`` is the plain complex average. Ignored without
        ``partition`` / ``partition_s2``.
    focal_backprop : float, "centroid" or None, optional
        Fresnel referencing of the partitioned ``S1`` parents: interpolate them
        at a plane back along the traversed depth, then propagate the
        reconstructed window forward again. ``"centroid"`` takes half the
        traversed depth and a number takes that fraction, rounded to whole
        slices. ``None`` (default) or ``0`` disables it. ``S2`` uses
        referencing only when ``partition_s2`` names it.
    scan_chunk : int or None, default 8192
        Most scan positions one atom contracts at once. It bounds the
        scan-dependent intermediate memory, so the scan can be sampled
        arbitrarily finely. The result does not depend on the value (to
        rounding); ``None`` processes all positions of an atom in one block.
    reconstruction_workspace : int or None, default None
        Additional device memory (bytes) that the partitioned legs may use to
        reuse one full-grid NNW reconstruction per slice. ``None`` uses the
        default of 2 GiB. ``0`` turns the reuse off: every atom reconstructs
        its own crop window, which is the behaviour of the BiP-PRISM paper and
        of v0.3, with the smallest peak memory. The result does not depend on
        the value (to rounding). Ignored without ``partition`` /
        ``partition_s2``.

    Returns
    -------
    Tensor or (Tensor, Tensor)
        With ``qeels_axis=None``: the STEM-EELS image, ``(P,)`` or ``scan_shape``.
        Otherwise ``(spectrum, q_pixels)``, where ``spectrum`` carries a trailing
        momentum axis (``(P, n_q)`` or ``scan_shape + (n_q,)``) and ``q_pixels``
        is each bin's sorted signed pixel frequency (physical ``q = q_pixels /
        gridsize[qeels_axis]``). For ``"both"``, ``q_pixels`` is ``(n_q, 2)`` in
        ``(qy, qx)`` order, one signed pair per detector beam; divide each column
        by its grid size to get inverse Angstrom.

    Raises
    ------
    ValueError
        If ``contract`` or ``qeels_axis`` is not one of the listed values, or
        if ``reconstruction_workspace`` is negative.
    """
    order = _ORDER_BY_CONTRACT.get(contract)
    if order is None:
        raise ValueError(f"contract must be 'auto', 'columns' or 'probe': {contract!r}")
    reconstruction_workspace = _workspace_bytes(reconstruction_workspace)
    scan_chunk = None if scan_chunk is None else max(1, int(scan_chunk))
    device = transmissions.device
    nz, ny, nx = transmissions.shape
    H = transition_potentials.array.to(device=device)  # (n_trans, Ny, Nx)
    n_trans = H.shape[0]
    kernel = propagator_kernel(
        (ny, nx), gridsize, wavelength, slice_distance, device=device, dtype=H.dtype
    )
    fy, fx = _factor_pair(interpolation_factor)

    sites = np.atleast_2d(np.asarray(sites, dtype=np.float64))
    site_slice = np.asarray(BeamPartition.uniform(nz).bin_index(sites[:, 2]))
    P = scan_pixels.shape[0]

    det_idx = _detector_beams(
        (ny, nx), gridsize, wavelength, detector_mrad, (fy, fx), device
    )

    img, q_bin, n_q, q_pix = _qeels_bins(det_idx, qeels_axis, (ny, nx), P, device)
    acc = _OrderedScanAccumulator(img)

    partitioned = partition is not None
    S1, S2 = _build_matrices(
        probe_q.to(transmissions.dtype),
        transmissions,
        kernel,
        det_idx,
        (fy, fx),
        partition,
        partition_s2,
        mag_preserve=mag_preserve,
        focal_backprop=focal_backprop,
        physics=(wavelength, slice_distance, gridsize),
    )

    wy, wx = _crop_window(inelastic_crop, ny, nx)

    ry, rx = scan_pixels[:, :2].to(torch.float64).unbind(1)

    # The crop window around the shifted origin, the fft of the transition
    # potentials and the shift-ramp frequencies are all atom-independent, so they
    # are built once instead of once per ionized atom.
    cy = _periodic_window(ny // 2, wy, ny, device)
    cx = _periodic_window(nx // 2, wx, nx, device)
    Hq = torch.fft.fft2(H, dim=(-2, -1))
    ky = torch.fft.fftfreq(ny, device=device, dtype=torch.float64).view(ny, 1)
    kx = torch.fft.fftfreq(nx, device=device, dtype=torch.float64).view(1, nx)
    # In `_SlicePlan`'s field order (all_shifts ... atom_counts_by_slice).
    tables = _slice_tables(
        sites,
        site_slice,
        nz,
        (ny, nx),
        (wy, wx),
        (ry, rx),
        (fy, fx),
        P,
        prism_mask,
        device,
    )

    plan = _SlicePlan._make(
        (
            S1,
            S2,
            acc,
            scan_pixels,
            q_bin,
            n_q,
            partitioned,
            n_trans,
            wy,
            wx,
            H,
            Hq,
            ky,
            kx,
            cy,
            cx,
            *tables,
            {},
            order,
            scan_chunk,
            reconstruction_workspace,
        )
    )
    for i in range(nz):
        _accumulate_slice(plan, i)

    img = acc.flush()
    if scan_shape is not None:
        shape = (int(scan_shape[0]), int(scan_shape[1]))
        img = img.reshape(*shape, n_q) if q_bin is not None else img.reshape(*shape)
    return (img, q_pix) if q_bin is not None else img
