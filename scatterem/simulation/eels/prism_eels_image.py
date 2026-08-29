"""Linear-scaling double-channeling PRISM STEM-EELS *image* (Brown & Ophus).

This is the detector-integrated counterpart of
:func:`scatterem.simulation.eels.prism_eels.prism_transition_potential`. It
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
* For each ionized atom the small matrix ``SHn0[d, b] = sum_r S2_d(r) H_n0(r) S1_b(r)``
  is formed over a crop window around the atom (the transition potential is
  localized), then applied to every scan illumination vector:
  ``I(R) += sum_d |sum_b SHn0[d, b] psi_illum[b, R]|^2`` over the PRISM crop region.

Because ``S2`` is built per slice (not per atom) and the per-atom work is a small
cropped matmul, the cost is independent of the number of probe positions and
scales benignly with the number of ionized atoms -- unlike the 4D dual-S, which
re-propagates per atom. The transpose-multislice ``S2`` is validated to reproduce
``fft(forward_exit(w))`` exactly, so with no crop this equals the
detector-integrated 4D dual-S.
"""

from __future__ import annotations

from math import pi
from typing import Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from .multislice_eels import _propagate, propagator_kernel
from .prism_eels import (
    PartitionedScatteringMatrix,
    ScatteringMatrix,
    _factor_pair,
    _partition_weights,
)
from .transition_potentials import TransitionPotentials

__all__ = ["DetectorExitSMatrix", "prism_eels_image"]


def _conj_propagate(waves: Tensor, kernel: Tensor) -> Tensor:
    """Inverse Fresnel step ``ifft(fft(.) * conj(kernel))`` (kernel is even)."""
    return torch.fft.ifft2(
        torch.fft.fft2(waves, dim=(-2, -1)) * kernel.conj(), dim=(-2, -1)
    )


class DetectorExitSMatrix:
    """S2: columns are detector-output beams back-propagated to the current slice.

    ``S2_d(r)`` is the adjoint (transpose) multislice of the conjugate detector
    plane wave ``e_d*`` from the exit surface back to the current slice, so that
    for any wave ``w`` at that slice ``sum_r S2_d(r) w(r) == fft(forward_exit(w))[d]``
    -- the amplitude reaching detector beam ``d``. Built once covering all slices
    (slice 0 -> exit) and :meth:`peel_to` advances the start slice forward (removing
    front slices) so it is reused across every atom.

    With ``partition`` the transpose multislice is built on only ``Bp`` *parent*
    detector beams (a hex-ring subsample); the full detector columns are
    NNW-reconstructed on a crop window via :meth:`columns_window`. By reciprocity
    the de-tilted back-propagated columns are locally similar, so the same
    partitioned-PRISM interpolation applies -- shrinking the resident S2 from
    ``ndet`` to ``Bp`` columns.
    """

    def __init__(
        self,
        det_idx: Tensor,
        transmissions: Tensor,
        kernel: Tensor,
        partition: Optional[dict] = None,
    ):
        device = transmissions.device
        dtype = transmissions.dtype
        self.transmissions = transmissions
        self.kernel = kernel
        nz, ny, nx = transmissions.shape
        self.ny, self.nx = ny, nx
        self.det_idx = det_idx  # (ndet, 2) corner-origin pixel indices
        self.ndet = det_idx.shape[0]
        self._dtype = dtype
        self._device = device

        dgy = ((det_idx[:, 0] + ny // 2) % ny - ny // 2).to(torch.float64)
        dgx = ((det_idx[:, 1] + nx // 2) % nx - nx // 2).to(torch.float64)
        self._det_signed = torch.stack([dgy, dgx], dim=1)  # (ndet, 2)

        self.partitioned = partition is not None
        if self.partitioned:
            # Parent detector beams + NNW weights (geometry on CPU, like S1).
            all_signed = self._det_signed.cpu().numpy().astype(np.float64)
            pidx, w = _partition_weights(
                det_idx.cpu().numpy(),
                all_signed,
                int(partition.get("n_radial", 4)),
                int(partition.get("n_angular", 6)),
            )
            self._w = torch.as_tensor(w, dtype=dtype, device=device)  # (ndet, Bp)
            build_signed = self._det_signed[torch.as_tensor(np.asarray(pidx))]
        else:
            build_signed = self._det_signed
        self._build_signed = build_signed.to(device)

        # Conjugate plane waves e_d*(r) = exp(-2pi i g_d . r / N) for the build beams.
        ry = torch.arange(ny, device=device, dtype=torch.float64).view(1, ny, 1)
        rx = torch.arange(nx, device=device, dtype=torch.float64).view(1, 1, nx)
        gy = self._build_signed[:, 0]
        gx = self._build_signed[:, 1]
        v = torch.exp(
            -2j * pi * (gy[:, None, None] * ry / ny + gx[:, None, None] * rx / nx)
        ).to(dtype)

        # Transpose multislice from exit back to slice 0 (adjoint of forward_exit:
        # transmit at every slice, propagate between, in reverse slice order).
        u = v
        for j in range(nz - 1, -1, -1):
            u = u * transmissions[j]
            if j > 0:
                u = _propagate(u, kernel)
        self.S = u  # (n_build, ny, nx): ndet columns, or Bp parents if partitioned
        self._start = 0

    def peel_to(self, start: int) -> None:
        """Advance the start slice to ``start`` (remove front slices, reused S2)."""
        for j in range(self._start, start):
            self.S = _conj_propagate(self.S * self.transmissions[j].conj(), self.kernel)
        self._start = start

    def columns_window(self, iy: Tensor, ix: Tensor) -> Tensor:
        """Detector columns on a crop window ``(ndet, wy, wx)``.

        Direct crop when full; NNW reconstruction from the ``Bp`` parents (de-tilt,
        combine, re-tilt -- all on the window, never the full grid) when partitioned.
        """
        if not self.partitioned:
            return self.S[:, iy][:, :, ix]
        ny, nx = self.ny, self.nx
        Sw = self.S[:, iy][:, :, ix]  # (Bp, wy, wx) parent columns on the window
        yy = iy.to(torch.float64).view(1, -1, 1)
        xx = ix.to(torch.float64).view(1, 1, -1)
        gp = self._build_signed
        detilt = torch.exp(
            -2j * pi * (gp[:, 0, None, None] * yy / ny + gp[:, 1, None, None] * xx / nx)
        ).to(self._dtype)
        recon = torch.einsum("dp,pwv->dwv", self._w, Sw * detilt)  # (ndet, wy, wx)
        gd = self._det_signed
        tilt = torch.exp(
            2j * pi * (gd[:, 0, None, None] * yy / ny + gd[:, 1, None, None] * xx / nx)
        ).to(self._dtype)
        return recon * tilt


def _window_indices(center: int, width: int, n: int, device) -> Tensor:
    """``width`` indices centred on ``center`` with periodic wraparound."""
    half = width // 2
    rng = torch.arange(-half, width - half, device=device)
    return (center + rng) % n


def _window_indices_stack(centers: np.ndarray, width: int, n: int, device) -> Tensor:
    """:func:`_window_indices` for many centres at once -- ``(len(centers), width)``.

    Same integer arithmetic as the scalar form (``(center + rng) % n`` in int64), so
    row ``j`` is elementwise equal to ``_window_indices(int(centers[j]), ...)``; the
    point is that the whole call's windows cost one launch instead of two per atom.
    """
    half = width // 2
    rng = torch.arange(-half, width - half, device=device)
    c = torch.as_tensor(np.asarray(centers, dtype=np.int64), device=device).view(-1, 1)
    return (c + rng) % n


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
    counts: Sequence[int], n_trans: int, ndet: int, nbeams: int, W: int
) -> list:
    """One slice's per-atom contraction order, decided entirely on the HOST.

    The per-atom coupling is a three-way contraction over the crop window, and it
    can be associated two ways.  Both compute the same sum; they differ only in
    which pair is contracted first:

    * ``"window"`` -- contract the window for every ``(t, d, b)`` triple to get
      ``SHn0[t,d,b] = sum_wv S2c[d,wv] h[t,wv] S1c[b,wv]``, then project onto the
      scan, ``amp[t,p,d] = sum_b SHn0[t,d,b] c[p,b]``.  ``SHn0`` is
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
    remaining points, and on the independent ``ndet = 137`` sweep recorded in
    ``QUEUE.md`` R11 (probe at ``Pm = 144``, window at ``Pm = 272`` and ``1024``).
    It is a **calibration**, not a derivation: it was fitted to the 128 px sweep and
    then checked against three further ``(n_trans, ndet, nbeams, W)`` geometries it
    was not fitted to.  Getting it wrong costs time and nothing else -- both orders
    are mathematically identical, so a mis-dispatch is a missed win, never a wrong
    answer, which is why a measured constant is an acceptable form of fix here.

    ``counts`` is the slice's per-atom ``Pm``, which the caller already has on the
    host (it is the ``bincount`` of the PRISM mask, transferred once before the
    slice loop).  So the decision costs no device work and NO SYNC -- the whole
    point of taking it here rather than off ``c_masked.shape[0]`` inside the loop.
    """
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
    S: Tensor, iy: Tensor, ix: Tensor, wy: int, wx: int
) -> Optional[Tensor]:
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
    """
    na = iy.shape[0]
    if na < 2:
        return None
    if na * S.shape[0] * wy * wx * S.element_size() > _EELS_CROP_BLOCK_BYTES:
        return None
    return S[:, iy[:, :, None], ix[:, None, :]].permute(1, 0, 2, 3)


def _hs1_stack_or_none(h_crops: Tensor, S1_win: Optional[Tensor]) -> Optional[Tensor]:
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
      order -- which is the thing D72 measured going wrong (up to 4.4e-04) when the
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
    coeffs: Tensor, sel_cat: Optional[Tensor], counts: Sequence[int]
) -> Optional[list]:
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
    sel_cat: Optional[Tensor],
    S1_win: Optional[Tensor],
    S2_win: Optional[Tensor],
    na: int,
    n_trans: int,
    ndet: int,
    W: int,
) -> Optional[int]:
    """This slice's common ``Pm`` when its whole "probe"-order body can be batched.

    The per-atom body of the ``"probe"`` order is ``psi = c @ s1``,
    ``hpsi = h * psi``, ``amp = hpsi @ s2^T`` and a float64 reduction -- and every
    one of those runs at the SAME shape for every atom of the slice as soon as the
    atoms' scan counts agree.  The operands are already stacked contiguously along
    the atom axis (``s1_flat``, ``s2_flat``, ``h_flat``, and the single
    ``index_select`` behind :func:`_coeff_stack_or_none`), so the whole slice is one
    launch train instead of ``na`` of them.

    **Bit-identical to the per-atom form, and this is D73's law rather than D72's
    exception.**  Batching the atom axis leaves both contractions' lengths
    (``Nbeams`` and ``W``) and both operands' leading strides exactly as the
    per-atom GEMMs see them -- it does not fold the atom axis *into* a contraction,
    which is the operation D72 measured reassociating cuBLAS by 4.4e-04.  Measured
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
    """``_fourier_shift(H, s)[:, cy][:, :, cx]`` for many shifts at once.

    ``Hq`` is ``fft2(H)`` -- the same tensor for every atom, so it is transformed
    once by the caller instead of once per atom.

    **Bitwise-identical to the per-atom loop it replaces**, for the reasons
    ``multislice_eels._fourier_shift_stack`` documents for the same restructuring
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
    inelastic_crop: Optional[int] = None,
    prism_mask: bool = True,
    scan_shape: Optional[Sequence[int]] = None,
    partition: Optional[dict] = None,
    partition_s2: Optional[dict] = None,
) -> Tensor:
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
        the NNW reconstruction error. ``None`` uses the exact per-pixel ``S1``.
    partition_s2 : dict, optional
        Same idea for ``S2`` (the detector matrix): build the transpose multislice
        on ``Bp`` parent detector beams and NNW-reconstruct the detector columns on
        the crop window. Reduces the (often dominant) resident ``S2`` storage from
        ``ndet`` to ``Bp`` columns; stacks a second NNW error on the detector side.
        ``None`` keeps the full per-beam ``S2``.

    Returns
    -------
    Tensor
        STEM-EELS image, ``(P,)`` or ``scan_shape``.
    """
    device = transmissions.device
    nz, ny, nx = transmissions.shape
    H = transition_potentials.array.to(device=device)  # (n_trans, Ny, Nx)
    n_trans = H.shape[0]
    kernel = propagator_kernel(
        (ny, nx), gridsize, wavelength, slice_distance, device=device, dtype=H.dtype
    )
    fy, fx = _factor_pair(interpolation_factor)

    sites = np.atleast_2d(np.asarray(sites, dtype=np.float64))
    site_slice = np.clip((sites[:, 2] % 1.0 * nz).astype(int), 0, nz - 1)
    P = scan_pixels.shape[0]
    img = torch.zeros(P, dtype=torch.float64, device=device)

    # Detector output beams: grid pixels within the collection semi-angle, PRISM-
    # subsampled by the interpolation factor (S2 is a PRISM scattering matrix too,
    # so its output beams are decimated like S1 -- without this S2 would have
    # ~f**2 too many columns and be intractable). f = 1 keeps every beam.
    qy = torch.fft.fftfreq(ny, d=gridsize[0] / ny, device=device)
    qx = torch.fft.fftfreq(nx, d=gridsize[1] / nx, device=device)
    q2 = qy[:, None] ** 2 + qx[None, :] ** 2
    beta_q = detector_mrad / 1000.0 / wavelength
    iy_grid = torch.arange(ny, device=device)[:, None].expand(ny, nx)
    ix_grid = torch.arange(nx, device=device)[None, :].expand(ny, nx)
    det_mask = (q2 <= beta_q**2) & (iy_grid % fy == 0) & (ix_grid % fx == 0)
    det_idx = torch.nonzero(det_mask, as_tuple=False)  # (ndet, 2)

    partitioned = partition is not None
    if partitioned:
        S1 = PartitionedScatteringMatrix(
            probe_q.to(transmissions.dtype),
            transmissions,
            kernel,
            interpolation_factor=(fy, fx),
            **partition,
        )
    else:
        S1 = ScatteringMatrix(
            probe_q.to(transmissions.dtype),
            transmissions,
            kernel,
            interpolation_factor=(fy, fx),
        )
    S2 = DetectorExitSMatrix(det_idx, transmissions, kernel, partition=partition_s2)

    wy = ny if inelastic_crop is None else max(1, ny // int(inelastic_crop))
    wx = nx if inelastic_crop is None else max(1, nx // int(inelastic_crop))

    ry = scan_pixels[:, 0].to(torch.float64)
    rx = scan_pixels[:, 1].to(torch.float64)

    # The crop window around the shifted origin, the fft of the transition
    # potentials and the shift-ramp frequencies are all atom-independent, so they
    # are built once instead of once per ionized atom.
    cy = _window_indices(ny // 2, wy, ny, device)
    cx = _window_indices(nx // 2, wx, nx, device)
    Hq = torch.fft.fft2(H, dim=(-2, -1))
    ky = torch.fft.fftfreq(ny, device=device, dtype=torch.float64).view(ny, 1)
    kx = torch.fft.fftfreq(nx, device=device, dtype=torch.float64).view(1, nx)
    # Every atom's sub-pixel shift, tabulated once (same float64 arithmetic as the
    # per-atom `ay - round(ay) + ny // 2`; numpy and python both round half to even).
    site_y = sites[:, 0] % 1.0 * ny
    site_x = sites[:, 1] % 1.0 * nx
    all_shifts = np.stack(
        [site_y - np.round(site_y) + ny // 2, site_x - np.round(site_x) + nx // 2],
        axis=1,
    )
    # Every atom's crop window, tabulated once (same int64 `(center + rng) % n`) --
    # but in SLICE-GROUPED order, so each slice's block of window rows is a
    # `torch.split` view of the same single host transfer.  Slicing a site-ordered
    # device stack with the numpy `in_slice` instead would be a *synchronous
    # pageable H2D of the index* twice per slice: measured at ~227 us of host stall
    # each, which is more than the batched gather below saves on a call whose host
    # is the critical path.
    slice_groups = [np.nonzero(site_slice == i)[0] for i in range(nz)]
    group_counts = [int(g.size) for g in slice_groups]
    order = (
        np.concatenate(slice_groups)
        if any(group_counts)
        else np.zeros(0, dtype=np.int64)
    )
    iy_by_slice = list(
        torch.split(
            _window_indices_stack(np.round(site_y)[order], wy, ny, device),
            group_counts,
        )
    )
    ix_by_slice = list(
        torch.split(
            _window_indices_stack(np.round(site_x)[order], wx, nx, device),
            group_counts,
        )
    )

    # The PRISM crop mask is a function of the SITE and the scan grid only -- both
    # known before the slice loop -- so the whole (Nsite, P) mask is built in one
    # broadcast pass and converted to per-atom scan-index lists in ONE host
    # transfer.  The per-atom form paid ~14 elementwise launches, a
    # `bool(mask.any())` DEVICE SYNC and two boolean-mask `nonzero`s (one behind
    # `coeffs[mask]`, one behind `img[mask] +=`) per ionized atom, on a call that is
    # ~2/3 host-issue.  The arithmetic is unchanged: broadcasting a (Nsite, 1)
    # float64 site coordinate against the (P,) scan coordinate forms the same
    # float64 products as the scalar-per-atom spelling, `nonzero` returns indices in
    # ascending order so `index_select` gathers exactly what the boolean mask did,
    # and each atom's indices are distinct so `index_add_` performs the same single
    # float64 addition per scan position as the masked read-add-write.
    #
    # The mask rows are built in the same SLICE-GROUPED order as the window tables
    # above, so each slice's atoms' indices are one contiguous block of the single
    # `nonzero` -- a `torch.split` view, costing nothing -- and the whole slice's
    # coefficient gather can be taken in one `index_select` (see
    # `_coeff_stack_or_none`).  Reordering the rows does not change any atom's
    # indices: `nonzero` is row-major and ascending within a row either way.
    n_site = int(sites.shape[0])
    offs = np.concatenate([[0], np.cumsum(group_counts)]).astype(np.int64)
    scan_by_slice: list = [None] * nz
    scan_idx_by_slice: list = [None] * nz
    atom_counts_by_slice: list = [None] * nz
    if prism_mask:
        sy_t = torch.as_tensor(sites[order, 0], dtype=torch.float64, device=device)
        sx_t = torch.as_tensor(sites[order, 1], dtype=torch.float64, device=device)
        mask_all = (
            torch.abs((ry / ny - sy_t.view(-1, 1) + 0.5) % 1.0 - 0.5) <= 0.5 / fy
        ) & (torch.abs((rx / nx - sx_t.view(-1, 1) + 0.5) % 1.0 - 0.5) <= 0.5 / fx)
        nzi = torch.nonzero(mask_all, as_tuple=False)  # (M, 2), row-major ascending
        counts = torch.bincount(nzi[:, 0], minlength=n_site).tolist()  # one transfer
        flat_scan = nzi[:, 1].contiguous()
        per_slice = [int(sum(counts[offs[i] : offs[i + 1]])) for i in range(nz)]
        blocks = list(torch.split(flat_scan, per_slice))
        for i in range(nz):
            if group_counts[i] == 0:
                continue
            atom_counts_by_slice[i] = counts[offs[i] : offs[i + 1]]
            scan_by_slice[i] = blocks[i]
            scan_idx_by_slice[i] = list(torch.split(blocks[i], atom_counts_by_slice[i]))
    else:
        every = torch.arange(P, device=device)
        for i in range(nz):
            if group_counts[i] == 0:
                continue
            atom_counts_by_slice[i] = [P] * group_counts[i]
            scan_idx_by_slice[i] = [every] * group_counts[i]

    for i in range(nz):
        in_slice = slice_groups[i]
        if in_slice.size == 0:
            continue
        S1.advance_to(i)
        S2.peel_to(i)
        coeffs = S1.coeffs_at(scan_pixels)  # (P, Nbeams)
        # Every atom in this slice shifts the SAME transition potentials, so the
        # sub-pixel shifts are done in one batched transform (bitwise-identical --
        # see _shift_crop_stack) rather than one launch train per atom.
        h_crops = _shift_crop_stack(
            Hq, ky, kx, all_shifts[in_slice], cy, cx, H.dtype
        )
        iy_sl = iy_by_slice[i]  # (na, wy)
        ix_sl = ix_by_slice[i]  # (na, wx)
        # Every crop window of this slice in ONE advanced index per scattering
        # matrix, instead of two chained ones per atom (see _crop_stack).
        S1_win = (
            None
            if partitioned
            else _crop_stack_or_none(S1.S, iy_sl, ix_sl, wy, wx)
        )
        S2_win = (
            None
            if S2.partitioned
            else _crop_stack_or_none(S2.S, iy_sl, ix_sl, wy, wx)
        )
        # Shapes that are fixed for the whole slice -- only `Pm` varies per atom --
        # so the contraction order of every atom is decided here, on the host, from
        # the scan counts that were already transferred before the slice loop.
        na = int(in_slice.size)
        ndet = S2.ndet
        Nbeams = int(coeffs.shape[1])
        W = wy * wx
        orders = _contract_orders(atom_counts_by_slice[i], n_trans, ndet, Nbeams, W)
        any_window = any(o == "window" for o in orders)
        # One elementwise `h * S1` product and one coefficient gather for the WHOLE
        # slice, both leaving every GEMM's problem shape and strides untouched (see
        # the two helpers).  `None` means the per-atom spelling below is kept.
        # The `h * S1` product exists ONLY to feed the "window" order's first GEMM,
        # so a slice whose atoms all take the "probe" order never builds it -- which
        # is the memory-traffic half of the reassociation's win, not just a guard.
        hs1_all = _hs1_stack_or_none(h_crops, S1_win) if any_window else None
        # The whole slice's "probe"-order body in ONE launch train instead of one
        # per atom (see :func:`_probe_batched_or_none` for what it computes and why
        # it is bit-identical).  Decided entirely on the host, from counts that were
        # already transferred before the slice loop, so it costs no device work and
        # no sync.  `None` means the per-atom loop below runs exactly as it shipped.
        Pm_b = _probe_batch_size(
            orders, atom_counts_by_slice[i], scan_by_slice[i], S1_win, S2_win,
            na, n_trans, ndet, W,
        )
        c_all = (
            None
            if Pm_b is not None
            else _coeff_stack_or_none(
                coeffs, scan_by_slice[i], atom_counts_by_slice[i]
            )
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
        # cuBLAS sees the same problem and D73's law applies unchanged.
        hs1_perm = None  # (na, wx, wy, n_trans, Nbeams), contiguous
        s2_perm = None  # (na, ndet, wx, wy), contiguous
        # The "probe" order's two GEMM operands, flattened over the window axis and
        # staged ONCE PER SLICE, lazily (a slice whose atoms all take the "window"
        # order never builds them).  They are made contiguous deliberately: the crop
        # stacks are permuted views whose row `j` has leading stride `na*W` instead of
        # `W`, so feeding them straight to the GEMM would hand cuBLAS a DIFFERENT `ld`
        # than the per-atom fallback below hands it for the same problem -- exactly the
        # "same shape, same strides, same kernel, same reduction order" invariant that
        # `_hs1_stack_or_none` documents and that D72 measured going wrong (4.4e-04)
        # when it was violated.  After the copy, `s1_flat[j]` / `s2_flat[j]` are
        # byte-for-byte the tensors the per-atom `reshape` produces, so the batched and
        # unbatched spellings of the "probe" order stay bit-identical to each other.
        # The copy is bounded by _EELS_CROP_BLOCK_BYTES (the stacks only exist below
        # it) and is the same one-per-slice staging the "window" order already pays for
        # `s2_perm`.
        s1_flat = None  # (na, Nbeams, W), contiguous
        s2_flat = None  # (na, ndet, W), contiguous
        if Pm_b is not None:
            # `S1_win.reshape(...).contiguous()` and `S2_win.reshape(...)` build
            # exactly the `s1_flat` / `s2_flat` stacks the per-atom branch below
            # builds lazily, and `coeffs.index_select(0, scan_by_slice[i])` is the
            # same single gather `_coeff_stack_or_none` takes -- viewed as
            # `(na, Pm, Nbeams)` instead of split into `na` contiguous rows.
            psi = torch.bmm(
                coeffs.index_select(0, scan_by_slice[i]).view(na, Pm_b, Nbeams),
                S1_win.reshape(na, Nbeams, W).contiguous(),
            )  # (na, Pm, W)
            hpsi = h_flat[:, :, None, :] * psi[:, None]  # (na, n_trans, Pm, W)
            amp = torch.bmm(
                hpsi.reshape(na, n_trans * Pm_b, W),
                S2_win.reshape(na, ndet, W).contiguous().transpose(1, 2),
            )  # (na, n_trans*Pm, ndet)
            contrib = (
                amp.abs()
                .pow(2)
                .to(torch.float64)
                .view(na, n_trans, Pm_b, ndet)
                .sum(dim=(1, 3))
            )  # (na, Pm)
            # The scatter stays PER ATOM: two atoms of one slice can contribute to
            # the same scan position, so concatenating their indices would turn `na`
            # scatters with distinct keys into one atomic scatter with duplicate
            # keys -- a different (and nondeterministic) float64 summation order.
            for j in range(na):
                img.index_add_(0, scan_idx_by_slice[i][j], contrib[j])
            continue
        for j, s in enumerate(in_slice):
            sel = scan_idx_by_slice[i][j]  # scan positions this atom contributes to
            if sel.numel() == 0:  # host-side: the mask was reduced once, above
                continue
            iy = iy_sl[j]
            ix = ix_sl[j]
            S2c = (
                S2_win[j] if S2_win is not None else S2.columns_window(iy, ix)
            )  # (ndet, wy, wx)

            c_masked = (
                c_all[j] if c_all is not None else coeffs.index_select(0, sel)
            )  # (Pm, Nbeams)
            Pm = c_masked.shape[0]

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
                elif partitioned:
                    s1f = S1.reconstruct_columns_window(iy, ix).reshape(Nbeams, W)
                else:
                    s1f = S1.S[:, iy][:, :, ix].reshape(Nbeams, W)
                if S2_win is not None:
                    if s2_flat is None:
                        s2_flat = S2_win.reshape(na, ndet, W).contiguous()
                    s2f = s2_flat[j]  # (ndet, W)
                else:
                    s2f = S2c.reshape(ndet, W)
                # (Pm, W): the PRISM probe of Eq. (7) restricted to the window, shared
                # by every transition of this atom.
                psi = c_masked @ s1f
                # `.transpose(0, 1)` is a free view that cuBLAS consumes as op=T -- no
                # staging copy, which is why this order does not need D80's treatment.
                s2t = s2f.transpose(0, 1)  # (W, ndet)
                hj = h_flat[j]  # (n_trans, W)
                # Spelled with `@` rather than einsum: D79 measured torch.einsum at
                # ~13.6 us of HOST parse per call for zero extra device work on a body
                # that is dispatch-bound, and matmul lowers straight to the GEMM.
                per_t = Pm * W + Pm * ndet
                tblk = max(1, min(n_trans, _EELS_TRANS_BLOCK_BYTES // (per_t * 16)))
                for t0 in range(0, n_trans, tblk):
                    hpsi = hj[t0 : t0 + tblk, None, :] * psi[None]  # (nt, Pm, W)
                    amp = hpsi @ s2t  # (nt, Pm, ndet)
                    # |amp|^2 summed over this block's transitions (they add
                    # incoherently) and the detector beams, in fp64.
                    contrib = amp.abs().pow(2).to(torch.float64).sum(dim=(0, 2))
                    img.index_add_(0, sel, contrib)
                continue

            # Scalar-combinable S1 columns on the crop window. When partitioned,
            # reconstruct ONLY the window from the Bp parents (no full-grid matrix)
            # so the Bp-parent memory saving is realised and compute is window-sized.
            if hs1_all is not None:
                S1c = None  # the product is already formed for the whole slice
            elif S1_win is not None:
                S1c = S1_win[j]  # (Nbeams, wy, wx)
            elif partitioned:
                S1c = S1.reconstruct_columns_window(iy, ix)  # (Nbeams, wy, wx)
            else:
                S1c = S1.S[:, iy][:, :, ix]  # (Nbeams, wy, wx)

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
            per_t = Nbeams * wy * wx + ndet * Nbeams + Pm * ndet
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
                # result is bit-identical (D73's law).  Guarded, because einsum
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
                            HS1.permute(3, 2, 0, 1)
                            .contiguous()
                            .view(1, kwin, nt_b * nb_b)
                        )
                    SHn0 = (
                        torch.bmm(lhs, rhs)
                        .view(ndet, nt_b, nb_b)
                        .permute(1, 0, 2)
                    )  # (nt, ndet, Nbeams)
                    # (1, nt*ndet, b) @ (1, b, Pm), with einsum's OWN row order (nt, ndet)
                    # -- which costs the same `reshape` copy einsum pays.  The cheaper
                    # (ndet, nt) order makes the left operand a free view of SHn0's
                    # buffer, but `amp` then has different STRIDES, and the consumer
                    # below reduces over two of its axes: measured, that reassociates
                    # the float64 sum by 1 ULP and breaks the `dense` sha256 contract on
                    # real data (random-valued probes cannot see it).  Same strides is
                    # what makes this bit-exact, so keep them.
                    amp = (
                        torch.bmm(
                            SHn0.reshape(1, nt_b * ndet, nb_b),
                            c_masked.permute(1, 0).unsqueeze(0),
                        )
                        .view(nt_b, ndet, Pm)
                        .permute(0, 2, 1)
                    )  # (nt, Pm, ndet)
                else:
                    SHn0 = torch.einsum("dwv,tbwv->tdb", S2c, HS1)  # (nt, ndet, Nbeams)
                    amp = torch.einsum("tdb,pb->tpd", SHn0, c_masked)  # (nt, Pm, ndet)
                # |amp|^2 summed over this block's transitions (they add
                # incoherently) and the detector beams, in fp64, as the per-t
                # loop did.
                contrib = amp.abs().pow(2).to(torch.float64).sum(dim=(0, 2))  # (Pm,)
                img.index_add_(0, sel, contrib)

    if scan_shape is not None:
        img = img.reshape(int(scan_shape[0]), int(scan_shape[1]))
    return img
