"""Sub-pixel shift and accumulation of the tcDF segment images.

Each azimuthal segment image is shifted by its mean parallax shift and
accumulated onto the up-sampled grid, per SSNR half-set, either by area-overlap
drizzle (:func:`_drizzle_streamed`) or by Fourier zero-pad upsampling plus a
phase-ramp shift (:func:`scatterem.physics.fourier.fourier_rescale` and
:func:`scatterem.physics.fourier.fourier_shift`). See ``shift_method`` in
:func:`scatterem.reconstruction.darkfield.core.tilt_corrected_dark_field`.
"""

import math

import numpy as np
import torch

from scatterem.physics.fourier import fourier_rescale, fourier_shift
from scatterem.reconstruction.darkfield.drizzle import drizzle_resample

#: Byte budget for one :func:`drizzle_resample` call's window temporaries, used
#: by :func:`_drizzle_streamed` to size its frame chunk.  The dominant
#: per-chunk allocations are the ``(P, chunk*H*W)`` weight/index/mask trio
#: (float32 + int64 + bool = 13 B per entry, taken as 16 with slack), where
#: ``P <= (2*radius+1)**2`` is the offset-pair count.
_DRIZZLE_WINDOW_BUDGET_BYTES = 512 * 1024 * 1024
_DRIZZLE_WINDOW_BYTES_PER_ENTRY = 16


def _drizzle_frame_chunk(stack, upsample, pixfrac, floor):
    """Largest frame chunk whose drizzle windows fit the byte budget, never
    below ``floor``.

    ``frame_chunk`` was a fixed 8 -- a memory guard that does not read the
    thing it guards.  The windows cost ``P * chunk * H * W`` entries with
    ``P <= (2*radius+1)**2``, and ``radius`` is a function of ``pixfrac`` and
    ``upsample`` alone (the same expression :func:`drizzle_resample` uses), so
    the bound is computable here.  On an up-sampled grid the fixed constant is
    the right order; on the small, ``upsample=1`` scan grids the tcDF figure
    path actually runs it splits a 16-frame half-set into two calls whose
    windows are ~30 MB, doubling the host dispatch of a ~70 %-GPU-idle path to
    save nothing measurable (the row peak is set by the resident 4D array).

    Clamped BELOW by ``floor`` (the shipped 8) on purpose: the budget may only
    ever make a chunk LARGER, so no configuration that runs today can become
    more finely chunked -- and hence slower -- than it is now.
    """
    n = int(stack.shape[0])
    radius = int(math.ceil(max(pixfrac * upsample / 2.0, 1e-6) + 0.5)) + 1
    per_frame = (
        (2 * radius + 1) ** 2
        * int(stack.shape[1])
        * int(stack.shape[2])
        * _DRIZZLE_WINDOW_BYTES_PER_ENTRY
    )
    fits = _DRIZZLE_WINDOW_BUDGET_BYTES // max(per_frame, 1)
    return max(1, min(n, max(int(fits), int(floor))))


def _drizzle_streamed(
    stack, shifts_hr, upsample, *, pixfrac=1.0, kde_sigma=0.0, frame_chunk=8
):
    """Drizzle a stack in frame groups: accumulate ``(accum, hits)`` per group
    and divide ONCE. Exactly equal to the single stack-level call (the splat is
    a plain sum over frames; the kde blur is linear), but the drizzle
    internals' ~8x (N,H,W) window temporaries shrink to (frame_chunk,H,W).

    ``frame_chunk`` is now a FLOOR rather than the chunk itself -- see
    :func:`_drizzle_frame_chunk`."""
    frame_chunk = _drizzle_frame_chunk(stack, int(upsample), pixfrac, frame_chunk)
    accum_total = None
    hits_total = None
    for f in range(0, stack.shape[0], frame_chunk):
        _, accum_f, hits_f = drizzle_resample(
            stack[f : f + frame_chunk],
            shifts_hr[f : f + frame_chunk],
            upsample,
            pixfrac=pixfrac,
            kde_sigma=kde_sigma,
            return_parts=True,
        )
        accum_total = accum_f if accum_total is None else accum_total + accum_f
        hits_total = hits_f if hits_total is None else hits_total + hits_f
    return accum_total / hits_total.clamp_min(1e-12)


def _segment_mean_shifts(shifts_flat, seg_px, device):
    """Per-wedge mean parallax shift ``(dy, dx)`` of one half-set.

    ``shifts_flat`` is the ``(ny * nx, 2)`` per-detector-pixel image shift and
    ``seg_px`` the flat member-pixel indices of each wedge (``None`` for a wedge
    with no detector pixels, which gets shift 0). Returns two ``(n_wedges,)``
    tensors on ``device``.
    """
    # ONE segmented reduction, not three launches per wedge.  The wedges are
    # a partition of their member pixels, so the per-wedge mean shift is an
    # ``index_add_`` over the concatenated member list divided by the member
    # counts -- ~7 launches instead of 3*n_dark_field_segments (96 at the
    # shipped default of 32, and this runs twice, once per half-set).  The
    # loop it replaces was pure host dispatch: the wedges hold 2-255
    # detector pixels each, so every one of those launches moved a few
    # hundred bytes.
    #
    # Everything the HOST reads here comes from tensor METADATA (``numel``,
    # ``len``), never from a device VALUE, so nothing in this function
    # syncs -- the same discipline the gather in
    # ``gather_detector_group_means`` needs, and for the same reason (this
    # path is ~100 % host-bound at small detector sizes, where one sync
    # costs more than the launches saved).
    n = len(seg_px)
    lens = [0 if idx is None else int(idx.numel()) for idx in seg_px]
    present = [i for i, m in enumerate(lens) if m > 0]
    acc = torch.zeros(
        (n, shifts_flat.shape[-1]), dtype=shifts_flat.dtype, device=device
    )
    if present:
        gid = torch.from_numpy(
            np.repeat(
                np.asarray(present, dtype=np.int64),
                np.asarray([lens[i] for i in present], dtype=np.int64),
            )
        ).to(device=device, non_blocking=True)
        acc.index_add_(
            0,
            gid,
            shifts_flat.index_select(0, torch.cat([seg_px[i] for i in present])),
        )
    # An empty wedge accumulated nothing and is divided by 1, giving 0 --
    # exactly the "no shift" the loop's ``zero`` gave it.
    counts = torch.tensor(
        [m if m else 1 for m in lens], dtype=shifts_flat.dtype, device=device
    )
    acc = acc / counts.unsqueeze(-1)
    return acc[:, 0], acc[:, 1]


def _shifts_hr(dataset, upsample_int, dy, dx):
    """Per-segment parallax shift ``(S, 2)`` in up-sampled pixels.

    ``dy`` and ``dx`` are the per-segment shifts in Angstrom. Both shift
    methods use this conversion.
    """
    # shift_HRpx = shift[Å] * upsample_int / scan_sampling[Å/px]. The shift is
    # already grad(chi) / 2 pi; a further 1 / (2 pi) here (as before 2026-10)
    # registered every segment with 1 / (2 pi) of its parallax.
    scan_sampling = np.asarray(dataset.sampling[:2], dtype=np.float64)
    cy = float(upsample_int) / scan_sampling[0]
    cx = float(upsample_int) / scan_sampling[1]
    return torch.stack((dy * cy, dx * cx), -1)


def _shift_accumulate_halfsets(
    dataset,
    upsample_int,
    vdf_stack1,
    vdf_stack2,
    df_shifts_dy1,
    df_shifts_dx1,
    df_shifts_dy2,
    df_shifts_dx2,
    *,
    shift_method,
    drizzle_pixfrac,
    drizzle_kde_sigma,
):
    """Shift both half-set segment stacks and accumulate each onto the grid.

    ``vdf_stack1`` / ``vdf_stack2`` are the ``(S, H, W)`` real-space segment
    images of the two half-sets and ``df_shifts_*`` their per-segment mean
    parallax shifts in Angstrom: means of
    ``scatterem.utils.transfer.aberrations_to_image_shifts``, which is
    already the physical image shift ``grad chi / 2 pi`` (``lambda C10 K`` for
    defocus). Each segment image is moved by its shift divided by the scan
    step, i.e. ``shift * upsample_int / scan_sampling`` output pixels. Returns
    the two half-set images on the ``upsample_int``-times up-sampled scan grid,
    by ``shift_method`` (``"drizzle"`` or ``"fourier"``; anything else raises
    ``ValueError``).
    """
    if shift_method not in ("drizzle", "fourier"):
        raise ValueError(
            f"Invalid shift_method: {shift_method!r} (expected 'fourier' or 'drizzle')"
        )
    shifts_hr1 = _shifts_hr(dataset, upsample_int, df_shifts_dy1, df_shifts_dx1)
    shifts_hr2 = _shifts_hr(dataset, upsample_int, df_shifts_dy2, df_shifts_dx2)
    if shift_method == "drizzle":
        # Two accumulators, one per half-set, so the SSNR half-split survives.
        tcDF1 = _drizzle_streamed(
            vdf_stack1,
            shifts_hr1,
            int(upsample_int),
            pixfrac=drizzle_pixfrac,
            kde_sigma=drizzle_kde_sigma,
        )
        tcDF2 = _drizzle_streamed(
            vdf_stack2,
            shifts_hr2,
            int(upsample_int),
            pixfrac=drizzle_pixfrac,
            kde_sigma=drizzle_kde_sigma,
        )
        return tcDF1, tcDF2
    # Value-preserving sinc upsample (mean kept, so the absolute scale matches the
    # drizzle path's hit-normalised one), then the same up-sampled-pixel shift.
    hr = (1.0 / upsample_int, 1.0 / upsample_int)
    up1, _ = fourier_rescale(vdf_stack1, (1.0, 1.0), hr)
    up2, _ = fourier_rescale(vdf_stack2, (1.0, 1.0), hr)
    tcDF1 = fourier_shift(up1, shifts_hr1).mean(0)
    tcDF2 = fourier_shift(up2, shifts_hr2).mean(0)
    return tcDF1, tcDF2
