"""Aberration-independent segment geometry of tilt-corrected dark/bright field.

The detector masks, the azimuthal wedge membership and the per-segment images
gathered from the 4D array: everything
:func:`scatterem.reconstruction.darkfield.core.tilt_corrected_dark_field`
builds before the aberrations enter, and which the depth section therefore
builds once for all planes.
"""

import numpy as np
import torch

from scatterem.data.datasets import Dataset4DStem, _normalized_mean_pattern
from scatterem.utils.lazy_plotting import vis

from ._scaling import _DEAD_FRAME_FRACTION, _dark_field_scaling


class _SegmentGeometry:
    """Everything in ``tilt_corrected_dark_field`` that does NOT depend on the
    aberrations: the detector masks, the azimuthal segment membership, and the
    per-segment images gathered from the 4D array.

    The aberrations enter the reconstruction at exactly one place — ``shifts``
    (:func:`aberrations_to_image_shifts`) — and from there only the per-segment
    mean shift and the splat.  A caller that sweeps the aberrations (the depth
    section, which walks ``aberrations.array[0]``) therefore recomputes this
    object identically on every step, including the full pass over the
    ``(ny, nx, M, M)`` array in ``gather_detector_group_means``.

    The identifying parameters are carried so a reused instance FAILS CLOSED
    rather than silently reconstructing from the wrong segments.

    ``seg_sizes`` holds the number of source pixels of each kept segment, in
    segment order (even entries: half-set 1, odd: half-set 2). Each segment
    image is the mean over these pixels.
    """

    __slots__ = (
        "dataset",
        "bright_field_mask_threshold",
        "n_dark_field_segments",
        "upsample",
        "upsample_int",
        "vdf_stack1",
        "vdf_stack2",
        "specific_radius_masks1",
        "specific_radius_masks2",
        "seg_px1",
        "seg_px2",
        "seg_sizes",
        "region",
        "dark_field_scaling",
        "dead_frame_fraction",
    )

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def _check(
        self,
        dataset,
        bright_field_mask_threshold,
        n_dark_field_segments,
        upsample,
        region,
        dark_field_scaling,
        dead_frame_fraction,
    ) -> None:
        if (
            dataset is not self.dataset
            or bright_field_mask_threshold != self.bright_field_mask_threshold
            or n_dark_field_segments != self.n_dark_field_segments
            or upsample != self.upsample
            or region != self.region
            or dark_field_scaling != self.dark_field_scaling
            or dead_frame_fraction != self.dead_frame_fraction
        ):
            raise ValueError(
                "precomputed tcDF segment geometry does not match this call "
                "(dataset / bright_field_mask_threshold / n_dark_field_segments / "
                "upsample / region / dark_field_scaling / dead_frame_fraction must "
                "all be the ones it was built with)"
            )


def _tcdf_segment_geometry(
    dataset: Dataset4DStem,
    n_dark_field_segments: int = 32,
    verbosity: int = 0,
    bright_field_mask_threshold: float = 0.3,
    upsample: float | str = "nyquist",
    dark_field_scaling: bool = False,
    dead_frame_fraction: float = _DEAD_FRAME_FRACTION,
    region: str = "dark_field",
) -> "_SegmentGeometry":
    """Build the aberration-independent half of :func:`tilt_corrected_dark_field`.

    Lifted VERBATIM out of that function — same statements, same order — so a
    caller that reuses the result runs the identical arithmetic on the identical
    inputs.  See ``_SegmentGeometry`` for why a sweep over the aberrations
    wants it.

    ``region`` selects which detector pixels feed the azimuthal segments.
    ``"dark_field"`` (default) uses everything outside the bright-field disk.
    ``"bright_field"`` uses the disk itself, which makes this tilt-corrected
    BRIGHT field -- the segment/shift/drizzle/half-split-SSNR machinery is not
    specific to the dark field. tcBF is dark where the specimen scatters, i.e.
    the OPPOSITE contrast sense to tcDF; ``fused_full_field`` flips it when it
    uses it as the phase channel. The bright-field disk holds far fewer pixels
    than the dark field, so use a smaller ``n_dark_field_segments`` (empty
    wedges are dropped rather than allowed to poison the image with NaN).

    ``dark_field_scaling`` accounts for the electrons scattered clean past the
    detector edge. A pixelated detector only records the dark field that lands
    on it, so on a thick, strongly scattering specimen the recorded dark field
    under-reports the scattering cross-section exactly where the specimen is
    thickest. Taking the incident probe current to be the largest total count
    recorded anywhere in the scan (a position that scatters nothing off the
    detector, i.e. vacuum), everything outside the bright-field disk must be
    dark field whether or not it was recorded, so each segment image is scaled
    per scan position by

        (max_p' I_total(p') - I_BF(p)) / I_DF_recorded(p)   >= 1.

    OFF by default: the assumption fails if the scan contains no such vacuum
    position, or if the detector already subtends the full scattering
    distribution (then the factor is ~1 and only adds noise).

    ``dead_frame_fraction`` guards the factor against dropped detector frames.
    A scan position recording less than this fraction of the incident current
    measured no probe -- its factor degenerates towards ``incident/0`` and its
    recorded angular distribution is meaningless -- so it is left uncorrected.
    The default only catches all-but-blank frames; raise it (e.g. 0.05) for a
    detector that reads its flyback row back *partially*, which clears the
    default guard while still being garbage. It must stay well below the
    smallest genuine ``recorded/incident`` in the scan (a thick specimen can
    legitimately lose over half its electrons past the detector edge).
    """
    from scatterem.reconstruction._sampling import _resolve_upsample_int

    if region not in ("dark_field", "bright_field"):
        raise ValueError(
            f"Invalid region: {region!r} (expected 'dark_field' or 'bright_field')"
        )
    if dark_field_scaling and region == "bright_field":
        raise ValueError(
            "dark_field_scaling is defined from the recorded dark field "
            "(it redistributes the intensity that missed the detector over the "
            "dark-field pixels) and is meaningless for region='bright_field'"
        )

    # tcDF uses ONE isotropic factor (max over the per-axis factors).
    upsample_int = np.max(_resolve_upsample_int(dataset, upsample, verbosity=verbosity))
    device = dataset.device
    wavelength = dataset.meta.wavelength
    semiangle_cutoff = dataset.meta.semiconvergence_angle
    # Polymorphic: the eager dataset computes this from its resident array,
    # the out-of-core dataset serves its streamed Pass-0 statistic.
    diff_mean = dataset.mean_cbed_tcdf.to(device)
    diff_mean = _normalized_mean_pattern(diff_mean)
    bright_field_mask = diff_mean > bright_field_mask_threshold
    if verbosity > 0:
        fig, ax = vis.show_2d(
            bright_field_mask.float(), cbar=True, title="Virtual Bright Field Mask"
        )
    bright_field_inds = torch.argwhere(bright_field_mask)
    bright_field_center = bright_field_inds.float().mean(0)
    dark_field_mask = ~bright_field_mask

    # Create circular dark field mask centered on bright_field_center
    y_coords, x_coords = torch.meshgrid(
        torch.arange(dark_field_mask.shape[0], device=dark_field_mask.device),
        torch.arange(dark_field_mask.shape[1], device=dark_field_mask.device),
        indexing="ij",
    )

    # Calculate radial distances from bright_field_center
    r = torch.sqrt(
        (y_coords - bright_field_center[0]) ** 2
        + (x_coords - bright_field_center[1]) ** 2
    )

    # Set dark field mask to include everything up to the edge of the array
    max_radius = min(
        dark_field_mask.shape[0] - bright_field_center[0],
        bright_field_center[0],
        dark_field_mask.shape[1] - bright_field_center[1],
        bright_field_center[1],
    )
    # Update dark_field_mask to be circular and centered on bright_field_center
    dark_field_mask = (r <= max_radius) & ~bright_field_mask
    # The pixels the segments actually draw from. Same inscribed-circle
    # restriction either way, so both regions stay azimuthally complete.
    source_mask = (
        ((r <= max_radius) & bright_field_mask)
        if region == "bright_field"
        else dark_field_mask
    )
    # Create n_dark_field_segments azimuthal masks
    center = torch.tensor(
        [dark_field_mask.shape[0] // 2, dark_field_mask.shape[1] // 2],
        device=dark_field_mask.device,
    )

    angles = torch.atan2(y_coords - center[0], x_coords - center[1])
    angles = (angles + torch.pi) / (2 * torch.pi)  # Normalize to [0,1]
    segment_size = 1.0 / n_dark_field_segments

    inner_radius = semiangle_cutoff / wavelength * 2 / 3
    outer_radius = inner_radius + dataset.sampling[-1] * 1.3
    ny, nx = dataset.shape[-2:]
    sy, sx = dataset.dr
    k_x = torch.fft.fftfreq(nx, sx, device=device)
    k_y = torch.fft.fftfreq(ny, sy, device=device)

    # Build (ny, nx) grid to match detector masks (y, x) layout.
    k = torch.sqrt(k_y[:, None] ** 2 + k_x[None, :] ** 2)
    radial_mask = torch.fft.fftshift((inner_radius <= k) & (k < outer_radius))

    # ---- segment geometry (data-independent given diff_mean + aberrations) --
    # BATCHED over the wedge axis.  The per-wedge form this replaces built one
    # (M, M) mask at a time and called ``argwhere`` inside the loop, with a
    # ``nonzero`` per wedge again below in ``_member_px``: 2*n+1 ops whose output
    # SHAPE is data-dependent, hence 2*n+1 device syncs, plus 32 iterations of
    # python -- on a path measured at 99.92 % HOST-bound, where the device is not
    # on the critical path at all (r14_ffstem/carbon/tcdf, 14.1 ms).
    #
    # This is the same arithmetic, not an approximation of it: the comparisons
    # are the SAME elementwise comparisons against the SAME wedge bounds cast to
    # ``angles.dtype``, and ``nonzero`` returns rows in ascending flat order, so
    # the rows belonging to wedge ``i`` are contiguous and in exactly the order
    # ``argwhere`` on that one wedge produced them.  Graded by equality, not
    # tolerance -- every mask and every index tensor is bit-identical.
    _seg_lo = torch.tensor(
        [i * segment_size for i in range(n_dark_field_segments)],
        dtype=angles.dtype,
        device=angles.device,
    ).view(-1, 1, 1)
    _seg_hi = torch.tensor(
        [(i + 1) * segment_size for i in range(n_dark_field_segments)],
        dtype=angles.dtype,
        device=angles.device,
    ).view(-1, 1, 1)
    _angles_b = angles.unsqueeze(0)
    # Normal case: segment doesn't cross the 0/2pi boundary
    segment_masks = (_angles_b >= _seg_lo) & (_angles_b < _seg_hi)
    for i in range(n_dark_field_segments):
        segment_end = (i + 1) * segment_size
        if segment_end < 1.0:
            continue
        # The last segment: segment crosses 0/2pi boundary
        # This includes angles from segment_start to 1.0 AND from 0.0 to (segment_end - 1.0)
        # Run the original expression for just these wedges (normally exactly
        # one), so the boundary case stays character for character.
        segment_start = i * segment_size
        high_angles = (angles >= segment_start) & (angles <= 1.0)
        low_angles = angles < (segment_end - 1.0)
        segment_masks[i] = high_angles | low_angles

    _source_seg_masks = segment_masks & source_mask
    _radius_seg_masks = segment_masks & radial_mask
    # ONE host round trip for both count vectors, in place of one per wedge.
    _counts = torch.stack(
        (
            _source_seg_masks.flatten(1).sum(1),
            _radius_seg_masks.flatten(1).sum(1),
        )
    ).tolist()

    def _split_leading(nz, counts):
        """Split an ascending ``nonzero`` result on its leading (wedge) axis,
        dropping that axis, so each chunk is what ``argwhere``/``nonzero`` on
        that single wedge would have returned -- contiguous, same dtype."""
        return [t.contiguous() for t in torch.split(nz[:, 1:], counts, dim=0)]

    # An empty wedge gathers nothing, and the mean of an empty gather is NaN --
    # which would propagate through the shift operator into the whole image.
    # Reachable whenever the source region is small relative to
    # n_dark_field_segments (the bright-field disk is only tens of pixels), so
    # drop the wedge instead of carrying the NaN. The counts are already on the
    # host, so this costs no extra sync.
    _keep = [i for i, c in enumerate(_counts[0]) if c > 0]
    if len(_keep) < 2:
        raise ValueError(
            f"region={region!r} left {len(_keep)} non-empty segment(s) of "
            f"{n_dark_field_segments}; the SSNR half-split needs at least 2. "
            f"Lower n_dark_field_segments or raise bright_field_mask_threshold."
        )

    _seg_inds_all = _split_leading(
        torch.nonzero(_source_seg_masks, as_tuple=False), _counts[0]
    )
    seg_inds_list = [_seg_inds_all[i] for i in _keep]
    specific_radius_masks_stack = _radius_seg_masks[_keep]

    # ---- per-segment images: per-scan-position mean over member px ----------
    # Polymorphic: eager gathers from the resident array; out-of-core runs ONE
    # streamed pass over the memmap fanned out to all segments. The two extra
    # off-detector-scaling groups ride along in the SAME pass -- gathering them
    # separately would re-read the whole memmap.
    scaling_groups = (
        [torch.argwhere(bright_field_mask), torch.argwhere(~bright_field_mask)]
        if dark_field_scaling
        else []
    )
    gathered = dataset.gather_detector_group_means(seg_inds_list + scaling_groups)
    segment_image_list = gathered[: len(seg_inds_list)]

    if dark_field_scaling:
        # Means -> sums over their (disjoint, together exhaustive) pixel sets.
        bf_group, df_group = scaling_groups
        bf_sum = gathered[len(seg_inds_list)] * bf_group.shape[0]
        df_sum = gathered[len(seg_inds_list) + 1] * df_group.shape[0]
        scaling = _dark_field_scaling(bf_sum, df_sum, dead_frame_fraction, verbosity)
        segment_image_list = [img * scaling for img in segment_image_list]

    # Even/odd segments form the two SSNR half-sets. Keep the REAL-space
    # segment images; the fourier path FFTs them below, the drizzle path
    # splats them directly (a Fourier phase-ramp shift of a single-count
    # image rings, a drizzle splat does not).
    dark_field_segment_images1 = segment_image_list[0::2]
    dark_field_segment_images2 = segment_image_list[1::2]

    vdf_stack1 = torch.stack(dark_field_segment_images1)  # (S1, H, W) real
    vdf_stack2 = torch.stack(dark_field_segment_images2)
    specific_radius_masks1 = specific_radius_masks_stack[0::2].contiguous()
    specific_radius_masks2 = specific_radius_masks_stack[1::2].contiguous()

    # Flat member-pixel indices of each wedge, so the per-segment mean shift can
    # be taken with an ``index_select`` instead of re-running boolean-mask
    # indexing (a ``nonzero``, hence a host sync) on every call.  ``nonzero``
    # returns ascending flat indices, which is the same order a bool mask
    # produces, so ``shifts.reshape(-1, 2).index_select(0, idx)`` is the same
    # tensor -- same values, same shape, same strides -- as ``shifts[mask]``.
    # ``None`` marks a wedge with no detector pixels (see ``_seg_mean``); the
    # emptiness is decided HERE, once, rather than by a ``bool(mask.any())``
    # sync per wedge per call.
    #
    # Taken from the batched wedge stack with ONE ``nonzero`` rather than one per
    # wedge, using the counts already brought back above -- same ascending flat
    # indices, same dtype, same shape.
    _flat_nz = torch.nonzero(
        _radius_seg_masks.reshape(n_dark_field_segments, -1), as_tuple=False
    )
    _member_px_kept = _split_leading(_flat_nz, _counts[1])
    _member_px_all = [
        _member_px_kept[i].squeeze(1) if _counts[1][i] else None for i in _keep
    ]

    return _SegmentGeometry(
        dataset=dataset,
        bright_field_mask_threshold=bright_field_mask_threshold,
        n_dark_field_segments=n_dark_field_segments,
        upsample=upsample,
        upsample_int=upsample_int,
        vdf_stack1=vdf_stack1,
        vdf_stack2=vdf_stack2,
        specific_radius_masks1=specific_radius_masks1,
        specific_radius_masks2=specific_radius_masks2,
        seg_px1=_member_px_all[0::2],
        seg_px2=_member_px_all[1::2],
        seg_sizes=[_counts[0][i] for i in _keep],
        region=region,
        dark_field_scaling=dark_field_scaling,
        dead_frame_fraction=dead_frame_fraction,
    )
