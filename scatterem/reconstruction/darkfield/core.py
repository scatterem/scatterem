"""Tilt-corrected (virtual) dark-field and bright-field reconstruction.

Segments the detector into azimuthal wedges, estimates each wedge's parallax
shift from the aberrations, and accumulates the shifted segment images onto
an upsampled grid to recover a tilt-corrected dark-field (or, with
``region="bright_field"``, bright-field) image, optionally with a half-split
SSNR estimate.
"""

import torch
from tqdm import tqdm

from scatterem.data.datasets import Dataset4DStem
from scatterem.reconstruction._sampling import _probe_aberrations
from scatterem.reconstruction._ssnr import (
    compute_ssnr_from_halfset_images,
    gaussian_blur1d,
)
from scatterem.utils.lazy_plotting import plt
from scatterem.utils.transfer import (
    aberrations_to_image_shifts,
    detector_to_shift_model_frame,
)

from ._geometry import _SegmentGeometry, _tcdf_segment_geometry
from ._scaling import _DEAD_FRAME_FRACTION
from ._shift import _segment_mean_shifts, _shift_accumulate_halfsets


def _halfset_segment_shifts(
    dataset: Dataset4DStem, geometry: _SegmentGeometry
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-segment mean parallax shifts ``(dy1, dx1, dy2, dx2)`` of both half-sets.

    The shifts are in Angstrom, on the device of ``dataset``. These are the
    shifts that :func:`tilt_corrected_dark_field` applies.
    """
    device = dataset.device
    wavelength = dataset.meta.wavelength

    # The ONLY place the aberrations enter tcDF, so the ``_segment_geometry`` hook
    # is exact. They are the probe's (C10 less ``defocus_guess``), on the device of
    # meta.aberrations.array; aligned to the dataset device for the index_select.
    aberrations_array = _probe_aberrations(dataset).detach().clone().to(device)
    aberrations_array[3:] = 0  # C10 and A1 only
    # Detector frame (as the bright-field fit publishes it) -> the shift model's
    # argument-rotated frame; the same object at rotation 0 (bit for bit).
    rotation = torch.tensor([dataset.meta.rotation], device=device)
    aberrations_array = detector_to_shift_model_frame(aberrations_array, rotation)
    shifts = aberrations_to_image_shifts(
        aberrations_array=aberrations_array,
        rotation=rotation,
        sampling=dataset.dr,
        wavelength=wavelength,
        shape=(dataset.shape[-2], dataset.shape[-1]),
    )

    # Per-segment mean shift. A segment whose annular wedge contains NO detector
    # pixels (thin dark-field annulus at low convergence angle split into many
    # segments) gives an empty slice -> .mean() is NaN, which propagates through
    # exp(-i grad.q) into an all-NaN tcDF. Use 0 (no shift) for empty segments.
    #
    # The wedges are aberration-independent, so their member pixels and their
    # emptiness are decided once in ``_SegmentGeometry``. What is left here is a
    # gather + two means per wedge, and it stays on the device: the old form paid
    # a ``bool(mask.any())`` sync AND a boolean-mask ``nonzero`` sync per wedge
    # per column, then handed 16 device scalars back through the host in
    # ``torch.tensor([...])`` -- 4x32 host round-trips for 4x16 numbers, on a
    # path whose wedges hold 2-4 detector pixels each.
    shifts_flat = shifts.reshape(-1, shifts.shape[-1])

    df_shifts_dy1, df_shifts_dx1 = _segment_mean_shifts(
        shifts_flat, geometry.seg_px1, device
    )
    df_shifts_dy2, df_shifts_dx2 = _segment_mean_shifts(
        shifts_flat, geometry.seg_px2, device
    )
    return df_shifts_dy1, df_shifts_dx1, df_shifts_dy2, df_shifts_dx2


def tilt_corrected_dark_field(
    dataset: Dataset4DStem,
    n_dark_field_segments: int = 32,
    verbosity: int = 0,
    bright_field_mask_threshold: float = 0.3,
    upsample: float | str = "nyquist",
    return_snr: bool = False,
    snr_blur_sigma: float = 0.0,
    shift_method: str = "drizzle",
    drizzle_pixfrac: float = 1.0,
    drizzle_kde_sigma: float = 0.0,
    dark_field_scaling: bool = False,
    dead_frame_fraction: float = _DEAD_FRAME_FRACTION,
    region: str = "dark_field",
    _segment_geometry: "_SegmentGeometry | None" = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Tilt-corrected dark-field reconstruction.

    The probe has the aberrations ``meta.aberrations`` with C10 less
    ``meta.defocus_guess``, as in
    :func:`~scatterem.reconstruction.ptychography.direct.direct_ptychography`.

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

    ``shift_method`` selects how each azimuthal dark-field segment is sub-pixel
    shifted (by its parallax shift) and accumulated onto the up-sampled grid:

    * ``"drizzle"`` (default) — area-overlap drizzle (box splat + hit-map
      normalization). Non-negative and ringing-free, which is the physically
      correct behaviour for counting data: at low dose the segment images are
      single electron counts and Fourier shifting turns each into a sinc with
      ~20% negative side-lobes. ``drizzle_pixfrac`` sets the drop size (fraction
      of an input pixel; 1.0 fills the grid from one frame) and
      ``drizzle_kde_sigma`` optionally Nadaraya-Watson-smooths sparse holes.
    * ``"fourier"`` — Fourier zero-pad upsample + phase-ramp shift. Marginally
      sharper on dense, well-sampled (high-dose) data, but rings on sparse data.

    Each segment image is moved by its full parallax: the
    ``aberrations_to_image_shifts`` shift (Angstrom, from C10 and A1 only)
    divided by the scan step. tcBF registers at the true defocus. Dark-field
    tcDF focuses at about 0.8-1.1 times the true defocus, depending on how the
    object scatters over angle (atomic number, feature size): a dark-field
    segment image moves with the incident-beam angle, while the published rule
    (Ma et al. 2025) assumes the mean of a uniformly filled aperture, 2/3 of
    its radius. On a Gaussian phase object the segment images move 0.80 times
    the applied shift at a width of 0.25 A and 1.04-1.11 times at 0.4-1.2 A.

    Parameters
    ----------
    dataset : Dataset4DStem
        The 4D-STEM dataset to reconstruct from.
    n_dark_field_segments : int, optional
        Number of azimuthal detector segments.
    verbosity : int, optional
        Diagnostic-plot verbosity; ``> 0`` shows the bright-field mask, ``> 1``
        also plots the SSNR curves and (with ``dark_field_scaling``) the
        scaling factor map.
    bright_field_mask_threshold : float, optional
        Fraction of the peak mean-CBED intensity above which a detector pixel
        is classified as bright field.
    upsample : float or str, optional
        Upsampling factor applied to the reconstructed grid, or ``"nyquist"``
        to resolve it from the acquisition geometry.
    return_snr : bool, optional
        Also compute and return the half-split SSNR map.
    snr_blur_sigma : float, optional
        Gaussian blur sigma (in radial bins) applied to the SSNR curve before
        it is scattered back onto the 2D grid.
    shift_method : {"drizzle", "fourier"}, optional
        How each segment is sub-pixel shifted and accumulated onto the
        upsampled grid.
    drizzle_pixfrac : float, optional
        Drizzle drop size, as a fraction of an input pixel; used only when
        `shift_method` is ``"drizzle"``.
    drizzle_kde_sigma : float, optional
        Drizzle Nadaraya-Watson smoothing sigma for sparse holes; used only
        when `shift_method` is ``"drizzle"``.
    dark_field_scaling : bool, optional
        Rescale each segment image to account for scattering past the
        detector edge.
    dead_frame_fraction : float, optional
        Minimum fraction of the incident current a scan position must record
        to receive `dark_field_scaling` correction.
    region : {"dark_field", "bright_field"}, optional
        Which detector pixels feed the azimuthal segments.
    _segment_geometry : _SegmentGeometry, optional
        Private. Precomputed aberration-independent segment geometry, to
        skip rebuilding it. Default builds it.

    Returns
    -------
    tuple of (torch.Tensor, torch.Tensor or None)
        The tilt-corrected (dark- or bright-field) image, and the half-split
        SSNR map when `return_snr` is True, else ``None``.

    Examples
    --------
    The image lies on the scan grid up-sampled by ``upsample`` along both scan
    axes, in ``(scan_y, scan_x)`` order. The second slot is ``None`` unless
    ``return_snr=True``, and then it is an SSNR map of the image's shape:

    >>> import numpy as np, torch
    >>> import warp
    >>> warp.config.log_level = warp.LOG_WARNING  # module-load chatter depends on the cache
    >>> from scatterem.data.datasets import Dataset4DStem
    >>> from scatterem.reconstruction.darkfield.core import (
    ...     tilt_corrected_dark_field,
    ... )
    >>> ky, kx = np.mgrid[:16, :16] - 8
    >>> disk = (np.hypot(ky, kx) <= 4).astype(np.float32)
    >>> rng = np.random.default_rng(0)
    >>> cube = disk + 0.05 * rng.random((6, 5, 16, 16), dtype=np.float32)
    >>> ds = Dataset4DStem.from_array(
    ...     cube,
    ...     energy=300e3,
    ...     semiconvergence_angle=4e-3,
    ...     scan_step=29.0,
    ...     reciprocal_step=8.3e-3,
    ...     device=torch.device("cpu"),
    ... )
    >>> image, ssnr = tilt_corrected_dark_field(
    ...     ds, n_dark_field_segments=8, upsample=2
    ... )
    >>> tuple(image.shape), image.dtype, ssnr is None
    ((12, 10), torch.float32, True)
    >>> image, ssnr = tilt_corrected_dark_field(
    ...     ds, n_dark_field_segments=8, upsample=2, return_snr=True
    ... )
    >>> ssnr.shape == image.shape
    True
    """

    if _segment_geometry is None:
        _segment_geometry = _tcdf_segment_geometry(
            dataset,
            n_dark_field_segments=n_dark_field_segments,
            verbosity=verbosity,
            bright_field_mask_threshold=bright_field_mask_threshold,
            dark_field_scaling=dark_field_scaling,
            dead_frame_fraction=dead_frame_fraction,
            region=region,
            upsample=upsample,
        )
    else:
        _segment_geometry._check(
            dataset,
            bright_field_mask_threshold,
            n_dark_field_segments,
            upsample,
            region,
            dark_field_scaling,
            dead_frame_fraction,
        )

    upsample_int = _segment_geometry.upsample_int
    vdf_stack1 = _segment_geometry.vdf_stack1
    vdf_stack2 = _segment_geometry.vdf_stack2
    df_shifts_dy1, df_shifts_dx1, df_shifts_dy2, df_shifts_dx2 = (
        _halfset_segment_shifts(dataset, _segment_geometry)
    )

    tcDF1, tcDF2 = _shift_accumulate_halfsets(
        dataset,
        upsample_int,
        vdf_stack1,
        vdf_stack2,
        df_shifts_dy1,
        df_shifts_dx1,
        df_shifts_dy2,
        df_shifts_dx2,
        shift_method=shift_method,
        drizzle_pixfrac=drizzle_pixfrac,
        drizzle_kde_sigma=drizzle_kde_sigma,
    )

    tcDF = tcDF2 + tcDF1

    if return_snr:
        sampling_tcdf = tuple((dataset.sampling[:2] / upsample_int).tolist())

        ssnr_tcdf_1d, q_rad, bin_idx = compute_ssnr_from_halfset_images(
            tcDF1,
            tcDF2,
            sampling_tcdf,
            verbosity=verbosity,
            gaussian_sigma=snr_blur_sigma,
        )

        # Cut above probe-support limit
        k_max = 1 * dataset.meta.semiconvergence_angle / dataset.meta.wavelength
        ssnr_tcdf_1d[q_rad > 1.95 * k_max] = 0
        # Raw half-split power ratio (dose- and object-bearing). NOT divided by
        # sqrt(fluence): both fusion channels must share this common reference so
        # the Eq. 39 Wiener crossover is physical (see FF-STEM SSNR recalibration).
        ssnr_tcdf = ssnr_tcdf_1d[bin_idx].reshape(tcDF1.shape)
        if verbosity > 1:
            NN = ssnr_tcdf_1d.shape[0]
            plt.figure()
            plt.plot(
                q_rad.cpu().numpy()[: NN // 2],
                ssnr_tcdf_1d.cpu().numpy()[: NN // 2],
                label="SSNRq_blur radial avg",
            )
            plt.xlabel("Spatial frequency |q| (1/Å)")
            plt.ylabel("SSNR_tcDF(|q|)")
            plt.grid(True)
            plt.tight_layout()
        return tcDF, ssnr_tcdf

    return tcDF, None


def tilt_corrected_dark_field_depth_section(
    dataset: Dataset4DStem,
    depth_angstroms=torch.tensor,
    bright_field_mask_threshold: float = 0.3,
    return_device: torch.device | None = None,
    upsample: float | str = "nyquist",
    n_dark_field_segments: int = 32,
    verbosity: int = 0,
    dark_field_scaling: bool = False,
    dead_frame_fraction: float = _DEAD_FRAME_FRACTION,
    region: str = "dark_field",
    return_snr: bool = False,
    ssnr_return_device: torch.device | None = None,
    _segment_geometry: "_SegmentGeometry | None" = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
    """Stack tcDF reconstructions swept over a defocus (depth) series.

    Runs :func:`tilt_corrected_dark_field` once per entry of
    ``depth_angstroms`` (each added to ``aberrations[0]``) and stacks the
    results as ``(n_depths, ny * up, nx * up)``, with ``up`` the one isotropic
    factor :func:`tilt_corrected_dark_field` uses (the larger of the two per-axis
    factors ``upsample`` resolves to). ``dark_field_scaling`` /
    ``dead_frame_fraction`` / ``region`` are those of
    :func:`tilt_corrected_dark_field` and are shared by all depths: the
    segment geometry that carries them is built once. The probe has the
    aberrations ``meta.aberrations`` with C10 less ``meta.defocus_guess``, as
    in :func:`~scatterem.reconstruction.ptychography.direct.direct_ptychography`.

    Parameters
    ----------
    dataset : Dataset4DStem
        The 4D-STEM dataset to reconstruct from.
    depth_angstroms : torch.Tensor
        Defocus offsets, in Angstrom, added to ``aberrations[0]`` at each
        step. The guess ``meta.defocus_guess`` is subtracted once.
    bright_field_mask_threshold : float, optional
        Fraction of the peak mean-CBED intensity above which a detector pixel
        is classified as bright field.
    return_device : torch.device, optional
        Device the returned stack is placed on. Defaults to CPU.
    upsample : float or str, optional
        Upsampling factor applied to the reconstructed grid, or ``"nyquist"``
        to resolve it from the acquisition geometry. On an anisotropic scan,
        ``"nyquist"`` resolves to two factors, and the larger one applies to
        both axes.
    n_dark_field_segments : int, optional
        Number of azimuthal detector segments.
    verbosity : int, optional
        Diagnostic-plot verbosity, forwarded to :func:`tilt_corrected_dark_field`.
    dark_field_scaling : bool, optional
        Rescale each segment image to account for scattering past the
        detector edge.
    dead_frame_fraction : float, optional
        Minimum fraction of the incident current a scan position must record
        to receive `dark_field_scaling` correction.
    region : {"dark_field", "bright_field"}, optional
        Which detector pixels feed the azimuthal segments.
    return_snr : bool, optional
        Also return the per-plane half-set SSNR stack -- a by-product of the
        two half-set images :func:`tilt_corrected_dark_field` builds either
        way. With the default ``False`` the return is the bare image stack,
        unchanged.
    ssnr_return_device : torch.device, optional
        Device the SSNR stack is placed on. Defaults to `return_device`, so a
        fusion that consumes the SSNR on the GPU can keep it resident while
        the image cube goes to the host.
    _segment_geometry : _SegmentGeometry, optional
        Precomputed segment geometry, built on `dataset` with these
        parameters. Private. Default builds it.

    Returns
    -------
    torch.Tensor or tuple of torch.Tensor
        The stacked tcDF images, shape ``(n_depths, ny * up, nx * up)``, on
        `return_device`. With ``return_snr=True``, that stack and the
        per-plane half-set SSNR stack on `ssnr_return_device`; the SSNR slot
        is ``None`` only when ``depth_angstroms`` is empty (no plane to size
        it by).
    """
    # Resolved here rather than in the signature: a torch.device() call in a
    # default is evaluated once at import (B008). None keeps the same public
    # default without the shared instance.
    return_device = torch.device("cpu") if return_device is None else return_device
    if ssnr_return_device is None:
        ssnr_return_device = return_device
    # Allocated LAZILY from the first plane's actual SSNR shape.
    ssnr_depth_section = None
    aberrations_array = dataset.meta.aberrations.array.detach().clone()
    # The sweep moves ONLY aberrations.array[0], and the aberrations reach tcDF
    # only through ``shifts``; the masks, the segment membership and the full
    # ``(ny, nx, M, M)`` gather behind them are identical at every depth.  Build
    # them once instead of ``len(depth_angstroms)`` times.
    segment_geometry = _segment_geometry or _tcdf_segment_geometry(
        dataset,
        n_dark_field_segments=n_dark_field_segments,
        verbosity=verbosity,
        bright_field_mask_threshold=bright_field_mask_threshold,
        upsample=upsample,
        region=region,
        dark_field_scaling=dark_field_scaling,
        dead_frame_fraction=dead_frame_fraction,
    )
    # The geometry's factor is the ONE isotropic factor every plane is built on.
    # A per-axis allocation does not hold an anisotropic Nyquist upsampling.
    up = int(segment_geometry.upsample_int)
    tcDF_depth_section = torch.zeros(
        len(depth_angstroms),
        dataset.shape[0] * up,
        dataset.shape[1] * up,
        device=return_device,
    )
    try:
        for i, depth in enumerate(
            tqdm(depth_angstroms, desc="Assembling tcDF depth section")
        ):
            dataset.meta.aberrations.array[:] = aberrations_array
            dataset.meta.aberrations.array[0] += depth
            dataset.meta.aberrations.array[3:] = 0
            # The MODULE-LEVEL function: it caches nothing on the dataset, so no
            # plane's image or SSNR is left behind between planes.
            tcDF_i, ssnr_i = tilt_corrected_dark_field(
                dataset,
                n_dark_field_segments=n_dark_field_segments,
                verbosity=verbosity,
                bright_field_mask_threshold=bright_field_mask_threshold,
                upsample=upsample,
                dark_field_scaling=dark_field_scaling,
                dead_frame_fraction=dead_frame_fraction,
                region=region,
                return_snr=return_snr,
                _segment_geometry=segment_geometry,
            )
            tcDF_depth_section[i].copy_(tcDF_i)
            if return_snr and ssnr_i is not None:
                if ssnr_depth_section is None:
                    ssnr_depth_section = torch.zeros(
                        (len(depth_angstroms), *ssnr_i.shape),
                        dtype=ssnr_i.dtype,
                        device=ssnr_return_device,
                    )
                ssnr_depth_section[i].copy_(ssnr_i)
    finally:
        # Restored on the exception path too: without this an OOM mid-sweep leaves
        # the dataset focused at C1_fit + depth_k with aberrations[3:] zeroed.
        dataset.meta.aberrations.array[:] = aberrations_array
    if return_snr:
        return tcDF_depth_section, ssnr_depth_section
    return tcDF_depth_section


__all__ = [
    "compute_ssnr_from_halfset_images",
    "gaussian_blur1d",
    "tilt_corrected_dark_field",
    "tilt_corrected_dark_field_depth_section",
]
