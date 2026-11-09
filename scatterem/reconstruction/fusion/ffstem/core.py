"""SSNR-weighted fusion of a ptychographic and a dark-field full-field image."""

import numpy as np
import torch

from scatterem.data.datasets import Dataset4DStem

from scatterem.utils.spectral_mesh import dual_mesh
from scatterem.reconstruction.fusion._band import fuse_images_fourier_weighted
from scatterem.utils.lazy_plotting import plt, vis
from scatterem.utils.spectral_mesh import shell_means


def _metadata_cutoff(dataset: Dataset4DStem):
    """Return the aperture cutoff from required dataset metadata."""
    if dataset.meta is None:
        raise ValueError("full-field fusion requires dataset metadata")
    return dataset.meta.semiconvergence_angle / dataset.meta.wavelength


def fused_full_field(
    dataset: Dataset4DStem,
    ptycho_image: torch.Tensor,
    tcdf_image: torch.Tensor,
    ptycho_ssnr: torch.Tensor,
    tcdf_ssnr: torch.Tensor,
    verbosity=0,
    taper: bool = False,
    q_soft_start_frac: float = 0.5,
    q_soft_end_frac: float = 0.7,
    q_hard_cutoff_frac: float = 1.0,
    ptycho_lowq_cutoff_frac: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Fuse a ptychographic and a dark-field image by their measured SSNR.

    Implements the SSNR-weighted full-field fusion of the FF-STEM paper: at
    each spatial frequency the two channels are combined with weights
    ``ssnr_i / (ssnr_1 + ssnr_2)``, so each contributes where it actually
    carries information. Neither channel transfers beyond the ``2 * alpha``
    band limit, which is imposed explicitly.

    A channel that is partly non-finite -- the tilt-correction can produce NaN
    shifts on low-convergence data -- is zeroed with a warning so the fusion
    falls back to the surviving channel instead of returning all-NaN. If both
    SSNRs are about zero inside the passband the result would be silently
    all-zero, so that case warns and falls back to ptychography-only weights.

    Parameters
    ----------
    dataset : Dataset4DStem
        Supplies the device, the real-space sampling and the
        ``semiconvergence_angle`` / ``wavelength`` that set the ``2 * alpha``
        band limit.
    ptycho_image : Tensor
        Real-space ptychographic phase image, shape ``(Ny, Nx)``, already on
        the up-sampled grid.
    tcdf_image : Tensor
        Tilt-corrected dark-field image, same shape as ``ptycho_image``.
    ptycho_ssnr : Tensor
        Spectral signal-to-noise ratio of ``ptycho_image``, on the same
        frequency grid.
    tcdf_ssnr : Tensor
        Spectral signal-to-noise ratio of ``tcdf_image``, same shape as
        ``ptycho_ssnr``.
    verbosity : int
        Above zero, plot the weights, the filtered channels and the fused
        image with matplotlib.
        Default 0.
    taper : bool
        Apply the legacy empirical apodization of the dark-field weight. Not
        part of the paper's equation; kept for reproducibility.
        Default False.
    q_soft_start_frac, q_soft_end_frac : float
        Start and end of that taper, as fractions of ``2 * alpha``. Ignored
        unless ``taper`` is set.
    q_hard_cutoff_frac : float
        Hard band limit as a fraction of ``2 * alpha``. Both weights are zero
        beyond it.
        Default 1.0.
    ptycho_lowq_cutoff_frac : float
        When positive, transfer the ptychographic weight to the dark-field
        below ``ptycho_lowq_cutoff_frac * q_nyquist`` with a cosine ramp, to
        suppress the low-frequency cupping halo of the direct-ptychography
        phase. ``0.0`` disables it.
        Default 0.0.

    Returns
    -------
    fused : Tensor
        The fused real-space image.
    ptycho_filtered : Tensor or None
        The ptychographic channel after its weight was applied. ``None`` unless
        ``verbosity > 1`` (the filtered channels are only materialised for the
        diagnostic plots).
    tcdf_filtered : Tensor or None
        The dark-field channel after its weight was applied. ``None`` unless
        ``verbosity > 1``.

    Raises
    ------
    ValueError
        If the two images, or the two SSNRs, do not have matching shapes.

    Examples
    --------
    Default ``verbosity=0`` returns the fused image and two ``None`` placeholders
    (filtered channels are only built when ``verbosity > 1``). Mismatched image
    shapes raise:

    >>> import numpy as np, torch
    >>> import warp
    >>> warp.config.log_level = warp.LOG_WARNING  # module-load chatter depends on the cache
    >>> from scatterem.data import Aberrations
    >>> from scatterem.data.datasets import Dataset4DStem
    >>> from scatterem.reconstruction.ptychography.direct import (
    ...     direct_ptychography,
    ... )
    >>> from scatterem.reconstruction.darkfield.core import (
    ...     tilt_corrected_dark_field,
    ... )
    >>> from scatterem.reconstruction.fusion.ffstem.core import fused_full_field
    >>> n_scan, n_det = 8, 16
    >>> rng = np.random.default_rng(0)
    >>> ky, kx = np.meshgrid(
    ...     np.arange(n_det) - n_det // 2,
    ...     np.arange(n_det) - n_det // 2,
    ...     indexing="ij",
    ... )
    >>> disk = (np.hypot(ky, kx) <= 4).astype(np.float32)
    >>> ry, rx = np.meshgrid(
    ...     np.arange(n_scan), np.arange(n_scan), indexing="ij"
    ... )
    >>> mod = 1 + 0.15 * np.cos(2 * np.pi * ry / n_scan) * np.cos(
    ...     2 * np.pi * rx / n_scan
    ... )
    >>> cube = disk[None, None] * mod[:, :, None, None].astype(np.float32)
    >>> cube = np.clip(
    ...     cube + 0.01 * rng.standard_normal(cube.shape).astype(np.float32),
    ...     0,
    ...     None,
    ... )
    >>> ab = Aberrations(array=torch.zeros(12))
    >>> ab.array[0] = -2.2e4
    >>> ds = Dataset4DStem.from_array(
    ...     np.ascontiguousarray(cube),
    ...     energy=300e3,
    ...     semiconvergence_angle=4e-3,
    ...     scan_step=29.0,
    ...     reciprocal_step=8.3e-3,
    ...     transform_to_amplitudes=True,
    ...     aberrations=ab,
    ...     device=torch.device("cpu"),
    ... )
    >>> phase, _ = direct_ptychography(
    ...     ds, upsample=2, n_batches=4, phase_sign="preserve"
    ... )
    >>> tcdf, _ = tilt_corrected_dark_field(
    ...     ds,
    ...     n_dark_field_segments=8,
    ...     upsample=2,
    ...     bright_field_mask_threshold=0.3,
    ... )
    >>> fused, ptycho_f, tcdf_f = fused_full_field(
    ...     ds,
    ...     phase,
    ...     tcdf,
    ...     torch.ones_like(phase),
    ...     torch.ones_like(tcdf),
    ...     verbosity=0,
    ... )
    >>> tuple(fused.shape), ptycho_f is None, tcdf_f is None
    ((16, 16), True, True)
    >>> fused_full_field(
    ...     ds, phase, tcdf[:, :8], torch.ones_like(phase), torch.ones_like(tcdf)
    ... )
    Traceback (most recent call last):
        ...
    ValueError: Input image shapes do not match: ptycho_image.shape=torch.Size([16, 16]), tcdf_image.shape=torch.Size([16, 8])
    """
    if ptycho_image.shape != tcdf_image.shape:
        raise ValueError(
            f"Input image shapes do not match: ptycho_image.shape={ptycho_image.shape}, tcdf_image.shape={tcdf_image.shape}"
        )
    if ptycho_ssnr.shape != tcdf_ssnr.shape:
        raise ValueError(
            f"Input SSNR shapes do not match: ptycho_ssnr.shape={ptycho_ssnr.shape}, tcdf_ssnr.shape={tcdf_ssnr.shape}"
        )
    device = dataset.device
    upsample = np.array(ptycho_image.shape[0]) / np.array(dataset.shape[0])

    # NaN-robustness: a broken channel (e.g. the tcDF tilt-correction producing
    # NaN shifts on low-convergence data -> NaN dark-field image/SSNR) must not
    # poison the fused result. Zero out any non-finite image/SSNR so the fusion
    # falls back to the surviving channel instead of returning all-NaN.
    import warnings as _warnings

    for _name, _img, _ssnr in (
        ("ptycho", ptycho_image, ptycho_ssnr),
        ("tcdf", tcdf_image, tcdf_ssnr),
    ):
        _bad = ~torch.isfinite(_img) | ~torch.isfinite(_ssnr)
        if bool(_bad.any()):
            _warnings.warn(
                f"fused_full_field: {_bad.float().mean().item() * 100:.0f}% of the "
                f"{_name} channel is non-finite; zeroing it so the fusion falls "
                f"back to the other channel.",
                stacklevel=2,
            )
    ptycho_image = torch.nan_to_num(ptycho_image, nan=0.0, posinf=0.0, neginf=0.0)
    tcdf_image = torch.nan_to_num(tcdf_image, nan=0.0, posinf=0.0, neginf=0.0)
    ptycho_ssnr = torch.nan_to_num(ptycho_ssnr, nan=0.0, posinf=0.0, neginf=0.0)
    tcdf_ssnr = torch.nan_to_num(tcdf_ssnr, nan=0.0, posinf=0.0, neginf=0.0)

    ssnr1 = ptycho_ssnr
    ssnr2 = tcdf_ssnr
    denominator = ssnr1 + ssnr2
    safe_denominator = denominator + torch.as_tensor(1e-12, device=device)
    w1 = ssnr1 / safe_denominator
    w2 = ssnr2 / safe_denominator
    w1[ssnr1 == 0] = 0
    w2[ssnr2 == 0] = 0
    q = dual_mesh(
        ptycho_image.shape,
        dataset.sampling[:2] / upsample,
        staggered=False,
        device=device,
    )
    qn = torch.linalg.vector_norm(q, dim=0)
    q_cutoff = _metadata_cutoff(dataset)

    two_alpha = 2.0 * q_cutoff

    # Low-q halo suppression. The direct-ptychography phase has a low-frequency
    # "cupping" artifact (SSB has no reliable DC/low-q transfer); the SSNR weights
    # otherwise leak it into the fused image at low q (a visible halo). Per the
    # documented design the low-freq band should come EXCLUSIVELY from the (clean)
    # dark-field, so below q_lp = ptycho_lowq_cutoff_frac * q_nyquist we smoothly
    # transfer the ptycho weight to the dark-field (cosine ramp: ptycho 0 at DC ->
    # full at q_lp). Preserves w1+w2. Default 0.0 = OFF (unchanged behavior).
    if ptycho_lowq_cutoff_frac and ptycho_lowq_cutoff_frac > 0.0:
        q_lp = float(ptycho_lowq_cutoff_frac) * float(qn.max())
        ramp = 0.5 * (1.0 - torch.cos(np.pi * (qn / (q_lp + 1e-30)).clamp(0.0, 1.0)))
        transfer = w1 * (1.0 - ramp)
        w1 = w1 * ramp
        w2 = w2 + transfer

    if taper:
        # Optional empirical apodization of the tcDF weight (legacy behavior; OFF by
        # default). Not part of paper Eq. 39 — kept for reproducibility only.
        q_soft_start = q_soft_start_frac * two_alpha
        q_soft_end = q_soft_end_frac * two_alpha
        t = ((qn - q_soft_start) / (q_soft_end - q_soft_start)).clamp(0.0, 1.0)
        soft_mask = 0.5 * (1.0 + torch.cos(np.pi * t))
        w2 = w2 * soft_mask
        denom_soft = w1 + w2 + torch.as_tensor(1e-12, device=device)
        w1 = w1 / denom_soft
        w2 = w2 / denom_soft

    # Physical band limit: neither channel transfers beyond 2α. Default cut is the
    # full 2α; the empirical SSNRs supply the smooth rolloff approaching it.
    band_mask = qn <= q_hard_cutoff_frac * two_alpha
    w1[~band_mask] = 0
    w2[~band_mask] = 0

    # Degeneracy guard: if both channels carry no SSNR in the passband, the fused
    # image would be silently all-zero. Fall back to ptychography-alone and warn.
    #
    # The passband restriction is already IN the operands: the two statements
    # above set w1 and w2 to exactly 0 wherever ``~band_mask``, so
    # ``(w1 + w2).abs()`` is exactly +0.0 outside the passband and the whole-array
    # sum is the passband sum.  Spelling it ``w1[band_mask] + w2[band_mask]``
    # cost TWO ``masked_select`` gathers over the full [Nqy,Nqx] grid, each of
    # which has to synchronize to size its own output, so this one guard was
    # three device round-trips for one scalar: measured 0.1385 -> 0.0505 ms at
    # the 245^2 registered shape, 7.4 % of ``r15/medium/fused``.
    #
    # NOT bit-identical to the gathered form -- the reduction runs over Nqy*Nqx
    # terms instead of |passband|, so the summation TREE differs and the
    # non-zero terms can round differently.  The added terms are exactly +0.0
    # (``abs`` maps -0.0 to +0.0, so no signed-zero case survives), and the
    # threshold is a degeneracy test at 1e-9 against a sum that is either
    # ~0 or O(Nqy*Nqx) -- a flip would need the true sum to sit within an ULP of
    # 1e-9.  Verified equal over randomised inputs in
    # a benchmark script.
    if float((w1 + w2).abs().sum()) <= 1e-9:
        import warnings

        warnings.warn(
            "fused_full_field: both SSNRs are ~zero in the passband; "
            "falling back to ptychography-only weights (check the input SSNRs).",
            stacklevel=2,
        )
        w1 = band_mask.to(w1.dtype)
        w2 = torch.zeros_like(w2)

    # assume_finite=True is load-bearing and provable HERE and nowhere else: both
    # images are the direct outputs of the ``nan_to_num`` above and nothing between
    # touches them, so the robust normalization does not have to ask the device
    # whether they contain NaN -- a blocking host sync, twice, worth 8.7 % of
    # ``r15_direct_ptychography/medium/fused``.
    fused, ptycho_filtered, tcdf_filtered = fuse_images_fourier_weighted(
        ptycho_image,
        tcdf_image,
        w1,
        w2,
        verbosity=verbosity,
        return_filtered=(verbosity > 1),
        assume_finite=True,
    )

    if verbosity > 1:
        sampling_tcdf = tuple((dataset.sampling[:2] / upsample).tolist())
        q_bins, ssnr_ptycho_1d = shell_means(ptycho_ssnr.cpu().numpy(), sampling_tcdf)
        q_bins, ssnr_tcdf_1d = shell_means(tcdf_ssnr.cpu().numpy(), sampling_tcdf)
        NN = ptycho_image.shape[0]
        plt.figure()

        snr_fused2 = ssnr_ptycho_1d + ssnr_tcdf_1d
        # plt.plot(q_radial[:NN//2].cpu().numpy(), ssnr_rad_pctf[:NN//2].cpu().numpy() , label='Ptychography Heuristic')
        dash_width = 3
        plt.plot(
            q_bins[: NN // 2],
            (snr_fused2)[: NN // 2],
            label="FF-STEM",
            linewidth=3,
            color="red",
        )
        plt.plot(
            q_bins[: NN // 2],
            ssnr_ptycho_1d[: NN // 2],
            label="Direct Ptychography",
            linestyle="--",
            linewidth=dash_width,
            color="green",
        )
        plt.plot(
            q_bins[: NN // 2],
            ssnr_tcdf_1d[: NN // 2],
            label="tcDF-STEM",
            linestyle="--",
            linewidth=dash_width,
            color="grey",
        )

        # plt.plot(q_radial[:NN//2].cpu().numpy(), ((snr_fused))[:NN//2].cpu().numpy(), label='FF-STEM Heuristic')

        plt.xlabel("Spatial frequency |q|")
        plt.ylabel("SSNR(|q|)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        nconf = vis.NormalizationConfig(interval_type="manual", vmin=0, vmax=1)
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        fig_bf_analytic2, ax_bf_analytic2 = vis.show_2d(
            [torch.fft.fftshift(w1), torch.fft.fftshift(w2)],
            cbar=True,
            title=["Weight Ptychography", "Weight tcDF"],
            figax=(fig, ax),
            norm=nconf,
            cmap="inferno",
            scalebar={
                "sampling": dataset.sampling[-1],
                "length": 1,
                "units": r"Å$^{-1}$",
            },
        )

        nconf = vis.NormalizationConfig(interval_type="manual", vmin=0, vmax=1)
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        fig_bf_analytic2, ax_bf_analytic2 = vis.show_2d(
            [torch.fft.fftshift(ssnr1)],
            cbar=True,
            title=["SSNR Ptychography"],
            figax=(fig, ax),
            norm=nconf,
            cmap="magma",
            scalebar={"sampling": sampling_tcdf[0], "length": 10, "units": "Å"},
        )
        plt.show()
        nconf = vis.NormalizationConfig(
            interval_type="manual", vmin=0, vmax=ssnr2.max().item()
        )
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        fig_bf_analytic2, ax_bf_analytic2 = vis.show_2d(
            [torch.fft.fftshift(ssnr2)],
            cbar=True,
            title=["SSNR tcDF"],
            figax=(fig, ax),
            norm=nconf,
            cmap="magma",
        )
        plt.show()
        nconf = vis.NormalizationConfig(interval_type="manual", vmin=0, vmax=1)
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        fig_bf_analytic2, ax_bf_analytic2 = vis.show_2d(
            [torch.fft.fftshift(denominator)],
            cbar=True,
            title=["SSNR FF-STEM"],
            figax=(fig, ax),
            norm=nconf,
            cmap="magma",
        )
        # α and 2α circles on FFT panels

        titles = ["Ptychography", "tcDF", "Fused"]
        vis.show_2d([ptycho_image, tcdf_image, fused], cbar=True, title=titles)
    return fused, ptycho_filtered, tcdf_filtered


__all__ = [
    "fused_full_field",
]
