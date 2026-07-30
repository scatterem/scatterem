import torch
import matplotlib.pyplot as plt
import numpy as np
import scatterem.vis as vis
from scatterem.utils.grids import fft_frequencies_2d
from scatterem.utils.grids import radial_average
from scatterem.utils.utils import fuse_images_fourier_weighted
from scatterem.utils.data.datasets import Dataset4dstem
from scatterem.vis.normalization import NormalizationConfig


def fused_full_field(
    dataset: Dataset4dstem,
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
) -> torch.Tensor:
    if ptycho_image.shape != tcdf_image.shape:
        raise ValueError(f"Input image shapes do not match: ptycho_image.shape={ptycho_image.shape}, tcdf_image.shape={tcdf_image.shape}")
    if ptycho_ssnr.shape != tcdf_ssnr.shape:
        raise ValueError(f"Input SSNR shapes do not match: ptycho_ssnr.shape={ptycho_ssnr.shape}, tcdf_ssnr.shape={tcdf_ssnr.shape}")
    device = dataset.device
    dk = dataset.sampling[-2:]
    upsample = np.array(ptycho_image.shape[0])/np.array(dataset.shape[0])

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
                f"fused_full_field: {_bad.float().mean().item()*100:.0f}% of the "
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
    q = fft_frequencies_2d(
        ptycho_image.shape, dataset.sampling[:2] / upsample, device=device
    )
    qn = torch.norm(q, dim=0)
    q_cutoff = dataset.meta.semiconvergence_angle / dataset.meta.wavelength

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
    passband = band_mask
    if float((w1[passband] + w2[passband]).abs().sum()) <= 1e-9:
        import warnings
        warnings.warn(
            "fused_full_field: both SSNRs are ~zero in the passband; "
            "falling back to ptychography-only weights (check the input SSNRs).",
            stacklevel=2,
        )
        w1 = band_mask.to(w1.dtype)
        w2 = torch.zeros_like(w2)

    fused, ptycho_filtered, tcdf_filtered = fuse_images_fourier_weighted(
        ptycho_image, tcdf_image, w1, w2, verbosity=verbosity, return_filtered=(verbosity > 1)
    )

    if verbosity > 1:
        sampling_tcdf = tuple((dataset.sampling[:2] / upsample).tolist())
        q_bins, ssnr_ptycho_1d = radial_average(ptycho_ssnr.cpu().numpy(), sampling_tcdf)
        q_bins, ssnr_tcdf_1d = radial_average(tcdf_ssnr.cpu().numpy(), sampling_tcdf)
        NN = ptycho_image.shape[0]
        plt.figure()

        snr_fused2 = ssnr_ptycho_1d + ssnr_tcdf_1d
        snr_fused_max = snr_fused2.max().item()
        # plt.plot(q_radial[:NN//2].cpu().numpy(), ssnr_rad_pctf[:NN//2].cpu().numpy() , label='Ptychography Heuristic')
        dash_width = 3
        plt.plot(q_bins[:NN//2], ((snr_fused2))[:NN//2], label='FF-STEM', linewidth=3, color='red')
        plt.plot(q_bins[:NN//2], ssnr_ptycho_1d[:NN//2] , label='Direct Ptychography',
                linestyle='--', linewidth=dash_width, color='green')
        plt.plot(q_bins[:NN//2], ssnr_tcdf_1d[:NN//2] , label='tcDF-STEM',
                linestyle='--', linewidth=dash_width, color='grey')

        # plt.plot(q_radial[:NN//2].cpu().numpy(), ((snr_fused))[:NN//2].cpu().numpy(), label='FF-STEM Heuristic')

        plt.xlabel("Spatial frequency |q|")
        plt.ylabel("SSNR(|q|)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        q_cutoff = dataset.meta.semiconvergence_angle / dataset.meta.wavelength
        r_cutoff = q_cutoff / dk[0]
        ds = denominator.shape[0]//4

        nconf = NormalizationConfig(interval_type="manual", vmin=0, vmax=1)
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        fig_bf_analytic2, ax_bf_analytic2 = vis.show_2d(
            [torch.fft.fftshift(w1),torch.fft.fftshift(w2)],
            cbar=True,
            title=["Weight Ptychography", "Weight tcDF"],
            figax=(fig, ax),
            norm=nconf,
            cmap='inferno',
            scalebar={"sampling":dataset.sampling[-1],"length":1,"units":r"Å$^{-1}$"}
        )

        nconf = NormalizationConfig(interval_type="manual", vmin=0, vmax=1)
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        fig_bf_analytic2, ax_bf_analytic2 = vis.show_2d(
            [torch.fft.fftshift(ssnr1)],
            cbar=True,
            title=["SSNR Ptychography"],
            figax=(fig, ax),
            norm=nconf,
            cmap='magma',
            scalebar={"sampling":sampling_tcdf[0],"length":10,"units":"Å"}
        )
        plt.show()
        nconf = NormalizationConfig(interval_type="manual", vmin=0, vmax=ssnr2.max().item())
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        fig_bf_analytic2, ax_bf_analytic2 = vis.show_2d(
            [torch.fft.fftshift(ssnr2)],
            cbar=True,
            title=["SSNR tcDF"],
            figax=(fig, ax),
            norm=nconf,
            cmap='magma'
        )
        plt.show()
        nconf = NormalizationConfig(interval_type="manual", vmin=0, vmax=1)
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        fig_bf_analytic2, ax_bf_analytic2 = vis.show_2d(
            [torch.fft.fftshift(denominator)],
            cbar=True,
            title=["SSNR FF-STEM"],
            figax=(fig, ax),
            norm=nconf,
            cmap='magma'
        )
        # α and 2α circles on FFT panels

        titles = ["Ptychography", "tcDF", "Fused"]
        vis.show_2d([ptycho_image,tcdf_image,fused], cbar=True, title=titles)
    return fused, ptycho_filtered, tcdf_filtered
