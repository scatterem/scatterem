import torch 
import matplotlib.pyplot as plt
import numpy as np



import scatterem2.vis as vis
from scatterem2.vis.visualization import show_2d
from scatterem2.utils.stem import fftfreq2
from scatterem2.utils.utils import radial_average2, fuse_images_fourier_weighted
from scatterem2.utils.data.datasets import Dataset4dstem
from scatterem2.vis.custom_normalizations import NormalizationConfig

def fused_full_field(
    dataset: Dataset4dstem, 
    ptycho_image: torch.Tensor, 
    tcdf_image: torch.Tensor, 
    ptycho_ssnr: torch.Tensor, 
    tcdf_ssnr: torch.Tensor, 
    verbosity=0
) -> torch.Tensor:
    if ptycho_image.shape != tcdf_image.shape:
        raise ValueError(f"Input image shapes do not match: ptycho_image.shape={ptycho_image.shape}, tcdf_image.shape={tcdf_image.shape}")
    if ptycho_ssnr.shape != tcdf_ssnr.shape:
        raise ValueError(f"Input SSNR shapes do not match: ptycho_ssnr.shape={ptycho_ssnr.shape}, tcdf_ssnr.shape={tcdf_ssnr.shape}")
    device = dataset.device
    dk = dataset.sampling[-2:]
    upsample = np.array(ptycho_image.shape[0])/np.array(dataset.shape[0])

    ssnr1 = ptycho_ssnr
    ssnr2 = tcdf_ssnr
    denominator = ssnr1 + ssnr2
    safe_denominator = denominator + torch.as_tensor(1e-12, device=device)
    w1 = ssnr1 / safe_denominator# / denominator
    w2 = ssnr2 / safe_denominator
    w1[ssnr1 == 0] = 0
    w2[ssnr2 == 0] = 0
    q = fftfreq2(ptycho_image.shape, dataset.sampling[:2]/upsample)
    qn = torch.norm(q, dim=0)
    q_cutoff = dataset.meta.semiconvergence_angle / dataset.meta.wavelength

    two_alpha = 2.0 * q_cutoff

    # Start tapering near the outer ~15–20% of frequencies
    q_soft_start = 0.5 * two_alpha     # start reducing tcDF here
    q_soft_end   = 0.7 * two_alpha           # fully killed at Nyquist

    soft_mask = torch.ones_like(qn)

    in_taper = (qn >= q_soft_start) & (qn <= q_soft_end)
    t = (qn[in_taper] - q_soft_start) / (q_soft_end - q_soft_start)  # 0 → 1
    soft_mask[in_taper] = 0.5 * (1.0 + torch.cos(np.pi * t))        # 1 → 0

    soft_mask[qn > q_soft_end] = 0.0
    soft_mask = soft_mask.to(device)
    # apply only to tcDF weights
    w2 = w2 * soft_mask

    # renormalize w1/w2 so they stay coupled
    denom_soft = w1 + w2 + torch.as_tensor(1e-12, device=device)
    w1 = w1 / denom_soft
    w2 = w2 / denom_soft

    
    two_alpha_mask = qn <=0.99 * 2 * q_cutoff
    
    w1[~two_alpha_mask] = 0
    w2[~two_alpha_mask] = 0

    fused, ptycho_filtered, tcdf_filtered = fuse_images_fourier_weighted(ptycho_image, tcdf_image, w1, w2)

    if verbosity > 1:
        sampling_tcdf = tuple((dataset.sampling[:2] / upsample).tolist())
        q_bins, ssnr_ptycho_1d = radial_average2(ptycho_ssnr.cpu().numpy(), sampling_tcdf)
        q_bins, ssnr_tcdf_1d = radial_average2(tcdf_ssnr.cpu().numpy(), sampling_tcdf)
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