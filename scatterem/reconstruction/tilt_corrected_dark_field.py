import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
 
import scatterem.vis as vis
from scatterem.utils.data.datasets import Dataset4dstem
from typing import Union
from scatterem.utils.transfer import aberrations_to_image_shifts
from scatterem.reconstruction.drizzle import drizzle_resample


def _fourier_upsample_stack(imgs: torch.Tensor, upsample_int: int) -> torch.Tensor:
    """FFT a stack of real segment images and zero-pad to ``upsample_int`` in
    Fourier space (corner-origin quadrant packing). Returns the padded complex
    spectrum stack.

    The positive-frequency block keeps ``(D+1)//2`` bins at the top/left and the
    negative-frequency block keeps ``D//2`` bins at the bottom/right, so source
    and destination slice lengths always agree — this is odd-size-safe and
    reduces to the plain ``D//2`` split for even ``D`` (unchanged behavior)."""
    F0 = torch.fft.fft2(imgs, dim=(-2, -1))
    if upsample_int <= 1:
        return F0
    S, H, W = F0.shape
    new_H, new_W = int(H * upsample_int), int(W * upsample_int)
    py, ny = (H + 1) // 2, H // 2  # positive / negative freq counts (rows)
    px, nx = (W + 1) // 2, W // 2  # positive / negative freq counts (cols)
    Fp = torch.zeros((S, new_H, new_W), dtype=F0.dtype, device=F0.device)
    Fp[:, :py, :px] = F0[:, :py, :px]  # (+y, +x)
    Fp[:, :py, new_W - nx :] = F0[:, :py, px:]  # (+y, -x)
    Fp[:, new_H - ny :, :px] = F0[:, py:, :px]  # (-y, +x)
    Fp[:, new_H - ny :, new_W - nx :] = F0[:, py:, px:]  # (-y, -x)
    return Fp


def gaussian_blur1d(x, sigma):
    """Apply 1D Gaussian blur to a 1D torch tensor."""
    if sigma <= 0:
        return x
    import math
    # 3 sigma on each side + 1 center
    k = int(math.ceil(3 * sigma))
    ksize = 2 * k + 1
    device = x.device
    dtype = x.dtype
    t = torch.arange(ksize, device=device, dtype=dtype) - k
    kernel = torch.exp(-0.5 * (t / sigma) ** 2)
    kernel = kernel / kernel.sum()
    # reflect padding currently requires at least 3D tensors, so pad after adding batch/channel dims
    x_3d = x.view(1, 1, -1)
    x_pad = torch.nn.functional.pad(x_3d, (k, k), mode='reflect')
    x_blur = torch.nn.functional.conv1d(x_pad, kernel[None, None, :], padding=0)[0, 0]
    return x_blur

def _drizzle_streamed(
    stack, shifts_hr, upsample, *, pixfrac=1.0, kde_sigma=0.0, frame_chunk=8
):
    """Drizzle a stack in frame groups: accumulate ``(accum, hits)`` per group
    and divide ONCE. Exactly equal to the single stack-level call (the splat is
    a plain sum over frames; the kde blur is linear), but the drizzle
    internals' ~8x (N,H,W) window temporaries shrink to (frame_chunk,H,W)."""
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


def compute_ssnr_from_halfset_images(tcDF1, tcDF2, sampling, gaussian_sigma=0.0, verbosity=0):
    FA = torch.fft.fftn(tcDF1, dim=(-2,-1), norm="ortho")
    FB = torch.fft.fftn(tcDF2, dim=(-2,-1), norm="ortho")

    N = 0.5 * (FA - FB)
    S = 0.5 * (FA + FB)

    power_N = N.abs()**2   # |N|^2
    power_S = S.abs()**2   # |S|^2

    ny, nx = N.shape[-2:]
    qy = torch.fft.fftfreq(ny, d=sampling[0]).to(N.device)
    qx = torch.fft.fftfreq(nx, d=sampling[1]).to(N.device)
    QY, QX = torch.meshgrid(qy, qx, indexing='ij')
    q_mag = torch.sqrt(QX**2 + QY**2)

    flat_q = q_mag.flatten()
    flat_N = power_N.flatten()
    flat_S = power_S.flatten()

    radial_bins = tcDF1.shape[0]//2
    q_min, q_max = flat_q.min(), flat_q.max()
    bin_edges = torch.linspace(q_min, q_max, steps=radial_bins+1, device=flat_q.device)
    bin_idx = torch.bucketize(flat_q, bin_edges) - 1
    bin_idx = bin_idx.clamp(min=0, max=radial_bins-1)

    # Noise power <|N|^2>
    num_N = torch.zeros(radial_bins, device=flat_N.device, dtype=flat_N.dtype)
    den_N = torch.zeros(radial_bins, device=flat_N.device, dtype=flat_N.dtype)
    num_N.index_add_(0, bin_idx, flat_N)
    den_N.index_add_(0, bin_idx, torch.ones_like(flat_N))
    den_safe_N = den_N.clamp_min(1.0)
    VarN_radial = num_N / den_safe_N

    # Total power <|S|^2>
    num_S = torch.zeros(radial_bins, device=flat_S.device, dtype=flat_S.dtype)
    den_S = torch.zeros(radial_bins, device=flat_S.device, dtype=flat_S.dtype)
    num_S.index_add_(0, bin_idx, flat_S)
    den_S.index_add_(0, bin_idx, torch.ones_like(flat_S))
    den_safe_S = den_S.clamp_min(1.0)
    S_power_radial = num_S / den_safe_S

    # Mean q for plotting
    qnum = torch.zeros(radial_bins, device=flat_q.device, dtype=flat_q.dtype)
    qnum.index_add_(0, bin_idx, flat_q)
    q_rad = qnum / den_safe_N

    # SSNR(q) ≈ (signal power) / (noise power)
    signal_power_radial = (S_power_radial - VarN_radial).clamp_min(0.0)
    SSNRq = signal_power_radial / VarN_radial.clamp_min(1e-20)
    if gaussian_sigma > 0:
        SSNRq = gaussian_blur1d(SSNRq, gaussian_sigma)
    if verbosity > 1:
        plt.figure()
        plt.semilogy(q_rad.cpu().numpy(), VarN_radial.cpu().numpy(), label="|N| radial avg")
        plt.semilogy(q_rad.cpu().numpy(), signal_power_radial.cpu().numpy(), label="|S| radial avg")
        plt.semilogy(q_rad.cpu().numpy(), SSNRq.cpu().numpy(), label="SSNRq radial avg")
        plt.xlabel("Spatial frequency |q|")
        plt.ylabel("Radial mean (|N|, |S|)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
    return SSNRq, q_rad, bin_idx

def tilt_corrected_dark_field(
        dataset : Dataset4dstem,  
        n_dark_field_segments : int = 32, 
        verbosity : int = 0,
        bright_field_mask_threshold : float = 0.3,
        upsample: Union[float, str] = "nyquist",
        return_snr: bool = False,
        snr_blur_sigma: float = 0.0,
        shift_method: str = "drizzle",
        drizzle_pixfrac: float = 1.0,
        drizzle_kde_sigma: float = 0.0,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Tilt-corrected dark-field reconstruction.

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
    """

    from scatterem.reconstruction.direct_ptychography import _resolve_upsample_int

    # tcDF uses ONE isotropic factor (max over the per-axis factors).
    upsample_int = np.max(_resolve_upsample_int(dataset, upsample, verbosity=verbosity))
    aberrations_array = dataset.meta.aberrations.array.detach().clone()
    device = dataset.device
    wavelength = dataset.meta.wavelength
    semiangle_cutoff = dataset.meta.semiconvergence_angle
    # set everything above and including C21 to 0
    # aberrations_array[0] = -200
    aberrations_array[3:] = 0
    # Polymorphic: the eager dataset computes this from its resident array,
    # the out-of-core dataset serves its streamed Pass-0 statistic.
    diff_mean = dataset.mean_cbed_tcdf.to(device)
    diff_mean = diff_mean / diff_mean.max()
    bright_field_mask = diff_mean > bright_field_mask_threshold
    if verbosity > 0:           
        fig, ax = vis.show_2d(bright_field_mask.float(), cbar=True, title="Virtual Bright Field Mask") 
    bright_field_inds = torch.argwhere(bright_field_mask)
    bright_field_mask = bright_field_mask
    bright_field_center = bright_field_inds.float().mean(0)
    dark_field_mask = ~bright_field_mask

    # Create circular dark field mask centered on bright_field_center
    y_coords, x_coords = torch.meshgrid(
        torch.arange(dark_field_mask.shape[0], device=dark_field_mask.device),
        torch.arange(dark_field_mask.shape[1], device=dark_field_mask.device),
        indexing="ij",
    )

    # Calculate radial distances from bright_field_center
    r = torch.sqrt((y_coords - bright_field_center[0]) ** 2 + (x_coords - bright_field_center[1]) ** 2)

    # Set dark field mask to include everything up to the edge of the array
    max_radius = min(
        dark_field_mask.shape[0] - bright_field_center[0],
        bright_field_center[0],
        dark_field_mask.shape[1] - bright_field_center[1],
        bright_field_center[1],
    )
    # Update dark_field_mask to be circular and centered on bright_field_center
    dark_field_mask = (r <= max_radius) & ~bright_field_mask
    # Create n_dark_field_segments azimuthal masks
    center = torch.tensor(
        [dark_field_mask.shape[0] // 2, dark_field_mask.shape[1] // 2],
        device=dark_field_mask.device,
    )
    
    angles = torch.atan2(y_coords - center[0], x_coords - center[1])
    angles = (angles + torch.pi) / (2 * torch.pi)  # Normalize to [0,1]
    segment_size = 1.0 / n_dark_field_segments 
    shifts = aberrations_to_image_shifts(
        aberrations_array=aberrations_array,
        rotation=torch.tensor([dataset.meta.rotation], device=device),
        sampling=dataset.dr,
        wavelength=wavelength,
        shape=dataset.shape[-2:],
    )
 
    inner_radius = semiangle_cutoff / wavelength * 2/3
    outer_radius= inner_radius + dataset.sampling[-1] * 1.3
    ny,nx = dataset.shape[-2:]
    sy,sx = dataset.dr
    k_x = torch.fft.fftfreq(nx,sx, device=device)
    k_y = torch.fft.fftfreq(ny,sy, device=device)

    # Build (ny, nx) grid to match detector masks (y, x) layout.
    k = torch.sqrt(k_y[:, None] ** 2 + k_x[None, :] ** 2)
    radial_mask = torch.fft.fftshift(((inner_radius <= k) & (k < outer_radius)))
 
    # ---- segment geometry (data-independent given diff_mean + aberrations) --
    specific_radius_masks_all = []
    seg_inds_list = []
    for i in range(n_dark_field_segments):
        segment_start = i * segment_size
        segment_end = (i + 1) * segment_size
        if segment_end < 1.0:
            # Normal case: segment doesn't cross the 0/2π boundary
            segment_mask = (angles >= segment_start) & (angles < segment_end)
        else:
            # The last segment: segment crosses 0/2π boundary
            # This includes angles from segment_start to 1.0 AND from 0.0 to (segment_end - 1.0)
            high_angles = (angles >= segment_start) & (angles <= 1.0)
            low_angles = angles < (segment_end - 1.0)
            segment_mask = high_angles | low_angles
        specific_radius_masks_all.append(segment_mask & radial_mask)
        seg_inds_list.append(torch.argwhere(segment_mask & dark_field_mask))

    # ---- per-segment images: per-scan-position mean over member px ----------
    # Polymorphic: eager gathers from the resident array; out-of-core runs ONE
    # streamed pass over the memmap fanned out to all segments.
    segment_image_list = dataset.gather_detector_group_means(seg_inds_list)

    # Even/odd segments form the two SSNR half-sets. Keep the REAL-space
    # segment images; the fourier path FFTs them below, the drizzle path
    # splats them directly (a Fourier phase-ramp shift of a single-count
    # image rings, a drizzle splat does not).
    dark_field_segment_images1 = segment_image_list[0::2]
    dark_field_segment_images2 = segment_image_list[1::2]

    vdf_stack1 = torch.stack(dark_field_segment_images1)  # (S1, H, W) real
    vdf_stack2 = torch.stack(dark_field_segment_images2)
    specific_radius_masks1 = specific_radius_masks_all[0::2]
    specific_radius_masks2 = specific_radius_masks_all[1::2]
 
    specific_radius_masks1 = torch.stack(specific_radius_masks1)
    specific_radius_masks2 = torch.stack(specific_radius_masks2)
 
    # Per-segment mean shift. A segment whose annular wedge contains NO detector
    # pixels (thin dark-field annulus at low convergence angle split into many
    # segments) gives an empty slice -> .mean() is NaN, which propagates through
    # exp(-i grad.q) into an all-NaN tcDF. Use 0 (no shift) for empty segments.
    def _seg_mean(col, masks):
        return torch.tensor(
            [shifts[m][:, col].mean() if bool(m.any()) else 0.0 for m in masks],
            device=device,
        )

    df_shifts_dx1 = _seg_mean(1, specific_radius_masks1)
    df_shifts_dy1 = _seg_mean(0, specific_radius_masks1)
    df_shifts_dx2 = _seg_mean(1, specific_radius_masks2)
    df_shifts_dy2 = _seg_mean(0, specific_radius_masks2)


    if shift_method == "drizzle":
        # Convert the per-segment parallax shift (Å) to HR pixels and drizzle the
        # two half-sets into their own accumulators (preserves the SSNR
        # half-split). Registration matches the fourier operator exactly:
        # shift_HRpx = fw_shift[Å] * upsample_int / (2*pi * scan_sampling[Å/px])
        # (the qvec's fftfreq(scan*U, sampling/U) makes the U cancel in the
        # frequency spacing, leaving the *scan* sampling as the denominator).
        scan_sampling = np.asarray(dataset.sampling[:2], dtype=np.float64)
        cy = float(upsample_int) / (2.0 * np.pi * scan_sampling[0])
        cx = float(upsample_int) / (2.0 * np.pi * scan_sampling[1])
        shifts_hr1 = torch.stack((df_shifts_dy1 * cy, df_shifts_dx1 * cx), -1)
        shifts_hr2 = torch.stack((df_shifts_dy2 * cy, df_shifts_dx2 * cx), -1)
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
    elif shift_method == "fourier":
        vdf_stack_fft1 = _fourier_upsample_stack(vdf_stack1, int(upsample_int))
        vdf_stack_fft2 = _fourier_upsample_stack(vdf_stack2, int(upsample_int))

        gpts = np.array(dataset.shape[:2]) * upsample_int
        sampling = dataset.sampling[:2] / upsample_int
        qxa = torch.fft.fftfreq(gpts[1], sampling[1], device=device, dtype=torch.float32)
        qya = torch.fft.fftfreq(gpts[0], sampling[0], device=device, dtype=torch.float32)
        qya = qya[:, None].broadcast_to(*gpts)
        qxa = qxa[None, :].broadcast_to(*gpts)
        qvec = torch.stack((qya, qxa), 0)

        grad_k_df1 = torch.stack((df_shifts_dy1, df_shifts_dx1), -1)
        grad_kq_df1 = torch.einsum("na,amp->nmp", grad_k_df1, qvec)
        operator_df1 = torch.exp(-1j * grad_kq_df1)

        grad_k_df2 = torch.stack((df_shifts_dy2, df_shifts_dx2), -1)
        grad_kq_df2 = torch.einsum("na,amp->nmp", grad_k_df2, qvec)
        operator_df2 = torch.exp(-1j * grad_kq_df2)

        # Value-preserving zero-pad normalization: torch's backward-norm ifft2
        # divides by the U^2-larger padded grid, so multiply by upsample_int**2
        # (not a single power). This makes the fourier path preserve the per-pixel
        # value of a flat/DC region -- matching the drizzle path's hit-normalized
        # (accum/hits) scale, so the two shift methods are absolute-scale identical.
        norm = float(upsample_int) ** 2
        tcDF1 = torch.fft.ifft2(vdf_stack_fft1 * operator_df1 * norm).real.mean(0)
        tcDF2 = torch.fft.ifft2(vdf_stack_fft2 * operator_df2 * norm).real.mean(0)
    else:
        raise ValueError(
            f"Invalid shift_method: {shift_method!r} (expected 'fourier' or 'drizzle')"
        )

    tcDF = tcDF2 + tcDF1

    if return_snr:        
        sampling_tcdf = tuple((dataset.sampling[:2] / upsample_int).tolist())
 
        ssnr_tcdf_1d, q_rad, bin_idx = compute_ssnr_from_halfset_images(tcDF1, tcDF2, 
            sampling_tcdf, verbosity=verbosity, gaussian_sigma=snr_blur_sigma)

        # Cut above probe-support limit
        k_max = 1 * dataset.meta.semiconvergence_angle / dataset.meta.wavelength
        ssnr_tcdf_1d[q_rad > 1.95 * k_max] = 0
        # Raw half-split power ratio (dose- and object-bearing). NOT divided by
        # sqrt(fluence): both fusion channels must share this common reference so
        # the Eq. 39 Wiener crossover is physical (see FF-STEM SSNR recalibration).
        ssnr_tcdf = ssnr_tcdf_1d[bin_idx].reshape(tcDF1.shape)
        if verbosity > 1:
            NN = ssnr_tcdf_1d.shape[0]
            fig = plt.figure()
            plt.plot(q_rad.cpu().numpy()[:NN//2], ssnr_tcdf_1d.cpu().numpy()[:NN//2], label="SSNRq_blur radial avg")
            plt.xlabel(f"Spatial frequency |q| (1/Å)")
            plt.ylabel("SSNR_tcDF(|q|)")
            plt.grid(True) 
            plt.tight_layout()
        return tcDF, ssnr_tcdf
     
    return tcDF, None
 

def tilt_corrected_dark_field_depth_section(
    dataset : Dataset4dstem,
    depth_angstroms=torch.tensor, 
    bright_field_mask_threshold : float = 0.3,
    return_device : torch.device = torch.device("cpu"),
    upsample : Union[float, str] = "nyquist",    
    n_dark_field_segments : int = 32,
    verbosity : int = 0,
) -> torch.Tensor:
    from scatterem.reconstruction.direct_ptychography import _resolve_upsample_int

    upsample_int = _resolve_upsample_int(dataset, upsample, verbosity=verbosity)
    tcDF_depth_section = torch.zeros(len(depth_angstroms), dataset.shape[0] * upsample_int[0], dataset.shape[1] * upsample_int[1], device=return_device)
    aberrations_array = dataset.meta.aberrations.array.detach().clone()        
    i = 0
    for depth in tqdm(depth_angstroms, desc="Assembling tcDF depth section"): 
        dataset.meta.aberrations.array[:] = aberrations_array 
        dataset.meta.aberrations.array[0] += depth
        dataset.meta.aberrations.array[3:] = 0
        tcDF_depth_section[i] = dataset.tilt_corrected_dark_field(
                n_dark_field_segments  = n_dark_field_segments, 
                verbosity  = verbosity,
                bright_field_mask_threshold = bright_field_mask_threshold,
                upsample=upsample)[0].to(return_device)
 
        i += 1
    dataset.meta.aberrations.array[:] = aberrations_array
    return tcDF_depth_section