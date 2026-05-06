import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
 
import scatterem2.vis as vis
from scatterem2.utils.data.datasets import Dataset4dstem, DatasetVirtualBrightField4dstem
from typing import Union
from scatterem2.utils.transfer import aberrations_to_image_shifts
 
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
        ) -> tuple[torch.Tensor, torch.Tensor]:

    if upsample == "nyquist":
        scan_sampling = np.array(dataset.sampling[:2])
        nyquist_sampling = dataset.meta.wavelength / (4 * dataset.meta.semiconvergence_angle)
        upsample_float = scan_sampling / nyquist_sampling
        upsample_int = np.ceil(upsample_float).astype(np.uint32)
        
        if upsample_float.any() < 1:
            upsample_float[upsample_float < 1] = 1. 
        if verbosity > 0:
            print(f"scan_sampling = {scan_sampling}")
            print(f"nyquist_sampling = {nyquist_sampling}")           
            print(f"Upsampling to Nyquist, upsample factor: {upsample_int}")
            
    elif isinstance(upsample, str):   
        raise ValueError(f"Invalid upsample: {upsample}")
    elif not isinstance(upsample, float):
        raise ValueError(f"Invalid upsample: {upsample}")
    elif isinstance(upsample, float) or isinstance(upsample, int):
        upsample_float = np.array([upsample, upsample])
        upsample_int = np.ceil(upsample_float).astype(np.uint32)
        if upsample_float.any() < 1:
            upsample_float[upsample_float < 1] = 1. 
        if verbosity > 0:
            print(f"Upsampling to Nyquist, upsample factor: {upsample_int}")
    upsample_int = np.max(upsample_int)
    aberrations_array = dataset.meta.aberrations.array.detach().clone()
    device = dataset.device
    wavelength = dataset.meta.wavelength
    semiangle_cutoff = dataset.meta.semiconvergence_angle
    # set everything above and including C21 to 0

    aberrations_array[3:] = 0
    n = dataset.array.shape[0]
    diff_mean = dataset.array[:n,:n].mean((0,1))
    diff_mean /= diff_mean.max()
    bright_field_mask = diff_mean > bright_field_mask_threshold
    if verbosity > 0:           
        fig, ax = vis.show_2d(bright_field_mask.float(), cbar=True, title="Virtual Bright Field Mask") 
    bright_field_inds = torch.argwhere(bright_field_mask)
    bright_field_mask = bright_field_mask
    bright_field_center = bright_field_inds.float().mean(0)
    dark_field_mask = ~bright_field_mask

    bright_field_intensity = dataset.array[:,:,bright_field_mask].sum(-1)
    dark_field_intensity = dataset.array[:,:,dark_field_mask].sum(-1)
    max_intensity_per_probe = dataset.array.sum((-2,-1)).max().item()
    real_dark_field_intensity = max_intensity_per_probe - bright_field_intensity
    dark_field_scaling = real_dark_field_intensity / (dark_field_intensity + 1e-12)

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

    k = torch.sqrt(k_x[:,None]**2 + k_y[None,:]**2)
    radial_mask = torch.fft.fftshift(((inner_radius <= k) & (k < outer_radius)))
 
    specific_radius_masks1 = []
    specific_radius_masks2 = []
 
    dark_field_segment_images1 = []
    dark_field_segment_images2 = []
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
        specific_radius_mask = segment_mask & radial_mask
        full_field_segments_i = segment_mask & dark_field_mask   
        dark_field_segment_inds = torch.argwhere(full_field_segments_i) 
        dark_field_segment_image = dataset.array[
            :, :, dark_field_segment_inds[:, 0], dark_field_segment_inds[:, 1]
        ].mean(-1) #* dark_field_scaling 
    
        if i % 2 == 0:
            specific_radius_masks1.append(specific_radius_mask)
        else:
            specific_radius_masks2.append(specific_radius_mask)

        if upsample_int > 1:        
            img_fft = torch.fft.fft2(dark_field_segment_image, dim=(-2, -1))
            H, W = img_fft.shape[-2], img_fft.shape[-1]
            new_H, new_W = int(H * upsample_int), int(W * upsample_int)
            # Fourier upsample: zero-pad in Fourier domain
            
            img_fft_padded = torch.zeros((new_H, new_W), dtype=img_fft.dtype, device=img_fft.device)            
            img_fft_padded[new_H-H//2:, new_W-W//2:] = img_fft[H//2:, W//2:]
            img_fft_padded[new_H-H//2:, :W-W//2] = img_fft[H//2:, :W//2]
            img_fft_padded[:H-H//2, new_W-W//2:] = img_fft[:H//2, W//2:]
            img_fft_padded[:H-H//2, :W-W//2] = img_fft[:H//2, :W//2]       
   
            if i % 2 == 0:
                dark_field_segment_images1.append(img_fft_padded)
            else:
                dark_field_segment_images2.append(img_fft_padded)
        else:
            img_fft = torch.fft.fft2(dark_field_segment_image, dim=(-2, -1))
            if i % 2 == 0:
                dark_field_segment_images1.append(img_fft)
            else:
                dark_field_segment_images2.append(img_fft)
 
    vdf_stack_fft1 = torch.stack(dark_field_segment_images1)
    vdf_stack_fft2 = torch.stack(dark_field_segment_images2)
 
    specific_radius_masks1 = torch.stack(specific_radius_masks1)
    specific_radius_masks2 = torch.stack(specific_radius_masks2)
 
    df_shifts_dx1 = torch.tensor(
        [
            shifts[mask][:, 1].mean()
            for mask in specific_radius_masks1
        ], device=device
    )
    df_shifts_dy1 = torch.tensor(
        [
            shifts[mask][:, 0].mean()
            for mask in specific_radius_masks1
        ], device=device
    )
    df_shifts_dx2 = torch.tensor(
        [
            shifts[mask][:, 1].mean()
            for mask in specific_radius_masks2
        ], device=device
    )
    df_shifts_dy2 = torch.tensor(
        [
            shifts[mask][:, 0].mean()
            for mask in specific_radius_masks2
        ], device=device
    )


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

    tcDF1 = torch.fft.ifft2(vdf_stack_fft1 * operator_df1 * upsample_int).real.mean(0)
    tcDF2 = torch.fft.ifft2(vdf_stack_fft2 * operator_df2 * upsample_int).real.mean(0)

    tcDF = tcDF2 + tcDF1  

    if return_snr:        
        sampling_tcdf = tuple((dataset.sampling[:2] / upsample_int).tolist())
 
        ssnr_tcdf_1d, q_rad, bin_idx = compute_ssnr_from_halfset_images(tcDF1, tcDF2, 
            sampling_tcdf, verbosity=verbosity, gaussian_sigma=snr_blur_sigma)

        # Cut above probe-support limit
        k_max = 1 * dataset.meta.semiconvergence_angle / dataset.meta.wavelength
        # ssnr_tcdf_1d = ssnr_tcdf_1d[:dataset.shape[0]//2]
        ssnr_tcdf_1d[q_rad > 1.95 * k_max] = 0
        ssnr_tcdf = ssnr_tcdf_1d[bin_idx].reshape(tcDF1.shape)  / np.sqrt(dataset.fluence_per_probe)
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
    if upsample == "nyquist":
        scan_sampling = np.array(dataset.sampling[:2])
        nyquist_sampling = dataset.meta.wavelength / (4 * dataset.meta.semiconvergence_angle)
        upsample_float = scan_sampling / nyquist_sampling
        upsample_int = np.ceil(upsample_float).astype(np.uint32)
        if upsample_float.any() < 1:
            upsample_float[upsample_float < 1] = 1. 
        if verbosity > 0:
            print(f"scan_sampling = {scan_sampling}")
            print(f"nyquist_sampling = {nyquist_sampling}")           
            print(f"Upsampling to Nyquist, upsample factor: {upsample_int}")
            
    elif isinstance(upsample, str):   
        raise ValueError(f"Invalid upsample: {upsample}")
    elif not isinstance(upsample, float):
        raise ValueError(f"Invalid upsample: {upsample}")
    elif isinstance(upsample, float):
        upsample_float = np.array([upsample, upsample])
        upsample_int = np.ceil(upsample_float).astype(np.uint32)
        if upsample_float.any() < 1:
            upsample_float[upsample_float < 1] = 1. 
        if verbosity > 0:
            print(f"Upsampling to Nyquist, upsample factor: {upsample_int}")
    tcDF_depth_section = torch.zeros(len(depth_angstroms), dataset.array.shape[0] * upsample_int[0], dataset.array.shape[1] * upsample_int[1], device=return_device)    
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
                upsample=upsample).to(return_device)
 
        i += 1
    dataset.meta.aberrations.array[:] = aberrations_array
    return tcDF_depth_section
