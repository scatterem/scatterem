from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import griddata
from torchmetrics.image import TotalVariation
from tqdm import tqdm

import scatterem2.vis as vis
from scatterem2.utils.data.aberrations import Aberrations
from scatterem2.utils.data.datasets import (
    Dataset4dstem,
    DatasetVirtualBrightField4dstem,
)
from scatterem2.utils.registration import relative_shifts
from scatterem2.utils.stem import (
    beamlet_samples,
    fftfreq2,
    natural_neighbor_weights,
)
from scatterem2.utils.transfer import cartesian2polar, aberrations_to_image_shifts
from scatterem2.nn.functional.ptychography import (
    CorrectAberrations,
    correct_aberrations_inplace,
    phase_contrast_transfer_function as _phase_contrast_transfer_function,
)
from tqdm.auto import tqdm
from scatterem2.utils.transfer import double_and_triple_pixel_counts
from scatterem2.vis.custom_normalizations import NormalizationConfig

def plot_bright_field_shifts(
    query_points,
    shift_values,
    sampling,
    wavelength,
    opt_rotation,
    arrow_scale,
    suptitle=None,
):
    """
    Plot bright field shifts before and after rotation correction.
    Args:
        query_points: torch.Tensor - query points (2D: N x 2), order by radius in the bright field, increasing order.
        shift_values: torch.Tensor - shift values (2D: N x 2), order by radius in the bright field, increasing order.
        sampling: torch.Tensor - sampling (2D: 2)
        wavelength: float - wavelength in Angstroms
        opt_rotation: float - rotation in degrees. The default is 0.
        arrow_scale: float - arrow scale. The default is 25e-2.

    Returns:
        None
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    if suptitle is not None:
        fig.suptitle(suptitle)
    # ax2 = ax[0].twinx()  # Create right axis
    ax3 = ax[0].twiny()  # Create top axis
    shift_values = shift_values.cpu().numpy()
    # Draw arrows for each position
    q_query_points = query_points.cpu().numpy() * sampling[0].cpu().numpy()

    ax[0].quiver(
        q_query_points[:, 1],
        q_query_points[:, 0],
        shift_values[:, 1] * arrow_scale,
        shift_values[:, 0] * arrow_scale,
        fc="r",
        ec="r",
        label="Shift vectors",
    )
    ax[0].scatter(
        q_query_points[:, 1],
        q_query_points[:, 0],
        color="b",
        s=1,
        marker="x",
        label="Query points",
    )

    f = 1.2
    # Calculate k ranges
    k_min_x = q_query_points[:, 1].min()
    k_max_x = q_query_points[:, 1].max()
    k_min_y = q_query_points[:, 0].min()
    k_max_y = q_query_points[:, 0].max()

    # Calculate alpha ranges (alpha = k*lambda)
    alpha_min_x = k_min_x * wavelength * 1e3  # Convert to mrad
    alpha_max_x = k_max_x * wavelength * 1e3
    alpha_min_y = k_min_y * wavelength * 1e3
    alpha_max_y = k_max_y * wavelength * 1e3

    # Create 5 evenly spaced ticks centered at zero
    k_ticks_x = np.linspace(k_min_x, k_max_x, 5)
    k_ticks_y = np.linspace(k_min_y, k_max_y, 5)
    alpha_ticks_x = np.linspace(alpha_min_x, alpha_max_x, 5)
    alpha_ticks_y = np.linspace(alpha_min_y, alpha_max_y, 5)

    # Set limits and ticks
    ax[0].set_ylim(k_min_y * f, k_max_y * f)
    ax[0].set_xlim(k_min_x * f, k_max_x * f)
    # ax2.set_ylim(alpha_min_y * f, alpha_max_y * f)
    ax3.set_xlim(alpha_min_x * f, alpha_max_x * f)

    ax[0].set_title("Bright Field Shifts")
    ax[0].set_ylabel(r"$k_y$ [$\mathrm{\AA^{-1}}$]")
    ax[0].set_xlabel(r"$k_x$ [$\mathrm{\AA^{-1}}$]")

    ax[0].set_yticks(k_ticks_y)
    ax[0].set_xticks(k_ticks_x)
    # ax2.set_yticks(alpha_ticks_y)
    ax3.set_xticks(alpha_ticks_x)

    # ax2.set_yticklabels([f'{a:.1f}' for a in alpha_ticks_y])
    ax3.set_xticklabels([f"{a:.1f}" for a in alpha_ticks_x])

    # ax2.set_ylabel(r"$\alpha_x$ [mrad]")
    ax3.set_xlabel(r"$\alpha_x$ [mrad]")

    ax[0].legend()

    # Create twin axes
    ax2 = ax[1].twinx()  # Create right axis
    ax3 = ax[1].twiny()  # Create top axis

    # Calculate rotation matrix
    angle_rad = torch.deg2rad(-opt_rotation).item()
    rot_matrix = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )

    # Apply rotation to query points
    q_query_points_rot = np.dot(q_query_points, rot_matrix)

    ax[1].quiver(
        q_query_points_rot[:, 1],
        q_query_points_rot[:, 0],
        shift_values[:, 1] * arrow_scale,
        shift_values[:, 0] * arrow_scale,
        fc="r",
        ec="r",
        label=f"Shift vectors (bin {bin})",
    )
    ax[1].scatter(
        q_query_points_rot[:, 1],
        q_query_points_rot[:, 0],
        color="b",
        s=1,
        marker="x",
        label="Query points",
    )

    # Calculate alpha ranges (alpha = k*lambda)
    alpha_min_x = k_min_x * wavelength * 1e3  # Convert to mrad
    alpha_max_x = k_max_x * wavelength * 1e3
    alpha_min_y = k_min_y * wavelength * 1e3
    alpha_max_y = k_max_y * wavelength * 1e3

    # Create 5 evenly spaced ticks centered at zero
    k_ticks_x = np.linspace(k_min_x, k_max_x, 5)
    k_ticks_y = np.linspace(k_min_y, k_max_y, 5)
    alpha_ticks_x = np.linspace(alpha_min_x, alpha_max_x, 5)
    alpha_ticks_y = np.linspace(alpha_min_y, alpha_max_y, 5)

    # Set limits and ticks
    ax[1].set_ylim(k_min_y * f, k_max_y * f)
    ax[1].set_xlim(k_min_x * f, k_max_x * f)
    ax2.set_ylim(alpha_min_y * f, alpha_max_y * f)
    ax3.set_xlim(alpha_min_x * f, alpha_max_x * f)

    ax[1].set_title("Bright Field Shifts (Rotated)")
    # ax[1].set_ylabel(r"$k_y$ [$\mathrm{\AA^{-1}}$]")
    ax[1].set_xlabel(r"$k_x$ [$\mathrm{\AA^{-1}}$]")

    # ax[1].set_yticks(k_ticks_y)
    ax[1].set_yticks([])
    ax[1].set_yticklabels([])
    ax[1].set_xticks(k_ticks_x)
    ax2.set_yticks(alpha_ticks_y)
    ax3.set_xticks(alpha_ticks_x)

    ax2.set_yticklabels([f"{a:.1f}" for a in alpha_ticks_y])
    ax3.set_xticklabels([f"{a:.1f}" for a in alpha_ticks_x])

    ax2.set_ylabel(r"$\alpha_y$ [mrad]")
    ax3.set_xlabel(r"$\alpha_x$ [mrad]")

    # ax[1].legend()
    plt.tight_layout()
    plt.show()

def fit_aberrations_and_rotation_to_bright_field_shifts(
    target_shifts: torch.Tensor,
    k_sampling: np.ndarray,
    wavelength: float,
    bright_field_mask: torch.Tensor,
    include_in_fit_mask: torch.Tensor | None = None,
    rotation_init: torch.Tensor | None = None,
    rotation_requires_grad: bool = True,
    max_iter: int = 250,
    lr: float = 1e-1,
    verbosity: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fit the aberrations and rotation to the bright field shifts.

    Args:
        target_shifts: torch.Tensor - target bright field shifts (2D: N x 2), order by radius in the bright field, increasing order.
        k_sampling: torch.Tensor - k-space sampling (2D: 2)
        wavelength: float - wavelength in Angstroms
        bright_field_mask: torch.Tensor - bright field mask (2D: H x W)
        include_in_fit_mask: torch.Tensor | None - mask for the bright field shifts to include in the fit. If None, all bright field shifts are included.
        rotation_init: torch.Tensor | None - initial rotation in degrees. If None, the rotation is initialized to 0.
        rotation_requires_grad: bool - whether to require the rotation to be optimized. The default is True.
        max_iter: int - maximum number of iterations. The default is 50.
        lr: float - learning rate. The default is 1.
        verbosity: int - verbosity level. The default is 0.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: aberrations and rotation. The aberrations are in units of Angstroms. The rotation is in degrees.
    """

    # Check that all tensors are on same device and dtype
    device = target_shifts.device
    target_shifts = target_shifts.to(torch.float32)
 
    if bright_field_mask.device != device:
        bright_field_mask = bright_field_mask.to(device)

    bright_field_inds = torch.argwhere(bright_field_mask)
  
    if include_in_fit_mask is not None:
        if include_in_fit_mask.device != device:
            include_in_fit_mask = include_in_fit_mask.to(device)
    else:
        include_in_fit_mask = torch.ones(
            bright_field_inds.shape, device=device, dtype=torch.bool
        )

    # Set up optimization
    aberrations = torch.zeros(
        12, device=device, requires_grad=True, dtype=torch.float32
    )
    rotation_opt = (
        torch.zeros(1, device=device, dtype=torch.float32)
        if rotation_init is None
        else rotation_init.to(device).to(torch.float32)
    )

    def zero_grad_hook(grad):
        # g_before = grad.detach().cpu().numpy()[0]
        if grad is not None:
            grad[3:] = 0
        # g_after = grad.detach().cpu().numpy()[0]
        # print(f"C10 Gradients before masking: {g_before}, after masking: {g_after}")
        return grad

    # Register the hook
    aberrations.register_hook(zero_grad_hook)
    opt_params = [
        aberrations,
    ]
    if rotation_requires_grad:
        rotation_opt.requires_grad = True
        opt_params.append(rotation_opt)
    optimizer = torch.optim.LBFGS(
        opt_params,
        lr=lr,
        max_iter=max_iter,
        max_eval=None,
        tolerance_grad=1e-10,
        tolerance_change=1e-10,
        history_size=50,
        line_search_fn="strong_wolfe",
    )
    
    
    def closure():
        optimizer.zero_grad()
        predicted_shifts = aberrations_to_image_shifts(
            aberrations, rotation_opt, k_sampling, wavelength, bright_field_mask.shape
        )
        predicted_shifts = predicted_shifts[bright_field_mask]
        loss = torch.nn.functional.huber_loss(
            predicted_shifts[include_in_fit_mask]  , target_shifts[include_in_fit_mask]
        )
        loss.backward()
        with torch.no_grad():
            print("grad C10,C12,phi12:",
                float(aberrations.grad[0].item()),
                float(aberrations.grad[1].item()),
                float(aberrations.grad[2].item()))
        return loss

    if verbosity > 0:
        print("Optimizing aberrations...")
    loss = optimizer.step(closure)

    with torch.no_grad():
        final_shifts = aberrations_to_image_shifts(
            aberrations, rotation_opt, k_sampling, wavelength, bright_field_mask.shape  
        )
        final_shifts = final_shifts[bright_field_mask]
        final_shifts_np = final_shifts.cpu().numpy()

    if verbosity > 0:
        print("\nFinal shifts max:", final_shifts_np.max(0))
        print(f"Initial rotation: {rotation_init.detach().item():.2f} deg")
        print(f"Final   rotation: {rotation_opt.detach().item():.2f} deg")
        print(f"Final aberrations: {aberrations.detach().cpu().numpy()}")
    return aberrations.detach(), rotation_opt.detach()

def aberrations_and_rotation_from_bright_field_shifts(
    dataset: DatasetVirtualBrightField4dstem,
    fit_rotation: bool = True,
    target_percentage_nonzero_pixels: float = 0.75,
    n_batches: int = 25,
    registration_upsample_factor: int = 10,
    lowpass_fwhm_bright_field: Optional[float] = None,
    bin_factors: tuple[int, ...] = (2, 1), 
    verbosity: int = 0,
    update_dataset: bool = True,
    n_center_indices: int = 25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    First determine the bright field shifts, then fit aberrations and rotation to the bright field shifts.
    Args:
        dataset: Dataset4dstem or DatasetVirtualBrightField4dstem object containing the diffraction pattern.
        fit_rotation: Whether to fit the rotation. The default is True.
        bright_field_mask_threshold: Threshold to determine the bright field pixels, relative to max=1. The default is 0.5.
        target_percentage_nonzero_pixels: Target percentage of nonzero pixels for the bright field. The default is 0.75.
        n_batches: Number of batches for the bright field shifts. The default is 25.
        registration_upsample_factor: Upsampling factor for the registration. The default is 10.
        lowpass_fwhm_bright_field: FWHM of the lowpass filter for the bright field. The default is None, which means no lowpass filter.
        bin_factors: Bin factors for the bright field. The default is (2, 1). 
        verbosity: Verbosity level. The default is 0.
        update_dataset: Whether to update the dataset. The default is True.
        n_center_indices: Number of center indices to use for the bright field shifts. The default is 25.
    Returns:
        tuple[torch.Tensor, torch.Tensor]: Aberrations, rotation.

    Raises:
        ValueError: If the dataset is not a valid Dataset4dstem object.
    """

    device = dataset.device 
    wavelength = dataset.meta.wavelength
    vBF = dataset
    bright_field = vBF.normalized_bright_field
    if target_percentage_nonzero_pixels is not None:
        percentage_nonzero = (bright_field != 0).float().mean()
        # Calculate minimum bin size needed to get percentage_nonzero >= 75%
        current_percentage = percentage_nonzero.item()    
        bin_factors_min = int(np.ceil(np.sqrt(target_percentage_nonzero_pixels / current_percentage)))
     
    G = vBF.G
    bright_field_center_image = torch.mean(
        bright_field[:, :, :n_center_indices], dim=-1
    )
    del bright_field

    G_ref = torch.fft.fft2(bright_field_center_image)
 
    sampling = torch.as_tensor(dataset.meta.sampling[:2], device=device, dtype=torch.float32)
    reciprocal_sampling = torch.as_tensor(
    dataset.parent_dataset.meta.sampling[-2:], device=device) 
    rot_tensor = torch.tensor(
        [dataset.meta.rotation], device=device, dtype=torch.float32
    )
    global_shifts = torch.zeros((len(vBF.bright_field_inds_centered), 2), device=device, dtype=torch.float32)
    kx = torch.fft.fftfreq(G.shape[1], device=device).reshape(1, -1)[:, :, None]
    ky = torch.fft.fftfreq(G.shape[0], device=device).reshape(-1, 1)[:, :, None]
    order = vBF.bright_field_inds_radial_order.to(device=device)
    desc = "Self-calibration (reference-based)"
    pbar = tqdm(bin_factors, desc=desc, disable=not verbosity)
    for bin in pbar:
        measured_incremental_shifts_px = bright_field_shifts(
            G_ref,
            G,
            vBF.bright_field_inds_centered,
            vBF.bright_field_inds_centered_ordered_by_radius,
            bin,
            n_batches,
            registration_upsample_factor,
            verbosity,
            lowpass_fwhm_bright_field,
        )
        measured_incremental_shifts_px -= torch.mean(measured_incremental_shifts_px, axis=0)  
        inc_shifts_angstroms = measured_incremental_shifts_px * sampling 
        pbar.set_postfix({'Res. Shift [A]': f"{inc_shifts_angstroms.max().item():.2f}"})
        global_shifts += inc_shifts_angstroms
        print(f'the bright field mask shape: {vBF.bright_field_mask.shape}')
        opt_aberrations, opt_rotation = fit_aberrations_and_rotation_to_bright_field_shifts(
            global_shifts,
            reciprocal_sampling,
            wavelength,
            vBF.bright_field_mask,
            rotation_init=rot_tensor,
            rotation_requires_grad=fit_rotation,
            verbosity=1,
        )

        with torch.no_grad():
            pred_shifts_full = aberrations_to_image_shifts(
                opt_aberrations, opt_rotation, reciprocal_sampling, wavelength, vBF.bright_field_mask.shape
            )
            pred_shifts = pred_shifts_full[vBF.bright_field_mask]

        fitted_global_shifts = aberrations_to_image_shifts(
            opt_aberrations, opt_rotation, reciprocal_sampling, 
            wavelength, vBF.bright_field_mask.shape
        )
        fitted_global_shifts = fitted_global_shifts[vBF.bright_field_mask]
        model_incremental_shifts = fitted_global_shifts - global_shifts
        global_shifts = fitted_global_shifts

        dx = measured_incremental_shifts_px[order, 1][None, None, :]
        dy = measured_incremental_shifts_px[order, 0][None, None, :]
        phase_ramp = torch.exp(-2j * np.pi * (dx * kx + dy * ky))
        
        G *= phase_ramp 
        G_ref = torch.mean(G[..., :n_center_indices], dim=-1)
 

    if verbosity > 0:
        print(
            f"Interpolated shifts for {len(model_incremental_shifts)} bright field indices"
        )
        cc_shifts_x = torch.zeros_like(vBF.diffraction_pattern_mean_normalized)
        cc_shifts_y = torch.zeros_like(vBF.diffraction_pattern_mean_normalized)
        cc_shifts_x[vBF.bright_field_inds[:, 0], vBF.bright_field_inds[:, 1]] = (
            global_shifts[:, 1]
        )
        cc_shifts_y[vBF.bright_field_inds[:, 0], vBF.bright_field_inds[:, 1]] = (
            global_shifts[:, 0]
        )
 
        fitted_shifts_x = torch.zeros_like(vBF.diffraction_pattern_mean_normalized)
        fitted_shifts_y = torch.zeros_like(vBF.diffraction_pattern_mean_normalized)
        fitted_shifts_x[vBF.bright_field_inds[:, 0], vBF.bright_field_inds[:, 1]] = (
            fitted_global_shifts[:, 1]
        )
        fitted_shifts_y[vBF.bright_field_inds[:, 0], vBF.bright_field_inds[:, 1]] = (
            fitted_global_shifts[:, 0]
        )
        titles = [
            "CC Shift Values X",
            "CC Shift Values Y",
            "Fitted Shifts X",
            "Fitted Shifts Y",
        ]
        plots = [cc_shifts_x, cc_shifts_y, fitted_shifts_x, fitted_shifts_y]
        vis.show_2d(plots, cbar=True, title=titles, cmap="RdBu")

    if update_dataset:
        vBF.meta.aberrations = Aberrations(array=opt_aberrations)
        vBF.meta.rotation = opt_rotation.item()
        vBF.meta.defocus_guess = -opt_aberrations[0].item()

    return opt_aberrations, opt_rotation

  


def bright_field_shifts(
    G_ref: torch.Tensor,
    G_moving: torch.Tensor,
    bright_field_inds_centered: torch.Tensor,
    bright_field_inds_centered_ordered_by_radius: torch.Tensor,
    bin: int,
    n_batches: int = 16,
    registration_upsample_factor: int = 10,
    verbosity: int = 1,
    fwhm_lowpass_bf=None,
) -> torch.Tensor:
    """
    Calculate the bright field shifts.
    The bright field shifts are calculated by performing a registration between the reference and moving images.
    The registration is performed by performing an upsampled cross-correlation between the reference and moving images.
    The bright field shifts are then returned. If binning is used, the bright field shifts are interpolated to all bright field indices.

    Args:
        G_ref: torch.Tensor - reference image (2D: H x W)
        G_moving: torch.Tensor - moving image (3D: N x H x W)
        bright_field_inds_centered: torch.Tensor - bright field indices centered (2D: N x 2)
        bright_field_inds_centered_ordered_by_radius: torch.Tensor - bright field indices centered ordered by radius (2D: N x 2)
        bin: int - bin size. If None, no binning is performed.
        n_batches: int - number of batches. The default is 16.
        registration_upsample_factor: int - registration upsample factor. The default is 10.
        verbosity: int - verbosity. The default is 1.
        fwhm_lowpass_bf: float - FWHM of the lowpass filter. If None, no lowpass filter is used.

    Returns:
        torch.Tensor - bright field shifts (2D: N x 2), order by radius in the bright field, increasing order.
    """
    device = G_moving.device
    kx = torch.fft.fftfreq(G_moving.shape[1], device=device).reshape(1, -1)
    ky = torch.fft.fftfreq(G_moving.shape[0], device=device).reshape(-1, 1)

    if fwhm_lowpass_bf is not None:
        sigma = fwhm_lowpass_bf / (2 * (2 * np.log(2)) ** 0.5)
        kx2 = kx**2  # shape (1, N)
        ky2 = ky**2  # shape (M, 1)
        k2 = kx2 + ky2  # shape (M, N)
        gaussian_filter = torch.exp(-2 * (np.pi**2) * (sigma**2) * k2)

    bright_field_inds_order = torch.argsort(
        torch.sum(bright_field_inds_centered**2, dim=1)
    )
    bright_field_inds_ordered_by_radius_binned = torch.ceil(
        bright_field_inds_centered_ordered_by_radius / bin
    ).int()
    bright_field_inds_binned = torch.ceil(bright_field_inds_centered / bin).int()
    bright_field_unique_inds_binned = torch.unique(bright_field_inds_binned, dim=0)
    bright_field_unique_inds_binned_order = torch.argsort(
        torch.sum(bright_field_unique_inds_binned**2, dim=1)
    )
    bright_field_unique_inds_binned_ordered = bright_field_unique_inds_binned[
        bright_field_unique_inds_binned_order
    ]

    # Calculate batch_size based on n_batches
    total_elements = len(bright_field_unique_inds_binned_ordered)
    batch_size = max(1, total_elements // n_batches)

    if verbosity > 0:
        bf_ref_before = torch.fft.ifft2(G_ref, dim=(0, 1), norm="ortho").real

    bright_field_shifts = torch.zeros(
        len(bright_field_unique_inds_binned_ordered), 2, device=device
    )
    N = len(bright_field_unique_inds_binned_ordered)
    desc = "Registering bright field images"
    pbar = tqdm(range(0, N, batch_size), desc=desc, disable=not verbosity)
    for i in pbar:
        # Determine batch indices and actual batch size upfront
        if bin > 1:
            batch_inds = bright_field_unique_inds_binned_ordered[i : i + batch_size]
        else:
            remaining_elements = len(bright_field_unique_inds_binned_ordered) - i
            actual_batch_size = min(batch_size, remaining_elements)
            batch_inds = np.arange(i, i + actual_batch_size)
            max_indices = G_moving.shape[-1] - 1
            batch_inds = np.clip(batch_inds, 0, max_indices)

            if batch_inds[-1] >= G_moving.shape[-1]:
                batch_inds = batch_inds[batch_inds < G_moving.shape[-1]]
                if len(batch_inds) == 0:
                    print(f"Warning: No valid indices for batch starting at {i}")
                    continue

        # Determine actual batch size and allocate G_moving appropriately
        actual_batch_size = len(batch_inds)
        # print(f"{i:02d} actual_batch_size = {actual_batch_size}")

        if bin > 1:
            # Allocate G_moving with the proper size for binning
            G_moving_batch = torch.zeros(
                actual_batch_size,
                G_moving.shape[0],
                G_moving.shape[1],
                device=device,
                dtype=G_moving.dtype,
            )   
            # perform the binning in a loop
            for j, ri in enumerate(batch_inds):
                take = (
                    bright_field_inds_ordered_by_radius_binned[:, 0] == ri[0]
                ).__and__(bright_field_inds_ordered_by_radius_binned[:, 1] == ri[1])
                G_moving_batch[j] = torch.mean(
                    G_moving[..., torch.where(take)[0]], dim=-1
                )

            if fwhm_lowpass_bf is not None:
                G_moving_batch = G_moving_batch * gaussian_filter
        else:
            # print(f"Taking indices: {bright_field_inds_centered_ordered_by_radius[batch_inds]}")
            G_moving_batch = torch.permute(G_moving[..., batch_inds], (2, 0, 1))

        xy_shift = relative_shifts(
            G_ref, G_moving_batch, upsample_factor=registration_upsample_factor
        )
        # Slice xy_shift to match the actual batch size we need
        xy_shift = xy_shift[:actual_batch_size]
        bright_field_shifts[i : i + actual_batch_size] = xy_shift

        dx = xy_shift[:, 1]
        dy = xy_shift[:, 0]
        phase_ramp = torch.exp(
            -1j
            * 2
            * np.pi
            * (dx[:, None, None] * kx[None, :, :] + dy[:, None, None] * ky[None, :, :])
        )
        G_ref = G_ref * i / (i + batch_size) + torch.mean(
            G_moving_batch * phase_ramp, dim=0
        ) * batch_size / (i + batch_size)

    if verbosity > 0:
        bf_ref_after = torch.fft.ifft2(G_ref, dim=(0, 1), norm="ortho").real
        fig, ax = vis.show_2d(
            [bf_ref_before, bf_ref_after],
            cbar=True,
            title=[
                "Bright Field reference image [Before]",
                "Bright Field reference image [After]",
            ],
        )

    if bin > 1:
        # if verbosity > 0:
        #     print(
        #         "Interpolating shifts from binned indices to all bright field indices..."
        #     )

        # Convert to numpy for scipy interpolation
        binned_points = bright_field_unique_inds_binned_ordered.cpu().numpy() * bin
        shift_values = bright_field_shifts.cpu().numpy()
        query_points = bright_field_inds_centered_ordered_by_radius.cpu().numpy()

        all_bright_field_shifts_np = griddata(
            binned_points, shift_values, query_points, method="cubic", fill_value=0.0
        )
        zero_shifts = torch.from_numpy(all_bright_field_shifts_np).sum(1) == 0

        inds_with_zero_shifts = torch.where(zero_shifts)[0]
        G_zero_shifts = G_moving[..., inds_with_zero_shifts]
        G_zero_shifts = torch.permute(G_zero_shifts, (2, 0, 1))
        xy_shift = relative_shifts(
            G_ref, G_zero_shifts, upsample_factor=registration_upsample_factor
        )
        all_bright_field_shifts_np[inds_with_zero_shifts] = xy_shift.cpu().numpy()
        # if verbosity > 0:
        #     print(f"Indices with zero shifts after interpolation= {zero_shifts.sum()}")
    else:
        all_bright_field_shifts_np = bright_field_shifts.cpu().numpy()
    # print(f"all_bright_field_shifts_np.shape = {all_bright_field_shifts_np.shape}")

    inverse_order = torch.argsort(bright_field_inds_order).cpu().numpy()
    all_bright_field_shifts_np = all_bright_field_shifts_np[inverse_order]
    # print(f"inverse_order.shape = {inverse_order.shape}")
    # print(f"all_bright_field_shifts_np.shape = {all_bright_field_shifts_np.shape}")
    result = all_bright_field_shifts_np.copy().astype(np.float32)
    # result[:,0] = all_bright_field_shifts_np[:,1]
    # result[:,1] = all_bright_field_shifts_np[:,0]
    return torch.as_tensor(result, device=device)


def aberrations_and_rotation_from_bright_field_shifts_interpolated(
    dataset: Dataset4dstem | DatasetVirtualBrightField4dstem,
    bright_field_mask_threshold: float,
    target_percentage_nonzero_pixels: float = 0.75,
    registration_upsample_factor: int = 10,
    lowpass_fwhm_bright_field: Optional[float] = None,
    bin: Optional[int] = None,
    arrow_scale: float = 25e-2,
    verbosity: int = 0,
    update_dataset: bool = True,
    return_G: bool = False,
    fit_max_iter: int = 50,
    fit_lr: float = 1,
    fit_rotation: bool = False,
    rot_tensor: torch.Tensor = None,
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    First determine the bright field shifts, then fit aberrations and rotation to the bright field shifts.
    Args:
        dataset: Dataset4dstem object containing the diffraction pattern.
        bright_field_mask: Mask of the bright field pixels.
        fit_rotation: Whether to fit the rotation. The default is True.
        target_percentage_nonzero_pixels: Target percentage of nonzero pixels for the bright field. The default is 0.75.
        n_batches: Number of batches for the bright field shifts. The default is 25.
        registration_upsample_factor: Upsampling factor for the registration. The default is 10.
        lowpass_fwhm_bright_field: FWHM of the lowpass filter for the bright field. The default is None, which means no lowpass filter.
        bin: Bin size for the bright field. The default is None, which means no binning.
        arrow_scale: Scale for the arrows in the plot. The default is 25e-2.
        verbosity: Verbosity level. The default is 0.
        update_dataset: Whether to update the dataset. The default is True.
        return_G: Whether to return the G. If True, the G is returned in addition to the aberrations and rotation.
        n_center_indices: Number of center indices to use for the bright field shifts. The default is 25.
        fit_max_iter: Maximum number of iterations for the fit. The default is 50.
        fit_lr: Learning rate for the fit. The default is 1.
        fit_rotation: Whether to fit the rotation. The default is False.
        rot_tensor: Tensor of the rotation. The default is None.
    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Aberrations, rotation, and G.

    Raises:
        ValueError: If the dataset is not a valid Dataset4dstem object.

    """

    rot_tensor = torch.tensor(
        [dataset.meta.rotation], device=device, dtype=torch.float32
    )
    device = dataset.device

    k_sampling = torch.as_tensor(dataset.meta.sampling[-2:], device=device)[None]
    wavelength = dataset.meta.wavelength
    # rotation_init = dataset.meta.rotation  # unused variable

    diff_mean = dataset.array[:25, :25].mean((0, 1))
    diff_mean /= diff_mean.max()
    bright_field_mask = diff_mean > bright_field_mask_threshold
    bright_field_inds = torch.argwhere(bright_field_mask)
    bright_field_inds_centered = (
        bright_field_inds.float() - torch.mean(bright_field_inds.float(), dim=0)[None]
    )
    bright_field_inds_order = torch.argsort(
        torch.sum(bright_field_inds_centered**2, dim=1)
    )
    bright_field_inds_ordered_by_radius = bright_field_inds[bright_field_inds_order]
    bright_field_inds_centered_ordered_by_radius = bright_field_inds_centered[
        bright_field_inds_order
    ]

    diffraction_intensities = diff_mean[
        bright_field_inds_ordered_by_radius[:, 0],
        bright_field_inds_ordered_by_radius[:, 1],
    ]
    bright_field = (
        dataset.array[
            :,
            :,
            bright_field_inds_ordered_by_radius[:, 0],
            bright_field_inds_ordered_by_radius[:, 1],
        ]
        / diffraction_intensities[None, None, :]
    )

    percentage_nonzero = (bright_field != 0).float().mean()
    # Calculate minimum bin size needed to get percentage_nonzero >= 75%
    current_percentage = percentage_nonzero.item()

    bin = (
        int(np.ceil(np.sqrt(target_percentage_nonzero_pixels / current_percentage)))
        if bin is None
        else bin
    )

    G = torch.fft.fft2(bright_field, dim=(0, 1), norm="ortho")

    numerical_aperture_radius_pixels = torch.max(
        torch.norm(bright_field_inds_centered, dim=1)
    ).item()
    n_angular_samples = 6
    n_radial_samples = 5

    parent_beams_coords = beamlet_samples(
        bright_field_mask.cpu().numpy(),
        numerical_aperture_radius_pixels,
        n_angular_samples,
        n_radial_samples,
    )

    M = dataset.array.shape[-2:]
    parent_beams = torch.zeros(tuple(M), dtype=torch.bool)
    for si in parent_beams_coords:
        parent_beams[si[0], si[1]] = 1
    parent_beams = torch.fft.fftshift(parent_beams)
    # Bp = parent_beams_coords.shape[0]
    B = bright_field_mask.sum()

    parent_beams_coords = torch.argwhere(parent_beams)
    parent_beams_coords_centered = (
        parent_beams_coords.float()
        - torch.mean(parent_beams_coords.float(), dim=0)[None]
    ).long()
    parent_beams_coords_order = torch.argsort(
        torch.sum(parent_beams_coords_centered.float() ** 2, dim=1)
    )
    parent_beams_coords_ordered_center = parent_beams_coords_centered[
        parent_beams_coords_order
    ]
    # parent_beams_coords_ordered = parent_beams_coords[parent_beams_coords_order]

    if n_radial_samples > 1:
        nnw_np = natural_neighbor_weights(
            parent_beams_coords_ordered_center,
            bright_field_inds_centered_ordered_by_radius.long().cpu().numpy(),
            minimum_weight_cutoff=20e-2,
        )
        nnw = torch.as_tensor(nnw_np).to(torch.float32).to(device)
    else:
        nnw = np.ones((B, 1))
    # %
    # i = 4
    # b = np.zeros(tuple(M))
    # indi = bright_field_inds_centered_ordered_by_radius.long().cpu().numpy()
    # b[indi[:,0], indi[:,1]] = nnw_np[:,i]
    # plt.figure(figsize=(8,8))
    # plt.imshow(np.fft.fftshift(b))
    # plt.colorbar()
    # plt.title('Natural Neighbor Weight Distribution')
    # plt.xlabel('X coordinate')
    # plt.ylabel('Y coordinate')
    # plt.show()

    # %
    device = G.device
    kx = torch.fft.fftfreq(G.shape[1], device=device).reshape(1, -1)
    ky = torch.fft.fftfreq(G.shape[0], device=device).reshape(-1, 1)

    if lowpass_fwhm_bright_field is not None:
        sigma = lowpass_fwhm_bright_field / (2 * (2 * np.log(2)) ** 0.5)
        kx2 = kx**2  # shape (1, N)
        ky2 = ky**2  # shape (M, 1)
        k2 = kx2 + ky2  # shape (M, N)
        gaussian_filter = torch.exp(-2 * (np.pi**2) * (sigma**2) * k2)

    G_refs = torch.zeros(
        nnw.shape[1], G.shape[0], G.shape[1], device=device, dtype=G.dtype
    )
    ## compute Grefs by
    for i_ref in range(nnw.shape[1]):
        nnw_i = nnw[:, i_ref]
        take = nnw_i > 0
        nnw_nonzero = nnw_i[take]
        # print(f"nnw_nonzero.shape = {nnw_nonzero.shape}")
        G_refs[i_ref] = torch.mean(G[:, :, take] * nnw_nonzero[None, None, :], dim=-1)
        # vis.show_2d(torch.fft.ifft2(G_refs[i_ref]).real, cbar=True, title=f"G_ref {i_ref}")

    parent_beam_shifts = torch.zeros(len(parent_beams_coords_order), 2, device=device)
    G_ref = G_refs[0]
    for i in tqdm(range(0, len(parent_beams_coords_order), 1), desc="First pass:"):
        if lowpass_fwhm_bright_field is not None:
            G_moving = G_refs[i] * gaussian_filter
        else:
            G_moving = G_refs[i]
        parent_beam_shifts[i] = relative_shifts(
            G_ref, G_moving, upsample_factor=registration_upsample_factor
        )
        dx = parent_beam_shifts[i, 1]
        dy = parent_beam_shifts[i, 0]
        phase_ramp = torch.exp(-1j * 2 * np.pi * (dx * kx + dy * ky))
        G_ref = G_ref * i / (i + 1) + G_refs[i] * phase_ramp * 1 / (i + 1)

    if verbosity > 0:
        s = torch.tensor(dataset.meta.sampling[:2], device=device, dtype=torch.float32)
        plot_bright_field_shifts(
            parent_beams_coords_ordered_center,
            parent_beam_shifts,
            s,
            wavelength,
            torch.tensor(dataset.meta.rotation, device=device, dtype=torch.float32),
            1,
        )

    all_bright_field_shifts = torch.zeros(
        len(bright_field_inds_order), 2, device=device
    )
    for i in tqdm(
        range(0, len(bright_field_inds_order), 1), desc="Interpolate BF shifts:"
    ):
        # Get the weights for this bright field index
        weights = nnw[i]
        # Find indices where weights are non-zero
        nonzero_weights = weights > 0
        # Get the corresponding shifts and weights
        relevant_shifts = parent_beam_shifts[nonzero_weights]
        relevant_weights = weights[nonzero_weights]
        # Calculate weighted average shift for this bright field index
        weighted_shift = torch.sum(
            relevant_shifts * relevant_weights.unsqueeze(1), dim=0
        )
        # Store the interpolated shift
        all_bright_field_shifts[i] = weighted_shift

    shift_values = all_bright_field_shifts
    shift_values -= shift_values.mean(axis=0)  # subtract mean shift
    shift_values *= torch.tensor(
        dataset.meta.sampling[:2], device=device, dtype=torch.float32
    )
    inverse_order = torch.argsort(bright_field_inds_order)
    shift_values = shift_values[inverse_order]

    if verbosity > 0:
        cc_shifts_x = torch.zeros_like(diff_mean)
        cc_shifts_y = torch.zeros_like(diff_mean)
        cc_shifts_x[bright_field_inds[:, 0], bright_field_inds[:, 1]] = shift_values[
            :, 1
        ]
        cc_shifts_y[bright_field_inds[:, 0], bright_field_inds[:, 1]] = shift_values[
            :, 0
        ]

    opt_aberrations, opt_rotation = fit_aberrations_and_rotation_to_bright_field_shifts(
        shift_values,
        k_sampling[0],
        wavelength,
        bright_field_mask,
        rotation_init=rot_tensor,
        rotation_requires_grad=fit_rotation,
        max_iter=fit_max_iter,
        lr=fit_lr,
        verbosity=verbosity,
    )
    fitted_shifts = aberrations_to_image_shifts(
        opt_aberrations, opt_rotation, k_sampling[0], wavelength, bright_field_mask
    )

    if verbosity > 0:
        fitted_shifts_x = torch.zeros_like(diff_mean)
        fitted_shifts_y = torch.zeros_like(diff_mean)
        fitted_shifts_x[bright_field_inds[:, 0], bright_field_inds[:, 1]] = (
            fitted_shifts[:, 1]
        )
        fitted_shifts_y[bright_field_inds[:, 0], bright_field_inds[:, 1]] = (
            fitted_shifts[:, 0]
        )
        titles = [
            "CC Shift Values X",
            "CC Shift Values Y",
            "Fitted Shifts X",
            "Fitted Shifts Y",
        ]
        plots = [cc_shifts_x, cc_shifts_y, fitted_shifts_x, fitted_shifts_y]
        vis.show_2d(plots, cbar=True, title=titles, cmap="RdBu")

    # additional refinement here
    # %
    fitted_shifts_ordered = fitted_shifts[bright_field_inds_order]
    # build G_refs for each parent beam
    G_refs = torch.zeros(
        nnw.shape[1], G.shape[0], G.shape[1], device=device, dtype=G.dtype
    )
    # parent_beam_mean_shifts = torch.zeros(nnw.shape[1], 2, device=device)
    for i_ref in range(nnw.shape[1]):
        nnw_i = nnw[:, i_ref]
        take = nnw_i > 0
        nnw_nonzero = nnw_i[take]
        # Get all indices that contribute to this parent beam (where weight > 0)
        contributing_indices = torch.where(take)[0]
        # Get the fitted shifts for all contributing indices
        shifts_x = fitted_shifts_ordered[contributing_indices, 1]  # x shifts
        shifts_y = fitted_shifts_ordered[contributing_indices, 0]  # y shifts
        # Create meshgrid of k-vectors
        kx = torch.fft.fftfreq(G.shape[1], device=device).reshape(1, -1)
        ky = torch.fft.fftfreq(G.shape[0], device=device).reshape(-1, 1)
        # Create phase ramps for each contributing index
        dx = shifts_x[None, None, :]  # Shape: (1, 1, n_contrib)
        dy = shifts_y[None, None, :]  # Shape: (1, 1, n_contrib)
        phase_ramps = torch.exp(
            -1j * 2 * np.pi * (dx * kx[..., None] + dy * ky[..., None])
        )
        G_refs[i_ref] = torch.mean(
            G[:, :, take] * phase_ramps * nnw_nonzero[None, None, :], dim=-1
        )
        # if i_ref < 10:
        #     b = torch.zeros(tuple(M))
        #     indi = bright_field_inds_centered_ordered_by_radius.long()
        #     b[indi[:,0], indi[:,1]] = nnw[:,i_ref].cpu()
        #     vis.show_2d([torch.fft.fftshift(b), torch.fft.ifft2(G_refs[i_ref]).real], cbar=True, title=f"G_ref {i_ref}")

    parent_beams_coords_ordered_dev = parent_beams_coords_ordered_center.to(device)
    bright_field_shifts2 = torch.zeros(len(bright_field_inds_order), 2, device=device)
    fwhm_lowpass_bf = None
    for i, bf_ind in tqdm(
        enumerate(bright_field_inds_centered_ordered_by_radius), desc="Final pass:"
    ):
        if fwhm_lowpass_bf is not None:
            G_moving = G[..., i] * gaussian_filter
        else:
            G_moving = G[..., i]

        index_of_closest_parent_beam = torch.argmin(
            torch.norm(bf_ind - parent_beams_coords_ordered_dev, dim=1)
        )
        pb = parent_beams_coords_ordered_dev[index_of_closest_parent_beam]
        parent_beam_index_in_all_beams = torch.argmin(
            torch.norm(pb - bright_field_inds_centered_ordered_by_radius, dim=1)
        )  # torch.argwhere((pb == bright_field_inds_centered_ordered_by_radius).all(1))[0].item()
        parent_beam_shift = fitted_shifts_ordered[parent_beam_index_in_all_beams]
        G_ref = G_refs[index_of_closest_parent_beam]
        # kg = bf_ind.cpu().numpy()
        relative_shift = relative_shifts(
            G_ref, G_moving, upsample_factor=registration_upsample_factor
        )
        # print(f'bf coords: {kg} pb coords: {pb.cpu().numpy()} pb shift: {parent_beam_shift.cpu().numpy()} rel shift: {relative_shift.cpu().numpy()}')
        bright_field_shifts2[i] = relative_shift + parent_beam_shift

    shift_values2 = bright_field_shifts2
    shift_values2 -= shift_values2.mean(axis=0)  # subtract mean shift
    shift_values2 *= torch.tensor(
        dataset.meta.sampling[:2], device=device, dtype=torch.float32
    )
    shift_values2 = shift_values2[inverse_order]

    if verbosity > 0:
        cc_shifts_x2 = torch.zeros_like(diff_mean)
        cc_shifts_y2 = torch.zeros_like(diff_mean)
        cc_shifts_x2[bright_field_inds[:, 0], bright_field_inds[:, 1]] = shift_values2[
            :, 1
        ]
        cc_shifts_y2[bright_field_inds[:, 0], bright_field_inds[:, 1]] = shift_values2[
            :, 0
        ]

    opt_aberrations2, opt_rotation2 = (
        fit_aberrations_and_rotation_to_bright_field_shifts(
            shift_values2,
            k_sampling[0],
            wavelength,
            bright_field_mask,
            rotation_init=rot_tensor,
            rotation_requires_grad=fit_rotation,
            max_iter=fit_max_iter,
            lr=fit_lr,
            verbosity=verbosity,
        )
    )

    if verbosity > 0:
        fitted_shifts2 = aberrations_to_image_shifts(
            opt_aberrations2,
            opt_rotation2,
            k_sampling[0],
            wavelength,
            bright_field_mask,
        )
        fitted_shifts_x2 = torch.zeros_like(diff_mean)
        fitted_shifts_y2 = torch.zeros_like(diff_mean)
        fitted_shifts_x2[bright_field_inds[:, 0], bright_field_inds[:, 1]] = (
            fitted_shifts2[:, 1]
        )
        fitted_shifts_y2[bright_field_inds[:, 0], bright_field_inds[:, 1]] = (
            fitted_shifts2[:, 0]
        )
        titles = [
            "CC Shift Values X",
            "CC Shift Values Y",
            "Fitted Shifts X",
            "Fitted Shifts Y",
        ]
        plots = [cc_shifts_x2, cc_shifts_y2, fitted_shifts_x2, fitted_shifts_y2]
        vis.show_2d(plots, cbar=True, title=titles, cmap="RdBu")

    if update_dataset:
        if fit_rotation:
            dataset.meta.rotation = opt_rotation2.detach().item()
        dataset.meta.aberrations.array = opt_aberrations2

    if return_G:
        return opt_aberrations2, opt_rotation2, G
    else:
        return opt_aberrations2, opt_rotation2

 


def phase_contrast_transfer_function(
    dataset: Dataset4dstem | DatasetVirtualBrightField4dstem,
    upsample: Union[float, str] = "nyquist",
    bright_field_mask_threshold: float = 0.5,
    verbosity: int = 0,
    **kwargs,
) -> torch.Tensor:
    """
    Performs a joint ptychography reconstruction and aberration determination.
    Args:
        dataset: Dataset4dstem or DatasetVirtualBrightField4dstem object containing the diffraction pattern.
        upsample: Upsampling factor for the diffraction pattern.
        bright_field_mask_threshold: Threshold for the bright field.
        verbosity: Verbosity level.
        store_image_in_dataset: Whether to store the image in the dataset.
        n_batches: Number of batches for the vBF.

    Returns:
        torch.Tensor: The reconstructed phase image (Imaginary part).
        DatasetVirtualBrightField4dstem: the vBF dataset.

    Raises:
        ValueError: If the dataset is not a valid Dataset4dstem object.
        ValueError: If the upsample is invalid.
    """
    if upsample == "nyquist":
        scan_sampling = np.array(dataset.sampling[:2])
        nyquist_sampling = dataset.meta.wavelength / (
            4 * dataset.meta.semiconvergence_angle
        )
        upsample_float = scan_sampling / nyquist_sampling
        upsample_int = np.ceil(upsample_float).astype(np.uint32)
        if upsample_float.any() < 1:
            upsample_float[upsample_float < 1] = 1.0
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
            upsample_float[upsample_float < 1] = 1.0
        if verbosity > 0:
            print(f"Upsampling to Nyquist, upsample factor: {upsample_int}")

    if isinstance(dataset, Dataset4dstem):
        vBF = DatasetVirtualBrightField4dstem.from_4dstem_dataset(
            dataset,
            bright_field_mask_threshold=bright_field_mask_threshold,
            device=dataset.device,
        )
    else:
        vBF = dataset
    ds_rotation = torch.tensor(vBF.meta.rotation, device=vBF.device)
    semiconvergence_angle = vBF.meta.semiconvergence_angle
    wavelength = vBF.meta.wavelength

    if (upsample_int[0] > 1) or (upsample_int[1] > 1):
        new_shape = (
            int(round(vBF.G.shape[0] * upsample_int[0])),
            int(round(vBF.G.shape[1] * upsample_int[1])),
        )
        Gprime = torch.tile(vBF.G, (upsample_int[0], upsample_int[1], 1))
    else:
        Gprime = vBF.G
        new_shape = tuple([int(vBF.G.shape[0]), int(vBF.G.shape[1])])
    Qy, Qx = vBF.get_q_1d(new_shape)
    Kx = vBF.k[:, 1]
    Ky = vBF.k[:, 0]
    aberrations = vBF.meta.aberrations.array
    pctf = _phase_contrast_transfer_function(
        Gprime,
        aberrations,
        ds_rotation,
        semiconvergence_angle,
        wavelength,
        Qx,
        Qy,
        Kx,
        Ky,
    )
    return pctf




def determine_aberrations(
    dataset: Dataset4dstem | DatasetVirtualBrightField4dstem,
    correction_method: str = "bright-field-shifts",
    fit_rotation: bool = True,
    registration_upsample_factor: int = 10,
    lowpass_fwhm_bright_field: Optional[float] = None,
    bin_factors: tuple[int, ...] = (2, 1, 1),
    upsample: Union[int, str] = "nyquist",
    n_batches: int = 25,
    roi_shape: tuple[int, int] = (128, 128),
    roi_center: Union[str, tuple[int, int]] = "center",
    num_iterations: int = 10,
    lr: float = 20,
    bright_field_mask_threshold: float = 0.5,
    target_percentage_nonzero_pixels: float = 0.75,
    correct_order: int = 1,
    gradient_mask: torch.Tensor = torch.ones(12, dtype=torch.bool),
    verbosity: int = 0,
    update_dataset: bool = True,
    n_center_indices: int = 25,
    **kwargs,
) -> tuple[torch.Tensor, DatasetVirtualBrightField4dstem]:
    """
    Performs a joint ptychography reconstruction and aberration determination.
    Args:
        dataset: Dataset4dstem or DatasetVirtualBrightField4dstem object containing the diffraction pattern.
        correction_method: Method to correct the aberrations. Either "bright-field-shifts" or "total-variation".
        fit_rotation: Whether to fit the rotation.
        registration_upsample_factor: Upsampling factor for the registration.
        lowpass_fwhm_bright_field: FWHM of the lowpass filter for the bright field.
        bin: Bin size for the bright field. Used for bright field shifts.
        upsample: Upsampling factor for the diffraction pattern.
        n_batches: Number of batches for the bright field shifts.
        roi_shape: Shape of the region of interest.
        roi_center: Center of the region of interest.
        num_iterations: Number of optimization iterations.
        lr: Learning rate for the optimizer.
        bright_field_mask_threshold: Threshold for the bright field. Used for bright field shifts.
        target_percentage_nonzero_pixels: Target percentage of nonzero pixels for the bright field.
        correct_order: Order of the aberrations to correct. Used for total variation.
        gradient_mask: Mask for the gradient. Used for total variation.
        verbosity: Verbosity level.
        n_center_indices: Number of center indices to use for the bright field shifts. The default is 25.
    Returns:
        torch.Tensor: The reconstructed weak phase image.
        DatasetVirtualBrightField4dstem: the input dataset with the aberrations determined.

    Raises:
        ValueError: If the roi_center is invalid.
        ValueError: If the dataset is not a valid Dataset4dstem or DatasetVirtualBrightField4dstem object.
        ValueError: If the upsample is invalid.
    """
    if roi_center == "center":
        roi_center = (dataset.shape[0] // 2, dataset.shape[1] // 2)
    elif roi_center == "dark_field_center_of_mass":
        roi_center = (dataset.shape[0] // 2, dataset.shape[1] // 2)
    elif isinstance(roi_center, tuple):
        roi_center = roi_center
    else:
        raise ValueError(f"Invalid roi_center: {roi_center}")

    ds = dataset

    zero_aberrations_index = 11
    if correct_order == 0:
        zero_aberrations_index = 1
    elif correct_order == 1:
        zero_aberrations_index = 3
    elif correct_order == 2:
        zero_aberrations_index = 7
    elif correct_order == 3:
        zero_aberrations_index = 11

    if correct_order > 0:
        gradient_mask[zero_aberrations_index:] = 0
    # Ensure upsample is an integer for torch.tile
    if upsample == "nyquist":
        scan_sampling = np.array(dataset.sampling[:2])
        nyquist_sampling = dataset.meta.wavelength / (
            4 * dataset.meta.semiconvergence_angle
        )
        upsample_float = scan_sampling / nyquist_sampling
        upsample_int = np.ceil(upsample_float).astype(np.uint32)
        if upsample_float.any() < 1:
            upsample_float[upsample_float < 1] = 1.0
        if verbosity > 0:
            print(f"Upsampling to Nyquist, upsample factor: {upsample_int}")

    elif isinstance(upsample, str):
        raise ValueError(f"Invalid upsample: {upsample}")
    elif isinstance(upsample, float) or isinstance(upsample, int):
        upsample_float = np.array([upsample, upsample])
        upsample_int = np.ceil(upsample_float).astype(np.uint32)
        if upsample_float.any() < 1:
            upsample_float[upsample_float < 1] = 1.0
        if verbosity > 0:
            print(f"Upsampling to Nyquist, upsample factor: {upsample_int}")
    else:
        raise ValueError(f"Invalid upsample: {upsample}")


    ds_rotation = ds.meta.rotation
    semiconvergence_angle = ds.meta.semiconvergence_angle
    wavelength = ds.meta.wavelength
    dataset = ds
    device = dataset.device
    wavelength = dataset.meta.wavelength

    if isinstance(dataset, Dataset4dstem):
        vBF = DatasetVirtualBrightField4dstem.from_4dstem_dataset(
            dataset, bright_field_mask_threshold=bright_field_mask_threshold
        )
    else:
        vBF = dataset
    new_shape = (
        int(round(vBF.G.shape[0] * upsample_int[0])),
        int(round(vBF.G.shape[1] * upsample_int[1])),
    )
    Qy, Qx = vBF.get_q_1d(new_shape)
    Kx = vBF.k[:, 1]
    Ky = vBF.k[:, 0]

    if correction_method == "bright-field-shifts":
        with torch.no_grad():
            if verbosity > 0:
                print("Fitting aberrations and rotation from bright field shifts")
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                start_time.record()
 

            aberrations, rotation = aberrations_and_rotation_from_bright_field_shifts(
                vBF,
                fit_rotation=fit_rotation,
                target_percentage_nonzero_pixels=target_percentage_nonzero_pixels,
                n_batches=n_batches,
                registration_upsample_factor=registration_upsample_factor,
                lowpass_fwhm_bright_field=lowpass_fwhm_bright_field,
                bin_factors=bin_factors, 
                verbosity=verbosity,
                update_dataset=update_dataset,
                n_center_indices=n_center_indices,
            )

            if verbosity > 0:
                end_time.record()
                torch.cuda.synchronize()
                elapsed_time = start_time.elapsed_time(end_time)
                print(
                    f"Time to fit aberrations and rotation: {elapsed_time/1000:.3f} s"
                )

    elif correction_method == "total-variation":
        G = vBF.G
        vBF.array.requires_grad = False

        if upsample_int[0] > 1 or upsample_int[1] > 1:
            Gprime = torch.tile(G, (upsample_int[0], upsample_int[1], 1))
            del G
        else:
            Gprime = G

        aberrations = vBF.meta.aberrations.array.clone().to(vBF.device)
        aberrations.requires_grad = True
        Gprime.requires_grad = False
        optimizer = torch.optim.LBFGS(
            [aberrations], lr=lr, max_iter=num_iterations, line_search_fn="strong_wolfe"
        )
        tv_loss = TotalVariation().to(vBF.device)
        direct_ptycho_image = None

        def closure():
            nonlocal direct_ptycho_image
            optimizer.zero_grad()

            Gprime_corrected = CorrectAberrations.apply(
                Gprime,
                aberrations,
                torch.tensor(vBF.meta.rotation, device=vBF.device),
                semiconvergence_angle,
                wavelength,
                Qx,
                Qy,
                Kx,
                Ky,
            )
            factor = np.sqrt(upsample_int[0] * upsample_int[1])
            G_bf = (
                torch.fft.ifft2(Gprime_corrected, dim=(0, 1), norm="ortho").imag
                * factor
            )
            direct_ptycho_image = torch.mean(G_bf, dim=(-1))
            loss = -tv_loss(direct_ptycho_image[roi_slice].unsqueeze(0).unsqueeze(0))
            loss.backward()
            return loss

        roi_slice = np.s_[
            roi_center[0] - roi_shape[0] // 2 : roi_center[0] + roi_shape[0] // 2,
            roi_center[1] - roi_shape[1] // 2 : roi_center[1] + roi_shape[1] // 2,
        ]

        # Add optimizer hook to zero gradients for specific indices
        def zero_grad_hook(grad):
            # g_before = grad.detach().cpu().numpy()[0]
            if grad is not None:
                grad[gradient_mask == 0] = 0
            return grad

        # Register the hook
        aberrations.register_hook(zero_grad_hook)

        # for i in range(num_iterations):
        optimizer.step(closure)

        print(f"aberrations    = {aberrations[0:4].detach().cpu().numpy()} ")
    else:
        raise ValueError(
            f"Invalid correction method: {correction_method}. Choose from 'bright-field-shifts' or 'total-variation'."
        )
    # Use the direct_ptycho_image from the last optimization iteration
    # No need to recompute since it's already available from closure()

    print("\nOptimized aberration coefficients:")
    polar = cartesian2polar(
        {
            "C10": aberrations[0].item(),
            "C12a": aberrations[1].item(),
            "C12b": aberrations[2].item(),
            "C21a": aberrations[3].item(),
            "C21b": aberrations[4].item(),
            "C23a": aberrations[5].item(),
            "C23b": aberrations[6].item(),
            "C30": aberrations[7].item(),
        }
    )
    print(f"  C10 (Defocus): {polar['C10']:.3f}")
    print(f"  C12 (Astigmatism magnitude): {polar['C12']:.3f}")
    print(f"  C21 (Coma magnitude): {polar['C21']:.3f}")
    print(f"  C30 (Spherical aberration): {polar['C30']:.3f}")
    print(f"  phi12 (Astigmatism angle): {polar['phi12']:.3f}")
    print(f"  phi21 (Coma angle): {polar['phi21']:.3f}")
    if update_dataset:
        vBF.meta.aberrations.array = aberrations.detach()
        if fit_rotation:
            vBF.meta.rotation = rotation
        if correction_method == "total-variation":
            vBF.weak_phase_image = direct_ptycho_image.detach()

    return aberrations.detach(), vBF


def _direct_ptychography(
    Gprime: torch.Tensor,
    aberrations: torch.Tensor,
    ds_rotation: float,
    semiconvergence_angle: float,
    wavelength: float,
    Qy: torch.Tensor,
    Qx: torch.Tensor,
    Kx: torch.Tensor,
    Ky: torch.Tensor,
    device: torch.device,
    upsample: np.ndarray,
    # shifts_batch: torch.Tensor,
) -> torch.Tensor:
    Gprime_corrected = correct_aberrations_inplace(
        Gprime,
        aberrations,
        torch.tensor(ds_rotation, device=device),
        semiconvergence_angle,
        wavelength,
        Qx,
        Qy,
        Kx,
        Ky,
    )
    factor = np.sqrt(upsample[0] * upsample[1])
    G_bf = torch.fft.ifft2(Gprime_corrected, dim=(0, 1), norm="ortho") * factor
    phase_image = torch.mean(G_bf.imag, dim=(-1))
    return phase_image


def direct_ptychography(
    dataset: Dataset4dstem | DatasetVirtualBrightField4dstem,
    upsample: Union[float, str] = "nyquist",
    bright_field_mask_threshold: float = 0.5,
    verbosity: int = 0,
    n_batches: int = 25,
    return_snr: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Performs a joint ptychography reconstruction and aberration determination.
    Args:
        dataset: Dataset4dstem or DatasetVirtualBrightField4dstem object containing the diffraction pattern.
        upsample: Upsampling factor for the diffraction pattern.
        bright_field_mask_threshold: Threshold for the bright field.
        verbosity: Verbosity level.
        store_image_in_dataset: Whether to store the image in the dataset.
        n_batches: Number of batches for the vBF.

    Returns:
        torch.Tensor: The reconstructed phase image (Imaginary part). 
        torch.Tensor | None: The SNR of the reconstructed phase image if return_snr is True, otherwise None.
    Raises:
        ValueError: If the dataset is not a valid Dataset4dstem object.
        ValueError: If the upsample is invalid.
    """
    if upsample == "nyquist":
        scan_sampling = np.array(dataset.sampling[:2])
        nyquist_sampling = dataset.meta.wavelength / (
            4 * dataset.meta.semiconvergence_angle
        )
        upsample_float = scan_sampling / nyquist_sampling
        upsample_int = np.ceil(upsample_float).astype(np.uint32)
        if upsample_float.any() < 1:
            upsample_float[upsample_float < 1] = 1.0
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
            upsample_float[upsample_float < 1] = 1.0
        if verbosity > 0:
            print(f"Upsampling to Nyquist, upsample factor: {upsample_int}")

    if isinstance(dataset, Dataset4dstem):
        vBF = DatasetVirtualBrightField4dstem.from_4dstem_dataset(
            dataset,
            bright_field_mask_threshold=bright_field_mask_threshold,
            device=dataset.device,
        )
    else:
        vBF = dataset

    ds_rotation = vBF.meta.rotation
    semiconvergence_angle = vBF.meta.semiconvergence_angle
    wavelength = vBF.meta.wavelength
    device = vBF.device
    G = vBF.G

    new_shape = (
        int(round(G.shape[0] * upsample_int[0])),
        int(round(G.shape[1] * upsample_int[1])),
    )
    Qy, Qx = vBF.get_q_1d(new_shape)

    with torch.no_grad():
        if verbosity > 0:
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()
        # Loop through batches if n_batches is specified and > 1, else process all at once
        if n_batches is not None and n_batches > 1:
            phase_image = torch.zeros(
                (G.shape[0] * upsample_int[0], G.shape[1] * upsample_int[1]),
                device=device,
            )
            batch_size = int(np.ceil(G.shape[-1] / n_batches))

            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, G.shape[-1])
                G_batch = G[..., start_idx:end_idx]

                if (upsample_int > 1).any():
                    Gprime_batch = torch.tile(
                        G_batch, (upsample_int[0], upsample_int[1], 1)
                    )
                else:
                    Gprime_batch = G_batch
                phase_image_batch = _direct_ptychography(
                    Gprime_batch,
                    vBF.meta.aberrations.array,
                    ds_rotation,
                    semiconvergence_angle,
                    wavelength,
                    Qy,
                    Qx,
                    vBF.k[start_idx:end_idx, 1],
                    vBF.k[start_idx:end_idx, 0],
                    device,
                    upsample_int,
                )
                phase_image[:] += phase_image_batch
            phase_image /= n_batches

        else:
            if (upsample_int[0] > 1) or (upsample_int[1] > 1):
                new_shape = (
                    int(round(vBF.G.shape[0] * upsample_int[0])),
                    int(round(vBF.G.shape[1] * upsample_int[1])),
                )
                Gprime = torch.tile(vBF.G, (upsample_int[0], upsample_int[1], 1))
            else:
                Gprime = vBF.G
                new_shape = tuple([int(vBF.G.shape[0]), int(vBF.G.shape[1])])
            phase_image = _direct_ptychography(
                Gprime,
                vBF.meta.aberrations.array,
                ds_rotation,
                semiconvergence_angle,
                wavelength,
                Qy,
                Qx,
                vBF.k[:, 1],
                vBF.k[:, 0],
                device,
                upsample_int,
            )

        if verbosity > 0:
            end_time.record()
            torch.cuda.synchronize()
            elapsed_time = start_time.elapsed_time(end_time)
            print(
                f"Time to reconstruct directptychography image: {elapsed_time/1000:.3f} s"
            )
        if return_snr:
            snr_ptycho_analytical = direct_ptychography_ssnr(dataset, upsample=upsample, verbosity=verbosity)
        else:
            snr_ptycho_analytical = None
    return phase_image, snr_ptycho_analytical
 

def direct_ptychography_depth_section(
    dataset: Dataset4dstem,
    depth_angstroms=torch.tensor,
    bright_field_mask_threshold: float = 0.3,
    return_device: torch.device = torch.device("cpu"),
    upsample: Union[float, str] = "nyquist",
    n_batches: int = 25,
    verbosity: int = 0,
) -> torch.Tensor:
    aberrations_array = dataset.meta.aberrations.array.detach().clone()
    wavelength = dataset.meta.wavelength
    rotation = dataset.meta.rotation
    semiconvergence_angle = dataset.meta.semiconvergence_angle
    if upsample == "nyquist":
        scan_sampling = np.array(dataset.sampling[:2])
        nyquist_sampling = dataset.meta.wavelength / (
            4 * dataset.meta.semiconvergence_angle
        )
        upsample_float = scan_sampling / nyquist_sampling
        upsample_int = np.ceil(upsample_float).astype(np.uint32)
        if upsample_float.any() < 1:
            upsample_float[upsample_float < 1] = 1.0
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
            upsample_float[upsample_float < 1] = 1.0
        if verbosity > 0:
            print(f"Upsampling to Nyquist, upsample factor: {upsample_int}")
    if isinstance(dataset, Dataset4dstem):
        vBF = DatasetVirtualBrightField4dstem.from_4dstem_dataset(
            dataset,
            bright_field_mask_threshold=bright_field_mask_threshold,
            device=dataset.device,
        )
    else:
        vBF = dataset

    new_shape = (
        int(round(vBF.G.shape[0] * upsample_int[0])),
        int(round(vBF.G.shape[1] * upsample_int[1])),
    )
    Qy, Qx = vBF.get_q_1d(new_shape)

    G_depth_sections = torch.zeros(
        len(depth_angstroms),
        vBF.G.shape[0] * upsample_int[0],
        vBF.G.shape[1] * upsample_int[1],
        device=return_device,
    )
    aberrations_array_depth_section = torch.zeros_like(aberrations_array)
    for i, depth in enumerate(depth_angstroms):
        aberrations_array_depth_section[:] = aberrations_array
        aberrations_array_depth_section[0] += depth
        if n_batches is not None and n_batches > 1:
            phase_image = torch.zeros(
                (vBF.G.shape[0] * upsample_int[0], vBF.G.shape[1] * upsample_int[1]),
                device=vBF.device,
            )
            batch_size = int(np.ceil(vBF.G.shape[-1] / n_batches))

            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, vBF.G.shape[-1])
                G_batch = vBF.G[..., start_idx:end_idx]
                if (upsample_int > 1).any():
                    Gprime_batch = torch.tile(
                        G_batch, (upsample_int[0], upsample_int[1], 1)
                    )
                else:
                    Gprime_batch = G_batch
                phase_image_batch = _direct_ptychography(
                    Gprime_batch,
                    aberrations_array_depth_section,
                    rotation,
                    semiconvergence_angle,
                    wavelength,
                    Qy,
                    Qx,
                    vBF.k[start_idx:end_idx, 1],
                    vBF.k[start_idx:end_idx, 0],
                    vBF.device,
                    upsample_int,
                )

                phase_image += phase_image_batch
            phase_image /= n_batches
        else:
            if (upsample_int[0] > 1) or (upsample_int[1] > 1):
                Gprime = torch.tile(vBF.G, (upsample_int[0], upsample_int[1], 1))
            else:
                Gprime = vBF.G
            phase_image = _direct_ptychography(
                Gprime,
                aberrations_array_depth_section,
                rotation,
                semiconvergence_angle,
                wavelength,
                Qy,
                Qx,
                vBF.k[:, 1],
                vBF.k[:, 0],
                vBF.device,
                upsample_int,
            )
        G_depth_sections[i] = phase_image.to(return_device)
    return G_depth_sections

def direct_ptychography_ssnr(
    dataset: Dataset4dstem | DatasetVirtualBrightField4dstem, 
    upsample: Union[float, str] = "nyquist",  
    verbosity: int = 0,
):
    dalpha0 = dataset.sampling[-1] * dataset.meta.wavelength
    pctf_corner_center = phase_contrast_transfer_function(dataset, verbosity=1, upsample=upsample) 
    pctf = torch.fft.fftshift(pctf_corner_center) 

    # Calculate ptycho noise
    q = fftfreq2(pctf.shape, dataset.sampling[:2])
    qn = torch.norm(q, dim=0)
    q1d = qn.view(-1).cpu().numpy()
    R = dataset.meta.semiconvergence_angle / dataset.meta.wavelength
    delta_k = dataset.sampling[-1]
    N2, N3 = double_and_triple_pixel_counts(q1d, R, delta_k)
    rBF = dataset.meta.semiconvergence_angle / dalpha0
    Nalpha = np.pi * rBF**2
    ptycho_noise_squared = (N2+N3).reshape(pctf.shape) / Nalpha
    ptycho_noise_2d = torch.as_tensor(np.sqrt(ptycho_noise_squared), device=dataset.device, dtype=torch.float32)
    snr_ptycho_analytical = pctf_corner_center / ptycho_noise_2d 
    snr_ptycho_analytical[ptycho_noise_2d == 0] = 0

    if verbosity > 1:
        nconf = NormalizationConfig(interval_type="manual", vmin=0, vmax=1)
        vis.show_2d([pctf], cbar=True, title=["pctf"], norm=nconf)  
        fig, ax = plt.subplots(1,4, figsize=(20, 5))
        ax[0].imshow(N2.reshape(pctf.shape))
        ax[1].imshow(N3.reshape(pctf.shape))
        
        ax[2].imshow(pctf_corner_center.cpu().numpy())  
        
        ax[3].imshow((ptycho_noise_2d == 0).cpu().numpy())
        
        plt.show()
        plt.tight_layout()        

        p2 = torch.fft.fftshift(snr_ptycho_analytical).cpu().numpy()
        # vis.show_2d([snr_ptycho_analytical, torch.as_tensor(analytical_ssb_ssnr)], cbar=True, title=["snr_ptycho", "snr_ptycho_analytical"])
        
        p2s = p2.shape[0]//4
        fig, ax = plt.subplots(1,1, figsize=(5, 5))
        imax = ax.imshow(p2[p2s:-p2s, p2s:-p2s], cmap='magma')
        plt.colorbar(imax)
        plt.title("snr_ptycho_analytical")
        plt.tight_layout()
        plt.show()
    return snr_ptycho_analytical