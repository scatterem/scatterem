import torch
import numpy as np


def dft_upsample_batched(cc_stack, upsample_factor, xy_shifts_stack, device="cpu"):
    """
    Fully vectorized batched version of dft_upsample for processing multiple correlation images.
    
    Args:
        cc_stack (torch.Tensor): Stack of correlation images (nbatch x height x width)
        upsample_factor (int): Upsampling factor
        xy_shifts_stack (torch.Tensor): Upsample centers (nbatch x 2)
        device (str): Device to use for computation
    
    Returns:
        torch.Tensor: Stack of upsampled images (nbatch x upsampled_height x upsampled_width)
    """
    nbatch, height, width = cc_stack.shape
    _ = nbatch  # Used for clarity in function signature
    pixelRadius = 1.5
    numRow = int(np.ceil(pixelRadius * upsample_factor))
    numCol = numRow
    
    # Pre-compute frequency grids for all images
    # These are the same for all images since they have the same size
    freq_x = torch.fft.ifftshift(torch.arange(width, device=device)) - np.floor(width / 2)  # (width,)
    freq_y = torch.fft.ifftshift(torch.arange(height, device=device)) - np.floor(height / 2)  # (height,)
    
    # Create coordinate grids for upsampling
    coord_x = torch.arange(numCol, device=device)  # (numCol,)
    coord_y = torch.arange(numRow, device=device)  # (numRow,)
    
    # Extract shift components for all images at once
    xy_shift_x = xy_shifts_stack[:, 0]  # (nbatch,)
    xy_shift_y = xy_shifts_stack[:, 1]  # (nbatch,)
    
    # Vectorized kernel computation
    # For column kernel: freq_x (width,) x coord_x (numCol,) -> (width, numCol)
    # We need to broadcast across batch dimension: (nbatch, width, numCol)
    coord_x_expanded = coord_x.unsqueeze(0).unsqueeze(0)  # (1, 1, numCol)
    xy_shift_y_expanded = xy_shift_y.unsqueeze(1).unsqueeze(2)  # (nbatch, 1, 1)
    freq_x_expanded = freq_x.unsqueeze(0).unsqueeze(2)  # (1, width, 1)
    
    # Compute column kernels for all images at once
    colKern_all = torch.exp(
        (-1j * 2 * np.pi / (width * upsample_factor))
        * freq_x_expanded * (coord_x_expanded - xy_shift_y_expanded)  # (nbatch, width, numCol)
    )
    
    # For row kernel: coord_y (numRow,) x freq_y (height,) -> (numRow, height)
    # We need to broadcast across batch dimension: (nbatch, numRow, height)
    coord_y_expanded = coord_y.unsqueeze(0).unsqueeze(2)  # (1, numRow, 1)
    xy_shift_x_expanded = xy_shift_x.unsqueeze(1).unsqueeze(2)  # (nbatch, 1, 1)
    freq_y_expanded = freq_y.unsqueeze(0).unsqueeze(1)  # (1, 1, height)
    
    # Compute row kernels for all images at once
    rowKern_all = torch.exp(
        (-1j * 2 * np.pi / (height * upsample_factor))
        * (coord_y_expanded - xy_shift_x_expanded) * freq_y_expanded  # (nbatch, numRow, height)
    )
    
    # Vectorized matrix multiplication for all images
    # For each image i: rowKern_all[i] @ cc_stack[i] @ colKern_all[i]
    # This can be done using batch matrix multiplication
    
    # First multiplication: cc_stack @ colKern_all
    # cc_stack: (nbatch, height, width), colKern_all: (nbatch, width, numCol)
    # Result: (nbatch, height, numCol)
    intermediate = torch.bmm(cc_stack, colKern_all)
    
    # Second multiplication: rowKern_all @ intermediate
    # rowKern_all: (nbatch, numRow, height), intermediate: (nbatch, height, numCol)
    # Result: (nbatch, numRow, numCol)
    upsampled_images = torch.real(torch.bmm(rowKern_all, intermediate))
    
    return upsampled_images


def relative_shifts(
    G1,
    G2,
    upsample_factor, 
):
    """
    Relative shift of images G1 and G2 using DFT upsampling of cross correlation.
    Supports both single images and stacks of images.

    Parameters
    -------
    G1: torch.Tensor
        fourier transform of reference image (2D: height x width)
    G2: torch.Tensor
        fourier transform of image(s) to align (2D: height x width or 3D: nbatch x height x width)
    upsample_factor: float
        upsampling for correlation. Must be greater than 2.
    device: str, optional
        calculation device will be perfomed on. Must be 'cpu' or 'gpu'

    Returns:
        xy_shift [pixels] - 2D tensor for single image, 3D tensor (nbatch x 2) for stack
    """
 
    # Handle both single images and stacks
    if G2.dim() == 2:
        # Single image case - add None dimension to make it 3D
        G2_stack = G2.unsqueeze(0)  # (1, height, width)
        xy_shifts = relative_shifts_dft_upsampled(G1, G2_stack, upsample_factor)
        return xy_shifts  # Return first (and only) shift as 2D tensor
    elif G2.dim() == 3:
        # Stack case - use directly
        return relative_shifts_dft_upsampled(G1, G2, upsample_factor)
    else:
        raise ValueError(f"G2 must be 2D (single image) or 3D (stack), got {G2.dim()}D")

def relative_shifts_dft_upsampled(G1, G2_stack, upsample_factor):
    """
    Align a stack of images to a reference image using batched operations.
    
    Parameters
    -------
    G1: torch.Tensor
        fourier transform of reference image (2D: height x width)
    G2_stack: torch.Tensor
        fourier transform of image stack (3D: nbatch x height x width)
    upsample_factor: float
        upsampling for correlation. Must be greater than 2.
    device: str, optional
        calculation device will be perfomed on. Must be 'cpu' or 'gpu'

    Returns:
        xy_shifts: torch.Tensor (nbatch x 2) - shifts for each image in the stack
    """
    nbatch, height, width = G2_stack.shape
    device = G1.device
    
    # Batched cross correlation: G1 * conj(G2_stack)
    # G1: (height, width), G2_stack: (nbatch, height, width)
    # We need to broadcast G1 to match G2_stack
    G1_expanded = G1.unsqueeze(0).expand(nbatch, -1, -1)  # (nbatch, height, width)
    cc_stack = G1_expanded * torch.conj(G2_stack)  # (nbatch, height, width)
    
    # Batched inverse FFT to get correlation images
    cc_real_stack = torch.real(torch.fft.ifft2(cc_stack))  # (nbatch, height, width)
    
    # Find peaks for all images at once
    # Get argmax for each image in the batch
    flat_indices = cc_real_stack.view(nbatch, -1).argmax(dim=1)  # (nbatch,)
    
    # CRITICAL: Ensure flat_indices are within bounds before unravel_index
    max_flat_index = height * width - 1
    flat_indices = torch.clamp(flat_indices, 0, max_flat_index)
    
    x0_stack, y0_stack = torch.unravel_index(flat_indices, (height, width))
    
    # CRITICAL: Ensure unraveled indices are within bounds
    x0_stack = torch.clamp(x0_stack, 0, height - 1)
    y0_stack = torch.clamp(y0_stack, 0, width - 1)
    
    torch.cuda.synchronize()
    # Batched half-pixel refinement
    # Create indices for all images at once
    x_inds_stack = torch.remainder(
        x0_stack.unsqueeze(1) + torch.arange(-1, 2, device=device), 
        height
    ).to(torch.int64)  # (nbatch, 3)
    
    y_inds_stack = torch.remainder(
        y0_stack.unsqueeze(1) + torch.arange(-1, 2, device=device), 
        width
    ).to(torch.int64)  # (nbatch, 3)
    
    # CRITICAL: Ensure indices are properly bounded before CUDA operations
    x_inds_stack = torch.clamp(x_inds_stack, 0, height - 1)
    y_inds_stack = torch.clamp(y_inds_stack, 0, width - 1)
    x0_stack = torch.clamp(x0_stack, 0, height - 1)
    y0_stack = torch.clamp(y0_stack, 0, width - 1)
    
    # Extract values for parabolic fitting
    batch_indices = torch.arange(nbatch, device=device).unsqueeze(1)  # (nbatch, 1)
    
    # Check for out-of-bounds indices and print a warning if found
    if (
        (x_inds_stack < 0).any() or (x_inds_stack >= height).any()
        or (y_inds_stack < 0).any() or (y_inds_stack >= width).any()
        or (x0_stack < 0).any() or (x0_stack >= height).any()
        or (y0_stack < 0).any() or (y0_stack >= width).any()
    ):
        print("Warning: One or more indices in x_inds_stack, y_inds_stack, x0_stack, or y0_stack are out of bounds.")
        print(f"x_inds_stack range: [{x_inds_stack.min().item()}, {x_inds_stack.max().item()}]")
        print(f"y_inds_stack range: [{y_inds_stack.min().item()}, {y_inds_stack.max().item()}]")
        print(f"x0_stack range: [{x0_stack.min().item()}, {x0_stack.max().item()}]")
        print(f"y0_stack range: [{y0_stack.min().item()}, {y0_stack.max().item()}]")
        print(f"height: {height}, width: {width}")
    
    # Get vx values: cc_real[batch_idx, x_inds, y0]
    vx_stack = cc_real_stack[batch_indices, x_inds_stack, y0_stack.unsqueeze(1)]  # (nbatch, 3)
     
    # Get vy values: cc_real[batch_idx, x0, y_inds]
    vy_stack = cc_real_stack[batch_indices, x0_stack.unsqueeze(1), y_inds_stack]  # (nbatch, 3)
     
    # Parabolic subpixel refinement (vectorized)
    dx_stack = (vx_stack[:, 2] - vx_stack[:, 0]) / (4 * vx_stack[:, 1] - 2 * vx_stack[:, 2] - 2 * vx_stack[:, 0])
    dy_stack = (vy_stack[:, 2] - vy_stack[:, 0]) / (4 * vy_stack[:, 1] - 2 * vy_stack[:, 2] - 2 * vy_stack[:, 0])
    
    # Round to half-pixel precision
    x0_refined = torch.round((x0_stack.float() + dx_stack) * 2.0) / 2.0
    y0_refined = torch.round((y0_stack.float() + dy_stack) * 2.0) / 2.0
    
    # Stack the refined coordinates for batched processing
    xy_shifts_initial = torch.stack([x0_refined, y0_refined], dim=1)  # (nbatch, 2)
    
    # Use batched upsampled correlation
    xy_shifts = upsampled_correlation_batched(cc_stack, upsample_factor, xy_shifts_initial, device)

    xy_shifts[:,0] = (
        torch.remainder(
            xy_shifts[:,0] + height / 2,
            height,
        )
        - height / 2
    )
    xy_shifts[:,1] = (
        torch.remainder(
            xy_shifts[:,1] + width / 2,
            width,
        )
        - width / 2
    )
    
    return xy_shifts


def upsampled_correlation_batched(cc_stack, upsample_factor, xy_shifts_stack, device="cpu"):
    """
    Upsampled cross-correlation for processing multiple cross-correlation images wit DFT upsampling.
    
    Args:
        cc_stack (torch.Tensor): Stack of correlation images (nbatch x height x width)
        upsample_factor (int): Upsampling factor
        xy_shifts_stack (torch.Tensor): Initial shift estimates (nbatch x 2)
        device (str): Device to use for computation
    
    Returns:
        torch.Tensor: Refined shifts (nbatch x 2)
    
    Note:
        For debugging CUDA out-of-bounds errors, set CUDA_LAUNCH_BLOCKING=1 environment variable
        before running your script to get more detailed error information.
    """
    nbatch = cc_stack.shape[0]
    
    # Round shifts to upsample_factor precision
    xy_shifts_rounded = torch.round(xy_shifts_stack * upsample_factor) / upsample_factor
    
    # Global shift calculation
    globalShift = np.fix(np.ceil(upsample_factor * 1.5) / 2)
    upsample_centers = globalShift - upsample_factor * xy_shifts_rounded  # (nbatch, 2)
    
    # Apply batched DFT upsampling
    image_corr_upsample = torch.conj(
        dft_upsample_batched(torch.conj(cc_stack), upsample_factor, upsample_centers, device=device)
    )
    
    # Find peaks for all images at once
    # Get argmax for each image in the batch
    flat_indices = image_corr_upsample.view(nbatch, -1).argmax(dim=1)  # (nbatch,)
    upsampled_height, upsampled_width = image_corr_upsample.shape[1], image_corr_upsample.shape[2]
    
    # CRITICAL: Ensure flat_indices are within bounds before unravel_index
    max_flat_index = upsampled_height * upsampled_width - 1
    flat_indices = torch.clamp(flat_indices, 0, max_flat_index)
    
    xy_subshift_stack = torch.unravel_index(flat_indices, (upsampled_height, upsampled_width))
    xy_subshift_stack = torch.stack(xy_subshift_stack, dim=1)  # (nbatch, 2)
    
    # CRITICAL: Ensure unraveled indices are within bounds
    xy_subshift_stack[:, 0] = torch.clamp(xy_subshift_stack[:, 0], 0, upsampled_height - 1)
    xy_subshift_stack[:, 1] = torch.clamp(xy_subshift_stack[:, 1], 0, upsampled_width - 1)
    
    # Batched parabolic subpixel fitting
    # Get 3x3 neighborhoods around peaks
    x_inds = xy_subshift_stack[:, 0].unsqueeze(1) + torch.arange(-1, 2, device=device)  # (nbatch, 3)
    y_inds = xy_subshift_stack[:, 1].unsqueeze(1) + torch.arange(-1, 2, device=device)  # (nbatch, 3)
    
    # Handle edge cases by clamping indices
    x_inds = torch.clamp(x_inds, 0, upsampled_height - 1)
    y_inds = torch.clamp(y_inds, 0, upsampled_width - 1)
    
    # INSERT_YOUR_CODE
    # Check if any indices are out of bounds and print a warning
    if (x_inds.min() < 0 or x_inds.max() >= upsampled_height or
        y_inds.min() < 0 or y_inds.max() >= upsampled_width):
        print(
            f"Warning: x_inds or y_inds out of bounds. "
            f"x_inds range    : [{x_inds.min().item()}, {x_inds.max().item()}], "
            f"y_inds range    : [{y_inds.min().item()}, {y_inds.max().item()}], "
            f"upsampled_height: {upsampled_height}, upsampled_width: {upsampled_width}"
        )

    # Extract 3x3 neighborhoods using gather operations (more CUDA-friendly)
    neighborhoods = torch.zeros(nbatch, 3, 3, device=device, dtype=image_corr_upsample.dtype)
    
    # Alternative approach: use gather operations to avoid double indexing
    for i in range(nbatch):
        # Get the current image's indices
        x_idx = x_inds[i]  # Shape: (3,)
        y_idx = y_inds[i]  # Shape: (3,)
        
        # Double-check bounds (defensive programming)
        x_idx = torch.clamp(x_idx, 0, upsampled_height - 1)
        y_idx = torch.clamp(y_idx, 0, upsampled_width - 1)
        
        # Extract neighborhood using single indexing operation
        try:
            # Method 1: Direct indexing (safer)
            neighborhoods[i] = image_corr_upsample[i][x_idx][:, y_idx]
        except (IndexError, RuntimeError) as e:
            print(f"IndexError at batch {i}: x_idx={x_idx}, y_idx={y_idx}")
            print(f"upsampled_height={upsampled_height}, upsampled_width={upsampled_width}")
            print(f"image_corr_upsample.shape={image_corr_upsample.shape}")
            print(f"x_idx range: [{x_idx.min().item()}, {x_idx.max().item()}]")
            print(f"y_idx range: [{y_idx.min().item()}, {y_idx.max().item()}]")
            
            # Fallback: use gather operations
            try:
                # Create coordinate grids
                batch_idx = torch.full((3, 3), i, device=device, dtype=torch.long)
                x_grid = x_idx.unsqueeze(1).expand(3, 3)  # (3, 3)
                y_grid = y_idx.unsqueeze(0).expand(3, 3)  # (3, 3)
                
                # Flatten indices for gather
                flat_indices = batch_idx * (upsampled_height * upsampled_width) + x_grid * upsampled_width + y_grid
                flat_image = image_corr_upsample.view(-1)
                
                # Gather values
                neighborhoods[i] = flat_image[flat_indices].view(3, 3)
            except Exception as e2:
                print(f"Fallback also failed: {e2}")
                raise e
        
        torch.cuda.synchronize()

    
    # Parabolic subpixel refinement (vectorized)
    # icc[0,1], icc[1,1], icc[2,1] for dx calculation
    vx = neighborhoods[:, :, 1]  # (nbatch, 3)
    # icc[1,0], icc[1,1], icc[1,2] for dy calculation  
    vy = neighborhoods[:, 1, :]  # (nbatch, 3)
    
    # Calculate dx and dy for all images
    denominator_x = 4 * vx[:, 1] - 2 * vx[:, 2] - 2 * vx[:, 0]
    denominator_y = 4 * vy[:, 1] - 2 * vy[:, 2] - 2 * vy[:, 0]
    
    # Avoid division by zero
    dx_stack = torch.where(
        torch.abs(denominator_x) > 1e-10,
        (vx[:, 2] - vx[:, 0]) / denominator_x,
        torch.zeros_like(denominator_x)
    )
    
    dy_stack = torch.where(
        torch.abs(denominator_y) > 1e-10,
        (vy[:, 2] - vy[:, 0]) / denominator_y,
        torch.zeros_like(denominator_y)
    )
    
    # Calculate final refined shifts
    xy_subshift_refined = xy_subshift_stack - globalShift
    refined_shifts = xy_shifts_rounded + (xy_subshift_refined + torch.stack([dx_stack, dy_stack], dim=1)) / upsample_factor
    
    return refined_shifts
    