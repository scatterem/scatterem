from __future__ import annotations


import numpy as np
import torch

from scatterem.utils.aberration_basis import cartesian_chi_gradient
from scatterem.utils.grids import fft_frequencies_2d























def aberrations_to_image_shifts(
    aberrations_array: torch.Tensor,
    rotation: torch.Tensor,
    sampling: np.ndarray,
    wavelength: float,
    shape: tuple[int, int] | torch.Size,
) -> torch.Tensor:
    """
    Calculate the bright field shifts from the aberrations and rotation.

    Args:
        aberrations_array: torch.Tensor - aberrations array (1D: 12)
        rotation: torch.Tensor - rotation in degrees (1D: 1)
        sampling: torch.Tensor - sampling (2D: 2)
        wavelength: float - wavelength in Angstroms
        shape: tuple[int, int] | torch.Size - shape of the bright field mask (2D: H x W)

    Returns:
        torch.Tensor - bright field shifts (2D: H x W x 2)
    """
    device = aberrations_array.device
    if rotation.device != device:
        rotation = rotation.to(device)
    if isinstance(shape, torch.Size):
        shape = tuple(shape)
    else:
        shape = tuple(shape)
    q = torch.fft.fftshift(
        fft_frequencies_2d(shape, sampling, False, device=device), dim=(-2, -1)
    )
    # fft_frequencies_2d is float32; honour the caller's precision instead, so a
    # float64 fit is not silently truncated to float32 by its own forward model.
    if aberrations_array.is_floating_point():
        q = q.to(aberrations_array.dtype)

    # The shift field is the gradient of chi, evaluated in the rotated detector
    # frame. Negative rotation because this is the detector plane while the
    # dataset's rotation is given in the real-space scan plane.
    #
    # Evaluated ANALYTICALLY at rotated coordinates, rather than by taking a
    # numerical gradient of chi on the grid and resampling that with
    # grid_sample. chi is a polynomial in the scattering angle, so its gradient
    # is closed-form; the previous route paid for that three times over. It
    # differenced an analytic function, it interpolated the result
    # (bicubic, padding_mode="border", so the array edge was clamped and wrong
    # by up to a third of full scale -- outside the bright-field disk, which is
    # why it never showed), and grid_sample's backward pass uses CUDA atomics,
    # which made the whole aberration fit NON-DETERMINISTIC on identical input.
    angle = torch.deg2rad(-rotation)
    cos_rot, sin_rot = torch.cos(angle), torch.sin(angle)
    qy_rot = sin_rot * q[1] + cos_rot * q[0]
    qx_rot = cos_rot * q[1] - sin_rot * q[0]

    gradient = cartesian_chi_gradient(qy_rot, qx_rot, wavelength, aberrations_array)
    return torch.stack([gradient[0], gradient[1]], dim=-1) / (2 * np.pi)










def pair_overlap_area_torch(d, R):
    """Torch port of :func:`pair_overlap_area` (stays on ``d``'s device).

    Area of overlap of two circles of radius ``R`` with centre separation
    ``d``. Returns 0 for ``d >= 2R``. Matches the numpy reference exactly:
    the transcendentals are only evaluated where in-domain (via clamping),
    then masked out, so no NaN can leak for ``d >= 2R``.
    """
    d = d.double()
    A = torch.zeros_like(d)

    mask = d < 2 * R
    # Clamp the arccos argument into [-1, 1] before evaluating so the
    # out-of-domain entries (d >= 2R, where the result is masked to 0
    # anyway) never produce NaN. In the in-domain region the clamp is a
    # no-op, reproducing the numpy output bit-for-bit.
    arg = torch.clamp(d / (2 * R), -1.0, 1.0)
    sqrt_arg = torch.clamp(4 * R**2 - d**2, min=0.0)
    A_full = 2 * R**2 * torch.arccos(arg) - 0.5 * d * torch.sqrt(sqrt_arg)
    A = torch.where(mask, A_full, A)
    return A


def triple_overlap_area_torch(q, R):
    """Torch port of :func:`triple_overlap_area` (stays on ``q``'s device).

    Triple-overlap area ``A3(q)`` for three circles of radius ``R`` centred
    at ``-q, 0, +q``. Nonzero only for ``0 <= q <= R``.
    """
    q = q.double()
    A3 = torch.zeros_like(q)

    mask = q <= R
    arg = torch.clamp(q / R, -1.0, 1.0)
    sqrt_arg = torch.clamp(R**2 - q**2, min=0.0)
    A3_full = (
        torch.pi * R**2 - 2 * R**2 * torch.arcsin(arg) - 2 * q * torch.sqrt(sqrt_arg)
    )
    A3 = torch.where(mask, A3_full, A3)
    return A3


def double_and_triple_overlap_areas_torch(q, R):
    """Torch port of :func:`double_and_triple_overlap_areas`.

    Return ``(A2(q), A3(q))`` as torch tensors on ``q``'s device, matching
    the numpy reference exactly (including the ``q >= 2R`` cutoff to zero).
    """
    q = q.double()

    A3 = triple_overlap_area_torch(q, R)

    A01 = pair_overlap_area_torch(q, R)  # central with +q
    A12 = pair_overlap_area_torch(q, R)  # central with -q
    A02 = pair_overlap_area_torch(2 * q, R)  # +q with -q

    A2 = A01 + A12 + A02 - 3 * A3

    cutoff = q >= 2 * R
    A2 = torch.where(cutoff, torch.zeros_like(A2), A2)
    A3 = torch.where(cutoff, torch.zeros_like(A3), A3)

    return A2, A3


def double_and_triple_pixel_counts_torch(q, R, delta_k):
    """Torch port of :func:`double_and_triple_pixel_counts`.

    Number of pixels in double- and triple-overlap regions, given k-space
    pixel size ``delta_k``. Returns torch tensors on ``q``'s device so the
    analytical-SSNR computation never round-trips to host.
    """
    A2, A3 = double_and_triple_overlap_areas_torch(q, R)
    pix_area = delta_k**2
    N2 = A2 / pix_area
    N3 = A3 / pix_area
    return N2, N3






