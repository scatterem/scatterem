"""Bright-field transfer helpers: the shift model and its aberration frames.

Two frames of the 12-vector of cartesian aberration coefficients
(:data:`scatterem.physics.optics.LENS_LAYOUT`) are in use, and they differ
only at a scan rotation ``rot != 0``:

* DETECTOR frame. ``meta.aberrations`` is a detector-frame vector: chi is
  evaluated at the unrotated detector coordinate ``K``. The direct-ptychography
  kernels, tilt-corrected dark field and the
  bright-field fits' published results all use it. The scan-frame image of the
  bright-field pixel at ``K`` moves by ``R(-rot) grad chi(K) / 2 pi`` (the
  VECTOR is rotated).
* SHIFT-MODEL frame. :func:`aberrations_to_image_shifts`, the forward model of
  the bright-field shift fits, evaluates ``grad chi(R(-rot) K) / 2 pi`` (the
  ARGUMENT is rotated).

Both give the same shift field once every m-fold pair ``Ca + i Cb`` of the
detector-frame vector is replaced by ``(Ca + i Cb) exp(-i m rot)``; the round
terms (C10, C30) are the same in both. :func:`detector_to_shift_model_frame`
and its inverse :func:`shift_model_to_detector_frame` convert between them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch

from scatterem.physics.optics import LENS_LAYOUT, LENS_TERMS, lens_wavefront_slope
from scatterem.utils.spectral_mesh import dual_mesh

#: ``(index of Ca, index of Cb, m)`` of every m-fold pair in the 12-vector.
_PAIRS: tuple[tuple[int, int, int], ...] = tuple(
    (i, i + 1, m) for i, (_, m, part) in enumerate(LENS_TERMS) if part == "c" and m > 0
)




def aberrations_to_image_shifts(
    aberrations_array: torch.Tensor,
    rotation: torch.Tensor,
    sampling: np.ndarray,
    wavelength: float,
    shape: Sequence[int] | torch.Size,
) -> torch.Tensor:
    """Bright-field image shifts of an aberration vector: the shift model.

    Evaluates ``grad chi(R(-rot) K) / 2 pi`` on the detector grid: the ARGUMENT
    of the gradient is rotated, so ``aberrations_array`` is read in the
    SHIFT-MODEL frame (see the module docstring). This is the forward model the
    bright-field shift fits are fitted with. For a detector-frame vector, such
    as ``meta.aberrations``, pass
    ``detector_to_shift_model_frame(aberrations, rotation)``; the result is
    then ``R(-rot) grad chi(K) / 2 pi``, the shift the direct-ptychography
    kernels imply.

    Parameters
    ----------
    aberrations_array : torch.Tensor
        Shape ``(12,)``, :data:`scatterem.physics.optics.LENS_LAYOUT`
        order, Angstrom, in the shift-model frame. Its dtype sets the
        precision of the result and its device the device.
    rotation : torch.Tensor
        Scan rotation in degrees, one element.
    sampling : numpy.ndarray
        Detector-grid real-space sampling ``(2,)`` in Angstrom; the detector
        frequencies are ``fftfreq(shape, sampling)``.
    wavelength : float
        Electron wavelength in Angstrom.
    shape : sequence of int or torch.Size
        Detector grid ``(H, W)``.

    Returns
    -------
    torch.Tensor
        Shape ``(H, W, 2)``, the ``(y, x)`` shift of every detector pixel in
        Angstrom, zero frequency at ``(H // 2, W // 2)``.
    """
    device = aberrations_array.device
    if rotation.device != device:
        rotation = rotation.to(device)
    if isinstance(shape, torch.Size):
        shape = tuple(shape)
    else:
        shape = tuple(shape)
    q = torch.fft.fftshift(
        dual_mesh(shape, sampling, staggered=False, device=device), dim=(-2, -1)
    )
    # dual_mesh is float32 by default; honour the caller's precision instead, so a
    # float64 fit is not silently truncated to float32 by its own forward model.
    if aberrations_array.is_floating_point():
        q = q.to(aberrations_array.dtype)
    else:
        # lens_wavefront_slope takes a floating lens vector; an integer one is
        # read in the grid dtype.
        aberrations_array = aberrations_array.to(q.dtype)

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

    slope = lens_wavefront_slope(aberrations_array, qy_rot, qx_rot, lambda_a=wavelength)
    return slope / (2 * np.pi)


def _pair_rotation(rotation_deg: float) -> torch.Tensor:
    """The ``(12, 12)`` float64 matrix that turns every m-fold pair by ``exp(-i m rot)``.

    Rows of the round terms (C10, C30) are identity rows. Built on the host in
    one ``torch.tensor`` call.
    """
    angle = math.radians(rotation_deg)
    n = len(LENS_LAYOUT)
    rows = [[float(i == j) for j in range(n)] for i in range(n)]
    for ia, ib, m in _PAIRS:
        c, s = math.cos(m * angle), math.sin(m * angle)
        rows[ia][ia], rows[ia][ib] = c, s
        rows[ib][ia], rows[ib][ib] = -s, c
    return torch.tensor(rows, dtype=torch.float64)


def detector_to_shift_model_frame(
    aberrations: torch.Tensor, rotation_deg: float | torch.Tensor
) -> torch.Tensor:
    """Re-express a detector-frame 12-vector in the shift model's frame.

    ``aberrations_to_image_shifts(out, rot)`` is then the shift the
    direct-ptychography kernels imply for the detector-frame ``aberrations``,
    ``R(-rot) grad chi(K) / 2 pi`` (see the module docstring). The inverse is
    :func:`shift_model_to_detector_frame`.

    Parameters
    ----------
    aberrations : torch.Tensor
        Shape ``(12,)``, :data:`scatterem.physics.optics.LENS_LAYOUT`
        order, Angstrom, in the detector frame (``meta.aberrations``). Any
        real dtype and device.
    rotation_deg : float or torch.Tensor
        Scan rotation (``meta.rotation``) in degrees; a one-element tensor (as
        ``determine_aberrations(fit_rotation=True)`` returns it) is read with
        ``float``.

    Returns
    -------
    torch.Tensor
        The vector with every m-fold pair ``Ca + i Cb`` turned by
        ``exp(-i m rot)``, as one float64 matrix product returned in the
        input's dtype and on its device. The round terms (C10, C30) come back
        unchanged, and ``aberrations`` itself is returned at
        ``rotation_deg == 0``, so a defocus-only vector or an unrotated scan
        is passed through bit for bit. An integer vector comes back in the
        default float dtype (the turned pair is not integer).

    Raises
    ------
    ValueError
        If ``rotation_deg`` is not finite.
    """
    rotation_deg = float(rotation_deg)
    if not math.isfinite(rotation_deg):
        raise ValueError(f"rotation_deg must be finite, got {rotation_deg}")
    if rotation_deg == 0:
        return aberrations
    turn = _pair_rotation(rotation_deg).to(aberrations.device)
    dtype = aberrations.dtype
    if not aberrations.is_floating_point():
        dtype = torch.get_default_dtype()
    return (turn @ aberrations.double()).to(dtype)


def shift_model_to_detector_frame(
    aberrations: torch.Tensor, rotation_deg: float | torch.Tensor
) -> torch.Tensor:
    """Re-express a shift-model-frame 12-vector in the detector frame.

    The inverse of :func:`detector_to_shift_model_frame`: a vector fitted with
    :func:`aberrations_to_image_shifts` (the bright-field shift fits) becomes
    the detector-frame vector ``meta.aberrations`` holds and the
    direct-ptychography kernels, the ptychography probe and tilt-corrected
    dark field read.

    Parameters
    ----------
    aberrations : torch.Tensor
        Shape ``(12,)``, :data:`scatterem.physics.optics.LENS_LAYOUT`
        order, Angstrom, in the shift-model frame. Any real dtype and device.
    rotation_deg : float or torch.Tensor
        Scan rotation in degrees, the one the vector was fitted at; a
        one-element tensor is read with ``float``.

    Returns
    -------
    torch.Tensor
        The vector with every m-fold pair ``Ca + i Cb`` turned by
        ``exp(+i m rot)``, in the input's dtype and on its device. The round
        terms come back unchanged, and ``aberrations`` itself is returned at
        ``rotation_deg == 0``.
    """
    return detector_to_shift_model_frame(aberrations, -float(rotation_deg))










def pair_overlap_area_torch(d, R):
    """Torch form of the pair overlap area (stays on ``d``'s device).

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
    """Torch form of the triple overlap area (stays on ``q``'s device).

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
    """Torch form of the double and triple overlap areas.

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
    """Torch form of the double and triple pixel counts.

    Number of pixels in double- and triple-overlap regions, given k-space
    pixel size ``delta_k``. Returns torch tensors on ``q``'s device so the
    analytical-SSNR computation never round-trips to host.
    """
    if torch.is_tensor(q) and q.is_cuda:
        # One bit-exact Warp kernel instead of ~70 float64 launches;
        # gates and the exactness argument are in the module.  None means
        # the ``overlap_counts`` backend row declined.
        from scatterem.utils._overlap_counts_warp import overlap_counts

        out = overlap_counts(q, R, delta_k)
        if out is not None:
            return out
    A2, A3 = double_and_triple_overlap_areas_torch(q, R)
    pix_area = delta_k**2
    N2 = A2 / pix_area
    N3 = A3 / pix_area
    return N2, N3




