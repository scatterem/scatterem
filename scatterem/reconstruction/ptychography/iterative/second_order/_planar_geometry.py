"""Scan geometry and probe seed of :class:`~.planar.PlanarPtychography`.

The rules are those of the unified ``Ptychography`` module for one
scan in the lab frame (``object_in_scan_frame=False``), so that both models
put a scan position on the same object pixel:

* the object pixel is the real-space pixel of the model grid,
  ``1 / (N dk)``;
* the raster is :func:`~scatterem.utils.stem.scan.advanced_raster_scan`,
  turned by the scan rotation, centred on its mean;
* the object canvas holds the raster plus one probe window, rounded up;
* each position splits into an integer top-left corner and a sub-pixel
  offset ``dr`` in ``[-0.5, 0.5]``.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from scatterem.utils.stem.scan import advanced_raster_scan


def scan_geometry(
    scan_shape: tuple[int, int],
    scan_step: float,
    dk: float,
    model_grid: int,
    rotation_deg: float,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor, tuple[int, int]]:
    """Integer corners, sub-pixel offsets and object shape of a raster scan.

    Parameters
    ----------
    scan_shape : tuple of int
        ``(Ny, Nx)`` scan positions, row-major.
    scan_step : float
        Scan step in angstroms (the same on both axes).
    dk : float
        Detector pixel in inverse angstroms.
    model_grid : int
        Side ``N`` of the model's far-field grid (the probe window).
    rotation_deg : float
        Scan rotation in degrees.
    device : torch.device or str or None, optional
        Device of the returned tensors.

    Returns
    -------
    positions : Tensor
        ``(K, 2)`` int64 top-left object pixels of the probe windows.
    offsets : Tensor
        ``(K, 2)`` float32 sub-pixel offsets ``dr`` in pixels.
    object_shape : tuple of int
        ``(H, W)`` of the object canvas.
    """
    sampling = np.array([scan_step, scan_step, dk, dk], dtype=np.float32)
    detector = np.array([model_grid, model_grid])
    pixel = 1 / (2 * (sampling[-2:] * detector / 2))  # AcquisitionMetadata.dr
    step_px = sampling[:2] / pixel
    raster = advanced_raster_scan(
        int(scan_shape[0]),
        int(scan_shape[1]),
        1,
        mirror=[1, 1],
        theta=rotation_deg,
        dy=step_px[0],
        dx=step_px[1],
        device=device,
    )
    raster = raster - raster.mean(0)
    window = torch.as_tensor(detector, device=raster.device)
    lo, hi = raster.min(0).values, raster.max(0).values
    canvas = torch.ceil((hi - lo) + window)
    shifted = raster + (canvas / 2 - window / 2)
    corners = torch.round(shifted).to(torch.int32)
    offsets = shifted - corners.float()
    shape = (int(canvas[0]), int(canvas[1]))
    return corners.to(torch.int64), offsets.to(torch.float32), shape


def pad_probe(probe: Tensor, model_grid: int) -> Tensor:
    """Zero-pad real-space probe modes in the centred aperture plane.

    The detector pixel stays the same and the real-space pixel becomes finer,
    as the ``detector_pad_to`` grid of ``Ptychography`` does. The power is
    not kept; normalise afterwards (:func:`normalise_probe`).

    Parameters
    ----------
    probe : Tensor
        ``(M, 1, n, n)`` complex modes.
    model_grid : int
        Target side ``N >= n``, with ``N - n`` even.

    Returns
    -------
    Tensor
        ``(M, 1, N, N)`` complex modes.
    """
    n = int(probe.shape[-1])
    if n == model_grid:
        return probe
    if model_grid < n or (model_grid - n) % 2:
        raise ValueError(f"cannot pad a {n} px probe to {model_grid} px")
    dims = (-2, -1)
    aperture = torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(probe, dim=dims)), dim=dims
    )
    o = (model_grid - n) // 2
    padded = torch.zeros(
        (*probe.shape[:-2], model_grid, model_grid),
        dtype=aperture.dtype,
        device=aperture.device,
    )
    padded[..., o : o + n, o : o + n] = aperture
    real = torch.fft.ifft2(torch.fft.ifftshift(padded, dim=dims))
    return torch.fft.fftshift(real, dim=dims).to(probe.dtype)


def normalise_probe(probe: Tensor, norm: float) -> Tensor:
    """Scale the modes to total power ``sum |P|^2 = norm``.

    Parameters
    ----------
    probe : Tensor
        Complex modes.
    norm : float
        Target power: the sum over the detector of the mean measured
        intensity pattern.

    Returns
    -------
    Tensor
        The scaled modes.
    """
    scale = (torch.sum(torch.abs(probe) ** 2) / norm) ** 0.5
    return probe / scale


def radial_support_mask(n: int, radius: float, margin: float, device=None) -> Tensor:
    """``(n, n)`` soft real-space disk: 1 inside ``radius``, a cosine ramp
    to 0 over ``margin`` pixels, 0 beyond; centred on ``(n / 2, n / 2)``.

    Parameters
    ----------
    n : int
        Grid side.
    radius : float
        Radius of the flat part, in pixels.
    margin : float
        Width of the ramp, in pixels.
    device : torch.device or str or None, optional
        Device of the result.

    Returns
    -------
    Tensor
        float32 mask.
    """
    yy = torch.arange(n, device=device, dtype=torch.float32) - n / 2
    r = torch.sqrt(yy[:, None] ** 2 + yy[None, :] ** 2)
    ramp = (r - radius).clamp(0.0, margin) / max(margin, 1e-6)
    mask = 0.5 * (1 + torch.cos(torch.pi * ramp))
    mask[r <= radius] = 1.0
    mask[r >= radius + margin] = 0.0
    return mask
