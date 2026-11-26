"""A small mixed-state scene for the ``PlanarPtychography`` tests."""

from __future__ import annotations

import torch

from scatterem.reconstruction.ptychography.iterative.second_order.planar import (
    PlanarPtychography,
)


def planar_truth(n_scan: int = 8, ndet: int = 24) -> tuple[PlanarPtychography, list]:
    """A small mixed-state scene and its noise-free data."""
    g = torch.Generator().manual_seed(3)
    n = ndet
    yy, xx = torch.meshgrid(
        torch.arange(n) - n / 2, torch.arange(n) - n / 2, indexing="ij"
    )
    aperture = (yy**2 + xx**2 < (n / 5) ** 2).to(torch.complex64)
    aperture = aperture * torch.exp(1j * torch.atan2(yy, xx))  # a vortex
    dims = (-2, -1)
    p0 = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(aperture)), dim=dims)
    p1 = torch.roll(p0, 1, dims=-1) * 0.4
    probe = torch.stack([p0, p1])[:, None]
    shape = (n_scan, n_scan, ndet, ndet)
    truth = PlanarPtychography.from_scan(
        torch.ones(shape), 0.4, 0.1, probe, rotation_deg=3.0
    )
    with torch.no_grad():
        h, w = truth.object.shape
        phase = 0.4 * torch.randn(h, w, generator=g)
        truth.object.copy_(torch.complex(phase, 0.02 * torch.rand(h, w, generator=g)))
        everything = torch.arange(n_scan * n_scan)
        intensities = truth(everything) ** 2
    return truth, intensities.reshape(shape)
