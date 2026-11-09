"""Fresnel / angular-spectrum transfer phases and tilt (shared by Propagator).

Kept beside :mod:`scatterem.physics.propagation` only because that module is
near the size limit.  Import the public names from ``propagation``.

All grids are corner-origin (``fftfreq`` order).  Distances are in Angstrom.
Phases are built in float64 from :func:`~scatterem.physics.periodic_sampling.dft_sample_frequencies`
(the same staging as :func:`~scatterem.physics.propagation.propagator_kernel`)
and cast by the caller when a narrower dtype is wanted.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import pi

import numpy as np
import torch
from torch import Tensor

from scatterem.physics.periodic_sampling import PeriodicGrid, dft_sample_frequencies

__all__ = [
    "angular_spectrum_kernel",
    "propagator_phase",
    "tilt_phase",
    "transfer_phase",
]


def _q2(gpts: Sequence[int], gridsize: Sequence[float], device) -> Tensor:
    """``|q|^2`` [1/A^2] on a corner-origin grid, float64."""
    qy, qx = np.broadcast_arrays(
        *dft_sample_frequencies(PeriodicGrid(tuple(gpts), tuple(gridsize)))
    )
    return torch.as_tensor(qy**2 + qx**2, dtype=torch.float64, device=device)


def propagator_phase(
    gpts: Sequence[int],
    gridsize: Sequence[float],
    wavelength: float,
    dz: float,
    method: str = "fresnel",
    device=None,
) -> Tensor:
    """Unwrapped free-space phase ``phi`` with ``kernel = exp(1j * phi)``.

    Parameters
    ----------
    gpts : sequence of int
        Grid size `(Ny, Nx)`.
    gridsize : sequence of float
        Real-space extent `(Ly, Lx)`, in Angstrom.
    wavelength : float
        Electron wavelength, in Angstrom.
    dz : float
        Propagation distance, in Angstrom (may be fractional or negative).
    method : str, optional
        ``"fresnel"``/``"paraxial"`` or ``"angular_spectrum"``/``"asm"``.
    device : torch.device, optional
        Device of the returned tensor.

    Returns
    -------
    Tensor
        Real float64 phase in radians, shape `(Ny, Nx)`, corner origin.
    """
    return transfer_phase(_q2(gpts, gridsize, device), wavelength, dz, method)


def transfer_phase(
    q2: Tensor, wavelength: float, dz: float, method: str = "fresnel"
) -> Tensor:
    """Free-space phase ``phi(|q|^2)`` of a propagation by ``dz``.

    Parameters
    ----------
    q2 : Tensor
        ``|q|^2`` in 1/Angstrom^2 (``q`` in cycles per Angstrom), float64, any
        shape and device.
    wavelength : float
        Electron wavelength, in Angstrom.
    dz : float
        Propagation distance, in Angstrom (may be negative).
    method : str, optional
        ``"fresnel"``/``"paraxial"`` or ``"angular_spectrum"``/``"asm"``.

    Returns
    -------
    Tensor
        Phase in radians, shape, dtype and device of ``q2``; the transfer
        function is ``exp(1j * phi)``.

    Raises
    ------
    ValueError
        If ``method`` is unknown.
    """
    if method in ("fresnel", "paraxial"):
        return (-pi * wavelength * dz) * q2
    if method in ("angular_spectrum", "asm"):
        # Reduced form ``kz - k = -(2π q)² / (kz + k)`` avoids the float32
        # cancellation of the direct ``sqrt(k² - (2π q)²) - k``; the
        # evanescent band (|q| > 1/λ) keeps the old clamp-to-zero-kz phase ``-k``.
        k = 2.0 * pi / wavelength
        arg = k**2 - (2.0 * pi) ** 2 * q2
        kz = torch.sqrt(torch.clamp(arg, min=0.0))
        return torch.where(arg > 0, dz * (-((2.0 * pi) ** 2) * q2) / (kz + k), -dz * k)
    raise ValueError(
        f"method must be 'angular_spectrum'/'asm' or 'fresnel'/'paraxial'; "
        f"got {method!r}"
    )


def angular_spectrum_kernel(
    gpts: Sequence[int],
    gridsize: Sequence[float],
    wavelength: float,
    dz: float,
    device=None,
    dtype: torch.dtype = torch.complex64,
) -> Tensor:
    """Exact angular-spectrum free-space propagator ``exp(i dz (kz - k))``.

    Parameters
    ----------
    gpts : sequence of int
        Grid size `(Ny, Nx)`.
    gridsize : sequence of float
        Real-space extent `(Ly, Lx)`, in Angstrom.
    wavelength : float
        Electron wavelength, in Angstrom.
    dz : float
        Propagation distance, in Angstrom.
    device : torch.device, optional
        Device of the returned tensor.
    dtype : torch.dtype, optional
        Output complex dtype. Default is `torch.complex64`.

    Returns
    -------
    Tensor
        Transfer function, shape `(Ny, Nx)`, corner-origin reciprocal grid.
        The evanescent band (``|q| > 1/lambda``) is clamped to zero phase.
    """
    return torch.exp(
        1j
        * propagator_phase(
            gpts, gridsize, wavelength, dz, method="angular_spectrum", device=device
        )
    ).to(dtype)


def tilt_phase(
    q: Tensor, slice_thickness: float | Tensor, tilt: Sequence[float] | Tensor
) -> Tensor:
    """Per-slice phase ``2*pi*dz*(qy*tan(theta_y) + qx*tan(theta_x))`` [rad].

    Parameters
    ----------
    q : torch.Tensor
        ``(2, Ny, Nx)`` spatial frequencies ``(qy, qx)`` in 1/length, as
        returned by :func:`scatterem.physics._frequency_grid.fft_frequency_grid`.
    slice_thickness : float or torch.Tensor
        Propagation distance per slice, in the length units of ``1/q``.
    tilt : Sequence of float or torch.Tensor
        ``(theta_y, theta_x)`` in radians; a tensor keeps the autograd graph.

    Returns
    -------
    torch.Tensor
        Real ``(Ny, Nx)`` phase, corner origin.
    """
    t = torch.tan(torch.as_tensor(tilt, dtype=q.dtype, device=q.device))
    return (2.0 * torch.pi) * slice_thickness * (q[0] * t[0] + q[1] * t[1])
