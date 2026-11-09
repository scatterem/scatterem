"""Fresnel referencing of partitioned PRISM parents on a crop window.

Partitioned PRISM interpolates de-tilted parent columns across the aperture.
The interpolation is more accurate at a plane where the propagation chirp is
smaller, so the parents can be Fresnel-propagated to a reference plane first,
combined there, and the result propagated back on a padded crop window (Pelz
2021). :class:`_FocalReferenceMixin` holds that procedure once for both
matrices that use it: the probe-side ``PartitionedScatteringMatrix`` and the
detector-side ``DetectorExitSMatrix``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import pi

import numpy as np
import torch
from torch import Tensor


def _fresnel_window_kernel(
    wy: int,
    wx: int,
    n_slices: int,
    dxy: float,
    dxx: float,
    wavelength: float,
    dz: float,
    device,
    dtype: torch.dtype,
) -> Tensor:
    """Fresnel propagator over ``n_slices`` slices on a ``(wy, wx)`` window grid.

    Same sign convention as
    :func:`~scatterem.physics.propagation.propagator_kernel` (forward is
    ``exp(-i pi lambda (n dz) |q|^2)``), but built on the window's own
    frequency grid, so that a windowed reconstruction can be propagated locally.

    Parameters
    ----------
    wy, wx : int
        Window shape in pixels.
    n_slices : int
        Number of slices to propagate; negative propagates backwards.
    dxy, dxx : float
        Pixel size along y and x, in Angstrom.
    wavelength : float
        Electron wavelength, in Angstrom.
    dz : float
        Slice distance, in Angstrom.
    device : torch.device
        Device of the returned kernel.
    dtype : torch.dtype
        Complex dtype of the returned kernel.

    Returns
    -------
    Tensor
        Unit-modulus kernel ``(wy, wx)`` with the corner origin.
    """
    qy = torch.fft.fftfreq(wy, d=dxy, device=device, dtype=torch.float64).view(wy, 1)
    qx = torch.fft.fftfreq(wx, d=dxx, device=device, dtype=torch.float64).view(1, wx)
    chi = -pi * wavelength * (n_slices * dz) * (qy**2 + qx**2)
    return torch.exp(1j * chi).to(dtype)


def _pad_window(idx: Tensor, n: int, m: int) -> tuple[Tensor, slice]:
    """Pad a contiguous periodic window ``idx`` by ``m`` pixels on each side.

    The windowed forward Fresnel step wraps around its window; the margin keeps
    that wrap-around outside the inner window.

    Parameters
    ----------
    idx : Tensor
        1-D window indices with step 1 (modulo ``n``).
    n : int
        Length of the full periodic axis.
    m : int
        Margin in pixels on each side.

    Returns
    -------
    padded : Tensor
        The widened window indices, capped to the full axis length.
    inner : slice
        Selects the original ``idx`` positions within ``padded``.
    """
    length = int(idx.shape[0])
    padded_length = min(length + 2 * m, n)
    left = (padded_length - length) // 2
    start = (int(idx[0].item()) - left) % n
    padded = (start + torch.arange(padded_length, device=idx.device)) % n
    return padded, slice(left, left + length)


def _focal_margin(
    n_slices: int,
    gmax: float,
    grid: tuple[int, int],
    physics: tuple[float, float, tuple[float, float]] | None,
) -> int:
    """Window margin in pixels that covers the Fresnel spread of ``n_slices``.

    The spread is about ``(n dz) lambda k_max``, with ``k_max`` the largest beam
    frequency. Without the physical scale the margin is a constant 24 pixels.

    Parameters
    ----------
    n_slices : int
        Number of propagated slices (sign ignored).
    gmax : float
        Largest absolute signed beam index of the matrix.
    grid : (int, int)
        Grid shape ``(ny, nx)``.
    physics : (float, float, (float, float)) or None
        ``(wavelength, slice_distance, gridsize)``, in Angstrom.

    Returns
    -------
    int
        Margin in pixels, at least 4 and at most half the smaller grid axis.
    """
    if physics is None:
        return 24
    wavelength, dz, gridsize = physics
    ny, nx = grid
    kmax = gmax / min(gridsize)  # 1/A
    dx = min(gridsize[0] / ny, gridsize[1] / nx)
    spread_px = abs(n_slices) * dz * wavelength * kmax / dx
    return int(min(max(4, np.ceil(1.5 * spread_px) + 2), min(ny, nx) // 2))


def _focal_slices(focal_backprop, depth: int) -> int:
    """Whole slices to Fresnel-reference before natural-neighbor interpolation.

    Parameters
    ----------
    focal_backprop : float, str or None
        ``"centroid"`` selects half of ``depth``; a float selects that fraction
        of ``depth``; ``None``, ``0``, ``""`` or ``"0"`` disables the
        referencing.
    depth : int
        Number of slices the columns have traversed.

    Returns
    -------
    int
        Number of slices, rounded to the nearest whole slice.

    Raises
    ------
    ValueError
        If ``focal_backprop`` is a string other than ``"centroid"``.
    """
    if not focal_backprop or focal_backprop == "0":
        return 0
    if isinstance(focal_backprop, str):
        if focal_backprop != "centroid":
            raise ValueError(
                f"focal_backprop str must be 'centroid', got {focal_backprop!r}"
            )
        return int(round(depth / 2))
    return int(round(float(focal_backprop) * depth))


def _focal_physics(
    wavelength: float | None,
    slice_distance: float | None,
    gridsize: Sequence[float] | None,
) -> tuple[float, float, tuple[float, float]] | None:
    """Pack the windowed-Fresnel scale, or ``None`` when any part is missing."""
    if not (wavelength and slice_distance and gridsize is not None):
        return None
    return (
        float(wavelength),
        float(slice_distance),
        (float(gridsize[0]), float(gridsize[1])),
    )


class _FocalReferenceMixin:
    """Fresnel referencing for a partitioned matrix with parents in ``self.S``.

    The host class sets ``S``, ``kernel``, ``ny``, ``nx``, ``_device`` and
    ``_dtype``, and calls :meth:`_init_focal_reference` in its constructor.
    """

    S: Tensor
    kernel: Tensor
    ny: int
    nx: int
    _device: torch.device
    _dtype: torch.dtype

    def _init_focal_reference(
        self,
        focal_backprop: float | str | None,
        wavelength: float | None,
        slice_distance: float | None,
        gridsize: Sequence[float] | None,
    ) -> None:
        """Store the referencing setting and the windowed-Fresnel scale."""
        self.focal_backprop = focal_backprop
        self._physics = _focal_physics(wavelength, slice_distance, gridsize)
        self._S_ref: Tensor | None = None  # referenced full-grid parents
        self._S_ref_key: tuple[int, int] | None = None

    def _fresnel(self, S: Tensor, n_slices: int) -> Tensor:
        """Fresnel-propagate ``S`` by ``n_slices`` kernel steps (negative: inverse)."""
        if n_slices == 0:
            return S
        K = self.kernel**n_slices  # |kernel| = 1: negative powers are conjugates
        return torch.fft.ifft2(torch.fft.fft2(S, dim=(-2, -1)) * K, dim=(-2, -1))

    def _referenced_parents(self, plane: int, n_ref: int) -> Tensor:
        """Full-grid parents propagated by ``n_ref`` steps, cached per plane.

        Only ``Bp`` columns, shared by every atom of the plane.
        """
        key = (int(plane), int(n_ref))
        if self._S_ref_key != key or self._S_ref is None:
            self._S_ref = self._fresnel(self.S, n_ref)
            self._S_ref_key = key
        return self._S_ref

    def _focal_window(
        self,
        iy: Tensor,
        ix: Tensor,
        plane: int,
        n_ref: int,
        gmax: float,
        nnw: Callable[[Tensor, Tensor, Tensor], Tensor],
    ) -> Tensor:
        """Columns on the window ``(iy, ix)``, interpolated at the reference plane.

        Parameters
        ----------
        iy, ix : Tensor
            1-D row and column indices of the crop window.
        plane : int
            The matrix's current plane, the cache key of the referenced parents.
        n_ref : int
            Kernel steps from the current plane to the reference plane.
        gmax : float
            Largest absolute signed beam index, for the window margin.
        nnw : callable
            ``nnw(Sw, py, px)`` combines the parents ``Sw`` on the window
            ``(py, px)`` into the output columns.

        Returns
        -------
        Tensor
            Output columns on the window ``(n_out, wy, wx)`` at the current plane.

        Raises
        ------
        ValueError
            If the wavelength, slice distance or grid size is missing.
        """
        if self._physics is None:
            raise ValueError(
                "focal_backprop windowed reconstruction needs wavelength, "
                "slice_distance and gridsize"
            )
        m = _focal_margin(n_ref, gmax, (self.ny, self.nx), self._physics)
        py, iny = _pad_window(iy, self.ny, m)
        px, inx = _pad_window(ix, self.nx, m)
        Sw = self._referenced_parents(plane, n_ref)[:, py][:, :, px]
        recon = nnw(Sw, py, px)  # at the reference plane, on the padded window
        lam, dz, (ly, lx) = self._physics
        K = _fresnel_window_kernel(
            int(py.shape[0]),
            int(px.shape[0]),
            -n_ref,  # back to the current plane
            ly / self.ny,
            lx / self.nx,
            lam,
            dz,
            self._device,
            self._dtype,
        )
        recon = torch.fft.ifft2(torch.fft.fft2(recon, dim=(-2, -1)) * K, dim=(-2, -1))
        return recon[:, iny][:, :, inx]
