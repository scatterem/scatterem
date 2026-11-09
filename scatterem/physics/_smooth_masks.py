"""Round masks with a logistic rim on a grid in DFT order.

This private module was written for scatterem from the description of the
Fermi (logistic) function and its Sommerfeld expansion in standard texts,
for example Ashcroft, N. W. and Mermin, N. D. (1976). Solid State Physics,
ch. 2. The public names are re-exported by ``scatterem.physics.optics``.

At the distance ``d`` [axis-0 pixel] from the grid origin, a disc of radius
``R`` and rim width ``w`` has the value ``1 / (1 + exp(4 (d - R) / w))``.
The value falls from about 0.881 to about 0.119 over the distance ``w``.
"""

from __future__ import annotations

import numbers
from collections.abc import Sequence

import torch

from scatterem.physics._checks import _checked

RIM_WIDTH_PX: float = 2.0
"""Default rim width ``w`` of the masks [axis-0 pixel].

Over ``2 w = 4`` pixels the rim goes from 0.982 to 0.018. A rim of 2 pixels
damps the ringing of a sharp disc by more than two decades, and it stays
narrow compared with typical radii of 10 to 100 pixels.
"""


def _check_shape(shape: object) -> tuple[int, int]:
    """Return ``(Ny, Nx)`` as Python ints, or raise."""
    if not isinstance(shape, Sequence) or isinstance(shape, str):
        raise TypeError(f"shape must be a sequence of ints, got {shape!r}")
    if len(shape) != 2:
        raise ValueError(f"shape must have two entries, got {len(shape)}")
    for n in shape:
        if isinstance(n, bool) or not isinstance(n, numbers.Integral):
            raise TypeError(f"shape must hold integers, got {shape!r}")
        if n < 1:
            raise ValueError(f"shape entries must be positive, got {shape!r}")
    return int(shape[0]), int(shape[1])


def _check_dtype(dtype: torch.dtype | None) -> torch.dtype:
    """Return the mask dtype, or raise ``TypeError``."""
    out = torch.get_default_dtype() if dtype is None else dtype
    if not isinstance(out, torch.dtype) or not out.is_floating_point:
        raise TypeError(f"dtype must be a real floating torch.dtype, got {dtype}")
    return out


def _distance(
    shape: tuple[int, int], px_ratio: float, device: torch.device
) -> torch.Tensor:
    """Return the float64 distance [axis-0 pixel] of each pixel from the origin."""
    ny, nx = shape
    f64 = torch.float64
    iy = torch.fft.fftfreq(ny, d=1.0 / ny, dtype=f64, device=device)
    ix = torch.fft.fftfreq(nx, d=1.0 / nx, dtype=f64, device=device)
    return torch.sqrt(iy[:, None] ** 2 + (px_ratio * ix[None, :]) ** 2)


def _pixel_ratio(px_ratio: float | None) -> float:
    """Return the checked pixel width ratio ``s`` [1]; ``None`` gives square pixels."""
    if px_ratio is None:
        return 1.0
    return _checked("px_ratio", px_ratio, zero_ok=False)


def _rim(d: torch.Tensor, radius: float, rim: float) -> torch.Tensor:
    """Return ``1 / (1 + exp(4 (d - R) / w))`` without overflow."""
    return torch.sigmoid(-4.0 * (d - radius) / rim)


def smooth_disc(
    shape: Sequence[int],
    radius_px: float,
    *,
    rim_px: float = RIM_WIDTH_PX,
    px_ratio: float | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return a disc with a logistic rim on a grid in DFT order.

    The value at distance ``d`` is ``1 / (1 + exp(4 (d - R) / w))``, with
    ``d = sqrt(i_y^2 + (s i_x)^2)`` and ``(i_y, i_x)`` the signed DFT indices.

    Parameters
    ----------
    shape : sequence of int
        Grid shape ``(Ny, Nx)``.
    radius_px : float
        Radius ``R`` [axis-0 pixel], finite and zero or larger.
    rim_px : float, optional
        Rim width ``w`` [axis-0 pixel], finite and larger than zero. Default
        ``RIM_WIDTH_PX``.
    px_ratio : float, optional
        Ratio ``s`` of the axis-1 pixel width to the axis-0 pixel width [1],
        finite and larger than zero. ``None`` means square pixels (``s = 1``).
    dtype : torch.dtype, optional
        Real dtype. ``None`` means ``torch.get_default_dtype()``.
    device : torch.device or str, optional
        Device. ``None`` means ``torch.get_default_device()``.

    Returns
    -------
    torch.Tensor
        Shape ``(Ny, Nx)``, values in ``[0, 1]``, no autograd graph.

    Raises
    ------
    TypeError
        If an input has a bad type.
    ValueError
        If an input has a bad value.
    """
    grid = _check_shape(shape)
    radius = _checked("radius_px", radius_px, zero_ok=True)
    rim = _checked("rim_px", rim_px, zero_ok=False)
    ratio = _pixel_ratio(px_ratio)
    out_dtype = _check_dtype(dtype)
    dev = torch.get_default_device() if device is None else torch.device(device)
    with torch.no_grad():
        mask = _rim(_distance(grid, ratio, dev), radius, rim)
        return mask.to(out_dtype)


def smooth_annulus(
    shape: Sequence[int],
    inner_px: float,
    outer_px: float,
    *,
    rim_px: float = RIM_WIDTH_PX,
    px_ratio: float | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return a ring with logistic rims on a grid in DFT order.

    The ring is ``f(d; q) * (1 - f(d; p))``, with ``f`` the disc profile of
    :func:`smooth_disc`, ``p`` the inner and ``q`` the outer radius.

    Parameters
    ----------
    shape : sequence of int
        Grid shape ``(Ny, Nx)``.
    inner_px, outer_px : float
        Radii ``p <= q`` [axis-0 pixel], finite and zero or larger.
    rim_px : float, optional
        Rim width ``w`` of both rims [axis-0 pixel]. Default ``RIM_WIDTH_PX``.
    px_ratio : float, optional
        Pixel width ratio ``s`` [1]. ``None`` means square pixels.
    dtype : torch.dtype, optional
        Real dtype. ``None`` means ``torch.get_default_dtype()``.
    device : torch.device or str, optional
        Device. ``None`` means ``torch.get_default_device()``.

    Returns
    -------
    torch.Tensor
        Shape ``(Ny, Nx)``, values in ``[0, 1]``, no autograd graph.

    Raises
    ------
    TypeError
        If an input has a bad type.
    ValueError
        If an input has a bad value or ``inner_px > outer_px``.
    """
    grid = _check_shape(shape)
    inner = _checked("inner_px", inner_px, zero_ok=True)
    outer = _checked("outer_px", outer_px, zero_ok=True)
    if inner > outer:
        raise ValueError(f"inner_px {inner} must not exceed outer_px {outer}")
    rim = _checked("rim_px", rim_px, zero_ok=False)
    ratio = _pixel_ratio(px_ratio)
    out_dtype = _check_dtype(dtype)
    dev = torch.get_default_device() if device is None else torch.device(device)
    with torch.no_grad():
        d = _distance(grid, ratio, dev)
        mask = _rim(d, outer, rim) * (1.0 - _rim(d, inner, rim))
        return mask.to(out_dtype)


def illumination_disc(
    shape: Sequence[int],
    step_inv_a: Sequence[float],
    opening_rad: float,
    *,
    lambda_a: float,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the disc of a convergent beam on a frequency grid.

    The disc has the radius ``R = a / (lambda g_y)`` [axis-0 pixel], the
    pixel width ratio ``s = g_x / g_y`` and the rim width ``RIM_WIDTH_PX``.
    Thus its rim is at the angle ``a`` in every direction.

    Parameters
    ----------
    shape : sequence of int
        Grid shape ``(Ny, Nx)``.
    step_inv_a : sequence of float
        Frequency steps ``(g_y, g_x)`` [1/Angstrom], finite and positive.
    opening_rad : float
        Semi-angle ``a`` of the condenser opening [rad], zero or larger.
    lambda_a : float
        Wavelength [Angstrom], finite and larger than zero.
    dtype : torch.dtype, optional
        Real dtype. ``None`` means ``torch.get_default_dtype()``.
    device : torch.device or str, optional
        Device. ``None`` means ``torch.get_default_device()``.

    Returns
    -------
    torch.Tensor
        Shape ``(Ny, Nx)``, values in ``[0, 1]``, no autograd graph.

    Raises
    ------
    TypeError
        If an input has a bad type.
    ValueError
        If an input has a bad value.
    """
    if not isinstance(step_inv_a, Sequence) or len(step_inv_a) != 2:
        raise TypeError("step_inv_a must be a sequence of two numbers")
    g_y = _checked("step_inv_a[0]", step_inv_a[0], zero_ok=False)
    g_x = _checked("step_inv_a[1]", step_inv_a[1], zero_ok=False)
    angle = _checked("opening_rad", opening_rad, zero_ok=True)
    lam = _checked("lambda_a", lambda_a, zero_ok=False)
    return smooth_disc(
        shape,
        angle / (lam * g_y),
        rim_px=RIM_WIDTH_PX,
        px_ratio=g_x / g_y,
        dtype=dtype,
        device=device,
    )
