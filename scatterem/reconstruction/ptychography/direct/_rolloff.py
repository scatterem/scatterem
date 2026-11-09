"""Raised-cosine roll-off of a phase spectrum at the 2-alpha transfer limit.

:func:`rolloff_2alpha` filters a finished image.
``DirectPtychographyConfig.rolloff_start`` applies it to the phase of the
``"phase"`` and ``"wiener"`` routes.
"""

from collections.abc import Sequence

import numpy as np
import torch

__all__ = ["rolloff_2alpha", "rolloff_weight"]


def _radial_frequency(
    shape: Sequence[int], pixel: Sequence[float], device: torch.device
) -> torch.Tensor:
    """``|q|`` in 1/Angstrom on the FFT grid of an image of ``shape``, float64."""
    qy = torch.fft.fftfreq(shape[0], d=float(pixel[0]), dtype=torch.float64)
    qx = torch.fft.fftfreq(shape[1], d=float(pixel[1]), dtype=torch.float64)
    return torch.hypot(qy[:, None], qx[None, :]).to(device)


def _pixel_pair(pixel: float | Sequence[float]) -> tuple[float, float]:
    """``pixel`` as a ``(y, x)`` pair of positive floats, or raise."""
    pair = np.broadcast_to(np.asarray(pixel, dtype=np.float64), (2,))
    if not np.all(np.isfinite(pair)) or np.any(pair <= 0):
        raise ValueError(f"pixel must be positive and finite, got {pixel!r}")
    return float(pair[0]), float(pair[1])


def rolloff_weight(
    shape: Sequence[int],
    pixel: float | Sequence[float],
    q_edge: float,
    start: float,
    device: torch.device,
) -> torch.Tensor:
    """The spectral weight of :func:`rolloff_2alpha`, per frequency.

    Parameters
    ----------
    shape : sequence of int
        ``(H, W)`` of the image.
    pixel : float or sequence of float
        Pixel size in Angstrom, one value or ``(y, x)``.
    q_edge : float
        Edge frequency in 1/Angstrom.
    start : float
        Start of the roll-off as a fraction of ``q_edge``, in ``[0, 1)``.
    device : torch.device
        Device of the result.

    Returns
    -------
    torch.Tensor
        ``float64`` weight in ``[0, 1]``, shape ``(H, W)``, corner-origin order.

    Raises
    ------
    ValueError
        If ``q_edge`` or ``pixel`` is not positive, or ``start`` is outside
        ``[0, 1)``.
    """
    if not q_edge > 0:
        raise ValueError(f"q_edge must be positive, got {q_edge!r}")
    if not 0.0 <= start < 1.0:
        raise ValueError(f"start must be in [0, 1), got {start!r}")
    q = _radial_frequency(shape, _pixel_pair(pixel), device)
    t = ((q / q_edge - start) / (1.0 - start)).clamp(0.0, 1.0)
    return 0.5 * (1.0 + torch.cos(torch.pi * t))


def rolloff_2alpha(
    image: torch.Tensor,
    pixel: float | Sequence[float],
    q_edge: float,
    start: float = 0.9,
) -> torch.Tensor:
    """Roll the spectrum of an image off to zero at the 2-alpha transfer limit.

    Direct-ptychography phases transfer up to ``2 alpha / lambda``. Near that
    edge the noise rises and the power spectrum shows a bright rim. This
    function multiplies the spectrum by 1 below ``start * q_edge``, by a raised
    cosine from 1 to 0 between ``start * q_edge`` and ``q_edge``, and by 0
    above ``q_edge``. The image mean is kept.

    Parameters
    ----------
    image : torch.Tensor
        Real image, shape ``(..., H, W)``; the roll-off acts on the last two
        axes.
    pixel : float or sequence of float
        Pixel size of ``image`` in Angstrom, one value or ``(y, x)``.
    q_edge : float
        Edge frequency in 1/Angstrom, normally ``2 * alpha / lambda`` (alpha
        in rad, lambda in Angstrom).
    start : float, optional
        Start of the roll-off as a fraction of ``q_edge``, in ``[0, 1)``.
        Default is 0.9.

    Returns
    -------
    torch.Tensor
        The filtered image, same shape, dtype and device as ``image``.

    Raises
    ------
    ValueError
        If ``q_edge`` or ``pixel`` is not positive, or ``start`` is outside
        ``[0, 1)``.
    """
    x = image.to(torch.float64)
    weight = rolloff_weight(tuple(x.shape[-2:]), pixel, q_edge, start, x.device)
    mean = x.mean(dim=(-2, -1), keepdim=True)
    out = mean + torch.fft.ifft2(torch.fft.fft2(x - mean) * weight).real
    return out.to(image.dtype)
