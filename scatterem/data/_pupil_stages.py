"""Front-end stages for the measurement of the bright-field disk.

The stages are a 3 x 3 running median (Tukey 1977), the iterative intermeans
threshold (Ridler and Calvard 1978), a coarse disk from the thresholded
median image, the cleaning of single bright pixels, smooth window functions
and the plausibility checks that refuse a pattern without a clear disk.

Literature: Tukey, J. W. (1977), *Exploratory Data Analysis* (running
medians); Ridler, T. W. and Calvard, S. (1978), Picture thresholding using an
iterative selection method, IEEE Trans. SMC-8, 630-632. This code was
written for scatterem from this literature.
"""

from __future__ import annotations

import logging
import math
from typing import NamedTuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Smallest fraction of the coarse circle that the thresholded disk fills.
# Disks give 0.8 to 1.0, median-filtered noise gives about 0.5 or less.
_MIN_COMPACTNESS = 0.7
# Smallest coarse radius [pixel] that counts as a disk.
_MIN_RADIUS = 3.0


class CoarseDisk(NamedTuple):
    """Result of the front end.

    Attributes
    ----------
    cleaned : torch.Tensor
        The pattern with single bright pixels replaced by the local median.
    center : tuple of float
        The centroid ``(y, x)`` of the thresholded disk [pixel].
    radius : float
        ``sqrt(area / pi)`` of the thresholded disk [pixel].
    low, high : float
        The two class means of the intermeans threshold.
    """

    cleaned: torch.Tensor
    center: tuple[float, float]
    radius: float
    low: float
    high: float


def median3(image: torch.Tensor) -> torch.Tensor:
    """Return the 3 x 3 running median of a 2-D image.

    The border is extended by replication. A group of at most 4 bright
    pixels in a 3 x 3 window does not change the median.

    Parameters
    ----------
    image : torch.Tensor
        Shape ``(Ny, Nx)``, real.

    Returns
    -------
    torch.Tensor
        Shape ``(Ny, Nx)``, the median image.
    """
    padded = F.pad(image[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
    windows = padded.unfold(0, 3, 1).unfold(1, 3, 1)
    return windows.reshape(*image.shape, 9).median(dim=-1).values


def intermeans(image: torch.Tensor) -> tuple[float, float, float]:
    """Return the intermeans threshold and the two class means.

    Start at the middle of the value range. Repeat: set the threshold to the
    mean of the two class means, until it changes by at most 1e-9 of the
    range, or 100 times (Ridler and Calvard 1978).

    Parameters
    ----------
    image : torch.Tensor
        Real image.

    Returns
    -------
    tuple of float
        ``(threshold, low, high)``.

    Raises
    ------
    ValueError
        If the image has no contrast.
    """
    v_min = float(image.min().item())
    v_max = float(image.max().item())
    if not v_max - v_min > 1e-12 * max(abs(v_max), abs(v_min)):
        raise ValueError(
            f"pupil_frame: no contrast; the pattern is constant (value {v_max:g}) "
            "after a 3 x 3 median filter"
        )
    thr = 0.5 * (v_min + v_max)
    low = high = thr
    for _ in range(100):
        upper = image > thr
        high = float(image[upper].mean().item())
        low = float(image[~upper].mean().item())
        new = 0.5 * (low + high)
        done = abs(new - thr) <= 1e-9 * (v_max - v_min)
        thr = new
        if done:
            break
    return thr, low, high


def front_end(pattern: torch.Tensor, radius_cap: float) -> CoarseDisk:
    """Find a coarse disk, clean single pixels and check plausibility.

    Parameters
    ----------
    pattern : torch.Tensor
        Shape ``(Ny, Nx)``, float64.
    radius_cap : float
        Largest plausible radius, relative to ``min(Ny, Nx)``.

    Returns
    -------
    CoarseDisk
        The coarse disk and the cleaned pattern.

    Raises
    ------
    ValueError
        If the pattern has no contrast, no compact disk, or a disk that is
        too small or too large.
    """
    med = median3(pattern)
    thr, low, high = intermeans(med)
    mask = med > thr
    area = float(mask.sum().item())
    yy, xx = pixel_grid(pattern.shape, pattern.device)
    weight = mask.to(torch.float64)
    cy = float((weight * yy).sum().item()) / area
    cx = float((weight * xx).sum().item()) / area
    radius = math.sqrt(area / math.pi)
    if radius < _MIN_RADIUS:
        raise ValueError(
            f"pupil_frame: no disk found; the bright region has a radius of "
            f"{radius:.2f} pixel, below {_MIN_RADIUS} pixel"
        )
    inside = (yy - cy) ** 2 + (xx - cx) ** 2 < radius**2
    compact = float((mask & inside).sum().item()) / (math.pi * radius**2)
    if compact < _MIN_COMPACTNESS:
        raise ValueError(
            f"pupil_frame: no compact disk found; the bright pixels fill {compact:.2f} "
            f"of their equivalent circle (a disk fills at least "
            f"{_MIN_COMPACTNESS})"
        )
    check_radius(radius, pattern.shape, radius_cap)
    cleaned = torch.where((pattern - med).abs() <= high - low, pattern, med)
    logger.debug(
        "coarse disk: centre (%.3f, %.3f), radius %.3f, levels %g / %g, "
        "compactness %.3f",
        cy,
        cx,
        radius,
        low,
        high,
        compact,
    )
    return CoarseDisk(cleaned, (cy, cx), radius, low, high)


def pixel_grid(
    shape: tuple[int, ...], device: torch.device, offset: tuple[int, int] = (0, 0)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the float64 pixel coordinates ``y`` and ``x`` of a 2-D array.

    Parameters
    ----------
    shape : tuple of int
        ``(Ny, Nx)``.
    device : torch.device
        The device of the result.
    offset : tuple of int, optional
        Coordinates [pixel] of element ``(0, 0)``.

    Returns
    -------
    tuple of torch.Tensor
        ``y`` of shape ``(Ny, 1)`` and ``x`` of shape ``(1, Nx)`` [pixel].
    """
    y = torch.arange(shape[0], dtype=torch.float64, device=device) + offset[0]
    x = torch.arange(shape[1], dtype=torch.float64, device=device) + offset[1]
    return y[:, None], x[None, :]


def check_radius(radius: float, shape: tuple[int, ...], radius_cap: float) -> None:
    """Refuse a radius above the largest plausible radius.

    Raises
    ------
    ValueError
        If ``radius > radius_cap * min(shape)``.
    """
    limit = radius_cap * min(shape[0], shape[1])
    if radius > limit:
        raise ValueError(
            f"pupil_frame: the disk radius {radius:.2f} pixel is above radius_cap="
            f"{radius_cap:g} times the shorter side ({limit:.2f} pixel); the "
            "disk fills nearly the whole detector"
        )


def check_levels(inside: float, outside: float, contrast_floor: float) -> None:
    """Refuse levels whose ratio is below the contrast floor.

    A non-positive outside level with a larger inside level counts as an
    infinite ratio.

    Raises
    ------
    ValueError
        If ``inside <= outside`` or the ratio is below ``contrast_floor``.
    """
    if inside <= outside:
        raise ValueError(
            f"pupil_frame: no disk; the inside level {inside:g} is not above the "
            f"outside level {outside:g}"
        )
    if outside > 0 and inside / outside < contrast_floor:
        raise ValueError(
            f"pupil_frame: the ratio of the inside level {inside:g} to the outside "
            f"level {outside:g} is {inside / outside:.3f}, below "
            f"contrast_floor={contrast_floor:g}"
        )


def ramp(z: torch.Tensor) -> torch.Tensor:
    """Return a smooth step: 0 for ``z <= 0``, 1 for ``z >= 1``, cosine between."""
    return 0.5 - 0.5 * torch.cos(math.pi * z.clamp(0.0, 1.0))


def edge_window(r: torch.Tensor, radius: float, half: float, soft: float):
    """Return the symmetric edge window ``g(r)``.

    ``g = 1`` on ``[radius - half, radius + half]``, with cosine ramps of
    width ``soft`` on both sides. Its integral over ``r`` is
    ``2 half + soft``.
    """
    return 1.0 - ramp(((r - radius).abs() - half) / soft)
