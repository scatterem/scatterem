"""Radius and centre of the bright-field disk in a diffraction pattern.

In 4D-STEM the mean diffraction pattern holds the bright-field disk, the
image of the probe-forming aperture (the pupil of the probe-forming lens).
Its radius ``R`` [pixel], the aperture semi-angle ``alpha`` [rad] and the
wavelength ``lambda`` [length] calibrate the detector: one pixel is
``alpha / (lambda R)`` [1/length] (Kirkland 2020; Zuo and Spence 2017). Its
centre gives the optical axis on the detector.

Two estimators measure the disk. Both use the normalized intensity
``u = (P - I_out) / (I_in - I_out)`` and the odd-edge integral: for a window
``g`` that is symmetric about ``rho`` and flat over the edge,
``int u g dr = int_0^R g dr``. This gives the radius of the half level
``(I_in + I_out) / 2`` for any edge shape that is odd about that level.

- ``"integral"`` (default) integrates over all pixels near the edge. The
  centre is the centroid of ``u`` in a smooth window. It needs a complete
  disk.
- ``"rayfit"`` integrates ``u`` along rays from the centre, which gives one
  edge point per ray. A robust geometric circle fit then rejects the rays
  that cross a beam stop or leave the detector.

Literature: Kirkland, E. J. (2020), *Advanced Computing in Electron
Microscopy*, 3rd ed.; Zuo, J. M. and Spence, J. C. H. (2017), *Advanced
Transmission Electron Microscopy*; Ridler and Calvard (1978); Tukey (1977);
and, for the circle fit, Kasa (1976), Pratt (1987), Taubin (1991), Chernov
(2010), Al-Sharadqah and Chernov (2009), Huber and Ronchetti (2009),
Rousseeuw and Leroy (1987). This code was written for scatterem from this
literature.
"""

from __future__ import annotations

import logging
import math
from typing import Any, NamedTuple

import numpy as np
import torch

from scatterem.data._pupil_stages import (
    check_levels,
    check_radius,
    edge_window,
    front_end,
    pixel_grid,
    ramp,
)
from scatterem.data._ring_regression import robust_circle
from scatterem.utils._checks import float64_tensor, real_scalar

__all__ = ["ESTIMATORS", "PupilEstimate", "measure_pupil"]

logger = logging.getLogger(__name__)

ESTIMATORS = ("integral", "rayfit")

_ITERATIONS = 4
_RAY_STEP = 0.25  # [pixel]
_MIN_RAYS = 64
_MIN_VALID_RAYS = 8
_DEFAULT_COVERAGE = 0.4


class PupilEstimate(NamedTuple):
    """Radius and centre of the bright-field disk.

    Parameters
    ----------
    rim_radius : float
        The radius of the half-level circle [pixel].
    rim_center : numpy.ndarray
        Shape ``(2,)``, float64, the centre ``(y, x)`` [pixel].
    """

    rim_radius: float
    rim_center: np.ndarray


def _band(radius: float, edge_reach: float) -> tuple[float, float]:
    """Return the half-width ``h`` and the ramp width ``a`` of the edge band."""
    half = min(edge_reach, 0.6 * radius)
    return half, 0.5 * half


def _region_masks(r: torch.Tensor, radius: float, half: float):
    """Return the masks of the inside and the outside level regions."""
    inner = r < radius - half
    start = radius + half
    outer = (r >= start) & (r < start + max(half, 0.5 * radius))
    return inner, outer


def _levels(cleaned, r, radius, half, robust: bool) -> tuple[float, float]:
    """Return ``(I_in, I_out)`` from the regions inside and outside the edge."""
    inner, outer = _region_masks(r, radius, half)
    n_in = int(inner.sum().item())
    n_out = int(outer.sum().item())
    if n_in < 4:
        raise ValueError(
            f"pupil_frame: the region inside the disk edge has {n_in} pixels, fewer "
            "than 4; the disk is too small for the edge band"
        )
    if n_out < 16:
        raise ValueError(
            f"pupil_frame: the region outside the disk edge has {n_out} pixels on the "
            "detector, fewer than 16; increase the detector area around the disk"
        )
    if robust:
        return (
            float(cleaned[inner].median().item()),
            float(cleaned[outer].median().item()),
        )
    return float(cleaned[inner].mean().item()), float(cleaned[outer].mean().item())


def _crop(shape, center, radius, edge_reach) -> tuple[slice, slice, tuple[int, int]]:
    """Return the box around the coarse disk that the later stages use."""
    margin = 1.5 * radius + 3.0 * edge_reach + 4.0
    y0 = max(0, math.floor(center[0] - margin))
    y1 = min(shape[0], math.ceil(center[0] + margin) + 1)
    x0 = max(0, math.floor(center[1] - margin))
    x1 = min(shape[1], math.ceil(center[1] + margin) + 1)
    return slice(y0, y1), slice(x0, x1), (y0, x0)


def _integral_estimate(coarse, shape, edge_reach, contrast_floor):
    """Run the ``"integral"`` estimator; return ``(radius, (c_y, c_x))``."""
    rows, cols, offset = _crop(shape, coarse.center, coarse.radius, edge_reach)
    cleaned = coarse.cleaned[rows, cols]
    yy, xx = pixel_grid(cleaned.shape, cleaned.device, offset)
    cy, cx = coarse.center
    radius = coarse.radius
    for _ in range(_ITERATIONS):
        half, soft = _band(radius, edge_reach)
        r = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        i_in, i_out = _levels(cleaned, r, radius, half, robust=False)
        check_levels(i_in, i_out, contrast_floor)
        u = (cleaned - i_out) / (i_in - i_out)
        uw = u * (1.0 - ramp((r - radius - half) / soft))
        total = float(uw.sum().item())
        if not total > 0:
            raise ValueError(
                "pupil_frame: the disk has no positive weight; no disk found"
            )
        cy = float((uw * yy).sum().item()) / total
        cx = float((uw * xx).sum().item()) / total
        r = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        g = edge_window(r, radius, half, soft)
        integral = float((u * g / r.clamp_min(1e-12)).sum().item()) / (2 * math.pi)
        radius = radius + integral - half - 0.5 * soft
        if not radius > 0:
            raise ValueError(
                "pupil_frame: the radius estimate is not positive; no disk"
            )
    logger.debug("integral: levels %g / %g, band half-width %.3f", i_in, i_out, half)  # pyright: ignore[reportPossiblyUnboundVariable]
    if (
        cy - radius - 1 < 0
        or cx - radius - 1 < 0
        or cy + radius + 1 > shape[0] - 1
        or cx + radius + 1 > shape[1] - 1
    ):
        raise ValueError(
            f"pupil_frame: the disk (centre ({cy:.2f}, {cx:.2f}), radius {radius:.2f}) is "
            "not complete on the detector; use estimator='rayfit'"
        )
    return radius, (cy, cx)


def _bilinear(image: torch.Tensor, y: torch.Tensor, x: torch.Tensor):
    """Interpolate a 2-D image bilinearly; return values and an inside mask."""
    ny, nx = image.shape
    inside = (y >= 0) & (y <= ny - 1) & (x >= 0) & (x <= nx - 1)
    yc = y.clamp(0, ny - 1)
    xc = x.clamp(0, nx - 1)
    y0 = torch.floor(yc).clamp(max=max(ny - 2, 0))
    x0 = torch.floor(xc).clamp(max=max(nx - 2, 0))
    fy = yc - y0
    fx = xc - x0
    iy = y0.to(torch.int64)
    ix = x0.to(torch.int64)
    value = (
        image[iy, ix] * (1 - fy) * (1 - fx)
        + image[iy + 1, ix] * fy * (1 - fx)
        + image[iy, ix + 1] * (1 - fy) * fx
        + image[iy + 1, ix + 1] * fy * fx
    )
    return value, inside


def _ray_edge_points(u, center, radius, half, soft):
    """Return the edge point of each ray, the validity mask and the ray count.

    The edge point of a ray is at ``R + int u g dr - half - soft / 2`` from
    the centre (odd-edge integral along the ray). A ray is valid only if all
    of its samples are inside the detector.
    """
    n_rays = max(_MIN_RAYS, 4 * math.ceil(2 * math.pi * (radius + half) / 4))
    dev = u.device
    theta = 2 * math.pi * torch.arange(n_rays, dtype=torch.float64, device=dev)
    theta = theta / n_rays
    r_start = max(radius - half - soft, 0.0)
    n_samples = math.ceil((radius + half + soft - r_start) / _RAY_STEP)
    steps = torch.arange(n_samples, dtype=torch.float64, device=dev) + 0.5
    rr = r_start + _RAY_STEP * steps
    sin, cos = torch.sin(theta)[:, None], torch.cos(theta)[:, None]
    values, inside = _bilinear(u, center[0] + rr * sin, center[1] + rr * cos)
    g = edge_window(rr, radius, half, soft)
    rho = radius + _RAY_STEP * (values * g).sum(dim=1) - half - 0.5 * soft
    points = torch.stack((center[0] + rho * sin[:, 0], center[1] + rho * cos[:, 0]), 1)
    return points, rho, inside.all(dim=1), n_rays


def _rayfit_estimate(coarse, shape, edge_reach, contrast_floor, arc_floor):
    """Run the ``"rayfit"`` estimator; return ``(radius, (c_y, c_x))``."""
    cleaned = coarse.cleaned
    yy, xx = pixel_grid(cleaned.shape, cleaned.device)
    center = coarse.center
    radius = coarse.radius
    for _ in range(_ITERATIONS):
        half, soft = _band(radius, edge_reach)
        r = torch.sqrt((yy - center[0]) ** 2 + (xx - center[1]) ** 2)
        i_in, i_out = _levels(cleaned, r, radius, half, robust=True)
        check_levels(i_in, i_out, contrast_floor)
        u = (cleaned - i_out) / (i_in - i_out)
        points, rho, valid, n_rays = _ray_edge_points(u, center, radius, half, soft)
        n_valid = int(valid.sum().item())
        if n_valid < _MIN_VALID_RAYS:
            raise ValueError(
                f"rayfit: only {n_valid} of {n_rays} rays are inside the "
                f"detector, fewer than {_MIN_VALID_RAYS}; the disk edge is "
                "missing"
            )
        start = (center[0], center[1], float(rho[valid].median().item()))
        fit = robust_circle(points, valid, start)
        center, radius = fit.center, fit.radius
        if not radius > 0:
            raise ValueError("rayfit: the fitted radius is not positive; no disk")
    coverage = float((fit.weights > 0).sum().item()) / n_rays  # pyright: ignore[reportPossiblyUnboundVariable]
    logger.debug(
        "rayfit: levels %g / %g, band half-width %.3f, coverage %.3f",
        i_in,  # pyright: ignore[reportPossiblyUnboundVariable]
        i_out,  # pyright: ignore[reportPossiblyUnboundVariable]
        half,  # pyright: ignore[reportPossiblyUnboundVariable]
        coverage,
    )
    if coverage < arc_floor:
        raise ValueError(
            f"rayfit: edge coverage {coverage:.2f} of the full circle is below "
            f"arc_floor={arc_floor:g}; too much of the disk edge is hidden "
            "or outside the detector"
        )
    return radius, center


# Default half-width of the edge band [pixel] per estimator. "integral": 2,
# because a wide band reads a disk with a sloped top too small (Gd2O3 mean
# pattern: -4.1 % at 10 px, -0.5 % at 2 px against the half-level crossing).
# "rayfit": 10, because a narrow band misses a disk with a 90-degree notch
# (planted R = 24 px: -10.7 % and a centre 4 px off at 2 px, -0.03 % at 10).
_DEFAULT_EDGE_REACH = {"integral": 2.0, "rayfit": 10.0}


def _settings(estimator, radius_cap, contrast_floor, edge_reach, arc_floor):
    """Validate the settings; return them as floats."""
    if not isinstance(estimator, str):
        raise TypeError(f"estimator: expected a str, got {estimator!r}")
    if estimator not in ESTIMATORS:
        raise ValueError(f"estimator: expected one of {ESTIMATORS}, got {estimator!r}")
    cap = real_scalar(radius_cap, "radius_cap")
    if not 0 < cap <= 1:
        raise ValueError(f"radius_cap: expected a value in (0, 1], got {cap}")
    floor = real_scalar(contrast_floor, "contrast_floor")
    if not (math.isfinite(floor) and floor >= 1):
        raise ValueError(f"contrast_floor: expected a finite value >= 1, got {floor}")
    if edge_reach is None:
        edge_reach = _DEFAULT_EDGE_REACH[estimator]
    reach = real_scalar(edge_reach, "edge_reach")
    if not (math.isfinite(reach) and reach > 0):
        raise ValueError(f"edge_reach: expected a finite value > 0, got {reach}")
    if arc_floor is None:
        coverage = _DEFAULT_COVERAGE
    else:
        if estimator != "rayfit":
            raise ValueError("arc_floor applies only to estimator='rayfit'")
        coverage = real_scalar(arc_floor, "arc_floor")
        if not 0 < coverage <= 1:
            raise ValueError(f"arc_floor: expected a value in (0, 1], got {coverage}")
    return cap, floor, reach, coverage


def _pattern(pupil_frame: Any) -> torch.Tensor:
    """Validate the pattern and return it as a float64 tensor."""
    if isinstance(pupil_frame, np.ndarray | torch.Tensor) and pupil_frame.ndim != 2:
        raise ValueError(
            "pupil_frame: expected one 2-D pattern (Ny, Nx), for example the mean over "
            f"all probe positions; got shape {tuple(pupil_frame.shape)}"
        )
    pattern = float64_tensor(pupil_frame, "pupil_frame")
    if pattern.numel() == 0:
        raise ValueError(
            f"pupil_frame: the pattern is empty, shape {tuple(pattern.shape)}"
        )
    return pattern


def measure_pupil(
    pupil_frame: Any,
    estimator: str = "integral",
    *,
    radius_cap: float = 0.45,
    contrast_floor: float = 1.5,
    edge_reach: float | None = None,
    arc_floor: float | None = None,
) -> PupilEstimate:
    """Measure the radius and the centre of the bright-field disk.

    The disk is the image of the probe-forming aperture (the pupil). Its
    radius is the radius of the circle on which the intensity is
    ``(I_in + I_out) / 2``. One detector pixel then corresponds to
    ``alpha / (lambda * radius)`` [1/length], for the aperture semi-angle
    ``alpha`` [rad] and the wavelength ``lambda`` [length].

    A 3 x 3 median filter and the intermeans threshold give a coarse disk.
    Pixels that differ from their local median by more than the contrast are
    replaced by that median, which removes bright defective pixels. The
    levels ``I_in`` and ``I_out`` come from the regions inside and outside
    the edge band. All steps are equivariant under an added constant and a
    positive factor, so the result does not change with them. The function
    uses no random numbers.

    Parameters
    ----------
    pupil_frame : numpy.ndarray or torch.Tensor
        One 2-D pattern ``(Ny, Nx)`` of intensities, integer or real, for
        example the mean over all probe positions. A tensor can be on any
        device and can have autograd enabled; it is only read.
    estimator : {"integral", "rayfit"}, optional
        ``"integral"`` (default) is for complete disks. ``"rayfit"`` also
        measures a disk with a missing sector or with a part outside the
        detector. Each estimator raises an error if it cannot give a result;
        it never gives the result of the other estimator.
    radius_cap : float, optional
        The largest plausible radius, relative to the shorter detector side,
        in ``(0, 1]``. Default 0.45: it accepts all complete disks that keep
        a margin to the border and refuses a disk that fills nearly the
        whole detector. Use a value up to 1 for a pattern cut down to the
        disk.
    contrast_floor : float, optional
        The smallest ratio ``I_in / I_out`` that counts as a disk, at least
        1. Default 1.5. If ``I_out <= 0`` the ratio counts as infinite.
    edge_reach : float, optional
        The half-width of the band around the edge [pixel] that the
        estimators integrate over. The band uses ``min(edge_reach, 0.6 R)``.
        Default (None): 2 for ``"integral"``, 10 for ``"rayfit"``. A wide
        band reads a disk with a sloped top too small: on the Gd2O3 mean
        pattern a 10-pixel band gives a radius 4 % small, a 2-pixel band
        0.5 %. ``"rayfit"`` needs the wide band on a disk with a large
        missing sector: with 2 pixels it reads a disk with a 90-degree notch
        11 % small. Increase it for soft edges: 10 covers edges up to a tanh
        width of 4 pixels.
    arc_floor : float, optional
        Only for ``"rayfit"``: the smallest fraction of the full circle,
        in ``(0, 1]``, that the accepted edge points must cover. Default
        (None) 0.4.

    Returns
    -------
    PupilEstimate
        ``(rim_radius, rim_center)``: the radius [pixel] as a float, and the centre
        ``(y, x)`` [pixel] as a new NumPy float64 array of shape ``(2,)``.

    Raises
    ------
    TypeError
        If ``pupil_frame`` is complex, bool or not an array, or a setting has a
        wrong type.
    ValueError
        If ``pupil_frame`` is not 2-D or is empty, a setting is out of range, or the
        pattern has no clear disk (no contrast, no compact disk, a radius
        above ``radius_cap``, a level ratio below ``contrast_floor``, an
        incomplete disk for ``"integral"``, or an edge coverage below
        ``arc_floor`` for ``"rayfit"``).
    """
    cap, floor, reach, coverage = _settings(
        estimator, radius_cap, contrast_floor, edge_reach, arc_floor
    )
    pattern = _pattern(pupil_frame)
    shape = tuple(pattern.shape)
    coarse = front_end(pattern, cap)
    if estimator == "integral":
        radius, center = _integral_estimate(coarse, shape, reach, floor)
    else:
        radius, center = _rayfit_estimate(coarse, shape, reach, floor, coverage)
    check_radius(radius, shape, cap)
    return PupilEstimate(float(radius), np.array(center, dtype=np.float64))
