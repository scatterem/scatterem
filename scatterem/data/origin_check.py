"""Detect whether a diffraction pattern uses the corner-origin convention.

Corner-origin data fed to the FF-STEM path fails SILENTLY: the bright-field disk
straddles the array corners, so the fitted disk radius is meaningless, ``dk`` comes out
badly wrong, and direct ptychography returns an essentially flat image -- with no
exception anywhere. These helpers turn that into a loud error.

All of scatterem's data entry points take centred patterns (zero frequency at pixel
``(My // 2, Mx // 2)``). The check runs where a consumer depends on that convention:
the FF-STEM / reciprocal-calibration entry points.
It is NOT applied in ``Dataset4DStem.from_array``.
"""

from __future__ import annotations

import torch


def circular_centroid(pattern: torch.Tensor) -> tuple[float, float]:
    """Wrapped first moment of a 2-D pattern, in pixels, one value per axis.

    Computed as the phase of the first Fourier coefficient along each axis, so a disk
    straddling the array edge reports a position near 0 rather than near the middle.

    A PLAIN intensity-weighted centroid is unusable for this job: for mass split
    symmetrically across the four corners it lands exactly at the array centre, which is
    indistinguishable from genuinely centred data.
    """
    p = pattern.detach().to(torch.float64)
    if p.ndim != 2:
        raise ValueError(f"expected a 2-D pattern, got shape {tuple(p.shape)}")
    # A constant background contributes ~0 to the first Fourier coefficient, but
    # removing it still improves contrast on low-dynamic-range data.
    p = torch.clamp(p - p.min(), min=0.0)

    out: list[float] = []
    for axis in (0, 1):
        profile = p.sum(dim=1 - axis).to(torch.complex128)
        n = profile.shape[0]
        k = torch.exp(
            2j * torch.pi * torch.arange(n, dtype=torch.float64, device=p.device) / n
        )
        phase = torch.angle((profile * k).sum())
        out.append(float((phase / (2 * torch.pi) * n) % n))
    return out[0], out[1]


def _toroidal_distance_from_centre(pattern: torch.Tensor) -> float:
    """Distance of the circular centroid from the fftshift centre, wrapping at the edges.

    The centre is ``n // 2`` on each axis -- where ``fftshift`` places the origin --
    not ``(n - 1) / 2``.
    """
    ny, nx = pattern.shape[-2:]
    y, x = circular_centroid(pattern)
    dy = abs(y - ny // 2)
    dx = abs(x - nx // 2)
    dy = min(dy, ny - dy)
    dx = min(dx, nx - dx)
    return float((dy**2 + dx**2) ** 0.5)


def corner_origin_margin(mean_pattern: torch.Tensor) -> float:
    """``raw_distance - fftshifted_distance``, in pixels. Positive => corner-origin.

    Exposed so callers and tests can assert a margin rather than only a boolean, and so
    an error message can quantify how far off the data is.
    """
    raw = _toroidal_distance_from_centre(mean_pattern)
    shifted = _toroidal_distance_from_centre(
        torch.fft.fftshift(mean_pattern, dim=(-2, -1))
    )
    return raw - shifted


def is_corner_origin(mean_pattern: torch.Tensor) -> bool:
    """True when an ``fftshift`` would move the intensity closer to the array centre.

    Threshold-free and relative, which is what keeps a merely off-centre disk -- the
    case guard G3 recentres -- from being mistaken for a corner-origin cube. The
    effective boundary is a disk offset of ``n / 4``.
    """
    return corner_origin_margin(mean_pattern) > 0.0


def first_harmonic_contrast(pattern: torch.Tensor) -> float:
    """How concentrated a 2-D pattern is, from 0 (flat or noise) to 1 (a point).

    The ratio ``|sum(p * exp(2 pi i y / n))| / sum(p)`` of the projected profile,
    after the minimum is removed, taken on the less concentrated axis. For a
    uniform disk of radius ``r`` it is ``2 J1(x) / x`` with ``x = 2 pi r / n``:
    about 0.72 at ``r = n / 4``, 0.18 when the disk touches the array edges, and
    near 0 when it fills the array or for noise. Below about 0.1 the centroid in
    :func:`circular_centroid` is not meaningful, so neither is
    :func:`is_corner_origin`.
    """
    p = pattern.detach().to(torch.float64)
    if p.ndim != 2:
        raise ValueError(f"expected a 2-D pattern, got shape {tuple(p.shape)}")
    p = torch.clamp(p - p.min(), min=0.0)
    total = float(p.sum())
    if total <= 0.0:
        return 0.0
    out = []
    for axis in (0, 1):
        profile = p.sum(dim=1 - axis).to(torch.complex128)
        n = profile.shape[0]
        k = torch.exp(
            2j * torch.pi * torch.arange(n, dtype=torch.float64, device=p.device) / n
        )
        out.append(float((profile * k).sum().abs()) / total)
    return min(out)


def is_clearly_corner_origin(
    mean_pattern: torch.Tensor, min_contrast: float = 0.1
) -> bool:
    """True when the pattern has a clear disk AND that disk is corner-origin.

    :func:`is_corner_origin` alone decides by chance on a pattern with no
    concentrated disk: noise, or a bright-field disk wider than the detector
    (a detector cropped to the disk). This version answers False for those, so
    a caller that raises on True never refuses data it cannot judge.

    Parameters
    ----------
    mean_pattern : torch.Tensor
        ``(My, Mx)`` mean diffraction pattern.
    min_contrast : float
        Smallest :func:`first_harmonic_contrast` that counts as a clear disk.
        Default 0.1, which a uniform disk keeps up to a radius of about
        ``0.53 * n``.

    Returns
    -------
    bool
        True if the pattern is clearly corner-origin.
    """
    if first_harmonic_contrast(mean_pattern) < min_contrast:
        return False
    return is_corner_origin(mean_pattern)
