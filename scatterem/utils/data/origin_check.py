"""Detect whether a diffraction pattern uses the corner-origin convention.

Corner-origin data fed to the FF-STEM path fails SILENTLY: the bright-field disk
straddles the array corners, so the fitted disk radius is meaningless, ``dk`` comes out
badly wrong, and direct ptychography returns an essentially flat image -- with no
exception anywhere. These helpers turn that into a loud error.

Deliberately NOT applied in ``Dataset4dstem.from_array``: iterative ptychography
legitimately consumes corner-origin measurements. The check belongs at the FF-STEM /
reciprocal-calibration entry points, which do assume a centred origin.
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
            2j
            * torch.pi
            * torch.arange(n, dtype=torch.float64, device=p.device)
            / n
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
