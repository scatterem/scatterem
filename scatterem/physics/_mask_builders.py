"""Private float64 builders of the band-limit masks of ``periodic_sampling.py``.

The FFT index order follows Press et al. (2007), sec. 12.1. The alias-free
ellipse follows Kirkland (2010), sec. 6.8. The edge is the Hann taper of
Harris (1978). Written for scatterem from this literature. The public
docstrings in ``periodic_sampling.py`` give the formulas.
"""

from __future__ import annotations

import operator

import numpy as np
import torch

from ._lru_store import STORE, device_key, real_dtype_key
from ._spectral_filter import raised_cosine


def fft_indices(n: int) -> np.ndarray:
    """Return the signed FFT-order integers ``j`` of an axis with ``n`` points."""
    j = np.arange(n, dtype=np.int64)
    return np.where(j <= (n - 1) // 2, j, j - n)


def count(n) -> int:
    """Return ``n`` as an ``int``; reject ``bool`` with ``TypeError``."""
    if isinstance(n, bool):
        raise TypeError(f"a point count must be an integer, not {n!r}")
    return operator.index(n)


def ellipse_axes(nyquist_fraction) -> tuple[float, float]:
    """Return ``(rho_y, rho_x)`` after a range check (``ValueError``)."""
    if np.ndim(nyquist_fraction) == 0:
        rho = (float(nyquist_fraction),) * 2
    else:
        rho = tuple(float(v) for v in nyquist_fraction)
    if len(rho) != 2 or not all(0.0 < v <= 1.0 for v in rho):
        raise ValueError(
            "nyquist_fraction must be one or two values in (0, 1]; "
            f"got {nyquist_fraction!r}"
        )
    return rho


def build_ellipse(shape, rho, fade_depth, half) -> np.ndarray:
    """Build the float64 ellipse mask on the CPU.

    The cut test ``num < den`` uses index units, so it is exact when the
    semi-axes ``R_i = rho_i N_i / 2`` are integers or half-integers.
    """
    ny, nx = shape
    r_y, r_x = rho[0] * ny / 2.0, rho[1] * nx / 2.0
    jy = fft_indices(ny).astype(np.float64)[:, None]
    if half:
        jx = np.arange(nx // 2 + 1, dtype=np.float64)[None, :]
    else:
        jx = fft_indices(nx).astype(np.float64)[None, :]
    num = (jy * r_x) ** 2 + (jx * r_y) ** 2
    den = (r_y * r_x) ** 2
    inside = num < den
    if fade_depth == 0.0:
        return inside.astype(np.float64)
    depth = 1.0 - np.sqrt(num / den)
    edge = raised_cosine(np.clip(depth, 0.0, fade_depth), fade_depth)
    return np.where(inside, np.where(depth >= fade_depth, 1.0, edge), 0.0)


def circle_radii(grid, cut_fraction, rim_samples) -> tuple[float, float, float]:
    """Return ``(k_c, w, k_f)`` [1/Angstrom] of a circular pass band.

    ``k_c`` depends only on the pixel sizes. ``w`` is in frequency steps
    ``1 / L``, so two grids with equal pixels but different lengths have the
    same ``k_c`` and different ``w`` and ``k_f``.
    """
    k_c = cut_fraction * min(grid.nyquist_frequencies)
    w = rim_samples * max(grid.frequency_steps)
    return k_c, w, k_c - w


def build_circle(points, lengths, radii) -> np.ndarray:
    """Build the float64 circular pass band on the CPU.

    The frequencies ``j / L`` are the same IEEE divisions as in
    ``dft_sample_frequencies``.
    """
    k_c, w, k_f = radii
    ky = (fft_indices(points[0]) / lengths[0])[:, None]
    kx = (fft_indices(points[1]) / lengths[1])[None, :]
    k = np.sqrt(ky**2 + kx**2)
    rim = raised_cosine(np.clip(k_c - k, 0.0, w), w)
    return np.where(k <= k_f, 1.0, np.where(k >= k_c, 0.0, rim))


def stored_ellipse(shape, nyquist_fraction, half, fade_depth, device, dtype):
    """Return the stored ellipse mask. Callers must not change it."""
    ny, nx = (count(n) for n in shape)
    if ny < 1 or nx < 1:
        raise ValueError(f"shape must hold two counts of at least 1; got {shape!r}")
    rho = ellipse_axes(nyquist_fraction)
    w = float(fade_depth)
    if not 0.0 <= w < 1.0:
        raise ValueError(f"fade_depth must be in [0, 1); got {fade_depth!r}")
    dt, dev = real_dtype_key(dtype), device_key(device)
    key = ("ellipse", (ny, nx), rho, w, bool(half), dt, dev)

    def build() -> torch.Tensor:
        arr = build_ellipse((ny, nx), rho, w, bool(half))
        return torch.from_numpy(arr).to(dtype=dt).to(device=dev)

    return STORE.get(key, build)
