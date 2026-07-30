"""Reciprocal-space coordinate grids.

Replaces ``utils/stem.fftfreq2``. The function itself is a thin arrangement of
``torch.fft.fftfreq``, but it lives in a module carrying abTEM (GPL-3.0) and
MetPy (BSD-3) derived code, so the published surface needs it somewhere clean.

One behaviour change: the sampling argument has no mutable default. The
replaced signature was ``dx: List[float] = [1.0, 1.0]``, a shared list that any
caller could mutate for every later caller.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
from torch import Tensor


def fft_frequencies_2d(
    shape: Sequence[int],
    sampling: Sequence[float] = (1.0, 1.0),
    half_pixel_shift: bool = False,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Stacked ``(qy, qx)`` FFT frequencies for a 2D grid.

    Args:
        shape: grid dimensions ``(ny, nx)``.
        sampling: real-space pixel size ``(dy, dx)``; frequencies come out in
            the reciprocal of whatever unit this is given in.
        half_pixel_shift: offset both axes by half a frequency bin. Use for
            quantities defined on pixel centres rather than pixel edges.
        device: device for the returned tensor.
        dtype: floating dtype for the returned tensor; ``torch``'s default if
            omitted.

    Returns:
        A tensor of shape ``(2, ny, nx)`` whose first slice is ``qy`` and second
        is ``qx``, matching ``torch.fft`` axis order.
    """
    ny, nx = int(shape[0]), int(shape[1])
    dy, dx = float(sampling[0]), float(sampling[1])

    qy = torch.fft.fftfreq(ny, d=dy, device=device, dtype=dtype)
    qx = torch.fft.fftfreq(nx, d=dx, device=device, dtype=dtype)
    if half_pixel_shift:
        qy = qy + 0.5 / (ny * dy)
        qx = qx + 0.5 / (nx * dx)

    grid_y, grid_x = torch.meshgrid(qy, qx, indexing="ij")
    return torch.stack([grid_y, grid_x], dim=0)


def frequency_magnitude_2d(
    shape: Sequence[int],
    sampling: Sequence[float] = (1.0, 1.0),
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> Tensor:
    """``|q|`` on a 2D FFT grid, the radial coordinate most callers actually want."""
    q = fft_frequencies_2d(shape, sampling, device=device, dtype=dtype)
    return torch.hypot(q[0], q[1])


def grid_shape_and_sampling(q: Tensor) -> Tuple[Tuple[int, int], Tuple[float, float]]:
    """Recover ``(shape, sampling)`` from a grid produced by :func:`fft_frequencies_2d`.

    Useful when a frequency grid is passed around without its provenance. The
    sampling follows from the largest representable frequency step:
    ``d = 1 / (n * dq)``.
    """
    if q.ndim != 3 or q.shape[0] != 2:
        raise ValueError(f"expected a (2, ny, nx) frequency grid; got {tuple(q.shape)}")
    ny, nx = int(q.shape[1]), int(q.shape[2])
    dqy = float(q[0, 1, 0] - q[0, 0, 0]) if ny > 1 else 0.0
    dqx = float(q[1, 0, 1] - q[1, 0, 0]) if nx > 1 else 0.0
    dy = 1.0 / (ny * dqy) if dqy else 1.0
    dx = 1.0 / (nx * dqx) if dqx else 1.0
    return (ny, nx), (dy, dx)


def radial_average(image, sampling: Sequence[float]):
    """Radial average of a corner-origin 2D array, over FFT frequency bins.

    Replaces ``utils/utils.radial_average2``, which is a py4DSTEM (GPL-3.0)
    derivative. Written here from the definition instead.

    Each pixel contributes to the two bins its ``|q|`` falls between, in
    proportion to how close it lies to each — a linear (area-weighted) partition
    rather than nearest-bin assignment, which is what keeps a radial profile from
    developing sawtooth ripple at small radii where few pixels land per bin. The
    two contributions are accumulated in one pass by concatenating them, so the
    normalising count and the weighted sum each need a single ``bincount``.

    Args:
        image: 2D array with the zero frequency at ``[0, 0]``, as ``torch.fft``
            produces. numpy array or torch tensor.
        sampling: real-space pixel size ``(dy, dx)``; the returned frequencies are
            in the reciprocal of that unit.

    Returns:
        ``(q_bins, profile)``, truncated to the largest frequency represented on
        both axes so the tail is not built from corner pixels alone.

    One deliberate difference from the replaced function: it truncated at the
    ``x`` axis maximum only, so with **anisotropic** sampling it returned bins
    past the shorter axis' Nyquist, where a "ring" is really a pair of corner
    wedges. This truncates at the smaller of the two. On a square grid with equal
    sampling — every case the published figures use — the two agree to machine
    precision (measured max abs difference 2e-15, same bin count); with
    anisotropic sampling this returns a shorter profile, which matters to a caller
    slicing the result by index.
    """
    import numpy as np

    values = np.asarray(
        image.detach().cpu().numpy() if hasattr(image, "detach") else image,
        dtype=np.float64,
    )
    if values.ndim != 2:
        raise ValueError(f"expected a 2D array; got shape {values.shape}")

    ny, nx = values.shape
    qy = np.fft.fftfreq(ny, float(sampling[0]))
    qx = np.fft.fftfreq(nx, float(sampling[1]))
    q = np.hypot(qy[:, None], qx[None, :]).ravel()

    step = float(qy[1] - qy[0]) if ny > 1 else 1.0
    q_bins = np.arange(0.0, q.max() + step, step)

    # Split each pixel between bin `lo` and bin `lo + 1` by the fractional part.
    exact = q / step
    lo = np.floor(exact).astype(np.intp)
    upper_share = exact - lo

    index = np.concatenate((lo, lo + 1))
    share = np.concatenate((1.0 - upper_share, upper_share))
    n_bins = q_bins.size

    count = np.bincount(index, weights=share, minlength=n_bins)[:n_bins]
    total = np.bincount(index, weights=share * np.tile(values.ravel(), 2), minlength=n_bins)[
        :n_bins
    ]
    profile = np.divide(total, count, out=np.zeros_like(total), where=count > 0)

    # Beyond min(|qy|max, |qx|max) only the corners contribute, so the profile
    # there averages an ever-thinner wedge of the plane rather than a ring.
    keep = q_bins <= min(np.abs(qy).max(), np.abs(qx).max())
    return q_bins[keep], profile[keep]
