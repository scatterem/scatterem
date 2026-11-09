"""Frequency grids of the discrete Fourier transform and band-limit masks.

Literature: the frequency order of the DFT follows Press et al. (2007),
*Numerical Recipes*, sec. 12.1, and Bracewell (2000), ch. 10. The alias-free
band limit follows Kirkland (2010), *Advanced Computing in Electron
Microscopy*, sec. 6.8, and Kirkland, Loane and Silcox (1987). The smooth edge
is the Hann taper of Harris (1978). Written for scatterem from this
literature.

Units: lengths [Angstrom], spatial frequencies [1/Angstrom] without a factor
``2 pi``. Masks are in FFT order (zero frequency at index 0).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import torch

from ._lru_store import STORE, MaskCacheStats, device_key, real_dtype_key
from ._mask_builders import (
    build_circle,
    circle_radii,
    count,
    fft_indices,
    stored_ellipse,
)
from ._spectral_filter import filter_in_fourier_space, real_dtype_of

__all__ = [
    "CircularPassBand",
    "MaskCacheStats",
    "PeriodicGrid",
    "apply_pass_band",
    "clear_mask_cache",
    "dft_sample_frequencies",
    "ellipse_mask",
    "lowpass_ellipse",
    "mask_cache_stats",
]

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PeriodicGrid:
    """A periodic sampling grid: point counts and periods per axis.

    Parameters
    ----------
    points : tuple of int
        Number of samples ``N_i`` per axis, axis order ``(..., y, x)``.
    lengths : tuple of float
        Period ``L_i`` of each axis [Angstrom].

    Raises
    ------
    ValueError
        If the tuples differ in length, are empty, if a count is not an
        integer of at least 1, or if a period is not finite and positive.

    Notes
    -----
    The grid stores the period, not the pixel size. One IEEE division of the
    exact integer ``j`` by ``L`` then gives the correctly rounded ``j / L``.
    The object is hashable.
    """

    points: tuple[int, ...]
    lengths: tuple[float, ...]

    def __post_init__(self) -> None:
        try:
            pts = tuple(count(n) for n in self.points)
            lens = tuple(float(v) for v in self.lengths)
        except TypeError as err:
            raise ValueError(f"invalid grid description: {err}") from None
        if len(pts) == 0 or len(pts) != len(lens):
            raise ValueError(
                f"points {pts} and lengths {lens} must be non-empty and of equal size"
            )
        if any(n < 1 for n in pts):
            raise ValueError(f"every point count must be at least 1; got {pts}")
        if any(not (math.isfinite(v) and v > 0) for v in lens):
            raise ValueError(f"every length must be finite and > 0; got {lens}")
        object.__setattr__(self, "points", pts)
        object.__setattr__(self, "lengths", lens)

    @classmethod
    def from_pixel_sizes(cls, points, pixel_sizes) -> PeriodicGrid:
        """Make a grid from point counts and pixel sizes.

        Parameters
        ----------
        points : sequence of int
            Number of samples per axis.
        pixel_sizes : sequence of float
            Sample spacing per axis [Angstrom].

        Returns
        -------
        PeriodicGrid
            The grid with ``L_i = N_i * d_i``, rounded once. The stored
            ``L_i`` is the reference for all frequencies.
        """
        pts = tuple(count(n) for n in points)
        sizes = tuple(float(d) for d in pixel_sizes)
        if len(pts) != len(sizes):
            raise ValueError(f"points {pts} and pixel_sizes {sizes} differ in size")
        return cls(pts, tuple(n * d for n, d in zip(pts, sizes, strict=True)))

    @property
    def ndim(self) -> int:
        """Number of axes."""
        return len(self.points)

    @property
    def pixel_sizes(self) -> tuple[float, ...]:
        """Sample spacing ``L_i / N_i`` per axis [Angstrom]."""
        return tuple(v / n for v, n in zip(self.lengths, self.points, strict=True))

    @property
    def frequency_steps(self) -> tuple[float, ...]:
        """Frequency step ``1 / L_i`` per axis [1/Angstrom]."""
        return tuple(1.0 / v for v in self.lengths)

    @property
    def nyquist_frequencies(self) -> tuple[float, ...]:
        """Nyquist frequency ``N_i / (2 L_i)`` per axis [1/Angstrom]."""
        return tuple(
            n / (2.0 * v) for v, n in zip(self.lengths, self.points, strict=True)
        )


def dft_sample_frequencies(grid: PeriodicGrid) -> tuple[np.ndarray, ...]:
    """Return the spatial frequencies that the DFT samples on a grid.

    On an axis with ``N`` points and period ``L`` [Angstrom], the DFT samples
    the spectrum at ``j / L`` for the FFT-order integers ``j``: ``0, 1, ...,
    (N - 1) // 2`` and then ``-(N // 2), ..., -1`` (Press et al. 2007,
    sec. 12.1).

    Parameters
    ----------
    grid : PeriodicGrid
        The grid.

    Returns
    -------
    tuple of numpy.ndarray
        One float64 array per axis [1/Angstrom]. The array of axis ``i`` has
        length ``N_i`` on axis ``i`` and length 1 on all other axes, so the
        arrays broadcast to the full grid. ``np.broadcast_arrays`` gives full
        views without a copy. Each value is one IEEE division of an exact
        integer and is correctly rounded. The arrays are new on each call.
    """
    out = []
    for axis, (n, length) in enumerate(zip(grid.points, grid.lengths, strict=True)):
        shape = [1] * grid.ndim
        shape[axis] = n
        out.append((fft_indices(n) / length).reshape(shape))
    return tuple(out)


def ellipse_mask(
    shape: tuple[int, int],
    nyquist_fraction: float | tuple[float, float],
    *,
    half_last_axis: bool,
    fade_depth: float = 0.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return a low-pass mask that keeps the frequencies inside an ellipse.

    The ellipse is centred on zero frequency, with its axes on the grid axes.
    Its semi-axis on axis ``i`` is ``rho_i`` times the Nyquist frequency. In
    index units it is ``R_i = rho_i N_i / 2``. For the FFT integers
    ``(j_y, j_x)`` the code computes, in float64::

        num = (j_y R_x)**2 + (j_x R_y)**2,   den = (R_y R_x)**2

    The normalised radius is ``r = sqrt(num / den)``. A point is inside if
    ``num < den``. A point with ``num >= den`` (``r >= 1``, also ``r`` exactly
    1) gets the value 0. For ``R_i`` integer or half-integer below about
    ``2**20`` this test is exact.

    Alias-free size: let two fields have spectra in the open ellipse ``E``.
    Their product has its spectrum in ``E + E = 2E``. The DFT wraps frequency
    ``j`` onto ``j + (m_y N_y, m_x N_x)`` with integer ``m``. A wrapped copy
    reaches ``E`` only if the shift lies in ``2E - E = 3E``. The shortest
    shifts are along one axis, with normalised length ``N_i / R_i``. So no
    copy reaches ``E`` if ``N_i / R_i >= 3`` on both axes, that is
    ``rho_i <= 2/3``. Boundary points are removed, so equality is safe. This
    is the 2/3 rule of Kirkland (2010, sec. 6.8).

    Smooth edge: with depth ``t = 1 - r`` and ``w = fade_depth > 0``, the value
    is 0 for ``num >= den``, 1 for ``t >= w``, and ``sin(pi/2 * t / w)**2``
    between. This is the Hann taper (Harris 1978). It is monotone in ``t``,
    has a continuous slope at both ends and stays inside the cut, so the
    alias-free result stays true.

    Parameters
    ----------
    shape : tuple of int
        Full grid shape ``(Ny, Nx)``, also for the half layout.
    nyquist_fraction : float or tuple of float
        ``rho`` [dimensionless], one value for both axes or ``(rho_y,
        rho_x)``, each in (0, 1]. The argument is required. The value 2/3 is
        the largest alias-free value; its float64 value is a little below 2/3.
    half_last_axis : bool
        Required. If True, return the ``torch.fft.rfft2`` layout ``(Ny, Nx //
        2 + 1)``; column ``n`` holds ``j_x = n``. If False, return the full
        layout ``(Ny, Nx)``.
    fade_depth : float, optional
        Depth ``w`` of the smooth edge [normalised radius], in [0, 1). The
        default 0 gives the hard edge: it keeps the full alias-free band and
        filtering twice equals filtering once.
    device : torch.device or str, optional
        Device of the result. Default: the CPU.
    dtype : torch.dtype, optional
        Real floating dtype of the result. Default: the torch default dtype.

    Returns
    -------
    torch.Tensor
        Real mask ``(Ny, Nx)`` in FFT order, values in [0, 1]. The caller
        owns the tensor and can change it.

    Raises
    ------
    ValueError
        If ``shape``, ``nyquist_fraction`` or ``fade_depth`` is out of range.
    TypeError
        If ``dtype`` is not a real floating dtype.

    Notes
    -----
    The code builds the mask once in float64 on the CPU and keeps the result
    of each dtype and device in a bounded LRU store (32 masks, 256 MiB). A
    repeat request returns a clone of the stored mask. A float32 mask is the
    float64 mask rounded once, so the two agree to 0.5 float32 ulp at every
    point. The brief allows differences for points within float32 rounding
    of the cut or of the inner end of the edge; this design has none. CPU
    and CUDA masks are equal bit for bit.
    """
    stored = stored_ellipse(
        shape, nyquist_fraction, half_last_axis, fade_depth, device, dtype
    )
    return stored.clone()


def lowpass_ellipse(
    field: torch.Tensor,
    nyquist_fraction: float | tuple[float, float],
    *,
    fade_depth: float = 0.0,
) -> torch.Tensor:
    """Return a complex real-space field band-limited by the ellipse mask.

    The function computes ``ifft2(fft2(field) * mask)`` with the mask of
    :func:`ellipse_mask` for the shape, real dtype and device of ``field``.
    The arguments and their reasons are those of :func:`ellipse_mask`.

    Parameters
    ----------
    field : torch.Tensor
        Complex field ``(..., Ny, Nx)`` in real space. The mask broadcasts
        over the leading axes.
    nyquist_fraction : float or tuple of float
        Semi-axes as fractions of the Nyquist frequency [dimensionless].
        Required; 2/3 is the largest alias-free value.
    fade_depth : float, optional
        Depth of the smooth edge [normalised radius]; 0 is the hard edge.

    Returns
    -------
    torch.Tensor
        New complex field with the shape and dtype of ``field``. The input
        does not change, and the autograd graph stays intact.

    Raises
    ------
    TypeError
        If ``field`` is not complex.
    """
    real = real_dtype_of(field)
    if field.dim() < 2:
        raise ValueError(f"field needs at least two axes; got {tuple(field.shape)}")
    mask_dtype = torch.float32 if real == torch.float16 else real
    mask = stored_ellipse(
        tuple(field.shape[-2:]),
        nyquist_fraction,
        False,
        fade_depth,
        field.device,
        mask_dtype,
    )
    return filter_in_fourier_space(field, mask)


@dataclass(frozen=True)
class CircularPassBand:
    """Isotropic pass band in physical frequency on a 2-D grid.

    The cut radius is ``k_c = cut_fraction * min(N_y / (2 L_y), N_x / (2
    L_x))`` [1/Angstrom]. The rim width is ``w = rim_samples * max(1 / L_y, 1
    / L_x)`` [1/Angstrom], and the flat radius is ``k_f = k_c - w``. With
    ``|k|^2 = (j_y / L_y)**2 + (j_x / L_x)**2`` (float64), the mask is 1 for
    ``|k| <= k_f``, 0 for ``|k| >= k_c`` and ``sin(pi/2 * (k_c - |k|) /
    w)**2`` on the rim. The rim is the Hann taper (Harris 1978). In exact
    arithmetic rim values are strictly between 0 and 1. A rim point within
    rounding of ``k_f`` or ``k_c`` can round to 1 or 0.

    Parameters
    ----------
    band_grid : PeriodicGrid
        A 2-D grid ``(y, x)``. Pixel sizes can differ.
    cut_fraction : float
        ``k_c`` as a fraction of the lower Nyquist frequency [dimensionless],
        in (0, 1]. The argument is required. A value of at most 1 keeps the
        circle inside the Nyquist rectangle. A value of at most 2/3 makes a
        product of two band-limited fields alias-free on both axes (see
        :func:`ellipse_mask`).
    rim_samples : float, optional
        Rim width in frequency steps of the coarser axis [dimensionless],
        greater than 0. The default 2 is the smallest width that puts at
        least one sample strictly on the rim on each axis. A wider rim gives
        less ringing and a smaller flat band.

    Raises
    ------
    ValueError
        If the grid is not 2-D, if a parameter is out of range, or if
        ``k_f < 0``.

    Notes
    -----
    Keep one object for one multislice run. The transmission step and the
    propagation step then use identical pass bands. The object has no
    mutable state, and equal inputs give equal masks.
    """

    band_grid: PeriodicGrid
    cut_fraction: float
    rim_samples: float = 2.0

    def __post_init__(self) -> None:
        if not isinstance(self.band_grid, PeriodicGrid) or self.band_grid.ndim != 2:
            raise ValueError(
                f"CircularPassBand needs a 2-D PeriodicGrid; got {self.band_grid!r}"
            )
        object.__setattr__(self, "cut_fraction", float(self.cut_fraction))
        object.__setattr__(self, "rim_samples", float(self.rim_samples))
        if not 0.0 < self.cut_fraction <= 1.0:
            raise ValueError(
                f"cut_fraction must be in (0, 1]; got {self.cut_fraction!r}"
            )
        if not self.rim_samples > 0.0:
            raise ValueError(f"rim_samples must be > 0; got {self.rim_samples!r}")
        k_c, w, k_f = circle_radii(self.band_grid, self.cut_fraction, self.rim_samples)
        if k_f < 0.0:
            raise ValueError(
                f"rim width {w:g} 1/Angstrom is larger than the cut radius "
                f"{k_c:g} 1/Angstrom; lower rim_samples or use a larger grid"
            )

    @cached_property
    def _radii(self) -> tuple[float, float, float]:
        return circle_radii(self.band_grid, self.cut_fraction, self.rim_samples)

    @property
    def cut_radius(self) -> float:
        """Cut radius ``k_c`` [1/Angstrom]; the mask is 0 at and above it."""
        return self._radii[0]

    @property
    def rim_width(self) -> float:
        """Rim width ``w`` [1/Angstrom]."""
        return self._radii[1]

    @property
    def flat_radius(self) -> float:
        """Flat radius ``k_f`` [1/Angstrom]; the mask is 1 at and below it."""
        return self._radii[2]

    def _stored(self, device, dtype) -> torch.Tensor:
        dt, dev = real_dtype_key(dtype), device_key(device)
        key = ("circle", self.band_grid, self.cut_fraction, self.rim_samples, dt, dev)
        return STORE.get(
            key,
            lambda: (
                torch.from_numpy(
                    build_circle(
                        self.band_grid.points, self.band_grid.lengths, self._radii
                    )
                )
                .to(dtype=dt)
                .to(device=dev)
            ),
        )

    def mask(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Return the pass-band mask.

        Parameters
        ----------
        device : torch.device or str, optional
            Device. Default: the CPU.
        dtype : torch.dtype, optional
            Real floating dtype. Default: the torch default dtype.

        Returns
        -------
        torch.Tensor
            Real mask ``(Ny, Nx)`` in FFT order, values in [0, 1]. The caller
            owns the tensor. A float32 mask is the float64 mask rounded once.
        """
        return self._stored(device, dtype).clone()

    def min_crop_shape(self) -> tuple[int, int]:
        """Return the smallest lossless centred crop box ``(M_y, M_x)``.

        The centred crop of ``M`` points keeps ``j`` from ``-(M // 2)`` to
        ``M - 1 - M // 2``. Padding back returns every kept value to its own
        frequency. With ``J_i`` the largest ``|j_i|`` where the float64 mask
        is not zero, the smallest box is ``M_i = 2 J_i + 1``, never above
        ``N_i``. A mask of lower precision has no more non-zero points, so the
        box is valid for every dtype.

        Returns
        -------
        tuple of int
            ``(M_y, M_x)`` [samples].
        """
        stored = self._stored("cpu", torch.float64).numpy()
        out = []
        for axis, n in enumerate(self.band_grid.points):
            used = np.any(stored != 0.0, axis=1 - axis)
            j_max = int(np.abs(fft_indices(n)[used]).max())
            out.append(min(n, 2 * j_max + 1))
        return out[0], out[1]


def apply_pass_band(field: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return the band-limited real-space field ``ifft2(fft2(field) * mask)``.

    Parameters
    ----------
    field : torch.Tensor
        Complex field ``(..., Ny, Nx)`` in real space.
    mask : torch.Tensor
        Real mask ``(Ny, Nx)`` in FFT order, for example from
        :meth:`CircularPassBand.mask`. The function moves it to the device
        and real dtype of ``field``.

    Returns
    -------
    torch.Tensor
        New field with the shape and dtype of ``field``. Neither input
        changes; the autograd graph of ``field`` stays intact. A
        ``complex32`` field is computed in ``complex64`` and cast back, so the
        FFT sum cannot overflow.

    Raises
    ------
    TypeError
        If ``field`` is not complex.
    ValueError
        If the mask shape does not match the last two axes of ``field``.
    OverflowError
        If a ``complex32`` result has a part above the float16 range.
    """
    return filter_in_fourier_space(field, mask)


def mask_cache_stats() -> MaskCacheStats:
    """Return the counters of the mask store.

    Returns
    -------
    MaskCacheStats
        ``hits``, ``misses`` (number of builds), ``entries`` and ``nbytes``
        [bytes].
    """
    return STORE.stats()


def clear_mask_cache() -> None:
    """Remove all stored masks and set the counters to zero."""
    STORE.clear()
