"""Integer FFT index grids and the fftshift checkerboard.

The physical frequency grid is :func:`scatterem.physics._frequency_grid.fft_frequency_grid`.
"""

from collections.abc import Sequence

import numpy as np
import torch
from torch import Size, Tensor


# Not fft_frequency_grid: integer beam indices, the S-matrix coordinate convention.
def fft_index_grid(
    shape: Sequence[int] | Size | np.ndarray,
    device: str | torch.device | None = "cpu",
) -> Tensor:
    """Signed integer FFT index of every pixel of a ``(ny, nx)`` grid.

    The integer twin of
    :func:`scatterem.physics._frequency_grid.fft_frequency_grid`: entry ``[:, i, j]`` is the signed
    frequency index ``(ky, kx)`` that ``torch.fft.fft2`` puts at pixel
    ``(i, j)`` -- ``0 .. (n - 1) // 2`` followed by ``-(n // 2) .. -1`` along
    each axis, for even and odd ``n`` alike.  This is the one definition of a
    beam coordinate; :class:`~scatterem.data.SMeta` and the S-matrix
    builders all take it from here.

    Parameters
    ----------
    shape : array-like
        Grid dimensions ``(ny, nx)``.
    device : str or torch.device or None, optional
        Device of the result. Default is ``"cpu"``.

    Returns
    -------
    Tensor
        ``(2, ny, nx)`` int64, ``[ky, kx]``.
    """
    axes = [
        (torch.arange(int(n), device=device) + int(n) // 2) % int(n) - int(n) // 2
        for n in (shape[0], shape[1])
    ]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=0)


def fftshift_checkerboard(ny: int, nx: int) -> np.ndarray:
    """Build the +-1 checkerboard that is an fftshift in the conjugate domain.

    Multiplying a field by ``-(-1) ** (i + j)`` shifts its transform by half the
    grid along both axes.  That is a +-1 pattern only for EVEN sizes (an odd
    axis would need the complex ramp ``exp(-2 pi i k (n // 2) / n)``), so odd
    sizes are refused here rather than answered with the wrong pattern.

    Parameters
    ----------
    ny : int
        Number of rows (must be even).
    nx : int
        Number of columns (must be even).

    Returns
    -------
    numpy.ndarray
        ``(ny, nx)`` array of alternating ``-1``/``+1`` values, ``-1`` at
        ``[0, 0]``.

    Raises
    ------
    ValueError
        If either size is odd.

    Notes
    -----
    Shapes are ``(ny, nx)`` here as everywhere else.  Until 2026-09-18 the
    signature was ``(w, h)`` returning ``(h, w)`` while every caller passed
    ``(ny, nx)``, so a non-square probe grid raised a broadcast error.
    """
    ny, nx = int(ny), int(nx)
    if ny % 2 or nx % 2:
        raise ValueError(
            f"fftshift_checkerboard needs even sizes, got {(ny, nx)}: a half-grid "
            f"shift is a +-1 pattern only when the grid has an even number of points."
        )
    i, j = np.indices((ny, nx))
    return np.where((i + j) % 2 == 0, -1, 1)
