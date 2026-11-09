"""Stacked torch FFT-frequency grids.

Import it as ``scatterem.physics._frequency_grid.fft_frequency_grid``.

Written for scatterem; it is ``torch.fft.fftfreq`` per axis, meshed and stacked,
and comes from no other implementation.  It replaces ``utils.stem.fftfreq2`` and
``fftfreq3`` with the same operations in the same order, so its output is
bit-identical to theirs for every argument they accepted.
"""

from collections.abc import Sequence

import numpy as np
import torch
from torch import Size, Tensor

__all__ = ["fft_frequency_grid"]


def fft_frequency_grid(
    shape: Sequence[int] | Size | np.ndarray,
    sampling: Sequence[float] | np.ndarray | None = None,
    half_pixel_shift: bool = False,
    device: str | torch.device | None = "cpu",
    *,
    dtype: torch.dtype | None = None,
    rfft: bool = False,
) -> Tensor:
    """FFT frequencies of an N-D grid, meshed and stacked, corner origin.

    Axis ``i`` holds ``torch.fft.fftfreq(shape[i], sampling[i])``: frequency
    ``j / (shape[i] * sampling[i])`` for ``j = 0, 1, ..., -2, -1`` in FFT
    order, in the reciprocal of the unit ``sampling`` is given in.  Unlike
    :func:`scatterem.physics.periodic_sampling.dft_sample_frequencies` (numpy, float64, from the
    extent), this is a torch tensor, built from the pixel size.

    Parameters
    ----------
    shape : (D,) sequence of int
        Grid dimensions, e.g. ``(ny, nx)`` or ``(nz, ny, nx)``.
    sampling : (D,) sequence of float, optional
        Real-space pixel size per dimension.  Only its first ``D`` entries are
        read.  Default: ``1.0`` on every axis (frequencies in cycles/pixel).
    half_pixel_shift : bool, optional
        Add half a frequency bin, ``0.5 / shape[i] / sampling[i]``, to every
        axis (the ``centered=True`` of the replaced ``fftfreq2``).  Default
        False.
    device : str or torch.device, optional
        Device of the result.  Default ``"cpu"``.
    dtype : torch.dtype, optional
        Floating dtype of the result.  Default: torch's default dtype.
    rfft : bool, optional
        Use ``torch.fft.rfftfreq`` on the last axis, the half spectrum of
        ``torch.fft.rfftn``, which has ``shape[-1] // 2 + 1`` non-negative
        frequencies.  Default False.

    Returns
    -------
    Tensor
        Shape ``(D, *shape)`` (``(D, *shape[:-1], shape[-1] // 2 + 1)`` with
        ``rfft``); ``[i]`` is the frequency along dimension ``i`` broadcast
        over the grid (for 2-D: ``[0]`` is ``qy``, ``[1]`` is ``qx``).
    """
    ndim = len(shape)
    if sampling is None:
        sampling = (1.0,) * ndim
    axes = []
    for dim in range(ndim):
        freq = torch.fft.rfftfreq if rfft and dim == ndim - 1 else torch.fft.fftfreq
        axis = freq(int(shape[dim]), sampling[dim], device=device, dtype=dtype)
        if half_pixel_shift:
            axis += 0.5 / shape[dim] / sampling[dim]
        axes.append(axis)
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=0)
