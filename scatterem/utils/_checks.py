"""Input checks for the grid and disk-measurement functions of scatterem.

This module validates inputs: sample pitches, point counts, dtypes and real
arrays. It follows no literature. This code was written for scatterem.
"""

from __future__ import annotations

import logging
import math
import numbers
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

REAL_DTYPES = (torch.float32, torch.float64)


def _is_bool(value: Any) -> bool:
    """Return True for a Python or NumPy bool."""
    return isinstance(value, bool | np.bool_)


def _scalar_to_float(value: Any, name: str) -> float:
    """Convert one real scalar to a Python float.

    Parameters
    ----------
    value : int, float, NumPy real scalar, 0-d array or 0-d tensor
        The value to convert.
    name : str
        The name of the input, for error messages.

    Returns
    -------
    float
        The value as a float64 number.

    Raises
    ------
    TypeError
        If the value is not a real number.
    """
    if isinstance(value, torch.Tensor):
        if value.ndim != 0 or value.is_complex() or value.dtype == torch.bool:
            raise TypeError(f"{name}: expected real numbers, got tensor {value}")
        return float(value.detach().cpu().double().item())
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise TypeError(f"{name}: expected real numbers, got array {value}")
        value = value[()]
    if _is_bool(value) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name}: expected real numbers, got {value!r}")
    return float(value)


def real_scalar(value: Any, name: str) -> float:
    """Return a real scalar setting as a float.

    Parameters
    ----------
    value : int or float
        The setting.
    name : str
        The name of the setting, for error messages.

    Returns
    -------
    float
        The value.

    Raises
    ------
    TypeError
        If the value is not a real number.
    """
    return _scalar_to_float(value, name)


def pitches(pitch: Any, n_axes: int, name: str = "pitch") -> tuple[float, ...]:
    """Validate a sample pitch and give one value per axis.

    Parameters
    ----------
    pitch : scalar, sequence, numpy.ndarray or torch.Tensor
        One pitch for all axes, or one pitch per axis [length].
    n_axes : int
        The number of axes.
    name : str
        The name of the input, for error messages.

    Returns
    -------
    tuple of float
        One finite, positive pitch per axis [length].

    Raises
    ------
    TypeError
        If a value is not a real number.
    ValueError
        If the count is wrong, or a value is not finite and positive.
    """
    if isinstance(pitch, str | bytes) or pitch is None:
        raise TypeError(f"{name}: expected real numbers, got {pitch!r}")
    is_array = isinstance(pitch, np.ndarray | torch.Tensor)
    if (is_array and pitch.ndim == 0) or not (
        is_array or isinstance(pitch, list | tuple)
    ):
        values = (_scalar_to_float(pitch, name),) * n_axes
    else:
        if is_array and pitch.ndim != 1:
            raise ValueError(f"{name}: expected a 1-D array, got shape {pitch.shape}")
        if isinstance(pitch, np.ndarray) and pitch.dtype.kind not in "iuf":
            raise TypeError(f"{name}: expected real numbers, got {pitch.dtype}")
        items = list(pitch.detach().cpu()) if isinstance(pitch, torch.Tensor) else pitch
        values = tuple(_scalar_to_float(v, name) for v in items)
        if len(values) != n_axes:
            raise ValueError(
                f"{name}: expected {n_axes} values (one per axis), got {len(values)}"
            )
    if not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError(f"{name}: expected finite positive values, got {list(values)}")
    return values


def point_counts(n_yx: Any, name: str = "n_yx") -> tuple[int, int]:
    """Validate the point counts of a 2-D grid.

    Parameters
    ----------
    n_yx : sequence of two int, numpy.ndarray or torch.Tensor
        The number of points along ``y`` and along ``x``.
    name : str
        The name of the input, for error messages.

    Returns
    -------
    tuple of int
        ``(Ny, Nx)``.

    Raises
    ------
    TypeError
        If the input is not a sequence of integers.
    ValueError
        If the length is not 2 or a count is smaller than 1.
    """
    if isinstance(n_yx, torch.Tensor | np.ndarray):
        if n_yx.ndim != 1:
            raise ValueError(
                f"{name}: expected 2 counts, got shape {tuple(n_yx.shape)}"
            )
        n_yx = n_yx.tolist()
    if not isinstance(n_yx, list | tuple):
        raise TypeError(f"{name}: expected a sequence of 2 integers, got {n_yx!r}")
    if len(n_yx) != 2:
        raise ValueError(f"{name}: expected 2 counts (Ny, Nx), got {len(n_yx)}")
    out = []
    for value in n_yx:
        if _is_bool(value) or not isinstance(value, numbers.Integral):
            raise TypeError(f"{name}: expected integers, got {value!r}")
        if value < 1:
            raise ValueError(f"{name}: expected counts of at least 1, got {value}")
        out.append(int(value))
    return out[0], out[1]


def real_dtype(dtype: Any) -> torch.dtype:
    """Return a real result dtype, or the torch default for None.

    Parameters
    ----------
    dtype : torch.dtype or None
        The requested dtype.

    Returns
    -------
    torch.dtype
        ``torch.float32`` or ``torch.float64``.

    Raises
    ------
    TypeError
        If the dtype is not float32 or float64.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    if dtype not in REAL_DTYPES:
        raise TypeError(f"dtype: expected torch.float32 or torch.float64, got {dtype}")
    return dtype


def float64_tensor(values: Any, name: str) -> torch.Tensor:
    """Return a float64 tensor view or copy of a real array.

    The input is never changed. A tensor stays on its device and loses its
    autograd graph. A NumPy array goes to the CPU.

    Parameters
    ----------
    values : numpy.ndarray or torch.Tensor
        Integer or real array.
    name : str
        The name of the input, for error messages.

    Returns
    -------
    torch.Tensor
        The values as float64.

    Raises
    ------
    TypeError
        If the input is not an array, or is bool or complex.
    """
    if isinstance(values, torch.Tensor):
        if values.is_complex() or values.dtype == torch.bool:
            raise TypeError(f"{name}: expected real values, got dtype {values.dtype}")
        return values.detach().to(torch.float64)
    if isinstance(values, np.ndarray):
        if values.dtype.kind not in "iuf":
            raise TypeError(f"{name}: expected real values, got dtype {values.dtype}")
        return torch.from_numpy(np.ascontiguousarray(values, dtype=np.float64))
    raise TypeError(
        f"{name}: expected a numpy.ndarray or a torch.Tensor, got {type(values)}"
    )
