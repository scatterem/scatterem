"""Private helpers: spectral filter of a complex field and the raised-cosine edge.

The filter is the discrete convolution theorem (Press et al. 2007, sec. 12.1;
Bracewell 2000, ch. 10): FFT, multiply by the mask, inverse FFT. The edge
profile is the Hann taper (Harris 1978). Written for scatterem from this
literature.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

_log = logging.getLogger(__name__)

_FLOAT16_MAX = float(torch.finfo(torch.float16).max)

_REAL_OF_COMPLEX = {
    torch.complex32: torch.float16,
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
}


def real_dtype_of(field: torch.Tensor) -> torch.dtype:
    """Return the real dtype that belongs to a complex field.

    Parameters
    ----------
    field : torch.Tensor
        A complex tensor.

    Returns
    -------
    torch.dtype
        ``float16``, ``float32`` or ``float64``.

    Raises
    ------
    TypeError
        If ``field`` is not complex.
    """
    try:
        return _REAL_OF_COMPLEX[field.dtype]
    except KeyError:
        raise TypeError(
            f"the field must be a complex tensor; got dtype {field.dtype}"
        ) from None


def raised_cosine(depth: np.ndarray, width: float) -> np.ndarray:
    """Return the Hann edge ``sin(pi/2 * depth / width)**2`` (float64).

    Parameters
    ----------
    depth : numpy.ndarray
        Depth inside the cut, in the unit of ``width``. Valid in
        ``[0, width]``.
    width : float
        Depth of the edge, greater than 0.

    Returns
    -------
    numpy.ndarray
        The edge values, in [0, 1] on the valid range.
    """
    return np.sin(0.5 * np.pi * (depth / width)) ** 2


def filter_in_fourier_space(field: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return ``ifft2(fft2(field) * mask)``, out of place.

    The mask moves to the device and the real dtype of ``field``. A
    ``complex32`` field is computed in ``complex64``, because the
    unnormalised FFT adds ``Ny * Nx`` values and can leave the float16 range.

    Parameters
    ----------
    field : torch.Tensor
        Complex field ``(..., Ny, Nx)`` in real space.
    mask : torch.Tensor
        Real mask ``(Ny, Nx)`` in FFT order.

    Returns
    -------
    torch.Tensor
        The filtered field, with the shape and dtype of ``field``.

    Raises
    ------
    TypeError
        If ``field`` is not complex.
    ValueError
        If the mask shape is not the shape of the last two axes of ``field``.
    OverflowError
        If a ``complex32`` result has a part above the float16 range.
    """
    real = real_dtype_of(field)
    if field.dim() < 2 or tuple(mask.shape) != tuple(field.shape[-2:]):
        raise ValueError(
            f"mask shape {tuple(mask.shape)} does not match the last two axes "
            f"of the field shape {tuple(field.shape)}"
        )
    # A complex32 field is summed in complex64; other fields keep their dtype.
    half = field.dtype == torch.complex32
    spectrum = torch.fft.fft2(field.to(torch.complex64) if half else field)
    weights = mask.to(device=field.device, dtype=torch.float32 if half else real)
    out = torch.fft.ifft2(spectrum * weights)
    if half:
        peak = torch.view_as_real(out.detach()).abs().amax()
        if bool(peak > _FLOAT16_MAX):
            raise OverflowError(
                f"the filtered field has a value of {float(peak):.6g}; "
                f"complex32 holds at most {_FLOAT16_MAX:g}"
            )
        out = out.to(torch.complex32)
    return out
