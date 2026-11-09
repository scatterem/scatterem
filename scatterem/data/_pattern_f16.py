"""float16 storage of noise-free intensities with a per-tilt scale (pattern streaming).

A tilt's intensities are stored once, at ingest, as
``float16(I * 2**e)`` with ``e`` chosen so the largest stored value lies in
``[2**13, 2**14]``; decoding (``float32(h) * 2**-e``) is exact, so only the
float16 rounding (at most half an ulp, a relative ``2**-11``) loses
information.

Integer counts are stored as uint8 or uint16 (:func:`as_count_encoding`).
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

__all__ = [
    "Float16Tilt",
    "as_count_encoding",
    "decode_scale",
    "encode_float16",
    "float16_exponent",
]

#: Above this fraction of nonzero intensities becoming subnormal or zero in
#: float16, ingest logs a warning (§4.7.1).
F16_WARN_FRACTION = 1e-3
_F16_MIN_NORMAL = 2.0**-14
#: |e| bound: both 2**e and 2**-e stay float32 normals.
_F16_MAX_EXPONENT = 126
#: Values converted per block at ingest (float64 scratch of 64 MiB).
_ENCODE_BLOCK = 1 << 23


def float16_exponent(max_value: float) -> int:
    """``e`` with ``max_value * 2**e`` in ``[2**13, 2**14]``; 0 for an empty max.

    Parameters
    ----------
    max_value : float
        The tilt's largest intensity (finite, non-negative).

    Returns
    -------
    int
        ``floor(log2(2**14 / max_value))``, computed exactly with ``frexp``.
    """
    if max_value <= 0.0:
        return 0
    m, q = math.frexp(float(max_value))  # max = m * 2**q, m in [0.5, 1)
    return 14 - q + (1 if m == 0.5 else 0)


def decode_scale(exponent: int, device: torch.device | str | None = None) -> Tensor:
    """The 0-dim float32 ``2**-exponent`` the float16 decode multiplies by."""
    return torch.tensor(2.0 ** (-int(exponent)), dtype=torch.float32, device=device)


@dataclass(frozen=True)
class Float16Tilt:
    """One tilt encoded as float16 (:func:`encode_float16`).

    Attributes
    ----------
    values : Tensor
        float16 ``intensity * 2**exponent``, the input's shape.
    exponent : int
        The per-tilt scale exponent ``e`` (scale ``s = 2**e``).
    subnormal_fraction : float
        Fraction of the nonzero intensities stored as float16 subnormals.
    zero_fraction : float
        Fraction of the nonzero intensities that became zero.
    """

    values: Tensor
    exponent: int
    subnormal_fraction: float
    zero_fraction: float


def encode_float16(intensity: Tensor | np.ndarray, *, label: str = "") -> Float16Tilt:
    """Quantise one tilt's noise-free intensities to float16 with its scale.

    Parameters
    ----------
    intensity : Tensor or numpy.ndarray
        Finite, non-negative float intensities (any shape).
    label : str, optional
        Names the tilt in the warning.

    Returns
    -------
    Float16Tilt
        The values, exponent and the subnormal/zero fractions (when their
        sum exceeds ``F16_WARN_FRACTION`` a warning is logged).

    Raises
    ------
    ValueError
        A negative or non-finite intensity, or a maximum whose scale exponent
        is outside +-126 (the decode scale must be a float32 normal).
    """
    with warnings.catch_warnings():
        # only read: a read-only memmap (np.load(mmap_mode="r")) is fine
        warnings.filterwarnings(
            "ignore", message=".*not writable.*", category=UserWarning
        )
        src = torch.as_tensor(intensity).detach().cpu()
    flat = src.reshape(-1)
    if flat.numel() and not bool((torch.isfinite(flat) & (flat >= 0)).all()):
        raise ValueError("float16 ingest needs finite, non-negative intensities")
    e = float16_exponent(float(flat.max()) if flat.numel() else 0.0)
    if not -_F16_MAX_EXPONENT <= e <= _F16_MAX_EXPONENT:
        raise ValueError(
            f"the tilt's maximum {float(flat.max()):.3g} needs a scale 2**{e}, "
            f"outside 2**+-{_F16_MAX_EXPONENT} (the decode scale must be a "
            "float32 normal)"
        )
    h = np.empty(flat.numel(), dtype=np.float16)
    n_nonzero = n_zero = n_sub = 0
    for i in range(0, flat.numel(), _ENCODE_BLOCK):
        x = flat[i : i + _ENCODE_BLOCK].to(torch.float64).numpy()
        # one rounding, float64 -> float16: numpy converts directly, torch goes
        # through float32 and can round twice
        hb = h[i : i + _ENCODE_BLOCK]
        hb[...] = (x * 2.0**e).astype(np.float16)
        nonzero = x != 0
        n_nonzero += int(nonzero.sum())
        n_zero += int((nonzero & (hb == 0)).sum())
        n_sub += int((nonzero & (hb != 0) & (np.abs(hb) < _F16_MIN_NORMAL)).sum())
    zero, sub = n_zero / max(n_nonzero, 1), n_sub / max(n_nonzero, 1)
    if sub + zero > F16_WARN_FRACTION:
        logger.warning(
            "float16 ingest%s: %.3g of nonzero intensities subnormal, %.3g zero",
            f" ({label})" if label else "",
            sub,
            zero,
        )
    h = torch.from_numpy(h).reshape(src.shape)
    return Float16Tilt(h, e, sub, zero)


def as_count_encoding(array: Tensor) -> Tensor:
    """Integer counts as uint8/uint16, refusing values that do not fit.

    Parameters
    ----------
    array : Tensor
        Integer counts (any integer dtype).

    Returns
    -------
    Tensor
        ``array`` itself when it is uint8 or uint16, else a uint16 copy.

    Raises
    ------
    ValueError
        Negative counts, counts above 65535, or a non-integer dtype.
    """
    if array.dtype in (torch.uint8, torch.uint16):
        return array
    if array.dtype.is_floating_point or array.dtype.is_complex:
        raise ValueError(
            f"{array.dtype} patterns are not counts; ingest noise-free float "
            "intensities as float16 (encode_float16)"
        )
    if array.dtype == torch.bool:
        raise ValueError("bool patterns are not counts")
    if array.numel():
        lo, hi = int(array.min()), int(array.max())
        if lo < 0:
            raise ValueError(f"negative counts ({lo}) in an unsigned source")
        if hi > 65535:
            raise ValueError(f"counts up to {hi} do not fit uint16 (max 65535)")
    return array.to(torch.uint16)
