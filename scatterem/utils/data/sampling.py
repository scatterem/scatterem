"""Lightweight sidecar carrying per-axis pixel size and units for images.

The numeric data stays a plain tensor/ndarray; this object is threaded
explicitly into the visualization layer so calibration cannot silently vanish
through tensor operations (the failure mode of the removed ``UnitTensor``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


def _to_tuple(x):
    if x is None:
        return None
    if isinstance(x, (str, bytes)):
        return (x,)
    if isinstance(x, Sequence):
        return tuple(x)
    return (x,)  # scalar (float/int) or single unit


@dataclass(frozen=True)
class Sampling:
    """Per-axis image calibration.

    Parameters
    ----------
    pixel_size : float or sequence of float, optional
        Physical size of one pixel along each axis, in axis order ``(dy, dx)``
        for a 2D image. ``None`` means uncalibrated.
    units : str or sequence of str, optional
        Unit string per axis, e.g. ``("Å", "Å")`` (real space) or
        ``("1/Å", "1/Å")`` (reciprocal space).
    value_unit : str, optional
        Unit of the stored values, used for the colorbar label.
    """

    pixel_size: Optional[Tuple[float, ...]] = None
    units: Optional[Tuple[str, ...]] = None
    value_unit: Optional[str] = None

    def __post_init__(self) -> None:
        ps = _to_tuple(self.pixel_size)
        un = _to_tuple(self.units)
        object.__setattr__(self, "pixel_size", ps)
        object.__setattr__(self, "units", un)
        if ps is not None and un is not None and len(ps) != len(un):
            raise ValueError(
                f"pixel_size has {len(ps)} axes but units has {len(un)}"
            )

    @property
    def is_calibrated(self) -> bool:
        return self.pixel_size is not None

    def scalebar_sampling(self) -> Tuple[Optional[float], Optional[str]]:
        """Return ``(pixel_size, unit)`` for the x-axis (last axis).

        The horizontal scale bar is measured along the column axis, the last
        axis of an ``(H, W)`` image.
        """
        ps = self.pixel_size[-1] if self.pixel_size else None
        un = self.units[-1] if self.units else None
        return ps, un
