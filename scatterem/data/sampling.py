"""Lightweight sidecar carrying per-axis pixel size and units for images.

The numeric data stays a plain tensor/ndarray; this object is threaded
explicitly into the visualization layer so calibration cannot silently vanish
through tensor operations (the failure mode of the removed ``UnitTensor``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


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

    Replaces the removed ``UnitTensor``: numeric data stays a plain
    tensor/ndarray; this sidecar is threaded explicitly so units cannot vanish
    through tensor ops.

    Parameters
    ----------
    pixel_size : float or tuple of float, optional
        Physical size of one pixel along each axis, in axis order ``(dy, dx)``
        for a 2D image. ``None`` means uncalibrated.
    units : str or tuple of str, optional
        Unit string per axis, e.g. ``("Å", "Å")`` (real space) or
        ``("1/Å", "1/Å")`` (reciprocal space).
    value_unit : str, optional
        Unit of the stored values, used for the colorbar label.

    Examples
    --------
    The scale bar reads the last axis; mismatched axis counts raise:

    >>> from scatterem.data import Sampling
    >>> s = Sampling(pixel_size=(0.1, 0.2), units=("A", "A"), value_unit="e")
    >>> s.is_calibrated, s.scalebar_sampling()
    (True, (0.2, 'A'))
    >>> Sampling(pixel_size=(1.0,), units=("A", "A"))
    Traceback (most recent call last):
        ...
    ValueError: pixel_size has 1 axes but units has 2
    """

    pixel_size: tuple[float, ...] | None = None
    units: tuple[str, ...] | None = None
    value_unit: str | None = None

    def __post_init__(self) -> None:
        ps = _to_tuple(self.pixel_size)
        un = _to_tuple(self.units)
        object.__setattr__(self, "pixel_size", ps)
        object.__setattr__(self, "units", un)
        if ps is not None and un is not None and len(ps) != len(un):
            raise ValueError(f"pixel_size has {len(ps)} axes but units has {len(un)}")

    @property
    def is_calibrated(self) -> bool:
        """Whether a physical pixel size is attached to this image."""
        return self.pixel_size is not None

    def scalebar_sampling(self) -> tuple[float | None, str | None]:
        """Return the calibration of the x-axis, i.e. the last axis.

        The horizontal scale bar is measured along the column axis, the last
        axis of an ``(H, W)`` image.

        Returns
        -------
        pixel_size : float or None
            Physical size of one column, or ``None`` when uncalibrated.
        unit : str or None
            Unit of ``pixel_size``, or ``None`` when uncalibrated.
        """
        ps = self.pixel_size[-1] if self.pixel_size else None
        un = self.units[-1] if self.units else None
        return ps, un


def reconstruction_sampling(
    dataset: Any,
    image: Any = None,
    *,
    shape: Sequence[int] | None = None,
) -> Sampling:
    """The real-space pixel of an image reconstructed from a scanned dataset.

    A reconstruction is up-sampled relative to the scan, so its pixel is
    smaller than the scan step. It covers the same field of view as the scan,
    so::

        pixel_size = scan_step * scan_px / image_px

    per axis. The formula reads the upsample factor off the output shape, so
    the caller need not know which factor ``upsample="nyquist"`` picked.

    Parameters
    ----------
    dataset : object
        The dataset the reconstruction came from. Its ``shape[:2]`` is the
        scan shape and its ``sampling[:2]`` the scan step. Its ``units[:2]``
        label the axes when present, else ``("Å", "Å")``.
    image : array-like, optional
        The reconstruction. Its last two dimensions give the output shape.
    shape : sequence of int, optional
        The output shape ``(H, W)``, in place of ``image``.

    Returns
    -------
    Sampling
        The pixel size ``(dy, dx)`` and the units of the two real-space axes.

    Raises
    ------
    ValueError
        If neither ``image`` nor ``shape`` is given, or the shape is not 2-D.
    """
    if shape is None:
        if image is None:
            raise ValueError("pass either the reconstructed image or its shape")
        shape = tuple(image.shape)[-2:]
    if len(shape) != 2:
        raise ValueError(f"expected a 2D output shape; got {tuple(shape)}")
    scan_shape = tuple(dataset.shape)[:2]
    step = (float(dataset.sampling[0]), float(dataset.sampling[1]))
    units = getattr(dataset, "units", None)
    axis_units = ("Å", "Å") if units is None else (str(units[0]), str(units[1]))
    return Sampling(
        pixel_size=tuple(
            s * int(n) / int(m) for s, n, m in zip(step, scan_shape, shape, strict=True)
        ),
        units=axis_units,
    )


__all__ = [
    "Sampling",
    "reconstruction_sampling",
]
