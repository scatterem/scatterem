"""Crop-width / axis validation for :meth:`scatterem.data.datasets.base.Dataset.crop`.

Owns the selection, width-checking and origin-shift arithmetic so ``Dataset.crop``
only applies the resulting slices to the array.
"""

from __future__ import annotations

import numpy as np


def _resolve_crop_slices_and_origin(
    ndim: int,
    shape: tuple[int, ...],
    origin,
    sampling,
    crop_widths: tuple[tuple[int, int], ...],
    axes: int | tuple[int, ...] | None,
) -> tuple[tuple[slice, ...], tuple[float, ...]]:
    """Validate crop widths/axes and return ``(slices, new_origin)``."""
    if axes is None:
        if len(crop_widths) != ndim:
            raise ValueError(
                f"got {len(crop_widths)} crop widths for {ndim} axes; when "
                f"axes is omitted every axis needs one"
            )
        selected = tuple(range(ndim))
        widths = tuple(crop_widths)
    else:
        if isinstance(axes, (int, np.integer)):
            selected = (int(axes),)
            # a lone axis may be given a bare (before, after)
            widths = (
                (crop_widths,)
                if len(crop_widths) == 2
                and all(isinstance(v, (int, np.integer)) for v in crop_widths)
                else tuple(crop_widths)
            )
        else:
            selected = tuple(int(a) for a in axes)
            widths = tuple(crop_widths)
        if len(widths) != len(selected):
            raise ValueError(
                f"got {len(widths)} crop widths for {len(selected)} axes; they "
                f"must correspond one to one"
            )

    selected = tuple(a + ndim if a < 0 else a for a in selected)
    for axis in selected:
        if not 0 <= axis < ndim:
            raise ValueError(
                f"axis {axis} is out of range for a {ndim}-dimensional dataset"
            )
    if len(set(selected)) != len(selected):
        raise ValueError(f"axes must be distinct; got {axes!r}")

    per_axis = {}
    for axis, width in zip(selected, widths, strict=True):
        before, after = (int(width[0]), int(width[1]))
        if before < 0 or after < 0:
            raise ValueError(
                f"crop widths must not be negative; got {(before, after)} for "
                f"axis {axis}"
            )
        if before + after >= shape[axis]:
            raise ValueError(
                f"cropping {before}+{after} from axis {axis} would consume all "
                f"{shape[axis]} of it"
            )
        per_axis[axis] = (before, after)

    slices = tuple(
        (
            slice(per_axis[axis][0], length - per_axis[axis][1])
            if axis in per_axis
            else slice(None)
        )
        for axis, length in enumerate(shape)
    )
    new_origin = tuple(
        float(o) + per_axis.get(axis, (0, 0))[0] * float(s)
        for axis, (o, s) in enumerate(
            zip(
                np.atleast_1d(origin),
                np.atleast_1d(sampling),
                strict=True,
            )
        )
    )
    return slices, new_origin
