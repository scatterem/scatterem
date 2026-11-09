"""Validators of the optional fields of ``AcquisitionMetadata``.

``parent_scan`` (a window of a parent raster) and ``detector_warp_centre``
(the disk centre, the pivot of the detector warp). ``data.datasets._meta``
uses them too, when a dataset operation moves either.
"""

from typing import Any

import numpy as np
from torch import Tensor


def _parent_scan(value: Any) -> tuple[int, int, int, int] | None:
    """``parent_scan`` as four ints ``(y0, x0, NY, NX)``, or None."""
    if value is None:
        return None
    arr = np.asarray(value)
    # Refuse a non-integral entry rather than truncate it: int(4.9999999) is 4,
    # which would move every position of the window by one scan step.
    if arr.ndim != 1 or not all(float(v).is_integer() for v in arr):
        raise ValueError(
            f"parent_scan must be four integers (y0, x0, NY, NX), got {value!r}"
        )
    out = tuple(int(v) for v in arr)
    if len(out) != 4 or min(out) < 0 or out[2] < 1 or out[3] < 1:
        raise ValueError(f"parent_scan must be (y0, x0, NY, NX) >= 0, got {value!r}")
    return out


def _warp_centre(value: Any) -> tuple[float, float] | None:
    """``detector_warp_centre`` as two finite floats ``(y, x)``, or None."""
    if value is None:
        return None
    arr = np.asarray(
        value.detach().cpu() if isinstance(value, Tensor) else value, dtype=float
    )
    if arr.shape != (2,) or not np.isfinite(arr).all():
        raise ValueError(
            f"detector_warp_centre must be two finite pixel coordinates (y, x), "
            f"got {value!r}"
        )
    return (float(arr[0]), float(arr[1]))
