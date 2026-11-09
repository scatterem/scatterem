"""The elliptical k-metric of the detector: which wave vector a pixel records.

Post-specimen optics (projector lenses, an energy filter) can stretch the
diffraction pattern along one axis. A round bright-field disk then lands on the
detector as an ellipse. The probe itself stays round, so the correction belongs
in the model, not in the data: the pixel at offset ``p`` (pixels, ``(row,
col)``) from the disk centre records the physical wave vector::

    k = W (p * dk)

with ``dk`` the dataset's reciprocal sampling (``sampling[-2:]``, 1/A per pixel)
and ``W`` the 2 x 2 detector warp in ``meta.detector_warp``. For an ellipse of
ellipticity ``e = 1 - b / a`` whose major axis is at ``tilt`` from the row axis
toward the column axis::

    W = R(tilt) diag(sqrt(1 - e), 1 / sqrt(1 - e)) R(tilt)^T

which is symmetric with ``det W = 1``, so the mean disk radius, and with it a
``dk`` calibrated from that radius, is unchanged (:func:`detector_warp_matrix`).
The disk centre is the pivot. A virtual bright field takes it from the centroid
of its mask, the centre of the ellipse.

A distortion beyond the ellipse (for example an "egg", a three-fold term) is a
polynomial ``k -> k + sum_ab c_ab (k_y / k_ref)^a (k_x / k_ref)^b``, with
``k_ref = WARP_REFERENCE = 1`` 1/A. ``meta.detector_warp`` then holds the
``(T, 2)`` table of ``c_ab = (c_y, c_x)`` in 1/A, in the order of
the polynomial exponents (degree 0, then 1, ...;
``a`` descending), ``T = (n + 1)(n + 2) / 2`` for degree ``n``. The map goes
from the nominal ``k = p dk`` to the true ``k``, as ``W`` does.

The direct-ptychography methods read the warped ``vBF.k``
(:func:`bright_field_k`): they evaluate the aperture and the aberration
function at the true ``K``, and the data stay on their pixels. The scan
rotation (``meta.rotation``) stays a separate, outer factor.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "as_detector_warp",
    "bright_field_k",
    "detector_warp_matrix",
    "polynomial_warp",
    "warp_table",
]


def detector_warp_matrix(ellipticity: float, tilt_deg: float) -> np.ndarray:
    """The area-preserving stretch ``W`` of an elliptical bright-field disk.

    Parameters
    ----------
    ellipticity : float
        ``1 - b / a`` of the disk on the detector, minor over major semi-axis,
        in ``[0, 1)``. ``0`` is a round disk and gives the identity.
    tilt_deg : float
        Angle of the major axis from the row (``k_y``) axis toward the column
        (``k_x``) axis, in degrees.

    Returns
    -------
    numpy.ndarray
        ``(2, 2)`` float64, ``(row, col)`` order. Symmetric, ``det W = 1``. It
        maps a pixel offset on the major axis to ``sqrt(1 - e)`` times that
        offset, and one on the minor axis to ``1 / sqrt(1 - e)`` times it, so
        the elliptical disk maps to a round one.

    Raises
    ------
    ValueError
        If ``ellipticity`` is not in ``[0, 1)`` or a value is not finite.
    """
    e, t = float(ellipticity), math.radians(float(tilt_deg))
    if not (math.isfinite(e) and math.isfinite(t) and 0.0 <= e < 1.0):
        raise ValueError(
            f"ellipticity must be finite and in [0, 1), tilt finite; got "
            f"{ellipticity!r}, {tilt_deg!r}"
        )
    if e == 0.0:
        return np.eye(2)  # exactly: no rounding from the rotation
    c, s = math.cos(t), math.sin(t)
    rot = np.array([[c, -s], [s, c]])
    stretch = math.sqrt(1.0 - e)
    return rot @ np.diag([stretch, 1.0 / stretch]) @ rot.T


#: ``k_ref`` of a polynomial warp table, 1/A.
WARP_REFERENCE = 1.0


def _degree_of(n_terms: int) -> int | None:
    """The degree of a ``(T, 2)`` table, or None if ``T`` is not triangular."""
    n = (math.isqrt(8 * n_terms + 1) - 3) // 2
    return n if n >= 0 and (n + 1) * (n + 2) // 2 == n_terms else None


def as_detector_warp(value: Any) -> Tensor | None:
    """Validate a detector warp: None, a ``(2, 2)`` ``W`` or a ``(T, 2)`` table.

    Parameters
    ----------
    value : array_like or None
        The warp ``W``, the polynomial table (see the module docstring), or
        None for no warp.

    Returns
    -------
    torch.Tensor or None
        A float64 tensor on the CPU, or None.

    Raises
    ------
    ValueError
        If the shape is neither ``(2, 2)`` nor ``(T, 2)`` with a triangular
        ``T >= 3``, an entry is not finite, or ``det W`` is not positive (a
        fold or a mirror is not a warp: a transposed detector is a separate
        setting).
    """
    if value is None:
        return None
    w = torch.as_tensor(np.asarray(value, dtype=np.float64), dtype=torch.float64)
    table = w.ndim == 2 and w.shape[1] == 2 and w.shape[0] >= 3
    if table and _degree_of(int(w.shape[0])) is None:
        table = False
    if not (table or tuple(w.shape) == (2, 2)) or not bool(torch.isfinite(w).all()):
        raise ValueError(
            "detector_warp must be a finite (2, 2) matrix or a (T, 2) polynomial "
            f"table with T = 3, 6, 10, ...; got {value!r}"
        )
    if table:
        return w
    if float(torch.linalg.det(w)) <= 0.0:
        raise ValueError(
            "detector_warp must have det > 0; a mirrored detector is a "
            f"transposed detector, not a warp (got {w.tolist()})"
        )
    return w


def bright_field_k(centered: Tensor, sampling: Tensor, meta: Any) -> Tensor:
    """The wave vector ``(ky, kx)`` each bright-field pixel records, float32.

    Parameters
    ----------
    centered : torch.Tensor
        ``(n, 2)`` pixel offsets from the bright-field centroid, ``(row, col)``.
    sampling : torch.Tensor
        ``(1, 2)`` reciprocal sampling ``dk`` per detector axis, 1/A per pixel,
        on the device of ``centered``.
    meta : object
        The acquisition metadata. Its ``detector_warp`` (absent or None: no
        warp) is applied as ``k = W (p dk)``, or as the polynomial.

    Returns
    -------
    torch.Tensor
        ``(n, 2)`` float32. With no warp it is ``(centered * sampling)``
        cast to float32, bit for bit as before the warp existed.
    """
    k = centered * sampling.expand_as(centered)
    # Validated here too: ``meta.detector_warp`` may be assigned directly.
    warp = as_detector_warp(getattr(meta, "detector_warp", None))
    if warp is not None:
        w = warp.to(device=k.device)
        k = k.to(torch.float64)
        k = k @ w.T if tuple(w.shape) == (2, 2) else polynomial_warp(k, w)
    return k.to(torch.float32)


def polynomial_warp(k: Tensor, table: Tensor) -> Tensor:
    """``k + sum_ab c_ab (k_y / k_ref)^a (k_x / k_ref)^b`` for a ``(T, 2)`` table.

    Parameters
    ----------
    k : torch.Tensor
        ``(..., 2)`` nominal ``(k_y, k_x)``, 1/A.
    table : torch.Tensor
        ``(T, 2)`` coefficients (module docstring), on the device of ``k``.

    Returns
    -------
    torch.Tensor
        ``(..., 2)`` true ``k``, dtype of ``k``.
    """
    degree = _degree_of(int(table.shape[0]))
    u = k / WARP_REFERENCE
    terms = [
        u[..., 0] ** a * u[..., 1] ** (d - a)
        for d in range(int(degree or 0) + 1)
        for a in range(d, -1, -1)
    ]
    return k + torch.stack(terms, -1) @ table.to(k.dtype)


def warp_table(warp: Any, reference: float = WARP_REFERENCE) -> Tensor:
    """The ``(T, 2)`` polynomial table of a warp; ``W`` becomes degree 1.

    Parameters
    ----------
    warp : array_like
        A ``(2, 2)`` ``W`` or a ``(T, 2)`` table (:func:`as_detector_warp`).
    reference : float, optional
        ``k_ref`` of the returned table, 1/A (default ``WARP_REFERENCE``).
        The same field: a degree-``d`` row scales by ``(reference /
        WARP_REFERENCE)^d``. A ``k_ref`` near the detector's reach keeps a
        Newton inverse well scaled.

    Returns
    -------
    torch.Tensor
        ``(T, 2)`` float64 on the CPU.

    Raises
    ------
    ValueError
        If ``warp`` is None or not a valid warp.
    """
    w = as_detector_warp(warp)
    if w is None:
        raise ValueError("warp_table needs a warp, got None")
    if tuple(w.shape) == (2, 2):
        table = torch.zeros(3, 2, dtype=torch.float64)
        # Rows (1, 0) and (0, 1) multiply k_y and k_x: the columns of W - I.
        table[1:] = WARP_REFERENCE * (w - torch.eye(2, dtype=torch.float64)).T
    else:
        table = w
    n = int(table.shape[0])
    degree = [d for d in range(n) for _ in range(d + 1)][:n]
    scale = torch.tensor([(reference / WARP_REFERENCE) ** d for d in degree])
    return table * scale.to(torch.float64)[:, None]
