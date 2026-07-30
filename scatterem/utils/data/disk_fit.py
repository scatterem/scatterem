"""Bright-field disk geometry from an averaged diffraction pattern.

Written from the geometry rather than adapted from an existing implementation:
``utils/data/bright_field.py::_area_method`` is a port of a GPL-3.0 upstream,
which an Apache-2.0 release cannot distribute.

The method is the natural one -- sweep an intensity threshold, convert the area
above it into an equivalent-circle radius, and read the radius off the flattest
part of that curve -- with two deliberate differences from the code it replaces:

1. **The threshold sweep is normalised by a median-filtered maximum**, not the
   raw maximum. A single stuck pixel is orders of magnitude brighter than the
   probe, so normalising by the raw max makes every threshold select roughly
   that one pixel; the radius curve is then perfectly flat and a plateau-seeking
   fit reports the flatness as a confident sub-pixel radius. A 3x3 median
   removes isolated spikes and leaves the disk plateau untouched.
2. **The sweep brackets the half-maximum symmetrically** (0.25 to 0.75 by
   default) and the radius is the *median* over the sweep. The half-maximum
   contour is the conventional edge of a soft-edged probe, and for an edge
   profile of the usual form the radius curve is antisymmetric about it --
   ``r(t) = R + b*atanh(1 - 2t)`` for a ``tanh`` roll-off of width ``b``, so
   thresholds paired symmetrically about 0.5 have radii that average to ``R``
   exactly. Taking the median therefore cancels the edge-softness bias to first
   order while staying robust to the tails, where ``atanh`` diverges.

   A "find the flattest window of the radius curve" rule looks equivalent and is
   not: pixel-count quantisation makes the curve a staircase, many windows tie on
   flatness, and any fixed tie-break biases the threshold systematically. That
   bias is toward larger radii (the curve decreases in ``t``) and reaches +3.6%
   on an 8 px disk -- straight into the reciprocal-space calibration.

Both checks that can fail do so loudly: a pattern with no bright compact region
raises rather than returning a number that looks like a measurement.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from numpy.typing import NDArray


def _as_2d_float_tensor(dp) -> torch.Tensor:
    tensor = dp if isinstance(dp, torch.Tensor) else torch.as_tensor(np.asarray(dp))
    if tensor.ndim != 2:
        raise ValueError(
            "expected a single 2D diffraction pattern (average over the scan "
            f"first); got an array of shape {tuple(tensor.shape)}"
        )
    if not tensor.is_floating_point():
        tensor = tensor.to(torch.float32)
    return tensor


def _spike_free_maximum(dp: torch.Tensor) -> torch.Tensor:
    """Maximum of a 3x3 median-filtered copy, so hot pixels cannot set the scale.

    A hot pixel is isolated by definition, so the median of any 3x3 window
    containing it is a neighbouring value. The interior of the bright-field disk
    is locally flat, so its median equals its value.
    """
    padded = torch.nn.functional.pad(dp[None, None], (1, 1, 1, 1), mode="replicate")
    windows = padded.unfold(2, 3, 1).unfold(3, 3, 1).reshape(*dp.shape, 9)
    return windows.median(dim=-1).values.max()


def fit_bright_field_disk(
    dp,
    *,
    threshold_range: Tuple[float, float] = (0.25, 0.75),
    n_thresholds: int = 100,
    max_radius_fraction: float = 0.4,
    min_contrast: float = 3.0,
) -> Tuple[float, NDArray]:
    """Radius and centre of the central bright-field disk, in pixels.

    Args:
        dp: 2D averaged diffraction pattern (torch tensor or numpy array).
        threshold_range: low and high sweep limits as fractions of the
            spike-free maximum. Keep it **symmetric about 0.5** -- that symmetry
            is what makes the median unbiased. Widen it for a very soft edge.
        n_thresholds: number of thresholds in the sweep.
        max_radius_fraction: reject a fit whose radius exceeds this fraction of
            the smaller detector dimension. Guards against "fitting" a pattern
            that has no compact disk, where the sweep returns a radius of order
            the detector itself.
        min_contrast: reject a fit whose median intensity inside the disk is not
            at least this many times the median outside it.

    Returns:
        ``(radius, centre)`` with ``radius`` in pixels as a ``float`` and
        ``centre`` as a 2-element ``numpy`` array ordered ``[y, x]``.

    Raises:
        ValueError: if ``dp`` is not 2D, or if the pattern contains no
            identifiable bright-field disk.
    """
    dp = _as_2d_float_tensor(dp)
    ny, nx = dp.shape

    reference = _spike_free_maximum(dp)
    if not float(reference) > 0.0:
        raise ValueError(
            "no bright-field disk found: the pattern has no positive intensity "
            "once isolated hot pixels are discounted"
        )

    low, high = threshold_range
    thresholds = torch.linspace(
        float(low), float(high), int(n_thresholds), device=dp.device, dtype=dp.dtype
    )
    # Equivalent-circle radius of the area above each threshold.
    areas = torch.stack([(dp >= t * reference).sum() for t in thresholds]).to(dp.dtype)
    radii = torch.sqrt(areas / torch.pi)

    # Median over a sweep centred on the half-maximum. See the module docstring:
    # the radius curve is antisymmetric about the half-maximum, so the median
    # cancels edge-softness bias, and it is robust to the diverging tails.
    radius = float(radii.median())
    threshold = 0.5 * (float(low) + float(high)) * float(reference)

    if radius > max_radius_fraction * min(ny, nx):
        raise ValueError(
            "no bright-field disk found: the fitted radius "
            f"({radius:.1f} px) exceeds {max_radius_fraction:.0%} of the "
            f"smaller detector dimension ({min(ny, nx)} px), which means the "
            "threshold sweep found no compact bright region -- check that this "
            "is an averaged diffraction pattern and not, say, a virtual image"
        )

    mask = dp >= threshold
    inside = dp[mask]
    outside = dp[~mask]
    if inside.numel() == 0:
        raise ValueError("no bright-field disk found: the fitted disk is empty")
    if outside.numel() > 0:
        median_out = float(outside.median())
        median_in = float(inside.median())
        tiny = torch.finfo(dp.dtype).tiny
        if median_in < min_contrast * max(median_out, tiny):
            raise ValueError(
                "no bright-field disk found: median intensity inside the "
                f"fitted disk ({median_in:.4g}) is not {min_contrast}x the "
                f"median outside it ({median_out:.4g}), so the pattern has no "
                "disk-versus-background contrast"
            )

    # Intensity-weighted centroid, with intensities clipped at the spike-free
    # maximum so a hot pixel inside the mask cannot drag the centre toward it.
    weights = torch.clamp(dp, max=reference) * mask
    total = weights.sum()
    ys = torch.arange(ny, device=dp.device, dtype=dp.dtype)
    xs = torch.arange(nx, device=dp.device, dtype=dp.dtype)
    y0 = float((weights.sum(dim=1) * ys).sum() / total)
    x0 = float((weights.sum(dim=0) * xs).sum() / total)

    return radius, np.array([y0, x0])
