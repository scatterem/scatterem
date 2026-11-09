"""The result of the bright-field-shift aberration fit, with its quality.

A bright-field-shift (parallax) fit can end far from the measured shifts and
still return numbers. :class:`BrightFieldShiftFit` carries the relative
residual and a ``reliable`` flag, so that a pipeline can check the fit before
it writes the values as results. The class docstring gives the fallback order.
"""

import warnings
from collections.abc import Iterator
from dataclasses import dataclass, replace
from math import isnan, nan

import torch

__all__ = ["BrightFieldShiftFit"]

#: Relative residual above which the fit is not reliable (the old warning level).
DEFAULT_RESIDUAL_RTOL = 0.15


@dataclass(frozen=True)
class BrightFieldShiftFit:
    """Aberrations and rotation fitted to bright-field shifts, with the fit quality.

    The object unpacks like the 2-tuple the fit returned before,
    ``aberrations, rotation = fit``, so callers that need only the values do
    not change. Callers that act on the values (a pipeline that writes them as
    results) check ``reliable`` first and use the fallback order in the
    notes below when it is ``False``.

    Parameters
    ----------
    aberrations : torch.Tensor
        The 12 fitted aberration coefficients, in Angstrom. Their frame is
        that of the function that returned the object:
        :func:`fit_aberrations_and_rotation_to_bright_field_shifts` returns the
        SHIFT-MODEL frame (the argument-rotated frame of
        ``scatterem.utils.transfer.aberrations_to_image_shifts``, the model it
        fits), and :func:`aberrations_and_rotation_from_bright_field_shifts`
        (which :func:`determine_aberrations` calls) returns the DETECTOR frame
        of ``meta.aberrations``. The two differ in the m-fold pairs at
        ``rotation != 0``; ``scatterem.utils.transfer.shift_model_to_detector_frame``
        at the fitted rotation converts the first to the second.
    rotation : torch.Tensor
        The fitted scan rotation, in degrees (0-dim or shape ``(1,)``).
    relative_residual : float
        ``RMS(fitted shifts - measured shifts) / RMS(measured shifts)`` over
        the pixels in the fit. ``nan`` when no fit ran.
    residual_rtol : float
        The residual limit that ``reliable`` uses.
    well_posed : bool
        ``False`` when the rotation is free and the closed-form seed found the
        defocus too small to fix it (the rotation is then the prior, not a
        fit). A held rotation (``fit_rotation=False``) is always well posed.

    Notes
    -----
    A good fit has a relative residual below about 0.15. On low-contrast or
    one-directional specimens the fit can end at
    0.8-1.05 and still return a C1 and a rotation; those values are not
    measurements. Fallback order when ``reliable`` is ``False``:

    1. Repair dropped frames, then fit again with the rotation held at a
       known value and a lower order (``fit_rotation=False``,
       ``correct_order=0``).
    2. Choose C10 by eye in an SSB defocus series, at a known rotation and
       with A1 held fixed, and record the choice.
    """

    aberrations: torch.Tensor
    rotation: torch.Tensor
    relative_residual: float
    residual_rtol: float = DEFAULT_RESIDUAL_RTOL
    well_posed: bool = True

    @property
    def reliable(self) -> bool:
        """``True`` when the residual is at most the limit and the fit is well posed.

        ``nan`` residuals count as unreliable. The fit emits a ``UserWarning``
        in exactly the cases where this is ``False``.

        Returns
        -------
        bool
            Whether the caller can use the values without a fallback.
        """
        return bool(self.relative_residual <= self.residual_rtol) and self.well_posed

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Yield ``aberrations`` and then ``rotation``, for 2-tuple unpacking.

        Returns
        -------
        Iterator of torch.Tensor
            The fitted aberrations, then the fitted rotation.
        """
        return iter((self.aberrations, self.rotation))


def _shift_fit_result(
    aberrations: torch.Tensor,
    rotation: torch.Tensor,
    final_shifts: torch.Tensor,
    target_shifts: torch.Tensor,
    include: torch.Tensor,
    residual_rtol: float,
    well_posed: bool,
) -> BrightFieldShiftFit:
    """Measure the relative residual, warn when it or the seed fails, and wrap.

    The two warnings are those the fit always gave; ``reliable`` is ``False``
    exactly when one of them is emitted.
    """
    with torch.no_grad():
        masked_target = target_shifts[include]
        rms_resid = (final_shifts[include] - masked_target).pow(2).mean().sqrt()
        rms_target = masked_target.pow(2).mean().sqrt().clamp_min(1e-12)
        rel = (rms_resid / rms_target).item()
    if not rel <= residual_rtol:
        warnings.warn(
            f"Aberration fit residual is high (relative residual {rel:.2f} > "
            f"{residual_rtol}); the fit may be unreliable. Check "
            "BrightFieldShiftFit.reliable and use the fallback order in its "
            "docstring.",
            UserWarning,
            stacklevel=3,
        )
    if not well_posed:
        warnings.warn(
            "Near-zero defocus: scan rotation is not determined by the bright-field "
            "shifts; using the rotation prior. Aberration/rotation estimate is "
            "degenerate.",
            UserWarning,
            stacklevel=3,
        )
    return BrightFieldShiftFit(
        aberrations=aberrations,
        rotation=rotation,
        relative_residual=rel,
        residual_rtol=float(residual_rtol),
        well_posed=bool(well_posed),
    )


def _worst_of_levels(fits: list[BrightFieldShiftFit]) -> BrightFieldShiftFit:
    """The last level's values with the worst quality over all levels.

    A bin level fits the previous level's model plus the new measured
    increment, so a misfit at an earlier level is not in the last level's
    residual. The result is ``reliable`` only when every level was: its
    residual is the largest (``nan`` if any is ``nan``) and it is well posed
    only when every level was.
    """
    residuals = [fit.relative_residual for fit in fits]
    worst = nan if any(isnan(r) for r in residuals) else max(residuals)
    return replace(
        fits[-1],
        relative_residual=worst,
        well_posed=all(fit.well_posed for fit in fits),
    )
