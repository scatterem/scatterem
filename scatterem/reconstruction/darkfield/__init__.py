"""Tilt-corrected dark and bright field: parallax-corrected shift-and-sum.

:mod:`.core` holds the reconstruction, :mod:`.drizzle` the area-overlap resampling it
uses by default, :mod:`._run` the :func:`~scatterem.reconstruction.run` runners.
"""

from ._run import TiltCorrectedDarkFieldConfig, TiltCorrectedDarkFieldResult

from .core import tilt_corrected_dark_field, tilt_corrected_dark_field_depth_section

__all__ = [
    "TiltCorrectedDarkFieldConfig",
    "TiltCorrectedDarkFieldResult",
    "tilt_corrected_dark_field",
    "tilt_corrected_dark_field_depth_section",
]
