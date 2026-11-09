"""Direct (non-iterative) ptychography: single-side-band reconstruction and its
aberration/rotation calibration.

This file re-exports the functions, and the
:func:`~scatterem.reconstruction.run` configs of the single-plane SSB runner
(``_run``) and the aberration fit (``_aberration_run``).

Names are re-exported with the redundant-alias idiom, and the module-level
tunables are served by ``__getattr__`` rather than copied, so that
``enable_chunk_sum_first()`` is still visible through the package instead of
being frozen at its import-time value.
"""

from . import _chunking as _chunking
from . import _config as _config
from ._aberration_run import AberrationFitConfig as AberrationFitConfig
from ._aberration_run import AberrationFitResult as AberrationFitResult
from ._autofocus_image import _image_sharpness as _image_sharpness
from ._autofocus_image import _windowed_roi as _windowed_roi
from ._bf_parallax import measure_bf_parallax as measure_bf_parallax
from ._bright_field_fit_result import BrightFieldShiftFit as BrightFieldShiftFit
from ._chunking import _direct_ptychography as _direct_ptychography
from ._chunking import _direct_ptychography_kouter as _direct_ptychography_kouter
from ._chunking import _g_chunk_kouter_view as _g_chunk_kouter_view
from ._chunking import _iter_chunk_images as _iter_chunk_images
from ._chunking import _upsampled_grid as _upsampled_grid
from ._com_rotation_prior import (
    estimate_scan_rotation_from_com as estimate_scan_rotation_from_com,
)
from ._common import PhaseSignMode as PhaseSignMode
from ._common import _chunk_ranges as _chunk_ranges
from ._common import _orient_phase_image as _orient_phase_image
from ._common import _phase_image_skewness as _phase_image_skewness
from ._common import _resolve_upsample_int as _resolve_upsample_int
from ._common import _resolve_vbf as _resolve_vbf
from ._config import _MIN_HALFSPLIT_PIXELS as _MIN_HALFSPLIT_PIXELS
from ._config import enable_chunk_sum_first as enable_chunk_sum_first
from ._gamma_weighting import DEFAULT_WIENER_ALPHA as DEFAULT_WIENER_ALPHA
from ._gamma_weighting import GammaWeighting as GammaWeighting
from ._gamma_weighting import (
    direct_ptychography_wiener as direct_ptychography_wiener,
)
from ._gamma_weighting import (
    reconstruct_wiener_weighted as reconstruct_wiener_weighted,
)
from ._plotting import plot_bright_field_shifts as plot_bright_field_shifts
from ._rolloff import rolloff_2alpha as rolloff_2alpha
from ._run import DirectPtychographyConfig as DirectPtychographyConfig
from ._run import DirectPtychographyResult as DirectPtychographyResult
from .aberration_fitting import (
    _aberration_precondition_scales as _aberration_precondition_scales,
)
from .aberration_fitting import _build_gradient_mask as _build_gradient_mask
from .aberration_fitting import _optimize_aberrations_tv as _optimize_aberrations_tv
from .aberration_fitting import _resolve_aberration_stages as _resolve_aberration_stages
from .aberration_fitting import determine_aberrations as determine_aberrations
from .bright_field import _rotation_matrix_2x2 as _rotation_matrix_2x2
from .bright_field import (
    aberrations_and_rotation_from_bright_field_shifts as aberrations_and_rotation_from_bright_field_shifts,
)
from .bright_field import (
    fit_aberrations_and_rotation_to_bright_field_shifts as fit_aberrations_and_rotation_to_bright_field_shifts,
)
from .bright_field import (
    fit_low_order_aberrations_and_rotation_closed_form as fit_low_order_aberrations_and_rotation_closed_form,
)
from .core import _checkerboard_parity as _checkerboard_parity
from .core import direct_ptychography as direct_ptychography
from .core import (
    direct_ptychography_empirical_ssnr as direct_ptychography_empirical_ssnr,
)
from .ctf import direct_ptychography_ssnr as direct_ptychography_ssnr
from .ctf import phase_contrast_transfer_function as phase_contrast_transfer_function

# Read module-level data through the module that owns
# it, so a runtime rebinding (enable_chunk_sum_first) is visible here.
_LIVE_ATTRS = {
    "_ABERRATION_RADIAL_POWER": _config,
    "_TV_CHUNK_CKPT_BUDGET_BYTES": _config,
    "_TV_CHUNK_CACHE_BUDGET_BYTES": _config,
    "_MIN_HALFSPLIT_PIXELS": _config,
    "_CHUNK_SUM_FIRST": _config,
    "_UPSAMPLED_GRID_CACHE": _chunking,
    "_UPSAMPLED_GRID_CACHE_MAX": _chunking,
}


def __getattr__(name: str):
    owner = _LIVE_ATTRS.get(name)
    if owner is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(owner, name)


def __dir__() -> list[str]:
    return sorted(list(globals()) + list(_LIVE_ATTRS))
