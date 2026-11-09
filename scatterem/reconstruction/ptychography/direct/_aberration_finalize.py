"""Log a finished aberration fit and write it onto the virtual bright field."""

import logging

import torch

from scatterem.physics.optics import AMPLITUDE_LAYOUT, lens_to_amplitude, unpack_lens

from ._common import _require_meta

logger = logging.getLogger(__name__)


def _log_and_finalize_aberration_fit(
    vBF,
    aberrations: torch.Tensor,
    rotation,
    direct_ptycho_image: torch.Tensor | None,
    update_dataset: bool,
    fit_rotation: bool,
) -> None:
    """Log the fitted coefficients (cartesian + polar), then write the fit
    onto ``vBF``.

    ``vBF.aberration_rotation`` gets ``rotation`` as a float in every mode: the
    scan rotation (deg) the aberrations hold at. With ``update_dataset=False``
    it is the only record of that rotation. With `update_dataset`, the write
    sets ``aberrations`` and resets ``defocus_guess`` to 0.0, as
    ``bright_field`` does.

    No alignment overlay reaches this point: the bright-field-shift
    self-calibration clears its own before returning (see
    ``bright_field._self_calibrate_bright_field_shifts``), so the autofocus
    stage and every reconstruction reusing the cached vBF see the pristine G.
    """
    logger.info("Optimized aberration coefficients:")
    polar = unpack_lens(
        lens_to_amplitude(aberrations.detach()), layout=AMPLITUDE_LAYOUT
    )
    logger.info("  C10 (Defocus): %.3f", polar["C10"])
    logger.info("  C12 (Astigmatism magnitude): %.3f", polar["C12"])
    logger.info("  C21 (Coma magnitude): %.3f", polar["C21"])
    logger.info("  C30 (Spherical aberration): %.3f", polar["C30"])
    logger.info("  phi12 (Astigmatism angle): %.3f", polar["phi12"])
    logger.info("  phi21 (Coma angle): %.3f", polar["phi21"])

    vBF.aberration_rotation = float(rotation)
    if update_dataset:
        _write_fitted_aberrations(
            vBF, aberrations.detach(), rotation if fit_rotation else None
        )
        if direct_ptycho_image is not None:
            vBF.weak_phase_image = direct_ptycho_image.detach()


def _write_fitted_aberrations(
    dataset, aberrations: torch.Tensor, rotation=None
) -> None:
    """Write a fit onto ``dataset.meta`` with the defocus counted once.

    The fitted C10 is in ``aberrations[0]`` alone, so ``defocus_guess`` is reset
    to 0.0. A non-zero guess would count the defocus twice: the probe has
    C10 = ``aberrations[0] - defocus_guess``. ``rotation`` is written when it is
    not None. ``bright_field`` resets the guess in the same way.
    """
    meta = _require_meta(dataset)
    meta.aberrations.array = aberrations
    meta.defocus_guess = 0.0
    if rotation is not None:
        meta.rotation = rotation
