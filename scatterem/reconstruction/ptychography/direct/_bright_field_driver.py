"""Helpers of the bright-field self-calibration in :mod:`.bright_field`.

The starting reference image, the low-pass width, the spectrum source over a
vBF, the detector-frame publication of the fit and the fit's log lines.
"""

import logging
import math
from dataclasses import replace

import torch

from scatterem.data.aberrations import Aberrations
from scatterem.data.datasets import DatasetVirtualBrightField4DStem
from scatterem.utils.transfer import shift_model_to_detector_frame

from ._bright_field_closed_form import low_order_to_detector_frame
from ._bright_field_fit_result import BrightFieldShiftFit
from ._common import _require_meta

logger = logging.getLogger(__name__)

__all__ = [
    "VbfSpectra",
    "center_reference",
    "fwhm_to_sigma",
    "log_fit",
    "publish_detector_frame",
]


def center_reference(
    vBF: DatasetVirtualBrightField4DStem, n_center_indices: int, device
) -> torch.Tensor:
    """``fft2`` of the mean normalised image of the ``n`` innermost BF pixels.

    The normalisation reads only the first ``n`` radius-ordered columns. The
    values are the same as slicing the full normalised stack (the per-pixel
    divide commutes with the column selection), and it works on the streaming
    provider, whose stack may live on the CPU.
    """
    inds_r = vBF.bright_field_inds_ordered_by_radius
    norm_cols = vBF.diffraction_pattern_mean_normalized[
        inds_r[:n_center_indices, 0], inds_r[:n_center_indices, 1]
    ].to(device)
    center_cols = (
        vBF.array[..., :n_center_indices].to(device) / norm_cols[None, None, :]
    )
    return torch.fft.fft2(center_cols.mean(-1))


def fwhm_to_sigma(fwhm: float | None) -> float | None:
    """The Gaussian standard deviation of a FWHM (None stays None)."""
    if fwhm is None:
        return None
    return float(fwhm) / (2.0 * math.sqrt(2.0 * math.log(2.0)))


class VbfSpectra:
    """The BF image spectra of a vBF provider as a ``SpectrumSource``.

    Index ``i`` is BF pixel ``i`` in the original order of
    ``bright_field_inds_centered``; the provider stores its BF axis in radial
    order, so :meth:`read` maps the indices through the radial rank. The
    provider applies its alignment overlay at fetch time.
    """

    def __init__(self, vbf: DatasetVirtualBrightField4DStem) -> None:
        self._vbf = vbf
        order = vbf.bright_field_inds_radial_order.to(device="cpu", dtype=torch.long)
        self._rank = torch.argsort(order)
        self.image_count = int(order.numel())
        self.scan_grid_shape = (int(vbf.array.shape[0]), int(vbf.array.shape[1]))
        self.run_device = torch.device(vbf.device)

    def read(self, indices) -> torch.Tensor:
        """The spectra of BF pixels ``indices``, shape ``(n, H, W)``."""
        idx = self._rank[torch.as_tensor(list(indices), dtype=torch.long)]
        cols = self._vbf.get_G_columns(idx.to(self.run_device))
        return torch.permute(cols, (2, 0, 1)).to(self.run_device)


def log_fit(rotation_init, rotation, aberrations, final_shifts) -> None:
    """Log the result of one shift fit (INFO)."""
    logger.info("Final shifts max: %s", final_shifts.max(0).values.cpu().numpy())
    logger.info("Initial rotation: %.2f deg", rotation_init.item())
    logger.info("Final   rotation: %.2f deg", rotation.item())
    logger.info("Final aberrations: %s", aberrations.cpu().numpy())


def publish_detector_frame(
    vBF: DatasetVirtualBrightField4DStem,
    fit: BrightFieldShiftFit,
    update_dataset: bool,
) -> BrightFieldShiftFit:
    """The shift-model-frame fit turned to the detector frame, stored if asked.

    Everything before this models the measured shifts, in the shift model's
    frame; the vector returned and written into ``meta.aberrations`` is the
    detector-frame one its readers use, turned at the FINAL rotation (the
    identity, bit for bit, at rotation 0). The covariances go through the
    Jacobian of the same turn. The fit's quality fields are kept.
    """
    turned = {
        name: low_order_to_detector_frame(fit.aberrations, fit.rotation, cov)[1]
        for name in ("covariance", "covariance_bootstrap")
        if (cov := getattr(fit, name)) is not None
    }
    fit = replace(
        fit,
        aberrations=shift_model_to_detector_frame(fit.aberrations, fit.rotation),
        **turned,
    )
    if update_dataset:
        meta = _require_meta(vBF)
        meta.aberrations = Aberrations(array=fit.aberrations)
        meta.rotation = fit.rotation.item()
        meta.defocus_guess = 0.0  # C10 is in aberrations alone
    return fit
