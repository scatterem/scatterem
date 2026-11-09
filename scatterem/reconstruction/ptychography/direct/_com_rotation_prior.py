"""The diffraction centre-of-mass (COM) prior on the scan rotation.

When ``meta.rotation`` is unset, the bright-field aberration fit in
:mod:`.bright_field` starts from the scan rotation that makes the
per-position diffraction COM field curl-free. :func:`_com_rotation_prior`
measures that field on the 4D dataset a virtual bright field was built from.
An eager dataset is measured on its in-memory cube. A dataset that reads in
blocks is never converted whole, so it is measured one
scan-row block at a time: a pattern's COM depends on that pattern alone, so
the blocks give the same field (and the same rotation) with one block on the
device at a time.
"""

import logging

import torch

from scatterem.data.calibration import (
    _calculate_intensities_center_of_mass,
    find_scan_rotation,
)
from scatterem.data.datasets import Dataset4DStem, DatasetVirtualBrightField4DStem
from scatterem.data.datasets._direct import reads_in_blocks

logger = logging.getLogger(__name__)


def estimate_scan_rotation_from_com(
    intensities: torch.Tensor,
    reciprocal_sampling: tuple[float, float] = (1.0, 1.0),
    dp_mask: torch.Tensor | None = None,
) -> float:
    """Independent scan<->detector rotation from the diffraction COM curl.

    Computes the per-scan-position diffraction COM and finds the rotation making the COM field
    curl-free, via the (validated) calibration utilities. Used as a prior for the bright-field
    aberration fit when ``meta.rotation`` is unset.

    Parameters
    ----------
    intensities : torch.Tensor
        Diffraction intensities, shape `(Rx, Ry, Qx, Qy)`.
    reciprocal_sampling : tuple of float, optional
        Detector reciprocal sampling `(dx, dy)`. Default is `(1.0, 1.0)`.
    dp_mask : torch.Tensor, optional
        Detector mask, shape `(Qx, Qy)`. Default is None.

    Returns
    -------
    float
        Estimated scan rotation, in degrees.
    """
    device = intensities.device
    (
        _,
        _,
        _,
        _,
        com_normalized_x,
        com_normalized_y,
    ) = _calculate_intensities_center_of_mass(
        intensities, reciprocal_sampling, dp_mask=dp_mask, device=device
    )
    return _curl_free_rotation_deg(com_normalized_x, com_normalized_y)


def _curl_free_rotation_deg(
    com_normalized_x: torch.Tensor, com_normalized_y: torch.Tensor
) -> float:
    """Rotation, in degrees, that makes a drift-corrected COM field curl-free."""
    found = find_scan_rotation(
        com_normalized_x,
        com_normalized_y,
        run_device=com_normalized_x.device,
        quiet=True,
        resolve_180=False,
    )
    return float(found.scan_turn_deg)


def _streamed_scan_rotation_from_com(
    dataset: Dataset4DStem, reciprocal_sampling: tuple[float, float]
) -> float:
    """:func:`estimate_scan_rotation_from_com` of a dataset that reads in blocks.

    Each block is read transformed, as the eager cube holds it, and its COM is measured
    by the routine the eager path uses; the drift fit and the rotation search then run
    once on the assembled ``(Rx, Ry)`` field.
    """
    device = dataset.device
    rows_x, rows_y = [], []
    for _, _, block in dataset.iter_scan_blocks(device="cpu"):
        com_x, com_y, *_ = _calculate_intensities_center_of_mass(
            block.to(device), reciprocal_sampling, fit_function=None
        )
        rows_x.append(com_x)
        rows_y.append(com_y)
    measured = (torch.cat(rows_x), torch.cat(rows_y))
    # Given ``com_measured`` the routine measures nothing: its first argument
    # only supplies the device.
    *_, com_normalized_x, com_normalized_y = _calculate_intensities_center_of_mass(
        measured[0], reciprocal_sampling, com_measured=measured
    )
    return _curl_free_rotation_deg(com_normalized_x, com_normalized_y)


def _com_rotation_prior(
    dataset: DatasetVirtualBrightField4DStem, fallback: float, verbosity: int = 0
) -> float:
    """Scan-rotation prior from the COM of the cube a virtual bright field came from.

    Parameters
    ----------
    dataset : DatasetVirtualBrightField4DStem
        Virtual bright field; its ``parent_dataset`` (eager or read in blocks) is
        measured.
    fallback : float
        Rotation returned, in degrees, when no prior can be measured: the vBF
        has no parent 4D dataset (built from an array or loaded on its own),
        or no pattern has a finite COM (e.g. a cube without signal). Logged as
        a warning when used. Any other failure propagates.
    verbosity : int, optional
        Logs the prior at INFO level when > 0. Default is 0.

    Returns
    -------
    float
        Scan rotation, in degrees.
    """
    parent = dataset.parent_dataset
    if parent is None:
        return _prior_unavailable("the vBF has no parent 4D dataset", fallback)
    # dr is the detector sampling; only the ratio of the two axes matters here, as the
    # COM-curl rotation search is invariant to uniform scaling of the COM field.
    recip = (float(parent.dr[0]), float(parent.dr[1]))
    try:
        if reads_in_blocks(parent):
            rotation = _streamed_scan_rotation_from_com(parent, recip)
        else:
            rotation = estimate_scan_rotation_from_com(
                parent.array, reciprocal_sampling=recip
            )
    except ValueError as e:  # the drift fit found no finite COM to fit
        return _prior_unavailable(str(e), fallback)
    if verbosity > 0:
        logger.info("COM-curl scan-rotation prior: %.2f deg", rotation)
    return rotation


def _prior_unavailable(reason: str, fallback: float) -> float:
    """Log why the COM prior cannot be measured, and return ``fallback``."""
    logger.warning(
        "COM rotation prior unavailable (%s); using meta.rotation = %s deg",
        reason,
        fallback,
    )
    return fallback
