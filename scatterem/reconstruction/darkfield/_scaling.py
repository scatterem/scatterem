"""Off-detector dark-field scaling and its dead-frame guard.

The per-scan-position factor ``dark_field_scaling`` applies to every dark-field
segment image in
:func:`scatterem.reconstruction.darkfield.core.tilt_corrected_dark_field`,
and the fraction of the incident current below which a scan position counts as a
dropped detector frame and is left uncorrected.
"""

import logging

import torch

from scatterem.utils.lazy_plotting import vis

logger = logging.getLogger(__name__)

# A scan position whose total recorded counts fall below this fraction of the
# incident probe current is a dropped/blank detector frame, not a measurement.
# ``dark_field_scaling`` leaves those uncorrected (factor 1); see the guard
# there for why an unguarded factor turns one stray count into a hot pixel.

_DEAD_FRAME_FRACTION = 1e-3


def _dark_field_scaling(bf_sum, df_sum, dead_frame_fraction, verbosity):
    """Per-scan-position factor for the dark field scattered past the detector.

    ``bf_sum`` / ``df_sum`` are the per-scan-position total counts recorded in
    the bright-field disk and outside it (disjoint, together exhaustive). The
    factor is ``(max_p' I_total(p') - I_BF(p)) / I_DF_recorded(p)``, set to 1 at
    a scan position that recorded less than ``dead_frame_fraction`` of the
    incident current (see :data:`_DEAD_FRAME_FRACTION`). Returns a tensor of the
    scan shape, to multiply each dark-field segment image by.
    """
    # Incident probe current: the brightest total recorded in the scan, i.e.
    # the position that scattered nothing past the detector edge.
    recorded = bf_sum + df_sum
    incident = recorded.max()
    scaling = (incident - bf_sum) / (df_sum + 1e-12)
    # Dead-frame guard. Detectors drop frames (the ARINA blanks its first
    # scan row); there bf_sum = df_sum = 0, the factor degenerates to
    # incident/eps ~ 1e13, and a single stray count in such a frame becomes
    # a hot pixel that dominates the image. A position that recorded
    # essentially nothing measured no probe, so the correction is undefined
    # there and must fall back to 1. The threshold is far below any real
    # off-detector loss (that still records a sizable fraction of incident).
    alive = recorded > dead_frame_fraction * incident
    n_dead = int((~alive).sum())
    if n_dead:
        scaling = torch.where(alive, scaling, torch.ones_like(scaling))
    if verbosity > 0:
        logger.info(
            "Dark-field scaling: factor %.3f-%.3f (median %.3f); "
            "incident %.4g counts/probe; %d dead frame(s) left uncorrected",
            scaling.min(),
            scaling.max(),
            scaling.median(),
            incident,
            n_dead,
        )
    if verbosity > 1:
        vis.show_2d(scaling, cbar=True, title="Dark field scaling factor")
    return scaling
