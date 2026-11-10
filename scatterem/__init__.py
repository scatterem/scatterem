"""Electron scattering simulation and 4D-STEM reconstruction.

This release holds the published methods of three papers:

* Fused full-field STEM: direct (single-side-band) ptychography,
  tilt-corrected dark-field imaging, and their SSNR-weighted Wiener fusion
  (`doi.org/10.1002/advs.76620 <https://doi.org/10.1002/advs.76620>`_). The
  names below are its entry points.
* PRISM-EELS with the beam-partitioned (BiP-PRISM) S-matrices
  (`doi.org/10.1088/2515-7639/aebdd0
  <https://doi.org/10.1088/2515-7639/aebdd0>`_), in
  :mod:`scatterem.simulation.eels`; the driver is
  :class:`scatterem.simulation.eels.PrismEelsSimulator`.
* Elastic PRISM, in :mod:`scatterem.nn.scan_exit_wave` (``PrismScanOp``,
  ``build_prism_smeta``) and through the same driver.

See ``README.md``.

This module sets no process-wide state itself: no
float32 matmul precision, no warnings filter. NVIDIA Warp is initialised by the
``scatterem.nn`` and ``scatterem.reconstruction`` package inits, which call
``warp.init()`` because their modules hand torch tensors to Warp through
``wp.from_torch``, which does not initialise Warp by itself; importing this
module imports ``reconstruction``, so Warp is initialised then. The float32
matmul precision is torch's default (``"highest"``) unless you set it; the
published figures were computed at ``"medium"`` (TF32), which
``torch.set_float32_matmul_precision("medium")`` restores, and so does
:func:`configure` (``scatterem.configure(matmul_precision="medium")``).
"""

from ._configure import configure
from .data.datasets import Dataset4DStem
from .data.public import (
    You2026AuLowDose,
    You2026Carbon,
    You2026Co3O4,
    You2026Gd2O3,
)
from .reconstruction import (
    direct_ptychography,
    fused_full_field,
    tilt_corrected_dark_field,
)

__version__ = "0.4.2"

__all__ = [
    "Dataset4DStem",
    "configure",
    "You2026AuLowDose",
    "You2026Carbon",
    "You2026Co3O4",
    "You2026Gd2O3",
    "direct_ptychography",
    "fused_full_field",
    "tilt_corrected_dark_field",
]
