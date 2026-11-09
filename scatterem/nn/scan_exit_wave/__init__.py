"""Scan-time exit-wave operators: elastic PRISM and per-position multislice.

Written for the public release. :class:`PrismScanOp` computes exit waves from
a (beam-partitioned) PRISM S-matrix; :class:`MultisliceScanOp` is the
per-position multislice reference; ``build_prism_smeta`` builds the beam basis.
"""

from scatterem.nn.scan_exit_wave.base import ScanExitWaveOperator
from scatterem.nn.scan_exit_wave.multislice import MultisliceScanOp
from scatterem.nn.scan_exit_wave.prism import PrismScanOp
from scatterem.nn.scan_exit_wave.prism_smeta import (
    build_prism_smeta,
    build_prism_smeta_for_probe,
)

__all__ = [
    "MultisliceScanOp",
    "PrismScanOp",
    "ScanExitWaveOperator",
    "build_prism_smeta",
    "build_prism_smeta_for_probe",
]
