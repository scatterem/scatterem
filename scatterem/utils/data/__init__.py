"""Domain data containers: the 4D-STEM dataset, its metadata, and calibration.

Written for the public release. The private tree's version is eight lines of
eager star-imports that pull in a PRISM S-matrix container, the tomography
sampler and the 6D sparse-Bragg containers — all belonging to papers that are
not out yet — so it cannot be exported.

Only the names the FF-STEM pipeline needs are re-exported here. Everything else
is reachable from its own module.
"""

from .data_classes import Metadata4dstem
from .datasets import Dataset4dstem, DatasetVirtualBrightField4dstem
from .sampling import Sampling

__all__ = [
    "Dataset4dstem",
    "DatasetVirtualBrightField4dstem",
    "Metadata4dstem",
    "Sampling",
]
