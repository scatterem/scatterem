"""Direct (non-iterative) reconstruction of 4D-STEM data."""

import warp as _wp

from ._runner import Provenance, ReconstructionConfig, Result, register_runner, run
from .darkfield import tilt_corrected_dark_field
from .fusion.ffstem import fused_full_field
from .ptychography.direct import direct_ptychography

__all__ = [
    "Provenance",
    "ReconstructionConfig",
    "Result",
    "direct_ptychography",
    "fused_full_field",
    "register_runner",
    "run",
    "tilt_corrected_dark_field",
]


# Initialise Warp here, once, for every module below this package: they hand
# torch tensors to Warp through `wp.from_torch`, which does not initialise Warp
# by itself. The top-level `scatterem/__init__.py` does not call it, but it
# imports `scatterem.reconstruction`, so `import scatterem` starts Warp.
_wp.init()
