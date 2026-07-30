"""Fused full-field STEM reconstruction.

Direct (single-side-band) ptychography, tilt-corrected dark-field imaging, and
their SSNR-weighted Wiener fusion. See ``README.md``; the method is described in
`doi.org/10.1002/advs.76620 <https://doi.org/10.1002/advs.76620>`_.

Written for the public release. The private tree's version re-exports the ``io``
subpackage, whose readers and preprocessing pipeline belong to other work.

Importing this module has two deliberate side effects, both inherited from the
full library so that behaviour matches: it initialises NVIDIA Warp, and it sets
torch's float32 matmul precision to "medium" (TF32), which is what the published
reconstructions ran with.
"""

import warnings

import torch
import warp

from .datasets import (
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
from .utils.data.datasets import Dataset4dstem

warnings.filterwarnings(
    "ignore", message="The 'train_dataloader' does not have many workers"
)

torch.set_float32_matmul_precision("medium")
warp.init()

__version__ = "0.2.0"

__all__ = [
    "Dataset4dstem",
    "You2026AuLowDose",
    "You2026Carbon",
    "You2026Co3O4",
    "You2026Gd2O3",
    "direct_ptychography",
    "fused_full_field",
    "tilt_corrected_dark_field",
]
