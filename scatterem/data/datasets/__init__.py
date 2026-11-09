"""4D-STEM dataset containers.

This file names the dataset containers of the package.

The surface matches the wildcard surface of the former flat module for these
classes: ``__all__`` holds the public names, and the private
helpers other exported modules import by name are re-exported with the
redundant-alias idiom so they stay reachable without entering ``__all__``.
``with_meta`` is public.
"""

from ._meta import with_meta
from ._vbf_subsets import _bf_mask_geometry as _bf_mask_geometry
from ._vbf_subsets import _normalized_mean_pattern as _normalized_mean_pattern
from .base import Dataset
from .base import _place_aberrations_on_device as _place_aberrations_on_device
from .dataset_4dstem import Dataset4DStem
from .dataset_4dstem import _metadata4dstem_from_physics as _metadata4dstem_from_physics
from .virtual_bright_field import DatasetVirtualBrightField4DStem

__all__ = [
    "Dataset",
    "Dataset4DStem",
    "DatasetVirtualBrightField4DStem",
    "with_meta",
]
