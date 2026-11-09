"""Where a dataset's array lives: in memory, memory-mapped, HDF5, zarr, or a blob.

Every source has the scan dimensions first and offers ``read`` (one slice per scan
dimension), ``gather`` (flat scan points), ``view`` (a lazy crop) and an explicit
``materialize``/``release``. A :class:`RowTransform` turns stored values into what a
consumer reads.
"""

from .._pattern_transform import RowTransform
from ._h5 import H5RowSource
from ._memmap import MemmapSource
from ._protocol import ArraySource
from ._slice import SlicedSource
from ._tensor import TensorSource
from ._zarr import ZarrSource

__all__ = [
    "ArraySource",
    "H5RowSource",
    "MemmapSource",
    "RowTransform",
    "SlicedSource",
    "TensorSource",
    "ZarrSource",
]

