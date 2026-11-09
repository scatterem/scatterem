"""``torch.nn`` modules of the elastic PRISM and multislice operators.

This package holds the propagator and the sub-pixel shift.

Setting an attribute here does not reach the module that defines it: code that
monkeypatches an internal must target :mod:`.propagator`.
"""

from . import propagator as propagator
from .propagator import Propagator as Propagator
from .propagator import _bcast_cmul_ as _bcast_cmul_
from .shift import SubpixShift as SubpixShift

__all__ = ["Propagator", "SubpixShift"]


def __getattr__(name: str) -> object:
    """Serve ``.propagator``'s Triton-shim cache live, not as a frozen binding."""
    if name == "_BCAST_CMUL":
        return propagator._BCAST_CMUL
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
