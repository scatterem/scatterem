"""A device move that moves every tensor a dataset holds, not only its array."""

import torch
from torch import Tensor

from .base import _place_aberrations_on_device

__all__ = ["_MovesTensorState"]


class _MovesTensorState:
    """Mixin: ``dataset.device = d`` moves all tensor state of the instance.

    The plain ``Dataset`` setter moves the array only. A dataset that keeps more
    tensors (indices, frequency grids, cached transforms) then holds them on two
    devices. This setter moves every tensor in the instance dict, and the
    aberrations of ``meta``. Two attributes that hold the same tensor still hold
    one tensor after the move.
    """

    _array: Tensor

    @property
    def device(self) -> torch.device:
        """Device the underlying array is stored on."""
        return self._array.device

    @device.setter
    def device(self, value: torch.device | str) -> None:
        target = torch.device(value)
        moved: dict[int, Tensor] = {}
        for name, held in list(vars(self).items()):
            if isinstance(held, Tensor):
                if id(held) not in moved:
                    moved[id(held)] = held.to(target)
                setattr(self, name, moved[id(held)])
        _place_aberrations_on_device(getattr(self, "meta", None), target)
