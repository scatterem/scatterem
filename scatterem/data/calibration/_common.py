"""Shared conversion helper, and the device rule every entry point follows.

``device=None`` means the device of the entry point's tensor input, else CPU
(:func:`scatterem.utils._device.resolve_device`); an explicit ``device`` wins.
Each entry point converts its first input with :func:`_to_tensor` and then
places everything else on that tensor's device.
"""

from __future__ import annotations

import numpy as np
import torch

from scatterem.utils._device import resolve_device


def _to_tensor(
    x,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Convert *x* (ndarray, tensor, scalar) to a torch tensor.

    ``device=None`` keeps a tensor on its own device and puts anything else on
    CPU.
    """
    device = resolve_device(device, x if isinstance(x, torch.Tensor) else None)
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)
