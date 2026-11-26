"""NumPy/torch interop for the probe-state metrics and the OAM sorter.

The metrics compute in torch, so a mode stack that already lives on the GPU in
a solver callback is measured where it is. NumPy input is converted (without a
copy where possible) and results are handed back as NumPy, so analysis scripts
keep working in NumPy end to end.
"""

import numpy as np
import torch

ArrayLike = np.ndarray | torch.Tensor


def as_tensor(x: ArrayLike) -> tuple[torch.Tensor, bool]:
    """Return ``x`` as a tensor and whether it was a NumPy array."""
    if isinstance(x, torch.Tensor):
        return x, False
    return torch.as_tensor(np.asarray(x)), True


def as_complex(x: torch.Tensor) -> torch.Tensor:
    """Promote a real tensor to the matching complex dtype (complex input unchanged)."""
    if x.is_complex():
        return x
    if x.dtype == torch.float64:
        return x.to(torch.complex128)
    return x.to(torch.complex64)


def restore(x: torch.Tensor, to_numpy: bool) -> ArrayLike:
    """Hand ``x`` back in the caller's array family."""
    return x.detach().cpu().numpy() if to_numpy else x
