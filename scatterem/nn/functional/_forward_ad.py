"""Forward-mode autograd helpers for kernels without a forward-mode derivative.

Fused kernels (Triton, Warp) and some torch ops (``grid_sampler_3d``) have no
forward-mode derivative. A fused kernel that is not a differentiable torch op
drops the tangent of a dual operand without an error; a torch op without a
formula raises. Callers check :func:`carries_tangent` and take a path that
keeps the tangent.
"""

import torch.autograd.forward_ad as fwad
from torch import Tensor


def carries_tangent(*tensors: Tensor) -> bool:
    """Whether a tensor is a forward-mode autograd dual at the current level.

    Parameters
    ----------
    *tensors : torch.Tensor
        The operands.

    Returns
    -------
    bool
        ``True`` if any operand has a tangent; always ``False`` outside a
        ``torch.autograd.forward_ad.dual_level``.
    """
    return any(fwad.unpack_dual(t).tangent is not None for t in tensors)
