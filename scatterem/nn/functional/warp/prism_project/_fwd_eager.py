"""Eager dispatch of the no-grad PRISM forward custom ops.

Every dense-route chunk issues two forward custom ops, ``prism_ramp_mul`` and
``prism_project_fwd``.  The ``torch.ops`` boundary matters only while something
traces or wraps the call (compile, a dispatch mode, functorch, a tensor
subclass) or while autograd must record the op.  In all other cases the caller
may call the op body (``CustomOpDef._init_fn``) directly and skip the custom-op
dispatch (lab: 34 + 39 us per chunk).  The body is the function the op runs, on
the same tensors, so the result is bit-identical.  The ``prism_fwd_eager``
backend row selects the route (:func:`prism_fwd_eager_supported` is its
predicate).
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils._python_dispatch import _get_current_dispatch_mode

from scatterem.nn._backends_launch import PRISM_FWD_EAGER
from scatterem.utils.env import resolve_backend

__all__ = ["fwd_eager", "prism_fwd_eager_supported"]


def prism_fwd_eager_supported(*tensors: Tensor) -> bool:
    """True if a forward op may run its body without the custom-op dispatch.

    All must hold: no ``torch.compile`` tracing, no operand that autograd must
    record, no functorch transform, no ``TorchDispatchMode`` and only plain
    ``torch.Tensor`` operands (no subclass).
    """
    return (
        not torch.compiler.is_compiling()
        and not (torch.is_grad_enabled() and any(t.requires_grad for t in tensors))
        and not torch._C._are_functorch_transforms_active()
        and _get_current_dispatch_mode() is None
        and all(type(t) is torch.Tensor for t in tensors)
    )


def fwd_eager(*tensors: Tensor) -> bool:
    """Whether to call a forward op's body directly on ``tensors``.

    The compile test comes first, so a traced call never enters the resolver.
    """
    return not torch.compiler.is_compiling() and (
        resolve_backend(PRISM_FWD_EAGER, None, *tensors) == "eager"
    )
