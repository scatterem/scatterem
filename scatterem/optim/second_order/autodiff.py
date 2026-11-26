"""Forward-mode derivative from two nested reverse-mode passes.

:func:`double_vjp` linearises a function at a point and returns the products
``J @ t`` (``push``) and ``J^T @ c`` (``pullback``) without building the
Jacobian ``J``. It does not call ``torch.func.jvp``. The forward-mode path of
that function loads decompositions that emit an upstream ``torch.jit.script``
deprecation warning on this torch version.

The pullback ``g(c) = J^T c`` is linear in ``c``. Pulling a tangent back
through ``g`` returns ``J`` applied to that tangent.

Requirements:

* The model must be twice differentiable. ``push`` differentiates through the
  pullback. A custom ``autograd.Function`` or a Warp kernel without a double
  backward raises ``RuntimeError``.
* For complex tensors, ``push`` is the real-linear derivative and ``pullback``
  is its adjoint under the inner product ``Re<x, y>`` of this package. That
  matches the gradient convention of PyTorch for real losses.

The tier-5 CRLB functions of ``diagnostics.advanced`` were the first callers.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


def double_vjp(
    model: Callable[[Tensor], Tensor], primal: Tensor
) -> tuple[Tensor, Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """Return ``model(primal)``, a ``push`` and the model's ``pullback``.

    ``push(tangent)`` computes ``J @ tangent`` (forward-mode, via two nested
    VJPs); ``pullback(cotangent)`` computes ``J^T @ cotangent`` (plain
    reverse-mode, from the same linearisation point). Both are cheap to
    reuse: neither re-traces ``model``.

    Parameters
    ----------
    model : callable
        A function of one tensor, differentiated at ``primal``.
    primal : torch.Tensor
        The point to linearise ``model`` at.

    Returns
    -------
    mu : torch.Tensor
        ``model(primal)``.
    push : callable
        ``push(tangent) -> J @ tangent``, ``tangent`` shaped like ``primal``.
    pullback : callable
        ``pullback(cotangent) -> J^T @ cotangent``, ``cotangent`` shaped like
        ``mu``.
    """
    # torch.func.vjp is overloaded on has_aux and pyright picks the 3-tuple
    # (has_aux=True) branch for this call site.
    mu, vjp_fn = torch.func.vjp(model, primal)  # pyright: ignore[reportAssignmentType]

    def pullback(cotangent: Tensor) -> Tensor:
        return vjp_fn(cotangent)[0]

    _, pullback_of_pullback = torch.func.vjp(  # pyright: ignore[reportAssignmentType]
        pullback, torch.zeros_like(mu)
    )

    def push(tangent: Tensor) -> Tensor:
        return pullback_of_pullback(tangent)[0]

    return mu, push, pullback
