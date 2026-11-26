"""The linear least-squares Hessian oracle (linear tomography's BH-CG path).

Split from :mod:`.oracles` (module size).  Besides the oracle protocol it
declares three capabilities that :func:`~.cg.bh_cg` reads:

* ``constant_hessian`` -- ``hess_op`` does not depend on the iterate, so the
  ``H s`` of one iteration is the ``H s_prev`` of the next;
* ``fused_grad_loss`` -- :meth:`LinearLeastSquaresOracle.grad_and_loss` gives
  ``grad()`` and ``loss()`` from one forward per batch;
* ``forward_hess_quad`` -- :meth:`LinearLeastSquaresOracle.hess_quad_forward`
  gives ``<H s, s>`` from forwards only.

A subclass that sets a flag to ``False`` gets the plain path back.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from .oracles import HessianForms


class LinearLeastSquaresOracle(HessianForms):
    """Oracle for a sum-of-mse loss with linear per-batch forwards.

    ``f(x) = sum_i mse(A_i(x), b_i) / n_batches`` with LINEAR
    per-batch forwards ``A_i`` (e.g. tomographic projection of a volume).

    ``batches`` is a sequence of ``(pred_fn, data)`` pairs where ``pred_fn()``
    evaluates that batch's forward at the *current* value of ``param`` and is
    linear in it.  The gradient is one forward + one backward (adjoint) per
    batch, accumulated — memory stays O(batch), no cross-batch graph and no
    double-backward.  ``hess_op`` exploits linearity: the Hessian is the
    constant mse-normalized ``2 A^T A``, so ``H v`` equals the gradient
    evaluated at ``x = v`` with zero data; it is computed by temporarily
    writing ``v`` into the parameter (restored afterwards, also on error).

    With this oracle, :func:`bh_cg` is exact-step CGLS on the normal
    equations — finite-step exact on the quadratic, zero hyperparameters.

    ``constant_hessian = True`` tells :func:`bh_cg` that it may reuse ``H s``
    of the previous iteration as ``H s_prev``.  ``fused_grad_loss = True``
    tells it that :meth:`grad_and_loss` returns the values of ``grad()`` and
    ``loss()`` at the same iterate.  ``forward_hess_quad = True`` tells it
    that :meth:`hess_quad_forward` gives ``<H s, s>`` without the adjoint.

    Parameters
    ----------
    param : Tensor
        Leaf tensor the loss depends on. ``requires_grad`` is enabled on
        it here.
    batches : sequence of tuple
        Sequence of ``(pred_fn, data)`` pairs where ``pred_fn()`` evaluates
        that batch's forward, linear in ``param``, at its current value,
        and ``data`` is the corresponding target tensor.
    """

    constant_hessian = True
    fused_grad_loss = True
    forward_hess_quad = True

    def __init__(self, param: Tensor, batches: Sequence[tuple]):
        self.params: list[Tensor] = [param.requires_grad_(True)]
        self.batches = list(batches)
        if not self.batches:
            raise ValueError("batches must be non-empty")

    def loss(self) -> float:
        """Evaluate the mean per-batch MSE loss under ``no_grad``.

        Returns
        -------
        float
            Mean squared-error loss, averaged over all batches.
        """
        n = len(self.batches)
        with torch.no_grad():
            total = sum(torch.nn.functional.mse_loss(fn(), b) for fn, b in self.batches)
        return float(total) / n

    def _grad_at_current(self, zero_data: bool) -> list[Tensor]:
        p = self.params[0]
        n = len(self.batches)
        g = torch.zeros_like(p)
        for fn, b in self.batches:
            pred = fn()
            target = torch.zeros_like(pred) if zero_data else b
            loss = torch.nn.functional.mse_loss(pred, target) / n
            (gb,) = torch.autograd.grad(loss, [p])
            g += gb
        return [g]

    def grad(self) -> list[Tensor]:
        """Accumulate the gradient of the mean MSE loss over all batches.

        Returns
        -------
        list of Tensor
            Single-element list holding the gradient with respect to
            ``self.params[0]``.
        """
        return self._grad_at_current(zero_data=False)

    def grad_and_loss(self) -> tuple[list[Tensor], float]:
        """Return ``(grad(), loss())`` from one forward per batch.

        The per-batch mse feeds both the autograd gradient and the loss sum,
        in the same order and arithmetic as :meth:`grad` and :meth:`loss`, so
        the values are the same words.

        Returns
        -------
        tuple of (list of Tensor, float)
            The gradient (single-element list) and the mean MSE loss.
        """
        p = self.params[0]
        n = len(self.batches)
        g = torch.zeros_like(p)
        total = 0
        for fn, b in self.batches:
            mse = torch.nn.functional.mse_loss(fn(), b)
            (gb,) = torch.autograd.grad(mse / n, [p])
            g += gb
            total = total + mse.detach()
        return [g], float(total) / n

    def hess_op(self, v: Sequence[Tensor]) -> list[Tensor]:
        """Apply the (constant) Hessian ``2 A^T A`` to direction ``v``.

        Exploits linearity of the per-batch forward: the Hessian-vector
        product equals the gradient evaluated at ``x = v`` with zero data,
        computed by temporarily writing ``v`` into the parameter (restored
        afterwards, also on error).

        Parameters
        ----------
        v : sequence of Tensor
            Single-element sequence holding the direction tensor, shaped
            like ``self.params[0]``.

        Returns
        -------
        list of Tensor
            Single-element list holding the Hessian-vector product
            ``H(v)``.
        """
        p = self.params[0]
        saved = p.detach().clone()
        with torch.no_grad():
            p.copy_(v[0])
        try:
            Hv = self._grad_at_current(zero_data=True)
        finally:
            with torch.no_grad():
                p.copy_(saved)
        return Hv

    def hess_quad_forward(self, v: Sequence[Tensor]) -> Tensor:
        """Compute ``<H v, v>`` from forwards only: ``(2 / n) sum_i mse(A_i v, 0)``.

        Equal to ``rdot(hess_op(v), v)`` in exact arithmetic (``H`` is
        ``2 A^T A`` with the mse normalisation), without the adjoint pass.  It
        rounds differently, so it is close to, not equal to, that value.

        Parameters
        ----------
        v : sequence of Tensor
            Single-element sequence holding the direction tensor.

        Returns
        -------
        Tensor
            Scalar ``H(v, v)``.
        """
        p = self.params[0]
        n = len(self.batches)
        saved = p.detach().clone()
        with torch.no_grad():
            p.copy_(v[0])
        try:
            with torch.no_grad():
                total = None
                for fn, _ in self.batches:
                    pred = fn()
                    q = torch.nn.functional.mse_loss(pred, torch.zeros_like(pred))
                    total = q if total is None else total + q
        finally:
            with torch.no_grad():
                p.copy_(saved)
        assert total is not None  # batches is non-empty
        return total * (2.0 / n)
