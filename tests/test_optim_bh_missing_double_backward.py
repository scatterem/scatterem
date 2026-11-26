"""Regression: the per-group BH step tolerates a forward whose custom-op backward
has no second-order (double-backward) autograd formula.

The fused Warp ``scatterem::transmit_bwd`` used by the ptychography forward is
registered with only a *first-order* autograd formula, so the exact bilinear
Hessian (formed by double-backward) raises "no autograd formula was registered"
at the second ``autograd.grad``. Whether the first-order grad graph is even
recorded as differentiable -- making ``t.requires_grad`` True so the second grad
is attempted at all -- depends on ambient autograd state, which made several
BH-solver tests crash only inside the full suite (and pass in isolation).

``_per_group_alphas`` now catches that specific failure and falls back to the
finite-difference secant Hessian (``hess_quad_matrix_fd`` -- first-order grads
only, still *real* curvature) rather than crashing or discarding curvature.

This reproduces the exact double-backward gap deterministically with a stand-in
op pair, independent of test ordering.
"""

from __future__ import annotations

import math

import pytest
import torch

from scatterem.optim.second_order import (
    ChunkedAutogradOracle,
    _is_missing_double_backward,
    _per_group_alphas,
)

# --- stand-ins for transmit_fwd / transmit_bwd -----------------------------
# An op with NO autograd formula (mirrors ``transmit_bwd``): a second backward
# through it has nothing to differentiate.


@torch.library.custom_op("bh_regress::no_grad_op", mutates_args=())
def _no_grad_op(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


@_no_grad_op.register_fake
def _(x):
    return torch.empty_like(x)


# A first-order-differentiable op (mirrors ``transmit_fwd``) whose backward routes
# through the formula-less op, so any double-backward through it has no formula.


@torch.library.custom_op("bh_regress::sq", mutates_args=())
def _sq(x: torch.Tensor) -> torch.Tensor:
    return x * x


@_sq.register_fake
def _(x):
    return torch.empty_like(x)


def _sq_setup(ctx, inputs, output):
    ctx.save_for_backward(inputs[0])


def _sq_bwd(ctx, grad):
    (x,) = ctx.saved_tensors
    return torch.ops.bh_regress.no_grad_op(2.0 * x * grad)


torch.library.register_autograd("bh_regress::sq", _sq_bwd, setup_context=_sq_setup)


def test_per_group_alphas_falls_back_to_fd_on_missing_double_backward():
    # Convex quadratic loss = sum(sq(x)) = sum(x^2): gradient 2x, Hessian 2 I.
    torch.manual_seed(0)
    x = torch.randn(4, requires_grad=True)

    # Sanity: this forward really does trigger the guarded-against condition --
    # the first-order grad graph is differentiable (so the second grad is
    # attempted), but the op in it has no second-order formula.
    g_graph = torch.autograd.grad(
        torch.ops.bh_regress.sq(x).sum(), x, create_graph=True
    )[0]
    assert g_graph.requires_grad

    oracle = ChunkedAutogradOracle([lambda: torch.ops.bh_regress.sq(x).sum()], [x])
    basis = [[torch.ones_like(x)]]

    # The exact (double-backward) Hessian is genuinely unavailable here...
    with pytest.raises(RuntimeError) as ei:
        oracle.hess_quad_matrix(basis)
    assert _is_missing_double_backward(ei.value)

    # ...but the FD Hessian recovers the true curvature (H = 2 I -> H(1,1) = 2*4).
    A_fd = oracle.hess_quad_matrix_fd(basis, oracle.grad())
    torch.testing.assert_close(float(A_fd[0, 0]), 8.0, atol=1e-3, rtol=1e-3)

    # ...and _per_group_alphas transparently falls back to it (no crash, finite
    # alphas, positive measured curvature) instead of double-backward-crashing.
    g = oracle.grad()
    alphas, hss = _per_group_alphas(oracle, g, [torch.ones_like(x)], fallback=1.0)
    assert all(math.isfinite(a) for a in alphas)
    assert math.isfinite(hss) and hss > 0.0


def test_hess_quad_matrix_still_measures_real_curvature():
    # Guard against the fix masking a genuinely broken exact path: a plain
    # twice-differentiable closure must still use the exact double-backward
    # Hessian. f = sum(x^3) -> H = diag(6 x), so H(v, v) = sum(6 x) for v = 1.
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    oracle = ChunkedAutogradOracle([lambda: (x**3).sum()], [x])

    hv = oracle.hess_op([torch.ones_like(x)])
    torch.testing.assert_close(hv[0], 6.0 * x.detach())

    A = oracle.hess_quad_matrix([[torch.ones_like(x)]])
    torch.testing.assert_close(float(A[0, 0]), float((6.0 * x.detach()).sum()))
