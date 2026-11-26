"""The bilinear-Hessian nonlinear CG solver (BH-CG).

Split from :mod:`.solvers` (module size); :mod:`.solvers` re-exports
:func:`bh_cg`.  Three oracle capabilities save passes over the data (see
:mod:`.linear_oracle`): ``constant_hessian``, ``fused_grad_loss`` and
``forward_hess_quad``.  An oracle that
does not declare a flag gets the plain path.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence

from torch import Tensor

from .groups import _per_group_alphas, _step_groups
from .log import BHLog, _callback_stop
from .primitives import _direction, _newton_alpha, _rho2, _step, rdot


def _flag(oracle, name: str) -> bool:
    """Whether ``oracle`` declares the capability ``name``."""
    return bool(getattr(oracle, name, False))


def _scalar_step(oracle, g, d, s, beta, fallback, fwd_quad):
    """Take the scalar Newton step along ``s``; return ``(s, beta, alpha, Hs)``.

    On a negative-curvature conjugate direction it restarts on ``d``.  With
    ``fwd_quad`` only ``<H s, s>`` is formed, from forwards (``Hs`` is None).
    """
    if fwd_quad:
        hss = oracle.hess_quad_forward(s)
        if float(hss) <= 0.0 and beta != 0.0:
            s, beta = d, 0.0
            hss = oracle.hess_quad_forward(s)
        alpha = _newton_alpha(g, s, None, fallback, hss=hss)
        _step(oracle.params, alpha, s)
        return s, beta, alpha, None
    Hs = oracle.hess_op(s)
    if float(rdot(Hs, s)) <= 0.0 and beta != 0.0:
        # conjugate direction hit negative curvature: restart on -rho^2 g
        s, beta = d, 0.0
        Hs = oracle.hess_op(s)
    alpha = _newton_alpha(g, s, Hs, fallback)
    _step(oracle.params, alpha, s)
    return s, beta, alpha, Hs


def bh_cg(
    oracle,
    n_iter: int,
    rho: Sequence[float] | None = None,
    fallback: float = 1e-2,
    per_group_alpha: bool = False,
    callback: Callable[[int, BHLog], bool | None] | None = None,
    precond: Callable[[Sequence[Tensor]], Sequence[Tensor]] | None = None,
) -> BHLog:
    """Nonlinear CG with Daniel's Hessian beta and Newton step size (BH-CG).

    Paper eqs. (16)-(18): s_k = -rho^2 g_k + beta s_{k-1},
    beta = H(rho^2 g_k, s_{k-1}) / H(s_{k-1}, s_{k-1}), evaluated at x_k.
    Restarts (beta = 0) on non-positive or non-finite curvature.

    ``per_group_alpha=True`` replaces the scalar Newton step size by the
    per-parameter-group G x G solve (see :func:`_per_group_alphas`); the
    conjugacy history then tracks the actually applied step
    ``sum_i alpha_i e_i`` (Daniel's beta is invariant to its scale).

    Oracle capabilities (scalar step size only, except the second):

    - ``constant_hessian = True`` (e.g. :class:`LinearLeastSquaresOracle`):
      ``H s_prev`` equals the ``H s`` of the previous iteration, so that
      product is reused, not computed again.
    - ``fused_grad_loss = True``: the logged loss and the next iteration's
      gradient come from one ``grad_and_loss()`` call, because both are
      evaluated at the same iterate.  This needs ``callback=None`` (a callback
      may move the iterate).  Bit-identical.
    - ``forward_hess_quad = True``: on the last iteration ``H s`` is used only
      through ``<H s, s>``, so ``hess_quad_forward(s)`` gives it from forwards
      only and the last adjoint pass is skipped.  It rounds differently from
      ``rdot(hess_op(s), s)``, so the last step is close to, not equal to, the
      plain one.

    Parameters
    ----------
    oracle : object
        Full-batch oracle exposing ``params``, ``grad()``, ``hess_op()`` and
        ``loss()``.
    n_iter : int
        Number of CG iterations.
    rho : Sequence[float] or None, optional
        Per-parameter preconditioner scale forwarded to `_rho2`. Default None.
    fallback : float, optional
        Step size when the Newton-rule curvature is non-positive. Default 1e-2.
    per_group_alpha : bool, optional
        Solve a per-parameter-group step size instead of one scalar Newton
        step. Default False.
    callback : callable, optional
        Called as ``callback(k, log)`` after every iteration; a truthy return
        stops the run (``log.early_stop``). Default None.
    precond : callable, optional
        Diagonal-preconditioner callable, as in :func:`bh_gd` (flexible
        preconditioning: Daniel's beta and the restarts keep the recurrence
        stable while the maps evolve).

    Returns
    -------
    BHLog
        Log of losses, step sizes, CG betas and timing across iterations.
    """
    log = BHLog()
    r2 = _rho2(rho, oracle.params)
    s_prev: list[Tensor] | None = None
    reuse_hs = _flag(oracle, "constant_hessian")
    fuse_loss = callback is None and _flag(oracle, "fused_grad_loss")
    fwd_quad = _flag(oracle, "forward_hess_quad")
    Hs_prev: list[Tensor] | None = None  # H s_prev, kept for a constant Hessian
    g_next: list[Tensor] | None = None
    work_time = 0.0
    for k in range(n_iter):
        t_iter_start = time.perf_counter()
        g = g_next if g_next is not None else oracle.grad()
        g_next = None
        d = _direction(g, r2, precond, oracle.params)
        beta = 0.0
        if s_prev is not None:
            Hsp = Hs_prev if Hs_prev is not None else oracle.hess_op(s_prev)
            denom = float(rdot(Hsp, s_prev))
            num = -float(rdot(Hsp, d))  # = H(rho^2 g, s_prev)
            if denom > 0.0 and math.isfinite(denom) and math.isfinite(num):
                beta = num / denom
        s = d
        if beta != 0.0:
            assert s_prev is not None  # beta stays 0.0 on the first iteration
            s = [di + beta * spi for di, spi in zip(d, s_prev, strict=False)]
        if per_group_alpha:
            alphas, hss = _per_group_alphas(oracle, g, s, fallback)
            if hss <= 0.0 and beta != 0.0:
                # conjugate direction hit negative curvature: restart on -rho^2 g
                s, beta = d, 0.0
                alphas, hss = _per_group_alphas(oracle, g, s, fallback)
            _step_groups(oracle.params, alphas, s)
            s_prev = [a * si for a, si in zip(alphas, s, strict=False)]
            alpha = alphas[0]
            log.group_alphas.append(alphas)
        else:
            last = k + 1 == n_iter
            s, beta, alpha, Hs = _scalar_step(
                oracle, g, d, s, beta, fallback, fwd_quad and last
            )
            s_prev = s
            Hs_prev = Hs if reuse_hs else None
        if fuse_loss and k + 1 < n_iter:
            # the next gradient is work, so the fused pass counts in the wall
            g_next, loss = oracle.grad_and_loss()
            work_time += time.perf_counter() - t_iter_start
        else:
            work_time += time.perf_counter() - t_iter_start
            loss = oracle.loss()
        log.wall.append(work_time)
        log.losses.append(loss)
        log.alphas.append(alpha)
        log.betas.append(beta)
        if _callback_stop(callback, k, log, n_iter):
            break
    return log
