"""Inner product, step application and the Newton line-search alpha."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor


def rdot(xs: Sequence[Tensor], ys: Sequence[Tensor]) -> Tensor:
    """Real inner product Re<x, y> summed over a parameter list.

    For real tensors this is the ordinary dot product; for complex tensors it
    is the real part of the complex inner product, i.e. the inner product of
    the underlying real space R^{2n}.

    Parameters
    ----------
    xs : Sequence of Tensor
        First list of parameter tensors.
    ys : Sequence of Tensor
        Second list of parameter tensors, same shapes as ``xs``.

    Returns
    -------
    Tensor
        Scalar sum of ``Re<x, y>`` over every ``(x, y)`` pair.
    """
    total: Tensor | None = None
    for x, y in zip(xs, ys, strict=False):
        t = (x.conj() * y).real.sum() if x.is_complex() else (x * y).sum()
        total = t if total is None else total + t
    if total is None:
        raise ValueError("rdot requires at least one tensor pair")
    return total


def _scalars_to_host(ts: Sequence[Tensor]) -> list[float]:
    """``[float(t) for t in ts]`` for 0-d tensors, with one device sync.

    Each value is widened to float64 on its device and the stack is copied back
    once, so the returned floats equal the per-tensor ``float()`` values exactly.
    Tensors on different devices fall back to one ``float()`` each.
    """
    ts = [t.detach() for t in ts]
    if len({t.device for t in ts}) != 1:
        return [float(t) for t in ts]
    return torch.stack([t.to(torch.float64) for t in ts]).tolist()


def _is_missing_double_backward(err: Exception) -> bool:
    """True if ``err`` is autograd reporting no second-order (double-backward)
    formula for an op in the backward graph.

    Custom ops registered with only a first-order ``register_autograd`` (e.g. the
    fused Warp ``scatterem::transmit_bwd``) are not themselves differentiable, so
    a Hessian formed by double-backward through them raises at the *second*
    ``autograd.grad``. The exact autograd message is "Trying to backward through
    <op> but no autograd formula was registered." Whether the first-order grad
    graph is even recorded as differentiable (so ``t.requires_grad`` is True and
    the second grad is attempted at all) can depend on ambient autograd state, so
    the Hessian oracles guard the second grad and treat this as zero curvature.
    """
    msg = str(err)
    return (
        "no autograd formula" in msg
        or "does not support double backward" in msg
        or "double backward" in msg
    )


def _rho2(rho: Sequence[float] | None, params: Sequence[Tensor]) -> list[float]:
    if rho is None:
        return [1.0] * len(params)
    if len(rho) != len(params):
        raise ValueError("rho must have one factor per parameter group")
    return [float(r) ** 2 for r in rho]


def _direction(g, r2, precond, params) -> list[Tensor]:
    """Steepest-descent direction: ``-precond(params)_i * g_i`` when a diagonal
    preconditioner callable is given (re-evaluated at the current iterate, so
    dynamic preconditioners like the overlap-energy maps stay current),
    otherwise the classical per-parameter scalar scaling ``-rho^2 g``."""
    if precond is not None:
        d = precond(params)
        return [-gi if di is None else -di * gi for di, gi in zip(d, g, strict=False)]
    return [-r * gi for r, gi in zip(r2, g, strict=False)]


def _step(params: Sequence[Tensor], alpha: float, s: Sequence[Tensor]) -> None:
    with torch.no_grad():
        for p, si in zip(params, s, strict=False):
            p += alpha * si


def _newton_alpha(
    g: Sequence[Tensor],
    s: Sequence[Tensor],
    Hs: Sequence[Tensor] | None,
    fallback: float,
    hss: Tensor | None = None,
) -> float:
    """alpha = -<g,s>/H(s,s) with a bounded fallback on non-convex curvature.

    Curvature must clear a scale-relative floor ``tiny * <s,s>`` (tiny=1e-30)
    so that vanishingly small positive curvature falls back rather than
    producing an unbounded Newton step.  A caller that already holds
    ``H(s,s)`` passes it as ``hss``; ``Hs`` is then not read.
    """
    tiny = 1e-30
    if hss is None:
        assert Hs is not None, "_newton_alpha needs Hs or hss"
        hss = rdot(Hs, s)
    hss_v = float(hss)
    gs = float(rdot(g, s))
    if hss_v > tiny * float(rdot(s, s)) and math.isfinite(hss_v) and math.isfinite(gs):
        return -gs / hss_v
    # Negative/zero curvature: bounded step of length `fallback` along s.
    s_norm = math.sqrt(max(float(rdot(s, s)), 1e-300))
    return fallback / s_norm


def _structural_zeros_like(t: Tensor) -> Tensor:
    """``torch.zeros_like(t)``, tagged as *known* to be identically zero.

    A group-restricted basis vector ``e_i`` is the full parameter-list layout
    with every block but one filled by ``torch.zeros_like``.  The producer knows
    those blocks are zero; the Hessian operator it is handed to does not, and
    cannot find out without reading the tensor.  So an analytic Hessian
    operator runs its full-size arithmetic against
    an identically-zero operand -- for the ptychography oracle that is two
    ``(K, M1, M2)`` multiply-adds, plus (for the object block) a whole patch
    gather of zeros.

    The tag lets the operator skip those terms instead.  It is a plain python
    attribute, so it does NOT survive a view or an arithmetic op -- deliberately:
    only a tensor this function produced is ever trusted, and any tensor derived
    from one is treated as unknown.  Reading it must therefore be an
    optimisation and never a correctness requirement.

    Dropping ``x * 0`` and ``+ 0`` is exact in IEEE-754 for finite ``x``, so an
    operator that honours the tag is bit-identical to one that ignores it.
    """
    z = torch.zeros_like(t)
    z._bh_structural_zero = True  # pyright: ignore[reportAttributeAccessIssue]
    return z


def _is_structural_zero(t: Tensor | None) -> bool:
    """Is ``t`` a block :func:`_structural_zeros_like` produced (or absent)?"""
    return t is None or getattr(t, "_bh_structural_zero", False) is True
