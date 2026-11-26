"""Hessian oracles: autograd and chunked autograd.

The linear least-squares oracle is in :mod:`.linear_oracle` (module size).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
from torch import Tensor

from .primitives import _is_structural_zero, _scalars_to_host, rdot


class HessianForms:
    """``hess_quad`` and ``hess_bilinear`` of an oracle, from its ``hess_op``.

    Every oracle defines the Hessian operator ``hess_op(v)``; both scalar forms
    follow from it with the real inner product :func:`rdot`.
    """

    def hess_op(self, v: Sequence[Tensor]) -> list[Tensor]:
        """Return the Hessian-vector product ``H(v)``; each oracle defines it."""
        raise NotImplementedError

    def hess_quad(self, s: Sequence[Tensor]) -> Tensor:
        """Compute the Hessian quadratic form H(s, s).

        Parameters
        ----------
        s : sequence of Tensor
            Direction tensors, in the layout of ``hess_op``'s input.

        Returns
        -------
        Tensor
            Scalar value of ``H(s, s) = <H(s), s>_R``.
        """
        return rdot(self.hess_op(s), s)

    def hess_bilinear(self, y: Sequence[Tensor], z: Sequence[Tensor]) -> Tensor:
        """Compute the Hessian bilinear form H(y, z) = <H(y), z>_R.

        Parameters
        ----------
        y : sequence of Tensor
            Left direction tensors, in the layout of ``hess_op``'s input.
        z : sequence of Tensor
            Right direction tensors, in the same layout.

        Returns
        -------
        Tensor
            Scalar value of ``H(y, z)``.
        """
        return rdot(self.hess_op(y), z)


class AutogradOracle(HessianForms):
    """Generic bilinear-Hessian oracle via double-backward autograd.

    Parameters
    ----------
    closure : callable
        Callable returning the scalar loss, built from ``params``. Must be
        re-evaluable (called fresh for every gradient / HVP).
    params : sequence of Tensor
        Leaf tensors the loss depends on. ``requires_grad`` is enabled on
        them here.
    """

    def __init__(self, closure: Callable[[], Tensor], params: Sequence[Tensor]):
        self.params: list[Tensor] = [p.requires_grad_(True) for p in params]
        self._closure = closure

    def loss(self) -> float:
        """Evaluate the closure under ``no_grad`` and return its scalar value.

        Returns
        -------
        float
            Loss value at the current parameters.
        """
        with torch.no_grad():
            return float(self._closure())

    def grad(self) -> list[Tensor]:
        """Evaluate the closure and return the gradient of the loss.

        Returns
        -------
        list of Tensor
            Gradient of the loss with respect to ``self.params``, one entry
            per parameter, in the same order.
        """
        value = self._closure()
        return list(torch.autograd.grad(value, self.params))

    def hess_op(self, v: Sequence[Tensor]) -> list[Tensor]:
        """Apply the Hessian operator H(v) = d/dt grad f(x + t v) |_{t=0}.

        Parameters
        ----------
        v : sequence of Tensor
            Direction tensors, one per entry of ``self.params``.

        Returns
        -------
        list of Tensor
            Hessian-vector product ``H(v)``, one entry per parameter.
        """
        value = self._closure()
        gs = torch.autograd.grad(value, self.params, create_graph=True)
        s = rdot(gs, [vi.detach() for vi in v])
        return list(torch.autograd.grad(s, self.params))


class ChunkedAutogradOracle(HessianForms):
    """Autograd oracle over per-batch loss closures; ``f = sum_b closure_b()``.

    ``loss``/``grad``/``hess_op`` accumulate exactly across batches — each
    batch's graph is built and freed within the loop, so peak memory is
    O(batch) regardless of how many batches the objective sums over (the
    full-batch requirement of BH is statistical, not a memory constraint).

    ``hess_quad_matrix(basis)`` is the fused per-group curvature pass: the
    G x G matrix ``A_ij = H(e_i, e_j)`` computed in ONE sweep over the
    batches (one ``create_graph`` backward per batch, then one directional
    double-backward per basis direction, reduced to scalars immediately).
    Together with the gradient pass this puts per-group BH-GD at 2 full
    data passes per iteration — the structural floor, since the search
    direction depends on the full-batch gradient.

    Parameters
    ----------
    closures : sequence of callable
        Per-batch callables, each returning that batch's scalar loss
        contribution, built from ``params``. Must be re-evaluable (called
        fresh for every gradient / HVP sweep).
    params : sequence of Tensor
        Leaf tensors the loss depends on. ``requires_grad`` is enabled on
        them here.
    """

    def __init__(
        self, closures: Sequence[Callable[[], Tensor]], params: Sequence[Tensor]
    ):
        self.params: list[Tensor] = [p.requires_grad_(True) for p in params]
        self.closures = list(closures)
        if not self.closures:
            raise ValueError("closures must be non-empty")

    def loss(self) -> float:
        """Evaluate all closures under ``no_grad`` and sum their values.

        Returns
        -------
        float
            Total loss, summed over all batches.
        """
        with torch.no_grad():
            return float(sum(c() for c in self.closures))

    @staticmethod
    def _add_into(acc: list[Tensor], contrib: Sequence[Tensor | None]) -> None:
        for a, c in zip(acc, contrib, strict=False):
            if c is not None:
                a += c

    def grad(self) -> list[Tensor]:
        """Accumulate the gradient of the summed loss over all batches.

        Each batch's graph is built and freed within the loop, so peak
        memory is O(batch) regardless of the number of batches.

        Returns
        -------
        list of Tensor
            Gradient of the summed loss with respect to ``self.params``,
            one entry per parameter.
        """
        g = [torch.zeros_like(p) for p in self.params]
        for c in self.closures:
            gb = torch.autograd.grad(c(), self.params, allow_unused=True)
            self._add_into(g, gb)
        return g

    def grad_and_loss(self) -> tuple:
        """Gradient and loss from ONE sweep (the closure value is free).

        Returns
        -------
        tuple of (list of Tensor, float)
            Gradient with respect to ``self.params`` (one entry per
            parameter) and the summed loss value.
        """
        g = [torch.zeros_like(p) for p in self.params]
        total = 0.0
        for c in self.closures:
            v = c()
            gb = torch.autograd.grad(v, self.params, allow_unused=True)
            self._add_into(g, gb)
            total += float(v.detach())
        return g, total

    @staticmethod
    def _directional(
        gs: Sequence[Tensor | None],
        v: Sequence[Tensor],
        zero: Sequence[bool] | None = None,
    ):
        """Re<grad, v> over the non-None entries (None = param unused).

        ``zero[k]`` True leaves entry k out: ``v[k]`` is known to be zero.
        """
        t = None
        for k, (gi, vi) in enumerate(zip(gs, v, strict=False)):
            if gi is None or (zero is not None and zero[k]):
                continue
            term = (gi.conj() * vi).real.sum() if gi.is_complex() else (gi * vi).sum()
            t = term if t is None else t + term
        return t

    def hess_op(self, v: Sequence[Tensor]) -> list[Tensor]:
        """Apply the Hessian of the summed loss to direction ``v``.

        Accumulates the per-batch Hessian-vector product across all
        batches; each batch's second-order graph is built and freed
        within the loop, so peak memory is O(batch).

        Parameters
        ----------
        v : sequence of Tensor
            Direction tensors, one per entry of ``self.params``.

        Returns
        -------
        list of Tensor
            Hessian-vector product ``H(v)``, one entry per parameter.
        """
        out = [torch.zeros_like(p) for p in self.params]
        vd = [vi.detach() for vi in v]
        for c in self.closures:
            gs = torch.autograd.grad(
                c(), self.params, create_graph=True, allow_unused=True
            )
            t = self._directional(gs, vd)
            if t is None or not t.requires_grad:
                continue  # batch is (locally) linear: zero curvature
            hb = torch.autograd.grad(t, self.params, allow_unused=True)
            self._add_into(out, hb)
        return out

    def hess_quad_matrix(self, basis: Sequence[Sequence[Tensor]]) -> Tensor:
        """Compute A_ij = H(e_i, e_j) for group directions, in one sweep.

        Basis blocks from :func:`._structural_zeros_like` are left out of both
        directional products.  In ``t_i = <g, e_i>`` a term
        ``<g_j, 0>`` adds nothing to the value, but it puts the gradient of
        group j in the graph, so ``grad(t_i)`` runs the double backward through
        that branch too.  The tag is read before ``detach()``, which drops it.
        Exact for finite values (``x + 0 == x``).

        Each ``A[i, j]`` term stays on the device as a float64 0-d tensor that
        takes ``0 + t_1 + t_2 + ...`` in batch order, and the matrix goes to
        the host once at the end.  A ``float(tij)`` per term
        made the host wait for the GPU once per batch, i and j.  The float32 to
        float64 cast is exact and a float64 add rounds the same on both
        devices, so ``A`` is the same words.

        Parameters
        ----------
        basis : sequence of sequence of Tensor
            Group directions ``e_i``, each shaped like ``self.params``.

        Returns
        -------
        Tensor
            ``(G, G)`` float64 curvature matrix ``A`` with
            ``A[i, j] = H(e_i, e_j)``, where ``G = len(basis)``.
        """
        G = len(basis)
        zero = [
            [_SKIP_ZERO_BLOCKS and _is_structural_zero(e) for e in ei] for ei in basis
        ]
        det = [[e.detach() for e in ei] for ei in basis]
        acc: list[list[Tensor | float]] = [[0.0] * G for _ in range(G)]
        for c in self.closures:
            gs = torch.autograd.grad(
                c(), self.params, create_graph=True, allow_unused=True
            )
            for i in range(G):
                t = self._directional(gs, det[i], zero[i])
                if t is None or not t.requires_grad:
                    continue
                Hi = torch.autograd.grad(
                    t, self.params, retain_graph=(i < G - 1), allow_unused=True
                )
                for j in range(G):
                    tij = self._directional(Hi, det[j], zero[j])
                    if tij is not None:
                        acc[i][j] = _accumulate(acc[i][j], tij)
            # A pruned branch is not freed by the last backward: drop this
            # batch's graph now, not after the next batch's graph is built.
            gs = t = Hi = None
        return _matrix_to_host(acc)

    def hess_quad_matrix_fd(
        self,
        basis: Sequence[Sequence[Tensor]],
        g0: Sequence[Tensor],
        eps_rel: float = 1e-4,
        eps_hints: Sequence[float | None] | None = None,
    ) -> Tensor:
        """Secant curvature: A_ij ~= <g(x + eps_j e_j) - g0, e_i> / eps_j.

        Plain backwards only — no ``create_graph`` second-order graph, so
        the memory transient equals a gradient pass and each column costs
        one gradient sweep (G sweeps total).  Exact on quadratics; O(eps)
        error otherwise, which the PD check / acceptance test absorb.
        The perturbation norm is ``eps_rel * (1 + |x|)``, unless
        ``eps_hints[j]`` gives a per-group scale — the natural choice is
        |previous solved alpha|, i.e. the secant is taken over the interval
        the step will actually cross (also keeps the difference well above
        float32 gradient noise when gradients get small).  Parameters are
        restored bit-exactly from saved copies.

        Parameters
        ----------
        basis : sequence of sequence of Tensor
            Group directions ``e_j``, each shaped like ``self.params``.
        g0 : sequence of Tensor
            Gradient at the current iterate (the caller already has it),
            one entry per entry of ``self.params``.
        eps_rel : float, optional
            Relative perturbation size used when no hint is given for a
            group, by default ``1e-4``.
        eps_hints : sequence of float or None, optional
            Per-group perturbation scale overriding ``eps_rel`` where not
            ``None``, by default ``None``.

        Returns
        -------
        Tensor
            ``(G, G)`` float64 curvature matrix ``A``, where
            ``G = len(basis)``.
        """
        G = len(basis)
        det = [[e.detach() for e in ei] for ei in basis]
        x_norm = math.sqrt(
            sum(float((p.detach().abs() ** 2).sum()) for p in self.params)
        )
        A = torch.zeros(G, G, dtype=torch.float64)
        for j in range(G):
            ej = det[j]
            ej_norm = math.sqrt(sum(float((e.abs() ** 2).sum()) for e in ej))
            if ej_norm == 0.0:
                continue
            hint = eps_hints[j] if eps_hints is not None else None
            if hint is not None and math.isfinite(hint) and abs(hint) > 0.0:
                eps = abs(hint)
            else:
                eps = eps_rel * (1.0 + x_norm) / ej_norm
            saved = [p.detach().clone() for p in self.params]
            with torch.no_grad():
                for p, e in zip(self.params, ej, strict=False):
                    p += eps * e
            try:
                g_eps = self.grad()
            finally:
                with torch.no_grad():
                    for p, sv in zip(self.params, saved, strict=False):
                        p.copy_(sv)
            dg = [ge - g0i for ge, g0i in zip(g_eps, g0, strict=False)]
            for i in range(G):
                t = self._directional(dg, det[i])
                if t is not None:
                    # <H e_j, e_i> = A_ji = A_ij (H symmetric)
                    A[j, i] = float(t) / eps
        return A


#: Whether :meth:`ChunkedAutogradOracle.hess_quad_matrix` leaves the tagged zero
#: basis blocks out.  A test hook, not a setting: tests set it
#: to ``False`` to run the reference path.  No environment variable reads it.
_SKIP_ZERO_BLOCKS = True

#: Whether :meth:`ChunkedAutogradOracle.hess_quad_matrix` keeps its terms on the
#: device.  A test hook like :data:`_SKIP_ZERO_BLOCKS`.
_DEVICE_ACCUMULATE = True


def _accumulate(a: Tensor | float, tij: Tensor) -> Tensor | float:
    """``a + tij`` as a float64 device scalar, or as a host float (reference)."""
    if not _DEVICE_ACCUMULATE:
        return a + float(tij)
    if isinstance(a, float):
        a = torch.zeros((), dtype=torch.float64, device=tij.device)
    return a + tij.detach().to(torch.float64)


def _matrix_to_host(acc: list[list[Tensor | float]]) -> Tensor:
    """The ``(G, G)`` float64 host matrix of ``acc``, read in one device sync."""
    G = len(acc)
    flat = [a for row in acc for a in row]
    dev = [a for a in flat if isinstance(a, Tensor)]
    vals = iter(_scalars_to_host(dev)) if dev else iter(())
    out = [next(vals) if isinstance(a, Tensor) else a for a in flat]
    return torch.tensor(out, dtype=torch.float64).reshape(G, G)
