"""Stop and return rules for :func:`.driver.solve_ptychography` passes.

:class:`PassStopRule` holds the stop logic of the mixed-state BH runs of the
fast-probe research driver, as pure state over the per-pass numbers.
:class:`PassStopCallback` is the ``callback(k, log)`` of the driver that feeds
it: after each accepted pass it applies the module's constraint schedule,
measures the gauge-free state change of the probe, ``1 - F(rho_k, rho_k-1)``,
and the change of the purity, asks the rule, and keeps a copy of the last
accepted state. After the solve, :meth:`PassStopCallback.restore_last` puts
that state back, in place of the candidate that the solver selected (the
lowest loss of the start, the best pass, the last pass and the tail average).

Example::

    rule = PassStopRule(probe_tol=0.005, min_iters=150)
    cb = PassStopCallback(module, rule, return_last=True)
    log = solve_ptychography(module, batches, n_iter=150, callback=cb)
    cb.restore_last()
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from ..regularization.probe_modes import _get_probe_parameter

logger = logging.getLogger(__name__)

__all__ = ["PassStopCallback", "PassStopRule", "mode_purity", "state_change"]


@dataclass
class PassStopRule:
    """When to stop a sequence of solver passes.

    The rule acts only when ``probe_tol`` is set and the pass has a state
    change ``rel`` (from the second pass on). A pass "meets" the rule when
    all of these hold:

    * ``rel < probe_tol`` (the state stopped changing);
    * ``|dloss| < loss_tol`` (the loss stopped changing);
    * the pass was not reverted;
    * ``dpurity < purity_tol``, when ``purity_tol`` is set;
    * the state has moved (one pass with ``rel > probe_tol``), when
      ``needs_move`` is set. Before that, a seed on the plateau before the
      transition has ``rel`` near 0 while the loss still falls.

    The solve stops after ``patience`` consecutive passes that meet the rule,
    but not before ``min_iters`` passes. Independently, with
    ``passes_after_move`` set, it stops after that number of accepted passes
    after the state moved.

    Parameters
    ----------
    probe_tol : float or None, optional
        Bound on the state change per pass. None disables the rule.
        Default None.
    loss_tol : float, optional
        Bound on the relative loss change per pass. Default 2e-3.
    min_iters : int, optional
        Earliest stop, in passes. Default 2.
    patience : int, optional
        Consecutive passes that must meet the rule. Default 1.
    purity_tol : float or None, optional
        Bound on the purity change per pass; None skips this test.
        Default None.
    needs_move : bool, optional
        Never stop before the state has moved. Default False.
    passes_after_move : int or None, optional
        Stop after this number of accepted passes after the move. Default
        None.
    """

    probe_tol: float | None = None
    loss_tol: float = 2e-3
    min_iters: int = 2
    patience: int = 1
    purity_tol: float | None = None
    needs_move: bool = False
    passes_after_move: int | None = None
    stopped_at: int | None = field(default=None, init=False)
    run: int = field(default=0, init=False)
    moved: bool = field(default=False, init=False)
    after_move: int = field(default=0, init=False)

    def update(
        self,
        k: int,
        rel: float | None,
        dloss: float,
        reverted: bool,
        dpurity: float = math.inf,
    ) -> bool:
        """Record pass ``k`` and say whether the solve stops after it.

        Parameters
        ----------
        k : int
            Pass index, from 0.
        rel : float or None
            State change against the previous pass; None for the first pass.
        dloss : float
            Relative loss change ``(L[k-1] - L[k]) / |L[k-1]|``; ``inf`` when
            there is no earlier loss.
        reverted : bool
            Whether the solver undid the pass.
        dpurity : float, optional
            Absolute purity change against the previous pass. Default inf.

        Returns
        -------
        bool
            True when the solve stops after this pass.
        """
        if self.probe_tol is None or rel is None or self.stopped_at is not None:
            return False
        if self.moved and not reverted:
            self.after_move += 1
        self.moved = self.moved or rel > self.probe_tol
        after = self.passes_after_move
        if after is not None and self.after_move >= after:
            self.stopped_at = k
            return True
        ok = (
            rel < self.probe_tol
            and abs(dloss) < self.loss_tol
            and not reverted
            and (self.purity_tol is None or dpurity < self.purity_tol)
            and (self.moved or not self.needs_move)
        )
        self.run = self.run + 1 if ok else 0
        if self.run >= self.patience and k + 1 >= self.min_iters:
            self.stopped_at = k
            return True
        return False


def mode_purity(probe: Tensor, drop_last: bool = True) -> float:
    """Purity ``sum(p_i^2)`` of the normalised mode powers ``p_i``.

    Parameters
    ----------
    probe : torch.Tensor
        ``(M, ...)`` complex modes.
    drop_last : bool, optional
        Leave the last mode out when ``M > 1`` (the free mode of a mixed
        state). Default True.

    Returns
    -------
    float
        The purity, in ``(0, 1]``.
    """
    modes = probe[:-1] if drop_last and probe.shape[0] > 1 else probe
    power = (modes.abs() ** 2).reshape(modes.shape[0], -1).sum(-1).double()
    p = power / power.sum()
    return float((p**2).sum())


def _gram_power(a: Tensor, b: Tensor) -> Tensor:
    """``sum_ij |<a_i, b_j>|^2`` over the modes of ``(M, N)`` matrices."""
    return (a.conj() @ b.transpose(0, 1)).abs().square().sum()


def state_change(previous: Tensor, current: Tensor) -> float:
    """Change ``1 - F`` of the mixed state between two passes.

    ``F = Tr(rho_1 rho_2) / sqrt(Tr rho_1^2 Tr rho_2^2)`` with
    ``rho = sum_i |m_i><m_i|``, the normalised Hilbert-Schmidt fidelity of
    the two mode sets. ``F`` does not change with the scale, the phase of a
    mode or a unitary mixing of the modes. No gauge is removed: consecutive
    passes share a frame. ``1 - F`` is quadratic in a small change of a mode,
    so it is computed in double precision.

    Parameters
    ----------
    previous, current : torch.Tensor
        ``(M, ...)`` complex modes; each mode is flattened.

    Returns
    -------
    float
        ``1 - F`` in ``[0, 1]``; 0 when the two density matrices are
        proportional.
    """
    a = previous.reshape(previous.shape[0], -1).to(torch.complex128)
    b = current.reshape(current.shape[0], -1).to(a.device, torch.complex128)
    fidelity = _gram_power(a, b) / torch.sqrt(_gram_power(a, a) * _gram_power(b, b))
    return float(1.0 - fidelity)


class PassStopCallback:
    """The ``callback(k, log)`` of the BH driver with a :class:`PassStopRule`.

    For each pass ``k``, in this order: if the pass was accepted, apply the
    module's constraints (``apply_constraints``) and keep a copy of its state
    (``return_last``); then compute the state change of probe 0 against the
    previous pass (:func:`state_change`, no gauge removed) and the loss
    change, and ask the rule.

    Parameters
    ----------
    module : PlanarPtychography or object
        The module that the driver optimises: a ``PlanarPtychography`` or a
        ``Ptychography`` module. Its constraints are
        ``module.apply_pass_constraints()`` if it has that method, else the
        constraint schedule ``module._apply_all_constraints()``, gated on
        the pass index: ``start_epoch`` and ``freq`` count passes.
    rule : PassStopRule
        The stop rule; it keeps its own state.
    apply_constraints : bool, optional
        Apply the module's constraints after each accepted pass. Default True.
    return_last : bool, optional
        Keep the state of the last accepted pass for
        :meth:`restore_last`. Default False.

    Attributes
    ----------
    history : list of dict
        Per pass: ``k``, ``loss``, ``rel``, ``dloss``, ``reverted`` and,
        with ``rule.purity_tol`` set, ``purity``.
    """

    def __init__(
        self,
        module: Any,
        rule: PassStopRule,
        *,
        apply_constraints: bool = True,
        return_last: bool = False,
    ) -> None:
        self.module = module
        self.rule = rule
        self.apply_constraints = apply_constraints
        self.return_last = return_last
        self.history: list[dict[str, Any]] = []
        self._previous: Tensor | None = None
        self._purity: float | None = None
        self._last: dict[str, Tensor] | None = None

    def __call__(self, k: int, log: Any) -> bool:
        """Handle pass ``k``; return True to stop the solve after it."""
        reverted = bool(getattr(log, "reverted", False))
        if not reverted:
            if self.apply_constraints:
                self._apply_constraints(k)
            if self.return_last:
                state = self.module.state_dict()
                self._last = {n: v.detach().clone() for n, v in state.items()}
        probe = _get_probe_parameter(self.module.model, 0).detach()
        rel = None
        if self._previous is not None:
            rel = state_change(self._previous, probe)
        losses = list(log.losses)
        dloss = math.inf
        if len(losses) > 1:
            dloss = (losses[-2] - losses[-1]) / abs(losses[-2])
        entry = {"k": k, "loss": losses[-1] if losses else math.nan}
        entry.update(rel=rel, dloss=dloss, reverted=reverted)
        dpurity = math.inf
        if self.rule.purity_tol is not None:
            purity = mode_purity(probe)
            if self._purity is not None:
                dpurity = abs(purity - self._purity)
            self._purity = purity
            entry["purity"] = purity
        self.history.append(entry)
        if self.rule.update(k, rel, dloss, reverted, dpurity):
            logger.info("pass stop rule: stop after pass %d", k)
            return True
        self._previous = probe.clone()
        return False

    def _apply_constraints(self, k: int) -> None:
        """Apply the module's constraints that are due in pass ``k``.

        A ``Ptychography`` module gates its schedule on the trainer epoch,
        which stays 0 without a trainer. Here the schedule counts passes:
        a constraint is due from pass ``start_epoch`` on, every ``freq``
        passes.
        """
        apply = getattr(self.module, "apply_pass_constraints", None)
        if apply is not None:
            apply()
            return
        schedule = getattr(self.module, "constraint_callback", None)
        if schedule is None:  # a module without a schedule gates itself
            self.module._apply_all_constraints()
            return

        def due(name: str) -> bool:
            return schedule.should_apply(name, epoch=k, batches_per_epoch=1)

        self.module.should_apply = due  # the instance gate shadows the epoch gate
        try:
            self.module._apply_all_constraints()
        finally:
            del self.module.should_apply

    def restore_last(self) -> bool:
        """Load the state of the last accepted pass into the module.

        Returns
        -------
        bool
            True when a state was kept and loaded, False otherwise (no
            accepted pass, or ``return_last`` is False).
        """
        if self._last is None:
            return False
        with torch.no_grad():
            self.module.load_state_dict(self._last)
        return True
