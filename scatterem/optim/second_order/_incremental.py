"""Parts of :func:`.solvers.bh_gd_incremental`: one batch step and the pass record.

:func:`_batch_direction` and :func:`_backtrack` are the batch step's direction
and acceptance test. :class:`_PassTracker` keeps the best pass, the tail
average and the selection score across the passes, and runs the final-state
selection (:func:`.selection._select_final_state`).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
from torch import Tensor

from .groups import GroupSlice, _step_groups_sliced
from .log import BHLog
from .selection import _select_final_state


def _full_loss(closures: Sequence[Callable[[], Tensor]]) -> float:
    """The full loss, the sum of all closures, without a graph."""
    with torch.no_grad():
        return float(sum(c() for c in closures))


def _batch_direction(
    g: list[Tensor],
    params: list[Tensor],
    direction: Callable[[list[Tensor], list[Tensor]], list[Tensor]] | None,
    pmaps: Sequence[Tensor | None] | None,
) -> list[Tensor]:
    """The batch step direction: ``direction(g, params)``, ``-pmap * g`` or ``-g``."""
    if direction is not None:
        return direction(g, params)
    if pmaps is None:
        return [-gi for gi in g]
    return [-gi if m is None else -m * gi for m, gi in zip(pmaps, g, strict=False)]


def _backtrack(
    oracle,
    params: list[Tensor],
    groups: Sequence[GroupSlice],
    alphas: list[float],
    s: list[Tensor],
    loss_before: float,
    max_backtracks: int,
) -> tuple[list[float], int]:
    """Halve a step that raised the batch loss; undo it if no halving helps.

    The step ``alphas * s`` is already applied. Returns the accepted step
    sizes (all 0 after an undo) and the number of halvings.
    """
    loss_after = oracle.loss()
    n_bt = 0
    while (
        not math.isfinite(loss_after) or loss_after > loss_before
    ) and n_bt < max_backtracks:
        _step_groups_sliced(params, groups, [-a / 2.0 for a in alphas], s)
        alphas = [a / 2.0 for a in alphas]
        loss_after = oracle.loss()
        n_bt += 1
    if loss_after > loss_before and n_bt >= max_backtracks:
        _step_groups_sliced(params, groups, [-a for a in alphas], s)
        alphas = [0.0] * len(alphas)
    return alphas, n_bt


class _PassTracker:
    """Best pass, tail average and selection score of one incremental run.

    Parameters
    ----------
    keep_best : bool
        Snapshot the best pass (``early_stop`` or ``polyak``). A ``score``
        always snapshots.
    score : callable or None
        The selection score at the current parameters, lower is better. None
        judges the passes by their full training loss. With a score,
        :meth:`start` makes the start state the first candidate.
    patience : int
        Passes without a new best score after which :meth:`score_stalled`
        is True.
    """

    def __init__(
        self, keep_best: bool, score: Callable[[], float] | None, patience: int
    ) -> None:
        self.keep_best = keep_best or score is not None
        self.score = score
        self.patience = patience
        self.best_key = float("inf")  # score (or loss) of the best pass
        self.best_total = float("inf")  # full loss of the best pass
        self.best_state: list[Tensor] | None = None
        self.best_index = -1  # pass of the best state; -1 is the start state
        self.passes = 0
        self.since_best = 0
        self.avg: list[Tensor] | None = None
        self.avg_n = 0

    def start(self, total: float, params: Sequence[Tensor]) -> None:
        """With a score: score the start state and keep it as the best so far.

        So a run in which no pass beats its start returns the start state,
        not a pass with a worse score. Without a score this does nothing.
        """
        if self.score is None:
            return
        self.best_key, self.best_total = float(self.score()), total
        with torch.no_grad():
            self.best_state = [p.detach().clone() for p in params]

    def end_pass(self, total: float, params: Sequence[Tensor], log: BHLog) -> None:
        """Score the pass (``log.scores``) and snapshot it if it is the best.

        Runs after the pass-level revert and before the user callback, so
        the parameters correspond to ``total``.
        """
        key = total
        index, self.passes = self.passes, self.passes + 1
        if self.score is not None:
            key = float(self.score())
            log.scores.append(key)
        if not self.keep_best:
            return
        if key < self.best_key:
            self.best_key, self.best_total, self.best_index = key, total, index
            with torch.no_grad():
                self.best_state = [p.detach().clone() for p in params]
            self.since_best = 0
        else:
            self.since_best += 1

    def score_stalled(self) -> bool:
        """True after ``patience`` passes without a new best score."""
        return self.score is not None and self.since_best >= self.patience

    def accumulate(self, params: Sequence[Tensor]) -> None:
        """Add the current parameters to the tail average."""
        with torch.no_grad():
            if self.avg is None:
                self.avg = [p.detach().clone() for p in params]
                self.avg_n = 1
            else:
                self.avg_n += 1
                for a, p in zip(self.avg, params, strict=False):
                    a += (p.detach() - a) / self.avg_n

    def finish(
        self,
        closures: Sequence[Callable[[], Tensor]],
        params: Sequence[Tensor],
        log: BHLog,
    ) -> None:
        """Set ``params`` to the best of {last, average, best pass}, if any.

        ``log.polyak["best_pass_index"]`` is the pass of the best-pass
        candidate; -1 is the start state (see :meth:`start`).
        """
        if self.avg is None and self.best_state is None:
            return
        _select_final_state(
            closures,
            params,
            self.avg,
            self.avg_n,
            self.best_state,
            self.best_total,
            log,
            score=self.score,
            best_score=self.best_key,
        )
        if log.polyak is not None and self.best_state is not None:
            log.polyak["best_pass_index"] = self.best_index
