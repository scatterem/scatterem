"""The per-iteration record the solvers append to."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BHLog:
    """Per-iteration log shared by all BH solvers.

    With ``per_group_alpha`` solvers, ``alphas`` records the first group's
    step size and ``group_alphas`` the full per-group vector; otherwise
    ``group_alphas`` stays empty.

    Parameters
    ----------
    losses : list[float]
        Loss value recorded at each pass.
    alphas : list[float]
        Step size recorded at each pass (first group's, under
        ``per_group_alpha``).
    betas : list[float]
        Momentum/curvature coefficient recorded at each pass.
    group_alphas : list[list[float]]
        Full per-group step-size vector at each pass, under
        ``per_group_alpha`` solvers; empty otherwise.
    wall : list[float]
        Cumulative solver work, in seconds, at each pass (diagnostic
        ``loss()`` evaluations excluded).
    polyak : dict, optional
        Final-state selection outcome, for incremental solvers.
    reverts : list[int]
        Pass indices that were reverted.
    batch_merge : list[int]
        Merge factor applied at each pass.
    early_stop : dict, optional
        Why and where a run ended early: ``reason="plateau"`` or
        ``reason="score"`` (``bh_gd_incremental(early_stop=True)``), or
        ``reason="callback"`` (any solver whose callback returned a truthy
        value).
    steps_per_pass : list[int]
        Number of steps actually run at each pass.
    curvature_freq : int, optional
        Number of steps between ``G x G`` curvature solves.
    reverted : bool
        Whether the most recent pass (or step) was reverted, i.e. its index
        is ``reverts[-1]``; set just before the callback runs, so a callback
        can skip constraints on a pass whose update was undone.
    scores : list[float]
        The selection ``score`` after each pass (``bh_gd_incremental(score=...)``);
        empty without a score.
    hook_steps : list[int]
        Number of ``step_hook`` calls per pass that returned True
        (``bh_gd_incremental(step_hook=...)``); empty without a hook. A
        reverted pass (``reverts``) undid the changes of its calls.
    solver : str
        The solver that ran, set by ``solve_ptychography`` (it can switch
        to ``"bh_gd_incremental"``); empty when a solver is called directly.
    """

    losses: list[float] = field(default_factory=list)
    alphas: list[float] = field(default_factory=list)
    betas: list[float] = field(default_factory=list)
    group_alphas: list[list[float]] = field(default_factory=list)
    wall: list[float] = field(
        default_factory=list
    )  # cumulative solver work seconds (diagnostic loss() evaluations excluded)
    polyak: dict | None = None  # final-state selection outcome (incremental)
    reverts: list[int] = field(default_factory=list)  # pass indices reverted
    batch_merge: list[int] = field(default_factory=list)  # merge factor per pass
    early_stop: dict | None = None  # plateau / callback stop info
    # cost accounting: backward sweeps = sum(steps_per_pass) * (1 + G / curvature_freq).
    # Without these two the run's compute is only expressible in seconds, which on a
    # contended card swings up to 1.8x on identical work.
    steps_per_pass: list[int] = field(default_factory=list)  # steps ACTUALLY run
    curvature_freq: int | None = None  # steps between G x G curvature solves
    reverted: bool = False  # the pass the callback is looking at was reverted
    scores: list[float] = field(default_factory=list)  # selection score per pass
    hook_steps: list[int] = field(default_factory=list)  # step_hook changes per pass
    solver: str = ""  # the solver solve_ptychography ran


def _callback_stop(callback, k: int, log: BHLog, cap: int) -> bool:
    """Run ``callback(k, log)`` if given; return True when it asks to stop.

    Sets ``log.reverted`` for pass ``k`` first. A truthy return records
    ``log.early_stop`` with ``reason="callback"``; the solver then breaks out
    of its loop after finishing pass ``k`` (final-state selection included).
    A ``None``/falsy return -- every pre-existing callback -- changes nothing.
    """
    if callback is None:
        return False
    log.reverted = bool(log.reverts) and log.reverts[-1] == k
    if not callback(k, log):
        return False
    _record_stop(log, k, cap, "callback")
    return True


def _record_stop(log: BHLog, k: int, cap: int, reason: str) -> None:
    """Record in ``log.early_stop`` that the run ended after pass ``k``."""
    log.early_stop = {
        "stopped_at": k,
        "passes_run": k + 1,
        "cap": cap,
        "reason": reason,
    }
