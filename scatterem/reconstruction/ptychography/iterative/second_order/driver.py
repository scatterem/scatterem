"""``solve_ptychography``: the one-call entry point over the solvers."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from torch import Tensor

from scatterem.optim.second_order.cg import bh_cg
from scatterem.optim.second_order.groups import GroupSlice
from scatterem.optim.second_order.log import BHLog
from scatterem.optim.second_order.oracles import ChunkedAutogradOracle
from scatterem.optim.second_order.solvers import bh_gd, bh_gd_incremental

from ._grad_scope import _refuse_multi_rank, _solve_requires_grad
from .target import SolveTarget, as_solve_target

logger = logging.getLogger(__name__)

#: The solvers ``solve_ptychography`` accepts as ``solver=``.
SOLVERS = ("bh_gd", "bh_cg", "bh_gd_incremental", "secant_gd")


def solve_ptychography(
    module,
    batches,
    n_iter: int,
    optimize_probe: bool = True,
    optimize_positions: bool = False,
    position_step_px: float | None = 0.5,
    per_group_alpha: bool = True,
    solver: str = "bh_gd",
    fallback: float = 1e-2,
    max_backtracks: int = 8,
    probe_mode_groups: bool = False,
    batch_growth: bool = False,
    early_stop: bool = False,
    compile_forward: bool = False,
    callback: Callable[[int, BHLog], bool | None] | None = None,
    precond: Callable | None = None,
    optimize_object: bool = True,
    position_preconditioner: Callable | None = None,
    heldout: Any = None,
    object_prior: Any = None,
) -> BHLog:
    """Cold-start second-order solver for a ptychography model.

    ``module`` is a :class:`~.target.SolveTarget`: the unified
    ``Ptychography`` module (through its registered adapter) or the 2D mixed-state model
    :class:`~.planar.PlanarPtychography`. The rest of this text describes
    the ``Ptychography`` module; the 2D model has its own batches and
    refuses `compile_forward`, `heldout` and `object_prior`.

    Optimizes the object (sampler parameters) and, with
    ``optimize_probe=True``, the probe parameters — each tensor its own
    step-size group — against the module's own assembled training loss
    (data term + enabled regularizers), so it minimizes exactly what Adam
    training minimizes.  With ``optimize_positions=True`` the per-angle
    position offsets (``model.dr``) join as their own group(s); ``dr``
    enters the forward unconditionally, so no extra flag is needed.
    ``free_path`` is never touched.  Meant to be called directly (cold
    start); calling it after a partial Adam fit works the same way.

    ``batches`` is a DataLoader or any iterable of the model's 6/8-tuple
    batches; it is materialized ONCE into a fixed full-batch list (BH needs
    deterministic full-data statistics — the requirement is statistical,
    not memory: gradients and curvature scalars accumulate per batch).
    The batches hold centred patterns, as for ``Ptychography.fit``; the
    model's ``prepare_data`` checks the first and resamples a padded grid.
    It does not stream: every batch's patterns sit on the device for the whole
    solve, also when the loader streams them from host RAM or disk (pattern
    streaming v1), so a streamed series larger than the GPU fails
    here; fit it with Adam (``Ptychography.fit``) instead.

    The solvers:

    - ``"bh_gd"`` (default), ``"bh_cg"``: full-batch gradient descent and
      conjugate gradients with step sizes from the exact bilinear Hessian.
      Cost: 2 full passes over the data per iteration (gradient pass + one
      fused curvature pass via ``ChunkedAutogradOracle.hess_quad_matrix``),
      roughly 2-3 Adam epochs, with no learning rates to tune.  ``"bh_cg"``
      is for object-only refinement (with ``optimize_probe=False``).
    - ``"bh_gd_incremental"``: minibatch gradient descent; every batch step
      solves its step sizes from the exact bilinear Hessian of that batch.
    - ``"secant_gd"``: minibatch gradient descent with per-group Newton step
      sizes.  Each step solves a small ``G x G`` system for one step size per
      parameter group (object, probe or probe mode, positions), so the step
      lies in the span of the group gradients, after Carlsson et al., "The
      bilinear Hessian for large scale optimization", arXiv:2502.03070.  The
      curvature in that system is a secant estimate from gradient
      differences (plain backward passes, no double-backward), not the
      bilinear Hessian, and it is solved again every 8th batch step only.
      Each step backtracks on the batch loss; adaptive acceptance tests only
      every few steps after a run of clean accepts.  The final state is the
      one of lowest full loss among the last iterate, the tail average of
      the iterates and the best pass (``log.polyak``).  This is
      ``bh_gd_incremental`` with ``curvature_freq=8``, ``fd_curvature``,
      ``adaptive_accept`` and ``polyak``.

    `probe_mode_groups`, `batch_growth`, `early_stop`,
    `position_preconditioner`, `heldout` and `object_prior` need a minibatch
    solver: with them, ``"bh_gd"`` and ``"bh_cg"`` change to
    ``"bh_gd_incremental"``.  ``log.solver`` records the solver that ran.

    ``compile_forward=True`` compiles the model's tilt-agnostic physics
    core (the forward model's ``compile_forward``, auto-selecting the fused
    real-waves path) and prepares the fixed batch list for it: per-batch
    gather indices are precomputed and PADDED to a common unique-count so
    every same-sized batch presents identical shapes to one compiled
    graph.  Requires ``solver="secant_gd"`` — the exact curvature path is a
    double-backward, which ``torch.compile``'s AOTAutograd rejects; the
    secant curvature uses plain gradients only.

    Parameters
    ----------
    module : SolveTarget or object
        The model to optimize: a ``SolveTarget``, or a ``Ptychography`` module
        through its registered adapter. Its assembled training loss (data term +
        enabled regularizers) is what the solver minimizes.
    batches : Iterable
        A DataLoader or any iterable of the model's batches (6/8-tuples for
        ``Ptychography``); materialized once into a fixed full-batch list.
    n_iter : int
        Number of solver iterations (full-data passes, see cost note above).
    optimize_probe : bool, optional
        Include the probe model's parameters as their own step-size group(s).
    optimize_positions : bool, optional
        Include the per-angle position offsets (``model.dr``) as their own
        step-size group(s), trust-bounded by `position_step_px`.
    position_step_px : float or None, optional
        Trust-region bound, in pixels, on each `optimize_positions` step.
        Default 0.5. None leaves the position steps unbounded.
    per_group_alpha : bool, optional
        Solve a separate step size per parameter group rather than one
        shared step size. ``"secant_gd"`` requires True.
    solver : {"bh_gd", "bh_cg", "bh_gd_incremental", "secant_gd"}, optional
        Solver to use, see above. The minibatch-only options change
        ``"bh_gd"`` and ``"bh_cg"`` to ``"bh_gd_incremental"``.
    fallback : float, optional
        Step size used when the curvature-based step size is rejected.
    max_backtracks : int, optional
        Maximum backtracking line-search steps per iteration.
    probe_mode_groups : bool, optional
        Give each probe mode (dim 0) its own step-size group instead of
        sharing the dominant mode's; selects `bh_gd_incremental`.
    batch_growth : bool, optional
        Enable incremental-solver batch growth; selects `bh_gd_incremental`.
    early_stop : bool, optional
        Enable incremental-solver early stopping; selects
        `bh_gd_incremental`.
    compile_forward : bool, optional
        Compile the model's tilt-agnostic physics core and pad the fixed
        batch list so every same-sized batch presents identical shapes to
        one compiled graph. Requires ``solver="secant_gd"``.
    callback : callable, optional
        Called as ``callback(iteration, log)`` after each iteration; a truthy
        return stops the solve after that iteration (``log.early_stop``,
        ``reason="callback"``) and ``log.reverted`` flags an undone pass.
    precond : callable, optional
        Diagonal preconditioner over the solver's parameter list (object
        sampler parameters, then probe parameters, then position offsets);
        ``precond(params)`` returns one map per parameter, ``None`` for
        identity. Forwarded to ``bh_gd_incremental`` (re-evaluated once per
        pass) or to ``bh_gd``/``bh_cg`` (per iteration).
    optimize_object : bool, optional
        Include the object (sampler parameters). ``False`` refines only the
        probe (and positions) against a fixed, known object -- e.g. one
        reconstructed once and reused across many probe settings. The
        ``requires_grad`` flag of every tensor the solve touches is restored
        on exit, including when it raises.
    position_preconditioner : callable or None, optional
        A ``direction(g, params)`` callable preconditioning each batch step's direction; selects
        ``"bh_gd_incremental"``, requires `optimize_positions` and excludes
        `precond`.
    heldout : object, optional
        Scores the held-out positions (its `dataset` and `held_out_indices`)
        after every pass; selects ``"bh_gd_incremental"``. Its ``history``
        gets the start state (``epoch=-1``) and one entry per pass, and the
        held-out deviance becomes the solver's selection ``score``: the
        final state is the candidate (start state, best pass, last pass or
        tail average) with the lowest deviance, so the solve never returns a
        state worse on the holdout than its start
        (``log.polyak["best_pass_index"] == -1`` when the start wins).
        `early_stop` also stops after 3 passes without a new best. The pass
        revert, the stall test, `batch_growth` and the plateau stop still
        judge the training loss. No batch may hold a held-out position.
    object_prior : object, optional
        A proximal denoiser prior on the object, configured as for an Adam
        fit (``cadence="prox"`` only; requires `optimize_object`); selects
        ``"bh_gd_incremental"``. Its step runs from the solver's step hook
        every ``every_n_patterns`` training patterns (or ``every_n_steps``
        batch steps), from pass ``start_epoch`` on, on the support of the
        training positions. A merged batch step (`batch_growth`) that covers
        several intervals applies one step only, so after a merge the prior
        runs less often than every ``every_n_patterns``. ``log.hook_steps``
        counts the steps per pass and ``object_prior.sigmas`` holds the
        ``sigma`` of each. A step raises the training loss. Pass `heldout`
        too: without it, the best-pass and final-state selection judge the
        training loss. Even with it, the pass revert, the stall test,
        `batch_growth` and the plateau stop judge the training loss, prox
        changes included: a pass whose loss rises by more than 0.5 % of the
        start loss is reverted, its prox steps too. No prior is
        attached by default; the module's ``object_denoiser="auto"`` affects
        Adam fits only.

    Returns
    -------
    BHLog
        Log of the solve (per-iteration losses, step sizes and diagnostics).

    Raises
    ------
    ValueError
        If `solver` is unknown, if `compile_forward` is True with another
        solver than ``"secant_gd"``, if ``"secant_gd"`` comes with
        `per_group_alpha` False, if `batches` is empty,
        if no trainable parameters are found, or if
        `position_preconditioner` is given without `optimize_positions` or
        together with `precond`, if `object_prior` has another cadence than
        ``"prox"`` or comes without `optimize_object`, if `heldout` has no
        held-out positions, or if a batch holds a held-out position.
    NotImplementedError
        If `heldout` scores held-out pixels, or the model does not support
        an option (the 2D model: `compile_forward`, `heldout`,
        `object_prior`).
    TypeError
        If `object_prior` cannot act on the module's object sampler, or
        `module` is neither a ``SolveTarget`` nor a ``Ptychography`` module.
    """
    _refuse_multi_rank()  # first: nothing compiled or flipped on a refused solve
    hooks = (position_preconditioner, heldout, object_prior)
    minibatch = any((probe_mode_groups, batch_growth, early_stop)) or any(
        h is not None for h in hooks
    )
    solver = _choose_solver(solver, compile_forward, minibatch, per_group_alpha)
    if position_preconditioner is not None and not optimize_positions:
        raise ValueError("position_preconditioner requires optimize_positions=True")
    target = as_solve_target(module)
    target.check_options(
        heldout=heldout,
        object_prior=object_prior,
        optimize_object=optimize_object,
        compile_forward=compile_forward,
    )
    fixed = target.solve_batches(batches, compile_forward)
    n = len(fixed)

    object_params = target.object_parameters()
    params, groups, step_clip = _solve_set(
        target,
        object_params if optimize_object else [],
        optimize_probe,
        probe_mode_groups,
        optimize_positions,
        position_step_px,
    )
    if not params:
        raise ValueError("no trainable parameters found on the module")
    score, step_hook, callback = target.solve_hooks(
        fixed, heldout, object_prior, callback
    )
    gates = target.grad_gates()
    with _solve_requires_grad(gates, object_params, params, optimize_object):
        closures = [_batch_closure(target, b, n) for b in fixed]
        log = _run_solver(
            solver,
            closures,
            params,
            groups,
            step_clip,
            n_iter,
            per_group_alpha=per_group_alpha,
            fallback=fallback,
            max_backtracks=max_backtracks,
            batch_growth=batch_growth,
            early_stop=early_stop,
            callback=callback,
            precond=precond,
            direction=position_preconditioner,
            score=score,
            step_hook=step_hook,
        )
    log.solver = solver
    target.finish_solve(heldout, log)
    return log


def _batch_closure(target: SolveTarget, batch: Any, n: int) -> Callable[[], Tensor]:
    """The loss closure of one batch: its training loss divided by ``n``."""

    def closure() -> Tensor:
        return target.batch_loss(batch) / n

    return closure


def _choose_solver(
    solver: str, compile_forward: bool, minibatch: bool, per_group_alpha: bool
) -> str:
    """Check `solver` and return the solver that runs.

    With `minibatch` (an option that only the minibatch solvers have),
    ``"bh_gd"`` and ``"bh_cg"`` change to ``"bh_gd_incremental"``.
    ``"secant_gd"`` needs `per_group_alpha`: its secant curvature is per
    group, and the shared step size is an exact Hessian product.
    """
    if solver not in SOLVERS:
        raise ValueError(f"unknown solver {solver!r}; use one of {SOLVERS}")
    if solver == "secant_gd" and not per_group_alpha:
        raise ValueError(
            "solver='secant_gd' requires per_group_alpha=True: the shared step "
            "size uses the exact Hessian product (a double backward), not the "
            "secant curvature"
        )
    if compile_forward and solver != "secant_gd":
        raise ValueError(
            "compile_forward=True requires solver='secant_gd': the exact-curvature "
            "solvers differentiate twice through the forward "
            "(hess_quad_matrix / hess_op are double-backwards), which "
            "torch.compile does not support. The secant curvature of "
            "'secant_gd' uses plain gradient sweeps only."
        )
    if minibatch and solver in ("bh_gd", "bh_cg"):
        return "bh_gd_incremental"
    return solver


def _solve_set(
    target: SolveTarget,
    params: list[Tensor],
    optimize_probe: bool,
    probe_mode_groups: bool,
    optimize_positions: bool,
    position_step_px: float | None,
) -> tuple[list[Tensor], list[GroupSlice], list[float | None] | None]:
    """The solve set: parameters, step-size groups and the position trust bounds.

    ``params`` holds the object parameters (empty for a probe-only solve);
    the probe and, with `optimize_positions`, the position offsets follow.
    A `position_step_px` of None leaves the position steps unbounded.
    """
    params = list(params)
    groups: list[GroupSlice] = [(i, None) for i in range(len(params))]
    if optimize_probe:
        for p in target.probe_parameters():
            pi = len(params)
            params.append(p)
            if probe_mode_groups and p.dim() >= 1 and p.shape[0] > 1:
                # one step-size group per probe mode (dim 0): weak tail modes get
                # their own solved alpha instead of sharing the dominant mode's
                groups += [(pi, m) for m in range(p.shape[0])]
            else:
                groups.append((pi, None))
    step_clip: list[float | None] | None = None
    if optimize_positions:
        # the forward is periodic in dr (FFT phase ramp): an unbounded Newton step
        # can wrap around and still pass backtracking, so trust-bound each dr step
        step_clip = [None for _ in range(len(groups))]
        for d in target.position_parameters():
            groups.append((len(params), None))
            params.append(d)
            step_clip.append(position_step_px)
    return params, groups, step_clip


def _run_solver(
    solver: str,
    closures: list,
    params: list[Tensor],
    groups: list[GroupSlice],
    step_clip: list[float | None] | None,
    n_iter: int,
    *,
    per_group_alpha: bool,
    fallback: float,
    max_backtracks: int,
    batch_growth: bool,
    early_stop: bool,
    callback: Callable[[int, BHLog], bool | None] | None,
    precond: Callable | None,
    direction: Callable | None,
    score: Callable[[], float] | None = None,
    step_hook: Callable[[range], bool] | None = None,
) -> BHLog:
    """Dispatch the assembled problem to the solver ``solve_ptychography`` chose."""
    if solver in ("bh_gd_incremental", "secant_gd"):
        secant = solver == "secant_gd"
        return bh_gd_incremental(
            closures,
            params,
            n_passes=n_iter,
            per_group_alpha=per_group_alpha,
            fallback=fallback,
            max_backtracks=max_backtracks,
            seed=0,
            step_clip=step_clip,
            curvature_freq=8 if secant else 1,
            fd_curvature=secant,
            adaptive_accept=secant,
            polyak=secant,
            group_slices=groups,
            batch_growth=batch_growth,
            early_stop=early_stop,
            callback=callback,
            precond=precond,
            direction=direction,
            score=score,
            step_hook=step_hook,
        )
    if step_clip is not None:
        logger.warning(
            "solve_ptychography: position trust bound is only enforced by "
            "the minibatch solvers; full-batch solvers take unbounded dr steps"
        )
    oracle = ChunkedAutogradOracle(closures, params)
    if solver == "bh_gd":
        return bh_gd(
            oracle,
            n_iter=n_iter,
            per_group_alpha=per_group_alpha,
            fallback=fallback,
            max_backtracks=max_backtracks,
            callback=callback,
            precond=precond,
        )
    if solver == "bh_cg":
        return bh_cg(
            oracle,
            n_iter=n_iter,
            per_group_alpha=per_group_alpha,
            fallback=fallback,
            callback=callback,
            precond=precond,
        )
    raise ValueError(f"unknown solver {solver!r}; use one of {SOLVERS}")
