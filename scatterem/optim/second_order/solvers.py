"""The bilinear-Hessian solvers: gradient descent, CG and quasi-Newton."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence

import torch
from torch import Tensor

from ._incremental import _backtrack, _batch_direction, _full_loss, _PassTracker
from .groups import (
    GroupSlice,
    _basis_from_groups,
    _default_groups,
    _per_group_alphas,
    _relax_and_clip,
    _step_groups,
    _step_groups_sliced,
)
from .log import BHLog, _callback_stop, _record_stop
from .oracles import ChunkedAutogradOracle
from .primitives import _direction, _newton_alpha, _rho2, _step, rdot

#: Whether :func:`bh_gd` reuses the losses it already holds.
#: A test hook, not a setting: tests set it to ``False`` to run the reference
#: path.  No environment variable reads it.
_BHGD_LOSS_REUSE = True


def bh_gd(
    oracle,
    n_iter: int,
    rho: Sequence[float] | None = None,
    fallback: float = 1e-2,
    per_group_alpha: bool = False,
    max_backtracks: int = 0,
    callback: Callable[[int, BHLog], bool | None] | None = None,
    precond: Callable[[Sequence[Tensor]], Sequence[Tensor]] | None = None,
) -> BHLog:
    """Gradient descent with the Newton step-size rule (BH-GD, paper eq. (15)).

    ``max_backtracks`` is a step-acceptance safeguard (as in :func:`bh_qn`):
    if the step increases the loss, halve it and retry, up to
    ``max_backtracks`` times, reverting fully if none is accepted; 0
    disables.  Essential on flat-tailed objectives (e.g. cold-start
    amplitude-loss ptychography), where near-zero positive curvature makes
    the raw Newton fallback overshoot catastrophically.

    With backtracking on, the loss at an accepted step is already in
    ``loss_after``, so it is logged without a new ``oracle.loss()`` call; and
    with ``callback=None`` the next iteration's ``loss_before`` is the logged
    loss, because ``grad()`` and the curvature solve do not move the
    parameters.  This matters for an oracle without a
    forward cache (:class:`ChunkedAutogradOracle`: one sweep per ``loss()``).

    Parameters
    ----------
    oracle : object
        Full-batch oracle exposing ``params``, ``grad()``, ``hess_op()`` and
        ``loss()``.
    n_iter : int
        Number of gradient-descent iterations.
    rho : Sequence[float] or None, optional
        Per-parameter preconditioner scale forwarded to `_rho2`. Default None.
    fallback : float, optional
        Step size when the Newton-rule curvature is non-positive. Default 1e-2.
    per_group_alpha : bool, optional
        Solve a per-parameter-group step size instead of one scalar Newton
        step. Default False.
    max_backtracks : int, optional
        Maximum halvings tried when a step increases the loss; 0 (the
        default) disables backtracking.
    callback : callable, optional
        Called as ``callback(k, log)`` after every iteration; a truthy return
        stops the run (``log.early_stop``, see :class:`BHLog`). Default None.
    precond : callable, optional
        Per-iteration diagonal preconditioner: ``precond(params)`` returns one
        positive map per parameter; the direction becomes ``-precond_i * g_i``
        (supersedes ``rho``; the Newton rule sizes any direction).

    Returns
    -------
    BHLog
        Log of losses, step sizes and timing across iterations.
    """
    log = BHLog()
    log.curvature_freq = 1  # full batch: one step per pass, curvature every step
    r2 = _rho2(rho, oracle.params)
    work_time = 0.0
    carried: float | None = None  # logged loss at the current parameters
    for k in range(n_iter):
        t_iter_start = time.perf_counter()
        g = oracle.grad()
        s = _direction(g, r2, precond, oracle.params)
        if per_group_alpha:
            alphas, _hss = _per_group_alphas(oracle, g, s, fallback)
        else:
            Hs = oracle.hess_op(s)
            alphas = [_newton_alpha(g, s, Hs, fallback)] * len(s)
        loss_before = None
        if max_backtracks > 0:
            loss_before = carried if carried is not None else oracle.loss()
        _step_groups(oracle.params, alphas, s)
        logged: float | None = None
        if max_backtracks > 0:
            loss_after = oracle.loss()
            n_bt = 0
            while (
                not math.isfinite(loss_after) or loss_after > loss_before
            ) and n_bt < max_backtracks:
                # halve the applied step
                _step_groups(oracle.params, [-a / 2.0 for a in alphas], s)
                alphas = [a / 2.0 for a in alphas]
                loss_after = oracle.loss()
                n_bt += 1
            if loss_after > loss_before and n_bt >= max_backtracks:
                _step_groups(oracle.params, [-a for a in alphas], s)  # revert
                alphas = [0.0] * len(alphas)
                log.reverts.append(k)
            elif _BHGD_LOSS_REUSE:
                logged = loss_after  # no parameter update since it was taken
        alpha = alphas[0]
        if per_group_alpha:
            log.group_alphas.append(alphas)
        work_time += time.perf_counter() - t_iter_start
        logged = oracle.loss() if logged is None else logged
        log.losses.append(logged)
        carried = logged if (_BHGD_LOSS_REUSE and callback is None) else None
        log.alphas.append(alpha)
        log.betas.append(0.0)
        log.wall.append(work_time)
        log.steps_per_pass.append(1)
        if _callback_stop(callback, k, log, n_iter):
            break
    return log


_ADAPTIVE_ACCEPT_WARMUP = 64  # tested clean accepts before sampling kicks in
_ADAPTIVE_ACCEPT_STRIDE = 8  # then test every N-th step
_PASS_REVERT_TOL = 5e-3  # full-loss rise, over the start loss, that reverts a pass
_STALL_FRAC = 0.01  # a pass "stalls" when it improves by less than this
# fraction of the observed DESCENT RANGE (initial - best loss), not the loss size:
# meaningful for offset-dominated losses (low-dose Poisson NLL floors).
_EARLY_STOP_PATIENCE = 2  # consecutive stalled passes that end an early_stop run
_MIN_STEPS_PER_PASS = 8  # batch growth caps so a pass keeps >= this many steps
# (one step per full-data sweep wastes compute; < 64 batches may merge to full)


def bh_gd_incremental(
    closures: Sequence[Callable[[], Tensor]],
    params: Sequence[Tensor],
    n_passes: int,
    per_group_alpha: bool = True,
    fallback: float = 1e-2,
    max_backtracks: int = 8,
    relax: float = 1.0,
    seed: int | None = None,
    step_clip: Sequence[float | None] | None = None,
    curvature_freq: int = 1,
    fd_curvature: bool = False,
    adaptive_accept: bool = False,
    polyak: bool = False,
    group_slices: list[GroupSlice] | None = None,
    batch_growth: bool = False,
    early_stop: bool = False,
    callback: Callable[[int, BHLog], bool | None] | None = None,
    precond: Callable[[Sequence[Tensor]], Sequence[Tensor | None]] | None = None,
    direction: Callable[[list[Tensor], list[Tensor]], list[Tensor]] | None = None,
    score: Callable[[], float] | None = None,
    score_patience: int = 3,
    step_hook: Callable[[range], bool] | None = None,
) -> BHLog:
    """Minibatch (incremental) per-group BH-GD — LSQML-style updates.

    ``step_clip`` (aligned with ``params``, ``None`` = unbounded) is a
    per-group trust radius on the inf-norm of a single batch step: the
    solved alpha is rescaled so ``max|alpha * step|`` never exceeds the
    bound.  Needed for parameters the forward is PERIODIC in (scan
    position offsets enter as FFT phase ramps), where a wrapped-around
    giant step can still lower the batch loss and defeat backtracking.

    Iterates the per-batch loss closures; for each batch, takes a Newton-rule
    step (per-group G x G solve when ``per_group_alpha``) computed from that
    batch's local gradient/curvature, so every mini-step uses the LATEST
    parameters.  This trades the full-batch quadratic-model exactness for the
    update cadence that makes minibatch methods win on real ptychography
    (each pass = ``len(closures)`` parameter updates instead of one).

    Safeguards: per-batch step-acceptance backtracking on the batch loss
    (``max_backtracks`` halvings, full revert on failure) and an optional
    global ``relax`` factor on the solved step sizes.  ``seed`` enables
    per-pass batch shuffling (deterministic).  With a single closure, one
    pass is exactly one full-batch :func:`bh_gd` iteration.

    Fast-path options (all default off; per-group mode only):

    - ``curvature_freq=k``: re-solve the G x G step sizes every k-th step
      and reuse the cached alphas between (step sizes drift slowly across
      batches); any backtrack invalidates the cache so the next step
      re-solves.  The clip and acceptance safeguards run on every step.
    - ``fd_curvature``: secant curvature (``hess_quad_matrix_fd``) — plain
      gradient sweeps instead of the ``create_graph`` double-backward;
      same memory as a gradient pass.
    - ``adaptive_accept``: after ``_ADAPTIVE_ACCEPT_WARMUP`` consecutive
      tested clean accepts, test only every ``_ADAPTIVE_ACCEPT_STRIDE``-th
      step; any backtrack resets to testing every step.
    - ``polyak``: average the iterates over the last half of the passes;
      on exit the parameters are set to whichever of (last iterate,
      average) has the lower full loss, recorded in ``log.polyak``.
    - ``group_slices``: explicit step-size groups as (param_index, mode)
      pairs — ``mode=None`` for the whole tensor, an int for one dim-0
      slice (e.g. per-probe-mode groups on a (modes,K,M,M) tensor).  When
      given, ``step_clip`` aligns with the GROUPS, not the params.
    - ``batch_growth``: variance annealing — whenever a pass stalls
      (improves by < ``_STALL_FRAC`` of the observed descent range) or is
      reverted, double the effective batch by merging adjacent closures
      (capped so a pass keeps at least ``_MIN_STEPS_PER_PASS`` steps,
      i.e. ``n_batches // 8`` for large batch counts — one step per
      full-data sweep is wasted compute); merge factor per pass in
      ``log.batch_merge``.
    - ``early_stop``: ``n_passes`` becomes a CAP.  The run ends after
      ``_EARLY_STOP_PATIENCE`` consecutive stalled passes — with
      ``batch_growth`` each non-terminal stall first buys a batch
      doubling AND resets the patience (every merge level gets a fresh
      chance), so the run only stops when growing stopped helping.
      With ``early_stop`` or ``polyak`` the best-loss pass is snapshotted
      and the final parameters are the best of {last iterate, tail
      average, best pass} (``log.polyak``); ``log.early_stop`` records
      where and why the run ended.

    ``log.losses``/``log.wall`` carry one entry per PASS (the exact full
    loss after the pass — costs one extra forward sweep); ``log.group_alphas``
    one vector per batch step.

    Parameters
    ----------
    closures : Sequence[Callable[[], Tensor]]
        Per-batch loss closures; each call re-evaluates that batch's loss at
        the current `params`.
    params : Sequence[Tensor]
        Parameters to optimize in place.
    n_passes : int
        Number of passes over `closures` (a cap when `early_stop` is set).
    per_group_alpha : bool, optional
        Whether to solve a per-group G x G Newton step instead of one scalar
        step. Default is True.
    fallback : float, optional
        Step size used when the curvature estimate is non-positive. Default
        is 1e-2.
    max_backtracks : int, optional
        Maximum halvings tried when a batch step increases its loss. Default
        is 8.
    relax : float, optional
        Global multiplier applied to every solved step size. Default is 1.0.
    seed : int or None, optional
        Seed for per-pass batch-order shuffling; `None` disables shuffling.
        Default is `None`.
    step_clip : Sequence[float or None] or None, optional
        Per-group (or per-param, without `group_slices`) inf-norm trust
        radius on a single batch step. Default is `None` (unbounded).
    curvature_freq : int, optional
        Re-solve per-group step sizes every k-th step, reusing cached alphas
        between. Default is 1.
    fd_curvature : bool, optional
        Use finite-difference secant curvature instead of the exact
        double-backward Hessian. Default is False.
    adaptive_accept : bool, optional
        After a warm-up of clean accepts, test step acceptance only every
        `_ADAPTIVE_ACCEPT_STRIDE`-th step. Default is False.
    polyak : bool, optional
        Average iterates over the tail of the run and keep whichever of
        {last iterate, average} has lower full loss. Default is False.
    group_slices : list[GroupSlice] or None, optional
        Explicit step-size groups as (param_index, mode) pairs; when given,
        `step_clip` aligns with the groups, not `params`. Default is `None`.
    batch_growth : bool, optional
        Double the effective batch size (merging adjacent closures) whenever
        a pass stalls or is reverted. Default is False.
    early_stop : bool, optional
        Stop after `_EARLY_STOP_PATIENCE` consecutive stalled passes,
        turning `n_passes` into a cap. Default is False.
    callback : callable, optional
        Called as ``callback(k, log)`` after every pass; ``log.reverted`` says
        whether pass ``k`` was undone. A truthy return stops the run after
        this pass, final-state selection included. Default is `None`.
    precond : callable, optional
        Diagonal preconditioner, re-evaluated once per PASS at the pass-start
        iterate: ``precond(params)`` returns one positive map per parameter
        (``None`` = identity); every batch direction becomes
        ``-precond_i * g_i``. The per-group Newton rule sizes the
        preconditioned direction, so the maps need no normalization.
    direction : callable, optional
        ``direction(g, params)`` replaces ``-g``; excludes `precond`. Default is `None`.
    score : callable, optional
        The selection score at the current parameters, lower is better (e.g. a
        held-out loss), at the start and after each pass revert, before
        `callback` (pass scores in ``log.scores``). The best state, start
        included, is kept; the final state is the lowest-score candidate;
        `early_stop` also ends after `score_patience` passes without a new
        best (``reason="score"``). Pass revert, stall test, `batch_growth` and
        plateau stop still judge the training loss. Default `None` judges by
        the training loss.
    score_patience : int, optional
        See `score`. Default is 3.
    step_hook : callable, optional
        ``step_hook(span)`` after every batch step (after the acceptance test),
        with the ``range`` of closure indices the step used. It may change
        the parameters in place (for example a proximal step) and returns
        True when it did (``log.hook_steps``). The next step's acceptance
        test evaluates the loss at the changed parameters, so no solver state
        needs an update. A reverted pass restores its snapshot and so also
        undoes the hook's changes in that pass. Default is `None`.

    Returns
    -------
    BHLog
        Log of losses, step sizes, timing and (when applicable) the
        early-stop/polyak selection across passes.
    """
    if precond is not None and direction is not None:
        raise ValueError("bh_gd_incremental: pass precond or direction, not both")
    log = BHLog()
    log.curvature_freq = curvature_freq
    params = [p.requires_grad_(True) for p in params]
    groups = group_slices if group_slices is not None else _default_groups(params)
    merge = 1

    def build_oracles(m: int) -> list[ChunkedAutogradOracle]:
        return [
            ChunkedAutogradOracle(closures[i : i + m], params)
            for i in range(0, len(closures), m)
        ]

    batch_oracles = build_oracles(merge)
    order = list(range(len(batch_oracles)))
    merge_cap = min(
        len(closures), max(_MIN_STEPS_PER_PASS, len(closures) // _MIN_STEPS_PER_PASS)
    )
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)
    work_time = 0.0
    method = "fd" if fd_curvature else "exact"
    cached_alphas: list[float] | None = None
    alphas: list[float] = []  # empty until the first batch step of the first pass
    step_idx = clean_accepts = 0
    tracker = _PassTracker(early_stop or polyak, score, score_patience)
    polyak_from = n_passes - max(1, n_passes // 2)
    prev_total = _full_loss(closures)
    tracker.start(prev_total, params)  # with a score: the start is a candidate
    loss_scale = max(abs(prev_total), 1e-300)
    initial_total = min_total = prev_total
    stall_count = 0
    for k in range(n_passes):
        t_pass = time.perf_counter()
        with torch.no_grad():
            snapshot = [p.detach().clone() for p in params]
        if seed is not None and len(order) > 1:
            order = torch.randperm(len(order), generator=gen).tolist()
        pass_aborted = False
        steps_this_pass = hooked = 0
        pmaps = precond(params) if precond is not None else None
        for bi in order:
            o = batch_oracles[bi]
            g, batch_loss = o.grad_and_loss()
            if not math.isfinite(batch_loss):
                # an untested step overflowed: abort the pass NOW (the pass-end revert
                # restores the finite snapshot) rather than drag non-finite params on
                pass_aborted = True
                break
            s = _batch_direction(g, params, direction, pmaps)
            if per_group_alpha:
                if cached_alphas is None or step_idx % curvature_freq == 0:
                    basis = _basis_from_groups(s, groups)
                    # secant interval = |last solved step size| per group:
                    # curvature is measured over the scale actually stepped
                    hints = (
                        [abs(a) for a in cached_alphas]
                        if (method == "fd" and cached_alphas is not None)
                        else None
                    )
                    alphas, _hss = _per_group_alphas(
                        o, g, s, fallback, method, eps_hints=hints, basis=basis
                    )
                    cached_alphas = list(alphas)
                else:
                    alphas = list(cached_alphas)
            else:
                Hs = o.hess_op(s)
                alphas = [_newton_alpha(g, s, Hs, fallback)] * len(groups)
            alphas = _relax_and_clip(alphas, relax, step_clip, groups, s)
            test_step = max_backtracks > 0 and (
                not adaptive_accept
                or clean_accepts < _ADAPTIVE_ACCEPT_WARMUP
                or step_idx % _ADAPTIVE_ACCEPT_STRIDE == 0
            )
            _step_groups_sliced(params, groups, alphas, s)
            if test_step:
                # the fused pass already evaluated the closure: acceptance reference
                alphas, n_bt = _backtrack(
                    o, params, groups, alphas, s, batch_loss, max_backtracks
                )
                clean_accepts = clean_accepts + 1 if n_bt == 0 else 0
                if n_bt:
                    cached_alphas = None  # stale step sizes: re-solve next
            if per_group_alpha:
                log.group_alphas.append(alphas)
            step_idx += 1
            steps_this_pass += 1
            if step_hook is not None:
                span = range(bi * merge, min((bi + 1) * merge, len(closures)))
                hooked += bool(step_hook(span))
        work_time += time.perf_counter() - t_pass
        total = _full_loss(closures)
        reverted = (
            pass_aborted
            or not math.isfinite(total)
            or total - prev_total > _PASS_REVERT_TOL * loss_scale
        )
        if reverted:
            # undo a regressed pass from its snapshot; retest every step, drop the
            # step sizes; re-measure the loss: a callback (constraints) may have
            # raised it after prev_total, and a stale reference undoes every pass
            with torch.no_grad():
                for p, sv in zip(params, snapshot, strict=False):
                    p.copy_(sv)
            log.reverts.append(k)
            total = _full_loss(closures)
            clean_accepts = 0
            cached_alphas = None
        log.batch_merge.append(merge)
        # record the steps that ran: an aborted pass ran fewer than `order`
        log.steps_per_pass.append(steps_this_pass)
        if step_hook is not None:
            log.hook_steps.append(hooked)
        min_total = min(min_total, total)
        descent = initial_total - min_total
        # no stall verdicts before any real descent exists (cold start)
        stalled = descent > 0.0 and (
            reverted or prev_total - total < _STALL_FRAC * descent
        )
        if batch_growth and merge < merge_cap and stalled:
            # variance annealing: stalled steps -> larger batches, fresh patience
            merge = min(2 * merge, merge_cap)
            batch_oracles = build_oracles(merge)
            order = list(range(len(batch_oracles)))
            cached_alphas = None
            clean_accepts = 0
            stall_count = -1  # incremented to 0 below: patience reset
        prev_total = total
        log.losses.append(total)
        log.alphas.append(alphas[0] if alphas else 0.0)
        log.betas.append(0.0)
        log.wall.append(work_time)
        tracker.end_pass(total, params, log)  # score + best snapshot, pre-callback
        stop = _callback_stop(callback, k, log, n_passes)
        if polyak and (
            (early_stop and (stalled or tracker.avg is not None)) or k >= polyak_from
        ):
            # accumulate AFTER the callback (average over post-constraint iterates);
            # with early_stop the window opens at plateau onset, not a fixed fraction
            tracker.accumulate(params)
        stall_count = max(0, stall_count + 1) if stalled else 0
        if stop:
            break
        if early_stop and stall_count >= _EARLY_STOP_PATIENCE:
            _record_stop(log, k, n_passes, "plateau")
            break
        if early_stop and tracker.score_stalled():
            _record_stop(log, k, n_passes, "score")
            break
    tracker.finish(closures, params, log)
    return log


def bh_qn(
    oracle,
    n_iter: int,
    inner_iters: int = 12,
    inner_tol: float = 1e-10,
    rho: Sequence[float] | None = None,
    fallback: float = 1e-2,
    max_backtracks: int = 8,
    callback: Callable[[int, BHLog], bool | None] | None = None,
) -> BHLog:
    """Truncated-Newton method (BH-QN): inner linear CG solves H y = -g.

    The inner loop is standard preconditioned CG on the (fixed) Hessian at the
    current iterate, truncated at the last positive-curvature iterate
    (Newton-CG practice).  The outer step uses the Newton step-size rule.
    Cost: 1 grad + (inner CG iterations + 1) hess_op per outer step.

    ``max_backtracks`` is a step-acceptance safeguard: if the Newton-rule step
    increases the loss, revert and halve alpha, up to ``max_backtracks`` times;
    0 disables.

    Parameters
    ----------
    oracle : object
        Full-batch oracle exposing ``params``, ``grad()``, ``hess_op()`` and
        ``loss()``.
    n_iter : int
        Number of outer (truncated-Newton) iterations.
    inner_iters : int, optional
        Maximum inner CG iterations per outer step. Default is 12.
    inner_tol : float, optional
        Relative residual tolerance that stops the inner CG solve early.
        Default is 1e-10.
    rho : Sequence[float] or None, optional
        Per-parameter preconditioner scale forwarded to `_rho2` and used as
        the inner-CG preconditioner. Default is `None`.
    fallback : float, optional
        Step size used when the outer Newton-rule curvature estimate is
        non-positive. Default is 1e-2.
    max_backtracks : int, optional
        Maximum halvings tried when the outer step increases the loss; 0
        disables. Default is 8.
    callback : callable, optional
        Called as ``callback(k, log)`` after every outer iteration; a truthy
        return stops the run (``log.early_stop``). Default is `None`.

    Returns
    -------
    BHLog
        Log of losses, step sizes and timing across outer iterations.
    """
    log = BHLog()
    r2 = _rho2(rho, oracle.params)
    work_time = 0.0
    for k in range(n_iter):
        t_iter_start = time.perf_counter()
        g = oracle.grad()
        # inner CG on H y = -g, preconditioner M^{-1} = diag(rho^2)
        y = [torch.zeros_like(p) for p in oracle.params]
        r = [-gi for gi in g]
        z = [ri * ri2 for ri, ri2 in zip(r, r2, strict=False)]
        p_dir = [zi.clone() for zi in z]
        rz = float(rdot(r, z))
        for j in range(inner_iters):
            Hp = oracle.hess_op(p_dir)
            pHp = float(rdot(Hp, p_dir))
            if pHp <= 0.0 or not math.isfinite(pHp):
                if j == 0:
                    y = z  # fall back to preconditioned steepest descent
                break
            t = rz / pHp
            y = [yi + t * pi for yi, pi in zip(y, p_dir, strict=False)]
            r = [ri - t * hpi for ri, hpi in zip(r, Hp, strict=False)]
            z = [ri * ri2 for ri, ri2 in zip(r, r2, strict=False)]
            rz_new = float(rdot(r, z))
            if rz_new < inner_tol * max(rz, 1e-300):
                break
            p_dir = [zi + (rz_new / rz) * pi for zi, pi in zip(z, p_dir, strict=False)]
            rz = rz_new
        s = y
        Hs = oracle.hess_op(s)
        alpha = _newton_alpha(g, s, Hs, fallback)
        loss_before = oracle.loss()
        _step(oracle.params, alpha, s)
        if max_backtracks > 0:
            loss_after = oracle.loss()
            n_bt = 0
            while (
                not math.isfinite(loss_after) or loss_after > loss_before
            ) and n_bt < max_backtracks:
                _step(oracle.params, -alpha / 2.0, s)  # halve the applied step
                alpha /= 2.0
                loss_after = oracle.loss()
                n_bt += 1
            if loss_after > loss_before and n_bt >= max_backtracks:
                _step(oracle.params, -alpha, s)  # give up: revert fully
                alpha = 0.0
                log.reverts.append(k)
        work_time += time.perf_counter() - t_iter_start
        log.wall.append(work_time)
        log.losses.append(oracle.loss())
        log.alphas.append(alpha)
        log.betas.append(0.0)
        if _callback_stop(callback, k, log, n_iter):
            break
    return log
