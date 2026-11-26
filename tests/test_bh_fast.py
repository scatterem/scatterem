"""Tests for the fast-path additions to scatterem.optim.second_order
(fused grad+loss, FD curvature, curvature amortization, adaptive
acceptance, Polyak averaging)."""

import math

import pytest
import torch

torch.manual_seed(0)


def _make_problem(n_batches=8, seed=0, noise=0.0):
    """Per-batch coupled quadratic closures over [complex a, real b], float64.

    f_i(a, b) = ||A_i a - y_i||^2 + ||B_i b - z_i||^2
                + Re<c_i, a> * <d_i, b>          (bilinear cross-group term)

    With ``noise > 0`` the per-batch targets are jittered so batch optima
    disagree — a stochastic steady state for cadence-noise tests.
    """
    gen = torch.Generator().manual_seed(seed)

    def cr(*shape):
        return torch.complex(
            torch.randn(*shape, generator=gen, dtype=torch.float64),
            torch.randn(*shape, generator=gen, dtype=torch.float64),
        )

    a = cr(6).requires_grad_(True)
    b = torch.randn(4, generator=gen, dtype=torch.float64).requires_grad_(True)
    params = [a, b]
    a_true, b_true = cr(6), torch.randn(4, generator=gen, dtype=torch.float64)

    closures = []
    for _ in range(n_batches):
        A = cr(6, 6) + 3.0 * torch.eye(6, dtype=torch.complex128)
        B = torch.randn(4, 4, generator=gen, dtype=torch.float64) + 3.0 * torch.eye(
            4, dtype=torch.float64
        )
        c = 0.1 * cr(6)
        d = 0.1 * torch.randn(4, generator=gen, dtype=torch.float64)
        y = A @ (a_true + noise * cr(6))
        z = B @ (b_true + noise * torch.randn(4, generator=gen, dtype=torch.float64))

        def closure(A=A, B=B, c=c, d=d, y=y, z=z):
            fa = (A @ a - y).abs().square().sum()
            fb = (B @ b - z).square().sum()
            cross = (c.conj() * a).real.sum() * (d * b).sum()
            return fa + fb + cross

        closures.append(closure)
    return closures, params


class TestGradAndLoss:
    def test_matches_separate_grad_and_loss(self):
        from scatterem.optim.second_order import ChunkedAutogradOracle

        closures, params = _make_problem()
        o = ChunkedAutogradOracle(closures, params)
        g_ref = o.grad()
        loss_ref = o.loss()
        g, loss = o.grad_and_loss()
        assert loss == pytest.approx(loss_ref, rel=1e-12)
        for gi, gr in zip(g, g_ref, strict=True):
            assert torch.allclose(gi, gr)


def _counting(closures):
    """Wrap closures so every evaluation is counted."""
    counter = {"n": 0}

    def wrap(c):
        def wrapped():
            counter["n"] += 1
            return c()

        return wrapped

    return [wrap(c) for c in closures], counter


class TestFusedLossReuse:
    def test_incremental_pass_costs_three_evals_per_batch(self):
        """grad+loss fused and reused as loss_before: a clean (no-backtrack)
        pass needs exactly 3 closure evals per batch (fused grad pass,
        curvature pass, acceptance loss_after) plus one full-loss sweep for
        the pass-revert baseline and one at end of pass."""
        from scatterem.optim.second_order import bh_gd_incremental

        closures, params = _make_problem(n_batches=4)
        wrapped, counter = _counting(closures)
        bh_gd_incremental(wrapped, params, n_passes=1, max_backtracks=4)
        n = len(closures)
        assert counter["n"] == 3 * n + 2 * n  # 3/batch + baseline + log sweep


class TestFDCurvature:
    def _basis(self, s):
        return [
            [sj if j == i else torch.zeros_like(sj) for j, sj in enumerate(s)]
            for i in range(len(s))
        ]

    def test_fd_matches_exact_on_quadratic(self):
        """Secant curvature is exact on a quadratic (float64): the FD
        G x G matrix must match the double-backward one."""
        from scatterem.optim.second_order import ChunkedAutogradOracle

        closures, params = _make_problem()
        o = ChunkedAutogradOracle(closures, params)
        g = o.grad()
        s = [-gi for gi in g]
        basis = self._basis(s)
        A_exact = o.hess_quad_matrix(basis)
        A_fd = o.hess_quad_matrix_fd(basis, g)
        assert torch.allclose(A_fd, A_exact, rtol=1e-6, atol=1e-8)

    def test_fd_restores_params_exactly(self):
        from scatterem.optim.second_order import ChunkedAutogradOracle

        closures, params = _make_problem()
        before = [p.detach().clone() for p in params]
        o = ChunkedAutogradOracle(closures, params)
        g = o.grad()
        s = [-gi for gi in g]
        o.hess_quad_matrix_fd(self._basis(s), g)
        for p, b in zip(params, before, strict=True):
            assert torch.equal(p.detach(), b)

    def test_fd_uses_no_double_backward(self):
        """FD curvature must work when the graph cannot be re-differentiated
        (a closure built under inference-style single-backward assumptions is
        emulated by counting: G columns cost exactly G closure sweeps)."""
        from scatterem.optim.second_order import ChunkedAutogradOracle

        closures, params = _make_problem(n_batches=3)
        wrapped, counter = _counting(closures)
        o = ChunkedAutogradOracle(wrapped, params)
        g = o.grad()
        counter["n"] = 0
        s = [-gi for gi in g]
        o.hess_quad_matrix_fd(self._basis(s), g)
        assert counter["n"] == len(params) * len(wrapped)  # G sweeps, no more


class TestFDCurvatureInSolver:
    def test_fd_solver_converges_like_exact(self):
        from scatterem.optim.second_order import bh_gd_incremental

        c1, p1 = _make_problem(seed=1)
        log_exact = bh_gd_incremental(c1, p1, n_passes=4)
        c2, p2 = _make_problem(seed=1)
        log_fd = bh_gd_incremental(c2, p2, n_passes=4, fd_curvature=True)
        assert log_fd.losses[-1] == pytest.approx(log_exact.losses[-1], rel=1e-3)


class TestCurvatureAmortization:
    def test_cached_alphas_reused_between_resolves(self):
        from scatterem.optim.second_order import bh_gd_incremental

        closures, params = _make_problem(n_batches=8)
        log = bh_gd_incremental(
            closures, params, n_passes=1, curvature_freq=4, max_backtracks=0
        )
        ga = log.group_alphas
        assert len(ga) == 8
        assert ga[1] == ga[0] and ga[2] == ga[0] and ga[3] == ga[0]
        assert ga[4] != ga[0]  # re-solved
        assert ga[5] == ga[4] and ga[7] == ga[4]

    def test_amortized_still_descends(self):
        from scatterem.optim.second_order import bh_gd_incremental

        closures, params = _make_problem(n_batches=8)
        with torch.no_grad():
            init = float(sum(c() for c in closures))
        log = bh_gd_incremental(closures, params, n_passes=3, curvature_freq=4)
        assert log.losses[-1] < 0.1 * init

    def test_backtrack_invalidates_alpha_cache(self):
        """A rejected/backtracked step means the cached step sizes are stale:
        the next batch must re-solve even mid-window."""
        from scatterem.optim.second_order import bh_gd_incremental

        closures, params = _make_problem(n_batches=6)
        # relax >> 1 forces overshoot -> backtracking on (nearly) every step
        log = bh_gd_incremental(
            closures,
            params,
            n_passes=1,
            curvature_freq=6,
            relax=50.0,
            max_backtracks=8,
        )
        ga = log.group_alphas
        # with the cache invalidated by backtracks, consecutive raw alpha
        # vectors cannot all be halved copies of the first solve
        halvings = {
            tuple(round(a / g, 6) if g else 0.0 for a, g in zip(v, ga[0], strict=True))
            for v in ga[1:]
        }
        assert len(halvings) > 1


class TestAdaptiveAcceptance:
    def test_skips_acceptance_tests_after_warmup(self):
        from scatterem.optim.second_order import bh_gd_incremental

        n_passes, n_batches = 18, 8  # 144 steps: warmup is 64 tested accepts
        c1, p1 = _make_problem(n_batches=n_batches, seed=2)
        w1, cnt_plain = _counting(c1)
        bh_gd_incremental(w1, p1, n_passes=n_passes)
        c2, p2 = _make_problem(n_batches=n_batches, seed=2)
        w2, cnt_adapt = _counting(c2)
        bh_gd_incremental(w2, p2, n_passes=n_passes, adaptive_accept=True)
        assert cnt_adapt["n"] < cnt_plain["n"] - 20  # real savings

    def test_stays_safe_when_steps_are_bad(self):
        from scatterem.optim.second_order import bh_gd_incremental

        closures, params = _make_problem(seed=3)
        with torch.no_grad():
            init = float(sum(c() for c in closures))
        log = bh_gd_incremental(
            closures, params, n_passes=3, relax=100.0, adaptive_accept=True
        )
        assert math.isfinite(log.losses[-1])
        assert log.losses[-1] <= init


class TestFDEpsHints:
    def test_eps_hints_accepted_and_exact_on_quadratic(self):
        """The secant interval can be set per group (|previous alpha| —
        curvature measured over the scale actually stepped); on a quadratic
        the result is eps-independent and must still match the exact matrix."""
        from scatterem.optim.second_order import ChunkedAutogradOracle

        closures, params = _make_problem()
        o = ChunkedAutogradOracle(closures, params)
        g = o.grad()
        s = [-gi for gi in g]
        basis = [
            [sj if j == i else torch.zeros_like(sj) for j, sj in enumerate(s)]
            for i in range(len(s))
        ]
        A_exact = o.hess_quad_matrix(basis)
        A_fd = o.hess_quad_matrix_fd(basis, g, eps_hints=[0.037, None])
        assert torch.allclose(A_fd, A_exact, rtol=1e-6, atol=1e-8)


class TestPassRevert:
    def test_regressing_pass_is_reverted(self):
        """With safeguards forced off (max_backtracks=0) and huge relax, a
        pass strictly increases the loss; the pass-level safety net must
        revert to the pass-start iterate, so the final full loss cannot be
        (much) worse than the initial one."""
        from scatterem.optim.second_order import bh_gd_incremental

        closures, params = _make_problem(seed=5)
        with torch.no_grad():
            init = float(sum(c() for c in closures))
        log = bh_gd_incremental(
            closures, params, n_passes=3, relax=1e4, max_backtracks=0
        )
        with torch.no_grad():
            final = float(sum(c() for c in closures))
        assert log.reverts, "regressing passes must be recorded"
        assert final <= init * 1.005

    def test_nonfinite_pass_is_reverted_not_propagated(self):
        """A step that overflows to inf/nan (huge relax, no backtracking)
        must never survive: the pass reverts to its finite snapshot and the
        run ends with finite parameters."""
        from scatterem.optim.second_order import bh_gd_incremental

        closures, params = _make_problem(seed=15)
        with torch.no_grad():
            init = float(sum(c() for c in closures))
        log = bh_gd_incremental(
            closures, params, n_passes=3, relax=1e60, max_backtracks=0
        )
        assert log.reverts
        for p in params:
            assert (
                torch.isfinite(p.detach()).all()
                if not p.is_complex()
                else (
                    torch.isfinite(p.detach().real).all()
                    and torch.isfinite(p.detach().imag).all()
                )
            )
        with torch.no_grad():
            final = float(sum(c() for c in closures))
        assert math.isfinite(final) and final <= init * 1.005

    def test_healthy_run_never_reverts(self):
        from scatterem.optim.second_order import bh_gd_incremental

        closures, params = _make_problem(seed=6)
        log = bh_gd_incremental(closures, params, n_passes=4)
        assert log.reverts == []
        assert log.losses[-1] < log.losses[0]


def _make_mode_problem(n_batches=6, seed=0, scales=(1.0, 10.0, 100.0, 1000.0)):
    """Quadratic over a (modes, n) complex 'probe' + a real vector, where each
    mode row has a curvature ~scales[m]^2: one shared step size cannot serve
    all modes, per-mode groups can."""
    gen = torch.Generator().manual_seed(seed)

    def cr(*shape):
        return torch.complex(
            torch.randn(*shape, generator=gen, dtype=torch.float64),
            torch.randn(*shape, generator=gen, dtype=torch.float64),
        )

    M = len(scales)
    a = cr(M, 5).requires_grad_(True)
    b = torch.randn(3, generator=gen, dtype=torch.float64).requires_grad_(True)
    params = [a, b]
    a_true, b_true = cr(M, 5), torch.randn(3, generator=gen, dtype=torch.float64)
    sc = torch.tensor(scales, dtype=torch.float64).unsqueeze(1)

    closures = []
    for _ in range(n_batches):
        B = torch.randn(3, 3, generator=gen, dtype=torch.float64) + 3.0 * torch.eye(
            3, dtype=torch.float64
        )
        z = B @ b_true

        def closure(B=B, z=z):
            fa = (sc * (a - a_true)).abs().square().sum()
            fb = (B @ b - z).square().sum()
            return fa + fb

        closures.append(closure)
    return closures, params


class TestGroupSlices:
    def test_per_mode_groups_get_distinct_alphas_and_win(self):
        from scatterem.optim.second_order import bh_gd_incremental

        c1, p1 = _make_mode_problem(seed=8)
        log_shared = bh_gd_incremental(c1, p1, n_passes=3)
        c2, p2 = _make_mode_problem(seed=8)
        groups = [(0, 0), (0, 1), (0, 2), (0, 3), (1, None)]
        log_modes = bh_gd_incremental(c2, p2, n_passes=3, group_slices=groups)
        # the FIRST solve's per-mode alphas span the curvature range (the
        # problem fully converges within 3 passes, so late alphas are ~0)
        al = log_modes.group_alphas[0]
        assert len(al) == 5
        nonzero = [abs(a) for a in al[:4] if a != 0.0]
        assert max(nonzero) / min(nonzero) > 1e2
        # and the mode-resolved solve converges further
        assert log_modes.losses[-1] < 0.1 * log_shared.losses[-1]

    def test_default_groups_unchanged(self):
        from scatterem.optim.second_order import bh_gd_incremental

        c1, p1 = _make_problem(seed=9)
        log_a = bh_gd_incremental(c1, p1, n_passes=2)
        c2, p2 = _make_problem(seed=9)
        log_b = bh_gd_incremental(
            c2, p2, n_passes=2, group_slices=[(0, None), (1, None)]
        )
        assert log_a.losses[-1] == pytest.approx(log_b.losses[-1], rel=1e-9)


class TestBatchGrowth:
    def test_batch_merges_on_stall_and_caps(self):
        from scatterem.optim.second_order import bh_gd_incremental

        # noisy batch optima: per-batch steps rattle, passes stall quickly
        closures, params = _make_problem(n_batches=8, seed=10, noise=0.5)
        log = bh_gd_incremental(closures, params, n_passes=12, batch_growth=True)
        assert log.batch_merge, "merge factor must be logged per pass"
        assert log.batch_merge[0] == 1
        assert log.batch_merge[-1] > 1  # growth happened
        assert max(log.batch_merge) <= len(closures)  # capped at full batch
        assert log.batch_merge == sorted(log.batch_merge)  # monotone

    def test_growth_caps_at_min_steps_per_pass(self):
        """A merged 'pass' that takes only one step per full-data sweep is
        wasted compute: growth must cap so a pass keeps delivering at least
        ~8 steps (cap = n_batches // 8 for large batch counts; small
        problems may still merge to full batch)."""
        from scatterem.optim.second_order import bh_gd_incremental

        closures, params = _make_problem(n_batches=64, seed=17, noise=0.6)
        log = bh_gd_incremental(closures, params, n_passes=14, batch_growth=True)
        assert log.batch_merge[-1] > 1  # growth happened
        assert max(log.batch_merge) <= 8  # 64 // 8

    def test_growth_escapes_the_stochastic_stall(self):
        """Per-batch steps stall on conflicting batch optima (the plain run
        freezes); merged batches average the conflict away and keep
        descending toward the full-batch optimum."""
        from scatterem.optim.second_order import bh_gd_incremental

        c1, p1 = _make_problem(n_batches=8, seed=11, noise=0.5)
        log_plain = bh_gd_incremental(c1, p1, n_passes=14)
        c2, p2 = _make_problem(n_batches=8, seed=11, noise=0.5)
        log_grow = bh_gd_incremental(c2, p2, n_passes=14, batch_growth=True)
        assert log_grow.batch_merge[-1] > 1
        assert log_grow.losses[-1] < 0.7 * log_plain.losses[-1]


class TestEarlyStopping:
    def test_stops_on_plateau_long_before_cap(self):
        """An exactly quadratic problem converges in a few passes; with
        early_stop the solver must notice the plateau and not burn the cap."""
        from scatterem.optim.second_order import bh_gd_incremental

        closures, params = _make_problem(seed=12)
        log = bh_gd_incremental(closures, params, n_passes=50, early_stop=True)
        assert log.early_stop is not None
        assert len(log.losses) < 10

    def test_cap_without_flag_unchanged(self):
        from scatterem.optim.second_order import bh_gd_incremental

        closures, params = _make_problem(seed=12)
        log = bh_gd_incremental(closures, params, n_passes=6)
        assert log.early_stop is None
        assert len(log.losses) == 6

    def test_growth_gets_a_chance_before_stopping(self):
        """With batch_growth, a stall first triggers a batch doubling; the
        run only stops when the ENLARGED batch stalls too — so any early
        stop on the noisy problem must have grown at least once."""
        from scatterem.optim.second_order import bh_gd_incremental

        closures, params = _make_problem(n_batches=8, seed=13, noise=0.5)
        log = bh_gd_incremental(
            closures, params, n_passes=30, batch_growth=True, early_stop=True
        )
        if log.early_stop is not None:
            assert log.batch_merge[-1] > 1

    def test_final_state_is_the_best_seen(self):
        """Robust stopping must not give back progress: the returned params
        can never be worse than the best logged pass."""
        from scatterem.optim.second_order import bh_gd_incremental

        closures, params = _make_problem(n_batches=8, seed=14, noise=0.4)
        log = bh_gd_incremental(
            closures, params, n_passes=20, early_stop=True, polyak=True
        )
        with torch.no_grad():
            final = float(sum(c() for c in closures))
        assert final <= min(log.losses) * (1 + 1e-9)


class TestOffsetLossEarlyStop:
    def test_offset_dominated_loss_does_not_stop_prematurely(self):
        """Losses with a large constant floor (low-dose Poisson NLL) make
        any tolerance anchored to the loss MAGNITUDE meaningless — the whole
        meaningful descent can be a few percent of the value.  The stall
        criterion must anchor to the observed descent range, so slow-but-real
        progress is not classified as a plateau."""
        from scatterem.optim.second_order import bh_gd_incremental

        closures, params = _make_problem(seed=16)
        with torch.no_grad():
            qscale = float(sum(c() for c in closures))
        OFFSET = 1e4 * qscale  # loss floor >> the reducible part

        def with_offset(c, n=len(closures)):  # noqa: B008 -- bound once, on purpose
            def closure():
                return c() + OFFSET / n

            return closure

        offset_closures = [with_offset(c) for c in closures]
        # relax damps the steps so progress is slow-but-steady, like the
        # low-dose real data — magnitude-anchored tolerances see only stalls
        log = bh_gd_incremental(
            offset_closures, params, n_passes=60, early_stop=True, relax=0.2
        )
        with torch.no_grad():
            final_q = float(sum(c() for c in closures))  # offset-free part
        # the reducible part must actually be reduced (>=100x), not abandoned
        # at the first sub-magnitude-tolerance pass
        assert final_q < 0.01 * qscale
        # and the run should still terminate itself once truly converged
        assert log.early_stop is not None


class TestPolyakWithCallback:
    def test_comparison_uses_post_callback_state(self):
        """Callbacks may mutate params (constraint projections); the final
        keep-better comparison must be against the params as they are NOW,
        not the loss logged before the callback ran."""
        from scatterem.optim.second_order import bh_gd_incremental

        closures, params = _make_problem(n_batches=6, seed=7, noise=0.2)

        def spoil(k, log):
            with torch.no_grad():
                params[0] += 0.5  # constraint-like mutation AFTER the log

        log = bh_gd_incremental(
            closures, params, n_passes=6, polyak=True, callback=spoil
        )
        with torch.no_grad():
            fresh_last_or_avg = float(sum(c() for c in closures))
        # the solver must have evaluated the spoiled last iterate freshly,
        # so the recorded loss_last cannot equal the pre-callback log entry
        assert log.polyak["loss_last"] != pytest.approx(log.losses[-1])
        assert fresh_last_or_avg <= min(
            log.polyak["loss_last"], log.polyak["loss_avg"]
        ) * (1 + 1e-9)


class TestPolyakAveraging:
    def test_keeps_better_of_average_and_last(self):
        from scatterem.optim.second_order import bh_gd_incremental

        # noisy per-batch optima -> steady-state oscillation
        closures, params = _make_problem(n_batches=8, seed=4, noise=0.3)
        log = bh_gd_incremental(closures, params, n_passes=10, polyak=True)
        assert log.polyak is not None
        with torch.no_grad():
            final = float(sum(c() for c in closures))
        assert final <= min(log.polyak["loss_last"], log.polyak["loss_avg"]) * (
            1 + 1e-9
        )
        assert final <= log.losses[-1] * (1 + 1e-9)
