"""Stop and return rules of the BH driver's passes (``second_order.stop_rule``).

The reference is the stop block of the ``cb`` callback of the fast-probe
research driver, copied below. Both see the same synthetic pass sequences.
"""

from __future__ import annotations

import math
import random
from types import SimpleNamespace

import pytest
import torch

from scatterem.reconstruction.ptychography.iterative.second_order import (
    PassStopCallback,
    PassStopRule,
)
from scatterem.reconstruction.ptychography.iterative.second_order.stop_rule import (
    mode_purity,
)


def _ref_rule(cfg: dict):
    """Reference: the stop block of the research driver's pass callback.

    Copied (shortened) from bundle 2026-11-21. It is called only when
    ``probe_tol`` is set and a previous probe exists.
    """
    stop = {"at": None, "run": 0, "moved": False, "after": 0}

    def step(k, rel, dloss, reverted, dpur):
        if stop["moved"] and not reverted:
            stop["after"] += 1
        stop["moved"] = stop["moved"] or rel > cfg["probe_tol"]
        pam = cfg["passes_after_move"]
        if pam is not None and stop["after"] >= pam and stop["at"] is None:
            stop["at"] = k
            return True
        ok = (
            rel < cfg["probe_tol"]
            and abs(dloss) < cfg["loss_tol"]
            and not reverted
            and (cfg["purity_tol"] is None or dpur < cfg["purity_tol"])
            and (stop["moved"] or not cfg["needs_move"])
        )
        stop["run"] = stop["run"] + 1 if ok else 0
        if stop["run"] >= cfg["patience"] and k + 1 >= cfg["min_iters"]:
            if stop["at"] is None:
                stop["at"] = k
                return True
        return False

    return step


def _run(rule_step, passes) -> int | None:
    for k, p in enumerate(passes):
        if rule_step(k, *p):
            return k
    return None


def _skip_first(step):
    """The research driver asks the rule only once a previous probe exists."""
    return lambda k, rel, *a: rel is not None and step(k, rel, *a)


def _sequence(rng: random.Random, n: int = 40) -> list[tuple]:
    out = []
    for k in range(n):
        rel = rng.choice([0.0, 1e-4, 3e-3, 0.02, 0.2]) if k else None
        dloss = rng.choice([math.inf, 0.05, 1e-3, -5e-4, 1e-5]) if k else math.inf
        out.append((rel, dloss, rng.random() < 0.15, rng.choice([1e-4, 0.01, 0.1])))
    return out


CONFIGS = [
    dict(probe_tol=0.005, loss_tol=2e-3, min_iters=3, patience=1),
    dict(probe_tol=0.005, loss_tol=2e-3, min_iters=2, patience=3, needs_move=True),
    dict(probe_tol=0.005, loss_tol=2e-3, min_iters=5, patience=2, purity_tol=0.02),
    dict(probe_tol=0.005, loss_tol=1e-2, min_iters=2, patience=1, passes_after_move=4),
    dict(
        probe_tol=0.005,
        loss_tol=2e-3,
        min_iters=150,
        patience=1,
        passes_after_move=1000,
    ),
]


@pytest.mark.parametrize("cfg", CONFIGS)
def test_rule_matches_the_research_rule_on_random_sequences(cfg):
    full = dict(purity_tol=None, needs_move=False, passes_after_move=None) | cfg
    rng = random.Random(7)
    for _ in range(200):
        seq = _sequence(rng)
        rule = PassStopRule(**cfg)
        got = _run(rule.update, seq)
        expected = _run(_skip_first(_ref_rule(full)), seq)
        assert got == expected
        assert rule.stopped_at == got


def test_rule_needs_patience_and_min_iters():
    rule = PassStopRule(probe_tol=0.01, loss_tol=1e-3, min_iters=4, patience=2)
    calm = (1e-4, 1e-5, False)
    assert not rule.update(0, None, math.inf, False)
    assert not rule.update(1, *calm)
    assert not rule.update(2, *calm)  # patience met, min_iters not
    assert rule.update(3, *calm)
    assert rule.stopped_at == 3


def test_reverted_pass_resets_the_run():
    rule = PassStopRule(probe_tol=0.01, min_iters=1, patience=2)
    assert not rule.update(1, 1e-4, 1e-5, False)
    assert not rule.update(2, 1e-4, 1e-5, True)
    assert not rule.update(3, 1e-4, 1e-5, False)
    assert rule.update(4, 1e-4, 1e-5, False)


def test_needs_move_waits_for_a_large_change_then_counts_passes():
    rule = PassStopRule(
        probe_tol=0.01, min_iters=1, needs_move=True, passes_after_move=3
    )
    for k in range(1, 6):  # plateau: no change, loss calm
        assert not rule.update(k, 0.0, 1e-5, False)
    assert not rule.update(6, 0.05, 0.1, False)  # moves
    assert not rule.update(7, 0.05, 0.1, False)  # after 1
    assert not rule.update(8, 0.05, 0.1, True)  # reverted: not counted
    assert not rule.update(9, 0.05, 0.1, False)  # after 2
    assert rule.update(10, 0.05, 0.1, False)  # after 3
    assert rule.stopped_at == 10


def test_no_probe_tol_never_stops():
    rule = PassStopRule()
    assert not any(rule.update(k, 0.0, 0.0, False) for k in range(1, 20))


def test_mode_purity_drops_the_last_mode():
    p = torch.zeros(3, 1, 2, 2, dtype=torch.complex64)
    p[0, 0, 0, 0], p[1, 0, 0, 1], p[2, 0, 1, 1] = 2.0, 1.0, 5.0
    assert mode_purity(p) == pytest.approx(0.8**2 + 0.2**2)
    assert mode_purity(p[:1]) == 1.0


class _Module(torch.nn.Module):
    """A stand-in for ``Ptychography``: a probe, an object and a counter."""

    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.probe_model = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.ones(3, 1, 4, 4, dtype=torch.complex64))]
        )
        self.obj = torch.nn.Parameter(torch.zeros(4))
        self.n_constraints = 0

    def _apply_all_constraints(self) -> None:
        self.n_constraints += 1


def test_callback_applies_constraints_and_returns_the_last_accepted_pass():
    module = _Module()
    rule = PassStopRule(probe_tol=1e-6, min_iters=100)
    cb = PassStopCallback(module, rule, return_last=True)
    losses: list[float] = []
    probe = module.model.probe_model[0]
    for k, (loss, reverted) in enumerate([(1.0, False), (0.9, False), (0.95, True)]):
        with torch.no_grad():
            module.obj.fill_(float(k))
            probe.mul_(1.0 + 0.1 * k)
        losses.append(loss)
        assert not cb(k, SimpleNamespace(losses=losses, reverted=reverted))
    assert module.n_constraints == 2  # not after the reverted pass
    assert cb.history[0]["rel"] is None
    assert cb.history[1]["dloss"] == pytest.approx(0.1)
    with torch.no_grad():
        module.obj.fill_(99.0)
    assert cb.restore_last()
    assert torch.equal(module.obj, torch.full((4,), 1.0))


def test_callback_rel_is_the_density_matrix_change():
    module = _Module()
    cb = PassStopCallback(
        module, PassStopRule(probe_tol=0.5, min_iters=1), apply_constraints=False
    )
    log = SimpleNamespace(losses=[1.0], reverted=False)
    assert not cb(0, log)
    with torch.no_grad():  # a global phase and scale: the state is unchanged
        module.model.probe_model[0].mul_(2.0j)
    log.losses = [1.0, 1.0]
    assert cb(1, log)
    assert cb.history[1]["rel"] == pytest.approx(0.0, abs=1e-12)
    assert not PassStopCallback(module, PassStopRule()).restore_last()




