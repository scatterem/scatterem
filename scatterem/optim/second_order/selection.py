"""Final-state selection for :func:`.solvers.bh_gd_incremental`."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor

from .log import BHLog


def _select_final_state(
    closures: Sequence[Callable[[], Tensor]],
    params: Sequence[Tensor],
    avg: list[Tensor] | None,
    avg_n: int,
    best_state: list[Tensor] | None,
    best_total: float,
    log: BHLog,
    score: Callable[[], float] | None = None,
    best_score: float = float("inf"),
) -> None:
    """Set ``params`` to the best candidate; record it in ``log.polyak``.

    Candidates are the last iterate (re-evaluated here, since a callback may
    have mutated the parameters after the logged total), the tail average
    ``avg`` over ``avg_n`` iterates, and the best pass snapshot
    ``best_state`` (whose loss is ``best_total``); ``None`` omits one.

    Without ``score`` the lowest full loss wins. With ``score`` the lowest
    score wins (``best_score`` is the snapshot's score); ``log.polyak`` then
    also holds the score of each candidate.
    """

    # final-state selection: last iterate (post-callback, freshly
    # evaluated — callbacks mutate params after the logged total) vs
    # tail average vs best pass; keep the lowest
    def evaluate() -> tuple[float, float]:
        loss = float(sum(c() for c in closures))
        return loss, (float(score()) if score is not None else loss)

    with torch.no_grad():
        loss_last, score_last = evaluate()
        candidates = [
            ("last", loss_last, score_last, [p.detach().clone() for p in params])
        ]
        if avg is not None:
            for p, a in zip(params, avg, strict=False):
                p.copy_(a)
            candidates.append(("average", *evaluate(), avg))
        if best_state is not None:
            best_key = best_score if score is not None else best_total
            candidates.append(("best_pass", best_total, best_key, best_state))
        name, loss_win, score_win, state = min(candidates, key=lambda t: t[2])
        for p, w in zip(params, state, strict=False):
            p.copy_(w)
    loss_avg = next((c[1] for c in candidates if c[0] == "average"), None)
    log.polyak = {
        "used_average": name == "average",
        "selected": name,
        "selected_loss": loss_win,
        "loss_last": loss_last,
        "loss_avg": loss_avg if loss_avg is not None else float("inf"),
        "loss_best_pass": best_total if best_state is not None else None,
        "n_averaged": avg_n,
    }
    if score is not None:
        scores = {c[0]: c[2] for c in candidates}
        log.polyak.update(
            selected_score=score_win,
            score_last=score_last,
            score_avg=scores.get("average", float("inf")),
            score_best_pass=scores.get("best_pass"),
        )
