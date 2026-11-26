"""The second-order driver's scoped ``requires_grad`` set.

:func:`~scatterem.reconstruction.ptychography.iterative.second_order.solve_ptychography`
differentiates with ``torch.autograd.grad`` over an explicit parameter list, so
it switches exactly that solve set on for the duration of the solve and
restores every flag it touched afterwards. The solve has no gradient
all-reduce, so it refuses a multi-rank ``torch.distributed`` group (spec 3).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

from torch import Tensor

from scatterem.training import _ranks


def _refuse_multi_rank() -> None:
    """Raise before any state change when a multi-rank group is active."""
    if _ranks.multi_rank():
        raise NotImplementedError(
            "solve_ptychography runs on one rank only: it has no gradient "
            "all-reduce across ranks (distributed second-order solve: spec 3)"
        )


@contextmanager
def _solve_requires_grad(
    gates: dict[str, bool] | None,
    sampler_params: Sequence[Tensor],
    params: Sequence[Tensor],
    optimize_object: bool,
) -> Iterator[None]:
    """Require grad on the solve set only; restore every flag on exit or raise.

    In the fixed trainable-set mode (``gates``, the model's gradient gates)
    every gate is opened for the solve too, so the forward does not detach a
    solved leaf.

    Raises
    ------
    NotImplementedError
        If a multi-rank ``torch.distributed`` group is active.
    """
    _refuse_multi_rank()
    saved = {id(p): (p, p.requires_grad) for p in (*sampler_params, *params)}
    saved_gates = None if gates is None else dict(gates)
    try:
        if not optimize_object:
            for p in sampler_params:
                p.requires_grad_(False)
        for p in params:
            p.requires_grad_(True)
        if gates is not None:
            gates.update(dict.fromkeys(gates, True))
        yield
    finally:
        for p, requires_grad in saved.values():
            p.requires_grad_(requires_grad)
        if gates is not None and saved_gates is not None:
            gates.update(saved_gates)
