"""What :func:`.driver.solve_ptychography` needs of the model it solves.

The driver works on a :class:`SolveTarget`: an object that gives the
parameter groups, turns the caller's batches into a fixed list and computes
the training loss of one batch. Two models satisfy it:

* :class:`~.planar.PlanarPtychography`, the 2D mixed-state model, implements
  the protocol itself;
* the unified ``Ptychography`` module is wrapped by an adapter that its package registers with
  :func:`register_solve_target` when it is imported.

:func:`as_solve_target` returns the target of a model: the model itself when
it implements the protocol, else the first registered adapter that accepts
it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Protocol, runtime_checkable

from torch import Tensor

from scatterem.optim.second_order.log import BHLog

__all__ = ["SolveTarget", "as_solve_target", "register_solve_target"]

#: ``(score, step_hook, callback)`` for ``bh_gd_incremental``.
SolveHooks = tuple[
    Callable[[], float] | None,
    Callable[[range], bool] | None,
    Callable[[int, BHLog], bool | None] | None,
]


@runtime_checkable
class SolveTarget(Protocol):
    """The model interface of :func:`.driver.solve_ptychography`.

    A batch is whatever :meth:`solve_batches` returns; the driver only
    counts the batches and passes each one back to :meth:`batch_loss`.

    Parameters
    ----------
    *args, **kwargs
        Unused: a protocol is a type, not instantiated.
    """

    def check_options(
        self,
        *,
        heldout: Any,
        object_prior: Any,
        optimize_object: bool,
        compile_forward: bool,
    ) -> None:
        """Raise for a driver option that the model does not support.

        Parameters
        ----------
        heldout, object_prior : object or None
            The driver's held-out scoring and object prior.
        optimize_object : bool
            Whether the object is solved.
        compile_forward : bool
            Whether the forward is to be compiled.
        """
        ...

    def solve_batches(self, batches: Iterable, compile_forward: bool) -> list:
        """The fixed batch list of the solve, data on the model's device.

        Parameters
        ----------
        batches : Iterable
            The caller's batches.
        compile_forward : bool
            Prepare the batches for a compiled forward.

        Returns
        -------
        list
            One entry per batch, passed back to :meth:`batch_loss`.

        Raises
        ------
        ValueError
            If `batches` is empty.
        """
        ...

    def object_parameters(self) -> list[Tensor]:
        """The object parameters, one step-size group each.

        Returns
        -------
        list of Tensor
            The parameters.
        """
        ...

    def probe_parameters(self) -> list[Tensor]:
        """The probe parameters; a mixed-state probe has its modes on dim 0.

        Returns
        -------
        list of Tensor
            The parameters.
        """
        ...

    def position_parameters(self) -> list[Tensor]:
        """The scan-position offsets ``(K, 2)`` in pixels, one tensor per scan.

        Returns
        -------
        list of Tensor
            The parameters.
        """
        ...

    def grad_gates(self) -> dict[str, bool] | None:
        """The model's gradient gates, opened for the solve.

        Returns
        -------
        dict or None
            The gates, or None when the model has none.
        """
        ...

    def solve_hooks(
        self,
        fixed: list,
        heldout: Any,
        object_prior: Any,
        callback: Callable[[int, BHLog], bool | None] | None,
    ) -> SolveHooks:
        """The solver's ``score``, ``step_hook`` and ``callback``.

        Parameters
        ----------
        fixed : list
            The batches of :meth:`solve_batches`.
        heldout, object_prior : object or None
            The driver's held-out scoring and object prior.
        callback : callable or None
            The caller's pass callback.

        Returns
        -------
        tuple
            ``(score, step_hook, callback)``.
        """
        ...

    def batch_loss(self, batch: Any) -> Tensor:
        """The training loss of one batch of :meth:`solve_batches`.

        Parameters
        ----------
        batch : object
            One entry of the fixed batch list.

        Returns
        -------
        Tensor
            Scalar loss.
        """
        ...

    def finish_solve(self, heldout: Any, log: BHLog) -> None:
        """Tidy up after the solve (for example the held-out history).

        Parameters
        ----------
        heldout : object or None
            The driver's held-out scoring.
        log : BHLog
            The solve log.
        """
        ...


_FACTORIES: list[Callable[[Any], SolveTarget | None]] = []


def register_solve_target(
    factory: Callable[[Any], SolveTarget | None],
) -> Callable[[Any], SolveTarget | None]:
    """Register an adapter that turns a model into a :class:`SolveTarget`.

    Parameters
    ----------
    factory : callable
        ``factory(model)`` returns the target, or None for a model it does
        not handle.

    Returns
    -------
    callable
        ``factory``, so the function can be a decorator.
    """
    if factory not in _FACTORIES:
        _FACTORIES.append(factory)
    return factory


def as_solve_target(model: Any) -> SolveTarget:
    """The :class:`SolveTarget` of ``model``.

    Parameters
    ----------
    model : object
        A model that implements :class:`SolveTarget`, or one that a
        registered adapter accepts.

    Returns
    -------
    SolveTarget
        ``model`` itself, or the target the first accepting adapter returns.

    Raises
    ------
    TypeError
        If no adapter accepts ``model``.
    """
    if isinstance(model, SolveTarget):
        return model
    for factory in _FACTORIES:
        target = factory(model)
        if target is not None:
            return target
    raise TypeError(
        f"solve_ptychography cannot solve a {type(model).__name__}: it "
        "implements neither SolveTarget nor a registered adapter"
    )


def fixed_batch_list(batches: Iterable, convert: Callable[[Any], Any]) -> list:
    """``[convert(b) for b in batches]``; raise for an empty list.

    Raises
    ------
    ValueError
        If ``batches`` is empty.
    """
    fixed = [convert(b) for b in batches]
    if not fixed:
        raise ValueError("batches must be non-empty")
    return fixed
