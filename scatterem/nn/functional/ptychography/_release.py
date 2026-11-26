"""Drop the large operands of the recorded Warp launches, keep the launches.

``_KOUTER_CMD`` (``kouter.py``) and ``_KSUM_CMD`` (``ksum.py``) keep a strong
reference to every tensor they bind, so that a raw pointer in a recorded
parameter block can never dangle (see the comment above ``_KOUTER_CMD``).  The
cost: after the last launch, the entry still holds its ``G`` operand.  That
operand is a chunk VIEW of the whole virtual-bright-field stack ``G``, so the
entry keeps all of ``G`` allocated after the reconstruction returns.  A second
run on the same dataset then builds a new ``G`` while the old one is still
allocated, and peaks one ``G`` higher.

:func:`release_launch_operands` replaces the stored ``G`` and output tensors
of every entry with a sentinel record that no tensor can match.  The recorded
command, its scalars and the small operands stay.  The next replay therefore
rebinds those two slots before it launches -- it does that on most calls
anyway, because each wrapper makes a new ``view_as_real`` object per call -- so
no launch reads a stale pointer.  The recorded ``wp.Launch`` itself holds the
tensor twice more, and the release clears both: the packed ``array_t`` in
``cmd.params`` (``wi.from_torch(..., return_ctype=True)`` keeps its source on
the struct) and the record-time ``wp.array`` in ``cmd.fwd_args``.

:func:`releases_launch_operands` wraps a function or a generator function so
that the release runs when it returns, raises, or (for a generator) closes.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import TypeVar

from . import kouter, ksum

F = TypeVar("F", bound=Callable)

#: ``_KOUTER_CMD`` slots that hold large tensors: ``G`` (0) and the output (13).
_KOUTER_LARGE_SLOTS = (0, 13)
#: ``_KSUM_CMD`` slots that hold large tensors: ``G`` (0) and the output (12).
_KSUM_LARGE_SLOTS = (0, 12)


def _unbind(cmd, slot: int) -> None:
    """Write a NULL array descriptor into ``slot`` of the recorded ``cmd``.

    The packed descriptor is not the only holder: the ``array_t`` that
    ``wi.from_torch(..., return_ctype=True)`` returns keeps its source tensor
    as an attribute, and the command keeps that struct in ``cmd.params``.  The
    record-time ``wp.array`` in ``cmd.fwd_args`` holds the tensor too.
    ``set_param_at_index_from_ctype`` replaces both with a fresh, empty struct
    of the same type (``data == 0``), so no tensor stays reachable and a launch
    without a rebind would read a NULL pointer, not freed memory.
    """
    # ``params[0]`` is the launch bounds; argument ``i`` sits at ``i + 1``.
    cmd.set_param_at_index_from_ctype(slot, type(cmd.params[slot + 1])())


def release_launch_operands() -> None:
    """Drop the ``G`` and output operands of every recorded k-outer / k-sum launch.

    Always safe: each replay rebinds a slot whose record is the sentinel
    before it launches, so the packed stale pointer is never read.  It does
    not change numbers, and it keeps the recorded commands, so the replay
    speed-up inside a reconstruction stays.
    """
    for cmd, _scalars, bound in kouter._KOUTER_CMD.values():
        for slot in _KOUTER_LARGE_SLOTS:
            # ``(tensor, data_ptr, shape, ndim)``: ``None is t`` is never true,
            # so the replay rebinds; the rank stays for its rank check.
            bound[slot] = (None, -1, None, bound[slot][3])
            _unbind(cmd, slot)
    for cmd, _scalars, bound, part in ksum._KSUM_CMD.values():
        for slot in _KSUM_LARGE_SLOTS:
            if part is not None and bound[slot][0] is part:
                # The entry owns ``part``; keeping it bound costs nothing.
                continue
            # ``(tensor, data_ptr, shape, stride, dtype)``: no live tensor has
            # ``data_ptr() == -1``, so the replay rebinds.
            bound[slot] = (None, -1, None, None, None)
            _unbind(cmd, slot)


def releases_launch_operands(fn: F) -> F:
    """Decorate ``fn`` to call :func:`release_launch_operands` when it ends.

    For a generator function the release runs when the generator is exhausted,
    closed, or collected.
    """
    if inspect.isgeneratorfunction(fn):

        @functools.wraps(fn)
        def gen_wrapper(*args, **kwargs):
            try:
                yield from fn(*args, **kwargs)
            finally:
                release_launch_operands()

        return gen_wrapper  # pyright: ignore[reportReturnType]

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        finally:
            release_launch_operands()

    return wrapper  # pyright: ignore[reportReturnType]
