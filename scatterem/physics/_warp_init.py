"""Initialise Warp without creating a CUDA context on every GPU of the node.

``wp.init()`` logs a start-up banner at Warp's default level (``LOG_INFO``) that
lists each CUDA device's total memory. Warp reads that by retaining the device's
primary context for the query and releasing it afterwards, so a process on a
node with several GPUs briefly creates a context (about 520 MiB on an H100) on
every one of them, one after the other, before it has bound its own device.
With N processes starting together on such a node, each GPU briefly holds N
of these contexts.

:func:`ensure_warp_initialized` raises Warp's log level to ``LOG_WARNING`` for
the call, which skips the banner and with it the per-device query, and restores
the level afterwards, so Warp's later messages (module loads, warnings) are
unchanged. Checked on Warp 1.17 and 1.18 (1.18 is the locked version), where
the banner is the only per-device context in ``wp.init``. To bring the banner
back, set ``warp.config.log_level = warp.LOG_INFO`` (or lower) before the call
(Warp 1.18 has no ``config.verbose``; on Warp 1.17 and older,
``warp.config.verbose = True`` did it). The
canonical helper is ``scatterem.utils.warp.launch.ensure_warp_initialized``,
which delegates here; the ``physics`` modules call this one because ``physics``
imports nothing of scatterem's outside itself.
"""

from __future__ import annotations

from collections.abc import Callable

import warp as wp

__all__ = ["ensure_warp_initialized", "quiet_implicit_start"]

#: Whether :func:`quiet_implicit_start` has wrapped Warp's start-up function.
_quiet_installed = False


def _quietly(start: Callable[[], None]) -> None:
    """Run ``start`` with Warp's log level raised past the start-up banner."""
    config = wp.config
    if hasattr(config, "log_level"):
        saved = config.log_level
        raised = max(saved, wp.LOG_WARNING)
        config.log_level = raised
        try:
            start()
        finally:
            # Warp <= 1.17's own init may set the level (its legacy
            # ``quiet``/``verbose``); keep what it chose rather than undo it
            if config.log_level == raised:
                config.log_level = saved
    else:  # older Warp: the banner is switched by ``config.quiet``
        # `quiet` is gone from Warp 1.18's typed config; this branch is for older Warp
        saved = config.quiet  # pyright: ignore[reportAttributeAccessIssue]
        config.quiet = True  # pyright: ignore[reportAttributeAccessIssue]
        try:
            start()
        finally:
            config.quiet = saved  # pyright: ignore[reportAttributeAccessIssue]


def ensure_warp_initialized() -> None:
    """Initialise the Warp runtime once, without the per-device banner query.

    Idempotent: ``wp.init`` returns at once after the first call.
    """
    _quietly(wp.init)


def quiet_implicit_start() -> None:
    """Make Warp's on-demand start skip the banner as well.

    The library does not start Warp at import (a started Warp has a thread, and
    a process with one cannot fork safely). ``wp.launch``, ``wp.array`` and the
    device queries then start Warp themselves, through the module-level
    ``init`` of ``warp._src.context``, banner and per-device query included.
    This wraps that function once so the first start runs through
    :func:`_quietly`. It starts nothing. ``wp.init`` (Warp's re-export of the
    original) is unchanged. On a Warp without that module layout it does
    nothing. Checked on Warp 1.17 and 1.18.
    """
    global _quiet_installed
    if _quiet_installed:
        return
    try:
        from warp._src import (
            context,  # noqa: PLC0415 -- Warp internals, checked 1.17/1.18
        )
    except ImportError:
        return
    original = getattr(context, "init", None)
    if original is None or not hasattr(context, "runtime"):
        return

    def init() -> None:
        if context.runtime is None:
            _quietly(original)
        else:
            original()

    context.init = init
    _quiet_installed = True
