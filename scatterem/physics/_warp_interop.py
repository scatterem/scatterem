"""Warp's torch-interop and stream calls, with Warp started on first use.

``wp.from_torch``, ``wp.device_from_torch``, ``wp.stream_from_torch``,
``wp.ScopedStream``, ``wp.Stream`` and ``wp.synchronize_device`` do not start
the Warp runtime themselves; without it they fail with ``AttributeError:
'NoneType' object has no attribute ...``. ``wp.launch``, ``wp.array`` and the
device queries do start it.

The library used to start Warp when ``scatterem.nn``, ``scatterem.simulation``
or ``scatterem.reconstruction`` was imported. Starting Warp starts a thread, so
a process that had imported them could no longer fork safely: the test
harness's fork server could not preload the library, and every rank paid
2.5-3.5 s of imports. Now nothing starts Warp at import. Library code calls
these six through this module (imported as ``wi``, the same length as ``wp``),
and the first call starts Warp. No library module calls them on ``wp``
directly.

It lives in ``physics`` because that is the lowest layer, so every package may
import it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ParamSpec, TypeVar

import warp as wp

from scatterem.physics._warp_init import ensure_warp_initialized, quiet_implicit_start

__all__ = [
    "ScopedStream",
    "Stream",
    "device_from_torch",
    "from_torch",
    "stream_from_torch",
    "synchronize_device",
    "warp_capture_stream",
    "warp_sees_cuda",
]

# Any first start, also Warp's own from wp.launch, skips the banner.
quiet_implicit_start()

#: Set once Warp is started; starting it is one-way, so the flag never resets.
_started = False


def _start() -> None:
    global _started
    if not _started:
        ensure_warp_initialized()
        _started = True


P = ParamSpec("P")
R = TypeVar("R")

def _lazy(name: str, signature: Callable[P, R]) -> Callable[P, R]:
    """``wp.<name>`` with Warp started first, typed like ``signature``.

    The function is looked up on ``wp`` at each call, so a test that
    monkeypatches ``wp.<name>`` still reaches its patch.
    """

    def call(*args: P.args, **kwargs: P.kwargs) -> R:
        _start()
        return getattr(wp, name)(*args, **kwargs)

    call.__name__ = call.__qualname__ = name
    call.__doc__ = f"``wp.{name}``, after starting Warp if needed."
    return call


from_torch = _lazy("from_torch", wp.from_torch)
device_from_torch = _lazy("device_from_torch", wp.device_from_torch)
stream_from_torch = _lazy("stream_from_torch", wp.stream_from_torch)
synchronize_device = _lazy("synchronize_device", wp.synchronize_device)
# Constructors only: ``isinstance`` and annotations use ``wp.ScopedStream`` and
# ``wp.Stream`` themselves.
ScopedStream = _lazy("ScopedStream", wp.ScopedStream)
Stream = _lazy("Stream", wp.Stream)


def warp_sees_cuda() -> bool:
    """Whether Warp can drive a CUDA device at all.

    torch can see a CUDA device that Warp's CUDA build cannot drive (a driver
    too old for Warp's toolkit, say). A launch then dies inside Warp's torch
    bridge with a bare ``IndexError``. A Warp row's ``available`` predicate
    calls this, so ``"auto"`` declines to the torch spelling instead.

    Returns
    -------
    bool
        Whether Warp has at least one CUDA device.
    """
    _start()
    # Looked up on ``wp`` at each call, so a test can monkeypatch it.
    return bool(wp.is_cuda_available())


def warp_capture_stream(stream):
    """Make Warp launch onto ``stream`` for the duration of a graph capture.

    Warp launches onto its **own** per-device stream, not torch's current one.
    Under ``torch.cuda.graph`` that is a *silent* failure and not an error: the
    Warp kernels run eagerly while the graph is recording and are simply absent
    from it, so every replay leaves their outputs at whatever the recorded
    buffers happened to hold. Measured, not supposed: a captured Warp region
    without this made a benchmark oracle fail on every element, with no
    exception anywhere.

    ``sync_enter=False`` is load-bearing: :class:`warp.ScopedStream` otherwise
    makes the entered stream wait on Warp's previous (non-capturing) stream,
    and a cross-stream dependency on a non-capturing stream invalidates the
    capture outright (``cudaErrorStreamCaptureInvalidated``). The caller has
    already ordered ``stream`` against the work it depends on.

    Parameters
    ----------
    stream : torch.cuda.Stream
        The capturing torch stream.

    Returns
    -------
    warp.ScopedStream
        A context manager that routes Warp launches onto ``stream``.
    """
    return ScopedStream(stream_from_torch(stream), sync_enter=False)
