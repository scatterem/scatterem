"""Streaming passes that do not stall when a foreign process keeps one core busy.

Pass 0 and the vBF copy run thousands of small torch ops. On the CPU each op is one
OpenMP parallel region over all intra-op threads, and the waiting threads spin. When
a foreign process keeps one core busy, the thread that shares its core runs only
when the scheduler gives it a time slice, and every region waits for it (about
10 ms on an 8-vCPU VM). A vBF build then took 26 s instead of 1.7 s.

Here the ops run on one OpenMP thread each, so no op waits for another thread:

- :func:`serial_here` runs the calling thread's ops on one thread for a while (the
  vBF copy).
- :func:`serial_threads` gives worker threads that do the same (exact pass 0 on the
  CPU). Each block goes to a free worker, so a worker on a busy core does fewer
  blocks and the other workers do more.

The float64 pass 0 (amplitudes, float data, a pre-divisor) still runs on all
intra-op threads, because its bits depend on the thread count; it can still stall.

The thread count is set with ``omp_set_num_threads``, which in libgomp belongs to
the calling thread. This module does not call ``torch.set_num_threads``: that call
changes process-wide state, and with n > 1 it breaks batched CPU LU in torch >= 2.10.
Where torch's OpenMP library cannot be found, :func:`workers` is 1 and the
passes run as before.
"""

from __future__ import annotations

import ctypes
import functools
import logging
import os
import threading
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from typing import TypeVar
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import torch

__all__ = ["ordered_map", "serial_here", "serial_threads", "workers"]

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


def _probe(lib: ctypes.CDLL) -> bool:
    """Whether ``lib.omp_set_num_threads`` sets torch's count in a new thread."""
    seen: list[int] = []

    def run() -> None:
        torch.get_num_threads()  # torch first sets this thread's count (lazily)
        lib.omp_set_num_threads(2)
        seen.append(torch.get_num_threads())
        lib.omp_set_num_threads(1)
        seen.append(torch.get_num_threads())

    t = threading.Thread(target=run)
    t.start()
    t.join()
    return seen == [2, 1]


@functools.cache
def _omp() -> ctypes.CDLL | None:
    """The OpenMP library whose ``omp_set_num_threads`` sets torch's thread count
    (torch's own first, then the loaded ones), or None (logged once)."""
    names = ("libgomp", "libiomp5", "libomp")
    paths = sorted((Path(torch.__file__).parent / "lib").glob("lib*omp*.so*"))
    try:
        with open("/proc/self/maps") as f:  # Linux; elsewhere torch's own lib dir only
            paths += sorted(
                {Path(ln.split()[-1]) for ln in f if "omp" in ln and "/" in ln}
            )
    except OSError:
        pass
    for p in dict.fromkeys(paths):
        if not (p.name.startswith(names) and os.path.exists(p)):
            continue
        try:
            lib = ctypes.CDLL(str(p))
            if _probe(lib):
                return lib
        except (OSError, AttributeError):
            continue
    logger.debug(
        "no OpenMP library sets torch's thread count per thread: pass 0 and the "
        "vBF copy use all intra-op threads and can stall on a busy core"
    )
    return None


def _serial() -> None:
    """Make the calling thread's torch CPU ops run on one OpenMP thread."""
    torch.get_num_threads()  # torch first sets this thread's count (lazily)
    lib = _omp()
    if lib is not None:
        lib.omp_set_num_threads(1)


# At most this many workers. On an 8-vCPU VM, 4 workers built the vBF benchmark
# faster than 8 (1.13 s against 1.38 s): the workers also run Python between the
# ops and wait for each other's GIL. Each worker also keeps its own heap (about
# 8 MB), so 4 workers keep the peak memory within about 25 MiB of one thread.
_MAX_WORKERS = 4


def workers() -> int:
    """Worker threads for a streaming pass: torch's intra-op thread count (at most
    ``_MAX_WORKERS``), or 1 where a thread cannot set its own OpenMP thread count."""
    n = min(torch.get_num_threads(), _MAX_WORKERS)
    return n if n > 1 and _omp() is not None else 1


@contextmanager
def serial_here() -> Iterator[None]:
    """Run the calling thread's torch CPU ops on one OpenMP thread in the block,
    then restore its count."""
    keep = torch.get_num_threads()
    lib = _omp() if keep > 1 else None
    if lib is None:
        yield
        return
    lib.omp_set_num_threads(1)
    try:
        yield
    finally:
        lib.omp_set_num_threads(keep)


@contextmanager
def serial_threads(n: int) -> Iterator[ThreadPoolExecutor]:
    """A pool of ``n`` threads whose torch CPU ops run on one OpenMP thread each.

    Parameters
    ----------
    n : int
        Worker threads (at least 1).

    Yields
    ------
    ThreadPoolExecutor
        The pool; it shuts down when the block ends.
    """
    pool = ThreadPoolExecutor(n, initializer=_serial, thread_name_prefix="serial-torch")
    try:
        yield pool
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        _release_free_heap()


def _release_free_heap() -> None:
    """Give the workers' free heap memory back to the system (glibc only).

    glibc gives each thread its own heap and keeps the blocks a thread freed for
    its next allocations. After the pool closes nothing reuses them, and they would
    add to the peak of the next pass (+60-130 MiB on the S3d bench). ``malloc_trim``
    only returns free pages; it changes no allocator setting.
    """
    try:
        ctypes.CDLL(None).malloc_trim(0)
    except (OSError, AttributeError):  # not glibc
        pass


def ordered_map(
    pool: ThreadPoolExecutor, fn: Callable[[T], R], items: Iterable[T], ahead: int
) -> Iterator[R]:
    """``fn(item)`` on the pool, the results in the order of ``items``.

    ``items`` is consumed in this thread. At most ``ahead`` items run or wait at a
    time, so their buffers stay bounded.
    """
    pending: deque[Future[R]] = deque()
    for item in items:
        pending.append(pool.submit(fn, item))
        if len(pending) >= ahead:
            yield pending.popleft().result()
    while pending:
        yield pending.popleft().result()
