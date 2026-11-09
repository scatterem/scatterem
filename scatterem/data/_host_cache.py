"""The host pattern cache of a streamed series.

A *row* is a flat scan index ``iy * sx + ix`` of one tilt. Each tilt is a 4-D
raster storage source ``(sy, sx, ky, kx)``. :class:`HostCache` holds each tilt's
owned rows: ``"all"`` keeps every tilt once filled; ``"tilt"`` keeps two, filling
the next tilt on a background thread while the current one is served (the disk
tier).
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor

from torch import Tensor

from scatterem.data.storage import ArraySource

__all__ = ["HOST_CACHE_MODES", "HostCache"]

HOST_CACHE_MODES = ("all", "tilt")


def _mem_available() -> int:
    """Bytes of host memory available (``MemAvailable``), for ``host_cache="auto"``."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))


class HostCache:
    """Each tilt's owned rows in host RAM: every tilt (``"all"``) or two (``"tilt"``).

    In ``"tilt"`` mode the cache holds the tilt being served plus the one
    :meth:`prefetch` is filling on its background thread: a
    new tilt replaces the old one when :meth:`get` takes it. :meth:`get` is
    meant for one consumer (the ring's gather thread).

    Parameters
    ----------
    sources : sequence of ArraySource
        One 4-D raster source ``(sy, sx, ky, kx)`` per tilt.
    rows : sequence of Tensor or None
        Per tilt, the global rows to hold (``None``: every row). With ``None``,
        a C-contiguous CPU ``TensorSource`` is served as a view of its tensor, no
        copy; every other source is read into a new tensor.
    mode : {"all", "tilt"}
        Keep every tilt, or the served one plus the prefetched one.

    Attributes
    ----------
    stall_seconds : float
        Time :meth:`get` spent waiting for a fill (the disk tier's visible stall).
    fills : int
        Tilt fills so far.

    Raises
    ------
    ValueError
        If ``mode`` is unknown, if ``rows`` does not hold one row set per source,
        or if a source is not a 4-D raster.
    """

    def __init__(
        self, sources: Sequence[ArraySource], rows: Sequence[Tensor | None], mode: str
    ) -> None:
        if mode not in HOST_CACHE_MODES:
            raise ValueError(
                f"host_cache mode must be one of {HOST_CACHE_MODES}, got {mode!r}"
            )
        if len(sources) != len(rows):
            raise ValueError("one row set per source")
        for s in sources:
            if len(s.shape) != 4 or s.scan_ndim != 2:
                raise ValueError(
                    f"a host cache holds 4-D raster sources, got shape {s.shape} "
                    f"with scan_ndim {s.scan_ndim}"
                )
        self.sources, self.rows, self.mode = list(sources), list(rows), mode
        self._held: OrderedDict[int, Tensor] = OrderedDict()
        self._pending: dict[int, Future] = {}
        self._lock = threading.Lock()
        self._pool: ThreadPoolExecutor | None = None
        self._closed = False
        self.stall_seconds = 0.0
        self.fills = 0

    def _fill(self, t: int) -> Tensor:
        rows, src = self.rows[t], self.sources[t]
        with self._lock:
            self.fills += 1
        if rows is None:
            # A C-contiguous CPU TensorSource returns its own tensor: a view.
            whole = src.read(None, device="cpu")
            return whole.reshape(-1, *whole.shape[2:])
        return src.gather(rows, device="cpu")

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("the host cache is closed")

    def prefetch(self, t: int) -> None:
        """Start filling tilt ``t`` in the background (``"tilt"``) unless held.

        Parameters
        ----------
        t : int
            The tilt to fill. In ``"all"`` mode this does nothing.

        Raises
        ------
        RuntimeError
            If the cache is closed.
        """
        if self.mode != "tilt":
            return
        with self._lock:
            self._check_open()
            if t in self._held or t in self._pending:
                return
            if self._pool is None:
                self._pool = ThreadPoolExecutor(1, thread_name_prefix="pattern-fill")
            self._pending[t] = self._pool.submit(self._fill, t)

    def get(self, t: int) -> Tensor:
        """Return tilt ``t``'s held rows, filling the tilt first if needed.

        Parameters
        ----------
        t : int
            The tilt to serve.

        Returns
        -------
        Tensor
            ``(n_t, ky, kx)`` in the stored dtype, on the CPU.

        Raises
        ------
        RuntimeError
            If the cache is closed.
        """
        with self._lock:
            self._check_open()
            held = self._held.get(t)
            if held is not None:
                self._held.move_to_end(t)
                return held
            fut = self._pending.pop(t, None)
        t0 = time.perf_counter()
        data = fut.result() if fut is not None else self._fill(t)
        with self._lock:
            self.stall_seconds += time.perf_counter() - t0
            if self.mode == "tilt":
                self._held.clear()  # the caller keeps the old tilt it still uses
            self._held[t] = data
        return data

    def held(self) -> list[int]:
        """Return the tilts held now.

        Returns
        -------
        list of int
            The held tilts, least recently used first.
        """
        with self._lock:
            return list(self._held)

    def close(self) -> None:
        """Stop the fill thread and drop every held tilt; the cache is then closed."""
        with self._lock:
            self._closed = True
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        with self._lock:
            self._held.clear()
            self._pending.clear()
