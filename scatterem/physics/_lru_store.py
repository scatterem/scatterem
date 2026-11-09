"""Private LRU store for mask tensors, with an entry limit and a byte budget.

The store follows the least-recently-used replacement rule of the computer
science literature (for example, Knuth, *The Art of Computer Programming*,
vol. 1, on lists). Written for scatterem from this literature.

The store keeps built masks so that a repeat request costs one lookup. A
lock makes the store safe for threads. Callers that give a mask to user code
must give a clone, because the stored tensor is shared.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable
from typing import NamedTuple

import torch

_log = logging.getLogger(__name__)

#: Maximum number of stored masks.
MAX_ENTRIES = 32
#: Maximum total size of the stored masks [bytes] (256 MiB).
MAX_BYTES = 256 * 1024**2


class MaskCacheStats(NamedTuple):
    """Counters of the mask store.

    Parameters
    ----------
    hits : int
        Number of requests that the store answered without a build.
    misses : int
        Number of requests that caused a build.
    entries : int
        Number of masks in the store now.
    nbytes : int
        Total size of the masks in the store now [bytes].
    """

    hits: int
    misses: int
    entries: int
    nbytes: int


class _LruStore:
    """Keep tensors by key, drop the least recently used one first."""

    def __init__(self, max_entries: int, max_bytes: int) -> None:
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._items: OrderedDict[Hashable, torch.Tensor] = OrderedDict()
        self._nbytes = 0
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

    def get(self, key: Hashable, build: Callable[[], torch.Tensor]) -> torch.Tensor:
        """Return the stored tensor for ``key``; build and store it if absent.

        Parameters
        ----------
        key : Hashable
            The key of the tensor.
        build : callable
            A function without arguments that returns the tensor.

        Returns
        -------
        torch.Tensor
            The stored tensor. The caller must not change it.
        """
        with self._lock:
            found = self._items.get(key)
            if found is not None:
                self._items.move_to_end(key)
                self._hits += 1
                return found
            self._misses += 1
            value = build()
            size = value.element_size() * value.nelement()
            if size > self._max_bytes:
                _log.debug("mask of %d bytes is above the budget; not stored", size)
                return value
            self._items[key] = value
            self._nbytes += size
            while (
                len(self._items) > self._max_entries or self._nbytes > self._max_bytes
            ):
                _, old = self._items.popitem(last=False)
                self._nbytes -= old.element_size() * old.nelement()
            return value

    def stats(self) -> MaskCacheStats:
        """Return the counters of the store."""
        with self._lock:
            return MaskCacheStats(
                self._hits, self._misses, len(self._items), self._nbytes
            )

    def clear(self) -> None:
        """Remove all tensors and set the counters to zero."""
        with self._lock:
            self._items.clear()
            self._nbytes = 0
            self._hits = 0
            self._misses = 0


#: The one store of the process.
STORE = _LruStore(MAX_ENTRIES, MAX_BYTES)


def device_key(device: torch.device | str | None) -> torch.device:
    """Return a device with an explicit index for CUDA.

    Parameters
    ----------
    device : torch.device, str or None
        The requested device. ``None`` means the CPU.

    Returns
    -------
    torch.device
        The device. A CUDA device without an index gets the current index.
    """
    dev = torch.device("cpu") if device is None else torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    return dev


def real_dtype_key(dtype: torch.dtype | None) -> torch.dtype:
    """Return the real floating dtype for a mask.

    Parameters
    ----------
    dtype : torch.dtype or None
        The requested dtype. ``None`` means ``torch.get_default_dtype()``.

    Returns
    -------
    torch.dtype
        The dtype.

    Raises
    ------
    TypeError
        If the dtype is not a real floating dtype.
    """
    dt = torch.get_default_dtype() if dtype is None else dtype
    if not isinstance(dt, torch.dtype) or not dt.is_floating_point:
        raise TypeError(f"a mask needs a real floating dtype; got {dt!r}")
    return dt
