"""Scope that reuses the layout spectra of potential builds.

Literature: a thermal diffuse scattering calculation builds many potentials
of one structure with different atomic form spectra (Kirkland 2020,
*Advanced Computing in Electron Microscopy*, 3rd ed., ch. 7). The layout
spectra are the smoothing-free particle-mesh spectra of Hockney and Eastwood
(1988), *Computer Simulation Using Particles*, ch. 7. Written for scatterem
from this literature.

Units: the stored layout spectra are dimensionless sums of weights.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import logging
import numbers
from collections import OrderedDict

import numpy as np
import torch

__all__ = ["SweepLedger", "spectrum_sweep"]

_log = logging.getLogger(__name__)

_ACTIVE: contextvars.ContextVar[SweepLedger | None] = contextvars.ContextVar(
    "scatterem_spectrum_sweep", default=None
)


class SweepLedger:
    """Store and counters of one reuse scope.

    Parameters
    ----------
    max_entries : int
        Largest number of stored entries. The least recently used entry
        goes first.

    Attributes
    ----------
    hits : int
        Builds that used stored layout spectra.
    misses : int
        Builds that computed and stored layout spectra.
    bypassed : int
        Builds that could not use the scope (thermal draws, or an input
        other than the form spectra that requires grad).
    entries : int
        Stored entries. 0 after the scope ends.
    nbytes : int
        Bytes of the stored tensors.
    """

    def __init__(self, max_entries: int) -> None:
        self._max = max_entries
        self._store: OrderedDict = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._bypassed = 0

    @property
    def hits(self) -> int:
        """Builds that used stored layout spectra."""
        return self._hits

    @property
    def misses(self) -> int:
        """Builds that computed and stored layout spectra."""
        return self._misses

    @property
    def bypassed(self) -> int:
        """Builds that could not use the scope."""
        return self._bypassed

    @property
    def entries(self) -> int:
        """Stored entries; 0 after the scope ends."""
        return len(self._store)

    @property
    def nbytes(self) -> int:
        """Bytes of the stored tensors."""
        total = 0
        for value in self._store.values():
            for row in value:
                for t in row:
                    if t is not None:
                        total += t.element_size() * t.numel()
        return total

    def __repr__(self) -> str:
        return (
            f"SweepLedger(hits={self._hits}, misses={self._misses}, "
            f"bypassed={self._bypassed}, entries={self.entries})"
        )

    # ---------------------------------------------------- used by the build
    def lookup(self, key):
        """Return the stored value of ``key`` (and count a hit) or None.

        Parameters
        ----------
        key : collections.abc.Hashable
            The key of the layout spectra.

        Returns
        -------
        list or None
            The stored rows of detached tensors, or None for a key that is
            not in the store.
        """
        value = self._store.get(key)
        if value is not None:
            self._store.move_to_end(key)
            self._hits += 1
        return value

    def keep(self, key, value) -> None:
        """Store detached layout spectra under ``key``; count a miss.

        Parameters
        ----------
        key : collections.abc.Hashable
            The key of the layout spectra.
        value : list
            Rows of tensors or None. The store keeps detached copies.
        """
        self._misses += 1
        frozen = [[None if t is None else t.detach() for t in row] for row in value]
        self._store[key] = frozen
        self._store.move_to_end(key)
        while len(self._store) > self._max:
            self._store.popitem(last=False)

    def note_bypass(self) -> None:
        """Count a build that could not use the scope."""
        self._bypassed += 1

    def _close(self) -> None:
        self._store.clear()


def active_ledger() -> SweepLedger | None:
    """Return the ledger of the innermost active scope, or None."""
    return _ACTIVE.get()


@contextlib.contextmanager
def spectrum_sweep(max_entries: int = 4):
    """Open a scope in which potential builds reuse their layout spectra.

    Inside the scope, a build keeps the layout spectra ``D_{s,e}`` (deposit
    and smoothing removal, with the prefactors inside, before the form
    spectra and the rolloff). A later build with the same key uses them again.
    The key holds content digests of the atom data, the coefficients and
    the replacement tensors (read at each build, so an in-place change of a
    caller array gives a new key), and every other input that changes
    ``D``. The form spectra are not in the key: they act afresh in each
    build, so their gradients stay correct.

    Only builds without thermal draws and without any input that requires
    grad (other than the form spectra) take part; others count as
    ``bypassed``. Results inside the scope equal results outside it bit for
    bit, because both use the same functions.

    Parameters
    ----------
    max_entries : int, optional
        Positive number of stored entries (default 4); the least recently
        used entry goes first. One entry holds ``S E Ny (Nx // 2 + 1)``
        complex values.

    Yields
    ------
    SweepLedger
        The counters of this scope.

    Notes
    -----
    Scopes nest. The inner scope starts empty and does not see the outer
    store. When a scope ends, also through an exception, the store is
    emptied and the outer scope is active again. The scope is local to a
    thread and to an asyncio task (``contextvars``). Outside all scopes, no
    build keeps anything.
    """
    if isinstance(max_entries, (bool, np.bool_)) or not isinstance(
        max_entries, numbers.Integral
    ):
        raise TypeError(f"max_entries: must be an integer; got {max_entries!r}")
    if int(max_entries) < 1:
        raise ValueError(f"max_entries: must be at least 1; got {max_entries}")
    ledger = SweepLedger(int(max_entries))
    token = _ACTIVE.set(ledger)
    try:
        yield ledger
    finally:
        _ACTIVE.reset(token)
        ledger._close()


def content_digest(value) -> str:
    """Return a blake2b digest of the bytes, dtype and shape of an array."""
    h = hashlib.blake2b(digest_size=20)
    if value is None:
        h.update(b"none")
        return h.hexdigest()
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().contiguous().numpy()
        h.update(f"tensor:{value.dtype}:{value.device}".encode())
    else:
        arr = np.ascontiguousarray(np.asarray(value))
        h.update(f"array:{arr.dtype}".encode())
    h.update(repr(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()
