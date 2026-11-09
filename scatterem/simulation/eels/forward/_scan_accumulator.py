"""The per-atom ``img.index_add_`` of :mod:`.prism_eels_image`, in ONE ordered pass.

The STEM-EELS image is accumulated one ``index_add_`` per atom (and per
transition block), in atom order -- 128 launches at ``r7_eels/dense`` for 16
floats each.  They cannot be merged into one ``index_add_``: two atoms can hit the
same scan position, and one scatter with duplicate keys sums them in atomic
order, not atom order.  They do not need atomics at all, though.  Every
call's keys are duplicate-free (a ``nonzero`` row or ``arange(P)``), so call ``k``
is exactly "add row ``D[k]``", where ``D[k]`` holds the call's values at its keys
and ``+0.0`` elsewhere, and adding ``+0.0`` to the non-negative image is the
identity.  The sequential result is therefore ``img + D[0] + D[1] + ...`` in call
order, which is a ``cumsum`` over the row axis: the outer-dim scan runs one
thread per column and accumulates the rows in order, so it is bitwise the same
float64 sum (75 cases ``torch.equal``; the reversed row order and
``D.sum(0)`` both differ, so the tests have teeth).

The choice is the ``eels_scan_acc`` row of ``scatterem/simulation/_backends.py``
(CUDA only: the per-call launch is what it deletes).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor

from scatterem.simulation._backends import EELS_SCAN_ACC
from scatterem.utils.env import resolve_backend

__all__ = ["_OrderedScanAccumulator", "scan_accumulate_supported"]

# Upper bound (bytes) on the pending ``(calls + 1, *img.shape)`` matrix; above it
# the pending calls are flushed early, which is exact because a flush is just an
# ordered prefix of the same sum.
_EELS_SCAN_ACC_BLOCK_BYTES = 1 << 26
# A flush is ~5 launches (zeros, row copy, H2D, index_put_, cumsum), so with fewer
# pending calls than this it replays them as the ordered per-call index_add_
# instead: r7_eels/small (2 calls) read 0.978x deferred, dense (128 calls)
# 1.065-1.083x.
_EELS_SCAN_ACC_MIN_CALLS = 8

#: ``[cumsum flushes]``; lets a test prove the deferred path was TAKEN.
_SCAN_FLUSHES = [0]


def scan_accumulate_supported(img: Tensor) -> bool:
    """True if the image ``img`` may be accumulated by the ordered ``cumsum``."""
    return img.is_cuda


def _cat(ts: Sequence[Tensor] | Tensor) -> Tensor:
    """``torch.cat(ts)``, without a copy for a single tensor."""
    return ts[0] if len(ts) == 1 else torch.cat(list(ts))


class _OrderedScanAccumulator:
    """``img.index_add_(0, sel, val)`` per call, deferred and done in ONE pass.

    ``img`` is ``(P,)`` or, for qEELS, ``(P, n_q)``; each call adds ``val``
    (``(len(sel),)`` or ``(len(sel), n_q)``) at the duplicate-free keys ``sel``.
    """

    def __init__(self, img: Tensor):
        self.img = img
        self.deferred = resolve_backend(EELS_SCAN_ACC, None, img) == "scan"
        self._calls: list = []  # per add(): (sels, vals), replayed below the floor
        self._sel: list = []  # per add(): its calls' keys, concatenated
        self._val: list = []  # per add(): its calls' values, concatenated
        self._len: list = []  # flat, per call

    def add(
        self,
        sels: Sequence[Tensor],
        vals: Sequence[Tensor] | Tensor,
        sel_cat: Tensor | None = None,
        val_cat: Tensor | None = None,
    ) -> None:
        """``index_add_(0, sels[k], vals[k])`` in order; ``*_cat`` = their concat."""
        if not self.deferred:
            for s, v in zip(sels, vals, strict=True):
                self.img.index_add_(0, s, v)
            return
        lens = [int(s.numel()) for s in sels]
        row_bytes = self.img.numel() * self.img.element_size()
        if (len(self._len) + len(lens) + 1) * row_bytes > _EELS_SCAN_ACC_BLOCK_BYTES:
            self.flush()
        self._calls.append((sels, vals))
        self._sel.append(sel_cat if sel_cat is not None else _cat(sels))
        self._val.append(val_cat if val_cat is not None else _cat(vals))
        self._len.extend(lens)

    def flush(self) -> Tensor:
        """Apply every pending call, in call order; return the image."""
        if 0 < len(self._len) < _EELS_SCAN_ACC_MIN_CALLS:
            for sels, vals in self._calls:
                for s, v in zip(sels, vals, strict=True):
                    self.img.index_add_(0, s, v)
        elif self._len:
            img = self.img
            k = len(self._len)
            rows = np.repeat(np.arange(1, k + 1), self._len)
            D = torch.zeros((k + 1, *img.shape), dtype=img.dtype, device=img.device)
            D[0] = img
            D.index_put_(
                (
                    torch.as_tensor(rows, device=img.device),
                    torch.cat(self._sel) if len(self._sel) > 1 else self._sel[0],
                ),
                torch.cat(self._val) if len(self._val) > 1 else self._val[0],
            )
            self.img = torch.cumsum(D, 0)[-1]
            _SCAN_FLUSHES[0] += 1
        self._calls, self._sel, self._val, self._len = [], [], [], []
        return self.img
