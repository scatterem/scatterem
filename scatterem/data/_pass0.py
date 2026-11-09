"""Pass 0: the whole-tilt statistics of a source-backed dataset.

:class:`Pass0` accumulates per-pixel sums of the step-5 values and of their squares,
exactly in int64 when they are integers, else in float64. With ``row_width`` it also
sums each scan row in float64 as ``Dataset4DStem.__init__`` does, so the total and the
dose are bitwise the eager ones.
"""

from __future__ import annotations

import contextlib
import math
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from . import _serial_threads
from ._pattern_transform import _scalar

__all__ = ["ExactPart", "Pass0", "Pass0Stats", "add_blocks", "pass0_block_rows"]


def pass0_block_rows(
    shape: tuple[int, ...], row_chunk: int | None, block_rows: int, exact: bool
) -> int:
    """Scan rows per pass-0 read, in whole storage chunks (``row_chunk``).

    Exact sums do not depend on the blocks, so they read at most about 16 MiB of
    float32 values at a time: the read buffer and the step-5 copy then come back
    from the heap, while larger ones are mapped fresh and fault in every page.
    Float64 sums keep ``block_rows`` (their bits depend on it).
    """
    if exact:
        row_bytes = max(1, math.prod(shape[1:]) * 4)
        block_rows = min(block_rows, max(1, (16 << 20) // row_bytes))
    rc = row_chunk or 0
    if rc:  # read whole storage chunks
        block_rows = -(-max(block_rows, rc) // rc) * rc
    return block_rows


@dataclass(frozen=True)
class Pass0Stats:
    """The whole-cube statistics of one tilt.

    Attributes
    ----------
    c : Tensor
        0-dim float32 CPU normalisation constant (1.0 when not normalising).
    measurement_norm : float
        ``sum_pixels mean_rows (A / c)**2``.
    total : float
        The total of the step-5 values without the pre-divisor
        (``_total_intensity``).
    n_rows : int
        Patterns summed.
    mean_pattern : Tensor or None
        float32 CPU ``(ky, kx)`` mean of the step-5 values (None when the
        statistics were read from a store).
    mean_window : Tensor or None
        float32 CPU ``(ky, kx)`` mean of the step-5 values over the window
        (``Pass0``'s ``window``), or None.
    """

    c: Tensor
    measurement_norm: float
    total: float
    n_rows: int
    mean_pattern: Tensor | None
    mean_window: Tensor | None = None

    @property
    def dose_per_probe(self) -> float:
        """``total / n_rows``, the resident ``dose_per_probe_unnormalized``."""
        return self.total / max(self.n_rows, 1)


@dataclass(frozen=True)
class ExactPart:
    """One block's exact sums (:meth:`Pass0.part`), not yet added to the accumulator.

    ``row_ints`` are the int64 scan-row totals when they give the float64 bits;
    otherwise ``values`` holds the block's values for the per-row float64 recipe.
    """

    n: int
    integer: bool
    top: float
    sum: Tensor | None
    sumsq: Tensor | None
    window_sum: Tensor | None
    row_ints: list[int] | None
    values: Tensor | None


def _cpu(t: Tensor | None) -> Tensor | None:
    return None if t is None else t.cpu()


def _pieces(
    n: int, w: int | None, per: int, cap: int, split: bool
) -> list[tuple[int, int]]:
    """Pattern ranges of at most ``cap`` values: whole scan rows (at least one), or
    with ``split`` also parts of one scan row."""
    width = w or 1
    rows = cap // (width * per)
    if rows >= 1 or not split or w is None:
        step = width * max(1, rows) if w is not None else max(1, cap // per)
        return [(a, min(a + step, n)) for a in range(0, n, step)]
    step = max(1, cap // per)
    return [
        (a, min(a + step, r + w)) for r in range(0, n, w) for a in range(r, r + w, step)
    ]


_PIECE = 2**18  # values per worker piece: about 3 MB of float32 and int64 copies


def add_blocks(
    acc: Pass0,
    blocks: Sequence[tuple[int, int]],
    read: Callable[[int, int], tuple[Tensor, Tensor]],
    convert: Callable[[Tensor, Tensor], Tensor],
    dose: Callable[[Tensor, Tensor], Tensor] | None,
    cpu: bool,
    concurrent_reads: bool = False,
) -> None:
    """Add the blocks of scan rows ``[y0, y1)`` in order.

    ``read(y0, y1)`` gives a block's raw patterns and their global rows;
    ``convert(raw, rows)`` gives their step-5 values and ``dose(raw, rows)`` their
    dose values (or None).

    For exact sums of ``cpu`` blocks, worker threads read the blocks and make their
    sums, one piece at a time, on one OpenMP thread each (``_serial_threads``);
    this thread adds them in scan order. A busy core then slows one worker, not
    every op. Without ``concurrent_reads`` (the source's ``read`` is not known to be
    safe from several threads) the workers read one block at a time. The float64
    path stays as it was: its bits depend on the thread count, so it can still stall
    on a busy core.
    """
    workers = _serial_threads.workers() if cpu and acc.exact else 1
    if workers <= 1 or not blocks:
        for y0, y1 in blocks:
            raw, rows = read(y0, y1)
            acc.add(convert(raw, rows), None if dose is None else dose(raw, rows))
        return
    first, w, row0 = blocks[0][0], acc.row_width, acc._scan_row
    one_read = contextlib.nullcontext() if concurrent_reads else threading.Lock()

    def work(block: tuple[int, int]) -> ExactPart | tuple[Tensor, Tensor | None]:
        with one_read:
            raw, rows = read(*block)
        n = int(raw.shape[0])
        if raw.device.type != "cpu" or not n or (w and n % w):
            # add() handles (or refuses) it
            return convert(raw, rows), None if dose is None else dose(raw, rows)
        get = lambda a, b: convert(raw[a:b], rows[a:b])  # noqa: E731
        dget = None if dose is None else (lambda a, b, d=dose: d(raw[a:b], rows[a:b]))
        return acc.part(get, dget, scan_row=row0 + block[0] - first, n=n,
                        cap=_PIECE, split=True)  # fmt: skip

    with _serial_threads.serial_threads(workers) as pool:
        for done in _serial_threads.ordered_map(pool, work, blocks, 2 * workers):
            if isinstance(done, ExactPart):
                acc.commit(done)
            else:
                acc.add(*done)


class Pass0:
    """Per-pixel sums of the step-5 values and of their squares.

    Exact int64 sums when ``exact`` (integer values), else float64. Without
    ``row_width``, blocks may come in any order and split; :meth:`merge` adds
    another accumulator (an int64 merge is exact for any partition; a float64
    one is a rank-order fold).

    Parameters
    ----------
    det_shape : tuple of int
        ``(ky, kx)``.
    exact : bool
        Whether the values are integers (``RowTransform.integer_values``).
    row_width : int or None, optional
        Patterns per scan row. With it, every block must be whole scan rows in
        scan order, and the total is the eager constructor's per-row float64 sum.
    window : tuple of int or None, optional
        ``(rows, cols)``: also sum the first ``cols`` patterns of each of the first
        ``rows`` scan rows (tcDF's ``array[:rows, :cols]``). Needs ``row_width``.

    Raises
    ------
    ValueError
        A ``window`` without ``row_width``.
    """

    def __init__(
        self,
        det_shape: tuple[int, int],
        exact: bool,
        row_width: int | None = None,
        window: tuple[int, int] | None = None,
    ) -> None:
        dtype = torch.int64 if exact else torch.float64
        self.exact = exact
        self.row_width = None if row_width is None else int(row_width)
        self.sum = torch.zeros(det_shape, dtype=dtype)
        self.sumsq = torch.zeros(det_shape, dtype=dtype)
        self.row_total = 0.0
        self.n_rows = 0
        if window is not None and row_width is None:
            raise ValueError("a window needs row_width (it counts scan rows)")
        self.window = None if window is None else (int(window[0]), int(window[1]))
        self.window_sum = torch.zeros(det_shape, dtype=dtype)
        self._scan_row = 0
        self._local = threading.local()  # scratch buffers, one set per thread

    def add(self, x: Tensor, dose_values: Tensor | None = None) -> None:
        """Add a block of step-5 values ``(n, ky, kx)`` (any device).

        Parameters
        ----------
        x : Tensor
            The block's step-5 values.
        dose_values : Tensor or None, optional
            The block's values without the pre-divisor; the row totals (the
            dose) sum these. Needs ``row_width``.

        Raises
        ------
        ValueError
            With ``row_width``, a block that is not whole scan rows (the sums
            stay unchanged). On the exact path, a value that is not an integer,
            or squares that could overflow the int64 sums.
        """
        if self.row_width is not None and x.shape[0] % self.row_width:
            raise ValueError(
                f"a block of {x.shape[0]} patterns is not whole scan rows of "
                f"{self.row_width}"
            )
        if self.exact and x.numel():
            self.commit(self.part(x, dose_values, scan_row=self._scan_row))
            return
        y = x.to(torch.int64) if self.exact else x.to(torch.float64)
        self.sum += y.sum(0).cpu()
        self.sumsq += (y * y).sum(0).cpu()
        if self.row_width is not None:
            w = self.row_width
            rows = x if dose_values is None else dose_values
            for i in range(0, rows.shape[0], w):
                # Dataset4DStem.__init__'s recipe: one float64 sum per scan row on the
                # values' device, added to a Python float in row order.
                self.row_total += float(rows[i : i + w].to(torch.float64).sum())
                if self.window is not None and self._scan_row < self.window[0]:
                    part = x[i : i + w][: self.window[1]]  # step-5 values, with pre
                    part = (
                        part.to(torch.int64) if self.exact else part.to(torch.float64)
                    )
                    self.window_sum += part.sum(0).cpu()
                self._scan_row += 1
        self.n_rows += int(x.shape[0])

    def _buf(self, like: Tensor, dtype: torch.dtype | None = None) -> Tensor:
        """A scratch tensor of ``like``'s shape, reused across blocks (per thread).

        Fresh memory for every block costs a page fault per page, and the faults
        of a multithreaded first touch dominated pass 0.
        """
        bufs = self._local.__dict__.setdefault("bufs", {})
        key = (dtype or like.dtype, like.device)
        buf = bufs.get(key)
        if buf is None or buf.numel() < like.numel():
            buf = bufs[key] = torch.empty(
                like.numel(), dtype=key[0], device=like.device
            )
        return buf[: like.numel()].view(like.shape)

    def part(
        self,
        x: Tensor | Callable[[int, int], Tensor],
        dose_values: Tensor | Callable[[int, int], Tensor] | None = None,
        *,
        scan_row: int,
        n: int | None = None,
        cap: int = 2**21,
        split: bool = False,
    ) -> ExactPart:
        """The exact sums of a block; :meth:`commit` adds them.

        It reads only the settings, so worker threads can run it for several
        blocks at a time (``add_blocks``). Integer sums do not depend on the order,
        so each piece of at most ``cap`` values is checked and cast once to
        int64 for its pixel sums, the window, the scan-row totals and (in place)
        the squares. Pieces are whole scan rows; with ``split``, a scan row above
        ``cap`` is cut into pieces too. A float64 sum of a scan row of integers
        whose magnitudes sum to less than 2**53 is exact in any order, so the
        row's int64 sum has its bits.

        Parameters
        ----------
        x : Tensor or callable
            Integer step-5 values ``(n, ky, kx)``, whole scan rows; or
            ``x(a, b)``, the values of patterns ``a:b`` (made one piece at a time).
        dose_values : Tensor or callable or None, optional
            As :meth:`add`, or as ``x``.
        scan_row : int
            The scan row of the first pattern (the window counts from it).
        n : int, optional
            Patterns, when ``x`` is a callable.
        cap : int, optional
            Values per piece.
        split : bool, optional
            Whether a scan row above ``cap`` is cut into pieces.

        Returns
        -------
        ExactPart
            The sums, or ``integer=False`` (and no sums) for a value that is not
            an integer.
        """
        get = x if callable(x) else (lambda a, b, x=x: x[a:b])
        n = int(n if callable(x) else x.shape[0])  # pyright: ignore[reportArgumentType]
        w, top, wn = self.row_width, 0.0, self.window
        s = q = win = None
        ints = [0] * (n // w) if w is not None else []
        per = max(1, get(0, 1)[0].numel()) if n else 1  # values per pattern
        for a, b in _pieces(n, w, per, cap, split):
            p = get(a, b)
            if not torch.equal(p, torch.trunc(p, out=self._buf(p))):
                return ExactPart(n, False, 0.0, None, None, None, None, None)
            if p.is_floating_point():  # no |x| copy (aminmax has no uint16 kernel)
                lo, hi = torch.aminmax(p)
                top = max(top, -float(lo), float(hi))
            else:
                top = max(top, float(p.abs().max()))
            y = self._buf(p, torch.int64).copy_(p)
            if s is None or q is None:
                s, q = torch.zeros_like(y[0]), torch.zeros_like(y[0])
                win = None if wn is None else torch.zeros_like(y[0])
            if w is None:
                s += y.sum(0)
            elif a % w == 0 and (b - a) % w == 0:  # whole scan rows
                rows = y.view(-1, w, *y.shape[1:])
                per_row = rows.sum(1)  # (scan rows, ky, kx)
                s += per_row.sum(0)
                if win is not None and wn is not None:
                    k = min(max(wn[0] - scan_row - a // w, 0), rows.shape[0])
                    if k and wn[1] >= w:
                        win += per_row[:k].sum(0)
                    elif k:
                        win += rows[:k, : wn[1]].sum((0, 1))
                for j, t in enumerate(per_row.flatten(1).sum(1).tolist()):
                    ints[a // w + j] += t
            else:  # a piece of the scan row a // w
                ps = y.sum(0)
                s += ps
                cols = min(b - a, wn[1] - a % w) if wn is not None else 0
                if win is not None and wn is not None:
                    if scan_row + a // w < wn[0] and cols > 0:
                        win += y[:cols].sum(0)
                ints[a // w] += int(ps.sum())
            q += y.mul_(y).sum(0)
        by_ints = w is not None and dose_values is None and top * w * per < 2.0**53
        vals = None
        if w is not None and not by_ints:  # the per-row float64 recipe needs them
            dose = dose_values
            vals = get(0, n) if dose is None else dose(0, n) if callable(dose) else dose
        return ExactPart(n, True, top, _cpu(s), _cpu(q), _cpu(win),
                         ints if by_ints else None, vals)  # fmt: skip

    def commit(self, part: ExactPart) -> None:
        """Add a block's :meth:`part` (blocks in scan order).

        Raises
        ------
        ValueError
            A value that is not an integer, or squares that could overflow the
            int64 sums. The sums stay unchanged.
        """
        if not part.integer:
            raise ValueError("exact pass-0 sums need integer values")
        if part.top * part.top * (self.n_rows + part.n) >= 2.0**62:
            raise ValueError("int64 pass-0 sums would overflow; sum in float64")
        assert part.sum is not None and part.sumsq is not None
        self.sum += part.sum
        self.sumsq += part.sumsq
        if part.window_sum is not None:
            self.window_sum += part.window_sum
        w = self.row_width
        if w is not None:
            if part.row_ints is not None:
                for t in part.row_ints:
                    self.row_total += float(t)
            elif part.values is not None:  # Dataset4DStem.__init__'s per-row recipe
                vals = part.values
                for i in range(0, part.n, w):
                    self.row_total += float(vals[i : i + w].to(torch.float64).sum())
            self._scan_row += part.n // w
        self.n_rows += part.n

    def merge(self, other: Pass0) -> None:
        """Add ``other``'s sums (same ``exact`` and detector shape).

        Parameters
        ----------
        other : Pass0
            Another accumulator without ``row_width``.

        Raises
        ------
        ValueError
            Either accumulator has ``row_width`` (scan-row totals keep the
            scan's order), or the two differ in ``exact`` or detector shape.
        """
        if self.row_width is not None or other.row_width is not None:
            raise ValueError("scan-row totals do not merge (their order is the scan's)")
        if other.exact != self.exact or other.sum.shape != self.sum.shape:
            raise ValueError("cannot merge pass-0 sums of different kinds")
        self.sum += other.sum
        self.sumsq += other.sumsq
        self.n_rows += other.n_rows

    def finish(self, *, normalize: bool, c: Tensor | None = None) -> Pass0Stats:
        """The statistics; an injected ``c`` (D1a) replaces the computed one.

        Parameters
        ----------
        normalize : bool
            Whether the constant is the mean pattern's maximum (else 1.0).
        c : Tensor or None
            An injected 0-dim float32 constant.

        Returns
        -------
        Pass0Stats
            The tilt's statistics.

        Raises
        ------
        ValueError
            ``normalize`` without ``c`` and a mean pattern whose maximum is not
            positive.
        TypeError
            An injected ``c`` that is not a 0-dim float32 tensor.
        """
        n = max(self.n_rows, 1)
        mean = self.sum.to(torch.float64) / n
        if c is not None:
            _scalar(c, "c")
            c_out = c.detach().cpu()
        elif normalize:
            c_out = mean.max().to(torch.float32)
            if not float(c_out) > 0:
                raise ValueError(
                    "the mean pattern's maximum is not positive: nothing to "
                    "normalise by (an empty or all-zero tilt)"
                )
        else:
            c_out = torch.tensor(1.0, dtype=torch.float32)
        cf = float(c_out)
        norm = float((self.sumsq.to(torch.float64) / n).sum()) / (cf * cf)
        if self.row_width is not None:
            total = self.row_total
        else:
            total = float(self.sum.to(torch.float64).sum())
        mean_window = None
        if self.window is not None:
            cells = self.window[0] * self.window[1]
            mean_window = (self.window_sum.to(torch.float64) / cells).to(torch.float32)
        return Pass0Stats(
            c_out, norm, total, self.n_rows, mean.to(torch.float32), mean_window
        )
