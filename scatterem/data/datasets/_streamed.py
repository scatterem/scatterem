"""The streamed tier of a source-backed ``Dataset4DStem``.

This module holds:

- :func:`fit_pass0`: the whole-tilt statistics over the step-5 values (``c``,
  ``measurement_norm``, the total, the dose), read in blocks of whole scan rows;
- :class:`StreamedRows`: a dataset restricted to some global rows (a rank's owned rows);
- :func:`resolve_host_cache`: ``"auto"`` to ``"all"`` or ``"tilt"`` by the host memory,
  and :func:`serves_as_view`, the sources it takes as in RAM;
- :func:`stream_transform`, :func:`check_stream` and :func:`init_statistics`: the
  configuration and the statistics of a source-backed dataset;
- ``_Streamed``: the mixin that ``Dataset4DStem`` inherits through ``_SourceBacked``.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any

import torch
from torch import Tensor

from scatterem.data._host_cache import HOST_CACHE_MODES, _mem_available
from scatterem.data._pass0 import Pass0, Pass0Stats, add_blocks, pass0_block_rows
from scatterem.data._pattern_f16 import decode_scale
from scatterem.data._pattern_transform import (
    PoissonSpec,
    RowTransform,
    cast_only,
    check_stream_encoding,
    poisson_key,
)
from scatterem.data.storage import ArraySource, TensorSource
from scatterem.data.storage._protocol import native_device

__all__ = [
    "HOST_FRACTION",
    "StreamedRows",
    "check_stream",
    "fit_pass0",
    "init_statistics",
    "native_device",
    "resolve_host_cache",
    "serves_as_view",
    "stream_transform",
]

#: Fraction of ``MemAvailable`` (per local rank) the ``"all"`` cache may take
#: before ``"auto"`` picks ``"tilt"``.
HOST_FRACTION = 0.6


def fit_pass0(
    source: ArraySource,
    transform: RowTransform,
    *,
    normalize: bool,
    block_rows: int,
    c: Tensor | None = None,
) -> Pass0Stats:
    """The tilt's statistics over the step-5 values of every row.

    One read pass gives ``c``, ``measurement_norm``, the total and the dose. Each
    block is whole scan rows, so the total is summed one scan row at a time as
    ``Dataset4DStem.__init__`` sums it. The total and the dose leave out the
    transform's ``pre``: for a transform with ``pre``, steps 1-5 run a second time
    on the same raw block without it.

    Parameters
    ----------
    source : ArraySource
        The tilt's patterns, ``(sy, sx, ky, kx)``.
    transform : RowTransform
        Its conversion (``c`` unused); steps 1-5 run on its
        ``transform_device`` (the CPU when it is not set); where ``source`` lives
        if it is device-exact with exact int64 sums.
    normalize : bool
        Whether ``c`` is the mean pattern's maximum (else 1).
    block_rows : int
        Scan rows per read (fewer for exact sums), rounded up to whole storage
        chunks (``source.row_chunk``); see ``pass0_block_rows``.
    c : Tensor, optional
        An injected 0-dim float32 constant (design D1a).

    Returns
    -------
    Pass0Stats
        The statistics.

    Raises
    ------
    ValueError
        As :meth:`~scatterem.data._pass0.Pass0.add` and
        :meth:`~scatterem.data._pass0.Pass0.finish`.
    """
    sy, sx, ky, kx = source.shape
    # Exact int64 sums for counts of at most 16 bits; wider integers (int32, uint32,
    # int64, uint64) could overflow the squares, so they sum in float64.
    exact = transform.integer_values and transform.encoding.itemsize <= 2
    block_rows = pass0_block_rows(source.shape, source.row_chunk, block_rows, exact)
    full = sx <= sy  # the tcDF window (sy, min(sy, sx)) is the scan: no second sum
    acc = Pass0((ky, kx), exact=exact, row_width=sx, window=None if full else (sy, sy))
    tdev = transform.steps_device  # unset: the CPU (C1)
    if transform.device_exact and exact:
        tdev = native_device(source)  # int64 sums: no bit depends on the device
    transform = replace(transform.with_c(None), transform_device=tdev).placed(tdev)
    dose_tr = None if transform.pre is None else replace(transform, pre=None)

    def read(y0: int, y1: int) -> tuple[Tensor, Tensor]:
        raw = source.read((slice(y0, y1), slice(0, sx)), device=tdev)
        return raw.reshape(-1, ky, kx), torch.arange(y0 * sx, y1 * sx)

    # Keyed Poisson draws depend only on the rows: both calls draw the same counts.
    dose = None if dose_tr is None else dose_tr.stage1
    blocks = [(y0, min(y0 + block_rows, sy)) for y0 in range(0, sy, block_rows)]
    concurrent = bool(getattr(source, "concurrent_reads", False))
    add_blocks(
        acc, blocks, read, transform.stage1, dose, tdev.type == "cpu", concurrent
    )
    st = acc.finish(normalize=normalize, c=c)
    return replace(st, mean_window=st.mean_pattern) if full else st


class StreamedRows:
    """A dataset restricted to the sorted global ``rows``: ``self[local]`` is the
    float32 batch of ``rows[local]`` on ``device`` (a synchronous gather).

    Parameters
    ----------
    item : Any
        A streamed dataset of one tilt (it has ``gather(rows, device=...)``).
    rows : Tensor
        Sorted global rows, e.g. a rank's ``owned_rows[t]``.
    device : torch.device
        Where batches go.

    Raises
    ------
    ValueError
        ``rows`` is not sorted or has duplicates.
    """

    def __init__(self, item: Any, rows: Tensor, device: torch.device) -> None:
        r = rows.detach().to("cpu", torch.int64)
        if r.numel() > 1 and not bool((r[1:] > r[:-1]).all()):
            raise ValueError("restricted rows must be sorted without duplicates")
        self.item, self.rows, self.batch_device = item, r, torch.device(device)
        #: The host side of the data: never moved with the module.
        self.device = torch.device("cpu")

    def __len__(self) -> int:
        return int(self.rows.numel())

    def __getitem__(self, local: Tensor) -> Tensor:
        glob = self.rows[local.detach().to("cpu", torch.int64)]
        return self.item.gather(glob, device=self.batch_device)


def resolve_host_cache(
    mode: str, nbytes: int, *, in_ram: bool, allow_tilt: bool, local_ranks: int = 0
) -> str:
    """``"all"`` or ``"tilt"`` for a requested ``mode``.

    Parameters
    ----------
    mode : {"auto", "all", "tilt"}
        The items' request.
    nbytes : int
        Bytes the ``"all"`` cache would hold on this rank.
    in_ram : bool
        Whether the sources already sit in RAM (then ``"auto"`` is ``"all"``).
    allow_tilt : bool
        False on the single-GPU path, which refuses the disk tier.
    local_ranks : int, optional
        Ranks sharing this node's memory; 0 reads ``LOCAL_WORLD_SIZE``.

    Returns
    -------
    str
        ``"all"`` or ``"tilt"``.

    Raises
    ------
    ValueError
        An unknown mode, or ``"tilt"`` (requested or resolved) when not allowed.
    """
    if mode not in ("auto", *HOST_CACHE_MODES):
        raise ValueError(f"host_cache must be 'auto', 'all' or 'tilt', got {mode!r}")
    if mode == "auto":
        ranks = local_ranks or int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        budget = HOST_FRACTION * _mem_available() / max(ranks, 1)
        mode = "all" if in_ram or nbytes <= budget else "tilt"
        if mode == "tilt" and not allow_tilt:
            raise ValueError(
                f"the patterns ({nbytes / 1e9:.1f} GB) do not fit this rank's host "
                f"RAM budget ({budget / 1e9:.1f} GB), and a single-GPU fit refuses "
                "the disk tier (host_cache='tilt')"
            )
    if mode == "tilt" and not allow_tilt:
        raise ValueError(
            "host_cache='tilt' (the disk tier) is refused by a single-GPU fit"
        )
    return mode


def serves_as_view(source: ArraySource) -> bool:
    """Whether the host cache serves ``source`` as a view, with no copy.

    Only a C-contiguous CPU ``TensorSource`` qualifies (``HostCache``). Every other
    source, resident or not, is copied into the cache, so ``host_cache="auto"`` checks
    the RAM budget for it.

    Parameters
    ----------
    source : ArraySource
        A tilt's patterns.

    Returns
    -------
    bool
        True for a C-contiguous CPU ``TensorSource``.
    """
    return (
        isinstance(source, TensorSource)
        and source.tensor.device.type == "cpu"
        and source.tensor.is_contiguous()
    )


def stream_transform(
    encoding: torch.dtype,
    *,
    sampling: Any = None,
    transform_to_amplitudes: bool = False,
    fourier_shift_dim: tuple[int, ...] | None = None,
    clip_neg_values: bool = True,
    scale_exponent: int | None = None,
    dose: float | None = None,
    noise_seed: int | None = None,
    noise_tilt: int = 0,
    transform_device: torch.device | str = "cpu",
) -> RowTransform:
    """The ``RowTransform`` of a streamed tilt, from the resident keywords.

    Parameters
    ----------
    encoding : torch.dtype
        The stored dtype: uint8, uint16 or float16.
    sampling : array_like, optional
        The dataset's sampling; its scan step converts ``dose`` to electrons per pattern.
    transform_to_amplitudes, fourier_shift_dim, clip_neg_values
        As for ``Dataset4DStem.from_array``.
    scale_exponent : int, optional
        float16 only: the tilt's exponent ``e`` (intensity ``= h * 2**-e``).
    dose : float, optional
        float16 only: draw keyed Poisson counts at this dose (e/A^2).
    noise_seed : int, optional
        The seed of the draws; required with ``dose``.
    noise_tilt : int, optional
        The tilt key of the draws (the tilt's index in its store).
    transform_device : torch.device or str, optional
        Where steps 1-5 run; mirror the device the resident cube is built on.

    Returns
    -------
    RowTransform
        The transform, without the normalisation constant ``c``.

    Raises
    ------
    ValueError
        ``dose`` without ``noise_seed`` or without a ``sampling``.
    """
    poisson = None
    if dose is not None:
        if noise_seed is None:
            raise ValueError("dose needs a noise_seed (the draws are keyed by it)")
        if sampling is None:
            raise ValueError("dose needs the scan step: give a sampling")
        n_e = float(dose) * float(sampling[0]) * float(sampling[1])
        poisson = PoissonSpec(n_e, poisson_key(int(noise_seed), int(noise_tilt)))
    inv = None if scale_exponent is None else decode_scale(int(scale_exponent))
    return RowTransform(
        encoding, fourier_shift_dim=fourier_shift_dim, clip_neg_values=clip_neg_values,
        transform_to_amplitudes=transform_to_amplitudes, inv_scale=inv,
        poisson=poisson, transform_device=torch.device(transform_device),
    )  # fmt: skip


def check_stream(
    source: ArraySource, transform: RowTransform | None, host_cache: str
) -> None:
    """Refuse a streamed configuration a fit cannot serve.

    Parameters
    ----------
    source : ArraySource
        The dataset's patterns.
    transform : RowTransform or None
        The dataset's conversion.
    host_cache : str
        The requested host cache mode.

    Raises
    ------
    ValueError
        An unknown ``host_cache``, a dtype that is not a stream encoding, no
        transform, or a ``TensorSource`` that is not on the host.
    """
    if host_cache not in ("auto", *HOST_CACHE_MODES):
        raise ValueError(
            f"host_cache must be 'auto', 'all' or 'tilt', got {host_cache!r}"
        )
    check_stream_encoding(source.dtype)
    if transform is None:
        raise ValueError(
            "a streamed dataset needs a RowTransform (its batches are converted)"
        )
    if isinstance(source, TensorSource) and source.tensor.device.type != "cpu":
        raise ValueError(
            "a streamed dataset keeps its patterns on the host; pass a CPU source "
            "and the batch device as device="
        )


def init_statistics(
    ds: Any, *, normalize: bool, dose: float | None, pass0: Pass0Stats | None
) -> None:
    """Pass 0 of a new source-backed dataset, or the injected values.

    One read pass at most. A dose that is not known here is deferred: the first read of
    ``dose_per_probe_unnormalized`` runs ``_fill_deferred_dose`` (rule R-dose).

    Parameters
    ----------
    ds : Dataset4DStem
        The new dataset (``from_source`` has set its source and transform).
    normalize : bool
        Whether ``c`` is pass 0's mean-pattern maximum when the transform has none.
    dose : float or None
        The given dose per probe, or None.
    pass0 : Pass0Stats or None
        Statistics recorded earlier; then nothing is read.

    Raises
    ------
    ValueError
        ``normalize=True`` without a transform, or over a source before ``materialize``.
    """
    from ._blocks import _readable, _streamed_dose  # noqa: PLC0415 - imports us

    tr: RowTransform | None = ds._transform
    if normalize and tr is None:
        raise ValueError("normalize=True needs a RowTransform: it holds the constant c")
    needs_c = normalize and tr is not None and tr.c is None
    readable = _readable(ds._source)  # rule R-dose: no read before materialize
    if needs_c and not readable:
        raise ValueError("normalize=True runs pass 0, which reads the source: "
                         "materialize it first, or put c into the RowTransform")  # fmt: skip
    injected = not needs_c and ds._measurement_norm is not None
    stats, wanted = pass0, needs_c or (not injected and dose is None)
    if stats is None and tr is not None and readable and wanted:
        stats = fit_pass0(
            ds._source, tr, normalize=normalize, block_rows=ds._row_block, c=tr.c
        )
    if stats is not None:
        if needs_c:
            assert tr is not None
            # A resolved transform device keeps following the dataset.
            follows = ds._transform_device_follows
            ds._set_transform(tr.with_c(stats.c), follows=follows)
        if ds._measurement_norm is None:
            ds._measurement_norm = float(stats.measurement_norm)
    ds._stats = stats
    total = None
    if dose is None and stats is not None:
        # Pass 0's row totals leave out pre and c, so they give the dose in one pass.
        dose, total = stats.dose_per_probe, stats.total
    elif dose is None and tr is None and readable:
        dose = _streamed_dose(ds._source, None)  # a dataset without a transform
    if dose is None:
        # Deferred: the property runs _fill_deferred_dose on first read (R-dose).
        ds._dose, ds._total_intensity, ds._dose_deferred = None, None, True
        return
    n = ds._source.shape[0] * ds._source.shape[1]
    ds.dose_per_probe_unnormalized = float(dose)  # the setter clears _dose_deferred
    ds._total_intensity = float(dose) * n if total is None else total


class _Streamed:
    """Mixin for ``Dataset4DStem``: the streamed tier and the fit statistics (U1).

    A dataset streams when it was built with ``host_cache``
    (``from_source(..., host_cache="auto" | "all" | "tilt")``): the feeders then serve its
    patterns through a ring from host RAM or disk. The feed protocol
    (``_row_transform``, ``_base_transform``) is private.
    """

    host_cache: str | None = None
    _measurement_norm: float | None = None
    _stats: Pass0Stats | None = None

    @property
    def streams(self) -> bool:
        """Whether a fit streams the patterns (the dataset has a ``host_cache``).

        Returns
        -------
        bool
            True when the dataset was built with ``host_cache``.
        """
        return self.host_cache is not None

    @property
    def normalization_const(self: Any) -> float | None:
        """The constant ``c`` reads divide by; None on an eager dataset (already divided).

        Returns
        -------
        float or None
            ``c``, 1.0 for a source-backed dataset without ``c``, or None when eager.
        """
        if self._is_eager():
            return None
        tr = self._transform
        return 1.0 if tr is None or tr.c is None else float(tr.c)

    @property
    def measurement_norm(self: Any) -> float:
        """``sum_pixels mean_rows (A / c)**2`` of what reads return.

        An eager dataset computes it from its cube on every call; a source-backed one
        takes pass 0's value or the injected one (pattern streaming D1a).

        Returns
        -------
        float
            The measurement norm.
        """
        if self._is_eager():
            return float((self.array**2).mean(dim=(0, 1)).sum().item())
        if self._measurement_norm is None:
            tr = self._transform or cast_only(self._source.dtype)
            st = fit_pass0(
                self._source, tr, normalize=False, block_rows=self._row_block, c=tr.c
            )
            self._measurement_norm = float(st.measurement_norm)
        return self._measurement_norm

    def restrict(
        self: Any, rows: Tensor, device: torch.device | str | None = None
    ) -> StreamedRows:
        """This dataset restricted to the sorted global ``rows`` (no data copied).

        Parameters
        ----------
        rows : Tensor
            Sorted global flat rows, e.g. a rank's owned rows.
        device : torch.device or str, optional
            Where batches go; the dataset's device by default.

        Returns
        -------
        StreamedRows
            ``part[local]`` gathers ``rows[local]`` synchronously.
        """
        dev = self.device if device is None else torch.device(device)
        return StreamedRows(self, rows, dev)

    def _row_transform(self: Any, device: torch.device | str) -> RowTransform:
        """The transform with ``c``, placed for ``device`` (cached per device)."""
        return self._placed_for(torch.device(device))

    def _base_transform(self: Any) -> RowTransform:
        """The transform without ``c``, its scalars on ``transform_device``."""
        tr = self._transform
        if tr is None:
            raise ValueError("this dataset has no RowTransform")
        tr = tr.with_c(None)
        return tr.placed(tr.steps_device)

