"""How ``Dataset4DStem`` reaches its data: an ArraySource plus an optional RowTransform.

``dataset_4dstem.py`` is frozen size debt, so the source plumbing lives here as a mixin.
``_array`` stays the name every method of the class uses: on a ``TensorSource``
without a transform it IS the stored tensor, so the in-place work in ``__init__`` and
the eager methods are unchanged bit for bit.
Assigning ``_array`` (``Dataset.__init__``, the ``array`` setter, ``__init__``'s
dtype and fftshift steps) always builds a NEW ``TensorSource``, so a shallow copy of a
dataset (the runner's isolated views) never shares a later assignment.
"""

from dataclasses import replace

import numpy as np
import torch
from torch import Tensor

from scatterem.data._pass0 import Pass0Stats
from scatterem.data._pattern_transform import RowTransform
from scatterem.data.storage import ArraySource, TensorSource
from scatterem.data.storage._protocol import as_flat_index, check_index

from ._direct import _DirectStats
from ._streamed import _Streamed, check_stream, init_statistics, native_device

__all__ = ["_SourceBacked"]


class _SourceBacked(_DirectStats, _Streamed):
    """Mixin for ``Dataset4DStem``: storage, shape and batch reads from a source."""

    _source: ArraySource
    _transform: RowTransform | None = None
    # Whether the transform's device was resolved here (not set by the caller), so it
    # follows the dataset's device on a move.
    _transform_device_follows: bool = False
    _read_device: torch.device | None = None
    _setter_copied: bool = False
    host_cache: str | None = None
    _row_block: int = 8
    _dose: float | None = None
    _dose_deferred: bool = False
    _total_intensity: float | None

    # --- storage -----------------------------------------------------------
    @property
    def source(self) -> ArraySource:
        """Where the data lives."""
        return self._source

    @property
    def read_transform(self) -> RowTransform | None:
        """What reads apply to the stored values; ``None`` reads them unchanged."""
        return self._transform

    def _set_transform(
        self, transform: RowTransform | None, *, follows: bool = False
    ) -> None:
        self._transform = (
            None if transform is None or transform.is_identity else transform
        )
        self._transform_device_follows = follows and self._transform is not None
        self._placed_read: dict[torch.device, RowTransform] = {}
        self._placed_native: dict[tuple[torch.device, ...], RowTransform] = {}

    def _is_eager(self) -> bool:
        return isinstance(self._source, TensorSource) and self._transform is None

    def _placed_for(self, device: torch.device) -> RowTransform:
        """The transform placed for ``device``, cached so its scalars move once."""
        assert self._transform is not None
        cache = self.__dict__.setdefault("_placed_read", {})
        tr = cache.get(device)
        if tr is None:
            tr = cache[device] = self._transform.placed(device)
        return tr

    def _read_transform(self, device: torch.device) -> RowTransform:
        """The transform reads use: a device-exact one runs steps 1-5 at the source."""
        tr = self._placed_for(device)
        native = native_device(self._source)
        # stage1 checks the device type only, so any CUDA device index routes alike.
        if not tr.device_exact or tr.steps_device.type == native.type:
            return tr
        cache = self.__dict__.setdefault("_placed_native", {})
        key = (device, native)  # a source may be materialised or released later
        if key not in cache:  # placed() moves inv_scale and pre to `native`
            cache[key] = replace(tr, transform_device=native).placed(device)
        return cache[key]

    def _converted(self, raw: Tensor, rows: Tensor, device: torch.device) -> Tensor:
        """Steps 1-5 on ``raw`` (on ``_read_transform``'s device), step 6 on ``device``.

        ``stage1`` of a transform that is not the identity returns a new tensor, so
        the result never aliases ``raw`` (a ``TensorSource`` read may alias the store).
        """
        tr = self._read_transform(device)
        frame = raw.shape[-2:]
        x = tr.stage1(raw.reshape(-1, *frame), rows)
        return tr.stage2(x.to(device)).reshape(*raw.shape[:-2], *frame)

    @property
    def _array(self) -> Tensor:
        if self._is_eager():
            return self._source.tensor  # pyright: ignore[reportAttributeAccessIssue]
        if self._source.is_resident:
            dev = self._data_device()
            if self._transform is None:
                # Always an owned copy: a TensorSource (or a crop of one, or a
                # materialised codec source) may alias its stored tensor.
                return self._source.read(device=dev).clone()
            raw = self._source.read(device=self._read_transform(dev).steps_device)
            # stage1 casts to a new tensor (B1 copies float32 input), so this is owned.
            return self._converted(raw, _block_rows(None, self._source.shape), dev)
        raise RuntimeError(
            f"{type(self).__name__} keeps its data in a {type(self._source).__name__} "
            "that is not resident; whole-cube access needs "
            "dataset.source.materialize(device) first. Batches (ds[idx], gather) and "
            "iter_scan_blocks() work without it."
        )

    @_array.setter
    def _array(self, value: Tensor) -> None:
        stored = value.contiguous()
        # A non-contiguous value is copied here. The copy is private, so __init__
        # does not need to clone it again before its in-place steps.
        self._setter_copied = stored is not value
        self._source = TensorSource(stored)
        self._set_transform(None)

    @property
    def _array3d(self) -> Tensor:
        a = self._array
        return a.reshape(a.shape[0] * a.shape[1], *a.shape[2:])

    def __getstate__(self) -> dict:
        """The pickled state without the placed transforms (may hold CUDA tensors)."""
        state = dict(self.__dict__)
        state["_placed_read"], state["_placed_native"] = {}, {}
        return state

    def __setstate__(self, state: dict) -> None:
        """Rebuild from a pickle; one from before the storage layer holds ``_array``.

        Such a pickle (a legacy store's pickled ``items``)
        carries ``_array`` and the flattened ``_array3d`` in ``__dict__``. Both
        properties shadow the instance dict, so the state is routed through the
        ``_array`` setter and the derived view is dropped. The placed transforms
        always start empty; the first read places them again. A pickle from before
        S4 holds the dose under its public name; it moves to ``_dose``.
        """
        state = dict(state)
        state.pop("_array3d", None)
        state["_placed_read"], state["_placed_native"] = {}, {}
        if "dose_per_probe_unnormalized" in state:  # a pickle from before S4
            state["_dose"] = state.pop("dose_per_probe_unnormalized")
        stored = state.pop("_array", None)
        self.__dict__.update(state)
        if stored is not None and "_source" not in state:
            self._array = stored

    # --- shape and device --------------------------------------------------
    @property
    def shape(self) -> torch.Size:
        """Shape of the data, ``(scan_y, scan_x, k_y, k_x)``."""
        return torch.Size(self._source.shape)

    @property
    def ndim(self) -> int:
        """Number of dimensions (4)."""
        return len(self._source.shape)

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of what reads return."""
        return self._source.dtype if self._transform is None else torch.float32

    def _data_device(self) -> torch.device:
        if isinstance(self._source, TensorSource) and self.host_cache is None:
            return self._source.tensor.device
        return self._read_device or torch.device("cpu")

    def _move_to(self, device: torch.device | str) -> None:
        """Move resident tensor storage; otherwise set where reads and batches go.

        A streamed dataset keeps its source on the host.

        A transform device that ``from_source`` resolved follows the move; one the
        caller set stays.
        """
        if isinstance(self._source, TensorSource) and self.host_cache is None:
            self._source = TensorSource(self._source.tensor.to(device).contiguous())
        else:
            self._read_device = torch.device(device)
        if self._transform_device_follows and self._transform is not None:
            moved = replace(self._transform, transform_device=self._data_device())
            self._set_transform(moved, follows=True)

    # --- reads -------------------------------------------------------------
    def gather(
        self, flat_points, *, device: torch.device | str | None = None
    ) -> Tensor:
        """Patterns at flat scan points ``iy * nx + ix``, transformed.

        Parameters
        ----------
        flat_points : list of int, numpy.ndarray, or Tensor
            Flat scan indices; negative values count from the end.
        device : torch.device or str, optional
            Where the batch goes; the dataset's device by default.

        Returns
        -------
        Tensor
            ``(len(flat_points), k_y, k_x)``.
        """
        dev = torch.device(self._data_device() if device is None else device)
        if self._transform is None:
            return self._source.gather(flat_points, device=dev)
        rows = as_flat_index(flat_points, self._source.shape[0] * self._source.shape[1])
        raw = self._source.gather(rows, device=self._read_transform(dev).steps_device)
        return self._converted(raw, rows, dev)

    def read(self, index=None, *, device: torch.device | str | None = None) -> Tensor:
        """The block ``index`` (one slice per scan dimension), transformed.

        Keyed Poisson draws see the block's global flat rows.

        Parameters
        ----------
        index : tuple of slice, optional
            One slice per scan dimension; the whole scan by default.
        device : torch.device or str, optional
            Where the block goes; the dataset's device by default.
        """
        dev = torch.device(self._data_device() if device is None else device)
        if self._transform is None:
            return self._source.read(index, device=dev)
        raw = self._source.read(index, device=self._read_transform(dev).steps_device)
        return self._converted(raw, _block_rows(index, self._source.shape), dev)

    # --- dose --------------------------------------------------------------
    @property
    def dose_per_probe_unnormalized(self) -> float | None:
        """Mean pattern sum per scan point, before normalisation.

        A dataset built without a dose and without a pass 0 defers it, and so does a
        source that needs ``materialize``. The first read (after ``materialize``) runs
        one dose pass and records the dose and the total.

        Returns
        -------
        float or None
            The dose; ``None`` while it is deferred and the source is not readable.
        """
        if self._dose_deferred and _readable(self._source):
            self._fill_deferred_dose()
        return self._dose

    @dose_per_probe_unnormalized.setter
    def dose_per_probe_unnormalized(self, value: float | None) -> None:
        self._dose = value
        self._dose_deferred = False

    def _fill_deferred_dose(self) -> None:
        """Run the deferred dose pass once: steps 1-5 without ``pre`` or ``c``."""
        if not _readable(self._source):
            raise RuntimeError(
                "the dose of this dataset is computed on first use; its "
                f"{type(self._source).__name__} needs "
                "dataset.source.materialize(device) first"
            )
        if self.__dict__.get("_frame_view"):  # the eager crop's recipe (_DirectStats)
            return self._fill_frame_dose()
        dose = _streamed_dose(self._source, self._transform)
        n = self._source.shape[0] * self._source.shape[1]
        self._dose, self._total_intensity = dose, dose * n
        self._dose_deferred = False

    def _lazy_total(self) -> float:
        """The total for ``total_intensity`` when none is recorded.

        A deferred dose runs its pass now (one read pass) and gives the
        pre-normalisation total. Any other dataset sums its reads, as before S4.
        """
        if self._dose_deferred:
            self._fill_deferred_dose()
            assert self._total_intensity is not None
            return self._total_intensity
        return sum(
            float(b.to(torch.float64).sum()) for _, _, b in self.iter_scan_blocks(1)
        )

    @classmethod
    def from_source(
        cls,
        source: ArraySource,
        meta=None,
        *,
        transform: RowTransform | None = None,
        name: str | None = None,
        origin=None,
        sampling=None,
        units=None,
        signal_units: str = "arb. units",
        device: torch.device | str | None = None,
        dose_per_probe_unnormalized: float | None = None,
        normalize: bool = False,
        measurement_norm: float | None = None,
        host_cache: str | None = None,
        row_block: int = 8,
        _pass0: Pass0Stats | None = None,
    ):
        """A dataset over ``source``; nothing is read but pass 0 (if needed).

        The stored values are taken as they are unless ``transform`` is given. A
        ``RowTransform`` with the eager constant ``c`` gives batches bitwise equal to
        ``from_array`` of the same raw cube, when its ``transform_device`` is of the
        device type the eager cube was built on (CPU and CUDA ``sqrt`` differ in the
        last bit). An unset ``transform_device`` becomes the dataset's device, so
        steps 1-5 run where the data is, and it follows a later ``ds.device = ...``.

        Pass 0 runs at construction when ``c`` is needed, or when neither
        ``measurement_norm`` nor the dose is given. It reads every scan row once and
        gives ``c``, the measurement norm and the dose, which is bitwise the eager
        value. With ``c`` and ``measurement_norm`` injected, nothing is read; the
        dose is then computed on first use (``total_intensity``,
        ``fluence_per_probe``).
        A source that needs ``materialize`` and is not resident is not read: its dose
        is computed on first use, after ``materialize``.

        Parameters
        ----------
        source : ArraySource
            A 4-D raster source (``scan_ndim == 2``).
        meta : AcquisitionMetadata, optional
            Acquisition metadata; sampling and units default from it.
        transform : RowTransform, optional
            Applied to every read: steps 1-5 on its ``transform_device``, the
            division by its ``c`` on the batch device. None reads the stored
            values as they are. An unset ``transform_device`` (None) becomes the
            dataset's device; a set one is kept, also on a move. A transform
            without ``sqrt`` and without Poisson draws runs steps 1-5 where the
            source lives (the same bits on every device); only the division by
            ``c`` runs on the batch device. A synchronous gather of such a dataset
            then copies float32 to the batch device.
        name, origin, sampling, units, signal_units
            As in :meth:`from_array`.
        device : torch.device or str, optional
            Where reads go; a ``TensorSource`` is materialised there unless the
            dataset streams.
        dose_per_probe_unnormalized : float, optional
            The dose per probe position; from pass 0, or computed on first use,
            when omitted.
        normalize : bool, default False
            Whether the constant ``c`` is pass 0's mean-pattern maximum when the
            transform has none. Needs a transform.
        measurement_norm : float, optional
            An injected measurement norm (D1a); with an injected ``c`` nothing is
            read at construction.
        host_cache : {"auto", "all", "tilt"}, optional
            Streams the dataset in a fit (``streams`` is then True): every row in
            host RAM (``"all"``), the served and the next tilt (``"tilt"``,
            decomposed fits only), or chosen by the host memory (``"auto"``). The
            source stays on the host; ``device`` is where batches go. None keeps
            the resident feeders.
        row_block : int, default 8
            Scan rows per read of pass 0 and ``iter_scan_blocks``.
        _pass0 : Pass0Stats, optional
            Statistics recorded earlier (a pattern store's).

        Returns
        -------
        Dataset4DStem
            The new dataset.

        Raises
        ------
        ValueError
            If ``source`` is not a 4-D raster source, if the encoding of
            ``transform`` is not the dtype of ``source``, for a streamed
            configuration ``check_stream`` refuses, or for ``normalize=True``
            without a transform.
        """
        if len(source.shape) != 4 or source.scan_ndim != 2:
            raise ValueError(
                f"Dataset4DStem needs a 4-D raster source, got shape {source.shape} "
                f"with scan_ndim {source.scan_ndim}"
            )
        if transform is not None and transform.encoding != source.dtype:
            raise ValueError(
                f"the transform's encoding {transform.encoding} is not the source's "
                f"dtype {source.dtype}"
            )
        if host_cache is not None:
            check_stream(source, transform, host_cache)
        obj = cls.__new__(cls)
        obj.host_cache = host_cache
        obj._row_block = int(row_block)
        obj._measurement_norm = (
            None if measurement_norm is None else float(measurement_norm)
        )
        _init_from_source(
            obj, source, meta, transform, device=device,
            labels=(name, origin, sampling, units, signal_units),
        )  # fmt: skip
        init_statistics(
            obj, normalize=normalize, dose=dose_per_probe_unnormalized, pass0=_pass0
        )
        return obj


def _block_rows(index, shape: tuple[int, ...]) -> Tensor:
    """Global flat rows ``iy * nx + ix`` of a raster block, in read order."""
    ys, xs = check_index(index, shape, 2)
    nx = shape[1]
    y = torch.arange(ys.start, ys.stop, dtype=torch.int64)
    x = torch.arange(xs.start, xs.stop, dtype=torch.int64)
    return (y[:, None] * nx + x[None, :]).reshape(-1)


def _readable(source: ArraySource) -> bool:
    """Whether a read works now. A codec blob reads only after ``materialize``.

    An attribute, not an ``isinstance`` test, so this module imports no codec
    class.
    """
    return source.is_resident or not getattr(source, "reads_need_materialize", False)


def _streamed_dose(source: ArraySource, transform: RowTransform | None) -> float:
    """Mean pattern sum per scan point after steps 1-5, without ``pre`` or ``c``.

    One scan row at a time in float64, as ``Dataset4DStem.__init__`` sums, so it agrees
    bit for bit with the eager value of the same data built on the same device type.
    """
    ny, nx = source.shape[:2]
    tr = None
    dev = torch.device("cpu")
    if transform is not None:
        dev = transform.steps_device
        tr = replace(transform, pre=None, c=None).placed(dev)
    total = 0.0
    for row in range(ny):
        block = source.read((slice(row, row + 1), slice(0, nx)), device=dev)[0]
        if tr is None:
            x = block.to(torch.float32)
        else:
            rows = torch.arange(row * nx, (row + 1) * nx, dtype=torch.int64)
            x = tr.stage1(block, rows)
        total += float(x.to(torch.float64).sum())
    return total / (ny * nx)


def _init_from_source(
    obj, source, meta, transform: RowTransform | None, *, device, labels: tuple
) -> None:
    """Set everything ``Dataset4DStem.__init__`` sets, without reading the data."""
    from ._meta import meta_sampling
    from .base import _place_aberrations_on_device

    name, origin, sampling, units, signal_units = labels
    obj._source = source
    obj._read_device = None if device is None else torch.device(device)
    if (
        device is not None
        and isinstance(source, TensorSource)
        and obj.host_cache is None
    ):
        source.materialize(device)
    follows = False
    if transform is not None and transform.transform_device is None:
        follows = True  # steps 1-5 of the pattern streaming on the dataset's device
        transform = replace(transform, transform_device=obj._data_device())
    obj._set_transform(transform, follows=follows)
    obj._shape = torch.Size(source.shape)
    obj.meta = meta
    obj.name = name or "4d dataset"
    obj.origin = np.zeros(4) if origin is None else origin
    s = meta_sampling(meta) if sampling is None else sampling
    obj.sampling = np.ones(4) if s is None else s
    u = getattr(meta, "units", None) if units is None else units
    obj.units = ["pixels"] * 4 if u is None else u
    obj.signal_units = signal_units
    obj.transform_to_amplitudes = bool(
        transform is not None and transform.transform_to_amplitudes
    )
    obj.fourier_shift_dim = None if transform is None else transform.fourier_shift_dim
    obj.probe_index = 0
    obj.bright_field_mask_threshold = 0.3
    if meta is not None:
        _place_aberrations_on_device(meta, obj._data_device())
