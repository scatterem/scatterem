"""The direct paths of a source-backed ``Dataset4DStem``.

``dataset_4dstem.py`` is frozen size debt, so the streaming branches live here:
:class:`_DirectStats` (the pass-0 statistics of the direct paths) and
the block-wise detector group means, the averaged pattern, the frame views, the
bright-field crop, the streaming vBF and ``from_array(..., out_of_core=True)``.
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
from torch import Tensor

from scatterem.data._pass0 import Pass0Stats
from scatterem.data._pattern_transform import STREAM_ENCODINGS, RowTransform
from scatterem.data.storage import ArraySource, MemmapSource, TensorSource

# Through the module, never a bound name: a test's
# monkeypatch.setattr(_streamed, "fit_pass0", ...) must reach _direct_stats too.
from . import _streamed
from ._meta import copy_meta, moved_warp_centre
from ._parent_link import VBF_CACHE_ATTRS

__all__ = [
    "AsFloat32",
    "_DirectStats",
    "averaged_pattern",
    "build_vbf",
    "crop_brightfield",
    "detector_group_means",
    "frame_child",
    "frame_view",
    "mean_cbed_tcdf",
    "out_of_core_dataset",
    "reads_in_blocks",
]


class _DirectStats:
    """Mixin for ``Dataset4DStem``: the block reads and statistics of the direct paths."""

    #: Where the streaming vBF keeps G (``DatasetVirtualBrightFieldStreaming``).
    g_residency: str = "auto"
    #: Set by ``frame_view``: the deferred dose is the crop's (``_fill_frame_dose``).
    _frame_view: bool = False

    def iter_scan_blocks(
        self: Any,
        block_rows: int | None = None,
        *,
        device: torch.device | str | None = None,
    ) -> Iterator[tuple[int, int, Tensor]]:
        """Yield ``(row_start, row_end, block)``: transformed blocks of whole scan rows.

        Parameters
        ----------
        block_rows : int, optional
            Scan rows per block; the dataset's ``row_block`` (8) by default.
        device : torch.device or str, optional
            Where the blocks go; the dataset's device by default. A numpy consumer asks
            for ``"cpu"`` and calls ``.numpy()``.

        Yields
        ------
        tuple of (int, int, Tensor)
            ``(row_start, row_end, block)``, ``block`` of shape
            ``(row_end - row_start, scan_x, k_y, k_x)``, as ``read`` returns it.
        """
        br = int(block_rows or self._row_block)
        ny, nx = self._source.shape[:2]
        for s in range(0, ny, br):
            e = min(s + br, ny)
            yield s, e, self.read((slice(s, e), slice(0, nx)), device=device)

    def _direct_stats(self: Any) -> Pass0Stats:
        """Pass 0 with the mean pattern and the tcDF window (run once, then kept)."""
        st = self._stats
        if st is None or st.mean_pattern is None or st.mean_window is None:
            tr = self._transform or RowTransform(
                self._source.dtype, clip_neg_values=False
            )
            st = _streamed.fit_pass0(self._source, tr, normalize=False,
                                     block_rows=self._row_block, c=tr.c)  # fmt: skip
            self._stats = st
        return st

    def _fill_frame_dose(self: Any) -> None:
        """A frame view's dose: the eager crop's sum of the cropped reads (``c`` divided).

        ``Dataset4DStem.__init__`` sums each scan row in float64 into a Python float,
        in row order, on the cube's device; this reads the same values the same way.
        """
        total = 0.0
        for _, _, block in self.iter_scan_blocks(device=self.device):
            for row in block:
                total += float(row.to(torch.float64).sum())
        self._dose = total / (self.shape[0] * self.shape[1])
        self._total_intensity, self._dose_deferred = total, False

    @property
    def mean_diffraction_pattern(self: Any) -> Tensor:
        """Mean pattern over the scan, as reads return it (``c`` divided).

        Returns
        -------
        Tensor
            ``(k_y, k_x)``: ``array.mean((0, 1))`` on an eager dataset; pass 0's float64
            mean, cast to float32 and divided by ``c``, on a source-backed one (CPU).
        """
        if self._is_eager():
            return self.array.mean((0, 1))
        return self._direct_stats().mean_pattern / self.normalization_const

    @property
    def sum_diffraction_pattern(self: Any) -> Tensor:
        """Sum pattern over the scan: ``mean_diffraction_pattern`` times the scan size.

        Returns
        -------
        Tensor
            ``(k_y, k_x)``.
        """
        return self.mean_diffraction_pattern * (self.shape[0] * self.shape[1])


def reads_in_blocks(ds: Any) -> bool:
    """Whether the direct paths read ``ds`` block by block.

    True for every dataset whose reads convert (a source with a ``RowTransform``, or a
    source that is not a ``TensorSource``), resident or not; such a dataset never
    converts its whole cube. False for an eager dataset, which uses ``.array``.
    """
    return not ds._is_eager()


def mean_cbed_tcdf(ds: Any) -> Tensor:
    """tcDF's ``array[:n, :n].mean((0, 1))``, n = scan rows (numpy clamps the columns)."""
    if reads_in_blocks(ds):
        return ds._direct_stats().mean_window / ds.normalization_const
    n = ds.array.shape[0]
    return ds.array[:n, :n].mean((0, 1))


def detector_group_means(ds: Any, groups: list[Tensor]) -> list[Tensor]:
    """Per-group detector-pixel means; one sequential pass when ``ds`` reads in blocks.

    One read pass is fanned out to every group (a per-group gather would read the
    source once per group); only the ``(ny, nx)`` group images go to the device. An
    eager dataset keeps the contraction and the per-group gather as before.
    """
    if not reads_in_blocks(ds):
        # A lazy import: dataset_4dstem imports this module.
        from ._virtual_detectors import (
            _detector_group_means_contracted,  # noqa: PLC0415
        )

        contracted = _detector_group_means_contracted(ds.array, groups)
        if contracted is not None:
            return contracted
        return [ds.array[:, :, g[:, 0], g[:, 1]].mean(-1) for g in groups]
    lens = [int(g.shape[0]) for g in groups]
    offs = np.concatenate(([0], np.cumsum(lens)))
    all_inds = torch.cat([g.cpu() for g in groups]).numpy()
    iy, ix = all_inds[:, 0], all_inds[:, 1]
    ny, nx = ds.shape[:2]
    out = np.empty((len(groups), ny, nx), dtype=np.float32)
    for s, e, block in ds.iter_scan_blocks(device="cpu"):
        cols = block.numpy()[:, :, iy, ix]
        for j, m in enumerate(lens):
            if m == 0:
                out[j, s:e] = np.nan  # eager parity: mean of an empty gather
            else:
                out[j, s:e] = cols[:, :, offs[j] : offs[j + 1]].mean(-1)
    return [torch.from_numpy(img).to(ds.device) for img in out]


def averaged_pattern(ds: Any) -> Tensor:
    """``_array[:50, :50].mean((0, 1))``; read as one block when ``ds`` reads in blocks."""
    if not reads_in_blocks(ds):
        return ds._array[:50, :50].mean((0, 1))  # slicing clamps to the scan
    ny, nx = ds.shape[:2]
    slab = ds.read((slice(0, min(ny, 50)), slice(0, min(nx, 50))), device="cpu")
    return slab.to(ds.device).mean((0, 1))


#: The caches a frame view must not share with its parent: the placed transforms of
#: ``_SourceBacked`` (``_placed_read``, ``_placed_native``) and the cached vBF.
_DROP = ("_placed_read", "_placed_native", *VBF_CACHE_ATTRS)


def _refuse_fftshifted(ds: Any) -> None:
    tr = ds.read_transform
    if tr is not None and tr.fourier_shift_dim is not None:
        raise ValueError(
            "a detector crop of fftshifted reads is refused: a crop of the stored "
            "frames is not a crop of the shifted ones; crop before the fftshift"
        )


def frame_view(ds: Any, frame: tuple[slice, slice]) -> Any:
    """``ds`` over a detector crop of its source (a lazy frame view).

    The view keeps the transform, and gets its own ``meta`` (with the crop's shape, as
    ``crop`` gives it) and its own ``origin`` (shifted by the frame start). It has no
    statistics and no cached vBF. Its dose and total are the crop's, computed on first
    use as the eager crop computes them (``_DirectStats._fill_frame_dose``).

    Raises
    ------
    ValueError
        The reads are fftshifted: a crop of the stored frames is not a crop of the
        shifted ones.
    """
    _refuse_fftshifted(ds)
    ny, nx = ds.shape[:2]
    child = copy.copy(ds)  # __getstate__/__setstate__ reset the placed transforms
    for key in _DROP:
        child.__dict__.pop(key, None)
    # The dose is the crop's, as the eager crop computes it, on first use.
    child._frame_view = True
    child._dose, child._total_intensity, child._dose_deferred = None, None, True
    child._source = ds.source.view((slice(0, ny), slice(0, nx)), frame=frame)
    child._shape = torch.Size(child._source.shape)
    child._stats = None
    child._measurement_norm = None
    shift = (0, 0, frame[0].start or 0, frame[1].start or 0)
    child.meta = copy_meta(
        ds.meta,
        shape=child._shape,
        warp_centre=moved_warp_centre(ds.meta, (-shift[2], -shift[3])),
    )
    child.origin = np.asarray(ds.origin, dtype=float) - np.asarray(shift)
    return child


def frame_child(ds: Any, frame: tuple[slice, slice]) -> Any:
    """A frame view that keeps the parent's ``c`` and its cropped means.

    ``measurement_norm`` is not cropped: the child computes it on first use.
    """
    st = ds._direct_stats()
    child = frame_view(ds, frame)
    child._stats = dataclasses.replace(
        st, mean_pattern=st.mean_pattern[frame], mean_window=st.mean_window[frame],
        measurement_norm=float("nan"),  # not used: child._measurement_norm is None
    )  # fmt: skip
    return child


def crop_brightfield(ds: Any, estimator: str, normalize: bool = True,
                     clip_neg_values: bool = True, *,
                     edge_reach: float | None = None) -> Any:  # fmt: skip
    """``Dataset4DStem.crop_brightfield_``: both branches.

    An eager dataset crops its cube with ``crop_`` (``normalize`` and
    ``clip_neg_values`` are forwarded), as before. A dataset that reads in blocks
    gets a lazy detector crop: the child keeps the parent's ``c`` (the eager path crops
    the already-normalised cube), its dose is the crop's (as the eager crop's), and its
    mean patterns are the parent's, cropped; ``normalize`` and ``clip_neg_values`` are inert there. The
    origin moves as in the eager crop.

    Raises
    ------
    ValueError
        A dataset that reads in blocks with fftshifted reads, or (on both branches) a
        bright-field box that starts outside the detector; before the parent changes.
    """
    blocks = reads_in_blocks(ds)
    if blocks:
        _refuse_fftshifted(ds)  # first: the parent must stay unchanged
    box = ds._bright_field_crop_box(estimator, edge_reach)
    if box.crop_slice[-2].start < 0 or box.crop_slice[-1].start < 0:
        raise ValueError(  # before the parent changes; a slice would wrap or be empty
            f"the bright-field box {box.crop_slice[-2:]} starts outside the detector: "
            "the disk is too close to its edge for a centred crop"
        )
    ds._radius_bright_field = box.radius_ceil
    ds.origin[-1] = box.center[0]  # the center of the bright-field region
    ds.origin[-2] = box.center[1]
    if blocks:
        det = (box.crop_slice[-2], box.crop_slice[-1])
        child = frame_child(ds, det)
        child.name = ds.name + str(box.crop_slice)  # as crop names it
        child._crop_sl = det
    else:
        child = ds.crop_(
            box.crop_slice, normalize=normalize, clip_neg_values=clip_neg_values
        )
    child._radius_bright_field = box.radius
    if getattr(child, "origin", None) is not None:  # into the cropped coordinates
        origin = list(child.origin)
        origin[-2] = box.center[0] - (box.y0 - box.radius)
        origin[-1] = box.center[1] - (box.x0 - box.radius)
        child.origin = tuple(origin)
    child.is_bright_field = True
    return child


class AsFloat32:
    """An ``ArraySource`` that reads ``inner`` and casts every block to float32.

    The cast is the eager constructor's (``array.to(torch.float32)``, element by element),
    so the values are bitwise the eager cube's before its clamp. It serves
    the dtypes outside ``_DIRECT_ENCODINGS`` (uint32, int16, int32, float64, ...): pass 0
    then sums their float32 values in float64, not in int64.

    Parameters
    ----------
    inner : ArraySource
        The stored cube.
    """

    def __init__(self, inner: ArraySource) -> None:
        self.inner = inner

    shape = property(lambda self: self.inner.shape)
    scan_ndim = property(lambda self: self.inner.scan_ndim)
    row_chunk = property(lambda self: self.inner.row_chunk)
    is_resident = property(lambda self: self.inner.is_resident)
    dtype = property(lambda self: torch.float32)

    def read(self, index: Any = None, *, device: torch.device | str) -> Tensor:
        """The block ``index`` as float32 on ``device`` (a new tensor)."""
        return self.inner.read(index, device=device).to(torch.float32)

    def gather(self, flat_points: Any, *, device: torch.device | str) -> Tensor:
        """The patterns at ``flat_points`` as float32 on ``device``."""
        return self.inner.gather(flat_points, device=device).to(torch.float32)

    def view(self, index: Any, *, frame: Any = None) -> AsFloat32:
        """A view of ``inner``, still cast."""
        return AsFloat32(self.inner.view(index, frame=frame))

    def materialize(self, device: torch.device | str) -> None:
        """Materialise ``inner``."""
        self.inner.materialize(device)

    def release(self) -> None:
        """Release ``inner``."""
        self.inner.release()


#: The dtypes the direct tier reads as they are stored; ``AsFloat32`` casts the others
#: (see :class:`AsFloat32`). Not ``ENCODINGS``: that holds every eager dtype.
_DIRECT_ENCODINGS = (*STREAM_ENCODINGS, torch.float32)


def _cube_source(array: Any) -> Any:
    """The source of ``from_array(..., out_of_core=True)``; the cube is never copied.

    A cube that streams (``host_cache="auto"``) and is a writable, C-contiguous in-RAM
    array becomes a ``TensorSource`` over the same memory, so the host cache serves it
    as a view. Any other cube becomes a ``MemmapSource``: a ``TensorSource`` without a
    host cache would be materialised on ``device`` by ``_init_from_source``.
    """
    if isinstance(array, torch.Tensor):
        if array.device.type != "cpu":
            raise ValueError(
                "out_of_core=True keeps the cube on the host; pass a CPU array"
            )
        if array.dtype == torch.bfloat16:  # numpy has no bfloat16
            return AsFloat32(TensorSource(array.detach()))
        array = array.detach().numpy()
    if not isinstance(array, np.ndarray) and hasattr(array, "shape"):
        raise ValueError(
            f"out_of_core=True reads a numpy array, a memmap or a CPU tensor, not a "
            f"{type(array).__name__} (np.asarray would read it whole); open the cube "
            "as a source (H5RowSource, ZarrSource) and call Dataset4DStem.from_source"
        )
    arr = array if isinstance(array, np.ndarray) else np.asarray(array)
    if not arr.dtype.isnative:
        raise ValueError(
            f"out_of_core=True needs a cube in native byte order, got {arr.dtype}; "
            "convert it with arr.astype(arr.dtype.newbyteorder('='))"
        )
    if arr.dtype.kind not in "biuf":
        raise ValueError(
            f"out_of_core=True reads bool, integer or float cubes, got {arr.dtype}"
        )
    source = MemmapSource(arr)
    in_ram = not isinstance(arr, np.memmap) and arr.flags.writeable
    if source.dtype in STREAM_ENCODINGS and in_ram and arr.flags.c_contiguous:
        return TensorSource(torch.from_numpy(arr))
    return source if source.dtype in _DIRECT_ENCODINGS else AsFloat32(source)


def out_of_core_dataset(cls: Any, array: Any, meta: Any, *, device: Any, normalize: bool,
                        transform_to_amplitudes: bool, fourier_shift_dim: Any,
                        clip_neg_values: bool, row_block: int, g_residency: str,
                        labels: tuple) -> Any:  # fmt: skip
    """``from_array(..., out_of_core=True)``: a dataset that reads its cube in pieces (U5).

    The cube stays where it is (a memmap stays on disk, a CPU array stays in host
    RAM); every read converts with a ``RowTransform`` built from the resident keywords,
    and pass 0 gives ``c``. Every dtype the eager constructor reads is read, cast as
    the eager constructor casts. A uint8, uint16 or float16 cube streams in a fit
    (``host_cache="auto"``); any other dtype feeds a fit by synchronous gathers.

    Raises
    ------
    ValueError
        A tensor on a GPU, a lazy array-like (an HDF5, Zarr or dask array), a cube
        not in native byte order, or a dtype that is not bool, integer or float.
    """
    source = _cube_source(array)
    name, origin, sampling, units, signal_units = labels
    tdev = torch.device("cpu" if device is None else device)
    f16 = source.dtype == torch.float16
    tr = RowTransform(
        source.dtype, fourier_shift_dim=fourier_shift_dim, clip_neg_values=clip_neg_values,
        transform_to_amplitudes=transform_to_amplitudes, transform_device=tdev,
        inv_scale=torch.tensor(1.0, dtype=torch.float32) if f16 else None,  # 2**0, exact
    )  # fmt: skip
    ds = cls.from_source(
        source, meta, transform=tr, name=name, origin=origin, sampling=sampling,
        units=units, signal_units=signal_units, device=device, normalize=normalize,
        host_cache="auto" if source.dtype in STREAM_ENCODINGS else None,
        row_block=row_block,
    )  # fmt: skip
    ds.g_residency = str(g_residency)
    return ds


def build_vbf(ds: Any, threshold: float) -> Any:
    """``Dataset4DStem._build_vbf``: the streaming provider when ``ds`` reads in blocks."""
    if not reads_in_blocks(ds):
        from . import virtual_bright_field as vbf  # noqa: PLC0415

        return vbf.DatasetVirtualBrightField4DStem.from_4dstem_dataset(
            ds, bright_field_mask_threshold=threshold, device=ds.device
        )
    # A lazy import: _vbf_streaming imports this package.
    from scatterem.data import _vbf_streaming  # noqa: PLC0415

    return _vbf_streaming.DatasetVirtualBrightFieldStreaming.from_4dstem_out_of_core(
        ds, bright_field_mask_threshold=threshold
    )
