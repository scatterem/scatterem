"""A numpy-array source, in practice a ``.npy`` memmap.

Over a C-ordered memmap that spans its whole file, reads go to the file with
``os.preadv``, one call per run of consecutive frames, straight into the result: no
page faults under the GIL. Any other array is read with numpy indexing. Such a source
pickles by its file, not by its data.
"""

import math
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from ._protocol import ScanIndex, as_flat_index, check_index, compose, take
from ._slice import SlicedSource

__all__ = ["MemmapSource"]


def _torch_dtype(dtype: np.dtype) -> torch.dtype:
    if not dtype.isnative:
        raise ValueError(f"{dtype} is not in native byte order")
    return torch.from_numpy(np.empty(0, dtype=dtype)).dtype


@dataclass(frozen=True)
class _File:
    """A whole C-ordered array in one file: where its data start, its shape, dtype."""

    path: str
    offset: int
    shape: tuple[int, ...]
    dtype: str


def _whole_file(array: np.ndarray) -> _File | None:
    """The file of a memmap that spans it whole, else None.

    numpy copies a parent's ``offset`` into every slice, so only an array whose bytes
    end exactly at the end of the file is trusted. A copy-on-write map (``mode="c"``)
    keeps its edits in private pages that the file does not see, and an unlinked file
    has no size, so both are read with numpy. A ``mode="r+"`` map shares the page
    cache with ``preadv``, so the file route sees its edits.
    """
    if not isinstance(array, np.memmap) or not array.filename:
        return None
    if array.mode == "c" or not array.flags.c_contiguous:
        return None
    try:
        size = os.path.getsize(array.filename)
    except OSError:
        return None
    if size != array.offset + array.nbytes:
        return None
    shape = tuple(int(n) for n in array.shape)
    return _File(os.fspath(array.filename), int(array.offset), shape, array.dtype.str)


def _runs(rows: np.ndarray) -> list[tuple[int, int, int]]:
    """``(position, first row, length)`` of each run of consecutive rows."""
    if rows.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(rows) != 1) + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [rows.size]))
    return [
        (int(s), int(rows[s]), int(e - s)) for s, e in zip(starts, ends, strict=True)
    ]


def _over_file(
    array: np.ndarray, file: _File, crop: ScanIndex, scan_ndim: int
) -> "MemmapSource":
    src = MemmapSource(array, scan_ndim=scan_ndim)
    src._file, src._crop = file, crop
    return src


def _reopen(file: _File, crop: ScanIndex, scan_ndim: int) -> "MemmapSource":
    full = np.memmap(
        file.path, dtype=np.dtype(file.dtype), mode="r", offset=file.offset,
        shape=file.shape,
    )  # fmt: skip
    return _over_file(full[crop], file, crop, scan_ndim)


class MemmapSource:
    """A numpy array read by slices; ``materialize`` keeps a tensor copy resident.

    Every read returns a writable tensor that owns its memory, never a view of a
    read-only memmap. An array that is not a memmap is in memory, so it is resident.

    A memmap slice made outside ``view`` (for example ``mm[2:5]``) pickles by its data
    (numpy gives it the parent's offset); ``view`` keeps the file route. A
    copy-on-write memmap (``mode="c"``) and a memmap whose file was unlinked are also
    read with numpy and pickle by their data, so in-memory edits are kept.

    Parameters
    ----------
    array : numpy.ndarray
        The data, scan dimensions first; usually ``np.load(path, mmap_mode="r")``.
    scan_ndim : int
        2 on a raster scan, 1 on a point scan.

    Raises
    ------
    ValueError
        A bad ``scan_ndim``, or an array not in native byte order.
    """

    # ``read`` is safe from several threads at once (each read opens its own file descriptor (``preadv``)).
    concurrent_reads = True

    def __init__(self, array: np.ndarray, *, scan_ndim: int = 2) -> None:
        if scan_ndim not in (1, 2) or array.ndim < scan_ndim:
            raise ValueError(f"bad scan_ndim {scan_ndim} for shape {array.shape}")
        self.array = array
        self.scan_ndim = scan_ndim
        self.shape = tuple(int(n) for n in array.shape)
        self.dtype = _torch_dtype(array.dtype)
        self.row_chunk: int | None = None
        self._resident: Tensor | None = None
        self._file = _whole_file(array)
        self._crop = check_index(None, self.shape, scan_ndim)

    @classmethod
    def open(cls, path, *, scan_ndim: int = 2) -> "MemmapSource":
        """Memory-map a ``.npy`` file.

        Parameters
        ----------
        path : str or os.PathLike
            The ``.npy`` file.
        scan_ndim : int
            2 on a raster scan, 1 on a point scan.

        Returns
        -------
        MemmapSource
            A source that reads the file with ``os.preadv``.

        Raises
        ------
        ValueError
            A bad ``scan_ndim``, or a file not in native byte order.
        """
        return cls(np.load(path, mmap_mode="r"), scan_ndim=scan_ndim)

    def __reduce__(self):
        """Pickle a whole-file memmap (or its crop) by its file, not its data."""
        if self._file is None:
            return object.__reduce__(self)
        return (_reopen, (self._file, self._crop, self.scan_ndim))

    @property
    def is_resident(self) -> bool:
        """Whether the data are in memory: materialised, or not a memmap."""
        return self._resident is not None or not isinstance(self.array, np.memmap)

    def _file_rows(self, flat: np.ndarray) -> np.ndarray:
        """Flat points of this crop as flat rows of the file's array."""
        assert self._file is not None
        scan = self.shape[: self.scan_ndim]
        coords = np.unravel_index(flat, scan)
        out = np.zeros_like(flat)
        dims = self._file.shape[: self.scan_ndim]
        for c, s, n in zip(coords, self._crop, dims, strict=True):
            out = out * n + (c + s.start)
        return out

    def _pread(self, rows: np.ndarray) -> Tensor:
        """The sorted, distinct file ``rows`` read with ``preadv`` into a new tensor."""
        file = self._file
        assert file is not None
        frame = self.shape[self.scan_ndim :]
        out = torch.empty((rows.size, *frame), dtype=self.dtype)
        row_bytes = math.prod(frame) * np.dtype(file.dtype).itemsize
        fd = os.open(file.path, os.O_RDONLY)
        try:
            for pos, row0, n in _runs(rows):
                view = memoryview(out[pos : pos + n].numpy()).cast("B")
                start = file.offset + row0 * row_bytes
                done = 0
                while done < len(view):
                    got = os.preadv(fd, [view[done:]], start + done)
                    if got <= 0:
                        raise OSError(f"{file.path}: short read at byte {start + done}")
                    done += got
        finally:
            os.close(fd)
        return out

    def read(
        self, index: ScanIndex | None = None, *, device: torch.device | str
    ) -> Tensor:
        """Read the block ``index`` selects, on ``device``.

        Parameters
        ----------
        index : tuple of slice, optional
            One step-1 slice per scan dimension. ``None`` reads the whole array.
        device : torch.device or str
            Device of the result.

        Returns
        -------
        Tensor
            A writable copy in the stored dtype that owns its memory.
        """
        idx = check_index(index, self.shape, self.scan_ndim)
        if self._resident is not None:
            return self._resident[idx].to(device, copy=True)
        if self._file is None:
            return torch.from_numpy(np.array(self.array[idx], copy=True)).to(device)
        block = tuple(s.stop - s.start for s in idx)
        grids = np.meshgrid(*(np.arange(s.start, s.stop) for s in idx), indexing="ij")
        flat = np.ravel_multi_index(
            tuple(g.reshape(-1) for g in grids), self.shape[: self.scan_ndim]
        ).astype(np.int64)
        frames = self._pread(self._file_rows(flat))
        return frames.reshape(*block, *self.shape[self.scan_ndim :]).to(device)

    def gather(self, flat_points, *, device: torch.device | str) -> Tensor:
        """Read the patterns at flat scan points, and only those frames.

        Parameters
        ----------
        flat_points : sequence of int or Tensor
            A 1-D integer index of flat scan points (``iy * nx + ix`` on a raster).
            Negative points count from the end.
        device : torch.device or str
            Device of the result.

        Returns
        -------
        Tensor
            ``(len(flat_points), *frame_shape)``, in the stored dtype. It owns its
            memory.

        Raises
        ------
        TypeError
            If ``flat_points`` is a boolean mask or a float index.
        ValueError
            If ``flat_points`` is not 1-D.
        IndexError
            If a point is out of range.
        """
        scan = self.shape[: self.scan_ndim]
        idx = as_flat_index(flat_points, math.prod(scan))
        if self._resident is not None:
            frames = self._resident.reshape(-1, *self.shape[self.scan_ndim :])
            return take(frames, idx).to(device)
        if self._file is None:
            coords = np.unravel_index(idx.numpy(), scan)
            return torch.from_numpy(np.array(self.array[coords], copy=True)).to(device)
        rows = self._file_rows(idx.numpy())
        uniq, inverse = np.unique(rows, return_inverse=True)
        frames = self._pread(uniq)
        if not np.array_equal(uniq, rows):
            frames = frames[torch.from_numpy(inverse.reshape(-1))]
        return frames.to(device)

    def view(
        self, index: ScanIndex, *, frame: ScanIndex | None = None
    ) -> "MemmapSource | SlicedSource":
        """Return a source over a crop; a memmap crop stays on disk.

        Parameters
        ----------
        index : tuple of slice
            One step-1 slice per scan dimension.
        frame : tuple of slice, optional
            One step-1 slice per frame dimension (a detector crop). ``None`` keeps the
            whole frame.

        Returns
        -------
        MemmapSource or SlicedSource
            MemmapSource, or a SlicedSource for a frame crop. A crop of a whole-file
            memmap still reads its file with ``os.preadv``.
        """
        if frame is not None:
            return SlicedSource(self, index, frame=frame)
        idx = check_index(index, self.shape, self.scan_ndim)
        if self._file is None:
            return MemmapSource(self.array[idx], scan_ndim=self.scan_ndim)
        crop = compose(self._crop, idx)
        return _over_file(self.array[idx], self._file, crop, self.scan_ndim)

    def materialize(self, device: torch.device | str) -> None:
        """Read the whole array once and keep it on ``device``.

        Parameters
        ----------
        device : torch.device or str
            Device that holds the resident copy.
        """
        self._resident = torch.from_numpy(
            np.array(self.array, copy=True, order="C")
        ).to(device)

    def release(self) -> None:
        """Drop the resident copy; reads go to the array again."""
        self._resident = None
