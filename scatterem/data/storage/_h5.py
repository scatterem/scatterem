"""A row-chunked HDF5 source, for files such as e4d that store one scan row per chunk.

Reads open the file for the call and close it again, so a source pickles (DataLoader
workers) and holds no handle. ``gather`` reads the rows that cover the requested points
once, in runs of consecutive rows, then picks the frames: on a row-chunked file a
shuffled batch decodes only the rows it touches.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from torch import Tensor

from ._protocol import ScanIndex, as_flat_index, check_index, take, unravel
from ._slice import SlicedSource

if TYPE_CHECKING:
    import h5py

__all__ = ["H5Part", "H5RowSource"]


def _h5py():
    import h5py
    import hdf5plugin  # noqa: F401  (registers the Blosc/Blosc2/bslz4 filters)

    return h5py


@dataclass(frozen=True)
class H5Part:
    """One file holding scan rows ``row_offset ...`` of the whole array."""

    path: str
    dataset: str
    row_offset: int


def _part_shapes(
    parts: Sequence[H5Part],
) -> tuple[list[tuple[int, ...]], np.dtype, int | None]:
    h5py = _h5py()
    shapes: list[tuple[int, ...]] = []
    dtype: np.dtype | None = None
    heights: set[int | None] = set()
    for p in parts:
        with h5py.File(p.path, "r") as f:
            d = cast("h5py.Dataset", f[p.dataset])
            shapes.append(tuple(int(n) for n in d.shape))
            heights.add(None if d.chunks is None else int(d.chunks[0]))
            dtype = d.dtype if dtype is None else dtype
            if d.dtype != dtype:
                raise ValueError(f"{p.path}: dtype {d.dtype} != {dtype}")
    if dtype is None:
        raise ValueError("an H5RowSource needs at least one part")
    height = heights.pop() if len(heights) == 1 else None
    if height is not None and any(p.row_offset % height for p in parts):
        return shapes, dtype, None  # a part off the chunk grid that starts at row 0
    return shapes, dtype, height


def _runs(sorted_rows: list[int]) -> list[tuple[int, int]]:
    """``[start, stop)`` runs of consecutive rows in a sorted list of distinct rows."""
    runs = [[sorted_rows[0], sorted_rows[0] + 1]]
    for r in sorted_rows[1:]:
        if r == runs[-1][1]:
            runs[-1][1] = r + 1
        else:
            runs.append([r, r + 1])
    return [(a, b) for a, b in runs]


class H5RowSource:
    """Scan rows split over one or more HDF5 datasets, read lazily.

    Every read returns a tensor that owns its memory: writing into it never changes the
    source, resident or not. ``row_chunk`` is the HDF5 chunk height when every part has
    the same one and every part's ``row_offset`` is a multiple of it.

    Parameters
    ----------
    parts : sequence of H5Part
        In row order; each part's ``row_offset`` must equal the rows before it.
    scan_ndim : int
        2 on a raster scan, 1 on a point scan (the "row" axis is then the point axis).
    """

    # ``read`` is safe from several threads at once (each read opens the file; h5py serializes the calls).
    concurrent_reads = True

    def __init__(self, parts: Sequence[H5Part], *, scan_ndim: int = 2) -> None:
        if not parts:
            raise ValueError("H5RowSource needs at least one part")
        self.parts = tuple(parts)
        self.scan_ndim = scan_ndim
        shapes, np_dtype, self.row_chunk = _part_shapes(self.parts)
        rows = 0
        for p, s in zip(self.parts, shapes, strict=True):
            if p.row_offset != rows or s[1:] != shapes[0][1:]:
                raise ValueError(
                    f"{p.path}: row_offset {p.row_offset} (expected {rows}) or "
                    f"frame shape {s[1:]} does not tile the rows"
                )
            rows += s[0]
        self.shape = (rows, *shapes[0][1:])
        self.dtype = torch.from_numpy(np.empty(0, dtype=np_dtype)).dtype
        self._np_dtype = np_dtype
        self._rows = [s[0] for s in shapes]
        self._resident: Tensor | None = None

    @classmethod
    def single(
        cls, path, dataset: str = "counts", *, scan_ndim: int = 2
    ) -> "H5RowSource":
        """Build a source over one dataset in one file.

        Parameters
        ----------
        path : str or os.PathLike
            The HDF5 file.
        dataset : str
            Path of the dataset inside the file.
        scan_ndim : int
            2 on a raster scan, 1 on a point scan.

        Returns
        -------
        H5RowSource
            A source with one part at row offset 0.
        """
        return cls([H5Part(str(path), dataset, 0)], scan_ndim=scan_ndim)

    @property
    def is_resident(self) -> bool:
        """Whether ``materialize`` has loaded the array."""
        return self._resident is not None

    def _read_np(self, idx: ScanIndex) -> np.ndarray:
        h5py = _h5py()
        rows, rest = idx[0], idx[1:]
        blocks = []
        for p, n in zip(self.parts, self._rows, strict=True):
            lo, hi = max(rows.start, p.row_offset), min(rows.stop, p.row_offset + n)
            if lo < hi:
                with h5py.File(p.path, "r") as f:
                    sel = (slice(lo - p.row_offset, hi - p.row_offset), *rest)
                    blocks.append(np.asarray(cast("h5py.Dataset", f[p.dataset])[sel]))
        if not blocks:
            frame = self.shape[len(idx) :]
            width = tuple(s.stop - s.start for s in rest)
            return np.empty((0, *width, *frame), dtype=self._np_dtype)
        return np.concatenate(blocks, axis=0)

    def read(
        self, index: ScanIndex | None = None, *, device: torch.device | str
    ) -> Tensor:
        """Read the block ``index`` selects from the rows that cover it.

        Parameters
        ----------
        index : tuple of slice, optional
            One step-1 slice per scan dimension. ``None`` reads the whole array.
        device : torch.device or str
            Device of the result.

        Returns
        -------
        Tensor
            The block in the stored dtype. It owns its memory.
        """
        idx = check_index(index, self.shape, self.scan_ndim)
        if self._resident is not None:
            return self._resident[idx].to(device, copy=True)
        return torch.from_numpy(self._read_np(idx)).to(device)

    def gather(self, flat_points, *, device: torch.device | str) -> Tensor:
        """Read the patterns at flat scan points, and only the rows they touch.

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
        frame = self.shape[self.scan_ndim :]
        idx = as_flat_index(flat_points, math.prod(scan))
        if self._resident is not None:
            frames = self._resident.reshape(-1, *frame)
            return take(frames, idx).to(device)
        if idx.numel() == 0:
            return torch.empty((0, *frame), dtype=self.dtype).to(device)
        coords = unravel(idx, scan)
        rows, inverse = torch.unique(coords[0], return_inverse=True)
        blocks = [
            torch.from_numpy(
                self._read_np((slice(a, b), *(slice(0, n) for n in scan[1:])))
            )
            for a, b in _runs(rows.tolist())
        ]
        block = torch.cat(blocks, dim=0)
        return block[(inverse, *coords[1:])].to(device)

    def view(self, index: ScanIndex, *, frame: ScanIndex | None = None) -> SlicedSource:
        """Return a lazy crop.

        Parameters
        ----------
        index : tuple of slice
            One step-1 slice per scan dimension.
        frame : tuple of slice, optional
            One step-1 slice per frame dimension (a detector crop). ``None`` keeps the
            whole frame.

        Returns
        -------
        SlicedSource
            A crop over this source. It reads no data.
        """
        return SlicedSource(self, index, frame=frame)

    def materialize(self, device: torch.device | str) -> None:
        """Read every row once and keep the array on ``device``.

        Parameters
        ----------
        device : torch.device or str
            Device that holds the resident array.
        """
        full = check_index(None, self.shape, self.scan_ndim)
        self._resident = torch.from_numpy(self._read_np(full)).to(device)

    def release(self) -> None:
        """Drop the resident copy."""
        self._resident = None
