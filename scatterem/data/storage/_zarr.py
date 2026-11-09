"""A zarr array source, or one tilt of a ``(T, *scan, *frame)`` zarr array.

Chunks hold whole scan rows (the simulator sinks), so ``gather`` reads every covering
scan-row chunk once and picks the frames in RAM. Reading frame by frame would decode a
chunk per frame. zarr itself is never imported here: the caller opens the array, and a
zarr array pickles by its store, not by its data.
"""

import math
from typing import Any

import numpy as np
import torch
from torch import Tensor

from ._protocol import ScanIndex, as_flat_index, check_index, take
from ._slice import SlicedSource

__all__ = ["ZarrSource"]


class ZarrSource:
    """A zarr array with its scan dimensions first, read lazily.

    Every read returns a tensor that owns its memory.

    Parameters
    ----------
    array : zarr.Array
        ``(*scan, *frame)``, or ``(T, *scan, *frame)`` with ``tilt``.
    tilt : int, optional
        The tilt this source reads from a 5-D (or longer) array.
    scan_ndim : int
        2 on a raster scan, 1 on a point scan.

    Raises
    ------
    ValueError
        A bad ``scan_ndim``, a ``tilt`` out of range, or a dtype not in native byte
        order.
    """

    # ``read`` is safe from several threads at once (zarr reads of one array are thread-safe).
    concurrent_reads = True

    def __init__(
        self, array: Any, *, tilt: int | None = None, scan_ndim: int = 2
    ) -> None:
        lead = 0 if tilt is None else 1
        nd = len(array.shape) - lead
        if scan_ndim not in (1, 2) or nd < scan_ndim:
            raise ValueError(f"bad scan_ndim {scan_ndim} for shape {array.shape}")
        if tilt is not None and not 0 <= int(tilt) < int(array.shape[0]):
            raise ValueError(f"tilt {tilt} out of range for {array.shape[0]} tilts")
        dt = np.dtype(array.dtype)
        if not dt.isnative:
            raise ValueError(f"zarr array: {dt} is not in native byte order")
        self.array, self.tilt, self.scan_ndim = array, tilt, scan_ndim
        self._lead: tuple[int, ...] = () if tilt is None else (int(tilt),)
        self.shape = tuple(int(n) for n in array.shape[lead:])
        self.dtype = torch.from_numpy(np.empty(0, dtype=dt)).dtype
        self._np_dtype = dt
        self.row_chunk: int | None = int(array.chunks[lead])
        self._resident: Tensor | None = None

    @property
    def is_resident(self) -> bool:
        """Whether ``materialize`` has loaded the array."""
        return self._resident is not None

    def read(
        self, index: ScanIndex | None = None, *, device: torch.device | str
    ) -> Tensor:
        """Read the block ``index`` selects; zarr decodes each covering chunk once.

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
        # zarr decodes into a new buffer on every __getitem__, so the block owns its
        # memory already; a second copy would double the peak RAM of a whole read.
        block = np.asarray(self.array[self._lead + idx])
        if not block.flags.writeable:
            block = block.copy()
        return torch.from_numpy(block).to(device)

    def gather(self, flat_points, *, device: torch.device | str) -> Tensor:
        """Read the patterns at flat scan points, one read per covering chunk.

        Parameters
        ----------
        flat_points : sequence of int or Tensor
            A 1-D integer index of flat scan points (``iy * nx + ix`` on a raster),
            in any order, repeats allowed. Negative points count from the end.
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
        scan, frame = self.shape[: self.scan_ndim], self.shape[self.scan_ndim :]
        idx = as_flat_index(flat_points, math.prod(scan))
        if self._resident is not None:
            frames = self._resident.reshape(-1, *frame)
            return take(frames, idx).to(device)
        pts = idx.numpy()
        per_row = math.prod(scan[1:])  # points per scan row; 1 on a point scan
        rc = self.row_chunk or scan[0]
        out = np.empty((pts.size, *frame), dtype=self._np_dtype)
        chunk = (pts // per_row) // rc
        for c in np.unique(chunk):
            y0 = int(c) * rc
            y1 = min(y0 + rc, scan[0])
            block = np.asarray(self.array[(*self._lead, slice(y0, y1))]).reshape(
                -1, *frame
            )
            hit = np.flatnonzero(chunk == c)
            out[hit] = block[pts[hit] - y0 * per_row]
        return torch.from_numpy(out).to(device)

    def view(self, index: ScanIndex, *, frame: ScanIndex | None = None) -> SlicedSource:
        """Return a lazy crop.

        Parameters
        ----------
        index : tuple of slice
            One step-1 slice per scan dimension.
        frame : tuple of slice, optional
            One step-1 slice per frame dimension.

        Returns
        -------
        SlicedSource
            A crop over this source. It reads no data.
        """
        return SlicedSource(self, index, frame=frame)

    def materialize(self, device: torch.device | str) -> None:
        """Read the whole array once and keep it on ``device``.

        Parameters
        ----------
        device : torch.device or str
            Device that holds the resident array.
        """
        self._resident = None
        self._resident = self.read(device=device)

    def release(self) -> None:
        """Drop the resident copy."""
        self._resident = None
