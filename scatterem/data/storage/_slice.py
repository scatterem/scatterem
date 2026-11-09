"""A lazy crop over any source (``view`` for sources that cannot slice natively)."""

import math

import torch
from torch import Tensor

from ._protocol import (
    ArraySource,
    ScanIndex,
    as_flat_index,
    check_frame,
    check_index,
    compose,
    unravel,
)

__all__ = ["SlicedSource"]


class SlicedSource:
    """The block ``index`` of ``base``; reads go to ``base`` with composed indices.

    Parameters
    ----------
    base : ArraySource
        The source cropped.
    index : tuple of slice
        One step-1 slice per scan dimension of ``base``.
    frame : tuple of slice, optional
        One step-1 slice per frame dimension of ``base`` (a detector crop). ``None``
        keeps the whole frame.
    """

    def __init__(
        self, base: ArraySource, index: ScanIndex, *, frame: ScanIndex | None = None
    ) -> None:
        self.base = base
        self.scan_ndim = base.scan_ndim
        self.index = check_index(index, base.shape, base.scan_ndim)
        self.frame = check_frame(frame, base.shape, base.scan_ndim)
        self._whole_frame = self.frame == check_frame(None, base.shape, base.scan_ndim)
        scan = tuple(s.stop - s.start for s in self.index)
        self.shape = scan + tuple(s.stop - s.start for s in self.frame)
        self.dtype = base.dtype

    @property
    def concurrent_reads(self) -> bool:
        """Whether ``read`` is safe from several threads at once (the base's)."""
        return bool(getattr(self.base, "concurrent_reads", False))

    @property
    def row_chunk(self) -> int | None:
        """The base's row chunk when the crop starts on its chunk grid, else None.

        Pass 0 counts whole chunks from the crop's first scan row, so a crop that
        starts inside a chunk has no row chunk.
        """
        rc = self.base.row_chunk
        if rc is None or self.index[0].start % rc:
            return None
        return rc

    def _crop_frame(self, x: Tensor, lead: int) -> Tensor:
        if self._whole_frame:
            return x
        return x[(slice(None),) * lead + self.frame].contiguous()

    @property
    def is_resident(self) -> bool:
        """Whether the base is resident."""
        return self.base.is_resident

    @property
    def reads_need_materialize(self) -> bool:
        """Whether reads raise until ``materialize`` (the base decides)."""
        return bool(getattr(self.base, "reads_need_materialize", False))

    def read(
        self, index: ScanIndex | None = None, *, device: torch.device | str
    ) -> Tensor:
        """Read the block ``index`` of this view from the base.

        Parameters
        ----------
        index : tuple of slice, optional
            One step-1 slice per scan dimension. ``None`` reads the whole view.
        device : torch.device or str
            Device of the result.

        Returns
        -------
        Tensor
            What the base's ``read`` returns for the composed index.
        """
        inner = (
            None if index is None else check_index(index, self.shape, self.scan_ndim)
        )
        return self._crop_frame(
            self.base.read(compose(self.index, inner), device=device), self.scan_ndim
        )

    def gather(self, flat_points, *, device: torch.device | str) -> Tensor:
        """Read the patterns at the view's flat scan points from the base.

        Parameters
        ----------
        flat_points : sequence of int or Tensor
            A 1-D integer index of flat scan points of the view (``iy * nx + ix`` on a raster).
            Negative points count from the end.
        device : torch.device or str
            Device of the result.

        Returns
        -------
        Tensor
            ``(len(flat_points), *frame_shape)``, from the base's ``gather``.

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
        parts = unravel(as_flat_index(flat_points, math.prod(scan)), scan)
        base_scan = self.base.shape[: self.scan_ndim]
        flat = torch.zeros_like(parts[0])
        for p, s, n in zip(parts, self.index, base_scan, strict=True):
            flat = flat * n + (p + s.start)
        return self._crop_frame(self.base.gather(flat, device=device), 1)

    def view(
        self, index: ScanIndex, *, frame: ScanIndex | None = None
    ) -> "SlicedSource":
        """Return a crop of this crop, still over the same base.

        Parameters
        ----------
        index : tuple of slice
            One step-1 slice per scan dimension of this view.
        frame : tuple of slice, optional
            One step-1 slice per frame dimension of this view.

        Returns
        -------
        SlicedSource
            A lazy crop of the base. It reads no data.
        """
        inner = check_index(index, self.shape, self.scan_ndim)
        inner_frame = check_frame(frame, self.shape, self.scan_ndim)
        return SlicedSource(
            self.base,
            compose(self.index, inner),
            frame=compose(self.frame, inner_frame),
        )

    def materialize(self, device: torch.device | str) -> None:
        """Materialise the base.

        Parameters
        ----------
        device : torch.device or str
            Device that holds the base's resident array.
        """
        self.base.materialize(device)

    def release(self) -> None:
        """Release the base."""
        self.base.release()
