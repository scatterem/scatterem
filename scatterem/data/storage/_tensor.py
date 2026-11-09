"""An in-memory source: the tensor itself."""

import math

import torch
from torch import Tensor

from ._protocol import (
    ScanIndex,
    as_flat_index,
    check_frame,
    check_index,
    take,
    unravel,
)

__all__ = ["TensorSource"]

# torch 2.11 cannot unpickle uint16, uint32 or uint64 tensors; they travel as the
# signed type of the same width (a bitwise view).
_PICKLE_AS = {
    torch.uint16: torch.int16,
    torch.uint32: torch.int32,
    torch.uint64: torch.int64,
}


class TensorSource:
    """A tensor with its scan dimensions first.

    ``read`` and ``gather`` return views or fresh tensors on ``device``; a read on the
    tensor's own device may alias it, so callers must not write into what they read.

    Parameters
    ----------
    tensor : Tensor
        The data, scan dimensions first.
    scan_ndim : int
        2 on a raster scan, 1 on a point scan.
    """

    # ``read`` is safe from several threads at once (reads index and copy the stored tensor).
    concurrent_reads = True

    def __init__(self, tensor: Tensor, *, scan_ndim: int = 2) -> None:
        if scan_ndim not in (1, 2) or tensor.ndim < scan_ndim:
            raise ValueError(
                f"scan_ndim must be 1 or 2 and at most tensor.ndim, got {scan_ndim} "
                f"for shape {tuple(tensor.shape)}"
            )
        self.tensor = tensor
        self.scan_ndim = scan_ndim

    def __getstate__(self) -> dict:
        """The state, with an unsigned tensor viewed as its signed twin."""
        state = dict(self.__dict__)
        signed = _PICKLE_AS.get(self.tensor.dtype)
        if signed is not None:
            state["tensor"] = self.tensor.view(signed)
            state["_unsigned"] = self.tensor.dtype
        return state

    def __setstate__(self, state: dict) -> None:
        """Rebuild from a pickle; an unsigned tensor gets its dtype back (a view)."""
        state = dict(state)
        unsigned = state.pop("_unsigned", None)
        if unsigned is not None:
            state["tensor"] = state["tensor"].view(unsigned)
        self.__dict__.update(state)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the stored tensor."""
        return tuple(self.tensor.shape)

    @property
    def dtype(self) -> torch.dtype:
        """Stored dtype."""
        return self.tensor.dtype

    @property
    def is_resident(self) -> bool:
        """Always True: the data is in memory."""
        return True

    @property
    def row_chunk(self) -> None:
        """Always None: a tensor has no storage chunks."""
        return None

    def read(
        self, index: ScanIndex | None = None, *, device: torch.device | str
    ) -> Tensor:
        """Read the tensor, or the block ``index`` selects, on ``device``.

        Parameters
        ----------
        index : tuple of slice, optional
            One step-1 slice per scan dimension. ``None`` reads the whole array.
        device : torch.device or str
            Device of the result.

        Returns
        -------
        Tensor
            The block in the stored dtype. On the tensor's own device it may alias
            the stored tensor.
        """
        if index is None:
            return self.tensor.to(device)
        idx = check_index(index, self.shape, self.scan_ndim)
        return self.tensor[idx].to(device)

    def gather(self, flat_points, *, device: torch.device | str) -> Tensor:
        """Read the patterns at flat scan points.

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
            ``(len(flat_points), *frame_shape)``, in the stored dtype.

        Raises
        ------
        TypeError
            If ``flat_points`` is a boolean mask or a float index.
        ValueError
            If ``flat_points`` is not 1-D.
        IndexError
            If a point is out of range.
        """
        scan_shape = self.shape[: self.scan_ndim]
        idx = as_flat_index(flat_points, math.prod(scan_shape))
        if self.tensor.is_contiguous():
            flat = self.tensor.view(-1, *self.shape[self.scan_ndim :])
            out = take(flat, idx)
        else:
            parts = unravel(idx, scan_shape)
            out = take(self.tensor, parts)
        return out.to(device)

    def view(
        self, index: ScanIndex, *, frame: ScanIndex | None = None
    ) -> "TensorSource":
        """Return a source over a crop of this tensor.

        Parameters
        ----------
        index : tuple of slice
            One step-1 slice per scan dimension.
        frame : tuple of slice, optional
            One step-1 slice per frame dimension (a detector crop). ``None`` keeps the
            whole frame.

        Returns
        -------
        TensorSource
            A source over the crop. It shares the tensor's storage.
        """
        idx = check_index(index, self.shape, self.scan_ndim)
        fr = check_frame(frame, self.shape, self.scan_ndim)
        return TensorSource(self.tensor[idx + fr], scan_ndim=self.scan_ndim)

    def materialize(self, device: torch.device | str) -> None:
        """Move the tensor to ``device`` (it is already resident).

        Parameters
        ----------
        device : torch.device or str
            Device that holds the tensor afterwards.
        """
        self.tensor = self.tensor.to(device)

    def release(self) -> None:
        """Nothing to release: the tensor is the data."""
