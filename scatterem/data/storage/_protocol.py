"""What every array source offers.

A source holds one array with its scan dimensions first: ``(scan_y, scan_x, ...)`` on a
raster scan (``scan_ndim == 2``) and ``(n_points, ...)`` on a point scan
(``scan_ndim == 1``). ``read`` takes one slice per scan dimension, ``gather`` takes flat
scan points (``iy * nx + ix`` on a raster), so nothing here assumes two scan axes.

Nothing decodes or moves silently: a source that is not resident (on disk, or a codec
blob) is read piece by piece, and keeping all of it in memory is an explicit
``materialize(device)``.
"""

from typing import Protocol, runtime_checkable

import numpy as np
import torch
from torch import Tensor

__all__ = ["ArraySource"]

ScanIndex = tuple[slice, ...]


@runtime_checkable
class ArraySource(Protocol):
    """Where a dataset's array lives.

    The four descriptors are read-only properties here so that an implementation may
    provide them as plain attributes or as properties (pyright accepts both).

    Parameters
    ----------
    *args, **kwargs
        A protocol is not instantiated; each implementation has its own constructor.
    """

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the whole array, scan dimensions first."""
        ...

    @property
    def scan_ndim(self) -> int:
        """Number of scan dimensions: 2 on a raster scan, 1 on a point scan."""
        ...

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the stored values."""
        ...

    @property
    def is_resident(self) -> bool:
        """Whether the whole array is in memory."""
        ...

    @property
    def row_chunk(self) -> int | None:
        """Scan rows per storage chunk; ``None`` when the storage has no chunks."""
        ...

    def read(
        self, index: ScanIndex | None = None, *, device: torch.device | str
    ) -> Tensor:
        """Read a block of the array.

        Parameters
        ----------
        index : tuple of slice, optional
            One step-1 slice per scan dimension. ``None`` reads the whole array.
        device : torch.device or str
            Device of the result.

        Returns
        -------
        Tensor
            The block, in the stored dtype, scan dimensions first.

        Notes
        -----
        A source whose ``read`` is safe from several threads at once may set the
        attribute ``concurrent_reads = True``; pass 0 then reads its blocks on worker
        threads. Without it, pass 0 reads one block at a time.
        """
        ...

    def gather(self, flat_points, *, device: torch.device | str) -> Tensor:
        """Read the patterns at flat scan points.

        Parameters
        ----------
        flat_points : sequence of int or Tensor
            A 1-D integer index of flat scan points (``iy * nx + ix`` on a raster).
        device : torch.device or str
            Device of the result.

        Returns
        -------
        Tensor
            ``(len(flat_points), *frame_shape)``, in the stored dtype.
        """
        ...

    def view(
        self, index: ScanIndex, *, frame: ScanIndex | None = None
    ) -> "ArraySource":
        """Return a lazy crop of this source.

        Parameters
        ----------
        index : tuple of slice
            One step-1 slice per scan dimension.
        frame : tuple of slice, optional
            One step-1 slice per frame dimension (a detector crop). ``None`` keeps the
            whole frame.

        Returns
        -------
        ArraySource
            A source over the crop. It reads no data.
        """
        ...

    def materialize(self, device: torch.device | str) -> None:
        """Load the whole array and keep it on ``device``.

        Parameters
        ----------
        device : torch.device or str
            Device that holds the resident array.
        """
        ...

    def release(self) -> None:
        """Drop the resident array; later reads go to the storage again."""
        ...


def check_index(
    index: ScanIndex | None, shape: tuple[int, ...], scan_ndim: int
) -> ScanIndex:
    """Normalise ``index`` to one in-bounds, step-1 slice per scan dimension."""
    if index is None:
        return tuple(slice(0, n) for n in shape[:scan_ndim])
    if not isinstance(index, tuple) or len(index) != scan_ndim:
        raise ValueError(
            f"index needs one slice per scan dimension ({scan_ndim}), got {index!r}"
        )
    out = []
    for s, n in zip(index, shape[:scan_ndim], strict=True):
        if not isinstance(s, slice):
            raise ValueError(f"index entries must be slices, got {s!r}")
        start, stop, step = s.indices(n)
        if step != 1:
            raise ValueError(f"slices must have step 1, got {s!r}")
        out.append(slice(start, max(start, stop)))
    return tuple(out)


def check_frame(
    frame: ScanIndex | None, shape: tuple[int, ...], scan_ndim: int
) -> ScanIndex:
    """Normalise ``frame`` to one in-bounds, step-1 slice per frame dimension."""
    dims = tuple(shape[scan_ndim:])
    if frame is None:
        return tuple(slice(0, n) for n in dims)
    if not isinstance(frame, tuple) or len(frame) != len(dims):
        raise ValueError(
            f"frame needs one slice per frame dimension ({len(dims)}), got {frame!r}"
        )
    return check_index(frame, dims, len(dims))


def compose(outer: ScanIndex, inner: ScanIndex | None) -> ScanIndex:
    """The slices of the base that ``inner`` selects inside the view ``outer``."""
    if inner is None:
        return outer
    return tuple(
        slice(o.start + i.start, o.start + i.stop)
        for o, i in zip(outer, inner, strict=True)
    )


def _kind(dtype: torch.dtype) -> str:
    """``"b"`` for bool, ``"f"`` for floating or complex, ``"i"`` for integers."""
    if dtype == torch.bool:
        return "b"
    return "f" if dtype.is_floating_point or dtype.is_complex else "i"


def as_flat_index(flat_points, n_points: int) -> Tensor:
    """``flat_points`` as a CPU int64 tensor, negatives wrapped, bounds checked.

    Raises
    ------
    TypeError
        If the index is boolean (a mask) or floating point.
    ValueError
        If it is not one-dimensional (a scalar or a 2-D index).
    IndexError
        If a point is out of range.
    """
    if isinstance(flat_points, Tensor):
        kind, ndim = _kind(flat_points.dtype), flat_points.ndim
    else:
        arr = np.asarray(flat_points)
        # An empty list has no dtype of its own (numpy says float64).
        empty_list = arr.size == 0 and not isinstance(flat_points, np.ndarray)
        kind, ndim = "i" if empty_list else arr.dtype.kind, arr.ndim
    if kind not in "iu":
        raise TypeError(
            "flat scan points must be integers (a boolean mask or float index is "
            f"not accepted; convert it first), got dtype kind {kind!r}"
        )
    if ndim != 1:
        raise ValueError(
            f"flat scan points must be a 1-D sequence of integers, got {ndim}-D input; "
            "pass a 1-D sequence"
        )
    if isinstance(flat_points, Tensor):
        idx = flat_points.detach().to(device="cpu", dtype=torch.int64)
    else:
        idx = torch.as_tensor(np.asarray(flat_points, dtype=np.int64))
    idx = torch.where(idx < 0, idx + n_points, idx)
    if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= n_points):
        raise IndexError(f"scan point index out of range for {n_points} points")
    return idx


# torch 2.11 has no CUDA index kernel for uint16/uint32/uint64, so index a signed view.
_SIGNED_TWIN = {
    torch.uint16: torch.int16,
    torch.uint32: torch.int32,
    torch.uint64: torch.int64,
}


def take(t: Tensor, index: Tensor | tuple[Tensor, ...]) -> Tensor:
    """``t[index]`` for integer index tensors on any device, bitwise, without a copy of ``t``."""
    index = (
        tuple(i.to(t.device) for i in index)
        if isinstance(index, tuple)
        else index.to(t.device)
    )
    twin = _SIGNED_TWIN.get(t.dtype)
    if twin is None:
        return t[index]
    return t.view(twin)[index].view(t.dtype)


def unravel(flat: Tensor, scan_shape: tuple[int, ...]) -> tuple[Tensor, ...]:
    """Row-major unravel of flat scan points (any number of scan dimensions)."""
    out = []
    rest = flat
    for n in reversed(scan_shape):
        out.append(rest % n)
        rest = rest // n
    return tuple(reversed(out))


def native_device(source: object) -> torch.device:
    """Where ``source`` reads without a copy: the device of the tensor it holds.

    A wrapper (``SlicedSource.base``, ``AsFloat32.inner``) asks the source it wraps. A
    ``TensorSource`` holds ``tensor``; a materialised codec, memmap, HDF5 or Zarr
    source holds its resident copy. Any other source reads on the CPU.

    Parameters
    ----------
    source : ArraySource
        The source.

    Returns
    -------
    torch.device
        The device of the held tensor, else the CPU.
    """
    for attr in ("base", "inner"):
        inner = getattr(source, attr, None)
        if inner is not None:
            return native_device(inner)
    held = getattr(source, "tensor", None)
    decoded = getattr(source, "_decoded", None)  # a wrapper that holds a TensorSource
    if held is None and decoded is not None:
        held = decoded.tensor
    if held is None:
        held = getattr(source, "_resident", None)  # memmap, HDF5, Zarr
    return held.device if isinstance(held, Tensor) else torch.device("cpu")
