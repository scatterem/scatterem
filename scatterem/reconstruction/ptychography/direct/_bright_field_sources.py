"""Input forms of the BF shift estimators: an in-memory stack or a source of parts.

The estimators ask for images by index list, in their own order. So the
internal chunking of a source never reaches the arithmetic.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from torch import Tensor

__all__ = ["ChunkedSpectrumSource", "SpectrumSource"]


@runtime_checkable
class SpectrumSource(Protocol):
    """A source that delivers the BF spectra in parts.

    Attributes
    ----------
    image_count : int
        Number of BF images ``N``.
    scan_grid_shape : tuple of int
        ``(n0, n1)``.
    run_device : torch.device
        Where :meth:`read` delivers data and where the work runs.
    """

    image_count: int
    scan_grid_shape: tuple[int, int]
    run_device: torch.device

    def read(self, indices: Sequence[int]) -> Tensor:
        """Return complex ``(len(indices), n0, n1)``, rows in the order given."""
        ...


class ChunkedSpectrumSource:
    """Adapt a contiguous-range reader (HDF5 dataset, memory map) to :class:`SpectrumSource`.

    Parameters
    ----------
    fetch : callable
        ``fetch(start, stop)`` returns a complex array or tensor
        ``(stop − start, n0, n1)``.
    image_count : int
    scan_grid_shape : (int, int)
    images_per_chunk : int
        Images per ``fetch`` call. The source caches the last chunk. The
        chunk size never changes the result of the estimators, because they
        ask for images by index.
    run_device : str or torch.device, optional
        Where the data go. Default CPU.
    """

    def __init__(
        self,
        fetch: Callable[[int, int], object],
        image_count: int,
        scan_grid_shape: tuple[int, int],
        *,
        images_per_chunk: int,
        run_device="cpu",
    ):
        if int(images_per_chunk) < 1:
            raise ValueError(f"images_per_chunk must be >= 1, got {images_per_chunk!r}")
        self._fetch = fetch
        self.image_count = int(image_count)
        self.scan_grid_shape = (int(scan_grid_shape[0]), int(scan_grid_shape[1]))
        self.images_per_chunk = int(images_per_chunk)
        self.run_device = torch.device(run_device)
        self._cache: tuple[int, Tensor] | None = None

    def _chunk(self, c: int) -> Tensor:
        if self._cache is None or self._cache[0] != c:
            start = c * self.images_per_chunk
            stop = min(self.image_count, start + self.images_per_chunk)
            data = self._fetch(start, stop)
            data = (
                data if isinstance(data, Tensor) else torch.as_tensor(np.asarray(data))
            )
            self._cache = (c, data.to(self.run_device))
        return self._cache[1]

    def read(self, indices: Sequence[int]) -> Tensor:
        rows = [
            self._chunk(int(i) // self.images_per_chunk)[int(i) % self.images_per_chunk]
            for i in indices
        ]
        if not rows:
            return torch.zeros(
                (0, *self.scan_grid_shape),
                dtype=torch.complex64,
                device=self.run_device,
            )
        return torch.stack(rows)


class _StackSource:
    """An in-memory stack as a :class:`SpectrumSource`."""

    def __init__(self, stack):
        if isinstance(stack, np.ndarray):
            stack = torch.from_numpy(stack)
        self._stack = stack
        self.image_count = int(stack.shape[0])
        self.scan_grid_shape = (int(stack.shape[1]), int(stack.shape[2]))
        self.run_device = stack.device

    def read(self, indices: Sequence[int]) -> Tensor:
        idx = torch.as_tensor(
            np.asarray(indices, dtype=np.int64), device=self.run_device
        )
        return self._stack.index_select(0, idx)


def _as_source(image_spectra):
    if isinstance(image_spectra, (Tensor, np.ndarray)):
        if not (
            image_spectra.is_complex()
            if isinstance(image_spectra, Tensor)
            else np.iscomplexobj(image_spectra)
        ):
            raise ValueError("image_spectra must be complex Fourier transforms")
        if image_spectra.ndim != 3:
            raise ValueError(
                f"image_spectra must be (N, n0, n1), got shape {tuple(image_spectra.shape)}"
            )
        source = _StackSource(image_spectra)
    elif isinstance(image_spectra, SpectrumSource):
        source = image_spectra
    else:
        raise ValueError("image_spectra must be a complex stack or a SpectrumSource")  # noqa: TRY004
    n0, n1 = source.scan_grid_shape
    if n0 < 1 or n1 < 1:
        raise ValueError(
            f"scan shape must be at least 1 x 1, got {source.scan_grid_shape}"
        )
    return source
