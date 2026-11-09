"""The streaming virtual-bright-field provider of a dataset that reads in blocks.

Moved out of the deleted ``out_of_core.py`` (frozen by the size ratchet) to make
room for pattern streaming: :class:`DatasetVirtualBrightFieldStreaming`, its residency policies
``RESIDENCIES`` and the pinned-or-pageable host allocator ``_empty_host``.
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.fft import fftfreq

from ._serial_threads import serial_here
from .datasets import (
    DatasetVirtualBrightField4DStem,
    _bf_mask_geometry,
    _normalized_mean_pattern,
    _place_aberrations_on_device,
)
from .detector_warp import bright_field_k
from .storage._protocol import native_device

if TYPE_CHECKING:
    from .datasets import Dataset4DStem

logger = logging.getLogger(__name__)

# The concrete residency policies of DatasetVirtualBrightFieldStreaming
# ("auto" resolves to one of these at build time). Single source of truth —
# tests parametrize over it.
RESIDENCIES = ("gpu", "vbf_gpu", "cpu_pinned")


def _empty_host(shape, dtype, pin: bool) -> torch.Tensor:
    """Host tensor, pinned when possible. Multi-GiB single ``cudaHostAlloc``
    calls can fail with 'invalid argument' on some driver/OS combinations
    (observed for ~9 GiB on this host despite an ample memlock limit) — fall
    back to pageable memory with a warning; transfers are then synchronous
    copies but everything stays correct."""
    if pin:
        try:
            return torch.empty(shape, dtype=dtype, pin_memory=True)
        except RuntimeError as exc:  # torch.AcceleratorError subclasses this
            import warnings

            warnings.warn(
                f"pinned host allocation of {tuple(shape)} {dtype} failed "
                f"({exc}); falling back to pageable memory (slower H2D).",
                stacklevel=2,
            )
    return torch.empty(shape, dtype=dtype)


def _bright_field_box(dataset: Dataset4DStem, iy, ix):
    """A frame view over the bright-field box, or the dataset when its reads are fftshifted."""
    # A lazy import: scatterem.data.datasets imports this module.
    from .datasets._direct import frame_view  # noqa: PLC0415

    tr = dataset.read_transform
    if tr is not None and tr.fourier_shift_dim is not None:
        return dataset, 0, 0
    y0, x0 = int(iy.min()), int(ix.min())
    frame = (slice(y0, int(iy.max()) + 1), slice(x0, int(ix.max()) + 1))
    return frame_view(dataset, frame), y0, x0


def _vbf_rows(sub: Dataset4DStem) -> int:
    """Scan rows per vBF read: whole storage chunks of about 16 MiB of whole frames.

    glibc maps a buffer above 32 MiB with fresh pages on every read; a smaller one is
    reused from the heap. So the pass-1 copy reads at most ``row_block`` rows, and
    fewer when they are large.
    """
    src = getattr(sub, "_source", None)
    shape = getattr(getattr(src, "base", src), "shape", sub.shape)  # whole frames
    row_bytes = max(1, int(sub.shape[1]) * int(np.prod(shape[-2:])) * 4)
    rows = min(int(getattr(sub, "_row_block", 8)), max(1, (16 << 20) // row_bytes))
    rc = getattr(src, "row_chunk", None) or 1
    return -(-rows // rc) * rc


def _copy_bright_field(sub: Dataset4DStem, out: np.ndarray, iy, ix) -> None:
    """``out[y, x] = pattern[y, x][iy, ix]`` for every scan point: pass 1.

    The copy is light (a crop, a cast and a gather per block), so on the CPU it runs
    on one OpenMP thread: thousands of small parallel regions stall when a foreign
    process keeps one core busy (``_serial_threads``).
    """
    cpu = native_device(sub.source).type == "cpu"
    with serial_here() if cpu else contextlib.nullcontext():
        for s, e, block in sub.iter_scan_blocks(_vbf_rows(sub), device="cpu"):
            out[s:e] = block.numpy()[:, :, iy, ix]


class DatasetVirtualBrightFieldStreaming(DatasetVirtualBrightField4DStem):
    """vBF whose real-space stack lives on CPU (optionally pinned) and whose G
    is served in BF-pixel chunks under a residency policy.

    Residency policies (``g_residency``):

    - ``"gpu"``        — full G resident on the compute device (~17 GiB for a
      1024^2 scan with ~2200 BF px); chunks are non-contiguous views. Fastest;
      needs a ~24 GiB card at full scale.
    - ``"vbf_gpu"``    — only the float32 vBF stack is resident (~8.5 GiB);
      each chunk's G is recomputed as ``fft2`` on the fly. Fits 16 GiB.
    - ``"cpu_pinned"`` — G lives in pinned host RAM, chunks are H2D-copied on
      demand. Minimal VRAM, PCIe-bound; avoid for fit/depth loops.
    - ``"auto"``       — pick by free VRAM at build time.

    Every consumer iterates ``get_G_chunk(s, e)``; the three policies are
    numerically interchangeable (the scan-plane FFT is independent per BF
    pixel, so a recomputed chunk equals a slice of the full transform up to
    cuFFT plan-level float variance).
    """

    # Set by ``from_4dstem_out_of_core`` on the instance it builds, and read by
    # every ``get_G_chunk`` path below. Declared here so the residency contract
    # is visible on the class instead of being three attributes that only exist
    # if you went through the one classmethod.
    _compute_device: torch.device
    _chunk_bf: int
    _g_residency: str

    @classmethod
    def from_4dstem_out_of_core(
        cls,
        dataset: Dataset4DStem,
        bright_field_mask_threshold: float = 0.3,
        num_indices_for_bright_field_mask: int = 625,
        g_residency: str | None = None,
        chunk_bf: int = 64,
        device: torch.device | None = None,
        pin_memory: bool = True,
        verbosity: int = 0,
    ) -> DatasetVirtualBrightFieldStreaming:
        cdev = torch.device(device) if device is not None else dataset.device
        ny, nx = dataset.shape[:2]
        # BF mask from the same scan-corner slab as the eager constructor
        # (n = ceil(sqrt(625)) = 25): identical mask, ordering and k.
        n = int(np.ceil(np.sqrt(num_indices_for_bright_field_mask)))
        slab = dataset.read((slice(0, n), slice(0, n)), device="cpu")
        diff_mean = slab.to(cdev).mean((0, 1))
        diff_mean = _normalized_mean_pattern(diff_mean)
        geo = _bf_mask_geometry(diff_mean, bright_field_mask_threshold)
        inds = geo["bright_field_inds_ordered_by_radius"]
        n_bf = int(inds.shape[0])
        iy = inds[:, 0].cpu().numpy()
        ix = inds[:, 1].cpu().numpy()

        # Pass 1: ONE sequential read of the source, fanned out to all BF columns.
        # Only the BF disk's bounding box is read and converted.
        sub, y0, x0 = _bright_field_box(dataset, iy, ix)
        pin = bool(pin_memory) and cdev.type == "cuda"
        vbf_cpu = _empty_host((ny, nx, n_bf), torch.float32, pin)
        vbf_np = vbf_cpu.numpy()  # shared memory: the writes land in place
        _copy_bright_field(sub, vbf_np, iy - y0, ix - x0)

        # Reads the dataset's own sampling, the way the Qx/Qy lines below always
        # have: the dataset's sampling is the source of truth ``meta`` mirrors.
        sampling_det = torch.as_tensor(dataset.sampling[-2:], device=cdev)[None]
        centered = geo["bright_field_inds_centered_ordered_by_radius"].to(cdev)
        k = bright_field_k(centered, sampling_det, dataset.meta)
        Qx = fftfreq(nx, dataset.sampling[1], dtype=torch.float32, device=cdev)
        Qy = fftfreq(ny, dataset.sampling[0], dtype=torch.float32, device=cdev)

        obj = cls(
            array=vbf_cpu,
            name=f"vBF of {dataset._name} (streaming)",
            origin=dataset.origin,
            sampling=dataset._sampling,
            units=dataset._units[:-1],
            signal_units=dataset._signal_units,
            _token=cls._token,
            meta=dataset.meta,
            device=torch.device("cpu"),  # keep the stack on CPU
            parent_dataset=dataset,
            diffraction_pattern_mean_normalized=diff_mean,
            bright_field_mask=geo["bright_field_mask"],
            bright_field_inds=geo["bright_field_inds"],
            bright_field_inds_centered=geo["bright_field_inds_centered"],
            bright_field_inds_radial_order=geo["bright_field_inds_radial_order"],
            bright_field_inds_ordered_by_radius=inds,
            bright_field_inds_centered_ordered_by_radius=centered,
            fourier_shift_dim=None,
            clip_neg_values=False,
            k=k,
            qx_1d=Qx,
            qy_1d=Qy,
        )
        obj._compute_device = cdev
        # DatasetVirtualBrightField4DStem.__init__ just placed the (shared)
        # meta's aberrations on the vBF's OWN array device, which is
        # deliberately CPU here ("keep the stack on CPU" above) -- not the
        # compute device the Warp kernels actually launch on. Re-place onto
        # cdev now that it is known, or direct_ptychography's kernel launch
        # sees a CPU aberration tensor again.
        _place_aberrations_on_device(obj.meta, cdev)
        obj._chunk_bf = int(chunk_bf)
        requested = g_residency if g_residency is not None else dataset.g_residency
        obj._g_residency = obj._resolve_residency(requested, ny, nx, n_bf, cdev)
        if verbosity > 0:
            logger.info(
                "streaming vBF: %d BF px, residency=%s, chunk_bf=%d, compute device=%s",
                n_bf,
                obj._g_residency,
                obj._chunk_bf,
                cdev,
            )
        obj._materialize_G()
        return obj

    # ---- residency ---------------------------------------------------------
    @staticmethod
    def _resolve_residency(requested, ny, nx, n_bf, cdev) -> str:
        if requested in RESIDENCIES:
            return requested
        if requested not in (None, "auto"):
            raise ValueError(f"unknown g_residency: {requested!r}")
        if cdev.type != "cuda":
            return "vbf_gpu"
        free, _ = torch.cuda.mem_get_info(cdev)
        g_bytes = ny * nx * n_bf * 8  # complex64
        if free >= 1.3 * g_bytes:
            return "gpu"
        if free >= 1.3 * (ny * nx * n_bf * 4):
            return "vbf_gpu"
        return "cpu_pinned"

    def _materialize_G(self) -> None:
        ny, nx, n_bf = self._array.shape
        cdev = self._compute_device
        cb = self._chunk_bf
        if self._g_residency == "gpu":
            self._G = torch.empty((ny, nx, n_bf), dtype=torch.complex64, device=cdev)
            for s in range(0, n_bf, cb):
                e = min(s + cb, n_bf)
                self._G[..., s:e] = torch.fft.fft2(
                    self._array[..., s:e].to(cdev), dim=(0, 1), norm="ortho"
                )
        elif self._g_residency == "vbf_gpu":
            # 4 B/px resident; G chunks are recomputed on the fly.
            self._array = self._array.to(cdev)
            self._array3d = self._array
        else:  # cpu_pinned: (n_bf, ny, nx) layout so chunk slices are contiguous
            self._G_cpu = _empty_host(
                (n_bf, ny, nx), torch.complex64, cdev.type == "cuda"
            )
            for s in range(0, n_bf, cb):
                e = min(s + cb, n_bf)
                g = torch.fft.fft2(
                    self._array[..., s:e].to(cdev), dim=(0, 1), norm="ortho"
                )
                self._G_cpu[s:e] = g.movedim(-1, 0).to("cpu")

    # ---- consumer surface ----------------------------------------------------
    @property
    def device(self) -> torch.device:  # pyright: ignore[reportIncompatibleMethodOverride]  # read-only by design
        """The COMPUTE device (the vBF stack itself may live on the CPU)."""
        return self._compute_device

    @property
    def n_bright_field(self) -> int:
        return int(self._array.shape[-1])

    @property
    def G(self) -> torch.Tensor:
        if self._g_residency == "gpu":
            return self._G
        raise RuntimeError(
            f"g_residency={self._g_residency!r} keeps no resident G; iterate "
            "get_G_chunk(s, e) instead."
        )

    def get_G_chunk(self, s: int, e: int) -> torch.Tensor:
        """Chunk of G on the compute device; ALWAYS an owned tensor the caller
        may mutate freely (never a view of the resident cache), with the
        alignment overlay applied when one is active. The copy this costs
        replaces the one the aberration kernel's ``.contiguous()`` would have
        made on a non-contiguous view — net zero extra copies."""
        if self._g_residency == "gpu":
            # resident-G slicing: the eager implementation already handles the
            # clone-vs-contiguous ownership subtlety (and the overlay).
            return super().get_G_chunk(s, e)
        if self._g_residency == "vbf_gpu":
            chunk = torch.fft.fft2(self._array[..., s:e], dim=(0, 1), norm="ortho")
        else:
            # ``copy=True``: on a CPU compute device ``.to`` would return a
            # view of the host G; to a GPU it is the one H2D copy either way.
            chunk = (
                self._G_cpu[s:e]
                .to(self._compute_device, non_blocking=True, copy=True)
                .movedim(0, -1)
            )
        return self._apply_alignment(chunk, slice(s, e))

    def get_G_columns(self, idx: torch.Tensor) -> torch.Tensor:
        """G columns for an arbitrary index tensor (radius order), owned, on
        the compute device, overlay applied. The gathers hit memory-resident
        tensors (VRAM or host RAM), never the disk memmap."""
        if self._g_residency == "gpu":
            return super().get_G_columns(idx)
        idx = torch.as_tensor(idx, dtype=torch.long, device=self.device)
        if self._g_residency == "vbf_gpu":
            chunk = torch.fft.fft2(self._array[..., idx], dim=(0, 1), norm="ortho")
        else:
            chunk = (
                self._G_cpu[idx.cpu()]
                .to(self._compute_device, non_blocking=True)
                .movedim(0, -1)
            )
        return self._apply_alignment(chunk, idx)
