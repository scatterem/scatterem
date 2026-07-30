"""Out-of-core 4D-STEM dataset.

The raw uint16 cube stays a CPU (mem-mapped) numpy array; it is never uploaded
to the GPU and never widened to float32 in memory. Normalization becomes a
stored scalar (division by the identical scalar is elementwise, so consumers
see bitwise-identical values to the eager constructor's whole-cube divide).
Pass-0 statistics (mean CBED, total intensity, the tcDF mean variant) are
float64 reductions streamed over contiguous scan-row blocks — reading the
memmap sequentially matters: the per-position detector frame spans only a few
pages, so any per-pixel fancy indexing re-reads the whole file.

Design: docs/superpowers/plans/2026-07-14-chunked-direct-ptycho.md and the
chunked_direct_ptycho_design.md it references.
"""

from __future__ import annotations

from typing import Any, Iterator, Optional

import numpy as np
import torch
from torch import Tensor

from torch.fft import fftfreq

from .datasets import (
    Dataset4dstem,
    DatasetVirtualBrightField4dstem,
    _bf_mask_geometry,
    _place_aberrations_on_device,
)

# The concrete residency policies of DatasetVirtualBrightFieldStreaming
# ("auto" resolves to one of these at build time). Single source of truth —
# tests parametrize over it.
RESIDENCIES = ("gpu", "vbf_gpu", "cpu_pinned")


class Dataset4dstemOutOfCore(Dataset4dstem):
    """``Dataset4dstem`` whose raw cube is a CPU memmap read in scan-row blocks.

    Full-cube accessors (``.array``, ``._array3d``) raise so stray call sites
    fail loudly instead of silently working on unnormalized uint16 data.
    """

    def __init__(
        self,
        raw: np.ndarray,
        name: str,
        origin,
        sampling,
        units,
        meta,
        device,
        normalize: bool = True,
        ooc_row_block: int = 8,
        g_residency: str = "auto",
        signal_units: str = "arb. units",
        _stats: Optional[tuple] = None,
        normalization_const: Optional[float] = None,
    ) -> None:
        # Deliberately do NOT call Dataset4dstem.__init__ — it uploads the
        # cube, widens it to float32 and normalizes in place. Set the metadata
        # surface the pipeline consumes directly.
        if raw.ndim != 4:
            raise ValueError(f"expected a 4D scan cube, got shape {raw.shape}")
        self._raw = raw
        self._name = str(name)
        self._origin = np.asarray(origin)
        self._sampling = np.asarray(sampling)
        self._units = list(units)
        self._signal_units = str(signal_units)
        self.meta = meta
        self.probe_index = 0
        self.fourier_shift_dim = None
        self.transform_to_amplitudes = False
        self.bright_field_mask_threshold = 0.3
        self._shape = tuple(int(s) for s in raw.shape)
        self._device = torch.device(device)
        _place_aberrations_on_device(meta, self._device)
        self._row_block = int(ooc_row_block)
        self.g_residency = str(g_residency)
        self._normalize = bool(normalize)
        if _stats is not None:
            (
                self._mean_cbed_raw,
                self._mean_cbed_tcdf_raw,
                raw_total,
                self.dose_per_probe_unnormalized,
            ) = _stats
        else:
            raw_total = self._compute_pass0_stats()
        if normalization_const is not None:
            # e.g. a detector crop inherits the PARENT's scalar (the eager
            # path crops the already-normalized cube).
            self._normalization_const = float(normalization_const)
        else:
            self._normalization_const = (
                float(self._mean_cbed_raw.max()) if self._normalize else 1.0
            )
        # The RAW (pre-normalization) total, matching the eager path. Storing the
        # normalized total here mirrored an eager-path bug that made `fluence` wrong by
        # 1/normalization_const; both paths now keep the physical total.
        self._total_intensity = raw_total

    def _compute_pass0_stats(self) -> float:
        """Streamed float64 reductions over scan-row blocks; returns the raw
        (pre-normalization) total intensity."""
        a = self._raw
        ny, nx = self._shape[:2]
        # tcDF uses arr[:n, :n] with n = ny (tilt_corrected_dark_field.py);
        # numpy clamps the column slice when nx < ny.
        nsq_rows = ny
        nsq_cols = min(ny, nx)
        sum_cbed = np.zeros(a.shape[-2:], dtype=np.float64)
        sum_tcdf = np.zeros(a.shape[-2:], dtype=np.float64)
        for s in range(0, ny, self._row_block):
            # Reduce the raw uint16 block with a float64 ACCUMULATOR — never
            # materialize a float64 copy (8x the block's memory traffic).
            block = np.asarray(a[s : s + self._row_block])
            sum_cbed += block.sum(axis=(0, 1), dtype=np.float64)
            if s < nsq_rows:
                bl = block[: max(0, nsq_rows - s), :nsq_cols]
                sum_tcdf += bl.sum(axis=(0, 1), dtype=np.float64)
        n_probe = ny * nx
        total = float(sum_cbed.sum())
        self._mean_cbed_raw = torch.from_numpy(
            (sum_cbed / n_probe).astype(np.float32)
        )
        self._mean_cbed_tcdf_raw = torch.from_numpy(
            (sum_tcdf / (nsq_rows * nsq_cols)).astype(np.float32)
        )
        self.dose_per_probe_unnormalized = total / n_probe
        return total

    # ---- guarded full-cube accessors --------------------------------------
    # Guard the STORAGE attribute: base-class code reads self._array directly
    # (bin_detector, pad_and_taper_*, mean/max_probe_intensity) as well as via
    # the .array property — one raising property covers both failure levels.
    @property
    def _array(self):
        raise RuntimeError(
            "Dataset4dstemOutOfCore keeps the raw cube on disk (CPU memmap); "
            "full-cube access is disabled. Use .shape, iter_scan_blocks(), "
            "mean_diffraction_pattern / sum_diffraction_pattern, or the "
            "streaming vBF/tcDF paths."
        )

    @property
    def _array3d(self):
        raise RuntimeError(
            "Dataset4dstemOutOfCore has no in-memory 3D view; use "
            "iter_scan_blocks()."
        )

    # ---- shape / device surface (base properties read self.array) ----------
    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def ndim(self) -> int:
        return 4

    @property
    def dtype(self) -> torch.dtype:
        # Consumers receive float32 (normalized) slabs.
        return torch.float32

    @property
    def device(self) -> torch.device:
        return self._device

    # ---- streamed statistics ------------------------------------------------
    @property
    def normalization_const(self) -> float:
        return self._normalization_const

    @property
    def mean_diffraction_pattern(self) -> Tensor:
        """Full-scan mean CBED of the NORMALIZED data (CPU float32)."""
        return self._mean_cbed_raw / self._normalization_const

    @property
    def sum_diffraction_pattern(self) -> Tensor:
        """Full-scan sum CBED of the normalized data — the streaming
        equivalent of ``dataset.array.sum((0, 1))``."""
        return self.mean_diffraction_pattern * (self._shape[0] * self._shape[1])

    @property
    def mean_cbed_tcdf(self) -> Tensor:
        """tcDF's ``array[:n, :n].mean((0, 1))`` variant (n = scan rows)."""
        return self._mean_cbed_tcdf_raw / self._normalization_const

    def iter_scan_blocks(
        self, block_rows: Optional[int] = None
    ) -> Iterator[tuple[int, int, np.ndarray]]:
        """Yield ``(row_start, row_end, raw_uint16_block)`` sequentially.

        Consumers normalize with ``block.astype(np.float32) /
        normalization_const`` — identical values to the eager path's
        whole-cube scalar divide.
        """
        br = int(block_rows or self._row_block)
        for s in range(0, self._shape[0], br):
            e = min(s + br, self._shape[0])
            yield s, e, np.asarray(self._raw[s:e])

    def gather_detector_group_means(
        self, groups: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """ONE sequential streamed pass over the memmap, fanned out to ALL
        groups (a per-group gather would re-read the whole file per group).
        Reduces on the CPU and uploads only the (ny, nx) group images — for a
        full-scan tcDF that is ~0.3 GiB of H2D instead of ~21 GiB of gathered
        columns."""
        lens = [int(g.shape[0]) for g in groups]
        offs = np.concatenate(([0], np.cumsum(lens)))
        all_inds = torch.cat([g.cpu() for g in groups]).numpy()
        iy, ix = all_inds[:, 0], all_inds[:, 1]
        ny, nx = self._shape[:2]
        out = np.empty((len(groups), ny, nx), dtype=np.float32)
        for s, e, block in self.iter_scan_blocks():
            cols = block[:, :, iy, ix].astype(np.float32)
            cols /= self._normalization_const
            for j, m in enumerate(lens):
                if m == 0:
                    out[j, s:e] = np.nan  # eager parity: mean of empty gather
                else:
                    out[j, s:e] = cols[:, :, offs[j] : offs[j + 1]].mean(-1)
        return [torch.from_numpy(img).to(self._device) for img in out]

    def _build_vbf(
        self, bright_field_mask_threshold: float
    ) -> "DatasetVirtualBrightFieldStreaming":
        # get_vbf() cache hook: the out-of-core dataset gets the streaming
        # provider (residency policy from self.g_residency).
        return DatasetVirtualBrightFieldStreaming.from_4dstem_out_of_core(
            self, bright_field_mask_threshold=bright_field_mask_threshold
        )

    def _averaged_diffraction_pattern(self) -> Tensor:
        """Streaming twin of the eager ``self._array[:50, :50].mean((0, 1))``."""
        sx = min(self._shape[0], 50)
        sy = min(self._shape[1], 50)
        slab = np.asarray(self._raw[:sx, :sy], dtype=np.float32)
        slab /= self._normalization_const
        return torch.from_numpy(slab).to(self._device).mean((0, 1))

    def crop_brightfield_(
        self,
        thresh_lower: float = 0.01,
        thresh_upper: float = 0.99,
        normalize: bool = True,
        clip_neg_values: bool = True,
    ) -> "Dataset4dstemOutOfCore":
        """Detector-crop to the bright-field box WITHOUT materializing data.

        Numpy basic slicing of the memmap stays lazy; the child shares the
        PARENT's normalization scalar (the eager path crops the
        already-normalized cube) and its Pass-0 CBED statistics are the
        cropped slices of the parent's.

        ``normalize``/``clip_neg_values`` are accepted only for signature
        compatibility with the eager ``Dataset4dstem.crop_brightfield_``:
        ``calibrate_reciprocal_from_bright_field`` (inherited, not overridden
        here) calls them polymorphically with ``normalize=False,
        clip_neg_values=False`` for its measurement-only scratch crop. They
        are no-ops in this class -- the child always inherits the parent's
        ``_normalization_const`` directly (see below) rather than recomputing
        one from the crop, and the raw memmap is never widened/divided in
        place, so there is nothing here for the eager path's aliasing bug
        (an unclonded view mutated by ``Dataset4dstem.__init__``) to apply to.
        """
        box = self._bright_field_crop_box(thresh_lower, thresh_upper)
        self._radius_bright_field = box.radius_ceil
        # set the origin to the center of the bright field region (mirrors the
        # eager crop_brightfield_)
        self.origin[-1] = box.center[0]
        self.origin[-2] = box.center[1]
        sl = box.crop_slice
        det_sl = (sl[-2], sl[-1])
        child = Dataset4dstemOutOfCore(
            raw=self._raw[:, :, det_sl[0], det_sl[1]],
            name=self._name + str(sl),
            origin=np.array(self._origin),
            sampling=self._sampling,
            units=self._units,
            meta=self.meta,
            device=self._device,
            normalize=self._normalize,
            ooc_row_block=self._row_block,
            g_residency=self.g_residency,
            signal_units=self._signal_units,
            _stats=(
                self._mean_cbed_raw[det_sl],
                self._mean_cbed_tcdf_raw[det_sl],
                self._total_intensity,  # already the raw total; no rescale needed
                self.dose_per_probe_unnormalized,
            ),
            normalization_const=self._normalization_const,
        )
        child._radius_bright_field = box.radius
        child._crop_sl = det_sl
        # Shift the origin into the cropped coordinate system (mirrors eager).
        new_origin = list(child.origin)
        new_origin[-2] = box.center[0] - (box.y0 - box.radius)
        new_origin[-1] = box.center[1] - (box.x0 - box.radius)
        child._origin = np.asarray(tuple(new_origin))
        child.is_bright_field = True
        return child

    def _summary_rows(self) -> dict[str, Any]:
        return {
            "name": self._name,
            "shape": self._shape,
            "dtype": "uint16 memmap (served as normalized float32)",
            "device": f"cpu memmap -> compute {self._device}",
            "origin": self._origin,
            "sampling": self._sampling,
            "units": self._units,
            "signal_units": self._signal_units,
            "normalization_const": self._normalization_const,
            "g_residency": self.g_residency,
        }

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


class DatasetVirtualBrightFieldStreaming(DatasetVirtualBrightField4dstem):
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

    @classmethod
    def from_4dstem_out_of_core(
        cls,
        dataset: Dataset4dstemOutOfCore,
        bright_field_mask_threshold: float = 0.3,
        num_indices_for_bright_field_mask: int = 625,
        g_residency: Optional[str] = None,
        chunk_bf: int = 64,
        device: Optional[torch.device] = None,
        pin_memory: bool = True,
        verbosity: int = 0,
    ) -> "DatasetVirtualBrightFieldStreaming":
        cdev = torch.device(device) if device is not None else dataset.device
        ny, nx = dataset.shape[:2]
        # BF mask from the same scan-corner slab as the eager constructor
        # (n = ceil(sqrt(625)) = 25): identical mask, ordering and k.
        n = int(np.ceil(np.sqrt(num_indices_for_bright_field_mask)))
        slab = np.asarray(dataset._raw[:n, :n], dtype=np.float32)
        slab /= dataset.normalization_const
        diff_mean = torch.from_numpy(slab).to(cdev).mean((0, 1))
        diff_mean = diff_mean / diff_mean.max()
        geo = _bf_mask_geometry(diff_mean, bright_field_mask_threshold)
        inds = geo["bright_field_inds_ordered_by_radius"]
        n_bf = int(inds.shape[0])
        iy = inds[:, 0].cpu().numpy()
        ix = inds[:, 1].cpu().numpy()

        # Pass 1: ONE sequential read of the memmap, fanned out to all BF
        # columns (per-chunk fancy indexing would re-read the whole file).
        pin = bool(pin_memory) and cdev.type == "cuda"
        vbf_cpu = _empty_host((ny, nx, n_bf), torch.float32, pin)
        vbf_np = vbf_cpu.numpy()  # shared memory: write casts land in place
        for s, e, block in dataset.iter_scan_blocks():
            dst = vbf_np[s:e]
            dst[...] = block[:, :, iy, ix]  # uint16 -> float32 cast on assign
            dst /= dataset.normalization_const

        sampling_det = torch.as_tensor(dataset.meta.sampling[-2:], device=cdev)[None]
        centered = geo["bright_field_inds_centered_ordered_by_radius"].to(cdev)
        k = (centered * sampling_det.expand_as(centered)).to(torch.float32)
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
        # DatasetVirtualBrightField4dstem.__init__ just placed the (shared)
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
            print(
                f"streaming vBF: {n_bf} BF px, residency={obj._g_residency}, "
                f"chunk_bf={obj._chunk_bf}, compute device={cdev}"
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
    def device(self) -> torch.device:
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
            chunk = (
                self._G_cpu[s:e]
                .to(self._compute_device, non_blocking=True)
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
