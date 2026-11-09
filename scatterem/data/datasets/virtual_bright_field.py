from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.fft import fftfreq

from scatterem.data.acquisition import AcquisitionMetadata
from scatterem.data.detector_warp import bright_field_k
from scatterem.data.validation import ensure_valid_array
from scatterem.physics.fourier import phase_ramp

from ._meta import meta_sampling
from ._parent_link import ParentLink
from ._tensor_state import _MovesTensorState
from ._vbf_subsets import _bf_mask_geometry, _normalized_mean_pattern, _VbfSubsets
from .base import Dataset, _place_aberrations_on_device

if TYPE_CHECKING:
    # Annotations only: a real import is circular (dataset_4dstem.py imports
    # this module for DatasetVirtualBrightField4DStem).
    from .dataset_4dstem import Dataset4DStem


# One tensor object for the __init__ parameter defaults below (B008 bans the
# call there); the class-body defaults above are not arguments.
_EMPTY_TENSOR = torch.empty(0)


class DatasetVirtualBrightField4DStem(_VbfSubsets, _MovesTensorState, Dataset):
    """A virtual bright field 4D STEM dataset with metadata for electron diffraction patterns.

    Parameters
    ----------
    array : torch.Tensor
        Gathered bright-field diffraction data, shape ``(Ry, Rx, N_bf)``.
    name : str
        Dataset name.
    origin : numpy.typing.NDArray or tuple or list or float or int
        Real-space origin.
    sampling : numpy.typing.NDArray or tuple or list or float or int
        Real-space sampling (pixel size) per axis.
    units : list of str or tuple or list
        Units of each axis.
    parent_dataset : Dataset4DStem or None, optional
        4D-STEM dataset this virtual bright field was derived from.
    diffraction_pattern_mean_normalized : torch.Tensor, optional
        Max-normalized mean CBED used to build the bright-field mask.
    bright_field_mask : torch.Tensor, optional
        Boolean bright-field detector mask.
    bright_field_inds : torch.Tensor, optional
        Detector pixel indices inside the bright-field mask.
    bright_field_inds_centered : torch.Tensor, optional
        ``bright_field_inds`` centered on their mean.
    bright_field_inds_radial_order : torch.Tensor, optional
        Permutation sorting ``bright_field_inds`` by radius.
    bright_field_inds_ordered_by_radius : torch.Tensor, optional
        ``bright_field_inds`` sorted by radius.
    bright_field_inds_centered_ordered_by_radius : torch.Tensor, optional
        ``bright_field_inds_centered`` sorted by radius.
    k : torch.Tensor, optional
        Physical detector k-vectors of the bright-field pixels.
    qx_1d : torch.Tensor, optional
        1D Fourier frequencies along the scan x-axis.
    qy_1d : torch.Tensor, optional
        1D Fourier frequencies along the scan y-axis.
    signal_units : str, optional
        Units of the signal values.
    meta : AcquisitionMetadata or None, optional
        Acquisition metadata.
    astype_float32 : bool, optional
        If True (default), cast the array to ``float32``.
    fourier_shift_dim : tuple, optional
        Dimensions to ``ifftshift``; unused here (always None from
        :meth:`from_4dstem_dataset`).
    probe_index : int, optional
        Probe index for multi-probe datasets.
    device : torch.device or str or None, optional
        ``None`` keeps a torch tensor's device; other inputs become CPU.
    clip_neg_values : bool, optional
        If True (default), clip negative array values to zero.
    _token : object or None, optional
        Internal construction guard.
    """

    # A view derived from its parent dataset, not saved on its own: save() refuses.
    SERIAL_FIELDS = ()
    meta: AcquisitionMetadata | None = None
    bright_field_mask: torch.Tensor = torch.empty(0)
    bright_field_inds: torch.Tensor = torch.empty(0)
    bright_field_inds_centered: torch.Tensor = torch.empty(0)
    bright_field_inds_radial_order: torch.Tensor = torch.empty(0)
    bright_field_inds_ordered_by_radius: torch.Tensor = torch.empty(0)
    bright_field_inds_centered_ordered_by_radius: torch.Tensor = torch.empty(0)
    k: torch.Tensor = torch.empty(0)
    qx_1d: torch.Tensor = torch.empty(0)
    qy_1d: torch.Tensor = torch.empty(0)
    q_2d: torch.Tensor = torch.empty(0)
    _G: torch.Tensor = torch.empty(0)
    _direct_ptychography_phase_image: torch.Tensor | None = None
    _direct_ptychography_amplitude_image: torch.Tensor | None = None
    diffraction_pattern_mean_normalized: torch.Tensor = torch.empty(0)
    parent_dataset = ParentLink()

    @property
    def direct_ptychography_phase_image(self) -> torch.Tensor | None:
        """Cached direct-ptychography phase image, or None if not computed."""
        return self._direct_ptychography_phase_image

    @direct_ptychography_phase_image.setter
    def direct_ptychography_phase_image(self, value: torch.Tensor | None) -> None:
        self._direct_ptychography_phase_image = value

    @property
    def direct_ptychography_amplitude_image(self) -> torch.Tensor | None:
        """Cached direct-ptychography amplitude image, or None if not computed."""
        return self._direct_ptychography_amplitude_image

    @direct_ptychography_amplitude_image.setter
    def direct_ptychography_amplitude_image(self, value: torch.Tensor | None) -> None:
        self._direct_ptychography_amplitude_image = value

    @property
    def normalized_bright_field(self) -> torch.Tensor:
        """Array divided by the mean CBED intensity at each bright-field pixel."""
        inds = self.bright_field_inds_ordered_by_radius
        norm = self.diffraction_pattern_mean_normalized[inds[:, 0], inds[:, 1]]
        norm = norm[None, None, :]
        return self.array / norm

    def _summary_rows(self) -> dict[str, Any]:
        inds = self.bright_field_inds_ordered_by_radius
        rows = {
            "name": self.name,
            "shape": tuple(self.shape),
            "n_bright_field": int(inds.shape[0]) if inds.numel() else 0,
            "dtype": self.dtype,
            "device": self.device,
            "G_cached": bool(self._G.numel()),
        }
        if self.parent_dataset is not None:
            rows["parent"] = self.parent_dataset.name
        return rows

    def __init__(
        self,
        array: Tensor,
        name: str,
        origin: NDArray | tuple | list | float | int,
        sampling: NDArray | tuple | list | float | int,
        units: list[str] | tuple | list,
        parent_dataset: Dataset4DStem | None = None,
        diffraction_pattern_mean_normalized: torch.Tensor = _EMPTY_TENSOR,
        bright_field_mask: torch.Tensor = _EMPTY_TENSOR,
        bright_field_inds: torch.Tensor = _EMPTY_TENSOR,
        bright_field_inds_centered: torch.Tensor = _EMPTY_TENSOR,
        bright_field_inds_radial_order: torch.Tensor = _EMPTY_TENSOR,
        bright_field_inds_ordered_by_radius: torch.Tensor = _EMPTY_TENSOR,
        bright_field_inds_centered_ordered_by_radius: torch.Tensor = _EMPTY_TENSOR,
        k: torch.Tensor = _EMPTY_TENSOR,
        qx_1d: torch.Tensor = _EMPTY_TENSOR,
        qy_1d: torch.Tensor = _EMPTY_TENSOR,
        signal_units: str = "arb. units",
        meta: AcquisitionMetadata | None = None,
        astype_float32: bool = True,
        fourier_shift_dim: tuple | None = None,
        probe_index: int = 0,
        device: torch.device | str | None = None,
        clip_neg_values: bool = True,
        _token: object | None = None,
    ) -> None:
        super().__init__(
            array=array,
            name=name,
            origin=origin,
            sampling=sampling,
            units=units,
            signal_units=signal_units,
            _token=_token,
            device=device,
        )

        self.meta = meta
        _place_aberrations_on_device(meta, self._array.device)
        self.k = k
        self.qx_1d = qx_1d
        self.qy_1d = qy_1d

        self.q_2d = torch.stack(
            torch.meshgrid(qy_1d[0], qx_1d[1], indexing="ij"), dim=0
        )
        self.fourier_shift_dim = fourier_shift_dim
        self.probe_index = probe_index
        self.parent_dataset = parent_dataset
        self.diffraction_pattern_mean_normalized = diffraction_pattern_mean_normalized
        self.bright_field_mask = bright_field_mask
        self.bright_field_inds = bright_field_inds
        self.bright_field_inds_centered = bright_field_inds_centered
        self.bright_field_inds_radial_order = bright_field_inds_radial_order
        self.bright_field_inds_ordered_by_radius = bright_field_inds_ordered_by_radius
        self.bright_field_inds_centered_ordered_by_radius = (
            bright_field_inds_centered_ordered_by_radius
        )

        self._shape = self._array.shape
        if astype_float32:
            self._array = self._array.to(torch.float32)
        if fourier_shift_dim is not None:
            self._array = torch.fft.ifftshift(self._array, dim=fourier_shift_dim)
        if clip_neg_values:
            self._array[self._array < 0] = 0

        self._array3d = self._array

        self._total_intensity = None

    @classmethod
    def from_4dstem_dataset(
        cls,
        dataset: Dataset4DStem,
        bright_field_mask_threshold: float = 0.3,
        num_indices_for_bright_field_mask: int = 625,
        device: torch.device | None = None,  # will use dataset.device if not provided
    ) -> DatasetVirtualBrightField4DStem:
        """Validate and create a DatasetVirtualBrightField4DStem from a Dataset4DStem.

        Parameters
        ----------
        dataset : Dataset4DStem
            The dataset to validate and create a DatasetVirtualBrightField4DStem from.
        bright_field_mask_threshold : float, optional
            Fraction of the mean-CBED reference level above which a detector
            pixel is in the bright-field mask. The reference level is the
            maximum, capped at 1.5 times the maximum of the 3 x 3-median-
            filtered mean CBED, so a few hot pixels cannot lift it.
        num_indices_for_bright_field_mask : int, optional
            Approximate number of detector pixels (a square region) used to
            estimate the mean CBED for the mask.
        device : torch.device or None, optional
            The device to create the DatasetVirtualBrightField4DStem on;
            defaults to ``dataset.device``.

        Returns
        -------
        DatasetVirtualBrightField4DStem
            A Dataset object with the validated array and metadata.
        """
        if device is None:
            device = dataset.device
        n = int(np.ceil(np.sqrt(num_indices_for_bright_field_mask)))
        diff_mean = dataset.array[:n, :n].mean((0, 1))
        diff_mean = _normalized_mean_pattern(diff_mean)
        geo = _bf_mask_geometry(diff_mean, bright_field_mask_threshold)
        bright_field_mask = geo["bright_field_mask"]
        bright_field_inds = geo["bright_field_inds"]
        bright_field_inds_centered = geo["bright_field_inds_centered"]
        bright_field_inds_radial_order = geo["bright_field_inds_radial_order"]
        bright_field_inds_ordered_by_radius = geo["bright_field_inds_ordered_by_radius"]
        bright_field_inds_centered_ordered_by_radius = geo[
            "bright_field_inds_centered_ordered_by_radius"
        ]

        bright_field = dataset.array[
            :,
            :,
            bright_field_inds_ordered_by_radius[:, 0],
            bright_field_inds_ordered_by_radius[:, 1],
        ]

        validated_array = ensure_valid_array(bright_field, device=device)
        _ndim = validated_array.ndim
        # The dataset's sampling, not ``meta``'s mirror: the dataset's is the
        # source of truth, and duck-typed metadata may have none (see ``_meta``).
        sampling = torch.as_tensor(dataset.sampling[-2:], device=device)[None]
        centered = bright_field_inds_centered_ordered_by_radius.to(device)
        k = bright_field_k(centered, sampling, dataset.meta)

        upsample_int = 1
        Qx = fftfreq(
            dataset.shape[1] * upsample_int,
            dataset.sampling[1] / upsample_int,
            dtype=torch.float32,
            device=dataset.device,
        )
        Qy = fftfreq(
            dataset.shape[0] * upsample_int,
            dataset.sampling[0] / upsample_int,
            dtype=torch.float32,
            device=dataset.device,
        )

        # Set defaults if None
        _name = f"vBF of {dataset.name}"
        dso = dataset.origin
        _origin = dso if dso is not None else np.zeros(_ndim)
        # ``meta_sampling`` handles both metadata families; ``meta.sampling`` does not.
        _ms = meta_sampling(dataset.meta)
        _sampling = dataset.sampling if dataset.sampling is not None else _ms
        if _sampling is None:
            _sampling = np.ones(_ndim)
        # The dataset's 4 units, else its meta's, less the last: the vBF's third
        # axis is one bright-field pixel index standing in for both detector axes.
        _all_units = dataset.units
        if _all_units is None:
            _all_units = getattr(dataset.meta, "units", None)
        _units = ["pixels"] * _ndim if _all_units is None else _all_units[:-1]

        return cls(
            array=validated_array,
            name=_name,
            origin=_origin,
            sampling=_sampling,
            units=_units,
            signal_units=dataset.signal_units,
            _token=cls._token,
            meta=dataset.meta,
            device=device,
            parent_dataset=dataset,
            diffraction_pattern_mean_normalized=diff_mean,
            bright_field_mask=bright_field_mask,
            bright_field_inds=bright_field_inds,
            bright_field_inds_centered=bright_field_inds_centered,
            bright_field_inds_radial_order=bright_field_inds_radial_order,
            bright_field_inds_ordered_by_radius=bright_field_inds_ordered_by_radius,
            bright_field_inds_centered_ordered_by_radius=bright_field_inds_centered_ordered_by_radius,
            # NOT dataset.fourier_shift_dim. That tuple names the parent's DETECTOR axes
            # (conventionally (2, 3)) of a 4-D (Ry,Rx,Ky,Kx) array, but this array is 3-D
            # (Ry,Rx,N_bf) -- the detector axes are flattened away by gathering the
            # bright-field pixels -- so shifting dim 3 is out of range and raised
            # IndexError for every detector size. The parent shifts its own array in
            # __init__, before those pixels are gathered, so there is nothing left to do.
            fourier_shift_dim=None,
            clip_neg_values=False,
            k=k,
            qx_1d=Qx,
            qy_1d=Qy,
        )

    def clear_memory(self) -> None:
        """
        Clear the memory of the dataset.
        """
        self._G = torch.empty(0)
        # Clear CUDA memory if using GPU
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        # Run garbage collection
        gc.collect()

    @property
    def G(self) -> torch.Tensor:
        """
        The Fourier transform of the bright field image.

        Returns:
            torch.Tensor: The Fourier transform of the bright field image
        """
        if self._G.numel() == 0:
            self._G = torch.fft.fft2(self.array, dim=(0, 1), norm="ortho")
        return self._G

    @property
    def n_bright_field(self) -> int:
        """Number of bright-field detector pixels (the chunkable axis)."""
        return int(self.bright_field_inds_ordered_by_radius.shape[0])

    # ---- alignment-shift overlay -------------------------------------------
    # The bright-field-shift self-calibration corrects G with per-pixel phase
    # ramps exp(-2πi(dx·kx + dy·ky)). Ramps compose ADDITIVELY in (dy, dx), so
    # instead of mutating the (large, possibly shared) G cache, a cumulative
    # (N_BF, 2) shift table is kept and the ramp is applied AT FETCH TIME by
    # get_G_chunk / get_G_columns. Exact for every residency, and the pristine
    # G is never touched.
    _align_shifts: torch.Tensor | None = None

    def add_alignment_shifts(self, shifts_px: torch.Tensor) -> None:
        """Accumulate per-BF-pixel alignment shifts.

        Parameters
        ----------
        shifts_px : torch.Tensor
            Per-BF-pixel ``(dy, dx)`` shifts in SCAN pixels, shape
            ``(N_bf, 2)``, ordered like the G last axis (radius order).
        """
        shifts_px = shifts_px.to(device=self.device, dtype=torch.float32)
        if self._align_shifts is None:
            self._align_shifts = shifts_px.clone()
        else:
            self._align_shifts = self._align_shifts + shifts_px

    def clear_alignment_shifts(self) -> None:
        """Drop the accumulated alignment-shift overlay."""
        self._align_shifts = None

    def _apply_alignment(self, chunk: torch.Tensor, idx) -> torch.Tensor:
        """Multiply an owned G chunk by the cumulative alignment ramp for its
        BF pixels (``idx``: slice or index tensor into the radius order).

        Cost note: while an overlay is active (only INSIDE the bright-field-shift
        self-calibration loop, which clears it on exit), every fetch pays one exp,
        one transposing copy of the ramp and one multiply over the chunk. Outside
        that loop the overlay is None and this is a no-op.
        """
        if self._align_shifts is None:
            return chunk
        # float32 like the shift table; contiguous (a strided multiply moves CPU bits)
        sh = self._align_shifts[idx].to(chunk.device)
        ramp = phase_ramp(
            chunk.shape[:2], sh, dtype=chunk.dtype, ramp_dtype=torch.float32
        )
        return chunk * ramp.permute(1, 2, 0).contiguous()

    def get_G_chunk(self, s: int, e: int) -> torch.Tensor:
        """Return an owned, contiguous slice of G with alignment applied.

        Never a view of the cached ``_G``: a partial last-axis slice is a
        non-contiguous view (in-place ops write through), a full-range slice
        is a contiguous alias. The copy made here replaces the one
        ``correct_aberrations_inplace``'s ``.contiguous()`` would have made on
        a non-contiguous view — net zero extra copies.

        Parameters
        ----------
        s : int
            Start index (inclusive) along the last (bright-field-pixel) axis.
        e : int
            End index (exclusive) along the last axis.

        Returns
        -------
        torch.Tensor
            ``G[..., s:e]`` as an owned tensor the caller may mutate freely
            (with the alignment overlay applied, when one is active).
        """
        chunk = self.G[..., s:e]
        chunk = chunk.clone() if chunk.is_contiguous() else chunk.contiguous()
        return self._apply_alignment(chunk, slice(s, e))

    def get_G_columns(self, idx: torch.Tensor) -> torch.Tensor:
        """Return owned G columns for an arbitrary index tensor.

        Random access is cheap here — G (or the vBF stack it derives from) is
        memory-resident, not on disk. Advanced indexing always copies, so the
        result is owned without a clone.

        Parameters
        ----------
        idx : torch.Tensor
            Index tensor (radius order) selecting columns along the last axis.

        Returns
        -------
        torch.Tensor
            G columns at ``idx``, with the alignment overlay applied.
        """
        idx = torch.as_tensor(idx, dtype=torch.long, device=self.device)
        return self._apply_alignment(self.G[..., idx], idx)

    def get_q_1d(
        self, shape: tuple[int, ...] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute 1D Fourier frequencies for a (possibly upsampled) shape.

        Parameters
        ----------
        shape : tuple of int, optional
            Leading ``(Ry, Rx)`` dimensions; defaults to ``self.shape``.
            Only the first two entries are used; the bright-field pixel count
            in the dataset's third dimension does not affect these axes.

        Returns
        -------
        tuple of torch.Tensor
            ``(Qy, Qx)`` 1D frequency axes.
        """
        if shape is None:
            shape = self.shape
        upsample_x = shape[1] / self.shape[1]
        upsample_y = shape[0] / self.shape[0]
        Qx = fftfreq(
            shape[1],
            self.sampling[1] / upsample_x,
            dtype=torch.float32,
            device=self.device,
        )
        Qy = fftfreq(
            shape[0],
            self.sampling[0] / upsample_y,
            dtype=torch.float32,
            device=self.device,
        )

        return Qy, Qx
