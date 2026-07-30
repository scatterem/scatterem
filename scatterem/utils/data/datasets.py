import copy
import gc
import warnings
from typing import Any, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from numpy.typing import DTypeLike, NDArray
from torch import Tensor
from torch.fft import fftfreq
from torch.utils.data import Dataset as TorchDataset

import scatterem.vis as vis
from scatterem.io.store import Serializable
from scatterem.utils.data._pretty import PrettyDatasetMixin
from scatterem.utils.data.data_classes import Aberrations, Metadata4dstem
from scatterem.utils.physics import electron_wavelength
from scatterem.utils.data.disk_fit import fit_bright_field_disk
from scatterem.utils.data.validation import (
    ensure_valid_array,
    validate_ndinfo,
    validate_units,
)



class _BrightFieldCropBox(NamedTuple):
    crop_slice: tuple
    radius: int  # crop half-width, clamped to the detector
    center: NDArray  # [y, x] bright-field center
    y0: int
    x0: int
    radius_ceil: float  # ceil of the detected radius, before clamping


class _PhysicsMetadata(NamedTuple):
    meta: Metadata4dstem
    sampling: tuple[float, float, float, float]
    origin: NDArray


def _metadata4dstem_from_physics(
    array_shape: Sequence[int],
    energy: float,
    semiconvergence_angle: float | None,
    scan_step: Union[float, tuple, list, NDArray, None],
    reciprocal_step: Union[float, tuple, list, NDArray, None] = None,
    rotation: float = 0.0,
    aberrations: Optional[Aberrations] = None,
) -> _PhysicsMetadata:
    """Build a ``Metadata4dstem`` from acquisition physics.

    The "physics convenience path" used by ``Dataset4dstem.from_array``; also
    shared with the downloadable public datasets (``datasets/public/base.py``).

    ``scan_step`` and ``reciprocal_step`` each accept a scalar or a length-2
    pair; ``reciprocal_step`` defaults to the ``(1.0, 1.0)`` placeholder that a
    later calibration (e.g. ``calibrate_reciprocal_from_bright_field``)
    overwrites.

    Returns:
        A ``_PhysicsMetadata(meta, sampling, origin)`` where ``sampling`` is
        ``(dr0, dr1, dk0, dk1)`` and ``origin`` is the array center.
    """
    shape = np.array(array_shape)

    def _as_pair(v, default):
        v = default if v is None else v
        if hasattr(v, "__len__"):
            return float(v[0]), float(v[1])
        return float(v), float(v)

    dr0, dr1 = _as_pair(scan_step, 1.0)
    dk0, dk1 = _as_pair(reciprocal_step, 1.0)
    sampling = (dr0, dr1, dk0, dk1)
    meta = Metadata4dstem(
        energy=energy,
        semiconvergence_angle=semiconvergence_angle,
        sampling=sampling,
        shape=shape,
        rotation=rotation,
        aberrations=aberrations,
    )
    return _PhysicsMetadata(meta=meta, sampling=sampling, origin=shape / 2.0)


def _place_aberrations_on_device(meta, device: torch.device) -> None:
    """Move ``meta.aberrations.array`` onto ``device``, in place.

    Warp kernels take the aberration array and the diffraction array in the
    same launch, so a CPU-resident aberration tensor on a CUDA dataset makes
    direct_ptychography fail with an opaque device-mismatch error unless
    determine_aberrations_ happened to run first and move it as a side effect.

    Mutates the caller's ``Aberrations`` rather than rebinding
    ``meta.aberrations``: the vBF shares ``meta`` by reference with its parent
    dataset so that determine_aberrations_ writing into ``vBF.meta.aberrations``
    is visible on the original -- a load-bearing convention a defensive copy
    would break.
    """
    aberrations = getattr(meta, "aberrations", None)
    if aberrations is not None and getattr(aberrations, "array", None) is not None:
        aberrations.array = aberrations.array.to(device)


class Dataset(PrettyDatasetMixin, TorchDataset, Serializable):
    #: Persisted by scatterem.io.store. `device` is deliberately absent: a file
    #: that pins a GPU will not open on someone else's machine.
    SERIAL_FIELDS = (
        "array",
        "name",
        "origin",
        "sampling",
        "units",
        "signal_units",
    )

    @classmethod
    def _from_fields(cls, fields):
        return cls(**fields, _token=cls._token)

    """
    An array plus the calibration needed to interpret its axes.

    Each axis carries an origin, a pitch and a unit, so a position in the array
    can be turned into a physical coordinate without the caller tracking the
    conversion. Values have their own unit, separate from the axes'.

    Attributes:
        array: the data itself.
        name: label used in reprs and figure titles.
        origin: coordinate of index zero on each axis.
        sampling: pitch per axis, in ``units``.
        units: one unit string per axis.
        signal_units: unit of the stored values.
    """

    _token = object()

    def __init__(
        self,
        array: Any,  # Input can be array-like
        name: str,
        origin: Union[NDArray, tuple, list, float, int],
        sampling: Union[NDArray, tuple, list, float, int],
        units: Union[list[str], tuple, list],
        signal_units: str = "arb. units",
        _token: object | None = None,
        device: torch.device = torch.device("cpu"),
    ):
        if _token is not self._token:
            raise RuntimeError("Use Dataset.from_array() to instantiate this class.")

        self._array: Tensor = ensure_valid_array(array, device=device)
        self.name = name
        self.origin = origin
        self.sampling = sampling
        self.units = units
        self.signal_units = signal_units

    @classmethod
    def from_array(
        cls,
        array: Any,  # Input can be array-like
        name: str | None = None,
        origin: Union[NDArray, tuple, list, float, int] | None = None,
        sampling: Union[NDArray, tuple, list, float, int] | None = None,
        units: Union[list[str], tuple, list] | None = None,
        signal_units: str = "arb. units",
    ) -> "Dataset":
        """
        Validates and creates a Dataset from an array.

        Parameters
        ----------
        array: Any
            The array to validate and create a Dataset from.
        name: str | None
            The name of the Dataset.
        origin: Union[NDArray, tuple, list, float, int] | None
            The origin of the Dataset.
        sampling: Union[NDArray, tuple, list, float, int] | None
            The sampling of the Dataset.
        units: Union[list[str], tuple, list] | None
            The units of the Dataset.
        signal_units: str
            The units of the signal.

        Returns
        -------
        Dataset
            The container, with any unset axis metadata filled in from the
            array's rank.
        """
        tensor = ensure_valid_array(array)
        rank = tensor.ndim
        return cls(
            array=tensor,
            name=f"{rank}d dataset" if name is None else name,
            origin=np.zeros(rank) if origin is None else origin,
            sampling=np.ones(rank) if sampling is None else sampling,
            units=["pixels"] * rank if units is None else units,
            signal_units=signal_units,
            _token=cls._token,
        )

    # --- Properties ---
    @property
    def array(self) -> Tensor:
        """The underlying n-dimensional array data."""
        return self._array

    @array.setter
    def array(self, value: Tensor) -> None:
        self._array = ensure_valid_array(
            value, dtype=self.dtype, ndim=self.ndim, device=self.device
        )

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = str(value)

    @property
    def origin(self) -> NDArray:
        return self._origin

    @origin.setter
    def origin(self, value: Union[NDArray, tuple, list, float, int]) -> None:
        self._origin = value

    @property
    def sampling(self) -> NDArray:
        return self._sampling

    @sampling.setter
    def sampling(self, value: Union[NDArray, tuple, list, float, int]) -> None:
        self._sampling = value

    @property
    def units(self) -> list[str]:
        return self._units

    @units.setter
    def units(self, value: Union[list[str], tuple, list]) -> None:
        self._units = validate_units(value, self.ndim)

    @property
    def signal_units(self) -> str:
        return self._signal_units

    @signal_units.setter
    def signal_units(self, value: str) -> None:
        self._signal_units = str(value)

    # --- Derived Properties ---
    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    @property
    def ndim(self) -> int:
        return self.array.ndim

    @property
    def dtype(self) -> torch.dtype:
        return self.array.dtype

    @property
    def device(self) -> torch.device:
        """
        Outputting a string is likely temporary -- once we have our use cases we can
        figure out a more permanent device solution that enables easier translation between

        """
        return self.array.device

    @device.setter
    def device(self, value: torch.device) -> None:
        self.array = self.array.to(value)

    # --- Summaries ---
    def _summary_rows(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": tuple(self.shape),
            "dtype": self.dtype,
            "device": self.device,
            "origin": self.origin,
            "sampling": self.sampling,
            "units": self.units,
            "signal_units": self.signal_units,
        }

    # --- Methods ---
    def copy(self) -> "Dataset":
        """
        Copies Dataset.

        Parameters
        ----------
        copy_attributes: bool
            If True, copies non-standard attributes. Standard attributes (array, metadata)
            are always deep-copied.
        """
        # Metadata arrays (origin, sampling) are numpy, use copy()
        # Units list is copied by slicing
        new_dataset = type(self).from_array(
            array=self.array.clone(),
            name=self.name,
            origin=copy.deepcopy(self.origin),
            sampling=copy.deepcopy(self.sampling),
            units=self.units[:],
            signal_units=self.signal_units,
        )

        return new_dataset

    def mean(self, axes: Optional[tuple[int, ...]] = None) -> Any:
        """
        Computes and returns mean of the data array.

        Parameters
        ----------
        axes: tuple, optional
            Axes over which to compute mean. If None specified, mean of all elements is computed.

        Returns
        --------
        mean: scalar or array (np.ndarray or cp.ndarray)
            Mean of the data.
        """
        return self.array.mean(axis=axes)




    def crop(
        self,
        crop_widths: tuple[tuple[int, int], ...],
        axes: Optional[Union[int, tuple[int, ...]]] = None,
        modify_in_place: bool = False,
    ) -> Optional["Dataset"]:
        """Trim elements from the ends of axes, keeping coordinates consistent.

        ``origin`` moves by ``+before * sampling`` on each cropped axis, so a
        physical coordinate computed from the cropped array still lands on the
        same feature -- the counterpart of the shift :meth:`pad` applies. The
        implementation this replaces left ``origin`` alone, which silently moved
        the data in physical space. ``sampling`` is unchanged.

        Args:
            crop_widths: a ``(before, after)`` pair per entry of ``axes``, giving
                how many elements to remove from each end. A single pair is
                accepted when ``axes`` is a single axis.
            axes: which axes to crop, defaulting to every axis -- in which case
                ``crop_widths`` must have one pair per axis. Negative indices
                count from the end.
            modify_in_place: crop ``self`` and return ``None`` rather than a copy.

        Returns:
            The cropped copy, or ``None`` when ``modify_in_place`` is set.

        Raises:
            ValueError: if the widths and axes do not correspond, an axis is
                repeated or out of range, a width is negative, or a crop would
                consume an entire axis.
        """
        if axes is None:
            if len(crop_widths) != self.ndim:
                raise ValueError(
                    f"got {len(crop_widths)} crop widths for {self.ndim} axes; when "
                    f"axes is omitted every axis needs one"
                )
            selected = tuple(range(self.ndim))
            widths = tuple(crop_widths)
        else:
            if isinstance(axes, (int, np.integer)):
                selected = (int(axes),)
                # a lone axis may be given a bare (before, after)
                widths = (
                    (crop_widths,)
                    if len(crop_widths) == 2
                    and all(isinstance(v, (int, np.integer)) for v in crop_widths)
                    else tuple(crop_widths)
                )
            else:
                selected = tuple(int(a) for a in axes)
                widths = tuple(crop_widths)
            if len(widths) != len(selected):
                raise ValueError(
                    f"got {len(widths)} crop widths for {len(selected)} axes; they "
                    f"must correspond one to one"
                )

        selected = tuple(a + self.ndim if a < 0 else a for a in selected)
        for axis in selected:
            if not 0 <= axis < self.ndim:
                raise ValueError(
                    f"axis {axis} is out of range for a {self.ndim}-dimensional dataset"
                )
        if len(set(selected)) != len(selected):
            raise ValueError(f"axes must be distinct; got {axes!r}")

        per_axis = {}
        for axis, width in zip(selected, widths):
            before, after = (int(width[0]), int(width[1]))
            if before < 0 or after < 0:
                raise ValueError(
                    f"crop widths must not be negative; got {(before, after)} for "
                    f"axis {axis}"
                )
            if before + after >= self.shape[axis]:
                raise ValueError(
                    f"cropping {before}+{after} from axis {axis} would consume all "
                    f"{self.shape[axis]} of it"
                )
            per_axis[axis] = (before, after)

        slices = tuple(
            (
                slice(per_axis[axis][0], length - per_axis[axis][1])
                if axis in per_axis
                else slice(None)
            )
            for axis, length in enumerate(self.shape)
        )
        origin = tuple(
            float(o) + per_axis.get(axis, (0, 0))[0] * float(s)
            for axis, (o, s) in enumerate(
                zip(np.atleast_1d(self.origin), np.atleast_1d(self.sampling))
            )
        )

        target = self if modify_in_place else self.copy()
        # clone(): a slice is a view, and writing it back to a dataset that shares
        # storage with the original has bitten this method before.
        target.array = self.array[slices].clone()
        target.origin = origin
        return None if modify_in_place else target






def _bf_mask_geometry(
    diff_mean: torch.Tensor, bright_field_mask_threshold: float
) -> dict[str, torch.Tensor]:
    """Bright-field mask + index/ordering tensors from a (max-normalized) mean
    CBED. Single source of truth shared by the eager vBF constructor and the
    out-of-core streaming provider — the two MUST produce identical masks,
    orderings and k-vectors for their reconstructions to be interchangeable.
    """
    bright_field_mask = diff_mean > bright_field_mask_threshold
    bright_field_inds = torch.argwhere(bright_field_mask)
    bright_field_inds_centered = (
        bright_field_inds.float() - torch.mean(bright_field_inds.float(), dim=0)[None]
    )
    bright_field_inds_radial_order = torch.argsort(
        torch.sum(bright_field_inds_centered**2, dim=1)
    )
    return {
        "bright_field_mask": bright_field_mask,
        "bright_field_inds": bright_field_inds,
        "bright_field_inds_centered": bright_field_inds_centered,
        "bright_field_inds_radial_order": bright_field_inds_radial_order,
        "bright_field_inds_ordered_by_radius": bright_field_inds[
            bright_field_inds_radial_order
        ],
        "bright_field_inds_centered_ordered_by_radius": bright_field_inds_centered[
            bright_field_inds_radial_order
        ],
    }


class DatasetVirtualBrightField4dstem(Dataset):
    """A virtual bright field 4D STEM dataset with metadata for electron diffraction patterns."""

    meta: Optional[Metadata4dstem] = None
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
    parent_dataset: Optional["Dataset4dstem"] = None

    @property
    def direct_ptychography_phase_image(self) -> torch.Tensor | None:
        return self._direct_ptychography_phase_image

    @direct_ptychography_phase_image.setter
    def direct_ptychography_phase_image(self, value: torch.Tensor | None) -> None:
        self._direct_ptychography_phase_image = value

    @property
    def direct_ptychography_amplitude_image(self) -> torch.Tensor | None:
        return self._direct_ptychography_amplitude_image

    @direct_ptychography_amplitude_image.setter
    def direct_ptychography_amplitude_image(self, value: torch.Tensor | None) -> None:
        self._direct_ptychography_amplitude_image = value

    @property
    def normalized_bright_field(self) -> torch.Tensor:
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
        origin: Union[NDArray, tuple, list, float, int],
        sampling: Union[NDArray, tuple, list, float, int],
        units: Union[list[str], tuple, list],
        parent_dataset: Optional["Dataset4dstem"] = None,
        diffraction_pattern_mean_normalized: torch.Tensor = torch.empty(0),
        bright_field_mask: torch.Tensor = torch.empty(0),
        bright_field_inds: torch.Tensor = torch.empty(0),
        bright_field_inds_centered: torch.Tensor = torch.empty(0),
        bright_field_inds_radial_order: torch.Tensor = torch.empty(0),
        bright_field_inds_ordered_by_radius: torch.Tensor = torch.empty(0),
        bright_field_inds_centered_ordered_by_radius: torch.Tensor = torch.empty(0),
        k: torch.Tensor = torch.empty(0),
        qx_1d: torch.Tensor = torch.empty(0),
        qy_1d: torch.Tensor = torch.empty(0),
        signal_units: str = "arb. units",
        meta: Optional[Metadata4dstem] = None,
        astype_float32: bool = True,
        fourier_shift_dim: Tuple = None,
        probe_index: int = 0,
        device: torch.device = torch.device("cpu"),
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
        dataset: "Dataset4dstem",
        verbosity: int = 0,
        bright_field_mask_threshold: float = 0.3,
        num_indices_for_bright_field_mask: int = 625,
        device: Optional[
            torch.device
        ] = None,  # will use dataset.device if not provided
    ) -> "DatasetVirtualBrightField4dstem":
        """
        Validates and creates a DatasetVirtualBrightField4dstem from a Dataset4dstem.

        Parameters
        ----------
        dataset: Dataset4dstem
            The dataset to validate and create a DatasetVirtualBrightField4dstem from.
        verbosity: int
            The verbosity of the validation.
        bright_field_mask_threshold: float
            The threshold for the bright field mask.
        num_indices_for_bright_field_mask: Optional[int]
            The number of indices for the bright field mask.
        device: torch.device
            The device to create the DatasetVirtualBrightField4dstem on.
        verbosity: int
            The verbosity of the validation.
        units: Union[list[str], tuple, list] | None
            The units of the Dataset.
        signal_units: str
            The units of the signal.

        Returns
        -------
        DatasetVirtualBrightField4dstem
            A Dataset object with the validated array and metadata.
        """
        if device is None:
            device = dataset.device
        n = int(np.ceil(np.sqrt(num_indices_for_bright_field_mask)))
        diff_mean = dataset.array[:n, :n].mean((0, 1))
        diff_mean /= diff_mean.max()
        geo = _bf_mask_geometry(diff_mean, bright_field_mask_threshold)
        bright_field_mask = geo["bright_field_mask"]
        if verbosity > 0:
            fig, ax = vis.show_2d(
                bright_field_mask.float(), cbar=True, title="Virtual Bright Field Mask"
            )
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
        sampling = torch.as_tensor(dataset.meta.sampling[-2:], device=device)[None]
        k = (
            bright_field_inds_centered_ordered_by_radius.to(device)
            * sampling.expand_as(bright_field_inds_centered_ordered_by_radius)
        ).to(torch.float32)

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
        _sampling = (
            dataset.sampling
            if dataset.sampling is not None
            else (
                dataset.meta.sampling
                if dataset.meta is not None and dataset.meta.sampling is not None
                else np.ones(_ndim)
            )
        )
        _units = (
            dataset.units[:-1]
            if dataset.units is not None
            else (
                dataset.meta.units
                if dataset.meta is not None and dataset.meta.units is not None
                else ["pixels"] * _ndim
            )
        )

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
    _align_shifts: Optional[torch.Tensor] = None

    def add_alignment_shifts(self, shifts_px: torch.Tensor) -> None:
        """Accumulate per-BF-pixel (dy, dx) shifts in SCAN pixels, ordered like
        the G last axis (radius order)."""
        shifts_px = shifts_px.to(device=self.device, dtype=torch.float32)
        if self._align_shifts is None:
            self._align_shifts = shifts_px.clone()
        else:
            self._align_shifts = self._align_shifts + shifts_px

    def clear_alignment_shifts(self) -> None:
        self._align_shifts = None

    def _apply_alignment(self, chunk: torch.Tensor, idx) -> torch.Tensor:
        """Multiply an owned G chunk by the cumulative alignment ramp for its
        BF pixels (``idx``: slice or index tensor into the radius order).

        Cost note: while an overlay is active (only DURING a bright-field-shift
        fit), every fetch pays one exp+multiply over the chunk — comparable to
        the aberration-correction kernel that follows it, and only the
        "combined" method's autofocus stage runs hot with it. Outside fits the
        overlay is None and this is a no-op.
        """
        if self._align_shifts is None:
            return chunk
        sh = self._align_shifts[idx]
        ny, nx = int(chunk.shape[0]), int(chunk.shape[1])
        kx = torch.fft.fftfreq(nx, device=chunk.device).reshape(1, -1, 1)
        ky = torch.fft.fftfreq(ny, device=chunk.device).reshape(-1, 1, 1)
        ramp = torch.exp(
            -2j * np.pi * (sh[:, 1][None, None, :] * kx + sh[:, 0][None, None, :] * ky)
        )
        return chunk * ramp

    def get_G_chunk(self, s: int, e: int) -> torch.Tensor:
        """``G[..., s:e]`` as an OWNED tensor the caller may mutate freely
        (with the alignment overlay applied, when one is active).

        Never a view of the cached ``_G``: a partial last-axis slice is a
        non-contiguous view (in-place ops write through), a full-range slice
        is a contiguous alias. The copy made here replaces the one
        ``correct_aberrations_inplace``'s ``.contiguous()`` would have made on
        a non-contiguous view — net zero extra copies.
        """
        chunk = self.G[..., s:e]
        chunk = chunk.clone() if chunk.is_contiguous() else chunk.contiguous()
        return self._apply_alignment(chunk, slice(s, e))

    def get_G_columns(self, idx: torch.Tensor) -> torch.Tensor:
        """G columns for an ARBITRARY index tensor (radius order), owned, with
        the alignment overlay applied. Random access is cheap here — G (or the
        vBF stack it derives from) is memory-resident, not on disk. Advanced
        indexing always copies, so the result is owned without a clone."""
        idx = torch.as_tensor(idx, dtype=torch.long, device=self.device)
        return self._apply_alignment(self.G[..., idx], idx)

    def get_q_1d(
        self, shape: tuple[int, int] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
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


class Dataset4dstem(Dataset):
    SERIAL_FIELDS = Dataset.SERIAL_FIELDS + ("meta",)
    SERIAL_NESTED = {"meta": Metadata4dstem}

    @classmethod
    def _from_fields(cls, fields):
        # normalize=False and clip_neg_values=False: the saved array was already
        # normalized on its way in, and running the constructor's normalization a
        # second time divides it again. The factor comes out near 1.0 for an
        # already-normalized array, so the damage is a ~1e-7 drift rather than an
        # obvious break -- the same reason the crop() aliasing bug went unnoticed.
        return cls(
            **fields,
            normalize=False,
            clip_neg_values=False,
            _token=cls._token,
        )

    """A 4D STEM dataset with metadata for electron diffraction patterns."""

    meta: "Metadata4dstem" = None
    _is_cropped: bool = False
    _is_bright_field: bool = False
    _direct_ptychography_phase_image: torch.Tensor = torch.empty(0)
    _direct_ptychography_amplitude_image: torch.Tensor = torch.empty(0)
    _tilt_corrected_dark_field_image: torch.Tensor = torch.empty(0)
    dose_per_probe_unnormalized: float = None
    dose_per_probe_normalized: float = None

    @property
    def direct_ptychography_phase_image(self) -> torch.Tensor:
        return self._direct_ptychography_phase_image

    @direct_ptychography_phase_image.setter
    def direct_ptychography_phase_image(self, value: torch.Tensor) -> None:
        self._direct_ptychography_phase_image = value

    @property
    def direct_ptychography_amplitude_image(self) -> torch.Tensor:
        return self._direct_ptychography_amplitude_image

    @direct_ptychography_amplitude_image.setter
    def direct_ptychography_amplitude_image(self, value: torch.Tensor) -> None:
        self._direct_ptychography_amplitude_image = value

    @property
    def tilt_corrected_dark_field_image(self) -> torch.Tensor:
        return self._tilt_corrected_dark_field_image

    @tilt_corrected_dark_field_image.setter
    def tilt_corrected_dark_field_image(self, value: torch.Tensor) -> None:
        self._tilt_corrected_dark_field_image = value

    def __init__(
        self,
        array: Tensor,
        name: str,
        origin: Union[NDArray, tuple, list, float, int],
        sampling: Union[NDArray, tuple, list, float, int],
        units: Union[list[str], tuple, list],
        signal_units: str = "arb. units",
        meta: "Metadata4dstem" = None,
        transform_to_amplitudes: bool = False,
        astype_float32: bool = True,
        fourier_shift_dim: Tuple = None,
        probe_index: int = 0,
        device: torch.device = torch.device("cpu"),
        normalize: bool = True,
        clip_neg_values: bool = True,
        copy: bool = True,
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
        self.transform_to_amplitudes = transform_to_amplitudes
        self.fourier_shift_dim = fourier_shift_dim
        self.probe_index = probe_index
        # Default bright-field mask threshold used by determine_aberrations_,
        # direct_ptychography, tilt_corrected_dark_field and fused_full_field
        # when the caller omits the kwarg. 0.3 matches the reconstruction
        # methods' historical default; override per-dataset as needed.
        self.bright_field_mask_threshold = 0.3

        self._shape = self._array.shape
        # Track whether we already hold a private buffer. `.to(float32)` copies only
        # when the dtype actually changes, and fftshift always returns a new tensor.
        _private = False
        if astype_float32:
            _private = self._array.dtype != torch.float32
            self._array = self._array.to(torch.float32)
        if fourier_shift_dim is not None:
            self._array = torch.fft.fftshift(self._array, dim=fourier_shift_dim)
            _private = True
        # Everything from here on mutates self._array IN PLACE (the clip below,
        # sqrt_(), and the normalization further down). If it still shares storage
        # with the buffer the caller handed us, those writes silently corrupt the
        # caller's array: a float32 numpy cube passed to from_array came back
        # divided by normalization_const. Integer cubes were accidentally safe
        # because the dtype conversion copied. Take one copy here instead.
        if copy and not _private and (clip_neg_values or transform_to_amplitudes or normalize):
            self._array = self._array.clone()

        if clip_neg_values:
            self._array[self._array < 0] = 0

        if transform_to_amplitudes:
            self._array.sqrt_()

        # Total intensity is a property of the DATA, so it must be captured HERE,
        # before the normalization below. Reading it afterwards (the old behaviour, via
        # the lazily-evaluated `total_intensity` property seeded to None) summed the
        # already-normalized array and made `fluence` wrong by exactly
        # 1/normalization_const on every dataset with the default normalize=True.
        # Accumulate in float64 per outer slice, matching the property's own precision
        # strategy, and reuse the one sum for both quantities.
        _total_unnormalized = 0.0
        for _i in range(self._array.shape[0]):
            _total_unnormalized += float(self._array[_i].to(torch.float64).sum())
        self.dose_per_probe_unnormalized = _total_unnormalized / (
            self._shape[0] * self._shape[1]
        )

        if normalize:
            # Calculate normalization before creating the view
            temp_3d = self._array.contiguous().view(
                self._shape[0] * self._shape[1], self._shape[2], self._shape[3]
            )
            normalization_const = temp_3d.mean(0).max()
            self._array /= normalization_const

        # Create _array3d AFTER all modifications
        self._array3d = self._array.contiguous().view(
            self._shape[0] * self._shape[1], self._shape[2], self._shape[3]
        )
        # Seeded with the pre-normalization float64 sum computed above -- NOT None. A
        # lazy sum here would read the already-normalized array.
        self._total_intensity = _total_unnormalized

    def get_vbf(
        self, bright_field_mask_threshold: float
    ) -> "DatasetVirtualBrightField4dstem":
        """Cached virtual-bright-field provider, keyed on the mask threshold.

        Keeps exactly ONE vBF (and its cached ~17 GiB G for full-scan data)
        alive per dataset across the aberration fit, the reconstruction, the
        empirical SSNR and the depth section. Subclasses override
        ``_build_vbf`` to pick their provider flavor.
        """
        vbf = getattr(self, "vBF", None)
        if (
            vbf is not None
            and getattr(self, "_vbf_threshold", None) == bright_field_mask_threshold
        ):
            return vbf
        vbf = self._build_vbf(bright_field_mask_threshold)
        self.vBF = vbf
        self._vbf_threshold = bright_field_mask_threshold
        return vbf

    def _build_vbf(
        self, bright_field_mask_threshold: float
    ) -> "DatasetVirtualBrightField4dstem":
        return DatasetVirtualBrightField4dstem.from_4dstem_dataset(
            self,
            bright_field_mask_threshold=bright_field_mask_threshold,
            device=self.device,
        )

    @property
    def mean_cbed_tcdf(self) -> torch.Tensor:
        """tcDF's mean CBED variant: ``array[:n, :n].mean((0, 1))`` with
        ``n`` = scan rows (numpy clamps the column slice when nx < n)."""
        n = self.array.shape[0]
        return self.array[:n, :n].mean((0, 1))

    def gather_detector_group_means(
        self, groups: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Per-scan-position mean over each group of detector pixels.

        ``groups`` is a list of ``(M_i, 2)`` integer index tensors; returns one
        ``(ny, nx)`` image per group on ``self.device`` (NaN for an empty
        group, matching ``mean`` over an empty gather). Out-of-core datasets
        override this with a single streamed pass over the memmap.
        """
        return [self.array[:, :, g[:, 0], g[:, 1]].mean(-1) for g in groups]

    def determine_aberrations_(
        self,
        correction_method: str = "combined",
        fit_rotation: bool = False,
        target_percentage_nonzero_pixels: float = 0.75,
        n_batches: int = 25,
        registration_upsample_factor: int = 10,
        lowpass_fwhm_bright_field: Optional[float] = None,
        bin_factors: tuple[int, ...] = (2, 1, 1),
        verbosity: int = 0,
        correct_order: int = 1,
        gradient_mask: Optional[torch.Tensor] = None,
        num_iterations: int = 10,
        lr: float = 20,
        bright_field_mask_threshold: Optional[float] = None,
        roi_shape: tuple[int, int] = (128, 128),
        roi_center: Union[str, tuple[int, int]] = "center",
        upsample: Union[int, str] = "nyquist",
        update_dataset: bool = True,
        n_center_indices: int = 25,
        phase_sign: str = "positive",
        sharpness_metric: str = "sparsity",
        reg_weight: float = 0.05,
        alignment_method: str = "reference",
    ) -> None:
        """
        Determine the aberrations with the method given in correction_method.

        Args:
            correction_method: How to determine the aberrations. "combined" (default) chains the bright-field-shift low-order+rotation fit with the sharpness autofocus (higher orders, when correct_order>=2); "bright-field-shifts" is low-order+rotation only; "autofocus" is the autofocus only (rotation fixed). The legacy "total-variation" is a deprecated alias for "autofocus".
            sharpness_metric: Autofocus objective for the higher-order stage: "sparsity" (default, L4/kurtosis) or "tv".
            reg_weight: L2 regularization (scaled units) toward the seed in the autofocus optimizer.
            fit_rotation: Whether to fit the rotation.
            target_percentage_nonzero_pixels: Target percentage of nonzero pixels for the bright field.
            n_batches: Number of batches for the bright field shifts.
            registration_upsample_factor: Upsampling factor for the registration.
            lowpass_fwhm_bright_field: FWHM of the lowpass filter for the bright field.
            bin_factors: Bin factors for the bright field. The default is (2, 1, 1).
            arrow_scale: Scale for the arrows in the plot.
            verbosity: Verbosity level.
            lr: Learning rate for the optimizer. Used for the autofocus stage.
            bright_field_mask_threshold: Threshold for the bright field. Used for bright field shifts.
            roi_shape: Shape of the region of interest. Used for bright field shifts.
            roi_center: Center of the region of interest.
            upsample: Upsampling factor for the diffraction pattern. Used for bright field shifts.
            correct_order: Order of the aberrations to correct. Used for the autofocus stage.
            num_iterations: Number of optimization iterations. Used for the autofocus stage.
            n_center_indices: Number of center indices to use for the bright field shifts. The default is 25.

        """
        if gradient_mask is None:
            gradient_mask = torch.ones(12, dtype=torch.bool)
        if bright_field_mask_threshold is None:
            bright_field_mask_threshold = self.bright_field_mask_threshold
        aberrations = self.determine_aberrations(
            correction_method=correction_method,
            fit_rotation=fit_rotation,
            registration_upsample_factor=registration_upsample_factor,
            lowpass_fwhm_bright_field=lowpass_fwhm_bright_field,
            bin_factors=bin_factors,
            upsample=upsample,
            n_batches=n_batches,
            roi_shape=roi_shape,
            roi_center=roi_center,
            num_iterations=num_iterations,
            lr=lr,
            bright_field_mask_threshold=bright_field_mask_threshold,
            target_percentage_nonzero_pixels=target_percentage_nonzero_pixels,
            correct_order=correct_order,
            gradient_mask=gradient_mask,
            verbosity=verbosity,
            update_dataset=update_dataset,
            n_center_indices=n_center_indices,
            phase_sign=phase_sign,
            sharpness_metric=sharpness_metric,
            reg_weight=reg_weight,
            alignment_method=alignment_method,
        )
        self.meta.aberrations = Aberrations(array=aberrations)
        if fit_rotation:
            self.meta.rotation = self.vBF.meta.rotation

    def determine_aberrations(
        self,
        correction_method: str = "combined",
        fit_rotation: bool = False,
        target_percentage_nonzero_pixels: float = 0.75,
        n_batches: int = 25,
        registration_upsample_factor: int = 10,
        lowpass_fwhm_bright_field: Optional[float] = None,
        bin_factors: tuple[int, ...] = (2, 1, 1),
        verbosity: int = 0,
        correct_order: int = 1,
        gradient_mask: Optional[torch.Tensor] = None,
        num_iterations: int = 10,
        lr: float = 20,
        bright_field_mask_threshold: float = 0.5,
        roi_shape: tuple[int, int] = (128, 128),
        roi_center: Union[str, tuple[int, int]] = "center",
        upsample: Union[int, str] = "nyquist",
        update_dataset: bool = True,
        n_center_indices: int = 25,
        phase_sign: str = "positive",
        sharpness_metric: str = "sparsity",
        reg_weight: float = 0.05,
        alignment_method: str = "reference",
    ) -> torch.Tensor:
        """
        Determine the aberrations with the method given in correction_method.

        Args:
            correction_method: How to determine the aberrations. "combined" (default) chains the bright-field-shift low-order+rotation fit with the sharpness autofocus (higher orders, when correct_order>=2); "bright-field-shifts" is low-order+rotation only; "autofocus" is the autofocus only (rotation fixed). The legacy "total-variation" is a deprecated alias for "autofocus".
            sharpness_metric: Autofocus objective for the higher-order stage: "sparsity" (default, L4/kurtosis — less gameable, unbiased for higher orders) or "tv".
            reg_weight: L2 regularization (scaled units) toward the seed in the autofocus optimizer.
            fit_rotation: Whether to fit the rotation.
            target_percentage_nonzero_pixels: Target percentage of nonzero pixels for the bright field.
            n_batches: Number of batches for the bright field shifts.
            registration_upsample_factor: Upsampling factor for the registration.
            lowpass_fwhm_bright_field: FWHM of the lowpass filter for the bright field.
            bin_factors: Bin factors for the bright field. The default is (2, 1, 1).
            verbosity: Verbosity level.
            lr: Learning rate for the optimizer. Used for the autofocus stage.
            bright_field_mask_threshold: Threshold for the bright field. Used for bright field shifts.
            roi_shape: Shape of the region of interest. Used for bright field shifts.
            roi_center: Center of the region of interest.
            upsample: Upsampling factor for the diffraction pattern. Used for bright field shifts.
            correct_order: Order of the aberrations to correct. Used for the autofocus stage.
            num_iterations: Number of optimization iterations. Used for the autofocus stage.
            n_center_indices: Number of center indices to use for the bright field shifts. The default is 25.
            update_dataset: Whether to update the dataset.
        """
        from scatterem.reconstruction.direct_ptychography import (
            determine_aberrations,
        )

        if gradient_mask is None:
            gradient_mask = torch.ones(12, dtype=torch.bool)

        try:
            aberrations, vBF = determine_aberrations(
                dataset=self,
                correction_method=correction_method,
                fit_rotation=fit_rotation,
                registration_upsample_factor=registration_upsample_factor,
                lowpass_fwhm_bright_field=lowpass_fwhm_bright_field,
                bin_factors=bin_factors,
                upsample=upsample,
                n_batches=n_batches,
                roi_shape=roi_shape,
                roi_center=roi_center,
                num_iterations=num_iterations,
                lr=lr,
                bright_field_mask_threshold=bright_field_mask_threshold,
                target_percentage_nonzero_pixels=target_percentage_nonzero_pixels,
                correct_order=correct_order,
                gradient_mask=gradient_mask,
                verbosity=verbosity,
                update_dataset=update_dataset,
                n_center_indices=n_center_indices,
                phase_sign=phase_sign,
                sharpness_metric=sharpness_metric,
                reg_weight=reg_weight,
                alignment_method=alignment_method,
            )
        except BaseException:
            # Exception backstop: the fit accumulates FIT-SCOPED alignment
            # shifts on the (dataset-cached) vBF and clears them on its normal
            # exit. If it raises mid-way, the overlay must not leak — a later
            # reconstruction reusing the cached vBF would silently double-apply
            # the measured shifts (overlay ramps + fitted aberrations).
            _vbf = getattr(self, "vBF", None)
            if _vbf is not None:
                _vbf.clear_alignment_shifts()
            raise
        if update_dataset:
            self.vBF = vBF
            # Cache key for _resolve_vbf: reconstruction/SSNR/depth calls with
            # the same threshold reuse this vBF instead of rebuilding it.
            self._vbf_threshold = bright_field_mask_threshold
            self.meta.aberrations = Aberrations(array=aberrations)
            if fit_rotation:
                self.meta.rotation = vBF.meta.rotation
        return aberrations


    def fused_full_field(
        self,
        n_batches: int = 32,
        upsample: Union[int, str] = "nyquist",
        verbosity: int = 0,
        bright_field_mask_threshold: Optional[float] = None,
        return_snr: bool = True,
        snr_blur_sigma: float = 0.0,
        ssnr_method: str = "analytical",
        ptycho_lowq_cutoff_frac: float = 0.0,
        tcdf_shift_method: str = "drizzle",
        drizzle_pixfrac: float = 1.0,
        drizzle_kde_sigma: float = 0.0,
        n_dark_field_segments: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform fused full field reconstruction. Combine direct ptychography and tilt corrected dark field
        with SSNR-weighted Fourier fusion.

        ``ptycho_lowq_cutoff_frac`` > 0 hands the low-q band (below this fraction of
        Nyquist) exclusively to the dark-field, removing the direct-ptychography
        low-frequency cupping halo that otherwise leaks into the fused image.

        Args:
            n_batches: Number of batches for processing.
            upsample: Upsampling factor for the diffraction pattern.
            verbosity: Verbosity level.
            bright_field_mask_threshold: Threshold for the bright field.
            return_snr: Whether to compute SSNR for weighting.
            snr_blur_sigma: Gaussian blur sigma for the empirical ptycho SSNR and the tcDF SSNR (the analytical ptycho SSNR ignores it).
            ssnr_method: How to compute the ptychography SSNR ("empirical" or
                "analytical"). Changing this invalidates the cached ptycho SSNR.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (fused, phase_weighted, tcdf_weighted)
        """
        from scatterem.reconstruction.fused_full_field import (
            fused_full_field as _fused_full_field,
        )

        import time as _time

        if bright_field_mask_threshold is None:
            bright_field_mask_threshold = self.bright_field_mask_threshold

        _t0 = _time.perf_counter()

        # Key the cached ptycho SSNR/image on every parameter that determines
        # it; a stale cache (e.g. a different ``upsample``) would otherwise be
        # silently reused and yield a wrong-resolution result.
        ptycho_key = (
            str(upsample),
            n_batches,
            bright_field_mask_threshold,
            ssnr_method,
        )
        if (
            self.direct_ptychography_phase_image is None
            or getattr(self, "_ssnr_ptycho_cache_key", None) != ptycho_key
        ):
            phase_image, ssnr_ptycho = self.direct_ptychography(
                upsample=upsample,
                verbosity=verbosity,
                n_batches=n_batches,
                return_snr=return_snr,
                bright_field_mask_threshold=bright_field_mask_threshold,
                ssnr_method=ssnr_method,
                snr_blur_sigma=snr_blur_sigma,
            )
            self._ssnr_ptycho_cache_key = ptycho_key
        else:
            phase_image = self.direct_ptychography_phase_image
            ssnr_ptycho = self.ssnr_ptycho

        # n_dark_field_segments is a SCIENTIFIC parameter (wedge membership,
        # per-segment shifts, half-set parity); it defaults to n_batches only
        # for backward compatibility. Pass it explicitly to change ptycho
        # chunking without changing the tcDF science.
        if n_dark_field_segments is None:
            warnings.warn(
                f"n_dark_field_segments not given; falling back to "
                f"n_batches={int(n_batches)}. This is a SCIENTIFIC parameter (wedge "
                f"membership, per-segment shifts, half-set parity), so changing "
                f"n_batches for a memory reason silently changes the tilt-corrected "
                f"dark field, not just its batching. Pass n_dark_field_segments "
                f"explicitly.",
                stacklevel=2,
            )
            segments = int(n_batches)
        else:
            segments = int(n_dark_field_segments)

        # The tcDF SSNR/image likewise depends on upsample / segments /
        # threshold / blur; key the cache on all of them.
        tcdf_key = (
            str(upsample),
            segments,
            bright_field_mask_threshold,
            snr_blur_sigma,
            tcdf_shift_method,
            drizzle_pixfrac,
            drizzle_kde_sigma,
        )
        if (
            self.tilt_corrected_dark_field_image is None
            or getattr(self, "_ssnr_tcdf_cache_key", None) != tcdf_key
        ):
            tcDF, ssnr_tcdf = self.tilt_corrected_dark_field(
                n_dark_field_segments=segments,
                verbosity=verbosity,
                bright_field_mask_threshold=bright_field_mask_threshold,
                upsample=upsample,
                return_snr=return_snr,
                snr_blur_sigma=snr_blur_sigma,
                shift_method=tcdf_shift_method,
                drizzle_pixfrac=drizzle_pixfrac,
                drizzle_kde_sigma=drizzle_kde_sigma,
            )
            self._ssnr_tcdf_cache_key = tcdf_key
        else:
            tcDF = self.tilt_corrected_dark_field_image
            ssnr_tcdf = self.ssnr_tcdf

        _t_fuse = _time.perf_counter()
        fused, phase_weighted, tcdf_weighted = _fused_full_field(
            self,
            phase_image,
            tcDF,
            ssnr_ptycho,
            ssnr_tcdf,
            verbosity=verbosity,
            ptycho_lowq_cutoff_frac=ptycho_lowq_cutoff_frac,
        )
        if verbosity > 0:
            print(
                f"FF-STEM timing: total (recon+fuse) "
                f"{_time.perf_counter() - _t0:.3f} s, fusion-only "
                f"{_time.perf_counter() - _t_fuse:.3f} s"
            )

        return fused, phase_weighted, tcdf_weighted


    def tilt_corrected_dark_field(
        self,
        n_dark_field_segments: int = 32,
        verbosity: int = 0,
        bright_field_mask_threshold: Optional[float] = None,
        upsample: Union[float, str] = "nyquist",
        return_snr: bool = False,
        snr_blur_sigma: float = 0.0,
        shift_method: str = "drizzle",
        drizzle_pixfrac: float = 1.0,
        drizzle_kde_sigma: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Perform tilt corrected dark field reconstruction.

        ``shift_method`` defaults to ``"drizzle"`` — a non-negative, ringing-free
        area-overlap splat (with ``drizzle_pixfrac`` / ``drizzle_kde_sigma``),
        preferable for sparse/low-dose data. Pass ``"fourier"`` for the Fourier
        phase-ramp path (marginally sharper on dense, high-dose data).
        """
        from scatterem.reconstruction.tilt_corrected_dark_field import (
            tilt_corrected_dark_field,
        )

        if bright_field_mask_threshold is None:
            bright_field_mask_threshold = self.bright_field_mask_threshold

        result, ssnr_tcdf = tilt_corrected_dark_field(
            self,
            n_dark_field_segments=n_dark_field_segments,
            verbosity=verbosity,
            bright_field_mask_threshold=bright_field_mask_threshold,
            upsample=upsample,
            return_snr=return_snr,
            snr_blur_sigma=snr_blur_sigma,
            shift_method=shift_method,
            drizzle_pixfrac=drizzle_pixfrac,
            drizzle_kde_sigma=drizzle_kde_sigma,
        )
        self.tilt_corrected_dark_field_image = result
        self.ssnr_tcdf = ssnr_tcdf
        return result, ssnr_tcdf



    def direct_ptychography(
        self,
        upsample: Union[float, str] = "nyquist",
        verbosity: int = 0,
        n_batches: int = 25,
        return_snr: bool = False,
        phase_sign: str = "positive",
        ssnr_method: str = "analytical",
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Perform direct ptychography reconstruction.
        """
        from scatterem.reconstruction.direct_ptychography import (
            direct_ptychography,
        )

        kwargs.setdefault(
            "bright_field_mask_threshold", self.bright_field_mask_threshold
        )

        phase_image, ssnr_ptycho = direct_ptychography(
            self,
            upsample=upsample,
            verbosity=verbosity,
            n_batches=n_batches,
            return_snr=return_snr,
            phase_sign=phase_sign,
            ssnr_method=ssnr_method,
            **kwargs,
        )

        self.direct_ptychography_phase_image = phase_image
        self.ssnr_ptycho = ssnr_ptycho
        self.ssnr_ptycho_method = ssnr_method

        return phase_image, ssnr_ptycho


    @property
    def radius_bright_field(self):
        """
        Returns the radius of the bright field (BF) region if available.
        """
        return getattr(self, "_radius_bright_field", None)

    def _bright_field_crop_box(
        self: "Dataset4dstem", thresh_lower: float, thresh_upper: float
    ) -> _BrightFieldCropBox:
        r, c = self.bright_field_radius_and_center(
            thresh_lower=thresh_lower, thresh_upper=thresh_upper
        )
        r_ceil = np.ceil(r)
        r_int = int(np.ceil(r_ceil)) + 1
        y0_int = int(np.round(c[0]))
        x0_int = int(np.round(c[1]))
        # self.shape (not self._array.shape) so the out-of-core subclass —
        # which has no in-memory array — can reuse this box computation.
        rmax = self.shape[-1] // 2
        r = min(r_int, rmax)
        crop_slice = np.s_[
            :, :, y0_int - r : y0_int + r + 1, x0_int - r : x0_int + r + 1
        ]
        return _BrightFieldCropBox(crop_slice, r, c, y0_int, x0_int, r_ceil)

    def crop_brightfield_(
        self: "Dataset4dstem",
        thresh_lower: float = 0.01,
        thresh_upper: float = 0.99,
        normalize: bool = True,
        clip_neg_values: bool = True,
    ) -> "Dataset4dstem":
        """
        Crop the dataset to the brightfield region without clone.

        ``normalize``/``clip_neg_values`` are forwarded to :meth:`crop_` --
        see its docstring; pass ``False`` for both when the crop is a
        measurement-only scratch object (e.g. inside
        ``calibrate_reciprocal_from_bright_field``) that must not mutate this
        dataset's array through the unclonded view.
        """
        box = self._bright_field_crop_box(thresh_lower, thresh_upper)
        self._radius_bright_field = box.radius_ceil
        # set the origin to the center of the bright field region
        self.origin[-1] = box.center[0]
        self.origin[-2] = box.center[1]

        data_bf = self.crop_(
            box.crop_slice, normalize=normalize, clip_neg_values=clip_neg_values
        )
        data_bf._radius_bright_field = box.radius
        # Shift the origin into the cropped coordinate system.
        if hasattr(data_bf, "origin") and data_bf.origin is not None:
            new_origin = list(data_bf.origin)
            new_origin[-2] = box.center[0] - (box.y0 - box.radius)
            new_origin[-1] = box.center[1] - (box.x0 - box.radius)
            data_bf.origin = tuple(new_origin)

        data_bf.is_bright_field = True
        return data_bf





    def _averaged_diffraction_pattern(self) -> Tensor:
        sx = min(self.array.shape[0], 50)
        sy = min(self.array.shape[1], 50)
        return self._array[:sx, :sy].mean((0, 1))

    def bright_field_radius_and_center(
        self,
        thresh_lower: float = 0.1,
        thresh_upper: float = 0.6,
        N: int = 100,
        method: str = "area",
        edge_method: str = "canny",
        min_edge_points: int = 100,
        ransac_iterations: int = 1000,
        ransac_threshold: float = 2.0,
    ) -> Tuple[float, NDArray]:
        """Center and radius of the bright-field disk in pixels; center is [y, x].

        Delegates to
        :func:`scatterem.utils.data.disk_fit.fit_bright_field_disk`. The
        ``edge_method`` and ``ransac_*`` arguments belong to a RANSAC circle fit
        that this release does not carry; they are accepted so callers forwarding
        the full argument list keep working, and are otherwise unused.
        """
        if method != "area":
            raise ValueError(
                f"method={method!r} is not available in this release; only "
                f"'area' (the default) is implemented"
            )
        return fit_bright_field_disk(
            self._averaged_diffraction_pattern(),
            threshold_range=(thresh_lower, thresh_upper),
            n_thresholds=N,
            # A dataset already cropped to the bright-field disk has the disk
            # filling most of the frame by construction, so the "is this really a
            # compact disk?" guard must be relaxed there -- otherwise it rejects
            # the very thing it is measuring. calibrate_reciprocal_from_bright_field
            # crops 96 -> 51 px and re-fits, where the disk is 46% of the frame.
            max_radius_fraction=0.49 if self.is_bright_field else 0.4,
        )

    def bright_field_radius_and_center_(
        self,
        thresh_lower: float = 0.1,
        thresh_upper: float = 0.6,
        N: int = 100,
        method: str = "area",
        edge_method: str = "canny",
        min_edge_points: int = 50,
        ransac_iterations: int = 1000,
        ransac_threshold: float = 2.0,
    ) -> Tuple[float, NDArray]:
        """In-place variant: caches the radius on the dataset as ``_radius_bright_field``."""
        r, c = self.bright_field_radius_and_center(
            thresh_lower,
            thresh_upper,
            N,
            method,
            edge_method,
            min_edge_points,
            ransac_iterations,
            ransac_threshold,
        )
        self._radius_bright_field = r
        return r, c

    def calibrate_reciprocal_from_bright_field(
        self,
        semiconvergence_angle: Optional[float] = None,
        thresh_lower: float = 0.1,
        thresh_upper: float = 0.99,
        method: str = "area",
    ) -> float:
        """Calibrate the reciprocal-space pixel size ``dk`` from the measured
        bright-field disk radius and the known semiconvergence angle.

        Encapsulates the crop -> measure -> ``dk = alpha / rBF / wavelength``
        -> ``self.sampling = (dr, dr, dk, dk)`` dance every demo repeats.
        ``self`` is updated in place (via the ``sampling`` setter, which
        propagates to ``self.meta.sampling``); the crop used to measure the
        disk is a scratch object (``crop_brightfield_`` returns a new
        dataset, built with ``normalize=False, clip_neg_values=False`` so
        it never writes back through its unclonded view into this dataset's
        storage) and this dataset's array is left untouched -- only
        ``self.origin``'s detector-plane entries are recentred and
        ``self._radius_bright_field`` is set, matching ``crop_brightfield_``'s
        existing (self-mutating) behavior.

        Args:
            semiconvergence_angle: Beam convergence semi-angle in radians.
                Defaults to ``self.meta.semiconvergence_angle``.
            thresh_lower, thresh_upper: Intensity-threshold sweep bounds used
                both for the bright-field crop and the radius measurement.
            method: Radius-measurement method, forwarded to
                ``bright_field_radius_and_center_`` ("area" or "circle").

        Returns:
            float: The calibrated reciprocal-space sampling ``dk``.
        """
        if semiconvergence_angle is None:
            semiconvergence_angle = self.meta.semiconvergence_angle

        # Corner-origin data would sail through the crop-and-measure below and yield a
        # meaningless radius, hence a badly wrong dk, with no error anywhere. Check
        # before measuring. `mean_cbed_tcdf` is the existing mean-CBED accessor (a scan
        # subset is sufficient -- the disk's position does not depend on which scan
        # positions are averaged).
        from scatterem.utils.data.origin_check import (
            corner_origin_margin,
            is_corner_origin,
        )

        _mean_cbed = self.mean_cbed_tcdf
        if is_corner_origin(_mean_cbed):
            raise ValueError(
                "the diffraction patterns use the CORNER-origin convention, but "
                "reciprocal calibration (and the whole FF-STEM path) assumes a CENTERED "
                "origin. Left uncaught this fails silently: the bright-field disk "
                "straddles the array corners, so the fitted radius and therefore dk are "
                "wrong, and direct ptychography returns an essentially flat image. Fix "
                "by fftshift-ing the detector axes before building the dataset, e.g. "
                "torch.fft.fftshift(cube, dim=(-2, -1)) "
                f"(corner-origin margin {corner_origin_margin(_mean_cbed):.1f} px)."
            )

        data_bf = self.crop_brightfield_(
            thresh_lower, thresh_upper, normalize=False, clip_neg_values=False
        )
        rBF, _ = data_bf.bright_field_radius_and_center_(
            thresh_lower, thresh_upper, method=method
        )

        dk = (semiconvergence_angle / rBF) / self.meta.wavelength
        dr = self.sampling[0]
        self.sampling = (dr, dr, dk, dk)
        return dk

    @property
    def total_intensity(self) -> float:
        """
        Total intensity of the probe over the dataset.
        """
        if self._total_intensity is None:
            total = 0.0
            for i in range(self._array.shape[0]):
                total += float(self._array[i].to(torch.float64).sum())
            self._total_intensity = total
        return self._total_intensity

    @total_intensity.setter
    def total_intensity(self, value: float) -> None:
        self._total_intensity = value

    @property
    def mean_probe_intensity(self) -> torch.Tensor:
        """
        Mean intensity of the probe over the dataset.
        """
        return self._array.sum(axis=(-2, -1)).mean()

    @property
    def max_probe_intensity(self) -> torch.Tensor:
        """
        Max intensity of the probe over the dataset.
        """
        return self._array.sum(axis=(-2, -1)).max()

    @property
    def fluence(self) -> float:
        """Calculate total electron fluence (electrons per square Angstrom) from total intensity.

        Returns:
            float: Total electron fluence in electrons per square Angstrom.
        """
        scan_area = float(np.prod(self.sampling[:2] * np.array(self._shape[:2])))
        return self.total_intensity / scan_area

    @property
    def fluence_per_probe(self) -> float:
        """Calculate the average electron dose per probe position.

        Returns:
            float: Average dose per probe (unnormalized).
        """
        return self.dose_per_probe_unnormalized



    def crop(
        self,
        index: tuple[slice, ...],
        clone: bool = True,
        normalize: bool = True,
        clip_neg_values: bool = True,
    ) -> "Dataset4dstem":
        """
        Simple indexing function to return Dataset4dstem view.

        Parameters
        ----------
        index : tuple[slice, ...]
            Index to access a subset of the dataset
        clone : bool
            If True, the array is cloned before returning.
        normalize, clip_neg_values : bool
            Forwarded to ``Dataset4dstem.from_array``. When ``clone=False``
            the child's array is an unclonded VIEW into ``self._array`` --
            ``normalize=True``/``clip_neg_values=True`` then mutate the
            parent's storage in place (``self._array /= ...`` / masked
            zeroing runs through the view). Pass ``False`` for a
            measurement-only scratch crop that must not touch the parent.

        Returns
        -------
        dataset
            A new Dataset4dstem instance containing the indexed data
        """
        array_view = self.array[index]
        if clone:
            array_view = array_view.clone()
        ndim = array_view.ndim

        # Calculate new origin based on slice info and old origin
        if hasattr(index[0], "start") and index[0].start is not None:
            origin_offset_y = index[0].start
        else:
            origin_offset_y = 0

        if hasattr(index[1], "start") and index[1].start is not None:
            origin_offset_x = index[1].start
        else:
            origin_offset_x = 0

        if hasattr(index[2], "start") and index[2].start is not None:
            origin_offset_z = index[2].start
        else:
            origin_offset_z = 0

        if hasattr(index[3], "start") and index[3].start is not None:
            origin_offset_k = index[3].start
        else:
            origin_offset_k = 0

        new_origin = np.array(self.origin) - np.array(
            [origin_offset_y, origin_offset_x, origin_offset_z, origin_offset_k]
        )

        if ndim == 4:
            cls = Dataset4dstem
        else:
            raise ValueError("only 4D slices are supported.")

        return cls.from_array(
            array=array_view,
            name=self.name + str(index),
            origin=new_origin,
            sampling=self.sampling,
            units=self.units,
            signal_units=self.signal_units,
            device=self.device,
            meta=self.meta,
            is_cropped=True,
            normalize=normalize,
            clip_neg_values=clip_neg_values,
        )

    def crop_(
        self,
        index: tuple[slice, ...],
        normalize: bool = True,
        clip_neg_values: bool = True,
    ) -> "Dataset4dstem":
        """
        Simple indexing function to return Dataset4dstem view.

        Parameters
        ----------
        index : tuple[slice, ...]
            Index to access a subset of the dataset
        normalize, clip_neg_values : bool
            Forwarded to :meth:`crop` (which is always called with
            ``clone=False`` here, i.e. an unclonded view -- see its docstring
            for why these default to ``True`` but must be ``False`` for a
            measurement-only scratch crop).

        Returns
        -------
        dataset
            A new Dataset4dstem instance containing the indexed data
        """
        return self.crop(
            index, clone=False, normalize=normalize, clip_neg_values=clip_neg_values
        )

    @property
    def detector_shape(self) -> NDArray:
        """ """
        return np.array(self._shape[-2:])

    @property
    def k_max(self) -> NDArray:
        """Calculate maximum scattering vector magnitude from semiconvergence angle and detector shape.

        Returns:
            float: Maximum scattering vector magnitude in inverse Angstroms.
        """
        return self.sampling[-2:] * self.detector_shape / 2

    @property
    def dr(self) -> NDArray:
        """Calculate real space sampling of the detector from k_max.

        Returns:
            float: Real space sampling of the detector in Angstroms.
        """
        return 1 / (2 * self.k_max)

    @property
    def dk(self) -> NDArray:
        """Calculate reciprocal space sampling of the detector from a bright field radius estimation.

        Returns:
            float: Reciprocal space sampling of the detector in inverse Angstroms.
        """
        rbf, _ = self.bright_field_radius_and_center()
        return (
            self.meta.semiconvergence_angle / rbf / electron_wavelength(self.meta.energy)
        )

    @classmethod
    def from_array(
        cls,
        array: Any,  # Input can be array-like
        name: str | None = None,
        origin: Union[NDArray, tuple, list, float, int] | None = None,
        sampling: Union[NDArray, tuple, list, float, int] | None = None,
        units: Union[list[str], tuple, list] | None = None,
        signal_units: str = "arb. units",
        meta: Optional[Metadata4dstem] = None,
        transform_to_amplitudes: bool = False,
        fourier_shift_dim: Tuple = None,
        normalize: bool = True,
        clip_neg_values: bool = True,
        copy: bool = True,
        device: torch.device = torch.device("cpu"),
        is_cropped: bool = False,
        out_of_core: bool = False,
        ooc_row_block: int = 8,
        g_residency: str = "auto",
        energy: float | None = None,
        semiconvergence_angle: float | None = None,
        scan_step: Union[float, tuple, list, NDArray, None] = None,
        reciprocal_step: Union[float, tuple, list, NDArray, None] = None,
        rotation: float = 0.0,
        aberrations: Optional["Aberrations"] = None,
    ) -> "Dataset4dstem":
        """
        Validates and creates a Dataset from an array.

        Parameters
        ----------
        array: Any
            The array to validate and create a Dataset from.
        name: str | None
            The name of the Dataset.
        origin: Union[NDArray, tuple, list, float, int] | None
            The origin of the Dataset.
        sampling: Union[NDArray, tuple, list, float, int] | None
            The sampling of the Dataset.
        units: Union[list[str], tuple, list] | None
            The units of the Dataset.
        signal_units: str
            The units of the signal.
        out_of_core: bool
            Keep the raw (uint16) cube as a CPU numpy array/memmap and stream
            it; nothing is uploaded to ``device`` here. See
            ``scatterem.utils.data.out_of_core``.
        ooc_row_block: int
            Scan-row block size for streamed reductions (out-of-core only).
        g_residency: str
            "gpu" | "vbf_gpu" | "cpu_pinned" | "auto" — where the streaming
            vBF provider keeps G (out-of-core only).
        energy, semiconvergence_angle, scan_step, reciprocal_step, rotation, aberrations:
            Physics convenience path: when ``meta`` is None and ``energy`` is
            given, a ``Metadata4dstem`` is built internally (shape derived from
            ``array``, ``sampling = (dr, dr, dk, dk)`` with ``dr`` from
            ``scan_step`` and ``dk`` from ``reciprocal_step`` -- both scalar or
            a length-2 pair, ``dk`` defaulting to a ``(1.0, 1.0)`` placeholder
            to be calibrated later, e.g. via
            ``calibrate_reciprocal_from_bright_field``). ``origin`` defaults to
            the array center in this path when not given explicitly.

        Returns
        -------
        Dataset
            The container, with any unset axis metadata filled in from the
            array's rank.
        """
        # Set defaults if None (np.ndim works on tensors, arrays and memmaps
        # alike — resolved once for both the eager and the out-of-core path)
        _ndim = np.ndim(array)
        _name = name if name is not None else f"{_ndim}d dataset"

        if meta is None and energy is not None:
            meta, _built_sampling, _built_origin = _metadata4dstem_from_physics(
                array.shape,
                energy=energy,
                semiconvergence_angle=semiconvergence_angle,
                scan_step=scan_step,
                reciprocal_step=reciprocal_step,
                rotation=rotation,
                aberrations=aberrations,
            )
            if sampling is None:
                sampling = _built_sampling
            if origin is None:
                origin = _built_origin

        _origin = origin if origin is not None else np.zeros(_ndim)
        _sampling = (
            sampling
            if sampling is not None
            else (
                meta.sampling
                if meta is not None and meta.sampling is not None
                else np.ones(_ndim)
            )
        )
        _units = (
            units
            if units is not None
            else (
                meta.units
                if meta is not None and meta.units is not None
                else ["pixels"] * _ndim
            )
        )

        if out_of_core:
            from .out_of_core import Dataset4dstemOutOfCore

            return Dataset4dstemOutOfCore(
                raw=array,
                name=_name,
                origin=_origin,
                sampling=_sampling,
                units=_units,
                meta=meta,
                device=device,
                normalize=normalize,
                ooc_row_block=ooc_row_block,
                g_residency=g_residency,
                signal_units=signal_units,
            )

        validated_array = ensure_valid_array(array, device=device)
        return cls(
            array=validated_array,
            name=_name,
            origin=_origin,
            sampling=_sampling,
            units=_units,
            signal_units=signal_units,
            _token=cls._token,
            meta=meta,
            device=device,
            transform_to_amplitudes=transform_to_amplitudes,
            fourier_shift_dim=fourier_shift_dim,
            normalize=normalize,
            copy=copy,
            clip_neg_values=clip_neg_values,
        )

    # --- Properties ---
    @property
    def array(self) -> Tensor:
        """The underlying n-dimensional array data. Tensor"""
        return self._array

    @array.setter
    def array(self, value: Tensor) -> None:
        self._array = ensure_valid_array(
            value, dtype=self.dtype, ndim=self.ndim, device=value.device
        )

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = str(value)

    @property
    def origin(self) -> NDArray:
        return self._origin

    @origin.setter
    def origin(self, value: Union[NDArray, tuple, list, float, int]) -> None:
        self._origin = validate_ndinfo(value, self.ndim, "origin")

    @property
    def sampling(self) -> NDArray:
        return self._sampling

    @sampling.setter
    def sampling(self, value: Union[NDArray, tuple, list, float, int]) -> None:
        self._sampling = validate_ndinfo(value, self.ndim, "sampling")
        if self.meta is not None:
            self.meta.sampling = self._sampling

    @property
    def units(self) -> list[str]:
        return self._units

    @units.setter
    def units(self, value: Union[list[str], tuple, list]) -> None:
        self._units = validate_units(value, self.ndim)

    @property
    def signal_units(self) -> str:
        return self._signal_units

    @signal_units.setter
    def signal_units(self, value: str) -> None:
        self._signal_units = str(value)

    # --- Derived Properties ---
    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    @property
    def ndim(self) -> int:
        return self.array.ndim

    @property
    def dtype(self) -> DTypeLike:
        return self.array.dtype

    @property
    def is_bright_field(self) -> bool:
        """Whether the dataset contains only the bright field region."""
        return self._is_bright_field if hasattr(self, "_is_bright_field") else False

    @is_bright_field.setter
    def is_bright_field(self, value: bool) -> None:
        self._is_bright_field = bool(value)

    @property
    def is_cropped(self) -> bool:
        """Whether the dataset has been cropped."""
        return self._is_cropped if hasattr(self, "_is_cropped") else False

    @is_cropped.setter
    def is_cropped(self, value: bool) -> None:
        self._is_cropped = bool(value)

    @property
    def device(self) -> torch.device:
        return self.array.device

    @device.setter
    def device(self, value: torch.device) -> None:
        """Set the device for the array."""
        self._array = self._array.to(value)
        self.meta.device = value

    # --- Summaries ---
    def _summary_rows(self) -> dict[str, Any]:
        rows = {
            "name": self.name,
            "scan": f"{self.shape[0]} x {self.shape[1]}",
            "detector": f"{self.shape[2]} x {self.shape[3]}",
            "dtype": self.dtype,
            "device": self.device,
            "amplitudes": self.transform_to_amplitudes,
            "dose/probe": self.dose_per_probe_unnormalized,
        }
        if self.meta is not None:
            rows["energy (eV)"] = getattr(self.meta, "energy", None)
            rows["semiconv (rad)"] = getattr(self.meta, "semiconvergence_angle", None)
        return rows


























