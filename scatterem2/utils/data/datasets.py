from typing import Any, Optional, Tuple, Union

import h5py
import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import DTypeLike, NDArray
from torch import Tensor
from torch.fft import fftfreq
from torch.utils.data import Dataset as TorchDataset
import scatterem2.vis as vis
from scatterem2.io.serialize import AutoSerialize
from scatterem2.utils.data.data_classes import Metadata4dstem
from scatterem2.utils.data.aberrations import Aberrations
from scatterem2.utils.stem import energy2wavelength
from scatterem2.utils.utils import (
    detect_edges,
    fit_circle_ransac,
    fuse_images_fourier_weighted,
    refine_circle_fit,
    select_best_circle,
)
from scatterem2.utils.validators import (
    ensure_valid_array,
    validate_ndinfo,
    validate_units,
)


class Dataset(TorchDataset, AutoSerialize):
    """
    A class representing a multi-dimensional dataset with metadata.
    Uses standard properties and validation within __init__ for type safety.

    Attributes (Properties):
        array (NDArray | Any): The underlying n-dimensional array data (Any for CuPy).
        name (str): A descriptive name for the dataset.
        origin (NDArray): The origin coordinates for each dimension (1D array).
        sampling (NDArray): The sampling rate/spacing for each dimension (1D array).
        units (list[str]): Units for each dimension.
        signal_units (str): Units for the array values.
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
            A Dataset object with the validated array and metadata.
        """
        validated_array = ensure_valid_array(array)
        _ndim = validated_array.ndim

        # Set defaults if None
        _name = name if name is not None else f"{_ndim}d dataset"
        _origin = origin if origin is not None else np.zeros(_ndim)
        _sampling = sampling if sampling is not None else np.ones(_ndim)
        _units = units if units is not None else ["pixels"] * _ndim

        return cls(
            array=validated_array,
            name=_name,
            origin=_origin,
            sampling=_sampling,
            units=_units,
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
        self._array = ensure_valid_array(value, dtype=self.dtype, ndim=self.ndim, device=self.device)

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
    def __repr__(self) -> str:
        description = [
            f"Dataset(shape={self.shape}, dtype={self.dtype}, name='{self.name}')",
            f"  sampling: {self.sampling}",
            f"  units: {self.units}",
            f"  signal units: '{self.signal_units}'",
        ]
        return "\n".join(description)

    def __str__(self) -> str:
        description = [
            f"Dataset named '{self.name}'",
            f"  shape: {self.shape}",
            f"  dtype: {self.dtype}",
            f"  device: {self.device}",
            f"  origin: {self.origin}",
            f"  sampling: {self.sampling}",
            f"  units: {self.units}",
            f"  signal units: '{self.signal_units}'",
        ]
        return "\n".join(description)


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
        norm = self.diffraction_pattern_mean_normalized[inds[:,0], inds[:,1]]
        norm = norm[None, None, :]
        return self.array / norm

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
        self.k = k
        self.qx_1d = qx_1d
        self.qy_1d = qy_1d
        self.q_2d = torch.stack(torch.meshgrid(qy_1d[0], qx_1d[1], indexing="ij"), dim=0)
        self.fourier_shift_dim = fourier_shift_dim
        self.probe_index = probe_index
        self.parent_dataset = parent_dataset
        self.diffraction_pattern_mean_normalized = diffraction_pattern_mean_normalized
        self.bright_field_mask = bright_field_mask
        self.bright_field_inds = bright_field_inds
        self.bright_field_inds_centered = bright_field_inds_centered
        self.bright_field_inds_radial_order = bright_field_inds_radial_order
        self.bright_field_inds_ordered_by_radius = bright_field_inds_ordered_by_radius
        self.bright_field_inds_centered_ordered_by_radius = bright_field_inds_centered_ordered_by_radius

        self._shape = self._array.shape
        if astype_float32:
            self._array = self._array.to(torch.float32)
        if fourier_shift_dim is not None:
            self._array = torch.fft.ifftshift(self._array, dim=(2, 3))
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
        device: Optional[torch.device] = None, # will use dataset.device if not provided
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
        diff_mean = dataset.array[:n,:n].mean((0,1))
        diff_mean /= diff_mean.max()
        bright_field_mask = diff_mean > bright_field_mask_threshold
        if verbosity > 0:           
            fig, ax = vis.show_2d(bright_field_mask.float(), cbar=True, title="Virtual Bright Field Mask") 
        bright_field_inds = torch.argwhere(bright_field_mask)
        bright_field_inds_centered = bright_field_inds.float() - torch.mean(bright_field_inds.float(), dim=0)[None]
        bright_field_inds_radial_order = torch.argsort(torch.sum(bright_field_inds_centered**2, dim=1))
        bright_field_inds_ordered_by_radius = bright_field_inds[bright_field_inds_radial_order]
        bright_field_inds_centered_ordered_by_radius = bright_field_inds_centered[bright_field_inds_radial_order]
    
        bright_field = dataset.array[:,:,bright_field_inds_ordered_by_radius[:,0], bright_field_inds_ordered_by_radius[:,1]]
 
        validated_array = ensure_valid_array(bright_field, device=device)
        _ndim = validated_array.ndim
        sampling = torch.as_tensor(dataset.meta.sampling[-2:], device=device)[None]
        k = (bright_field_inds_centered_ordered_by_radius.to(device) * sampling.expand_as(bright_field_inds_centered_ordered_by_radius)).to(torch.float32)
 
        upsample_int = 1
        Qx = fftfreq(dataset.shape[1] * upsample_int, dataset.sampling[1] / upsample_int, dtype=torch.float32, device=dataset.device,)
        Qy = fftfreq(dataset.shape[0] * upsample_int, dataset.sampling[0] / upsample_int, dtype=torch.float32, device=dataset.device,)

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
            fourier_shift_dim=dataset.fourier_shift_dim,
            clip_neg_values=False,
            k=k,
            qx_1d=Qx,
            qy_1d=Qy,
        )

    @property
    def G(self) -> torch.Tensor:
        """
        The Fourier transform of the bright field image.

        Returns:
            torch.Tensor: The Fourier transform of the bright field image
        """
        if self._G.numel() == 0:
            self._G = torch.fft.fft2(self.array, dim=(0,1), norm="ortho") 
        return self._G

  
    def get_q_1d(self, shape: tuple[int, int] = None) -> tuple[torch.Tensor, torch.Tensor]:
        if shape is None:
            shape = self.shape
        upsample_x = shape[1] / self.shape[1]
        upsample_y = shape[0] / self.shape[0]
        Qx = fftfreq(shape[1], self.sampling[1] / upsample_x, dtype=torch.float32, device=self.device,)
        Qy = fftfreq(shape[0], self.sampling[0] / upsample_y, dtype=torch.float32, device=self.device,)
        return Qy, Qx

class Dataset4dstem(Dataset):
    """A 4D STEM dataset with metadata for electron diffraction patterns."""

    meta: Metadata4dstem = None
    dose_per_probe_unnormalized: float = None
    dose_per_probe_normalized: float = None
    _is_cropped: bool = False
    _is_bright_field: bool = False
    _direct_ptychography_phase_image: torch.Tensor = torch.empty(0)
    _direct_ptychography_amplitude_image: torch.Tensor = torch.empty(0)
    _tilt_corrected_dark_field_image: torch.Tensor = torch.empty(0)
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
        meta: Metadata4dstem = None,
        transform_to_amplitudes: bool = False,
        astype_float32: bool = True,
        fourier_shift_dim: Tuple = None,
        probe_index: int = 0,
        device: torch.device = torch.device("cpu"),
        normalize: bool = True,
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
        self.transform_to_amplitudes = transform_to_amplitudes
        self.fourier_shift_dim = fourier_shift_dim
        self.probe_index = probe_index

        self._shape = self._array.shape
        if astype_float32:
            self._array = self._array.to(torch.float32)
        if fourier_shift_dim is not None:
            self._array = torch.fft.ifftshift(self._array, dim=(2, 3))
        if clip_neg_values:
            self._array[self._array < 0] = 0

        self._array3d = self._array.contiguous().view(
            self._shape[0] * self._shape[1], self._shape[2], self._shape[3]
        )
        if transform_to_amplitudes:
            self._array.sqrt_()
        self.dose_per_probe_unnormalized = self._array.sum().item() / (self._shape[0] * self._shape[1])        
        if normalize:
            normalization_const = self._array3d.mean(0).max()
            self._array /= normalization_const
        
        self._total_intensity = None
        # Calculate total intensity using float64 precision and loop

    def determine_aberrations_(self, 
            correction_method: str = "bright-field-shifts",
            fit_rotation: bool = False,
            target_percentage_nonzero_pixels: float = 0.75,
            n_batches: int = 25,
            registration_upsample_factor: int = 10,         
            lowpass_fwhm_bright_field: Optional[float] = None,
            bin: Optional[int] = None,
            arrow_scale: float = 25e-2, 
            verbosity: int = 0,         
            correct_order: int = 1,
            gradient_mask: torch.Tensor = torch.ones(12, dtype=torch.bool),
            num_iterations: int = 10,
            lr: float = 20,
            bright_field_mask_threshold: float = 0.5,
            roi_shape: tuple[int, int] = (128, 128),
            roi_center: Union[str, tuple[int, int]] = "center",
            upsample: Union[int, str] = "nyquist",
            n_center_indices: int = 25,) -> None:
        """
        Determine the aberrations with the method given in correction_method.

        Args:
            correction_method: Method to correct the aberrations. Either "bright-field-shifts" or "total-variation" or "bright-field-shifts-interpolated".
            fit_rotation: Whether to fit the rotation.
            target_percentage_nonzero_pixels: Target percentage of nonzero pixels for the bright field.
            n_batches: Number of batches for the bright field shifts.
            registration_upsample_factor: Upsampling factor for the registration.
            lowpass_fwhm_bright_field: FWHM of the lowpass filter for the bright field.
            bin: Bin size for the bright field.
            arrow_scale: Scale for the arrows in the plot.
            verbosity: Verbosity level.
            lr: Learning rate for the optimizer. Used for total variation.
            bright_field_mask_threshold: Threshold for the bright field. Used for bright field shifts.
            roi_shape: Shape of the region of interest. Used for bright field shifts.
            roi_center: Center of the region of interest.
            upsample: Upsampling factor for the diffraction pattern. Used for bright field shifts.
            correct_order: Order of the aberrations to correct. Used for total variation.
            num_iterations: Number of optimization iterations. Used for total variation.
            n_center_indices: Number of center indices to use for the bright field shifts. The default is 25.
            
        """
        aberrations = self.determine_aberrations(
            correction_method=correction_method,
            fit_rotation=fit_rotation,
            registration_upsample_factor=registration_upsample_factor,         
            lowpass_fwhm_bright_field=lowpass_fwhm_bright_field,
            bin=bin,
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
            update_dataset=True,
            n_center_indices=n_center_indices,)
        self.meta.aberrations = Aberrations(array=aberrations)
        if fit_rotation:
            self.meta.rotation = self.vBF.meta.rotation
 

    def determine_aberrations(self, 
            correction_method: str = "bright-field-shifts",
            fit_rotation: bool = False,
            target_percentage_nonzero_pixels: float = 0.75,
            n_batches: int = 25,
            registration_upsample_factor: int = 10,         
            lowpass_fwhm_bright_field: Optional[float] = None,
            bin: Optional[int] = None,
            arrow_scale: float = 25e-2, 
            verbosity: int = 0,         
            correct_order: int = 1,
            gradient_mask: torch.Tensor = torch.ones(12, dtype=torch.bool),
            num_iterations: int = 10,
            lr: float = 20,
            bright_field_mask_threshold: float = 0.5,
            roi_shape: tuple[int, int] = (128, 128),
            roi_center: Union[str, tuple[int, int]] = "center",
            upsample: Union[int, str] = "nyquist",
            update_dataset: bool = True,
            n_center_indices: int = 25,) -> torch.Tensor:
        """
        Determine the aberrations with the method given in correction_method.

        Args:
            correction_method: Method to correct the aberrations. Either "bright-field-shifts" or "total-variation" or "bright-field-shifts-interpolated".
            fit_rotation: Whether to fit the rotation.
            target_percentage_nonzero_pixels: Target percentage of nonzero pixels for the bright field.
            n_batches: Number of batches for the bright field shifts.
            registration_upsample_factor: Upsampling factor for the registration.
            lowpass_fwhm_bright_field: FWHM of the lowpass filter for the bright field.
            bin: Bin size for the bright field.
            arrow_scale: Scale for the arrows in the plot.
            verbosity: Verbosity level.
            lr: Learning rate for the optimizer. Used for total variation.
            bright_field_mask_threshold: Threshold for the bright field. Used for bright field shifts.
            roi_shape: Shape of the region of interest. Used for bright field shifts.
            roi_center: Center of the region of interest.
            upsample: Upsampling factor for the diffraction pattern. Used for bright field shifts.
            correct_order: Order of the aberrations to correct. Used for total variation.
            num_iterations: Number of optimization iterations. Used for total variation.
            n_center_indices: Number of center indices to use for the bright field shifts. The default is 25.
            update_dataset: Whether to update the dataset.
        """
        from scatterem2.reconstruction.direct_ptychography import determine_aberrations
        aberrations, vBF = determine_aberrations(
            dataset=self,
            correction_method=correction_method,
            fit_rotation=fit_rotation,
            registration_upsample_factor=registration_upsample_factor,         
            lowpass_fwhm_bright_field=lowpass_fwhm_bright_field,
            bin=bin,
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
            n_center_indices=n_center_indices,)
        if update_dataset:
            self.vBF = vBF
            self.meta.aberrations = Aberrations(array=aberrations)
            if fit_rotation:
                self.meta.rotation = vBF.meta.rotation
        return aberrations

    def _fuse_images_pctf_weighted(
        self, im_phase: torch.Tensor, im_dark_field: torch.Tensor, upsample="nyquist", verbosity: int = 0
    ) -> torch.Tensor:
        from scatterem2.reconstruction.direct_ptychography import (
            phase_contrast_transfer_function,
        )

        pctf = phase_contrast_transfer_function(self, upsample=upsample, verbosity=1)
        pctf = torch.fft.fftshift(pctf)
        p02, p98 = torch.quantile(pctf.cpu(), torch.tensor([0.02, 0.98]))
        pctf -= p02
        pctf /= p98
        Qy, Qx = self.vBF.get_q_1d(pctf.shape)
        Q = torch.sqrt(Qy[:, None] ** 2 + Qx[None, :] ** 2).to(self.device)
        alpha_mask = torch.fft.fftshift(
            Q < self.meta.semiconvergence_angle / self.meta.wavelength
        )

        weight_tcDF = 1 - (alpha_mask.float() * pctf)
        # vis.show_2d(pctf.cpu(), cbar=True, title="Pctf")
        weight_tcDF = torch.clip(weight_tcDF, 0, 1)
        weight_tcDF[alpha_mask == 0] = 0
        weight_tcDF /= weight_tcDF.max()
        if verbosity > 1:
            fig, ax = plt.subplots(1, 2, figsize=(8, 8))
            fig_bf_analytic2, ax_bf_analytic2 = vis.show_2d(
                [weight_tcDF, 1 - weight_tcDF],
                cbar=True,
                title=["Weight Dark field", "Weight Phase"],
                figax=(fig, ax),
            )
            plt.show()
        weight_tcDF = weight_tcDF.to(self.device)
        im_phase = im_phase.to(self.device)
        im_dark_field = im_dark_field.to(self.device)
        fused, phase_weighted, tcdf_weighted = fuse_images_fourier_weighted(im_phase, im_dark_field, weight_tcDF)
        return fused, phase_weighted, tcdf_weighted
    
    def fused_full_field(
        self,
        n_batches: int = 32,
        upsample: Union[int, str] = "nyquist",
        verbosity: int = 0,
        bright_field_mask_threshold: float = 0.3,
        store_image_in_dataset: bool = True,
        return_snr: bool = True,
        snr_blur_sigma: float = 0.0,
    ) -> torch.Tensor:
        """
        Perform fused full field reconstruction. Combine direct ptychography and tilt corrected dark field.
        Use the wavelet transform to fuse the two images.
        Args:
            n_batches: Number of batches for processing.
            upsample: Upsampling factor for the diffraction pattern.
            verbosity: Verbosity level.
            bright_field_mask_threshold: Threshold for the bright field.
            store_image_in_dataset: Whether to store the image in the dataset.
            method: Method to use for fusing the images. Either "dwt" or "fourier".
        Returns:
            torch.Tensor: The fused full field image.
        """
        from scatterem2.reconstruction.fused_full_field import fused_full_field
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
        if self.direct_ptychography_phase_image is None or self.ssnr_ptycho is None:
            phase_image, ssnr_ptycho = self.direct_ptychography(self, 
            upsample=upsample, 
            verbosity=verbosity,
            bright_field_mask_threshold=bright_field_mask_threshold,
            n_batches=n_batches,
            return_snr=return_snr)
        else:
            phase_image = self.direct_ptychography_phase_image
            ssnr_ptycho = self.ssnr_ptycho
        if self.tilt_corrected_dark_field_image is None or self.ssnr_tcdf is None:
            result, ssnr_tcdf = self.tilt_corrected_dark_field(self, 
            n_dark_field_segments=n_batches, 
            verbosity=verbosity,
            bright_field_mask_threshold=bright_field_mask_threshold,
            upsample=upsample,
            return_snr=return_snr,
            snr_blur_sigma=snr_blur_sigma)
        else:
            tcDF = self.tilt_corrected_dark_field_image
            ssnr_tcdf = self.ssnr_tcdf
        
        fused, phase_weighted, tcdf_weighted = fused_full_field(
            self, phase_image, tcDF, ssnr_ptycho, ssnr_tcdf, verbosity=verbosity)

        end_time.record()
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time)
        print(
            f"Time to Full field fusing image: {elapsed_time/1000:.3f} s"
        )
        return fused, phase_weighted, tcdf_weighted

    def tilt_corrected_dark_field(self, 
        n_dark_field_segments : int = 32, 
        verbosity : int = 0,
        bright_field_mask_threshold : float = 0.3,
        store_image_in_dataset: bool = True,
        upsample: Union[float, str] = "nyquist",
        return_snr: bool = False,
        snr_blur_sigma: float = 0.0,
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform tilt corrected dark field reconstruction.
        """
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
        from scatterem2.reconstruction.tilt_corrected_dark_field import tilt_corrected_dark_field
        tcdf, ssnr_tcdf = tilt_corrected_dark_field(self, 
            n_dark_field_segments=n_dark_field_segments, 
            verbosity=verbosity,
            bright_field_mask_threshold=bright_field_mask_threshold,
            upsample=upsample,
            return_snr=return_snr,
            snr_blur_sigma=snr_blur_sigma,
        )
        if store_image_in_dataset:
            self.tilt_corrected_dark_field_image = tcdf
            self.ssnr_tcdf = ssnr_tcdf
        end_time.record()
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time)
        print(
            f"Time to reconstruct tcDF image: {elapsed_time/1000:.3f} s"
        )
        return tcdf, ssnr_tcdf

    def direct_ptychography(self, 
        upsample: Union[float, str] = "nyquist",
        bright_field_mask_threshold: float = 0.5,
        verbosity: int = 0,
        store_image_in_dataset: bool = True,
        n_batches: int = 25,
        return_snr: bool = False,
        **kwargs,) -> torch.Tensor:
        """
        Perform direct ptychography reconstruction.
        """
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
        from scatterem2.reconstruction.direct_ptychography import direct_ptychography
        phase_image, snr_ptycho = direct_ptychography(self, 
            upsample=upsample, 
            verbosity=verbosity,
            bright_field_mask_threshold=bright_field_mask_threshold,
            n_batches=n_batches,
            return_snr=return_snr)
        if store_image_in_dataset:
            self.direct_ptychography_phase_image = phase_image
            self.ssnr_ptycho = snr_ptycho
        end_time.record()
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time)
        print(
            f"Time to reconstruct directptychography image: {elapsed_time/1000:.3f} s"
        )
        return phase_image, snr_ptycho
 

    def bin_detector(self, bin_factor: int, pad_if_necessary: bool = True) -> "Dataset4dstem":
        """
        Bin the last two dimensions (detector dimensions) by bin_factor.
        If the last two dimensions are not integer divisible by bin_factor and pad_if_necessary is True,
        pad the last two dimensions before binning. Otherwise, raise an error.

        Args:
            bin_factor (int): The binning factor for the last two dimensions.
            pad_if_necessary (bool): Whether to pad the last two dimensions if not divisible.

        Returns:
            Dataset4dstem: New dataset with binned detector dimensions.
        """
        arr = self._array
        shape = arr.shape
        h, w = shape[-2], shape[-1]
        pad_h_total = (bin_factor - h % bin_factor) % bin_factor
        pad_w_total = (bin_factor - w % bin_factor) % bin_factor

        if (h % bin_factor != 0 or w % bin_factor != 0):
            if pad_if_necessary:
                # Pad evenly on both sides
                pad_h_left = pad_h_total // 2
                pad_h_right = pad_h_total - pad_h_left
                pad_w_left = pad_w_total // 2
                pad_w_right = pad_w_total - pad_w_left
                
                # For 4D tensor: pad last two dims (h, w)
                # Format: (w_left, w_right, h_top, h_bottom)
                arr = torch.nn.functional.pad(arr, pad=(pad_w_left, pad_w_right, pad_h_left, pad_h_right))
            else:
                raise ValueError(
                    f"Detector dimensions ({h}, {w}) are not divisible by bin_factor={bin_factor} "
                    "and pad_if_necessary is False."
                )

        new_h = arr.shape[-2] // bin_factor
        new_w = arr.shape[-1] // bin_factor

        # Reshape and sum to bin
        arr_binned = arr.reshape(
            *arr.shape[:-2],
            new_h, bin_factor,
            new_w, bin_factor
        ).sum(dim=(-1, -3))

        # Update sampling for detector dimensions
        if hasattr(self, "sampling") and self.sampling is not None:
            sampling = list(self.sampling)
            if len(sampling) >= 4:
                sampling[2] = sampling[2] * bin_factor
                sampling[3] = sampling[3] * bin_factor
            else:
                # fallback: don't update if not enough dims
                sampling = tuple(sampling)
        else:
            sampling = None

        # Update origin for detector dimensions
        if hasattr(self, "origin") and self.origin is not None:
            origin = list(self.origin)
            if len(origin) >= 4:
                origin[2] = origin[2] * bin_factor
                origin[3] = origin[3] * bin_factor
            else:
                origin = tuple(origin)
        else:
            origin = None

        # Create a copy of metadata and update it with new shape and sampling
        import copy
        new_meta = copy.deepcopy(self.meta)
        new_meta.shape = np.array(arr_binned.shape)
        
        # Update sampling for detector dimensions if available
        if hasattr(new_meta, 'sampling') and new_meta.sampling is not None:
            sampling = list(new_meta.sampling)
            if len(sampling) >= 4:
                sampling[2] = sampling[2] * bin_factor
                sampling[3] = sampling[3] * bin_factor
                new_meta.sampling = tuple(sampling)

        # Create new Dataset4dstem
        return type(self).from_array(
            arr_binned,
            name=self.name + f" (binned x{bin_factor})",
            origin=origin if origin is not None else self.origin,
            sampling=sampling if sampling is not None else self.sampling,
            units=self.units,
            signal_units=self.signal_units,
            meta=new_meta,
            transform_to_amplitudes=self.transform_to_amplitudes,
            device=self.device,
            normalize=False,
        )

    def crop_brightfield_(
        self: "Dataset4dstem",
        thresh_lower: float = 0.01,
        thresh_upper: float = 0.99,
    ) -> "Dataset4dstem":
        """
        Crop the dataset to the brightfield region without clone.
        """
        r, c = self.bright_field_radius_and_center(
            thresh_lower=thresh_lower, thresh_upper=thresh_upper
        )
        r = np.ceil(r)
        self._radius_bright_field = r

        r_int = int(np.ceil(r)) + 1
        y0_int = int(np.round(c[0]))
        x0_int = int(np.round(c[1]))
        # set the origin to the center of the bright field region
        self.origin[-1] = c[0]
        self.origin[-2] = c[1]
        rmax = self._array.shape[-1] // 2
        # crop the data to the bright field region
        r = min(r_int, rmax)
        print(f"Radius BF: {r}")
        print(f"Center BF: {c}")

        # Calculate the crop slice
        crop_slice = np.s_[:, :, y0_int - r : y0_int + r + 1, x0_int - r : x0_int + r + 1]
        
        # Crop the data
        data_bf = self.crop_(crop_slice)
        data_bf._radius_bright_field = r
        # Update the origin for the cropped dataset
        # Calculate the new origin based on the crop offset from the original center
        if hasattr(data_bf, 'origin') and data_bf.origin is not None:
            new_origin = list(data_bf.origin)
            
            # Calculate the offset from the original center to the cropped region
            crop_start_y = y0_int - r
            crop_start_x = x0_int - r
            
            # The new origin should be the original center minus the crop offset
            # This gives us the position of the original center in the new coordinate system
            new_origin_y = c[0] - crop_start_y
            new_origin_x = c[1] - crop_start_x
            
            # Update the detector dimensions (last two) to reflect the new origin
            new_origin[-2] = new_origin_y  # y-coordinate of new origin
            new_origin[-1] = new_origin_x   # x-coordinate of new origin
            data_bf.origin = tuple(new_origin)
        
        data_bf.is_bright_field = True
        return data_bf

    def bright_field_radius_and_center(
        self,
        thresh_lower: float = 0.1,
        thresh_upper: float = 0.6,
        N: int = 100,
        method: str = "area",
        edge_method: str = "canny",
        min_edge_points: int = 50,
        ransac_iterations: int = 1000,
        ransac_threshold: float = 2.0,
        plot_rbf: bool = False,
    ) -> Tuple[float, NDArray]:
        """
        Gets the center and radius of the probe in the diffraction plane using circle fitting.
        This method is robust to notches and missing sections by fitting a circle to edge points.

        The algorithm:
        1. Create binary masks using multiple thresholds
        2. For each mask, detect edges using specified method
        3. Fit circles to edge points using RANSAC for robustness
        4. Select the best circle based on consensus across thresholds
        5. Refine the result using least-squares fitting

        Args:
            thresh_lower (float): Lower threshold limit (0 to 1)
            thresh_upper (float): Upper threshold limit (0 to 1)
            N (int): Number of thresholds to test
            method (str): Method to use for finding the bright field ('circle', 'area')
            edge_method (str): Edge detection method ('canny', 'sobel', 'gradient')
            min_edge_points (int): Minimum edge points required for circle fitting
            ransac_iterations (int): Number of RANSAC iterations
            ransac_threshold (float): RANSAC inlier threshold in pixels

        Returns:
            r (float): Central disk radius in pixels
            center (NDArray): [y, x] position of disk center
        """
        thresh_vals = torch.linspace(thresh_lower, thresh_upper, N, device=self.device)

        # Get averaged diffraction pattern
        sx = min(self.array.shape[0], 50)
        sy = min(self.array.shape[1], 50)
        DP = self._array[:sx, :sy].mean((0, 1))
        DPmax = torch.max(DP)

        # Convert to numpy for OpenCV operations
        DP_np = (DP / DPmax).cpu().numpy().astype(np.float32)

        if method == "circle":
            circle_candidates = []

            for i, thresh in enumerate(thresh_vals):
                # Create binary mask
                mask = (DP_np > thresh.item()).astype(np.uint8)

                # Detect edges
                edges = detect_edges(mask, method=edge_method)

                # Find edge points
                edge_points = np.column_stack(np.where(edges))

                if len(edge_points) < min_edge_points:
                    continue

                # Fit circle using RANSAC
                circle = fit_circle_ransac(
                    edge_points, iterations=ransac_iterations, threshold=ransac_threshold
                )

                if circle is not None:
                    cy, cx, r = circle
                    # Validate circle is reasonable
                    h, w = DP_np.shape
                    if 0 <= cx <= w and 0 <= cy <= h and r > 5 and r < min(h, w) / 2:
                        circle_candidates.append((cy, cx, r, thresh.item(), i))
            
            if not circle_candidates:
                # Fallback to original method if no circles found
                return self._fallback_area_method(DP, thresh_lower, thresh_upper, N)

            # Select best circle based on consensus
            best_circle = select_best_circle(circle_candidates, DP_np)

            # Refine with least-squares fitting
            cy, cx, r = refine_circle_fit(best_circle, DP_np, edge_method)
        elif method == "area":
            r, c = self._fallback_area_method(DP, thresh_lower, thresh_upper, N)
            cy, cx = c
        else:
            raise ValueError(f"Unknown method: {method}")

        if plot_rbf:
            
            fig, ax = vis.show_2d(torch.from_numpy(DP_np), title="Diffraction Pattern")
            ax.add_patch(
                mlp.patches.Circle(
                    (cx, cy),
                    r,
                    fill=False,
                    linewidth=2,
                )
            )
            mlp.pyplot.show()
 
        print(
            f"Radius and center of the bright field disk (pixels): , {float(r):.2f}, {cx:.2f}, {cy:.2f}"
        )
        return float(r), np.array([cy, cx])

    def bright_field_radius_and_center_(self, thresh_lower: float = 0.1, thresh_upper: float = 0.6, N: int = 100, method: str = "area", edge_method: str = "canny", min_edge_points: int = 50, ransac_iterations: int = 1000, ransac_threshold: float = 2.0, plot_rbf: bool = False) -> Tuple[float, NDArray]:
        """
        Gets the center and radius of the probe in the diffraction plane using circle fitting.
        This method is robust to notches and missing sections by fitting a circle to edge points.
        """
        r, c = self.bright_field_radius_and_center(thresh_lower, thresh_upper, N, method, edge_method, min_edge_points, ransac_iterations, ransac_threshold, plot_rbf)
        self._radius_bright_field = r
        return r, c

    def _fallback_area_method(self, DP, thresh_lower: float, thresh_upper: float, N: int = 50
    ) -> Tuple[float, np.ndarray]:
        """Fallback to original area-based method if circle fitting fails."""
        # This is the original method from your code
        thresh_vals = torch.linspace(thresh_lower, thresh_upper, N, device=self.device)
        r_vals = torch.zeros(N, device=self.device)

        ind = min(1000, DP.shape[0])
        
        DPmax = torch.max(DP)

        for i in range(len(thresh_vals)):
            thresh = thresh_vals[i]
            mask = DP > DPmax * thresh
            r_vals[i] = torch.sqrt(torch.sum(mask) / torch.pi)

        dr_dtheta = torch.gradient(r_vals, dim=0)[0]
        mask = (dr_dtheta <= 0) * (dr_dtheta >= 2 * torch.median(dr_dtheta))
        r = torch.mean(r_vals[mask])

        thresh = torch.mean(thresh_vals[mask])
        mask = DP > DPmax * thresh
        ar = DP * mask
        nx, ny = ar.shape
        ry, rx = torch.meshgrid(
            torch.arange(nx, device=self.device), torch.arange(ny, device=self.device), indexing="ij"
        )
        print(ry.shape, rx.shape, ar.shape)
        tot_intens = torch.sum(ar)
        x0 = torch.sum(rx * ar) / tot_intens
        y0 = torch.sum(ry * ar) / tot_intens

        return float(r), np.array([y0.item(), x0.item()])

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
    def fluence_per_area(self) -> float:
        """Calculate total electron fluence (electrons per square Angstrom) from total intensity.

        Returns:
            float: Total electron fluence in electrons per square Angstrom.
        """
        scan_area = float(np.prod(self.sampling[:2] * np.array(self._shape[:2])))
        return self.total_intensity / scan_area

    @property
    def fluence_per_probe(self) -> float:
        """Calculate total electron fluence (electrons per square Angstrom) from total intensity.

        Returns:
            float: Total electron fluence in electrons per square Angstrom.
        """
        
        return self.dose_per_probe_unnormalized

    def __len__(self) -> int:
        return len(self._array3d)

    # def __getitem__(self, idx: int) -> Tensor:
    #     return self._array3d[idx]
    def __getitem__(
        self, item: list[int]
    ) -> tuple[int, int, int, list[int], int, Tensor]:
        """
        Expects batched indices in item, so a List
        Expects a 6-tuple as output (batch_index, probe_index, angles_index, r_indices, translation_index, amplitudes_target)
        """
        r_indices = item
        return (
            item[0],
            self.probe_index,
            0,
            r_indices,
            0,
            self._array3d[r_indices],
        )

    def crop(self, index: tuple[slice, ...], clone: bool = True) -> "Dataset4dstem":
        """
        Simple indexing function to return Dataset4dstem view.

        Parameters
        ----------
        index : tuple[slice, ...]
            Index to access a subset of the dataset
        clone : bool
            If True, the array is cloned before returning.

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
        print(f"New origin: {new_origin}")

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
        )

    def crop_(self, index: tuple[slice, ...]) -> "Dataset4dstem":
        """
        Simple indexing function to return Dataset4dstem view.

        Parameters
        ----------
        index : tuple[slice, ...]
            Index to access a subset of the dataset
        clone : bool
            If True, the array is cloned before returning.

        Returns
        -------
        dataset
            A new Dataset4dstem instance containing the indexed data
        """
        return self.crop(index, clone = False)

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
            self.meta.semiconvergence_angle / rbf / energy2wavelength(self.meta.energy)
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
        device: torch.device = torch.device("cpu"),
        is_cropped: bool = False,
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

        Returns
        -------
        Dataset
            A Dataset object with the validated array and metadata.
        """
        validated_array = ensure_valid_array(array, device=device)
        _ndim = validated_array.ndim

        # Set defaults if None
        _name = name if name is not None else f"{_ndim}d dataset"
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
        return self._is_bright_field if hasattr(self, '_is_bright_field') else False

    @is_bright_field.setter
    def is_bright_field(self, value: bool) -> None:
        self._is_bright_field = bool(value)

    @property
    def is_cropped(self) -> bool:
        """Whether the dataset has been cropped."""
        return self._is_cropped if hasattr(self, '_is_cropped') else False

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
    def __repr__(self) -> str:
        description = [
            f"Dataset(shape={self.shape}, dtype={self.dtype}, name='{self.name}')",
            f"  sampling: {self.sampling}",
            f"  units: {self.units}",
            f"  signal units: '{self.signal_units}'",
        ]
        return "\n".join(description)

    def __str__(self) -> str:
        description = [
            f"Dataset4dstem named '{self.name}'",
            f"  shape: {self.shape}",
            f"  dtype: {self.dtype}",
            f"  device: {self.device}",
            f"  origin: {self.origin}",
            f"  sampling: {self.sampling}",
            f"  units: {self.units}",
            f"  signal units: '{self.signal_units}'",
        ]
        return "\n".join(description)
