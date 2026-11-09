import copy
from typing import Any, ClassVar, Optional

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset

from scatterem.data._pretty import PrettyDatasetMixin
from scatterem.data.validation import coerce_ndinfo, ensure_valid_array, validate_units
from scatterem.utils.store import Serializable

from ._crop import _resolve_crop_slices_and_origin


def _place_aberrations_on_device(meta, device: torch.device) -> None:
    """Move ``meta.aberrations.array`` onto ``device``, in place.

    Warp kernels take the aberration array and the diffraction array in the
    same launch, so a CPU-resident aberration tensor on a CUDA dataset makes
    direct_ptychography fail with an opaque device-mismatch error unless the
    aberration fit happened to run first and move it as a side effect.

    Mutates the caller's ``Aberrations`` rather than rebinding
    ``meta.aberrations``: the vBF shares ``meta`` by reference with its parent
    dataset so that the aberration fit writing into ``vBF.meta.aberrations``
    is visible on the original -- a load-bearing convention a defensive copy
    would break.
    """
    aberrations = getattr(meta, "aberrations", None)
    if aberrations is not None and getattr(aberrations, "array", None) is not None:
        aberrations.array = aberrations.array.to(device)


class Dataset(PrettyDatasetMixin, TorchDataset, Serializable):
    """
    An array plus the calibration needed to interpret its axes.

    Each axis carries an origin, a pitch and a unit, so a position in the array
    can be turned into a physical coordinate without the caller tracking the
    conversion. Values have their own unit, separate from the axes'.

    Parameters
    ----------
    array : array-like
        The data itself.
    name : str
        Label used in reprs and figure titles.
    origin : numpy.ndarray, tuple, list, float, or int
        Coordinate of index zero on each axis.
    sampling : numpy.ndarray, tuple, list, float, or int
        Pitch per axis, in `units`.
    units : list of str, or tuple
        One unit string per axis.
    signal_units : str, optional
        Unit of the stored values. Default is "arb. units".
    _token : object, optional
        Internal sentinel guarding direct construction; use
        :meth:`Dataset.from_array` instead of calling this constructor.
    device : torch.device or str or None, optional
        ``None`` keeps a torch tensor's device; other inputs become CPU tensors.

    Attributes
    ----------
    array : torch.Tensor
        The underlying n-dimensional array data.
    name : str
        Label used in reprs and figure titles.
    origin : numpy.ndarray
        Coordinate of index zero on each axis.
    sampling : numpy.ndarray
        Pitch per axis, in `units`.
    units : list of str
        One unit string per axis.
    signal_units : str
        Unit of the stored values.
    """

    _token = object()
    #: What save() persists. ``device`` is not: a file must open on any machine.
    SERIAL_FIELDS = ("array", "name", "origin", "sampling", "units", "signal_units")
    SERIAL_INIT_KWARGS: ClassVar[dict[str, Any]] = {"_token": _token}

    def __init__(
        self,
        array: Any,  # Input can be array-like
        name: str,
        origin: NDArray | tuple | list | float | int,
        sampling: NDArray | tuple | list | float | int,
        units: list[str] | tuple | list,
        signal_units: str = "arb. units",
        _token: object | None = None,
        device: torch.device | str | None = None,
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
        origin: NDArray | tuple | list | float | int | None = None,
        sampling: NDArray | tuple | list | float | int | None = None,
        units: list[str] | tuple | list | None = None,
        signal_units: str = "arb. units",
        *,
        device: torch.device | str | None = None,
    ) -> "Dataset":
        """
        Validate an array and wrap it in a Dataset.

        Parameters
        ----------
        array : array-like
            The array to validate and create a Dataset from.
        name : str, optional
            The name of the Dataset. Defaults to ``"{rank}d dataset"``.
        origin : numpy.ndarray, tuple, list, float, int, or None
            The origin of the Dataset. Defaults to zeros.
        sampling : numpy.ndarray, tuple, list, float, int, or None
            The sampling of the Dataset. Defaults to ones.
        units : list of str, tuple, list, or None
            The units of the Dataset. Defaults to ``"pixels"`` per axis.
        signal_units : str
            The units of the signal.
        device : torch.device or str or None, optional
            ``None`` keeps a torch tensor's device; other inputs become CPU.

        Returns
        -------
        Dataset
            The container, with unset axis metadata filled from the array's rank.
        """
        tensor = ensure_valid_array(array, device=device)
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
        # A tensor keeps its own device; passing self.device undid ``.device = "cuda"``.
        device = value.device if isinstance(value, Tensor) else self.device
        self._array = ensure_valid_array(value, self.dtype, self.ndim, device)

    @property
    def name(self) -> str:
        """Label used in reprs and figure titles."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = str(value)

    @property
    def origin(self) -> NDArray:
        """Coordinate of index zero on each axis."""
        return self._origin

    @origin.setter
    def origin(self, value: NDArray | tuple | list | float | int) -> None:
        self._origin = coerce_ndinfo(value, self.ndim)

    @property
    def sampling(self) -> NDArray:
        """Pitch per axis, in `units`."""
        return self._sampling

    @sampling.setter
    def sampling(self, value: NDArray | tuple | list | float | int) -> None:
        self._sampling = coerce_ndinfo(value, self.ndim)

    @property
    def units(self) -> list[str]:
        """One unit string per axis."""
        return self._units

    @units.setter
    def units(self, value: list[str] | tuple | list) -> None:
        self._units = validate_units(value, self.ndim)

    @property
    def signal_units(self) -> str:
        """Unit of the stored values."""
        return self._signal_units

    @signal_units.setter
    def signal_units(self, value: str) -> None:
        self._signal_units = str(value)

    # --- Derived Properties ---
    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the underlying array."""
        return self.array.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions of the underlying array."""
        return self.array.ndim

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the underlying array."""
        return self.array.dtype

    @property
    def device(self) -> torch.device:
        """Device the underlying array is stored on."""
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
        Deep-copy the dataset.

        Returns
        -------
        Dataset
            A new dataset with a cloned array and deep-copied axis metadata.
        """
        return type(self).from_array(
            array=self.array.clone(),
            name=self.name,
            # numpy arrays or tuples (see the note above), so no autograd
            # graph to sever -- the measurement tensor is cloned separately.
            origin=copy.deepcopy(self.origin),  # noqa: TID251
            sampling=copy.deepcopy(self.sampling),  # noqa: TID251
            units=self.units[:],
            signal_units=self.signal_units,
        )

    def mean(self, axes: tuple[int, ...] | None = None) -> Any:
        """
        Compute the mean of the data array.

        Parameters
        ----------
        axes : tuple of int, optional
            Axes over which to compute the mean. If None, the mean of all
            elements is computed.

        Returns
        -------
        numpy.ndarray, torch.Tensor, or float
            Mean of the data.
        """
        return self.array.mean(dim=axes)




    def crop(
        self,
        crop_widths: tuple[tuple[int, int], ...],
        axes: int | tuple[int, ...] | None = None,
        modify_in_place: bool = False,
    ) -> Optional["Dataset"]:
        """Trim elements from the ends of axes, keeping coordinates consistent.

        ``origin`` moves by ``+before * sampling`` on each cropped axis, so a
        physical coordinate computed from the cropped array still lands on the
        same feature -- the counterpart of the shift :meth:`pad` applies. The
        implementation this replaces left ``origin`` alone, which silently moved
        the data in physical space. ``sampling`` is unchanged.

        Parameters
        ----------
        crop_widths : tuple of (int, int)
            A ``(before, after)`` pair per entry of ``axes``, giving how many
            elements to remove from each end. A single pair is accepted when
            ``axes`` is a single axis.
        axes : int, tuple of int, or None, optional
            Which axes to crop, defaulting to every axis -- in which case
            ``crop_widths`` must have one pair per axis. Negative indices
            count from the end.
        modify_in_place : bool, optional
            Crop ``self`` and return ``None`` rather than a copy.

        Returns
        -------
        Dataset or None
            The cropped copy, or ``None`` when ``modify_in_place`` is set.

        Raises
        ------
        ValueError
            If the widths and axes do not correspond, an axis is repeated or
            out of range, a width is negative, or a crop would consume an
            entire axis.
        """
        slices, origin = _resolve_crop_slices_and_origin(
            self.ndim,
            tuple(self.shape),
            self.origin,
            self.sampling,
            crop_widths,
            axes,
        )

        target = self if modify_in_place else self.copy()
        # clone(): a slice is a view, and writing it back to a dataset that shares
        # storage with the original has bitten this method before.
        target.array = self.array[slices].clone()
        target.origin = origin
        return None if modify_in_place else target


