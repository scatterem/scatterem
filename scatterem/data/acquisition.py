"""Acquisition metadata of a 4D-STEM measurement.

:class:`AcquisitionMetadata` (formerly ``Metadata4dstem``) records the beam,
the per-axis sampling/units/shape and the lens state of one acquisition. It is
the one metadata class: its former base ``MetadataNew`` and the legacy
``Metadata``/``Metadata4D`` were deleted in M3.
Files written under the former name still load: it is listed in
:attr:`AcquisitionMetadata.SERIAL_FORMER_NAMES`, which the declared-schema
store matches alongside the class name. So is ``Metadata4dstemAndGeometry``,
the subclass that added a ``geometry`` by inheritance until the class held one
by composition (M2); files written without ``geometry`` load with ``None``
(:attr:`AcquisitionMetadata.SERIAL_OPTIONAL`).
"""

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from scatterem.data._acquisition_constructors import _AcquisitionConstructors
from scatterem.data._acquisition_fields import _parent_scan, _warp_centre
from scatterem.data.aberrations import Aberrations
from scatterem.data.detector_warp import as_detector_warp
from scatterem.data.geometry import Geometry
from scatterem.data.validation import validate_ndinfo, validate_units
from scatterem.physics.constants import debroglie_at
from scatterem.utils.store import Serializable








class _DetectorReach:
    """Mixin: ``k_max`` and ``dr`` from ``sampling`` and ``detector_shape``.

    :class:`AcquisitionMetadata` and
    :class:`~scatterem.data.datasets.Dataset4DStem` share it.
    """

    @property
    def k_max(self: Any) -> NDArray:
        """Maximum scattering vector magnitude on the detector, per axis.

        Returns
        -------
        numpy.ndarray
            Half the detector extent in reciprocal space, in inverse
            angstroms: ``sampling[-2:] * detector_shape / 2``.
        """
        return self.sampling[-2:] * self.detector_shape / 2

    @property
    def dr(self) -> NDArray:
        """Real-space sampling of the *detector* plane (Nyquist, ``1/(2*k_max)``).

        Despite the name, this is NOT the real-space scan step. See
        ``detector_sampling`` of
        :class:`~scatterem.data.acquisition.AcquisitionMetadata` for a less
        ambiguous alias of the same value.

        Returns
        -------
        numpy.ndarray
            Real-space sampling of the detector in angstroms.
        """
        return 1 / (2 * self.k_max)


@dataclass
class AcquisitionMetadata(Serializable, _AcquisitionConstructors, _DetectorReach):
    """Metadata class containing physical parameters and scan geometry for 4D-STEM.

    The one acquisition-metadata class: the beam, the per-axis sampling, units
    and shape of the 4D-STEM array, and the lens state.

    Parameters
    ----------
    energy : float
        Electron beam energy in electron volts.
    semiconvergence_angle : float
        Beam convergence semi-angle in radians.
    sampling : array-like or float or int
        Per-axis sampling (real- and/or reciprocal-space) of the 4D-STEM array.
    shape : numpy.ndarray
        Shape of the 4D-STEM array.
    units : list of str, optional
        Per-axis units for ``sampling``. Default is inferred from `sampling`'s
        dimensionality.
    rotation : float, optional
        Sample rotation angle in degrees. Default is 0.
    defocus_guess : float, optional
        Initial guess of the defocus in Angstroms. Default is 0.
    aberrations : Aberrations, optional
        Lens aberration coefficients. Default is all zeros.
    slice_thickness : float, optional
        Multislice slice thickness in Angstroms. Default is 0.
    sample_thickness_guess : float, optional
        Initial guess of sample thickness in Angstroms. Default is 0.
    vacuum_probe : Tensor, optional
        Complex vacuum probe wavefunction. Default is None.
    scan_affine : Tensor, optional
        Affine correction applied to the scan grid. Default is None.
    geometry : Geometry, optional
        Keyword only. Scan/projection pose of this acquisition. Default is
        None, which the ptychography models read as zero rotation and
        translation (see ``with_geometry``).
    parent_scan : sequence of int, optional
        Keyword only. ``(y0, x0, NY, NX)``: this scan is the window of a
        parent raster of ``NY x NX`` positions (same step and rotation) that
        starts at row ``y0`` and column ``x0``. The ptychography models then
        place each position where the parent raster puts it, and size the
        object field of view for the whole parent raster, so tilts with
        different windows share one frame. Default is None: the scan is its
        own raster, centred in the field of view.
    detector_warp : array_like, optional
        Keyword only. The detector warp: the ``(2, 2)`` elliptical k-metric
        ``W`` (a pixel at offset ``p`` from the disk centre records
        ``k = W (p dk)``) or a ``(T, 2)`` polynomial table. See
        ``scatterem.data.detector_warp``. Default is None (no warp).
    detector_warp_centre : (float, float), optional
        Keyword only. The disk centre ``(y, x)`` in pixels of the stored
        detector, the pivot of the warp. Detector crops, bins and pads move it
        (``data.datasets._meta``). Default is None: ``(H // 2, W // 2)``.
    """

    SERIAL_FIELDS = ("energy", "defocus_guess", "sample_thickness_guess")
    SERIAL_FIELDS += ("semiconvergence_angle", "units")
    SERIAL_FIELDS += ("sampling", "shape", "rotation", "aberrations", "vacuum_probe")
    SERIAL_FIELDS += ("slice_thickness", "scan_affine", "geometry", "parent_scan")
    SERIAL_FIELDS += ("detector_warp", "detector_warp_centre")
    SERIAL_NESTED: ClassVar[dict] = {"aberrations": Aberrations, "geometry": Geometry}
    SERIAL_OPTIONAL: ClassVar[tuple[str, ...]] = (
        "geometry",
        "parent_scan",
        "detector_warp",
        "detector_warp_centre",
    )
    SERIAL_FORMER_NAMES: ClassVar[tuple[str, ...]] = (
        "Metadata4dstem",
        "Metadata4dstemAndGeometry",
    )
    energy: float
    defocus_guess: float
    sample_thickness_guess: float
    vacuum_probe: Tensor | None
    semiconvergence_angle: float
    rotation: float
    aberrations: Aberrations
    # A class-level default too: a legacy store restores instances without
    # calling ``__init__``, and one written before M2 records no geometry.
    geometry: Geometry | None = None
    parent_scan: tuple[int, int, int, int] | None = None
    detector_warp: Tensor | None = None
    detector_warp_centre: tuple[float, float] | None = None

    def __init__(
        self,
        energy: float,
        semiconvergence_angle: float,
        sampling: NDArray | tuple | list | float | int,
        shape: NDArray | None = None,
        units: list[str] | tuple | list | None = None,
        rotation: float | None = 0,
        defocus_guess: float | None = 0,
        aberrations: Aberrations | None = None,
        slice_thickness: float | None = 0,
        sample_thickness_guess: float | None = 0,
        vacuum_probe: Tensor | None = None,
        scan_affine: Tensor | NDArray | list[float] | None = None,
        *,
        geometry: Geometry | None = None,
        parent_scan: Any = None,
        detector_warp: Any = None,
        detector_warp_centre: Any = None,
    ) -> None:
        """Initialize from the parameters the class docstring documents."""
        self.energy = energy
        self.defocus_guess = defocus_guess or 0
        self.sample_thickness_guess = sample_thickness_guess or 0
        self.vacuum_probe = vacuum_probe
        self.semiconvergence_angle = semiconvergence_angle
        self.rotation = rotation or 0
        _ndim = len(sampling) if isinstance(sampling, (np.ndarray, tuple, list)) else 4
        if units is None:
            _base_units = ["A", "A", "A^-1", "A^-1"]
            units = (_base_units * (_ndim // len(_base_units) + 1))[:_ndim]
        self.sampling = validate_ndinfo(sampling, _ndim, "sampling")
        self.units = validate_units(units, _ndim)
        self.ndim = len(self.sampling)
        self.shape = shape
        self.aberrations = (
            aberrations if aberrations is not None else Aberrations(torch.zeros((12,)))
        )
        self.scan_affine = scan_affine
        self.slice_thickness = slice_thickness
        self.geometry = geometry
        self.parent_scan = _parent_scan(parent_scan)
        self.detector_warp = as_detector_warp(detector_warp)
        self.detector_warp_centre = _warp_centre(detector_warp_centre)


    @property
    def wavelength(self) -> float:
        """Relativistic electron wavelength at ``energy``, in angstroms."""
        return debroglie_at(self.energy)

    @property
    def detector_shape(self) -> NDArray:
        """Shape of the detector, i.e. the last two axes of `shape`."""
        if self.shape is None:
            return np.array([1, 1])
        return np.array(self.shape[-2:])

    @property
    def detector_sampling(self) -> NDArray:
        """Real-space sampling of the detector plane (Nyquist, ``1/(2*k_max)``).

        A less ambiguous alias for ``dr``, which is the detector's real-space
        sampling, not the real-space scan step.

        Returns:
            float: Real space sampling of the detector in Angstroms.
        """
        return self.dr






    def __str__(self) -> str:
        """Generate string representation of metadata.

        Returns:
            str: Formatted string containing metadata values.
        """
        base_str = (
            "AcquisitionMetadata:\n"
            f"  alpha_rad:     {self.semiconvergence_angle}\n"
            f"  rotation_deg:  {self.rotation:2.2f}\n"
            f"  E_ev:          {self.energy:2.2f}\n"
            f"  wavelength:    {self.wavelength:2.2f}\n"
            f"  defocus_guess: {self.defocus_guess:2.2f}\n"
            f"  sample_thickness_guess: {self.sample_thickness_guess:2.2f}\n"
            f"  sampling:      {self.sampling}\n"
            f"  units:         {self.units}\n"
            f"  shape:         {self.shape}\n"
        )
        if self.aberrations is not None:
            base_str += f"\n  aberrations:   {self.aberrations.array.cpu().numpy()}\n"
        if self.vacuum_probe is not None:
            base_str += f"\n  vacuum_probe:  {self.vacuum_probe.shape}"
        if self.geometry is not None:
            base_str += f"\n{self.geometry}"
        return base_str

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            str: Same as __str__().
        """
        return self.__str__()

    @property
    def device(self):
        """Device the aberration coefficients are stored on.

        Returns
        -------
        torch.device or None
            Device of ``aberrations.array``; ``None`` without aberrations.
            Assigning moves them and the vacuum probe.
        """
        ab = getattr(self, "aberrations", None)
        return ab.array.device if ab is not None and hasattr(ab, "array") else None

    @device.setter
    def device(self, value):
        ab = getattr(self, "aberrations", None)
        if ab is not None and hasattr(ab, "array"):
            ab.array = ab.array.to(value)
        if self.vacuum_probe is not None:
            self.vacuum_probe = self.vacuum_probe.to(value)
