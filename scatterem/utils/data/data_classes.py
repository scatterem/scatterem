from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from scatterem.utils.physics import electron_wavelength
from scatterem.io.store import Serializable
from scatterem.utils.data.aberrations import Aberrations
from scatterem.utils.data.validation import validate_ndinfo, validate_units




@dataclass
class MetadataNew(Serializable):
    """Base metadata class containing physical parameters for electron microscopy.

    Attributes:
        semiconvergence_angle (float): Beam convergence semi-angle in radians.
        energy (float): Electron beam energy in electron volts.
        rotation (float): Sample rotation angle in degrees.
        defocus_guess (float): Initial guess of sample thickness in Angstroms.
        sample_thickness_guess (float): Initial guess of sample thickness in Angstroms.
    """

    energy: float

    defocus_guess: float
    sample_thickness_guess: float

    @property
    def wavelength(self) -> float:
        """Calculate electron wavelength from energy.

        Returns:
            float: Electron wavelength in Angstroms.
        """
        return electron_wavelength(self.energy)

    def __str__(self) -> str:
        """Pretty print all class members"""
        return (
            "Metadata:\n"
            f"  energy: {self.energy}\n"
            f"  defocus_guess: {self.defocus_guess}\n"
            f"  sample_thickness_guess: {self.sample_thickness_guess}"
        )

    @property
    def device(self):
        """
        Returns the device of the aberrations tensor if present, otherwise None.
        """
        if hasattr(self, "aberrations") and self.aberrations is not None and hasattr(self.aberrations, "array"):
            return self.aberrations.array.device
        return None

    @device.setter
    def device(self, value):
        """
        Sets the device for the aberrations tensor if present.
        """
        if hasattr(self, "aberrations") and self.aberrations is not None and hasattr(self.aberrations, "array"):
            self.aberrations.array = self.aberrations.array.to(value)






@dataclass
class Metadata4dstem(MetadataNew):
    SERIAL_FIELDS = (
        "energy",
        "semiconvergence_angle",
        "sampling",
        "shape",
        "units",
        "rotation",
        "defocus_guess",
        "aberrations",
        "sample_thickness_guess",
        "vacuum_probe",
    )
    SERIAL_NESTED = {"aberrations": Aberrations}

    """Metadata class containing physical parameters and scan geometry for 4D-STEM.

    Extends the base Metadata class with additional attributes specific to 4D-STEM scanning.

    Attributes:
        vacuum_probe (Tensor): Complex vacuum probe wavefunction.
    """

    vacuum_probe: Optional[Tensor]
    semiconvergence_angle: float
    rotation: float
    aberrations: Optional[Aberrations]

    def __init__(
        self,
        energy: float,
        semiconvergence_angle: float,
        sampling: Union[NDArray, tuple, list, float, int],
        shape: NDArray,
        units: Union[list[str], tuple, list, None] = None,
        rotation: Optional[float] = 0,
        defocus_guess: Optional[float] = 0,
        aberrations: Optional[Aberrations] = None,
        slice_thickness: Optional[float] = 0,
        sample_thickness_guess: Optional[float] = 0,
        vacuum_probe: Optional[Tensor] = None,
        scan_affine: Optional[Tensor] = None,
    ) -> None:
        """Initialize Metadata4D object with experimental parameters.

        Args:
            energy (float): Electron beam energy in electron volts.
            semiconvergence_angle (float): Beam convergence semi-angle in radians.
            rotation (Optional[float], optional): Sample rotation angle in degrees. Defaults to 0.
            defocus_guess (Optional[float], optional): Initial guess of sample thickness in Angstroms. Defaults to 0.
            sample_thickness_guess (Optional[float], optional): Initial guess of sample thickness in Angstroms. Defaults to 0.
            vacuum_probe (Optional[Tensor], optional): Complex vacuum probe wavefunction. Defaults to None.


        """
        super().__init__(energy, defocus_guess or 0, sample_thickness_guess or 0)
        self.vacuum_probe = vacuum_probe
        self.semiconvergence_angle = semiconvergence_angle
        self.rotation = rotation or 0
        _ndim = len(sampling) if hasattr(sampling, "__len__") else 4
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

    @property
    def detector_shape(self) -> NDArray:
        """ """
        return np.array(self.shape[-2:])

    @property
    def k_max(self) -> NDArray:
        """Calculate maximum scattering vector magnitude from semiconvergence angle and detector shape.

        Returns:
            float: Maximum scattering vector magnitude in inverse Angstroms.
        """
        return self.sampling[-2:] * self.detector_shape / 2

    @property
    def dr(self) -> NDArray:
        """Real-space sampling of the *detector* plane (Nyquist, ``1/(2*k_max)``).

        Note: despite the name, this is NOT the real-space scan step -- see
        ``detector_sampling`` for a less ambiguous alias of the same value.

        Returns:
            float: Real space sampling of the detector in Angstroms.
        """
        return 1 / (2 * self.k_max)

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
            "Metadata4dstem:\n"
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
        return base_str

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            str: Same as __str__().
        """
        return self.__str__()

    @property
    def device(self):
        """
        Returns the device of the aberrations tensor if present, otherwise None.
        """
        if hasattr(self, "aberrations") and self.aberrations is not None and hasattr(self.aberrations, "array"):
            return self.aberrations.array.device
            
        return None

    @device.setter
    def device(self, value):
        """
        Sets the device for the aberrations tensor if present.
        """
        if hasattr(self, "aberrations") and self.aberrations is not None and hasattr(self.aberrations, "array"):
            self.aberrations.array = self.aberrations.array.to(value)

        if hasattr(self, "vacuum_probe") and self.vacuum_probe is not None:
            self.vacuum_probe = self.vacuum_probe.to(value)
 
 





