from __future__ import annotations

import copy
from collections import defaultdict
from typing import Mapping, SupportsFloat

import numpy as np
import torch

from scatterem2.utils.energy import (
    HasAcceleratorMixin,
    energy2wavelength,
)
from scatterem2.utils.utils import get_dtype
from scatterem2.utils.stem import fftfreq2, _cartesian_aberrations

def aberrations_to_image_shifts(
    aberrations_array: torch.Tensor,
    rotation: torch.Tensor,
    sampling: np.ndarray,
    wavelength: float,
    shape: tuple[int, int] | torch.Size,
) -> torch.Tensor:
    """
    Calculate the bright field shifts from the aberrations and rotation.
    The bright field shifts are calculated by performing a rotation of the aberrations and then calculating the gradient of the aberrations with respect to the kx and ky.
    The gradient of the aberrations with respect to the kx and ky is then used to calculate the bright field shifts.
    The bright field shifts are then returned.

    Args:
        aberrations_array: torch.Tensor - aberrations array (1D: 12)
        rotation: torch.Tensor - rotation in degrees (1D: 1)
        sampling: torch.Tensor - sampling (2D: 2)
        wavelength: float - wavelength in Angstroms
        shape: tuple[int, int] | torch.Size - shape of the bright field mask (2D: H x W)

    Returns:
        torch.Tensor - bright field shifts (2D: N x 2), order by radius in the bright field, increasing order.
    """
    device = aberrations_array.device 
    if rotation.device != device:
        rotation = rotation.to(device)
    if isinstance(shape, torch.Size):
        shape = tuple(shape)
    else:
        shape = tuple(shape)
    q = torch.fft.fftshift(fftfreq2(shape, sampling, False, device=device), dim=(-2, -1))
    samy = torch.fft.fftshift(torch.fft.fftfreq(shape[0],d=sampling[0], device=device))#torch.linspace(-f, f, binary_mask.shape[0])
    samx = torch.fft.fftshift(torch.fft.fftfreq(shape[1],d=sampling[1], device=device))#torch.linspace(-f, f, binary_mask.shape[1])
    chi = _cartesian_aberrations(q[0], q[1], wavelength, aberrations_array)
    spacing = (samy, samx)
    dchi_dk0 = torch.stack(torch.gradient(chi, dim=(0, 1), spacing=spacing))
    # negative rotation because this is in the detector plane and we specify the dataset rotation in the real space scanning plane
    cos_rot = torch.cos(torch.deg2rad(-rotation))
    sin_rot = torch.sin(torch.deg2rad(-rotation))

    theta = torch.stack(
        [
            torch.stack([cos_rot, -sin_rot, torch.zeros_like(cos_rot)], dim=-1),
            torch.stack([sin_rot, cos_rot, torch.zeros_like(cos_rot)], dim=-1),
        ],
        dim=-2,
    )

    theta = theta.expand(dchi_dk0.shape[0], -1, -1)
    grid = torch.nn.functional.affine_grid(
        theta,
        size=(dchi_dk0.shape[0], 1, dchi_dk0.shape[1], dchi_dk0.shape[2]),
        align_corners=False,
    )
    # Rotate dchi_dk using grid_sample
    dchi_dk = torch.nn.functional.grid_sample(
        dchi_dk0.unsqueeze(1),
        grid,
        align_corners=False,
        mode="bicubic",
        padding_mode="border",
    ).squeeze()

    dchi_dkx_bf = dchi_dk[1] / (2 * np.pi)
    dchi_dky_bf = dchi_dk[0] / (2 * np.pi)
    fw_shifts = torch.stack(
        [dchi_dky_bf , dchi_dkx_bf ], dim=-1
    )
    return fw_shifts

def aberration_function_polar(
    alpha: torch.Tensor,
    phi: torch.Tensor,
    wavelength: float,
    aberrations: torch.Tensor,
) -> torch.Tensor:
    """
    Zernike polynomials in the polar coordinate system
    """
    chi = torch.zeros_like(phi)

    C10 = 0
    C12 = 1
    C21 = 2
    C23 = 3
    C30 = 4
    C32 = 5
    C34 = 6
    phi12 = 7
    phi21 = 8
    phi23 = 9
    phi32 = 10
    phi34 = 11

    if aberrations[C10] != 0 or aberrations[C12] != 0 or aberrations[phi12] != 0:
        chi = chi + (
            0.5
            * alpha**2.0
            * (
                aberrations[C10]
                + aberrations[C12] * torch.cos(2 * (phi - aberrations[phi12]))
            )
        )

    if (
        aberrations[C21] != 0
        or aberrations[phi21] != 0
        or aberrations[C23] != 0
        or aberrations[phi23] != 0
    ):
        chi = chi + (
            0.3333333333333333
            * alpha**3.0
            * (
                aberrations[C21] * torch.cos(phi - aberrations[phi21])
                + aberrations[C23] * torch.cos(3 * (phi - aberrations[phi23]))
            )
        )

    if (
        aberrations[C30] != 0
        or aberrations[C32] != 0
        or aberrations[phi32] != 0
        or aberrations[C34] != 0
        or aberrations[phi34] != 0
    ):
        chi = chi + (
            0.25
            * alpha**4.0
            * (
                aberrations[C30]
                + aberrations[C32] * torch.cos(2 * (phi - aberrations[phi32]))
                + aberrations[C34] * torch.cos(4 * (phi - aberrations[phi34]))
            )
        )
    chi *= 2 * torch.pi / wavelength
    return chi


def aberration_function_cartesian(
    qy: torch.Tensor, qx: torch.Tensor, wavelength: float, aberrations: torch.Tensor
) -> torch.Tensor:
    """
    Zernike polynomials in the cartesian coordinate system
    """
    u = qx * wavelength
    v = qy * wavelength
    u2 = u * u
    u3 = u2 * u
    u4 = u3 * u

    v2 = v * v
    v3 = v2 * v
    v4 = v3 * v

    chi = 0.0

    # r^2
    chi += 0.5 * aberrations[0] * (u2 + v2)

    # r^2 cos(2 phi) + r^2 sin(2 phi)
    chi += 0.5 * (aberrations[1] * (u2 - v2) + 2.0 * aberrations[2] * u * v)

    # r^3 cos(3phi) + r^3 sin(3 phi)
    chi += (1.0 / 3.0) * (
        aberrations[5] * (u3 - 3.0 * u * v2) + aberrations[6] * (3.0 * u2 * v - v3)
    )

    # r^3 cos(phi) + r^3 sin(phi)
    chi += (1.0 / 3.0) * (
        aberrations[3] * (u3 + u * v2) + aberrations[4] * (v3 + u2 * v)
    )

    # r^4
    chi += 0.25 * aberrations[7] * (u4 + v4 + 2.0 * u2 * v2)

    # r^4 cos(4 phi)
    chi += 0.25 * aberrations[10] * (u4 - 6.0 * u2 * v2 + v4)

    # r^4 sin(4 phi)
    chi += 0.25 * aberrations[11] * (4.0 * u3 * v - 4.0 * u * v3)

    # r^4 cos(2 phi)
    chi += 0.25 * aberrations[8] * (u4 - v4)

    # r^4 sin(2 phi)
    chi += 0.25 * aberrations[9] * (2.0 * u3 * v + 2.0 * u * v3)

    chi *= 2.0 * torch.pi / wavelength

    return chi


# def soft_aperture(
#     alpha: np.ndarray,
#     phi: np.ndarray,
#     semiangle_cutoff: float | np.ndarray,
#     angular_sampling: tuple[float, float],
# ) -> np.ndarray:
#     """
#     Calculates an array with a disk of ones and a soft edge.

#     Parameters
#     ----------
#     alpha : 2D array
#         Array of radial angles [mrad].
#     phi : 2D array
#         Array of azimuthal angles [rad].
#     semiangle_cutoff : float or 1D array
#         Semiangle cutoff(s) of the aperture(s). If given as an array, a 3D array is
#         returned where the first dimension represents a different aperture for each
#         item in the array of semiangle cutoffs.
#     angular_sampling : tuple of float
#         Reciprocal-space sampling in units of scattering angles [mrad].

#     Returns
#     -------
#     soft_aperture_array : 2D or 3D np.ndarray
#     """

#     semiangle_cutoff_array = torch.array(
#         semiangle_cutoff, dtype=get_dtype(complex=False)
#     )

#     base_ndims = len(alpha.shape)

#     semiangle_cutoff_array, alpha = expand_dims_to_broadcast(
#         semiangle_cutoff_array, alpha
#     )

#     semiangle_cutoff, phi = expand_dims_to_broadcast(
#         semiangle_cutoff_array, phi, match_dims=((-2, -1), (-2, -1))
#     )

#     angular_sampling = (
#         torch.tensor(angular_sampling, dtype=get_dtype(complex=False)) * 1e-3
#     )

#     denominator = torch.sqrt(
#         (torch.cos(phi) * angular_sampling[0]) ** 2
#         + (torch.sin(phi) * angular_sampling[1]) ** 2
#     )

#     ndims = len(alpha.shape)

#     zeros = (slice(None),) * (ndims - base_ndims) + (0,) * base_ndims

#     denominator[zeros] = 1.0

#     array = torch.clip(
#         (semiangle_cutoff - alpha) / denominator + 0.5, a_min=0.0, a_max=1.0
#     )

#     array[zeros] = 1.0
#     return array


def hard_aperture(alpha: np.ndarray, semiangle_cutoff: float) -> np.ndarray:
    """
    Calculates an array with a disk of ones and a soft edge.

    Parameters
    ----------
    alpha : 2D array
        Array of radial angles [mrad].
    semiangle_cutoff : float or 1D array
        Semiangle cutoff(s) of the aperture(s). If given as an array, a 3D array is
        returned where the first dimension represents a different aperture for each
        item in the array of semiangle cutoffs.

    Returns
    -------
    hard_aperture_array : 2D or 3D np.ndarray
    """

    return torch.array(alpha <= semiangle_cutoff).astype(get_dtype(complex=False))


def symbol_to_tex_symbol(symbol: str) -> str:
    tex_symbol = symbol.replace("C", "C_{").replace("phi", "\\phi_{") + "}"
    return f"${tex_symbol}$"


polar_aliases = {
    "defocus": "C10",
    "Cs": "C30",
    "C5": "C50",
    "astigmatism": "C12",
    "astigmatism_angle": "phi12",
    "astigmatism3": "C32",
    "astigmatism3_angle": "phi32",
    "astigmatism5": "C52",
    "astigmatism5_angle": "phi52",
    "coma": "C21",
    "coma_angle": "phi21",
    "coma4": "C41",
    "coma4_angle": "phi41",
    "trefoil": "C23",
    "trefoil_angle": "phi23",
    "trefoil4": "C43",
    "trefoil4_angle": "phi43",
    "quadrafoil": "C34",
    "quadrafoil_angle": "phi34",
    "quadrafoil5": "C54",
    "quadrafoil5_angle": "phi54",
    "pentafoil": "C45",
    "pentafoil_angle": "phi45",
    "hexafoil": "C56",
    "hexafoil_angle": "phi56",
}

polar_symbols = {value: key for key, value in polar_aliases.items()}


class _HasAberrations(HasAcceleratorMixin):
    C10: float
    C12: float
    phi12: float
    C21: float
    phi21: float
    C23: float
    phi23: float
    C30: float
    C32: float
    phi32: float
    C34: float
    phi34: float
    C41: float
    phi41: float
    C43: float
    phi43: float
    C45: float
    phi45: float
    C50: float
    C52: float
    phi52: float
    C54: float
    phi54: float
    C56: float
    phi56: float
    Cs: float
    C5: float
    astigmatism: float
    astigmatism_angle: float
    astigmatism3: float
    astigmatism3_angle: float
    astigmatism5: float
    astigmatism5_angle: float
    coma: float
    coma_angle: float
    coma4: float
    coma4_angle: float
    trefoil: float
    trefoil_angle: float
    trefoil4: float
    trefoil4_angle: float
    quadrafoil: float
    quadrafoil_angle: float
    quadrafoil5: float
    quadrafoil5_angle: float
    pentafoil: float
    pentafoil_angle: float
    hexafoil: float
    hexafoil_angle: float

    def __init__(self, *args, **kwargs):
        self._aberration_coefficients = {symbol: 0.0 for symbol in polar_symbols.keys()}
        super().__init__(*args, **kwargs)

    def __getattr__(self, name: str) -> float:
        name = polar_aliases.get(name, name)

        if name not in polar_symbols:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

        return self._aberration_coefficients.get(name, 0.0)

    def __setattr__(self, name: str, value: float) -> None:
        if name == "defocus":
            super().__setattr__(name, value)
            return

        name = polar_aliases.get(name, name)

        if name in polar_symbols:
            self._aberration_coefficients[name] = value
        else:
            super().__setattr__(name, value)

    @property
    def defocus(self) -> float:
        """Defocus equivalent to negative C10."""
        return -self.C10

    @defocus.setter
    def defocus(self, value: float) -> None:
        self.C10 = -value

    def _nonzero_coefficients(self, symbols: tuple[str, ...]) -> bool:
        for symbol in symbols:
            if not np.isscalar(self._aberration_coefficients[symbol]):
                return True

            if not self._aberration_coefficients[symbol] == 0.0:
                return True

        return False

    @property
    def aberration_coefficients(self) -> Mapping[str, float]:
        """The aberration coefficients as a dictionary."""
        return copy.deepcopy(self._aberration_coefficients)

    @property
    def _has_aberrations(self) -> bool:
        if np.all(
            [np.all(value == 0.0) for value in self._aberration_coefficients.values()]
        ):
            return False
        else:
            return True

    def set_aberrations(
        self, aberration_coefficients: Mapping[str, str | float]
    ) -> None:
        """
        Set the phase of the phase aberration.

        Parameters
        ----------
        aberration_coefficients : dict
            Mapping from aberration symbols to their corresponding values.
        """
        for symbol, value in aberration_coefficients.items():
            if symbol in ("defocus", "C10"):
                if isinstance(value, str) and value.lower() == "scherzer":
                    if self.energy is None:
                        raise RuntimeError(
                            "energy undefined, Scherzer defocus cannot be evaluated"
                        )
                    C30 = self._aberration_coefficients["C30"]
                    assert isinstance(C30, SupportsFloat)
                    value = scherzer_defocus(float(C30), self._valid_energy)

            if isinstance(value, str):
                raise ValueError("string values only allowed for defocus")

            setattr(self, symbol, value)


def nyquist_sampling(semiangle_cutoff: float, energy: float) -> float:
    """
    Calculate the Nyquist sampling.

    Parameters
    ----------
    semiangle_cutoff: float
        Semiangle cutoff [mrad].
    energy: float
        Electron energy [eV].
    """
    wavelength = energy2wavelength(energy)
    return 1 / (4 * semiangle_cutoff / wavelength * 1e-3)


def scherzer_defocus(Cs: float, energy: float) -> float:
    """
    Calculate the Scherzer defocus.

    Parameters
    ----------
    Cs: float
        Spherical aberration [Å].
    energy: float
        Electron energy [eV].
    """
    return np.sign(Cs) * np.sqrt(3 / 2 * np.abs(Cs) * energy2wavelength(energy))


def polar2cartesian(polar: dict) -> dict:
    """
    Convert between polar and Cartesian aberration coefficients.

    Parameters
    ----------
    polar : dict
        Mapping from polar aberration symbols to their corresponding values.

    Returns
    -------
    cartesian : dict
        Mapping from Cartesian aberration symbols to their corresponding values.
    """

    polar = defaultdict(lambda: 0, polar)

    cartesian = dict()
    cartesian["C10"] = polar["C10"]
    cartesian["C12a"] = -polar["C12"] * np.cos(2 * polar["phi12"])
    cartesian["C12b"] = polar["C12"] * np.sin(2 * polar["phi12"])
    cartesian["C21a"] = polar["C21"] * np.sin(polar["phi21"])
    cartesian["C21b"] = polar["C21"] * np.cos(polar["phi21"])
    cartesian["C23a"] = -polar["C23"] * np.sin(3 * polar["phi23"])
    cartesian["C23b"] = polar["C23"] * np.cos(3 * polar["phi23"])
    cartesian["C30"] = polar["C30"]
    cartesian["C32a"] = -polar["C32"] * np.cos(2 * polar["phi32"])
    cartesian["C32b"] = polar["C32"] * np.cos(np.pi / 2 - 2 * polar["phi32"])
    cartesian["C34a"] = polar["C34"] * np.cos(-4 * polar["phi34"])
    k = np.sqrt(3 + np.sqrt(8.0))
    cartesian["C34b"] = (
        1
        / 4.0
        * (1 + k**2) ** 2
        / (k**3 - k)
        * polar["C34"]
        * np.cos(4 * np.arctan(1 / k) - 4 * polar["phi34"])
    )

    return cartesian


def cartesian2polar(cartesian: dict) -> dict:
    """
    Convert between Cartesian and polar aberration coefficients.

    Parameters
    ----------
    cartesian : dict
        Mapping from Cartesian aberration symbols to their corresponding values.

    Returns
    -------
    polar : dict
        Mapping from polar aberration symbols to their corresponding values.
    """

    cartesian = defaultdict(lambda: 0, cartesian)

    polar = dict()
    polar["C10"] = cartesian["C10"]
    polar["C12"] = -np.sqrt(cartesian["C12a"] ** 2 + cartesian["C12b"] ** 2)
    polar["phi12"] = -np.arctan2(cartesian["C12b"], cartesian["C12a"]) / 2.0
    polar["C21"] = np.sqrt(cartesian["C21a"] ** 2 + cartesian["C21b"] ** 2)
    polar["phi21"] = np.arctan2(cartesian["C21a"], cartesian["C21b"])
    polar["C23"] = np.sqrt(cartesian["C23a"] ** 2 + cartesian["C23b"] ** 2)
    polar["phi23"] = -np.arctan2(cartesian["C23a"], cartesian["C23b"]) / 3.0
    polar["C30"] = cartesian["C30"]
    polar["C32"] = -np.sqrt(cartesian["C32a"] ** 2 + cartesian["C32b"] ** 2)
    polar["phi32"] = -np.arctan2(cartesian["C32b"], cartesian["C32a"]) / 2.0
    polar["C34"] = np.sqrt(cartesian["C34a"] ** 2 + cartesian["C34b"] ** 2)
    polar["phi34"] = np.arctan2(cartesian["C34b"], cartesian["C34a"]) / 4

    return polar


def pair_overlap_area(d, R):
    """
    Area of overlap of two circles of radius R with centre separation d.
    Returns 0 for d >= 2R.
    """
    d = np.asarray(d, dtype=np.float64)
    A = np.zeros_like(d)

    mask = d < 2*R
    dm = d[mask]
    A[mask] = (
        2 * R**2 * np.arccos(dm / (2*R))
        - 0.5 * dm * np.sqrt(4*R**2 - dm**2)
    )
    return A

def triple_overlap_area(q, R):
    """
    Triple-overlap area A3(q) for three circles of radius R
    centred at -q, 0, +q along one axis.
    Nonzero only for 0 <= q <= R.
    """
    q = np.asarray(q, dtype=np.float64)
    A3 = np.zeros_like(q)

    mask = q <= R
    qm = q[mask]
    A3[mask] = (
        np.pi * R**2
        - 2 * R**2 * np.arcsin(qm / R)
        - 2 * qm * np.sqrt(R**2 - qm**2)
    )
    return A3

def double_and_triple_overlap_areas(q, R):
    """
    Return (A2(q), A3(q)) for all q, where:
      A2 = area with exactly double overlap
      A3 = area with triple overlap
    for three circles of radius R at -q, 0, +q.
    """
    q = np.asarray(q, dtype=np.float64)

    # Triple overlap
    A3 = triple_overlap_area(q, R)

    # Pairwise overlaps
    A01 = pair_overlap_area(q, R)      # central with +q
    A12 = pair_overlap_area(q, R)      # central with -q
    A02 = pair_overlap_area(2*q, R)    # +q with -q

    # Exactly double-overlap area
    A2 = A01 + A12 + A02 - 3 * A3

    # For q > 2R, enforce zero
    A2[q >= 2*R] = 0.0
    A3[q >= 2*R] = 0.0

    return A2, A3

def double_and_triple_pixel_counts(q, R, delta_k):
    """
    Number of pixels in double- and triple-overlap regions, given
    k-space pixel size delta_k.
    """
    A2, A3 = double_and_triple_overlap_areas(q, R)
    pix_area = delta_k**2
    N2 = A2 / pix_area
    N3 = A3 / pix_area
    return N2, N3

def soft_aperture(
    alpha: torch.Tensor,
    phi: torch.Tensor,
    semiangle_cutoff: float,
    angular_sampling: tuple[float, float],
) -> torch.Tensor:
    """
    Calculates circular aperture with soft edges.

    Parameters
    ----------
    alpha: torch.Tensor
        Radial component of the polar frequencies [rad].
    phi: torch.Tensor
        Angular component of the polar frequencies.
    semiangle_cutoff: float
        The semiangle cutoff describes the sharp Fourier space cutoff due to the objective aperture [mrad].
    angular_sampling: Tuple[float,float]
        Sampling of the polar frequencies grid in mrad.

    Returns
    -------
    aperture: torch.Tensor
        circular aperture tensor with soft edges.
    """
    semiangle_rad = semiangle_cutoff * 1e-3
    denominator = torch.sqrt(
        (torch.cos(phi) * angular_sampling[0] * 1e-3).square()
        + (torch.sin(phi) * angular_sampling[1] * 1e-3).square()
    )
    array = torch.clip(
        (semiangle_rad - alpha) / denominator + 0.5,
        0,
        1,
    )
    return array.to(torch.float32)
    
def polar_coordinates(kx: torch.Tensor, ky: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """ """
    k = torch.sqrt(kx.square() + ky.square())
    phi = torch.arctan2(ky, kx)
    return k, phi

def aperture_autocorrelation(q, k0, normalize=False):
    """
    Analytical aperture autocorrelation for a circular top-hat aperture.

    Parameters
    ----------
    q : array-like
        Radial spatial frequency values |q|.
    k0 : float
        Aperture radius (probe semiangle in frequency units).
    normalize : bool, optional
        If True, normalizes the autocorrelation by (π k0^2)^2 so that the
        autocorrelation integrates to 1, consistent with a unit-integral
        aperture function.

    Returns
    -------
    AstarA : array-like
        Radial aperture autocorrelation values A ⋆ A (q).
    """

    q = np.asarray(q, dtype=np.float64)

    # Initialize output
    AstarA = np.zeros_like(q)

    # Valid region: |q| < 2 k0
    mask = (q < 2*k0)

    # Geometric overlap area of two disks of radius k0 separated by distance q
    # Only evaluate where mask=True, avoid invalid sqrt
    qm = q[mask]
    term1 = 2 * k0**2 * np.arccos(qm / (2*k0))
    term2 = 0.5 * qm * np.sqrt(4*k0**2 - qm**2)
    AstarA[mask] = term1 - term2

    # Optional normalization (because ∫A d k = 1 → A = 1/(π k0^2))
    if normalize:
        AstarA /= (np.pi * k0**2)**2

    return AstarA