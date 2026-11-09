"""Physical constants and relativistic electron kinematics.

This module was written for scatterem from the following literature:

- Tiesinga, E., Mohr, P. J., Newell, D. B. and Taylor, B. N. (2021). CODATA
  recommended values of the fundamental physical constants: 2018. Rev. Mod.
  Phys. 93, 025010.
- BIPM (2019). The International System of Units (SI), 9th ed.
- Kirkland, E. J. (2020). Advanced Computing in Electron Microscopy, 3rd ed.,
  ch. 2 (wavelength and mass) and ch. 5 (interaction parameter).
- Reimer, L. and Kohl, H. (2008). Transmission Electron Microscopy, 5th ed.,
  ch. 2 (relativistic kinematics).
- Aharonov, Y. and Bohm, D. (1959). Phys. Rev. 115, 485-491.
- Lichte, H. and Lehmann, M. (2008). Rep. Prog. Phys. 71, 016102.

Naming scheme of the constants. A prefix tells the source of the value:

- ``EXACT_``: a defining constant of the SI (BIPM 2019). It has no uncertainty.
- ``CODATA18_``: a CODATA 2018 recommended value with all digits of the table.
  The docstring comment gives its relative standard uncertainty ``u_r``.
- ``DERIVED_``: this module computes the value from the values above.

The module imports only the Python standard library.
"""

from __future__ import annotations

import math
import numbers

__all__ = [
    "CODATA18_DALTON",
    "CODATA18_ELECTRON_MASS",
    "CODATA18_ELECTRON_RADIUS",
    "CODATA18_FINE_STRUCTURE",
    "CODATA18_VACUUM_MU",
    "DERIVED_AB_COUPLING",
    "DERIVED_ELECTRON_REST_EV",
    "DERIVED_PLANCK_BAR",
    "EXACT_AVOGADRO",
    "EXACT_BOLTZMANN",
    "EXACT_E_CHARGE",
    "EXACT_LIGHT_SPEED",
    "EXACT_PLANCK",
    "bend_per_tesla_at",
    "debroglie_at",
    "gamma_at",
    "moving_mass_at",
    "phase_per_potential_at",
]

# --- Defining constants of the SI (BIPM 2019), exact -----------------------

EXACT_PLANCK = 6.62607015e-34
"""Planck constant ``h`` [J s]. Exact in the SI."""

EXACT_E_CHARGE = 1.602176634e-19
"""Elementary charge ``e`` [C]. Exact in the SI."""

EXACT_LIGHT_SPEED = 299792458.0
"""Speed of light in vacuum ``c`` [m/s]. Exact in the SI."""

EXACT_AVOGADRO = 6.02214076e23
"""Avogadro constant ``N_A`` [1/mol]. Exact in the SI."""

EXACT_BOLTZMANN = 1.380649e-23
"""Boltzmann constant ``k_B`` [J/K]. Exact in the SI."""

# --- CODATA 2018 recommended values (Tiesinga et al. 2021) -----------------

CODATA18_ELECTRON_MASS = 9.1093837015e-31
"""Electron rest mass ``m0`` [kg]. CODATA 2018, u_r = 3.0e-10."""

CODATA18_DALTON = 1.66053906660e-27
"""Atomic mass constant ``m_u`` [kg]. CODATA 2018, u_r = 3.0e-10."""

CODATA18_FINE_STRUCTURE = 7.2973525693e-3
"""Fine-structure constant ``alpha`` [1]. CODATA 2018, u_r = 1.5e-10."""

CODATA18_VACUUM_MU = 1.25663706212e-6
"""Magnetic constant ``mu_0`` [N/A^2]. CODATA 2018, u_r = 1.5e-10."""

CODATA18_ELECTRON_RADIUS = 2.8179403262e-15
"""Classical electron radius ``r_e`` [m]. CODATA 2018, u_r = 4.5e-10."""

# --- Derived values --------------------------------------------------------

DERIVED_PLANCK_BAR = EXACT_PLANCK / (2.0 * math.pi)
"""Reduced Planck constant ``hbar = h / (2 pi)`` [J s]. Exact in principle."""

DERIVED_ELECTRON_REST_EV = (
    CODATA18_ELECTRON_MASS * EXACT_LIGHT_SPEED**2 / EXACT_E_CHARGE
)
"""Electron rest energy ``m0 c^2 / e`` [eV]. u_r = 3.0e-10 (from ``m0``)."""

DERIVED_AB_COUPLING = EXACT_E_CHARGE / DERIVED_PLANCK_BAR * 1e-20
"""Coupling ``e / hbar`` [rad / (Tesla Angstrom^2)]. Exact in principle.

The Aharonov-Bohm phase of a path is ``-DERIVED_AB_COUPLING`` times the line
integral of the vector potential [Tesla Angstrom]. The factor ``1e-20``
converts 1/m^2 to 1/Angstrom^2.
"""

_M_PER_A = 1e-10  # metres per Angstrom
_REST_J = CODATA18_ELECTRON_MASS * EXACT_LIGHT_SPEED**2  # m0 c^2 [J]
_HC = EXACT_PLANCK * EXACT_LIGHT_SPEED  # h c [J m]


def _energy_joule(kinetic_ev: object, allow_zero: bool) -> float:
    """Check a kinetic energy and convert it to joule.

    Parameters
    ----------
    kinetic_ev : real scalar or 0-d tensor
        Kinetic energy ``E`` [eV].
    allow_zero : bool
        If True, accept ``E = 0``.

    Returns
    -------
    float
        ``E_J = e E`` [J].

    Raises
    ------
    TypeError
        If the energy is a bool, a complex number, a string or not real.
    ValueError
        If the energy is NaN, infinite, negative, or zero when not allowed.
    """
    value = kinetic_ev
    if getattr(value, "ndim", None) == 0 and hasattr(value, "item"):
        value = value.item()  # pyright: ignore[reportAttributeAccessIssue]
    if isinstance(value, bool) or type(value).__name__ == "bool_":
        raise TypeError(f"kinetic_ev must be a real number, got bool {value!r}")
    if not isinstance(value, numbers.Real):
        raise TypeError(f"kinetic_ev must be a real number, got {type(value).__name__}")
    energy = float(value)
    if not math.isfinite(energy):
        raise ValueError(f"kinetic_ev must be finite, got {energy}")
    if energy < 0.0:
        raise ValueError(f"kinetic_ev must not be negative, got {energy}")
    if energy == 0.0 and not allow_zero:
        raise ValueError("kinetic_ev must be larger than zero, got 0")
    return energy * EXACT_E_CHARGE


_ENERGY_NOTE = """
    Notes
    -----
    An electron that a potential difference of ``U`` volts accelerates has
    ``E = U`` eV. Thus the caller can give the high tension of the
    microscope in volts. The function accepts a Python or numpy real scalar
    and a 0-d tensor. For a 0-d tensor it calls ``.item()``, which waits for
    the device if the tensor is on a GPU.
"""


def debroglie_at(kinetic_ev: float) -> float:
    """Return the relativistic de Broglie wavelength of an electron.

    ``lambda = h c / sqrt(E_J (E_J + 2 m0 c^2))`` (Kirkland 2020, ch. 2).

    Parameters
    ----------
    kinetic_ev : float
        Kinetic energy ``E`` [eV], larger than zero.

    Returns
    -------
    float
        Wavelength ``lambda`` [Angstrom].

    Raises
    ------
    TypeError
        If ``kinetic_ev`` is not a real number.
    ValueError
        If ``kinetic_ev`` is zero, negative, NaN or infinite.
    """
    e_j = _energy_joule(kinetic_ev, allow_zero=False)
    return _HC / math.sqrt(e_j * (e_j + 2.0 * _REST_J)) / _M_PER_A


def gamma_at(kinetic_ev: float) -> float:
    """Return the Lorentz factor ``gamma = 1 + E_J / (m0 c^2)`` [1].

    The function accepts ``E = 0`` and then returns 1.0, because an electron
    at rest has a defined Lorentz factor.

    Parameters
    ----------
    kinetic_ev : float
        Kinetic energy ``E`` [eV], zero or larger.

    Returns
    -------
    float
        Lorentz factor ``gamma`` [1].

    Raises
    ------
    TypeError
        If ``kinetic_ev`` is not a real number.
    ValueError
        If ``kinetic_ev`` is negative, NaN or infinite.
    """
    return 1.0 + _energy_joule(kinetic_ev, allow_zero=True) / _REST_J


def moving_mass_at(kinetic_ev: float) -> float:
    """Return the mass ``m = gamma m0`` [kg] of a moving electron.

    The function accepts ``E = 0`` and then returns ``m0``.

    Parameters
    ----------
    kinetic_ev : float
        Kinetic energy ``E`` [eV], zero or larger.

    Returns
    -------
    float
        Mass ``m`` [kg].

    Raises
    ------
    TypeError
        If ``kinetic_ev`` is not a real number.
    ValueError
        If ``kinetic_ev`` is negative, NaN or infinite.
    """
    return gamma_at(kinetic_ev) * CODATA18_ELECTRON_MASS


def phase_per_potential_at(kinetic_ev: float) -> float:
    """Return the interaction parameter ``sigma = 2 pi m e lambda / h^2``.

    A projected potential ``v`` [Volt Angstrom] gives the transmitted electron
    the extra phase ``sigma v`` [rad] (Kirkland 2020, ch. 5).

    Parameters
    ----------
    kinetic_ev : float
        Kinetic energy ``E`` [eV], larger than zero.

    Returns
    -------
    float
        Interaction parameter ``sigma`` [rad / (Volt Angstrom)].

    Raises
    ------
    TypeError
        If ``kinetic_ev`` is not a real number.
    ValueError
        If ``kinetic_ev`` is zero, negative, NaN or infinite.
    """
    e_j = _energy_joule(kinetic_ev, allow_zero=False)
    mass = (1.0 + e_j / _REST_J) * CODATA18_ELECTRON_MASS
    lambda_m = _HC / math.sqrt(e_j * (e_j + 2.0 * _REST_J))
    sigma_si = 2.0 * math.pi * mass * EXACT_E_CHARGE * lambda_m / EXACT_PLANCK**2
    return sigma_si * _M_PER_A


def bend_per_tesla_at(kinetic_ev: float) -> float:
    """Return ``e / p`` [1 / (Tesla Angstrom)] of an electron.

    ``p c = sqrt(E_J^2 + 2 E_J m0 c^2)`` is the relativistic momentum. The
    value ``e / p = 1 / (B r)`` is the curvature ``1 / r`` of the path per
    Tesla of transverse magnetic field.

    Parameters
    ----------
    kinetic_ev : float
        Kinetic energy ``E`` [eV], larger than zero.

    Returns
    -------
    float
        ``e / p`` [1 / (Tesla Angstrom)].

    Raises
    ------
    TypeError
        If ``kinetic_ev`` is not a real number.
    ValueError
        If ``kinetic_ev`` is zero, negative, NaN or infinite.
    """
    e_j = _energy_joule(kinetic_ev, allow_zero=False)
    momentum = math.sqrt(e_j * (e_j + 2.0 * _REST_J)) / EXACT_LIGHT_SPEED
    return EXACT_E_CHARGE / momentum * _M_PER_A


for _func in (
    debroglie_at,
    gamma_at,
    moving_mass_at,
    phase_per_potential_at,
    bend_per_tesla_at,
):
    _func.__doc__ = (_func.__doc__ or "") + _ENERGY_NOTE
del _func
