"""Relativistic electron-optical quantities, written from the physics.

Each quantity is written directly from its textbook definition, with CODATA
values taken from :mod:`scipy.constants` rather than transcribed as literals,
so the derivation is visible in the source:

* Kirkland, *Advanced Computing in Electron Microscopy*, 2nd ed., Eq. (2.5)
  for the relativistic de Broglie wavelength and Eq. (5.6) for the
  interaction parameter.

Numerical agreement with the module this replaces is asserted in
``tests/test_electron_optics.py`` by evaluating both over 1 keV--1 MeV; the
agreement is a *consequence* of both being correct physics, not of shared
code.
"""

from __future__ import annotations

import numpy as np
from scipy import constants as _const

__all__ = [
    "relativistic_mass_correction",
    "energy2wavelength",
    "energy2sigma",
    "ELECTRON_REST_ENERGY_EV",
]

#: Electron rest energy m_e c^2, in eV (~5.11e5).
ELECTRON_REST_ENERGY_EV: float = float(
    _const.electron_mass * _const.speed_of_light**2 / _const.elementary_charge
)

#: Planck constant times c, in eV*Angstrom (~1.2398e4).
_HC_EV_ANGSTROM: float = float(
    _const.Planck * _const.speed_of_light / _const.elementary_charge * 1e10
)


def relativistic_mass_correction(energy):
    """Lorentz factor gamma of an electron accelerated through ``energy`` volts.

    The kinetic energy of an electron accelerated through a potential ``V`` is
    ``T = eV``, and its total energy is ``T + m_e c^2``, so

        gamma = (T + m_e c^2) / (m_e c^2) = 1 + T / (m_e c^2).

    Parameters
    ----------
    energy : float or array_like
        Accelerating voltage in volts (equivalently, kinetic energy in eV).

    Returns
    -------
    float or ndarray
        Dimensionless gamma >= 1.
    """
    return 1.0 + np.asarray(energy, dtype=float) / ELECTRON_REST_ENERGY_EV


def energy2wavelength(energy):
    """Relativistic de Broglie wavelength in Angstrom.

    From ``lambda = h / p`` with the relativistic momentum obtained from the
    energy-momentum relation ``E^2 = (pc)^2 + (m_e c^2)^2`` and ``E = T + m_e c^2``:

        (pc)^2 = T^2 + 2 T m_e c^2,

    so, working throughout in eV and Angstrom,

        lambda = hc / sqrt(T^2 + 2 T m_e c^2).

    At 300 kV this gives 0.0196874 Angstrom.

    Parameters
    ----------
    energy : float or array_like
        Accelerating voltage in volts.

    Returns
    -------
    float or ndarray
        Wavelength in Angstrom.
    """
    T = np.asarray(energy, dtype=float)
    pc = np.sqrt(T * (T + 2.0 * ELECTRON_REST_ENERGY_EV))
    return _HC_EV_ANGSTROM / pc


def energy2sigma(energy):
    """Interaction parameter sigma, in rad / (V * Angstrom).

    sigma converts a projected electrostatic potential (V*Angstrom) into the
    phase shift it imprints on the electron wave, which is what the multislice
    transmission function needs. Kirkland Eq. (5.6):

        sigma = (2 pi / (lambda V)) * (m_e c^2 + eV) / (2 m_e c^2 + eV),

    with ``lambda`` in Angstrom and ``V`` in volts. The trailing ratio is the
    relativistic correction; it tends to 1/2 in the non-relativistic limit and
    to 1 as ``eV`` grows large compared with the rest energy.

    Parameters
    ----------
    energy : float or array_like
        Accelerating voltage in volts.

    Returns
    -------
    float or ndarray
        Interaction parameter in rad / (V * Angstrom).
    """
    T = np.asarray(energy, dtype=float)
    wavelength = energy2wavelength(T)
    m0c2 = ELECTRON_REST_ENERGY_EV
    return (2.0 * np.pi / (wavelength * T)) * (m0c2 + T) / (2.0 * m0c2 + T)
