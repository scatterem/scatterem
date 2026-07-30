"""Relativistic electron optics for the published reconstruction path.

Written from the de Broglie relation rather than adapted from any existing
implementation: ``scatterem.utils.energy`` is a verbatim copy of a
GPL-3.0-or-later upstream, which an Apache-2.0 release cannot distribute. Only
the wavelength is reproduced here, because that is the one quantity the
published FF-STEM path needs (it sets the reciprocal-space sampling ``dk``).

Constants are the 2019 SI values, in which the Planck constant and the
elementary charge are exact by definition.
"""

from __future__ import annotations

import math

#: Planck constant [J s] -- exact (SI 2019).
PLANCK_CONSTANT = 6.62607015e-34
#: Elementary charge [C] -- exact (SI 2019).
ELEMENTARY_CHARGE = 1.602176634e-19
#: Electron rest mass [kg] -- CODATA 2018.
ELECTRON_REST_MASS = 9.1093837015e-31
#: Speed of light in vacuum [m/s] -- exact.
SPEED_OF_LIGHT = 299792458.0

#: Electron rest energy [J], the scale the relativistic correction is measured against.
_REST_ENERGY = ELECTRON_REST_MASS * SPEED_OF_LIGHT**2

_METRE_TO_ANGSTROM = 1e10


def electron_wavelength(accelerating_voltage: float) -> float:
    """Return the relativistic electron wavelength in Angstrom.

    An electron accelerated through ``V`` volts gains kinetic energy ``eV``. Its
    momentum follows from the relativistic energy-momentum relation, and the de
    Broglie relation turns that into a wavelength::

        lambda = h / sqrt(2 m0 e V (1 + e V / (2 m0 c^2)))

    The bracketed factor is the relativistic correction; it is negligible below
    a few hundred volts and reaches ~1.2 at 200 kV, where ignoring it would
    overstate the wavelength by roughly 10%.

    Args:
        accelerating_voltage: the accelerating voltage in volts (e.g. ``200e3``
            for a 200 kV microscope). Must be positive.

    Returns:
        The wavelength in Angstrom.

    Raises:
        ValueError: if ``accelerating_voltage`` is not positive.
    """
    if not accelerating_voltage > 0:
        raise ValueError(
            "accelerating_voltage must be positive, in volts; "
            f"got {accelerating_voltage!r}"
        )

    kinetic_energy = ELEMENTARY_CHARGE * float(accelerating_voltage)
    relativistic_factor = 1.0 + kinetic_energy / (2.0 * _REST_ENERGY)
    momentum = math.sqrt(
        2.0 * ELECTRON_REST_MASS * kinetic_energy * relativistic_factor
    )
    return PLANCK_CONSTANT / momentum * _METRE_TO_ANGSTROM
