"""The interaction constant, checked against physics rather than against pyms.

This file previously asserted agreement with ``pyms.structure_routines`` to
1e-9. That was removed for two reasons. py_multislice carries no licence of any
kind, so pinning our arithmetic to its output documents a derivation we cannot
publish; and the assertion had in any case stopped running -- ``pyms`` is not
installed here, so the module failed at collection rather than passing.

What follows checks the quantity itself: sigma is fixed by the relativistic
electron optics, so it can be verified from its own definition without
reference to any other implementation.
"""

import numpy as np
import pytest

from scatterem.simulation._electron_optics import (
    ELECTRON_REST_ENERGY_EV,
    energy2sigma,
    energy2wavelength,
)
from scatterem.simulation.scattering_factors import (
    _relativistic_mass_correction,
    _wavev,
    interaction_constant,
)

ENERGIES = [80e3, 200e3, 300e3]


@pytest.mark.parametrize("eV", ENERGIES)
def test_matches_kirkland_expression(eV):
    """sigma = (2 pi / (lambda V)) * (m0c2 + eV) / (2 m0c2 + eV), Kirkland Eq. (5.6)."""
    lam = float(energy2wavelength(eV))
    m0c2 = ELECTRON_REST_ENERGY_EV
    expected = (2 * np.pi / (lam * eV)) * (m0c2 + eV) / (2 * m0c2 + eV)
    assert interaction_constant(eV) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("eV", ENERGIES)
def test_agrees_with_the_electron_optics_helper(eV):
    """Both names return the same quantity; nothing should be able to drift."""
    assert interaction_constant(eV) == pytest.approx(float(energy2sigma(eV)), rel=1e-12)


@pytest.mark.parametrize("eV", ENERGIES)
def test_rad_per_angstrom_branch_is_gamma_over_k0(eV):
    gamma = _relativistic_mass_correction(eV)
    assert interaction_constant(eV, "rad/A") == pytest.approx(gamma / _wavev(eV), rel=1e-12)


@pytest.mark.parametrize("eV", ENERGIES)
def test_wavenumber_is_the_reciprocal_wavelength(eV):
    assert _wavev(eV) == pytest.approx(1.0 / float(energy2wavelength(eV)), rel=1e-12)


def test_sigma_falls_with_energy():
    """Faster electrons interact more weakly with the same potential."""
    sigmas = [interaction_constant(eV) for eV in ENERGIES]
    assert sigmas == sorted(sigmas, reverse=True)


def test_300kv_sigma_is_physically_sane():
    """Order of magnitude check: sigma(300 kV) ~ 6.5e-4 rad/(V*Angstrom)."""
    assert interaction_constant(300e3) == pytest.approx(6.5e-4, rel=0.05)


def test_unknown_units_rejected():
    with pytest.raises(ValueError, match="rad/VA"):
        interaction_constant(300e3, "furlongs")
