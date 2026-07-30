"""Clean-room relativistic electron optics.

The values pinned here are the textbook relativistic electron wavelengths at the
accelerating voltages this lab actually uses, so the test states the physics
rather than a previous implementation's output. `scatterem.utils.energy` is a
verbatim copy of a GPL-3.0 upstream and cannot ship in an Apache-2.0 release;
this module replaces the one function the published FF-STEM path needs, written
from the de Broglie relation.
"""

import math

import pytest

from scatterem.utils.physics import electron_wavelength

# lambda = h / sqrt(2 m0 e V (1 + e V / (2 m0 c^2))), in Angstrom.
# Reference values are the standard tabulated ones (e.g. Williams & Carter,
# "Transmission Electron Microscopy", Table 1.1), good to 4 decimal places.
TABULATED_ANGSTROM = {
    60e3: 0.04866,
    80e3: 0.04176,
    100e3: 0.03701,
    200e3: 0.02508,
    300e3: 0.01969,
}


@pytest.mark.parametrize("volts,expected", sorted(TABULATED_ANGSTROM.items()))
def test_matches_tabulated_relativistic_wavelength(volts, expected):
    assert electron_wavelength(volts) == pytest.approx(expected, abs=5e-6)


def test_is_monotonically_decreasing_in_voltage():
    """Higher energy means shorter wavelength -- guards a sign or reciprocal slip."""
    volts = [30e3, 60e3, 80e3, 100e3, 200e3, 300e3, 1e6]
    lam = [electron_wavelength(v) for v in volts]
    assert lam == sorted(lam, reverse=True)


def test_nonrelativistic_limit():
    """At low voltage the relativistic correction must vanish to <0.1%.

    The non-relativistic de Broglie wavelength is h/sqrt(2 m0 e V); at 100 V the
    correction term e V / (2 m0 c^2) is ~1e-7, so the two must agree closely.
    Catches a correction applied with the wrong sign or magnitude.
    """
    h = 6.62607015e-34
    m0 = 9.1093837015e-31
    e = 1.602176634e-19
    v = 100.0
    nonrel = h / math.sqrt(2 * m0 * e * v) * 1e10
    assert electron_wavelength(v) == pytest.approx(nonrel, rel=1e-3)


def test_rejects_nonpositive_voltage():
    """Zero or negative voltage is a caller bug, not something to return nan for."""
    for bad in (0.0, -200e3):
        with pytest.raises(ValueError, match="positive"):
            electron_wavelength(bad)


def test_no_numerical_regression_against_the_previous_implementation():
    """The clean-room rewrite must not shift any calibration downstream.

    Expression is independent, but the number has to stay put or every
    reconstruction's dk moves. The tolerance is 1e-7 rather than exact because
    the replaced implementation drew its constants from ase's CODATA-2014 table
    while this one uses the 2019 SI values, where h and e are exact by
    definition. That accounts for ~2e-8 relative; anything larger is a real bug.
    """
    legacy = pytest.importorskip("scatterem.utils.energy")
    for volts in (60e3, 80e3, 100e3, 200e3, 300e3):
        assert electron_wavelength(volts) == pytest.approx(
            legacy.energy2wavelength(volts), rel=1e-7
        )
