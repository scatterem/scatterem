"""Tests of the constants and the electron kinematics (brief TESTS.md, part A).

The oracles are the CODATA 2018 table and the SI brochure (values typed here),
an mpmath evaluation at 50 digits, Kirkland (2020) table values, limits and
second formulas.
"""

from __future__ import annotations

import math

import mpmath
import numpy as np
import pytest
import torch

from scatterem.physics import constants as C

KINEMATIC = (
    C.debroglie_at,
    C.gamma_at,
    C.moving_mass_at,
    C.phase_per_potential_at,
    C.bend_per_tesla_at,
)
ZERO_REFUSED = (C.debroglie_at, C.phase_per_potential_at, C.bend_per_tesla_at)


def rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b)


# --- A.1 published values ---------------------------------------------------


@pytest.mark.parametrize(
    ("value", "literature"),
    [
        (C.EXACT_PLANCK, 6.62607015e-34),
        (C.EXACT_E_CHARGE, 1.602176634e-19),
        (C.EXACT_LIGHT_SPEED, 299792458),
        (C.EXACT_AVOGADRO, 6.02214076e23),
        (C.EXACT_BOLTZMANN, 1.380649e-23),
        (C.CODATA18_ELECTRON_MASS, 9.1093837015e-31),
        (C.CODATA18_DALTON, 1.66053906660e-27),
        (C.CODATA18_FINE_STRUCTURE, 7.2973525693e-3),
        (C.CODATA18_VACUUM_MU, 1.25663706212e-6),
        (C.CODATA18_ELECTRON_RADIUS, 2.8179403262e-15),
    ],
)
def test_constant_equals_published_value(value: float, literature: float) -> None:
    assert value == literature


def test_published_value_check_detects_a_wrong_digit() -> None:
    assert C.CODATA18_ELECTRON_MASS != 9.1093837016e-31


# --- A.2 relations ----------------------------------------------------------


def test_relations_between_constants() -> None:
    h, e, c = C.EXACT_PLANCK, C.EXACT_E_CHARGE, C.EXACT_LIGHT_SPEED
    assert rel(C.DERIVED_PLANCK_BAR, h / (2 * math.pi)) < 1e-15
    alpha = e**2 * C.CODATA18_VACUUM_MU * c / (2 * h)
    assert rel(C.CODATA18_FINE_STRUCTURE, alpha) < 2e-9
    r_e = alpha * C.DERIVED_PLANCK_BAR / (C.CODATA18_ELECTRON_MASS * c)
    assert rel(C.CODATA18_ELECTRON_RADIUS, r_e) < 2e-9
    assert rel(C.DERIVED_ELECTRON_REST_EV, C.CODATA18_ELECTRON_MASS * c**2 / e) < 1e-15
    assert rel(C.DERIVED_ELECTRON_REST_EV, 510998.95000) < 1e-10


def test_relation_check_detects_a_wrong_magnetic_constant() -> None:
    h, e, c = C.EXACT_PLANCK, C.EXACT_E_CHARGE, C.EXACT_LIGHT_SPEED
    mu0_wrong = 4 * math.pi * 1e-7 * (1 + 1e-6)
    assert rel(C.CODATA18_FINE_STRUCTURE, e**2 * mu0_wrong * c / (2 * h)) > 2e-9


# --- A.3 high-precision oracle ----------------------------------------------


def _mp_oracle(energy_ev: float, relativistic: bool = True) -> dict[str, mpmath.mpf]:
    """Evaluate the contract formulas at 50 digits with the same constants."""
    with mpmath.workdps(50):
        h = mpmath.mpf("6.62607015e-34")
        e = mpmath.mpf("1.602176634e-19")
        c = mpmath.mpf("299792458")
        m0 = mpmath.mpf("9.1093837015e-31")
        e_j = e * mpmath.mpf(energy_ev)
        rest = m0 * c**2
        if relativistic:
            pc = mpmath.sqrt(e_j * (e_j + 2 * rest))
        else:
            pc = mpmath.sqrt(2 * rest * e_j)
        lam = h * c / pc
        gamma = 1 + e_j / rest
        mass = gamma * m0
        return {
            "lambda": lam * mpmath.mpf(10) ** 10,
            "gamma": gamma,
            "mass": mass,
            "sigma": 2 * mpmath.pi * mass * e * lam / h**2 * mpmath.mpf("1e-10"),
            "curvature_per_tesla": e / (pc / c) * mpmath.mpf("1e-10"),
        }


ENERGIES = [float(x) for x in np.logspace(0, 7, 50)]


@pytest.mark.parametrize("energy", ENERGIES)
def test_kinematics_match_50_digit_oracle(energy: float) -> None:
    ref = _mp_oracle(energy)
    got = {
        "lambda": C.debroglie_at(energy),
        "gamma": C.gamma_at(energy),
        "mass": C.moving_mass_at(energy),
        "sigma": C.phase_per_potential_at(energy),
        "curvature_per_tesla": C.bend_per_tesla_at(energy),
    }
    for key, value in got.items():
        assert isinstance(value, float)
        assert rel(value, float(ref[key])) <= 1e-13, key


def test_energy_zero_for_gamma_and_mass() -> None:
    assert C.gamma_at(0) == 1.0
    assert C.moving_mass_at(0.0) == C.CODATA18_ELECTRON_MASS


def test_oracle_check_detects_missing_relativistic_term() -> None:
    wrong = float(_mp_oracle(200e3, relativistic=False)["lambda"])
    assert rel(C.debroglie_at(200e3), wrong) > 0.05


# --- A.4 second path for sigma ----------------------------------------------


def _sigma_second_path(energy: float, relativistic: bool = True) -> float:
    lam = C.debroglie_at(energy)
    rest = C.CODATA18_ELECTRON_MASS * C.EXACT_LIGHT_SPEED**2 / C.EXACT_E_CHARGE
    frac = (rest + energy) / (2 * rest + energy) if relativistic else 0.5
    return 2 * math.pi / (lam * energy) * frac


@pytest.mark.parametrize("energy", [10.0, 3e3, 80e3, 200e3, 1e6, 1e7])
def test_sigma_matches_second_path(energy: float) -> None:
    assert rel(C.phase_per_potential_at(energy), _sigma_second_path(energy)) < 1e-12


def test_sigma_check_detects_missing_relativistic_fraction() -> None:
    wrong = _sigma_second_path(200e3, relativistic=False)
    assert rel(C.phase_per_potential_at(200e3), wrong) > 1e-12


# --- A.5 literature values (Kirkland 2020) ----------------------------------


@pytest.mark.parametrize(
    ("volts", "lam", "sigma"),
    [
        (100e3, 0.03701, 9.244e-4),
        (200e3, 0.02508, 7.288e-4),
        (300e3, 0.01969, 6.526e-4),
    ],
)
def test_kirkland_table_values(volts: float, lam: float, sigma: float) -> None:
    assert round(C.debroglie_at(volts), 5) == lam
    assert round(C.phase_per_potential_at(volts) * 1e4, 3) == round(sigma * 1e4, 3)


def test_kirkland_check_detects_wrong_energy() -> None:
    assert round(C.debroglie_at(200e3), 5) != 0.03701


# --- A.6 limits -------------------------------------------------------------


@pytest.mark.parametrize("energy", [1e-3, 0.1, 1.0, 9.9])
def test_low_energy_limit(energy: float) -> None:
    e_j = C.EXACT_E_CHARGE * energy
    rest = C.CODATA18_ELECTRON_MASS * C.EXACT_LIGHT_SPEED**2
    classic = C.EXACT_PLANCK / math.sqrt(2 * C.CODATA18_ELECTRON_MASS * e_j) * 1e10
    assert rel(C.debroglie_at(energy), classic) < e_j / rest


@pytest.mark.parametrize("energy", [1.01e8, 1e9, 1e11])
def test_high_energy_limit(energy: float) -> None:
    e_j = C.EXACT_E_CHARGE * energy
    rest = C.CODATA18_ELECTRON_MASS * C.EXACT_LIGHT_SPEED**2
    photon_like = C.EXACT_PLANCK * C.EXACT_LIGHT_SPEED / e_j * 1e10
    assert rel(C.debroglie_at(energy), photon_like) < rest / e_j


def test_limit_check_detects_wrong_limit_formula() -> None:
    e_j = C.EXACT_E_CHARGE * 5.0
    rest = C.CODATA18_ELECTRON_MASS * C.EXACT_LIGHT_SPEED**2
    wrong = C.EXACT_PLANCK / math.sqrt(C.CODATA18_ELECTRON_MASS * e_j) * 1e10
    assert rel(C.debroglie_at(5.0), wrong) > e_j / rest


# --- A.7 momentum -----------------------------------------------------------


@pytest.mark.parametrize("energy", [50.0, 2e4, 300e3, 5e6])
def test_momentum_relation(energy: float) -> None:
    e_j = C.EXACT_E_CHARGE * energy
    rest = C.CODATA18_ELECTRON_MASS * C.EXACT_LIGHT_SPEED**2
    p = math.sqrt(e_j**2 + 2 * e_j * rest) / C.EXACT_LIGHT_SPEED
    lam_m = C.debroglie_at(energy) * 1e-10
    assert rel(C.EXACT_PLANCK / lam_m, p) < 1e-13
    expected = C.EXACT_E_CHARGE * lam_m / C.EXACT_PLANCK * 1e-10
    assert rel(C.bend_per_tesla_at(energy), expected) < 1e-13


# --- A.8 Aharonov-Bohm ------------------------------------------------------


def test_flux_quantum_gives_two_pi() -> None:
    flux_t_a2 = C.EXACT_PLANCK / C.EXACT_E_CHARGE * 1e20
    assert rel(C.DERIVED_AB_COUPLING * flux_t_a2, 2 * math.pi) < 1e-14


def test_flux_check_detects_e_over_h() -> None:
    flux_t_a2 = C.EXACT_PLANCK / C.EXACT_E_CHARGE * 1e20
    wrong = C.EXACT_E_CHARGE / C.EXACT_PLANCK * 1e-20 * flux_t_a2
    assert rel(wrong, 2 * math.pi) > 0.5


# --- A.9 refusals and accepted types ----------------------------------------


@pytest.mark.parametrize("func", KINEMATIC)
@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), -float("inf")])
def test_bad_energy_value_is_refused(func, bad: float) -> None:
    with pytest.raises(ValueError, match="kinetic_ev"):
        func(bad)


@pytest.mark.parametrize("func", ZERO_REFUSED)
def test_zero_energy_is_refused(func) -> None:
    with pytest.raises(ValueError, match="kinetic_ev"):
        func(0)


@pytest.mark.parametrize("func", KINEMATIC)
@pytest.mark.parametrize(
    "bad", [True, np.bool_(True), "200", 1j, None, torch.tensor(True)]
)
def test_bad_energy_type_is_refused(func, bad) -> None:
    with pytest.raises(TypeError, match="kinetic_ev"):
        func(bad)


@pytest.mark.parametrize(
    "good", [200000, 200e3, np.float32(200e3), np.int64(200000), torch.tensor(200e3)]
)
def test_accepted_energy_types(good) -> None:
    assert rel(C.debroglie_at(good), C.debroglie_at(200e3)) < 1e-15


def test_refusal_check_detects_a_missing_refusal() -> None:
    with pytest.raises(pytest.fail.Exception):
        with pytest.raises(ValueError):
            C.gamma_at(0)
