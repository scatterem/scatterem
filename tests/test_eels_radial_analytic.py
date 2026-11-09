"""Analytic checks of the EELS radial states (hydrogenic family and Numerov).

Every expected value is a closed form: hydrogenic norms, energies, ``<r>`` and
orthogonality; the per-Ry normalisation of the free continuum (Manson 1972);
and the fourth-order convergence of the Numerov method.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import spherical_jn, spherical_yn

from scatterem.simulation.eels.physics.radial import (
    _HARTREE_EV,
    _RYDBERG_EV,
    _numerov,
    free_continuum_wavefunction,
    hydrogenic_bound_wavefunction,
)

_STATES = [(8, 1, 0), (6, 2, 0), (6, 2, 1), (26, 3, 2)]


@pytest.mark.parametrize("Z,n,ell", _STATES)
def test_hydrogenic_norm_and_energy(Z, n, ell):
    wf = hydrogenic_bound_wavefunction(Z, n, ell)
    assert float(np.trapezoid(wf.u**2, wf.r)) == pytest.approx(1.0, abs=2e-5)
    assert wf.energy == pytest.approx(-(Z**2) / (2.0 * n**2) * _HARTREE_EV, rel=1e-12)


@pytest.mark.parametrize("Z,n,ell", [(79, 1, 0), (79, 2, 1), (79, 3, 2)])
def test_the_default_grid_resolves_heavy_inner_shells(Z, n, ell):
    """Au 1s on the old fixed 4000-point grid had a trapezoid norm of 0.9786."""
    wf = hydrogenic_bound_wavefunction(Z, n, ell)
    assert float(np.trapezoid(wf.u**2, wf.r)) == pytest.approx(1.0, abs=2e-5)
    assert (wf.r[1] - wf.r[0]) <= 0.1 * n / Z


def test_light_elements_keep_the_4000_point_grid():
    assert len(hydrogenic_bound_wavefunction(8, 1, 0).r) == 4000
    assert len(hydrogenic_bound_wavefunction(14, 2, 1).r) == 4000


def test_hydrogenic_1s_closed_form():
    Z = 8
    wf = hydrogenic_bound_wavefunction(Z, 1, 0)
    np.testing.assert_allclose(
        wf.u, 2.0 * Z**1.5 * wf.r * np.exp(-Z * wf.r), rtol=1e-10, atol=1e-14
    )


@pytest.mark.parametrize(
    "Z,ell,n1,n2", [(6, 0, 1, 2), (6, 0, 2, 3), (6, 1, 2, 3), (26, 2, 3, 4)]
)
def test_hydrogenic_states_of_one_l_are_orthogonal(Z, ell, n1, n2):
    a = hydrogenic_bound_wavefunction(Z, n1, ell)
    b = hydrogenic_bound_wavefunction(Z, n2, ell)
    assert abs(float(np.trapezoid(a.u * b.u, a.r))) < 1e-6


@pytest.mark.parametrize("Z,n,ell", _STATES)
def test_hydrogenic_mean_radius(Z, n, ell):
    wf = hydrogenic_bound_wavefunction(Z, n, ell, n_points=40000)
    mean_r = float(np.trapezoid(wf.u**2 * wf.r, wf.r))
    assert mean_r == pytest.approx((3 * n**2 - ell * (ell + 1)) / (2.0 * Z), rel=1e-6)


@pytest.mark.parametrize("ell", [0, 1, 3])
@pytest.mark.parametrize("eps", [1.0, 10.0, 150.0])
def test_free_continuum_is_the_per_rydberg_bessel_state(ell, eps):
    wf = free_continuum_wavefunction(ell, eps)
    k = math.sqrt(eps / _RYDBERG_EV)
    expected = k * wf.r * spherical_jn(ell, k * wf.r) / math.sqrt(math.pi * k)
    np.testing.assert_allclose(wf.u, expected, rtol=1e-12, atol=1e-14)
    # Asymptotic amplitude from a least-squares fit on the last 10 Bohr.
    tail = wf.r > wf.rmax - 10.0
    z = k * wf.r[tail]
    basis = np.stack([z * spherical_jn(ell, z), z * spherical_yn(ell, z)], axis=1)
    (a, b), *_ = np.linalg.lstsq(basis, wf.u[tail], rcond=None)
    assert math.hypot(a, b) == pytest.approx(1.0 / math.sqrt(math.pi * k), rel=1e-6)


@pytest.mark.parametrize("ell", [0, 2])
def test_free_continuum_box_norm_grows_as_r_over_2_pi_k(ell):
    """Energy normalisation: ``int_0^R u_E^2 dr -> R / (2 pi k)`` (Ry units)."""
    eps, rmax = 10.0, 2000.0
    wf = free_continuum_wavefunction(ell, eps, n_points=400001, rmax=rmax)
    k = math.sqrt(eps / _RYDBERG_EV)
    box = float(np.trapezoid(wf.u**2, wf.r))
    assert box == pytest.approx(rmax / (2.0 * math.pi * k), rel=1e-3)


def _harmonic_error(npts):
    k = 1.3
    x = np.linspace(0.0, 10.0, npts)
    dx = float(x[1] - x[0])
    y = _numerov(np.full(npts, -k * k), 0.0, math.sin(k * dx), dx)
    return float(np.abs(y - np.sin(k * x)).max())


def test_numerov_is_fourth_order():
    coarse, fine = _harmonic_error(201), _harmonic_error(401)
    assert 14.0 < coarse / fine < 18.0
