"""Analytic checks of the EELS form factor and the transition potentials.

The expected values are closed forms or independent computations: scipy
special functions, the Gaunt integral by quadrature, the dipole integrals by
Laplace transform, rotation and mirror symmetry, and the Born-Bethe prefactor.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy.special import lpmv, roots_legendre, spherical_jn

from scatterem.physics.constants import (
    debroglie_at,
    gamma_at,
)
from scatterem.simulation.eels.physics import transition_potentials as tpmod
from scatterem.simulation.eels.physics._angular import spherical_harmonic
from scatterem.simulation.eels.physics._special_torch import (
    interp_torch as _interp_torch,
)
from scatterem.simulation.eels.physics._special_torch import (
    spherical_harmonic_torch as _spherical_harmonic_torch,
)
from scatterem.simulation.eels.physics.radial import (
    _BOHR_ANG,
    _RYDBERG_EV,
    free_continuum_wavefunction,
    hydrogenic_bound_wavefunction,
)
from scatterem.simulation.eels.physics.transition_potentials import (
    Transition,
    _assoc_legendre_torch,
    _form_factor,
    _radial_overlap,
    _spherical_bessel_torch,
    build_transition_potentials,
)


# --------------------------------------------------------------------------- #
# Torch special functions against scipy and numpy
# --------------------------------------------------------------------------- #
def test_spherical_bessel_torch_to_order_ten_on_zero_to_three_hundred():
    x = torch.cat(
        [
            torch.tensor([0.0, 1e-12, 1e-6], dtype=torch.float64),
            torch.linspace(1e-3, 300.0, 30001, dtype=torch.float64),
        ]
    )
    got = _spherical_bessel_torch(10, x)
    for ell in range(11):
        ref = spherical_jn(ell, x.numpy())
        np.testing.assert_allclose(got[ell].numpy(), ref, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("ell", range(9))
def test_assoc_legendre_torch_matches_lpmv(ell):
    theta = torch.linspace(0.0, math.pi, 37, dtype=torch.float64)
    ct, st = torch.cos(theta), torch.sin(theta)
    for am in range(ell + 1):
        ref = lpmv(am, ell, ct.numpy())
        scale = max(1.0, float(np.abs(ref).max()))
        np.testing.assert_allclose(
            _assoc_legendre_torch(ell, am, ct, st).numpy(), ref, atol=1e-13 * scale
        )


@pytest.mark.parametrize("ell", range(6))
def test_spherical_harmonic_torch_matches_sph_harm_y(ell):
    g = torch.Generator().manual_seed(ell)
    phi = torch.rand(50, generator=g, dtype=torch.float64) * 2 * math.pi
    theta = torch.rand(50, generator=g, dtype=torch.float64) * math.pi
    for m in range(-ell, ell + 1):
        ref = spherical_harmonic(m, ell, phi.numpy(), theta.numpy())
        got = _spherical_harmonic_torch(m, ell, phi, theta).numpy()
        np.testing.assert_allclose(got, ref, rtol=0, atol=1e-13)


def test_interp_torch_equals_numpy_interp():
    xp = torch.linspace(0.0, 3.0, 31, dtype=torch.float64)
    fp = torch.sin(7.0 * xp)
    xq = torch.linspace(-1.0, 4.0, 997, dtype=torch.float64)
    np.testing.assert_allclose(
        _interp_torch(xq, xp, fp).numpy(),
        np.interp(xq.numpy(), xp.numpy(), fp.numpy()),
        rtol=0,
        atol=1e-14,
    )


# --------------------------------------------------------------------------- #
# Angular algebra
# --------------------------------------------------------------------------- #
def test_gaunt_integral_matches_quadrature_up_to_l_four():
    """``int Y*_(l'm') Y_(lambda mu) Y_(lm) dOmega`` on an exact product grid."""
    nodes, weights = roots_legendre(16)
    n_phi = 16
    phi = 2.0 * math.pi * np.arange(n_phi) / n_phi
    theta = np.arccos(nodes)
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    w = np.outer(weights, np.full(n_phi, 2.0 * math.pi / n_phi))
    y = {
        (ell, m): spherical_harmonic(m, ell, ph, th)
        for ell in range(5)
        for m in range(-ell, ell + 1)
    }
    for (lp, mlp), y_lp in y.items():
        for (lam, mu), y_lam in y.items():
            for (ell, ml), y_l in y.items():
                quad = complex(np.sum(w * np.conj(y_lp) * y_lam * y_l))
                exact = tpmod._gaunt(lp, mlp, lam, mu, ell, ml)
                assert abs(quad - exact) < 1e-12, (lp, mlp, lam, mu, ell, ml)


def _channels(Z, n, ell, lp, eps):
    bound = hydrogenic_bound_wavefunction(Z, n, ell)
    excited = free_continuum_wavefunction(lp, eps)
    return [
        Transition(bound, excited, ml, mlp)
        for ml in range(-ell, ell + 1)
        for mlp in range(-lp, lp + 1)
    ]


@pytest.mark.parametrize(
    "Z,n,ell,lp,eps", [(8, 1, 0, 1, 10.0), (13, 2, 1, 2, 20.0), (13, 2, 1, 1, 5.0)]
)
def test_sum_over_m_is_rotationally_invariant(Z, n, ell, lp, eps):
    g = np.random.default_rng(3)
    qabs = np.full(5, 2.3)
    phi = g.uniform(0.0, 2.0 * math.pi, 5)
    theta = g.uniform(0.0, math.pi, 5)
    total = sum(
        np.abs(_form_factor(tr, qabs, phi, theta)) ** 2
        for tr in _channels(Z, n, ell, lp, eps)
    )
    np.testing.assert_allclose(total, np.full(5, total[0]), rtol=1e-12)


@pytest.mark.parametrize("Z,n,ell,lp,eps", [(8, 1, 0, 1, 10.0), (13, 2, 1, 2, 20.0)])
def test_mirror_conjugation_identity(Z, n, ell, lp, eps):
    """``eps_(-m',-m)(phi) = (-1)^(m+m') eps_(m',m)(-phi)``.

    Flipping every m leaves the Gaunt factor unchanged (``l + l' + lambda`` is
    even), and ``Y*_(lambda,-mu)(theta, phi) = (-1)^mu Y*_(lambda mu)(theta,
    -phi)``. ``i^lambda`` stays, so there is no complex conjugate.
    """
    qabs = np.array([0.4, 1.7, 3.1])
    phi = np.array([0.3, 2.0, 4.4])
    theta = np.array([0.2, 1.3, 2.6])
    bound = hydrogenic_bound_wavefunction(Z, n, ell)
    excited = free_continuum_wavefunction(lp, eps)
    for ml in range(-ell, ell + 1):
        for mlp in range(-lp, lp + 1):
            a = _form_factor(Transition(bound, excited, -ml, -mlp), qabs, phi, theta)
            b = _form_factor(Transition(bound, excited, ml, mlp), qabs, -phi, theta)
            np.testing.assert_allclose(
                a, (-1.0) ** (ml + mlp) * b, rtol=1e-12, atol=1e-15
            )


# --------------------------------------------------------------------------- #
# Radial integral limits
# --------------------------------------------------------------------------- #
def test_dipole_limit_one_s_to_p():
    Z, eps = 8, 10.0
    k = math.sqrt(eps / _RYDBERG_EV)
    amp = 1.0 / math.sqrt(math.pi * k)
    d = 16.0 * Z**2.5 * amp * k**2 / (Z**2 + k**2) ** 3
    kk = 1e-3  # 1/Bohr
    tr = _channels(Z, 1, 0, 1, eps)[0]
    i1 = _radial_overlap(1, tr.bound, tr.excited, np.array([kk / _BOHR_ANG]))[0]
    assert i1 / kk == pytest.approx(d / 3.0, rel=1e-6)


def test_dipole_limit_two_p_to_d():
    Z, eps = 13, 20.0
    k = math.sqrt(eps / _RYDBERG_EV)
    amp = 1.0 / math.sqrt(math.pi * k)
    p = Z / 2.0
    d = Z**2.5 / (2.0 * math.sqrt(6.0)) * amp * k * 48.0 * p * k**2 / (p**2 + k**2) ** 4
    kk = 1e-3
    tr = _channels(Z, 2, 1, 2, eps)[0]
    i1 = _radial_overlap(1, tr.bound, tr.excited, np.array([kk / _BOHR_ANG]))[0]
    assert i1 / kk == pytest.approx(d / 3.0, rel=1e-6)


def test_monopole_term_of_non_orthogonal_states_vanishes_as_k_squared():
    """1s (Coulomb) and a free s wave overlap, but ``j_0 - 1`` removes it."""
    tr = _channels(8, 1, 0, 0, 10.0)[0]
    kk = np.array([1e-3, 2e-3]) / _BOHR_ANG
    i0 = _radial_overlap(0, tr.bound, tr.excited, kk)
    slope = math.log(abs(i0[1]) / abs(i0[0])) / math.log(2.0)
    assert slope == pytest.approx(2.0, abs=1e-3)


@pytest.mark.parametrize("Z,n,ell,eps,lam", [(8, 1, 0, 20.0, 0), (13, 2, 1, 5.0, 2)])
def test_same_l_continuum_is_orthogonalised_to_the_bound_orbital(Z, n, ell, eps, lam):
    """For ``l' = l`` the integral uses ``c - <b|c> b`` (Gram-Schmidt).

    Reference: dense trapezoid sums, written here. Without the projection the
    monopole integral tends to ``-<c|b>`` at large K instead of 0.
    """
    bound = hydrogenic_bound_wavefunction(Z, n, ell)
    excited = free_continuum_wavefunction(ell, eps)
    # Exact samples on a dense grid (no interpolation in the reference).
    b = hydrogenic_bound_wavefunction(Z, n, ell, n_points=240001, rmax=12.0).u
    dense = free_continuum_wavefunction(ell, eps, n_points=240001, rmax=12.0)
    r, c = dense.r, dense.u
    overlap = float(np.trapezoid(b * c, r))
    assert abs(overlap) > 1e-2  # the model states are far from orthogonal
    c_perp = c - overlap / float(np.trapezoid(b * b, r)) * b
    kb = np.array([0.5, 2.0, 8.0, 30.0])  # 1/Bohr
    ref = np.array([np.trapezoid(c_perp * spherical_jn(lam, k * r) * b, r) for k in kb])
    got = _radial_overlap(lam, bound, excited, kb / _BOHR_ANG)
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-7 * np.abs(ref).max())
    if lam == 0:
        # Unprojected, ``j_0 - 1`` leaves about ``-<c|b>`` at large K.
        bare = float(np.trapezoid(c * (spherical_jn(0, kb[-1] * r) - 1.0) * b, r))
        assert bare == pytest.approx(-overlap, rel=0.1)
        assert abs(got[-1]) < 0.1 * abs(overlap)


# --------------------------------------------------------------------------- #
# Transition potentials: shape and absolute scale
# --------------------------------------------------------------------------- #
def _recip(transitions, Z, gpts, sampling, eV):
    tp = build_transition_potentials(
        transitions,
        Z,
        gpts,
        sampling,
        eV,
        bandlimit=1.0,
        dtype=torch.complex128,
        method="cpu",
    )
    return np.fft.fft2(tp.array.numpy(), axes=(-2, -1)) * sampling[0] * sampling[1], tp


def _q_grid(gpts, sampling):
    qa = 2.0 * math.pi * np.fft.fftfreq(gpts[0], sampling[0])[:, None]
    qb = 2.0 * math.pi * np.fft.fftfreq(gpts[1], sampling[1])[None, :]
    return np.broadcast_arrays(qa, qb)


def test_s_to_p_channel_ratio_is_sqrt_two_qz_over_qperp():
    eV, gpts, sampling = 200e3, (24, 20), (0.2, 0.25)
    tr = _channels(8, 1, 0, 1, 10.0)
    h, _ = _recip(tr[:2], 8, gpts, sampling, eV)  # m' = -1 and m' = 0
    qa, qb = _q_grid(gpts, sampling)
    qperp = np.hypot(qa, qb)
    qz = 2 * math.pi / debroglie_at(eV) - 2 * math.pi / debroglie_at(
        eV - tr[0].energy_loss
    )
    # Points outside the circular band edge (limit 1.0) are zero.
    keep = (qperp > 0) & (np.abs(h[0]) > 1e-8 * np.abs(h[0]).max())
    ratio = np.abs(h[1][keep]) / np.abs(h[0][keep])
    np.testing.assert_allclose(ratio, math.sqrt(2.0) * qz / qperp[keep], rtol=1e-10)


@pytest.mark.parametrize("method", ["cpu", "auto"])
@pytest.mark.parametrize(
    "Z,n,ell,lp,eps", [(8, 1, 0, 0, 20.0), (8, 1, 0, 1, 20.0), (22, 2, 1, 1, 20.0)]
)
def test_grid_build_equals_the_direct_form_factor_at_every_pixel(
    Z, n, ell, lp, eps, method
):
    """The K table and its interpolation add no error above 1e-6 (cubic Hermite table).

    ``_form_factor`` integrates ``I_lambda`` directly at every ``|K|``. The
    ``l' = l`` channels start as ``K^2`` (orthogonalised monopole); a linear
    table gave 4.5e-4 there at the first pixels.
    """
    eV, gpts, sampling = 200e3, (32, 32), (0.15, 0.15)
    tr = _channels(Z, n, ell, lp, eps)
    tp = build_transition_potentials(
        tr,
        Z,
        gpts,
        sampling,
        eV,
        bandlimit=1.0,
        dtype=torch.complex128,
        method=method,
    )
    h = np.fft.fft2(tp.array.numpy(), axes=(-2, -1))
    qa, qb = _q_grid(gpts, sampling)
    qz = tpmod._q_z(eV, tr[0].energy_loss)
    qperp = np.hypot(qa, qb)
    kabs = np.sqrt(qperp**2 + qz**2)
    phi = np.where(qperp > 0, np.arctan2(-qb, np.where(qperp > 0, -qa, 1.0)), 0.0)
    theta = np.arctan2(qperp, qz)
    scale = tpmod._scale(eV, tr[0].energy_loss) / (sampling[0] * sampling[1])
    keep = qperp <= 0.9 * np.abs(qa).max()  # inside the circular band edge
    for i, t in enumerate(tr):
        ref = scale * _form_factor(t, kabs, phi, theta) / kabs**2
        atol = 1e-9 * np.abs(ref[keep]).max()
        np.testing.assert_allclose(h[i][keep], ref[keep], rtol=1e-6, atol=atol)


def test_sum_over_channels_is_isotropic_on_the_grid():
    eV, gpts, sampling = 200e3, (16, 16), (0.25, 0.25)
    h, _ = _recip(_channels(13, 2, 1, 2, 20.0), 13, gpts, sampling, eV)
    total = (np.abs(h) ** 2).sum(axis=0)
    for a, b in [(1, 2), (3, 5), (0, 4)]:
        group = [total[a, b], total[b, a], total[-a, b], total[a, -b], total[-b, -a]]
        np.testing.assert_allclose(group, np.full(5, group[0]), rtol=1e-12)


def test_prefactor_is_the_first_born_bethe_constant():
    """``|H~|^2 d^2Q/(2 pi)^2`` is the first-Born double-differential cross-section.

    Bethe, per electron: ``d^2 sigma/(dOmega dE) = (k_n/k_0) 4 gamma^2 /
    (a0^2 K^4) |eps_eV|^2``. The scattered electron has wavenumber ``k_n``,
    so ``dOmega = d^2Q / k_n^2``. With 2 electrons per closed orbital,
    ``|H~|^2 = 2 (2 pi)^2 4 gamma^2 |eps_eV|^2 / (a0^2 K^4 k_0 k_n)``. The
    module gives ``H~ = scale eps_Ry / K^2`` and ``eps_eV = eps_Ry / sqrt(Ry)``,
    so ``scale^2 = 2 (2 pi)^2 4 gamma^2 Ry / (a0^2 k_0 k_n)``.
    """
    for eV in (60e3, 200e3, 300e3):
        for loss in (100.0, 532.0, 2000.0):
            gamma = gamma_at(eV)
            k0 = 2.0 * math.pi / debroglie_at(eV)
            kn = 2.0 * math.pi / debroglie_at(eV - loss)
            bethe = 2.0 * 4.0 * gamma**2 / _BOHR_ANG**2 * (2.0 * math.pi) ** 2
            bethe /= k0 * kn
            got = tpmod._scale(eV, loss) ** 2 * _RYDBERG_EV
            assert got == pytest.approx(bethe, rel=1e-8), (eV, loss)


def test_plane_wave_cross_section_equals_the_q_integral():
    """``int |H psi|^2 d^2r`` for ``psi = 1`` equals ``int d^2Q/(2 pi)^2 |H~|^2``.

    The m-summed ``|H~|^2`` depends only on ``Q_perp``, so the reference is a
    1-D integral of the direct form factor (no table). The cell is large, so
    the Riemann sum of the grid is exponentially accurate.
    """
    eV, gpts, sampling = 200e3, (128, 128), (0.1, 0.1)
    transitions = _channels(8, 1, 0, 1, 10.0)
    tp = build_transition_potentials(
        transitions, 8, gpts, sampling, eV, bandlimit=1.0,
        dtype=torch.complex128, method="cpu",
    )  # fmt: skip
    on_grid = float((tp.array.abs() ** 2).sum()) * sampling[0] * sampling[1]

    qz = 2 * math.pi / debroglie_at(eV) - 2 * math.pi / debroglie_at(
        eV - transitions[0].energy_loss
    )
    scale = tpmod._scale(eV, transitions[0].energy_loss)
    # The band limit 1.0 keeps the disc |Q| <= pi / dx. Composite Gauss rule on
    # geometric pieces resolves the peak near Q ~ q_z.
    q_max = math.pi / sampling[0]
    edges = np.concatenate(([0.0], np.geomspace(0.01, q_max, 12)))
    x, w = roots_legendre(48)
    q = np.concatenate(
        [
            0.5 * (b - a) * (x + 1) + a
            for a, b in zip(edges[:-1], edges[1:], strict=True)
        ]
    )
    wq = np.concatenate(
        [0.5 * (b - a) * w for a, b in zip(edges[:-1], edges[1:], strict=True)]
    )
    kabs = np.sqrt(q**2 + qz**2)
    theta = np.arctan2(q, qz)
    phi = np.zeros_like(q)
    s = sum(
        np.abs(scale * _form_factor(tr, kabs, phi, theta) / kabs**2) ** 2
        for tr in transitions
    )
    disc = float(np.sum(wq * q * s)) / (2.0 * math.pi)
    # The grid reads I_lambda from the K table; measured 9e-6.
    assert on_grid == pytest.approx(disc, rel=1e-4)


