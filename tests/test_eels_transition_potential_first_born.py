"""``build_transition_potentials`` against a direct first-Born quadrature.

The oracle in this file does not use the partial-wave route of the library
(Rayleigh expansion, Gaunt integral, radial K tables). It evaluates the
first-Born form factor

    epsilon_n0(K) = int phi_n*(r) exp(i K.r) phi_0(r) d^3r

as a plain three-dimensional quadrature in spherical coordinates. The states
are the closed-form hydrogenic bound orbital ``R_nl Y_lm`` and the free
spherical-Bessel continuum ``R_(eps l') Y_(l'm')`` per Rydberg, with scipy's
``sph_harm_y`` and ``genlaguerre``. The transition potential then follows from
the constants in the module docstring of
:mod:`scatterem.simulation.eels.physics.transition_potentials`:

    fft2(H)[f] = sigma sqrt(2 k_0/k_n) 4 pi E_h a_0 epsilon_eV(K) / (K^2 dy dx)

with ``K = (-2 pi f_a, -2 pi f_b, k_0 - k_n)`` and ``epsilon_eV =
epsilon_Ry / sqrt(Ry)``.

The three cases are O K (200 kV, 10 eV above the edge), C K (300 kV, 5 eV)
and Al L2,3 (200 kV, 20 eV) on a 32 x 32 grid with 0.2 Angstrom pixels. Every
channel ``(m, m')`` of each case is checked, as a complex number, at eight
reciprocal-space pixels that include ``q_perp = 0``. No value in this file
comes from another simulation code.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from numpy.polynomial.legendre import leggauss
from scipy.special import genlaguerre, sph_harm_y, spherical_jn

from scatterem.physics.constants import debroglie_at, phase_per_potential_at
from scatterem.simulation.eels.physics.radial import (
    free_continuum_wavefunction,
    hydrogenic_bound_wavefunction,
)
from scatterem.simulation.eels.physics.transition_potentials import (
    Transition,
    build_transition_potentials,
)

#: CODATA 2018 Hartree energy [eV] and Bohr radius [Angstrom].
_HARTREE = 27.211386245988
_BOHR = 0.529177210903
_RYDBERG = _HARTREE / 2.0

_GPTS = (32, 32)
_SAMPLING = (0.2, 0.2)

#: Reciprocal-space pixels (axis -2, axis -1), negative indices included.
_PIXELS = ((0, 0), (0, 1), (1, 0), (1, 1), (2, 3), (4, 1), (3, -2), (-5, 2))

#: Maximum error relative to the largest ``|fft2(H)|`` of the channel. The
#: measured agreement is about 2e-8; the bound leaves room for the quadrature.
_TOL = 1e-6

#: (label, Z, n, l, l', epsilon [eV], beam energy [eV])
_CASES = (
    ("O_K_200kV", 8, 1, 0, 1, 10.0, 200_000.0),
    ("C_K_300kV", 6, 1, 0, 1, 5.0, 300_000.0),
    ("Al_L23_200kV", 13, 2, 1, 2, 20.0, 200_000.0),
)


def _hydrogenic_R(Z: int, n: int, ell: int, r: np.ndarray) -> np.ndarray:
    """Closed-form hydrogenic ``R_nl(r)``, ``r`` in Bohr."""
    rho = 2.0 * Z * r / n
    norm = math.sqrt(
        (2.0 * Z / n) ** 3
        * math.factorial(n - ell - 1)
        / (2 * n * math.factorial(n + ell))
    )
    return (
        norm
        * rho**ell
        * np.exp(-rho / 2.0)
        * genlaguerre(n - ell - 1, 2 * ell + 1)(rho)
    )


class _BornQuadrature:
    """First-Born form factors of one ``(Z, n, l) -> (eps, l')`` subshell.

    Gauss-Legendre in ``r`` and ``cos(theta)``, the trapezoid rule (exact for
    trigonometric polynomials) in ``phi``. The radial sum is done once per
    ``K``; the channels ``(m, m')`` then differ only in the angular weights.
    """

    def __init__(self, Z: int, n: int, ell: int, lp: int, epsilon: float):
        self.l, self.lp = ell, lp
        k = math.sqrt(epsilon / _RYDBERG)  # 1/Bohr
        xr, wr = leggauss(240)
        rmax = 60.0 * n / Z  # the bound state is below 1e-20 of its peak here
        self.r = 0.5 * rmax * (xr + 1.0)
        u_c = k * self.r * spherical_jn(lp, k * self.r) / math.sqrt(math.pi * k)
        u_b = self.r * _hydrogenic_R(Z, n, ell, self.r)
        self.radial = 0.5 * rmax * wr * u_c * u_b

        xc, wc = leggauss(48)
        n_phi = 64
        theta = np.arccos(xc)
        phi = np.arange(n_phi) * (2.0 * np.pi / n_phi)
        self.theta, self.phi = np.meshgrid(theta, phi, indexing="ij")
        self.w_ang = wc[:, None] * np.full(n_phi, 2.0 * np.pi / n_phi)[None, :]
        self.rhat = np.stack(
            [
                np.sin(self.theta) * np.cos(self.phi),
                np.sin(self.theta) * np.sin(self.phi),
                np.cos(self.theta),
            ],
            axis=-1,
        )

    def radial_part(self, K_bohr: np.ndarray) -> np.ndarray:
        """``sum_r w u_c u_b exp(i K.r)`` on the angular grid."""
        kdot = self.rhat @ K_bohr  # (n_theta, n_phi)
        return np.tensordot(
            self.radial, np.exp(1j * self.r[:, None, None] * kdot[None]), axes=1
        )

    def channel(self, radial: np.ndarray, ml: int, mlp: int) -> complex:
        """``epsilon_n0`` per Rydberg^(1/2) for one channel."""
        ang = np.conj(sph_harm_y(self.lp, mlp, self.theta, self.phi)) * sph_harm_y(
            self.l, ml, self.theta, self.phi
        )
        return complex(np.sum(radial * ang * self.w_ang))


def _oracle(case) -> tuple[list[tuple[int, int]], np.ndarray]:
    """``fft2(H)`` of every channel at :data:`_PIXELS`, from the quadrature."""
    _, Z, n, ell, lp, epsilon, eV = case
    channels = [(ml, mlp) for ml in range(-ell, ell + 1) for mlp in range(-lp, lp + 1)]
    k0 = 2.0 * np.pi / debroglie_at(eV)
    binding = 0.5 * Z**2 / n**2 * _HARTREE
    kn = 2.0 * np.pi / debroglie_at(eV - (binding + epsilon))
    fa = np.fft.fftfreq(_GPTS[0], _SAMPLING[0])
    fb = np.fft.fftfreq(_GPTS[1], _SAMPLING[1])
    scale = (
        phase_per_potential_at(eV)
        * math.sqrt(2.0 * k0 / kn)
        * 4.0
        * np.pi
        * _HARTREE
        * _BOHR
        / math.sqrt(_RYDBERG)
        / (_SAMPLING[0] * _SAMPLING[1])
    )
    quad = _BornQuadrature(Z, n, ell, lp, epsilon)
    out = np.zeros((len(channels), len(_PIXELS)), dtype=np.complex128)
    for p, (i, j) in enumerate(_PIXELS):
        K = np.array([-2.0 * np.pi * fa[i], -2.0 * np.pi * fb[j], k0 - kn])
        radial = quad.radial_part(K * _BOHR)
        for t, (ml, mlp) in enumerate(channels):
            out[t, p] = scale * quad.channel(radial, ml, mlp) / float(K @ K)
    return channels, out


@pytest.mark.parametrize("case", _CASES, ids=[c[0] for c in _CASES])
def test_transition_potential_matches_the_first_born_quadrature(case):
    """Every channel and sampled pixel, as a complex number, within 1e-6."""
    label, Z, n, ell, lp, epsilon, eV = case
    channels, expected = _oracle(case)
    bound = hydrogenic_bound_wavefunction(Z=Z, n=n, l=ell)
    excited = free_continuum_wavefunction(l=lp, epsilon=epsilon)
    tp = build_transition_potentials(
        [Transition(bound, excited, ml, mlp) for ml, mlp in channels],
        Z,
        _GPTS,
        _SAMPLING,
        eV=eV,
        method="cpu",
        bandlimit=1.0,
        dtype=torch.complex128,
    )
    H = np.fft.fft2(tp.array.detach().cpu().numpy(), axes=(-2, -1))
    for t, (ml, mlp) in enumerate(channels):
        peak = float(np.abs(H[t]).max())
        assert peak > 0.0
        got = np.array([H[t][ij] for ij in _PIXELS])
        err = np.abs(got - expected[t]) / peak
        worst = int(np.argmax(err))
        assert err[worst] <= _TOL, (
            f"{label} (m={ml}, m'={mlp}) pixel {_PIXELS[worst]}: "
            f"got {got[worst]!r}, first Born {expected[t, worst]!r}, "
            f"error {err[worst]:.3e} of the channel peak"
        )
